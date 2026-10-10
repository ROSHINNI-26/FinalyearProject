
# ======================================================================
# FILE: __init__.py
# ======================================================================




# ======================================================================
# FILE: adoption.py
# ======================================================================

"""Phase 2B adoption rule (src/adoption.py).

Workflow: run the BASELINE first, read its seed std, set MIN_GAIN below, log/commit this
file, and only then run the residual model. decide() refuses to run while MIN_GAIN is None.

Primary metric = macro-F1 over N, S, V. F is reported but excluded from the rule because it
is dominated by one record (descriptive only).

    oof_base = {seed: logits [n,4]}, oof_res = {seed: logits [n,4]}  (out-of-fold, same folds)
    decide(y, oof_base, oof_res)
"""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from .calibration import fit_temperature, ece

CLASSES = ["N", "S", "V", "F"]
PRIMARY = [0, 1, 2]                      # N, S, V

RULE = {
    "MIN_GAIN": 0.03,
    "MIN_SEEDS_WON": 2,
    "MAX_S_DROP": 0.01,
    "MAX_V_DROP": 0.04,
    "MAX_ECE_INCREASE": 0.025,
}
def metrics(y, logits):
    ok = ~np.isnan(logits).any(axis=1)
    y, lg = y[ok], logits[ok]
    pred = lg.argmax(1)
    f1 = f1_score(y, pred, labels=range(4), average=None, zero_division=0)
    T = fit_temperature(lg, y)
    return {
        "macro_f1": float(f1[PRIMARY].mean()),
        "f1_S": float(f1[1]), "f1_V": float(f1[2]), "f1_F": float(f1[3]),
        "ece_ts": float(ece(softmax(lg / T, 1), y)),
        "T": T,
    }


def decide(y, oof_base, oof_res, rule=RULE):
    if rule["MIN_GAIN"] is None:
        raise ValueError("Set RULE['MIN_GAIN'] from the baseline seed std first, then log it.")
    seeds = sorted(oof_base)
    mb = {s: metrics(y, oof_base[s]) for s in seeds}
    mr = {s: metrics(y, oof_res[s]) for s in seeds}
    mean = lambda d, k: float(np.mean([d[s][k] for s in seeds]))

    gain = mean(mr, "macro_f1") - mean(mb, "macro_f1")
    wins = sum(mr[s]["macro_f1"] > mb[s]["macro_f1"] for s in seeds)
    d_s, d_v = mean(mr, "f1_S") - mean(mb, "f1_S"), mean(mr, "f1_V") - mean(mb, "f1_V")
    d_ece = mean(mr, "ece_ts") - mean(mb, "ece_ts")

    checks = {
        f"mean macro-F1(NSV) gain >= {rule['MIN_GAIN']}": gain >= rule["MIN_GAIN"],
        f"wins on >= {rule['MIN_SEEDS_WON']} seeds": wins >= rule["MIN_SEEDS_WON"],
        f"S-F1 drop <= {rule['MAX_S_DROP']}": d_s >= -rule["MAX_S_DROP"],
        f"V-F1 drop <= {rule['MAX_V_DROP']}": d_v >= -rule["MAX_V_DROP"],
        f"ECE(after TS) increase <= {rule['MAX_ECE_INCREASE']}": d_ece <= rule["MAX_ECE_INCREASE"],
    }
    bf = [mb[s]["macro_f1"] for s in seeds]
    print(f"baseline macro-F1(NSV) per seed {np.round(bf, 4)} (std {np.std(bf, ddof=1):.4f})")
    print(f"residual macro-F1(NSV) per seed {np.round([mr[s]['macro_f1'] for s in seeds], 4)}")
    print(f"gain {gain:+.4f} | wins {wins}/{len(seeds)} | dS {d_s:+.4f} | dV {d_v:+.4f} | dECE {d_ece:+.4f}"
          f" | F1_F base {mean(mb, 'f1_F'):.3f} res {mean(mr, 'f1_F'):.3f} (descriptive)")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    adopt = all(checks.values())
    print("DECISION:", "ADOPT residual CNN" if adopt else "KEEP baseline CNN", "-> freeze classifier")
    return adopt


# ======================================================================
# FILE: calibration.py
# ======================================================================

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

def fit_temperature(logits, y):
    f = lambda logT: -log_softmax(logits / np.exp(logT), axis=1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method='bounded').x))

def ece(probs, y, n_bins=15):
    conf, acc = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges, e = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def brier(probs, y):
    return ((probs - np.eye(probs.shape[1])[y]) ** 2).sum(1).mean()


# ======================================================================
# FILE: config.py
# ======================================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'mitdb'
PROC = ROOT / 'data' / 'processed'
SIG = PROC / 'signals'
BEATS_CSV = PROC / 'beats.csv'

SEED = 42
FS = 360
PRE, POST = 90, 90                      # 0.25 s each side -> 180 samples

PACED = [102, 104, 107, 217]            # excluded
DS1_ORIG = [101,106,108,109,112,114,115,116,118,119,122,124,
            201,203,205,207,208,209,215,220,223,230]
DS2_ORIG = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,
            219,221,222,228,231,232,233,234]

# Primary split: 202 moved to DS1 (same subject as 201). Original split kept for a secondary run.
DS1_PRIMARY = DS1_ORIG + [202]
DS2_PRIMARY = [r for r in DS2_ORIG if r != 202]

CLASSES = ['N', 'S', 'V', 'F']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
AAMI = {**{s: 'N' for s in ['N','L','R','e','j']},
        **{s: 'S' for s in ['A','a','J','S']},
        **{s: 'V' for s in ['V','E']},
        'F': 'F'}
Q_SYMBOLS = {'/', 'f', 'Q'}             # dropped (almost absent); counted for the report


# ======================================================================
# FILE: data.py
# ======================================================================

import numpy as np, pandas as pd, wfdb
from . import config as C

def build_beat_table():
    C.SIG.mkdir(parents=True, exist_ok=True)
    rows, dropped_q = [], 0
    for rec in C.DS1_ORIG + C.DS2_ORIG:
        r = wfdb.rdrecord(str(C.RAW / str(rec)))
        a = wfdb.rdann(str(C.RAW / str(rec)), 'atr')
        assert r.fs == C.FS, (rec, r.fs)
        ch = r.sig_name.index('MLII')            # record 114 has leads swapped
        np.save(C.SIG / f'{rec}.npy', r.p_signal[:, ch].astype(np.float32))
        for samp, sym in zip(a.sample, a.symbol):
            if sym in C.AAMI:
                rows.append((rec, int(samp), sym, C.AAMI[sym]))
            elif sym in C.Q_SYMBOLS:
                dropped_q += 1
    df = pd.DataFrame(rows, columns=['record', 'r_idx', 'symbol', 'aami'])
    df['split'] = df.record.map(lambda r: 'DS1' if r in C.DS1_PRIMARY else 'DS2')
    df['split_orig'] = df.record.map(lambda r: 'DS1' if r in C.DS1_ORIG else 'DS2')
    df.to_csv(C.BEATS_CSV, index=False)
    return df, dropped_q

def load_beats():
    return pd.read_csv(C.BEATS_CSV)

def load_signal(rec):
    return np.load(C.SIG / f'{rec}.npy')


# ======================================================================
# FILE: model.py
# ======================================================================

import torch, torch.nn as nn

class BeatCNN(nn.Module):
    def __init__(self, n_classes=4, n_extra=4):
        super().__init__()
        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2),
                                 nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(1, 16, 7), block(16, 32, 5),
                                      block(32, 64, 5), block(64, 64, 3))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + n_extra, n_classes))

    def forward(self, x, rr):
        return self.head(torch.cat([self.pool(self.features(x)), rr], 1))

# ---- APPEND to src/model.py (below BeatCNN) ----
class ResBlock(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if (cin == cout and stride == 1) else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm1d(cout)))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.skip(x))


class ResBeatCNN(nn.Module):
    """Same widths (16,32,64,64), global pooling, RR concat and head as BeatCNN.
    Only the conv blocks are residual, so the architecture is the one thing that changes."""
    def __init__(self, n_classes=4, n_extra=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            ResBlock(16, 32, 5, 2), ResBlock(32, 64, 5, 2), ResBlock(64, 64, 3, 2))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + n_extra, n_classes))

    def forward(self, x, rr):
        return self.head(torch.cat([self.pool(self.features(x)), rr], 1))


# ======================================================================
# FILE: noise.py
# ======================================================================

import zlib
import numpy as np
import wfdb
from . import config as C

NST = C.ROOT / 'data' / 'raw' / 'nstdb'
NOISE_TYPES = ['bw', 'em', 'ma']
BLOCK, CHUNK = 60 * C.FS, 10 * C.FS
_cache = {}

# C31: first 45% development, middle 10% guard, final 45% test.
DEV_FRAC, GAP_FRAC = 0.45, 0.10

def regions(n_samples):
    """Return half-open sample ranges for devA, devB, gap and test."""
    assert n_samples > 0, n_samples
    dev_end = int(round(DEV_FRAC * n_samples))
    test_start = n_samples - dev_end
    half = dev_end // 2
    return {
        'devA': (0, half),
        'devB': (half, dev_end),
        'gap': (dev_end, test_start),
        'test': (test_start, n_samples),
    }

def load_noise(kind):
    if kind not in NOISE_TYPES:
        raise ValueError(f'Unknown noise type: {kind}')
    if kind not in _cache:
        r = wfdb.rdrecord(str(NST / kind))
        assert r.fs == C.FS, r.fs
        _cache[kind] = r.p_signal[:, 0].astype(np.float32)
    return _cache[kind]

def _starts(lo, hi):
    """Starts of complete CHUNK-sized chunks contained in [lo, hi)."""
    if hi - lo < CHUNK:
        return np.array([], dtype=int)
    return np.arange(lo, hi - CHUNK + 1, CHUNK, dtype=int)

def chunk_starts(n_samples, pool):
    """
    C31 pools:
      devA = first contiguous half of the first 45%
      devB = second contiguous half of the first 45%
      dev   = devA + devB
      test  = final 45%; reserved for held-out evaluation
    The middle 10% is a guard gap. Chunks must fit entirely in a region.
    """
    reg = regions(n_samples)
    if pool == 'dev':
        return np.concatenate([
            _starts(*reg['devA']),
            _starts(*reg['devB'])
        ])
    if pool not in ('devA', 'devB', 'test'):
        raise ValueError(
            f"Invalid pool {pool!r}; use 'devA', 'devB', 'dev', or 'test'"
        )
    return _starts(*reg[pool])

def make_noise(kind, n, pool, rng):
    nz = load_noise(kind)
    starts = chunk_starts(len(nz), pool)
    if len(starts) == 0:
        raise ValueError(f'No complete noise chunks for {kind}/{pool}')
    pick = rng.choice(starts, int(np.ceil(n / CHUNK)), replace=True)
    return np.concatenate([nz[s:s + CHUNK] for s in pick])[:n]

def add_noise(sig, kind, snr_db, pool, seed):
    nz = make_noise(kind, len(sig), pool, np.random.default_rng(seed))
    nz = nz - nz.mean()
    denom = np.var(nz) * 10 ** (snr_db / 10)
    if denom <= 0:
        raise ValueError('Noise variance must be positive')
    a = np.sqrt(np.var(sig) / denom)
    return (sig + a * nz).astype(np.float32)

def noise_fn(kind, snr_db, pool):
    """Return a deterministic signal function for the requested noise pool."""
    if kind not in NOISE_TYPES:
        raise ValueError(f'Unknown noise type: {kind}')
    if pool not in ('devA', 'devB', 'dev', 'test'):
        raise ValueError(f'Invalid pool: {pool}')
    def fn(rec, sig):
        seed = zlib.crc32(f'{kind}-{snr_db}-{rec}-{pool}'.encode())
        return add_noise(sig, kind, snr_db, pool, seed)
    return fn



# ======================================================================
# FILE: noise_evenodd_backup.py
# ======================================================================

import zlib
import numpy as np, wfdb
from . import config as C

NST = C.ROOT / 'data' / 'raw' / 'nstdb'
NOISE_TYPES = ['bw', 'em', 'ma']
BLOCK, CHUNK = 60 * C.FS, 10 * C.FS
_cache = {}

def load_noise(kind):
    if kind not in _cache:
        r = wfdb.rdrecord(str(NST / kind))
        assert r.fs == C.FS, r.fs
        _cache[kind] = r.p_signal[:, 0].astype(np.float32)
    return _cache[kind]

def chunk_starts(n_samples, pool):
    """pool 'dev' = even 60 s blocks, 'test' = odd blocks (disjoint)."""
    starts = []
    for b in range(n_samples // BLOCK):
        if (b % 2 == 0) == (pool == 'dev'):
            starts += list(range(b * BLOCK, (b + 1) * BLOCK - CHUNK + 1, CHUNK))
    return np.array(starts)

def make_noise(kind, n, pool, rng):
    nz = load_noise(kind)
    starts = chunk_starts(len(nz), pool)
    pick = rng.choice(starts, int(np.ceil(n / CHUNK)), replace=True)
    return np.concatenate([nz[s:s + CHUNK] for s in pick])[:n]

def add_noise(sig, kind, snr_db, pool, seed):
    nz = make_noise(kind, len(sig), pool, np.random.default_rng(seed))
    nz = nz - nz.mean()
    a = np.sqrt(np.var(sig) / (np.var(nz) * 10 ** (snr_db / 10)))
    return (sig + a * nz).astype(np.float32)

def noise_fn(kind, snr_db, pool):
    """Plug into build_dataset(..., signal_fn=noise_fn('em', 6, 'dev'))."""
    def fn(rec, sig):
        seed = zlib.crc32(f'{kind}-{snr_db}-{rec}-{pool}'.encode())
        return add_noise(sig, kind, snr_db, pool, seed)
    return fn


# ======================================================================
# FILE: segment.py
# ======================================================================

import numpy as np
from scipy.signal import butter, filtfilt
from . import config as C
from .data import load_signal

def bandpass(sig, fs=C.FS, lo=0.5, hi=40.0):
    b, a = butter(4, [lo, hi], btype='band', fs=fs)
    return filtfilt(b, a, sig).astype(np.float32)

def segment_record(signal, r_idx, pre=C.PRE, post=C.POST):
    keep = (r_idx - pre >= 0) & (r_idx + post <= len(signal))
    idx = r_idx[keep]
    beats = (np.stack([signal[i - pre:i + post] for i in idx])
             if len(idx) else np.empty((0, pre + post)))
    beats = (beats - beats.mean(1, keepdims=True)) / (beats.std(1, keepdims=True) + 1e-8)
    return beats.astype(np.float32), keep

def rr_features(r_idx, fs=C.FS):
    """Per beat: previous RR, next RR, local mean RR (last 10 beats), prev RR / local mean.
    r_idx must be sorted within one record."""
    r = np.asarray(r_idx, dtype=float)
    rr = np.diff(r) / fs
    med = np.median(rr)
    pre = np.r_[med, rr]
    post = np.r_[rr, med]
    local = np.array([pre[max(0, i - 10):i + 1].mean() for i in range(len(r))])
    return np.stack([pre, post, local, pre / local], 1).astype(np.float32)

def build_dataset(df, signal_fn=None, filt=True):
    """Returns X [n,180], RR [n,4], y, rec, ridx.
    signal_fn(rec, sig) -> sig is applied BEFORE filtering (used for noise injection in Phase 4)."""
    X, RR, y, rec, ridx = [], [], [], [], []
    for r, g in df.sort_values(['record', 'r_idx']).groupby('record'):
        sig = load_signal(r)
        if signal_fn is not None:
            sig = signal_fn(r, sig)
        if filt:
            sig = bandpass(sig)
        b, keep = segment_record(sig, g.r_idx.values)
        rr = rr_features(g.r_idx.values)[keep]
        X.append(b); RR.append(rr)
        y += [C.CLS2IDX[c] for c in g.aami.values[keep]]
        rec += [r] * len(b)
        ridx += list(g.r_idx.values[keep])
    return (np.concatenate(X), np.concatenate(RR), np.array(y),
            np.array(rec), np.array(ridx))


# ======================================================================
# FILE: selective.py
# ======================================================================

import numpy as np

def risk_coverage(score, correct):
    """Accept highest-score samples first. Returns coverage and selective risk at each k."""
    order = np.argsort(-score, kind='stable')
    err = 1.0 - correct[order].astype(float)
    k = np.arange(1, len(score) + 1)
    return k / len(score), np.cumsum(err) / k

def aurc(score, correct):
    return risk_coverage(score, correct)[1].mean()

def accept_at_coverage(score, coverage):
    n_keep = int(round(coverage * len(score)))
    mask = np.zeros(len(score), bool)
    mask[np.argsort(-score, kind='stable')[:n_keep]] = True
    return mask

def selective_risk(accept, correct):
    return 1.0 - correct[accept].mean()

def class_coverage(accept, y, n_classes=4):
    return np.array([accept[y == c].mean() if (y == c).any() else np.nan for c in range(n_classes)])

def oracle_accept(correct, coverage):
    """Rejects errors first (risk lower bound)."""
    return accept_at_coverage(correct.astype(float), coverage)


# ======================================================================
# FILE: sqi.py
# ======================================================================

"""src/sqi.py -- beat-level signal-quality indices (SQI) and two SQI gates.

Unit of analysis: one annotated beat. Each beat gets its own SQI values from a 10 s context window
(3600 samples at 360 Hz) centred on the beat; at record edges the window is shifted inward so it always
holds 10 s. Annotated R-peak positions are used (no detector), so detector failure under noise is NOT
modelled here (state this limitation in the report).

Every feature is oriented so that HIGHER = BETTER QUALITY.
  kSQI       kurtosis of the 0.5-40 Hz filtered window            (clean ECG is peaky, noise is closer to Gaussian)
  sSQI       |skewness| of the filtered window
  pSQI       power in 5-15 Hz / power in 5-40 Hz   (raw window, Hann)   (QRS band share)
  basSQI     power in 1-40 Hz / power in 0-40 Hz   (raw window, Hann)   (low = baseline wander)
  tSQI_beat  Pearson correlation of THIS beat with the median beat template of the window (filtered)
  tSQI_win   median correlation of all beats in the window with that template
These follow the families of Li/Clifford/Behar (2008-2013); the exact formulas here are this project's own
definitions and must be cited as "SQIs in the style of Behar et al. 2013", not as a reproduction of that paper.

Morphology-dependent features (tSQI_beat, and to a lesser degree pSQI) can penalise ectopic beats. That is the
effect the study measures (class-dependent rejection), so it is reported, not hidden.

Invalid measurements fail closed: a window that is non-finite or flat, or with fewer than MIN_BEATS full
annotated beats, has NaN features. Rule gates map NaN to -inf (rejected first); FusedSQI does the same.
"""
import hashlib
import numpy as np
from scipy.stats import kurtosis, skew
from . import config as C

FS = C.FS
WIN = 10 * FS
PRE, POST = C.PRE, C.POST
FEATURES = ['kSQI', 'sSQI', 'pSQI', 'basSQI', 'tSQI_beat', 'tSQI_win']
MIN_BEATS = 4
FLAT_EPS = 1e-8
SNR_CLEAN_CAP = 30.0                      # regression target (dB) assigned to un-noised signals

_FREQ = np.fft.rfftfreq(WIN, 1.0 / FS)
_HANN = np.hanning(WIN)
_BANDS = {name: (_FREQ >= lo) & (_FREQ <= hi)
          for name, (lo, hi) in dict(qrs=(5, 15), ecg=(5, 40), nobas=(1, 40), full=(0, 40)).items()}


def window_start(r, n, win=WIN):
    """First sample of the 10 s window centred on r, shifted inward at record edges."""
    return int(np.clip(r - win // 2, 0, max(n - win, 0)))


def _spectral(x):
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x * _HANN)) ** 2
    ecg, full = p[_BANDS['ecg']].sum(), p[_BANDS['full']].sum()
    if ecg <= 0 or full <= 0:
        return np.nan, np.nan
    return p[_BANDS['qrs']].sum() / ecg, p[_BANDS['nobas']].sum() / full


def record_beat_matrix(sig_f, r_all, pre=PRE, post=POST):
    """z-scored beat segments [m, pre+post] of the filtered signal for every annotated beat; ok marks full segments."""
    r_all = np.asarray(r_all)
    ok = (r_all - pre >= 0) & (r_all + post <= len(sig_f))
    B = np.zeros((len(r_all), pre + post), np.float32)
    idx = np.where(ok)[0]
    if len(idx):
        B[idx] = np.stack([sig_f[r - pre:r + post] for r in r_all[idx]])
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-8)
    return B, ok


def beat_features(sig_raw, sig_f, r_all, target_idx):
    """Features [len(target_idx), len(FEATURES)] (float32) for the annotated beats r_all[target_idx].

    sig_raw: noisy UNFILTERED record; sig_f: the same record after the 0.5-40 Hz band-pass; r_all: sorted
    annotated R positions of the record (all beats, used for the template)."""
    r_all = np.asarray(r_all)
    assert np.all(np.diff(r_all) > 0), 'r_all must be sorted and unique'
    n = len(sig_raw)
    B, ok = record_beat_matrix(sig_f, r_all)
    F = np.full((len(target_idx), len(FEATURES)), np.nan, np.float32)
    for i, j in enumerate(target_idx):
        s = window_start(int(r_all[j]), n)
        xr, xf = sig_raw[s:s + WIN], sig_f[s:s + WIN]
        if len(xr) < WIN or not (np.isfinite(xr).all() and np.isfinite(xf).all()) or xr.std() < FLAT_EPS:
            continue                                                    # invalid window: all NaN
        p, b = _spectral(xr)
        F[i, 0], F[i, 1], F[i, 2], F[i, 3] = kurtosis(xf, fisher=False), abs(skew(xf)), p, b
        lo = np.searchsorted(r_all, s + PRE, 'left'); hi = np.searchsorted(r_all, s + WIN - POST, 'right')
        rows = np.arange(lo, hi)[ok[lo:hi]]
        if len(rows) < MIN_BEATS:
            continue
        t = np.median(B[rows], axis=0); tz = (t - t.mean()) / (t.std() + 1e-8)
        c = B[rows] @ tz / B.shape[1]
        F[i, 5] = np.median(c)
        if ok[j]:
            F[i, 4] = B[j] @ tz / B.shape[1]
    return F


def rule_score(F, name):
    """Rule gate: a single published-style SQI; NaN (invalid) -> -inf (rejected first)."""
    v = F[:, FEATURES.index(name)].astype(float)
    v[~np.isfinite(v)] = -np.inf
    return v


class FusedSQI:
    """Ridge regression from the SQI features to the injected SNR (dB); the score is the predicted SNR.
    Invalid rows (any NaN feature) score -inf. Closed form, no external dependency, JSON-serialisable."""

    def __init__(self, features=FEATURES, alpha=1.0):
        self.features, self.alpha = list(features), float(alpha)
        self.mu = self.sd = self.coef = None
        self.intercept = None

    def _X(self, F):
        return F[:, [FEATURES.index(f) for f in self.features]].astype(float)

    def fit(self, F, snr_db):
        X, y = self._X(F), np.asarray(snr_db, float)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        self.intercept = float(y.mean())
        self.coef = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - self.intercept))
        return self

    def score(self, F):
        X = self._X(F)
        ok = np.isfinite(X).all(1)
        out = np.full(len(X), -np.inf)
        out[ok] = ((X[ok] - self.mu) / self.sd) @ self.coef + self.intercept
        return out

    def to_dict(self):
        return dict(features=self.features, alpha=self.alpha, mu=self.mu.tolist(), sd=self.sd.tolist(),
                    coef=self.coef.tolist(), intercept=self.intercept)

    @classmethod
    def from_dict(cls, d):
        m = cls(d['features'], d['alpha'])
        m.mu, m.sd, m.coef, m.intercept = np.array(d['mu']), np.array(d['sd']), np.array(d['coef']), d['intercept']
        return m


def source_hash():
    """sha256 of this file, stored in the lock file so the gate definition cannot change silently."""
    with open(__file__, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ======================================================================
# FILE: train.py
# ======================================================================

import numpy as np, torch, torch.nn as nn
from .model import BeatCNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def class_weights(y, n=4):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)

def fit(X, RR, y, epochs=15, bs=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = BeatCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=class_weights(y).to(DEVICE))
    Xt = torch.tensor(X).unsqueeze(1)
    Rt = torch.tensor(RR)
    yt = torch.tensor(y)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm, tot = torch.randperm(n), 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, rb, yb = Xt[idx].to(DEVICE), Rt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb, rb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f'  epoch {ep + 1}: loss {tot / n:.4f}')
    return model

@torch.no_grad()
def get_logits(model, X, RR, bs=1024):
    model.eval()
    out = [model(torch.tensor(X[i:i + bs]).unsqueeze(1).to(DEVICE),
                 torch.tensor(RR[i:i + bs]).to(DEVICE)).cpu()
           for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


# ======================================================================
# FILE: train2.py
# ======================================================================

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

from .model import BeatCNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def class_weights(y, n=4):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def augment(xb):
    """xb: [B,1,L] z-scored beats. Small random time-stretch, shift, and noise."""
    B, _, L = xb.shape
    t = torch.linspace(-1, 1, L, device=xb.device)[None, :]
    scale = torch.empty(B, 1, device=xb.device).uniform_(0.9, 1.1)
    shift = torch.empty(B, 1, device=xb.device).uniform_(-0.05, 0.05)
    gx = ((t / scale) + shift).clamp(-1, 1)
    grid = torch.stack([gx, torch.zeros_like(gx)], -1).unsqueeze(1)     # [B,1,L,2]
    xb = F.grid_sample(xb.unsqueeze(2), grid, align_corners=True).squeeze(2)
    sigma = torch.empty(B, 1, 1, device=xb.device).uniform_(0.0, 0.1)
    return xb + sigma * torch.randn_like(xb)


def fit(X, RR, y, epochs=15, bs=256, lr=1e-3, seed=42, aug=False, wd=0.0 , model_fn=BeatCNN):
    torch.manual_seed(seed)
    model = model_fn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=class_weights(y).to(DEVICE))
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Rt = torch.tensor(RR, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm, tot = torch.randperm(n), 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, rb, yb = Xt[idx].to(DEVICE), Rt[idx].to(DEVICE), yt[idx].to(DEVICE)
            if aug:
                xb = augment(xb)
            opt.zero_grad()
            loss = lossf(model(xb, rb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f'  epoch {ep + 1}: loss {tot / n:.4f}')
    return model

@torch.no_grad()
def get_logits(model, X, RR, bs=1024):

    model.eval()

    out = [
        model(
            torch.tensor(
                X[i:i + bs],
                dtype=torch.float32
            ).unsqueeze(1).to(DEVICE),

            torch.tensor(
                RR[i:i + bs],
                dtype=torch.float32
            ).to(DEVICE)
        ).cpu()

        for i in range(0, len(X), bs)
    ]

    return torch.cat(out).numpy()
def fit_curve(Xtr, Rtr, ytr, Xva, Rva, yva, epochs=40, bs=256, lr=1e-3, seed=42, wd=0.0,
              model_fn=BeatCNN):
    """Train and record held-out metrics after every epoch."""
    torch.manual_seed(seed)
    model = model_fn().to(DEVICE)  

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    lossf = nn.CrossEntropyLoss(
        weight=class_weights(ytr).to(DEVICE)
    )

    Xt = torch.tensor(
        Xtr, dtype=torch.float32
    ).unsqueeze(1)

    Rt = torch.tensor(
        Rtr, dtype=torch.float32
    )

    yt = torch.tensor(
        ytr, dtype=torch.long
    )

    n = len(Xt)

    hist = []

    for ep in range(epochs):

        model.train()

        perm = torch.randperm(n)

        for i in range(0, n, bs):

            idx = perm[i:i + bs]

            xb = Xt[idx].to(DEVICE)
            rb = Rt[idx].to(DEVICE)
            yb = yt[idx].to(DEVICE)

            opt.zero_grad()

            loss = lossf(
                model(xb, rb),
                yb
            )

            loss.backward()
            opt.step()

        # -----------------------------
        # Validation after this epoch
        # -----------------------------

        lg = get_logits(
            model,
            Xva,
            Rva
        )

        pr = lg.argmax(1)

        # Unweighted validation NLL
        nll = F.cross_entropy(
            torch.tensor(lg, dtype=torch.float32),
            torch.tensor(yva, dtype=torch.long)
        ).item()

        # N / S / V recall
        rec_ = [
            (pr[yva == c] == c).mean()
            if (yva == c).any()
            else np.nan
            for c in range(3)
        ]

        hist.append([
            nll,
            (pr == yva).mean(),
            *rec_
        ])

    return np.array(hist)

