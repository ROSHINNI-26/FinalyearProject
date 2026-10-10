
# ======================================================================
# FILE: __init__.py
# ======================================================================




# ======================================================================
# FILE: adoption.py
# ======================================================================

"""Phase 2B adoption rule. Fill in MIN_GAIN, commit/log this file BEFORE running Phase 2B,
then do not edit it after seeing results.

Usage (after run_cv for both models and all seeds):
    oof_base = {seed: logits, ...}   # baseline out-of-fold logits per seed
    oof_res  = {seed: logits, ...}   # residual CNN out-of-fold logits per seed
    decide(y, oof_base, oof_res)
"""
import numpy as np
from sklearn.metrics import f1_score

CLASSES = ["N", "S", "V", "F"]

# ---- PRE-DECLARED RULE (edit once, before running) ----
RULE = {
    "MIN_GAIN": 0.02,        # mean macro-F1 gain required. Set ~2x baseline seed std.
    "MIN_SEEDS_WON": 2,      # residual must beat baseline on >= this many seeds
    "MAX_SV_DROP": 0.0,      # mean S-F1 and V-F1 may not fall by more than this
    "MAX_ECE_INCREASE": 0.02,  # mean ECE may not rise by more than this
}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def metrics(y, logits):
    ok = ~np.isnan(logits).any(axis=1)  # only rows that were in a validation fold
    y, p = y[ok], softmax(logits[ok])
    pred = p.argmax(1)
    per_class = f1_score(y, pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    return {
        "macro_f1": float(per_class.mean()),
        "f1_S": float(per_class[1]),
        "f1_V": float(per_class[2]),
        "per_class": per_class,
        "ece": float(ece(p, y)),
    }


def decide(y, oof_base, oof_res, rule=RULE):
    seeds = sorted(oof_base)
    mb = {s: metrics(y, oof_base[s]) for s in seeds}
    mr = {s: metrics(y, oof_res[s]) for s in seeds}

    def mean(d, k):
        return float(np.mean([d[s][k] for s in seeds]))

    gain = mean(mr, "macro_f1") - mean(mb, "macro_f1")
    wins = sum(mr[s]["macro_f1"] > mb[s]["macro_f1"] for s in seeds)
    d_s = mean(mr, "f1_S") - mean(mb, "f1_S")
    d_v = mean(mr, "f1_V") - mean(mb, "f1_V")
    d_ece = mean(mr, "ece") - mean(mb, "ece")

    checks = {
        f"mean macro-F1 gain >= {rule['MIN_GAIN']}": gain >= rule["MIN_GAIN"],
        f"wins on >= {rule['MIN_SEEDS_WON']} seeds": wins >= rule["MIN_SEEDS_WON"],
        "S-F1 not worse": d_s >= -rule["MAX_SV_DROP"],
        "V-F1 not worse": d_v >= -rule["MAX_SV_DROP"],
        f"ECE increase <= {rule['MAX_ECE_INCREASE']}": d_ece <= rule["MAX_ECE_INCREASE"],
    }

    print(f"baseline macro-F1 per seed: {[round(mb[s]['macro_f1'], 4) for s in seeds]}"
          f"  (std {np.std([mb[s]['macro_f1'] for s in seeds]):.4f})")
    print(f"residual macro-F1 per seed: {[round(mr[s]['macro_f1'], 4) for s in seeds]}")
    print(f"gain {gain:+.4f} | wins {wins}/{len(seeds)} | dS {d_s:+.4f} | dV {d_v:+.4f} | dECE {d_ece:+.4f}")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
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


# ======================================================================
# FILE: noice.py
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


def fit(X, RR, y, epochs=15, bs=256, lr=1e-3, seed=42, aug=False, wd=0.0):
    torch.manual_seed(seed)
    model = BeatCNN().to(DEVICE)
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


def fit_curve(
    Xtr, Rtr, ytr,
    Xva, Rva, yva,
    epochs=40,
    bs=256,
    lr=1e-3,
    seed=42,
    wd=0.0
):

    """Train and record held-out metrics after every epoch."""

    torch.manual_seed(seed)

    model = BeatCNN().to(DEVICE)

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

