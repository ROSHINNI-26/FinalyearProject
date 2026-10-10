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