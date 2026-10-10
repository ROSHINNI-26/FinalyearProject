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