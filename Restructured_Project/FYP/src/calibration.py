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