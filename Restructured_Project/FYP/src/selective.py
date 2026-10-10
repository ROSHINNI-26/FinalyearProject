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