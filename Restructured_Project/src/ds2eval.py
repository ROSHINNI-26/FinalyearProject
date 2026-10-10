"""src/ds2eval.py -- DS2 selective-evaluation logic.
Labels are used ONLY to compute metrics. No threshold, gate or calibration choice is made here.
Scores: higher = more likely to be ACCEPTED. Invalid SQI rows are -inf (rejected first)."""
import numpy as np, pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score

COVS = (1.0, 0.95, 0.90, 0.80, 0.70)
CLASSES = ['N', 'S', 'V', 'F']


# ---------- scores ----------
def conf_score(logits, T):
    """Calibrated max-softmax confidence."""
    return softmax(logits / T, axis=1).max(1)

def pct_rank(score):
    return rankdata(score, method='average') / len(score)

def d_score(s_sqi, s_conf):
    """System D (pre-specified, label-free): accept iff BOTH gates accept, with both gates rejecting the same
    fraction r. 'rank_sqi >= r and rank_conf >= r' is the same as 'min(rank_sqi, rank_conf) >= r', so D is
    the top-coverage set under min(percentile ranks). Ranks are computed on the evaluated set (curve mode)."""
    return np.minimum(pct_rank(s_sqi), pct_rank(s_conf))


# ---------- coverage sweep ----------
def order_random_ties(score, rng):
    """Highest score first; ties broken at random (stable argsort would favour early records)."""
    assert not np.isnan(score).any(), 'NaN score'
    return np.lexsort((rng.random(len(score)), -score))

def sweep_arrays(score, correct, y, rng, covs=COVS):
    """One sort -> selective risk and per-class coverage at every coverage level, plus AURC and AUGRC.
    correct: float array (1 = correct)."""
    n = len(score)
    o = order_random_ties(score, rng)
    cum_err = np.cumsum(1.0 - correct[o])
    cum_cls = np.cumsum(np.eye(4, dtype=np.float32)[y[o]], axis=0)
    ks = np.maximum(1, np.rint(np.asarray(covs) * n).astype(int))
    risk = cum_err[ks - 1] / ks
    cn = np.bincount(y, minlength=4).astype(float)
    cov_cls = cum_cls[ks - 1] / np.where(cn > 0, cn, np.nan)
    aurc = float(np.mean(cum_err / np.arange(1, n + 1)))
    augrc = float(np.mean(cum_err) / n)           # area under generalisation-risk curve (Traub 2024)
    return risk, cov_cls, aurc, augrc

def gap_nsv(cov_cls):
    return np.nanmax(cov_cls[..., :3], -1) - np.nanmin(cov_cls[..., :3], -1)

def point_table(cond, scores, correct, y, covs=COVS, seed=0):
    """Point estimates for every system; scores = {name: score array}."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, sc in scores.items():
        risk, cc, aurc, augrc = sweep_arrays(sc, correct, y, rng, covs)
        gap = gap_nsv(cc)
        for i, c in enumerate(covs):
            rows.append(dict(cond=cond, system=name, coverage=c, risk=risk[i],
                             covN=cc[i, 0], covS=cc[i, 1], covV=cc[i, 2], covF=cc[i, 3],
                             gap_NSV=gap[i], AURC=aurc, AUGRC=augrc))
    return pd.DataFrame(rows)


# ---------- patient-level bootstrap ----------
def record_bootstrap(rec, scores, correct, y, covs=COVS, B=300, seed=2028):
    """Resample DS2 RECORDS with replacement. The same resample is used for every system (paired), so
    differences have valid CIs. Ranks for D are fixed from the full set (see d_score)."""
    rng = np.random.default_rng(seed)
    recs = np.unique(rec)
    idx_by = {r: np.where(rec == r)[0] for r in recs}
    out = {s: dict(risk=np.zeros((B, len(covs))), cov=np.zeros((B, len(covs), 4)), aurc=np.zeros(B))
           for s in scores}
    for b in range(B):
        idx = np.concatenate([idx_by[r] for r in rng.choice(recs, len(recs), replace=True)])
        for s, sc in scores.items():
            risk, cc, aurc, _ = sweep_arrays(sc[idx], correct[idx], y[idx], rng, covs)
            out[s]['risk'][b], out[s]['cov'][b], out[s]['aurc'][b] = risk, cc, aurc
    return out

def _ci(a, axis=0):
    return np.nanpercentile(a, 2.5, axis=axis), np.nanpercentile(a, 97.5, axis=axis)

def bootstrap_summary(boot, covs=COVS, pairs=(('B', 'C'), ('D', 'C'), ('D', 'B'))):
    rows = []
    for s, d in boot.items():
        lo, hi = _ci(d['risk'])
        for i, c in enumerate(covs):
            rows.append(dict(kind='risk', name=s, coverage=c, mean=np.nanmean(d['risk'][:, i]), lo=lo[i], hi=hi[i]))
        g = gap_nsv(d['cov'])
        glo, ghi = _ci(g)
        for i, c in enumerate(covs):
            rows.append(dict(kind='gap_NSV', name=s, coverage=c, mean=np.nanmean(g[:, i]), lo=glo[i], hi=ghi[i]))
    for a, b in pairs:
        if a in boot and b in boot:
            diff = boot[a]['risk'] - boot[b]['risk']
            lo, hi = _ci(diff)
            for i, c in enumerate(covs):
                rows.append(dict(kind='risk_diff', name=f'{a}-{b}', coverage=c, mean=diff[:, i].mean(),
                                 lo=lo[i], hi=hi[i], excludes_zero=bool(lo[i] > 0 or hi[i] < 0)))
            dA = boot[a]['aurc'] - boot[b]['aurc']
            l, h = _ci(dA)
            rows.append(dict(kind='AURC_diff', name=f'{a}-{b}', coverage=np.nan, mean=dA.mean(), lo=l, hi=h,
                             excludes_zero=bool(l > 0 or h < 0)))
    return pd.DataFrame(rows)


# ---------- random-rejection null (beat level) ----------
def random_null(y, covs=COVS, R=5000, seed=2028):
    """Class coverage if beats were rejected at random at the same overall coverage (exact hypergeometric)."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=4)
    n = len(y)
    out = {}
    for c in covs:
        k = int(round(c * n))
        draws = rng.multivariate_hypergeometric(counts, k, size=R) / np.where(counts > 0, counts, np.nan)
        out[c] = draws
    return out

def class_deviation(cond, system, cov_obs, null, covs=COVS):
    """cov_obs[i, 4] observed class coverage at covs[i]. Flags classes outside the null 95% interval."""
    rows = []
    for i, c in enumerate(covs):
        lo, hi = _ci(null[c])
        gap_null = gap_nsv(null[c])
        for k, name in enumerate(CLASSES):
            rows.append(dict(cond=cond, system=system, coverage=c, cls=name, observed=cov_obs[i, k],
                             null_mean=np.nanmean(null[c][:, k]), null_lo=lo[k], null_hi=hi[k],
                             outside_null=bool(cov_obs[i, k] < lo[k] or cov_obs[i, k] > hi[k]),
                             gap_obs=float(gap_nsv(cov_obs[i])), gap_null95=float(np.nanpercentile(gap_null, 95))))
    return pd.DataFrame(rows)


# ---------- other analyses ----------
def accept_mask(score, coverage, rng):
    k = max(1, int(round(coverage * len(score))))
    m = np.zeros(len(score), bool)
    m[order_random_ties(score, rng)[:k]] = True
    return m

def disagreement(acc_sqi, acc_conf, y):
    rows = []
    for k, name in enumerate(CLASSES):
        m = y == k
        rows.append(dict(cls=name, n=int(m.sum()), both_accept=int((acc_sqi & acc_conf)[m].sum()),
                         sqi_only=int((acc_sqi & ~acc_conf)[m].sum()),
                         conf_only=int((~acc_sqi & acc_conf)[m].sum()),
                         both_reject=int((~acc_sqi & ~acc_conf)[m].sum())))
    return pd.DataFrame(rows)

def accepted_f1(y, pred, accept):
    """Per-class F1 on accepted beats, with accepted counts shown next to it."""
    f = f1_score(y[accept], pred[accept], labels=range(4), average=None, zero_division=0)
    n = np.bincount(y[accept], minlength=4)
    return {f'F1_{c}': float(f[i]) for i, c in enumerate(CLASSES)} | {f'n_{c}': int(n[i]) for i, c in enumerate(CLASSES)}

def error_auroc(score, correct):
    """How well a score separates correct from wrong predictions (0.5 = no information). Descriptive only."""
    s = np.asarray(score, float).copy()
    fin = np.isfinite(s)
    if fin.sum() == 0 or len(np.unique(correct)) < 2:
        return np.nan
    s[~fin] = s[fin].min() - 1.0
    return float(roc_auc_score(correct, s))


# ---------- building a condition (needs the project's src modules) ----------
def build_condition(df_ds2, kind=None, snr=None, pool='test'):
    """Beat table for DS2 under one condition (clean if kind is None). Mirrors segment.build_dataset exactly
    (noise -> band-pass -> segment) and also returns the SQI features, so rows are aligned by construction."""
    from . import config as C
    from .data import load_signal
    from .noise import noise_fn
    from .segment import bandpass, segment_record, rr_features
    from .sqi import beat_features
    fn = None if kind is None else noise_fn(kind, snr, pool)
    X, RR, y, rec, ridx, F = [], [], [], [], [], []
    for r, g in df_ds2.sort_values(['record', 'r_idx']).groupby('record'):
        sig = load_signal(r)
        if fn is not None:
            sig = fn(r, sig)
        sig_f = bandpass(sig)
        r_all = g.r_idx.values
        b, keep = segment_record(sig_f, r_all)
        X.append(b); RR.append(rr_features(r_all)[keep])
        F.append(beat_features(sig, sig_f, r_all, np.where(keep)[0]))
        y += [C.CLS2IDX[c] for c in g.aami.values[keep]]
        rec += [r] * int(keep.sum()); ridx += list(r_all[keep])
    return dict(X=np.concatenate(X), RR=np.concatenate(RR), y=np.array(y), rec=np.array(rec),
                ridx=np.array(ridx), F=np.concatenate(F))
