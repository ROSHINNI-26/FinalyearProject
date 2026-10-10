
# ================================================================================
# NOTEBOOK: 01_data_and_counts.ipynb
# ================================================================================

%load_ext autoreload
%autoreload 2
import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src import config as C
from src.data import build_beat_table, load_beats
from src.segment import build_dataset

df, dropped_q = build_beat_table()
print('beats kept:', len(df), '| Q beats dropped:', dropped_q)

assert not set(C.PACED) & set(df.record)
assert not set(C.DS1_PRIMARY) & set(C.DS2_PRIMARY)
assert df.record.nunique() == 44
assert len(C.DS1_PRIMARY) == 23 and len(C.DS2_PRIMARY) == 21
assert (df.aami == 'F').sum() == 803 and (df.aami == 'S').sum() == 2781
print('all checks passed')

print(pd.crosstab(df.split, df.aami, margins=True)[C.CLASSES + ['All']])
print(pd.crosstab(df.split_orig, df.aami, margins=True)[C.CLASSES + ['All']])

t = pd.crosstab([df.split, df.record], df.aami)[C.CLASSES]
display(t)
for sp in ['DS1', 'DS2']:
    s = t.loc[sp]
    print(sp, 'top-record share:', (s.max() / s.sum()).round(2).to_dict())
    print(sp, 'records with >=20 beats:', (s >= 20).sum().to_dict())
print(t.loc['DS1'].idxmax().to_dict(), t.loc['DS2'].idxmax().to_dict())

d1 = df[df.split == 'DS1']
X, RR, y, rec, ridx = build_dataset(d1)
fig, ax = plt.subplots(1, 4, figsize=(14, 2.5))
for k, c in enumerate(C.CLASSES):
    i = np.where(y == k)[0][0]
    ax[k].plot(X[i]); ax[k].set_title(f'{c} (record {rec[i]})')
plt.tight_layout()
print(X.shape, RR.shape, np.bincount(y))

d2 = df[df.split == 'DS2']
print(d2[d2.aami == 'N'].groupby(['record', 'symbol']).size().unstack(fill_value=0).to_string())


# ================================================================================
# NOTEBOOK: 02_train_cnn.ipynb
# ================================================================================

from google.colab import drive
drive.mount('/content/drive')

!pip install -q wfdb

import os
import sys

# Set the working directory to your project folder
project_path = '/content/drive/MyDrive/FYP'
if os.path.exists(project_path):
    os.chdir(project_path)
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Warning: {project_path} not found.")

sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src import config as C
from src.data import load_beats
from src.segment import build_dataset
from src.train2 import fit, get_logits, DEVICE
print('device:', DEVICE)

df = load_beats()
d1 = df[df.split == 'DS1']
X, RR, y, rec, ridx = build_dataset(d1)
print(X.shape, RR.shape, np.bincount(y))

FOLDS = {
    0: [109, 118, 119, 115, 230],
    1: [207, 106, 202, 112, 122],
    2: [208, 124, 114, 116],
    3: [203, 209, 101, 108],
    4: [201, 205, 215, 220, 223],
}
rec2fold = {r: k for k, rs in FOLDS.items() for r in rs}
assert sorted(rec2fold) == sorted(set(int(r) for r in rec))   # every DS1 record exactly once

oof = np.zeros((len(X), 4), dtype=np.float32)
fold_id = np.array([rec2fold[int(r)] for r in rec])
for k in range(5):
    va = np.where(fold_id == k)[0]
    tr = np.where(fold_id != k)[0]
    print(f'fold {k}: val {FOLDS[k]} | train F {(y[tr]==3).sum()}, S {(y[tr]==1).sum()}, V {(y[tr]==2).sum()}')
    m = fit(X[tr], RR[tr], y[tr])
    oof[va] = get_logits(m, X[va], RR[va])

pred = oof.argmax(1)
print(classification_report(y, pred, target_names=C.CLASSES, digits=3))
print(confusion_matrix(y, pred))
np.savez(C.PROC / 'oof_ds1_v3.npz', logits=oof, y=y, rec=rec, ridx=ridx, fold=fold_id)

for k in range(5):
    m_ = fold_id == k
    nrec = (pred[m_][y[m_] == 0] == 0).mean()
    print(f'fold {k}: acc {(pred[m_] == y[m_]).mean():.3f} | N recall {nrec:.3f} | records {sorted(set(int(r) for r in rec[m_]))}')

r_ = pd.DataFrame({'rec': rec, 'y': y, 'pred': pred})
n_rec = r_[r_.y == 0].assign(ok=lambda d: d.pred == 0).groupby('rec').ok.mean().round(3)
print(n_rec.sort_values().to_string())

from scipy.special import softmax
pr = softmax(oof, 1); conf = pr.max(1)
for r in [109, 207, 220, 203, 101, 112]:
    m_ = (rec == r) & (y == 0)
    wrong = m_ & (pred != 0)
    print(r, 'N beats:', m_.sum(), '| mean conf all:', conf[m_].mean().round(3),
          '| mean conf when wrong:', conf[wrong].mean().round(3) if wrong.any() else '-')

# from src.train2 import fit, get_logits, DEVICE
# from src.train2 import fit_curve
# H = []
# for k in range(5):
#     va = np.where(fold_id == k)[0]; tr = np.where(fold_id != k)[0]
#     H.append(fit_curve(X[tr], RR[tr], y[tr], X[va], RR[va], y[va], epochs=40))
# H = np.stack(H)                       # [fold, epoch, metric]
# mean = np.nanmean(H, axis=0)
# tab = pd.DataFrame(mean, columns=['val_NLL', 'acc', 'rec_N', 'rec_S', 'rec_V'],
#                    index=np.arange(1, 41))
# print(tab.iloc[[0, 2, 4, 7, 9, 14, 19, 24, 29, 34, 39]].round(3))

import torch, contextlib, io
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEEDS, EPOCHS = [42, 43, 44], 10
oofs = []
for s in SEEDS:
    o = np.zeros((len(X), 4), np.float32)
    for k in range(5):
        va = np.where(fold_id == k)[0]; tr = np.where(fold_id != k)[0]
        with contextlib.redirect_stdout(io.StringIO()):          # hide per-epoch lines
            m = fit(X[tr], RR[tr], y[tr], epochs=EPOCHS, seed=s)
        o[va] = get_logits(m, X[va], RR[va])
    oofs.append(o)
    p = o.argmax(1)
    print(f'seed {s}: acc {(p == y).mean():.3f} | recall N/S/V/F',
          [round((p[y == c] == c).mean(), 3) for c in range(4)])

np.savez(C.PROC / 'oof_ds1_v4_seeds.npz', logits=np.stack(oofs), y=y, rec=rec,
         ridx=ridx, fold=fold_id, seeds=SEEDS)

tabs = []
for o in oofs:
    p = o.argmax(1)
    d = pd.DataFrame({'rec': rec, 'y': y, 'ok': p == y})
    tabs.append(d[d.y == 0].groupby('rec').ok.mean())
T = pd.concat(tabs, axis=1, keys=[f'seed{s}' for s in SEEDS]).round(3)
T['range'] = T.max(1) - T.min(1)
print(T.sort_values('range', ascending=False).head(10))

from scipy.special import softmax
P = np.mean([softmax(o, 1) for o in oofs], axis=0)       # ensemble probabilities
pe = P.argmax(1)
print('ensemble acc', round((pe == y).mean(), 3),
      '| recall N/S/V/F', [round((pe[y == c] == c).mean(), 3) for c in range(4)])

def n_recall_by_record(preds_list):
    cols = []
    for p in preds_list:
        d = pd.DataFrame({'rec': rec, 'y': y, 'ok': p == y})
        cols.append(d[d.y == 0].groupby('rec').ok.mean())
    return pd.concat(cols, axis=1)

base = n_recall_by_record([o.argmax(1) for o in oofs])
print('baseline: mean per-record range of N recall across seeds:',
      (base.max(1) - base.min(1)).mean().round(3))
print(n_recall_by_record([pe]).round(3).T.to_string())    # ensemble, per record

def run_cv(seeds=(42, 43, 44), epochs=10, **kw):
    outs = []
    for s in seeds:
        o = np.zeros((len(X), 4), np.float32)
        for k in range(5):
            va = np.where(fold_id == k)[0]; tr = np.where(fold_id != k)[0]
            with contextlib.redirect_stdout(io.StringIO()):
                m = fit(X[tr], RR[tr], y[tr], epochs=epochs, seed=s, **kw)
            o[va] = get_logits(m, X[va], RR[va])
        outs.append(o)
    return outs

def summarize(name, outs):
    accs = [(o.argmax(1) == y).mean() for o in outs]
    rc = np.array([[(o.argmax(1)[y == c] == c).mean() for c in range(3)] for o in outs])
    t = n_recall_by_record([o.argmax(1) for o in outs])
    Pe = np.mean([softmax(o, 1) for o in outs], axis=0).argmax(1)
    print(f'{name:10s} acc {np.mean(accs):.3f}±{np.std(accs):.3f} | recall N/S/V {rc.mean(0).round(3)} '
          f'| N-recall range/record {(t.max(1) - t.min(1)).mean():.3f} | ensemble acc {(Pe == y).mean():.3f}')

res = {}
res['aug']    = run_cv(aug=True);            summarize('aug', res['aug'])
res['aug_wd'] = run_cv(aug=True, wd=1e-3);   summarize('aug_wd', res['aug_wd'])
summarize('baseline', oofs)

np.savez(C.PROC / 'oof_ds1_v5_augwd.npz', logits=np.stack(res['aug_wd']), y=y, rec=rec,
         ridx=ridx, fold=fold_id, seeds=[42, 43, 44])

# final models on ALL of DS1, for the DS2 evaluation in Phase 6 (DS2 is not touched here)
for s in [42, 43, 44]:
    with contextlib.redirect_stdout(io.StringIO()):
        m = fit(X, RR, y, epochs=10, seed=s, aug=True, wd=1e-3)
    torch.save(m.state_dict(), C.PROC / f'cnn_ds1_seed{s}.pt')
print('saved')


# ================================================================================
# NOTEBOOK: 02b_phase2b.ipynb
# ================================================================================

# ---- cell 1: imports (your existing cell, plus importlib) ----
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, io, contextlib, hashlib, inspect, importlib
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, torch
from scipy.special import softmax
from src import config as C
from src.data import load_beats
from src.segment import build_dataset
from src.model import BeatCNN, ResBeatCNN
from src.train2 import fit, get_logits, DEVICE
import src.adoption as A

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('device:', DEVICE)



# ---- cell 2: data + FIXED folds (201 and 202 are the same subject -> same fold) ----
df = load_beats()
d1 = df[df.split == 'DS1']
X, RR, y, rec, ridx = build_dataset(d1)

FOLDS = {
    0: [109, 118, 119, 115, 230],
    1: [207, 106, 112, 122],
    2: [208, 124, 114, 116],
    3: [203, 209, 101, 108],
    4: [201, 202, 205, 215, 220, 223],
}
rec2fold = {r: k for k, rs in FOLDS.items() for r in rs}
assert sorted(rec2fold) == sorted(set(int(r) for r in rec))
assert rec2fold[201] == rec2fold[202]
fold_id = np.array([rec2fold[int(r)] for r in rec])
for k in range(5):
    print(f'fold {k}: val beats N/S/V/F', np.bincount(y[fold_id == k], minlength=4))



# ---- cell 3: CV runner; the recipe is IDENTICAL for both models ----
SEEDS = [42, 43, 44]
RECIPE = dict(epochs=10, aug=True, wd=1e-3)       # your current baseline recipe

def run_cv(model_fn, seeds=SEEDS, **kw):
    outs = {}
    for s in seeds:
        o = np.zeros((len(X), 4), np.float32)
        for k in range(5):
            va = np.where(fold_id == k)[0]; tr = np.where(fold_id != k)[0]
            with contextlib.redirect_stdout(io.StringIO()):
                m = fit(X[tr], RR[tr], y[tr], seed=s, model_fn=model_fn, **kw)
            o[va] = get_logits(m, X[va], RR[va])
        outs[s] = o
        print('seed', s, 'done')
    return outs


print()

# ---- cell 4: BASELINE on the fixed folds, then read its seed std ----
oof_base = run_cv(BeatCNN, **RECIPE)
np.savez(C.PROC / 'oof_ds1_v6_base.npz', logits=np.stack([oof_base[s] for s in SEEDS]),
         y=y, rec=rec, ridx=ridx, fold=fold_id, seeds=SEEDS)
f1s = [A.metrics(y, oof_base[s])['macro_f1'] for s in SEEDS]
print('baseline macro-F1(NSV) per seed', np.round(f1s, 4), '| std', round(float(np.std(f1s, ddof=1)), 4))
print('suggested MIN_GAIN ~', round(max(0.01, 2 * float(np.std(f1s, ddof=1))), 3))



# ---- cell 5: lock the rule, then run the residual model ----
importlib.reload(A)
print(A.RULE)
print('adoption.py sha256:', hashlib.sha256(inspect.getsource(A).encode()).hexdigest()[:12])
assert A.RULE['MIN_GAIN'] is not None
print('params BeatCNN', sum(p.numel() for p in BeatCNN().parameters()),
      '| ResBeatCNN', sum(p.numel() for p in ResBeatCNN().parameters()))

oof_res = run_cv(ResBeatCNN, **RECIPE)
np.savez(C.PROC / 'oof_ds1_v6_res.npz', logits=np.stack([oof_res[s] for s in SEEDS]),
         y=y, rec=rec, ridx=ridx, fold=fold_id, seeds=SEEDS)


importlib.reload(A)
# ---- cell 6: decision + informational ensemble comparison ----
adopt = A.decide(y, oof_base, oof_res)
ens = lambda o: np.mean([o[s] for s in SEEDS], 0)
print('ensemble (info only): base', {k: round(v, 3) for k, v in A.metrics(y, ens(oof_base)).items()},
      '\n                      res ', {k: round(v, 3) for k, v in A.metrics(y, ens(oof_res)).items()})

import os; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)

d = np.load(C.PROC / 'oof_ds1_v6_base.npz')
print([round(A.metrics(d['y'], d['logits'][i])['macro_f1'], 4) for i in range(3)])  # expect 0.5122, 0.5736, 0.4882

## SAMPLE CHECK EXPERIMENT
## TO SET THE BASE LINE
# NO NEED TO RUN THIS
from sklearn.metrics import f1_score
rows = []
for s in SEEDS:
    p = oof_base[s].argmax(1)
    f = f1_score(y, p, labels=range(4), average=None, zero_division=0)
    m = A.metrics(y, oof_base[s])
    rows.append([s, *f.round(3), round(m['macro_f1'], 4), round(m['ece_ts'], 4), round(m['T'], 2)])
tab = pd.DataFrame(rows, columns=['seed','F1_N','F1_S','F1_V','F1_F','macroNSV','ECE_TS','T'])
print(tab.to_string(index=False))
print(tab[['F1_S','F1_V','macroNSV','ECE_TS']].std(ddof=1).round(4))

# per-fold S and V F1 for seed 42 (shows which fold drags S down)
p = oof_base[42].argmax(1)
for k in range(5):
    m_ = fold_id == k
    f = f1_score(y[m_], p[m_], labels=range(4), average=None, zero_division=0)
    print(f'fold {k}: F1 N/S/V', f[:3].round(3))

from sklearn.metrics import precision_recall_fscore_support as prfs
p = oof_base[42].argmax(1)
print(prfs(y, p, labels=[0,1,2], zero_division=0)[:2])   # precision, recall per class


# ================================================================================
# NOTEBOOK: 03_confidence_rejection.ipynb
# ================================================================================

# ---- cell 1: imports (your existing cell, plus importlib) ----
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, io, contextlib, hashlib, inspect, importlib
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())



import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import softmax
from src import config as C
from src.calibration import fit_temperature, ece, brier
from src.selective import *


d = np.load(C.PROC / 'oof_ds1_v6_base.npz')
L = d['logits']                    # [seed, n, 4]
y, rec = d['y'], d['rec']
Lens = L.mean(0)                   # ensemble by logit averaging
pred = Lens.argmax(1); correct = pred == y
print('ensemble acc', correct.mean().round(3))

T = fit_temperature(Lens, y)
P_raw, P_cal = softmax(Lens, 1), softmax(Lens / T, 1)
print('T =', round(T, 3))
for name, P in [('raw', P_raw), ('temp-scaled', P_cal)]:
    print(f'{name:12s} ECE {ece(P, y):.4f} | Brier {brier(P, y):.4f}')

scores = {'MSP raw': P_raw.max(1), 'MSP calibrated': P_cal.max(1),
          'neg. entropy (cal.)': (P_cal * np.log(P_cal + 1e-12)).sum(1)}
levels = [1.0, 0.95, 0.9, 0.8, 0.7]
base_risk = 1 - correct.mean()
rows = []
for name, s in scores.items():
    rows.append([name, round(aurc(s, correct), 4)] +
                [round(selective_risk(accept_at_coverage(s, c), correct), 4) for c in levels])
rows.append(['random', round(base_risk, 4)] + [round(base_risk, 4)] * len(levels))
rows.append(['oracle', round(aurc(correct.astype(float), correct), 4)] +
            [round(selective_risk(oracle_accept(correct, c), correct), 4) for c in levels])
print(pd.DataFrame(rows, columns=['method', 'AURC'] + [f'risk@{int(c*100)}%' for c in levels]).to_string(index=False))

s = scores['MSP calibrated']
out = []
for c in levels[1:]:
    cc = class_coverage(accept_at_coverage(s, c), y)
    out.append([f'{int(c*100)}%', *cc.round(3), round(np.nanmax(cc[:3]) - np.nanmin(cc[:3]), 3)])
print(pd.DataFrame(out, columns=['coverage'] + C.CLASSES + ['gap N/S/V']).to_string(index=False))
print('F is descriptive only (one dominant record).')

fig, ax = plt.subplots(figsize=(5, 3.5))
for name, s_ in scores.items():
    cov, risk = risk_coverage(s_, correct); ax.plot(cov, risk, label=name)
cov, risk = risk_coverage(correct.astype(float), correct); ax.plot(cov, risk, 'k--', label='oracle')
ax.axhline(base_risk, color='grey', ls=':', label='random')
ax.set_xlabel('coverage'); ax.set_ylabel('selective risk'); ax.legend(fontsize=7); plt.tight_layout()


# ---- cell 6: per-record errors and how many a confidence gate removes (80% coverage) ----
s_cal = scores['MSP calibrated']
acc80 = accept_at_coverage(s_cal, 0.8)
df = pd.DataFrame({'rec': rec, 'err': ~correct, 'rej80': ~acc80})
t = df.groupby('rec').agg(n=('err', 'size'), errors=('err', 'sum'), rejected=('rej80', 'sum'))
t['acc'] = 1 - t.errors / t.n
t['err_share'] = t.errors / t.errors.sum()
t['errors_removed'] = df[df.err & df.rej80].groupby('rec').size().reindex(t.index).fillna(0)
t['frac_err_removed'] = np.where(t.errors > 0, t.errors_removed / t.errors, np.nan)
top = t.sort_values('err_share', ascending=False)
print(top.head(8).round(3).to_string())
print('top-3 records share of all errors:', top.err_share.head(3).sum().round(3), list(top.index[:3]))
print('overall: errors removed at 80% coverage =',
      round(df[df.err & df.rej80].shape[0] / df.err.sum(), 3), '(random 0.20, oracle 1.00)')



# ---- cell 7: subject-level bootstrap at 80% coverage (201 and 202 are one subject) ----
subj = np.where(rec == 202, 201, rec)
subjects = np.unique(subj)
idx_by = {s: np.where(subj == s)[0] for s in subjects}
s_ent = scores['neg. entropy (cal.)']

def stats(idx, cov=0.8):
    yy, cc = y[idx], correct[idx]
    a = accept_at_coverage(s_cal[idx], cov)
    k = class_coverage(a, yy)
    return {'risk': selective_risk(a, cc),
            'cov_N': k[0], 'cov_S': k[1], 'cov_V': k[2], 'cov_F': k[3],
            'gap_NSV': np.nanmax(k[:3]) - np.nanmin(k[:3]),
            'AURC_msp': aurc(s_cal[idx], cc), 'AURC_ent': aurc(s_ent[idx], cc)}

point = stats(np.arange(len(y)))
B, rng, boot = 1000, np.random.default_rng(0), []          # use 2000-5000 for the final analysis
for _ in range(B):
    pick = rng.choice(subjects, len(subjects), replace=True)
    r = stats(np.concatenate([idx_by[s] for s in pick]))
    r['AURC_ent_minus_msp'] = r['AURC_ent'] - r['AURC_msp']
    boot.append(r)
boot = pd.DataFrame(boot)
rows = []
for k, v in point.items():
    lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
    note = ''
    if k.startswith('cov_'):
        note = 'includes 0.80 (class-neutral)' if lo <= 0.8 <= hi else 'EXCLUDES 0.80'
    rows.append([k, round(v, 4), round(lo, 4), round(hi, 4), int(boot[k].notna().sum()), note])
lo, hi = np.nanpercentile(boot['AURC_ent_minus_msp'], [2.5, 97.5])
rows.append(['AURC_ent - AURC_msp', round(point['AURC_ent'] - point['AURC_msp'], 4),
             round(lo, 4), round(hi, 4), B, 'includes 0' if lo <= 0 <= hi else 'EXCLUDES 0'])
print(f'{len(subjects)} subjects, {B} resamples (95% percentile intervals; cov_* is coverage of that true class at 80% overall coverage)')
print(pd.DataFrame(rows, columns=['stat', 'point', 'ci_lo', 'ci_hi', 'n_valid', 'note']).to_string(index=False))



# ---- cell 8: reliability diagram, saved for the report ----
def reliability(P, n_bins=10):
    conf, ok = P.max(1), (P.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 30:
            xs.append(conf[m].mean()); ys.append(ok[m].mean())
    return xs, ys

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
for name, P in [('raw', P_raw), ('temperature-scaled', P_cal)]:
    xs, ys = reliability(P); ax.plot(xs, ys, marker='o', ms=3, label=name)
ax.set_xlabel('confidence'); ax.set_ylabel('accuracy'); ax.legend(fontsize=8); plt.tight_layout()
fig.savefig(C.PROC / 'reliability_v6.png', dpi=200)

# ===== 03_confidence_rejection.ipynb : add these cells after cell 5 =====
# Needs the variables from cells 1-4: y, rec, correct, P_raw, P_cal, scores, C, np, pd, plt

# ---- cell 6: per-record errors and how many a confidence gate removes (80% coverage) ----
s_cal = scores['MSP calibrated']
acc80 = accept_at_coverage(s_cal, 0.8)
df = pd.DataFrame({'rec': rec, 'err': ~correct, 'rej80': ~acc80})
t = df.groupby('rec').agg(n=('err', 'size'), errors=('err', 'sum'), rejected=('rej80', 'sum'))
t['acc'] = 1 - t.errors / t.n
t['err_share'] = t.errors / t.errors.sum()
t['errors_removed'] = df[df.err & df.rej80].groupby('rec').size().reindex(t.index).fillna(0)
t['frac_err_removed'] = np.where(t.errors > 0, t.errors_removed / t.errors, np.nan)
top = t.sort_values('err_share', ascending=False)
print(top.head(8).round(3).to_string())
print('top-3 records share of all errors:', top.err_share.head(3).sum().round(3), list(top.index[:3]))
print('overall: errors removed at 80% coverage =',
      round(df[df.err & df.rej80].shape[0] / df.err.sum(), 3), '(random 0.20, oracle 1.00)')

# ---- cell 7: subject-level bootstrap at 80% coverage (201 and 202 are one subject) ----
subj = np.where(rec == 202, 201, rec)
subjects = np.unique(subj)
idx_by = {s: np.where(subj == s)[0] for s in subjects}
s_ent = scores['neg. entropy (cal.)']

def stats(idx, cov=0.8):
    yy, cc = y[idx], correct[idx]
    a = accept_at_coverage(s_cal[idx], cov)
    k = class_coverage(a, yy)
    return {'risk': selective_risk(a, cc),
            'cov_N': k[0], 'cov_S': k[1], 'cov_V': k[2], 'cov_F': k[3],
            'gap_NSV': np.nanmax(k[:3]) - np.nanmin(k[:3]),
            'AURC_msp': aurc(s_cal[idx], cc), 'AURC_ent': aurc(s_ent[idx], cc)}

point = stats(np.arange(len(y)))
B, rng, boot = 1000, np.random.default_rng(0), []          # use 2000-5000 for the final analysis
for _ in range(B):
    pick = rng.choice(subjects, len(subjects), replace=True)
    r = stats(np.concatenate([idx_by[s] for s in pick]))
    r['AURC_ent_minus_msp'] = r['AURC_ent'] - r['AURC_msp']
    boot.append(r)
boot = pd.DataFrame(boot)
rows = []
for k, v in point.items():
    lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
    note = ''
    if k.startswith('cov_'):
        note = 'includes 0.80 (class-neutral)' if lo <= 0.8 <= hi else 'EXCLUDES 0.80'
    rows.append([k, round(v, 4), round(lo, 4), round(hi, 4), int(boot[k].notna().sum()), note])
lo, hi = np.nanpercentile(boot['AURC_ent_minus_msp'], [2.5, 97.5])
rows.append(['AURC_ent - AURC_msp', round(point['AURC_ent'] - point['AURC_msp'], 4),
             round(lo, 4), round(hi, 4), B, 'includes 0' if lo <= 0 <= hi else 'EXCLUDES 0'])
print(f'{len(subjects)} subjects, {B} resamples (95% percentile intervals; cov_* is coverage of that true class at 80% overall coverage)')
print(pd.DataFrame(rows, columns=['stat', 'point', 'ci_lo', 'ci_hi', 'n_valid', 'note']).to_string(index=False))

# ---- cell 8: reliability diagram, saved for the report ----
def reliability(P, n_bins=10):
    conf, ok = P.max(1), (P.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 30:
            xs.append(conf[m].mean()); ys.append(ok[m].mean())
    return xs, ys

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
for name, P in [('raw', P_raw), ('temperature-scaled', P_cal)]:
    xs, ys = reliability(P); ax.plot(xs, ys, marker='o', ms=3, label=name)
ax.set_xlabel('confidence'); ax.set_ylabel('accuracy'); ax.legend(fontsize=8); plt.tight_layout()
fig.savefig(C.PROC / 'reliability_v6.png', dpi=200)


import os
import hashlib

TARGETS = {
    "oof_ds1_v6_subjectfolds.npz":
        "7138c8cf977be01aa1116ab78cab49da344413fd8ec5d8551846f39e37ab1e6c"
}

SEARCH_ROOTS = [
    "/content",
    "/content/drive/MyDrive/FYP",
]

for filename, expected_hash in TARGETS.items():
    print(f"\nLooking for: {filename}")
    found = False

    for root in SEARCH_ROOTS:
        if not os.path.exists(root):
            continue

        for current, dirs, files in os.walk(root):
            # Avoid searching large unrelated cache folders
            dirs[:] = [
                d for d in dirs
                if d not in {"node_modules", ".cache", "__pycache__"}
            ]

            if filename not in files:
                continue

            path = os.path.join(current, filename)
            found = True

            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha256.update(chunk)

            actual_hash = sha256.hexdigest()
            print("Path:", path)
            print("SHA256:", actual_hash)
            print("Matches project log:", actual_hash == expected_hash)

    if not found:
        print("NOT FOUND in the searched folders.")


from pathlib import Path
import hashlib
import numpy as np

folder = Path("/content/drive/MyDrive/FYP/data/processed")
names = [
    "oof_ds1_v6_subjectfolds.npz",
    "oof_ds1_v6_base.npz",
]

loaded = {}

for name in names:
    path = folder / name
    print(f"\n{'=' * 60}\n{name}")

    if not path.exists():
        print("NOT FOUND")
        continue

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print("SHA256:", digest)

    archive = np.load(path, allow_pickle=False)
    loaded[name] = {key: archive[key].copy() for key in archive.files}

    for key, arr in loaded[name].items():
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

a_name, b_name = names
if a_name in loaded and b_name in loaded:
    a, b = loaded[a_name], loaded[b_name]
    print("\nCOMPARISON")
    print("Only in subjectfolds:", sorted(set(a) - set(b)))
    print("Only in base:", sorted(set(b) - set(a)))

    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if x.shape != y.shape:
            print(f"{key}: DIFFERENT SHAPES {x.shape} vs {y.shape}")
        elif np.array_equal(x, y, equal_nan=True):
            print(f"{key}: IDENTICAL")
        elif np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            print(
                f"{key}: DIFFERENT; "
                f"max_abs_diff={np.max(np.abs(x.astype(float)-y.astype(float))):.8g}"
            )
        else:
            print(f"{key}: DIFFERENT VALUES")


# ================================================================================
# NOTEBOOK: 03_v6_subjectfolds_cells.ipynb
# ================================================================================

# =====================================================================================
# 03_confidence_rejection.ipynb  --  rerun on oof_ds1_v6_subjectfolds.npz
# No model is trained. The input artifact is only read. All outputs go to a NEW versioned directory.
# CELL 1 and CELL 2 REPLACE the old cell 1. CELLS 3-5 keep the old definitions (cells 2-4 of the old
# notebook, same formulas) with the additions listed in the comments. CELLS 6-10 are new.
# =====================================================================================

# ===== CELL 1: setup, artifact path, versioned output directory =====
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, json, hashlib, platform, datetime
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, scipy, matplotlib.pyplot as plt
from scipy.special import softmax, log_softmax
from src import config as C
from src.data import load_beats
from src.calibration import fit_temperature, ece, brier
from src.selective import *

ARTIFACT = C.PROC / 'oof_ds1_v6_subjectfolds.npz'            # PRIMARY artifact (the only file read)
ARTIFACT_SHA256 = '7138c8cf977be01aa1116ab78cab49da344413fd8ec5d8551846f39e37ab1e6c'

# expected properties of the primary artifact (values from log sections 12 and 12.1)
EXPECT = dict(
    n=53138, seeds=[42, 43, 44], class_counts=[47917, 999, 3807, 415], n_dropped_edge=11,
    folds={0: [109, 118, 119, 115, 230], 1: [207, 106, 202, 201, 112, 122],
           2: [208, 124, 114, 116], 3: [203, 209, 101, 108], 4: [205, 215, 220, 223]},
    fold_beats={0: 11002, 1: 13000, 2: 8860, 3: 9605, 4: 10671},
    seed_acc=[0.752, 0.718, 0.748], ens_acc=0.782)

LEVELS = [1.0, 0.95, 0.9, 0.8, 0.7]          # unchanged from the old notebook
COV_LEVELS = [0.95, 0.9, 0.8, 0.7]           # matched-coverage levels for class coverage / gap
B_BOOT, R_NULL = 2000, 5000                  # subject bootstrap resamples; beat-level random-rejection replicates
BOOT_SEED, PAIRED_NULL_SEED, BEAT_NULL_SEED = 2026, 2027, 2028

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

assert ARTIFACT.exists(), f'missing {ARTIFACT}'
sha_before = sha256_of(ARTIFACT)
print('LOADED ARTIFACT:', ARTIFACT)
print('SHA256          :', sha_before)
assert sha_before == ARTIFACT_SHA256, 'SHA256 differs from the audited primary artifact'

k_ver = 1
while (C.PROC / f'phase3_v6_subjectfolds_r{k_ver}').exists():
    k_ver += 1
OUT = C.PROC / f'phase3_v6_subjectfolds_r{k_ver}'
OUT.mkdir(parents=True, exist_ok=False)                      # never reuse or overwrite a previous run
assert OUT.parent == C.PROC and OUT != ARTIFACT
print('OUTPUT DIRECTORY:', OUT)

def save_df(name, df, index=False):
    p = OUT / name
    assert not p.exists(), p
    df.to_csv(p, index=index)
    return p

def save_fig(fig, name):
    p = OUT / name
    assert not p.exists(), p
    fig.savefig(p, dpi=200, bbox_inches='tight')
    return p


# ===== CELL 2: load the artifact and audit it (replaces the old data-loading lines) =====
d = np.load(ARTIFACT)
print('keys:', d.files)
L, y, rec = d['logits'], d['y'], d['rec']                    # L: [seed, n, 4]
n = EXPECT['n']
assert L.shape == (len(EXPECT['seeds']), n, 4), L.shape
assert y.shape == (n,) and rec.shape == (n,)
assert np.isfinite(L).all(), 'non-finite logits: some rows were not out-of-fold predicted'
assert set(np.unique(y)) == {0, 1, 2, 3}
assert np.bincount(y, minlength=4).tolist() == EXPECT['class_counts'], np.bincount(y, minlength=4)
if 'seeds' in d.files:
    assert [int(s) for s in d['seeds']] == EXPECT['seeds'], d['seeds']

# records: exactly the 23 DS1 primary records
assert set(int(r) for r in np.unique(rec)) == set(C.DS1_PRIMARY), 'record set differs from DS1_PRIMARY'
assert len(np.unique(rec)) == 23

# labels and record ids must agree with the beats table (independent of the artifact)
assert 'ridx' in d.files, 'artifact has no ridx; cannot cross-check labels'
ridx = d['ridx']
assert ridx.shape == (n,)
tbl = load_beats(); tbl = tbl[tbl.split == 'DS1']
assert len(tbl) - n == EXPECT['n_dropped_edge'], (len(tbl), n)     # Audit A: 11 edge beats
chk = pd.DataFrame({'record': rec.astype(int), 'r_idx': ridx.astype(int)})
assert not chk.duplicated(['record', 'r_idx']).any(), 'duplicate (record, r_idx) rows'
chk = chk.merge(tbl[['record', 'r_idx', 'aami']], on=['record', 'r_idx'], how='left', validate='one_to_one')
assert chk.aami.notna().all(), 'artifact rows missing from the beats table'
assert (chk.aami.map(C.CLS2IDX).values == y).all(), 'artifact labels disagree with the beats table'

# corrected subject-grouped fold layout (log section 12, Audit B)
rec2fold = {r: k for k, rs in EXPECT['folds'].items() for r in rs}
assert set(rec2fold) == set(C.DS1_PRIMARY) and len(rec2fold) == 23
assert rec2fold[201] == rec2fold[202], '201 and 202 are one subject and must share a fold'
fold_id = np.array([rec2fold[int(r)] for r in rec])
if 'fold' in d.files:
    assert np.array_equal(d['fold'], fold_id), 'stored fold ids differ from the corrected layout'
else:
    print('NOTE: artifact has no stored fold array; fold ids come from the corrected layout above')
assert {k: int((fold_id == k).sum()) for k in range(5)} == EXPECT['fold_beats']

# statistical unit = subject (201 and 202 merged)
subj = np.where(rec == 202, 201, rec)
subjects = np.unique(subj)
idx_by = {s: np.where(subj == s)[0] for s in subjects}
assert len(subjects) == 22

# informational only: the other artifact (not used for any result below)
_b = C.PROC / 'oof_ds1_v6_base.npz'
if _b.exists():
    _bd = np.load(_b)
    print('other artifact (NOT used): logits identical?', _bd['logits'].shape == L.shape and np.allclose(_bd['logits'], L),
          '| folds identical?', ('fold' in _bd.files) and np.array_equal(_bd['fold'], fold_id))
print('artifact audit passed: shapes, labels vs beats table, record ids, fold layout')


# ===== CELL 3: ensemble and calibration (old cell 2 + cross-fitted estimates) =====
Lens = L.mean(0)                                            # ensemble by logit averaging (unchanged)
pred = Lens.argmax(1); correct = pred == y
seed_acc = [(L[s].argmax(1) == y).mean() for s in range(L.shape[0])]
print('ensemble acc', correct.mean().round(3), '| seed acc', np.round(seed_acc, 3))
assert abs(correct.mean() - EXPECT['ens_acc']) < 0.002, 'ensemble accuracy differs from the audited 0.782: wrong artifact?'
assert np.allclose(np.round(seed_acc, 3), EXPECT['seed_acc'], atol=0.0011)

T = fit_temperature(Lens, y)                                # CALIBRATION-FIT (in-sample): T fitted on the same logits it is evaluated on
P_raw, P_cal = softmax(Lens, 1), softmax(Lens / T, 1)

# CROSS-FITTED (held-out T): for each fold, T is fitted on the OTHER folds' out-of-fold logits and applied to that fold
T_cf, Lcf = {}, np.empty_like(Lens)
for k in range(5):
    m = fold_id == k
    T_cf[k] = fit_temperature(Lens[~m], y[~m])
    Lcf[m] = Lens[m] / T_cf[k]
P_cf = softmax(Lcf, 1)

nll = lambda P: float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
cal_tab = pd.DataFrame([
    ['raw (no calibration)', ece(P_raw, y), brier(P_raw, y), nll(P_raw)],
    [f'temperature-scaled, IN-SAMPLE T={T:.3f} (calibration-fit, optimistic)', ece(P_cal, y), brier(P_cal, y), nll(P_cal)],
    ['temperature-scaled, CROSS-FITTED T (held-out folds)', ece(P_cf, y), brier(P_cf, y), nll(P_cf)]],
    columns=['calibration', 'ECE', 'Brier', 'NLL'])
print(cal_tab.round(4).to_string(index=False))
print('cross-fitted T per held-out fold:', {k: round(v, 3) for k, v in T_cf.items()})
print('Only the CROSS-FITTED row is a held-out calibration estimate. The in-sample row is a fit metric.')
save_df('calibration_table.csv', cal_tab)

# per-seed spread (each seed with its own in-sample T)
rows = []
for s, sd in enumerate(EXPECT['seeds']):
    Ts = fit_temperature(L[s], y); Pr, Pc = softmax(L[s], 1), softmax(L[s] / Ts, 1); cs = Pr.argmax(1) == y
    rows.append([sd, cs.mean(), Ts, aurc(Pr.max(1), cs), aurc(Pc.max(1), cs), ece(Pr, y), ece(Pc, y)])
seed_tab = pd.DataFrame(rows, columns=['seed', 'acc', 'T(in-sample)', 'AURC MSP raw', 'AURC MSP cal.', 'ECE raw', 'ECE TS (in-sample)'])
print(seed_tab.round(4).to_string(index=False)); save_df('per_seed_table.csv', seed_tab)


# ===== CELL 4: risk-coverage (old cell 3 + one added sensitivity row) =====
scores = {'MSP raw': P_raw.max(1), 'MSP calibrated': P_cal.max(1),
          'neg. entropy (cal.)': (P_cal * np.log(P_cal + 1e-12)).sum(1),
          'MSP cross-fitted (sensitivity)': P_cf.max(1)}      # added row; primary score is still 'MSP calibrated'
base_risk = 1 - correct.mean()
rows = []
for name, s in scores.items():
    rows.append([name, round(aurc(s, correct), 4)] +
                [round(selective_risk(accept_at_coverage(s, c), correct), 4) for c in LEVELS])
rows.append(['random', round(base_risk, 4)] + [round(base_risk, 4)] * len(LEVELS))
rows.append(['oracle', round(aurc(correct.astype(float), correct), 4)] +
            [round(selective_risk(oracle_accept(correct, c), correct), 4) for c in LEVELS])
rc_tab = pd.DataFrame(rows, columns=['method', 'AURC'] + [f'risk@{int(c*100)}%' for c in LEVELS])
print(rc_tab.to_string(index=False)); save_df('risk_coverage_table.csv', rc_tab)


# ===== CELL 5: class coverage at matched overall coverage (old cell 4) =====
s = scores['MSP calibrated']
out = []
for c in COV_LEVELS:
    cc = class_coverage(accept_at_coverage(s, c), y)
    out.append([f'{int(c*100)}%', *cc.round(3), round(np.nanmax(cc[:3]) - np.nanmin(cc[:3]), 3)])
cov_tab = pd.DataFrame(out, columns=['coverage'] + C.CLASSES + ['gap N/S/V'])
print(cov_tab.to_string(index=False)); print('F is descriptive only (one dominant record).')
save_df('class_coverage_table.csv', cov_tab)


# ===== CELL 6 (new): error by record and by fold, 80% coverage, calibrated MSP =====
acc80 = accept_at_coverage(scores['MSP calibrated'], 0.8)
dfr = pd.DataFrame({'rec': rec, 'err': ~correct, 'rej80': ~acc80})
t = dfr.groupby('rec').agg(n=('err', 'size'), errors=('err', 'sum'), rejected=('rej80', 'sum'))
t['fold'] = [rec2fold[int(r)] for r in t.index]
t['acc'] = 1 - t.errors / t.n
t['err_share'] = t.errors / t.errors.sum()
t['frac_rejected'] = t.rejected / t.n
t['errors_removed'] = dfr[dfr.err & dfr.rej80].groupby('rec').size().reindex(t.index).fillna(0)
t['frac_err_removed'] = np.where(t.errors > 0, t.errors_removed / t.errors, np.nan)
top = t.sort_values('err_share', ascending=False)
print(top.head(10).round(3).to_string())
print('top-3 records share of all errors:', round(top.err_share.head(3).sum(), 3), list(top.index[:3]))
print('overall errors removed at 80% coverage:', round(dfr[dfr.err & dfr.rej80].shape[0] / dfr.err.sum(), 3), '(random 0.20, oracle 1.00)')
print('Per-record rows are descriptive (the record is the sampling unit; no interval is given for a single record).')
save_df('error_by_record.csv', top.reset_index(), index=False)
ft = dfr.assign(fold=fold_id).groupby('fold').agg(n=('err', 'size'), acc=('err', lambda e: 1 - e.mean()))
print(ft.round(3).to_string()); save_df('accuracy_by_fold.csv', ft.reset_index())


# ===== CELL 7 (new): reliability diagram and risk-coverage plot =====
conf_ok = {'raw': P_raw, 'TS in-sample T (fit metric)': P_cal, 'TS cross-fitted T (held-out)': P_cf}
def reliability(P, n_bins=10, min_n=30):
    conf, ok = P.max(1), (P.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= min_n:
            rows.append([lo, hi, int(m.sum()), conf[m].mean(), ok[m].mean()])
    return pd.DataFrame(rows, columns=['bin_lo', 'bin_hi', 'n', 'mean_conf', 'accuracy'])
fig, ax = plt.subplots(figsize=(4.6, 4.6)); ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
rel_all = []
for name, P in conf_ok.items():
    r_ = reliability(P); ax.plot(r_.mean_conf, r_.accuracy, marker='o', ms=3, label=name); rel_all.append(r_.assign(calibration=name))
ax.set_xlabel('confidence'); ax.set_ylabel('accuracy'); ax.legend(fontsize=7); plt.tight_layout()
save_fig(fig, 'reliability_diagram.png'); save_df('reliability_bins.csv', pd.concat(rel_all))
fig, ax = plt.subplots(figsize=(5, 3.5))
for name, s_ in scores.items():
    cov, risk = risk_coverage(s_, correct); ax.plot(cov, risk, label=name)
cov, risk = risk_coverage(correct.astype(float), correct); ax.plot(cov, risk, 'k--', label='oracle')
ax.axhline(base_risk, color='grey', ls=':', label='random')
ax.set_xlabel('coverage'); ax.set_ylabel('selective risk'); ax.legend(fontsize=6); plt.tight_layout()
save_fig(fig, 'risk_coverage_curves.png')


# ===== CELL 8 (new): subject-level bootstrap (22 subjects, 201+202 merged), B = 2000 =====
# Resampling unit = subject. T is NOT refitted inside the bootstrap (the intervals are conditional on the fitted T / T_cf).
SC = scores
cls_names = C.CLASSES
edges15 = np.linspace(0, 1, 16)
LPcal = {'raw': P_raw, 'TS in-sample': P_cal, 'TS cross-fitted': P_cf}
conf_i = {k: P.max(1) for k, P in LPcal.items()}
ok_i = {k: (P.argmax(1) == y).astype(float) for k, P in LPcal.items()}
nll_i = {k: -np.log(P[np.arange(n), y] + 1e-12) for k, P in LPcal.items()}
brier_i = {k: ((P - np.eye(4)[y]) ** 2).sum(1) for k, P in LPcal.items()}

def ece_fast(conf, ok):
    b = np.clip(np.searchsorted(edges15, conf, side='left') - 1, 0, 14)
    return np.abs(np.bincount(b, weights=ok, minlength=15) - np.bincount(b, weights=conf, minlength=15)).sum() / len(conf)

def null_gap(yy, rng):                                      # one random rejection replicate at each matched coverage
    m = len(yy); perm = rng.permutation(m); cn = np.bincount(yy, minlength=4).astype(float); out = {}
    for c in COV_LEVELS:
        kk = int(round(c * m)); cv = np.bincount(yy[perm[:kk]], minlength=4) / np.where(cn > 0, cn, np.nan)
        out[c] = np.nanmax(cv[:3]) - np.nanmin(cv[:3])
    return out

def resample_stats(idx):
    yy, cc, m = y[idx], correct[idx], len(idx)
    ks = {c: int(round(c * m)) for c in COV_LEVELS}; kall = np.arange(1, m + 1); tot_err = (~cc).sum(); o = {}
    for nm in SC:
        order = np.argsort(-SC[nm][idx], kind='stable'); cum = np.cumsum(~cc[order])
        o[f'AURC|{nm}'] = (cum / kall).mean()
        for c in COV_LEVELS:
            o[f'risk@{int(c*100)}|{nm}'] = cum[ks[c] - 1] / ks[c]
        if nm == 'MSP calibrated':
            for c in COV_LEVELS:
                acc = np.zeros(m, bool); acc[order[:ks[c]]] = True
                cv = np.array([acc[yy == q].mean() if (yy == q).any() else np.nan for q in range(4)])
                for q, cl in enumerate(cls_names):
                    o[f'cov@{int(c*100)}|{cl}'] = cv[q]
                    o[f'dev@{int(c*100)}|{cl}'] = cv[q] - c            # coverage minus the class-neutral level
                o[f'gap@{int(c*100)}'] = np.nanmax(cv[:3]) - np.nanmin(cv[:3])
                o[f'err_removed@{int(c*100)}'] = 1 - cum[ks[c] - 1] / tot_err if tot_err > 0 else np.nan
    for k in LPcal:
        o[f'ECE|{k}'] = ece_fast(conf_i[k][idx], ok_i[k][idx]); o[f'Brier|{k}'] = brier_i[k][idx].mean(); o[f'NLL|{k}'] = nll_i[k][idx].mean()
    o['ECE diff|cross-fitted - raw'] = o['ECE|TS cross-fitted'] - o['ECE|raw']
    o['AURC diff|entropy - MSP cal.'] = o['AURC|neg. entropy (cal.)'] - o['AURC|MSP calibrated']
    o['AURC diff|MSP cal. - MSP raw'] = o['AURC|MSP calibrated'] - o['AURC|MSP raw']
    o['AURC diff|MSP cross-fitted - MSP cal.'] = o['AURC|MSP cross-fitted (sensitivity)'] - o['AURC|MSP calibrated']
    return o

point = resample_stats(np.arange(n))
# the fast statistics must reproduce the reference implementations on the full data
assert abs(point['AURC|MSP calibrated'] - aurc(scores['MSP calibrated'], correct)) < 1e-9
assert abs(point['risk@80|MSP calibrated'] - selective_risk(accept_at_coverage(scores['MSP calibrated'], 0.8), correct)) < 1e-9
assert abs(point['ECE|TS in-sample'] - ece(P_cal, y)) < 1e-6 and abs(point['ECE|raw'] - ece(P_raw, y)) < 1e-6

rng, null_rng = np.random.default_rng(BOOT_SEED), np.random.default_rng(PAIRED_NULL_SEED)
draws = []
for b in range(B_BOOT):
    pick = rng.choice(subjects, len(subjects), replace=True)
    idx = np.concatenate([idx_by[s_] for s_ in pick])
    r = resample_stats(idx); ng = null_gap(y[idx], null_rng)
    for c in COV_LEVELS:
        r[f'nullgap@{int(c*100)}'] = ng[c]; r[f'gap_minus_null@{int(c*100)}'] = r[f'gap@{int(c*100)}'] - ng[c]
    draws.append(r)
    if (b + 1) % 500 == 0: print('bootstrap', b + 1, '/', B_BOOT)
boot = pd.DataFrame(draws)

rows = []
for k in boot.columns:
    lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
    is_contrast = k.startswith(('dev@', 'gap_minus_null@', 'AURC diff', 'ECE diff'))     # flag only contrasts that have a natural 0
    rows.append([k, point.get(k, np.nan), np.nanmedian(boot[k]), lo, hi, int(boot[k].notna().sum()),
                 (bool(lo > 0 or hi < 0) if is_contrast else None)])
boot_tab = pd.DataFrame(rows, columns=['stat', 'point', 'boot_median', 'ci_lo', 'ci_hi', 'n_valid', 'contrast_interval_excludes_0'])
save_df('bootstrap_summary.csv', boot_tab)
np.savez_compressed(OUT / 'bootstrap_draws.npz', cols=np.array(boot.columns), values=boot.values,
                    seed=BOOT_SEED, paired_null_seed=PAIRED_NULL_SEED, B=B_BOOT)
show = lambda pre: print(boot_tab[boot_tab.stat.str.startswith(pre)].round(4).to_string(index=False))
print(f'{len(subjects)} subjects, {B_BOOT} resamples, 95% percentile intervals, T not refitted')
print('\n-- calibration (ECE/Brier/NLL; in-sample = fit metric, cross-fitted = held-out T) --')
for p_ in ['ECE|', 'Brier|', 'NLL|', 'ECE diff']: show(p_)
print('\n-- AURC and risk --');  [show(p_) for p_ in ['AURC|', 'AURC diff', 'risk@']]
print('\n-- class coverage minus overall coverage (0 = class-neutral); F descriptive --'); show('dev@')
print('\n-- N/S/V coverage gap, paired null, errors removed --'); [show(p_) for p_ in ['gap@', 'nullgap@', 'gap_minus_null@', 'err_removed@']]
print('NOTE: gap = max-min coverage is >= 0 by construction; use gap_minus_null (paired random-rejection null), not gap > 0.')


# ===== CELL 9 (new): random-rejection null for the N/S/V coverage gap, R = 5000, beat level =====
# Procedure: for each replicate, draw one uniformly random permutation of all DS1 beats (seed BEAT_NULL_SEED). At each matched
# overall coverage c, accept the first round(c*n) beats of that permutation (nested prefixes across c; each level is marginally a
# uniform random subset of size round(c*n)). Class coverage = accepted/total per true class; gap = max - min over N, S, V.
# p = (1 + #{null gap >= observed gap}) / (1 + R), one-sided. This null treats beats as exchangeable (ignores subject clustering),
# so it asks whether THIS gate's gap exceeds random beat selection in this sample; the paired subject bootstrap in CELL 8 handles subject variability.
rngn = np.random.default_rng(BEAT_NULL_SEED); cls_n = np.bincount(y, minlength=4)
null_cov = {c: np.empty((R_NULL, 4)) for c in COV_LEVELS}
for r_ in range(R_NULL):
    perm = rngn.permutation(n)
    for c in COV_LEVELS:
        null_cov[c][r_] = np.bincount(y[perm[:int(round(c * n))]], minlength=4) / cls_n
rows = []; fig, axs = plt.subplots(1, 4, figsize=(13, 2.6))
for ax, c in zip(axs, COV_LEVELS):
    ng = null_cov[c][:, :3].max(1) - null_cov[c][:, :3].min(1); obs = point[f'gap@{int(c*100)}']
    p = (1 + (ng >= obs).sum()) / (1 + R_NULL)
    rows.append([f'{int(c*100)}%', obs, ng.mean(), *np.percentile(ng, [2.5, 97.5]), ng.max(), p])
    ax.hist(ng, bins=40, color='0.7'); ax.axvline(obs, color='r'); ax.set_title(f'coverage {int(c*100)}%', fontsize=8)
plt.tight_layout(); save_fig(fig, 'gap_null_hist.png')
null_tab = pd.DataFrame(rows, columns=['coverage', 'observed_gap', 'null_mean', 'null_2.5%', 'null_97.5%', 'null_max', 'p_one_sided'])
print(f'Random-rejection null: R={R_NULL} replicates, seed={BEAT_NULL_SEED}; paired bootstrap null seed={PAIRED_NULL_SEED}')
print(null_tab.round(4).to_string(index=False)); save_df('gap_null_summary.csv', null_tab)
np.savez_compressed(OUT / 'gap_null_draws.npz', **{f'cov{int(c*100)}': null_cov[c] for c in COV_LEVELS}, seed=BEAT_NULL_SEED, R=R_NULL)


# ===== CELL 10 (new): manifest and final integrity check =====
sha_after = sha256_of(ARTIFACT)
assert sha_after == sha_before == ARTIFACT_SHA256, 'primary artifact changed during the run'
manifest = dict(
    artifact=str(ARTIFACT), artifact_sha256=sha_after, output_dir=str(OUT), n_beats=int(n), n_subjects=int(len(subjects)),
    seeds=EXPECT['seeds'], fold_layout=EXPECT['folds'], T_in_sample=float(T), T_cross_fitted={int(k): float(v) for k, v in T_cf.items()},
    ensemble_acc=float(correct.mean()), bootstrap=dict(B=B_BOOT, seed=BOOT_SEED, paired_null_seed=PAIRED_NULL_SEED, unit='subject (201+202 merged)'),
    beat_level_null=dict(R=R_NULL, seed=BEAT_NULL_SEED),
    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
    created=datetime.datetime.now().isoformat(timespec='seconds'))
with open(OUT / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(sorted(p.name for p in OUT.iterdir()))
print('primary artifact unchanged (SHA256 verified after the run)')


# ================================================================================
# NOTEBOOK: 03_v6_subjectfolds_cells_reviewed.ipynb
# ================================================================================

# =====================================================================================
# 03_confidence_rejection.ipynb  --  rerun on oof_ds1_v6_subjectfolds.npz
# No model is trained. The input artifact is only read. All outputs go to a NEW versioned directory.
# CELL 1 and CELL 2 REPLACE the old cell 1. CELLS 3-5 keep the old definitions (cells 2-4 of the old
# notebook, same formulas) with the additions listed in the comments. CELLS 6-10 are new.
# =====================================================================================

# ===== CELL 1: setup, artifact path, versioned output directory =====
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, json, hashlib, platform, datetime
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, scipy, matplotlib.pyplot as plt
from scipy.special import softmax, log_softmax
from src import config as C
from src.data import load_beats
from src.calibration import fit_temperature, ece, brier
from src.selective import *

ARTIFACT = C.PROC / 'oof_ds1_v6_subjectfolds.npz'            # PRIMARY artifact (the only file read)
ARTIFACT_SHA256 = '7138c8cf977be01aa1116ab78cab49da344413fd8ec5d8551846f39e37ab1e6c'

# expected properties of the primary artifact (values from log sections 12 and 12.1)
EXPECT = dict(
    n=53138, seeds=[42, 43, 44], class_counts=[47917, 999, 3807, 415], n_dropped_edge=11,
    folds={0: [109, 118, 119, 115, 230], 1: [207, 106, 202, 201, 112, 122],
           2: [208, 124, 114, 116], 3: [203, 209, 101, 108], 4: [205, 215, 220, 223]},
    fold_beats={0: 11002, 1: 13000, 2: 8860, 3: 9605, 4: 10671},
    seed_acc=[0.752, 0.718, 0.748], ens_acc=0.782)

LEVELS = [1.0, 0.95, 0.9, 0.8, 0.7]          # unchanged from the old notebook
COV_LEVELS = [0.95, 0.9, 0.8, 0.7]           # matched-coverage levels for class coverage / gap
B_BOOT, R_NULL = 2000, 5000                  # subject bootstrap resamples; beat-level random-rejection replicates
BOOT_SEED, PAIRED_NULL_SEED, BEAT_NULL_SEED = 2026, 2027, 2028

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

assert ARTIFACT.exists(), f'missing {ARTIFACT}'
sha_before = sha256_of(ARTIFACT)
print('LOADED ARTIFACT:', ARTIFACT)
print('SHA256          :', sha_before)
assert sha_before == ARTIFACT_SHA256, 'SHA256 differs from the audited primary artifact'

k_ver = 1
while (C.PROC / f'phase3_v6_subjectfolds_r{k_ver}').exists():
    k_ver += 1
OUT = C.PROC / f'phase3_v6_subjectfolds_r{k_ver}'
OUT.mkdir(parents=True, exist_ok=False)                      # never reuse or overwrite a previous run
assert OUT.parent == C.PROC and OUT != ARTIFACT
print('OUTPUT DIRECTORY:', OUT)

def save_df(name, df, index=False):
    p = OUT / name
    assert not p.exists(), p
    df.to_csv(p, index=index)
    return p

def save_fig(fig, name):
    p = OUT / name
    assert not p.exists(), p
    fig.savefig(p, dpi=200, bbox_inches='tight')
    return p


# ===== CELL 2: load the artifact and audit it (replaces the old data-loading lines) =====
d = np.load(ARTIFACT)
print('keys:', d.files)
L, y, rec = d['logits'], d['y'], d['rec']                    # L: [seed, n, 4]
n = EXPECT['n']
assert L.shape == (len(EXPECT['seeds']), n, 4), L.shape
assert y.shape == (n,) and rec.shape == (n,)
assert np.isfinite(L).all(), 'non-finite logits: some rows were not out-of-fold predicted'
assert set(np.unique(y)) == {0, 1, 2, 3}
assert np.bincount(y, minlength=4).tolist() == EXPECT['class_counts'], np.bincount(y, minlength=4)
if 'seeds' in d.files:
    assert [int(s) for s in d['seeds']] == EXPECT['seeds'], d['seeds']

# records: exactly the 23 DS1 primary records
assert set(int(r) for r in np.unique(rec)) == set(C.DS1_PRIMARY), 'record set differs from DS1_PRIMARY'
assert len(np.unique(rec)) == 23

# labels and record ids must agree with the beats table (independent of the artifact)
assert 'ridx' in d.files, 'artifact has no ridx; cannot cross-check labels'
ridx = d['ridx']
assert ridx.shape == (n,)
tbl = load_beats(); tbl = tbl[tbl.split == 'DS1']
assert len(tbl) - n == EXPECT['n_dropped_edge'], (len(tbl), n)     # Audit A: 11 edge beats
chk = pd.DataFrame({'record': rec.astype(int), 'r_idx': ridx.astype(int)})
assert not chk.duplicated(['record', 'r_idx']).any(), 'duplicate (record, r_idx) rows'
chk = chk.merge(tbl[['record', 'r_idx', 'aami']], on=['record', 'r_idx'], how='left', validate='one_to_one')
assert chk.aami.notna().all(), 'artifact rows missing from the beats table'
assert (chk.aami.map(C.CLS2IDX).values == y).all(), 'artifact labels disagree with the beats table'

# corrected subject-grouped fold layout (log section 12, Audit B)
rec2fold = {r: k for k, rs in EXPECT['folds'].items() for r in rs}
assert set(rec2fold) == set(C.DS1_PRIMARY) and len(rec2fold) == 23
assert rec2fold[201] == rec2fold[202], '201 and 202 are one subject and must share a fold'
fold_id = np.array([rec2fold[int(r)] for r in rec])
if 'fold' in d.files:
    assert np.array_equal(d['fold'], fold_id), 'stored fold ids differ from the corrected layout'
else:
    print('NOTE: artifact has no stored fold array; fold ids come from the corrected layout above')
assert {k: int((fold_id == k).sum()) for k in range(5)} == EXPECT['fold_beats']

# statistical unit = subject (201 and 202 merged)
subj = np.where(rec == 202, 201, rec)
subjects = np.unique(subj)
idx_by = {s: np.where(subj == s)[0] for s in subjects}
assert len(subjects) == 22

# informational only: the other artifact (not used for any result below)
_b = C.PROC / 'oof_ds1_v6_base.npz'
if _b.exists():
    _bd = np.load(_b)
    print('other artifact (NOT used): logits identical?', _bd['logits'].shape == L.shape and np.allclose(_bd['logits'], L),
          '| folds identical?', ('fold' in _bd.files) and np.array_equal(_bd['fold'], fold_id))
print('artifact audit passed: shapes, labels vs beats table, record ids, fold layout')


# ===== CELL 3: ensemble and calibration (old cell 2 + cross-fitted estimates) =====
Lens = L.mean(0)                                            # ensemble by logit averaging (unchanged)
pred = Lens.argmax(1); correct = pred == y

# Beat-to-prediction alignment audit (especially records flagged for unusually high error).
audit_beats = pd.DataFrame({
    'record': rec.astype(int), 'r_idx': ridx.astype(int), 'fold': fold_id.astype(int),
    'y_true_idx': y.astype(int), 'y_pred_idx': pred.astype(int),
    'confidence_msp_raw': softmax(Lens, axis=1).max(axis=1),
    'correct': correct.astype(bool)
})
audit_beats = audit_beats.merge(
    tbl[['record', 'r_idx', 'aami']], on=['record', 'r_idx'], how='left',
    validate='one_to_one', suffixes=('', '_table')
)
assert audit_beats['aami'].notna().all()
assert (audit_beats['y_true_idx'] == audit_beats['aami'].map(C.CLS2IDX).to_numpy()).all()
audit_beats['y_true'] = audit_beats['y_true_idx'].map({v: k for k, v in C.CLS2IDX.items()})
audit_beats['y_pred'] = audit_beats['y_pred_idx'].map({v: k for k, v in C.CLS2IDX.items()})
save_df('beat_prediction_alignment_audit.csv', audit_beats, index=False)

suspect_records = [118, 115, 207]
suspect_summary = (audit_beats[audit_beats.record.isin(suspect_records)]
    .groupby('record')
    .agg(n=('correct', 'size'), correct=('correct', 'sum'),
         mean_confidence=('confidence_msp_raw', 'mean'),
         first_r_idx=('r_idx', 'min'), last_r_idx=('r_idx', 'max'),
         unique_labels=('y_true', 'nunique'), unique_predictions=('y_pred', 'nunique'))
    .reset_index())
suspect_summary['accuracy'] = suspect_summary['correct'] / suspect_summary['n']
print('Suspicious-record alignment audit (descriptive; inspect beat_prediction_alignment_audit.csv):')
print(suspect_summary.round(4).to_string(index=False))
save_df('suspect_record_alignment_summary.csv', suspect_summary, index=False)

seed_acc = [(L[s].argmax(1) == y).mean() for s in range(L.shape[0])]
print('ensemble acc', correct.mean().round(3), '| seed acc', np.round(seed_acc, 3))
assert abs(correct.mean() - EXPECT['ens_acc']) < 0.002, 'ensemble accuracy differs from the audited 0.782: wrong artifact?'
assert np.allclose(np.round(seed_acc, 3), EXPECT['seed_acc'], atol=0.0011)

T = fit_temperature(Lens, y)                                # CALIBRATION-FIT (in-sample): T fitted on the same logits it is evaluated on
P_raw, P_cal = softmax(Lens, 1), softmax(Lens / T, 1)

# CROSS-FITTED (held-out T): for each fold, T is fitted on the OTHER folds' out-of-fold logits and applied to that fold
T_cf, Lcf = {}, np.empty_like(Lens)
for k in range(5):
    m = fold_id == k
    T_cf[k] = fit_temperature(Lens[~m], y[~m])
    Lcf[m] = Lens[m] / T_cf[k]
P_cf = softmax(Lcf, 1)

nll = lambda P: float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
cal_tab = pd.DataFrame([
    ['raw (no calibration)', ece(P_raw, y), brier(P_raw, y), nll(P_raw)],
    [f'temperature-scaled, IN-SAMPLE T={T:.3f} (calibration-fit, optimistic)', ece(P_cal, y), brier(P_cal, y), nll(P_cal)],
    ['temperature-scaled, CROSS-FITTED T (held-out folds)', ece(P_cf, y), brier(P_cf, y), nll(P_cf)]],
    columns=['calibration', 'ECE', 'Brier', 'NLL'])
print(cal_tab.round(4).to_string(index=False))
print('cross-fitted T per held-out fold:', {k: round(v, 3) for k, v in T_cf.items()})
print('Only the CROSS-FITTED row is a held-out calibration estimate. The in-sample row is a fit metric.')
save_df('calibration_table.csv', cal_tab)

# per-seed spread (each seed with its own in-sample T)
rows = []
for s, sd in enumerate(EXPECT['seeds']):
    Ts = fit_temperature(L[s], y); Pr, Pc = softmax(L[s], 1), softmax(L[s] / Ts, 1); cs = Pr.argmax(1) == y
    rows.append([sd, cs.mean(), Ts, aurc(Pr.max(1), cs), aurc(Pc.max(1), cs), ece(Pr, y), ece(Pc, y)])
seed_tab = pd.DataFrame(rows, columns=['seed', 'acc', 'T(in-sample)', 'AURC MSP raw', 'AURC MSP cal.', 'ECE raw', 'ECE TS (in-sample)'])
print(seed_tab.round(4).to_string(index=False)); save_df('per_seed_table.csv', seed_tab)


# ===== CELL 4: risk-coverage (old cell 3 + one added sensitivity row) =====
scores = {'MSP raw': P_raw.max(1), 'MSP calibrated': P_cal.max(1),
          'neg. entropy (cal.)': (P_cal * np.log(P_cal + 1e-12)).sum(1),
          'MSP cross-fitted (sensitivity)': P_cf.max(1)}      # added row; primary score is still 'MSP calibrated'
base_risk = 1 - correct.mean()
rows = []
for name, s in scores.items():
    rows.append([name, round(aurc(s, correct), 4)] +
                [round(selective_risk(accept_at_coverage(s, c), correct), 4) for c in LEVELS])
rows.append(['random', round(base_risk, 4)] + [round(base_risk, 4)] * len(LEVELS))
rows.append(['oracle', round(aurc(correct.astype(float), correct), 4)] +
            [round(selective_risk(oracle_accept(correct, c), correct), 4) for c in LEVELS])
rc_tab = pd.DataFrame(rows, columns=['method', 'AURC'] + [f'risk@{int(c*100)}%' for c in LEVELS])
print(rc_tab.to_string(index=False)); save_df('risk_coverage_table.csv', rc_tab)


# ===== CELL 5: class coverage at matched overall coverage (old cell 4) =====
s = scores['MSP calibrated']
out = []
for c in COV_LEVELS:
    cc = class_coverage(accept_at_coverage(s, c), y)
    out.append([f'{int(c*100)}%', *cc.round(3), round(np.nanmax(cc[:3]) - np.nanmin(cc[:3]), 3)])
cov_tab = pd.DataFrame(out, columns=['coverage'] + C.CLASSES + ['gap N/S/V'])
print(cov_tab.to_string(index=False)); print('F is descriptive only (one dominant record).')
save_df('class_coverage_table.csv', cov_tab)


# ===== CELL 6 (new): error by record and by fold, 80% coverage, calibrated MSP =====
acc80 = accept_at_coverage(scores['MSP calibrated'], 0.8)
dfr = pd.DataFrame({'rec': rec, 'err': ~correct, 'rej80': ~acc80})
t = dfr.groupby('rec').agg(n=('err', 'size'), errors=('err', 'sum'), rejected=('rej80', 'sum'))
t['fold'] = [rec2fold[int(r)] for r in t.index]
t['acc'] = 1 - t.errors / t.n
t['err_share'] = t.errors / t.errors.sum()
t['frac_rejected'] = t.rejected / t.n
t['errors_removed'] = dfr[dfr.err & dfr.rej80].groupby('rec').size().reindex(t.index).fillna(0)
t['frac_err_removed'] = np.where(t.errors > 0, t.errors_removed / t.errors, np.nan)
top = t.sort_values('err_share', ascending=False)
print(top.head(10).round(3).to_string())
print('top-3 records share of all errors:', round(top.err_share.head(3).sum(), 3), list(top.index[:3]))
print('overall errors removed at 80% coverage:', round(dfr[dfr.err & dfr.rej80].shape[0] / dfr.err.sum(), 3), '(random 0.20, oracle 1.00)')
print('Per-record rows are descriptive (the record is the sampling unit; no interval is given for a single record).')
save_df('error_by_record.csv', top.reset_index(), index=False)
ft = dfr.assign(fold=fold_id).groupby('fold').agg(n=('err', 'size'), acc=('err', lambda e: 1 - e.mean()))
print(ft.round(3).to_string()); save_df('accuracy_by_fold.csv', ft.reset_index())


# ===== CELL 7 (new): reliability diagram and risk-coverage plot =====
conf_ok = {'raw': P_raw, 'TS in-sample T (fit metric)': P_cal, 'TS cross-fitted T (held-out)': P_cf}
def reliability(P, n_bins=10, min_n=30):
    conf, ok = P.max(1), (P.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= min_n:
            rows.append([lo, hi, int(m.sum()), conf[m].mean(), ok[m].mean()])
    return pd.DataFrame(rows, columns=['bin_lo', 'bin_hi', 'n', 'mean_conf', 'accuracy'])
fig, ax = plt.subplots(figsize=(4.6, 4.6)); ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
rel_all = []
for name, P in conf_ok.items():
    r_ = reliability(P); ax.plot(r_.mean_conf, r_.accuracy, marker='o', ms=3, label=name); rel_all.append(r_.assign(calibration=name))
ax.set_xlabel('confidence'); ax.set_ylabel('accuracy'); ax.legend(fontsize=7); plt.tight_layout()
save_fig(fig, 'reliability_diagram.png'); save_df('reliability_bins.csv', pd.concat(rel_all))
fig, ax = plt.subplots(figsize=(5, 3.5))
for name, s_ in scores.items():
    cov, risk = risk_coverage(s_, correct); ax.plot(cov, risk, label=name)
cov, risk = risk_coverage(correct.astype(float), correct); ax.plot(cov, risk, 'k--', label='oracle')
ax.axhline(base_risk, color='grey', ls=':', label='random')
ax.set_xlabel('coverage'); ax.set_ylabel('selective risk'); ax.legend(fontsize=6); plt.tight_layout()
save_fig(fig, 'risk_coverage_curves.png')


# ===== CELL 8 (new): subject-level bootstrap (22 subjects, 201+202 merged), B = 2000 =====
# Resampling unit = subject. T is NOT refitted inside the bootstrap (the intervals are conditional on the fitted T / T_cf).
SC = scores
cls_names = C.CLASSES
edges15 = np.linspace(0, 1, 16)
LPcal = {'raw': P_raw, 'TS in-sample': P_cal, 'TS cross-fitted': P_cf}
conf_i = {k: P.max(1) for k, P in LPcal.items()}
ok_i = {k: (P.argmax(1) == y).astype(float) for k, P in LPcal.items()}
nll_i = {k: -np.log(P[np.arange(n), y] + 1e-12) for k, P in LPcal.items()}
brier_i = {k: ((P - np.eye(4)[y]) ** 2).sum(1) for k, P in LPcal.items()}

def ece_fast(conf, ok):
    b = np.clip(np.searchsorted(edges15, conf, side='left') - 1, 0, 14)
    return np.abs(np.bincount(b, weights=ok, minlength=15) - np.bincount(b, weights=conf, minlength=15)).sum() / len(conf)

def null_gap(yy, rng):                                      # one random rejection replicate at each matched coverage
    m = len(yy); perm = rng.permutation(m); cn = np.bincount(yy, minlength=4).astype(float); out = {}
    for c in COV_LEVELS:
        kk = int(round(c * m)); cv = np.bincount(yy[perm[:kk]], minlength=4) / np.where(cn > 0, cn, np.nan)
        out[c] = np.nanmax(cv[:3]) - np.nanmin(cv[:3])
    return out

def resample_stats(idx):
    yy, cc, m = y[idx], correct[idx], len(idx)
    ks = {c: int(round(c * m)) for c in COV_LEVELS}; kall = np.arange(1, m + 1); tot_err = (~cc).sum(); o = {}
    for nm in SC:
        order = np.argsort(-SC[nm][idx], kind='stable'); cum = np.cumsum(~cc[order])
        o[f'AURC|{nm}'] = (cum / kall).mean()
        for c in COV_LEVELS:
            o[f'risk@{int(c*100)}|{nm}'] = cum[ks[c] - 1] / ks[c]
        if nm == 'MSP calibrated':
            for c in COV_LEVELS:
                acc = np.zeros(m, bool); acc[order[:ks[c]]] = True
                cv = np.array([acc[yy == q].mean() if (yy == q).any() else np.nan for q in range(4)])
                for q, cl in enumerate(cls_names):
                    o[f'cov@{int(c*100)}|{cl}'] = cv[q]
                    o[f'dev@{int(c*100)}|{cl}'] = cv[q] - c            # coverage minus the class-neutral level
                o[f'gap@{int(c*100)}'] = np.nanmax(cv[:3]) - np.nanmin(cv[:3])
                o[f'err_removed@{int(c*100)}'] = 1 - cum[ks[c] - 1] / tot_err if tot_err > 0 else np.nan
    for k in LPcal:
        o[f'ECE|{k}'] = ece_fast(conf_i[k][idx], ok_i[k][idx]); o[f'Brier|{k}'] = brier_i[k][idx].mean(); o[f'NLL|{k}'] = nll_i[k][idx].mean()
    o['ECE diff|cross-fitted - raw'] = o['ECE|TS cross-fitted'] - o['ECE|raw']
    o['AURC diff|entropy - MSP cal.'] = o['AURC|neg. entropy (cal.)'] - o['AURC|MSP calibrated']
    o['AURC diff|MSP cal. - MSP raw'] = o['AURC|MSP calibrated'] - o['AURC|MSP raw']
    o['AURC diff|MSP cross-fitted - MSP cal.'] = o['AURC|MSP cross-fitted (sensitivity)'] - o['AURC|MSP calibrated']
    return o

point = resample_stats(np.arange(n))
# the fast statistics must reproduce the reference implementations on the full data
assert abs(point['AURC|MSP calibrated'] - aurc(scores['MSP calibrated'], correct)) < 1e-9
assert abs(point['risk@80|MSP calibrated'] - selective_risk(accept_at_coverage(scores['MSP calibrated'], 0.8), correct)) < 1e-9
assert abs(point['ECE|TS in-sample'] - ece(P_cal, y)) < 1e-6 and abs(point['ECE|raw'] - ece(P_raw, y)) < 1e-6

rng, null_rng = np.random.default_rng(BOOT_SEED), np.random.default_rng(PAIRED_NULL_SEED)
draws = []
for b in range(B_BOOT):
    pick = rng.choice(subjects, len(subjects), replace=True)
    idx = np.concatenate([idx_by[s_] for s_ in pick])
    r = resample_stats(idx); ng = null_gap(y[idx], null_rng)
    for c in COV_LEVELS:
        r[f'nullgap@{int(c*100)}'] = ng[c]; r[f'gap_minus_null@{int(c*100)}'] = r[f'gap@{int(c*100)}'] - ng[c]
    draws.append(r)
    if (b + 1) % 500 == 0: print('bootstrap', b + 1, '/', B_BOOT)
boot = pd.DataFrame(draws)

rows = []
for k in boot.columns:
    lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
    is_contrast = k.startswith(('dev@', 'gap_minus_null@', 'AURC diff', 'ECE diff'))     # flag only contrasts that have a natural 0
    rows.append([k, point.get(k, np.nan), np.nanmedian(boot[k]), lo, hi, int(boot[k].notna().sum()),
                 (bool(lo > 0 or hi < 0) if is_contrast else None)])
boot_tab = pd.DataFrame(rows, columns=['stat', 'point', 'boot_median', 'ci_lo', 'ci_hi', 'n_valid', 'contrast_interval_excludes_0'])
save_df('bootstrap_summary.csv', boot_tab)
np.savez_compressed(OUT / 'bootstrap_draws.npz', cols=np.array(boot.columns), values=boot.values,
                    seed=BOOT_SEED, paired_null_seed=PAIRED_NULL_SEED, B=B_BOOT)
show = lambda pre: print(boot_tab[boot_tab.stat.str.startswith(pre)].round(4).to_string(index=False))
print(f'{len(subjects)} subjects, {B_BOOT} resamples, 95% percentile intervals, T not refitted')
print('\n-- calibration (ECE/Brier/NLL; in-sample = fit metric, cross-fitted = held-out T) --')
for p_ in ['ECE|', 'Brier|', 'NLL|', 'ECE diff']: show(p_)
print('\n-- AURC and risk --');  [show(p_) for p_ in ['AURC|', 'AURC diff', 'risk@']]
print('\n-- class coverage minus overall coverage (0 = class-neutral); F descriptive --'); show('dev@')
print('\n-- N/S/V coverage gap, paired null, errors removed --'); [show(p_) for p_ in ['gap@', 'nullgap@', 'gap_minus_null@', 'err_removed@']]
print('NOTE: gap = max-min coverage is >= 0 by construction; use gap_minus_null (paired random-rejection null), not gap > 0.')


# ===== CELL 9 (new): random-rejection null for the N/S/V coverage gap, R = 5000, beat level =====
# Procedure: for each replicate, draw one uniformly random permutation of all DS1 beats (seed BEAT_NULL_SEED). At each matched
# overall coverage c, accept the first round(c*n) beats of that permutation (nested prefixes across c; each level is marginally a
# uniform random subset of size round(c*n)). Class coverage = accepted/total per true class; gap = max - min over N, S, V.
# p = (1 + #{null gap >= observed gap}) / (1 + R), one-sided. This null treats beats as exchangeable (ignores subject clustering),
# so its one-sided p-value is descriptive/exploratory, NOT subject-cluster-robust. The primary uncertainty summary is the subject bootstrap contrast in CELL 8; it still uses a beat-randomization null within each resampled subject-bootstrap sample.
rngn = np.random.default_rng(BEAT_NULL_SEED); cls_n = np.bincount(y, minlength=4)
null_cov = {c: np.empty((R_NULL, 4)) for c in COV_LEVELS}
for r_ in range(R_NULL):
    perm = rngn.permutation(n)
    for c in COV_LEVELS:
        null_cov[c][r_] = np.bincount(y[perm[:int(round(c * n))]], minlength=4) / cls_n
rows = []; fig, axs = plt.subplots(1, 4, figsize=(13, 2.6))
for ax, c in zip(axs, COV_LEVELS):
    ng = null_cov[c][:, :3].max(1) - null_cov[c][:, :3].min(1); obs = point[f'gap@{int(c*100)}']
    p = (1 + (ng >= obs).sum()) / (1 + R_NULL)
    rows.append([f'{int(c*100)}%', obs, ng.mean(), *np.percentile(ng, [2.5, 97.5]), ng.max(), p])
    ax.hist(ng, bins=40, color='0.7'); ax.axvline(obs, color='r'); ax.set_title(f'coverage {int(c*100)}%', fontsize=8)
plt.tight_layout(); save_fig(fig, 'gap_null_hist.png')
null_tab = pd.DataFrame(rows, columns=['coverage', 'observed_gap', 'null_mean', 'null_2.5%', 'null_97.5%', 'null_max', 'p_one_sided'])
print(f'Random-rejection null: R={R_NULL} replicates, seed={BEAT_NULL_SEED}; paired bootstrap null seed={PAIRED_NULL_SEED}')
print(null_tab.round(4).to_string(index=False)); save_df('gap_null_summary.csv', null_tab)

# Fill the previously missing observed point estimates in bootstrap_summary.csv.
# Point contrast = observed gap - Monte Carlo mean gap under random beat rejection.
# The percentile interval remains the subject-bootstrap distribution of gap minus one
# paired random-rejection replicate per resample; it includes random-null Monte Carlo noise.
for _, row in null_tab.iterrows():
    c_pct = int(row['coverage'].rstrip('%'))
    gkey, nkey, dkey = f'gap@{c_pct}', f'nullgap@{c_pct}', f'gap_minus_null@{c_pct}'
    null_mean = float(row['null_mean'])
    obs_gap = float(point[gkey])
    point[nkey] = null_mean
    point[dkey] = obs_gap - null_mean
    boot_tab.loc[boot_tab.stat == nkey, 'point'] = null_mean
    boot_tab.loc[boot_tab.stat == dkey, 'point'] = obs_gap - null_mean
tmp_summary = OUT / 'bootstrap_summary_corrected.tmp.csv'
assert not tmp_summary.exists()
boot_tab.to_csv(tmp_summary, index=False)
os.replace(tmp_summary, OUT / 'bootstrap_summary.csv')
print('Corrected point estimates for gap-minus-random-null written to bootstrap_summary.csv')

np.savez_compressed(OUT / 'gap_null_draws.npz', **{f'cov{int(c*100)}': null_cov[c] for c in COV_LEVELS}, seed=BEAT_NULL_SEED, R=R_NULL)


# ===== CELL 10 (new): manifest and final integrity check =====
sha_after = sha256_of(ARTIFACT)
assert sha_after == sha_before == ARTIFACT_SHA256, 'primary artifact changed during the run'
manifest = dict(
    artifact=str(ARTIFACT), artifact_sha256=sha_after, output_dir=str(OUT), n_beats=int(n), n_subjects=int(len(subjects)),
    seeds=EXPECT['seeds'], fold_layout=EXPECT['folds'], T_in_sample=float(T), T_cross_fitted={int(k): float(v) for k, v in T_cf.items()},
    ensemble_acc=float(correct.mean()), bootstrap=dict(B=B_BOOT, seed=BOOT_SEED, paired_null_seed=PAIRED_NULL_SEED, unit='subject (201+202 merged)'),
    beat_level_null=dict(R=R_NULL, seed=BEAT_NULL_SEED, inference='exploratory beat-level exchangeability; p-values not subject-cluster-robust', point_contrast='observed gap minus Monte Carlo mean random-rejection gap'),
    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
    created=datetime.datetime.now().isoformat(timespec='seconds'))
with open(OUT / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(sorted(p.name for p in OUT.iterdir()))
print('primary artifact unchanged (SHA256 verified after the run)')

# ===== CELL 11 (new): verify that artifact rows line up with beats and labels; per-record prediction tables =====
# Run in the SAME session after CELL 10 (so OUT is the current run folder). Read-only diagnostic: no training.
import torch
from src.segment import build_dataset
from src.model import BeatCNN
from src.train2 import get_logits, DEVICE

d1 = load_beats(); d1 = d1[d1.split == 'DS1']
X, RR, y2, rec2, ridx2 = build_dataset(d1)

# (a) the artifact's row order must be exactly the order of the dataset the models were trained and evaluated on
assert X.shape[0] == n
assert np.array_equal(rec2, rec) and np.array_equal(ridx2, ridx) and np.array_equal(y2, y), 'artifact row order differs from build_dataset'
print('row order, record ids, r_idx and labels identical to build_dataset: OK')

# (b) the R peak must sit at the window centre (index 90): offset of the largest |amplitude| within +/-10 samples
off = np.abs(X[:, 80:101]).argmax(1) - 10
centred = np.abs(off) <= 4
cen = pd.DataFrame({'rec': rec, 'centred': centred}).groupby('rec').centred.mean()
print('beats with R peak within 4 samples of the window centre: overall', round(centred.mean(), 3))
print(cen.loc[[115, 118, 207]].round(3).to_string())

# (c) in-sample accuracy of the saved final models (trained on ALL of DS1, so they have seen these beats)
Lf = []
for sd in EXPECT['seeds']:
    m = BeatCNN().to(DEVICE)
    m.load_state_dict(torch.load(C.PROC / f'cnn_ds1_seed{sd}.pt', map_location=DEVICE)); m.eval()
    Lf.append(get_logits(m, X, RR))
pf = np.mean(Lf, 0).argmax(1)
rows = []
for r in [115, 118, 207]:
    mk = rec == r
    rows.append([r, int(mk.sum()), float((pred[mk] == y[mk]).mean()), float((pf[mk] == y[mk]).mean())])
align = pd.DataFrame(rows, columns=['record', 'n', 'acc_out_of_fold', 'acc_final_models_in_sample'])
print(align.round(3).to_string(index=False))
print('If label and waveform are aligned, the in-sample accuracy should be clearly higher than the out-of-fold accuracy.')
save_df('alignment_verification_summary.csv', align)

# (d) true versus predicted counts for the suspicious records (out-of-fold ensemble)
names = C.CLASSES
tabs = []
for r in [115, 118, 207]:
    mk = rec == r
    ct = pd.crosstab(pd.Series(np.array(names)[y[mk]], name='true'), pd.Series(np.array(names)[pred[mk]], name='pred'))
    print(f'\nrecord {r}: true (rows) vs predicted (columns), out-of-fold ensemble'); print(ct.to_string())
    tabs.append(ct.reset_index().assign(record=r))
save_df('suspect_record_confusion.csv', pd.concat(tabs, ignore_index=True))




import numpy as np
from scipy.special import softmax
from pathlib import Path

ARTIFACT = Path(
    "/content/drive/MyDrive/FYP/data/processed/"
    "oof_ds1_v6_subjectfolds.npz"
)

with np.load(ARTIFACT, allow_pickle=False) as d:
    saved_logits = d["logits"].copy()
    y_null = d["y"].copy()
    rec_null = d["rec"].copy()
    fold_null = d["fold"].copy()

# Average logits across the 3 seeds, matching the project's ensemble rule.
ensemble_logits = saved_logits.mean(axis=0)

# Cross-fitted temperatures from the earlier calibration output.
# Confirm these are the exact values recorded for this artifact.
T_BY_FOLD = {
    0: 1.252,
    1: 1.387,
    2: 1.501,
    3: 1.403,
    4: 1.545,
}

msp_crossfit = np.empty(len(y_null), dtype=np.float64)

for f, temperature in T_BY_FOLD.items():
    mask = fold_null == f
    assert mask.any(), f"No beats found for fold {f}"
    msp_crossfit[mask] = softmax(
        ensemble_logits[mask] / temperature, axis=1
    ).max(axis=1)

assert np.isfinite(msp_crossfit).all()
assert len(msp_crossfit) == len(y_null) == len(rec_null)

print("Cross-fitted MSP created.")
print("Beats:", len(msp_crossfit))
print("Fold counts:", {
    int(f): int((fold_null == f).sum())
    for f in np.unique(fold_null)
})
print("MSP range:", round(msp_crossfit.min(), 4),
      "to", round(msp_crossfit.max(), 4))


# ===== CELL 12: within-record random-rejection null =====
# Purpose:
# Test whether the N/S/V class-coverage gap exceeds a null that preserves
# each record's accepted-beat count but randomizes accepted beats within records.
#
# No training. Does not modify the OOF artifact.
# Run after Cell 11, in the same notebook session.

import numpy as np
import pandas as pd
from pathlib import Path

# IMPORTANT:
# Use EXACTLY the confidence score used for the existing class-coverage table.
# Change this line if your notebook uses a different variable.
score = msp_crossfit
y_arr = y_null
rec_arr = rec_null

assert len(score) == len(y) == len(rec), "score/y/rec lengths differ"
assert np.isfinite(score).all(), "Confidence score contains non-finite values"

# Keep the same AAMI label mapping as the earlier cells.
class_names = list(C.CLASSES)
target_classes = ["N", "S", "V"]
class_ids = {name: class_names.index(name) for name in target_classes}

y_arr = np.asarray(y)
rec_arr = np.asarray(rec)
n = len(y_arr)

coverages = [0.95, 0.90, 0.80, 0.70]
R = 5000
rng = np.random.default_rng(2029)

# Stable global ranking: higher confidence = more likely to be accepted.
# The number accepted is fixed exactly at each requested coverage.
ranked = np.argsort(-score, kind="stable")

def class_coverage_gap(accepted):
    """Return per-class coverage and max-minus-min gap for N/S/V."""
    cov = {}
    for name in target_classes:
        mask = (y_arr == class_ids[name])
        cov[name] = (
            float(accepted[mask].mean()) if mask.any() else np.nan
        )
    values = np.array([cov[c] for c in target_classes], dtype=float)
    gap = float(values.max() - values.min())
    return cov, gap

# Record indices are fixed once; randomization happens only within each record.
record_indices = {
    r: np.flatnonzero(rec_arr == r)
    for r in np.unique(rec_arr)
}

results = []
null_rows = []

for target in coverages:
    n_accept = int(round(target * n))
    accepted_obs = np.zeros(n, dtype=bool)
    accepted_obs[ranked[:n_accept]] = True

    obs_cov, obs_gap = class_coverage_gap(accepted_obs)

    # Preserve the observed accepted count in EVERY record.
    accepted_per_record = {
        r: int(accepted_obs[idx].sum())
        for r, idx in record_indices.items()
    }

    null_gaps = np.empty(R, dtype=float)
    null_class_cov = {c: np.empty(R, dtype=float)
                      for c in target_classes}

    for b in range(R):
        accepted_null = np.zeros(n, dtype=bool)

        for r, idx in record_indices.items():
            k = accepted_per_record[r]
            if k > 0:
                chosen = rng.choice(idx, size=k, replace=False)
                accepted_null[chosen] = True

        cov_null, gap_null = class_coverage_gap(accepted_null)
        null_gaps[b] = gap_null

        for c in target_classes:
            null_class_cov[c][b] = cov_null[c]

    # One-sided test: is the observed gap unusually LARGE?
    p_value = (
        1 + np.count_nonzero(null_gaps >= obs_gap)
    ) / (R + 1)

    null_mean = float(null_gaps.mean())
    null_lo, null_hi = np.quantile(null_gaps, [0.025, 0.975])

    row = {
        "target_coverage": target,
        "actual_coverage": float(accepted_obs.mean()),
        "observed_gap_NSV": obs_gap,
        "null_mean_gap": null_mean,
        "observed_minus_null_mean": obs_gap - null_mean,
        "null_2.5pct": float(null_lo),
        "null_97.5pct": float(null_hi),
        "p_one_sided": float(p_value),
        "n_null_replicates": R,
    }

    for c in target_classes:
        row[f"coverage_{c}"] = obs_cov[c]
        row[f"null_mean_coverage_{c}"] = float(
            null_class_cov[c].mean()
        )

    results.append(row)

    for value in null_gaps:
        null_rows.append({
            "target_coverage": target,
            "null_gap_NSV": float(value)
        })

summary = pd.DataFrame(results)
null_draws = pd.DataFrame(null_rows)

out_dir = Path(OUT)
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "within_record_null_summary.csv"
draws_path = out_dir / "within_record_null_draws.csv"

summary.to_csv(summary_path, index=False)
null_draws.to_csv(draws_path, index=False)

print("Within-record null complete")
print(f"Records: {len(record_indices)}")
print(f"Beats: {n}")
print(f"Null replicates per coverage: {R}")
print(summary.round(4).to_string(index=False))
print(f"\nSaved: {summary_path}")
print(f"Saved: {draws_path}")

# ===== CELL 12 (new): independent within-record randomization null (cross-check) =====
# Run in the same session after CELL 10 (needs: scores, point, rec, y, n, COV_LEVELS, OUT, accept_at_coverage, class_coverage, save_df).
# Null: for each record keep the number of beats the REAL gate accepted, but choose WHICH beats at random inside the record.
# The observed gap is computed from the same calibrated-MSP gate as the primary analysis, so it must equal the primary gap.
R_WR, WR_SEED = 5000, 2029
rng_wr = np.random.default_rng(WR_SEED)
s_cal = scores['MSP calibrated']
idx_rec = {r: np.where(rec == r)[0] for r in np.unique(rec)}
cls_n = np.bincount(y, minlength=4).astype(float)

summary, perclass, draws = [], [], {}
for c in COV_LEVELS:
    acc = accept_at_coverage(s_cal, c)                                  # SAME gate and rule as the primary analysis
    k_rec = {r: int(acc[ix].sum()) for r, ix in idx_rec.items()}        # accepted beats per record
    assert sum(k_rec.values()) == int(round(c * n))
    assert all(0 <= k_rec[r] <= len(idx_rec[r]) for r in k_rec)
    cov_obs = class_coverage(acc, y)
    gap_obs = np.nanmax(cov_obs[:3]) - np.nanmin(cov_obs[:3])
    assert abs(gap_obs - point[f'gap@{int(c*100)}']) < 1e-9, 'observed gap differs from the primary analysis'
    ncov = np.empty((R_WR, 4))
    for i in range(R_WR):
        cnt = np.zeros(4)
        for r, ix in idx_rec.items():
            if k_rec[r]:
                cnt += np.bincount(y[rng_wr.choice(ix, k_rec[r], replace=False)], minlength=4)
        ncov[i] = cnt / cls_n
    ng = ncov[:, :3].max(1) - ncov[:, :3].min(1)
    p_up = (1 + (ng >= gap_obs).sum()) / (1 + R_WR)                     # H1: observed gap larger than the null
    p_lo = (1 + (ng <= gap_obs).sum()) / (1 + R_WR)                     # H1: observed gap smaller than the null
    lo, hi = np.percentile(ng, [2.5, 97.5])
    summary.append([f'{int(c*100)}%', gap_obs, ng.mean(), lo, hi, p_up, p_lo])
    for q, cl in enumerate(C.CLASSES):                                  # class-specific effect beyond record composition
        qlo, qhi = np.nanpercentile(ncov[:, q], [2.5, 97.5])
        perclass.append([f'{int(c*100)}%', cl, cov_obs[q], ncov[:, q].mean(), cov_obs[q] - ncov[:, q].mean(), qlo, qhi,
                         bool(cov_obs[q] < qlo or cov_obs[q] > qhi)])
    draws[f'cov{int(c*100)}'] = ncov

wr_tab = pd.DataFrame(summary, columns=['coverage', 'observed_gap', 'null_mean', 'null_2.5%', 'null_97.5%', 'p_obs_larger', 'p_obs_smaller'])
pc_tab = pd.DataFrame(perclass, columns=['coverage', 'class', 'observed_cov', 'null_mean_cov', 'obs_minus_null', 'null_2.5%', 'null_97.5%', 'outside_null_95%'])
print(f'Within-record null: R={R_WR}, seed={WR_SEED}, per-record accepted counts taken from the calibrated-MSP gate')
print(wr_tab.round(4).to_string(index=False))
print(pc_tab.round(4).to_string(index=False))
print('Conditional on these 22 subjects and their per-record acceptance counts; not a test of generalisation to new patients.')
save_df('within_record_null_crosscheck_summary.csv', wr_tab); save_df('within_record_null_crosscheck_per_class.csv', pc_tab)
np.savez_compressed(OUT / 'within_record_null_crosscheck_draws.npz', **draws, seed=WR_SEED, R=R_WR)


# ================================================================================
# NOTEBOOK: 04_noise_injection.ipynb
# ================================================================================

from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, json, zlib, hashlib, time
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())


import sys; sys.path.insert(0, '..')
import numpy as np, matplotlib.pyplot as plt
from src import config as C
from src.data import load_signal
from src.noise import *

# cell 2: noise records as loaded (check these numbers against PhysioNet's description)
for k in NOISE_TYPES:
    nz = load_noise(k); print(k, nz.shape, 'std', nz.std().round(3))

# cell 3: achieved SNR must match the target
sig = load_signal(106)
for k in NOISE_TYPES:
    for snr in [24, 12, 0, -6]:
        noisy = noise_fn(k, snr, 'dev')(106, sig)
        got = 10 * np.log10(np.var(sig) / np.var(noisy - sig))
        assert abs(got - snr) < 0.5, (k, snr, got)
print('SNR check passed')

# cell 4: visual check
fig, ax = plt.subplots(3, 3, figsize=(14, 6), sharex=True)
seg = slice(72000, 72000 + 5 * C.FS)
for i, k in enumerate(NOISE_TYPES):
    for j, snr in enumerate([24, 6, -6]):
        ax[i, j].plot(noise_fn(k, snr, 'dev')(106, sig)[seg], lw=0.7)
        ax[i, j].set_title(f'{k} @ {snr} dB')
plt.tight_layout()

# ===== notebook 04, ADDENDUM cell (run after the existing cells; reads no DS2 data) =====
# Checks still missing before DS2: all six SNRs, the 'test' pool that DS2 will use, pool separation, reproducibility,
# and the provenance of the added noise (it must really be built from the pool's chunks). Record 106 is a DS1 record.
import zlib
sig = load_signal(106)

def rebuild(kind, snr, rec_id, pool, signal):
    """Re-derive the added noise from src.noise's own recipe (same seed, same chunk draw, same scaling)."""
    nz = load_noise(kind); starts = chunk_starts(len(nz), pool)
    rng = np.random.default_rng(zlib.crc32(f'{kind}-{snr}-{rec_id}-{pool}'.encode()))
    pick = rng.choice(starts, int(np.ceil(len(signal) / CHUNK)), replace=True)
    z = np.concatenate([nz[s:s + CHUNK] for s in pick])[:len(signal)]
    z = z - z.mean()
    return pick, np.sqrt(np.var(signal) / (np.var(z) * 10 ** (snr / 10))) * z

for k in NOISE_TYPES:
    nz = load_noise(k)
    dev_s, test_s = set(chunk_starts(len(nz), 'dev')), set(chunk_starts(len(nz), 'test'))
    assert dev_s and test_s and not (dev_s & test_s), f'{k}: dev and test chunk pools overlap'
    for pool, own, other in [('dev', dev_s, test_s), ('test', test_s, dev_s)]:
        for snr in [24, 18, 12, 6, 0, -6]:
            out = noise_fn(k, snr, pool)(106, sig)
            got = 10 * np.log10(np.var(sig) / np.var(out - sig))
            assert abs(got - snr) < 0.5, (k, pool, snr, got)
            pick, added = rebuild(k, snr, 106, pool, sig)
            assert set(pick.tolist()) <= own and not (set(pick.tolist()) & other), (k, pool, snr, 'wrong pool')
            assert np.allclose(out - sig, added, atol=1e-4), (k, pool, snr, 'noise is not built from the pool chunks')
    a, b = noise_fn(k, 6, 'test')(106, sig), noise_fn(k, 6, 'test')(106, sig)
    assert np.array_equal(a, b), f'{k}: noise is not reproducible'
    c = noise_fn(k, 6, 'test')(108, load_signal(108))
    assert not np.allclose(c[:len(a)] - load_signal(108)[:len(a)], a - sig), f'{k}: records share the same noise draw'
    print(k, f'chunks dev/test: {len(dev_s)}/{len(test_s)} | six SNRs x two pools, provenance, reproducibility: OK')
print('Note: every record draws from the same', len(test_s), 'test chunks, so noise is not independent across records.')


# ================================================================================
# NOTEBOOK: 05_sqi_dev_cells.ipynb
# ================================================================================

# =====================================================================================
# 05_sqi_dev.ipynb  --  SQI gate development on DS1 ONLY (no DS2 data is read anywhere in this notebook)
# Needs src/sqi.py and src/noise.py (the file was called noice.py; rename it). Nothing is trained except the
# low-capacity SQI regression. The classifier artifact is only read. Outputs go to a NEW versioned directory.
# =====================================================================================

# ===== CELL 1: setup, primary artifact (read only), versioned output directory =====
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, json, zlib, hashlib, platform, datetime, time
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, scipy, matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src import config as C
from src.data import load_beats, load_signal
from src.segment import bandpass
from src.noise import NOISE_TYPES, load_noise, CHUNK, BLOCK, chunk_starts
from src.calibration import fit_temperature
from src.selective import *
import src.sqi as Q
from src.sqi import FEATURES, beat_features, rule_score, FusedSQI, SNR_CLEAN_CAP

ARTIFACT = C.PROC / 'oof_ds1_v6_subjectfolds.npz'
ARTIFACT_SHA256 = '7138c8cf977be01aa1116ab78cab49da344413fd8ec5d8551846f39e37ab1e6c'
FOLDS = {0: [109, 118, 119, 115, 230], 1: [207, 106, 202, 201, 112, 122],
         2: [208, 124, 114, 116], 3: [203, 209, 101, 108], 4: [205, 215, 220, 223]}
EXPECTED_ENS_ACC = 0.782
SNRS = [24, 18, 12, 6, 0, -6]            # same grid as the planned DS2 evaluation
N_TARGET = 200                           # beats sampled per record and condition for the development table
COV_LEVELS = [0.95, 0.9, 0.8, 0.7]
R_WR, WR_SEED = 2000, 2031               # within-record null replicates (use 5000 for the final DS2 analysis)

sha256_of = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
assert sha256_of(ARTIFACT) == ARTIFACT_SHA256, 'primary artifact hash differs'
k_ver = 1
while (C.PROC / f'sqi_dev_r{k_ver}').exists():
    k_ver += 1
OUT = C.PROC / f'sqi_dev_r{k_ver}'; OUT.mkdir(parents=True, exist_ok=False)
print('ARTIFACT:', ARTIFACT, '\nOUTPUT  :', OUT)

def save_df(name, df, index=False):
    p = OUT / name; assert not p.exists(), p; df.to_csv(p, index=index); return p

d = np.load(ARTIFACT)
L, y, rec, ridx, fold = d['logits'], d['y'], d['rec'], d['ridx'], d['fold']
n = len(y); assert L.shape == (3, n, 4)
rec2fold = {r: k for k, rs in FOLDS.items() for r in rs}
assert set(rec2fold) == set(C.DS1_PRIMARY) and rec2fold[201] == rec2fold[202]
assert np.array_equal(fold, np.array([rec2fold[int(r)] for r in rec])), 'fold layout differs from section 12'
Lens = L.mean(0); correct = Lens.argmax(1) == y
assert abs(correct.mean() - EXPECTED_ENS_ACC) < 0.002, 'ensemble accuracy differs from the primary baseline'
T = fit_temperature(Lens, y); conf_cal = softmax(Lens / T, 1).max(1)       # calibrated MSP, as in notebook 03
tbl = load_beats(); tbl = tbl[tbl.split == 'DS1'].sort_values(['record', 'r_idx'])
print('ensemble acc', correct.mean().round(3), '| T', round(T, 3), '| beats', n)


# ===== CELL 2: development noise pools (A for training, B for validation) and SNR validation =====
# C31: A = first half of the development region; B = second half; test = final 45%.
# The middle 10% is a guard gap; the final 45% is reserved for test.
def pool_starts(kind, label):
    """C31: A is the first dev region; B is the second dev region."""
    assert label in ("A", "B"), f"Invalid development pool: {label!r}"
    pool = "devA" if label == "A" else "devB"
    return chunk_starts(len(load_noise(kind)), pool)
def noisy(rec_id, sig, kind, snr, label):
    """Same SNR definition as src.noise.add_noise (record-level signal variance), but drawing chunks from pool A or B."""
    rng = np.random.default_rng(zlib.crc32(f'sqi-{kind}-{snr}-{rec_id}-{label}'.encode()))
    nz = load_noise(kind); st = pool_starts(kind, label)
    z = np.concatenate([nz[s:s + CHUNK] for s in rng.choice(st, int(np.ceil(len(sig) / CHUNK)), replace=True)])[:len(sig)]
    z = z - z.mean()
    return (sig + np.sqrt(np.var(sig) / (np.var(z) * 10 ** (snr / 10))) * z).astype(np.float32)

sig106 = load_signal(106)
for kind in NOISE_TYPES:
    A, B = set(pool_starts(kind, 'A')), set(pool_starts(kind, 'B'))
    test = set(chunk_starts(len(load_noise(kind)), 'test'))
    assert not (A & B) and not ((A | B) & test), f'noise pools overlap for {kind}'
    for snr in [24, 6, -6]:
        for lab in ['A', 'B']:
            nsy = noisy(106, sig106, kind, snr, lab)
            got = 10 * np.log10(np.var(sig106) / np.var(nsy - sig106))
            assert abs(got - snr) < 0.5, (kind, snr, lab, got)
    print(kind, 'chunks A/B/test:', len(A), len(B), len(test), '| achieved SNR within 0.5 dB, pools disjoint: OK')


# ===== CELL 3: SQI features on DS1 under clean and development-noise conditions (sampled beats) =====
conds = [('clean', None, 'clean')] + [(k, s, lab) for k in NOISE_TYPES for s in SNRS for lab in ('A', 'B')]
rows, t0 = [], time.time()
for r in sorted(tbl.record.unique()):
    g = tbl[tbl.record == r]; r_all, yc = g.r_idx.values, g.aami.map(C.CLS2IDX).values
    rr = np.random.default_rng(zlib.crc32(f'sqi-targets-{r}'.encode()))
    tgt = np.sort(rr.choice(len(r_all), size=min(N_TARGET, len(r_all)), replace=False))   # same beats in every condition
    sig = load_signal(r)
    for kind, snr, lab in conds:
        s_n = sig if kind == 'clean' else noisy(r, sig, kind, snr, lab)
        F = beat_features(s_n, bandpass(s_n), r_all, tgt)
        rows.append(pd.DataFrame(F, columns=FEATURES).assign(
            record=r, fold=rec2fold[r], kind=kind, snr=float(SNR_CLEAN_CAP if kind == 'clean' else snr),
            label=lab, r_idx=r_all[tgt], y=yc[tgt]))
    print('record', r, 'done', round(time.time() - t0), 's')
dev = pd.concat(rows, ignore_index=True)
dev.to_csv(OUT / 'dev_features.csv.gz', index=False)
inv = dev.groupby('kind')[FEATURES].apply(lambda x: x.isna().mean()).round(4)
print('rows', len(dev), '| fraction of invalid (NaN) feature values by condition family:'); print(inv.to_string())


# ===== CELL 4: how each single SQI responds to the injected SNR (validation noise B + clean) =====
def fc(v):                                              # fail closed: NaN/-inf -> just below the smallest finite value
    v = np.asarray(v, float); fin = np.isfinite(v)
    return np.where(fin, v, (v[fin].min() - 1.0) if fin.any() else 0.0)

def auc_sp(df, score):
    s = np.asarray(score, float); snr = df.snr.values
    m = (snr <= 6) | (snr >= 18)
    auc = roc_auc_score(snr[m] >= 18, fc(s[m]))         # bad (<= 6 dB) versus good (>= 18 dB or clean)
    fin = np.isfinite(s)
    sp = spearmanr(s[fin], snr[fin])[0]
    return auc, sp

Brows = dev[(dev.label == 'B') | (dev.label == 'clean')]
res = []
for f in FEATURES:
    for kind in NOISE_TYPES:
        sub = Brows[(Brows.kind == 'clean') | (Brows.kind == kind)]
        a, s_ = auc_sp(sub, rule_score(sub[FEATURES].values, f))
        res.append([f, kind, a, s_])
single = pd.DataFrame(res, columns=['feature', 'noise', 'AUROC_bad_vs_good', 'Spearman_vs_SNR'])
print(single.pivot(index='feature', columns='noise', values='AUROC_bad_vs_good').round(3).assign(
    mean=lambda t: t.mean(1)).to_string())
print(single.pivot(index='feature', columns='noise', values='Spearman_vs_SNR').round(3).to_string())
save_df('single_feature_response.csv', single)
# PRE-DECLARED selection of the rule gate: the single SQI with the highest mean AUROC over the three noise types (DS1 validation noise only)
RULE_FEATURE = single.groupby('feature').AUROC_bad_vs_good.mean().idxmax()
print('RULE GATE FEATURE (selected on DS1 development data only):', RULE_FEATURE)


# ===== CELL 5: fused gate (ridge on SNR): subject-grouped CV, and leave-one-noise-type-out =====
A_rows = dev[(dev.label == 'A') | (dev.label == 'clean')]            # training noise
def fit_on(df): return FusedSQI().fit(df[FEATURES].values, df.snr.values)

parts = []
for k in range(5):                                                   # train on other folds' A noise, validate on this fold's B noise
    g = fit_on(A_rows[A_rows.fold != k]); va = Brows[Brows.fold == k]
    parts.append(va.assign(fused=g.score(va[FEATURES].values)))
cv = pd.concat(parts)
res = []
for kind in NOISE_TYPES:
    sub = cv[(cv.kind == 'clean') | (cv.kind == kind)]
    res.append(['fused (record-grouped CV)', kind, *auc_sp(sub, sub.fused.values)])
    res.append([f'rule: {RULE_FEATURE}', kind, *auc_sp(sub, rule_score(sub[FEATURES].values, RULE_FEATURE))])
for h in NOISE_TYPES:                                                # held-out noise TYPE (never seen in training)
    ps = []
    for k in range(5):
        tr = A_rows[(A_rows.fold != k) & (A_rows.kind != h)]
        g = fit_on(tr); va = Brows[(Brows.fold == k) & ((Brows.kind == 'clean') | (Brows.kind == h))]
        ps.append(va.assign(fused=g.score(va[FEATURES].values)))
    sub = pd.concat(ps); res.append(['fused (held-out noise type)', h, *auc_sp(sub, sub.fused.values)])
cvtab = pd.DataFrame(res, columns=['gate', 'noise', 'AUROC_bad_vs_good', 'Spearman_vs_SNR'])
print(cvtab.pivot(index='gate', columns='noise', values='AUROC_bad_vs_good').round(3).to_string())
print(cvtab.pivot(index='gate', columns='noise', values='Spearman_vs_SNR').round(3).to_string())
save_df('gate_validation.csv', cvtab)
gate_final = fit_on(dev)                                             # final fit on ALL DS1 development rows (clean + A + B)
print('final fused gate, standardised coefficients (dB per SD):',
      dict(zip(FEATURES, np.round(gate_final.coef, 3))), '| intercept', round(gate_final.intercept, 2))


# ===== CELL 6: clean DS1, matched coverage: confidence gate versus SQI gates (what each gate rejects when there is no added noise) =====
F_clean = np.full((n, len(FEATURES)), np.nan, np.float32)
for r in sorted(set(rec.tolist())):
    g = tbl[tbl.record == r]; r_all = g.r_idx.values
    pos = np.where(rec == r)[0]; tg = np.searchsorted(r_all, ridx[pos]); assert np.array_equal(r_all[tg], ridx[pos])
    sig = load_signal(r); F_clean[pos] = beat_features(sig, bandpass(sig), r_all, tg)
print('invalid (NaN) feature rows on clean DS1:', round(float(np.isnan(F_clean).any(1).mean()), 4))

sc = {'confidence (calibrated MSP)': conf_cal, f'SQI rule ({RULE_FEATURE})': rule_score(F_clean, RULE_FEATURE)}
sc_f = np.empty(n)
for k in range(5):
    gk = fit_on(A_rows[A_rows.fold != k]); m = fold == k; sc_f[m] = gk.score(F_clean[m])   # held-out fold only
sc['SQI fused (fold-held-out)'] = sc_f

idx_rec = {r: np.where(rec == r)[0] for r in np.unique(rec)}
cls_n = np.bincount(y, minlength=4).astype(float)
def wr_null(acc, R=R_WR, seed=WR_SEED):
    """Keep each record's accepted count, choose WHICH beats at random inside the record."""
    rng = np.random.default_rng(seed); k_rec = {r: int(acc[ix].sum()) for r, ix in idx_rec.items()}
    cov = np.array([acc[y == q].mean() for q in range(4)])
    nc = np.empty((R, 4))
    for i in range(R):
        cnt = np.zeros(4)
        for r, ix in idx_rec.items():
            if k_rec[r]: cnt += np.bincount(y[rng.choice(ix, k_rec[r], replace=False)], minlength=4)
        nc[i] = cnt / cls_n
    gap = cov[:3].max() - cov[:3].min(); ng = nc[:, :3].max(1) - nc[:, :3].min(1)
    return cov, nc.mean(0), gap, ng.mean(), (1 + (ng >= gap).sum()) / (R + 1), (1 + (ng <= gap).sum()) / (R + 1)

rows, rows_pc = [], []
for gname, s in sc.items():
    for c in COV_LEVELS:
        acc = accept_at_coverage(s, c)
        cov, ncov, gap, ngap, p_up, p_lo = wr_null(acc)
        rows.append([gname, f'{int(c*100)}%', selective_risk(acc, correct), *cov, gap, ngap, p_up, p_lo])
        for q, cl in enumerate(C.CLASSES):
            rows_pc.append([gname, f'{int(c*100)}%', cl, cov[q], ncov[q], cov[q] - ncov[q]])
cmp_tab = pd.DataFrame(rows, columns=['gate', 'coverage', 'selective_risk', 'cov_N', 'cov_S', 'cov_V', 'cov_F',
                                      'gap_NSV', 'within_record_null_gap', 'p_obs_larger', 'p_obs_smaller'])
pc_tab = pd.DataFrame(rows_pc, columns=['gate', 'coverage', 'class', 'observed_cov', 'within_record_null_cov', 'obs_minus_null'])
aurc_tab = pd.DataFrame([[g_, aurc(s, correct)] for g_, s in sc.items()], columns=['gate', 'AURC_clean_DS1'])
print(aurc_tab.round(4).to_string(index=False)); print(cmp_tab.round(4).to_string(index=False))
print(pc_tab[pc_tab['class'].isin(['S', 'V', 'F'])].round(4).to_string(index=False))
print('F is descriptive only. Within-record null: conditional on these 22 subjects and each gate\'s own per-record counts; exploratory.')
save_df('clean_ds1_gate_comparison.csv', cmp_tab); save_df('clean_ds1_per_class_deviation.csv', pc_tab); save_df('clean_ds1_aurc.csv', aurc_tab)

# 2 x 2 agreement at 80% coverage between the confidence gate and the fused SQI gate, by true class
a_conf = accept_at_coverage(sc['confidence (calibrated MSP)'], 0.8); a_sqi = accept_at_coverage(sc['SQI fused (fold-held-out)'], 0.8)
dis = pd.DataFrame({'class': np.array(C.CLASSES)[y], 'conf': np.where(a_conf, 'accept', 'reject'), 'sqi': np.where(a_sqi, 'accept', 'reject')})
agree = dis.groupby(['class', 'conf', 'sqi']).size().unstack(['conf', 'sqi'], fill_value=0)
print(agree.to_string()); save_df('clean_ds1_agreement_2x2.csv', agree.reset_index())




# ===== CELL 7: lock file (written only after the choices are recorded in the project log) =====
LOCK = True        # set True ONLY after the decisions below are written into the log; DS2 must not be touched before this
lock = dict(
    created=datetime.datetime.now().isoformat(timespec='seconds'), sqi_source_sha256=Q.source_hash(),
    features=FEATURES, window_samples=Q.WIN, fs=Q.FS, min_beats=Q.MIN_BEATS, flat_eps=Q.FLAT_EPS, snr_clean_cap=SNR_CLEAN_CAP,
    invalid_policy='NaN features -> score -inf (rejected first)',
    noise=dict(types=NOISE_TYPES, snrs=SNRS, training_pool='dev blocks 0,4,8,...', validation_pool='dev blocks 2,6,10,...',
               reserved_for_ds2='test pool (odd blocks), never read here'),
    rule_gate=dict(feature=RULE_FEATURE, selection='highest mean AUROC (bad <= 6 dB vs good >= 18 dB or clean) over noise types, DS1 validation noise'),
    fused_gate=gate_final.to_dict(),
    operating_thresholds_clean_ds1={g_: {f'{int(c*100)}%': float(np.sort(s)[::-1][int(round(c * n)) - 1]) for c in COV_LEVELS}
                                    for g_, s in sc.items() if g_.startswith('SQI')},
    artifact=str(ARTIFACT), artifact_sha256=ARTIFACT_SHA256, ds2_data_used=False,
    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
if LOCK:
    p = OUT / 'sqi_gate_lock.json'; assert not p.exists()
    json.dump(lock, open(p, 'w'), indent=2)
    print('LOCKED:', p, '| sha256', sha256_of(p))
else:
    print('NOT LOCKED (LOCK = False). Review cells 4-6 and record the gate choices in the log first.')


from pathlib import Path
import json, hashlib

root = Path("/content/drive/MyDrive/FYP/data/processed")
lock_path = root / "sqi_dev_r5" / "sqi_gate_lock.json"

print("Lock exists:", lock_path.exists())

if lock_path.exists():
    lock = json.loads(lock_path.read_text())
    print("Lock SHA256:",
          hashlib.sha256(lock_path.read_bytes()).hexdigest())
    print("Noise metadata:")
    print(json.dumps(lock.get("noise", {}), indent=2))
    print("DS2 data used:", lock.get("ds2_data_used"))
    print("Rule gate:", lock.get("rule_gate"))
    print("Fused gate present:", "fused_gate" in lock)




# ================================================================================
# NOTEBOOK: 05_sqi_dev_cells_pre_C31_backup.ipynb
# ================================================================================

# =====================================================================================
# 05_sqi_dev.ipynb  --  SQI gate development on DS1 ONLY (no DS2 data is read anywhere in this notebook)
# Needs src/sqi.py and src/noise.py (the file was called noice.py; rename it). Nothing is trained except the
# low-capacity SQI regression. The classifier artifact is only read. Outputs go to a NEW versioned directory.
# =====================================================================================

# ===== CELL 1: setup, primary artifact (read only), versioned output directory =====
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, json, zlib, hashlib, platform, datetime, time
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, scipy, matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src import config as C
from src.data import load_beats, load_signal
from src.segment import bandpass
from src.noise import NOISE_TYPES, load_noise, CHUNK, BLOCK, chunk_starts
from src.calibration import fit_temperature
from src.selective import *
import src.sqi as Q
from src.sqi import FEATURES, beat_features, rule_score, FusedSQI, SNR_CLEAN_CAP

ARTIFACT = C.PROC / 'oof_ds1_v6_subjectfolds.npz'
ARTIFACT_SHA256 = '7138c8cf977be01aa1116ab78cab49da344413fd8ec5d8551846f39e37ab1e6c'
FOLDS = {0: [109, 118, 119, 115, 230], 1: [207, 106, 202, 201, 112, 122],
         2: [208, 124, 114, 116], 3: [203, 209, 101, 108], 4: [205, 215, 220, 223]}
EXPECTED_ENS_ACC = 0.782
SNRS = [24, 18, 12, 6, 0, -6]            # same grid as the planned DS2 evaluation
N_TARGET = 200                           # beats sampled per record and condition for the development table
COV_LEVELS = [0.95, 0.9, 0.8, 0.7]
R_WR, WR_SEED = 2000, 2031               # within-record null replicates (use 5000 for the final DS2 analysis)

sha256_of = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
assert sha256_of(ARTIFACT) == ARTIFACT_SHA256, 'primary artifact hash differs'
k_ver = 1
while (C.PROC / f'sqi_dev_r{k_ver}').exists():
    k_ver += 1
OUT = C.PROC / f'sqi_dev_r{k_ver}'; OUT.mkdir(parents=True, exist_ok=False)
print('ARTIFACT:', ARTIFACT, '\nOUTPUT  :', OUT)

def save_df(name, df, index=False):
    p = OUT / name; assert not p.exists(), p; df.to_csv(p, index=index); return p

d = np.load(ARTIFACT)
L, y, rec, ridx, fold = d['logits'], d['y'], d['rec'], d['ridx'], d['fold']
n = len(y); assert L.shape == (3, n, 4)
rec2fold = {r: k for k, rs in FOLDS.items() for r in rs}
assert set(rec2fold) == set(C.DS1_PRIMARY) and rec2fold[201] == rec2fold[202]
assert np.array_equal(fold, np.array([rec2fold[int(r)] for r in rec])), 'fold layout differs from section 12'
Lens = L.mean(0); correct = Lens.argmax(1) == y
assert abs(correct.mean() - EXPECTED_ENS_ACC) < 0.002, 'ensemble accuracy differs from the primary baseline'
T = fit_temperature(Lens, y); conf_cal = softmax(Lens / T, 1).max(1)       # calibrated MSP, as in notebook 03
tbl = load_beats(); tbl = tbl[tbl.split == 'DS1'].sort_values(['record', 'r_idx'])
print('ensemble acc', correct.mean().round(3), '| T', round(T, 3), '| beats', n)


# ===== CELL 2: development noise pools (A for training, B for validation) and SNR validation =====
# Both pools are subsets of the 'dev' pool of src/noise.py (even 60 s blocks): A = blocks 0, 4, 8, ...; B = blocks 2, 6, 10, ...
# so training noise and validation noise never share a segment, and neither touches the 'test' pool (odd blocks) reserved for DS2.
_ps = {}
def pool_starts(kind, label):
    if (kind, label) not in _ps:
        nz = load_noise(kind); st = []
        for b in range(len(nz) // BLOCK):
            if b % 4 == (0 if label == 'A' else 2):
                st += list(range(b * BLOCK, (b + 1) * BLOCK - CHUNK + 1, CHUNK))
        _ps[(kind, label)] = np.array(st)
    return _ps[(kind, label)]

def noisy(rec_id, sig, kind, snr, label):
    """Same SNR definition as src.noise.add_noise (record-level signal variance), but drawing chunks from pool A or B."""
    rng = np.random.default_rng(zlib.crc32(f'sqi-{kind}-{snr}-{rec_id}-{label}'.encode()))
    nz = load_noise(kind); st = pool_starts(kind, label)
    z = np.concatenate([nz[s:s + CHUNK] for s in rng.choice(st, int(np.ceil(len(sig) / CHUNK)), replace=True)])[:len(sig)]
    z = z - z.mean()
    return (sig + np.sqrt(np.var(sig) / (np.var(z) * 10 ** (snr / 10))) * z).astype(np.float32)

sig106 = load_signal(106)
for kind in NOISE_TYPES:
    A, B = set(pool_starts(kind, 'A')), set(pool_starts(kind, 'B'))
    test = set(chunk_starts(len(load_noise(kind)), 'test'))
    assert not (A & B) and not ((A | B) & test), f'noise pools overlap for {kind}'
    for snr in [24, 6, -6]:
        for lab in ['A', 'B']:
            nsy = noisy(106, sig106, kind, snr, lab)
            got = 10 * np.log10(np.var(sig106) / np.var(nsy - sig106))
            assert abs(got - snr) < 0.5, (kind, snr, lab, got)
    print(kind, 'chunks A/B/test:', len(A), len(B), len(test), '| achieved SNR within 0.5 dB, pools disjoint: OK')


# ===== CELL 3: SQI features on DS1 under clean and development-noise conditions (sampled beats) =====
conds = [('clean', None, 'clean')] + [(k, s, lab) for k in NOISE_TYPES for s in SNRS for lab in ('A', 'B')]
rows, t0 = [], time.time()
for r in sorted(tbl.record.unique()):
    g = tbl[tbl.record == r]; r_all, yc = g.r_idx.values, g.aami.map(C.CLS2IDX).values
    rr = np.random.default_rng(zlib.crc32(f'sqi-targets-{r}'.encode()))
    tgt = np.sort(rr.choice(len(r_all), size=min(N_TARGET, len(r_all)), replace=False))   # same beats in every condition
    sig = load_signal(r)
    for kind, snr, lab in conds:
        s_n = sig if kind == 'clean' else noisy(r, sig, kind, snr, lab)
        F = beat_features(s_n, bandpass(s_n), r_all, tgt)
        rows.append(pd.DataFrame(F, columns=FEATURES).assign(
            record=r, fold=rec2fold[r], kind=kind, snr=float(SNR_CLEAN_CAP if kind == 'clean' else snr),
            label=lab, r_idx=r_all[tgt], y=yc[tgt]))
    print('record', r, 'done', round(time.time() - t0), 's')
dev = pd.concat(rows, ignore_index=True)
dev.to_csv(OUT / 'dev_features.csv.gz', index=False)
inv = dev.groupby('kind')[FEATURES].apply(lambda x: x.isna().mean()).round(4)
print('rows', len(dev), '| fraction of invalid (NaN) feature values by condition family:'); print(inv.to_string())


# ===== CELL 4: how each single SQI responds to the injected SNR (validation noise B + clean) =====
def fc(v):                                              # fail closed: NaN/-inf -> just below the smallest finite value
    v = np.asarray(v, float); fin = np.isfinite(v)
    return np.where(fin, v, (v[fin].min() - 1.0) if fin.any() else 0.0)

def auc_sp(df, score):
    s = np.asarray(score, float); snr = df.snr.values
    m = (snr <= 6) | (snr >= 18)
    auc = roc_auc_score(snr[m] >= 18, fc(s[m]))         # bad (<= 6 dB) versus good (>= 18 dB or clean)
    fin = np.isfinite(s)
    sp = spearmanr(s[fin], snr[fin])[0]
    return auc, sp

Brows = dev[(dev.label == 'B') | (dev.label == 'clean')]
res = []
for f in FEATURES:
    for kind in NOISE_TYPES:
        sub = Brows[(Brows.kind == 'clean') | (Brows.kind == kind)]
        a, s_ = auc_sp(sub, rule_score(sub[FEATURES].values, f))
        res.append([f, kind, a, s_])
single = pd.DataFrame(res, columns=['feature', 'noise', 'AUROC_bad_vs_good', 'Spearman_vs_SNR'])
print(single.pivot(index='feature', columns='noise', values='AUROC_bad_vs_good').round(3).assign(
    mean=lambda t: t.mean(1)).to_string())
print(single.pivot(index='feature', columns='noise', values='Spearman_vs_SNR').round(3).to_string())
save_df('single_feature_response.csv', single)
# PRE-DECLARED selection of the rule gate: the single SQI with the highest mean AUROC over the three noise types (DS1 validation noise only)
RULE_FEATURE = single.groupby('feature').AUROC_bad_vs_good.mean().idxmax()
print('RULE GATE FEATURE (selected on DS1 development data only):', RULE_FEATURE)


# ===== CELL 5: fused gate (ridge on SNR): subject-grouped CV, and leave-one-noise-type-out =====
A_rows = dev[(dev.label == 'A') | (dev.label == 'clean')]            # training noise
def fit_on(df): return FusedSQI().fit(df[FEATURES].values, df.snr.values)

parts = []
for k in range(5):                                                   # train on other folds' A noise, validate on this fold's B noise
    g = fit_on(A_rows[A_rows.fold != k]); va = Brows[Brows.fold == k]
    parts.append(va.assign(fused=g.score(va[FEATURES].values)))
cv = pd.concat(parts)
res = []
for kind in NOISE_TYPES:
    sub = cv[(cv.kind == 'clean') | (cv.kind == kind)]
    res.append(['fused (record-grouped CV)', kind, *auc_sp(sub, sub.fused.values)])
    res.append([f'rule: {RULE_FEATURE}', kind, *auc_sp(sub, rule_score(sub[FEATURES].values, RULE_FEATURE))])
for h in NOISE_TYPES:                                                # held-out noise TYPE (never seen in training)
    ps = []
    for k in range(5):
        tr = A_rows[(A_rows.fold != k) & (A_rows.kind != h)]
        g = fit_on(tr); va = Brows[(Brows.fold == k) & ((Brows.kind == 'clean') | (Brows.kind == h))]
        ps.append(va.assign(fused=g.score(va[FEATURES].values)))
    sub = pd.concat(ps); res.append(['fused (held-out noise type)', h, *auc_sp(sub, sub.fused.values)])
cvtab = pd.DataFrame(res, columns=['gate', 'noise', 'AUROC_bad_vs_good', 'Spearman_vs_SNR'])
print(cvtab.pivot(index='gate', columns='noise', values='AUROC_bad_vs_good').round(3).to_string())
print(cvtab.pivot(index='gate', columns='noise', values='Spearman_vs_SNR').round(3).to_string())
save_df('gate_validation.csv', cvtab)
gate_final = fit_on(dev)                                             # final fit on ALL DS1 development rows (clean + A + B)
print('final fused gate, standardised coefficients (dB per SD):',
      dict(zip(FEATURES, np.round(gate_final.coef, 3))), '| intercept', round(gate_final.intercept, 2))


# ===== CELL 6: clean DS1, matched coverage: confidence gate versus SQI gates (what each gate rejects when there is no added noise) =====
F_clean = np.full((n, len(FEATURES)), np.nan, np.float32)
for r in sorted(set(rec.tolist())):
    g = tbl[tbl.record == r]; r_all = g.r_idx.values
    pos = np.where(rec == r)[0]; tg = np.searchsorted(r_all, ridx[pos]); assert np.array_equal(r_all[tg], ridx[pos])
    sig = load_signal(r); F_clean[pos] = beat_features(sig, bandpass(sig), r_all, tg)
print('invalid (NaN) feature rows on clean DS1:', round(float(np.isnan(F_clean).any(1).mean()), 4))

sc = {'confidence (calibrated MSP)': conf_cal, f'SQI rule ({RULE_FEATURE})': rule_score(F_clean, RULE_FEATURE)}
sc_f = np.empty(n)
for k in range(5):
    gk = fit_on(A_rows[A_rows.fold != k]); m = fold == k; sc_f[m] = gk.score(F_clean[m])   # held-out fold only
sc['SQI fused (fold-held-out)'] = sc_f

idx_rec = {r: np.where(rec == r)[0] for r in np.unique(rec)}
cls_n = np.bincount(y, minlength=4).astype(float)
def wr_null(acc, R=R_WR, seed=WR_SEED):
    """Keep each record's accepted count, choose WHICH beats at random inside the record."""
    rng = np.random.default_rng(seed); k_rec = {r: int(acc[ix].sum()) for r, ix in idx_rec.items()}
    cov = np.array([acc[y == q].mean() for q in range(4)])
    nc = np.empty((R, 4))
    for i in range(R):
        cnt = np.zeros(4)
        for r, ix in idx_rec.items():
            if k_rec[r]: cnt += np.bincount(y[rng.choice(ix, k_rec[r], replace=False)], minlength=4)
        nc[i] = cnt / cls_n
    gap = cov[:3].max() - cov[:3].min(); ng = nc[:, :3].max(1) - nc[:, :3].min(1)
    return cov, nc.mean(0), gap, ng.mean(), (1 + (ng >= gap).sum()) / (R + 1), (1 + (ng <= gap).sum()) / (R + 1)

rows, rows_pc = [], []
for gname, s in sc.items():
    for c in COV_LEVELS:
        acc = accept_at_coverage(s, c)
        cov, ncov, gap, ngap, p_up, p_lo = wr_null(acc)
        rows.append([gname, f'{int(c*100)}%', selective_risk(acc, correct), *cov, gap, ngap, p_up, p_lo])
        for q, cl in enumerate(C.CLASSES):
            rows_pc.append([gname, f'{int(c*100)}%', cl, cov[q], ncov[q], cov[q] - ncov[q]])
cmp_tab = pd.DataFrame(rows, columns=['gate', 'coverage', 'selective_risk', 'cov_N', 'cov_S', 'cov_V', 'cov_F',
                                      'gap_NSV', 'within_record_null_gap', 'p_obs_larger', 'p_obs_smaller'])
pc_tab = pd.DataFrame(rows_pc, columns=['gate', 'coverage', 'class', 'observed_cov', 'within_record_null_cov', 'obs_minus_null'])
aurc_tab = pd.DataFrame([[g_, aurc(s, correct)] for g_, s in sc.items()], columns=['gate', 'AURC_clean_DS1'])
print(aurc_tab.round(4).to_string(index=False)); print(cmp_tab.round(4).to_string(index=False))
print(pc_tab[pc_tab['class'].isin(['S', 'V', 'F'])].round(4).to_string(index=False))
print('F is descriptive only. Within-record null: conditional on these 22 subjects and each gate\'s own per-record counts; exploratory.')
save_df('clean_ds1_gate_comparison.csv', cmp_tab); save_df('clean_ds1_per_class_deviation.csv', pc_tab); save_df('clean_ds1_aurc.csv', aurc_tab)

# 2 x 2 agreement at 80% coverage between the confidence gate and the fused SQI gate, by true class
a_conf = accept_at_coverage(sc['confidence (calibrated MSP)'], 0.8); a_sqi = accept_at_coverage(sc['SQI fused (fold-held-out)'], 0.8)
dis = pd.DataFrame({'class': np.array(C.CLASSES)[y], 'conf': np.where(a_conf, 'accept', 'reject'), 'sqi': np.where(a_sqi, 'accept', 'reject')})
agree = dis.groupby(['class', 'conf', 'sqi']).size().unstack(['conf', 'sqi'], fill_value=0)
print(agree.to_string()); save_df('clean_ds1_agreement_2x2.csv', agree.reset_index())




# ===== CELL 7: lock file (written only after the choices are recorded in the project log) =====
LOCK = True        # set True ONLY after the decisions below are written into the log; DS2 must not be touched before this
lock = dict(
    created=datetime.datetime.now().isoformat(timespec='seconds'), sqi_source_sha256=Q.source_hash(),
    features=FEATURES, window_samples=Q.WIN, fs=Q.FS, min_beats=Q.MIN_BEATS, flat_eps=Q.FLAT_EPS, snr_clean_cap=SNR_CLEAN_CAP,
    invalid_policy='NaN features -> score -inf (rejected first)',
    noise=dict(types=NOISE_TYPES, snrs=SNRS, training_pool='dev blocks 0,4,8,...', validation_pool='dev blocks 2,6,10,...',
               reserved_for_ds2='test pool (odd blocks), never read here'),
    rule_gate=dict(feature=RULE_FEATURE, selection='highest mean AUROC (bad <= 6 dB vs good >= 18 dB or clean) over noise types, DS1 validation noise'),
    fused_gate=gate_final.to_dict(),
    operating_thresholds_clean_ds1={g_: {f'{int(c*100)}%': float(np.sort(s)[::-1][int(round(c * n)) - 1]) for c in COV_LEVELS}
                                    for g_, s in sc.items() if g_.startswith('SQI')},
    artifact=str(ARTIFACT), artifact_sha256=ARTIFACT_SHA256, ds2_data_used=False,
    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
if LOCK:
    p = OUT / 'sqi_gate_lock.json'; assert not p.exists()
    json.dump(lock, open(p, 'w'), indent=2)
    print('LOCKED:', p, '| sha256', sha256_of(p))
else:
    print('NOT LOCKED (LOCK = False). Review cells 4-6 and record the gate choices in the log first.')


##  SAMPLE EXPERIMENTS


import inspect
from src import selective

print(inspect.getsource(selective.accept_at_coverage))




# ================================================================================
# NOTEBOOK: 06_classifier_noise_sensitivity_cells.ipynb
# ================================================================================

# =====================================================================================
# 06_classifier_noise_sensitivity.ipynb  --  DIAGNOSTIC ONLY (in-sample): how do the saved FINAL models react to injected noise on DS1?
# The final models (cnn_ds1_seed42/43/44.pt) were trained on ALL of DS1, so every number here is IN-SAMPLE. Use it to check the
# noise pipeline and to see which noise types change the classifier's predictions; do NOT use it to claim robustness on unseen
# patients and do NOT use it to pick a gate on its own. DS2 and the test noise pool are not read.
# Uses the same preprocessing as training: src.segment.build_dataset(..., signal_fn=<noise>) injects noise BEFORE the 0.5-40 Hz band-pass.
# =====================================================================================

# ===== CELL 1: setup =====
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, json, zlib, hashlib, time
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from scipy.special import softmax
from src import config as C
from src.data import load_beats, load_signal
from src.noise import NOISE_TYPES, load_noise, CHUNK, BLOCK, chunk_starts
from src.segment import build_dataset
from src.model import BeatCNN
from src.train2 import get_logits, DEVICE

ARTIFACT = C.PROC / 'oof_ds1_v6_subjectfolds.npz'            # used only to check beat order; no OOF number is used
ARTIFACT_SHA256 = '7138c8cf977be01aa1116ab78cab49da344413fd8ec5d8551846f39e37ab1e6c'
SEEDS, SNRS = [42, 43, 44], [24, 18, 12, 6, 0, -6]
sha256_of = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
assert sha256_of(ARTIFACT) == ARTIFACT_SHA256
k_ver = 1
while (C.PROC / f'noise_sensitivity_r{k_ver}').exists():
    k_ver += 1
OUT = C.PROC / f'noise_sensitivity_r{k_ver}'; OUT.mkdir(parents=True, exist_ok=False)
print('OUTPUT:', OUT, '| device', DEVICE)

d = np.load(ARTIFACT); y_art, rec_art, ridx_art = d['y'], d['rec'], d['ridx']
tbl = load_beats(); d1 = tbl[tbl.split == 'DS1']
X0, RR0, y, rec, ridx = build_dataset(d1)                     # clean DS1, identical to training preprocessing
assert np.array_equal(y, y_art) and np.array_equal(rec, rec_art) and np.array_equal(ridx, ridx_art), 'beat order differs from the artifact'
print('clean dataset', X0.shape, RR0.shape, '| order identical to the artifact')

models = []
for sd in SEEDS:
    m = BeatCNN().to(DEVICE)
    m.load_state_dict(torch.load(C.PROC / f'cnn_ds1_seed{sd}.pt', map_location=DEVICE)); m.eval(); models.append(m)
def ens_logits(X, RR): return np.mean([get_logits(m, X, RR) for m in models], axis=0)

L0 = ens_logits(X0, RR0); pred0 = L0.argmax(1); ok0 = pred0 == y
print('clean in-sample accuracy (3-model ensemble):', ok0.mean().round(4))
assert ok0.mean() > 0.9, 'in-sample clean accuracy is unexpectedly low: wrong checkpoints or preprocessing?'


# ===== CELL 2: development noise (C31 pool B) =====
def pool_starts(kind, label):
    """C31: A is the first dev region; B is the second dev region."""
    assert label in ("A", "B"), f"Invalid development pool: {label!r}"
    pool = "devA" if label == "A" else "devB"
    return chunk_starts(len(load_noise(kind)), pool)
def noisy(rec_id, sig, kind, snr, label):
    rng = np.random.default_rng(zlib.crc32(f'sqi-{kind}-{snr}-{rec_id}-{label}'.encode()))
    nz = load_noise(kind); st = pool_starts(kind, label)
    z = np.concatenate([nz[s:s + CHUNK] for s in rng.choice(st, int(np.ceil(len(sig) / CHUNK)), replace=True)])[:len(sig)]
    z = z - z.mean()
    return (sig + np.sqrt(np.var(sig) / (np.var(z) * 10 ** (snr / 10))) * z).astype(np.float32)

sig106 = load_signal(106)
for kind in NOISE_TYPES:
    A, B = set(pool_starts(kind, 'A')), set(pool_starts(kind, 'B')); test = set(chunk_starts(len(load_noise(kind)), 'test'))
    assert not (A & B) and not ((A | B) & test)
    for snr in [24, -6]:
        got = 10 * np.log10(np.var(sig106) / np.var(noisy(106, sig106, kind, snr, 'B') - sig106)); assert abs(got - snr) < 0.5
print('noise pools disjoint from each other and from the test pool; achieved SNR within 0.5 dB: OK')


# ===== CELL 3: predictions under noise (all DS1 beats, 3-model ensemble) =====
rows, t0 = [], time.time()
rows.append(dict(kind='clean', snr=np.nan, n_beats=len(y), n_records=len(np.unique(rec)), acc=ok0.mean(),
                 **{f'recall_{c}': (pred0[y == i] == i).mean() for i, c in enumerate(C.CLASSES)},
                 changed=0.0, flip_to_wrong=0.0, mean_msp=softmax(L0, 1).max(1).mean()))
for kind in NOISE_TYPES:
    for snr in SNRS:
        X, RR, y2, rec2, ridx2 = build_dataset(d1, signal_fn=lambda r, sg, k=kind, s=snr: noisy(r, sg, k, s, 'B'))
        assert np.array_equal(y2, y) and np.array_equal(ridx2, ridx)
        Ln = ens_logits(X, RR); pn = Ln.argmax(1); okn = pn == y
        rows.append(dict(kind=kind, snr=snr, n_beats=len(y), n_records=len(np.unique(rec)), acc=okn.mean(),
                         **{f'recall_{c}': (pn[y == i] == i).mean() for i, c in enumerate(C.CLASSES)},
                         changed=(pn != pred0).mean(),                         # any change in the predicted class
                         flip_to_wrong=(~okn[ok0]).mean(),                     # beats correct on clean that become wrong
                         mean_msp=softmax(Ln, 1).max(1).mean()))
        print(kind, snr, 'acc', round(okn.mean(), 4), '| changed', round((pn != pred0).mean(), 4), '|', round(time.time() - t0), 's')
res = pd.DataFrame(rows)
res.to_csv(OUT / 'noise_sensitivity_final_models_in_sample.csv', index=False)
print(res.round(4).to_string(index=False))
print('IN-SAMPLE diagnostic (final models trained on all of DS1): likely understates sensitivity on unseen patients.')
print('"changed" = predicted class differs from the clean prediction; "flip_to_wrong" = share of clean-correct beats that become wrong.')


# ===== CELL 4: figure =====
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for ax, col, ttl in zip(axs, ['acc', 'changed', 'flip_to_wrong'], ['accuracy', 'predictions changed vs clean', 'clean-correct beats that become wrong']):
    for kind in NOISE_TYPES:
        s = res[res.kind == kind].sort_values('snr'); ax.plot(s.snr, s[col], marker='o', label=kind)
    ax.set_xlabel('injected SNR (dB, before filtering)'); ax.set_title(ttl, fontsize=9)
axs[0].axhline(ok0.mean(), color='grey', ls=':'); axs[0].legend(fontsize=7); plt.tight_layout()
p = OUT / 'noise_sensitivity.png'; assert not p.exists(); fig.savefig(p, dpi=200, bbox_inches='tight')
json.dump(dict(models=[f'cnn_ds1_seed{s}.pt' for s in SEEDS], artifact_sha256=ARTIFACT_SHA256, noise_pool='B (development)', in_sample=True,
               snrs=SNRS), open(OUT / 'manifest.json', 'w'), indent=2)
print('saved to', OUT)



# ================================================================================
# NOTEBOOK: 06_classifier_noise_sensitivity_cells_pre_C31_backup.ipynb
# ================================================================================

# =====================================================================================
# 06_classifier_noise_sensitivity.ipynb  --  DIAGNOSTIC ONLY (in-sample): how do the saved FINAL models react to injected noise on DS1?
# The final models (cnn_ds1_seed42/43/44.pt) were trained on ALL of DS1, so every number here is IN-SAMPLE. Use it to check the
# noise pipeline and to see which noise types change the classifier's predictions; do NOT use it to claim robustness on unseen
# patients and do NOT use it to pick a gate on its own. DS2 and the test noise pool are not read.
# Uses the same preprocessing as training: src.segment.build_dataset(..., signal_fn=<noise>) injects noise BEFORE the 0.5-40 Hz band-pass.
# =====================================================================================

# ===== CELL 1: setup =====
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wfdb
import os, sys, json, zlib, hashlib, time
os.chdir('/content/drive/MyDrive/FYP'); sys.path.insert(0, os.getcwd())

import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from scipy.special import softmax
from src import config as C
from src.data import load_beats, load_signal
from src.noise import NOISE_TYPES, load_noise, CHUNK, BLOCK, chunk_starts
from src.segment import build_dataset
from src.model import BeatCNN
from src.train2 import get_logits, DEVICE

ARTIFACT = C.PROC / 'oof_ds1_v6_subjectfolds.npz'            # used only to check beat order; no OOF number is used
ARTIFACT_SHA256 = '7138c8cf977be01aa1116ab78cab49da344413fd8ec5d8551846f39e37ab1e6c'
SEEDS, SNRS = [42, 43, 44], [24, 18, 12, 6, 0, -6]
sha256_of = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
assert sha256_of(ARTIFACT) == ARTIFACT_SHA256
k_ver = 1
while (C.PROC / f'noise_sensitivity_r{k_ver}').exists():
    k_ver += 1
OUT = C.PROC / f'noise_sensitivity_r{k_ver}'; OUT.mkdir(parents=True, exist_ok=False)
print('OUTPUT:', OUT, '| device', DEVICE)

d = np.load(ARTIFACT); y_art, rec_art, ridx_art = d['y'], d['rec'], d['ridx']
tbl = load_beats(); d1 = tbl[tbl.split == 'DS1']
X0, RR0, y, rec, ridx = build_dataset(d1)                     # clean DS1, identical to training preprocessing
assert np.array_equal(y, y_art) and np.array_equal(rec, rec_art) and np.array_equal(ridx, ridx_art), 'beat order differs from the artifact'
print('clean dataset', X0.shape, RR0.shape, '| order identical to the artifact')

models = []
for sd in SEEDS:
    m = BeatCNN().to(DEVICE)
    m.load_state_dict(torch.load(C.PROC / f'cnn_ds1_seed{sd}.pt', map_location=DEVICE)); m.eval(); models.append(m)
def ens_logits(X, RR): return np.mean([get_logits(m, X, RR) for m in models], axis=0)

L0 = ens_logits(X0, RR0); pred0 = L0.argmax(1); ok0 = pred0 == y
print('clean in-sample accuracy (3-model ensemble):', ok0.mean().round(4))
assert ok0.mean() > 0.9, 'in-sample clean accuracy is unexpectedly low: wrong checkpoints or preprocessing?'


# ===== CELL 2: development noise (pool B, same definition as notebook 05) =====
_ps = {}
def pool_starts(kind, label):
    if (kind, label) not in _ps:
        nz = load_noise(kind); st = []
        for b in range(len(nz) // BLOCK):
            if b % 4 == (0 if label == 'A' else 2):
                st += list(range(b * BLOCK, (b + 1) * BLOCK - CHUNK + 1, CHUNK))
        _ps[(kind, label)] = np.array(st)
    return _ps[(kind, label)]

def noisy(rec_id, sig, kind, snr, label):
    rng = np.random.default_rng(zlib.crc32(f'sqi-{kind}-{snr}-{rec_id}-{label}'.encode()))
    nz = load_noise(kind); st = pool_starts(kind, label)
    z = np.concatenate([nz[s:s + CHUNK] for s in rng.choice(st, int(np.ceil(len(sig) / CHUNK)), replace=True)])[:len(sig)]
    z = z - z.mean()
    return (sig + np.sqrt(np.var(sig) / (np.var(z) * 10 ** (snr / 10))) * z).astype(np.float32)

sig106 = load_signal(106)
for kind in NOISE_TYPES:
    A, B = set(pool_starts(kind, 'A')), set(pool_starts(kind, 'B')); test = set(chunk_starts(len(load_noise(kind)), 'test'))
    assert not (A & B) and not ((A | B) & test)
    for snr in [24, -6]:
        got = 10 * np.log10(np.var(sig106) / np.var(noisy(106, sig106, kind, snr, 'B') - sig106)); assert abs(got - snr) < 0.5
print('noise pools disjoint from each other and from the test pool; achieved SNR within 0.5 dB: OK')


# ===== CELL 3: predictions under noise (all DS1 beats, 3-model ensemble) =====
rows, t0 = [], time.time()
rows.append(dict(kind='clean', snr=np.nan, n_beats=len(y), n_records=len(np.unique(rec)), acc=ok0.mean(),
                 **{f'recall_{c}': (pred0[y == i] == i).mean() for i, c in enumerate(C.CLASSES)},
                 changed=0.0, flip_to_wrong=0.0, mean_msp=softmax(L0, 1).max(1).mean()))
for kind in NOISE_TYPES:
    for snr in SNRS:
        X, RR, y2, rec2, ridx2 = build_dataset(d1, signal_fn=lambda r, sg, k=kind, s=snr: noisy(r, sg, k, s, 'B'))
        assert np.array_equal(y2, y) and np.array_equal(ridx2, ridx)
        Ln = ens_logits(X, RR); pn = Ln.argmax(1); okn = pn == y
        rows.append(dict(kind=kind, snr=snr, n_beats=len(y), n_records=len(np.unique(rec)), acc=okn.mean(),
                         **{f'recall_{c}': (pn[y == i] == i).mean() for i, c in enumerate(C.CLASSES)},
                         changed=(pn != pred0).mean(),                         # any change in the predicted class
                         flip_to_wrong=(~okn[ok0]).mean(),                     # beats correct on clean that become wrong
                         mean_msp=softmax(Ln, 1).max(1).mean()))
        print(kind, snr, 'acc', round(okn.mean(), 4), '| changed', round((pn != pred0).mean(), 4), '|', round(time.time() - t0), 's')
res = pd.DataFrame(rows)
res.to_csv(OUT / 'noise_sensitivity_final_models_in_sample.csv', index=False)
print(res.round(4).to_string(index=False))
print('IN-SAMPLE diagnostic (final models trained on all of DS1): likely understates sensitivity on unseen patients.')
print('"changed" = predicted class differs from the clean prediction; "flip_to_wrong" = share of clean-correct beats that become wrong.')


# ===== CELL 4: figure =====
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for ax, col, ttl in zip(axs, ['acc', 'changed', 'flip_to_wrong'], ['accuracy', 'predictions changed vs clean', 'clean-correct beats that become wrong']):
    for kind in NOISE_TYPES:
        s = res[res.kind == kind].sort_values('snr'); ax.plot(s.snr, s[col], marker='o', label=kind)
    ax.set_xlabel('injected SNR (dB, before filtering)'); ax.set_title(ttl, fontsize=9)
axs[0].axhline(ok0.mean(), color='grey', ls=':'); axs[0].legend(fontsize=7); plt.tight_layout()
p = OUT / 'noise_sensitivity.png'; assert not p.exists(); fig.savefig(p, dpi=200, bbox_inches='tight')
json.dump(dict(models=[f'cnn_ds1_seed{s}.pt' for s in SEEDS], artifact_sha256=ARTIFACT_SHA256, noise_pool='B (development)', in_sample=True,
               snrs=SNRS), open(OUT / 'manifest.json', 'w'), indent=2)
print('saved to', OUT)


