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
        f"S-F1 drop <= {rule['MAX_SV_DROP']}": d_s >= -rule["MAX_SV_DROP"],
        f"V-F1 drop <= {rule['MAX_SV_DROP']}": d_v >= -rule["MAX_SV_DROP"],
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