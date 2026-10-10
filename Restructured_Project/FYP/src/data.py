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