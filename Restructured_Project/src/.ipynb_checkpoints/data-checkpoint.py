import numpy as np, pandas as pd, wfdb
from pathlib import Path
import wfdb
wfdb.dl_database('mitdb', 'data/raw/mitdb')
# 4 paced records (102, 104, 107, 217) excluded.
# Primary experiment: 202 moved to the DS1 side because 201 and 202
# came from the same subject. Keep the original split as a secondary run.
DS1_ORIG = [101,106,108,109,112,114,115,116,118,119,122,124,
            201,203,205,207,208,209,215,220,223,230]
DS2_ORIG = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,
            219,221,222,228,231,232,233,234]

def get_split(primary=True):
    if not primary:
        return DS1_ORIG, DS2_ORIG
    return DS1_ORIG + [202], [r for r in DS2_ORIG if r != 202]

# AAMI mapping (Q is kept as its own label so you can decide later)
AAMI = {}
for s in ['N','L','R','e','j']: AAMI[s] = 'N'
for s in ['A','a','J','S']:     AAMI[s] = 'S'
for s in ['V','E']:             AAMI[s] = 'V'
AAMI['F'] = 'F'
for s in ['/','f','Q']:         AAMI[s] = 'Q'
# every other annotation symbol is a non-beat marker and is ignored

def download_and_cache(records, out='data/signals'):
    Path(out).mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        r = wfdb.rdrecord(str(rec), pn_dir='mitdb')
        a = wfdb.rdann(str(rec), 'atr', pn_dir='mitdb')
        # pick MLII by name (record 114 has the leads swapped)
        ch = r.sig_name.index('MLII')
        np.save(f'{out}/{rec}.npy', r.p_signal[:, ch].astype(np.float32))
        for samp, sym in zip(a.sample, a.symbol):
            if sym in AAMI:
                rows.append((rec, int(samp), sym, AAMI[sym]))
    df = pd.DataFrame(rows, columns=['record','r_idx','symbol','aami'])
    df.to_csv('data/beats.csv', index=False)
    return df

if __name__ == '__main__':
    ds1, ds2 = get_split(primary=True)
    df = download_and_cache(ds1 + ds2)
    df['split'] = np.where(df.record.isin(ds1), 'DS1', 'DS2')
    print(pd.crosstab(df.split, df.aami))
    print(df.groupby('split').record.nunique())