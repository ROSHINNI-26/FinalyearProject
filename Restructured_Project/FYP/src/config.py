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