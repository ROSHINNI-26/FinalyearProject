import zlib
import numpy as np, wfdb
from . import config as C

NST = C.ROOT / 'data' / 'raw' / 'nstdb'
NOISE_TYPES = ['bw', 'em', 'ma']
BLOCK, CHUNK = 60 * C.FS, 10 * C.FS
_cache = {}

def load_noise(kind):
    if kind not in _cache:
        r = wfdb.rdrecord(str(NST / kind))
        assert r.fs == C.FS, r.fs
        _cache[kind] = r.p_signal[:, 0].astype(np.float32)
    return _cache[kind]

def chunk_starts(n_samples, pool):
    """pool 'dev' = even 60 s blocks, 'test' = odd blocks (disjoint)."""
    starts = []
    for b in range(n_samples // BLOCK):
        if (b % 2 == 0) == (pool == 'dev'):
            starts += list(range(b * BLOCK, (b + 1) * BLOCK - CHUNK + 1, CHUNK))
    return np.array(starts)

def make_noise(kind, n, pool, rng):
    nz = load_noise(kind)
    starts = chunk_starts(len(nz), pool)
    pick = rng.choice(starts, int(np.ceil(n / CHUNK)), replace=True)
    return np.concatenate([nz[s:s + CHUNK] for s in pick])[:n]

def add_noise(sig, kind, snr_db, pool, seed):
    nz = make_noise(kind, len(sig), pool, np.random.default_rng(seed))
    nz = nz - nz.mean()
    a = np.sqrt(np.var(sig) / (np.var(nz) * 10 ** (snr_db / 10)))
    return (sig + a * nz).astype(np.float32)

def noise_fn(kind, snr_db, pool):
    """Plug into build_dataset(..., signal_fn=noise_fn('em', 6, 'dev'))."""
    def fn(rec, sig):
        seed = zlib.crc32(f'{kind}-{snr_db}-{rec}-{pool}'.encode())
        return add_noise(sig, kind, snr_db, pool, seed)
    return fn