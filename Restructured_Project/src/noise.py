import zlib
import numpy as np
import wfdb
from . import config as C

NST = C.ROOT / 'data' / 'raw' / 'nstdb'
NOISE_TYPES = ['bw', 'em', 'ma']
BLOCK, CHUNK = 60 * C.FS, 10 * C.FS
_cache = {}

# C31: first 45% development, middle 10% guard, final 45% test.
DEV_FRAC, GAP_FRAC = 0.45, 0.10

def regions(n_samples):
    """Return half-open sample ranges for devA, devB, gap and test."""
    assert n_samples > 0, n_samples
    dev_end = int(round(DEV_FRAC * n_samples))
    test_start = n_samples - dev_end
    half = dev_end // 2
    return {
        'devA': (0, half),
        'devB': (half, dev_end),
        'gap': (dev_end, test_start),
        'test': (test_start, n_samples),
    }

def load_noise(kind):
    if kind not in NOISE_TYPES:
        raise ValueError(f'Unknown noise type: {kind}')
    if kind not in _cache:
        r = wfdb.rdrecord(str(NST / kind))
        assert r.fs == C.FS, r.fs
        _cache[kind] = r.p_signal[:, 0].astype(np.float32)
    return _cache[kind]

def _starts(lo, hi):
    """Starts of complete CHUNK-sized chunks contained in [lo, hi)."""
    if hi - lo < CHUNK:
        return np.array([], dtype=int)
    return np.arange(lo, hi - CHUNK + 1, CHUNK, dtype=int)

def chunk_starts(n_samples, pool):
    """
    C31 pools:
      devA = first contiguous half of the first 45%
      devB = second contiguous half of the first 45%
      dev   = devA + devB
      test  = final 45%; reserved for held-out evaluation
    The middle 10% is a guard gap. Chunks must fit entirely in a region.
    """
    reg = regions(n_samples)
    if pool == 'dev':
        return np.concatenate([
            _starts(*reg['devA']),
            _starts(*reg['devB'])
        ])
    if pool not in ('devA', 'devB', 'test'):
        raise ValueError(
            f"Invalid pool {pool!r}; use 'devA', 'devB', 'dev', or 'test'"
        )
    return _starts(*reg[pool])

def make_noise(kind, n, pool, rng):
    nz = load_noise(kind)
    starts = chunk_starts(len(nz), pool)
    if len(starts) == 0:
        raise ValueError(f'No complete noise chunks for {kind}/{pool}')
    pick = rng.choice(starts, int(np.ceil(n / CHUNK)), replace=True)
    return np.concatenate([nz[s:s + CHUNK] for s in pick])[:n]

def add_noise(sig, kind, snr_db, pool, seed):
    nz = make_noise(kind, len(sig), pool, np.random.default_rng(seed))
    nz = nz - nz.mean()
    denom = np.var(nz) * 10 ** (snr_db / 10)
    if denom <= 0:
        raise ValueError('Noise variance must be positive')
    a = np.sqrt(np.var(sig) / denom)
    return (sig + a * nz).astype(np.float32)

def noise_fn(kind, snr_db, pool):
    """Return a deterministic signal function for the requested noise pool."""
    if kind not in NOISE_TYPES:
        raise ValueError(f'Unknown noise type: {kind}')
    if pool not in ('devA', 'devB', 'dev', 'test'):
        raise ValueError(f'Invalid pool: {pool}')
    def fn(rec, sig):
        seed = zlib.crc32(f'{kind}-{snr_db}-{rec}-{pool}'.encode())
        return add_noise(sig, kind, snr_db, pool, seed)
    return fn
