"""All twelve recordings in one place: loading, weak labels, per-file scaling.

Files (all 20 kHz, columns X feed, Y normal-to-feed, Z axial):
  A  Oct 9   data/raw/Acce_{20k,60k,100k}_1_r1.lvm        (training runs)
  B  Oct 15  data/raw/test/Acce_{20k,60k,100k}_3_r1.lvm   (test runs)
  C  Oct 19  six scope recordings (4 inputs; the 4th has nothing plugged in)

The Oct 19 spindle speeds are not in the file names. They were read from the
spindle line in each file's spectrum (see FILES; checked in 08 step 0).

Labels come from labels.make_labels (the entropy rule), which needs the
spindle speed. The models never get the speed.
"""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pipeline import denoise as dn
from pipeline import labels as lb

DATA = Path(__file__).resolve().parents[1] / 'data'
CACHE = DATA / 'cache'
WIN, HOP, FS = lb.WIN, lb.HOP, lb.FS

# key: (path inside data/, rpm, session)
FILES = {
    'A20k':   ('raw/Acce_20k_1_r1.lvm', 20_000, 'A'),
    'A60k':   ('raw/Acce_60k_1_r1.lvm', 60_000, 'A'),
    'A100k':  ('raw/Acce_100k_1_r1.lvm', 100_000, 'A'),
    'B20k':   ('raw/test/Acce_20k_3_r1.lvm', 20_000, 'B'),
    'B60k':   ('raw/test/Acce_60k_3_r1.lvm', 60_000, 'B'),
    'B100k':  ('raw/test/Acce_100k_3_r1.lvm', 100_000, 'B'),
    'C2316':  ('recordings/scope_20261019_022316.lvm', 20_000, 'C'),
    'C3037':  ('recordings/scope_20261019_023037.lvm', 20_000, 'C'),
    'C3210':  ('recordings/scope_20261019_023210.lvm', 20_000, 'C'),
    'C3328':  ('recordings/scope_20261019_023328.lvm', 60_000, 'C'),
    'C3440':  ('recordings/scope_20261019_023440.lvm', 60_000, 'C'),
    'C3504':  ('recordings/scope_20261019_023504.lvm', 100_000, 'C'),
}


def raw(key):
    """(n, 3) raw X, Y, Z in volts. Parsed once, then read from data/cache/."""
    CACHE.mkdir(exist_ok=True)
    npy = CACHE / f'{key}_raw.npy'
    if not npy.exists():
        x = np.loadtxt(DATA / FILES[key][0], dtype=np.float32)[:, 1:4]
        np.save(npy, x)
    return np.load(npy)


def load(key):
    """Windows of the denoised signal plus their weak labels.

    The signal is divided by the file's own no-cutting level (per axis), so the
    models see "times the idle background" as in 04-06. The idle level is the
    median RMS of the quietest 20 % of windows, the same reference labels.py
    uses, because the Oct 19 files do not all start idle.
    """
    npz = CACHE / f'{key}_win.npz'
    if npz.exists():
        return dict(np.load(npz))
    x = raw(key).astype(float)
    L = lb.make_labels(x, FILES[key][1])
    clean, _ = dn.denoise(x)
    W = sliding_window_view(clean, WIN, axis=0)[::HOP]          # (n, 3, WIN)
    rms = np.sqrt((W ** 2).mean(-1))                            # (n, 3)
    tot = np.sqrt((rms ** 2).sum(1))
    idle = rms[tot <= np.quantile(tot, 0.20)].mean(0)
    W = (W / idle[None, :, None]).astype(np.float32)
    assert len(W) == len(L['t'])
    d = dict(X=W, y=L['label'], t=L['t'], score=L['entropy_drop'],
             level_ratio=L['level_ratio'], idle=idle)
    np.savez(npz, **d)
    return d


def load_all():
    """Concatenate every file. Adds `file` (index into FILES) per window."""
    parts = []
    for i, k in enumerate(FILES):
        d = load(k)
        d['file'] = np.full(len(d['y']), i)
        parts.append(d)
    keys = ['X', 'y', 't', 'score', 'level_ratio', 'file']
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}
