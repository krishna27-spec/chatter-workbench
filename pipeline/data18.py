"""The 18 runs of the severity study: 12 to train, 6 to test.

Files: data/raw/Acce_60k_1_r2/Acce_<rpm>_<feed per tooth>_r<run>.lvm (20 kHz, columns X feed, Y normal,
Z axial). Runs picked with random.Random(323): 2 of the 3 runs for each training condition,
and 2 of the 3 runs of the feed rate each speed never trains on as the test set.

  train  20k feed 1, 3   60k feed 1, 5   100k feed 3, 5
  test   20k feed 5      60k feed 3      100k feed 1

EXTRA holds the remaining run of each test condition, scored afterwards with the saved models.

Labels come from the Excel (excel_labels.py). Windows are cleaned with causal.py, the cleaning
the live stream uses, and divided by the file's idle RMS per axis, as export_model.py does.
"""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pipeline import causal
from pipeline import excel_labels as el
from pipeline import labels as lb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'Acce_60k_1_r2'
CACHE = ROOT / 'data' / 'cache' / 'v2'

TRAIN = ['20k_1_r1', '20k_1_r3', '20k_3_r1', '20k_3_r2', '60k_1_r1', '60k_1_r2',
         '60k_5_r2', '60k_5_r3', '100k_3_r2', '100k_3_r3', '100k_5_r1', '100k_5_r2']
TEST = ['20k_5_r1', '20k_5_r2', '60k_3_r2', '60k_3_r3', '100k_1_r1', '100k_1_r2']
EXTRA = ['20k_5_r3', '60k_3_r1', '100k_1_r3']      # the third run of each test condition


def rpm(run):
    return int(run.split('k_')[0]) * 1000


def raw(run):
    """(n, 3) raw X, Y, Z in volts. Parsed once, then read from data/cache/v2/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    npy = CACHE / f'{run}_raw.npy'
    if not npy.exists():
        np.save(npy, np.loadtxt(DATA / f'Acce_{run}.lvm', dtype=np.float32)[:, 1:4])
    return np.load(npy)


def idle_rms(W):
    """Per-axis idle level: mean RMS of the quietest 20 % of windows."""
    rms = np.sqrt((W ** 2).mean(-1))
    tot = np.sqrt((rms ** 2).sum(1))
    return rms[tot <= np.quantile(tot, 0.20)].mean(0)


def load(run):
    """Scaled windows X (n, 3, WIN), Excel labels y, window centres t, plus the level trace
    and the cut end time used for the label figures."""
    npz = CACHE / f'{run}_win.npz'
    if npz.exists():
        return dict(np.load(npz))
    x = raw(run).astype(float)
    L = lb.make_labels(x, rpm(run))
    y, t_end = el.window_labels(run, L['t'], L['cutting'], L['level_ratio'])
    clean = causal.clean_offline(x)
    W = sliding_window_view(clean, lb.WIN, axis=0)[::lb.HOP]
    n = min(len(W), len(y))
    W = (W[:n] / idle_rms(W[:n])[None, :, None]).astype(np.float32)
    d = dict(X=W, y=y[:n], t=L['t'][:n], level_ratio=L['level_ratio'][:n],
             entropy_drop=L['entropy_drop'][:n], t_end=t_end)
    np.savez(npz, **d)
    return d
