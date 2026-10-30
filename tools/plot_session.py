"""Plot one logged session of scope.py: raw signal, models, chatter frequency, output voltages.

    .venv/bin/python -m tools.plot_session data/recordings/session_20261023_023158

Reads raw.lvm, control.csv and events.txt of the folder (session_log.py) and writes overview.png
into the same folder. Both files use the same clock (seconds since the server started), so the
rows line up even if the models ran late at the time.
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CODE = {'no cutting': 0, 'stable': 1, 'chatter': 2}
COL = {0: '#aeaeb2', 1: '#34c759', 2: '#ff3b30'}
MODELS = [('extra_trees', 'Extra trees'), ('random_forest', 'Random forest'), ('svm_y', 'SVM (Y)'), ('vote', 'Vote')]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8f8e88', 'figure.dpi': 130})


def load_raw(path):
    """(t, x) with x (n, n_ch). Reads the tab-separated text in blocks (fast enough for ~10 min)."""
    data = np.loadtxt(path, dtype=np.float32)
    return data[:, 0].astype(float), data[:, 1:]


def envelope(t, v, cols=3000):
    """Min/max per column, so millions of samples draw like the real trace."""
    n = len(v) // cols * cols
    if n < cols:
        return t, v, v
    tt = t[:n].reshape(cols, -1).mean(1)
    vv = v[:n].reshape(cols, -1)
    return tt, vv.min(1), vv.max(1)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main(folder):
    folder = Path(folder)
    t_raw, x = load_raw(folder / 'raw.lvm')
    rows = list(csv.DictReader(open(folder / 'control.csv')))
    t = np.array([num(r['t']) for r in rows])
    lab = {k: np.array([CODE.get(r[k], -1) for r in rows]) for k, _ in MODELS}
    level = np.array([num(r['level_ratio']) for r in rows])
    f = np.array([num(r['chatter_f_Hz']) for r in rows])
    vs = np.array([num(r['spindle_V']) for r in rows])
    vf = np.array([num(r['feed_V']) for r in rows])
    writing = np.array([r['writing'] == '1' for r in rows])
    events = []
    if (folder / 'events.txt').exists():
        for line in open(folder / 'events.txt'):
            m = re.match(r'\s*(\d\d:\d\d:\d\d)\s+(.*)', line)
            events.append(line.strip())

    fig, axs = plt.subplots(7, 1, figsize=(12, 13), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 1, 1.0, 1.1, 1.3, 1.3]))
    for c, (name, colr) in enumerate(zip('XYZ', ['#2a78d6', '#0b7a75', '#7a45b0'])):   # own scale per axis
        if c < x.shape[1]:
            v = x[:, c] - np.median(x[:, c])
            tt, lo, hi = envelope(t_raw, v)
            axs[c].fill_between(tt, lo, hi, color=colr, alpha=0.7, lw=0)
            axs[c].set_ylabel(f'{name} ai{c} (V)')
            axs[c].text(0.995, 0.85, f'rms {v.std() * 1000:.1f} mV', transform=axs[c].transAxes, ha='right', fontsize=8)
    axs[0].set_title(f'{folder.name}: raw signal (each axis on its own scale), models, chatter frequency and output',
                     loc='left')
    ax = axs[2:]

    ax[1].plot(t, level, color='#52514e', lw=0.8)
    ax[1].set_yscale('log')
    ax[1].axhline(2, color='#8f8e88', ls='--', lw=0.8)
    ax[1].set_ylabel('level / idle')

    for r, (k, name) in enumerate(MODELS):
        for c in (0, 1, 2):
            m = lab[k] == c
            ax[2].scatter(t[m], np.full(m.sum(), len(MODELS) - 1 - r), s=6, c=COL[c], marker='s', lw=0)
    ax[2].set_yticks(range(len(MODELS)), [n for _, n in MODELS][::-1])
    ax[2].set_ylim(-0.6, len(MODELS) - 0.4)
    ax[2].grid(False)
    for c, name in ((0, 'no cutting'), (1, 'stable'), (2, 'chatter')):
        ax[2].scatter([], [], c=COL[c], marker='s', label=name)
    ax[2].legend(loc='upper left', ncol=3, frameon=False, bbox_to_anchor=(0, 1.3))

    for c in (1, 2):
        m = lab['vote'] == c
        ax[3].scatter(t[m], f[m] / 1000, s=4, c=COL[c], lw=0)
    ax[3].set_ylabel('chatter f (kHz)')
    ax[3].set_ylim(0, 10)
    ax[3].text(0.995, 0.92, 'shown only while the vote says stable / chatter', transform=ax[3].transAxes,
               ha='right', fontsize=8, color='#6e6e73')

    ax[4].step(t, vs, where='post', color='#c2410c', lw=1.5, label='spindle ao0 (V)')
    if np.isfinite(vf).any():
        ax[4].step(t, vf, where='post', color='#6b7c8a', lw=1.2, ls='--', label='feed ao1 (V)')
    lo, hi = np.nanmin(np.r_[vs, vf]), np.nanmax(np.r_[vs, vf])
    ax[4].fill_between(t, lo - 0.3, hi + 0.3, where=writing, color='#ff3b30', alpha=0.12, lw=0,
                       label='output ON (really written)')
    ax[4].set_ylim(lo - 0.3, hi + 0.6)
    ax[4].set_ylabel('command (V)')
    ax[4].set_xlabel('time since the server started (s)')
    ax[4].legend(loc='upper left', ncol=3, frameon=False)
    fig.tight_layout()
    out = folder / 'overview.png'
    fig.savefig(out)

    cut = lab['vote'] > 0
    print(f'wrote {out}')
    print(f'duration {t_raw[-1]:.1f} s raw, {len(t)} model rows')
    print('vote: ' + ', '.join(f'{n} {np.mean(lab["vote"] == c) * 100:.1f} %' for n, c in CODE.items()))
    if cut.any():
        print(f'cutting (vote stable/chatter) from {t[cut][0]:.1f} to {t[cut][-1]:.1f} s')
    print(f'output written during {writing.mean() * len(t) * 0.0256:.1f} s')
    for e in events:
        print('event:', e)


if __name__ == '__main__':
    main(sys.argv[1])
