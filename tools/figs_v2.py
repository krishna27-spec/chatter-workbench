"""Figures for the v2 severity study. Each function writes one guide/v2_training/*.png.
Run: .venv/bin/python -m tools.figs_v2 [name ...]   (no names = all)
     .venv/bin/python -m tools.figs_v2 --chatter3 [name ...]   3-class run (mild + normal + severe = chatter)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pipeline import data18 as D
from pipeline import excel_labels as el

OUT = Path('guide/v2_training')
RES = Path('results/v2')
# no cutting grey, stable blue, mild yellow, normal orange, severe red, ignored light grey
CC = {0: '#8f8e88', 1: '#2a78d6', 2: '#e0b000', 3: '#eb6834', 4: '#c0262d', -1: '#d6d5cf'}
CN = {0: 'no cutting', 1: 'stable', 2: 'mild', 3: 'normal', 4: 'severe', -1: 'ignored'}
MODELS = ['extra_trees', 'random_forest', 'svm_y']
NAMES, PRE, LAB = el.NAMES, '', lambda y: y          # switched by --chatter3
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8f8e88', 'figure.dpi': 130})


def labels_timeline(runs=None, name='labels_timeline'):
    """Level / idle over time, coloured by the Excel label, zoomed to the cut (2 s either side)."""
    runs = runs or D.TRAIN + D.TEST
    rows = -(-len(runs) // 3)
    fig, axs = plt.subplots(rows, 3, figsize=(12, 2.2 * rows), sharey=True, squeeze=False)
    for ax, r in zip(axs.ravel(), runs):
        d = D.load(r)
        t0, t1 = el.SEGMENTS[r][0][0] - 2, float(d['t_end']) + 2
        m = (d['t'] >= t0) & (d['t'] <= t1)
        y = LAB(d['y'])
        for c in CC:
            k = m & (y == c)
            ax.scatter(d['t'][k], d['level_ratio'][k], s=4, c=CC[c], label=CN[c], lw=0)
        for s, _ in el.SEGMENTS[r]:
            ax.axvline(s, color='#52514e', lw=0.5, ls=':')
        ax.set_yscale('log')
        ax.set_title(f"{r}  ({'train' if r in D.TRAIN else 'test'})", loc='left', fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel('time (s)')
    for ax in axs[:, 0]:
        ax.set_ylabel('level / idle')
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=6, markerscale=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / rows))
    fig.savefig(OUT / f'{PRE}{name}.png')


def class_counts():
    """Labelled windows per class: train (12 files) vs test (6 files)."""
    cnt = {}
    for part, runs in (('train', D.TRAIN), ('test', D.TEST)):
        y = np.concatenate([D.load(r)['y'] for r in runs])
        cnt[part] = np.bincount(y[y >= 0], minlength=5)
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(5)
    for i, part in enumerate(('train', 'test')):
        b = ax.bar(x + (i - 0.5) * 0.38, cnt[part], 0.38, label=part,
                   color=[CC[c] for c in range(5)], alpha=1 if part == 'train' else 0.5)
        ax.bar_label(b, fontsize=8)
    ax.set_xticks(x, el.NAMES)
    ax.set_yscale('log')
    ax.set_ylabel('windows (log)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'class_counts.png')


def confusion():
    """Row-normalised confusion matrices: leave-one-file-out (top) and unseen-feed test (bottom)."""
    res = json.loads((RES / 'results.json').read_text())
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.6) if len(NAMES) == 5 else (12, 6.4))
    for j, m in enumerate(MODELS):
        for i, part in enumerate(('lofo', 'test')):
            C = np.array(res[m][part]['confusion'], float)
            k = len(NAMES)
            P = C / np.maximum(C.sum(1, keepdims=True), 1)
            ax = axs[i, j]
            ax.imshow(P, cmap='Blues', vmin=0, vmax=1)
            for a in range(k):
                for b in range(k):
                    if C[a].sum():
                        ax.text(b, a, f'{P[a, b]:.2f}\n({int(C[a, b])})', ha='center', va='center', fontsize=7,
                                color='white' if P[a, b] > 0.6 else '#15212b')
            ax.set_xticks(range(k), NAMES, rotation=30)
            ax.set_yticks(range(k), NAMES)
            ax.grid(False)
            ax.set_title(f"{m}  {'LOFO (12 train files)' if part == 'lofo' else 'test (unseen feed)'}  "
                         f"macro F1 {res[m][part]['macro_f1']:.3f}", fontsize=9, loc='left')
            if j == 0:
                ax.set_ylabel('true')
            if i == 1:
                ax.set_xlabel('predicted')
    fig.tight_layout()
    fig.savefig(OUT / f'{PRE}confusion.png')


def per_file():
    """Macro F1 per file for each model: training files (LOFO) and test files."""
    res = json.loads((RES / 'results.json').read_text())
    runs = D.TRAIN + D.TEST
    fig, ax = plt.subplots(figsize=(12, 3.6))
    x = np.arange(len(runs))
    for i, m in enumerate(MODELS):
        v = [res[m]['lofo']['per_file'].get(r, res[m]['test']['per_file'].get(r)) for r in runs]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, label=m, color=['#2a78d6', '#0b7a75', '#7a45b0'][i])
    ax.axvline(len(D.TRAIN) - 0.5, color='#52514e', lw=1)
    ax.text(len(D.TRAIN) - 0.4, 1.02, 'test →', fontsize=8)
    ax.set_xticks(x, runs, rotation=40, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('macro F1')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / f'{PRE}per_file.png')


def timelines(runs=None, name='test_timelines'):
    """True label vs best model's prediction over time, for each test file."""
    runs = runs or D.TEST
    res = json.loads((RES / 'results.json').read_text())
    best = max(MODELS, key=lambda m: res[m]['test']['macro_f1'])
    pred = np.load(RES / f'pred_{best}.npz')
    fig, axs = plt.subplots(len(runs), 1, figsize=(11, 1.5 * len(runs) + 0.4))
    for ax, r in zip(axs, runs):
        d = D.load(r)
        y = LAB(d['y'])
        p = pred[r]
        t0, t1 = el.SEGMENTS[r][0][0] - 1, float(d['t_end']) + 1
        m = (d['t'] >= t0) & (d['t'] <= t1)
        for c in CC:
            k = m & (y == c)
            ax.scatter(d['t'][k], np.full(k.sum(), 1), s=10, c=CC[c], marker='s', lw=0)
            k = m & (p == c)
            ax.scatter(d['t'][k], np.zeros(k.sum()), s=10, c=CC[c], marker='s', lw=0, label=CN[c])
        ax.set_yticks([0, 1], [f'{best}', 'Excel'])
        ax.set_ylim(-0.6, 1.6)
        ax.grid(False)
        ax.set_title(r, loc='left', fontsize=9)
    axs[-1].set_xlabel('time (s)')
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=6, markerscale=1.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / len(runs)))
    fig.savefig(OUT / f'{PRE}{name}.png')


def extra():
    """Labels and predictions for the 3 extra test runs."""
    labels_timeline(D.EXTRA, 'extra_labels')
    timelines(D.EXTRA, 'extra_timelines')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    if '--chatter3' in args:
        args.remove('--chatter3')
        RES, NAMES, PRE, LAB = Path('results/v2_chatter3'), el.NAMES3, 'c3_', el.merge
        CC[2], CN[2] = CC[4], 'chatter'
        CC.pop(3), CC.pop(4)
    names = args or ['labels_timeline', 'class_counts', 'confusion', 'per_file', 'timelines']
    for n in names:
        globals()[n]()
        print('wrote', n)
