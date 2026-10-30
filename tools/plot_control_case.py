"""One test run through the speed controller (dry-run replay): raw signal, chatter state, output voltages.

Reads the replay log written by
    realtime.py --rpm <rpm> --control --replay data/raw/Acce_60k_1_r2/Acce_<run>.lvm --log results/v2_control/<run>.csv
(and its printed decisions, results/v2_control/<run>.txt) and writes guide/speed_control/case_<run>.png.
The chatter frequency panel recomputes, for every hop, what speed_control.chatter_frequency would
return: the same 2048-sample spectrum, the same air reference (first 2 s) and notches.

Run: .venv/bin/python -m tools.plot_control_case 100k_1_r1 [log folder, default results/v2_control]
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pipeline import data18 as D
from pipeline import excel_labels as el
from rig import speed_control as sc

RPM_PER_VOLT, FZ, Z, PROGRAM_FEED = 14000.0, 3.0, 2, 5.0     # realtime.py defaults (lab script)
STATE = {'no cutting': 0, 'stable': 1, 'chatter': 2}
COL = {0: '#8f8e88', 1: '#2a78d6', 2: '#c0262d', -1: '#d6d5cf'}
# Excel "Freq." row for the chatter segments of the 9 test runs: (segment start s, Hz)
EXCEL_F = {
    '20k_5_r1': [(9.8184, 7666), (10.0608, 7746)], '20k_5_r2': [(10.2797, 7786)], '20k_5_r3': [(9.9892, 7826)],
    '60k_3_r1': [(9.64493, 8600), (9.9961, 8600), (10.12293, 8600), (10.74543, 8820), (11.0036, 7940)],
    '60k_3_r2': [(6.7592, 8460), (7.1727, 8420), (7.21853, 8420), (7.82703, 8480)],
    '60k_3_r3': [(6.77537, 7360), (7.08253, 7540)],
    '100k_1_r1': [(27.5908, 8200), (27.7201, 8233)], '100k_1_r2': [(9.3047, 3333), (9.7187, 8333)],
    '100k_1_r3': [(24.9402, 8333)],
}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8f8e88', 'figure.dpi': 130})


def freq_track(x, rpm, calib_s=2.0):
    """Chatter frequency (Hz, NaN when refused) and window end time for every hop of a raw (n, 3) record.
    rpm = the speed the recording really ran at (the replay never changes it)."""
    fr = sliding_window_view(x.T.astype(float), sc.FFT_WIN, axis=1)[:, ::D.lb.HOP]     # (3, n, 2048)
    fr = (fr - fr.mean(-1, keepdims=True)) * np.hanning(sc.FFT_WIN)
    P = (np.abs(np.fft.rfft(fr, axis=-1)) ** 2).sum(0)                               # (n, bins)
    f = np.fft.rfftfreq(sc.FFT_WIN, 1 / D.lb.FS)
    t_end = (np.arange(len(P)) * D.lb.HOP + sc.FFT_WIN) / D.lb.FS
    air = P[t_end <= calib_s + sc.FFT_WIN / D.lb.FS].mean(0)
    fc = np.array([sc.chatter_frequency(p, air, f, rpm)[0] or np.nan for p in P], float)
    return t_end, fc


def decisions(run, folder='results/v2_control'):
    """(t, f, action) for each controller decision printed in the replay log."""
    txt = re.sub(r'\x1b\[[0-9;]*m', '', Path(f'{folder}/{run}.txt').read_text())
    return [(float(m[1]), float(m[2]), 'refused' if 'no change' in m[0] else 'changed')
            for m in re.finditer(r'\s*([\d.]+) s\s+chatter at\s+([\d.]+) Hz.*', txt)]


def main(run, folder='results/v2_control'):
    rows = list(csv.DictReader(open(f'{folder}/{run}.csv')))
    rows = [r for r in rows if r['vote']]                         # after calibration
    t = np.array([float(r['t']) for r in rows])
    vote = np.array([STATE[r['vote']] for r in rows])
    rpm = np.array([float(r['rpm_cmd']) for r in rows])
    v_spindle = np.clip(rpm / RPM_PER_VOLT, 0, 10)
    v_feed = np.array([sc.SpindleOutput('', RPM_PER_VOLT, z=Z, fz_um=FZ, program_feed=PROGRAM_FEED).feed_volts(n)
                       for n in rpm])
    changes = [(t[i], rpm[i - 1], rpm[i], rows[i]['f_chatter']) for i in range(1, len(t)) if rpm[i] != rpm[i - 1]]

    d = D.load(run)
    excel = el.merge(d['y'])
    x = D.raw(run)
    tx = np.arange(len(x)) / D.lb.FS
    t0, t1 = el.SEGMENTS[run][0][0] - 1.5, float(d['t_end']) + 1.0

    fig, ax = plt.subplots(4, 1, figsize=(11, 9.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1.6, 1.8]))
    m = (tx >= t0) & (tx <= t1)
    for c, name, colr in zip(range(3), 'XYZ', ['#2a78d6', '#0b7a75', '#7a45b0']):
        ax[0].plot(tx[m], x[m, c], lw=0.3, color=colr, label=name, alpha=0.8)
    ax[0].set_ylabel('accelerometer (V)')
    ax[0].legend(loc='upper left', ncol=3, frameon=False)
    ax[0].set_title(f'{run}: raw signal, chatter state and controller output (dry-run replay)', loc='left')

    k = (d['t'] >= t0) & (d['t'] <= t1)
    for c in COL:
        e = k & (excel == c)
        ax[1].scatter(d['t'][e], np.full(e.sum(), 1), s=14, c=COL[c], marker='s', lw=0)
    k = (t >= t0) & (t <= t1)
    for c in (0, 1, 2):
        e = k & (vote == c)
        ax[1].scatter(t[e], np.zeros(e.sum()), s=14, c=COL[c], marker='s', lw=0)
    ax[1].set_yticks([0, 1], ['top-3 vote', 'Excel'])
    ax[1].set_ylim(-0.6, 1.6)
    ax[1].grid(False)
    for c, name in ((0, 'no cutting'), (1, 'stable'), (2, 'chatter')):
        ax[1].scatter([], [], c=COL[c], marker='s', label=name)
    ax[1].legend(loc='upper left', ncol=3, frameon=False, bbox_to_anchor=(0, 1.35))

    ax[2].step(t[k], v_spindle[k], where='post', color='#c2410c', lw=1.6, label='spindle ao0 (V) = rpm / 14000')
    ax[2].step(t[k], v_feed[k], where='post', color='#6b7c8a', lw=1.2, ls='--', label='feed ao1 (V), fz kept at 3 µm')
    ax[2].set_ylabel('output (V)')
    lo, hi = min(v_spindle[k].min(), v_feed[k].min()), max(v_spindle[k].max(), v_feed[k].max())
    pad = max(0.15 * (hi - lo), 0.1)
    ax[2].set_ylim(lo - pad, hi + 2.5 * pad)               # zoom: at 20k a step is only ~0.05 V
    ax[2].legend(loc='upper left', ncol=2, frameon=False)

    tf, fc = freq_track(x, D.rpm(run))
    k = (tf >= t0) & (tf <= t1)
    vote_at = vote[np.clip(np.searchsorted(t, tf[k]), 0, len(t) - 1)]
    for c, name in ((0, 'no cutting'), (1, 'stable'), (2, 'chatter')):
        e = vote_at == c
        ax[3].scatter(tf[k][e], fc[k][e] / 1000, s=6, c=COL[c], lw=0, label=f'f, vote = {name}')
    seg = el.SEGMENTS[run]
    for i, (s0, fx) in enumerate(EXCEL_F.get(run, [])):
        s1 = next((s for s, _ in seg if s > s0), float(d['t_end']))
        ax[3].plot([s0, s1], [fx / 1000] * 2, color='#15212b', lw=2.2, solid_capstyle='butt',
                   label='Excel Freq.' if i == 0 else None)
    decs = [d_ for d_ in decisions(run, folder) if not np.isnan(d_[1])]
    for i, (td, fd, act) in enumerate(decs):
        ax[3].scatter(td, fd / 1000, s=70, marker='o' if act == 'changed' else 'X', facecolor='none' if act == 'changed'
                      else '#c2410c', edgecolor='#c2410c', lw=1.6, zorder=5,
                      label=('decision: speed changed' if act == 'changed' else 'decision: refused (hold)')
                      if i == [a for *_, a in decs].index(act) else None)
    ax[3].set_ylabel('chatter f (kHz)')
    ax[3].set_ylim(0, 10)
    ax[3].legend(loc='lower left', ncol=3, frameon=False, fontsize=8, markerscale=1.2)
    ax[3].set_xlabel('time (s)')
    for tc, *_ in (changes if len(changes) <= 3 else []):
        for a_ in ax:
            a_.axvline(tc, color='#c2410c', lw=0.8, ls=':')
    kt = (t >= t0) & (t <= t1)
    ax[2].set_title(('speed commands:   ' + '   ·   '.join(
        f'{tc:.2f} s  {a:.0f} → {b:.0f} rpm  ({a / RPM_PER_VOLT:.3f} → {b / RPM_PER_VOLT:.3f} V, f = {float(f or "nan"):.0f} Hz)'
        for tc, a, b, f in changes)) if len(changes) <= 3 else
        f'{len(changes)} output changes · spindle {v_spindle[kt].min():.3f}–{v_spindle[kt].max():.3f} V · '
        f'feed {v_feed[kt].min():.3f}–{v_feed[kt].max():.3f} V', loc='left', fontsize=8.5, color='#c2410c')
    ax[2].set_xlim(t0, t1)
    fig.tight_layout()
    out = Path('guide/speed_control') / f"case_{run}{'' if folder == 'results/v2_control' else '_' + Path(folder).name}.png"
    fig.savefig(out)
    print('wrote', out, 'changes:', [(round(c[0], 2), round(c[2])) for c in changes])


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['100k_1_r1']))
