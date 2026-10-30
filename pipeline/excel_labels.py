"""Severity labels from the lab's Excel (data/raw/surface analysis.xlsx, sheet "Surface chatter onset").

For each run the sheet gives the surface condition seen under the microscope and the time
(s, on the recording clock) where it starts. Copied by hand for the 18 runs we use into
data/labels/severity_segments.json, so the xlsx is never parsed at run time.

    0 = no cutting   1 = stable   2 = mild   3 = normal   4 = severe chatter   -1 = ignored

The last condition continues until the cut ends; the end of cutting comes from the signal
level (labels.make_labels `cutting`). Notes without a time ("later some severe chatter
jerks") cannot be placed and are left out.
"""
import json
from pathlib import Path

import numpy as np

AIR, STABLE, MILD, NORMAL, SEVERE, IGNORE = 0, 1, 2, 3, 4, -1
NAMES = ['no cutting', 'stable', 'mild', 'normal', 'severe']
NAMES3 = ['no cutting', 'stable', 'chatter']     # mild, normal and severe merged
GUARD_S = 0.05      # windows this close to a label change are ignored (about 2 hops)
EXIT_FRAC = 0.3     # exit tail: level below this fraction of the last segment's loud level

# Excel text -> class. Numbered repeats (sc2, mc2) are the same class seen again further on.
MAP = {
    's': STABLE, 'ms': STABLE, 'tool marks and shift to s': STABLE,
    'mc': MILD, 'mc2': MILD,
    'nc': NORMAL, 'tool marks and shift to nc': NORMAL,
    'sc': SEVERE, 'sc2': SEVERE, 'sc3': SEVERE, 'sc jerk': SEVERE, 'sc jerks': SEVERE,
    'many sc jerk': SEVERE, 'tool marks with sc': SEVERE,
}

# run -> [[start time s, Excel text], ...]. These are the lab's labels, so they live with the private
# data, not in the code. A trailing "…" with a time (100k_1_r1 29.52 s "… but few severe chatter marks
# jerk", 100k_5_r2 7.61 s "end point") is not a new condition.
SEGMENTS = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'labels' / 'severity_segments.json').read_text())


def cut_end(t, cutting, t_start):
    """Time of the last cutting window of the first continuous cut that runs past t_start."""
    after = np.flatnonzero((t >= t_start) & cutting)
    if not len(after):
        return t_start
    i = after[0]
    while i + 1 < len(t) and cutting[i + 1]:
        i += 1
    return t[i]


def exit_tail(t, level, t_last, t_end):
    """Windows where the tool has left the cut but the level is still dying away: after the
    last window of the final segment whose level reaches 0.3 x the segment's 90th percentile."""
    seg = (t >= t_last) & (t <= t_end)
    thr = EXIT_FRAC * np.quantile(level[seg], 0.9)
    loud = np.flatnonzero(seg & (level >= thr))
    return seg & (t > t[loud[-1]])


def window_labels(run, t, cutting, level):
    """Per-window class for window centres t (s). `cutting` and `level` come from labels.py and
    are used only to find where the cut ends. Returns (labels, t_end)."""
    seg = SEGMENTS[run]
    starts = np.array([s for s, _ in seg])
    cls = np.array([MAP[name] for _, name in seg])
    t_end = cut_end(t, cutting, starts[0])
    y = np.full(len(t), AIR)
    inside = (t >= starts[0]) & (t <= t_end)
    y[inside] = cls[np.searchsorted(starts, t[inside], side='right') - 1]
    edges = np.r_[starts, t_end]
    near = np.abs(t[:, None] - edges[None, :]).min(1) < GUARD_S
    y[near | exit_tail(t, level, starts[-1], t_end)] = IGNORE
    return y, t_end


def merge(y):
    """5-class labels -> 3 classes: mild, normal and severe all become chatter (2)."""
    return np.where(y >= MILD, MILD, y)
