"""Everything scope.py sees and sends, saved from the moment it starts.

One folder per run: data/recordings/session_<date>_<time>/
  raw.lvm       time (s), X, Y, Z (V), every sample (20 kHz), tab separated: the layout of the
                lab recordings, so it can be replayed (realtime.py / scope.py --replay raw.lvm)
  control.csv   one row per 25.6 ms hop: top-3 vote and each model, entropy rule, the speed
                command, spindle and feed volts, override %, whether the output was really
                being written, chatter frequency, phase nu, controller state
  events.txt    every controller decision and every Start / Stop output
  session.json  the settings, the NI modules and pins, the model package, the start time
Times in raw.lvm and control.csv are on the same clock: seconds since the server started.

The files are written by a background thread, so saving never holds up the acquisition.
raw.lvm grows by about 3 GB per hour at 20 kHz.
"""
import csv
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

COLUMNS = ['t', 'wall_time', 'vote', 'extra_trees', 'random_forest', 'svm_y', 'rule', 'level_ratio',
           'entropy_drop', 'rpm_cmd', 'spindle_V', 'feed_V', 'feed_override_pct', 'writing',
           'chatter_f_Hz', 'nu', 'controller_state']


class SessionLog:
    def __init__(self, root, meta):
        self.dir = Path(root) / f"session_{datetime.now():%Y%m%d_%H%M%S}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / 'session.json').write_text(json.dumps(meta, indent=1, default=str))
        self.f_raw = open(self.dir / 'raw.lvm', 'w', buffering=1 << 20)
        self.f_ctl = open(self.dir / 'control.csv', 'w', newline='', buffering=1 << 16)
        self.ctl = csv.DictWriter(self.f_ctl, COLUMNS, extrasaction='ignore')
        self.ctl.writeheader()
        self.f_ev = open(self.dir / 'events.txt', 'w', buffering=1)
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # called from the acquisition / analysis / web threads: only queue, never write here
    def raw(self, start, chunk, fs):
        self.q.put(('raw', start, np.array(chunk, np.float32), fs))

    def control(self, row):
        self.q.put(('ctl', row))

    def event(self, text):
        self.q.put(('ev', text))

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            kind = item[0]
            if kind == 'raw':
                _, start, chunk, fs = item
                t = (start + np.arange(chunk.shape[1])) / fs
                np.savetxt(self.f_raw, np.column_stack([t, chunk.T]), fmt='%.6f', delimiter='\t')
            elif kind == 'ctl':
                self.ctl.writerow(item[1])
            else:
                self.f_ev.write(item[1] + '\n')

    def size_mb(self):
        return round(sum(f.stat().st_size for f in self.dir.iterdir()) / 1e6, 1)

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=10)
        for f in (self.f_raw, self.f_ctl, self.f_ev):
            f.close()
