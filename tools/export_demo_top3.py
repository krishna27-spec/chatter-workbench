"""Add the top-3 models' predictions to the demo recordings in rig/dashboard/data/, so the website's Demo
shows the same models as Live.

Each demo signal (rig/dashboard/data/<run>.i16, made by export_site_data.py) is streamed through
realtime.Top3Predictor exactly as scope.py does live: 512-sample hops, causal cleaning, idle RMS
from the first 2 s (the demo starts 4 s before the cut), 10-window smoothing. Writes, per window,
each model's label and probabilities and the vote into labels.top3 of rig/dashboard/data/<run>.json.

Run: .venv/bin/python -m tools.export_demo_top3
"""
import json
from pathlib import Path

import numpy as np

from rig import realtime as rt

DATA = Path('rig/dashboard/data')
CODE = {'no cutting': 0, 'stable': 1, 'chatter': 2}


def main():
    index = json.loads((DATA / 'index.json').read_text())
    for run in [r['id'] for r in index['runs']]:
        meta = json.loads((DATA / f'{run}.json').read_text())
        n, fs = meta['n'], meta['fs']
        i16 = np.fromfile(DATA / f'{run}.i16', '<i2').reshape(3, n)
        x = i16 * np.array(meta['scale'])[:, None]
        top3 = rt.Top3Predictor()
        names = list(top3.models)
        out = dict(t=[], vote=[], models={m: [] for m in names}, probs={m: [] for m in names})
        for i in range(0, n - rt.HOP + 1, rt.HOP):
            r = top3.update(x[:, i:i + rt.HOP])
            if r is None:
                continue
            out['t'].append(round((i + rt.HOP - rt.WIN / 2) / fs, 4))     # window centre, as scope.py
            out['vote'].append(CODE[r['vote']])
            for m in names:
                out['models'][m].append(CODE[r[m]])
                out['probs'][m].append([round(float(p), 3) for p in top3.last_probs[m]])
        meta['labels']['top3'] = out
        (DATA / f'{run}.json').write_text(json.dumps(meta))
        v = np.array(out['vote'])
        print(f'{run}: {len(v)} windows, vote no cutting {np.sum(v == 0)}, stable {np.sum(v == 1)}, chatter {np.sum(v == 2)}')
    index['note'] += '; top3 (extra trees, random forest, SVM-Y, vote): 0 no cutting, 1 stable, 2 chatter'
    (DATA / 'index.json').write_text(json.dumps(index, indent=1))


if __name__ == '__main__':
    main()
