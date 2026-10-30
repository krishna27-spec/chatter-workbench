"""Score the saved v2 models on the 3 extra test runs (data18.EXTRA), then on all 9 unseen-feed runs.

The models in models/chatter_model_v2[_chatter3]/ were fitted on the 12 training runs only, so this
is a clean test. Writes results/v2[_chatter3]/extra.json and adds the 3 runs to pred_<model>.npz.

Run: .venv/bin/python -m training.test_extra
"""
import json
from pathlib import Path

import joblib
import numpy as np

from pipeline import data18 as D
from pipeline import dataset as ds
from training import train18 as T


def main():
    parts = [T.file_features(r) for r in D.EXTRA]
    names = parts[0][2]
    col = {n: i for i, n in enumerate(names)}
    for scheme, (classes, transform, suffix) in T.SCHEMES.items():
        n = len(classes)
        out, res = Path('models/chatter_model_v2' + suffix), Path('results/v2' + suffix)
        report = {}
        for model in T.ex.MAKE:
            clf = joblib.load(out / f'{model}.joblib')
            idx = [col[c] for c in T.ex.feature_names(model, names)]
            pf = res / f'pred_{model}.npz'
            pred = dict(np.load(pf))
            for r, (F, _, _) in zip(D.EXTRA, parts):
                pred[r] = clf.predict(F[:, idx])
            np.savez(pf, **pred)
            per_file, ys, ps = {}, [], []
            for r in D.TEST + D.EXTRA:
                y = transform(D.load(r)['y'])
                k = y >= 0
                per_file[r] = round(ds.metrics(y[k], pred[r][k], n)['macro_f1'], 3)
                ys.append(y[k]), ps.append(pred[r][k])
            m3 = ds.metrics(np.concatenate(ys[-3:]), np.concatenate(ps[-3:]), n)
            m9 = ds.metrics(np.concatenate(ys), np.concatenate(ps), n)
            f1 = lambda m: {classes[c]: (None if np.isnan(v) else round(float(v), 3)) for c, v in enumerate(m['f1'])}
            report[model] = dict(extra3=dict(macro_f1=round(m3['macro_f1'], 3), f1=f1(m3),
                                             confusion=m3['confusion'].tolist()),
                                 all9=dict(macro_f1=round(m9['macro_f1'], 3), f1=f1(m9),
                                           support=m9['support'].tolist(), confusion=m9['confusion'].tolist()),
                                 per_file=per_file)
            print(f'{scheme:9s} {model:14s} extra 3 macro F1 {m3["macro_f1"]:.3f}   all 9 {m9["macro_f1"]:.3f}', flush=True)
            print('   all 9 per class', report[model]['all9']['f1'], flush=True)
            print('   per file', {r: per_file[r] for r in D.EXTRA}, flush=True)
        (res / 'extra.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
