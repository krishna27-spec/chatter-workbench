"""Retrain the top 3 models (export_model.py) on the 12 severity-labelled runs.

  severity5 (default)  0 no cutting   1 stable   2 mild   3 normal   4 severe chatter   (excel_labels.py)
  chatter3             0 no cutting   1 stable   2 chatter (mild + normal + severe merged)

Same features, cleaning and classifiers as models/chatter_model; only the files and labels change.
  1. leave-one-file-out (LOFO) on the 12 training runs: each run is predicted by models that never saw it
  2. fit on all 12, predict the 6 test runs (a feed rate never seen in training at that speed)
Writes results/v2[_chatter3]/results.json, pred_<model>.npz and the package models/chatter_model_v2[_chatter3]/.

Run: .venv/bin/python -m training.train18 [severity5|chatter3]
"""
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

from pipeline import data18 as D
from pipeline import dataset as ds
from pipeline import excel_labels as el
from training import export_model as ex

cm = ex.cm                                      # models/chatter_model/chatter_models.py
# scheme -> (class names, label transform, output suffix)
SCHEMES = {'severity5': (el.NAMES, lambda y: y, ''),
           'chatter3': (el.NAMES3, el.merge, '_chatter3')}
CACHE = D.CACHE / 'features.npz'
RUNS = D.TRAIN + D.TEST


def file_features(run):
    d = D.load(run)
    F, names = cm.full(d['X'])
    print('features', run, F.shape, flush=True)
    return np.nan_to_num(F).astype(np.float32), d['y'], names


def features():
    """Feature table of all 18 runs, computed once (one process per file)."""
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        return c['F'], c['y'], c['fid'], list(c['names'])
    with mp.get_context('fork').Pool(6) as p:
        parts = p.map(file_features, RUNS)
    F = np.concatenate([q[0] for q in parts])
    y = np.concatenate([q[1] for q in parts])
    fid = np.concatenate([np.full(len(q[1]), i) for i, q in enumerate(parts)])
    names = parts[0][2]
    np.savez(CACHE, F=F, y=y, fid=fid, names=np.array(names))
    return F, y, fid, names


def summary(y, pred, fid, files, classes):
    """Pooled metrics plus macro F1 per file (over the classes present in that file)."""
    n = len(classes)
    m = ds.metrics(y, pred, n_classes=n)
    return dict(macro_f1=round(m['macro_f1'], 3), accuracy=round(m['accuracy'], 3),
                f1={classes[c]: (None if np.isnan(v) else round(float(v), 3)) for c, v in enumerate(m['f1'])},
                support=m['support'].tolist(), confusion=m['confusion'].tolist(),
                per_file={RUNS[f]: round(ds.metrics(y[fid == f], pred[fid == f], n)['macro_f1'], 3) for f in files})


def main(scheme='severity5'):
    classes, transform, suffix = SCHEMES[scheme]
    OUT = Path('models/chatter_model_v2' + suffix)
    RES = Path('results/v2' + suffix)
    F, y, fid, names = features()
    y = transform(y)
    n = len(classes)
    col = {n: i for i, n in enumerate(names)}
    lab = y >= 0
    train_ids = [RUNS.index(r) for r in D.TRAIN]
    test_ids = [RUNS.index(r) for r in D.TEST]
    is_train = np.isin(fid, train_ids)
    is_test = np.isin(fid, test_ids)
    print('labelled windows  train', np.bincount(y[lab & is_train], minlength=n),
          ' test', np.bincount(y[lab & is_test], minlength=n), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    meta = dict(classes=classes, window=dict(samples=D.lb.WIN, hop=D.lb.HOP, fs=D.lb.FS),
                cleaning='causal.py: despike, 50 Hz high-pass, mains notches 50-400 Hz',
                scaling='each axis divided by its idle RMS (quietest 20 % of windows of the file)',
                labels='data/raw/surface analysis.xlsx, sheet "Surface chatter onset" (excel_labels.py)',
                trained_on=D.TRAIN, tested_on=D.TEST,
                class_counts_train=np.bincount(y[lab & is_train], minlength=n).tolist(),
                sklearn_version=sklearn.__version__, models={})
    for model in ex.MAKE:
        fn = ex.feature_names(model, names)
        X = F[:, [col[n] for n in fn]]
        pred = np.full(len(y), -1)
        for f in train_ids:                                         # 1. LOFO on the 12
            tr = lab & is_train & (fid != f)
            pred[fid == f] = ex.MAKE[model]().fit(X[tr], y[tr]).predict(X[fid == f])
            print(f'  {model} LOFO {RUNS[f]} done', flush=True)
        clf = ex.MAKE[model]().fit(X[lab & is_train], y[lab & is_train])   # 2. all 12 -> test
        pred[is_test] = clf.predict(X[is_test])
        joblib.dump(clf, OUT / f'{model}.joblib', compress=3)

        k = lab & is_train
        results[model] = dict(lofo=summary(y[k], pred[k], fid[k], train_ids, classes))
        k = lab & is_test
        results[model]['test'] = summary(y[k], pred[k], fid[k], test_ids, classes)
        np.savez(RES / f'pred_{model}.npz', **{r: pred[fid == i] for i, r in enumerate(RUNS)})
        print(f'{model:14s} LOFO macro F1 {results[model]["lofo"]["macro_f1"]:.3f}   '
              f'test macro F1 {results[model]["test"]["macro_f1"]:.3f}', flush=True)
        print('   LOFO per class', results[model]['lofo']['f1'], flush=True)
        print('   test per class', results[model]['test']['f1'], flush=True)
        meta['models'][model] = dict(description=cm.MODELS[model], classifier=ex.HYPER[model],
                                     feature_names=fn, n_features=len(fn),
                                     lofo_macro_f1=results[model]['lofo']['macro_f1'],
                                     test_macro_f1=results[model]['test']['macro_f1'])
    (RES / 'results.json').write_text(json.dumps(results, indent=1))
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
    for f in ('denoise.py', 'causal.py', 'features.py'):
        shutil.copy(Path('pipeline') / f, OUT / f)
    shutil.copy(ex.OUT / 'chatter_models.py', OUT / 'chatter_models.py')
    print('wrote', OUT, RES)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
