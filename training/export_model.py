"""Build the portable chatter package in models/chatter_model/ with the top 3 models of part 2.

  extra_trees     extra trees on the 75 wave-1 features          (part 2: 0.865)
  random_forest   random forest on the 75 wave-1 features        (part 2: 0.859)
  svm_y           RBF SVM on the 79 features ending in Y         (part 2: 0.861)

Part 2 scored them with the offline cleaning (denoise.py). The package must also run live,
so here the windows are cleaned with causal.py, the code the live stream uses, and each
model's leave-one-file-out score is measured again for exactly this pipeline before the
final models are fitted on all 12 files. Also writes the feature reference (features.csv).

Run: .venv/bin/python -m training.export_model
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path('models/chatter_model')
sys.path.insert(0, str(OUT))                    # pickles must refer to chatter_models.SVMSoft
import chatter_models as cm                     # noqa: E402

from pipeline import causal                                   # noqa: E402
from pipeline import data12 as D                              # noqa: E402
from pipeline import dataset as ds                            # noqa: E402
from pipeline import feature_docs                             # noqa: E402
from pipeline import labels as lb                             # noqa: E402

CACHE = Path('data/cache/deploy_features_full.npz')
CLASSES = ['no cutting', 'stable', 'chatter']

MAKE = {
    'extra_trees': lambda: ExtraTreesClassifier(300, min_samples_leaf=5, class_weight='balanced',
                                                random_state=0, n_jobs=-1),
    'random_forest': lambda: RandomForestClassifier(300, min_samples_leaf=5, max_features='sqrt',
                                                    class_weight='balanced_subsample', random_state=0, n_jobs=-1),
    'svm_y': lambda: make_pipeline(StandardScaler(), cm.SVMSoft(C=1.0, class_weight='balanced',
                                                                decision_function_shape='ovr')),
}
HYPER = {
    'extra_trees': 'ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight=balanced)',
    'random_forest': 'RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=sqrt, '
                     'class_weight=balanced_subsample)',
    'svm_y': 'StandardScaler -> SVC(kernel=rbf, C=1, gamma=scale, class_weight=balanced); '
             'probabilities = softmax(2 x one-vs-rest decision values)',
}


def idle_rms(W):
    """Per-axis idle level: mean RMS of the quietest 20 % of windows (as data12.load)."""
    rms = np.sqrt((W ** 2).mean(-1))
    tot = np.sqrt((rms ** 2).sum(1))
    return rms[tot <= np.quantile(tot, 0.20)].mean(0)


def file_table(key):
    x = D.raw(key).astype(float)
    y = lb.make_labels(x, D.FILES[key][1])['label']
    clean = causal.clean_offline(x)
    W = sliding_window_view(clean, lb.WIN, axis=0)[::lb.HOP]           # (n, 3, WIN)
    n = min(len(W), len(y))
    W, y = W[:n], y[:n]
    F, names = cm.full(W / idle_rms(W)[None, :, None])
    print('features', key, F.shape, flush=True)
    return F, y, names


def feature_names(model, names):
    if model in ('extra_trees', 'random_forest'):
        return [n for n in names if n in cm.WAVE1_NAMES]
    return [n for n in names if n.endswith('Y')]       # the part-2 "Y axis only" filter


def main():
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        F, y, fid, names = c['F'], c['y'], c['fid'], list(c['names'])
    else:
        parts = [file_table(k) for k in D.FILES]
        F = np.concatenate([p[0] for p in parts]).astype(np.float32)
        y = np.concatenate([p[1] for p in parts])
        fid = np.concatenate([np.full(len(p[1]), i) for i, p in enumerate(parts)])
        names = parts[0][2]
        np.savez(CACHE, F=F, y=y, fid=fid, names=np.array(names))
    F = np.nan_to_num(F)
    lab = y >= 0
    keys = list(D.FILES)
    col = {n: i for i, n in enumerate(names)}

    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(classes=CLASSES, window=dict(samples=lb.WIN, hop=lb.HOP, fs=lb.FS),
                cleaning='causal.py: despike, 50 Hz high-pass, mains notches 50-400 Hz',
                scaling='each axis divided by its idle RMS (first seconds live, quietest 20 % of windows for a file)',
                trained_on={k: v[0] for k, v in D.FILES.items()},
                n_windows=int(lab.sum()), class_counts=np.bincount(y[lab], minlength=3).tolist(),
                sklearn_version=sklearn.__version__, models={})
    for model in MAKE:
        fn = feature_names(model, names)
        X = F[:, [col[n] for n in fn]]
        pred = np.full(len(y), -1)
        for f in np.unique(fid):
            tr, te = lab & (fid != f), fid == f
            pred[te] = MAKE[model]().fit(X[tr], y[tr]).predict(X[te])
        m = ds.metrics(y[lab], pred[lab])
        per_file = {keys[f]: round(ds.metrics(y[lab & (fid == f)], pred[lab & (fid == f)])['macro_f1'], 3)
                    for f in np.unique(fid)}
        print(f'{model:14s} {len(fn)} features, held-out macro F1 {m["macro_f1"]:.3f}', flush=True)
        joblib.dump(MAKE[model]().fit(X[lab], y[lab]), OUT / f'{model}.joblib', compress=3)
        meta['models'][model] = dict(description=cm.MODELS[model], classifier=HYPER[model],
                                     feature_names=fn, n_features=len(fn),
                                     heldout_macro_f1=round(m['macro_f1'], 3), heldout_per_file=per_file)
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
    (OUT / 'model.joblib').unlink(missing_ok=True)                    # the old single-model file

    # feature reference: one row per feature column
    with open(OUT / 'features.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'group', 'axis', 'used_by', 'description', 'computed_in'])
        for n in names:
            used = [k for k in MAKE if n in meta['models'][k]['feature_names']]
            if used:
                group, axis, text, where = feature_docs.describe(n)
                w.writerow([n, group, axis, ' '.join(used), text, where])
    for f in ('denoise.py', 'causal.py', 'features.py'):
        shutil.copy(Path('pipeline') / f, OUT / f)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
