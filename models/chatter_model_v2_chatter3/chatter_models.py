"""The three shipped models: which features each one reads, and how those features are computed.

All three see the same input: 51 ms windows (3 axes x 1024 samples at 20 kS/s), cleaned with
causal.py and divided by each axis's idle RMS. They differ in the feature set and the classifier.

  extra_trees     extra trees, 300 trees            wave-1 set, all axes      75 features
  random_forest   random forest, 300 trees          wave-1 set, all axes      75 features
  svm_y           RBF SVM (scaled inputs)           every feature ending in Y 79 features
"""
import numpy as np
from scipy.special import softmax
from sklearn.svm import SVC

import features as ft

MODELS = {
    'extra_trees':   'Extra trees (300 trees) on the wave-1 set, all three axes',
    'random_forest': 'Random forest (300 trees) on the wave-1 set, all three axes',
    'svm_y':         'RBF SVM on every feature whose name ends in Y (Y axis, plus corr_XY and coh_XY)',
}


class SVMSoft(SVC):
    """RBF SVM; class scores turned into probabilities with a softmax of the
    one-vs-rest decision values (cheaper than Platt scaling's internal 5-fold CV)."""
    def predict_proba(self, X):
        return softmax(2 * self.decision_function(X), axis=1)


def wave1(Ws):
    """(n, 75) wave-1 features of scaled windows Ws (n, 3, 1024), and their names."""
    F, names, _ = ft.extract(Ws, groups=ft.WAVE1)
    return F, names


def full(Ws):
    """(n, 237) every label-free feature plus the 9 sequence features, and their names.
    Sequence features look back over the previous 18 windows of the same stream,
    so Ws must be consecutive windows of one recording."""
    F, names, _ = ft.extract(Ws, groups=tuple(ft.AGNOSTIC))
    S, s_names = ft.sequence_features(Ws)
    return np.concatenate([F, S], 1), list(names) + s_names


def matrix(Ws, names):
    """Feature matrix in exactly the column order `names` (as stored in meta.json)."""
    F, have = wave1(Ws) if all(n in WAVE1_NAMES for n in names) else full(Ws)
    idx = {n: i for i, n in enumerate(have)}
    return F[:, [idx[n] for n in names]]


WAVE1_NAMES = set(wave1(np.random.default_rng(0).standard_normal((2, 3, ft.WIN)))[1])
