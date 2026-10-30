"""The same tree ensemble, evaluated fast for one window at a time.

scikit-learn's predict_proba on a 300-tree forest walks the trees one by one; for a single
window that costs about 7 ms, and the live loop has 25.6 ms per window for everything. Here all
trees are packed into flat arrays (one node table with offsets) and walked together with NumPy:
one step down every tree at once, repeated until every tree has reached a leaf (at most the
depth of the deepest tree). The trees, thresholds and leaf values are the fitted ones, so the
probabilities are the same as predict_proba (checked by check() to 1e-12).

Leaf value = class fractions of the training windows in that leaf (with the class weights), as
sklearn's tree predict_proba; the forest averages the trees. A pipeline is not handled here;
the extra trees and random forest of the models/ packages are plain estimators.
"""
import numpy as np


class FastForest:
    def __init__(self, forest):
        trees = [e.tree_ for e in forest.estimators_]
        sizes = np.array([t.node_count for t in trees])
        off = np.r_[0, np.cumsum(sizes)[:-1]]
        left, right, feat, thr, val = [], [], [], [], []
        for t, o in zip(trees, off):
            leaf = t.children_left == -1
            left.append(np.where(leaf, -1, t.children_left + o))
            right.append(np.where(leaf, -1, t.children_right + o))
            feat.append(np.where(leaf, 0, t.feature))
            thr.append(t.threshold)
            v = t.value[:, 0, :]
            val.append(v / v.sum(1, keepdims=True))
        self.left, self.right = np.concatenate(left), np.concatenate(right)
        self.feat, self.thr = np.concatenate(feat), np.concatenate(thr)
        self.val = np.concatenate(val)
        self.roots = off
        self.classes_ = forest.classes_
        self.n_features_in_ = forest.n_features_in_

    def predict_proba(self, X):
        X = np.asarray(X, float)
        out = np.empty((len(X), self.val.shape[1]))
        for i, x in enumerate(X):
            node = self.roots.copy()
            go = self.left[node] != -1
            while go.any():
                n = node[go]
                node[go] = np.where(x[self.feat[n]] <= self.thr[n], self.left[n], self.right[n])
                go = self.left[node] != -1
            out[i] = self.val[node].mean(0)
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def check(forest, X):
    """Largest difference to the forest's own predict_proba on the rows of X."""
    return float(np.abs(FastForest(forest).predict_proba(X) - forest.predict_proba(X)).max())
