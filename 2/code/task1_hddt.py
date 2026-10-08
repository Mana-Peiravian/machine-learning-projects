"""Task 1: Hellinger Distance Decision Tree from scratch."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import load_data, prepare_split, metric_dict, summarize

class Node:
    __slots__ = ('is_leaf','prediction','proba','feature','threshold','left','right','depth')
    def __init__(self, is_leaf=False, prediction=-1, proba=0.0, feature=None, threshold=None, left=None, right=None, depth=0):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.proba = proba
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth

class HDDT:
    def __init__(self, min_samples_split=10, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.root_ = self._build(X, y, depth=0)
        return self

    def _leaf(self, y, depth):
        pos = np.sum(y == 1)
        neg = np.sum(y == -1)
        pred = 1 if pos >= neg else -1
        proba = pos / len(y) if len(y) else 0.0
        return Node(True, pred, proba, depth=depth)

    def _build(self, X, y, depth):
        if len(y) < self.min_samples_split or len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return self._leaf(y, depth)
        feat, thr, score = self._best_split(X, y)
        if feat is None or score <= 0:
            return self._leaf(y, depth)
        mask = X[:, feat] <= thr
        if mask.sum() == 0 or mask.sum() == len(y):
            return self._leaf(y, depth)
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        pos = np.sum(y == 1)
        pred = 1 if pos >= len(y) - pos else -1
        return Node(False, pred, pos / len(y), feat, thr, left, right, depth)

    def _best_split(self, X, y):
        P = np.sum(y == 1)
        N = np.sum(y == -1)
        if P == 0 or N == 0:
            return None, None, -np.inf
        best_feature, best_threshold, best_score = None, None, -np.inf
        n_features = X.shape[1]
        for f in range(n_features):
            order = np.argsort(X[:, f], kind='mergesort')
            xs = X[order, f]
            ys = y[order]
            # Candidate cuts after positions where feature value changes.
            diff_idx = np.flatnonzero(xs[:-1] != xs[1:])
            if len(diff_idx) == 0:
                continue
            pos_cum = np.cumsum(ys == 1)
            neg_cum = np.cumsum(ys == -1)
            pL = pos_cum[diff_idx].astype(float)
            nL = neg_cum[diff_idx].astype(float)
            pR = P - pL
            nR = N - nL
            scores = (np.sqrt(pL / P) - np.sqrt(nL / N))**2 + (np.sqrt(pR / P) - np.sqrt(nR / N))**2
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                cut = diff_idx[idx]
                best_score = float(scores[idx])
                best_feature = f
                best_threshold = float((xs[cut] + xs[cut + 1]) / 2.0)
        return best_feature, best_threshold, best_score

    def _predict_one(self, x, proba=False):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.proba if proba else node.prediction

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, False) for x in X], dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = np.array([self._predict_one(x, True) for x in X], dtype=float)
        return np.vstack([1 - p, p]).T


def run_hddt_experiments(data_path='Covid.csv', depths=(None,2,3,4,5), seeds=range(10), save_dir=None):
    df = load_data(data_path)
    all_rows = []
    for depth in depths:
        rows = []
        for seed in seeds:
            Xtr, Xte, ytr, yte, pp, *_ = prepare_split(df, seed=seed)
            clf = HDDT(max_depth=depth, min_samples_split=10).fit(Xtr, ytr)
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]
            m = metric_dict(yte, pred, proba)
            m['max_depth'] = 'None' if depth is None else depth
            m['seed'] = seed
            rows.append(m); all_rows.append(m)
        if save_dir:
            pd.DataFrame(rows).to_csv(os.path.join(save_dir, f'task1_depth_{"None" if depth is None else depth}.csv'), index=False)
    result = pd.DataFrame(all_rows)
    if save_dir:
        summary = result.groupby('max_depth').agg(['mean','std'])
        summary.to_csv(os.path.join(save_dir, 'task1_pruning_summary.csv'))
        plot_df = result.groupby('max_depth')[['G-mean','F1 (+1)']].agg(['mean','std'])
        order = ['None',2,3,4,5]
        labels = [str(x) for x in order]
        xs = np.arange(len(labels))
        plt.figure(figsize=(7,4.2))
        plt.errorbar(xs, [plot_df.loc[x,('G-mean','mean')] for x in order], yerr=[plot_df.loc[x,('G-mean','std')] for x in order], marker='o', label='G-mean')
        plt.errorbar(xs, [plot_df.loc[x,('F1 (+1)','mean')] for x in order], yerr=[plot_df.loc[x,('F1 (+1)','std')] for x in order], marker='s', label='F1 (+1)')
        plt.xticks(xs, labels); plt.xlabel('max_depth'); plt.ylabel('Score'); plt.title('HDDT pruning experiment'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '../figures/task1_pruning.png'), dpi=160)
        plt.close()
    return result

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'Covid.csv'
    out = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out, exist_ok=True)
    res = run_hddt_experiments(path, save_dir=out)
    print(res.groupby('max_depth')[['Accuracy','Precision (+1)','Recall (+1)','F1 (+1)','AUC-ROC','G-mean']].agg(['mean','std']))
