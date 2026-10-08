"""NumPy/SciPy implementations used by the complete ML homework notebook.

The core estimators in this file do not depend on scikit-learn.  The emphasis
is transparent algorithms and controlled experiments rather than production
optimisation.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import expit, logsumexp


EPS = 1e-12


def _rng(random_state=42):
    return np.random.default_rng(random_state)


def stratified_split_indices(y, test_size=0.25, random_state=42):
    """Return train/test positions while preserving each class proportion."""
    y = np.asarray(y)
    rng = _rng(random_state)
    train, test = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = max(1, int(round(test_size * len(idx))))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return rng.permutation(train), rng.permutation(test)


def random_split_indices(n, test_size=0.25, random_state=42):
    idx = _rng(random_state).permutation(n)
    n_test = max(1, int(round(test_size * n)))
    return idx[n_test:], idx[:n_test]


def stratified_subsample_indices(y, max_n, random_state=42):
    """Proportional sample with at least one observation per represented class."""
    y = np.asarray(y)
    if len(y) <= max_n:
        return np.arange(len(y))
    rng = _rng(random_state)
    labels, counts = np.unique(y, return_counts=True)
    raw = counts / counts.sum() * max_n
    allocation = np.maximum(1, np.floor(raw).astype(int))
    while allocation.sum() > max_n:
        candidates = np.flatnonzero(allocation > 1)
        allocation[candidates[np.argmax(allocation[candidates] - raw[candidates])]] -= 1
    while allocation.sum() < max_n:
        candidates = np.flatnonzero(allocation < counts)
        allocation[candidates[np.argmax(raw[candidates] - allocation[candidates])]] += 1
    selected = []
    for label, k in zip(labels, allocation):
        idx = np.flatnonzero(y == label)
        selected.extend(rng.choice(idx, size=k, replace=False))
    return rng.permutation(selected)


class TabularPreprocessor:
    """Median/robust-scale numeric data and one-hot encode bounded categories.

    All statistics are learned in fit(), so using this object after a split
    avoids preprocessing leakage. Rare levels are collapsed into ``__OTHER__``.
    """

    def __init__(self, max_categories=30, min_category_count=5):
        self.max_categories = max_categories
        self.min_category_count = min_category_count

    def fit(self, X):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.numeric_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = [c for c in self.columns_ if c not in self.numeric_cols_]
        self.medians_, self.centers_, self.scales_, self.levels_ = {}, {}, {}, {}
        self.constant_cols_ = []
        for c in self.numeric_cols_:
            s = pd.to_numeric(X[c], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            q1, q3 = s.quantile([0.25, 0.75]) if s.notna().any() else (0.0, 1.0)
            scale = float(q3 - q1)
            if not np.isfinite(scale) or scale < EPS:
                scale = float(s.std(ddof=0))
            if not np.isfinite(scale) or scale < EPS:
                scale = 1.0
                self.constant_cols_.append(c)
            self.medians_[c], self.centers_[c], self.scales_[c] = med, med, scale
        for c in self.categorical_cols_:
            s = X[c].astype("string").fillna("__MISSING__")
            counts = s.value_counts()
            levels = counts[counts >= self.min_category_count].head(self.max_categories).index.tolist()
            if "__OTHER__" not in levels:
                levels.append("__OTHER__")
            self.levels_[c] = levels
        self.feature_names_ = list(self.numeric_cols_)
        for c in self.categorical_cols_:
            self.feature_names_.extend([f"{c}={v}" for v in self.levels_[c]])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reindex(columns=self.columns_).copy()
        blocks = []
        if self.numeric_cols_:
            numeric = np.column_stack([
                (pd.to_numeric(X[c], errors="coerce").fillna(self.medians_[c]).to_numpy(float)
                 - self.centers_[c]) / self.scales_[c]
                for c in self.numeric_cols_
            ])
            blocks.append(np.clip(numeric, -20, 20))
        for c in self.categorical_cols_:
            levels = self.levels_[c]
            valid = set(levels[:-1])
            values = X[c].astype("string").fillna("__MISSING__")
            values = values.where(values.isin(valid), "__OTHER__")
            blocks.append(np.column_stack([(values == level).to_numpy(float) for level in levels]))
        return np.hstack(blocks).astype(np.float64) if blocks else np.empty((len(X), 0))

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LabelEncoderScratch:
    def fit(self, y):
        self.classes_ = np.array(sorted(pd.Series(y).dropna().unique().tolist()), dtype=object)
        self.mapping_ = {v: i for i, v in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self.mapping_[v] for v in y], dtype=int)

    def inverse_transform(self, y):
        return self.classes_[np.asarray(y, dtype=int)]

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class LogisticRegressionScratch:
    """Binary or multinomial logistic regression trained with Adam."""

    def __init__(self, learning_rate=0.03, epochs=500, l2=1e-3,
                 batch_size=None, tol=1e-7, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.classes_ = np.unique(y)
        n, d = X.shape
        k = len(self.classes_)
        Y = np.eye(k)[np.searchsorted(self.classes_, y)]
        Xb = np.column_stack([np.ones(n), X])
        W = np.zeros((d + 1, k))
        m = np.zeros_like(W)
        v = np.zeros_like(W)
        rng = _rng(self.random_state)
        batch = min(n, self.batch_size or n)
        self.loss_history_ = []
        last = np.inf
        for epoch in range(1, self.epochs + 1):
            idx = rng.choice(n, batch, replace=False) if batch < n else np.arange(n)
            Z = Xb[idx] @ W
            P = np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
            grad = Xb[idx].T @ (P - Y[idx]) / len(idx)
            grad[1:] += self.l2 * W[1:]
            m = .9 * m + .1 * grad
            v = .999 * v + .001 * grad * grad
            mhat, vhat = m / (1 - .9 ** epoch), v / (1 - .999 ** epoch)
            W -= self.learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            if epoch == 1 or epoch % 10 == 0:
                Zall = Xb @ W
                loss = np.mean(logsumexp(Zall, axis=1) - np.sum(Y * Zall, axis=1))
                loss += .5 * self.l2 * np.sum(W[1:] ** 2)
                self.loss_history_.append(float(loss))
                if abs(last - loss) < self.tol:
                    break
                last = loss
        self.coef_ = W[1:].T
        self.intercept_ = W[0]
        return self

    def predict_proba(self, X):
        Z = np.asarray(X) @ self.coef_.T + self.intercept_
        return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LinearRegressionScratch:
    """Ridge-stabilised ordinary least squares solved by a linear system."""

    def __init__(self, l2=1e-6):
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        Xb = np.column_stack([np.ones(len(X)), X])
        penalty = np.eye(Xb.shape[1]) * self.l2
        penalty[0, 0] = 0
        try:
            beta = np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(Xb.T @ Xb + penalty) @ Xb.T @ y
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_


class KNNScratch:
    def __init__(self, n_neighbors=7, task="classification", weights="distance",
                 batch_size=1000):
        self.n_neighbors = n_neighbors
        self.task = task
        self.weights = weights
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_) if self.task == "classification" else None
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        outputs = []
        for start in range(0, len(X), self.batch_size):
            D = cdist(X[start:start + self.batch_size], self.X_, metric="euclidean")
            idx = np.argpartition(D, min(self.n_neighbors, len(self.X_)) - 1, axis=1)[:, :self.n_neighbors]
            dist = np.take_along_axis(D, idx, axis=1)
            targets = self.y_[idx]
            w = 1 / np.maximum(dist, 1e-9) if self.weights == "distance" else np.ones_like(dist)
            if self.task == "regression":
                pred = np.sum(w * targets, axis=1) / np.sum(w, axis=1)
            else:
                scores = np.stack([np.sum(w * (targets == c), axis=1) for c in self.classes_], axis=1)
                pred = self.classes_[np.argmax(scores, axis=1)]
            outputs.append(pred)
        return np.concatenate(outputs)


@dataclass
class _TreeNode:
    prediction: float | int
    feature: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class DecisionTreeScratch:
    """Greedy CART tree with Gini or variance reduction."""

    def __init__(self, task="classification", max_depth=8, min_samples_split=20,
                 min_samples_leaf=8, max_features=None, max_thresholds=32,
                 random_state=42):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y)
        self.rng_ = _rng(self.random_state)
        self.classes_ = np.unique(self.y_) if self.task == "classification" else None
        self.root_ = self._grow(np.arange(len(self.y_)), 0)
        del self.X_, self.y_
        return self

    def _prediction(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        if self.task == "regression":
            return float(np.var(y))
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(1 - np.sum(p * p))

    def _grow(self, idx, depth):
        y = self.y_[idx]
        node = _TreeNode(self._prediction(y))
        if (depth >= self.max_depth or len(idx) < self.min_samples_split
                or self._impurity(y) < EPS):
            return node
        n_features = self.X_.shape[1]
        if self.max_features is None:
            features = np.arange(n_features)
        else:
            m = min(n_features, int(self.max_features))
            features = self.rng_.choice(n_features, m, replace=False)
        parent = self._impurity(y)
        best_gain, best = 0.0, None
        for f in features:
            vals = self.X_[idx, f]
            unique = np.unique(vals)
            if len(unique) < 2:
                continue
            if len(unique) > self.max_thresholds:
                q = np.linspace(0.02, 0.98, self.max_thresholds)
                thresholds = np.unique(np.quantile(vals, q))
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2
            for threshold in thresholds:
                mask = vals <= threshold
                nl, nr = int(mask.sum()), int((~mask).sum())
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gain = parent - nl / len(idx) * self._impurity(y[mask]) - nr / len(idx) * self._impurity(y[~mask])
                if gain > best_gain:
                    best_gain, best = gain, (f, float(threshold), mask)
        if best is None:
            return node
        f, threshold, mask = best
        node.feature, node.threshold = f, threshold
        node.left = self._grow(idx[mask], depth + 1)
        node.right = self._grow(idx[~mask], depth + 1)
        return node

    def _predict_one(self, row):
        node = self.root_
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, X):
        return np.array([self._predict_one(row) for row in np.asarray(X)])


def pairwise_sq_dists(X, Y=None):
    X = np.asarray(X, float)
    Y = X if Y is None else np.asarray(Y, float)
    D = np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2 * X @ Y.T
    return np.maximum(D, 0)


def kernel_matrix(X, Y=None, kind="rbf", gamma=None, degree=2, coef0=1.0):
    X = np.asarray(X, float)
    Y = X if Y is None else np.asarray(Y, float)
    if kind == "linear":
        return X @ Y.T
    if kind == "poly":
        gamma = gamma if gamma is not None else 1 / max(1, X.shape[1])
        return (gamma * (X @ Y.T) + coef0) ** degree
    if kind == "rbf":
        gamma = gamma if gamma is not None else 1 / max(1, X.shape[1])
        return np.exp(-gamma * pairwise_sq_dists(X, Y))
    if kind == "laplacian":
        gamma = gamma if gamma is not None else 1 / math.sqrt(max(1, X.shape[1]))
        return np.exp(-gamma * cdist(X, Y, metric="cityblock"))
    raise ValueError(f"Unknown kernel: {kind}")


def median_gamma(X, max_points=1000, random_state=42):
    X = np.asarray(X)
    if len(X) > max_points:
        X = X[_rng(random_state).choice(len(X), max_points, replace=False)]
    d2 = pairwise_sq_dists(X)
    values = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(values[values > EPS]) if np.any(values > EPS) else 1.0
    return float(1 / max(med, EPS))


class KernelRidgeScratch:
    def __init__(self, kernel="rbf", gamma=None, degree=2, l2=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.l2 = l2

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.y_mean_ = float(np.mean(y))
        yc = np.asarray(y, float) - self.y_mean_
        self.gamma_ = self.gamma if self.gamma is not None else median_gamma(self.X_)
        K = kernel_matrix(self.X_, kind=self.kernel, gamma=self.gamma_, degree=self.degree)
        self.alpha_ = np.linalg.solve(K + self.l2 * np.eye(len(K)), yc)
        return self

    def predict(self, X):
        K = kernel_matrix(np.asarray(X), self.X_, kind=self.kernel,
                          gamma=self.gamma_, degree=self.degree)
        return K @ self.alpha_ + self.y_mean_


class KernelSVMScratch:
    """Multiclass one-vs-rest soft-margin SVM in a kernel basis.

    The representer coefficients minimise regularised squared hinge loss via
    Adam. This is an SVM objective; no external optimiser or ML library is used.
    """

    def __init__(self, kernel="rbf", gamma=None, degree=2, l2=1e-2,
                 learning_rate=0.03, epochs=250, random_state=42):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.gamma_ = self.gamma if self.gamma is not None else median_gamma(self.X_)
        K = kernel_matrix(self.X_, kind=self.kernel, gamma=self.gamma_, degree=self.degree)
        n = len(K)
        A = np.zeros((n, len(self.classes_)))
        b = np.zeros(len(self.classes_))
        mA, vA = np.zeros_like(A), np.zeros_like(A)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        for t in range(1, self.epochs + 1):
            F = K @ A + b
            Y = np.column_stack([np.where(y == c, 1.0, -1.0) for c in self.classes_])
            margin = 1 - Y * F
            active = margin > 0
            # squared hinge gives a stable differentiable subproblem
            Gf = -2 * Y * np.where(active, margin, 0) / n
            gradA = K @ Gf + self.l2 * (K @ A)
            gradb = Gf.sum(axis=0)
            mA, vA = .9 * mA + .1 * gradA, .999 * vA + .001 * gradA * gradA
            mb, vb = .9 * mb + .1 * gradb, .999 * vb + .001 * gradb * gradb
            A -= self.learning_rate * (mA / (1 - .9 ** t)) / (np.sqrt(vA / (1 - .999 ** t)) + 1e-8)
            b -= self.learning_rate * (mb / (1 - .9 ** t)) / (np.sqrt(vb / (1 - .999 ** t)) + 1e-8)
        self.alpha_, self.intercept_ = A, b
        return self

    def decision_function(self, X):
        K = kernel_matrix(np.asarray(X), self.X_, kind=self.kernel,
                          gamma=self.gamma_, degree=self.degree)
        return K @ self.alpha_ + self.intercept_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class KernelKNNScratch:
    def __init__(self, n_neighbors=7, task="classification", kernel="rbf",
                 gamma=None, degree=2):
        self.n_neighbors = n_neighbors
        self.task = task
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def fit(self, X, y):
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(self.y_) if self.task == "classification" else None
        self.gamma_ = self.gamma if self.gamma is not None else median_gamma(self.X_)
        Ktrain = kernel_matrix(self.X_, kind=self.kernel, gamma=self.gamma_, degree=self.degree)
        self.diag_train_ = np.diag(Ktrain)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        K = kernel_matrix(X, self.X_, kind=self.kernel, gamma=self.gamma_, degree=self.degree)
        diag_test = np.diag(kernel_matrix(X, kind=self.kernel, gamma=self.gamma_, degree=self.degree))
        D2 = np.maximum(diag_test[:, None] + self.diag_train_[None, :] - 2 * K, 0)
        k = min(self.n_neighbors, len(self.X_))
        idx = np.argpartition(D2, k - 1, axis=1)[:, :k]
        dist = np.sqrt(np.take_along_axis(D2, idx, axis=1))
        target = self.y_[idx]
        w = 1 / np.maximum(dist, 1e-9)
        if self.task == "regression":
            return np.sum(w * target, axis=1) / np.sum(w, axis=1)
        scores = np.stack([np.sum(w * (target == c), axis=1) for c in self.classes_], axis=1)
        return self.classes_[np.argmax(scores, axis=1)]


class KernelPCAScratch:
    def __init__(self, n_components=10, kernel="rbf", gamma=None, degree=2):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def fit(self, X):
        self.X_ = np.asarray(X, float)
        self.gamma_ = self.gamma if self.gamma is not None else median_gamma(self.X_)
        K = kernel_matrix(self.X_, kind=self.kernel, gamma=self.gamma_, degree=self.degree)
        self.train_col_mean_ = K.mean(axis=0)
        self.train_mean_ = K.mean()
        Kc = K - K.mean(axis=0)[None, :] - K.mean(axis=1)[:, None] + K.mean()
        vals, vecs = eigh(Kc, subset_by_index=[max(0, len(Kc) - self.n_components), len(Kc) - 1])
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        keep = vals > 1e-10
        self.eigenvalues_ = vals[keep]
        self.alphas_ = vecs[:, keep] / np.sqrt(self.eigenvalues_)[None, :]
        return self

    def transform(self, X):
        K = kernel_matrix(np.asarray(X), self.X_, kind=self.kernel,
                          gamma=self.gamma_, degree=self.degree)
        row_mean = K.mean(axis=1)
        Kc = K - self.train_col_mean_[None, :] - row_mean[:, None] + self.train_mean_
        return Kc @ self.alphas_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class IsolationForestScratch:
    """Small isolation forest implementation for anomaly diagnostics."""

    def __init__(self, n_trees=100, sample_size=256, max_depth=None, random_state=42):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, float)
        self.rng_ = _rng(self.random_state)
        self.sample_size_ = min(self.sample_size, len(X))
        self.max_depth_ = self.max_depth or int(math.ceil(math.log2(max(2, self.sample_size_))))
        self.trees_ = []
        for _ in range(self.n_trees):
            sample = X[self.rng_.choice(len(X), self.sample_size_, replace=False)]
            self.trees_.append(self._build(sample, 0))
        return self

    def _build(self, X, depth):
        if depth >= self.max_depth_ or len(X) <= 1 or np.all(np.ptp(X, axis=0) < EPS):
            return ("leaf", len(X))
        valid = np.flatnonzero(np.ptp(X, axis=0) > EPS)
        feature = int(self.rng_.choice(valid))
        lo, hi = X[:, feature].min(), X[:, feature].max()
        split = float(self.rng_.uniform(lo, hi))
        mask = X[:, feature] < split
        if mask.all() or (~mask).all():
            return ("leaf", len(X))
        return ("node", feature, split, self._build(X[mask], depth + 1),
                self._build(X[~mask], depth + 1))

    @staticmethod
    def _c(n):
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def _path(self, row, node, depth=0):
        if node[0] == "leaf":
            return depth + self._c(node[1])
        _, feature, split, left, right = node
        return self._path(row, left if row[feature] < split else right, depth + 1)

    def score_samples(self, X):
        X = np.asarray(X, float)
        paths = np.array([[self._path(row, tree) for tree in self.trees_] for row in X])
        return 2 ** (-paths.mean(axis=1) / max(self._c(self.sample_size_), EPS))

    def predict(self, X, contamination=0.01):
        score = self.score_samples(X)
        return score >= np.quantile(score, 1 - contamination)


def zscore_flags(values, threshold=3.0):
    x = np.asarray(values, float)
    z = np.abs((x - np.nanmean(x)) / max(np.nanstd(x), EPS))
    return z > threshold


def iqr_flags(values, factor=1.5):
    x = np.asarray(values, float)
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - factor * iqr) | (x > q3 + factor * iqr)


def outlier_comparison(frame, columns, contamination=0.01, random_state=42):
    """Compare univariate Z/IQR flags with multivariate isolation anomalies."""
    data = frame[columns].apply(pd.to_numeric, errors="coerce")
    data = data.fillna(data.median()).fillna(0)
    scale = data.quantile(.75) - data.quantile(.25)
    Z = (data - data.median()) / scale.replace(0, 1)
    forest = IsolationForestScratch(random_state=random_state).fit(Z.to_numpy())
    iso = forest.predict(Z.to_numpy(), contamination=contamination)
    rows = []
    for c in columns:
        z = zscore_flags(data[c])
        iqr = iqr_flags(data[c])
        rows.append({
            "feature": c,
            "zscore_count": int(z.sum()),
            "iqr_count": int(iqr.sum()),
            "isolation_count": int(iso.sum()),
            "z_iqr_jaccard": float((z & iqr).sum() / max(1, (z | iqr).sum())),
            "z_iso_jaccard": float((z & iso).sum() / max(1, (z | iso).sum())),
            "iqr_iso_jaccard": float((iqr & iso).sum() / max(1, (iqr | iso).sum())),
        })
    return pd.DataFrame(rows), pd.Series(iso, index=frame.index, name="isolation_anomaly")


def missingness_associations(frame, column, candidate_columns, max_rows=20000,
                             random_state=42):
    """Test whether a column's missingness is associated with observed fields.

    Numeric fields use Mann-Whitney U and rank-biserial effect. Categorical
    fields use chi-square and Cramer's V. Small p-values reject independence,
    providing evidence against MCAR (but cannot prove MNAR).
    """
    df = frame[[column] + list(candidate_columns)].copy()
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state)
    missing = df[column].isna()
    rows = []
    if missing.nunique() < 2:
        return pd.DataFrame(columns=["missing_column", "observed_feature", "test", "p_value", "effect"])
    for c in candidate_columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            a, b = s[missing].dropna(), s[~missing].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect = 2 * u / (len(a) * len(b)) - 1
            test = "Mann-Whitney U"
        else:
            table = pd.crosstab(s.astype("string").fillna("__MISSING__"), missing)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2, p, _, _ = stats.chi2_contingency(table)
            n = table.to_numpy().sum()
            effect = math.sqrt(chi2 / max(n * (min(table.shape) - 1), EPS))
            test = "Chi-square / Cramer's V"
        rows.append({"missing_column": column, "observed_feature": c, "test": test,
                     "p_value": float(p), "effect": float(effect)})
    return pd.DataFrame(rows).sort_values(["p_value", "effect"], ascending=[True, False])


def near_constant_report(frame, threshold=0.995):
    rows = []
    for c in frame.columns:
        counts = frame[c].value_counts(dropna=False, normalize=True)
        rows.append({
            "feature": c,
            "n_unique": int(frame[c].nunique(dropna=False)),
            "dominant_fraction": float(counts.iloc[0]) if len(counts) else 1.0,
            "near_constant": bool(len(counts) <= 1 or counts.iloc[0] >= threshold),
        })
    return pd.DataFrame(rows).sort_values(["near_constant", "dominant_fraction"], ascending=False)


def redundant_numeric_report(frame, threshold=0.90):
    numeric = frame.select_dtypes(include=np.number)
    corr = numeric.corr(method="spearman").abs()
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if corr.loc[a, b] >= threshold:
                rows.append({"feature_1": a, "feature_2": b,
                             "abs_spearman": float(corr.loc[a, b])})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "abs_spearman"])


def mutual_information_ranking(frame, target, numeric_bins=8):
    """Estimate univariate mutual information by empirical discretisation."""
    y = pd.Series(target).reset_index(drop=True).astype("string")
    rows = {}
    for c in frame.columns:
        s = frame[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > numeric_bins:
            try:
                x = pd.qcut(s.rank(method="first"), q=numeric_bins, duplicates="drop").astype("string")
            except ValueError:
                x = s.astype("string")
        else:
            x = s.astype("string")
        x = x.fillna("__MISSING__")
        joint = pd.crosstab(x, y, normalize=True)
        px = joint.sum(axis=1).to_numpy()[:, None]
        py = joint.sum(axis=0).to_numpy()[None, :]
        pxy = joint.to_numpy()
        mask = pxy > 0
        rows[c] = float(np.sum(pxy[mask] * np.log(
            pxy[mask] / np.broadcast_to(px @ py, pxy.shape)[mask]
        )))
    return pd.Series(rows, name="mutual_information").sort_values(ascending=False)


def accuracy_score(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def confusion_matrix(y, pred, labels=None):
    y, pred = np.asarray(y), np.asarray(pred)
    labels = np.unique(np.r_[y, pred]) if labels is None else np.asarray(labels)
    mapping = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(y, pred):
        cm[mapping[a], mapping[b]] += 1
    return cm, labels


def classification_metrics(y, pred):
    cm, labels = confusion_matrix(y, pred)
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def regression_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": float(1 - np.sum(err * err) / max(np.sum((y - y.mean()) ** 2), EPS)),
    }


def centered_kernel_alignment(K, y, task="classification"):
    K = np.asarray(K, float)
    H = np.eye(len(K)) - np.ones_like(K) / len(K)
    Kc = H @ K @ H
    if task == "classification":
        y = np.asarray(y)
        Y = (y[:, None] == y[None, :]).astype(float)
    else:
        yc = np.asarray(y, float) - np.mean(y)
        Y = np.outer(yc, yc)
    Yc = H @ Y @ H
    return float(np.sum(Kc * Yc) / max(np.linalg.norm(Kc) * np.linalg.norm(Yc), EPS))


def kernel_geometry(X, y, kernels=("linear", "poly", "rbf"), task="classification"):
    rows = []
    gamma = median_gamma(X)
    for kind in kernels:
        K = kernel_matrix(X, kind=kind, gamma=gamma)
        vals = np.linalg.eigvalsh((K + K.T) / 2)
        vals = np.maximum(vals, 0)
        effective_rank = (vals.sum() ** 2) / max(np.sum(vals * vals), EPS)
        off = K[~np.eye(len(K), dtype=bool)]
        rows.append({
            "kernel": kind,
            "gamma": gamma,
            "alignment": centered_kernel_alignment(K, y, task),
            "effective_rank": float(effective_rank),
            "off_diagonal_mean": float(np.mean(off)),
            "off_diagonal_std": float(np.std(off)),
        })
    return pd.DataFrame(rows)


def benchmark_fit_predict(model, X_train, y_train, X_test):
    tracemalloc.start()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test)
    predict_seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return pred, {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_python_memory_mb": peak / 1024 ** 2,
    }


def worst_classification_cases(raw_test, y, pred, n=10):
    result = raw_test.copy().reset_index(drop=True)
    result["actual"] = np.asarray(y)
    result["predicted"] = np.asarray(pred)
    result["error"] = result["actual"] != result["predicted"]
    return result.loc[result["error"]].head(n)


def worst_regression_cases(raw_test, y, pred, n=10):
    result = raw_test.copy().reset_index(drop=True)
    result["actual"] = np.asarray(y, float)
    result["predicted"] = np.asarray(pred, float)
    result["absolute_error"] = np.abs(result["actual"] - result["predicted"])
    result["relative_error"] = result["absolute_error"] / np.maximum(np.abs(result["actual"]), 1)
    return result.sort_values(["absolute_error", "relative_error"], ascending=False).head(n)
