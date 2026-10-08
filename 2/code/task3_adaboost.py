"""Task 3: AdaBoost.M1 and SMOTE from scratch (except decision stumps)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import load_data, prepare_split, metric_dict

class AdaBoostM1:
    def __init__(self, T=50, learning_rate=1.0, random_state=42):
        self.T = T
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.learners_ = []
        self.alphas_ = []
        self.train_errors_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
        n = len(y)
        w = np.ones(n) / n
        rng = np.random.default_rng(self.random_state)
        self.learners_, self.alphas_, self.train_errors_ = [], [], []
        for t in range(self.T):
            stump = DecisionTreeClassifier(max_depth=1, random_state=int(rng.integers(0, 2**31-1)))
            stump.fit(X, y, sample_weight=w)
            pred = stump.predict(X).astype(int)
            eps = float(np.sum(w * (pred != y)))
            eps = min(max(eps, 1e-12), 1 - 1e-12)
            if eps >= 0.5:
                break
            alpha = self.learning_rate * 0.5 * np.log((1 - eps) / eps)
            w *= np.exp(-alpha * y * pred)
            w /= w.sum()
            self.learners_.append(stump); self.alphas_.append(alpha)
            self.train_errors_.append(float(np.mean(self.predict(X) != y)))
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if not self.learners_:
            return np.zeros(X.shape[0])
        scores = np.zeros(X.shape[0])
        for a, h in zip(self.alphas_, self.learners_):
            scores += a * h.predict(X)
        return scores

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def predict_proba(self, X):
        s = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-2.0 * s))
        return np.vstack([1 - p, p]).T


def smote(X, y, k=5, random_state=42):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(random_state)
    min_X = X[y == 1]
    maj_count = np.sum(y == -1)
    min_count = len(min_X)
    need = maj_count - min_count
    if need <= 0:
        return X.copy(), y.copy()
    # Pairwise Euclidean distances among minority samples, no sklearn neighbors.
    diff = min_X[:, None, :] - min_X[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    kk = min(k, max(1, min_count - 1))
    neigh = np.argsort(dist, axis=1)[:, :kk]
    synthetic = []
    for _ in range(need):
        i = int(rng.integers(0, min_count))
        j = int(rng.choice(neigh[i]))
        lam = rng.random()
        synthetic.append(min_X[i] + lam * (min_X[j] - min_X[i]))
    X_new = np.vstack([X, np.vstack(synthetic)])
    y_new = np.concatenate([y, np.ones(need, dtype=int)])
    perm = rng.permutation(len(y_new))
    return X_new[perm], y_new[perm]


def run_adaboost_experiments(data_path='Covid.csv', seeds=range(10), save_dir=None):
    df = load_data(data_path)
    # Precompute splits once; all preprocessing is still fitted only on each training fold.
    splits = {}
    for seed in seeds:
        splits[seed] = prepare_split(df, seed=seed)[:4]

    rows_A = []
    for method in ['AdaBoost (no SMOTE)', 'AdaBoost + SMOTE']:
        for seed in seeds:
            Xtr, Xte, ytr, yte = splits[seed]
            if method == 'AdaBoost + SMOTE':
                Xfit, yfit = smote(Xtr, ytr, k=5, random_state=seed)
            else:
                Xfit, yfit = Xtr, ytr
            clf = AdaBoostM1(T=50, random_state=seed).fit(Xfit, yfit)
            pred = clf.predict(Xte); prob = clf.predict_proba(Xte)[:,1]
            m = metric_dict(yte, pred, prob); m['Method'] = method; m['seed'] = seed; rows_A.append(m)
    res_A = pd.DataFrame(rows_A)

    rows_B = []
    for T in [10,25,50,100]:
        for seed in seeds:
            Xtr, Xte, ytr, yte = splits[seed]
            Xfit, yfit = smote(Xtr, ytr, k=5, random_state=seed)
            clf = AdaBoostM1(T=T, random_state=seed).fit(Xfit, yfit)
            train_pred = clf.predict(Xfit); test_pred = clf.predict(Xte)
            m = metric_dict(yte, test_pred, clf.predict_proba(Xte)[:,1])
            m['T'] = T; m['seed'] = seed; m['Train error'] = float(np.mean(train_pred != yfit)); m['Test error'] = float(np.mean(test_pred != yte)); rows_B.append(m)
    res_B = pd.DataFrame(rows_B)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, '../figures'), exist_ok=True)
        res_A.to_csv(os.path.join(save_dir, 'task3_smote_comparison.csv'), index=False)
        res_B.to_csv(os.path.join(save_dir, 'task3_rounds.csv'), index=False)
        summ = res_B.groupby('T')[['Train error','Test error','F1 (+1)']].agg(['mean','std'])
        summ.to_csv(os.path.join(save_dir, 'task3_rounds_summary.csv'))
        xs = [10,25,50,100]
        plt.figure(figsize=(7,4.2))
        plt.errorbar(xs, [summ.loc[x,('Train error','mean')] for x in xs], yerr=[summ.loc[x,('Train error','std')] for x in xs], marker='o', label='Train error')
        plt.errorbar(xs, [summ.loc[x,('Test error','mean')] for x in xs], yerr=[summ.loc[x,('Test error','std')] for x in xs], marker='s', label='Test error')
        plt.errorbar(xs, [summ.loc[x,('F1 (+1)','mean')] for x in xs], yerr=[summ.loc[x,('F1 (+1)','std')] for x in xs], marker='^', label='F1 (+1)')
        plt.xlabel('T (boosting rounds)'); plt.ylabel('Score / error'); plt.title('AdaBoost + SMOTE: effect of T'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '../figures/task3_rounds.png'), dpi=160); plt.close()
    return res_A, res_B

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'Covid.csv'
    out = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out, exist_ok=True)
    res_A, res_B = run_adaboost_experiments(path, save_dir=out)
    print(res_A.groupby('Method')[['Accuracy','Precision (+1)','Recall (+1)','F1 (+1)','AUC-ROC','G-mean']].agg(['mean','std']))
    print(res_B.groupby('T')[['Train error','Test error','F1 (+1)','AUC-ROC','G-mean']].agg(['mean','std']))
