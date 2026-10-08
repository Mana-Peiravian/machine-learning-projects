"""Task 2: Bagging with undersampling for imbalanced data."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import load_data, prepare_split, metric_dict
from task1_hddt import HDDT

class BaggingImbalanced:
    def __init__(self, base_learner='hddt', T=31, max_depth=3, min_samples_split=10, random_state=42):
        self.base_learner = base_learner
        self.T = T
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.models_ = []

    def _new_model(self, seed):
        if self.base_learner == 'hddt':
            return HDDT(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        if self.base_learner == 'dt':
            return DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=seed)
        raise ValueError("base_learner must be 'hddt' or 'dt'")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.random_state)
        min_idx = np.where(y == 1)[0]
        maj_idx = np.where(y == -1)[0]
        m = len(min_idx)
        self.models_ = []
        for t in range(self.T):
            seed = int(rng.integers(0, 2**31 - 1))
            min_sample = rng.choice(min_idx, size=m, replace=True)
            maj_sample = rng.choice(maj_idx, size=m, replace=False)
            idx = np.concatenate([min_sample, maj_sample])
            rng.shuffle(idx)
            model = self._new_model(seed)
            model.fit(X[idx], y[idx])
            self.models_.append(model)
        return self

    def predict_proba(self, X):
        probs = []
        for model in self.models_:
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X)
                if p.ndim == 2 and p.shape[1] == 2:
                    # sklearn may order classes as [-1, +1]
                    if hasattr(model, 'classes_'):
                        pos_col = list(model.classes_).index(1)
                        probs.append(p[:, pos_col])
                    else:
                        probs.append(p[:, 1])
                else:
                    probs.append(p)
            else:
                probs.append((model.predict(X) == 1).astype(float))
        pavg = np.mean(np.vstack(probs), axis=0)
        return np.vstack([1 - pavg, pavg]).T

    def predict(self, X):
        votes = np.vstack([m.predict(X) for m in self.models_])
        score = votes.sum(axis=0)
        return np.where(score >= 0, 1, -1)


def run_bagging_experiments(data_path='Covid.csv', seeds=range(10), save_dir=None):
    df = load_data(data_path)
    rows_T = []
    for T in [11,31,51,101]:
        for seed in seeds:
            Xtr, Xte, ytr, yte, *_ = prepare_split(df, seed=seed)
            clf = BaggingImbalanced(base_learner='hddt', T=T, max_depth=3, random_state=seed).fit(Xtr, ytr)
            pred = clf.predict(Xte); prob = clf.predict_proba(Xte)[:, 1]
            m = metric_dict(yte, pred, prob); m['T'] = T; m['seed'] = seed; rows_T.append(m)
    res_T = pd.DataFrame(rows_T)
    rows_base = []
    for base in ['hddt', 'dt']:
        for seed in seeds:
            Xtr, Xte, ytr, yte, *_ = prepare_split(df, seed=seed)
            clf = BaggingImbalanced(base_learner=base, T=51, max_depth=3, random_state=seed).fit(Xtr, ytr)
            pred = clf.predict(Xte); prob = clf.predict_proba(Xte)[:, 1]
            m = metric_dict(yte, pred, prob); m['Base Learner'] = 'HDDT (yours)' if base=='hddt' else 'Standard DT (sklearn)'; m['seed'] = seed; rows_base.append(m)
    res_base = pd.DataFrame(rows_base)
    if save_dir:
        res_T.to_csv(os.path.join(save_dir, 'task2_ensemble_size.csv'), index=False)
        res_base.to_csv(os.path.join(save_dir, 'task2_base_learner.csv'), index=False)
        summ = res_T.groupby('T')[['Accuracy','Precision (+1)','Recall (+1)','F1 (+1)','AUC-ROC','G-mean']].agg(['mean','std'])
        summ.to_csv(os.path.join(save_dir, 'task2_ensemble_size_summary.csv'))
        xs = [11,31,51,101]
        plt.figure(figsize=(7,4.2))
        plt.errorbar(xs, [summ.loc[x,('G-mean','mean')] for x in xs], yerr=[summ.loc[x,('G-mean','std')] for x in xs], marker='o', label='G-mean')
        plt.errorbar(xs, [summ.loc[x,('F1 (+1)','mean')] for x in xs], yerr=[summ.loc[x,('F1 (+1)','std')] for x in xs], marker='s', label='F1 (+1)')
        plt.xlabel('T (number of classifiers)'); plt.ylabel('Score'); plt.title('Bagging with undersampling: effect of T'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '../figures/task2_T.png'), dpi=160); plt.close()
    return res_T, res_base

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'Covid.csv'
    out = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out, exist_ok=True)
    res_T, res_base = run_bagging_experiments(path, save_dir=out)
    print(res_T.groupby('T')[['Accuracy','Precision (+1)','Recall (+1)','F1 (+1)','AUC-ROC','G-mean']].agg(['mean','std']))
    print(res_base.groupby('Base Learner')[['Accuracy','Precision (+1)','Recall (+1)','F1 (+1)','AUC-ROC','G-mean']].agg(['mean','std']))
