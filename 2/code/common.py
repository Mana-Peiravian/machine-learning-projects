"""Shared utilities for HW2 Ensemble Learning on Covid.csv."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

RANDOM_STATE = 42


def load_data(path='Covid.csv'):
    df = pd.read_csv(path)
    # Normalize labels to {-1,+1}; keep column names otherwise unchanged.
    df['Label'] = df['Label'].astype(int)
    return df


def missing_rates(df):
    return df.isnull().mean().sort_values(ascending=False) * 100.0


def infer_feature_types(df, target='Label'):
    features = [c for c in df.columns if c != target]
    # In this dataset age and PO2 are ordinal/continuous clinical variables; most other
    # columns are binary symptoms/comorbidities encoded as 0/1 with missing values.
    continuous = [c for c in features if c.strip().lower() in {'age', 'po2'}]
    categorical = [c for c in features if c not in continuous]
    return features, continuous, categorical


class Preprocessor:
    """Fit-on-train preprocessing: impute, Kendall feature selection, scale continuous features."""
    def __init__(self, target='Label', corr_threshold=0.5):
        self.target = target
        self.corr_threshold = corr_threshold
        self.features_ = None
        self.continuous_ = None
        self.categorical_ = None
        self.impute_values_ = {}
        self.selected_features_ = None
        self.dropped_features_ = []
        self.scaler_ = None
        self.selected_continuous_ = None

    def fit(self, train_df):
        self.features_, continuous, categorical = infer_feature_types(train_df, self.target)
        self.continuous_ = continuous
        self.categorical_ = categorical
        # Impute values learned only from the training split.
        for c in self.features_:
            s = train_df[c]
            if c in self.continuous_:
                self.impute_values_[c] = float(s.median())
            else:
                m = s.mode(dropna=True)
                self.impute_values_[c] = float(m.iloc[0]) if len(m) else 0.0
        filled = train_df[self.features_].fillna(self.impute_values_)
        corr_df = pd.concat([filled, train_df[[self.target]]], axis=1).corr(method='kendall')
        label_corr = corr_df[self.target].drop(self.target).abs().fillna(0.0)
        to_drop = set()
        feats = list(self.features_)
        for i in range(len(feats)):
            f1 = feats[i]
            if f1 in to_drop:
                continue
            for j in range(i + 1, len(feats)):
                f2 = feats[j]
                if f2 in to_drop:
                    continue
                tau = corr_df.loc[f1, f2]
                if pd.notna(tau) and abs(tau) > self.corr_threshold:
                    # Keep the feature with stronger absolute Kendall association to Label.
                    if label_corr.get(f1, 0.0) >= label_corr.get(f2, 0.0):
                        to_drop.add(f2)
                    else:
                        to_drop.add(f1)
                        break
        self.dropped_features_ = sorted(to_drop)
        self.selected_features_ = [f for f in self.features_ if f not in to_drop]
        self.selected_continuous_ = [f for f in self.continuous_ if f in self.selected_features_]
        self.scaler_ = StandardScaler() if self.selected_continuous_ else None
        if self.scaler_ is not None:
            X_sel = filled[self.selected_features_].copy()
            self.scaler_.fit(X_sel[self.selected_continuous_])
        return self

    def transform(self, df):
        X = df[self.features_].fillna(self.impute_values_)
        X = X[self.selected_features_].astype(float).copy()
        if self.scaler_ is not None and self.selected_continuous_:
            X.loc[:, self.selected_continuous_] = self.scaler_.transform(X[self.selected_continuous_])
        return X.to_numpy(dtype=float)

    def fit_transform(self, train_df):
        return self.fit(train_df).transform(train_df)


def prepare_split(df, seed=42, test_size=0.30):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['Label'], random_state=seed)
    pp = Preprocessor().fit(train_df)
    X_train = pp.transform(train_df)
    X_test = pp.transform(test_df)
    y_train = train_df['Label'].to_numpy(dtype=int)
    y_test = test_df['Label'].to_numpy(dtype=int)
    return X_train, X_test, y_train, y_test, pp, train_df, test_df


def metric_dict(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    gmean = float(np.sqrt(sens * spec))
    try:
        auc = roc_auc_score((y_true == 1).astype(int), y_score if y_score is not None else (y_pred == 1).astype(float))
    except ValueError:
        auc = np.nan
    return {'Accuracy': acc, 'Precision (+1)': prec, 'Recall (+1)': rec, 'F1 (+1)': f1, 'AUC-ROC': auc, 'G-mean': gmean}


def summarize(rows):
    df = pd.DataFrame(rows)
    return pd.DataFrame({'Mean': df.mean(numeric_only=True), 'Std Dev': df.std(numeric_only=True, ddof=1)})


def format_mean_std(mean, std):
    return f"{mean:.4f} ± {std:.4f}"
