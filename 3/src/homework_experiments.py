"""Experiment orchestration for the integrated homework notebook."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.scratch_ml import (
    DecisionTreeScratch,
    KernelKNNScratch,
    KernelPCAScratch,
    KernelRidgeScratch,
    KernelSVMScratch,
    KNNScratch,
    LabelEncoderScratch,
    LinearRegressionScratch,
    LogisticRegressionScratch,
    TabularPreprocessor,
    classification_metrics,
    kernel_geometry,
    random_split_indices,
    regression_metrics,
    stratified_split_indices,
    stratified_subsample_indices,
    worst_classification_cases,
    worst_regression_cases,
)


SEED = 42


@dataclass
class PreparedTask:
    name: str
    dataset: str
    task_type: str
    target_name: str
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train_model: np.ndarray
    y_test_model: np.ndarray
    y_test_eval: np.ndarray
    inverse_target: object
    label_encoder: object
    preprocessor: TabularPreprocessor
    full_rows: int


def _subsample_positions(y, max_n, classification, seed):
    if len(y) <= max_n:
        return np.arange(len(y))
    if classification:
        return stratified_subsample_indices(y, max_n, seed)
    return np.random.default_rng(seed).choice(len(y), max_n, replace=False)


def _split_frame(df, target, task_type, strategy="random", test_size=.25,
                 seed=SEED, group_column=None):
    if group_column is not None:
        groups = df[group_column].astype("string").fillna("__MISSING_GROUP__")
        unique_groups = groups.unique()
        shuffled = np.random.default_rng(seed).permutation(unique_groups)
        n_test_groups = max(1, int(round(test_size * len(shuffled))))
        test_groups = set(shuffled[:n_test_groups])
        test_mask = groups.isin(test_groups).to_numpy()
        return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    if strategy == "time":
        order = np.argsort(df["days_from_start"].to_numpy())
        cut = int((1 - test_size) * len(order))
        return order[:cut], order[cut:]
    if task_type == "classification":
        return stratified_split_indices(df[target], test_size, seed)
    return random_split_indices(len(df), test_size, seed)


def _prepare_from_frame(name, dataset, df, target, task_type, drop_columns,
                        model_target=None, inverse_target=None, split_strategy="random",
                        train_cap=700, test_cap=300, seed=SEED, group_column=None):
    df = df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    train_idx, test_idx = _split_frame(
        df, target, task_type, split_strategy, seed=seed,
        group_column=group_column,
    )
    if group_column is not None:
        train_groups = set(df.iloc[train_idx][group_column].astype("string"))
        test_groups = set(df.iloc[test_idx][group_column].astype("string"))
        assert train_groups.isdisjoint(test_groups), "Group leakage across train/test split"
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    classification = task_type == "classification"
    train_pos = _subsample_positions(train_df[target], train_cap, classification, seed)
    test_pos = _subsample_positions(test_df[target], test_cap, classification, seed + 1)
    train_df, test_df = train_df.iloc[train_pos].copy(), test_df.iloc[test_pos].copy()

    feature_cols = [c for c in df.columns if c not in set(drop_columns)]
    X_train_raw, X_test_raw = train_df[feature_cols], test_df[feature_cols]
    prep = TabularPreprocessor(max_categories=18, min_category_count=4)
    X_train = prep.fit_transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    if classification:
        encoder = LabelEncoderScratch().fit(train_df[target])
        y_train_model = encoder.transform(train_df[target])
        # All labels should be represented due proportional splitting.
        y_test_model = encoder.transform(test_df[target])
        y_test_eval = y_test_model.copy()
        inverse = lambda z: encoder.inverse_transform(np.asarray(z, int))
    else:
        encoder = None
        model_target = model_target or target
        y_train_model = train_df[model_target].to_numpy(float)
        y_test_model = test_df[model_target].to_numpy(float)
        y_test_eval = test_df[target].to_numpy(float)
        inverse = inverse_target or (lambda z: np.asarray(z, float))

    return PreparedTask(
        name=name, dataset=dataset, task_type=task_type, target_name=target,
        X_train_raw=X_train_raw.reset_index(drop=True),
        X_test_raw=X_test_raw.reset_index(drop=True),
        X_train=X_train, X_test=X_test,
        y_train_model=np.asarray(y_train_model),
        y_test_model=np.asarray(y_test_model),
        y_test_eval=np.asarray(y_test_eval),
        inverse_target=inverse, label_encoder=encoder, preprocessor=prep,
        full_rows=len(df),
    )


def load_prepared_tasks(processed_dir="processed/model_tables", seed=SEED):
    base = Path(processed_dir)
    tasks = {}

    air_reg = pd.read_csv(base / "airbnb_regression.csv")
    tasks["airbnb_price"] = _prepare_from_frame(
        "airbnb_price", "Airbnb", air_reg, "price", "regression",
        drop_columns=["host_group_id", "price", "log_price", "price_winsorized"],
        model_target="log_price", inverse_target=lambda z: np.maximum(np.expm1(z), 0),
        train_cap=700, test_cap=300, seed=seed, group_column="host_group_id",
    )
    air_cls = pd.read_csv(base / "airbnb_superhost_classification.csv")
    tasks["airbnb_superhost"] = _prepare_from_frame(
        "airbnb_superhost", "Airbnb", air_cls, "superhost", "classification",
        drop_columns=["host_group_id", "superhost"], train_cap=800, test_cap=300,
        seed=seed, group_column="host_group_id",
    )

    nyc_cls = pd.read_csv(base / "nyc311_complaint_classification_sample.csv")
    tasks["nyc_complaint"] = _prepare_from_frame(
        "nyc_complaint", "NYC 311", nyc_cls, "complaint_category", "classification",
        drop_columns=["complaint_category"], split_strategy="time",
        train_cap=420, test_cap=260, seed=seed,
    )
    nyc_reg = pd.read_csv(base / "nyc311_resolution_regression_sample.csv")
    tasks["nyc_resolution"] = _prepare_from_frame(
        "nyc_resolution", "NYC 311", nyc_reg, "resolution_hours", "regression",
        drop_columns=["resolution_hours", "resolution_hours_capped", "log_resolution_hours"],
        model_target="log_resolution_hours", inverse_target=lambda z: np.maximum(np.expm1(z), 0),
        split_strategy="time", train_cap=650, test_cap=300, seed=seed,
    )

    hr_train = pd.read_csv(base / "hr_train_original.csv")
    hr_test = pd.read_csv(base / "hr_test_untouched.csv")
    feature_cols = [c for c in hr_train.columns if c != "Attrition"]
    prep = TabularPreprocessor(max_categories=20, min_category_count=3)
    Xtr, Xte = prep.fit_transform(hr_train[feature_cols]), prep.transform(hr_test[feature_cols])
    tasks["hr_attrition"] = PreparedTask(
        name="hr_attrition", dataset="IBM HR", task_type="classification",
        target_name="Attrition", X_train_raw=hr_train[feature_cols].reset_index(drop=True),
        X_test_raw=hr_test[feature_cols].reset_index(drop=True),
        X_train=Xtr, X_test=Xte,
        y_train_model=hr_train["Attrition"].to_numpy(int),
        y_test_model=hr_test["Attrition"].to_numpy(int),
        y_test_eval=hr_test["Attrition"].to_numpy(int),
        inverse_target=lambda z: np.asarray(z, int), label_encoder=None,
        preprocessor=prep, full_rows=len(hr_train) + len(hr_test),
    )

    retail_cls = pd.read_csv(base / "retail_future_segment_classification.csv")
    tasks["retail_segment"] = _prepare_from_frame(
        "retail_segment", "Online Retail", retail_cls, "customer_segment", "classification",
        drop_columns=["CustomerID", "customer_segment"],
        train_cap=800, test_cap=300, seed=seed,
    )
    retail_reg = pd.read_csv(base / "retail_future_spending_regression.csv")
    tasks["retail_spending"] = _prepare_from_frame(
        "retail_spending", "Online Retail", retail_reg, "future_net_spending", "regression",
        drop_columns=["CustomerID", "future_net_spending", "log_future_spending"],
        model_target="log_future_spending", inverse_target=lambda z: np.maximum(np.expm1(z), 0),
        train_cap=700, test_cap=300, seed=seed,
    )
    return tasks


class KPCAClassifier:
    def __init__(self, kernel="rbf", n_components=12, gamma=None):
        self.kernel = kernel
        self.n_components = n_components
        self.gamma = gamma

    def fit(self, X, y):
        self.kpca_ = KernelPCAScratch(
            n_components=min(self.n_components, len(X) - 1),
            kernel=self.kernel, gamma=self.gamma,
        )
        Z = self.kpca_.fit_transform(X)
        self.classifier_ = LogisticRegressionScratch(
            learning_rate=.03, epochs=250, l2=1e-2, batch_size=None
        ).fit(Z, y)
        return self

    def predict(self, X):
        return self.classifier_.predict(self.kpca_.transform(X))


def _benchmark(model, task):
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(task.X_train, task.y_train_model)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_model = model.predict(task.X_test)
    predict_seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if task.task_type == "classification":
        pred_eval = np.asarray(pred_model, int)
        metrics = classification_metrics(task.y_test_eval, pred_eval)
    else:
        lower, upper = np.quantile(task.y_train_model, [0.001, 0.999])
        clipped_model = np.clip(np.asarray(pred_model, float), lower, upper)
        pred_eval = np.asarray(task.inverse_target(clipped_model), float)
        pred_eval = np.nan_to_num(pred_eval, nan=0.0, posinf=np.finfo(float).max / 100)
        metrics = regression_metrics(task.y_test_eval, pred_eval)
        metrics["prediction_clipped_fraction"] = float(np.mean(clipped_model != pred_model))
    timing = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_python_memory_mb": peak / 1024 ** 2,
    }
    return pred_eval, metrics, timing


def models_for_task(task):
    d = task.X_train.shape[1]
    tree = DecisionTreeScratch(
        task=task.task_type, max_depth=7, min_samples_split=24,
        min_samples_leaf=8, max_features=min(45, d), max_thresholds=20,
        random_state=SEED,
    )
    knn = KNNScratch(n_neighbors=9, task=task.task_type)
    models = {}
    if task.task_type == "classification":
        svm_epochs = 55 if len(np.unique(task.y_train_model)) > 5 else 90
        models.update({
            "Logistic Regression": LogisticRegressionScratch(
                learning_rate=.03, epochs=300, l2=1e-2
            ),
            "KNN": knn,
            "Decision Tree": tree,
            "Kernel SVM [linear]": KernelSVMScratch(
                kernel="linear", l2=2e-2, learning_rate=.02, epochs=svm_epochs
            ),
            "Kernel SVM [poly]": KernelSVMScratch(
                kernel="poly", degree=2, l2=2e-2, learning_rate=.02, epochs=svm_epochs
            ),
            "Kernel SVM [rbf]": KernelSVMScratch(
                kernel="rbf", l2=2e-2, learning_rate=.02, epochs=svm_epochs
            ),
            "Kernel KNN [poly]": KernelKNNScratch(
                n_neighbors=9, task="classification", kernel="poly", degree=2
            ),
            "Kernel KNN [rbf]": KernelKNNScratch(
                n_neighbors=9, task="classification", kernel="rbf"
            ),
            "KPCA [rbf] + Logistic": KPCAClassifier(kernel="rbf", n_components=12),
        })
    else:
        models.update({
            "Linear Regression": LinearRegressionScratch(l2=1e-3),
            "KNN": knn,
            "Decision Tree": tree,
            "KRR [linear]": KernelRidgeScratch(kernel="linear", l2=2.0),
            "KRR [poly]": KernelRidgeScratch(kernel="poly", degree=2, l2=2.0),
            "KRR [rbf]": KernelRidgeScratch(kernel="rbf", l2=2.0),
            "Kernel KNN [poly]": KernelKNNScratch(
                n_neighbors=9, task="regression", kernel="poly", degree=2
            ),
            "Kernel KNN [rbf]": KernelKNNScratch(
                n_neighbors=9, task="regression", kernel="rbf"
            ),
        })
    return models


def run_all_experiments(tasks, output_dir="artifacts", continue_on_error=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_rows, geometry_rows = [], []
    predictions = {}
    failures = {}

    for task_name, task in tasks.items():
        print(f"\n=== {task_name}: train={len(task.X_train)}, test={len(task.X_test)}, "
              f"features={task.X_train.shape[1]} ===")
        geometry_n = min(260, len(task.X_train))
        geom_idx = (stratified_subsample_indices(task.y_train_model, geometry_n, SEED)
                    if task.task_type == "classification"
                    else np.random.default_rng(SEED).choice(len(task.X_train), geometry_n, replace=False))
        geometry = kernel_geometry(
            task.X_train[geom_idx], task.y_train_model[geom_idx],
            task=task.task_type,
        )
        geometry.insert(0, "task", task_name)
        geometry_rows.extend(geometry.to_dict("records"))

        for model_name, model in models_for_task(task).items():
            print(" ", model_name, end="", flush=True)
            try:
                pred, metrics, timing = _benchmark(model, task)
                row = {
                    "task": task_name, "dataset": task.dataset,
                    "task_type": task.task_type, "model": model_name,
                    "train_rows": len(task.X_train), "test_rows": len(task.X_test),
                    "feature_dimension": task.X_train.shape[1],
                    **metrics, **timing,
                }
                result_rows.append(row)
                predictions[(task_name, model_name)] = pred
                print(" ->", {k: round(v, 4) for k, v in metrics.items()})
            except Exception as exc:
                print(" -> FAILED:", repr(exc))
                result_rows.append({
                    "task": task_name, "dataset": task.dataset,
                    "task_type": task.task_type, "model": model_name,
                    "error": repr(exc),
                })
                if not continue_on_error:
                    raise

        task_results = pd.DataFrame([r for r in result_rows if r["task"] == task_name and "error" not in r])
        if len(task_results):
            if task.task_type == "classification":
                best_name = task_results.sort_values(
                    ["macro_f1", "balanced_accuracy"], ascending=False
                ).iloc[0]["model"]
                worst = worst_classification_cases(
                    task.X_test_raw, task.y_test_eval, predictions[(task_name, best_name)], n=10
                )
                if task.label_encoder is not None and len(worst):
                    worst["actual_label"] = task.label_encoder.inverse_transform(worst["actual"].astype(int))
                    worst["predicted_label"] = task.label_encoder.inverse_transform(worst["predicted"].astype(int))
            else:
                best_name = task_results.sort_values("mae").iloc[0]["model"]
                worst = worst_regression_cases(
                    task.X_test_raw, task.y_test_eval, predictions[(task_name, best_name)], n=10
                )
            worst.insert(0, "selected_model", best_name)
            failures[task_name] = worst
            worst.to_csv(output / f"worst_10_{task_name}.csv", index=False)

    results = pd.DataFrame(result_rows)
    geometry = pd.DataFrame(geometry_rows)
    results.to_csv(output / "model_results.csv", index=False)
    geometry.to_csv(output / "kernel_geometry.csv", index=False)
    return results, geometry, predictions, failures


def kernel_complexity_table(tasks):
    rows = []
    for name, task in tasks.items():
        for label, n in [("full engineered table", task.full_rows),
                         ("experiment subset", len(task.X_train))]:
            rows.append({
                "task": name,
                "scale": label,
                "n": n,
                "kernel_matrix_float64_GB": n * n * 8 / 1024 ** 3,
                "cubic_operation_proxy_n3": float(n) ** 3,
            })
    return pd.DataFrame(rows)


def grouped_error_table(task, prediction, group_column):
    raw = task.X_test_raw.reset_index(drop=True)
    if group_column not in raw:
        return pd.DataFrame()
    group = raw[group_column].astype("string").fillna("Missing")
    rows = []
    for value, idx in group.groupby(group).groups.items():
        idx = np.asarray(list(idx), int)
        if len(idx) < 5:
            continue
        if task.task_type == "classification":
            metric = classification_metrics(task.y_test_eval[idx], prediction[idx])
        else:
            metric = regression_metrics(task.y_test_eval[idx], prediction[idx])
        rows.append({"group_feature": group_column, "group": value, "n": len(idx), **metric})
    return pd.DataFrame(rows)
