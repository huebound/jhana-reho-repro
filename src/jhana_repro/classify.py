from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .schemas import public_roi_columns, read_reho_csv


@dataclass(frozen=True)
class ClassificationConfig:
    test_size: float = 0.25
    random_state: int = 42
    n_splits: int = 5
    rfe_fraction: float = 0.5
    n_permutations: int = 1000
    compat_target_per_subject: int = 8


def run_modes(
    reho_csv: Path,
    output_dir: Path,
    mode: str = "both",
    config: ClassificationConfig = ClassificationConfig(),
) -> dict[str, dict]:
    df = read_reho_csv(reho_csv)
    roi_columns = public_roi_columns(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["leak-safe", "compat"] if mode == "both" else [mode]
    results = {}
    split = subject_train_test_split(df, config)
    for selected_mode in modes:
        mode_dir = output_dir / selected_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        result = run_public_classification(
            df=df,
            roi_columns=roi_columns,
            output_dir=mode_dir,
            mode=selected_mode,
            split=split,
            config=config,
        )
        results[selected_mode] = result

    diagnostics = diagnostics_frame(results)
    diagnostics.to_csv(output_dir / "method_diagnostics.csv", index=False)
    (output_dir / "method_diagnostics.json").write_text(
        json.dumps(diagnostics.to_dict(orient="records"), indent=2) + "\n"
    )
    if set(results) == {"leak-safe", "compat"}:
        comparison = comparison_frame(results)
        comparison.to_csv(output_dir / "method_comparison.csv", index=False)
        (output_dir / "method_comparison.json").write_text(
            json.dumps(comparison.to_dict(orient="records"), indent=2) + "\n"
        )
    return results


def subject_train_test_split(df: pd.DataFrame, config: ClassificationConfig) -> dict[str, np.ndarray]:
    subject_labels = df.groupby("subid")["segment"].agg(lambda x: x.astype(str).mode().iloc[0])
    subjects = subject_labels.index.to_numpy()
    labels = subject_labels.to_numpy()
    stratify = labels if pd.Series(labels).value_counts().min() >= 2 else None
    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    if set(train_subjects) & set(test_subjects):
        raise AssertionError("Subject leakage: train and test subject sets overlap")
    return {"train_subjects": train_subjects, "test_subjects": test_subjects}


def run_public_classification(
    df: pd.DataFrame,
    roi_columns: list[str],
    output_dir: Path,
    mode: str,
    split: dict[str, np.ndarray],
    config: ClassificationConfig = ClassificationConfig(),
) -> dict:
    train_df = df[df["subid"].isin(split["train_subjects"])].reset_index(drop=True)
    test_df = df[df["subid"].isin(split["test_subjects"])].reset_index(drop=True)
    if set(train_df["subid"]) & set(test_df["subid"]):
        raise AssertionError("Subject leakage after dataframe split")

    encoder = LabelEncoder().fit(sorted(df["segment"].astype(str).unique()))
    if len(encoder.classes_) != 2:
        raise ValueError(f"Public classifier expects exactly two segment classes, got {list(encoder.classes_)}")

    if mode == "leak-safe":
        result = run_leak_safe(train_df, test_df, roi_columns, encoder, output_dir, config)
    elif mode == "compat":
        result = run_compat(train_df, test_df, roi_columns, encoder, output_dir, config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result["mode"] = mode
    result["train_subjects"] = sorted(map(str, split["train_subjects"]))
    result["test_subjects"] = sorted(map(str, split["test_subjects"]))
    write_outputs(output_dir, result)
    return result


def run_leak_safe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    roi_columns: list[str],
    encoder: LabelEncoder,
    output_dir: Path,
    config: ClassificationConfig,
) -> dict:
    X = train_df[roi_columns].to_numpy(dtype=float)
    y = encoder.transform(train_df["segment"].astype(str))
    groups = train_df["subid"].astype(str).to_numpy()
    X_test = test_df[roi_columns].to_numpy(dtype=float)
    y_test = encoder.transform(test_df["segment"].astype(str))
    model_defs = make_models(config.random_state)

    cv_rows = []
    splitter = make_group_splitter(y, groups, config)
    for fold_index, (train_idx, val_idx) in enumerate(splitter, start=1):
        selected, selector = fit_selector(X[train_idx], y[train_idx], roi_columns, config)
        X_fold_train = selector.transform(X[train_idx])
        X_fold_val = selector.transform(X[val_idx])
        X_fold_bal, y_fold_bal = balance_matrix(X_fold_train, y[train_idx], config.random_state)
        for name, model in model_defs.items():
            fitted = fit_model(model, X_fold_bal, y_fold_bal)
            proba = predict_proba_like(fitted, X_fold_val)
            preds = np.argmax(proba, axis=1)
            metrics = compute_metrics(y[val_idx], preds, proba)
            cv_rows.append({"fold": fold_index, "model": name, "n_features": len(selected), **metrics})

    cv_df = pd.DataFrame(cv_rows)
    ranking = (
        cv_df.groupby("model")[["kappa", "auc", "accuracy"]]
        .mean()
        .sort_values(["kappa", "auc", "accuracy"], ascending=False)
    )
    top_models = ranking.head(3).index.tolist()

    selected_features, selector = fit_selector(X, y, roi_columns, config)
    X_train_rfe = selector.transform(X)
    X_test_rfe = selector.transform(X_test)
    X_train_bal, y_train_bal = balance_matrix(X_train_rfe, y, config.random_state)
    fitted_models = {name: fit_model(model_defs[name], X_train_bal, y_train_bal) for name in top_models}
    ensemble = np.mean([predict_proba_like(model, X_test_rfe) for model in fitted_models.values()], axis=0)
    preds = np.argmax(ensemble, axis=1)
    ensemble_metrics = compute_metrics(y_test, preds, ensemble, config.n_permutations, config.random_state)
    individual = []
    for name, model in fitted_models.items():
        proba = predict_proba_like(model, X_test_rfe)
        individual.append({"model": name, **compute_metrics(y_test, np.argmax(proba, axis=1), proba)})

    write_feature_importances(output_dir, fitted_models, X_train_rfe, y_train_bal, selected_features)
    cv_df.to_csv(output_dir / "cv_metrics.csv", index=False)
    return {
        "pair": f"{encoder.classes_[0]}_vs_{encoder.classes_[1]}",
        "classes": encoder.classes_.tolist(),
        "top_models": top_models,
        "selected_features": selected_features,
        "ensemble": ensemble_metrics,
        "individual_models": individual,
        "cv_summary": ranking.reset_index().to_dict(orient="records"),
    }


def run_compat(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    roi_columns: list[str],
    encoder: LabelEncoder,
    output_dir: Path,
    config: ClassificationConfig,
) -> dict:
    augmented = compat_upsample_training_rows(train_df, roi_columns, config)
    X = augmented[roi_columns].to_numpy(dtype=float)
    y = encoder.transform(augmented["segment"].astype(str))
    X_test = test_df[roi_columns].to_numpy(dtype=float)
    y_test = encoder.transform(test_df["segment"].astype(str))
    model_defs = make_models(config.random_state)

    selected_features, selector = fit_selector(X, y, roi_columns, config)
    X_rfe = selector.transform(X)
    X_test_rfe = selector.transform(X_test)
    cv = StratifiedKFold(n_splits=min(config.n_splits, np.bincount(y).min()), shuffle=True, random_state=config.random_state)

    individual = []
    fitted_models = {}
    cv_rows = []
    for name, model in model_defs.items():
        fold_metrics = []
        for fold_index, (tr_idx, val_idx) in enumerate(cv.split(X_rfe, y), start=1):
            fitted_fold = fit_model(model, X_rfe[tr_idx], y[tr_idx])
            proba = predict_proba_like(fitted_fold, X_rfe[val_idx])
            metrics = compute_metrics(y[val_idx], np.argmax(proba, axis=1), proba)
            fold_metrics.append(metrics)
            cv_rows.append({"fold": fold_index, "model": name, **metrics})
        fitted = fit_model(model, X_rfe, y)
        fitted_models[name] = fitted
        proba = predict_proba_like(fitted, X_test_rfe)
        metrics = compute_metrics(y_test, np.argmax(proba, axis=1), proba)
        individual.append(
            {
                "model": name,
                "cv_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
                **metrics,
            }
        )

    top_models = [row["model"] for row in sorted(individual, key=lambda row: row["accuracy"], reverse=True)[:3]]
    ensemble = np.mean([predict_proba_like(fitted_models[name], X_test_rfe) for name in top_models], axis=0)
    preds = np.argmax(ensemble, axis=1)
    ensemble_metrics = compute_metrics(y_test, preds, ensemble, config.n_permutations, config.random_state)
    write_feature_importances(
        output_dir,
        {name: fitted_models[name] for name in top_models},
        X_rfe,
        y,
        selected_features,
    )
    pd.DataFrame(cv_rows).to_csv(output_dir / "cv_metrics.csv", index=False)
    augmented.to_csv(output_dir / "compat_upsampled_train.csv", index=False)
    return {
        "pair": f"{encoder.classes_[0]}_vs_{encoder.classes_[1]}",
        "classes": encoder.classes_.tolist(),
        "top_models": top_models,
        "selected_features": selected_features,
        "ensemble": ensemble_metrics,
        "individual_models": individual,
        "compat_upsampled_rows": int(len(augmented)),
    }


def make_group_splitter(y: np.ndarray, groups: np.ndarray, config: ClassificationConfig):
    min_class = int(np.bincount(y).min())
    n_splits = max(2, min(config.n_splits, min_class))
    if len(np.unique(groups)) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
        return splitter.split(np.zeros_like(y), y, groups)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    return splitter.split(np.zeros_like(y), y)


def make_models(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM": SVC(kernel="linear", probability=False, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=random_state),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=random_state),
}


def fit_model(model, X: np.ndarray, y: np.ndarray):
    fitted = clone(model)
    if isinstance(fitted, KNeighborsClassifier):
        fitted.set_params(n_neighbors=max(1, min(fitted.n_neighbors, len(y))))
    return fitted.fit(X, y)


def fit_selector(
    X: np.ndarray,
    y: np.ndarray,
    roi_columns: list[str],
    config: ClassificationConfig,
) -> tuple[list[str], RFE]:
    n_features = max(1, int(len(roi_columns) * config.rfe_fraction))
    selector = RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=config.random_state),
        n_features_to_select=n_features,
    )
    selector.fit(X, y)
    selected = [feature for feature, keep in zip(roi_columns, selector.get_support()) if keep]
    return selected, selector


def balance_matrix(X: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() == counts.max():
        return X, y
    if counts.min() >= 2:
        sampler = SMOTE(k_neighbors=min(5, int(counts.min() - 1)), random_state=random_state)
    else:
        sampler = RandomOverSampler(random_state=random_state)
    return sampler.fit_resample(X, y)


def compat_upsample_training_rows(
    train_df: pd.DataFrame,
    roi_columns: list[str],
    config: ClassificationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    rows = []
    metadata = [col for col in train_df.columns if col not in roi_columns]
    for _, row in train_df.iterrows():
        rows.append(row.copy())
        values = row[roi_columns].to_numpy(dtype=float)
        scale = np.nanstd(values)
        noise_scale = 0.05 * (scale if scale > 0 else 1.0)
        for index in range(config.compat_target_per_subject - 1):
            new_row = row[metadata].copy()
            noisy = values + rng.normal(0.0, noise_scale, size=values.shape)
            for col, value in zip(roi_columns, noisy):
                new_row[col] = value
            new_row["run"] = f"{row['run']}_aug{index + 1:02d}"
            rows.append(new_row)
    return pd.DataFrame(rows, columns=train_df.columns)


def predict_proba_like(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scaled = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            proba = np.column_stack([1.0 - scaled, scaled])
        else:
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
    else:
        preds = model.predict(X)
        proba = np.column_stack([1 - preds, preds]).astype(float)
    if proba.shape[1] != 2:
        raise ValueError("Only binary classification is supported")
    return proba


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    n_permutations: int = 0,
    random_state: int = 42,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sensitivity = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
    specificity = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    try:
        auc = roc_auc_score(y_true, proba[:, 1])
    except ValueError:
        auc = np.nan
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(auc),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
    if n_permutations:
        metrics["p_value"] = float(permutation_p_value(y_true, y_pred, n_permutations, random_state))
    return metrics


def permutation_p_value(y_true: np.ndarray, y_pred: np.ndarray, n_permutations: int, random_state: int) -> float:
    rng = np.random.default_rng(random_state)
    observed = accuracy_score(y_true, y_pred)
    scores = [accuracy_score(rng.permutation(y_true), y_pred) for _ in range(n_permutations)]
    return float(np.mean(np.asarray(scores) >= observed))


def write_feature_importances(
    output_dir: Path,
    fitted_models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_features: list[str],
) -> None:
    frames = []
    for name, model in fitted_models.items():
        if hasattr(model, "feature_importances_"):
            weights = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            weights = np.abs(np.asarray(model.coef_)).ravel()
        else:
            weights = np.zeros(len(selected_features), dtype=float)
        weights = normalize(weights)
        frame = pd.DataFrame({"Atlas": "Schaefer2018", "ROI": selected_features, "Weights": weights})
        frame.sort_values("Weights", ascending=False).to_csv(
            output_dir / f"feature_importance_{safe_name(name)}_depr_vs_control.csv",
            index=False,
        )
        frames.append(frame.set_index("ROI")["Weights"].rename(name))
    average = pd.concat(frames, axis=1).mean(axis=1).reset_index()
    average.columns = ["ROI", "Weights"]
    average.insert(0, "Atlas", "Schaefer2018")
    average.sort_values("Weights", ascending=False).to_csv(
        output_dir / "average_feature_importance_depr_vs_control.csv",
        index=False,
    )


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def write_outputs(output_dir: Path, result: dict) -> None:
    serializable = make_jsonable(result)
    (output_dir / "metrics.json").write_text(json.dumps(serializable, indent=2) + "\n")
    pd.DataFrame([{"mode": result["mode"], **result["ensemble"]}]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / f"results_{result['mode']}.txt").write_text(format_result_text(result))


def format_result_text(result: dict) -> str:
    lines = [
        f"[Performance Summary for Pair: ('{result['classes'][0]}', '{result['classes'][1]}')]",
        f"Ensembled Models: {result['top_models']}",
        "Ensemble Performance:",
    ]
    for key in ("accuracy", "auc", "kappa", "precision", "recall", "f1", "sensitivity", "specificity", "p_value"):
        if key in result["ensemble"]:
            lines.append(f"  {display_metric(key)}: {result['ensemble'][key]}")
    lines.append("Individual Model Performances:")
    for row in result["individual_models"]:
        lines.append(f"  Model: {row['model']}")
        for key in ("cv_accuracy", "accuracy", "kappa", "f1"):
            if key in row:
                lines.append(f"    {display_metric(key)}: {row[key]}")
    return "\n".join(lines) + "\n"


def display_metric(key: str) -> str:
    names = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "kappa": "Cohen Kappa",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "p_value": "P-Value",
        "cv_accuracy": "CV Accuracy",
    }
    return names[key]


def comparison_frame(results: dict[str, dict]) -> pd.DataFrame:
    safe = results["leak-safe"]["ensemble"]
    compat = results["compat"]["ensemble"]
    metrics = sorted(set(safe) & set(compat))
    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "leak_safe": safe[metric],
                "compat": compat[metric],
                "compat_minus_safe": compat[metric] - safe[metric],
            }
        )
    return pd.DataFrame(rows)


def diagnostics_frame(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for mode, result in results.items():
        cv_all, cv_top = validation_accuracy_summary(result)
        rows.append(
            {
                "mode": mode,
                "train_subjects": len(result["train_subjects"]),
                "test_subjects": len(result["test_subjects"]),
                "training_rows_after_augmentation": result.get("compat_upsampled_rows", len(result["train_subjects"])),
                "top_models": ";".join(result["top_models"]),
                "cv_accuracy_mean_all_models": cv_all,
                "cv_accuracy_mean_top_models": cv_top,
                "test_accuracy": result["ensemble"]["accuracy"],
                "test_auc": result["ensemble"]["auc"],
                "test_f1": result["ensemble"]["f1"],
                "cv_minus_test_accuracy": cv_top - result["ensemble"]["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def validation_accuracy_summary(result: dict) -> tuple[float, float]:
    top_models = set(result["top_models"])
    if "cv_summary" in result:
        rows = result["cv_summary"]
        all_scores = [row["accuracy"] for row in rows]
        top_scores = [row["accuracy"] for row in rows if row["model"] in top_models]
    else:
        rows = result["individual_models"]
        all_scores = [row["cv_accuracy"] for row in rows if "cv_accuracy" in row]
        top_scores = [row["cv_accuracy"] for row in rows if row["model"] in top_models and "cv_accuracy" in row]
    return float(np.mean(all_scores)), float(np.mean(top_scores))


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def make_jsonable(value):
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
