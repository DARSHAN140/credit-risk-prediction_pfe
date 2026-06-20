from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MPL_CACHE_DIR = BASE_DIR / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import MAX_TRAIN_ROWS, MODEL_DIR, RANDOM_STATE, RAW_DATA_PATH, REPORT_DIR
from src.data_processing import infer_column_types, prepare_model_frame, read_credit_data, temporal_or_stratified_split
from src.evaluation import ks_statistic
from src.modeling import GOVERNED_XGBOOST_COLUMNS, SMOTE_SAMPLING_STRATEGY, build_preprocessor

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover - dependency is declared by the project
    raise RuntimeError("LightGBM est requis pour comparer les cinq modeles.") from exc


TARGET_MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
)
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
COLORS = {
    "logistic_regression": "#2f5597",
    "decision_tree": "#c77c2b",
    "random_forest": "#4f7d5a",
    "xgboost": "#b04a4a",
    "lightgbm": "#6b4fa3",
}

OUTPUT_CSV = REPORT_DIR / "calibrated_model_comparison.csv"
OUTPUT_JSON = REPORT_DIR / "calibrated_model_comparison.json"
OUTPUT_TEX = REPORT_DIR / "tableau_comparaison_modeles_recalibres.tex"
OUTPUT_TEX_COMPAT = REPORT_DIR / "tableau_comparaison_modeles.tex"
CALIBRATION_POINTS_CSV = REPORT_DIR / "calibrated_models_calibration_points.csv"
TEST_PREDICTIONS_CSV = REPORT_DIR / "calibrated_models_test_predictions.csv"
CALIBRATED_MODEL_DIR = MODEL_DIR / "calibrated"
COMPARISON_FIGURE = BASE_DIR / "img" / "60_calibrated_models_comparison.png"


def load_governed_data():
    raw_df = read_credit_data(RAW_DATA_PATH)
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = (
            raw_df.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw_df), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    X, y = prepare_model_frame(raw_df)
    missing = sorted(set(GOVERNED_XGBOOST_COLUMNS) - set(X.columns))
    if missing:
        raise ValueError(f"Variables gouvernees absentes: {missing}")
    X = X[GOVERNED_XGBOOST_COLUMNS]
    X_train, X_test, y_train, y_test = temporal_or_stratified_split(raw_df, X, y)
    return X_train, X_test, y_train, y_test


def split_training_period(X_train, y_train):
    """Chronological 70/15/15 split: fit, calibration, threshold selection."""
    n_rows = len(X_train)
    fit_end = int(n_rows * 0.70)
    calibration_end = int(n_rows * 0.85)
    parts = (
        X_train.iloc[:fit_end],
        X_train.iloc[fit_end:calibration_end],
        X_train.iloc[calibration_end:],
        y_train.iloc[:fit_end],
        y_train.iloc[fit_end:calibration_end],
        y_train.iloc[calibration_end:],
    )
    for target in parts[3:]:
        if target.nunique() != 2:
            raise ValueError("Chaque sous-ensemble doit contenir les deux classes.")
    return parts


def model_estimators() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            C=0.05,
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=35,
            min_samples_split=80,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=7,
            min_samples_leaf=15,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=70,
            max_depth=2,
            learning_rate=0.025,
            subsample=0.65,
            colsample_bytree=0.65,
            reg_lambda=30.0,
            min_child_weight=20,
            gamma=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            objective="binary",
            n_estimators=80,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=40,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=25.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def build_training_pipeline(estimator, numeric_columns, categorical_columns) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_columns, categorical_columns)),
            (
                "smote",
                SMOTE(
                    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("model", estimator),
        ]
    )


def best_f1_threshold(y_true, y_score) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def calibration_bins(y_true, y_score, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    frame = pd.DataFrame({"target": np.asarray(y_true), "score": np.asarray(y_score)})
    frame["bin"] = pd.qcut(frame["score"], q=n_bins, duplicates="drop")
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            mean_predicted_probability=("score", "mean"),
            observed_positive_rate=("target", "mean"),
            count=("target", "size"),
        )
        .reset_index()
    )
    grouped["absolute_error"] = (
        grouped["observed_positive_rate"] - grouped["mean_predicted_probability"]
    ).abs()
    grouped["weight"] = grouped["count"] / len(frame)
    ece = float((grouped["weight"] * grouped["absolute_error"]).sum())
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped, ece


def evaluate(y_true, y_score, threshold: float, ece: float) -> dict[str, float | int]:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = float(roc_auc_score(y_true, y_score))
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "gini": 2 * roc_auc - 1,
        "ks": float(ks_statistic(y_true, y_score)),
        "pr_auc": float(auc(pr_recall, pr_precision)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "ece": ece,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def latex_table(results: pd.DataFrame) -> str:
    indexed = results.set_index("model")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparaison des modeles recalibres sur le jeu de test}",
        r"\label{tab:comparaison_modeles}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{|l|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Logistic Regression} & \textbf{Decision Tree} & \textbf{Random Forest} & \textbf{XGBoost} & \textbf{LightGBM} \\",
        r"\hline",
    ]

    percentage_rows = (
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("Gini", "gini"),
        ("KS", "ks"),
        ("PR-AUC", "pr_auc"),
    )
    for label, metric in percentage_rows:
        values = [f"{100 * indexed.loc[name, metric]:.2f}\\%" for name in TARGET_MODELS]
        lines.extend([f"{label} & " + " & ".join(values) + r" \\", r"\hline"])

    for label, metric in (
        (r"Brier Score $\downarrow$", "brier_score"),
        (r"ECE $\downarrow$", "ece"),
        ("Selected threshold", "threshold"),
    ):
        values = [f"{indexed.loc[name, metric]:.3f}" for name in TARGET_MODELS]
        lines.extend([f"{label} & " + " & ".join(values) + r" \\", r"\hline"])

    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def plot_calibration_comparison(points: pd.DataFrame, results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#66727f", label="Perfect calibration")
    metrics = results.set_index("model")
    for model_name in TARGET_MODELS:
        model_points = points.loc[points["model"] == model_name]
        ax.plot(
            model_points["mean_predicted_probability"],
            model_points["observed_positive_rate"],
            marker="o",
            linewidth=2,
            color=COLORS[model_name],
            label=(
                f"{DISPLAY_NAMES[model_name]} "
                f"(Brier={metrics.loc[model_name, 'brier_score']:.3f}, "
                f"ECE={metrics.loc[model_name, 'ece']:.3f})"
            ),
        )
    ax.set_title("Calibration comparison on the untouched test set", fontsize=15, fontweight="bold", loc="left")
    ax.set_xlabel("Mean calibrated probability")
    ax.set_ylabel("Observed risky-class rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    COMPARISON_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", message=".*cv='prefit'.*")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_governed_data()
    X_fit, X_calibration, X_threshold, y_fit, y_calibration, y_threshold = split_training_period(
        X_train,
        y_train,
    )
    numeric_columns, categorical_columns = infer_column_types(X_fit)

    results: list[dict[str, float | int | str]] = []
    all_calibration_points: list[pd.DataFrame] = []
    test_predictions = pd.DataFrame({"target": y_test.to_numpy()})

    for model_name, estimator in model_estimators().items():
        print(f"Training and calibrating {DISPLAY_NAMES[model_name]}...", flush=True)
        pipeline = build_training_pipeline(estimator, numeric_columns, categorical_columns)
        pipeline.fit(X_fit, y_fit)

        calibrated = CalibratedClassifierCV(
            estimator=pipeline,
            method="sigmoid",
            cv="prefit",
        )
        calibrated.fit(X_calibration, y_calibration)

        threshold_scores = calibrated.predict_proba(X_threshold)[:, 1]
        threshold, validation_f1 = best_f1_threshold(y_threshold, threshold_scores)

        test_scores = calibrated.predict_proba(X_test)[:, 1]
        points, ece = calibration_bins(y_test, test_scores, n_bins=10)
        points.insert(0, "model", model_name)
        all_calibration_points.append(points)

        metrics = evaluate(y_test, test_scores, threshold, ece)
        metrics.update(
            {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "calibration_method": "sigmoid",
                "threshold_criterion": "maximum F1 on threshold-validation set",
                "validation_f1_at_selected_threshold": validation_f1,
            }
        )
        results.append(metrics)
        test_predictions[f"score_{model_name}"] = test_scores
        test_predictions[f"prediction_{model_name}"] = (test_scores >= threshold).astype(int)
        joblib.dump(calibrated, CALIBRATED_MODEL_DIR / f"{model_name}.joblib")
        print(
            f"  AUC={metrics['roc_auc']:.4f} | Brier={metrics['brier_score']:.4f} "
            f"| ECE={metrics['ece']:.4f} | threshold={threshold:.4f}",
            flush=True,
        )

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    calibration_df = pd.concat(all_calibration_points, ignore_index=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    calibration_df.to_csv(CALIBRATION_POINTS_CSV, index=False)
    test_predictions.to_csv(TEST_PREDICTIONS_CSV, index=False)

    best = results_df.iloc[0]
    payload = {
        "methodology": {
            "features": list(GOVERNED_XGBOOST_COLUMNS),
            "n_fit": int(len(X_fit)),
            "n_calibration": int(len(X_calibration)),
            "n_threshold_validation": int(len(X_threshold)),
            "n_test": int(len(X_test)),
            "calibration_method": "sigmoid (Platt scaling)",
            "threshold_selection": "maximum F1 on a separate validation period",
            "test_usage": "metrics only; no calibration or threshold fitting",
        },
        "best_model_by_roc_auc": str(best["model"]),
        "results": results_df.to_dict(orient="records"),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = latex_table(results_df)
    best_text = (
        f"\n\nThe best ranking model is {best['display_name']}, with a test ROC-AUC of "
        f"{100 * best['roc_auc']:.2f}\\%. At the validation-selected threshold of "
        f"{best['threshold']:.3f}, its accuracy is {100 * best['accuracy']:.2f}\\%, "
        f"precision is {100 * best['precision']:.2f}\\%, recall is "
        f"{100 * best['recall']:.2f}\\%, and F1-score is {100 * best['f1']:.2f}\\%. "
        "Thresholds were selected before examining the test labels."
    )
    OUTPUT_TEX.write_text(table + best_text, encoding="utf-8")
    OUTPUT_TEX_COMPAT.write_text(table + best_text, encoding="utf-8")
    plot_calibration_comparison(calibration_df, results_df)

    print("\nFinal comparison:")
    print(
        results_df[
            [
                "display_name",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "ks",
                "brier_score",
                "ece",
            ]
        ].to_string(index=False)
    )
    print(f"\nLaTeX table: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
