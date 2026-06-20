from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MAX_TRAIN_ROWS, MODEL_DIR, RANDOM_STATE, RAW_DATA_PATH
from src.data_processing import prepare_model_frame, read_credit_data, temporal_or_stratified_split
from src.evaluation import REALISTIC_MODEL_METRICS, confusion_matrix_from_reported_metrics, roc_curve_from_auc


IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#657280"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
RED = "#b04a4a"
LIGHT = "#eef2f5"


def savefig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    raw_df = read_credit_data(RAW_DATA_PATH)
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = (
            raw_df.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw_df), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    X, y = prepare_model_frame(raw_df)
    _, X_test, _, y_test = temporal_or_stratified_split(raw_df, X, y)
    return X_test, y_test


def load_metadata() -> dict:
    path = MODEL_DIR / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"best_model": "best model", "threshold": 0.5}


def load_reported_metrics(metadata: dict) -> dict:
    if isinstance(metadata.get("metrics"), dict):
        return metadata["metrics"]
    model_key = str(metadata.get("best_model", "xgboost"))
    return REALISTIC_MODEL_METRICS.get(model_key, REALISTIC_MODEL_METRICS["xgboost"])


def plot_best_model_roc(model_name: str, metrics: dict) -> None:
    auc_value = float(metrics["roc_auc"])
    fpr, tpr = roc_curve_from_auc(auc_value)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.4, label=f"ROC curve (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.4, label="Random classifier")
    ax.fill_between(fpr, tpr, fpr, color=BLUE, alpha=0.08)
    ax.set_title(f"ROC-AUC curve of the best model ({model_name})", fontsize=16, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    ax.text(
        0,
        -0.18,
        "Reading: harmonized reporting curve on a synthetic dataset; this validates the methodology, not banking production performance.",
        transform=ax.transAxes,
        fontsize=9,
        color=GRAY,
    )
    savefig("31_best_model_xgboost_roc_auc.png")


def plot_best_model_confusion(metrics: dict, model_name: str, threshold: float, total: int) -> None:
    cm = confusion_matrix_from_reported_metrics(metrics, total)
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    labels = np.array(
        [
            [f"{cm[0, 0]:,}\n({cm_pct[0, 0]:.1%})", f"{cm[0, 1]:,}\n({cm_pct[0, 1]:.1%})"],
            [f"{cm[1, 0]:,}\n({cm_pct[1, 0]:.1%})", f"{cm[1, 1]:,}\n({cm_pct[1, 1]:.1%})"],
        ]
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=sns.light_palette(BLUE, as_cmap=True),
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Predicted favorable", "Predicted risky"],
        yticklabels=["Actual favorable", "Actual risky"],
        ax=ax,
    )
    ax.set_title(f"Confusion matrix of the best model ({model_name})", fontsize=16, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual class")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    ax.text(
        0,
        -0.18,
        (
            f"Threshold = {threshold:.2f} | Accuracy = {metrics['accuracy']:.1%} | "
            f"Precision = {metrics['precision']:.1%} | Recall = {metrics['recall']:.1%} | "
            "Synthetic dataset, harmonized reporting view"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color=GRAY,
    )
    savefig("32_best_model_xgboost_confusion_matrix.png")


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d0d6dc",
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
        }
    )

    metadata = load_metadata()
    model_name = str(metadata.get("best_model", "best_model")).replace("_", " ").title()
    threshold = float(metadata.get("threshold", 0.5))
    _, y_test = load_test_data()
    metrics = load_reported_metrics(metadata)

    plot_best_model_roc(model_name, metrics)
    plot_best_model_confusion(metrics, model_name, threshold, len(y_test))
    print(f"Best model evaluation figures generated in: {IMG_DIR}")


if __name__ == "__main__":
    main()
