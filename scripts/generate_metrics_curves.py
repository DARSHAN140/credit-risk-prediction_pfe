from __future__ import annotations

import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MPL_CACHE_DIR = BASE_DIR / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import confusion_matrix_from_reported_metrics


METRICS_PATH = BASE_DIR / "reports" / "metrics.csv"
THRESHOLD_METRICS_PATH = BASE_DIR / "reports" / "xgboost_threshold_metrics.csv"
IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#66727f"
GRID = "#d8dee4"
BLUE = "#2f5597"
ORANGE = "#c77c2b"
GREEN = "#4f7d5a"
RED = "#b04a4a"
PURPLE = "#6b4fa3"

COLORS = {
    "logistic_regression": BLUE,
    "decision_tree": ORANGE,
    "random_forest": GREEN,
    "xgboost": RED,
    "lightgbm": PURPLE,
}

DISPLAY_ORDER = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
]
TARGET_MODEL = "xgboost"

BENEFIT_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "gini", "ks", "pr_auc"]
ERROR_METRICS = ["brier_score", "ece"]
ALL_METRICS = BENEFIT_METRICS + ERROR_METRICS

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "gini": "Gini",
    "ks": "KS",
    "pr_auc": "PR-AUC",
    "brier_score": "Brier score",
    "ece": "ECE",
}


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH)
    df["model"] = pd.Categorical(df["model"], DISPLAY_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    for metric in ALL_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="raise")
    return df


def savefig(name: str, aliases: tuple[str, ...] = ()) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    output = IMG_DIR / name
    plt.savefig(output, dpi=220, bbox_inches="tight")
    for alias in aliases:
        alias_output = IMG_DIR / alias
        plt.savefig(alias_output, dpi=220, bbox_inches="tight")
        print(f"Generated image: {alias_output.relative_to(BASE_DIR)}")
    plt.close()
    print(f"Generated image: {output.relative_to(BASE_DIR)}")


def add_reporting_note(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.01,
        "Curves reconstructed from aggregated reporting metrics in reports/metrics.csv; synthetic dataset, methodological validation only.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=GRAY,
    )


def roc_curve_from_auc(target_auc: float, points: int = 180) -> tuple[np.ndarray, np.ndarray]:
    fpr = np.linspace(0, 1, points)
    alpha = target_auc / max(1 - target_auc, 1e-6)
    tpr = 1 - (1 - fpr) ** alpha
    return fpr, tpr


def pr_curve_from_auc(target_pr_auc: float, baseline: float = 0.44, points: int = 180) -> tuple[np.ndarray, np.ndarray]:
    recall = np.linspace(0, 1, points)
    alpha = max((1 - baseline) / max(target_pr_auc - baseline, 1e-6) - 1, 0.05)
    precision = baseline + (1 - baseline) * (1 - recall) ** alpha
    return recall, precision


def ks_gap_curve(target_ks: float, points: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    population = np.linspace(0, 1, points)
    exponent = np.log(max((1 - target_ks) / 2, 1e-6)) / np.log(0.5)
    cumulative_risky = 1 - (1 - population) ** exponent
    cumulative_favorable = population**exponent
    gap = cumulative_risky - cumulative_favorable
    return population, cumulative_risky, cumulative_favorable, gap


def calibration_curve_from_errors(ece: float, brier_score: float, min_brier: float) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.linspace(0.05, 0.95, 10)
    brier_penalty = max(brier_score - min_brier, 0)
    slope = 1 - min(0.28, ece * 2.4 + brier_penalty * 0.35)
    wave = ece * 0.9 * np.sin(2 * np.pi * predicted)
    observed = 0.5 + slope * (predicted - 0.5) + wave
    return predicted, np.clip(observed, 0.01, 0.99)


def plot_metrics_heatmap(df: pd.DataFrame) -> None:
    values = df[ALL_METRICS].to_numpy(dtype=float)
    normalized = values.copy()
    for col_index, metric in enumerate(ALL_METRICS):
        column = values[:, col_index]
        low, high = column.min(), column.max()
        if high == low:
            normalized[:, col_index] = 0.5
            continue
        scaled = (column - low) / (high - low)
        normalized[:, col_index] = 1 - scaled if metric in ERROR_METRICS else scaled

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    im = ax.imshow(normalized, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title("XGBoost metrics used for report curves", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xticks(np.arange(len(ALL_METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in ALL_METRICS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["display_name"])
    ax.tick_params(colors=DARK)

    for row in range(values.shape[0]):
        for col, metric in enumerate(ALL_METRICS):
            value = values[row, col]
            label = f"{value:.1%}" if metric not in ERROR_METRICS else f"{value:.3f}"
            text_color = "white" if normalized[row, col] > 0.62 else DARK
            ax.text(col, row, label, ha="center", va="center", fontsize=8.5, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.018)
    cbar.set_label("Relative score within each metric", color=DARK)
    add_reporting_note(fig)
    savefig("52_xgboost_metrics_summary.png")


def plot_roc_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.9))
    ax.set_facecolor("#f7f9fb")
    for _, row in df.iterrows():
        fpr, tpr = roc_curve_from_auc(row["roc_auc"])
        width = 3.0 if row["model"] == "xgboost" else 2.1
        ax.plot(
            fpr,
            tpr,
            color=COLORS[row["model"]],
            lw=width,
            label=f"{row['display_name']} (AUC = {row['roc_auc']:.1%})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8d99a6", lw=1.5, label="Random classifier")
    ax.set_title("XGBoost ROC curve reconstructed from reporting ROC-AUC", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.30, color=GRID)
    ax.legend(loc="lower right", fontsize=8.8, frameon=True)
    add_reporting_note(fig)
    savefig("53_xgboost_roc_curve.png")


def plot_precision_recall_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.9))
    ax.set_facecolor("#f7f9fb")
    baseline = 0.44
    for _, row in df.iterrows():
        recall, precision = pr_curve_from_auc(row["pr_auc"], baseline=baseline)
        width = 3.0 if row["model"] == "xgboost" else 2.1
        ax.plot(
            recall,
            precision,
            color=COLORS[row["model"]],
            lw=width,
            label=f"{row['display_name']} (PR-AUC = {row['pr_auc']:.1%})",
        )
        ax.scatter(row["recall"], row["precision"], color=COLORS[row["model"]], s=34, edgecolor="white", zorder=5)
    ax.axhline(baseline, linestyle="--", color="#8d99a6", lw=1.4, label="Risky-class baseline")
    ax.set_title("XGBoost Precision-Recall curve reconstructed from PR-AUC", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.38, 1.02)
    ax.grid(alpha=0.30, color=GRID)
    ax.legend(loc="lower left", fontsize=8.6, frameon=True)
    add_reporting_note(fig)
    savefig("54_xgboost_precision_recall_curve.png")


def plot_ks_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    ax.set_facecolor("#f7f9fb")
    for _, row in df.iterrows():
        population, cumulative_risky, cumulative_favorable, gap = ks_gap_curve(row["ks"])
        max_index = int(np.argmax(gap))
        max_population = float(population[max_index])
        max_ks = float(gap[max_index])

        ax.plot(
            population,
            cumulative_risky,
            color=RED,
            lw=2.6,
            label="Cumulative risky class",
        )
        ax.plot(
            population,
            cumulative_favorable,
            color=GREEN,
            lw=2.6,
            label="Cumulative favorable class",
        )
        ax.vlines(
            max_population,
            cumulative_favorable[max_index],
            cumulative_risky[max_index],
            colors=BLUE,
            linestyles="--",
            lw=2.4,
            label=f"KS = {max_ks:.3f}",
        )
    ax.set_title("KS curve of the selected model (XGBoost)", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Population sorted by predicted risk")
    ax.set_ylabel("Cumulative distribution")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.30, color=GRID)
    ax.legend(loc="lower right", fontsize=9.2, frameon=True)
    add_reporting_note(fig)
    savefig(
        "55_xgboost_ks_curve.png",
        aliases=("35_ks_curve_xgboost.png",),
    )


def plot_calibration_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    ax.set_facecolor("#f7f9fb")
    min_brier = float(df["brier_score"].min())
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8d99a6", lw=1.5, label="Perfect calibration")
    for _, row in df.iterrows():
        predicted, observed = calibration_curve_from_errors(row["ece"], row["brier_score"], min_brier)
        width = 3.0 if row["model"] == "xgboost" else 2.0
        ax.plot(
            predicted,
            observed,
            marker="o",
            ms=4.2,
            color=COLORS[row["model"]],
            lw=width,
            label=f"{row['display_name']} (ECE = {row['ece']:.3f}, Brier = {row['brier_score']:.3f})",
        )
    ax.set_title("XGBoost calibration curve reconstructed from ECE and Brier score", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.30, color=GRID)
    ax.legend(loc="upper left", fontsize=8.2, frameon=True)
    add_reporting_note(fig)
    savefig("56_xgboost_calibration_curve.png")


def plot_brier_ece_bars(df: pd.DataFrame) -> None:
    sorted_df = df.sort_values("brier_score", ascending=False)
    y = np.arange(len(sorted_df))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    fig.suptitle("XGBoost calibration error metrics - lower is better", fontsize=15, fontweight="bold", color=DARK, x=0.02, ha="left")

    for ax, metric, label in [
        (axes[0], "brier_score", "Brier score"),
        (axes[1], "ece", "Expected Calibration Error"),
    ]:
        ax.barh(y, sorted_df[metric], color=[COLORS[m] for m in sorted_df["model"]], alpha=0.92)
        ax.set_title(label, color=DARK, fontweight="bold")
        ax.set_xlabel("Score")
        ax.grid(axis="x", alpha=0.30, color=GRID)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for idx, value in enumerate(sorted_df[metric]):
            ax.text(value + 0.003, idx, f"{value:.3f}", va="center", fontsize=9, color=DARK)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sorted_df["display_name"])
    add_reporting_note(fig)
    savefig("57_xgboost_brier_ece.png")


def plot_threshold_analysis() -> None:
    threshold_df = pd.read_csv(THRESHOLD_METRICS_PATH)
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    ax.set_facecolor("#f7f9fb")

    styles = [
        ("accuracy", "Accuracy", BLUE),
        ("precision", "Precision", GREEN),
        ("recall", "Recall", RED),
        ("f1", "F1-score", PURPLE),
    ]
    for metric, label, color in styles:
        ax.plot(
            threshold_df["threshold"],
            threshold_df[metric],
            marker="o",
            markersize=5.5,
            linewidth=2.4,
            color=color,
            label=label,
        )

    selected = threshold_df.loc[np.isclose(threshold_df["threshold"], 0.50)].iloc[0]
    ax.axvline(0.50, color=ORANGE, linestyle="--", linewidth=2.0, label="Selected threshold = 0.50")
    ax.scatter(
        [0.50] * len(styles),
        [selected[metric] for metric, _, _ in styles],
        color=[color for _, _, color in styles],
        edgecolor="white",
        linewidth=1.2,
        s=72,
        zorder=5,
    )

    ax.text(
        0.515,
        0.705,
        (
            "At threshold 0.50:\n"
            f"Accuracy = {selected['accuracy']:.1%}\n"
            f"Precision = {selected['precision']:.1%}\n"
            f"Recall = {selected['recall']:.1%}\n"
            f"F1-score = {selected['f1']:.1%}"
        ),
        fontsize=9.3,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID},
    )

    ax.set_title("Threshold analysis for the final XGBoost model", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0.18, 0.82)
    ax.set_ylim(0.65, 0.98)
    ax.grid(alpha=0.30, color=GRID)
    ax.legend(loc="lower left", fontsize=9.0, frameon=True)
    add_reporting_note(fig)
    savefig(
        "59_xgboost_threshold_analysis.png",
        aliases=("37_threshold_tradeoff_xgboost.png",),
    )


def plot_confusion_matrices(df: pd.DataFrame, total: int = 4000) -> None:
    vmax = 0
    matrices: list[tuple[pd.Series, np.ndarray]] = []
    for _, row in df.iterrows():
        metrics = {key: float(row[key]) for key in ["accuracy", "precision", "recall", "f1"]}
        cm = confusion_matrix_from_reported_metrics(metrics, total)
        matrices.append((row, cm))
        vmax = max(vmax, int(cm.max()))

    if len(matrices) == 1:
        fig, ax = plt.subplots(figsize=(6.8, 5.9))
        axes = np.asarray([ax])
        title = "XGBoost confusion matrix reconstructed from reporting metrics"
    else:
        fig, axes_grid = plt.subplots(2, 3, figsize=(13.2, 8.0))
        axes = axes_grid.ravel()
        title = "Confusion matrices reconstructed from reporting metrics"

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        color=DARK,
        x=0.03,
        ha="left",
    )

    for ax, (row, cm) in zip(axes, matrices):
        color = COLORS[row["model"]]
        row_totals = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(row["display_name"], color=DARK, fontweight="bold", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred. favorable", "Pred. risky"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual favorable", "Actual risky"], fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)

        for i in range(2):
            for j in range(2):
                value = int(cm[i, j])
                pct = cm_pct[i, j]
                text_color = "white" if value > vmax * 0.55 else DARK
                ax.text(
                    j,
                    i,
                    f"{value:,}\n({pct:.1%})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold",
                )

        ax.text(
            0.5,
            -0.22,
            (
                f"Acc. {row['accuracy']:.1%} | Prec. {row['precision']:.1%} | "
                f"Recall {row['recall']:.1%}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            color=GRAY,
        )

    for ax in axes[len(matrices) :]:
        ax.axis("off")

    fig.text(
        0.5,
        0.01,
        (
            "Matrix reconstructed from aggregated XGBoost metrics on a fixed "
            f"test base of {total:,} cases; harmonized report figure only."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=GRAY,
    )

    savefig("58_xgboost_confusion_matrix.png")


def main() -> None:
    df = load_metrics()
    df = df.loc[df["model"] == TARGET_MODEL].reset_index(drop=True)
    plot_metrics_heatmap(df)
    plot_roc_curves(df)
    plot_precision_recall_curves(df)
    plot_ks_curves(df)
    plot_calibration_curves(df)
    plot_brier_ece_bars(df)
    plot_threshold_analysis()
    plot_confusion_matrices(df)


if __name__ == "__main__":
    main()
