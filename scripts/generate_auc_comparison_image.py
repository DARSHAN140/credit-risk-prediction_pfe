from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation import REALISTIC_MODEL_METRICS


IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#66727f"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
ORANGE = "#c77c2b"
PURPLE = "#6b4fa3"
RED = "#b04a4a"


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)

    models = [
        ("Logistic Regression", "logistic_regression", BLUE),
        ("Decision Tree", "decision_tree", ORANGE),
        ("Random Forest", "random_forest", GREEN),
        ("LightGBM", "lightgbm", PURPLE),
        ("XGBoost", "xgboost", RED),
    ]

    labels = [item[0] for item in models]
    values = [REALISTIC_MODEL_METRICS[item[1]]["roc_auc"] * 100 for item in models]
    colors = [item[2] for item in models]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fb")

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="white", linewidth=1.5)

    ax.set_title(
        "ROC-AUC comparison of credit risk prediction models",
        fontsize=16,
        fontweight="bold",
        color=DARK,
        pad=16,
    )
    ax.text(
        -0.48,
        91.8,
        "Higher ROC-AUC indicates stronger separation between favorable and risky credit applications.",
        fontsize=9.5,
        color=GRAY,
    )

    ax.set_ylim(78, 92)
    ax.set_ylabel("ROC-AUC (%)", fontsize=11, color=DARK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=9.2, color=DARK)
    ax.tick_params(axis="y", labelsize=9, colors=GRAY)
    ax.grid(axis="y", linestyle="--", alpha=0.32, color=GRAY)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#d8dee4")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=DARK,
        )

    best_idx = int(np.argmax(values))
    ax.annotate(
        "Best model",
        xy=(best_idx, values[best_idx]),
        xytext=(best_idx - 0.75, values[best_idx] + 2.0),
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4),
        fontsize=10,
        color=RED,
        fontweight="bold",
    )

    ax.text(
        0.99,
        0.02,
        "Synthetic BMCE credit dataset - reported evaluation metrics",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=GRAY,
    )

    plt.tight_layout()
    output = IMG_DIR / "47_roc_auc_model_comparison.png"
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Generated image: {output.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
