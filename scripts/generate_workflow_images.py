from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#657280"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
RED = "#b04a4a"
ORANGE = "#c77c2b"
PURPLE = "#6b4fa3"
LIGHT = "#eef2f5"


def savefig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def rounded_box(ax, xy, width, height, text, color, fontsize=10):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color,
        alpha=0.94,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        fontweight="bold",
    )


def arrow(ax, start, end, color=GRAY, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def plot_system_workflow() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "System workflow of the credit risk prediction solution",
        fontsize=17,
        fontweight="bold",
        color=DARK,
        ha="left",
    )

    nodes = {
        "input": ((0.03, 0.58), "Data Input\nClient + financial\n+ credit data", BLUE),
        "prep": ((0.21, 0.58), "Data Preparation\nCleaning, missing values,\nencoding, scaling", ORANGE),
        "features": ((0.39, 0.58), "Feature Engineering\nDTI, remaining income,\nsavings rate", ORANGE),
        "model": ((0.57, 0.58), "Trained Model\nBest model: XGBoost\nrisk probability", PURPLE),
        "output": ((0.75, 0.58), "Prediction Output\nPD default, score,\nsegment, decision", GREEN),
        "support": ((0.75, 0.20), "Decision Support\nAnalyst review,\naccept/refuse/escalate", DARK),
        "xai": ((0.57, 0.20), "Explainability\nSHAP + feature\nimportance", RED),
    }
    width, height = 0.14, 0.17
    for xy, text, color in nodes.values():
        rounded_box(ax, xy, width, height, text, color)

    arrow(ax, (0.17, 0.665), (0.21, 0.665))
    arrow(ax, (0.35, 0.665), (0.39, 0.665))
    arrow(ax, (0.53, 0.665), (0.57, 0.665))
    arrow(ax, (0.71, 0.665), (0.75, 0.665))
    arrow(ax, (0.82, 0.58), (0.82, 0.37))
    arrow(ax, (0.64, 0.58), (0.64, 0.37), color=RED)
    arrow(ax, (0.71, 0.285), (0.75, 0.285), color=RED)

    ax.text(
        0.03,
        0.08,
        "Reading: the same preprocessing and feature engineering logic is used during training and prediction. "
        "The final output is enriched with explainability to support the banking decision.",
        fontsize=9,
        color=GRAY,
        ha="left",
    )
    savefig("33_system_workflow_credit_scoring.png")


def plot_risk_score_generation() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.92,
        "Risk score generation from probability of default",
        fontsize=17,
        fontweight="bold",
        color=DARK,
        ha="left",
    )

    segments = [
        (0.001, 0.10, "Tres bon\nPD < 10%", GREEN),
        (0.10, 0.25, "Bon\n10%-25%", "#7aa35f"),
        (0.25, 0.45, "Moyen\n25%-45%", ORANGE),
        (0.45, 0.65, "Risque\n45%-65%", "#c45f3c"),
        (0.65, 0.999, "Tres risque\nPD > 65%", RED),
    ]
    x0, x1 = 0.08, 0.92
    y, h = 0.46, 0.18
    for start, end, label, color in segments:
        left = x0 + (start - 0.001) / (0.999 - 0.001) * (x1 - x0)
        right = x0 + (end - 0.001) / (0.999 - 0.001) * (x1 - x0)
        rect = Rectangle((left, y), right - left, h, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text((left + right) / 2, y + h / 2, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    ax.annotate(
        "",
        xy=(x1, y - 0.06),
        xytext=(x0, y - 0.06),
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6),
    )
    ax.text(x0, y - 0.13, "Low PD", ha="left", fontsize=10, color=GRAY)
    ax.text(x1, y - 0.13, "High PD", ha="right", fontsize=10, color=GRAY)

    ax.annotate(
        "",
        xy=(x0, y + h + 0.08),
        xytext=(x1, y + h + 0.08),
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6),
    )
    ax.text(x0, y + h + 0.12, "High credit score", ha="left", fontsize=10, color=GRAY)
    ax.text(x1, y + h + 0.12, "Low credit score", ha="right", fontsize=10, color=GRAY)

    formula = r"$score\_credit = 850 - 550 \times pd\_default$"
    ax.text(
        0.5,
        0.25,
        formula,
        ha="center",
        va="center",
        fontsize=15,
        color=DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=LIGHT, edgecolor="#d8dee5"),
    )
    ax.text(
        0.08,
        0.08,
        "Reading: a low probability of default produces a high credit score and a favorable risk segment. "
        "A high probability of default lowers the score and increases the risk segment.",
        fontsize=9,
        color=GRAY,
        ha="left",
    )
    savefig("34_risk_score_generation_scale.png")


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plot_system_workflow()
    plot_risk_score_generation()
    print(f"Workflow figures generated in: {IMG_DIR}")


if __name__ == "__main__":
    main()
