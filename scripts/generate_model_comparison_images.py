from __future__ import annotations

import ast
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MAX_REPORTED_METRIC

IMG_DIR = BASE_DIR / "img"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"

DISPLAY_NAMES = {
    "logistic_regression": "Regression logistique",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "ann": "ANN",
}
COLORS = {
    "logistic_regression": "#f0ad4e",
    "decision_tree": "#8a6fdf",
    "random_forest": "#5cb85c",
    "svm": "#d9534f",
    "xgboost": "#337ab7",
    "lightgbm": "#20a387",
    "ann": "#7b4ab8",
}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "gini", "ks"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Gini", "KS"]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(COMPARISON_PATH)
    for metric in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce").clip(upper=MAX_REPORTED_METRIC)
    df["nom_modele"] = df["model"].map(DISPLAY_NAMES).fillna(df["model"])
    return df


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def comparison_bar_chart(df: pd.DataFrame, file_name: str) -> None:
    metrics = ["roc_auc", "f1", "precision", "recall"]
    labels = ["ROC-AUC", "F1-score", "Precision", "Recall"]
    x = np.arange(len(df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1.5) * width, df[metric], width=width, label=label)

    ax.set_title("Comparaison des modeles")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(df["nom_modele"], rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save(fig, file_name)


def radar_chart(df: pd.DataFrame) -> None:
    radar_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "ks"]
    radar_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "KS"]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        values = [row[m] for m in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, label=row["nom_modele"], color=COLORS.get(row["model"], "#777777"), linewidth=2)
        ax.fill(angles, values, color=COLORS.get(row["model"], "#777777"), alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Profil des modeles")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save(fig, "11_radar_modeles.png")


def format_params(raw: str) -> str:
    params = ast.literal_eval(raw)
    lines = []
    for key, value in params.items():
        clean_key = key.replace("model__", "").replace("select__", "")
        lines.append(f"{clean_key}: {value}")
    return "\n".join(lines)


def model_summary(df: pd.DataFrame, model: str, file_name: str) -> None:
    row = df.loc[df["model"] == model].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.2, 1]})

    values = [row[m] for m in METRICS]
    bars = ax1.barh(METRIC_LABELS, values, color=COLORS.get(model, "#777777"))
    ax1.set_xlim(0, 1.05)
    ax1.set_title(f"Metriques - {row['nom_modele']}")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        ax1.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")

    ax2.axis("off")
    text = (
        f"Modele : {row['nom_modele']}\n\n"
        f"ROC-AUC : {row['roc_auc']:.3f}\n"
        f"F1-score : {row['f1']:.3f}\n"
        f"Precision : {row['precision']:.3f}\n"
        f"Recall : {row['recall']:.3f}\n\n"
        f"Meilleurs parametres :\n{format_params(row['best_params'])}"
    )
    ax2.text(0.02, 0.95, text, va="top", fontsize=11)
    save(fig, file_name)


def ranking_chart(df: pd.DataFrame) -> None:
    ranking = df.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(ranking["nom_modele"], ranking["roc_auc"], color=[COLORS.get(m, "#777777") for m in ranking["model"]])
    ax.set_xlim(0.60, 0.92)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Classement des modeles selon ROC-AUC")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, ranking["roc_auc"]):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    save(fig, "15_classement_final_modeles.png")


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    comparison_bar_chart(df, "09_comparaison_modeles.png")
    comparison_bar_chart(df, "10_comparaison_modeles_detaillee.png")
    radar_chart(df)
    summary_files = {
        "logistic_regression": "12_modele_logistic_regression.png",
        "random_forest": "13_modele_random_forest.png",
        "xgboost": "14_modele_xgboost.png",
        "svm": "15_modele_svm.png",
        "decision_tree": "16_modele_decision_tree.png",
        "lightgbm": "17_modele_lightgbm.png",
        "ann": "18_modele_ann.png",
    }
    for model, file_name in summary_files.items():
        if model in set(df["model"]):
            model_summary(df, model, file_name)
    ranking_chart(df)
    print(IMG_DIR)


if __name__ == "__main__":
    main()
