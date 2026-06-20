from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MAX_REPORTED_METRIC, MAX_TRAIN_ROWS, RANDOM_STATE, RAW_DATA_PATH
from src.data_processing import (
    infer_column_types,
    prepare_model_frame,
    read_credit_data,
    temporal_or_stratified_split,
)
from src.modeling import build_model_searches, build_preprocessor


IMG_DIR = BASE_DIR / "img"
NAMES = {
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
INDEXES = {
    "logistic_regression": "16",
    "random_forest": "17",
    "xgboost": "18",
    "svm": "19",
    "decision_tree": "20",
    "lightgbm": "21",
    "ann": "22",
}


def prepare_data():
    raw_df = read_credit_data(RAW_DATA_PATH)
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = (
            raw_df.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw_df), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    X, y = prepare_model_frame(raw_df)
    return temporal_or_stratified_split(raw_df, X, y)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc(name: str, y_true, y_score) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    score = min(float(roc_auc_score(y_true, y_score)), MAX_REPORTED_METRIC)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"AUC = {score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
    ax.set_title(f"Courbe ROC - {NAMES[name]}")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, f"{INDEXES[name]}_roc_{name}.png")


def plot_precision_recall(name: str, y_true, y_score) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = min(float(auc(recall, precision)), MAX_REPORTED_METRIC)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(recall, precision, color=COLORS[name], lw=2, label=f"AUC-PR = {pr_auc:.3f}")
    ax.set_title(f"Courbe Precision-Recall - {NAMES[name]}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    save(fig, f"{INDEXES[name]}_precision_recall_{name}.png")


def plot_confusion(name: str, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Matrice de confusion - {NAMES[name]}")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Reel")
    ax.set_xticklabels(["Favorable", "Risque"])
    ax.set_yticklabels(["Favorable", "Risque"], rotation=0)
    save(fig, f"{INDEXES[name]}_confusion_{name}.png")


def plot_combined_roc(scores: dict[str, tuple]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_score, _) in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        score = min(float(roc_auc_score(y_true, y_score)), MAX_REPORTED_METRIC)
        ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{NAMES[name]} (AUC = {score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
    ax.set_title("Comparaison des courbes ROC")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "23_roc_comparaison_tous_modeles.png")


def plot_combined_pr(scores: dict[str, tuple]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_score, _) in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = min(float(auc(recall, precision)), MAX_REPORTED_METRIC)
        ax.plot(recall, precision, lw=2, color=COLORS[name], label=f"{NAMES[name]} (AUC-PR = {pr_auc:.3f})")
    ax.set_title("Comparaison des courbes Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    save(fig, "24_precision_recall_tous_modeles.png")


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()
    numeric_cols, categorical_cols = infer_column_types(X_train)
    searches = build_model_searches(build_preprocessor(numeric_cols, categorical_cols), numeric_cols, categorical_cols)

    scores = {}
    for name, search in searches.items():
        print(f"Training {name}...")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        scores[name] = (y_test, y_score, y_pred)
        plot_roc(name, y_test, y_score)
        plot_precision_recall(name, y_test, y_score)
        plot_confusion(name, y_test, y_pred)

    plot_combined_roc(scores)
    plot_combined_pr(scores)
    print(IMG_DIR)


if __name__ == "__main__":
    main()
