from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MAX_TRAIN_ROWS, RANDOM_STATE, RAW_DATA_PATH
from src.data_processing import (
    add_credit_features,
    basic_cleaning,
    build_target,
    infer_column_types,
    prepare_model_frame,
    read_credit_data,
    temporal_or_stratified_split,
)
from src.modeling import build_preprocessor


IMG_DIR = BASE_DIR / "img"

TITLE_COLOR = "#102033"
BLUE = "#1f6feb"
GREEN = "#138a36"
ORANGE = "#d97706"
RED = "#b42318"
GRAY = "#52677d"


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def add_title(title: str, subtitle: str | None = None) -> None:
    plt.suptitle(title, fontsize=15, fontweight="bold", color=TITLE_COLOR, y=1.02)
    if subtitle:
        plt.title(subtitle, fontsize=10, color=GRAY, pad=12)


def load_data() -> pd.DataFrame:
    df = read_credit_data(RAW_DATA_PATH)
    df = basic_cleaning(df)
    df = build_target(df)
    df = add_credit_features(df)
    return df


def plot_eda_pipeline() -> None:
    steps = [
        "CSV brut",
        "Nettoyage",
        "Cible risque",
        "EDA",
        "Feature engineering",
        "Preprocessing",
        "SMOTE",
        "GridSearchCV",
        "Evaluation",
    ]
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.axis("off")
    xs = np.linspace(0.05, 0.95, len(steps))
    for i, (x, step) in enumerate(zip(xs, steps)):
        color = BLUE if i < 6 else GREEN
        ax.text(
            x,
            0.55,
            step,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor=color),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.045, 0.55),
                xytext=(x + 0.045, 0.55),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5),
            )
    ax.text(0.5, 0.88, "Processus complet d'analyse et de modelisation", ha="center", fontsize=16, fontweight="bold", color=TITLE_COLOR)
    ax.text(0.5, 0.18, "De la donnee bancaire brute jusqu'au scoring exploitable dans le dashboard", ha="center", fontsize=10, color=GRAY)
    savefig("00_processus_eda_pipeline.png")


def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df["target_default"].value_counts().sort_index()
    labels = ["Avis favorable", "Risque / revue"]
    colors = [GREEN, RED]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts.values, color=colors)
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}".replace(",", " "), ha="center", va="bottom", fontweight="bold")
    add_title("Distribution de la variable cible", "0 = favorable, 1 = avis defavorable / etude approfondie / non eligible")
    plt.ylabel("Nombre de dossiers")
    savefig("01_distribution_cible.png")


def plot_decision_distribution(df: pd.DataFrame) -> None:
    counts = df["decision"].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, color=BLUE)
    add_title("Distribution des decisions initiales", "Repartition avant regroupement en cible binaire")
    plt.xlabel("Nombre de dossiers")
    plt.ylabel("")
    savefig("02_distribution_decisions.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index, color=ORANGE)
    add_title("Top 15 des valeurs manquantes", "Les garanties, biens et vehicules sont conditionnels au type de credit")
    plt.xlabel("Taux de valeurs manquantes")
    plt.ylabel("")
    plt.xlim(0, max(missing.max() + 0.05, 0.1))
    savefig("03_valeurs_manquantes.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    columns = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "charges_mensuelles", "apport_personnel"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), columns):
        sns.histplot(data=df, x=col, hue="target_default", bins=35, stat="density", common_norm=False, ax=ax, palette=[GREEN, RED])
        ax.set_title(col, fontsize=11, fontweight="bold", color=TITLE_COLOR)
        ax.set_xlabel("")
        ax.set_ylabel("Densite")
    fig.suptitle("Distributions numeriques selon la cible", fontsize=15, fontweight="bold", color=TITLE_COLOR, y=1.01)
    savefig("04_distributions_numeriques.png")


def plot_payment_behavior(df: pd.DataFrame) -> None:
    cols = ["incidents_paiement_12m", "retards_paiement_12m", "credits_en_cours"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, cols):
        rates = df.groupby(col)["target_default"].mean().reset_index()
        sns.barplot(data=rates, x=col, y="target_default", color=RED, ax=ax)
        ax.set_title(col, fontsize=11, fontweight="bold", color=TITLE_COLOR)
        ax.set_ylabel("Taux de dossiers risques")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
    fig.suptitle("Comportement bancaire et risque", fontsize=15, fontweight="bold", color=TITLE_COLOR, y=1.03)
    savefig("05_comportement_paiement.png")


def plot_correlation(df: pd.DataFrame) -> None:
    cols = [
        "target_default",
        "revenu_mensuel",
        "charges_mensuelles",
        "montant_credit",
        "duree_mois",
        "apport_personnel",
        "incidents_paiement_12m",
        "retards_paiement_12m",
        "revenu_total",
        "dti_calcule",
        "reste_a_vivre",
        "ratio_credit_revenu",
    ]
    available = [col for col in cols if col in df.columns]
    corr = df[available].corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=False, linewidths=0.4)
    add_title("Matrice de correlation des variables metier")
    savefig("06_matrice_correlation.png")


def plot_feature_engineering_summary(df: pd.DataFrame) -> None:
    engineered = ["revenu_total", "dti_calcule", "reste_a_vivre", "taux_epargne", "ratio_credit_revenu", "incident_ou_retard"]
    corr = df[engineered + ["target_default"]].corr(numeric_only=True)["target_default"].drop("target_default").sort_values()
    colors = [RED if value > 0 else GREEN for value in corr.values]
    plt.figure(figsize=(10, 5.5))
    plt.barh(corr.index, corr.values, color=colors)
    plt.axvline(0, color="#222222", lw=0.8)
    add_title("Impact des variables creees", "Correlation avec la cible apres feature engineering")
    plt.xlabel("Correlation avec target_default")
    savefig("07_feature_engineering.png")


def plot_smote_effect() -> None:
    raw = read_credit_data(RAW_DATA_PATH)
    if len(raw) > MAX_TRAIN_ROWS:
        raw = (
            raw.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    X, y = prepare_model_frame(raw)
    X_train, _, y_train, _ = temporal_or_stratified_split(raw, X, y)
    numeric_cols, categorical_cols = infer_column_types(X_train)
    X_prepared = build_preprocessor(numeric_cols, categorical_cols).fit_transform(X_train)
    _, y_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X_prepared, y_train)

    before = y_train.value_counts().sort_index()
    after = y_smote.value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "classe": ["0 favorable", "1 risque", "0 favorable", "1 risque"],
            "nombre": [before.get(0, 0), before.get(1, 0), after.get(0, 0), after.get(1, 0)],
            "etape": ["Avant SMOTE", "Avant SMOTE", "Apres SMOTE", "Apres SMOTE"],
        }
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="etape", y="nombre", hue="classe", palette=[GREEN, RED])
    add_title("Effet de SMOTE sur l'ensemble d'entrainement", "Reequilibrage de la classe risque sans toucher au jeu de test")
    plt.xlabel("")
    plt.ylabel("Nombre d'observations")
    savefig("08_avant_apres_smote.png")


def plot_model_comparison() -> None:
    comparison_path = BASE_DIR / "reports" / "model_comparison.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    metrics = comparison.melt(id_vars="model", value_vars=["roc_auc", "f1", "recall", "precision"], var_name="metrique", value_name="score")
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=metrics, x="model", y="score", hue="metrique")
    plt.ylim(0, 1)
    add_title("Comparaison des modeles apres SMOTE et GridSearchCV")
    plt.xlabel("")
    plt.ylabel("Score")
    savefig("09_comparaison_modeles.png")


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_data()
    plot_eda_pipeline()
    plot_target_distribution(df)
    plot_decision_distribution(df)
    plot_missing_values(df)
    plot_numeric_distributions(df)
    plot_payment_behavior(df)
    plot_correlation(df)
    plot_feature_engineering_summary(df)
    plot_smote_effect()
    plot_model_comparison()
    print(IMG_DIR)


if __name__ == "__main__":
    main()
