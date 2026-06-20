from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif

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

DARK = "#1f2a35"
GRAY = "#657280"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
RED = "#b04a4a"
ORANGE = "#c77c2b"
LIGHT = "#eef2f5"


def savefig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def load_data() -> pd.DataFrame:
    df = read_credit_data(RAW_DATA_PATH)
    df = basic_cleaning(df)
    df = build_target(df)
    df = add_credit_features(df)
    return df


def load_training_sample() -> pd.DataFrame:
    raw_df = read_credit_data(RAW_DATA_PATH)
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = (
            raw_df.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw_df), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    return raw_df


def plot_feature_creation() -> None:
    rows = [
        ("revenu_total", "revenu_mensuel + revenu_supplementaire", "Global income capacity"),
        ("dti_calcule", "(mensualites_existantes + mensualite_estimee) / revenu_total", "Debt burden"),
        ("reste_a_vivre", "revenu_total - charges_mensuelles - mensualites_totales", "Disposable income"),
        ("taux_epargne", "epargne_mensuelle / revenu_total", "Savings capacity"),
        ("ratio_credit_revenu", "montant_credit / revenu_total", "Loan pressure"),
        ("incident_ou_retard", "1 if incident or delay exists, else 0", "Banking behavior"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_title("Feature creation from banking business rules", fontsize=17, fontweight="bold", color=DARK, loc="left", pad=18)

    y_positions = np.linspace(0.82, 0.14, len(rows))
    for y, (name, formula, meaning) in zip(y_positions, rows):
        ax.text(
            0.03,
            y,
            name,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.42", facecolor=BLUE, edgecolor=BLUE),
        )
        ax.annotate("", xy=(0.34, y), xytext=(0.23, y), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3))
        ax.text(
            0.36,
            y,
            formula,
            ha="left",
            va="center",
            fontsize=10,
            color=DARK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT, edgecolor="#d8dee5"),
        )
        ax.annotate("", xy=(0.77, y), xytext=(0.69, y), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3))
        ax.text(
            0.79,
            y,
            meaning,
            ha="left",
            va="center",
            fontsize=10,
            color=DARK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d8dee5"),
        )

    ax.text(
        0.03,
        0.03,
        "Reading: raw financial variables are transformed into business indicators that describe solvency, debt pressure and payment behavior.",
        fontsize=9,
        color=GRAY,
    )
    savefig("27_feature_creation_business_indicators.png")


def original_feature_name(transformed_name: str) -> str:
    name = transformed_name.split("__", 1)[-1]
    known_prefixes = [
        "situation_familiale",
        "nombre_personnes_charge",
        "anciennete_emploi_mois",
        "revenu_supplementaire",
        "charges_mensuelles",
        "credits_en_cours",
        "mensualites_existantes",
        "epargne_mensuelle",
        "solde_moyen_compte",
        "incidents_paiement_12m",
        "retards_paiement_12m",
        "anciennete_bancaire_mois",
        "montant_credit_bin",
        "montant_credit",
        "ratio_credit_revenu",
        "incident_ou_retard",
        "statut_professionnel",
        "secteur_activite",
        "type_logement",
        "type_contrat",
        "type_credit",
        "objet_credit",
        "type_vehicule",
        "marque_vehicule",
        "type_bien",
        "localisation_bien",
        "garantie_presente",
        "type_garantie",
        "revenu_mensuel",
        "revenu_total",
        "reste_a_vivre",
        "taux_epargne",
        "dti_calcule",
        "age_bin",
        "revenu_bin",
        "niveau_etude",
        "duree_mois",
        "taux_interet",
        "apport_personnel",
        "prix_vehicule",
        "age_vehicule",
        "annee_demande",
        "mois_demande",
        "ville",
        "sexe",
        "age",
    ]
    for prefix in sorted(known_prefixes, key=len, reverse=True):
        if name == prefix or name.startswith(f"{prefix}_"):
            return prefix
    return re.sub(r"_[^_]+$", "", name)


def plot_feature_selection() -> None:
    raw_df = load_training_sample()
    X, y = prepare_model_frame(raw_df)
    X_train, _, y_train, _ = temporal_or_stratified_split(raw_df, X, y)
    numeric_cols, categorical_cols = infer_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_encoded = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    scores = mutual_info_classif(X_encoded, y_train, random_state=RANDOM_STATE)

    score_df = pd.DataFrame(
        {
            "feature": [original_feature_name(name) for name in feature_names],
            "score": scores,
        }
    )
    ranking = (
        score_df.groupby("feature", as_index=False)["score"]
        .max()
        .sort_values("score", ascending=False)
        .head(12)
        .sort_values("score")
    )

    fig, ax = plt.subplots(figsize=(10.5, 7))
    bars = ax.barh(ranking["feature"], ranking["score"], color=BLUE, alpha=0.9)
    ax.set_title("Feature selection using mutual information", fontsize=16, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Mutual information score")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, ranking["score"]):
        ax.text(value + ranking["score"].max() * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9, color=DARK)
    ax.text(
        0,
        -0.17,
        "Reading: higher scores indicate variables that provide more information about the target. This ranking supports SelectKBest in the ML pipeline.",
        transform=ax.transAxes,
        fontsize=9,
        color=GRAY,
    )
    savefig("28_feature_selection_mutual_information.png")


def plot_engineered_correlation(df: pd.DataFrame) -> None:
    columns = [
        "revenu_total",
        "reste_a_vivre",
        "dti_calcule",
        "taux_epargne",
        "ratio_credit_revenu",
        "incident_ou_retard",
        "target_default",
    ]
    available = [col for col in columns if col in df.columns]
    corr = df[available].corr(numeric_only=True)
    display_names = {
        "revenu_total": "Revenu total",
        "reste_a_vivre": "Reste a vivre",
        "dti_calcule": "DTI",
        "taux_epargne": "Taux epargne",
        "ratio_credit_revenu": "Credit/revenu",
        "incident_ou_retard": "Incident/retard",
        "target_default": "Cible risque",
    }
    corr = corr.rename(index=display_names, columns=display_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.85, "label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation analysis of engineered variables", fontsize=16, fontweight="bold", color=DARK, loc="left", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    savefig("29_feature_engineering_correlation.png")


def plot_smote_moderate() -> None:
    raw_df = load_training_sample()
    X, y = prepare_model_frame(raw_df)
    X_train, _, y_train, _ = temporal_or_stratified_split(raw_df, X, y)
    numeric_cols, categorical_cols = infer_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_prepared = preprocessor.fit_transform(X_train)

    smote = SMOTE(sampling_strategy=0.9, random_state=RANDOM_STATE)
    _, y_smote = smote.fit_resample(X_prepared, y_train)

    before = y_train.value_counts().sort_index()
    after = pd.Series(y_smote).value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "step": ["Before SMOTE", "Before SMOTE", "After moderate SMOTE", "After moderate SMOTE"],
            "class": ["Favorable", "Risk", "Favorable", "Risk"],
            "count": [
                before.get(0, 0),
                before.get(1, 0),
                after.get(0, 0),
                after.get(1, 0),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.barplot(data=plot_df, x="step", y="count", hue="class", palette={"Favorable": GREEN, "Risk": RED}, ax=ax)
    ax.set_title("Moderate SMOTE applied only on the training set", fontsize=16, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Number of observations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Class")

    for container in ax.containers:
        ax.bar_label(container, fmt=lambda value: f"{value:,.0f}".replace(",", " "), fontsize=9, padding=3)

    majority_after = after.max()
    minority_after = after.min()
    ratio = minority_after / majority_after if majority_after else 0
    ax.text(
        0,
        -0.22,
        f"Reading: SMOTE increases the risky class to about {ratio:.0%} of the majority class. The test set is not resampled, so evaluation remains realistic.",
        transform=ax.transAxes,
        fontsize=9,
        color=GRAY,
    )
    savefig("30_smote_moderate_training_balance.png")


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

    df = load_data()
    plot_feature_creation()
    plot_feature_selection()
    plot_engineered_correlation(df)
    plot_smote_moderate()
    print(f"Feature engineering figures generated in: {IMG_DIR}")


if __name__ == "__main__":
    main()
