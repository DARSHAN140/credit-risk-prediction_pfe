from __future__ import annotations

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

from src.config import RAW_DATA_PATH
from src.data_processing import add_credit_features, basic_cleaning, build_target, read_credit_data


IMG_DIR = BASE_DIR / "img"

BLUE = "#2f5597"
RED = "#b04a4a"
GREEN = "#4f7d5a"
ORANGE = "#c77c2b"
GRAY = "#5f6b7a"
DARK = "#1f2a35"
LIGHT_GRAY = "#eef2f5"


def savefig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def format_number(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def load_project_data() -> pd.DataFrame:
    df = read_credit_data(RAW_DATA_PATH)
    df = basic_cleaning(df)
    df = build_target(df)
    df = add_credit_features(df)
    df["classe_risque"] = np.where(df["target_default"] == 1, "Risque", "Favorable")
    return df


def add_explanation(ax, text: str) -> None:
    ax.text(
        0.01,
        -0.22,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color=GRAY,
        va="top",
    )


def plot_dataset_overview(df: pd.DataFrame) -> None:
    raw_variable_count = pd.read_csv(RAW_DATA_PATH, nrows=0).shape[1]
    decision_counts = df["decision"].value_counts()
    default_rate = df["target_default"].mean()
    missing_rate = df.isna().mean().mean()
    duplicate_rows = df.duplicated().sum()

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 2.3])

    cards = [
        ("Dossiers", format_number(len(df))),
        ("Variables initiales", str(raw_variable_count)),
        ("Classe risque", f"{default_rate:.1%}"),
        ("Valeurs manquantes", f"{missing_rate:.1%}"),
    ]
    for i, (label, value) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(LIGHT_GRAY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=20, fontweight="bold", color=DARK)
        ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=10, color=GRAY)

    ax = fig.add_subplot(gs[1, :])
    order = decision_counts.index.tolist()
    colors = [GREEN if "favorable" in str(label).lower() else RED for label in order]
    bars = ax.barh(order, decision_counts.values, color=colors, alpha=0.9)
    ax.set_title("Distribution des decisions initiales", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Nombre de dossiers")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, decision_counts.values):
        ax.text(value + max(decision_counts.values) * 0.01, bar.get_y() + bar.get_height() / 2, format_number(value), va="center", fontsize=10, color=DARK)
    add_explanation(
        ax,
        f"Lecture: la cible binaire regroupe les avis defavorables, non eligibles et etudes approfondies en classe risque. Doublons detectes: {duplicate_rows}.",
    )
    savefig("21_eda_resume_dataset.png")


def plot_numeric_analysis(df: pd.DataFrame) -> None:
    columns = [
        ("revenu_total", "Revenu total"),
        ("reste_a_vivre", "Reste a vivre"),
        ("dti_calcule", "DTI calcule"),
        ("ratio_credit_revenu", "Credit / revenu"),
        ("montant_credit", "Montant credit"),
        ("epargne_mensuelle", "Epargne mensuelle"),
    ]
    available = [(col, label) for col, label in columns if col in df.columns]
    sample = df.sample(min(len(df), 30000), random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (col, label) in zip(axes.flatten(), available):
        low, high = sample[col].quantile([0.01, 0.99])
        work = sample[(sample[col] >= low) & (sample[col] <= high)]
        sns.histplot(
            data=work,
            x=col,
            hue="classe_risque",
            bins=35,
            stat="density",
            common_norm=False,
            palette={"Favorable": GREEN, "Risque": RED},
            alpha=0.45,
            ax=ax,
        )
        medians = work.groupby("target_default")[col].median()
        if 0 in medians:
            ax.axvline(medians[0], color=GREEN, linestyle="--", linewidth=1.3)
        if 1 in medians:
            ax.axvline(medians[1], color=RED, linestyle="--", linewidth=1.3)
        ax.set_title(label, fontsize=11, fontweight="bold", color=DARK)
        ax.set_xlabel("")
        ax.set_ylabel("Densite")
        ax.grid(alpha=0.2)

    for ax in axes.flatten()[len(available):]:
        ax.axis("off")

    fig.suptitle("Analyse exploratoire des variables numeriques", fontsize=16, fontweight="bold", color=DARK, y=1.02)
    fig.text(
        0.01,
        -0.01,
        "Lecture: les lignes pointillees indiquent les medianes par classe. Les valeurs extremes sont limitees aux percentiles 1-99 pour garder une lecture lisible.",
        fontsize=9,
        color=GRAY,
    )
    savefig("22_eda_numerique_profils_financiers.png")


def risk_rate_by_category(df: pd.DataFrame, column: str, min_count: int = 100) -> pd.DataFrame:
    work = df.copy()
    work[column] = work[column].fillna("Non applicable")
    grouped = (
        work.groupby(column, dropna=False)["target_default"]
        .agg(nombre="count", taux_risque="mean")
        .reset_index()
    )
    grouped = grouped[grouped["nombre"] >= min_count].sort_values("taux_risque", ascending=False)
    return grouped


def plot_categorical_analysis(df: pd.DataFrame) -> None:
    columns = [
        ("type_credit", "Type de credit"),
        ("statut_professionnel", "Statut professionnel"),
        ("type_contrat", "Type de contrat"),
        ("objet_credit", "Objet du credit"),
    ]
    available = [(col, label) for col, label in columns if col in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, label) in zip(axes.flatten(), available):
        rates = risk_rate_by_category(df, col).head(8)
        rates[col] = rates[col].astype(str)
        sns.barplot(data=rates, x="taux_risque", y=col, color=BLUE, ax=ax)
        ax.axvline(df["target_default"].mean(), color=RED, linestyle="--", linewidth=1.2)
        ax.set_title(label, fontsize=11, fontweight="bold", color=DARK)
        ax.set_xlabel("Taux de dossiers risques")
        ax.set_ylabel("")
        ax.set_xlim(0, min(1, max(rates["taux_risque"].max() + 0.12, 0.6)))
        ax.grid(axis="x", alpha=0.25)
        for idx, row in rates.reset_index(drop=True).iterrows():
            ax.text(
                row["taux_risque"] + 0.01,
                idx,
                f"{row['taux_risque']:.0%} | n={int(row['nombre'])}",
                va="center",
                fontsize=8.5,
                color=DARK,
            )

    for ax in axes.flatten()[len(available):]:
        ax.axis("off")

    fig.suptitle("Analyse exploratoire des variables categorielles", fontsize=16, fontweight="bold", color=DARK, y=1.02)
    fig.text(
        0.01,
        -0.01,
        "Lecture: la ligne rouge represente le taux de risque moyen du dataset. Les barres au-dessus de cette ligne indiquent des categories plus sensibles.",
        fontsize=9,
        color=GRAY,
    )
    savefig("23_eda_categorique_taux_risque.png")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    columns = [
        "target_default",
        "revenu_total",
        "reste_a_vivre",
        "dti_calcule",
        "ratio_credit_revenu",
        "revenu_mensuel",
        "charges_mensuelles",
        "mensualites_existantes",
        "montant_credit",
        "duree_mois",
        "epargne_mensuelle",
        "solde_moyen_compte",
        "incidents_paiement_12m",
        "retards_paiement_12m",
        "anciennete_bancaire_mois",
    ]
    available = [col for col in columns if col in df.columns]
    corr = df[available].corr(numeric_only=True)
    display_names = {
        "target_default": "Cible risque",
        "revenu_total": "Revenu total",
        "reste_a_vivre": "Reste a vivre",
        "dti_calcule": "DTI",
        "ratio_credit_revenu": "Credit/revenu",
        "revenu_mensuel": "Revenu mensuel",
        "charges_mensuelles": "Charges",
        "mensualites_existantes": "Mensualites",
        "montant_credit": "Montant credit",
        "duree_mois": "Duree",
        "epargne_mensuelle": "Epargne",
        "solde_moyen_compte": "Solde moyen",
        "incidents_paiement_12m": "Incidents",
        "retards_paiement_12m": "Retards",
        "anciennete_bancaire_mois": "Anciennete banque",
    }
    corr = corr.rename(index=display_names, columns=display_names)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12.8, 9.4))
    ax = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )
    ax.set_title("Matrice de correlation des variables metier", fontsize=16, fontweight="bold", color=DARK, loc="left", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    savefig("24_matrice_correlation_variables_metier.png")


def plot_target_correlation_rank(df: pd.DataFrame) -> None:
    columns = [
        "revenu_total",
        "reste_a_vivre",
        "dti_calcule",
        "ratio_credit_revenu",
        "revenu_mensuel",
        "charges_mensuelles",
        "mensualites_existantes",
        "montant_credit",
        "duree_mois",
        "epargne_mensuelle",
        "solde_moyen_compte",
        "incidents_paiement_12m",
        "retards_paiement_12m",
        "anciennete_bancaire_mois",
        "taux_epargne",
        "incident_ou_retard",
    ]
    available = [col for col in columns if col in df.columns]
    corr = df[available + ["target_default"]].corr(numeric_only=True)["target_default"].drop("target_default")
    corr = corr.reindex(corr.abs().sort_values().index)
    display_names = {
        "revenu_total": "Revenu total",
        "reste_a_vivre": "Reste a vivre",
        "dti_calcule": "DTI calcule",
        "ratio_credit_revenu": "Credit / revenu",
        "revenu_mensuel": "Revenu mensuel",
        "charges_mensuelles": "Charges mensuelles",
        "mensualites_existantes": "Mensualites existantes",
        "montant_credit": "Montant credit",
        "duree_mois": "Duree",
        "epargne_mensuelle": "Epargne mensuelle",
        "solde_moyen_compte": "Solde moyen",
        "incidents_paiement_12m": "Incidents 12 mois",
        "retards_paiement_12m": "Retards 12 mois",
        "anciennete_bancaire_mois": "Anciennete bancaire",
        "taux_epargne": "Taux epargne",
        "incident_ou_retard": "Incident ou retard",
    }
    corr.index = [display_names.get(index, index) for index in corr.index]

    colors = [RED if value > 0 else GREEN for value in corr.values]
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.barh(corr.index, corr.values, color=colors, alpha=0.9)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.set_title("Variables numeriques les plus liees a la cible", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Correlation avec target_default")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(corr.values):
        ha = "left" if value >= 0 else "right"
        offset = 0.015 if value >= 0 else -0.015
        ax.text(value + offset, y, f"{value:.2f}", va="center", ha=ha, fontsize=9, color=DARK)
    add_explanation(
        ax,
        "Lecture: rouge = relation positive avec le risque; vert = relation negative. Cette figure aide a expliquer les facteurs metier les plus influents.",
    )
    savefig("25_correlation_cible_top_variables.png")


def plot_missing_values_context(df: pd.DataFrame) -> None:
    missing = df.isna().mean().sort_values(ascending=True)
    missing = missing[missing > 0].tail(14)

    plt.figure(figsize=(10, 6.5))
    ax = plt.gca()
    bars = ax.barh(missing.index, missing.values, color=ORANGE, alpha=0.85)
    ax.set_title("Valeurs manquantes liees au type de credit", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Taux de valeurs manquantes")
    ax.set_ylabel("")
    ax.set_xlim(0, min(1, missing.max() + 0.12))
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, missing.values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.0%}", va="center", fontsize=9, color=DARK)
    add_explanation(
        ax,
        "Lecture: ces absences sont principalement structurelles. Par exemple, les variables vehicule ne concernent pas tous les credits, et les variables bien/garantie concernent surtout l'immobilier.",
    )
    savefig("26_valeurs_manquantes_contextuelles.png")


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d0d6dc",
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    df = load_project_data()
    plot_dataset_overview(df)
    plot_numeric_analysis(df)
    plot_categorical_analysis(df)
    plot_correlation_matrix(df)
    plot_target_correlation_rank(df)
    plot_missing_values_context(df)
    print(f"EDA figures generated in: {IMG_DIR}")


if __name__ == "__main__":
    main()
