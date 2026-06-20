import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURE_DIR, RAW_DATA_PATH, REPORT_DIR
from src.data_processing import build_target, read_credit_data


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = read_credit_data(RAW_DATA_PATH)
    df = build_target(df)

    summary = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "default_rate": df["target_default"].mean(),
        "missing_rate_mean": df.isna().mean().mean(),
        "duplicate_rows": df.duplicated().sum(),
    }
    pd.Series(summary).to_json(REPORT_DIR / "eda_summary.json", indent=2)
    df.isna().mean().sort_values(ascending=False).to_csv(REPORT_DIR / "missing_values.csv")

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="target_default")
    plt.title("Distribution de la cible")
    plt.xlabel("Defaut: 1 = avis defavorable")
    plt.ylabel("Nombre de dossiers")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_distribution.png", dpi=150)
    plt.close()

    numeric_cols = [
        col
        for col in ["age", "revenu_mensuel", "montant_credit", "duree_mois", "taux_endettement"]
        if col in df.columns
    ]
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue="target_default", bins=40, kde=True, stat="density", common_norm=False)
        plt.title(f"Distribution de {col}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"distribution_{col}.png", dpi=150)
        plt.close()

    if len(numeric_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols + ["target_default"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation des variables numeriques")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "correlation_matrix.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
