from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_PATH
from src.data_processing import prepare_model_frame, read_credit_data


def main() -> None:
    raw_df = read_credit_data(RAW_DATA_PATH)
    X, y = prepare_model_frame(raw_df)
    model_df = X.copy()
    model_df["target_default"] = y

    numeric_df = model_df.select_dtypes(include=["number"])
    corr = numeric_df.corr(numeric_only=True)

    figure_paths = [
        BASE_DIR / "reports" / "figures" / "correlation_matrix_after_processing.png",
        BASE_DIR / "img" / "correlation_matrix_after_processing.png",
    ]
    for path in figure_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
        linecolor="#eeeeee",
        square=False,
        cbar_kws={"shrink": 0.75, "label": "Correlation"},
    )
    plt.title(
        "Matrice de correlation apres traitement et suppression des variables exclues",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    for path in figure_paths:
        plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

    corr.to_csv(BASE_DIR / "reports" / "correlation_matrix_after_processing.csv")

    target_corr = (
        corr["target_default"]
        .drop("target_default")
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )
    target_corr.columns = ["variable", "correlation_abs_target"]
    target_corr.to_csv(BASE_DIR / "reports" / "target_correlation_after_processing.csv", index=False)

    print(figure_paths[0])
    print(figure_paths[1])


if __name__ == "__main__":
    main()
