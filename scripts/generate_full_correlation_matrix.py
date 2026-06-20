from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_PATH
from src.data_processing import add_credit_features, basic_cleaning, build_target, read_credit_data


def main() -> None:
    df = read_credit_data(RAW_DATA_PATH)
    df = basic_cleaning(df)
    df = build_target(df)
    df = add_credit_features(df)

    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr(numeric_only=True)

    output_paths = [
        BASE_DIR / "reports" / "figures" / "correlation_matrix_all_variables.png",
        BASE_DIR / "img" / "correlation_matrix_all_variables.png",
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(22, 18))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
        linecolor="#eeeeee",
        square=False,
        cbar_kws={"shrink": 0.75, "label": "Correlation"},
    )
    plt.title("Matrice de correlation complete des variables numeriques", fontsize=18, fontweight="bold", pad=18)
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    for path in output_paths:
        plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

    corr.to_csv(BASE_DIR / "reports" / "correlation_matrix_all_variables.csv")
    print(output_paths[0])
    print(output_paths[1])


if __name__ == "__main__":
    main()
