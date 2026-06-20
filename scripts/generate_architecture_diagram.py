from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT = BASE_DIR / "img" / "architecture_systeme.png"


def box(ax, x, y, w, h, title, subtitle, color):
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            w,
            h,
            facecolor=color,
            edgecolor="#8bbcff",
            linewidth=1.4,
            alpha=0.95,
        )
    )
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", color="#cfe2f3", fontsize=8.5)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#d7e7ff", lw=1.8))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#07111f")
    ax.set_facecolor("#07111f")
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    ax.text(7, 6.55, "Architecture du systeme de scoring credit", ha="center", color="white", fontsize=18, fontweight="bold")
    ax.text(7, 6.18, "Donnees -> Machine Learning -> API -> Dashboard -> Monitoring", ha="center", color="#91a4b8", fontsize=10)

    box(ax, 0.5, 4.6, 2.2, 1.0, "Dataset CSV", "100k demandes credit", "#12365a")
    box(ax, 3.2, 4.6, 2.2, 1.0, "EDA", "distributions, correlations", "#164469")
    box(ax, 5.9, 4.6, 2.2, 1.0, "Feature engineering", "ratios, bins, signaux", "#15546a")
    box(ax, 8.6, 4.6, 2.2, 1.0, "Pipeline ML", "SMOTE + GridSearchCV", "#1f5f7d")
    box(ax, 11.3, 4.6, 2.2, 1.0, "Modele", "XGBoost / RF / RL", "#1d6a89")

    box(ax, 3.2, 2.35, 2.2, 1.0, "FastAPI", "/health /predict /docs", "#173a5e")
    box(ax, 5.9, 2.35, 2.2, 1.0, "Streamlit", "dashboard premium", "#174d6c")
    box(ax, 8.6, 2.35, 2.2, 1.0, "Logs", "tracabilite scoring", "#175c68")
    box(ax, 11.3, 2.35, 2.2, 1.0, "Rapports", "PDF, figures, XAI", "#14635c")

    box(ax, 4.6, 0.65, 4.8, 0.85, "Docker / Docker Compose", "environnement reproductible et deploiement local", "#102033")

    arrow(ax, 2.7, 5.1, 3.2, 5.1)
    arrow(ax, 5.4, 5.1, 5.9, 5.1)
    arrow(ax, 8.1, 5.1, 8.6, 5.1)
    arrow(ax, 10.8, 5.1, 11.3, 5.1)
    arrow(ax, 12.4, 4.6, 4.3, 3.35)
    arrow(ax, 12.4, 4.6, 7.0, 3.35)
    arrow(ax, 7.0, 2.35, 9.7, 2.35)
    arrow(ax, 9.7, 2.35, 12.4, 2.35)
    arrow(ax, 7.0, 2.35, 7.0, 1.5)
    arrow(ax, 4.3, 2.35, 6.2, 1.5)
    arrow(ax, 12.4, 2.35, 8.2, 1.5)

    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
