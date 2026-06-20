from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#66727f"
LIGHT = "#eef2f5"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
ORANGE = "#c77c2b"
PURPLE = "#6b4fa3"
TEAL = "#2f7f7f"


def savefig(filename: str) -> None:
    IMG_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def box(ax, xy, width, height, text, color, fontsize=9.2, text_color="white"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=color,
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
    )


def outline_box(ax, xy, width, height, text, edge, fill="#ffffff", fontsize=8.8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.022",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
        fontweight="bold",
    )


def arrow(ax, start, end, color=GRAY, rad=0.0, lw=1.45, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def plot_docker_configuration() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.96, "Docker configuration of the BMCE credit risk project", fontsize=17, color=DARK, fontweight="bold", va="top")
    ax.text(
        0.03,
        0.91,
        "Two containers are built from the same Python image and expose the API and Streamlit dashboard.",
        fontsize=9.5,
        color=GRAY,
        va="top",
    )

    # Host layer
    ax.add_patch(
        FancyBboxPatch(
            (0.035, 0.08),
            0.93,
            0.76,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor="#ffffff",
            edgecolor="#d8dee4",
            linewidth=1.4,
        )
    )
    ax.text(0.06, 0.805, "Docker host", fontsize=12.5, color=DARK, fontweight="bold")

    # Build context and Dockerfile
    box(ax, (0.07, 0.62), 0.17, 0.11, "Project source\nDocker build context", BLUE, fontsize=8.8)
    outline_box(ax, (0.07, 0.48), 0.17, 0.085, "Dockerfile\npython:3.11-slim\nrequirements.txt", BLUE, fill=LIGHT, fontsize=8.0)
    arrow(ax, (0.155, 0.62), (0.155, 0.565), color=BLUE)

    # Shared image
    box(ax, (0.33, 0.55), 0.18, 0.13, "Application image\n/app workspace\nFastAPI + Streamlit code", ORANGE, fontsize=8.4)
    arrow(ax, (0.24, 0.675), (0.33, 0.62), color=GRAY)
    arrow(ax, (0.24, 0.522), (0.33, 0.60), color=GRAY)

    # Containers
    api_xy = (0.62, 0.61)
    dash_xy = (0.62, 0.36)
    box(ax, api_xy, 0.20, 0.13, "bmce-credit-api\nuvicorn api.main:app\n0.0.0.0:8000", GREEN, fontsize=8.2)
    box(ax, dash_xy, 0.20, 0.13, "bmce-credit-dashboard\nstreamlit dashboard/app.py\n0.0.0.0:8501", PURPLE, fontsize=8.0)
    arrow(ax, (0.51, 0.615), (0.62, 0.675), color=ORANGE)
    arrow(ax, (0.51, 0.585), (0.62, 0.425), color=ORANGE)
    arrow(ax, (0.72, 0.49), (0.72, 0.61), color=PURPLE, style="<|-", lw=1.4)
    ax.text(0.735, 0.55, "depends_on", fontsize=8.5, color=GRAY, rotation=90, va="center")

    # User-facing ports
    outline_box(ax, (0.84, 0.64), 0.10, 0.075, "Port 8000\nFastAPI /docs", GREEN, fill="#f4fbf6", fontsize=7.7)
    outline_box(ax, (0.84, 0.39), 0.10, 0.075, "Port 8501\nDashboard UI", PURPLE, fill="#f8f5ff", fontsize=7.7)
    arrow(ax, (0.82, 0.675), (0.84, 0.675), color=GREEN)
    arrow(ax, (0.82, 0.425), (0.84, 0.425), color=PURPLE)

    # Volumes
    ax.text(0.08, 0.35, "Mounted volumes", fontsize=11.5, color=DARK, fontweight="bold")
    volumes = [
        ("./models", "/app/models\ntrained model artifact", TEAL),
        ("./logs", "/app/logs\nprediction logs", ORANGE),
        ("./reports", "/app/reports\nmetrics and audit files", BLUE),
    ]
    for i, (host, cont, color) in enumerate(volumes):
        x = 0.08 + i * 0.18
        outline_box(ax, (x, 0.22), 0.15, 0.08, f"{host}\n{cont}", color, fill=LIGHT, fontsize=7.3)
        arrow(ax, (x + 0.15, 0.26), (0.62, 0.66 - i * 0.11), color=color, rad=0.05, lw=1.1)
        if i > 0:
            arrow(ax, (x + 0.15, 0.24), (0.62, 0.41 - (i - 1) * 0.03), color=color, rad=-0.03, lw=1.1)

    # .dockerignore note
    ax.add_patch(Rectangle((0.08, 0.125), 0.36, 0.045, facecolor="#f7f9fb", edgecolor="#d8dee4"))
    ax.text(
        0.095,
        0.147,
        ".dockerignore excludes virtual environments, cache files, Git metadata, processed data and generated logs.",
        fontsize=7.6,
        color=DARK,
        va="center",
    )

    # External user
    box(ax, (0.84, 0.18), 0.10, 0.075, "User\nBrowser", DARK, fontsize=8.2)
    arrow(ax, (0.89, 0.255), (0.89, 0.39), color=GRAY)
    arrow(ax, (0.89, 0.715), (0.89, 0.255), color=GRAY, style="<|-", lw=1.1)

    savefig("45_docker_configuration_project.png")


def main() -> None:
    plot_docker_configuration()
    print("Generated Docker configuration image:")
    print(" - img/45_docker_configuration_project.png")


if __name__ == "__main__":
    main()
