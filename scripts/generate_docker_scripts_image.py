from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#66727f"
LIGHT = "#f4f6f8"
PANEL = "#111827"
PANEL_HEADER = "#263241"
GREEN = "#8fd694"
BLUE = "#88c0d0"
ORANGE = "#f0b35f"
PURPLE = "#c4a7e7"
WHITE = "#f8fafc"


def read_file(name: str) -> list[str]:
    return (BASE_DIR / name).read_text(encoding="utf-8").splitlines()


def savefig(filename: str) -> None:
    IMG_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def draw_code_panel(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    lines: list[str],
    accent: str,
    font_size: float = 8.3,
    max_lines: int | None = None,
) -> None:
    x, y = xy
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor="#334155",
        facecolor=PANEL,
    )
    ax.add_patch(panel)

    header_h = 0.075
    header = FancyBboxPatch(
        (x, y + height - header_h),
        width,
        header_h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0,
        facecolor=PANEL_HEADER,
    )
    ax.add_patch(header)
    ax.add_patch(Rectangle((x, y + height - header_h), width, header_h * 0.55, facecolor=PANEL_HEADER, edgecolor=PANEL_HEADER))

    ax.add_patch(plt.Circle((x + 0.026, y + height - 0.038), 0.008, color="#ff5f57"))
    ax.add_patch(plt.Circle((x + 0.048, y + height - 0.038), 0.008, color="#ffbd2e"))
    ax.add_patch(plt.Circle((x + 0.070, y + height - 0.038), 0.008, color="#28c840"))
    ax.text(x + 0.095, y + height - 0.040, title, color=WHITE, fontsize=9.5, fontweight="bold", va="center")

    visible_lines = lines if max_lines is None else lines[:max_lines]
    if max_lines is not None and len(lines) > max_lines:
        visible_lines = visible_lines + ["..."]

    top = y + height - header_h - 0.035
    line_h = (height - header_h - 0.06) / max(len(visible_lines), 1)
    line_h = min(line_h, 0.038)

    for idx, line in enumerate(visible_lines, start=1):
        yy = top - (idx - 1) * line_h
        if yy < y + 0.025:
            break

        line_no = f"{idx:>2}"
        ax.text(x + 0.022, yy, line_no, color="#7d8590", fontsize=font_size, family="DejaVu Sans Mono", va="center")

        color = WHITE
        stripped = line.strip()
        if stripped.startswith(("FROM", "WORKDIR", "ENV", "COPY", "RUN", "EXPOSE", "CMD", "services:", "api:", "dashboard:")):
            color = BLUE
        elif stripped.startswith(("-", "command:", "depends_on:", "volumes:", "ports:", "build:", "container_name:")):
            color = ORANGE
        elif stripped.startswith("#"):
            color = GRAY
        elif stripped.startswith(("__pycache__", "*.pyc", ".venv", "venv", ".git", "data", "reports", "logs")):
            color = GREEN

        ax.text(
            x + 0.060,
            yy,
            line,
            color=color,
            fontsize=font_size,
            family="DejaVu Sans Mono",
            va="center",
        )

    ax.add_patch(Rectangle((x, y), 0.010, height, facecolor=accent, edgecolor=accent))


def main() -> None:
    dockerfile = read_file("Dockerfile")
    compose = read_file("docker-compose.yml")
    dockerignore = read_file(".dockerignore")

    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.965, "Docker deployment scripts used in the project", fontsize=17, color=DARK, fontweight="bold", va="top")
    ax.text(
        0.035,
        0.925,
        "Real configuration files used to build and run the FastAPI and Streamlit services.",
        fontsize=9.5,
        color=GRAY,
        va="top",
    )

    draw_code_panel(
        ax,
        (0.045, 0.46),
        0.40,
        0.39,
        "Dockerfile",
        dockerfile,
        BLUE,
        font_size=8.5,
    )
    draw_code_panel(
        ax,
        (0.475, 0.20),
        0.48,
        0.65,
        "docker-compose.yml",
        compose,
        ORANGE,
        font_size=7.8,
    )
    draw_code_panel(
        ax,
        (0.045, 0.18),
        0.40,
        0.24,
        ".dockerignore",
        dockerignore,
        GREEN,
        font_size=6.9,
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.045, 0.055),
            0.91,
            0.075,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=LIGHT,
            edgecolor="#d8dee4",
            linewidth=1.0,
        )
    )
    ax.text(
        0.065,
        0.092,
        "Execution command: docker compose up --build",
        fontsize=10,
        color=DARK,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="center",
    )
    ax.text(
        0.43,
        0.092,
        "API: http://localhost:8000/docs    Dashboard: http://localhost:8501",
        fontsize=9.2,
        color=GRAY,
        family="DejaVu Sans Mono",
        va="center",
    )

    savefig("46_docker_real_scripts.png")
    print("Generated Docker scripts image:")
    print(" - img/46_docker_real_scripts.png")


if __name__ == "__main__":
    main()
