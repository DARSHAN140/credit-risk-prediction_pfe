from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE = BASE_DIR / "reports" / "rapport_analyse_projet.md"
OUTPUT = BASE_DIR / "reports" / "rapport_analyse_projet.pdf"

PAGE_SIZE = (8.27, 11.69)
LEFT = 0.08
RIGHT = 0.92
TOP = 0.94
BOTTOM = 0.07
LINE = 0.024


def new_page(pdf: PdfPages):
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax, TOP


def finish_page(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def clean_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")
    return text


def add_text_block(pdf: PdfPages, fig, ax, y: float, text: str, size: int = 10, weight: str = "normal"):
    width = 100 if size <= 10 else 78
    lines = []
    for raw_line in text.splitlines() or [""]:
        if raw_line.strip() == "":
            lines.append("")
            continue
        lines.extend(textwrap.wrap(clean_inline(raw_line), width=width, break_long_words=False))

    for line in lines:
        if y < BOTTOM:
            finish_page(pdf, fig)
            fig, ax, y = new_page(pdf)
        ax.text(LEFT, y, line, fontsize=size, fontweight=weight, va="top", ha="left", family="DejaVu Sans")
        y -= LINE * (size / 10)
    return fig, ax, y


def add_image(pdf: PdfPages, fig, ax, y: float, image_path: Path, caption: str):
    if not image_path.exists():
        return add_text_block(pdf, fig, ax, y, f"[Figure introuvable: {image_path.name}]", size=9)

    if y < 0.42:
        finish_page(pdf, fig)
        fig, ax, y = new_page(pdf)

    fig_img = mpimg.imread(image_path)
    image_height = 0.34
    image_width = RIGHT - LEFT
    ax.imshow(fig_img, extent=[LEFT, LEFT + image_width, y - image_height, y], aspect="auto")
    ax.text(LEFT, y - image_height - 0.018, caption, fontsize=9, color="#444444", va="top", ha="left")
    y -= image_height + 0.055
    return fig, ax, y


def convert() -> None:
    content = SOURCE.read_text(encoding="utf-8").splitlines()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT) as pdf:
        fig, ax, y = new_page(pdf)

        for line in content:
            image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if image_match:
                caption = image_match.group(1)
                path = SOURCE.parent / image_match.group(2)
                fig, ax, y = add_image(pdf, fig, ax, y, path, caption)
                continue

            if line.startswith("# "):
                if y < 0.82:
                    finish_page(pdf, fig)
                    fig, ax, y = new_page(pdf)
                title = clean_inline(line[2:].strip())
                ax.text(LEFT, y, title, fontsize=20, fontweight="bold", va="top", ha="left", color="#102033")
                y -= 0.055
                continue

            if line.startswith("## "):
                if y < 0.18:
                    finish_page(pdf, fig)
                    fig, ax, y = new_page(pdf)
                fig, ax, y = add_text_block(pdf, fig, ax, y - 0.01, line[3:].strip(), size=14, weight="bold")
                y -= 0.01
                continue

            if line.startswith("### "):
                fig, ax, y = add_text_block(pdf, fig, ax, y, line[4:].strip(), size=12, weight="bold")
                continue

            if line.startswith("|"):
                fig, ax, y = add_text_block(pdf, fig, ax, y, line, size=8)
                continue

            if line.strip().startswith("- "):
                fig, ax, y = add_text_block(pdf, fig, ax, y, "• " + line.strip()[2:], size=10)
                continue

            fig, ax, y = add_text_block(pdf, fig, ax, y, line, size=10)

        finish_page(pdf, fig)


if __name__ == "__main__":
    convert()
    print(OUTPUT)
