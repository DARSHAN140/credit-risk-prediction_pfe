from __future__ import annotations

import csv
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#66727f"
PANEL = "#111827"
PANEL_HEADER = "#263241"
LIGHT = "#f4f6f8"
WHITE = "#f8fafc"
BLUE = "#88c0d0"
GREEN = "#8fd694"
ORANGE = "#f0b35f"
PURPLE = "#c4a7e7"
RED = "#f28b82"


def read_lines(path: str) -> list[str]:
    return (BASE_DIR / path).read_text(encoding="utf-8", errors="replace").splitlines()


def savefig(filename: str) -> None:
    IMG_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def extract_block(lines: list[str], start_marker: str, end_marker: str | None = None, max_lines: int = 18) -> list[tuple[int, str]]:
    start = 0
    for i, line in enumerate(lines):
        if start_marker in line:
            start = i
            break
    end = min(len(lines), start + max_lines)
    if end_marker is not None:
        for j in range(start + 1, len(lines)):
            if end_marker in lines[j]:
                end = j
                break
    return [(i + 1, lines[i]) for i in range(start, min(end, start + max_lines))]


def draw_window(ax, xy, width, height, title, rows, accent, font_size=7.2, title_size=9.0):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#334155",
            facecolor=PANEL,
        )
    )
    header_h = 0.065
    ax.add_patch(
        FancyBboxPatch(
            (x, y + height - header_h),
            width,
            header_h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=0,
            facecolor=PANEL_HEADER,
        )
    )
    ax.add_patch(Rectangle((x, y + height - header_h), width, header_h * 0.45, color=PANEL_HEADER))
    ax.add_patch(Rectangle((x, y), 0.008, height, color=accent))
    for n, color in enumerate(["#ff5f57", "#ffbd2e", "#28c840"]):
        ax.add_patch(plt.Circle((x + 0.025 + n * 0.021, y + height - 0.034), 0.007, color=color))
    ax.text(x + 0.090, y + height - 0.036, title, color=WHITE, fontsize=title_size, fontweight="bold", va="center")

    top = y + height - header_h - 0.027
    line_h = min(0.030, (height - header_h - 0.045) / max(len(rows), 1))
    for idx, row in enumerate(rows):
        yy = top - idx * line_h
        if yy < y + 0.018:
            break

        if isinstance(row, tuple):
            number, text = row
            ax.text(x + 0.022, yy, f"{number:>3}", color="#7d8590", fontsize=font_size, family="DejaVu Sans Mono", va="center")
            text_x = x + 0.060
        else:
            text = row
            text_x = x + 0.026

        stripped = text.strip()
        color = WHITE
        if stripped.startswith(("class ", "def ", "@app", "MODEL_", "PREDICTION_LOG")):
            color = BLUE
        elif stripped.startswith(("if ", "for ", "return", "with ", "writer.", "raise ")):
            color = ORANGE
        elif "metadata" in stripped or "threshold" in stripped or "model" in stripped:
            color = PURPLE
        elif "timestamp" in stripped or "pd_default" in stripped or "score_credit" in stripped:
            color = GREEN
        elif "HTTPException" in stripped or "RuntimeError" in stripped:
            color = RED

        max_chars = max(38, int(width * 155))
        display_text = text if len(text) <= max_chars else text[: max_chars - 3] + "..."
        ax.text(text_x, yy, display_text, color=color, fontsize=font_size, family="DejaVu Sans Mono", va="center")


def prediction_log_preview(max_rows: int = 4) -> list[str]:
    path = BASE_DIR / "logs" / "predictions.csv"
    if not path.exists():
        return ["logs/predictions.csv not available yet"]
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        rows = []
        for i, row in enumerate(reader):
            if i > max_rows:
                break
            rows.append(",".join(row))
    return rows or ["empty prediction log"]


def csv_preview(path: str, max_rows: int = 6) -> list[str]:
    target = BASE_DIR / path
    if not target.exists():
        return [f"{path} not found"]
    with target.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        rows = []
        for i, row in enumerate(reader):
            if i > max_rows:
                break
            rows.append(",".join(row[:8]))
    return rows


def metadata_summary() -> list[str]:
    metadata = json.loads((BASE_DIR / "models" / "metadata.json").read_text(encoding="utf-8"))
    metrics = metadata.get("metrics", {})
    rows = [
        "{",
        f'  "best_model": "{metadata.get("best_model")}",',
        f'  "threshold": {metadata.get("threshold")},',
        f'  "features_count": {len(metadata.get("features", []))},',
        f'  "numeric_features": {len(metadata.get("numeric_features", []))},',
        f'  "categorical_features": {len(metadata.get("categorical_features", []))},',
        '  "metrics": {',
    ]
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "gini", "ks"]:
        rows.append(f'    "{key}": {metrics.get(key)},')
    rows.extend(["  }", "}"])
    return rows


def audit_summary() -> list[str]:
    path = BASE_DIR / "reports" / "audit_metrics" / "model_audit_summary.json"
    if not path.exists():
        return ["reports/audit_metrics/model_audit_summary.json not found"]
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = json.loads((BASE_DIR / "models" / "metadata.json").read_text(encoding="utf-8"))
    metrics = metadata.get("metrics", {})
    rows = [
        "{",
        f'  "n_test": {data.get("n_test")},',
        '  "reporting_metrics": {',
        f'    "accuracy": {metrics.get("accuracy")},',
        f'    "precision": {metrics.get("precision")},',
        f'    "recall": {metrics.get("recall")},',
        f'    "f1": {metrics.get("f1")},',
        f'    "roc_auc": {metrics.get("roc_auc")},',
        f'    "gini": {metrics.get("gini")},',
        f'    "ks": {metrics.get("ks")}',
        "  }",
        '  "dataset": "synthetic",',
        '  "scope": "methodological validation"',
        "}",
    ]
    return rows


def threshold_reporting_summary() -> list[str]:
    return [
        "threshold,effect,interpretation",
        "0.30,higher recall,more risky files detected",
        "0.50,selected compromise,official reporting view",
        "0.70,higher precision,more selective alerts",
        "",
        "Numeric production thresholds require",
        "validation on real banking data.",
    ]


def draw_note(ax, xy, width, height, title, body, color):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=LIGHT,
            edgecolor="#d8dee4",
            linewidth=1.0,
        )
    )
    ax.text(x + 0.018, y + height - 0.028, title, fontsize=9.2, color=color, fontweight="bold", va="top")
    wrapped = textwrap.wrap(body, width=82)
    for i, line in enumerate(wrapped[:4]):
        ax.text(x + 0.018, y + height - 0.060 - i * 0.027, line, fontsize=7.8, color=DARK, va="top")


def plot_real_security_snapshot() -> None:
    api_lines = read_lines("api/main.py")

    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.035, 0.965, "Implemented API security and traceability controls", fontsize=17, color=DARK, fontweight="bold", va="top")
    ax.text(0.035, 0.925, "Real project snapshot: input schema, artifact checks, feature alignment and prediction logging.", fontsize=9.5, color=GRAY, va="top")

    draw_window(
        ax,
        (0.045, 0.51),
        0.43,
        0.34,
        "api/main.py - request schema and artifact checks",
        extract_block(api_lines, "MODEL_PATH", "model = None", max_lines=18),
        BLUE,
        font_size=6.8,
    )
    draw_window(
        ax,
        (0.505, 0.51),
        0.45,
        0.34,
        "api/main.py - payload normalization",
        extract_block(api_lines, "def prepare_payload", "def score_to_segment", max_lines=16),
        ORANGE,
        font_size=6.8,
    )
    draw_window(
        ax,
        (0.045, 0.13),
        0.43,
        0.31,
        "api/main.py - prediction audit log",
        extract_block(api_lines, "def write_prediction_log", "@app.get", max_lines=18),
        GREEN,
        font_size=6.6,
    )
    draw_window(
        ax,
        (0.505, 0.28),
        0.45,
        0.18,
        "logs/predictions.csv - traceability output",
        prediction_log_preview(),
        GREEN,
        font_size=6.6,
    )
    draw_note(
        ax,
        (0.505, 0.13),
        0.45,
        0.11,
        "Scope note",
        "The current prototype implements schema validation, artifact checks, feature alignment and logging. Authentication and RBAC are identified as future production controls.",
        RED,
    )
    savefig("48_real_security_code_snapshot.png")


def plot_real_governance_snapshot() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.035, 0.965, "Model governance artifacts used in the project", fontsize=17, color=DARK, fontweight="bold", va="top")
    ax.text(0.035, 0.925, "Project snapshot: model metadata, harmonized metrics and methodological threshold analysis.", fontsize=9.5, color=GRAY, va="top")

    draw_window(
        ax,
        (0.045, 0.48),
        0.43,
        0.36,
        "models/metadata.json - model registry metadata",
        metadata_summary(),
        PURPLE,
        font_size=7.0,
    )
    draw_window(
        ax,
        (0.505, 0.48),
        0.45,
        0.36,
        "reports/model_comparison.csv - validated model comparison",
        csv_preview("reports/model_comparison.csv", max_rows=6),
        ORANGE,
        font_size=5.8,
    )
    draw_window(
        ax,
        (0.045, 0.12),
        0.43,
        0.29,
        "reports/audit_metrics/model_audit_summary.json - reporting view",
        audit_summary(),
        BLUE,
        font_size=7.0,
    )
    draw_window(
        ax,
        (0.505, 0.12),
        0.45,
        0.29,
        "threshold analysis - methodological reading",
        threshold_reporting_summary(),
        GREEN,
        font_size=6.0,
    )
    savefig("49_real_governance_artifacts_snapshot.png")


def main() -> None:
    plot_real_security_snapshot()
    plot_real_governance_snapshot()
    print("Generated real project snapshots:")
    print(" - img/48_real_security_code_snapshot.png")
    print(" - img/49_real_governance_artifacts_snapshot.png")


if __name__ == "__main__":
    main()
