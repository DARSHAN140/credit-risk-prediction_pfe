from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "img"

DARK = "#1f2a35"
GRAY = "#66727f"
LIGHT = "#eef2f5"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
RED = "#b04a4a"
ORANGE = "#c77c2b"
PURPLE = "#6b4fa3"
TEAL = "#2f7f7f"


def savefig(filename: str) -> None:
    IMG_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def box(ax, xy, width, height, text, color, text_color="white", fontsize=9.5, lw=1.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
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
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
    )


def outline_box(ax, xy, width, height, text, edge, fill="#ffffff", fontsize=9.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.3,
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
        color=DARK,
        fontsize=fontsize,
        fontweight="bold",
    )


def arrow(ax, start, end, color=GRAY, rad=0.0, lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def title(ax, text, subtitle=None):
    ax.text(0.02, 0.96, text, fontsize=16, fontweight="bold", color=DARK, va="top")
    if subtitle:
        ax.text(0.02, 0.91, subtitle, fontsize=9.5, color=GRAY, va="top")


def plot_access_control() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        ax,
        "Secure access control for the credit risk application",
        "Authentication, role-based authorization and controlled API access.",
    )

    roles = [
        ("Analyst", "Submit credit file\nRead predictions", BLUE),
        ("Administrator", "Manage users\nDeploy model versions", PURPLE),
        ("Auditor", "Read audit logs\nReview justifications", TEAL),
    ]
    y_positions = [0.72, 0.52, 0.32]
    for (role, desc, color), y in zip(roles, y_positions):
        box(ax, (0.05, y), 0.16, 0.095, f"{role}\n{desc}", color, fontsize=8.2)
        arrow(ax, (0.21, y + 0.048), (0.31, 0.56), color=color, rad=0.12)

    box(ax, (0.32, 0.49), 0.16, 0.14, "OAuth2 / JWT\nLogin + token\nSession expiry", ORANGE, fontsize=9)
    box(ax, (0.54, 0.49), 0.16, 0.14, "API Gateway\nHTTPS/TLS\nRBAC rules", GREEN, fontsize=9)
    arrow(ax, (0.48, 0.56), (0.54, 0.56), color=GRAY)

    endpoints = [
        (0.78, 0.68, "/predict\ncredit scoring", BLUE),
        (0.78, 0.50, "/audit\nlogs and reports", TEAL),
        (0.78, 0.32, "/admin\nmodel governance", PURPLE),
    ]
    for x, y, text, color in endpoints:
        outline_box(ax, (x, y), 0.15, 0.095, text, color, fill="#f7f9fb", fontsize=8.7)
        arrow(ax, (0.70, 0.56), (x, y + 0.048), color=color, rad=0.08)

    controls = [
        ("Token expiration", "30 min inactivity"),
        ("Least privilege", "role-limited actions"),
        ("Input validation", "Pydantic schema"),
        ("Traceability", "user + timestamp"),
    ]
    x0 = 0.08
    for i, (head, sub) in enumerate(controls):
        x = x0 + i * 0.22
        outline_box(ax, (x, 0.12), 0.17, 0.075, f"{head}\n{sub}", GRAY, fill=LIGHT, fontsize=8)

    savefig("39_secure_access_control_flow.png")


def plot_validation_audit_trail() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        ax,
        "Input validation and audit trail",
        "Each request is checked before prediction and logged after decision support output.",
    )

    box(ax, (0.05, 0.70), 0.16, 0.11, "Credit request\nJSON payload", BLUE)
    box(ax, (0.30, 0.70), 0.19, 0.11, "Pydantic schema\nTypes + required fields", ORANGE, fontsize=9)
    box(ax, (0.58, 0.70), 0.18, 0.11, "Business controls\nAmounts + ratios", GREEN, fontsize=9)
    box(ax, (0.82, 0.70), 0.13, 0.11, "Prediction\naccepted input", PURPLE, fontsize=9)
    arrow(ax, (0.21, 0.755), (0.30, 0.755))
    arrow(ax, (0.49, 0.755), (0.58, 0.755))
    arrow(ax, (0.76, 0.755), (0.82, 0.755))

    headers = ["Field", "Control", "Example rule"]
    rows = [
        ["income", "numeric", "1,500 <= value <= 300,000"],
        ["credit amount", "numeric", "value > 0 and coherent with income"],
        ["DTI ratio", "derived", "0 <= ratio <= 1.2"],
        ["credit type", "categorical", "Personal / Auto / Mortgage"],
        ["employment status", "categorical", "known banking segment"],
    ]
    x_cols = [0.08, 0.31, 0.52]
    col_w = [0.20, 0.18, 0.36]
    ax.text(0.08, 0.57, "Validation controls", fontsize=12, fontweight="bold", color=DARK)
    for i, head in enumerate(headers):
        ax.add_patch(Rectangle((x_cols[i], 0.51), col_w[i], 0.045, facecolor=DARK, edgecolor=DARK))
        ax.text(x_cols[i] + 0.01, 0.533, head, color="white", fontsize=8.5, fontweight="bold", va="center")
    for r, row in enumerate(rows):
        y = 0.465 - r * 0.055
        fill = "#ffffff" if r % 2 == 0 else "#f5f7f9"
        for i, val in enumerate(row):
            ax.add_patch(Rectangle((x_cols[i], y), col_w[i], 0.05, facecolor=fill, edgecolor="#d8dee4"))
            ax.text(x_cols[i] + 0.01, y + 0.025, val, color=DARK, fontsize=8, va="center")

    ax.text(0.08, 0.17, "Audit record", fontsize=12, fontweight="bold", color=DARK)
    audit_items = [
        "request_id",
        "user_role",
        "model_version",
        "dataset_version",
        "pd_default",
        "credit_score",
        "decision_support",
        "timestamp",
        "top_SHAP_factors",
    ]
    for i, item in enumerate(audit_items):
        x = 0.08 + (i % 5) * 0.165
        y = 0.105 - (i // 5) * 0.055
        outline_box(ax, (x, y), 0.145, 0.038, item, GRAY, fill="#f7f9fb", fontsize=7.2)

    savefig("40_input_validation_audit_trail.png")


def plot_data_protection() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        ax,
        "Data protection lifecycle",
        "Security measures required for handling banking and credit-risk data.",
    )

    steps = [
        ("Collection", "limited fields\nbusiness purpose", BLUE),
        ("Anonymization", "remove direct identifiers\nhash request ids", TEAL),
        ("Encrypted storage", "database encryption\nsecure backups", GREEN),
        ("Access control", "RBAC + audit logs\nleast privilege", PURPLE),
        ("Retention", "defined retention period\narchiving rules", ORANGE),
        ("Deletion / review", "GDPR requests\nperiodic audit", RED),
    ]
    centers = [(0.16, 0.62), (0.40, 0.74), (0.64, 0.62), (0.64, 0.34), (0.40, 0.22), (0.16, 0.34)]
    for (head, sub, color), (cx, cy) in zip(steps, centers):
        box(ax, (cx - 0.095, cy - 0.055), 0.19, 0.11, f"{head}\n{sub}", color, fontsize=8.5)
    for i in range(len(centers)):
        start = centers[i]
        end = centers[(i + 1) % len(centers)]
        arrow(ax, start, end, color=GRAY, rad=0.08)

    ax.add_patch(FancyBboxPatch((0.77, 0.20), 0.17, 0.52, boxstyle="round,pad=0.02", facecolor=LIGHT, edgecolor="#d8dee4"))
    ax.text(0.795, 0.67, "Banking controls", fontsize=11.5, color=DARK, fontweight="bold")
    bullets = [
        "GDPR compliance",
        "Need-to-know access",
        "Encrypted backups",
        "Incident response",
        "Periodic access review",
        "Data minimization",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.795, 0.61 - i * 0.065, f"- {b}", fontsize=8.8, color=DARK)

    savefig("41_data_protection_lifecycle.png")


def plot_mlops_governance() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        ax,
        "Model governance and MLOps lifecycle",
        "A controlled lifecycle supports validation, deployment, monitoring and rollback.",
    )

    steps = [
        ("Data versioning", "dataset snapshot\nquality checks", BLUE),
        ("Training pipeline", "preprocessing\nSMOTE + models", ORANGE),
        ("Validation", "AUC, recall, KS\ncalibration", GREEN),
        ("Model registry", "versioned artifact\nmetadata", PURPLE),
        ("Approval", "risk review\nbusiness sign-off", TEAL),
        ("Deployment", "FastAPI service\nStreamlit dashboard", BLUE),
        ("Monitoring", "drift, stability\nperformance alerts", RED),
    ]
    x_positions = np.linspace(0.06, 0.86, len(steps))
    for i, ((head, sub, color), x) in enumerate(zip(steps, x_positions)):
        y = 0.61 if i % 2 == 0 else 0.43
        box(ax, (x, y), 0.105, 0.12, f"{head}\n{sub}", color, fontsize=7.6)
        if i < len(steps) - 1:
            next_y = 0.61 if (i + 1) % 2 == 0 else 0.43
            arrow(ax, (x + 0.105, y + 0.06), (x_positions[i + 1], next_y + 0.06), color=GRAY)

    arrow(ax, (0.915, 0.49), (0.43, 0.25), color=RED, rad=-0.22, lw=1.7)
    outline_box(ax, (0.31, 0.17), 0.25, 0.10, "Rollback or retraining\ntriggered by drift or validation failure", RED, fill="#fff5f5", fontsize=8.8)

    metrics = [
        ("Population Stability Index", "feature drift"),
        ("Brier score", "probability calibration"),
        ("Recall risky class", "missed-risk control"),
        ("Approval status", "go/no-go decision"),
    ]
    for i, (metric, desc) in enumerate(metrics):
        x = 0.10 + i * 0.22
        outline_box(ax, (x, 0.08), 0.18, 0.06, f"{metric}\n{desc}", GRAY, fill=LIGHT, fontsize=7.6)

    savefig("42_mlops_model_governance_lifecycle.png")


def plot_decision_support_thresholds() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        ax,
        "Decision-support thresholds for credit risk",
        "The model supports the analyst; final approval remains under human and policy control.",
    )

    x0, y0, w, h = 0.10, 0.60, 0.80, 0.08
    segments = [
        (0.00, 0.35, GREEN, "Low risk\npropose accept"),
        (0.35, 0.65, ORANGE, "Medium risk\nmanual review"),
        (0.65, 1.00, RED, "High risk\nescalate / reject"),
    ]
    for start, end, color, label in segments:
        ax.add_patch(Rectangle((x0 + start * w, y0), (end - start) * w, h, facecolor=color, edgecolor="white"))
        ax.text(x0 + (start + end) * w / 2, y0 + h / 2, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    for val in [0.00, 0.35, 0.50, 0.65, 1.00]:
        ax.plot([x0 + val * w, x0 + val * w], [y0 - 0.018, y0 + h + 0.018], color=DARK, lw=1)
        ax.text(x0 + val * w, y0 - 0.052, f"{val:.2f}", ha="center", va="top", fontsize=8.5, color=DARK)

    np.random.seed(7)
    low = np.random.beta(2, 6, 220)
    high = np.random.beta(6, 2, 170)
    scores = np.concatenate([low, high])
    bins = np.linspace(0, 1, 26)
    counts, edges = np.histogram(scores, bins=bins)
    counts = counts / counts.max() * 0.18
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        ax.add_patch(Rectangle((x0 + left * w, 0.31), (right - left) * w * 0.88, count, facecolor="#b8c3cf", edgecolor="#8d99a6", lw=0.4))
    ax.text(0.10, 0.52, "Predicted probability of default (PD)", fontsize=10, fontweight="bold", color=DARK)
    ax.text(0.10, 0.25, "Example score distribution used to choose business thresholds", fontsize=9, color=GRAY)

    flow = [
        ("Model score", BLUE),
        ("Risk policy", PURPLE),
        ("Analyst review", ORANGE),
        ("Final decision", GREEN),
    ]
    for i, (label, color) in enumerate(flow):
        x = 0.12 + i * 0.22
        box(ax, (x, 0.10), 0.15, 0.07, label, color, fontsize=8.7)
        if i < len(flow) - 1:
            arrow(ax, (x + 0.15, 0.135), (x + 0.22, 0.135), color=GRAY)

    savefig("43_decision_support_thresholds.png")


def plot_local_shap_report() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        ax,
        "Individual prediction explanation report",
        "A local SHAP-style view gives the analyst a concise justification for each credit file.",
    )

    ax.add_patch(FancyBboxPatch((0.06, 0.12), 0.88, 0.73, boxstyle="round,pad=0.02", facecolor="#ffffff", edgecolor="#d8dee4", linewidth=1.4))
    ax.text(0.09, 0.79, "Credit file: REQ-2026-01482", fontsize=12, color=DARK, fontweight="bold")
    ax.text(0.09, 0.745, "Model version: xgboost_v1.4  |  Dataset version: bmce_credit_2026_05", fontsize=8.8, color=GRAY)

    cards = [
        ("PD default", "0.62", RED),
        ("Credit score", "438 / 850", ORANGE),
        ("Segment", "High risk", RED),
        ("Action", "Manual review", PURPLE),
    ]
    for i, (head, val, color) in enumerate(cards):
        x = 0.09 + i * 0.21
        outline_box(ax, (x, 0.64), 0.17, 0.075, f"{head}\n{val}", color, fill="#f7f9fb", fontsize=8.8)

    features = [
        ("High DTI ratio", 0.23),
        ("Payment incidents", 0.18),
        ("Low remaining income", 0.14),
        ("Mortgage credit type", 0.09),
        ("Stable employment", -0.12),
        ("Existing savings", -0.08),
    ]
    ax.text(0.10, 0.56, "Main local explanation factors", fontsize=11, color=DARK, fontweight="bold")
    baseline_x = 0.49
    ax.plot([baseline_x, baseline_x], [0.25, 0.54], color=DARK, lw=1)
    for i, (name, value) in enumerate(features):
        y = 0.51 - i * 0.045
        color = RED if value > 0 else GREEN
        width = abs(value) * 0.55
        if value > 0:
            ax.add_patch(Rectangle((baseline_x, y - 0.014), width, 0.026, facecolor=color, edgecolor=color, alpha=0.9))
            ax.text(baseline_x + width + 0.012, y, f"+{value:.2f}", va="center", fontsize=8.2, color=color, fontweight="bold")
        else:
            ax.add_patch(Rectangle((baseline_x - width, y - 0.014), width, 0.026, facecolor=color, edgecolor=color, alpha=0.9))
            ax.text(baseline_x - width - 0.012, y, f"{value:.2f}", va="center", ha="right", fontsize=8.2, color=color, fontweight="bold")
        ax.text(0.12, y, name, va="center", fontsize=8.7, color=DARK)

    ax.add_patch(FancyBboxPatch((0.66, 0.27), 0.23, 0.25, boxstyle="round,pad=0.018", facecolor=LIGHT, edgecolor="#d8dee4"))
    ax.text(0.685, 0.485, "Generated justification", fontsize=10.5, color=DARK, fontweight="bold")
    notes = [
        "Risk is mainly driven by",
        "repayment capacity indicators.",
        "The file should be reviewed",
        "before any final decision.",
    ]
    for i, line in enumerate(notes):
        ax.text(0.685, 0.445 - i * 0.04, line, fontsize=8.7, color=DARK)

    savefig("44_local_shap_individual_report.png")


def main() -> None:
    plot_access_control()
    plot_validation_audit_trail()
    plot_data_protection()
    plot_mlops_governance()
    plot_decision_support_thresholds()
    plot_local_shap_report()
    print("Generated governance images in img/:")
    for name in [
        "39_secure_access_control_flow.png",
        "40_input_validation_audit_trail.png",
        "41_data_protection_lifecycle.png",
        "42_mlops_model_governance_lifecycle.png",
        "43_decision_support_thresholds.png",
        "44_local_shap_individual_report.png",
    ]:
        print(f" - {name}")


if __name__ == "__main__":
    main()
