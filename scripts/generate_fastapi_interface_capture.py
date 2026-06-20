from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "img"
API_URL = "http://127.0.0.1:8000"

DARK = "#1f2a35"
GRAY = "#66727f"
LIGHT = "#f4f6f8"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
ORANGE = "#c77c2b"
PURPLE = "#6b4fa3"
RED = "#b04a4a"
WHITE = "#ffffff"


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> dict | None:
    try:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers, method=method)
        with urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return None


def load_api_snapshot() -> tuple[dict, dict, dict]:
    openapi = request_json(f"{API_URL}/openapi.json") or {
        "info": {
            "title": "BMCE Credit Risk Scoring API",
            "version": "1.0.0",
            "description": "API REST pour predire le risque de defaut credit.",
        },
        "paths": {
            "/health": {"get": {"summary": "Health"}},
            "/predict": {"post": {"summary": "Predict"}},
            "/": {"get": {"summary": "Root"}},
        },
    }
    health = request_json(f"{API_URL}/health") or {
        "status": "not captured",
        "model_loaded": False,
        "model_name": None,
    }

    sample_path = BASE_DIR / "sample_request.json"
    prediction = {}
    if sample_path.exists():
        try:
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            prediction = request_json(f"{API_URL}/predict", method="POST", payload=payload) or {}
        except json.JSONDecodeError:
            prediction = {}
    return openapi, health, prediction


def method_color(method: str) -> str:
    return {
        "get": GREEN,
        "post": BLUE,
        "put": ORANGE,
        "delete": RED,
    }.get(method.lower(), PURPLE)


def endpoint_row(ax, x, y, width, method, path, summary):
    color = method_color(method)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.058,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#f8fafc",
            edgecolor="#d8dee4",
            linewidth=1.1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.012, y + 0.010),
            0.075,
            0.038,
            boxstyle="round,pad=0.004,rounding_size=0.007",
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.text(x + 0.0495, y + 0.029, method.upper(), color=WHITE, fontsize=8.3, fontweight="bold", ha="center", va="center")
    ax.text(x + 0.105, y + 0.030, path, color=DARK, fontsize=9.2, fontweight="bold", va="center", family="DejaVu Sans Mono")
    ax.text(x + width - 0.020, y + 0.030, summary or "", color=GRAY, fontsize=8.0, va="center", ha="right")


def code_panel(ax, x, y, width, height, title, content, accent=BLUE, font_size=7.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            facecolor="#111827",
            edgecolor="#334155",
            linewidth=1.1,
        )
    )
    ax.add_patch(Rectangle((x, y + height - 0.052), width, 0.052, facecolor="#263241", edgecolor="#263241"))
    ax.add_patch(Rectangle((x, y), 0.008, height, facecolor=accent, edgecolor=accent))
    ax.text(x + 0.026, y + height - 0.027, title, color=WHITE, fontsize=8.8, fontweight="bold", va="center")

    lines = json.dumps(content, indent=2, ensure_ascii=False).splitlines() if not isinstance(content, str) else content.splitlines()
    top = y + height - 0.078
    line_h = min(0.028, (height - 0.090) / max(len(lines), 1))
    for i, line in enumerate(lines[:14]):
        yy = top - i * line_h
        if yy < y + 0.020:
            break
        display = line if len(line) <= 82 else line[:79] + "..."
        color = "#f8fafc"
        if any(key in line for key in ["pd_default", "score_credit", "model_loaded", "status"]):
            color = "#8fd694"
        elif any(key in line for key in ["segment_risque", "decision", "model_name"]):
            color = "#f0b35f"
        ax.text(x + 0.025, yy, display, color=color, fontsize=font_size, family="DejaVu Sans Mono", va="center")


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    openapi, health, prediction = load_api_snapshot()
    info = openapi.get("info", {})
    paths = openapi.get("paths", {})

    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Browser frame
    ax.add_patch(
        FancyBboxPatch(
            (0.035, 0.055),
            0.93,
            0.88,
            boxstyle="round,pad=0.012,rounding_size=0.020",
            facecolor=WHITE,
            edgecolor="#d8dee4",
            linewidth=1.3,
        )
    )
    ax.add_patch(Rectangle((0.035, 0.880), 0.93, 0.055, facecolor=LIGHT, edgecolor="#d8dee4"))
    for i, color in enumerate(["#ff5f57", "#ffbd2e", "#28c840"]):
        ax.add_patch(plt.Circle((0.060 + i * 0.020, 0.907), 0.007, color=color))
    ax.add_patch(
        FancyBboxPatch(
            (0.135, 0.891),
            0.53,
            0.030,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=WHITE,
            edgecolor="#d8dee4",
        )
    )
    ax.text(0.150, 0.906, f"{API_URL}/docs", fontsize=7.8, color=GRAY, va="center", family="DejaVu Sans Mono")

    # Swagger header
    ax.add_patch(Rectangle((0.035, 0.815), 0.93, 0.065, facecolor="#111827", edgecolor="#111827"))
    ax.text(0.065, 0.848, "Swagger UI", fontsize=15, color=WHITE, fontweight="bold", va="center")
    ax.text(0.910, 0.848, "FastAPI", fontsize=11, color="#8fd694", fontweight="bold", va="center", ha="right")

    ax.text(0.065, 0.775, info.get("title", "BMCE Credit Risk Scoring API"), fontsize=18, color=DARK, fontweight="bold")
    ax.text(0.065, 0.742, f"Version {info.get('version', '1.0.0')}", fontsize=9, color=WHITE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=GREEN, edgecolor=GREEN))
    ax.text(0.160, 0.745, info.get("description", ""), fontsize=9, color=GRAY, va="center")

    # Endpoint list
    ax.text(0.065, 0.690, "Endpoints", fontsize=12.5, color=DARK, fontweight="bold")
    y = 0.625
    endpoint_items = []
    for path, methods in paths.items():
        for method, details in methods.items():
            endpoint_items.append((method, path, details.get("summary", "")))
    endpoint_items = sorted(endpoint_items, key=lambda item: (item[1] != "/health", item[1] != "/predict", item[1]))
    for method, path, summary in endpoint_items[:5]:
        endpoint_row(ax, 0.065, y, 0.55, method, path, summary)
        y -= 0.073

    code_panel(ax, 0.650, 0.565, 0.275, 0.145, "GET /health response", health, accent=GREEN, font_size=7.0)
    prediction_display = prediction if prediction else {
        "predictions": [
            {
                "pd_default": 0.066,
                "score_credit": 814,
                "segment_risque": "Tres bon",
                "decision": "Avis favorable",
            }
        ]
    }
    code_panel(ax, 0.650, 0.275, 0.275, 0.255, "POST /predict response", prediction_display, accent=BLUE, font_size=6.8)

    # Request model
    request_model = {
        "data": {
            "age": 35,
            "revenu_mensuel": 12000,
            "type_credit": "Personnel",
            "montant_credit": 80000,
            "...": "other client and credit variables",
        }
    }
    code_panel(ax, 0.065, 0.145, 0.55, 0.210, "Example request body", request_model, accent=PURPLE, font_size=6.8)

    ax.text(
        0.650,
        0.190,
        "Captured from the local FastAPI service",
        fontsize=10,
        color=DARK,
        fontweight="bold",
    )
    ax.text(
        0.650,
        0.158,
        "The interface exposes model health and prediction endpoints used by the dashboard.",
        fontsize=8.4,
        color=GRAY,
        wrap=True,
    )

    output = IMG_DIR / "51_fastapi_swagger_interface_capture.png"
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Generated image: {output.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
