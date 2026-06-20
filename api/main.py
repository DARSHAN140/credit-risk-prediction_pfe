import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.auth import Role, admin_router, auth_router, init_auth_db, require_roles
from src.config import BASE_DIR, LOG_DIR, MODEL_DIR
from src.data_processing import add_credit_features, basic_cleaning


MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
PREDICTION_LOG = LOG_DIR / "predictions.csv"
STATIC_DIR = BASE_DIR / "api" / "static"

app = FastAPI(
    title="BMCE Credit Risk Scoring API",
    description="API REST pour prédire le risque de défaut crédit.",
    version="1.0.0",
)
app.include_router(auth_router)
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    data: dict[str, Any] | list[dict[str, Any]]


def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Model artifacts not found. Run `python train.py` first.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


model = None
metadata = None


@app.on_event("startup")
def startup_event():
    global model, metadata
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    init_auth_db()
    model, metadata = load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": metadata.get("best_model") if metadata else None,
    }


def prepare_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    rows = payload if isinstance(payload, list) else [payload]
    df = pd.DataFrame(rows)
    df = basic_cleaning(df)
    df = add_credit_features(df)

    expected_features = metadata["features"]
    for col in expected_features:
        if col not in df.columns:
            df[col] = float("nan")
    return df[expected_features]


def score_to_segment(pd_default: float) -> str:
    if pd_default < 0.10:
        return "Tres bon"
    if pd_default < 0.25:
        return "Bon"
    if pd_default < 0.45:
        return "Moyen"
    if pd_default < 0.65:
        return "Risque"
    return "Tres risque"


def write_prediction_log(predictions: list[dict[str, Any]]) -> None:
    file_exists = PREDICTION_LOG.exists()
    with PREDICTION_LOG.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "pd_default", "score_credit", "segment_risque", "decision"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(predictions)


@app.post("/predict")
def predict(
    request: PredictionRequest,
    _current_user=Depends(require_roles(Role.admin, Role.analyste, Role.conseiller)),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    X = prepare_payload(request.data)
    probabilities = model.predict_proba(X)[:, 1]

    response = []
    logs = []
    for pd_default in probabilities:
        score = int(round(850 - 550 * float(pd_default)))
        segment = score_to_segment(float(pd_default))
        decision = "Avis defavorable" if pd_default >= metadata["threshold"] else "Avis favorable"
        item = {
            "pd_default": round(float(pd_default), 4),
            "score_credit": score,
            "segment_risque": segment,
            "decision": decision,
        }
        response.append(item)
        logs.append({"timestamp": datetime.utcnow().isoformat(), **item})

    write_prediction_log(logs)
    return {"predictions": response}


@app.get("/")
def root():
    return {
        "message": "BMCE Credit Risk Scoring API",
        "login": "/login",
        "docs": "/docs",
    }


@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse(STATIC_DIR / "login.html")
