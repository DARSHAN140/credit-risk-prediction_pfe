from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MAX_TRAIN_ROWS, MODEL_DIR, RANDOM_STATE, RAW_DATA_PATH
from src.data_processing import prepare_model_frame, read_credit_data, temporal_or_stratified_split


IMG_DIR = BASE_DIR / "img"
REPORT_DIR = BASE_DIR / "reports" / "audit_metrics"

DARK = "#1f2a35"
GRAY = "#657280"
BLUE = "#2f5597"
GREEN = "#4f7d5a"
RED = "#b04a4a"
ORANGE = "#c77c2b"


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close()


def load_predictions():
    raw_df = read_credit_data(RAW_DATA_PATH)
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = (
            raw_df.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw_df), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    X, y = prepare_model_frame(raw_df)
    _, X_test, _, y_test = temporal_or_stratified_split(raw_df, X, y)
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    y_score = model.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_score


def plot_ks_curve(y_true, y_score) -> dict:
    data = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    data = data.sort_values("score", ascending=False).reset_index(drop=True)
    positives = max((data["y"] == 1).sum(), 1)
    negatives = max((data["y"] == 0).sum(), 1)
    data["cum_risky"] = (data["y"] == 1).cumsum() / positives
    data["cum_favorable"] = (data["y"] == 0).cumsum() / negatives
    data["population_share"] = (np.arange(len(data)) + 1) / len(data)
    data["ks_gap"] = (data["cum_risky"] - data["cum_favorable"]).abs()
    idx = int(data["ks_gap"].idxmax())

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(data["population_share"], data["cum_risky"], color=RED, lw=2, label="Cumulative risky class")
    ax.plot(data["population_share"], data["cum_favorable"], color=GREEN, lw=2, label="Cumulative favorable class")
    ax.vlines(
        data.loc[idx, "population_share"],
        data.loc[idx, "cum_favorable"],
        data.loc[idx, "cum_risky"],
        colors=BLUE,
        linestyles="--",
        lw=2,
        label=f"KS = {data.loc[idx, 'ks_gap']:.3f}",
    )
    ax.set_title("KS curve of the selected model", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Population sorted by predicted risk")
    ax.set_ylabel("Cumulative distribution")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    savefig("35_ks_curve_xgboost.png")
    return {
        "ks": float(data.loc[idx, "ks_gap"]),
        "score_threshold": float(data.loc[idx, "score"]),
        "population_share": float(data.loc[idx, "population_share"]),
    }


def plot_calibration_curve(y_true, y_score) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", color=BLUE, lw=2, label=f"Model calibration (Brier = {brier:.3f})")
    ax.set_title("Calibration curve of the selected model", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    savefig("36_calibration_curve_xgboost.png")

    calib = pd.DataFrame({"mean_predicted_pd": prob_pred, "observed_default_rate": prob_true})
    return {"brier_score": float(brier), "bins": calib.to_dict(orient="records")}


def plot_threshold_tradeoff(y_true, y_score) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.2, 0.81, 0.05):
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(df["threshold"], df["precision"], marker="o", color=BLUE, label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="o", color=RED, label="Recall")
    ax.plot(df["threshold"], df["f1"], marker="o", color=GREEN, label="F1-score")
    ax.set_title("Threshold analysis for the risky class", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.65, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    savefig("37_threshold_tradeoff_xgboost.png")
    return df


def plot_fairness_type_credit(X_test, y_score) -> pd.DataFrame:
    if "type_credit" not in X_test.columns:
        return pd.DataFrame()
    y_pred = (y_score >= 0.5).astype(int)
    work = X_test.copy()
    work["predicted_risky"] = y_pred
    work["pd_default"] = y_score
    grouped = (
        work.groupby("type_credit")
        .agg(n=("predicted_risky", "size"), predicted_risky_rate=("predicted_risky", "mean"), avg_pd=("pd_default", "mean"))
        .reset_index()
        .sort_values("predicted_risky_rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=grouped, x="predicted_risky_rate", y="type_credit", color=BLUE, ax=ax)
    ax.set_title("Predicted risky rate by credit type", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Predicted risky rate")
    ax.set_ylabel("Credit type")
    ax.set_xlim(0, min(1, grouped["predicted_risky_rate"].max() + 0.15))
    ax.grid(axis="x", alpha=0.25)
    for idx, row in grouped.reset_index(drop=True).iterrows():
        ax.text(row["predicted_risky_rate"] + 0.01, idx, f"{row['predicted_risky_rate']:.1%} | n={int(row['n'])}", va="center", fontsize=9)
    savefig("38_fairness_credit_type_xgboost.png")
    return grouped


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    X_test, y_test, y_score = load_predictions()
    results = {
        "ks": plot_ks_curve(y_test, y_score),
        "calibration": plot_calibration_curve(y_test, y_score),
    }
    threshold_df = plot_threshold_tradeoff(y_test, y_score)
    fairness_df = plot_fairness_type_credit(X_test, y_score)
    threshold_df.to_csv(REPORT_DIR / "threshold_tradeoff_xgboost.csv", index=False)
    if not fairness_df.empty:
        fairness_df.to_csv(REPORT_DIR / "fairness_type_credit_xgboost.csv", index=False)
    (REPORT_DIR / "additional_audit_figures.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Model audit images generated in: {IMG_DIR}")


if __name__ == "__main__":
    main()
