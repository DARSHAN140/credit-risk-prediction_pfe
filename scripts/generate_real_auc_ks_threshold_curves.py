from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MPL_CACHE_DIR = BASE_DIR / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import MAX_TRAIN_ROWS, MODEL_DIR, RANDOM_STATE, RAW_DATA_PATH
from src.data_processing import prepare_model_frame, read_credit_data, temporal_or_stratified_split


IMG_DIR = BASE_DIR / "img"
FIGURE_DIR = BASE_DIR / "reports" / "figures"
REPORT_DIR = BASE_DIR / "reports"
CURVE_DATA_DIR = REPORT_DIR / "real_curves"
AUDIT_DATA_DIR = REPORT_DIR / "audit_metrics"

DARK = "#1f2a35"
GRAY = "#66727f"
GRID = "#d8dee4"
BLUE = "#2f5597"
ORANGE = "#c77c2b"
GREEN = "#4f7d5a"
RED = "#b04a4a"
PURPLE = "#6b4fa3"


def load_test_predictions() -> tuple[pd.Series, np.ndarray, float, str]:
    """Recreate the training test split and score it with the saved model."""
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
    calibrated_model_path = MODEL_DIR / "calibrated" / "xgboost.joblib"
    comparison_path = REPORT_DIR / "calibrated_model_comparison.csv"
    selected_threshold = 0.5
    model_source = "modele XGBoost gouverne non recalibre"
    if calibrated_model_path.exists() and comparison_path.exists():
        model = joblib.load(calibrated_model_path)
        comparison = pd.read_csv(comparison_path)
        xgboost_row = comparison.loc[comparison["model"] == "xgboost"]
        if xgboost_row.empty:
            raise ValueError("Le seuil du modele XGBoost recalibre est introuvable.")
        selected_threshold = float(xgboost_row.iloc[0]["threshold"])
        model_source = "modele XGBoost recalibre par sigmoid sur validation separee"
    else:
        model = joblib.load(MODEL_DIR / "best_model.joblib")
    metadata_path = MODEL_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected_features = metadata.get("features", X_test.columns.tolist())
        X_test = X_test[selected_features]
    y_score = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)

    if len(y_test) != len(y_score):
        raise ValueError("Le nombre de probabilites ne correspond pas au nombre de cibles de test.")
    if pd.Series(y_test).nunique() != 2:
        raise ValueError("La courbe ROC exige les deux classes dans l'echantillon de test.")
    if not np.isfinite(y_score).all() or ((y_score < 0) | (y_score > 1)).any():
        raise ValueError("Les probabilites predites doivent etre comprises entre 0 et 1.")
    return y_test.reset_index(drop=True), y_score, selected_threshold, model_source


def compute_roc_points(y_true: pd.Series, y_score: np.ndarray) -> tuple[pd.DataFrame, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    points = pd.DataFrame(
        {
            "threshold": thresholds,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
        }
    )
    points["youden_j"] = points["true_positive_rate"] - points["false_positive_rate"]
    return points, float(roc_auc_score(y_true, y_score))


def compute_precision_recall_points(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    points = pd.DataFrame(
        {
            "threshold": np.append(thresholds, np.nan),
            "recall": recall,
            "precision": precision,
        }
    )
    return points, float(auc(recall, precision))


def compute_ks_points(roc_points: pd.DataFrame, y_score: np.ndarray) -> tuple[pd.DataFrame, int]:
    points = roc_points.copy()
    points["cumulative_risky"] = points["true_positive_rate"]
    points["cumulative_favorable"] = points["false_positive_rate"]
    points["ks_gap"] = points["cumulative_risky"] - points["cumulative_favorable"]
    points["population_share"] = [
        0.0 if np.isinf(threshold) else float(np.mean(y_score >= threshold))
        for threshold in points["threshold"]
    ]
    ks_index = int(points["ks_gap"].idxmax())
    return points, ks_index


def compute_threshold_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

    rows: list[dict[str, float | int]] = []
    y_array = np.asarray(y_true, dtype=int)
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn = int(np.sum((y_array == 0) & (y_pred == 0)))
        fp = int(np.sum((y_array == 0) & (y_pred == 1)))
        fn = int(np.sum((y_array == 1) & (y_pred == 0)))
        tp = int(np.sum((y_array == 1) & (y_pred == 1)))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_array, y_pred),
                "precision": precision_score(y_array, y_pred, zero_division=0),
                "recall": recall_score(y_array, y_pred, zero_division=0),
                "f1": f1_score(y_array, y_pred, zero_division=0),
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
                "approval_rate": float(np.mean(y_pred == 0)),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    return pd.DataFrame(rows)


def compute_calibration_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Calculate reliability bins, Brier score and quantile-based ECE."""
    data = pd.DataFrame(
        {
            "target": np.asarray(y_true, dtype=int),
            "predicted_probability": np.asarray(y_score, dtype=float),
        }
    )
    data["bin"] = pd.qcut(
        data["predicted_probability"],
        q=n_bins,
        duplicates="drop",
    )
    grouped = data.groupby("bin", observed=True)
    calibration = grouped.agg(
        mean_predicted_probability=("predicted_probability", "mean"),
        observed_positive_rate=("target", "mean"),
        count=("target", "size"),
    ).reset_index()
    calibration["bin_lower"] = calibration["bin"].map(lambda interval: float(interval.left))
    calibration["bin_upper"] = calibration["bin"].map(lambda interval: float(interval.right))
    calibration["absolute_calibration_error"] = (
        calibration["observed_positive_rate"]
        - calibration["mean_predicted_probability"]
    ).abs()
    calibration["weight"] = calibration["count"] / len(data)

    ece = float(
        np.sum(
            calibration["weight"]
            * calibration["absolute_calibration_error"]
        )
    )
    brier = float(brier_score_loss(data["target"], data["predicted_probability"]))
    metrics: dict[str, float | int] = {
        "brier_score": brier,
        "ece": ece,
        "n_bins": int(len(calibration)),
    }
    calibration["bin"] = calibration["bin"].astype(str)
    return calibration, metrics


def save_figure(
    fig: plt.Figure,
    name: str,
    aliases: tuple[Path, ...] = (),
    use_tight_layout: bool = True,
) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    output = IMG_DIR / name
    if use_tight_layout:
        fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    for alias in aliases:
        alias.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(alias, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output.relative_to(BASE_DIR)}")


def add_source_note(fig: plt.Figure, n_test: int) -> None:
    fig.text(
        0.5,
        0.012,
        f"Calcul direct sur {n_test:,} predictions individuelles du jeu de test; aucune courbe reconstruite.",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )


def plot_roc(points: pd.DataFrame, auc_value: float, n_test: int) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.plot(
        points["false_positive_rate"],
        points["true_positive_rate"],
        color=BLUE,
        linewidth=2.6,
        label=f"XGBoost recalibre (ROC-AUC = {auc_value:.2%})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1.5, label="Classifieur aleatoire")
    ax.set_title("Courbe ROC-AUC réelle sur le jeu de test", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(loc="lower right")
    add_source_note(fig, n_test)
    save_figure(fig, "53_xgboost_roc_curve.png", aliases=(FIGURE_DIR / "roc_curve.png",))


def plot_precision_recall(
    points: pd.DataFrame,
    pr_auc: float,
    selected_metrics: pd.Series,
    positive_rate: float,
    n_test: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.plot(
        points["recall"],
        points["precision"],
        color=PURPLE,
        linewidth=2.5,
        label=f"XGBoost recalibre (PR-AUC = {pr_auc:.2%})",
    )
    ax.scatter(
        [selected_metrics["recall"]],
        [selected_metrics["precision"]],
        color=ORANGE,
        edgecolor="white",
        s=80,
        zorder=5,
        label=(
            f"Seuil valide : Precision={selected_metrics['precision']:.2%}, "
            f"Recall={selected_metrics['recall']:.2%}"
        ),
    )
    ax.axhline(
        positive_rate,
        linestyle="--",
        color=GRAY,
        label=f"Baseline = {positive_rate:.3f}",
    )
    ax.set_title("Courbe Precision-Recall empirique sur le jeu de test", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(loc="lower left")
    add_source_note(fig, n_test)
    save_figure(fig, "54_xgboost_precision_recall_curve.png")


def plot_metrics_summary(summary: dict) -> None:
    selected = summary["metrics_at_selected_threshold"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Gini", "KS", "PR-AUC"]
    values = [
        selected["accuracy"],
        selected["precision"],
        selected["recall"],
        selected["f1"],
        summary["roc_auc"],
        summary["gini"],
        summary["ks"],
        summary["pr_auc"],
    ]
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    bars = ax.barh(labels, values, color=[BLUE, GREEN, RED, PURPLE, ORANGE, BLUE, GREEN, RED])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Valeur mesuree")
    ax.set_title("Metriques empiriques du XGBoost recalibre", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.grid(axis="x", alpha=0.3, color=GRID)
    for bar, value in zip(bars, values):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2%}", va="center")
    fig.text(
        0.5,
        0.055,
        (
            f"Brier={summary['calibration']['brier_score']:.4f} | "
            f"ECE={summary['calibration']['ece']:.4f} | "
            f"Seuil valide={summary['configured_threshold']:.4f}"
        ),
        ha="center",
        fontsize=9,
        color=DARK,
    )
    add_source_note(fig, summary["n_test"])
    save_figure(fig, "52_xgboost_metrics_summary.png")


def plot_calibration_errors(summary: dict) -> None:
    labels = ["Brier score", "ECE"]
    values = [summary["calibration"]["brier_score"], summary["calibration"]["ece"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    for ax, label, value, color in zip(axes, labels, values, (BLUE, ORANGE)):
        ax.bar([label], [value], color=color, width=0.55)
        ax.set_ylim(0, max(0.15, value * 1.35))
        ax.set_ylabel("Erreur - plus faible est meilleur")
        ax.grid(axis="y", alpha=0.25, color=GRID)
        ax.text(0, value + max(0.006, value * 0.05), f"{value:.4f}", ha="center", fontweight="bold")
    fig.suptitle("Erreurs de calibration empiriques du XGBoost", fontsize=15, fontweight="bold", color=DARK)
    add_source_note(fig, summary["n_test"])
    save_figure(fig, "57_xgboost_brier_ece.png")


def plot_ks(points: pd.DataFrame, ks_index: int, n_test: int) -> None:
    selected = points.loc[ks_index]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.plot(points["population_share"], points["cumulative_risky"], color=RED, linewidth=2.5, label="Classe risquee cumulee")
    ax.plot(points["population_share"], points["cumulative_favorable"], color=GREEN, linewidth=2.5, label="Classe favorable cumulee")
    ax.vlines(
        selected["population_share"],
        selected["cumulative_favorable"],
        selected["cumulative_risky"],
        color=BLUE,
        linestyle="--",
        linewidth=2.4,
        label=f"KS = {selected['ks_gap']:.2%}",
    )
    ax.scatter(
        [selected["population_share"]] * 2,
        [selected["cumulative_favorable"], selected["cumulative_risky"]],
        color=BLUE,
        s=45,
        zorder=5,
    )
    ax.set_title("Courbe KS réelle sur le jeu de test", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Part de population triee par risque predit")
    ax.set_ylabel("Distribution cumulee")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(loc="lower right")
    ax.text(
        min(float(selected["population_share"]) + 0.025, 0.72),
        0.33,
        f"Seuil KS = {selected['threshold']:.4f}\nPopulation = {selected['population_share']:.1%}",
        fontsize=9.5,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID},
    )
    add_source_note(fig, n_test)
    save_figure(fig, "55_xgboost_ks_curve.png", aliases=(IMG_DIR / "35_ks_curve_xgboost.png",))


def plot_thresholds(metrics: pd.DataFrame, n_test: int, selected_threshold: float = 0.5) -> None:
    selected_index = int((metrics["threshold"] - selected_threshold).abs().idxmin())
    selected = metrics.loc[selected_index]

    fig, ax = plt.subplots(figsize=(9.6, 6.5))
    for metric, label, color in (
        ("accuracy", "Accuracy", BLUE),
        ("precision", "Precision", GREEN),
        ("recall", "Recall", RED),
        ("f1", "F1-score", PURPLE),
    ):
        ax.plot(metrics["threshold"], metrics[metric], color=color, linewidth=2.2, label=label)

    ax.axvline(
        selected["threshold"],
        color=ORANGE,
        linestyle="--",
        linewidth=2,
        label=f"Seuil valide = {selected['threshold']:.4f}",
    )
    ax.set_title("Analyse réelle du seuil de décision", fontsize=15, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel("Seuil de decision")
    ax.set_ylabel("Score")
    ax.set_xlim(float(metrics["threshold"].min()), float(metrics["threshold"].max()))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(loc="lower left", ncol=2)
    ax.text(
        0.62,
        0.46,
        (
            f"Au seuil {selected['threshold']:.4f}\n"
            f"Accuracy : {selected['accuracy']:.2%}\n"
            f"Precision : {selected['precision']:.2%}\n"
            f"Recall : {selected['recall']:.2%}\n"
            f"F1-score : {selected['f1']:.2%}"
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID},
    )
    add_source_note(fig, n_test)
    save_figure(fig, "59_xgboost_threshold_analysis.png", aliases=(IMG_DIR / "37_threshold_tradeoff_xgboost.png",))


def plot_calibration(
    calibration: pd.DataFrame,
    calibration_metrics: dict[str, float | int],
    y_true: pd.Series,
    y_score: np.ndarray,
) -> None:
    fig, (ax, histogram_ax) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=GRAY,
        linewidth=1.5,
        label="Calibration parfaite",
    )
    ax.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_positive_rate"],
        marker="o",
        markersize=7,
        color=BLUE,
        linewidth=2.5,
        label=(
            f"Modele (Brier = {float(calibration_metrics['brier_score']):.4f}, "
            f"ECE = {float(calibration_metrics['ece']):.4f})"
        ),
    )
    for _, row in calibration.iterrows():
        ax.vlines(
            row["mean_predicted_probability"],
            min(row["mean_predicted_probability"], row["observed_positive_rate"]),
            max(row["mean_predicted_probability"], row["observed_positive_rate"]),
            color=ORANGE,
            alpha=0.35,
            linewidth=1.2,
        )
    ax.set_title(
        "Courbe de calibration réelle sur le jeu de test",
        fontsize=15,
        fontweight="bold",
        color=DARK,
        loc="left",
    )
    ax.set_ylabel("Frequence positive observee")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(loc="upper left")

    y_array = np.asarray(y_true, dtype=int)
    histogram_ax.hist(
        y_score[y_array == 0],
        bins=np.linspace(0, 1, 21),
        color=GREEN,
        alpha=0.65,
        label="Classe favorable",
    )
    histogram_ax.hist(
        y_score[y_array == 1],
        bins=np.linspace(0, 1, 21),
        color=RED,
        alpha=0.60,
        label="Classe risquee",
    )
    histogram_ax.set_xlabel("Probabilite predite de la classe risquee")
    histogram_ax.set_ylabel("Effectif")
    histogram_ax.grid(axis="y", alpha=0.25, color=GRID)
    histogram_ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    add_source_note(fig, len(y_true))
    save_figure(
        fig,
        "56_xgboost_calibration_curve.png",
        aliases=(FIGURE_DIR / "calibration_curve.png",),
        use_tight_layout=False,
    )


def plot_confusion_matrix_real(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_array = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tn = int(np.sum((y_array == 0) & (y_pred == 0)))
    fp = int(np.sum((y_array == 0) & (y_pred == 1)))
    fn = int(np.sum((y_array == 1) & (y_pred == 0)))
    tp = int(np.sum((y_array == 1) & (y_pred == 1)))
    matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    row_percentages = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Nombre de dossiers")
    ax.set_title(
        f"Matrice de confusion réelle au seuil {threshold:.4f}",
        fontsize=15,
        fontweight="bold",
        color=DARK,
        loc="left",
    )
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_xticks([0, 1], labels=["Favorable (0)", "Risquée (1)"])
    ax.set_yticks([0, 1], labels=["Favorable (0)", "Risquée (1)"])

    maximum = matrix.max()
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > maximum * 0.55 else DARK
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}\n({row_percentages[row, column]:.1%})",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=color,
            )

    labels = {
        (0, 0): "TN",
        (0, 1): "FP",
        (1, 0): "FN",
        (1, 1): "TP",
    }
    for (row, column), label in labels.items():
        ax.text(
            column,
            row + 0.28,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if matrix[row, column] > maximum * 0.55 else GRAY,
        )
    add_source_note(fig, len(y_true))
    save_figure(
        fig,
        "58_xgboost_confusion_matrix.png",
        aliases=(FIGURE_DIR / "confusion_matrix.png",),
    )

    return pd.DataFrame(
        [
            {"actual": 0, "predicted": 0, "label": "TN", "count": tn},
            {"actual": 0, "predicted": 1, "label": "FP", "count": fp},
            {"actual": 1, "predicted": 0, "label": "FN", "count": fn},
            {"actual": 1, "predicted": 1, "label": "TP", "count": tp},
        ]
    )


def main() -> None:
    CURVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    y_true, y_score, selected_threshold, model_source = load_test_predictions()

    roc_points, auc_value = compute_roc_points(y_true, y_score)
    precision_recall_points, pr_auc = compute_precision_recall_points(y_true, y_score)
    ks_points, ks_index = compute_ks_points(roc_points, y_score)
    threshold_grid = np.unique(
        np.append(np.round(np.arange(0.05, 0.951, 0.01), 2), selected_threshold)
    )
    threshold_metrics = compute_threshold_metrics(y_true, y_score, threshold_grid)
    calibration_points, calibration_metrics = compute_calibration_metrics(y_true, y_score)
    confusion_points = plot_confusion_matrix_real(
        y_true,
        y_score,
        threshold=selected_threshold,
    )

    roc_points.to_csv(CURVE_DATA_DIR / "roc_curve_points.csv", index=False)
    precision_recall_points.to_csv(CURVE_DATA_DIR / "precision_recall_curve_points.csv", index=False)
    ks_points.to_csv(CURVE_DATA_DIR / "ks_curve_points.csv", index=False)
    threshold_metrics.to_csv(CURVE_DATA_DIR / "threshold_analysis.csv", index=False)
    threshold_metrics.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    threshold_metrics.to_csv(REPORT_DIR / "xgboost_threshold_metrics.csv", index=False)
    calibration_points.to_csv(CURVE_DATA_DIR / "calibration_curve_points.csv", index=False)
    calibration_points.to_csv(AUDIT_DATA_DIR / "calibration_bins.csv", index=False)
    confusion_points.to_csv(CURVE_DATA_DIR / "confusion_matrix.csv", index=False)

    ks_row = ks_points.loc[ks_index]
    threshold_05 = threshold_metrics.loc[np.isclose(threshold_metrics["threshold"], 0.5)].iloc[0]
    threshold_selected = threshold_metrics.loc[
        np.isclose(threshold_metrics["threshold"], selected_threshold)
    ].iloc[0]
    summary = {
        "source": f"predict_proba du {model_source} sur le jeu de test",
        "dataset_is_synthetic": True,
        "n_test": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": auc_value,
        "gini": 2 * auc_value - 1,
        "ks": float(ks_row["ks_gap"]),
        "ks_threshold": float(ks_row["threshold"]),
        "pr_auc": pr_auc,
        "configured_threshold": selected_threshold,
        "metrics_at_selected_threshold": {
            key: float(threshold_selected[key])
            for key in ("accuracy", "precision", "recall", "f1", "specificity", "approval_rate")
        },
        "metrics_at_0_5": {
            key: float(threshold_05[key])
            for key in ("accuracy", "precision", "recall", "f1", "specificity", "approval_rate")
        },
        "calibration": calibration_metrics,
        "confusion_matrix_at_selected_threshold": {
            row["label"].lower(): int(row["count"])
            for row in confusion_points.to_dict(orient="records")
        },
    }
    (CURVE_DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    audit_summary = {
        "source": summary["source"],
        "dataset_is_synthetic": True,
        "n_test": summary["n_test"],
        "positive_rate_test": summary["positive_rate"],
        "auc": summary["roc_auc"],
        "accuracy": summary["metrics_at_selected_threshold"]["accuracy"],
        "precision_risky": summary["metrics_at_selected_threshold"]["precision"],
        "recall_risky": summary["metrics_at_selected_threshold"]["recall"],
        "f1_risky": summary["metrics_at_selected_threshold"]["f1"],
        "gini": summary["gini"],
        "ks": summary["ks"],
        "ks_threshold": summary["ks_threshold"],
        "configured_threshold": summary["configured_threshold"],
        "brier_score": summary["calibration"]["brier_score"],
        "ece": summary["calibration"]["ece"],
        "calibration_bins": summary["calibration"]["n_bins"],
        "confusion_matrix_at_selected_threshold": summary["confusion_matrix_at_selected_threshold"],
    }
    (AUDIT_DATA_DIR / "model_audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2),
        encoding="utf-8",
    )
    threshold_metrics.to_csv(AUDIT_DATA_DIR / "threshold_analysis_extended.csv", index=False)
    threshold_metrics.to_csv(AUDIT_DATA_DIR / "threshold_tradeoff_xgboost.csv", index=False)

    plot_roc(roc_points, auc_value, len(y_true))
    plot_precision_recall(
        precision_recall_points,
        pr_auc,
        threshold_selected,
        float(np.mean(y_true)),
        len(y_true),
    )
    plot_ks(ks_points, ks_index, len(y_true))
    plot_thresholds(threshold_metrics, len(y_true), selected_threshold=selected_threshold)
    plot_calibration(calibration_points, calibration_metrics, y_true, y_score)
    plot_metrics_summary(summary)
    plot_calibration_errors(summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
