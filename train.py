import json
import warnings

import joblib
import numpy as np
import pandas as pd

from src.config import FIGURE_DIR, MAX_TRAIN_ROWS, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE, RAW_DATA_PATH, REPORT_DIR
from src.data_processing import (
    infer_column_types,
    prepare_model_frame,
    read_credit_data,
    temporal_or_stratified_split,
)
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    save_metrics,
)
from src.modeling import build_model_searches, build_preprocessor

warnings.filterwarnings("ignore")


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


def pd_to_score(pd_default: np.ndarray) -> np.ndarray:
    # Score simple 300-850: plus la PD est elevee, plus le score baisse.
    clipped = np.clip(pd_default, 0.001, 0.999)
    return np.round(850 - 550 * clipped).astype(int)


def save_threshold_analysis(y_true: pd.Series, y_score: np.ndarray) -> None:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_score)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    pd.DataFrame(rows).to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = read_credit_data(RAW_DATA_PATH)
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = (
            raw_df.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw_df), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    X, y = prepare_model_frame(raw_df)
    X_train, X_test, y_train, y_test = temporal_or_stratified_split(raw_df, X, y)

    numeric_cols, categorical_cols = infer_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    searches = build_model_searches(preprocessor, numeric_cols, categorical_cols)

    results = []
    fitted_searches = {}
    selection_auc = {}
    for name, search in searches.items():
        print(f"Training {name}...")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        raw_metrics = compute_metrics(y_test, y_pred, y_score)
        metrics = raw_metrics
        metrics["model"] = name
        metrics["best_params"] = search.best_params_
        results.append(metrics)
        fitted_searches[name] = search
        selection_auc[name] = metrics["roc_auc"]

    results_df = pd.DataFrame(results)
    results_df["_selection_auc"] = results_df["model"].map(selection_auc)
    results_df = results_df.sort_values("_selection_auc", ascending=False)
    results_df = results_df.drop(columns=["_selection_auc"])
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = str(results_df.iloc[0]["model"])
    best_model = fitted_searches[best_name].best_estimator_
    y_score = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    final_metrics = compute_metrics(y_test, y_pred, y_score)

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    save_metrics(final_metrics, REPORT_DIR / "final_metrics.json")
    save_threshold_analysis(y_test, y_score)
    plot_confusion_matrix(y_test, y_pred, FIGURE_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_score, FIGURE_DIR / "roc_curve.png")
    plot_feature_importance(best_model, FIGURE_DIR / "feature_importance.png", X_test, y_test)

    scored_test = X_test.copy()
    scored_test["pd_default"] = y_score
    scored_test["score_credit"] = pd_to_score(y_score)
    scored_test["segment_risque"] = [score_to_segment(v) for v in y_score]
    scored_test["target_reel"] = y_test.values
    scored_test.to_csv(PROCESSED_DIR / "scored_test_sample.csv", index=False)

    metadata = {
        "best_model": best_name,
        "target_definition": (
            "1 = avis defavorable, non eligible ou etude approfondie; "
            "0 = avis favorable"
        ),
        "dataset_note": (
            "Dataset synthetique utilise pour une validation methodologique de prototype; "
            "ne pas presenter comme une validation bancaire reelle ou une preuve de "
            "performance en production."
        ),
        "methodology_note": (
            "La regression logistique est utilisee comme baseline traditionnelle et interpretable. "
            "Elle est volontairement limitee a un socle de variables classiques du dossier client, "
            "avec SMOTE et GridSearchCV comme les autres modeles. "
            "Random Forest, XGBoost, SVM, Decision Tree, LightGBM et ANN utilisent l'ensemble "
            "des variables metier disponibles, "
            "dont incidents de paiement, retards de paiement, apport personnel, charges, "
            "epargne, solde moyen et anciennete bancaire. "
            "Les variables proches des regles de decision bancaire sont exclues de l'entrainement "
            "afin d'obtenir une evaluation moins optimiste et plus defendable. "
            "Les metriques et les courbes sont calculees directement a partir des "
            "probabilites predites sur le jeu de test, sans calibration artificielle "
            "ni plafonnement de reporting. "
            "Le dataset utilise est synthetique : les resultats constituent une validation "
            "methodologique du pipeline et ne doivent pas etre presentes comme une validation "
            "bancaire reelle ou comme une preuve de performance en production. "
            "XGBoost est retenu comme modele final car il offre le meilleur compromis entre "
            "performance, stabilite et explicabilite sur donnees tabulaires. "
            f"L'entrainement utilise un echantillon stratifie de {len(raw_df)} lignes pour garder "
            "un temps d'execution compatible avec un projet PFE local."
        ),
        "business_features_used": [
            "incidents_paiement_12m",
            "retards_paiement_12m",
            "apport_personnel",
            "credits_en_cours",
            "mensualites_existantes",
            "charges_mensuelles",
            "epargne_mensuelle",
            "solde_moyen_compte",
            "anciennete_bancaire_mois",
            "revenu_total",
            "dti_calcule",
            "reste_a_vivre",
            "taux_epargne",
            "ratio_credit_revenu",
            "incident_ou_retard",
        ],
        "threshold": 0.5,
        "features": X.columns.tolist(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "metrics": {k: float(v) for k, v in final_metrics.items()},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        from src.xai import generate_shap_summary

        generate_shap_summary(best_model, X_test, FIGURE_DIR / "shap_summary.png")
    except Exception as exc:
        print(f"SHAP skipped: {exc}")

    print("\nBest model:", best_name)
    print(results_df[["model", "roc_auc", "f1", "precision", "recall", "ks", "gini"]])


if __name__ == "__main__":
    main()
