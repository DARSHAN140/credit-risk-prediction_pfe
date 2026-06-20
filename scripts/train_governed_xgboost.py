from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MAX_TRAIN_ROWS, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE, RAW_DATA_PATH, REPORT_DIR
from src.data_processing import infer_column_types, prepare_model_frame, read_credit_data, temporal_or_stratified_split
from src.evaluation import compute_metrics, save_metrics
from src.modeling import GOVERNED_XGBOOST_COLUMNS, SMOTE_SAMPLING_STRATEGY, build_preprocessor


def load_split():
    raw_df = read_credit_data(RAW_DATA_PATH)
    if len(raw_df) > MAX_TRAIN_ROWS:
        raw_df = (
            raw_df.groupby("decision", group_keys=False)
            .sample(frac=MAX_TRAIN_ROWS / len(raw_df), random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    X, y = prepare_model_frame(raw_df)
    selected = [column for column in GOVERNED_XGBOOST_COLUMNS if column in X.columns]
    missing = sorted(set(GOVERNED_XGBOOST_COLUMNS) - set(selected))
    if missing:
        raise ValueError(f"Variables gouvernees absentes du dataset: {missing}")
    X = X[selected]
    return raw_df, temporal_or_stratified_split(raw_df, X, y)


def build_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_columns, categorical_columns)),
            (
                "smote",
                SMOTE(
                    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("select", SelectKBest(score_func=mutual_info_classif, k="all")),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    n_estimators=70,
                    max_depth=2,
                    learning_rate=0.025,
                    subsample=0.65,
                    colsample_bytree=0.65,
                    reg_lambda=30.0,
                    min_child_weight=20,
                    gamma=2.0,
                ),
            ),
        ]
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, split = load_split()
    X_train, X_test, y_train, y_test = split
    numeric_columns, categorical_columns = infer_column_types(X_train)
    model = build_model(numeric_columns, categorical_columns)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_score)

    joblib.dump(model, MODEL_DIR / "best_model.joblib")
    save_metrics(metrics, REPORT_DIR / "final_metrics.json")

    scored_test = X_test.copy()
    scored_test["pd_default"] = y_score
    scored_test["target_reel"] = y_test.to_numpy()
    scored_test.to_csv(PROCESSED_DIR / "scored_test_sample.csv", index=False)

    metadata = {
        "best_model": "xgboost",
        "model_variant": "governed_conservative_features",
        "target_definition": (
            "1 = avis defavorable, non eligible ou etude approfondie; "
            "0 = avis favorable"
        ),
        "dataset_note": (
            "Dataset synthetique utilise pour une validation methodologique; "
            "les metriques ne constituent pas une validation bancaire reelle."
        ),
        "methodology_note": (
            "Metriques calculees directement sur les probabilites du jeu de test. "
            "Le modele final utilise un jeu conservateur de 18 variables et exclut "
            "les variables qui reproduisent trop directement la regle synthetique "
            "de decision. Aucune metrique n'est plafonnee ou remplacee apres calcul."
        ),
        "excluded_dominant_proxies": [
            "dti_calcule",
            "reste_a_vivre",
            "ratio_credit_revenu",
            "revenu_total",
            "taux_epargne",
            "mensualites_existantes",
            "epargne_mensuelle",
            "incidents_paiement_12m",
            "retards_paiement_12m",
            "incident_ou_retard",
        ],
        "threshold": 0.5,
        "features": X_train.columns.tolist(),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    audit = {
        "model": "xgboost",
        "variant": "governed_conservative_features",
        "features": X_train.columns.tolist(),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    output = REPORT_DIR / "real_curves" / "governed_model_training.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
