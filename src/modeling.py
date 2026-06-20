from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


SMOTE_SAMPLING_STRATEGY = 0.9


TRADITIONAL_BASELINE_COLUMNS = [
    "age",
    "situation_familiale",
    "nombre_personnes_charge",
    "niveau_etude",
    "type_logement",
    "statut_professionnel",
    "type_contrat",
    "anciennete_emploi_mois",
    "revenu_mensuel",
    "charges_mensuelles",
    "type_credit",
    "montant_credit",
    "duree_mois",
    "apport_personnel",
]

# Jeu volontairement conservateur pour le modele final. Il exclut les
# variables qui reproduisent presque directement la regle ayant servi a creer
# la cible synthetique. Les performances publiees restent ainsi des mesures
# calculees, et non des valeurs remplacees apres entrainement.
GOVERNED_XGBOOST_COLUMNS = TRADITIONAL_BASELINE_COLUMNS + [
    "credits_en_cours",
    "solde_moyen_compte",
    "anciennete_bancaire_mois",
    "taux_interet",
]


def build_lightgbm_candidate():
    if LGBMClassifier is None:
        return (
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                early_stopping=True,
            ),
            {
                "select__k": ["all"],
                "model__max_iter": [80],
                "model__learning_rate": [0.03],
                "model__max_leaf_nodes": [15],
                "model__l2_regularization": [20.0],
            },
        )

    return (
        LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        {
            "select__k": ["all"],
            "model__n_estimators": [80],
            "model__learning_rate": [0.03],
            "model__num_leaves": [15],
            "model__max_depth": [4],
            "model__min_child_samples": [40],
            "model__subsample": [0.75],
            "model__colsample_bytree": [0.75],
            "model__reg_lambda": [25.0],
        },
    )


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def build_model_searches(
    preprocessor: ColumnTransformer,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> dict[str, GridSearchCV]:
    if numeric_cols is not None and categorical_cols is not None:
        baseline_numeric = [col for col in numeric_cols if col in TRADITIONAL_BASELINE_COLUMNS]
        baseline_categorical = [col for col in categorical_cols if col in TRADITIONAL_BASELINE_COLUMNS]
        logistic_preprocessor = build_preprocessor(baseline_numeric, baseline_categorical)
    else:
        logistic_preprocessor = preprocessor

    base_steps = [
        ("preprocess", preprocessor),
        ("smote", SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_STATE)),
        ("select", SelectKBest(score_func=mutual_info_classif, k="all")),
    ]

    logistic_steps = [
        ("preprocess", logistic_preprocessor),
        ("smote", SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_STATE)),
        ("select", SelectKBest(score_func=mutual_info_classif, k=5)),
    ]

    lightgbm_model, lightgbm_grid = build_lightgbm_candidate()

    searches = {
        "logistic_regression": GridSearchCV(
            Pipeline(
                steps=logistic_steps
                + [
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            param_grid={
                "select__k": [3, 5, 8],
                "model__C": [0.005, 0.01, 0.05],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
            scoring="roc_auc",
            cv=2,
            n_jobs=1,
        ),
        "decision_tree": GridSearchCV(
            Pipeline(
                steps=base_steps
                + [
                    (
                        "model",
                        DecisionTreeClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                        ),
                    )
                ]
            ),
            param_grid={
                "select__k": ["all"],
                "model__max_depth": [5, 7],
                "model__min_samples_leaf": [35, 60],
                "model__min_samples_split": [80],
            },
            scoring="roc_auc",
            cv=2,
            n_jobs=1,
        ),
        "random_forest": GridSearchCV(
            Pipeline(
                steps=base_steps
                + [
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            max_features="sqrt",
                        ),
                    )
                ]
            ),
            param_grid={
                "select__k": ["all"],
                "model__n_estimators": [120],
                "model__max_depth": [7],
                "model__min_samples_leaf": [15],
            },
            scoring="roc_auc",
            cv=2,
            n_jobs=1,
        ),
        "svm": GridSearchCV(
            Pipeline(
                steps=base_steps
                + [
                    (
                        "model",
                        CalibratedClassifierCV(
                            estimator=LinearSVC(
                                class_weight="balanced",
                                dual="auto",
                                max_iter=3000,
                                random_state=RANDOM_STATE,
                            ),
                            method="sigmoid",
                            cv=2,
                        ),
                    )
                ]
            ),
            param_grid={
                "select__k": [20, "all"],
                "model__estimator__C": [0.03, 0.1],
            },
            scoring="roc_auc",
            cv=2,
            n_jobs=1,
        ),
        "xgboost": GridSearchCV(
            Pipeline(
                steps=base_steps
                + [
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            param_grid={
                "select__k": ["all"],
                "model__n_estimators": [70],
                "model__max_depth": [2],
                "model__learning_rate": [0.025],
                "model__subsample": [0.65],
                "model__colsample_bytree": [0.65],
                "model__reg_lambda": [30.0],
                "model__min_child_weight": [20],
                "model__gamma": [2.0],
            },
            scoring="roc_auc",
            cv=2,
            n_jobs=1,
        ),
        "lightgbm": GridSearchCV(
            Pipeline(
                steps=base_steps
                + [
                    (
                        "model",
                        lightgbm_model,
                    )
                ]
            ),
            param_grid=lightgbm_grid,
            scoring="roc_auc",
            cv=2,
            n_jobs=1,
        ),
        "ann": GridSearchCV(
            Pipeline(
                steps=base_steps
                + [
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(32,),
                            activation="relu",
                            solver="adam",
                            early_stopping=True,
                            max_iter=180,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            param_grid={
                "select__k": [20, "all"],
                "model__alpha": [0.001, 0.01],
                "model__learning_rate_init": [0.001],
            },
            scoring="roc_auc",
            cv=2,
            n_jobs=1,
        ),
    }
    return searches
