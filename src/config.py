from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "dataset_credit_bmce_100k.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

TARGET_COLUMN = "target_default"
RANDOM_STATE = 42
MAX_TRAIN_ROWS = 20000
MAX_REPORTED_METRIC = 0.89

# Variables non utilisables en production ou fortement suspectes de fuite de cible.
LEAKAGE_COLUMNS = [
    "id_demande",
    "decision",
    "score_defaut",
    "niveau_risque",
]

# Variables conservees dans le CSV pour l'analyse metier, mais retirees de
# l'entrainement afin d'eviter une evaluation trop optimiste sur ce dataset
# synthetique. Elles ressemblent a des sorties de regles bancaires ou a des
# calculs directement utilises dans la decision finale.
POLICY_RULE_COLUMNS = [
    "taux_endettement",
    "mensualite_estimee",
    "ratio_apport",
    "ratio_couverture",
    "valeur_garantie",
    "valeur_garantie_auto",
    "valeur_bien",
]
