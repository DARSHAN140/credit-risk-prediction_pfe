import unicodedata

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from sklearn.model_selection import train_test_split

from src.config import LEAKAGE_COLUMNS, POLICY_RULE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def read_credit_data(path: str) -> pd.DataFrame:
    """Read the credit CSV with a small encoding fallback strategy."""
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, ParserError, MemoryError):
            try:
                return pd.read_csv(path, encoding=encoding, engine="python")
            except (UnicodeDecodeError, ParserError, MemoryError):
                continue
        except OSError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            try:
                return pd.read_csv(path, encoding=encoding, engine="python")
            except (ParserError, MemoryError, OSError):
                continue
            continue
    return pd.read_csv(path, encoding="latin1")


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary credit-risk target.

    1 = risky decision: unfavorable, non eligible, or requiring deeper review.
    0 = favorable decision.
    """
    df = df.copy()
    if "decision" not in df.columns:
        raise ValueError("La colonne cible 'decision' est introuvable dans le dataset.")

    normalized = df["decision"].map(normalize_text)
    df[TARGET_COLUMN] = normalized.apply(
        lambda x: 1
        if isinstance(x, str)
        and (
            "defavorable" in x
            or "refus" in x
            or "non eligible" in x
            or "etude approfondie" in x
        )
        else 0
    )
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    if "date_demande" in df.columns:
        df["date_demande"] = pd.to_datetime(df["date_demande"], errors="coerce")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    df = df.drop_duplicates()
    return df


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering simple, défendable et réalisable en trois semaines."""
    df = df.copy()

    revenu_total = numeric_series(df, "revenu_mensuel").fillna(0) + numeric_series(
        df, "revenu_supplementaire"
    ).fillna(0)
    mensualites_total = numeric_series(df, "mensualites_existantes").fillna(0) + numeric_series(
        df, "mensualite_estimee"
    ).fillna(0)
    charges = numeric_series(df, "charges_mensuelles").fillna(0)

    df["revenu_total"] = revenu_total
    df["dti_calcule"] = np.where(revenu_total > 0, mensualites_total / revenu_total, np.nan)
    df["reste_a_vivre"] = revenu_total - charges - mensualites_total
    df["taux_epargne"] = np.where(
        revenu_total > 0, numeric_series(df, "epargne_mensuelle").fillna(0) / revenu_total, np.nan
    )
    df["ratio_credit_revenu"] = np.where(
        revenu_total > 0, numeric_series(df, "montant_credit").fillna(0) / revenu_total, np.nan
    )
    df["incident_ou_retard"] = (
        (numeric_series(df, "incidents_paiement_12m").fillna(0) > 0)
        | (numeric_series(df, "retards_paiement_12m").fillna(0) > 0)
    ).astype(int)

    if "date_demande" in df.columns:
        df["annee_demande"] = df["date_demande"].dt.year
        df["mois_demande"] = df["date_demande"].dt.month

    df["age_bin"] = pd.cut(
        numeric_series(df, "age", np.nan),
        bins=[17, 25, 35, 45, 55, 70, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-70", "70+"],
    )
    df["revenu_bin"] = pd.cut(
        numeric_series(df, "revenu_mensuel", np.nan),
        bins=[-1, 4000, 8000, 15000, 30000, np.inf],
        labels=["tres_faible", "faible", "moyen", "eleve", "tres_eleve"],
    )
    df["montant_credit_bin"] = pd.cut(
        numeric_series(df, "montant_credit", np.nan),
        bins=[-1, 50000, 150000, 400000, 900000, np.inf],
        labels=["petit", "moyen", "important", "tres_important", "exceptionnel"],
    )
    return df


def prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = basic_cleaning(df)
    df = build_target(df)
    df = add_credit_features(df)

    drop_cols = [col for col in LEAKAGE_COLUMNS + POLICY_RULE_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols + [TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN].astype(int)

    # La date brute n'est pas donnée directement au modèle; on garde année/mois.
    X = X.drop(columns=["date_demande"], errors="ignore")
    return X, y


def temporal_or_stratified_split(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if "date_demande" in df.columns:
        dates = pd.to_datetime(df.loc[X.index, "date_demande"], errors="coerce")
        valid_idx = dates.dropna().sort_values().index
        if len(valid_idx) == len(X):
            cutoff = int(len(valid_idx) * (1 - test_size))
            train_idx = valid_idx[:cutoff]
            test_idx = valid_idx[cutoff:]
            return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols
