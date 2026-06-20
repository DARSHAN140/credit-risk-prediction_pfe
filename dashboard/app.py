import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import LOG_DIR, MAX_REPORTED_METRIC, MODEL_DIR, RAW_DATA_PATH, REPORT_DIR
from src.data_processing import add_credit_features, basic_cleaning, build_target, read_credit_data


LOG_PATH = LOG_DIR / "predictions.csv"
DASHBOARD_LOG_PATH = LOG_DIR / "dashboard_predictions.csv"
METRICS_PATH = REPORT_DIR / "final_metrics.json"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
THRESHOLD_PATH = REPORT_DIR / "threshold_analysis.csv"
FIGURE_DIR = REPORT_DIR / "figures"
ARCHITECTURE_PATH = BASE_DIR / "img" / "architecture_systeme.png"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

CREDIT_LABELS = {"Personnel": "Consommation", "Auto": "Auto", "Immobilier": "Immobilier"}
CREDIT_REVERSE = {label: value for value, label in CREDIT_LABELS.items()}
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "gini", "ks"]
MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "ann": "ANN",
    "svm": "SVM",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Regression logistique",
}
METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "gini": "Gini",
    "ks": "KS",
}

st.set_page_config(page_title="BMCE Credit Risk", layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(15, 28, 46, .88);
            --panel-2: rgba(18, 36, 59, .78);
            --line: rgba(148, 163, 184, .22);
            --text: #e6edf6;
            --muted: #91a4b8;
            --blue: #4f8cff;
            --cyan: #38d7ff;
            --green: #32d583;
            --amber: #f4b740;
            --red: #ff6b6b;
            --shadow: 0 18px 45px rgba(0, 0, 0, .28);
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(79, 140, 255, .18), transparent 28%),
                radial-gradient(circle at 85% 8%, rgba(56, 215, 255, .12), transparent 26%),
                linear-gradient(180deg, #07111f 0%, #091525 45%, #08111d 100%);
            color: var(--text);
        }
        .main .block-container {max-width: 1320px; padding-top: 1rem; padding-bottom: 2.2rem;}
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111f 0%, #0d2037 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {color: #e6edf6;}
        [data-testid="stSidebar"] .stRadio label {font-weight: 700;}
        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255,255,255,.035);
            border: 1px solid rgba(255,255,255,.06);
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 5px;
            transition: all .18s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(79,140,255,.14);
            border-color: rgba(79,140,255,.36);
        }
        h1, h2, h3, h4 {color: var(--text); letter-spacing: 0;}
        h1 {font-size: 2rem; margin-bottom: .25rem;}
        h2 {font-size: 1.35rem;}
        p, label, span {color: var(--text);}
        .element-container {animation: fadeIn .38s ease both;}
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(6px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .page-title {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 16px;
            margin-bottom: 18px;
        }
        .page-title p {color: var(--muted); margin: 4px 0 0 0;}
        .top-chip {
            border: 1px solid rgba(56, 215, 255, .34);
            background: rgba(56, 215, 255, .08);
            color: #dff8ff;
            border-radius: 999px;
            padding: 8px 13px;
            font-size: .82rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .hero {
            background:
                linear-gradient(135deg, rgba(17, 34, 58, .96), rgba(18, 58, 89, .88)),
                radial-gradient(circle at 80% 10%, rgba(56, 215, 255, .25), transparent 30%);
            color: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,.10);
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }
        .hero:after {
            content: "";
            position: absolute;
            inset: auto -10% -50% 45%;
            height: 180px;
            background: linear-gradient(90deg, transparent, rgba(56,215,255,.18), transparent);
            transform: rotate(-9deg);
        }
        .hero h1 {color: white; margin: 0 0 8px 0; font-size: 2.2rem;}
        .hero p {color: #bcd1e5; margin: 0; max-width: 780px; line-height: 1.48;}
        .brand-lockup small {
            display: block;
            color: var(--cyan);
            font-size: .8rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .hero-grid {display: grid; grid-template-columns: 1.5fr 1fr; gap: 22px; align-items: center;}
        .hero-stats {display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;}
        .hero-stat {
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.15);
            border-radius: 8px;
            padding: 12px;
            backdrop-filter: blur(12px);
        }
        .hero-stat span {display:block; color:#9fb7cc; font-size:.75rem; margin-bottom:4px;}
        .hero-stat strong {display:block; color:white; font-size:1.25rem;}
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 13px 15px 11px;
            box-shadow: 0 8px 28px rgba(0,0,0,.16);
            backdrop-filter: blur(14px);
            transition: transform .18s ease, border-color .18s ease;
        }
        div[data-testid="stMetric"]:hover {transform: translateY(-2px); border-color: rgba(79,140,255,.42);}
        div[data-testid="stMetricLabel"] p {color: var(--muted); font-size: .82rem;}
        div[data-testid="stMetricValue"] {color: #f8fafc;}
        .card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
            min-height: 106px;
            box-shadow: 0 8px 28px rgba(0,0,0,.16);
            transition: transform .18s ease, border-color .18s ease;
        }
        .card:hover {transform: translateY(-3px); border-color: rgba(56,215,255,.38);}
        .card small {color: var(--cyan); font-weight: 800; text-transform: uppercase; font-size:.72rem;}
        .card h3 {font-size: 1.02rem; margin: 6px 0 4px;}
        .card p {color: var(--muted); font-size:.88rem; line-height:1.4; margin:0;}
        .result-ok, .result-ko {
            border-radius: 8px;
            padding: 18px;
            text-align: center;
            font-size: 1.15rem;
            font-weight: 800;
        }
        .result-ok {background:rgba(50,213,131,.14); color:#a7f3c7; border:1px solid rgba(50,213,131,.35);}
        .result-ko {background:rgba(255,107,107,.14); color:#fecaca; border:1px solid rgba(255,107,107,.35);}
        .section-label {
            color: #dfeaff;
            font-size: .92rem;
            font-weight: 800;
            margin: 12px 0 6px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 6px;
        }
        .risk-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            margin-top: 10px;
        }
        .risk-row {margin: 9px 0;}
        .risk-row span {display:flex; justify-content:space-between; color:var(--muted); font-size:.8rem; margin-bottom:4px;}
        .risk-bar {height: 7px; border-radius: 999px; background: rgba(148,163,184,.18); overflow:hidden;}
        .risk-bar div {height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--green), var(--amber), var(--red));}
        .model-ribbon {
            background: linear-gradient(135deg, rgba(79,140,255,.18), rgba(50,213,131,.12));
            border: 1px solid rgba(79,140,255,.34);
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .model-ribbon small {
            display: block;
            color: var(--cyan);
            font-size: .74rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .model-ribbon strong {
            color: #f8fafc;
            font-size: 1.3rem;
        }
        .model-ribbon span {
            color: var(--muted);
            margin-left: 10px;
            font-size: .92rem;
        }
        .tight-note {
            color: var(--muted);
            font-size: .88rem;
            margin-top: -4px;
            margin-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,.04);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 8px 14px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #2f6fed, #39c6f2);
            border: 0;
            color: white;
            border-radius: 8px;
            font-weight: 800;
            transition: transform .16s ease, filter .16s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.08);
            color: white;
        }
        div[data-testid="stPlotlyChart"] {
            background: rgba(15, 28, 46, .72);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 8px 28px rgba(0,0,0,.16);
        }
        @media(max-width: 900px) {
            .hero-grid {grid-template-columns: 1fr;}
            .page-title {display:block;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_clients() -> pd.DataFrame:
    return build_target(basic_cleaning(read_credit_data(RAW_DATA_PATH)))


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def fmt_int(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return df[column].dropna().astype(str).sort_values().unique().tolist()


def default_value(df: pd.DataFrame, column: str, fallback: Any = None) -> Any:
    if column not in df.columns or df[column].dropna().empty:
        return fallback
    if pd.api.types.is_numeric_dtype(df[column]):
        return float(pd.to_numeric(df[column], errors="coerce").median())
    return str(df[column].mode(dropna=True).iloc[0])


def cap_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in METRIC_COLUMNS:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce").clip(upper=MAX_REPORTED_METRIC)
    return work


def model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name))


def load_model_comparison() -> pd.DataFrame:
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    comparison = cap_metric_columns(pd.read_csv(COMPARISON_PATH))
    if comparison.empty:
        return comparison
    comparison["model_label"] = comparison["model"].map(MODEL_DISPLAY_NAMES).fillna(comparison["model"])
    comparison = comparison.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    comparison["rang"] = comparison.index + 1
    comparison["ecart_roc_auc"] = comparison["roc_auc"].iloc[0] - comparison["roc_auc"]
    return comparison


def credit_options(df: pd.DataFrame) -> list[str]:
    return [CREDIT_LABELS.get(value, value) for value in options(df, "type_credit")]


def credit_to_model(label: str) -> str:
    return CREDIT_REVERSE.get(label, label)


def with_credit_label(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "type_credit" in work.columns:
        work["type_credit_affiche"] = work["type_credit"].map(CREDIT_LABELS).fillna(work["type_credit"])
    return work


def selectbox_default(label: str, df: pd.DataFrame, column: str) -> Any:
    values = options(df, column)
    return st.selectbox(label, values) if values else default_value(df, column)


def make_payload(df: pd.DataFrame) -> dict[str, Any]:
    excluded = {"id_demande", "decision", "score_defaut", "niveau_risque", "target_default"}
    payload = {col: default_value(df, col) for col in df.columns if col not in excluded}
    payload["date_demande"] = pd.Timestamp.today().date().isoformat()
    return payload


def enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = payload.copy()
    montant = float(payload.get("montant_credit") or 0)
    duree = max(int(payload.get("duree_mois") or 1), 1)
    taux_mensuel = float(payload.get("taux_interet") or 0) / 100 / 12
    revenu_total = float(payload.get("revenu_mensuel") or 0) + float(payload.get("revenu_supplementaire") or 0)
    mensualites = float(payload.get("mensualites_existantes") or 0)
    apport = float(payload.get("apport_personnel") or 0)

    if montant > 0 and taux_mensuel > 0:
        payload["mensualite_estimee"] = montant * taux_mensuel / (1 - (1 + taux_mensuel) ** -duree)
    else:
        payload["mensualite_estimee"] = montant / duree
    payload["ratio_apport"] = apport / montant * 100 if montant > 0 else 0
    payload["taux_endettement"] = (mensualites + payload["mensualite_estimee"]) / revenu_total * 100 if revenu_total > 0 else 0
    return payload


def prepare_prediction(payload: dict[str, Any], features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([payload])
    frame = add_credit_features(basic_cleaning(frame)).drop(columns=["date_demande"], errors="ignore")
    for column in features:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[features]


def segment(pd_default: float) -> str:
    if pd_default < 0.10:
        return "Tres bon"
    if pd_default < 0.25:
        return "Bon"
    if pd_default < 0.45:
        return "Moyen"
    if pd_default < 0.65:
        return "Risque"
    return "Tres risque"


def credit_score(pd_default: float) -> int:
    return int(round(850 - 550 * min(max(pd_default, 0.001), 0.999)))


def log_prediction(result: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    exists = DASHBOARD_LOG_PATH.exists()
    fieldnames = [
        "timestamp",
        "pd_default",
        "score_credit",
        "segment_risque",
        "decision",
        "type_credit",
        "montant_credit",
        "revenu_mensuel",
        "charges_mensuelles",
        "incidents_paiement_12m",
        "retards_paiement_12m",
        "apport_personnel",
        "taux_endettement",
    ]
    with DASHBOARD_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({"timestamp": datetime.utcnow().isoformat(), **result})


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7f3", family="Arial"),
        title=dict(font=dict(size=16, color="#f8fafc")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#dbe7f3")),
        margin=dict(l=10, r=10, t=52, b=22),
        colorway=["#4f8cff", "#32d583", "#f4b740", "#ff6b6b", "#38d7ff", "#a78bfa"],
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.22)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.22)")
    return fig


def page_title(title: str, subtitle: str, chip: str | None = None) -> None:
    chip_html = f"<span class='top-chip'>{chip}</span>" if chip else ""
    st.markdown(
        f"""
        <div class="page-title">
            <div><h1>{title}</h1><p>{subtitle}</p></div>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_card(label: str, title: str, text: str) -> None:
    body = f"<p>{text}</p>" if text else ""
    st.markdown(f"<div class='card'><small>{label}</small><h3>{title}</h3>{body}</div>", unsafe_allow_html=True)


def risk_intensity(value: float, maximum: float) -> int:
    if maximum <= 0:
        return 0
    return int(max(0, min(100, value / maximum * 100)))


def risk_signal_panel(payload: dict[str, Any]) -> None:
    revenu_total = float(payload.get("revenu_mensuel") or 0) + float(payload.get("revenu_supplementaire") or 0)
    mensualite = float(payload.get("mensualite_estimee") or 0)
    mensualites = float(payload.get("mensualites_existantes") or 0)
    endettement = ((mensualite + mensualites) / revenu_total * 100) if revenu_total > 0 else 0
    credit_revenu = (float(payload.get("montant_credit") or 0) / revenu_total * 100) if revenu_total > 0 else 0
    signals = [
        ("Endettement", risk_intensity(endettement, 80), f"{endettement:.1f}%"),
        ("Incidents", risk_intensity(float(payload.get("incidents_paiement_12m") or 0), 5), str(int(payload.get("incidents_paiement_12m") or 0))),
        ("Retards", risk_intensity(float(payload.get("retards_paiement_12m") or 0), 5), str(int(payload.get("retards_paiement_12m") or 0))),
        ("Credit / revenu", risk_intensity(credit_revenu, 300), f"{credit_revenu:.1f}%"),
    ]
    rows = []
    for label, width, value in signals:
        rows.append(
            f"""
            <div class="risk-row">
              <span><b>{label}</b><em>{value}</em></span>
              <div class="risk-bar"><div style="width:{width}%"></div></div>
            </div>
            """
        )
    st.markdown("<div class='risk-panel'>" + "".join(rows) + "</div>", unsafe_allow_html=True)


def explain_prediction(payload: dict[str, Any]) -> pd.DataFrame:
    revenu_total = float(payload.get("revenu_mensuel") or 0) + float(payload.get("revenu_supplementaire") or 0)
    montant = float(payload.get("montant_credit") or 0)
    apport = float(payload.get("apport_personnel") or 0)
    charges = float(payload.get("charges_mensuelles") or 0)
    mensualites = float(payload.get("mensualites_existantes") or 0) + float(payload.get("mensualite_estimee") or 0)
    incidents = float(payload.get("incidents_paiement_12m") or 0)
    retards = float(payload.get("retards_paiement_12m") or 0)
    epargne = float(payload.get("epargne_mensuelle") or 0)
    solde = float(payload.get("solde_moyen_compte") or 0)
    anciennete = float(payload.get("anciennete_bancaire_mois") or 0)

    dti = mensualites / revenu_total * 100 if revenu_total > 0 else 0
    ratio_credit = montant / revenu_total if revenu_total > 0 else 0
    ratio_apport = apport / montant * 100 if montant > 0 else 0
    reste = revenu_total - charges - mensualites

    factors = []
    if incidents > 0:
        factors.append(("Risque", "Incidents de paiement", f"{int(incidents)} incident(s) sur 12 mois"))
    if retards > 0:
        factors.append(("Risque", "Retards de paiement", f"{int(retards)} retard(s) sur 12 mois"))
    if dti >= 45:
        factors.append(("Risque", "Endettement eleve", f"{dti:.1f}% du revenu total"))
    elif dti <= 25 and revenu_total > 0:
        factors.append(("Protection", "Endettement maitrise", f"{dti:.1f}% du revenu total"))
    if ratio_credit >= 18:
        factors.append(("Risque", "Montant important", f"{ratio_credit:.1f} fois le revenu mensuel total"))
    if ratio_apport >= 20:
        factors.append(("Protection", "Apport personnel solide", f"{ratio_apport:.1f}% du montant demande"))
    elif montant > 0 and ratio_apport < 5:
        factors.append(("Risque", "Apport faible", f"{ratio_apport:.1f}% du montant demande"))
    if reste < 0:
        factors.append(("Risque", "Reste a vivre negatif", f"{reste:,.0f}".replace(",", " ")))
    elif reste > revenu_total * 0.35 and revenu_total > 0:
        factors.append(("Protection", "Reste a vivre confortable", f"{reste:,.0f}".replace(",", " ")))
    if epargne > revenu_total * 0.08 and revenu_total > 0:
        factors.append(("Protection", "Epargne reguliere", f"{epargne:,.0f}".replace(",", " ")))
    if solde > revenu_total:
        factors.append(("Protection", "Solde moyen favorable", f"{solde:,.0f}".replace(",", " ")))
    if anciennete >= 60:
        factors.append(("Protection", "Anciennete bancaire", f"{int(anciennete)} mois"))

    if not factors:
        factors.append(("Neutre", "Profil standard", "Aucun signal metier dominant"))
    return pd.DataFrame(factors, columns=["Effet", "Facteur", "Lecture"])


def gauge(pd_default: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_default * 100,
            number={"suffix": "%", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f6feb"},
                "steps": [
                    {"range": [0, 25], "color": "#dff3e6"},
                    {"range": [25, 45], "color": "#fff4cc"},
                    {"range": [45, 65], "color": "#ffe1cc"},
                    {"range": [65, 100], "color": "#ffd6d1"},
                ],
                "threshold": {"line": {"color": "#b42318", "width": 4}, "value": 50},
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=255,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7f3"),
    )
    return fig


def similar_clients(df: pd.DataFrame, payload: dict[str, Any], n: int = 6) -> pd.DataFrame:
    fields = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "charges_mensuelles"]
    work = with_credit_label(df)
    distance = pd.Series(0.0, index=work.index)
    for field in fields:
        values = pd.to_numeric(work[field], errors="coerce")
        scale = values.std() or 1
        distance += ((values - float(payload.get(field) or 0)).abs() / scale).fillna(5)
    work["distance"] = distance
    cols = ["age", "ville", "type_credit_affiche", "revenu_mensuel", "montant_credit", "decision"]
    return work.sort_values("distance").head(n)[cols]


def render_home(df: pd.DataFrame, metadata: dict[str, Any] | None) -> None:
    model_name = model_display_name(metadata.get("best_model", "-")) if metadata else "-"
    risk_rate = df["target_default"].mean()
    revenue = pd.to_numeric(df["revenu_mensuel"], errors="coerce").median()
    amount = pd.to_numeric(df["montant_credit"], errors="coerce").median()
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-grid">
            <div class="brand-lockup">
              <small>BMCE</small>
              <h1>Prediction Credit</h1>
            </div>
            <div class="hero-stats">
              <div class="hero-stat"><span>Modele</span><strong>{model_name}</strong></div>
              <div class="hero-stat"><span>Dossiers</span><strong>{fmt_int(len(df))}</strong></div>
              <div class="hero-stat"><span>Risque</span><strong>{risk_rate:.1%}</strong></div>
              <div class="hero-stat"><span>Seuil</span><strong>{metadata.get("threshold", .5):.2f}</strong></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Clients", fmt_int(len(df)))
    cols[1].metric("Taux risque", f"{risk_rate:.1%}")
    cols[2].metric("Revenu median", fmt_int(revenue))
    cols[3].metric("Montant median", fmt_int(amount))

    left, right = st.columns([1.25, 1])
    with left:
        counts = df["decision"].value_counts().reset_index()
        counts.columns = ["decision", "nombre"]
        fig = px.pie(
            counts,
            names="decision",
            values="nombre",
            color="decision",
            hole=0.48,
            title="Repartition des decisions",
        )
        fig = style_fig(fig, 350)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        comparison = load_model_comparison()
        if not comparison.empty:
            ranking = comparison.sort_values("roc_auc", ascending=True)
            fig = px.bar(
                ranking,
                x="roc_auc",
                y="model_label",
                color="model_label",
                orientation="h",
                text=ranking["roc_auc"].map(lambda v: f"{v:.3f}"),
                title="Classement ROC-AUC",
            )
            fig = style_fig(fig, 350)
            fig.update_xaxes(range=[0.74, 0.91], title="ROC-AUC")
            fig.update_yaxes(title="")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    a, b, c, d = st.columns(4)
    with a:
        mini_card("01", "Prediction", "")
    with b:
        mini_card("02", "Lot", "")
    with c:
        mini_card("03", "Portefeuille", "")
    with d:
        mini_card("04", "Suivi", "")


def render_prediction(df: pd.DataFrame, model, metadata: dict[str, Any] | None) -> None:
    page_title("Prediction", "Scoring d'une nouvelle demande", model_display_name(metadata.get("best_model", "-")) if metadata else None)
    if model is None or metadata is None:
        st.warning("Modele indisponible. Lancez `python train.py`.")
        return

    tab_one, tab_batch = st.tabs(["Demande unique", "Scoring par lot"])
    with tab_one:
        payload = make_payload(df)
        with st.form("prediction_form"):
            st.markdown("<div class='section-label'>Profil</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                payload["age"] = st.number_input("Age", 18, 90, int(default_value(df, "age", 35)))
                payload["sexe"] = st.selectbox("Sexe", options(df, "sexe"))
            with c2:
                payload["situation_familiale"] = st.selectbox("Situation", options(df, "situation_familiale"))
                payload["nombre_personnes_charge"] = st.number_input("Charges familiales", 0, 10, int(default_value(df, "nombre_personnes_charge", 0)))
            with c3:
                payload["niveau_etude"] = st.selectbox("Niveau", options(df, "niveau_etude"))
                payload["ville"] = st.selectbox("Ville", options(df, "ville"))
            with c4:
                payload["type_logement"] = st.selectbox("Logement", options(df, "type_logement"))
                payload["statut_professionnel"] = st.selectbox("Statut pro", options(df, "statut_professionnel"))

            st.markdown("<div class='section-label'>Revenus & comportement</div>", unsafe_allow_html=True)
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                payload["revenu_mensuel"] = st.number_input("Revenu", min_value=0.0, value=float(default_value(df, "revenu_mensuel", 8000)), step=500.0)
                payload["revenu_supplementaire"] = st.number_input("Revenu supp.", min_value=0.0, value=float(default_value(df, "revenu_supplementaire", 0)), step=250.0)
            with c6:
                payload["charges_mensuelles"] = st.number_input("Charges", min_value=0.0, value=float(default_value(df, "charges_mensuelles", 2500)), step=250.0)
                payload["epargne_mensuelle"] = st.number_input("Epargne", min_value=0.0, value=float(default_value(df, "epargne_mensuelle", 0)), step=250.0)
            with c7:
                payload["incidents_paiement_12m"] = st.number_input("Incidents 12m", 0, 20, int(default_value(df, "incidents_paiement_12m", 0)))
                payload["retards_paiement_12m"] = st.number_input("Retards 12m", 0, 20, int(default_value(df, "retards_paiement_12m", 0)))
            with c8:
                payload["credits_en_cours"] = st.number_input("Credits en cours", 0, 10, int(default_value(df, "credits_en_cours", 0)))
                payload["anciennete_bancaire_mois"] = st.number_input("Anc. bancaire", 0, 600, int(default_value(df, "anciennete_bancaire_mois", 24)))

            st.markdown("<div class='section-label'>Credit</div>", unsafe_allow_html=True)
            c9, c10, c11, c12 = st.columns(4)
            with c9:
                selected_credit = st.selectbox("Type credit", credit_options(df))
                payload["type_credit"] = credit_to_model(selected_credit)
                payload["objet_credit"] = st.selectbox("Objet", options(df, "objet_credit"))
            with c10:
                payload["montant_credit"] = st.number_input("Montant", min_value=0.0, value=float(default_value(df, "montant_credit", 100000)), step=5000.0)
                payload["duree_mois"] = st.number_input("Duree", 1, 360, int(default_value(df, "duree_mois", 48)))
            with c11:
                payload["taux_interet"] = st.number_input("Taux", 0.0, 30.0, float(default_value(df, "taux_interet", 7.0)), step=0.1)
                payload["apport_personnel"] = st.number_input("Apport", min_value=0.0, value=float(default_value(df, "apport_personnel", 0)), step=1000.0)
            with c12:
                payload["mensualites_existantes"] = st.number_input("Mensualites", min_value=0.0, value=float(default_value(df, "mensualites_existantes", 0)), step=250.0)
                payload["solde_moyen_compte"] = st.number_input("Solde moyen", value=float(default_value(df, "solde_moyen_compte", 0)), step=500.0)

            submitted = st.form_submit_button("Predire le risque", use_container_width=True)

        if submitted:
            payload = enrich_payload(payload)
            score_frame = prepare_prediction(payload, metadata["features"])
            pd_default = float(model.predict_proba(score_frame)[:, 1][0])
            decision = "Avis defavorable" if pd_default >= float(metadata.get("threshold", 0.5)) else "Avis favorable"
            result = {
                "pd_default": round(pd_default, 4),
                "score_credit": credit_score(pd_default),
                "segment_risque": segment(pd_default),
                "decision": decision,
                "type_credit": payload.get("type_credit"),
                "montant_credit": payload.get("montant_credit"),
                "revenu_mensuel": payload.get("revenu_mensuel"),
                "charges_mensuelles": payload.get("charges_mensuelles"),
                "incidents_paiement_12m": payload.get("incidents_paiement_12m"),
                "retards_paiement_12m": payload.get("retards_paiement_12m"),
                "apport_personnel": payload.get("apport_personnel"),
                "taux_endettement": payload.get("taux_endettement"),
            }
            st.session_state["prediction"] = {"payload": payload, "result": result}
            log_prediction(result)

        prediction = st.session_state.get("prediction")
        if prediction:
            result = prediction["result"]
            payload = prediction["payload"]
            left, right = st.columns([.95, 1.05])
            with left:
                st.plotly_chart(gauge(result["pd_default"]), use_container_width=True)
            with right:
                css = "result-ko" if result["decision"] == "Avis defavorable" else "result-ok"
                st.markdown(f"<div class='{css}'>{result['decision']}</div>", unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                k1.metric("Score", result["score_credit"])
                k2.metric("Segment", result["segment_risque"])
                k3.metric("PD", f"{result['pd_default']:.1%}")
                risk_signal_panel(payload)
                st.dataframe(explain_prediction(payload), use_container_width=True, hide_index=True)
                st.dataframe(similar_clients(df, payload), use_container_width=True, hide_index=True)
                export = pd.DataFrame([{**result, **{key: payload.get(key) for key in ["age", "ville", "duree_mois"]}}])
                st.download_button("Exporter", export.to_csv(index=False).encode("utf-8"), "prediction_credit.csv", "text/csv")

    with tab_batch:
        uploaded = st.file_uploader("Importer un CSV de demandes", type=["csv"])
        if uploaded:
            batch = pd.read_csv(uploaded)
            prepared = add_credit_features(basic_cleaning(batch)).drop(columns=["date_demande"], errors="ignore")
            for column in metadata["features"]:
                if column not in prepared.columns:
                    prepared[column] = pd.NA
            scores = model.predict_proba(prepared[metadata["features"]])[:, 1]
            scored = batch.copy()
            scored["pd_default"] = scores.round(4)
            scored["score_credit"] = [credit_score(float(value)) for value in scores]
            scored["segment_risque"] = [segment(float(value)) for value in scores]
            scored["decision_predite"] = ["Avis defavorable" if value >= metadata.get("threshold", .5) else "Avis favorable" for value in scores]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Demandes", len(scored))
            c2.metric("Defavorables", f"{(scored['decision_predite'] == 'Avis defavorable').mean():.1%}")
            c3.metric("PD moyenne", f"{scored['pd_default'].mean():.1%}")
            c4.metric("Score moyen", f"{scored['score_credit'].mean():.0f}")
            st.dataframe(scored.head(300), use_container_width=True, hide_index=True)
            st.download_button("Exporter le lot", scored.to_csv(index=False).encode("utf-8"), "scoring_batch.csv", "text/csv")


def filtered_clients(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        if st.session_state.get("page") == "Portefeuille":
            st.markdown("### Filtres")
            selected_villes = st.multiselect("Ville", options(df, "ville"))
            selected_types = st.multiselect("Type credit", credit_options(df))
            selected_decisions = st.multiselect("Decision", options(df, "decision"))
            ages = pd.to_numeric(df["age"], errors="coerce").dropna()
            amounts = pd.to_numeric(df["montant_credit"], errors="coerce").dropna()
            age_range = st.slider("Age", int(ages.min()), int(ages.max()), (int(ages.min()), int(ages.max())))
            amount_range = st.slider("Montant", int(amounts.min()), int(amounts.max()), (int(amounts.quantile(.05)), int(amounts.quantile(.95))), step=5000)
        else:
            selected_villes, selected_types, selected_decisions = [], [], []
            age_range, amount_range = (0, 200), (0, 10**9)
    work = df.copy()
    if selected_villes:
        work = work[work["ville"].astype(str).isin(selected_villes)]
    if selected_types:
        work = work[work["type_credit"].astype(str).isin([credit_to_model(value) for value in selected_types])]
    if selected_decisions:
        work = work[work["decision"].astype(str).isin(selected_decisions)]
    work = work[pd.to_numeric(work["age"], errors="coerce").between(*age_range)]
    work = work[pd.to_numeric(work["montant_credit"], errors="coerce").between(*amount_range)]
    return with_credit_label(work)


def render_portfolio(df: pd.DataFrame) -> None:
    page_title("Portefeuille", "Exploration des demandes clients", "CSV")
    data = filtered_clients(df)
    if data.empty:
        st.warning("Aucun dossier pour ces filtres.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dossiers", fmt_int(len(data)))
    c2.metric("Risque", f"{data['target_default'].mean():.1%}")
    c3.metric("Revenu median", fmt_int(pd.to_numeric(data["revenu_mensuel"], errors="coerce").median()))
    c4.metric("Montant median", fmt_int(pd.to_numeric(data["montant_credit"], errors="coerce").median()))
    sample = data.sample(min(len(data), 5000), random_state=42) if len(data) > 5000 else data
    left, right = st.columns(2)
    with left:
        fig = px.histogram(sample, x="revenu_mensuel", color="decision", nbins=35, title="Revenus")
        fig = style_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(sample, x="montant_credit", y="revenu_mensuel", color="decision", hover_data=["age", "ville", "type_credit_affiche"], title="Montant vs revenu")
        fig = style_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)
    cols = [col for col in ["id_demande", "age", "ville", "type_credit_affiche", "revenu_mensuel", "montant_credit", "incidents_paiement_12m", "retards_paiement_12m", "decision"] if col in data.columns]
    st.dataframe(data[cols].head(500), use_container_width=True, hide_index=True)
    st.download_button("Exporter", data.to_csv(index=False).encode("utf-8"), "clients_filtres.csv", "text/csv")


def render_monitoring(metadata: dict[str, Any] | None) -> None:
    page_title("Monitoring", "Performances, seuils et predictions", "ML")
    comparison = load_model_comparison()
    if not comparison.empty:
        best = comparison.iloc[0]
        runner_up = comparison.iloc[1] if len(comparison) > 1 else best
        gap = best["roc_auc"] - runner_up["roc_auc"]
        st.markdown(
            f"""
            <div class="model-ribbon">
              <small>Modele champion</small>
              <strong>{best['model_label']}</strong>
              <span>ROC-AUC {best['roc_auc']:.3f} / ecart +{gap:.3f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best["model_label"])
        c2.metric("ROC-AUC", f"{best['roc_auc']:.3f}")
        c3.metric("F1", f"{best['f1']:.3f}")
        c4.metric("Modeles testes", len(comparison))

        left, right = st.columns([1, 1.15])
        with left:
            ranking = comparison.sort_values("roc_auc", ascending=True)
            fig = px.bar(
                ranking,
                x="roc_auc",
                y="model_label",
                color="model_label",
                orientation="h",
                text=ranking["roc_auc"].map(lambda v: f"{v:.3f}"),
                title="Classement par ROC-AUC",
            )
            fig = style_fig(fig, 390)
            fig.update_xaxes(range=[0.74, 0.91], title="ROC-AUC")
            fig.update_yaxes(title="")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            long = comparison.melt(
                id_vars=["model_label"],
                value_vars=["precision", "recall", "f1"],
                var_name="metrique",
                value_name="score",
            )
            long["metrique"] = long["metrique"].map(METRIC_DISPLAY_NAMES)
            fig = px.bar(
                long,
                x="model_label",
                y="score",
                color="metrique",
                barmode="group",
                title="Precision, recall et F1",
            )
            fig = style_fig(fig, 390)
            fig.update_yaxes(range=[0.70, 0.90])
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True)

        table = comparison[
            ["rang", "model_label", "accuracy", "precision", "recall", "f1", "roc_auc", "gini", "ks", "ecart_roc_auc"]
        ].copy()
        table.columns = [
            "Rang",
            "Modele",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "Gini",
            "KS",
            "Ecart ROC-AUC",
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)
    if THRESHOLD_PATH.exists():
        thresholds = pd.read_csv(THRESHOLD_PATH)
        fig = px.line(thresholds, x="threshold", y=["precision", "recall", "f1"], markers=True, title="Analyse des seuils")
        fig = style_fig(fig, 360)
        fig.update_yaxes(range=[0.70, 0.91])
        st.plotly_chart(fig, use_container_width=True)
    if metadata and metadata.get("business_features_used"):
        st.caption("Variables metier: " + ", ".join(metadata["business_features_used"]))
    if LOG_PATH.exists():
        st.dataframe(pd.read_csv(LOG_PATH).tail(150), use_container_width=True, hide_index=True)
    if DASHBOARD_LOG_PATH.exists():
        st.dataframe(pd.read_csv(DASHBOARD_LOG_PATH).tail(150), use_container_width=True, hide_index=True)


def render_explainability(metadata: dict[str, Any] | None) -> None:
    page_title("Explicabilite", "Variables, architecture et lecture metier", "XAI")
    top, side = st.columns([1.15, .85])
    with top:
        if (FIGURE_DIR / "feature_importance.png").exists():
            st.image(str(FIGURE_DIR / "feature_importance.png"), caption="Importance globale des variables")
    with side:
        if (FIGURE_DIR / "shap_summary.png").exists():
            st.image(str(FIGURE_DIR / "shap_summary.png"), caption="Synthese SHAP globale")

    left, right = st.columns(2)
    with left:
        if ARCHITECTURE_PATH.exists():
            st.image(str(ARCHITECTURE_PATH), caption="Architecture fonctionnelle")
    with right:
        if metadata and metadata.get("business_features_used"):
            features = pd.DataFrame({"Variables metier": metadata["business_features_used"]})
            st.dataframe(features, use_container_width=True, hide_index=True)
        mini_card("XAI", "Lecture individuelle", "Chaque prediction affiche les signaux qui poussent le risque vers le haut ou vers le bas.")



def render_quality(df: pd.DataFrame) -> None:
    page_title("Qualite donnees", "Controle du fichier source", "DATA")
    missing = df.isna().mean().sort_values(ascending=False).head(18).reset_index()
    missing.columns = ["variable", "taux"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Colonnes", df.shape[1])
    c2.metric("Doublons", int(df.duplicated().sum()))
    c3.metric("Manquants moyens", f"{df.isna().mean().mean():.1%}")
    left, right = st.columns(2)
    with left:
        fig = px.bar(missing, x="taux", y="variable", orientation="h", title="Valeurs manquantes")
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)
    with right:
        fig = px.histogram(df, x="age", color="decision", nbins=30, title="Age par decision")
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)


inject_css()
clients = load_clients()
model, metadata = load_artifacts()

with st.sidebar:
    st.markdown("## BMCE Risk")
    st.caption("Credit decision intelligence")
    page = st.radio("Navigation", ["Accueil", "Prediction", "Portefeuille", "Monitoring", "Explicabilite", "Qualite donnees"], key="page")
    st.divider()
    st.caption("Modele actif")
    st.markdown(f"**{model_display_name(metadata.get('best_model', 'Non disponible')) if metadata else 'Non disponible'}**")

if page == "Accueil":
    render_home(clients, metadata)
elif page == "Prediction":
    render_prediction(clients, model, metadata)
elif page == "Portefeuille":
    render_portfolio(clients)
elif page == "Monitoring":
    render_monitoring(metadata)
elif page == "Explicabilite":
    render_explainability(metadata)
else:
    render_quality(clients)
