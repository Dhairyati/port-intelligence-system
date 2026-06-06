"""
data_loader.py
--------------
Centralised cached data and artifact loader for the dashboard.

Every dashboard page imports from here — never directly from disk.
This means:
  - Models load once at startup, shared across all sessions
  - DataFrames cache by content hash, reload only when files change
  - File paths are defined in one place — change here, all pages update
  - @st.cache_resource for models (shared across users, never reloaded)
  - @st.cache_data for DataFrames (cached per content, safe to reload)

Usage in any dashboard page:
    from dashboard.utils.data_loader import (
        load_master_dataset,
        load_port_features,
        load_xgb_classifier,
        load_shap_explainer,
        ...
    )
"""

import json
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root — works regardless of where Streamlit is launched from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Path registry — all file paths in one place
# ---------------------------------------------------------------------------
PATHS = {
    # Processed data
    "master_dataset":     PROJECT_ROOT / "data/processed/master_dataset.parquet",
    "port_features":      PROJECT_ROOT / "data/processed/port_features.parquet",
    "external_features":  PROJECT_ROOT / "data/processed/external_features.parquet",

    # Model artifacts
    "xgb_classifier":     PROJECT_ROOT / "models/xgb_classifier.joblib",
    "baseline_lr":        PROJECT_ROOT / "models/baseline_lr.joblib",
    "shap_explainer":     PROJECT_ROOT / "models/shap_explainer.joblib",
    "label_encoder":      PROJECT_ROOT / "models/label_encoder.joblib",
    "season_encoder":     PROJECT_ROOT / "models/season_encoder.joblib",
    "feature_columns":    PROJECT_ROOT / "models/feature_columns.json",
    "feature_dtypes":     PROJECT_ROOT / "models/feature_dtypes.json",
    "training_metadata":  PROJECT_ROOT / "models/training_metadata.json",

    # Evaluation outputs
    "per_port_metrics":   PROJECT_ROOT / "outputs/per_port_metrics.csv",
    "lopo_results":       PROJECT_ROOT / "outputs/lopo_results.csv",
    "evaluation_report":  PROJECT_ROOT / "outputs/evaluation_report.png",
}

# Prophet model paths are dynamic — one per port
def get_prophet_path(port: str) -> Path:
    slug = port.lower().replace(" ", "_")
    return PROJECT_ROOT / f"models/prophet_{slug}.pkl"

# SHAP waterfall paths are dynamic — one per port
def get_shap_waterfall_path(port: str) -> Path:
    slug = port.lower().replace(" ", "_")
    return PROJECT_ROOT / f"outputs/shap_waterfall_{slug}.png"

# Forecast chart paths are dynamic — one per port
def get_forecast_path(port: str) -> Path:
    slug = port.lower().replace(" ", "_")
    return PROJECT_ROOT / f"outputs/forecast_{slug}.png"


# ---------------------------------------------------------------------------
# Data loaders — @st.cache_data
# Cached by content hash. Safe to call multiple times — returns cached
# copy after first load. Automatically invalidates if file changes.
# ---------------------------------------------------------------------------

@st.cache_data
def load_master_dataset() -> pd.DataFrame:
    """
    Load master_dataset.parquet — primary data source for the dashboard.
    One row per port per day, 450 rows, 41 columns.
    Contains congestion labels, scores, weather features, and calendar features.
    """
    path = PATHS["master_dataset"]
    _assert_exists(path, "master_dataset.parquet")

    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "nearest_port"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data
def load_port_features() -> pd.DataFrame:
    """
    Load port_features.parquet — daily port-level congestion signals.
    Used by Page 2 (Port Deep Dive) for the 90-day history chart.
    """
    path = PATHS["port_features"]
    _assert_exists(path, "port_features.parquet")

    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["nearest_port", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data
def load_external_features() -> pd.DataFrame:
    """
    Load external_features.parquet — weather and calendar features.
    Used by Page 2 for weather severity overlay on history chart.
    """
    path = PATHS["external_features"]
    _assert_exists(path, "external_features.parquet")

    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_per_port_metrics() -> pd.DataFrame:
    """Load per_port_metrics.csv — evaluation metrics per port."""
    path = PATHS["per_port_metrics"]
    _assert_exists(path, "per_port_metrics.csv")
    return pd.read_csv(path)


@st.cache_data
def load_lopo_results() -> pd.DataFrame:
    """Load lopo_results.csv — leave-one-port-out AUC results."""
    path = PATHS["lopo_results"]
    _assert_exists(path, "lopo_results.csv")
    return pd.read_csv(path)


@st.cache_data
def load_feature_columns() -> list:
    """Load ordered feature column list used during training."""
    path = PATHS["feature_columns"]
    _assert_exists(path, "feature_columns.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_dtypes() -> dict:
    """Load feature dtype mapping used during training."""
    path = PATHS["feature_dtypes"]
    _assert_exists(path, "feature_dtypes.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_training_metadata() -> dict:
    """Load training metadata — split dates, params, notes."""
    path = PATHS["training_metadata"]
    _assert_exists(path, "training_metadata.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model loaders — @st.cache_resource
# Loaded once per app session, shared across all users.
# Never re-serialised — safe for sklearn/XGBoost/SHAP objects.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_xgb_classifier():
    """
    Load trained XGBoost classifier.
    384 KB — loaded once, shared across all dashboard pages and users.
    """
    path = PATHS["xgb_classifier"]
    _assert_exists(path, "xgb_classifier.joblib")
    return joblib.load(path)


@st.cache_resource
def load_baseline_lr():
    """Load Logistic Regression baseline (Pipeline with StandardScaler)."""
    path = PATHS["baseline_lr"]
    _assert_exists(path, "baseline_lr.joblib")
    return joblib.load(path)


@st.cache_resource
def load_shap_explainer():
    """
    Load SHAP TreeExplainer.
    384 KB — most expensive artifact to load. Cached permanently.
    """
    path = PATHS["shap_explainer"]
    _assert_exists(path, "shap_explainer.joblib")
    return joblib.load(path)


@st.cache_resource
def load_label_encoder():
    """Load port name → integer LabelEncoder."""
    path = PATHS["label_encoder"]
    _assert_exists(path, "label_encoder.joblib")
    return joblib.load(path)


@st.cache_resource
def load_season_encoder():
    """Load season string → integer LabelEncoder."""
    path = PATHS["season_encoder"]
    _assert_exists(path, "season_encoder.joblib")
    return joblib.load(path)


@st.cache_resource
def load_prophet_model(port: str):
    """
    Load a fitted Prophet model for a given port.

    Parameters
    ----------
    port : str
        Port name as stored in the dataset (e.g. 'Los Angeles').

    Returns
    -------
    Prophet model or None if file not found.
    """
    path = get_prophet_path(port)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Computed / derived data — not cached (fast to compute)
# ---------------------------------------------------------------------------

def get_available_ports(df: pd.DataFrame) -> list:
    """Return sorted list of port names from the dataset."""
    return sorted(df["nearest_port"].unique().tolist())


def get_latest_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the most recent congestion score and label per port.
    Used by Page 1 (Port Risk Map) for current risk display.

    Returns
    -------
    pd.DataFrame
        One row per port with columns:
        nearest_port, date, port_congestion_score,
        congestion_label, slow_vessel_ratio, avg_speed_in_zone
    """
    cols = [
        "nearest_port", "date",
        "port_congestion_score", "congestion_label",
        "slow_vessel_ratio", "avg_speed_in_zone",
        "vessels_in_zone", "max_weather_severity_score",
    ]
    available_cols = [c for c in cols if c in df.columns]

    latest = (
        df[available_cols]
        .sort_values("date")
        .groupby("nearest_port")
        .last()
        .reset_index()
    )
    return latest


def get_trend_direction(df: pd.DataFrame, port: str) -> str:
    """
    Compute trend direction for a port by comparing the mean
    congestion score of the last 7 days vs the previous 7 days.

    Returns
    -------
    str
        '↑ Worsening', '↓ Improving', or '→ Stable'
    """
    port_df = df[df["nearest_port"] == port].sort_values("date")

    if len(port_df) < 14:
        return "→ Stable"

    recent   = port_df.tail(7)["port_congestion_score"].mean()
    previous = port_df.iloc[-14:-7]["port_congestion_score"].mean()

    delta = recent - previous

    if delta > 0.02:
        return "↑ Worsening"
    elif delta < -0.02:
        return "↓ Improving"
    else:
        return "→ Stable"


def get_trend_arrow_color(trend: str) -> str:
    """Return hex color for a trend direction string."""
    if "Worsening" in trend:
        return "#F44336"   # red
    elif "Improving" in trend:
        return "#4CAF50"   # green
    else:
        return "#FF9800"   # amber


def get_risk_color(score: float) -> str:
    """
    Map congestion score to a risk colour for map markers and badges.

    Thresholds:
      score > 0.70 → Red    (high risk)
      score > 0.40 → Amber  (moderate)
      score ≤ 0.40 → Green  (low risk)
    """
    if score > 0.70:
        return "#F44336"   # red
    elif score > 0.40:
        return "#FF9800"   # amber
    else:
        return "#4CAF50"   # green


def get_risk_label(score: float) -> str:
    """Map congestion score to a human-readable risk label."""
    if score > 0.70:
        return "🔴 High Risk"
    elif score > 0.40:
        return "🟡 Moderate"
    else:
        return "🟢 Low Risk"


def generate_auto_insights(df: pd.DataFrame) -> list:
    """
    Generate auto-derived operational insights from the dataset.
    Displayed in the sidebar of app.py.

    Returns
    -------
    list of str
        Each string is one insight sentence.
    """
    insights = []

    try:
        # Highest congestion port
        avg_scores = df.groupby("nearest_port")["port_congestion_score"].mean()
        highest_port = avg_scores.idxmax()
        highest_score = avg_scores.max()
        insights.append(
            f"**{highest_port}** has the highest mean congestion "
            f"({highest_score:.3f}) across Jan–Mar 2023."
        )

        # New York anomaly — hardcoded from LOPO findings
        insights.append(
            "**New York** exhibits anomalous congestion behaviour "
            "relative to other ports (LOPO AUC=0.50 — Atlantic corridor outlier)."
        )

        # Weather influence
        if "max_weather_severity_score" in df.columns:
            weather_corr = (
                df.groupby("nearest_port")
                .apply(lambda x: x["max_weather_severity_score"]
                       .corr(x["port_congestion_score"]))
                .sort_values(ascending=False)
            )
            top_weather_port = weather_corr.idxmax()
            insights.append(
                f"Weather influence on congestion is strongest at "
                f"**{top_weather_port}** (highest weather-score correlation)."
            )

        # Trend direction
        falling_ports = []
        for port in df["nearest_port"].unique():
            if get_trend_direction(df, port) == "↓ Improving":
                falling_ports.append(port)
        if falling_ports:
            insights.append(
                f"Congestion trending **↓ Improving** at: "
                f"{', '.join(sorted(falling_ports))}."
            )

        # Congestion persistence — port with most consecutive congested days
        max_streak = 0
        streak_port = ""
        for port in df["nearest_port"].unique():
            port_labels = df[df["nearest_port"] == port].sort_values("date")["congestion_label"]
            streak = 0
            best   = 0
            for label in port_labels:
                streak = streak + 1 if label == 1 else 0
                best = max(best, streak)
            if best > max_streak:
                max_streak = best
                streak_port = port

        if streak_port:
            insights.append(
                f"**{streak_port}** shows the highest congestion persistence "
                f"({max_streak} consecutive congested days)."
            )

    except Exception:
        # Insights are non-critical — silently skip on any data issue
        pass

    return insights


# ---------------------------------------------------------------------------
# Port coordinate registry
# Used by Page 1 for map marker placement.
# Must match config.yaml exactly.
# ---------------------------------------------------------------------------
PORT_COORDINATES = {
    "Los Angeles": {"lat": 33.7361, "lon": -118.2922},
    "Houston":     {"lat": 29.7355, "lon": -95.0146},
    "Savannah":    {"lat": 32.0835, "lon": -81.0998},
    "Seattle":     {"lat": 47.6062, "lon": -122.3321},
    "New York":    {"lat": 40.6840, "lon": -74.0440},
}

# Ports flagged from LOPO evaluation — shown with caution banners
HIGH_CAUTION_PORTS = {"New York"}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _assert_exists(path: Path, label: str) -> None:
    """Raise a clear Streamlit error if a required file is missing."""
    if not path.exists():
        st.error(
            f"**Required file not found:** `{path}`\n\n"
            f"Make sure `{label}` exists before running the dashboard.\n\n"
            f"Run the Phase 2 pipeline (Colab) and Phase 3 training scripts first."
        )
        st.stop()
