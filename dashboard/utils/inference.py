"""
inference.py
------------
XGBoost inference and SHAP computation for the dashboard.

Used exclusively by Page 3 (Day Explainer) to:
  1. Prepare a single row from master_dataset for model input
  2. Run XGBoost predict_proba to get congestion probability
  3. Compute SHAP values for that specific row
  4. Return structured results for dashboard rendering

Design principles:
  - Reproduces EXACTLY the same feature preparation as train_classifier.py
  - Uses feature_columns.json to enforce column order
  - Uses saved encoders (never re-fits anything at inference time)
  - Returns typed dataclasses for clean dashboard consumption
  - All functions are pure (no side effects, no st.* calls)

SHAP note:
  SHAP values computed here explain model behaviour for one specific
  port-day combination. They show which features pushed the prediction
  toward congested or not congested FOR THIS ROW ONLY.
  They do NOT imply causality in the real world.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Leakage columns — must match train_classifier.py exactly
# These are dropped before inference, same as during training
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = [
    "congestion_label",
    "port_congestion_score",
    "port_congestion_score_norm",
    "congestion_pressure_index",
    "weather_congestion_interaction",
    "vessels_7d_rolling_norm",
    "nearest_port",
    "date",
    "season",
]


# ---------------------------------------------------------------------------
# Result dataclasses — typed return values for dashboard pages
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Result of running XGBoost inference on one port-day row.

    Attributes
    ----------
    port : str
        Port name.
    date : str
        Date string (YYYY-MM-DD).
    probability : float
        XGBoost predicted probability of congestion (0–1).
    label : int
        Binary prediction (1=congested, 0=not congested).
    label_str : str
        Human-readable label string.
    risk_level : str
        High / Moderate / Low based on probability thresholds.
    actual_label : Optional[int]
        Ground truth label from dataset if available.
    actual_score : float
        Raw port_congestion_score from dataset.
    calibration_caveat : str
        Honest note about XGBoost probability calibration.
    """
    port:               str
    date:               str
    probability:        float
    label:              int
    label_str:          str
    risk_level:         str
    actual_label:       Optional[int]
    actual_score:       float
    calibration_caveat: str


@dataclass
class SHAPResult:
    """
    SHAP explanation for one prediction.

    Attributes
    ----------
    feature_names : list of str
        Feature names sorted by absolute SHAP value (descending).
    shap_values : np.ndarray
        SHAP values aligned with feature_names.
    feature_values : np.ndarray
        Actual feature values for this row, aligned with feature_names.
    base_value : float
        SHAP base value (expected model output).
    top_n : int
        Number of top features shown (default 12).
    causality_note : str
        Mandatory note that SHAP ≠ causality.
    """
    feature_names:  list
    shap_values:    np.ndarray
    feature_values: np.ndarray
    base_value:     float
    top_n:          int
    causality_note: str


# ---------------------------------------------------------------------------
# Feature preparation — must match train_classifier.py exactly
# ---------------------------------------------------------------------------

def prepare_row_for_inference(
    row: pd.Series,
    feature_columns: list,
    port_encoder,
    season_encoder,
) -> pd.DataFrame:
    """
    Prepare a single dataset row for XGBoost inference.

    Applies the exact same transformations as train_classifier.py:
      1. Encode nearest_port → port_encoded
      2. Encode season → season_encoded
      3. Drop leakage columns
      4. Select and order columns per feature_columns.json
      5. Fill any nulls with a safe default (0 for most features)

    Parameters
    ----------
    row : pd.Series
        One row from master_dataset DataFrame.
    feature_columns : list
        Ordered list of feature column names from feature_columns.json.
    port_encoder : LabelEncoder
        Fitted port name → integer encoder from training.
    season_encoder : LabelEncoder
        Fitted season string → integer encoder from training.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame ready for model.predict_proba().
    """
    # Convert Series to single-row DataFrame for consistent manipulation
    df = pd.DataFrame([row])

    # ── Encode categorical columns ────────────────────────────────────────────
    if "nearest_port" in df.columns:
        port_val = df["nearest_port"].iloc[0]
        if port_val in port_encoder.classes_:
            df["port_encoded"] = port_encoder.transform([port_val])[0]
        else:
            # Unseen port — use -1 as sentinel (same as LOPO test)
            df["port_encoded"] = -1

    if "season" in df.columns:
        season_val = df["season"].iloc[0]
        if season_val in season_encoder.classes_:
            df["season_encoded"] = season_encoder.transform([season_val])[0]
        else:
            df["season_encoded"] = 0

    # ── Drop leakage and admin columns ───────────────────────────────────────
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ── Keep numeric columns only ─────────────────────────────────────────────
    df = df.select_dtypes(include="number")

    # ── Enforce exact column order from training ──────────────────────────────
    # Add any missing columns with 0 (handles schema evolution between retrains)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # ── Fill nulls ────────────────────────────────────────────────────────────
    df = df.fillna(0)

    return df


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def run_prediction(
    row: pd.Series,
    feature_columns: list,
    port_encoder,
    season_encoder,
    xgb_model,
) -> PredictionResult:
    """
    Run XGBoost inference on one port-day row and return a PredictionResult.

    Probability thresholds for risk level:
      probability > 0.70 → High Risk
      probability > 0.40 → Moderate
      probability ≤ 0.40 → Low Risk

    Calibration caveat is always included in the result because
    XGBoost probabilities are known to be overconfident (pushed
    toward 0 and 1). The displayed probability should be treated
    as a relative risk indicator, not a precise probability estimate.

    Parameters
    ----------
    row : pd.Series
        One row from master_dataset.
    feature_columns : list
        Ordered feature column list from feature_columns.json.
    port_encoder : LabelEncoder
        Fitted port encoder.
    season_encoder : LabelEncoder
        Fitted season encoder.
    xgb_model : XGBClassifier
        Trained XGBoost classifier.

    Returns
    -------
    PredictionResult
    """
    X = prepare_row_for_inference(row, feature_columns, port_encoder, season_encoder)

    # Get probability of positive class (congested = 1)
    proba = float(xgb_model.predict_proba(X)[0, 1])
    label = int(xgb_model.predict(X)[0])

    # Risk level from probability
    if proba > 0.70:
        risk_level = "🔴 High Risk"
    elif proba > 0.40:
        risk_level = "🟡 Moderate Risk"
    else:
        risk_level = "🟢 Low Risk"

    label_str = "⚠️ Congested" if label == 1 else "✅ Not Congested"

    # Actual ground truth from dataset if available
    actual_label = int(row["congestion_label"]) if "congestion_label" in row.index else None
    actual_score = float(row.get("port_congestion_score", 0.0))

    calibration_caveat = (
        "XGBoost probabilities tend to be overconfident — "
        "values are pushed toward 0 and 1. "
        "Treat this as a relative risk indicator, not a precise probability. "
        "Platt scaling would improve calibration but requires ≥500 samples."
    )

    return PredictionResult(
        port               = str(row.get("nearest_port", "Unknown")),
        date               = str(row.get("date", "Unknown"))[:10],
        probability        = round(proba, 4),
        label              = label,
        label_str          = label_str,
        risk_level         = risk_level,
        actual_label       = actual_label,
        actual_score       = round(actual_score, 4),
        calibration_caveat = calibration_caveat,
    )


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_for_row(
    row: pd.Series,
    feature_columns: list,
    port_encoder,
    season_encoder,
    shap_explainer,
    top_n: int = 12,
) -> SHAPResult:
    """
    Compute SHAP values for one port-day row and return a SHAPResult.

    SHAP values are computed using the pre-fitted TreeExplainer loaded
    from models/shap_explainer.joblib. The explainer was fitted on the
    full training dataset — computing values for a new row is fast
    (milliseconds) because the tree structure is pre-computed.

    CRITICAL: SHAP values explain model behaviour, not real-world causality.
    A positive SHAP value for anchored_ratio means the model used this
    feature to push the prediction toward congested. It does NOT mean
    that anchored vessels caused the congestion — correlation in the
    training data drives SHAP values, not causal mechanisms.

    Parameters
    ----------
    row : pd.Series
        One row from master_dataset.
    feature_columns : list
        Ordered feature column list.
    port_encoder, season_encoder : LabelEncoder
        Fitted encoders.
    shap_explainer : shap.TreeExplainer
        Pre-fitted SHAP TreeExplainer.
    top_n : int
        Number of top features to return (sorted by |SHAP|).

    Returns
    -------
    SHAPResult
    """
    X = prepare_row_for_inference(row, feature_columns, port_encoder, season_encoder)

    # Compute SHAP values — returns array of shape (1, n_features)
    shap_vals = shap_explainer.shap_values(X)

    # Handle both 1D and 2D SHAP output shapes
    if hasattr(shap_vals, "__len__") and len(shap_vals) == 2:
        # Some SHAP versions return [negative_class, positive_class]
        sv = shap_vals[1][0] if len(shap_vals[1].shape) > 1 else shap_vals[1]
    else:
        sv = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals

    sv = np.array(sv)

    # Get base value (expected model output across training set)
    base_value = float(shap_explainer.expected_value)
    if isinstance(shap_explainer.expected_value, (list, np.ndarray)):
        base_value = float(shap_explainer.expected_value[1])

    # Sort features by absolute SHAP value, take top_n
    all_feature_names = X.columns.tolist()
    feature_values    = X.iloc[0].values

    sorted_idx     = np.argsort(np.abs(sv))[::-1][:top_n]
    top_features   = [all_feature_names[i] for i in sorted_idx]
    top_shap_vals  = sv[sorted_idx]
    top_feat_vals  = feature_values[sorted_idx]

    causality_note = (
        "SHAP values explain model behaviour — which features the model "
        "relied on for this prediction. They do NOT imply real-world causality. "
        "High SHAP for 'anchored_ratio' means the model used this feature "
        "heavily for this specific day — not that anchored vessels caused congestion."
    )

    return SHAPResult(
        feature_names  = top_features,
        shap_values    = top_shap_vals,
        feature_values = top_feat_vals,
        base_value     = round(base_value, 4),
        top_n          = top_n,
        causality_note = causality_note,
    )


# ---------------------------------------------------------------------------
# SHAP waterfall chart builder
# ---------------------------------------------------------------------------

def build_shap_waterfall_figure(shap_result: SHAPResult, title: str = ""):
    """
    Build a matplotlib figure for the SHAP waterfall chart.

    Called by Page 3 (Day Explainer) to render live SHAP explanations
    for the user-selected port and date.

    Design:
      - Horizontal bar chart (features on y-axis, SHAP values on x-axis)
      - Red bars = positive SHAP (pushed toward congested)
      - Blue bars = negative SHAP (pushed toward not congested)
      - Feature value shown in label (e.g. "anchored_ratio = 0.923")
      - Vertical line at zero
      - Causality note in subtitle

    Parameters
    ----------
    shap_result : SHAPResult
        Output from compute_shap_for_row().
    title : str
        Optional chart title (port + date).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv   = shap_result.shap_values
    fn   = shap_result.feature_names
    fv   = shap_result.feature_values
    n    = len(sv)

    # Reverse order for horizontal bar (most important at top)
    sv_plot = sv[::-1]
    fn_plot = fn[::-1]
    fv_plot = fv[::-1]

    colors = ["#F44336" if v > 0 else "#2196F3" for v in sv_plot]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.5)))

    bars = ax.barh(
        range(n),
        sv_plot,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    # Y-axis labels: "feature_name = value"
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{f} = {v:.4f}" for f, v in zip(fn_plot, fv_plot)],
        fontsize=9,
    )

    # Zero reference line
    ax.axvline(0, color="white" if True else "black", lw=0.8, alpha=0.5)

    # Value labels on bars
    for i, (val, bar) in enumerate(zip(sv_plot, bars)):
        x_pos  = val + (0.003 if val >= 0 else -0.003)
        ha     = "left" if val >= 0 else "right"
        ax.text(x_pos, i, f"{val:+.3f}", va="center", ha=ha,
                fontsize=8, color="white", alpha=0.9)

    ax.set_xlabel("SHAP value  (← Not Congested  |  Congested →)", fontsize=9)
    ax.set_title(
        f"{title}\n"
        f"Red = pushed toward congested  |  Blue = pushed toward not congested\n"
        f"Base value: {shap_result.base_value:.4f}  "
        f"(model's average prediction across training data)",
        fontsize=9,
        pad=10,
    )

    # Causality note as figure text
    fig.text(
        0.01, -0.03,
        f"⚠ {shap_result.causality_note}",
        fontsize=7,
        style="italic",
        wrap=True,
        color="gray",
    )

    ax.grid(axis="x", alpha=0.2, linestyle="--")
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Utility: get one row from master dataset by port + date
# ---------------------------------------------------------------------------

def get_row_by_port_date(
    df: pd.DataFrame,
    port: str,
    date: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Retrieve a single row from master_dataset for a given port and date.

    Parameters
    ----------
    df : pd.DataFrame
        master_dataset DataFrame.
    port : str
        Port name.
    date : pd.Timestamp
        Date to look up.

    Returns
    -------
    pd.Series or None
        The matching row, or None if not found.
    """
    mask = (df["nearest_port"] == port) & (df["date"].dt.date == date.date())
    result = df[mask]

    if len(result) == 0:
        return None

    return result.iloc[0]
