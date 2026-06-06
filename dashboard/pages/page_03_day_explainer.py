"""
page_03_day_explainer.py
------------------------
Page 3 — Day Explainer

Allows users to select any port and date and understand the model's
congestion prediction and the reasoning behind it.

Structure:
  - Controls: port selector, date slider, SHAP feature count
  - Prediction Card: probability, label, ground truth, calibration note
  - Prediction Summary: plain-English narrative from top SHAP drivers
  - SHAP Waterfall: live feature attribution chart
  - Top Drivers: 3-item ranked card for non-technical viewers
  - Top 10 Influential Features: compact table sorted by |SHAP|
  - Model Interpretation: 2-sentence auto-generated narrative
  - Historical Context: selected date highlighted on 90-day chart

SHAP note: values explain model behaviour, not real-world causality.

All colors from theme.T — no hex values defined here.
Called via render() from app.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.data_loader import (
    HIGH_CAUTION_PORTS,
    get_shap_waterfall_path,
    load_feature_columns,
    load_label_encoder,
    load_master_dataset,
    load_season_encoder,
    load_shap_explainer,
    load_training_metadata,
    load_xgb_classifier,
)
from dashboard.utils.inference import (
    build_shap_waterfall_figure,
    compute_shap_for_row,
    get_row_by_port_date,
    run_prediction,
)
from dashboard.utils.theme import (
    T,
    apply_matplotlib_theme,
    banner_info,
    banner_note,
    banner_warn,
    card,
    plotly_layout,
    risk_color,
    risk_label,
    section_title,
)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render():
    """Main render function called from app.py."""

    # ── Load artifacts ────────────────────────────────────────────────────────
    try:
        df              = load_master_dataset()
        xgb_model       = load_xgb_classifier()
        shap_explainer  = load_shap_explainer()
        port_encoder    = load_label_encoder()
        season_encoder  = load_season_encoder()
        feature_columns = load_feature_columns()
        metadata        = load_training_metadata()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return

    ports = sorted(df["nearest_port"].unique().tolist())

    # ── How-it-works banner ───────────────────────────────────────────────────
    st.markdown(
        banner_info(
            "<b>🔍 How this page works</b><br>"
            "<span style='font-size:12px;'>"
            "Select a port and date to see the XGBoost model's prediction and "
            "a SHAP explanation of which features drove it. "
            "<b>SHAP explains model behaviour — not real-world causality.</b> "
            "High SHAP for a feature means the model relied on it heavily for "
            "this prediction, not that it caused congestion in the real world."
            "</span>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Controls row ──────────────────────────────────────────────────────────
    ctrl_port, ctrl_date, ctrl_shap = st.columns([1, 1, 2])

    with ctrl_port:
        selected_port = st.selectbox(
            "Port",
            options = ports,
            index   = 0,
            key     = "explainer_port",
        )

    with ctrl_date:
        port_dates = sorted(
            df[df["nearest_port"] == selected_port]["date"].dt.date.unique()
        )

        if not port_dates:
            st.error(f"No dates found for {selected_port}.")
            return

        selected_date = st.select_slider(
            "Date",
            options     = port_dates,
            value       = port_dates[-1],
            key         = "explainer_date",
            format_func = lambda d: d.strftime("%b %d, %Y"),
        )

    with ctrl_shap:
        top_n = st.slider(
            "Number of features to show in SHAP chart",
            min_value = 5,
            max_value = 20,
            value     = 12,
            step      = 1,
            key       = "shap_top_n",
            help      = "Shows the N most influential features for this prediction",
        )

    # ── New York caution ──────────────────────────────────────────────────────
    if selected_port in HIGH_CAUTION_PORTS:
        st.markdown(
            banner_warn(
                "<b>⚠ New York — Reduced Reliability</b><br>"
                "<span style='font-size:12px;'>"
                "LOPO evaluation showed this port has anomalous congestion dynamics "
                "(LOPO AUC = 0.50). Predictions for this port are less reliable "
                "than for other ports."
                "</span>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:14px 0 16px 0;">',
        unsafe_allow_html=True,
    )

    # ── Get row ───────────────────────────────────────────────────────────────
    row = get_row_by_port_date(df, selected_port, pd.Timestamp(selected_date))

    if row is None:
        st.warning(
            f"No data found for **{selected_port}** on **{selected_date}**.\n\n"
            "This date may not be in the dataset. Try selecting a different date."
        )
        return

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        prediction = run_prediction(
            row             = row,
            feature_columns = feature_columns,
            port_encoder    = port_encoder,
            season_encoder  = season_encoder,
            xgb_model       = xgb_model,
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # ── Run SHAP ──────────────────────────────────────────────────────────────
    try:
        shap_result = compute_shap_for_row(
            row             = row,
            feature_columns = feature_columns,
            port_encoder    = port_encoder,
            season_encoder  = season_encoder,
            shap_explainer  = shap_explainer,
            top_n           = top_n,
        )
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        shap_result = None

    # ── Prediction card + SHAP chart ──────────────────────────────────────────
    pred_col, shap_col = st.columns([1, 2], gap="large")

    with pred_col:
        _render_prediction_card(prediction, row)

    with shap_col:
        _render_shap_chart(shap_result, prediction)

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:18px 0 14px 0;">',
        unsafe_allow_html=True,
    )

    # ── Prediction Summary ────────────────────────────────────────────────────
    if shap_result is not None:
        _render_prediction_summary(prediction, shap_result)

        st.markdown(
            f'<hr style="border-color:{T["border"]};margin:18px 0 14px 0;">',
            unsafe_allow_html=True,
        )

    # ── Top Drivers + Top 10 feature table ───────────────────────────────────
    if shap_result is not None:
        drivers_col, table_col = st.columns([1, 2], gap="large")

        with drivers_col:
            _render_top_drivers(shap_result)

        with table_col:
            _render_top_features_table(row, feature_columns, shap_result)

        st.markdown(
            f'<hr style="border-color:{T["border"]};margin:18px 0 14px 0;">',
            unsafe_allow_html=True,
        )

        # ── Model Interpretation ──────────────────────────────────────────────
        _render_model_interpretation(shap_result, prediction)

        st.markdown(
            f'<hr style="border-color:{T["border"]};margin:18px 0 14px 0;">',
            unsafe_allow_html=True,
        )

    # ── Historical context ────────────────────────────────────────────────────
    _render_historical_context(df, selected_port, selected_date, prediction)


# ---------------------------------------------------------------------------
# Prediction card
# ---------------------------------------------------------------------------

def _render_prediction_card(prediction, row: pd.Series) -> None:
    """
    Render the prediction result card showing:
    - Port and date
    - Prediction label and probability
    - Risk level badge
    - Comparison with ground truth
    - Calibration caveat
    """
    rc = risk_color(prediction.probability)

    is_correct = (
        prediction.actual_label is not None
        and prediction.label == prediction.actual_label
    )
    correct_str      = "✅ Correct" if is_correct else "❌ Incorrect"
    actual_label_str = (
        "⚠️ Congested" if prediction.actual_label == 1 else "✅ Not Congested"
    )

    # Header
    st.markdown(
        section_title(f"🎯 Prediction — {prediction.port}"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:12px;color:{T["text_secondary"]};'
        f'margin-bottom:10px;">{prediction.date}</div>',
        unsafe_allow_html=True,
    )

    # Probability + label block
    st.markdown(
        f'<div style="background:{T["bg_card"]};border:1px solid {T["border"]};'
        f'border-radius:10px;padding:16px 14px;text-align:center;margin-bottom:8px;">'
        f'<div style="font-size:11px;color:{T["text_secondary"]};'
        f'letter-spacing:0.4px;margin-bottom:6px;">PREDICTED CONGESTION PROBABILITY</div>'
        f'<div style="font-size:36px;font-weight:800;color:{rc};'
        f'line-height:1.1;">{prediction.probability:.1%}</div>'
        f'<div style="font-size:15px;font-weight:600;color:{T["text_primary"]};'
        f'margin-top:8px;">{prediction.label_str}</div>'
        f'<div style="font-size:13px;color:{rc};margin-top:4px;">'
        f'{prediction.risk_level}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Ground truth block
    st.markdown(
        f'<div style="background:{T["bg_input"]};border:1px solid {T["border_light"]};'
        f'border-radius:8px;padding:12px 14px;margin-bottom:8px;">'
        f'<div style="font-size:10px;color:{T["text_secondary"]};'
        f'letter-spacing:0.5px;margin-bottom:6px;">ACTUAL (GROUND TRUTH)</div>'
        f'<div style="font-size:14px;color:{T["text_primary"]};'
        f'font-weight:600;margin-bottom:4px;">{actual_label_str}</div>'
        f'<div style="font-size:13px;color:{T["text_secondary"]};'
        f'margin-bottom:8px;">{correct_str}</div>'
        f'<div style="font-size:10px;color:{T["text_muted"]};'
        f'letter-spacing:0.4px;margin-bottom:3px;">RAW CONGESTION SCORE</div>'
        f'<div style="font-size:15px;color:{T["accent"]};font-weight:600;">'
        f'{prediction.actual_score:.4f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Calibration note
    with st.expander("⚠️ About this probability", expanded=False):
        st.markdown(
            banner_note(
                f"{prediction.calibration_caveat}<br><br>"
                "The calibration curve shows that XGBoost probabilities are pushed "
                "toward 0 and 1. These probabilities should be interpreted as model "
                "confidence rather than calibrated real-world frequencies."
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# SHAP chart
# ---------------------------------------------------------------------------

def _render_shap_chart(shap_result, prediction) -> None:
    """
    Render the live SHAP waterfall chart for the selected port-day.
    Falls back to pre-computed PNG if live computation failed.
    """
    st.markdown(section_title("📊 Prediction Drivers"), unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:12px;color:{T["text_secondary"]};margin-bottom:12px;">'
        f'<span style="color:{T["risk_high"]};">■</span> Red bars push toward '
        f'<b style="color:{T["text_primary"]};">Congested</b>&nbsp;&nbsp;'
        f'<span style="color:{T["accent"]};">■</span> Blue bars push toward '
        f'<b style="color:{T["text_primary"]};">Not Congested</b>&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<b>Explains model behaviour — not real-world causality</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if shap_result is not None:
        try:
            title = f"{prediction.port} — {prediction.date}"
            fig   = build_shap_waterfall_figure(shap_result, title=title)
            apply_matplotlib_theme(fig)
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Live SHAP chart failed ({e}). Showing pre-computed chart.")
            _show_precomputed_shap(prediction.port)
    else:
        _show_precomputed_shap(prediction.port)


def _show_precomputed_shap(port: str) -> None:
    """Fallback: show the pre-computed SHAP waterfall PNG."""
    path = get_shap_waterfall_path(port)
    if path.exists():
        st.image(
            str(path),
            use_column_width = True,
            caption = (
                f"Pre-computed SHAP waterfall for {port} "
                "(from model evaluation — not specific to selected date)."
            ),
        )
    else:
        st.info(
            f"SHAP chart not available for {port}. "
            "Run `python src/models/evaluate_classifier.py` to generate."
        )




# ---------------------------------------------------------------------------
# Prediction Summary
# ---------------------------------------------------------------------------

def _render_prediction_summary(prediction, shap_result) -> None:
    """
    Plain-English narrative generated from the top SHAP drivers.
    Answers: what does this prediction mean and what drove it?
    """
    st.markdown(section_title("📋 Prediction Summary"), unsafe_allow_html=True)

    port      = prediction.port
    date      = prediction.date
    label_str = prediction.label_str
    prob      = prediction.probability

    # Classify prediction confidence
    if prob >= 0.75 or prob <= 0.25:
        confidence = "high confidence"
    elif prob >= 0.60 or prob <= 0.40:
        confidence = "moderate confidence"
    else:
        confidence = "low confidence (borderline)"

    direction_phrase = (
        "elevated congestion risk" if prediction.label == 1 else "low congestion risk"
    )

    # Opening sentence
    opening = (
        f"The model predicts <b>{direction_phrase}</b> for <b>{port}</b> on <b>{date}</b> "
        f"({prob:.0%} probability, {confidence})."
    )

    # Extract top drivers — sorted by absolute SHAP, already ordered in shap_result
    names  = shap_result.feature_names
    values = shap_result.shap_values
    n      = min(4, len(names))

    # Separate into supporting (toward predicted class) and opposing
    predicted_positive = prediction.label == 1
    supporting = [
        (names[i], values[i]) for i in range(n)
        if (predicted_positive and values[i] > 0)
        or (not predicted_positive and values[i] < 0)
    ]
    opposing = [
        (names[i], values[i]) for i in range(n)
        if (predicted_positive and values[i] < 0)
        or (not predicted_positive and values[i] > 0)
    ]

    # Build bullet lines for supporting evidence (max 3)
    def _describe(name: str, shap_val: float) -> str:
        abs_val = abs(shap_val)
        magnitude = "strongly" if abs_val > 0.3 else ("moderately" if abs_val > 0.1 else "slightly")
        direction = "increased" if shap_val > 0 else "reduced"
        # Humanise common feature names
        label_map = {
            "anchored_ratio":       "High proportion of anchored vessels",
            "slow_vessel_ratio":    "High slow-vessel ratio",
            "avg_speed_in_zone":    "Low average vessel speed in zone",
            "avg_idle_duration":    "Extended idle vessel duration",
            "vessels_in_zone":      "Vessel density in zone",
            "max_weather_severity_score": "Weather severity",
            "is_weekend":           "Weekend day",
            "is_holiday":           "Holiday",
        }
        readable = label_map.get(name, name.replace("_", " ").title())
        return f"{readable} ({magnitude} {direction} predicted risk)"

    evidence_lines = [_describe(n, v) for n, v in supporting[:3]]

    # Build HTML
    bullets_html = "".join(
        f'<li style="margin-bottom:4px;">{line}</li>'
        for line in evidence_lines
    ) if evidence_lines else "<li>No dominant features identified.</li>"

    opposing_sentence = ""
    if opposing:
        opp_name = label_map_get(opposing[0][0])
        opposing_sentence = (
            f'<br><span style="color:{T["text_secondary"]};font-size:12px;">'
            f'Partially offset by: {opp_name}.</span>'
        )

    push_direction = "toward <b>Congested</b>" if prediction.label == 1 else "toward <b>Not Congested</b>"

    st.markdown(
        f'<div style="background:{T["bg_card"]};border:1px solid {T["border"]};'
        f'border-radius:10px;padding:16px 18px;">'
        f'<div style="font-size:13px;color:{T["text_primary"]};margin-bottom:10px;">'
        f'{opening}</div>'
        f'<div style="font-size:12px;color:{T["text_secondary"]};margin-bottom:6px;">'
        f'<b style="color:{T["text_primary"]};">Strongest evidence pushing this prediction '
        f'{push_direction}:</b></div>'
        f'<ul style="margin:0;padding-left:18px;font-size:13px;'
        f'color:{T["text_primary"]};line-height:1.7;">'
        f'{bullets_html}</ul>'
        f'{opposing_sentence}'
        f'</div>',
        unsafe_allow_html=True,
    )


def label_map_get(name: str) -> str:
    """Return human-readable feature name."""
    label_map = {
        "anchored_ratio":             "High anchored vessel ratio",
        "slow_vessel_ratio":          "High slow-vessel ratio",
        "avg_speed_in_zone":          "Low average vessel speed",
        "avg_idle_duration":          "Extended idle duration",
        "vessels_in_zone":            "Vessel density",
        "max_weather_severity_score": "Weather severity",
        "is_weekend":                 "Weekend day",
        "is_holiday":                 "Holiday",
    }
    return label_map.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Top Drivers card
# ---------------------------------------------------------------------------

def _render_top_drivers(shap_result) -> None:
    """
    3-item ranked card. Designed for non-technical viewers.
    Each item: rank, readable feature name, plain-English direction.
    """
    st.markdown(section_title("🏆 Top Drivers"), unsafe_allow_html=True)

    magnitude_words = {
        True:  ["Strongly increased", "Increased", "Slightly increased"],
        False: ["Strongly reduced",   "Reduced",   "Slightly reduced"],
    }

    rows_html = ""
    for i in range(min(3, len(shap_result.feature_names))):
        fname   = shap_result.feature_names[i]
        sval    = shap_result.shap_values[i]
        abs_val = abs(sval)
        toward_congested = sval > 0

        mag_idx   = 0 if abs_val > 0.3 else (1 if abs_val > 0.1 else 2)
        magnitude = magnitude_words[toward_congested][mag_idx]
        outcome   = "congestion risk" if toward_congested else "congestion risk"
        clr       = T["risk_high"] if toward_congested else T["risk_low"]
        readable  = label_map_get(fname)

        rows_html += (
            f'<div style="padding:10px 0;border-bottom:1px solid {T["border_light"]};">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="font-size:16px;font-weight:700;color:{T["text_muted"]};'
            f'min-width:20px;">{i+1}</div>'
            f'<div>'
            f'<div style="font-size:13px;font-weight:600;color:{T["text_primary"]};">'
            f'{readable}</div>'
            f'<div style="font-size:12px;color:{clr};margin-top:2px;">'
            f'{magnitude} {outcome}</div>'
            f'</div></div></div>'
        )

    st.markdown(card(rows_html, padding="14px 16px"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top 10 influential features table
# ---------------------------------------------------------------------------

def _render_top_features_table(
    row:             pd.Series,
    feature_columns: list,
    shap_result,
) -> None:
    """
    Top 10 features by absolute SHAP impact.
    Columns: Feature | Value | SHAP Impact | Direction
    Much tighter than the old all-features dump.
    """
    st.markdown(section_title("📋 Top 10 Influential Features"), unsafe_allow_html=True)

    if shap_result is None:
        st.info("SHAP values not available.")
        return

    # Build sorted list of (name, value, shap) by |shap| descending
    shap_pairs = sorted(
        zip(shap_result.feature_names, shap_result.feature_values, shap_result.shap_values),
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:10]

    table_rows = []
    for fname, fval, sval in shap_pairs:
        direction = "↑ Toward Congested" if sval > 0 else "↓ Toward Not Congested"
        table_rows.append({
            "Feature":     fname,
            "Value":       round(float(fval), 4) if isinstance(fval, (int, float)) else fval,
            "SHAP Impact": round(float(sval), 4),
            "Direction":   direction,
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width = True,
        hide_index          = True,
    )
    st.caption("Sorted by absolute SHAP impact. SHAP = model behaviour, not causality.")


# ---------------------------------------------------------------------------
# Model Interpretation card
# ---------------------------------------------------------------------------

def _render_model_interpretation(shap_result, prediction) -> None:
    """
    2-sentence auto-generated narrative classifying what kind of evidence
    drove the prediction: vessel behaviour, weather, or temporal signals.
    """
    st.markdown(section_title("🧭 Model Interpretation"), unsafe_allow_html=True)

    # Classify each top feature into a category
    vessel_features  = {"anchored_ratio", "slow_vessel_ratio", "avg_speed_in_zone",
                        "avg_idle_duration", "vessels_in_zone", "idle_duration_norm"}
    weather_features = {"max_weather_severity_score", "adverse_weather_day",
                        "weather_congestion_interaction"}
    temporal_features= {"is_weekend", "is_holiday", "month", "weekday",
                        "quarter", "season_encoded"}

    top_n = min(6, len(shap_result.feature_names))
    category_impact  = {"vessel": 0.0, "weather": 0.0, "temporal": 0.0, "other": 0.0}

    for i in range(top_n):
        fname = shap_result.feature_names[i]
        abs_s = abs(shap_result.shap_values[i])
        if fname in vessel_features:
            category_impact["vessel"]   += abs_s
        elif fname in weather_features:
            category_impact["weather"]  += abs_s
        elif fname in temporal_features:
            category_impact["temporal"] += abs_s
        else:
            category_impact["other"]    += abs_s

    dominant = max(category_impact, key=category_impact.get)
    weather_pct = (
        category_impact["weather"] / sum(category_impact.values()) * 100
        if sum(category_impact.values()) > 0 else 0
    )

    # Sentence 1: what dominated
    if dominant == "vessel":
        s1 = (
            "This prediction was driven primarily by <b>vessel behaviour indicators</b> — "
            "vessel speed, anchoring patterns, and idle duration in the port zone."
        )
    elif dominant == "weather":
        s1 = (
            "This prediction was driven primarily by <b>weather conditions</b> — "
            "adverse weather was the dominant signal the model relied on."
        )
    elif dominant == "temporal":
        s1 = (
            "This prediction was driven primarily by <b>temporal patterns</b> — "
            "day-of-week and seasonal signals were the strongest model inputs."
        )
    else:
        s1 = (
            "This prediction was influenced by a <b>mix of signals</b> — "
            "no single feature category dominated."
        )

    # Sentence 2: weather contribution context
    if weather_pct >= 30:
        s2 = f"Weather severity was a significant contributing factor ({weather_pct:.0f}% of top-feature influence)."
    elif weather_pct >= 10:
        s2 = f"Weather contributed moderately ({weather_pct:.0f}% of top-feature influence)."
    else:
        s2 = "Weather conditions contributed only marginally to this prediction."

    st.markdown(
        f'<div style="background:{T["bg_card"]};border:1px solid {T["border"]};'
        f'border-radius:10px;padding:14px 18px;">'
        f'<div style="font-size:13px;color:{T["text_primary"]};line-height:1.7;">'
        f'{s1}<br>{s2}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        banner_note(f"⚠️ {shap_result.causality_note}"),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Historical context chart
# ---------------------------------------------------------------------------

def _render_historical_context(
    df:            pd.DataFrame,
    port:          str,
    selected_date,
    prediction,
) -> None:
    """
    Show the selected date in context of the full 90-day history.
    """
    st.markdown(
        section_title("📅 Selected Date in Historical Context"),
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed.")
        return

    port_data = df[df["nearest_port"] == port].sort_values("date")

    if len(port_data) == 0:
        return

    rc = risk_color(float(port_data["port_congestion_score"].mean()))

    fig = go.Figure()

    # Full history line
    fig.add_trace(go.Scatter(
        x             = port_data["date"],
        y             = port_data["port_congestion_score"],
        mode          = "lines",
        name          = "Congestion Score",
        line          = dict(color=rc, width=2),
        opacity       = 0.6,
        hovertemplate = "%{x|%b %d}<br>Score: %{y:.4f}<extra></extra>",
    ))

    # Selected date star marker
    selected_ts  = pd.Timestamp(selected_date)
    selected_row = port_data[port_data["date"].dt.date == selected_date]

    if len(selected_row) > 0:
        selected_score = float(selected_row["port_congestion_score"].iloc[0])
        pred_color     = T["risk_high"] if prediction.label == 1 else T["risk_low"]

        fig.add_trace(go.Scatter(
            x    = [selected_ts],
            y    = [selected_score],
            mode = "markers",
            name = "Selected date",
            marker = dict(
                size   = 14,
                color  = pred_color,
                symbol = "star",
                line   = dict(color=T["text_primary"], width=2),
            ),
            hovertemplate = (
                f"<b>Selected: {selected_date}</b><br>"
                f"Score: {selected_score:.4f}<br>"
                f"Prediction: {prediction.label_str}<extra></extra>"
            ),
        ))

        fig.add_vline(
            x          = selected_ts,
            line_width = 1.5,
            line_dash  = "dash",
            line_color = pred_color,
        )

    layout = plotly_layout(height=260, margin=dict(l=40, r=20, t=30, b=40))
    layout["yaxis"]["range"] = [
        max(0, float(port_data["port_congestion_score"].min()) - 0.05),
        min(1.05, float(port_data["port_congestion_score"].max()) + 0.05),
    ]
    layout["yaxis"]["title"] = "Score"
    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"⭐ Star = selected date ({selected_date}).  "
        f"{'🔴 Red' if prediction.label == 1 else '🟢 Green'} = "
        f"{'Congested' if prediction.label == 1 else 'Not Congested'} prediction."
    )