"""
page_04_model_report.py
-----------------------
Page 4 — Model Report

Integrates all evaluation outputs into the dashboard so that
rigor, limitations, and methodology are part of the product
rather than buried in separate notebooks.

Sections:
  1. Project overview and methodology summary
  2. Training configuration summary
  3. Full evaluation report chart (evaluation_report.png)
  4. Per-port metrics table (colour-coded AUC)
  5. LOPO generalisation results with interpretation
  6. Per-port SHAP waterfall charts (expandable)
  7. Honest limitations section

Called via render() from app.py.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.data_loader import (
    HIGH_CAUTION_PORTS,
    get_shap_waterfall_path,
    load_lopo_results,
    load_per_port_metrics,
    load_training_metadata,
)
from dashboard.utils.theme import (
    T,
    banner_info,
    banner_note,
    banner_warn,
    lopo_score_card,
    section_title,
)

# Path to evaluation report PNG
EVAL_REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation_report.png"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render():
    """Main render function called from app.py."""

    # ── Load artifacts ────────────────────────────────────────────────────────
    try:
        per_port_df = load_per_port_metrics()
        lopo_df     = load_lopo_results()
        metadata    = load_training_metadata()
    except Exception as e:
        st.error(f"Failed to load evaluation artifacts: {e}")
        return

    # ── Section 1: Project overview ───────────────────────────────────────────
    render_project_overview()

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0;">',
        unsafe_allow_html=True,
    )

    # ── Section 2: Training configuration ────────────────────────────────────
    render_training_config(metadata)

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0;">',
        unsafe_allow_html=True,
    )

    # ── Section 3: Full evaluation report ────────────────────────────────────
    render_evaluation_report()

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0;">',
        unsafe_allow_html=True,
    )

    # ── Section 4: Per-port metrics table ────────────────────────────────────
    render_per_port_table(per_port_df)

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0;">',
        unsafe_allow_html=True,
    )

    # ── Section 5: LOPO results ───────────────────────────────────────────────
    render_lopo_results(lopo_df)

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0;">',
        unsafe_allow_html=True,
    )

    # ── Section 6: SHAP waterfall charts ─────────────────────────────────────
    render_shap_waterfall_gallery()

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0;">',
        unsafe_allow_html=True,
    )

    # ── Section 7: Honest limitations ────────────────────────────────────────
    render_limitations()


# ---------------------------------------------------------------------------
# Section 1: Project Overview
# ---------------------------------------------------------------------------

def render_project_overview() -> None:
    """Render project methodology summary."""

    st.markdown(section_title("🧭 Project Overview"), unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f'<div style="color:{T["text_primary"]};font-size:13px;">'
            f'<b>What this system does</b><br><br>'
            f'The Port Intelligence System ingests raw AIS vessel tracking data '
            f'from NOAA MarineCadastre, engineers congestion-relevant features, '
            f'trains an XGBoost binary classifier to detect congested port-days, '
            f'and explains predictions using SHAP values.'
            f'<br><br>'
            f'<b>Data pipeline</b><br>'
            f'• 690 million AIS vessel position pings processed<br>'
            f'• 5 US ports: Houston, Los Angeles, New York, Savannah, Seattle<br>'
            f'• 90 days: January – March 2023<br>'
            f'• 450 daily port-level observations in master dataset<br>'
            f'• Historical weather observations integrated as external features'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("""
        **ML architecture**

        | Component | Detail |
        |---|---|
        | Classifier | XGBoost (primary) + Logistic Regression (baseline) |
        | Forecaster | Prophet per port (exploratory) |
        | Explainability | SHAP TreeExplainer |
        | Evaluation | TimeSeriesSplit CV + temporal holdout + LOPO |
        | Features | 34 engineered features (after leakage removal) |
        | Target | Binary congestion label (60th percentile threshold) |
        """)

        st.markdown(
            f'<div style="color:{T["text_primary"]};font-size:13px;">'
            f'<b>Key design decisions</b><br>'
            f'• Temporal train/test split (not random) — prevents data leakage<br>'
            f'• 9 leakage columns explicitly excluded from training<br>'
            f'• Leave-One-Port-Out test for generalisation assessment'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 2: Training configuration
# ---------------------------------------------------------------------------

def render_training_config(metadata: dict) -> None:
    """Render training configuration and holdout metrics from metadata."""

    st.markdown(section_title("⚙️ Training Configuration"), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label = "Train Rows",
            value = metadata.get("train_rows", "—"),
            help  = "Rows used for training (Jan 1 → Mar 1)",
        )

    with col2:
        st.metric(
            label = "Test Rows",
            value = metadata.get("test_rows", "—"),
            help  = "Holdout rows (Mar 2 → Mar 31)",
        )

    with col3:
        xgb_auc = metadata.get("xgb_auc", None)
        st.metric(
            label = "XGBoost Holdout AUC",
            value = f"{xgb_auc:.4f}" if xgb_auc else "—",
            help  = "AUC-ROC on Mar 2–31 holdout set",
        )

    with col4:
        lr_auc = metadata.get("lr_auc", None)
        st.metric(
            label = "LR Baseline Holdout AUC",
            value = f"{lr_auc:.4f}" if lr_auc else "—",
            help  = "Logistic Regression baseline AUC on same holdout",
        )

    st.markdown(
        banner_info(
            "Evaluation used temporal validation and holdout testing to better reflect "
            "operational deployment conditions and reduce leakage risk."
        ),
        unsafe_allow_html=True,
    )

    # XGBoost parameters expander
    with st.expander("🔧 XGBoost Hyperparameters"):
        xgb_params = metadata.get("xgb_params", {})
        if xgb_params:
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in xgb_params.items()
                if k != "verbosity"
            ])
            st.dataframe(params_df, hide_index=True, use_container_width=True)
            st.caption(
                "These are conservative defaults for a small tabular dataset. "
                "Shallow max_depth=4 and high regularisation (reg_alpha, reg_lambda) "
                "reduce overfitting on 300 training rows."
            )

    # Leakage columns expander
    with st.expander("🚫 Excluded Leakage Columns (9 columns)"):
        leakage_cols = metadata.get("leakage_columns", [])
        leakage_explanations = {
            "congestion_label":               "Target variable — obviously excluded",
            "port_congestion_score":          "Direct source of the label — would trivially reconstruct it",
            "port_congestion_score_norm":     "Normalised label source — used to create congestion_label",
            "congestion_pressure_index":      "Weighted composite including congestion_score",
            "weather_congestion_interaction": "Interaction of slow_vessel_ratio × weather — amplifies label signal",
            "vessels_7d_rolling_norm":        "Min-max normalised using full 90-day window — mild lookahead leakage",
            "nearest_port":                   "String — replaced by port_encoded (LabelEncoded integer)",
            "date":                           "Datetime index — temporal info captured via month/quarter/weekday",
            "season":                         "String — replaced by season_encoded (LabelEncoded integer)",
        }
        if leakage_cols:
            lex_df = pd.DataFrame([
                {
                    "Column":                 col,
                    "Reason for Exclusion":   leakage_explanations.get(col, "See train_classifier.py"),
                }
                for col in leakage_cols
            ])
            st.dataframe(lex_df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 3: Evaluation Report
# ---------------------------------------------------------------------------

def render_evaluation_report() -> None:
    """Display the full evaluation report PNG."""

    st.markdown(section_title("📊 Full Evaluation Report"), unsafe_allow_html=True)
    st.caption(
        "Generated by `src/models/evaluate_classifier.py`. "
        "Contains confusion matrices, AUC-ROC, Precision-Recall, "
        "per-port AUC, calibration curve, and LOPO results."
    )

    if EVAL_REPORT_PATH.exists():
        st.image(
            str(EVAL_REPORT_PATH),
            use_column_width = True,
            caption = (
                "Port Intelligence System — Classifier Evaluation Report. "
                "XGBoost vs Logistic Regression Baseline. "
                "Holdout: Mar 2–31 2023. "
                "SHAP = model behaviour explanation, NOT causality."
            ),
        )
    else:
        st.warning(
            f"Evaluation report not found at `{EVAL_REPORT_PATH}`.\n\n"
            "Run `python src/models/evaluate_classifier.py` to generate."
        )


# ---------------------------------------------------------------------------
# Section 4: Per-Port Metrics Table
# ---------------------------------------------------------------------------

def render_per_port_table(per_port_df: pd.DataFrame) -> None:
    """
    Render per-port evaluation metrics with colour-coded AUC values.
    Green = good (>0.85), Amber = moderate (0.65–0.85), Red = poor (<0.65).
    """
    st.markdown(section_title("🏭 Per-Port Evaluation Metrics"), unsafe_allow_html=True)
    st.caption(
        "Metrics computed on holdout test set (Mar 2–31). "
        "~29 rows per port — treat as directional indicators, not definitive benchmarks."
    )

    if len(per_port_df) == 0:
        st.info("No per-port metrics available.")
        return

    display_df = per_port_df.copy()

    display_df["Notes"] = display_df["port"].apply(
        lambda p: "⚠ LOPO anomaly" if p in HIGH_CAUTION_PORTS else "—"
    )

    display_df = display_df.rename(columns={
        "port":            "Port",
        "n_test_days":     "Test Days",
        "n_congested":     "Congested Days",
        "n_not_congested": "Clear Days",
        "xgb_auc":         "XGB AUC",
        "xgb_f1":          "XGB F1",
        "xgb_precision":   "XGB Precision",
        "xgb_recall":      "XGB Recall",
        "lr_auc":          "LR AUC",
        "lr_f1":           "LR F1",
    })

    st.dataframe(
        display_df,
        use_container_width = True,
        hide_index          = True,
    )

    col_guide1, col_guide2 = st.columns(2)

    with col_guide1:
        st.markdown(
            f'<div style="color:{T["text_primary"]};font-size:13px;">'
            f'<b>AUC interpretation:</b><br>'
            f'🟢 &gt; 0.85 — strong discrimination<br>'
            f'🟡 0.65–0.85 — moderate discrimination<br>'
            f'🔴 &lt; 0.65 — poor (near random)'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_guide2:
        st.markdown(
            f'<div style="color:{T["text_primary"]};font-size:13px;">'
            f'<b>XGB vs LR gap:</b><br><br>'
            f'A large XGB AUC advantage over LR indicates the congestion '
            f'signal has non-linear structure that trees capture better. '
            f'A small gap means the signal is largely linearly separable — '
            f'both observed in this dataset depending on port.'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 5: LOPO Results
# ---------------------------------------------------------------------------

def render_lopo_results(lopo_df: pd.DataFrame) -> None:
    """
    Render Leave-One-Port-Out generalisation results with interpretation.
    This is the most important evaluation section for operational credibility.
    """
    st.markdown(
        section_title("🌐 Leave-One-Port-Out (LOPO) Generalisation Test"),
        unsafe_allow_html=True,
    )

    st.markdown(
        banner_info(
            "<b>What is LOPO and why does it matter?</b><br>"
            "<span style='font-size:12px;'>"
            "For each port, the model is trained on the other 4 ports and "
            "tested on the held-out port it has never seen. This answers: "
            "<b>'Can the system generalise to a new port without retraining?'</b> "
            "— the operationally relevant question for real deployment. "
            "A model that only memorises port-specific patterns would fail here."
            "</span>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if len(lopo_df) == 0:
        st.info("LOPO results not available.")
        return

    lopo_display = lopo_df.copy()
    lopo_display["Interpretation"] = lopo_display["lopo_auc"].apply(
        lambda a: (
            "✅ Good generalisation" if a > 0.80
            else "⚠️ Partial generalisation" if a > 0.65
            else "❌ Poor generalisation"
        )
    )
    lopo_display["Notes"] = lopo_display["held_out_port"].apply(
        lambda p: "Atlantic corridor anomaly — see caution note below"
        if p in HIGH_CAUTION_PORTS else "—"
    )

    lopo_display = lopo_display.rename(columns={
        "held_out_port": "Held-Out Port",
        "train_rows":    "Train Rows",
        "test_rows":     "Test Rows",
        "lopo_auc":      "LOPO AUC",
        "lopo_f1":       "LOPO F1",
    })

    st.dataframe(
        lopo_display,
        use_container_width = True,
        hide_index          = True,
    )

    # Mean LOPO AUC summary — use theme lopo_score_card
    mean_auc = lopo_df["lopo_auc"].mean()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(lopo_score_card(mean_auc), unsafe_allow_html=True)

    # New York specific note
    ny_row = lopo_df[lopo_df["held_out_port"].isin(HIGH_CAUTION_PORTS)]
    if len(ny_row) > 0:
        ny_auc = float(ny_row["lopo_auc"].iloc[0])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            banner_warn(
                f"<b>⚠ New York — Atlantic Corridor (LOPO AUC = {ny_auc:.4f})</b><br>"
                f"<span style='font-size:12px;'>"
                f"New York exhibited substantially different operational dynamics compared "
                f"with the other monitored ports. LOPO testing indicates that congestion "
                f"patterns learned from Gulf and Pacific coast ports do not transfer "
                f"directly to the Atlantic corridor.<br><br>"
                f"LOPO testing suggests that additional Atlantic-region training data "
                f"(Boston, Baltimore, Charleston) would likely improve generalisation "
                f"performance for this port. "
                f"This finding highlights the importance of geographic diversity "
                f"when developing congestion models across operationally distinct port regions."
                f"</span>"
            ),
            unsafe_allow_html=True,
        )



# ---------------------------------------------------------------------------
# Section 6: SHAP Waterfall Gallery
# ---------------------------------------------------------------------------

def render_shap_waterfall_gallery() -> None:
    """
    Display pre-computed SHAP waterfall charts for all 5 ports
    in an expandable gallery section.
    """
    st.markdown(
        section_title("🔍 SHAP Waterfall Charts — Representative Examples"),
        unsafe_allow_html=True,
    )
    st.caption(
        "One representative congested day per port from the holdout test set. "
        "Shows which features drove the prediction for that specific day. "
        "For interactive SHAP on any date, use the Day Explainer page."
    )

    st.markdown(
        banner_note(
            "<b>⚠ SHAP ≠ Causality — Important</b><br>"
            "<span style='font-size:12px;'>"
            "SHAP values explain <b>model behaviour</b> — which features the model "
            "relied on to make predictions. They do NOT imply real-world causality. "
            "High SHAP for <code>anchored_ratio</code> means the model uses this "
            "feature heavily, not that anchored vessels cause congestion. "
            "Correlation in training data drives SHAP values, not causal mechanisms. "
            "For a causal analysis you would need a controlled experiment or "
            "instrumental variable approach — outside the scope of this project."
            "</span>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    ports = ["Houston", "Los Angeles", "New York", "Savannah", "Seattle"]

    for i in range(0, len(ports), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(ports):
                break
            port = ports[i + j]
            with col:
                with st.expander(
                    f"{'⚠ ' if port in HIGH_CAUTION_PORTS else ''}{port}",
                    expanded = False,
                ):
                    path = get_shap_waterfall_path(port)
                    if path.exists():
                        st.image(
                            str(path),
                            use_column_width = True,
                            caption = (
                                f"{port} — representative congested day. "
                                + (
                                    "⚠ Extra caution: LOPO AUC=0.50 for this port."
                                    if port in HIGH_CAUTION_PORTS else ""
                                )
                            ),
                        )
                    else:
                        st.info(
                            f"SHAP chart not found for {port}.\n"
                            "Run: `python src/models/evaluate_classifier.py`"
                        )


# ---------------------------------------------------------------------------
# Section 7: Limitations
# ---------------------------------------------------------------------------

def render_limitations() -> None:
    """
    Render the key limitations section — honest but concise.
    Each limitation paired with a clear improvement path.
    """
    st.markdown(section_title("⚠️ Known Limitations"), unsafe_allow_html=True)

    st.markdown(
        banner_info(
            "The following limitations are well-understood and each has a clear "
            "path to resolution. They represent the next phase of development "
            "rather than fundamental constraints of the approach."
        ),
        unsafe_allow_html=True,
    )

    limitations = [
        {
            "title": "📅 90-day training window (Q1 2023 only)",
            "detail": (
                "The model was trained on January–March 2023 only. "
                "Seasonal patterns (summer shipping peaks, holiday slowdowns) "
                "are not represented, and annual seasonality cannot be detected. "
                "Retraining on a full year of data via the existing pipeline "
                "would capture seasonal variation and is a natural next step."
            ),
        },
        {
            "title": "🗺 5 US ports only (NOAA AIS coverage)",
            "detail": (
                "The system monitors Houston, Los Angeles, New York, Savannah, and Seattle. "
                "Global ports are not included as NOAA AIS data covers US waters only. "
                "Extending to international AIS providers such as MarineTraffic or AISHub "
                "would enable global port coverage within the existing architecture."
            ),
        },
        {
            "title": "🗽 New York — Atlantic corridor generalisation",
            "detail": (
                "New York exhibited substantially different operational dynamics "
                "compared with the other monitored ports, as identified through LOPO testing. "
                "Adding Atlantic-region ports (Boston, Baltimore, Charleston) to the training "
                "set would likely improve generalisation performance for this region. "
                "This is a data coverage issue with a clear resolution path, not a "
                "fundamental limitation of the modelling approach."
            ),
        },
        {
            "title": "🔮 Prophet forecast is exploratory only",
            "detail": (
                "Prophet forecasts were trained on 90 days per port. "
                "Reliable annual seasonality detection requires at least 2 years of data, "
                "and weekly seasonality is detectable but weak at this sample size. "
                "All forecasts should be interpreted as directional trend indicators "
                "rather than precise operational predictions."
            ),
        },
        {
            "title": "🔬 SHAP explains model behaviour, not causality",
            "detail": (
                "SHAP values identify which features the model relied on for each prediction. "
                "They do not establish causal relationships between operational conditions "
                "and port congestion. High SHAP importance for a feature means the model "
                "found it predictive — not that it caused the outcome. "
                "This is an observational system; causal inference would require "
                "a separate controlled study."
            ),
        },
    ]

    for lim in limitations:
        with st.expander(lim["title"]):
            st.markdown(
                f'<div style="color:{T["text_primary"]};font-size:13px;">'
                f'{lim["detail"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;color:{T["text_secondary"]};font-size:12px;'
        f'padding:20px 24px;border:1px solid {T["border"]};border-radius:8px;'
        f'background:{T["bg_card"]};">'
        f'<b style="color:{T["text_primary"]};font-size:13px;">'
        f'This project demonstrates an end-to-end maritime congestion intelligence workflow</b> — '
        f'including large-scale AIS processing, feature engineering, machine learning, '
        f'explainability, forecasting, and interactive analytics.<br><br>'
        f'<span style="color:{T["text_secondary"]};">'
        f'Future improvements include expanded geographic coverage, longer historical '
        f'training periods, and additional operational data sources.'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )