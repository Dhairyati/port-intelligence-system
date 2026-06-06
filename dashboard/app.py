"""
app.py
------
Port Intelligence System — Streamlit Dashboard Entry Point

Handles:
  - Page configuration
  - Global CSS injection (dark theme)
  - Sidebar: branding, navigation, auto-generated insights, data provenance
  - Routing to the four page render() functions

Run locally:
    streamlit run dashboard/app.py

Deploy:
    Push to GitHub → connect Streamlit Cloud → set main file: dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page config — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title            = "Port Intelligence System",
    page_icon             = "🚢",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Theme — inject global CSS and import component helpers
# ---------------------------------------------------------------------------
from dashboard.utils.theme import (
    T,
    banner_warn,
    brand_header,
    inject_global_css,
    insight_item,
    section_label,
    section_title,
)

inject_global_css()

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
from dashboard.utils.data_loader import (
    HIGH_CAUTION_PORTS,
    generate_auto_insights,
    load_master_dataset,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown(brand_header(), unsafe_allow_html=True)
    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:12px 0;">',
        unsafe_allow_html=True,
    )

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown(section_label("Navigation"), unsafe_allow_html=True)

    selected_page = st.radio(
        label            = "Go to",
        options          = [
            "🗺  Port Risk Overview",
            "📈  Port Deep Dive",
            "🔍  Day Explainer",
            "📊  Model Report",
        ],
        index            = 0,
        label_visibility = "collapsed",
    )

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:12px 0;">',
        unsafe_allow_html=True,
    )

    # ── Auto-generated insights ───────────────────────────────────────────────
    st.markdown(section_label("💡 System Insights"), unsafe_allow_html=True)

    try:
        df       = load_master_dataset()
        insights = generate_auto_insights(df)
        for insight in insights:
            st.markdown(insight_item(insight), unsafe_allow_html=True)
    except Exception:
        st.caption("Insights unavailable — check data files.")

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:12px 0;">',
        unsafe_allow_html=True,
    )

    # ── Data provenance ───────────────────────────────────────────────────────
    st.markdown(section_label("Data Provenance"), unsafe_allow_html=True)

    for line in [
        "📡 AIS: NOAA MarineCadastre",
        "🌤 Weather: Open-Meteo API",
        "📅 Period: Jan–Mar 2023",
        "🏭 5 US ports monitored",
    ]:
        st.markdown(
            f'<div style="font-size:12px;color:{T["text_secondary"]};'
            f'padding:2px 0;">{line}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:12px 0;">',
        unsafe_allow_html=True,
    )

    # ── Model info ────────────────────────────────────────────────────────────
    st.markdown(section_label("Model Info"), unsafe_allow_html=True)

    for line in [
        "🤖 Classifier: XGBoost",
        "📈 Forecaster: Prophet (exploratory)",
        "🔍 Explainability: SHAP",
        "✅ LOPO mean AUC: 0.895",
        "⚠️ New York: LOPO AUC = 0.50",
    ]:
        st.markdown(
            f'<div style="font-size:12px;color:{T["text_secondary"]};'
            f'padding:2px 0;">{line}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Main content — page header + routing
# ---------------------------------------------------------------------------

PAGE_META = {
    "🗺  Port Risk Overview": (
        "🗺 Port Risk Overview",
        "Current congestion risk across all 5 monitored US ports",
    ),
    "📈  Port Deep Dive": (
        "📈 Port Deep Dive",
        "90-day history, weather overlay, and 14-day exploratory forecast per port",
    ),
    "🔍  Day Explainer": (
        "🔍 Day Explainer",
        "SHAP-powered explanation of the model prediction for any port and date",
    ),
    "📊  Model Report": (
        "📊 Model Report",
        "Evaluation metrics, LOPO generalisation test, SHAP gallery, and limitations",
    ),
}

# Page title and subtitle
if selected_page in PAGE_META:
    title, subtitle = PAGE_META[selected_page]
    st.markdown(
        section_title(title),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:13px;color:{T["text_secondary"]};'
        f'margin-bottom:12px;">{subtitle}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:0 0 16px 0;">',
        unsafe_allow_html=True,
    )

# ── Route to page ─────────────────────────────────────────────────────────────
try:
    if selected_page == "🗺  Port Risk Overview":
        from dashboard.pages.page_01_port_risk_map import render

    elif selected_page == "📈  Port Deep Dive":
        from dashboard.pages.page_02_port_deep_dive import render

    elif selected_page == "🔍  Day Explainer":
        from dashboard.pages.page_03_day_explainer import render

    elif selected_page == "📊  Model Report":
        from dashboard.pages.page_04_model_report import render

    render()

except ImportError as e:
    st.error(
        f"Page failed to load: `{e}`\n\n"
        "Check that all files in `dashboard/pages/` exist."
    )
except Exception as e:
    st.error(f"Page error: {e}")
    st.exception(e)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f'<hr style="border-color:{T["border"]};margin:24px 0 8px 0;">',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="text-align:center;color:{T["text_muted"]};'
    f'font-size:11px;padding:4px 0;">'
    f'Port Intelligence System &nbsp;·&nbsp; '
    f'XGBoost · Prophet · SHAP · Streamlit &nbsp;·&nbsp; '
    f'AIS: NOAA MarineCadastre &nbsp;·&nbsp; Weather: Open-Meteo &nbsp;·&nbsp; '
    f'<em>Exploratory research project — not for operational use</em>'
    f'</div>',
    unsafe_allow_html=True,
)