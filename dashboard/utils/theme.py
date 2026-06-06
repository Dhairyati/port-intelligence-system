"""
theme.py
--------
Centralised dark theme and style system for Port Intelligence Dashboard.

Single source of truth for all colors, component styles, and chart theming.
All pages import from here — no color values defined anywhere else.

Design philosophy:
  - Deep navy/slate background — professional maritime feel
  - Blue accent — consistent with evaluation charts
  - Red/amber/green risk colors — operationally intuitive
  - All components use inline styles — Streamlit cannot override them

Usage:
    from dashboard.utils.theme import (
        T, card, banner_info, banner_warn, banner_note,
        metric_card, plotly_layout, apply_matplotlib_theme,
        risk_color, risk_label, trend_color, inject_global_css,
    )
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Color palette — single dark theme
# ---------------------------------------------------------------------------

T = {
    # ── Backgrounds ──────────────────────────────────────────────────────────
    "bg_primary":   "#0D1117",   # near-black — main content area
    "bg_card":      "#161B2E",   # deep navy — cards, panels
    "bg_input":     "#1C2333",   # slightly lighter — inputs, tables
    "bg_sidebar":   "#0D1117",   # matches primary

    # ── Borders ──────────────────────────────────────────────────────────────
    "border":       "#2D3561",   # subtle blue-tinted border
    "border_light": "#1E2347",   # very subtle divider

    # ── Text ─────────────────────────────────────────────────────────────────
    "text_primary":   "#E8EAF6",   # soft white — main text
    "text_secondary": "#8892B0",   # muted blue-grey — labels, captions
    "text_muted":     "#4A5568",   # very muted — dividers, footnotes

    # ── Accent ───────────────────────────────────────────────────────────────
    "accent":       "#4FC3F7",   # bright cyan-blue — primary interactive
    "accent_dim":   "#1E3A5F",   # dim accent for backgrounds

    # ── Risk colors ──────────────────────────────────────────────────────────
    "risk_high":       "#EF5350",   # red
    "risk_moderate":   "#FFA726",   # amber
    "risk_low":        "#66BB6A",   # green
    "risk_high_bg":    "#2D1515",   # dark red background
    "risk_moderate_bg":"#2D1F00",   # dark amber background
    "risk_low_bg":     "#0D2B0D",   # dark green background

    # ── Banners ──────────────────────────────────────────────────────────────
    "banner_info_bg":      "#0A1929",
    "banner_info_border":  "#4FC3F7",
    "banner_info_text":    "#B3E5FC",

    "banner_warn_bg":      "#1A0A0A",
    "banner_warn_border":  "#EF5350",
    "banner_warn_text":    "#FFCDD2",

    "banner_note_bg":      "#12071A",
    "banner_note_border":  "#CE93D8",
    "banner_note_text":    "#E1BEE7",

    # ── Charts ───────────────────────────────────────────────────────────────
    "chart_bg":    "#0D1117",
    "chart_grid":  "#1E2347",
    "chart_text":  "#8892B0",

    # ── Map ──────────────────────────────────────────────────────────────────
    "map_style": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",

    # ── Port colors (consistent across all charts) ───────────────────────────
    "port_colors": {
        "Houston":     "#4FC3F7",   # cyan
        "Los Angeles": "#81C784",   # green
        "New York":    "#EF5350",   # red  (matches caution status)
        "Savannah":    "#FFB74D",   # amber
        "Seattle":     "#CE93D8",   # purple
    },
}


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------

def risk_color(score: float) -> str:
    """Return hex color for a congestion score."""
    if score > 0.70:
        return T["risk_high"]
    elif score > 0.40:
        return T["risk_moderate"]
    return T["risk_low"]


def risk_label(score: float) -> str:
    """Return emoji + text risk label."""
    if score > 0.70:
        return "🔴 High Risk"
    elif score > 0.40:
        return "🟡 Moderate"
    return "🟢 Low Risk"


def trend_color(trend: str) -> str:
    """Return hex color for a trend direction string."""
    if "Worsening" in trend:
        return T["risk_high"]
    elif "Improving" in trend:
        return T["risk_low"]
    return T["risk_moderate"]


def lopo_color(auc: float) -> str:
    """Return hex color for a LOPO AUC value."""
    if auc > 0.80:
        return T["risk_low"]
    elif auc > 0.65:
        return T["risk_moderate"]
    return T["risk_high"]


# ---------------------------------------------------------------------------
# HTML component builders — inline styles only
# Inline styles have highest CSS specificity — Streamlit cannot override them.
# All return strings — pass to st.markdown(..., unsafe_allow_html=True)
# ---------------------------------------------------------------------------

def card(content: str, padding: str = "16px", extra: str = "") -> str:
    """Base card component — dark navy background with border."""
    return (
        f'<div style="'
        f'background:{T["bg_card"]};'
        f'border:1px solid {T["border"]};'
        f'border-radius:10px;'
        f'padding:{padding};'
        f'color:{T["text_primary"]};'
        f'{extra}'
        f'">{content}</div>'
    )


def banner_info(content: str) -> str:
    """Cyan info banner — methodology notes and neutral information."""
    return (
        f'<div style="'
        f'background:{T["banner_info_bg"]};'
        f'border-left:4px solid {T["banner_info_border"]};'
        f'border-radius:0 6px 6px 0;'
        f'padding:12px 16px;margin:8px 0;'
        f'color:{T["banner_info_text"]};font-size:13px;">'
        f'{content}</div>'
    )


def banner_warn(content: str) -> str:
    """Red warning banner — cautions, anomalies, limitations."""
    return (
        f'<div style="'
        f'background:{T["banner_warn_bg"]};'
        f'border-left:4px solid {T["banner_warn_border"]};'
        f'border-radius:0 6px 6px 0;'
        f'padding:12px 16px;margin:8px 0;'
        f'color:{T["banner_warn_text"]};font-size:13px;">'
        f'{content}</div>'
    )


def banner_note(content: str) -> str:
    """Purple note banner — SHAP causality disclaimers."""
    return (
        f'<div style="'
        f'background:{T["banner_note_bg"]};'
        f'border-left:4px solid {T["banner_note_border"]};'
        f'border-radius:0 6px 6px 0;'
        f'padding:12px 16px;margin:8px 0;'
        f'color:{T["banner_note_text"]};font-size:13px;">'
        f'{content}</div>'
    )


def metric_card(
    label:      str,
    value:      str,
    value_color: str = None,
    sub:        str = "",
    trend:      str = "",
    trend_clr:  str = "",
) -> str:
    """
    Metric display card used on Page 1 and Page 2.

    Parameters
    ----------
    label       : small label above the value
    value       : large primary value
    value_color : hex color for value (defaults to accent)
    sub         : small text below value
    trend       : trend arrow string
    trend_clr   : hex color for trend text
    """
    vc = value_color or T["accent"]
    tc = trend_clr or T["text_secondary"]

    sub_html = (
        f'<div style="font-size:11px;color:{T["text_secondary"]};margin-top:3px;">'
        f'{sub}</div>'
    ) if sub else ""

    trend_html = (
        f'<div style="font-size:12px;color:{tc};font-weight:600;margin-top:6px;">'
        f'{trend}</div>'
    ) if trend else ""

    inner = (
        f'<div style="font-size:11px;color:{T["text_secondary"]};'
        f'letter-spacing:0.4px;margin-bottom:5px;">{label}</div>'
        f'<div style="font-size:24px;font-weight:700;color:{vc};'
        f'line-height:1.1;">{value}</div>'
        f'{sub_html}{trend_html}'
    )
    return card(inner, padding="14px 12px", extra="text-align:center;")


def insight_item(content: str) -> str:
    """Insight bullet for the sidebar."""
    return (
        f'<div style="'
        f'background:{T["bg_input"]};'
        f'border-left:3px solid {T["accent"]};'
        f'border-radius:0 6px 6px 0;'
        f'padding:9px 12px;margin:5px 0;'
        f'font-size:12px;color:{T["text_primary"]};">'
        f'{content}</div>'
    )


def section_label(text: str) -> str:
    """Uppercase section label for sidebar groupings."""
    return (
        f'<div style="'
        f'color:{T["text_secondary"]};font-size:10px;'
        f'font-weight:600;letter-spacing:1.2px;'
        f'text-transform:uppercase;'
        f'margin:4px 0 6px 0;">'
        f'{text}</div>'
    )


def brand_header() -> str:
    """Sidebar brand title and subtitle."""
    return (
        f'<div style="font-size:19px;font-weight:700;'
        f'color:{T["text_primary"]};letter-spacing:-0.3px;">'
        f'🚢 Port Intelligence</div>'
        f'<div style="font-size:11px;color:{T["text_secondary"]};margin-top:3px;">'
        f'Maritime Congestion Analytics</div>'
    )


def lopo_score_card(mean_auc: float) -> str:
    """Large centered LOPO AUC score display for Page 4."""
    c     = lopo_color(mean_auc)
    label = (
        "GOOD — model learned cross-port patterns" if mean_auc > 0.80
        else "PARTIAL — some port-specific memorisation" if mean_auc > 0.65
        else "POOR — model relies on port-specific signals"
    )
    return card(
        content=(
            f'<div style="text-align:center;">'
            f'<div style="font-size:11px;color:{T["text_secondary"]};'
            f'margin-bottom:6px;letter-spacing:0.5px;">'
            f'MEAN LOPO AUC ACROSS ALL PORTS</div>'
            f'<div style="font-size:42px;font-weight:800;color:{c};">'
            f'{mean_auc:.4f}</div>'
            f'<div style="font-size:13px;color:{c};font-weight:600;">'
            f'{label}</div>'
            f'</div>'
        ),
        padding="22px",
    )


def section_title(text: str) -> str:
    """Consistent section heading used across all pages."""
    return (
        f'<div style="font-size:16px;font-weight:600;'
        f'color:{T["text_primary"]};margin-bottom:8px;">'
        f'{text}</div>'
    )


# ---------------------------------------------------------------------------
# Plotly layout — consistent chart theming across all pages
# ---------------------------------------------------------------------------

def plotly_layout(height: int = 350, margin: dict = None) -> dict:
    """
    Return Plotly update_layout kwargs matching the dark theme.

    Usage:
        fig.update_layout(**plotly_layout(height=320))

    Parameters
    ----------
    height : chart height in pixels
    margin : optional margin override
    """
    m = margin or dict(l=50, r=20, t=36, b=40)
    return dict(
        height        = height,
        plot_bgcolor  = T["chart_bg"],
        paper_bgcolor = T["chart_bg"],
        font          = dict(color=T["chart_text"], size=11),
        xaxis = dict(
            showgrid   = True,
            gridcolor  = T["chart_grid"],
            gridwidth  = 0.5,
            color      = T["chart_text"],
            tickformat = "%b %d",
            linecolor  = T["border"],
        ),
        yaxis = dict(
            showgrid  = True,
            gridcolor = T["chart_grid"],
            gridwidth = 0.5,
            color     = T["chart_text"],
            linecolor = T["border"],
        ),
        legend = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
            font        = dict(size=10),
            bgcolor     = "rgba(0,0,0,0)",
        ),
        margin    = m,
        hovermode = "x unified",
        hoverlabel = dict(
            bgcolor   = T["bg_card"],
            font_size = 12,
            font_color = T["text_primary"],
        ),
    )


# ---------------------------------------------------------------------------
# Matplotlib theme — for SHAP waterfall on Page 3
# ---------------------------------------------------------------------------

def apply_matplotlib_theme(fig, axes=None):
    """
    Apply dark theme colors to a matplotlib figure in-place.
    Call after build_shap_waterfall_figure() in Page 3.
    """
    fig.patch.set_facecolor(T["chart_bg"])
    for ax in (axes or fig.axes):
        ax.set_facecolor(T["bg_card"])
        ax.tick_params(colors=T["chart_text"], labelsize=8)
        ax.xaxis.label.set_color(T["chart_text"])
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_color(T["chart_text"])
        ax.title.set_color(T["text_primary"])
        for spine in ax.spines.values():
            spine.set_color(T["border"])
    return fig


# ---------------------------------------------------------------------------
# Global CSS injection — structural only, called once from app.py
# ---------------------------------------------------------------------------

def inject_global_css() -> None:
    """
    Inject minimal structural CSS that cannot be done with inline styles.
    Does not inject any color values — those are all inline.
    Called once at the top of app.py on every rerun.
    """
    st.markdown(
        f"""
        <style>
            /* Font smoothing */
            * {{ -webkit-font-smoothing: antialiased; }}

            /* Remove default top padding */
            .block-container {{
                padding-top: 0.8rem;
                padding-bottom: 1rem;
            }}

            /* Sidebar background */
            section[data-testid="stSidebar"] > div:first-child {{
                background-color: {T["bg_sidebar"]};
                border-right: 1px solid {T["border"]};
            }}

            /* Thin scrollbar */
            ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
            ::-webkit-scrollbar-track {{ background: {T["bg_primary"]}; }}
            ::-webkit-scrollbar-thumb {{
                background: {T["border"]};
                border-radius: 3px;
            }}

            /* Dataframe header row */
            .stDataFrame thead tr th {{
                background-color: {T["bg_input"]} !important;
                color: {T["text_secondary"]} !important;
                font-size: 11px !important;
                font-weight: 600 !important;
            }}

            /* Expander header */
            .streamlit-expanderHeader {{
                color: {T["text_primary"]} !important;
                font-size: 13px !important;
                background-color: {T["bg_card"]} !important;
            }}

            /* st.metric value */
            [data-testid="stMetricValue"] {{
                color: {T["accent"]} !important;
            }}

            /* Caption text */
            .stCaption {{
                color: {T["text_secondary"]} !important;
            }}

            /* Horizontal rule */
            hr {{
                border-color: {T["border"]} !important;
                margin: 1rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )