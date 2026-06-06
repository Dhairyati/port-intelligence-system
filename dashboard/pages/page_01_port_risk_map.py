"""
page_01_port_risk_map.py
------------------------
Page 1 — Port Risk Overview

Sections:
  1. Top metrics row — score, risk label, trend arrow per port
  2. PyDeck world map — coloured markers, dark-matter style
  3. Port risk summary table
  4. Operational Insights box
  5. 90-day score history chart — all 5 ports on one chart
  6. Methodology expander (condensed)

All colors from theme.T — no hex values defined here.
Called via render() from app.py.
"""

import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.data_loader import (
    HIGH_CAUTION_PORTS,
    PORT_COORDINATES,
    get_latest_scores,
    get_trend_direction,
    load_master_dataset,
)
from dashboard.utils.theme import (
    T,
    banner_warn,
    card,
    metric_card,
    plotly_layout,
    risk_color,
    risk_label,
    section_title,
    trend_color,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _build_map_data(latest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in latest.iterrows():
        port   = row["nearest_port"]
        score  = float(row.get("port_congestion_score", 0.5))
        coords = PORT_COORDINATES.get(port, {"lat": 0, "lon": 0})
        rgb    = _hex_to_rgb(risk_color(score))
        rows.append({
            "port":    port,
            "lat":     coords["lat"],
            "lon":     coords["lon"],
            "score":   round(score, 4),
            "risk":    risk_label(score),
            "color_r": rgb[0],
            "color_g": rgb[1],
            "color_b": rgb[2],
            "color_a": 220,
            "radius":  72000,
        })
    return pd.DataFrame(rows)


def _build_operational_insights(df: pd.DataFrame, latest: pd.DataFrame) -> list[str]:
    """
    Derive 4–6 prioritised plain-English insight bullets from live data.
    Fixed order: highest risk → elevated summary → top increase → top decrease
                 → most stable → NY warning.
    Returns a list of strings (no HTML) — rendered inside the insights card.
    """
    insights = []
    ports = latest["nearest_port"].tolist()

    scores = {
        row["nearest_port"]: float(row.get("port_congestion_score", 0.5))
        for _, row in latest.iterrows()
    }

    # 1. Highest congestion risk port
    highest_port = max(scores, key=scores.get)
    highest_score = scores[highest_port]
    insights.append(
        f"🔴 {highest_port} currently shows the highest congestion risk "
        f"in the network ({risk_label(highest_score)})"
    )

    # 2. Elevated ports summary
    elevated = [p for p, s in scores.items() if s > 0.70]
    if elevated:
        insights.append(
            f"🔴 {len(elevated)} of {len(ports)} monitored ports currently "
            f"show elevated congestion ({', '.join(elevated)})"
        )
    else:
        moderate = [p for p, s in scores.items() if s > 0.40]
        if moderate:
            insights.append(
                f"🟡 {len(moderate)} of {len(ports)} ports at moderate congestion "
                f"— no ports currently in the high-risk zone"
            )
        else:
            insights.append(
                f"🟢 All {len(ports)} monitored ports currently below "
                f"elevated congestion threshold"
            )

    # 3 & 4. Largest week-over-week increase and decrease — one bullet each, max two
    deltas = {}
    for port in ports:
        port_data = df[df["nearest_port"] == port].sort_values("date")
        if len(port_data) >= 14:
            last7  = port_data["port_congestion_score"].iloc[-7:].mean()
            prior7 = port_data["port_congestion_score"].iloc[-14:-7].mean()
            if prior7 > 0:
                deltas[port] = (last7 - prior7) / prior7 * 100  # signed %

    if deltas:
        biggest_rise = max(deltas, key=deltas.get)
        biggest_fall = min(deltas, key=deltas.get)

        if deltas[biggest_rise] >= 5:
            insights.append(
                f"📈 {biggest_rise} congestion risk increased "
                f"{deltas[biggest_rise]:.0f}% over the past week"
            )
        if deltas[biggest_fall] <= -5 and biggest_fall != biggest_rise:
            insights.append(
                f"📉 {biggest_fall} congestion risk decreased "
                f"{abs(deltas[biggest_fall]):.0f}% over the past week"
            )

    # 5. Most stable port (lowest 7-day std dev)
    stds = {}
    for port in ports:
        port_data = df[df["nearest_port"] == port].sort_values("date")
        if len(port_data) >= 7:
            stds[port] = port_data["port_congestion_score"].iloc[-7:].std()
    if stds:
        most_stable = min(stds, key=stds.get)
        insights.append(
            f"🟢 {most_stable} remains the most stable port this week"
        )

    # 6. New York reliability warning — always last if present
    if any(p in HIGH_CAUTION_PORTS for p in ports):
        insights.append(
            "⚠️ New York predictions should be treated cautiously (LOPO AUC = 0.50)"
        )

    return insights


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render():

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        df = load_master_dataset()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    latest = get_latest_scores(df)
    ports  = latest["nearest_port"].tolist()

    last_date = df["date"].max().strftime("%B %d, %Y")
    st.caption(
        f"📅 Data current as of **{last_date}**  "
        f"|  Latest available score per port  "
        f"|  {len(ports)} ports monitored"
    )

    # ── Metrics row ───────────────────────────────────────────────────────────
    cols = st.columns(len(ports))
    for i, (_, row) in enumerate(latest.iterrows()):
        port   = row["nearest_port"]
        score  = float(row.get("port_congestion_score", 0.5))
        trend  = get_trend_direction(df, port)
        rc     = risk_color(score)
        tc     = trend_color(trend)
        label  = port + (" ⚠" if port in HIGH_CAUTION_PORTS else "")
        with cols[i]:
            st.markdown(
                metric_card(
                    label       = label,
                    value       = f"{score:.3f}",
                    value_color = rc,
                    sub         = risk_label(score),
                    trend       = trend,
                    trend_clr   = tc,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Map + table ───────────────────────────────────────────────────────────
    map_col, tbl_col = st.columns([2, 1], gap="large")

    with map_col:
        st.markdown(section_title("🗺 Live Risk Map"), unsafe_allow_html=True)

        map_df = _build_map_data(latest)

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data                  = map_df,
            get_position          = ["lon", "lat"],
            get_color             = ["color_r", "color_g", "color_b", "color_a"],
            get_radius            = "radius",
            pickable              = True,
            stroked               = True,
            get_line_color        = [255, 255, 255, 60],
            line_width_min_pixels = 1,
        )

        labels = pdk.Layer(
            "TextLayer",
            data                   = map_df,
            get_position           = ["lon", "lat"],
            get_text               = "port",
            get_size               = 13,
            get_color              = [232, 234, 246, 210],
            get_pixel_offset       = [0, -60],
            get_anchor             = "'middle'",
            get_alignment_baseline = "'bottom'",
        )

        view = pdk.ViewState(
            latitude  = 37.5,
            longitude = -96.0,
            zoom      = 3.1,
            pitch     = 0,
        )

        tooltip = {
            "html": (
                f"<b style='font-size:13px;color:{T['text_primary']};'>"
                "{port}</b><br>"
                f"<span style='color:{T['text_secondary']};'>Risk:</span> "
                "{risk}<br>"
                f"<span style='color:{T['text_secondary']};'>Score:</span> "
                "{score}"
            ),
            "style": {
                "backgroundColor": T["bg_card"],
                "color":           T["text_primary"],
                "fontSize":        "12px",
                "padding":         "8px 12px",
                "borderRadius":    "6px",
                "border":          f"1px solid {T['border']}",
            },
        }

        st.pydeck_chart(
            pdk.Deck(
                layers             = [scatter, labels],
                initial_view_state = view,
                map_style          = T["map_style"],
                tooltip            = tooltip,
            ),
            use_container_width=True,
        )

        st.markdown(
            f'<div style="display:flex;gap:18px;font-size:12px;'
            f'color:{T["text_secondary"]};margin-top:6px;">'
            f'<span>🔴 High Risk (&gt; 0.70)</span>'
            f'<span>🟡 Moderate (0.40–0.70)</span>'
            f'<span>🟢 Low Risk (≤ 0.40)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with tbl_col:
        st.markdown(section_title("📋 Port Risk Summary"), unsafe_allow_html=True)

        tbl_rows = []
        for _, row in latest.iterrows():
            port  = row["nearest_port"]
            score = float(row.get("port_congestion_score", 0.5))
            trend = get_trend_direction(df, port)
            clf   = int(row.get("congestion_label", 0))
            tbl_rows.append({
                "Port":       port + (" ⚠" if port in HIGH_CAUTION_PORTS else ""),
                "Score":      round(score, 3),
                "Risk":       risk_label(score),
                "Trend":      trend,
                "Classifier": "Congested" if clf == 1 else "Clear",
            })

        st.dataframe(
            pd.DataFrame(tbl_rows),
            use_container_width = True,
            hide_index          = True,
        )

        st.markdown(
            f'<div style="font-size:11px;color:{T["text_secondary"]};margin-top:6px;">'
            f'<b style="color:{T["text_primary"]};">Trend:</b><br>'
            f'↑ Worsening — last 7d &gt; prior 7d<br>'
            f'↓ Improving — last 7d &lt; prior 7d<br>'
            f'→ Stable — change within ±0.02'
            f'</div>',
            unsafe_allow_html=True,
        )

        if any(p in HIGH_CAUTION_PORTS for p in ports):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                banner_warn(
                    "<b>⚠ New York — Caution</b><br>"
                    "<span style='font-size:12px;'>"
                    "Atlantic corridor anomaly (LOPO AUC = 0.50). "
                    "Predictions less reliable than other ports."
                    "</span>"
                ),
                unsafe_allow_html=True,
            )

    # ── Operational Insights ──────────────────────────────────────────────────
    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0 16px 0;">',
        unsafe_allow_html=True,
    )
    st.markdown(section_title("💡 Operational Insights"), unsafe_allow_html=True)

    insights = _build_operational_insights(df, latest)
    bullets  = "".join(
        f'<div style="padding:5px 0;border-bottom:1px solid {T["border_light"]};'
        f'font-size:13px;color:{T["text_primary"]};">{item}</div>'
        for item in insights
    )
    st.markdown(
        card(bullets, padding="14px 18px"),
        unsafe_allow_html=True,
    )

    # ── Score history chart ───────────────────────────────────────────────────
    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:20px 0 16px 0;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        section_title("📊 90-Day Congestion Score History — All Ports"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Scores are relative within each port — 0.9 at New York and 0.9 at "
        "Seattle represent different absolute congestion levels. "
        "Dashed vertical line = train/test split (Mar 2, 2023). "
        "New York shown as dotted line (LOPO anomaly)."
    )

    _render_history_chart(df)

    # ── Methodology expander ──────────────────────────────────────────────────
    with st.expander("ℹ️ About the Congestion Score"):
        st.markdown(
            f'<div style="color:{T["text_primary"]};font-size:13px;line-height:1.7;">'
            f'• <b>Congestion Score</b> = weighted combination of vessel speed (45%), '
            f'anchoring behavior (35%), and idle duration (20%) — derived from raw AIS pings<br>'
            f'• <b>Congestion label</b> = top 40% most congested days per port '
            f'(60th percentile threshold, port-specific)<br>'
            f'• <b>Coverage</b> = 5 US ports, Jan–Mar 2023, 25 km anchorage zone<br>'
            f'• <b>Scores are relative</b> — 0.9 at Houston and 0.9 at Seattle '
            f'reflect different absolute congestion levels<br>'
            f'• See the <b>Model Report</b> for full limitations and evaluation details'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# History chart helper
# ---------------------------------------------------------------------------

def _render_history_chart(df: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed. Run: pip install plotly")
        return

    fig = go.Figure()

    for port in sorted(df["nearest_port"].unique()):
        port_data = df[df["nearest_port"] == port].sort_values("date")
        color     = T["port_colors"].get(port, T["accent"])
        dash      = "dot" if port in HIGH_CAUTION_PORTS else "solid"
        name      = port + (" ⚠" if port in HIGH_CAUTION_PORTS else "")

        fig.add_trace(
            go.Scatter(
                x    = port_data["date"],
                y    = port_data["port_congestion_score"],
                mode = "lines",
                name = name,
                line = dict(color=color, width=2, dash=dash),
                hovertemplate = (
                    f"<b>{port}</b><br>"
                    "Date: %{x|%b %d, %Y}<br>"
                    "Score: %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x          = "2023-03-02",
        line_width = 1.2,
        line_dash  = "dash",
        line_color = T["text_muted"],
    )

    layout = plotly_layout(height=300)
    layout["yaxis"]["range"] = [0, 1.05]
    layout["yaxis"]["title"] = "Congestion Score"
    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True)