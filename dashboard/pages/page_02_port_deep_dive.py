"""
page_02_port_deep_dive.py
-------------------------
Page 2 — Port Deep Dive

Provides detailed operational analysis for a selected port:
  - Port selector dropdown + inline current-status strip
  - Key Takeaway card (auto-generated, single sentence)
  - Key metrics row (avg, max, congested days, adverse weather)
  - 90-day congestion score history (Plotly line chart)
    · Optional weather-severity overlay on secondary y-axis
    · Congested-day background bands
    · 7-day rolling average
    · Train/test split vline
  - Port Insights box (4 auto-generated operational bullets)
  - Exploratory Congestion Outlook (Prophet 14-day forecast PNG)
  - Monthly congestion-rate bar chart
  - Weekday vs weekend congestion bar chart (only if meaningful)
  - New York caution banner (conditional)

All colors from theme.T — no hex values defined here.
Called via render() from app.py.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.data_loader import (
    HIGH_CAUTION_PORTS,
    get_forecast_path,
    get_trend_direction,
    load_external_features,
    load_master_dataset,
    load_port_features,
)
from dashboard.utils.theme import (
    T,
    banner_info,
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
# Render
# ---------------------------------------------------------------------------

def render():
    """Main render function called from app.py."""

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df          = load_master_dataset()
        port_df     = load_port_features()
        external_df = load_external_features()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    ports = sorted(df["nearest_port"].unique().tolist())

    # ── Port selector + inline status strip ──────────────────────────────────
    col_select, col_info = st.columns([1, 3])

    with col_select:
        selected_port = st.selectbox(
            "Select Port",
            options = ports,
            index   = 0,
            key     = "deep_dive_port",
        )

    # Filter for selected port
    port_master   = df[df["nearest_port"] == selected_port].sort_values("date")
    port_features = port_df[port_df["nearest_port"] == selected_port].sort_values("date")

    ext_col       = "nearest_port" if "nearest_port" in external_df.columns else "port"
    port_external = external_df[external_df[ext_col] == selected_port].sort_values("date")

    with col_info:
        if len(port_master) > 0:
            latest_score = float(port_master["port_congestion_score"].iloc[-1])
            latest_date  = port_master["date"].iloc[-1].strftime("%b %d, %Y")
            trend        = get_trend_direction(df, selected_port)
            rc           = risk_color(latest_score)
            tc           = trend_color(trend)
            rl           = risk_label(latest_score)

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:24px;
                            padding:10px 0;flex-wrap:wrap;">
                    <div>
                        <div style="font-size:11px;color:{T['text_secondary']};
                                    letter-spacing:0.4px;">LATEST SCORE</div>
                        <div style="font-size:26px;font-weight:700;
                                    color:{rc};line-height:1.1;">{latest_score:.3f}</div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:{T['text_secondary']};
                                    letter-spacing:0.4px;">RISK LEVEL</div>
                        <div style="font-size:16px;color:{T['text_primary']};
                                    font-weight:600;">{rl}</div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:{T['text_secondary']};
                                    letter-spacing:0.4px;">TREND</div>
                        <div style="font-size:16px;color:{tc};
                                    font-weight:600;">{trend}</div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:{T['text_secondary']};
                                    letter-spacing:0.4px;">AS OF</div>
                        <div style="font-size:14px;color:{T['text_primary']};
                                    font-weight:500;">{latest_date}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── New York caution banner ───────────────────────────────────────────────
    if selected_port in HIGH_CAUTION_PORTS:
        st.markdown(
            banner_warn(
                "<b>⚠ High Caution — New York (Atlantic Corridor Anomaly)</b><br>"
                "<span style='font-size:12px;'>"
                "Leave-One-Port-Out evaluation revealed that congestion patterns at "
                "New York are fundamentally different from the other 4 ports in the "
                "training data (LOPO AUC = 0.50 — equivalent to random prediction). "
                "The classifier and forecaster for this port were trained on its own "
                "historical data but may not generalise to future conditions. "
                "Treat all predictions and forecasts with extra caution."
                "</span>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Takeaway ──────────────────────────────────────────────────────────
    if len(port_master) > 0:
        _render_key_takeaway(port_master, selected_port, df)

    # ── Key metrics row ───────────────────────────────────────────────────────
    _render_metrics_row(port_master)

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:18px 0 14px 0;">',
        unsafe_allow_html=True,
    )

    # ── History chart ─────────────────────────────────────────────────────────
    st.markdown(section_title("📈 90-Day Operational Trend"), unsafe_allow_html=True)

    weather_overlay = st.checkbox(
        "Show weather severity overlay",
        value = True,
        key   = "weather_overlay",
        help  = "Overlays max daily weather severity score on the congestion history",
    )

    _render_history_chart(port_master, port_external, selected_port, weather_overlay)

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:18px 0 14px 0;">',
        unsafe_allow_html=True,
    )

    # ── Port Insights ─────────────────────────────────────────────────────────
    st.markdown(section_title("💡 Port Insights"), unsafe_allow_html=True)
    _render_port_insights(port_master, port_external, selected_port, df)

    st.markdown(
        f'<hr style="border-color:{T["border"]};margin:18px 0 14px 0;">',
        unsafe_allow_html=True,
    )

    # ── Forecast + breakdown ──────────────────────────────────────────────────
    forecast_col, breakdown_col = st.columns([3, 2], gap="large")

    with forecast_col:
        _render_forecast_section(selected_port)

    with breakdown_col:
        _render_label_breakdown(port_master, selected_port)


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

def _render_metrics_row(port_master: pd.DataFrame) -> None:
    """Render 4 key metric cards for the selected port."""
    if len(port_master) == 0:
        st.warning("No data available for this port.")
        return

    scores        = port_master["port_congestion_score"]
    n_days        = len(port_master)
    n_congested   = (
        int(port_master["congestion_label"].sum())
        if "congestion_label" in port_master.columns
        else 0
    )
    pct_congested = n_congested / n_days * 100 if n_days > 0 else 0

    adverse_col = "adverse_weather_day"
    n_adverse   = (
        int(port_master[adverse_col].sum())
        if adverse_col in port_master.columns
        else 0
    )

    cols = st.columns(4)
    metrics = [
        (cols[0], "Avg Congestion Score", f"{scores.mean():.3f}", risk_color(scores.mean())),
        (cols[1], "Peak Congestion Score", f"{scores.max():.3f}",  risk_color(scores.max())),
        (cols[2], "Congested Days",
            f"{n_congested}/{n_days} ({pct_congested:.0f}%)", T["accent"]),
        (cols[3], "Adverse Weather Days",  f"{n_adverse} days",   T["risk_moderate"]),
    ]

    for col, label, value, color in metrics:
        with col:
            st.markdown(
                metric_card(label=label, value=value, value_color=color),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# History chart
# ---------------------------------------------------------------------------

def _render_history_chart(
    port_master:   pd.DataFrame,
    port_external: pd.DataFrame,
    port:          str,
    show_weather:  bool,
) -> None:
    """
    90-day congestion score history as a Plotly chart.
    Optionally overlays weather severity on a secondary y-axis.
    Marks congested days with subtle red background bands.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("Plotly not installed.  Run: pip install plotly")
        return

    if len(port_master) == 0:
        st.info("No data available for this port.")
        return

    rc = risk_color(float(port_master["port_congestion_score"].mean()))

    has_weather = (
        show_weather
        and len(port_external) > 0
        and "max_weather_severity_score" in port_external.columns
    )

    if has_weather:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
        show_weather = False

    # ── Congested-day background bands ────────────────────────────────────────
    if "congestion_label" in port_master.columns:
        for day in port_master[port_master["congestion_label"] == 1]["date"]:
            fig.add_vrect(
                x0         = day - pd.Timedelta(hours=12),
                x1         = day + pd.Timedelta(hours=12),
                fillcolor  = T["risk_high"],
                opacity    = 0.08,
                layer      = "below",
                line_width = 0,
            )

    # ── Main congestion score line ────────────────────────────────────────────
    score_trace = go.Scatter(
        x             = port_master["date"],
        y             = port_master["port_congestion_score"],
        mode          = "lines+markers",
        name          = "Congestion Score",
        line          = dict(color=rc, width=2.5),
        marker        = dict(size=4, color=rc),
        hovertemplate = "Date: %{x|%b %d, %Y}<br>Score: %{y:.4f}<extra></extra>",
    )

    rolling_trace = go.Scatter(
        x             = port_master["date"],
        y             = port_master["port_congestion_score"].rolling(7, min_periods=3).mean(),
        mode          = "lines",
        name          = "7-day rolling avg",
        line          = dict(color=T["text_secondary"], width=1.5, dash="dash"),
        opacity       = 0.55,
        hovertemplate = "7d avg: %{y:.4f}<extra></extra>",
    )

    if has_weather:
        fig.add_trace(score_trace,   secondary_y=False)
        fig.add_trace(rolling_trace, secondary_y=False)
    else:
        fig.add_trace(score_trace)
        fig.add_trace(rolling_trace)

    # ── Weather severity overlay ──────────────────────────────────────────────
    if has_weather:
        fig.add_trace(
            go.Bar(
                x             = port_external["date"],
                y             = port_external["max_weather_severity_score"],
                name          = "Weather Severity",
                marker_color  = T["risk_moderate"],
                opacity       = 0.40,
                hovertemplate = "Weather severity: %{y:.3f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text  = "Weather Severity",
            range       = [0, 2],
            secondary_y = True,
            showgrid    = False,
            title_font  = dict(color=T["risk_moderate"], size=10),
            tickfont    = dict(color=T["risk_moderate"], size=9),
        )

    # ── Train/test split marker ───────────────────────────────────────────────
    fig.add_vline(
        x          = "2023-03-02",
        line_width = 1.2,
        line_dash  = "dash",
        line_color = T["text_muted"],
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = plotly_layout(height=380, margin=dict(l=50, r=50, t=40, b=40))

    score_min = float(port_master["port_congestion_score"].min())
    score_max = float(port_master["port_congestion_score"].max())
    primary_yaxis = dict(
        showgrid   = True,
        gridcolor  = T["chart_grid"],
        gridwidth  = 0.5,
        range      = [max(0, score_min - 0.05), min(1.05, score_max + 0.05)],
        title      = "Congestion Score",
        title_font = dict(size=11),
        color      = T["chart_text"],
        linecolor  = T["border"],
    )

    if has_weather:
        fig.update_yaxes(**primary_yaxis, secondary_y=False)
        # Remove secondary from the shared layout dict to avoid conflict
        layout.pop("yaxis", None)
    else:
        layout["yaxis"].update(primary_yaxis)

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True)

    if "congestion_label" in port_master.columns:
        n_cong = int(port_master["congestion_label"].sum())
        st.caption(
            f"🔴 Red bands = {n_cong} congested days (above 60th percentile of normalised score)  "
            "| Dashed line = 7-day rolling average  "
            "| Vertical dash = train/test split (Mar 2, 2023)"
        )


# ---------------------------------------------------------------------------
# Forecast section
# ---------------------------------------------------------------------------

def _render_forecast_section(port: str) -> None:
    """Display the pre-computed Prophet forecast PNG for the selected port."""
    st.markdown(
        section_title("🔮 Exploratory Congestion Outlook — 14 Days"),
        unsafe_allow_html=True,
    )

    if port in HIGH_CAUTION_PORTS:
        st.markdown(
            banner_warn(
                "<b>⚠ Extra Caution</b> — This port's forecast is less reliable "
                "than others due to anomalous congestion dynamics identified "
                "during model evaluation (LOPO AUC = 0.50)."
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            banner_info(
                "Forecast generated from 90 days of historical data. "
                "Use as directional guidance rather than operational prediction."
            ),
            unsafe_allow_html=True,
        )

    forecast_path = get_forecast_path(port)

    if forecast_path.exists():
        st.image(
            str(forecast_path),
            use_column_width = True,
            caption = (
                f"Prophet forecast for {port} — "
                "Upper panel: actual vs fitted + 14-day forecast with 80% CI. "
                "Lower panel: trend component + weekly seasonality."
            ),
        )
    else:
        st.warning(
            f"Forecast chart not found for **{port}**.\n\n"
            f"Expected: `{forecast_path}`\n\n"
            "Run `python src/models/train_forecaster.py` to generate."
        )


# ---------------------------------------------------------------------------
# Label breakdown charts
# ---------------------------------------------------------------------------

def _render_label_breakdown(port_master: pd.DataFrame, port: str) -> None:
    """Monthly congestion rate and weekday vs weekend breakdown charts."""
    st.markdown(section_title("📊 Congestion Breakdown"), unsafe_allow_html=True)

    if "congestion_label" not in port_master.columns or len(port_master) == 0:
        st.info("No label data available.")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed.")
        return

    pm = port_master.copy()

    # ── Monthly congestion rate ───────────────────────────────────────────────
    pm["month_name"] = pm["date"].dt.strftime("%b %Y")

    monthly = (
        pm.groupby("month_name")
        .agg(congested_days=("congestion_label", "sum"),
             total_days    =("congestion_label", "count"))
        .reset_index()
    )
    monthly["pct"] = monthly["congested_days"] / monthly["total_days"] * 100
    monthly = monthly.sort_values("month_name")

    bar_colors = [
        T["risk_high"]     if p > 50
        else T["risk_moderate"] if p > 30
        else T["risk_low"]
        for p in monthly["pct"]
    ]

    fig_monthly = go.Figure(go.Bar(
        x             = monthly["month_name"],
        y             = monthly["pct"],
        marker_color  = bar_colors,
        text          = [f"{p:.0f}%" for p in monthly["pct"]],
        textposition  = "auto",
        hovertemplate = "%{x}<br>Congested: %{y:.1f}%<extra></extra>",
    ))

    monthly_layout = plotly_layout(height=210, margin=dict(l=40, r=20, t=36, b=30))
    monthly_layout["title"] = dict(
        text      = "Monthly Congestion Rate",
        font      = dict(size=13, color=T["text_primary"]),
        x         = 0,
        xanchor   = "left",
        pad       = dict(b=4),
    )
    monthly_layout["yaxis"].update(range=[0, 105], title="% Days Congested")
    monthly_layout["xaxis"].update(showgrid=False)
    monthly_layout.pop("legend", None)

    fig_monthly.update_layout(**monthly_layout)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ── Weekday vs weekend pattern — only if the difference is meaningful ────
    if "is_weekend" in pm.columns:
        wd_stats  = pm.groupby("is_weekend")["congestion_label"].mean() * 100
        wd_rate   = float(wd_stats.get(0, 0))
        we_rate   = float(wd_stats.get(1, 0))
        values_wd = [wd_rate, we_rate]

        if abs(wd_rate - we_rate) >= 5:
            fig_wd = go.Figure(go.Bar(
                x             = ["Weekday", "Weekend"],
                y             = values_wd,
                marker_color  = [T["accent"], T["risk_moderate"]],
                text          = [f"{v:.1f}%" for v in values_wd],
                textposition  = "auto",
                hovertemplate = "%{x}: %{y:.1f}% congested<extra></extra>",
            ))

            wd_layout = plotly_layout(height=190, margin=dict(l=40, r=20, t=36, b=30))
            wd_layout["title"] = dict(
                text    = "Weekday vs Weekend Congestion Rate",
                font    = dict(size=13, color=T["text_primary"]),
                x       = 0,
                xanchor = "left",
                pad     = dict(b=4),
            )
            wd_layout["yaxis"].update(range=[0, 105], title="% Days Congested")
            wd_layout["xaxis"].update(showgrid=False)
            wd_layout.pop("legend", None)

            fig_wd.update_layout(**wd_layout)
            st.plotly_chart(fig_wd, use_container_width=True)
            st.caption(
                "Higher weekend congestion may indicate reduced port staffing "
                "creating backlogs resolved on weekdays."
            )



# ---------------------------------------------------------------------------
# Key Takeaway card
# ---------------------------------------------------------------------------

def _render_key_takeaway(
    port_master:   pd.DataFrame,
    port:          str,
    df:            pd.DataFrame,
) -> None:
    """
    Single-sentence operational summary shown immediately below the status strip.
    Generated from live data — never hardcoded.
    """
    latest_score = float(port_master["port_congestion_score"].iloc[-1])
    trend        = get_trend_direction(df, port)
    rl           = risk_label(latest_score)

    # Compose sentence based on risk level + trend direction
    if latest_score > 0.70:
        if "Worsening" in trend:
            sentence = (
                f"{port} shows elevated congestion risk "
                f"with worsening week-over-week conditions."
            )
        elif "Improving" in trend:
            sentence = (
                f"{port} shows elevated congestion risk "
                f"but conditions are improving week-over-week."
            )
        else:
            sentence = (
                f"{port} shows elevated congestion risk "
                f"with stable recent conditions."
            )
    elif latest_score > 0.40:
        if "Worsening" in trend:
            sentence = (
                f"{port} is at moderate congestion risk "
                f"with a worsening trend — monitor closely."
            )
        else:
            sentence = (
                f"{port} is operating at moderate congestion levels "
                f"with {trend.lower().replace('→ ', 'stable')} conditions."
            )
    else:
        sentence = (
            f"{port} is operating normally "
            f"with low congestion risk and stable conditions."
        )

    st.markdown(
        f'<div style="'
        f'background:{T["accent_dim"]};'
        f'border-left:4px solid {T["accent"]};'
        f'border-radius:0 6px 6px 0;'
        f'padding:10px 16px;margin:0 0 16px 0;">'
        f'<span style="font-size:11px;color:{T["text_secondary"]};'
        f'letter-spacing:0.5px;font-weight:600;">KEY TAKEAWAY</span><br>'
        f'<span style="font-size:14px;color:{T["text_primary"]};'
        f'font-weight:500;">{sentence}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Port Insights box
# ---------------------------------------------------------------------------

def _render_port_insights(
    port_master:   pd.DataFrame,
    port_external: pd.DataFrame,
    port:          str,
    df:            pd.DataFrame,
) -> None:
    """
    Up to 4 auto-generated operational insight bullets for the selected port.
    Rendered inside a card — no hardcoded values.
    """
    insights = []

    if len(port_master) == 0:
        return

    scores = port_master["port_congestion_score"]

    # 1. Week-over-week trend
    if len(port_master) >= 14:
        last7  = scores.iloc[-7:].mean()
        prior7 = scores.iloc[-14:-7].mean()
        if prior7 > 0:
            pct = (last7 - prior7) / prior7 * 100
            if abs(pct) >= 5:
                direction = "increased" if pct > 0 else "decreased"
                icon = "📈" if pct > 0 else "📉"
                insights.append(
                    f"{icon} {port} congestion risk {direction} "
                    f"{abs(pct):.0f}% week-over-week"
                )

    # 2. Recent history context — is the latest score near the observed peak?
    latest = float(scores.iloc[-1])
    peak   = float(scores.max())
    if peak > 0 and latest / peak >= 0.90:
        insights.append(
            f"🔴 {port} is currently near its highest observed congestion "
            f"level in the monitoring period ({latest:.3f} vs peak {peak:.3f})"
        )

    # 3. Weather–congestion coincidence
    if (
        len(port_external) > 0
        and "max_weather_severity_score" in port_external.columns
        and "congestion_label" in port_master.columns
    ):
        merged = port_master[["date", "congestion_label"]].merge(
            port_external[["date", "max_weather_severity_score"]],
            on="date", how="inner",
        )
        if len(merged) > 0:
            # Congested days with above-median weather severity
            median_sev     = merged["max_weather_severity_score"].median()
            adverse_during = (
                (merged["congestion_label"] == 1)
                & (merged["max_weather_severity_score"] > median_sev)
            ).sum()
            total_congested = int(merged["congestion_label"].sum())

            if total_congested > 0:
                pct_co = adverse_during / total_congested * 100
                if pct_co >= 50:
                    insights.append(
                        f"🌦 Adverse weather coincided with {pct_co:.0f}% "
                        f"of congested days — weather may be a contributing factor"
                    )

    # 4. Congested-day percentage
    if "congestion_label" in port_master.columns:
        n_days      = len(port_master)
        n_congested = int(port_master["congestion_label"].sum())
        pct         = n_congested / n_days * 100 if n_days > 0 else 0
        insights.append(
            f"📊 {pct:.0f}% of observed days were classified as congested "
            f"({n_congested} of {n_days} days)"
        )

    if not insights:
        insights.append("No significant operational signals detected for this period.")

    bullets = "".join(
        f'<div style="padding:5px 0;border-bottom:1px solid {T["border_light"]};'
        f'font-size:13px;color:{T["text_primary"]};">{item}</div>'
        for item in insights[:4]   # hard cap at 4
    )
    st.markdown(card(bullets, padding="14px 18px"), unsafe_allow_html=True)