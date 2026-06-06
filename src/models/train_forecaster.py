"""
train_forecaster.py
-------------------
Phase 3 — Exploratory Congestion Trend Forecaster (Prophet)

Trains one Prophet time series model per port using the daily
port_congestion_score from port_features.parquet.

IMPORTANT FRAMING — READ BEFORE INTERPRETING RESULTS:
------------------------------------------------------
This forecaster is EXPLORATORY, not production-grade. Reasons:

1. Only 90 days of training data.
   Prophet works best with 2+ years to detect annual seasonality
   and multiple seasonal repetitions. With 90 days you get:
     - Trend component (reliable)
     - Weekly seasonality (weak — only ~13 weekly cycles observed)
     - NO reliable annual seasonality (disabled intentionally)

2. In-sample MAE is reported but overstates true forecast accuracy.
   A model evaluated on data it was trained on always looks better
   than it will on genuinely new data.

3. New York anomaly:
   LOPO evaluation showed New York has fundamentally different
   congestion dynamics from other ports (LOPO AUC=0.50).
   Its Prophet forecast should be treated with extra caution —
   the trend it learned may not reflect actual future behaviour.

What this forecaster IS useful for:
  - Visualising recent congestion trend per port
  - Identifying weekly operational patterns (weekend dips)
  - Dashboard "what direction is congestion heading?" indicator
  - Demonstrating end-to-end ML pipeline capability

What it is NOT:
  - A production forecast with quantified accuracy guarantees
  - A replacement for domain expert judgment
  - Reliable beyond 7 days with only 90 training observations

Model artifacts saved
---------------------
  models/prophet_houston.pkl
  models/prophet_los_angeles.pkl
  models/prophet_new_york.pkl
  models/prophet_savannah.pkl
  models/prophet_seattle.pkl

Chart outputs saved
-------------------
  outputs/forecast_houston.png
  outputs/forecast_los_angeles.png
  outputs/forecast_new_york.png
  outputs/forecast_savannah.png
  outputs/forecast_seattle.png

Usage (from project root):
  python src/models/train_forecaster.py
"""

import logging
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Suppress Prophet's verbose Stan output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Forecast horizon in days
FORECAST_HORIZON = 14

# Minimum training rows required per port to fit Prophet
# Prophet needs at least 2 full weekly cycles to detect weekly seasonality
MIN_TRAINING_ROWS = 14

# Prophet configuration
# yearly_seasonality=False — honest: 90 days cannot detect annual cycle
# weekly_seasonality=True  — 13 weekly cycles in 90 days, weak but present
# daily_seasonality=False  — we have daily data, not hourly
PROPHET_CONFIG = {
    "yearly_seasonality":  False,
    "weekly_seasonality":  True,
    "daily_seasonality":   False,
    "seasonality_mode":    "additive",  # additive fits operational data better
                                        # than multiplicative when values near 0
    "changepoint_prior_scale": 0.05,    # conservative — prevents overfitting
                                        # trend to noise in 90-day window
    "seasonality_prior_scale": 5.0,     # moderate weekly seasonality flexibility
    "uncertainty_samples":     500,     # number of samples for confidence intervals
}

# Ports where forecasts should carry extra caution warnings
# Based on LOPO evaluation results from evaluate_classifier.py
HIGH_CAUTION_PORTS = {"New York"}

# Colors for chart components
COLOR_ACTUAL    = "#1565C0"   # dark blue
COLOR_FORECAST  = "#E53935"   # red
COLOR_CI        = "#FFCDD2"   # light red for confidence interval
COLOR_TREND     = "#43A047"   # green for trend component


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------

def load_port_features(processed_dir: Path) -> pd.DataFrame:
    """
    Load port_features.parquet and extract the daily congestion score
    time series per port.

    We use port_congestion_score (raw, not normalised) as the Prophet
    target variable. The raw score is bounded [0, 1] which Prophet
    handles well in additive mode.

    Parameters
    ----------
    processed_dir : Path
        Path to data/processed/ directory.

    Returns
    -------
    pd.DataFrame
        Port features with 'nearest_port', 'date', 'port_congestion_score'.
    """
    path = processed_dir / "port_features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"port_features.parquet not found at {path}.\n"
            "Run port_features.py before train_forecaster.py."
        )

    log.info(f"Loading port_features.parquet ...")
    df = pd.read_parquet(path, engine="pyarrow")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["nearest_port", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    ports = sorted(df["nearest_port"].unique())
    log.info(f"  Ports   : {ports}")
    log.info(f"  Rows    : {len(df):,}")
    log.info(f"  Dates   : {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"  Score range: {df['port_congestion_score'].min():.4f} → "
             f"{df['port_congestion_score'].max():.4f}")

    return df


# ---------------------------------------------------------------------------
# 2. Prepare Prophet Input for One Port
# ---------------------------------------------------------------------------

def prepare_prophet_input(df: pd.DataFrame, port: str) -> pd.DataFrame:
    """
    Extract and format the time series for a single port into
    Prophet's required format: columns 'ds' (date) and 'y' (value).

    Handles missing dates by forward-filling with the previous day's
    score — Prophet requires a complete date range without gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Full port features DataFrame.
    port : str
        Port name to extract.

    Returns
    -------
    pd.DataFrame
        Prophet-formatted DataFrame with 'ds' and 'y' columns.
    """
    port_df = df[df["nearest_port"] == port][["date", "port_congestion_score"]].copy()
    port_df = port_df.rename(columns={"date": "ds", "port_congestion_score": "y"})
    port_df.sort_values("ds", inplace=True)
    port_df.reset_index(drop=True, inplace=True)

    # Fill any missing dates in the range
    full_range = pd.date_range(port_df["ds"].min(), port_df["ds"].max(), freq="D")
    if len(full_range) > len(port_df):
        port_df = port_df.set_index("ds").reindex(full_range).rename_axis("ds").reset_index()
        port_df["y"] = port_df["y"].fillna(method="ffill")
        log.warning(f"  {port}: {len(full_range) - len(port_df)} missing dates filled via ffill")

    return port_df


# ---------------------------------------------------------------------------
# 3. Train Prophet Model
# ---------------------------------------------------------------------------

def train_prophet(prophet_df: pd.DataFrame, port: str) -> Prophet:
    """
    Fit a Prophet model on the port's congestion score time series.

    Configuration rationale:
    - yearly_seasonality=False: 90 days is insufficient to estimate
      annual patterns. Enabling it would overfit to noise.
    - weekly_seasonality=True: 13 weekly cycles in 90 days provides
      weak but real signal — ports are slower on weekends.
    - changepoint_prior_scale=0.05: conservative trend flexibility.
      Prevents overfitting trend changes to short-window noise.
    - seasonality_mode='additive': appropriate when the seasonal
      variation is roughly constant regardless of trend level.

    Parameters
    ----------
    prophet_df : pd.DataFrame
        Prophet-formatted DataFrame (ds, y).
    port : str
        Port name (for logging).

    Returns
    -------
    Prophet
        Fitted Prophet model.
    """
    log.info(f"  Fitting Prophet for {port} ...")
    log.info(f"    Training rows : {len(prophet_df)}")
    log.info(f"    y range       : {prophet_df['y'].min():.4f} → {prophet_df['y'].max():.4f}")
    log.info(f"    y mean        : {prophet_df['y'].mean():.4f}")

    model = Prophet(**PROPHET_CONFIG)
    model.fit(prophet_df)

    return model


# ---------------------------------------------------------------------------
# 4. Generate Forecast
# ---------------------------------------------------------------------------

def generate_forecast(
    model: Prophet,
    prophet_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Generate a forecast for the next `horizon` days beyond the
    training period end date.

    Returns a DataFrame containing both historical fitted values
    and future predictions with confidence intervals.

    Parameters
    ----------
    model : Prophet
        Fitted Prophet model.
    prophet_df : pd.DataFrame
        Original training data (used to align historical plot).
    horizon : int
        Number of days to forecast beyond training data.

    Returns
    -------
    pd.DataFrame
        Prophet forecast DataFrame with columns:
        ds, yhat, yhat_lower, yhat_upper, trend, weekly
    """
    # Make future dataframe extending beyond training period
    future   = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)

    # Clip predictions to valid score range [0, 1]
    # Prophet can predict outside this range for bounded targets
    forecast["yhat"]       = forecast["yhat"].clip(0, 1)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(0, 1)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(0, 1)

    return forecast


# ---------------------------------------------------------------------------
# 5. Compute In-Sample Metrics
# ---------------------------------------------------------------------------

def compute_in_sample_metrics(
    prophet_df: pd.DataFrame,
    forecast: pd.DataFrame,
) -> dict:
    """
    Compute in-sample MAE and RMSE by comparing Prophet's fitted
    values against actual training observations.

    CRITICAL CAVEAT: In-sample metrics always overstate true forecast
    accuracy because the model was trained on this exact data.
    These metrics indicate goodness-of-fit, not predictive power.
    True predictive accuracy requires out-of-sample evaluation,
    which is not reliable with only 90 training days.

    Parameters
    ----------
    prophet_df : pd.DataFrame
        Actual training data (ds, y).
    forecast : pd.DataFrame
        Prophet forecast output.

    Returns
    -------
    dict
        MAE and RMSE on training data.
    """
    # Align forecast with actual values on training dates only
    train_forecast = forecast[forecast["ds"].isin(prophet_df["ds"])].copy()
    merged = prophet_df.merge(train_forecast[["ds", "yhat"]], on="ds")

    mae  = (merged["y"] - merged["yhat"]).abs().mean()
    rmse = ((merged["y"] - merged["yhat"]) ** 2).mean() ** 0.5

    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4)}


# ---------------------------------------------------------------------------
# 6. Prophet Cross-Validation (if enough data)
# ---------------------------------------------------------------------------

def run_prophet_cv(
    model: Prophet,
    prophet_df: pd.DataFrame,
    port: str,
) -> dict | None:
    """
    Run Prophet's built-in cross-validation if sufficient data exists.

    Prophet CV uses a sliding window: trains on initial period,
    forecasts horizon, steps forward, repeats.

    With only 90 days, this is very constrained:
      initial = 60 days (minimum training window)
      horizon = 7 days  (short — 14 days leaves too little data)
      period  = 15 days (step size between CV cutoffs)

    This gives approximately 2-3 CV cutpoints — extremely small.
    Results are directional only.

    Returns None if insufficient data for even one CV fold.

    Parameters
    ----------
    model : Prophet
        Fitted Prophet model.
    prophet_df : pd.DataFrame
        Training data.
    port : str
        Port name for logging.

    Returns
    -------
    dict or None
        CV MAE and coverage, or None if CV was skipped.
    """
    n_days = len(prophet_df)

    # Need at least 75 days for meaningful CV with these parameters
    if n_days < 75:
        log.warning(f"    {port}: only {n_days} days — skipping CV (need ≥75)")
        return None

    try:
        log.info(f"    Running Prophet CV (initial=60d, horizon=7d, period=15d) ...")
        df_cv = cross_validation(
            model,
            initial="60 days",
            horizon="7 days",
            period="15 days",
            disable_tqdm=True,
        )
        metrics_cv = performance_metrics(df_cv)

        mae_cv      = metrics_cv["mae"].mean()
        coverage_cv = metrics_cv["coverage"].mean()

        log.info(f"    CV MAE (7-day horizon): {mae_cv:.4f}")
        log.info(f"    CV coverage (80% CI)  : {coverage_cv:.2f}")

        return {
            "cv_mae_7d":  round(float(mae_cv), 4),
            "cv_coverage": round(float(coverage_cv), 4),
            "cv_cutpoints": len(df_cv["cutoff"].unique()),
        }

    except Exception as e:
        log.warning(f"    {port}: Prophet CV failed — {e}")
        return None


# ---------------------------------------------------------------------------
# 7. Plot Forecast
# ---------------------------------------------------------------------------

def plot_forecast(
    prophet_df: pd.DataFrame,
    forecast: pd.DataFrame,
    model: Prophet,
    port: str,
    metrics: dict,
    cv_metrics: dict | None,
    outputs_dir: Path,
) -> None:
    """
    Generate a two-panel forecast chart:
      Panel 1: Actual vs fitted/forecast with confidence interval
      Panel 2: Trend + weekly seasonality components

    Chart includes honest framing text noting the exploratory nature
    of the forecast and the 90-day training data limitation.

    Parameters
    ----------
    prophet_df : pd.DataFrame
        Actual training data.
    forecast : pd.DataFrame
        Prophet forecast output.
    model : Prophet
        Fitted Prophet model (for component plots).
    port : str
        Port name.
    metrics : dict
        In-sample MAE and RMSE.
    cv_metrics : dict or None
        Cross-validation metrics if available.
    outputs_dir : Path
        Directory to save the chart.
    """
    is_high_caution = port in HIGH_CAUTION_PORTS
    caution_text    = (
        "⚠ HIGH CAUTION: LOPO AUC=0.50 — Atlantic corridor anomaly detected.\n"
        "This port's congestion pattern differs from other ports in training data.\n"
        "Forecast reliability is lower than for other ports."
        if is_high_caution else
        "Exploratory forecast — 90 days training data.\n"
        "Weekly seasonality only. No annual cycle. Directional indicator only."
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Port Intelligence — Congestion Trend Forecast: {port}\n"
        f"EXPLORATORY ONLY — Prophet trained on 90 days (Jan–Mar 2023)",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: Actual vs Forecast ───────────────────────────────────────────
    ax1 = axes[0]

    train_end  = prophet_df["ds"].max()
    future_mask = forecast["ds"] > train_end
    hist_mask   = ~future_mask

    # Confidence interval — historical fitted
    ax1.fill_between(
        forecast[hist_mask]["ds"],
        forecast[hist_mask]["yhat_lower"],
        forecast[hist_mask]["yhat_upper"],
        alpha=0.2, color=COLOR_ACTUAL, label="_nolegend_"
    )

    # Confidence interval — future forecast
    ax1.fill_between(
        forecast[future_mask]["ds"],
        forecast[future_mask]["yhat_lower"],
        forecast[future_mask]["yhat_upper"],
        alpha=0.3, color=COLOR_CI, label=f"{FORECAST_HORIZON}-day CI (80%)"
    )

    # Fitted line — historical
    ax1.plot(
        forecast[hist_mask]["ds"],
        forecast[hist_mask]["yhat"],
        color=COLOR_ACTUAL, lw=1.5, alpha=0.7, label="Fitted (in-sample)"
    )

    # Forecast line — future
    ax1.plot(
        forecast[future_mask]["ds"],
        forecast[future_mask]["yhat"],
        color=COLOR_FORECAST, lw=2.5, linestyle="--",
        label=f"{FORECAST_HORIZON}-day forecast"
    )

    # Actual observations
    ax1.scatter(
        prophet_df["ds"], prophet_df["y"],
        color=COLOR_ACTUAL, s=18, zorder=5, alpha=0.7, label="Actual score"
    )

    # Vertical line at train/forecast boundary
    ax1.axvline(train_end, color="gray", linestyle=":", lw=1.5, alpha=0.7)
    ax1.text(
        train_end, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1.0,
        "  Forecast →", fontsize=8, color="gray", va="top"
    )

    # Score range
    ax1.set_ylim([0, 1.05])
    ax1.set_ylabel("Port Congestion Score", fontsize=10)
    ax1.set_xlabel("")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Metrics annotation box
    metrics_text = f"In-sample MAE: {metrics['mae']:.4f}\nIn-sample RMSE: {metrics['rmse']:.4f}"
    if cv_metrics:
        metrics_text += (
            f"\nCV MAE (7-day): {cv_metrics['cv_mae_7d']:.4f}"
            f"\nCV cutpoints: {cv_metrics['cv_cutpoints']}"
        )
    metrics_text += "\n⚠ In-sample metrics overstate accuracy"

    ax1.text(
        0.01, 0.98, metrics_text,
        transform=ax1.transAxes, fontsize=7.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Caution box
    caution_color = "#FFEBEE" if is_high_caution else "#E8F5E9"
    ax1.text(
        0.99, 0.98, caution_text,
        transform=ax1.transAxes, fontsize=7,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor=caution_color, alpha=0.9),
    )

    # ── Panel 2: Trend + Weekly Seasonality Components ────────────────────────
    ax2 = axes[1]

    # Plot trend
    ax2.plot(
        forecast["ds"], forecast["trend"],
        color=COLOR_TREND, lw=2, label="Trend"
    )

    # Overlay weekly seasonality as shaded signal if available
    if "weekly" in forecast.columns:
        weekly_scaled = forecast["weekly"] * 3   # scale for visibility
        ax2_twin = ax2.twinx()
        ax2_twin.fill_between(
            forecast["ds"], weekly_scaled,
            alpha=0.15, color="purple", label="Weekly seasonality (scaled)"
        )
        ax2_twin.axhline(0, color="purple", lw=0.5, alpha=0.5)
        ax2_twin.set_ylabel("Weekly seasonality (scaled ×3)", fontsize=8, color="purple")
        ax2_twin.tick_params(axis="y", labelcolor="purple", labelsize=7)
        ax2_twin.legend(fontsize=7, loc="upper right")

    # Mark train/forecast boundary on trend panel
    ax2.axvline(train_end, color="gray", linestyle=":", lw=1.5, alpha=0.7)

    ax2.set_ylabel("Congestion Score — Trend", fontsize=10)
    ax2.set_xlabel("Date", fontsize=9)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Add note about components
    ax2.text(
        0.01, 0.02,
        "Trend: long-term direction | Weekly: weekend vs weekday patterns\n"
        "Annual seasonality disabled — insufficient data (90 days) to estimate reliably",
        transform=ax2.transAxes, fontsize=7, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()

    port_slug = port.lower().replace(" ", "_")
    save_path = outputs_dir / f"forecast_{port_slug}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

    log.info(f"    ✓ forecast_{port_slug}.png saved")


# ---------------------------------------------------------------------------
# 8. Save Prophet Model
# ---------------------------------------------------------------------------

def save_prophet_model(model: Prophet, port: str, models_dir: Path) -> Path:
    """
    Save a fitted Prophet model to disk using pickle.

    Prophet models are saved as .pkl files because Prophet's internal
    Stan model is not joblib-compatible.

    Parameters
    ----------
    model : Prophet
        Fitted Prophet model.
    port : str
        Port name (used for filename).
    models_dir : Path
        Directory to save the model.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    port_slug  = port.lower().replace(" ", "_")
    model_path = models_dir / f"prophet_{port_slug}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    size_kb = model_path.stat().st_size / 1e3
    log.info(f"    ✓ prophet_{port_slug}.pkl saved ({size_kb:.1f} KB)")

    return model_path


# ---------------------------------------------------------------------------
# 9. Print Forecast Summary
# ---------------------------------------------------------------------------

def print_forecast_summary(all_results: list) -> None:
    """
    Print a concise summary table of all port forecasts including
    the 14-day predicted direction and confidence.
    """
    print("\n" + "=" * 75)
    print("  CONGESTION TREND FORECAST SUMMARY")
    print("  EXPLORATORY ONLY — 90 days training data, weekly seasonality only")
    print("=" * 75)
    print(f"  {'Port':<20} {'Last Score':>11} {'14-Day Forecast':>16} "
          f"{'Direction':>10} {'In-Samp MAE':>12}")
    print(f"  {'─'*70}")

    for r in all_results:
        port         = r["port"]
        last_actual  = r["last_actual"]
        forecast_end = r["forecast_end"]
        direction    = "↑ Rising " if forecast_end > last_actual + 0.01 else \
                       "↓ Falling" if forecast_end < last_actual - 0.01 else \
                       "→ Stable "
        caution      = " ⚠" if port in HIGH_CAUTION_PORTS else ""

        print(
            f"  {port + caution:<20} {last_actual:>11.4f} {forecast_end:>16.4f} "
            f"{direction:>10} {r['metrics']['mae']:>12.4f}"
        )

    print(f"  {'─'*70}")
    print(f"\n  ⚠ Forecast direction is exploratory — treat as trend indicator only")
    print(f"  ⚠ New York marked with ⚠ — LOPO anomaly, extra caution advised")
    print(f"  ⚠ In-sample MAE overstates true predictive accuracy")
    print("=" * 75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    processed_dir = PROJECT_ROOT / config["paths"]["processed"]
    models_dir    = PROJECT_ROOT / config["paths"]["models"]
    outputs_dir   = PROJECT_ROOT / "outputs"

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("Phase 3 — Exploratory Congestion Forecaster (Prophet)")
    log.info("FRAMING: Exploratory trend indicators only.")
    log.info("  - 90 days training data")
    log.info("  - Weekly seasonality only (annual disabled)")
    log.info("  - In-sample metrics overstate true accuracy")
    log.info("  - New York: extra caution (LOPO anomaly from evaluation)")
    log.info("=" * 65)

    # Load port features
    df    = load_port_features(processed_dir)
    ports = sorted(df["nearest_port"].unique())

    all_results = []

    for port in ports:
        print(f"\n{'─'*55}")
        log.info(f"Processing: {port}")

        if port in HIGH_CAUTION_PORTS:
            log.warning(
                f"  {port}: HIGH CAUTION PORT — LOPO AUC=0.50 detected during "
                f"classifier evaluation. This port has anomalous congestion "
                f"dynamics. Forecast is trained but should be treated with "
                f"extra scepticism."
            )

        # Step 1: Prepare Prophet input
        prophet_df = prepare_prophet_input(df, port)

        if len(prophet_df) < MIN_TRAINING_ROWS:
            log.warning(f"  {port}: only {len(prophet_df)} rows — skipping (need ≥{MIN_TRAINING_ROWS})")
            continue

        # Step 2: Train Prophet
        model = train_prophet(prophet_df, port)

        # Step 3: Generate forecast
        forecast = generate_forecast(model, prophet_df, FORECAST_HORIZON)

        # Step 4: In-sample metrics
        metrics = compute_in_sample_metrics(prophet_df, forecast)
        log.info(f"    In-sample MAE  : {metrics['mae']:.4f}")
        log.info(f"    In-sample RMSE : {metrics['rmse']:.4f}")
        log.info(f"    ⚠ In-sample metrics overstate true forecast accuracy")

        # Step 5: Prophet cross-validation (if enough data)
        cv_metrics = run_prophet_cv(model, prophet_df, port)

        # Step 6: Plot forecast
        plot_forecast(
            prophet_df, forecast, model,
            port, metrics, cv_metrics,
            outputs_dir,
        )

        # Step 7: Save model artifact
        save_prophet_model(model, port, models_dir)

        # Collect summary info
        last_actual  = float(prophet_df["y"].iloc[-1])
        forecast_end = float(
            forecast[forecast["ds"] > prophet_df["ds"].max()]["yhat"].iloc[-1]
        )

        all_results.append({
            "port":         port,
            "last_actual":  last_actual,
            "forecast_end": forecast_end,
            "metrics":      metrics,
            "cv_metrics":   cv_metrics,
        })

    # Final summary table
    print_forecast_summary(all_results)

    # Artifact inventory
    print("\n" + "=" * 65)
    print("  PROPHET ARTIFACTS SAVED")
    print("=" * 65)
    prophet_files = sorted(models_dir.glob("prophet_*.pkl"))
    for f in prophet_files:
        size_kb = f.stat().st_size / 1e3
        print(f"  {f.name:<45} {size_kb:>8.1f} KB")

    forecast_files = sorted(outputs_dir.glob("forecast_*.png"))
    print()
    for f in forecast_files:
        size_kb = f.stat().st_size / 1e3
        print(f"  {f.name:<45} {size_kb:>8.1f} KB")
    print("=" * 65)

    log.info("\nPhase 3 complete.")
    log.info("All three Phase 3 scripts have run successfully.")
    log.info("Artifacts ready for Phase 5 dashboard.")
    log.info("")
    log.info("Models directory contents:")
    for f in sorted(models_dir.iterdir()):
        log.info(f"  {f.name}")


if __name__ == "__main__":
    main()
