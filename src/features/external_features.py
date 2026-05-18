"""
external_features.py
--------------------
Phase 2 — External Feature Engineering (Weather + Calendar)

Loads raw hourly weather data from data/raw/weather/, engineers
operational condition features, aggregates to daily port-level signals,
adds calendar features, then merges with port_features.parquet to
produce a combined external feature table.

These features capture congestion drivers that vessel AIS behavior
cannot explain: storms that halt crane operations, holiday slowdowns,
seasonal shipping patterns.

Features engineered
-------------------
Weather severity (hourly → daily aggregated):
    wind_severity          : normalised wind speed component (0–1)
    precip_severity        : normalised precipitation component (0–1)
    weathercode_severity   : severity derived from WMO weather code (0–1)
    weather_severity_score : composite of above three (0–1)

Operational risk flags (daily max):
    gale_warning_flag      : 1 if max wind speed ≥ 34 knots (Beaufort 8)
    heavy_rain_flag        : 1 if total daily precipitation ≥ 10 mm
    storm_flag             : 1 if any hourly weathercode ≥ 80 (heavy rain/storm)
    adverse_weather_day    : 1 if any of the above flags are set

Daily weather aggregations:
    max_windspeed          : maximum hourly wind speed that day (km/h)
    mean_windspeed         : mean hourly wind speed (km/h)
    total_precipitation    : sum of hourly precipitation (mm)
    max_weather_severity   : worst-hour severity score of the day
    mean_weather_severity  : average severity across all hours

Calendar features:
    month                  : 1–12
    quarter                : 1–4
    season                 : winter / spring / summer / autumn
    is_weekend             : 1 if Saturday or Sunday
    is_holiday             : 1 for major international shipping holidays

Usage (from project root):
    python src/features/external_features.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

# Wind speed thresholds (km/h)
# Beaufort scale: Force 8 (Gale) = 62–74 km/h → port crane operations cease
GALE_WIND_KMH       = 62.0
# Wind above this is severe but below gale (slows operations noticeably)
HIGH_WIND_KMH       = 40.0
# Normalisation ceiling — above this wind is treated as maximum severity
MAX_WIND_NORM_KMH   = 100.0

# Precipitation thresholds (mm per day)
HEAVY_RAIN_MM_DAY   = 10.0     # heavy rain flag threshold
MAX_PRECIP_NORM_MM  = 50.0     # normalisation ceiling for precipitation severity

# WMO Weather Interpretation Code severity mapping
# Codes: https://open-meteo.com/en/docs (weathercode variable)
# Groups → severity level (0.0 = clear, 1.0 = severe storm)
WMO_SEVERITY = {
    # Clear / mostly clear
    0: 0.00, 1: 0.05, 2: 0.10, 3: 0.15,
    # Fog
    45: 0.30, 48: 0.35,
    # Drizzle
    51: 0.15, 53: 0.20, 55: 0.25,
    # Freezing drizzle
    56: 0.30, 57: 0.35,
    # Rain
    61: 0.25, 63: 0.40, 65: 0.55,
    # Freezing rain
    66: 0.50, 67: 0.60,
    # Snow
    71: 0.35, 73: 0.50, 75: 0.65,
    77: 0.40,    # Snow grains
    # Rain showers
    80: 0.40, 81: 0.50, 82: 0.65,
    # Snow showers
    85: 0.55, 86: 0.70,
    # Thunderstorms
    95: 0.85,
    96: 0.90, 99: 1.00,
}

# Storm threshold — WMO codes at or above this indicate severe conditions
STORM_WMO_THRESHOLD = 80

# Weights for composite weather_severity_score (must sum to 1.0)
SEVERITY_WEIGHTS = {
    "wind":   0.50,    # wind is the primary port operations disruptor
    "precip": 0.25,
    "wmo":    0.25,
}

# Calendar: major dates when international port activity drops
# Format: (month, day) — year-independent
INTERNATIONAL_HOLIDAYS = {
    (1, 1),    # New Year's Day
    (1, 2),    # New Year's holiday (many countries)
    (5, 1),    # International Labour Day (major in EU/Asia ports)
    (12, 24),  # Christmas Eve
    (12, 25),  # Christmas Day
    (12, 26),  # Boxing Day
    (12, 31),  # New Year's Eve
}

# Season mapping by month (Northern Hemisphere conventions)
SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3:  "spring", 4: "spring", 5: "spring",
    6:  "summer", 7: "summer", 8: "summer",
    9:  "autumn", 10: "autumn", 11: "autumn",
}


# ---------------------------------------------------------------------------
# 1. Load Weather Data
# ---------------------------------------------------------------------------

def load_weather_data(raw_weather_dir: Path) -> pd.DataFrame:
    """
    Load all per-port weather CSVs and combine into a single DataFrame.

    Each file was saved by fetch_weather.py as:
        data/raw/weather/weather_<port_name>.csv

    Parameters
    ----------
    raw_weather_dir : Path
        Directory containing weather CSVs.

    Returns
    -------
    pd.DataFrame
        Combined hourly weather data for all ports.

    Raises
    ------
    FileNotFoundError
        If no weather CSV files are found.
    """
    csv_files = sorted(raw_weather_dir.glob("weather_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No weather CSV files found in {raw_weather_dir}.\n"
            "Run: python src/ingestion/fetch_weather.py"
        )

    log.info(f"Found {len(csv_files)} weather file(s)")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        log.info(f"  {path.name}: {len(df):,} rows, port='{df['port'].iloc[0]}'")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total weather rows: {len(combined):,}")

    return combined


# ---------------------------------------------------------------------------
# 2. Parse and Validate Timestamps
# ---------------------------------------------------------------------------

def parse_weather_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'timestamp' column to datetime and validate coverage.

    Open-Meteo returns ISO 8601 strings: "2023-01-01T00:00"

    Parameters
    ----------
    df : pd.DataFrame
        Raw weather DataFrame with 'timestamp' string column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'timestamp' as datetime64[ns].
    """
    log.info("Parsing weather timestamps ...")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    failed = df["timestamp"].isnull().sum()
    if failed > 0:
        log.warning(f"  {failed:,} rows failed timestamp parsing — dropping")
        df = df.dropna(subset=["timestamp"])

    log.info(f"  Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"  Hourly rows per port:")

    rows_per_port = df.groupby("port").size()
    for port, count in rows_per_port.items():
        expected = 24 * 365    # 8,760 for a full year
        pct = count / expected * 100
        log.info(f"    {port:<25} {count:>6,} rows  ({pct:.0f}% of full year)")

    return df


# ---------------------------------------------------------------------------
# 3. Hourly Weather Severity Features
# ---------------------------------------------------------------------------

def add_weather_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite hourly weather_severity_score (0–1) from wind,
    precipitation, and WMO weather code.

    Each component is normalised independently to [0, 1] before weighting,
    so no single variable can dominate due to scale differences.

    Components
    ----------
    wind_severity        : windspeed_10m / MAX_WIND_NORM_KMH, clipped to [0, 1]
    precip_severity      : precipitation (hourly mm) / (MAX_PRECIP_NORM_MM / 24),
                           clipped to [0, 1]
    weathercode_severity : looked up from WMO_SEVERITY dict; unknown codes → 0.1

    Parameters
    ----------
    df : pd.DataFrame
        Hourly weather DataFrame with windspeed_10m, precipitation,
        weathercode columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with severity component columns and composite score added.
    """
    log.info("Computing hourly weather severity scores ...")

    # ── Check which columns are available ───────────────────────────────────
    available = df.columns.tolist()
    log.info(f"  Available weather columns: {available}")

    # ── Wind severity ────────────────────────────────────────────────────────
    if "windspeed_10m" in df.columns:
        wind = pd.to_numeric(df["windspeed_10m"], errors="coerce").fillna(0.0)
        df["wind_severity"] = (wind / MAX_WIND_NORM_KMH).clip(0, 1).round(4)
    else:
        log.warning("  'windspeed_10m' not found — wind_severity set to 0")
        df["wind_severity"] = 0.0

    # ── Precipitation severity ───────────────────────────────────────────────
    # Open-Meteo gives hourly mm. Normalise against hourly equivalent of daily cap.
    hourly_precip_cap = MAX_PRECIP_NORM_MM / 24.0    # ~2.08 mm/hour at maximum

    if "precipitation" in df.columns:
        precip = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0.0)
        df["precip_severity"] = (precip / hourly_precip_cap).clip(0, 1).round(4)
    else:
        log.warning("  'precipitation' not found — precip_severity set to 0")
        df["precip_severity"] = 0.0

    # ── WMO code severity ────────────────────────────────────────────────────
    if "weathercode" in df.columns:
        wmo = pd.to_numeric(df["weathercode"], errors="coerce").fillna(0).astype(int)
        # Map codes to severity; unknown codes default to 0.1 (slight uncertainty)
        df["weathercode_severity"] = wmo.map(WMO_SEVERITY).fillna(0.1).round(4)
    else:
        log.warning("  'weathercode' not found — weathercode_severity set to 0")
        df["weathercode_severity"] = 0.0

    # ── Composite score ──────────────────────────────────────────────────────
    df["weather_severity_score"] = (
        SEVERITY_WEIGHTS["wind"]   * df["wind_severity"]
        + SEVERITY_WEIGHTS["precip"] * df["precip_severity"]
        + SEVERITY_WEIGHTS["wmo"]    * df["weathercode_severity"]
    ).round(4)

    log.info(f"  Severity score — mean: {df['weather_severity_score'].mean():.4f}, "
             f"max: {df['weather_severity_score'].max():.4f}")

    return df


# ---------------------------------------------------------------------------
# 4. Hourly Operational Risk Flags
# ---------------------------------------------------------------------------

def add_operational_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary hourly flags for conditions that directly halt or impair
    port crane and vessel operations.

    Flags
    -----
    gale_warning_flag : 1 if windspeed_10m ≥ GALE_WIND_KMH (62 km/h)
                        At Beaufort 8, most container crane operations cease.
    heavy_rain_flag   : computed at daily level — placeholder here (set to 0)
                        (daily total is more meaningful than hourly threshold)
    storm_flag        : 1 if weathercode ≥ STORM_WMO_THRESHOLD (80)
                        WMO ≥ 80 = heavy rain showers, thunderstorms

    Parameters
    ----------
    df : pd.DataFrame
        Hourly weather DataFrame with wind and weathercode columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with risk flag columns added.
    """
    log.info("Adding hourly operational risk flags ...")

    # Gale warning flag
    if "windspeed_10m" in df.columns:
        wind = pd.to_numeric(df["windspeed_10m"], errors="coerce").fillna(0.0)
        df["gale_warning_flag"] = (wind >= GALE_WIND_KMH).astype(np.int8)
    else:
        df["gale_warning_flag"] = np.int8(0)

    # Storm flag from WMO code
    if "weathercode" in df.columns:
        wmo = pd.to_numeric(df["weathercode"], errors="coerce").fillna(0).astype(int)
        df["storm_flag"] = (wmo >= STORM_WMO_THRESHOLD).astype(np.int8)
    else:
        df["storm_flag"] = np.int8(0)

    log.info(f"  gale_warning_flag : {df['gale_warning_flag'].sum():,} hourly pings "
             f"({df['gale_warning_flag'].mean()*100:.2f}%)")
    log.info(f"  storm_flag        : {df['storm_flag'].sum():,} hourly pings "
             f"({df['storm_flag'].mean()*100:.2f}%)")

    return df


# ---------------------------------------------------------------------------
# 5. Aggregate Hourly → Daily
# ---------------------------------------------------------------------------

def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly weather features to daily port-level observations.

    Aggregation strategy (per column):
    - Severity scores  → max (worst hour of the day matters most)
    - Wind speed       → max AND mean (both are informative)
    - Precipitation    → sum (total daily accumulation)
    - Risk flags       → max (1 if any hour triggered the flag)

    Parameters
    ----------
    df : pd.DataFrame
        Hourly weather DataFrame with engineered severity and flag columns.

    Returns
    -------
    pd.DataFrame
        Daily weather DataFrame, one row per port per day.
    """
    log.info("Aggregating hourly weather to daily port observations ...")

    df["date"] = df["timestamp"].dt.normalize()

    # Build aggregation spec dynamically — only include columns that exist
    agg_spec = {}

    # Severity aggregations
    for col in ["weather_severity_score", "wind_severity",
                "precip_severity", "weathercode_severity"]:
        if col in df.columns:
            agg_spec[f"max_{col}"]  = (col, "max")
            agg_spec[f"mean_{col}"] = (col, "mean")

    # Wind speed
    if "windspeed_10m" in df.columns:
        agg_spec["max_windspeed"]  = ("windspeed_10m", "max")
        agg_spec["mean_windspeed"] = ("windspeed_10m", "mean")

    # Precipitation (daily total)
    if "precipitation" in df.columns:
        agg_spec["total_precipitation"] = ("precipitation", "sum")

    # Risk flags (any-hour → 1)
    for flag in ["gale_warning_flag", "storm_flag"]:
        if flag in df.columns:
            agg_spec[flag] = (flag, "max")

    daily = (
        df.groupby(["port", "date"])
        .agg(**agg_spec)
        .reset_index()
    )

    # ── Heavy rain flag (daily total precipitation threshold) ────────────────
    if "total_precipitation" in daily.columns:
        daily["heavy_rain_flag"] = (
            daily["total_precipitation"] >= HEAVY_RAIN_MM_DAY
        ).astype(np.int8)
    else:
        daily["heavy_rain_flag"] = np.int8(0)

    # ── Adverse weather day composite flag ───────────────────────────────────
    # 1 if any of the three primary risk flags are triggered
    flag_cols = [c for c in ["gale_warning_flag", "heavy_rain_flag", "storm_flag"]
                 if c in daily.columns]
    if flag_cols:
        daily["adverse_weather_day"] = (
            daily[flag_cols].max(axis=1)
        ).astype(np.int8)
    else:
        daily["adverse_weather_day"] = np.int8(0)

    # Round numeric columns for clean output
    numeric_cols = daily.select_dtypes(include="number").columns
    daily[numeric_cols] = daily[numeric_cols].round(4)

    log.info(f"  Daily rows: {len(daily):,}")
    log.info(f"  Ports: {daily['port'].nunique()}")
    log.info(f"  Adverse weather days: {daily['adverse_weather_day'].sum():,} "
             f"({daily['adverse_weather_day'].mean()*100:.1f}% of port-days)")

    return daily


# ---------------------------------------------------------------------------
# 6. Calendar Features
# ---------------------------------------------------------------------------

def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based operational pattern features to the daily table.

    Ports follow predictable calendar rhythms:
    - Public holidays reduce throughput significantly (skeleton crew)
    - Q4 (Oct–Dec) is peak shipping season globally
    - Winter storms are more common in Q1 at northern hemisphere ports
    - Weekend operations run on reduced staff at most ports

    Features added
    --------------
    month        : int 1–12
    quarter      : int 1–4
    season       : string (winter / spring / summer / autumn)
    is_weekend   : 1 if Saturday (5) or Sunday (6)
    is_holiday   : 1 if (month, day) in INTERNATIONAL_HOLIDAYS

    Parameters
    ----------
    daily : pd.DataFrame
        Daily weather DataFrame with 'date' column (datetime).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with calendar feature columns added.
    """
    log.info("Adding calendar features ...")

    daily["month"]    = daily["date"].dt.month
    daily["quarter"]  = daily["date"].dt.quarter
    daily["season"]   = daily["month"].map(SEASON_MAP)
    daily["weekday"]  = daily["date"].dt.dayofweek     # 0=Mon, 6=Sun
    daily["is_weekend"] = (daily["weekday"] >= 5).astype(np.int8)

    # Holiday flag — year-independent check on (month, day) tuples
    month_day = list(zip(daily["date"].dt.month, daily["date"].dt.day))
    daily["is_holiday"] = pd.array(
        [1 if md in INTERNATIONAL_HOLIDAYS else 0 for md in month_day],
        dtype=np.int8,
    )

    log.info(f"  Weekend days : {daily['is_weekend'].sum():,}")
    log.info(f"  Holiday days : {daily['is_holiday'].sum():,}")
    log.info(f"  Season distribution:")
    for season, count in daily["season"].value_counts().items():
        log.info(f"    {season:<10} {count:>6,} port-days")

    return daily


# ---------------------------------------------------------------------------
# 7. Merge with Port Features
# ---------------------------------------------------------------------------

def merge_with_port_features(
    weather_daily: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Left-join weather + calendar features onto port_features.parquet.

    The join key is (port name, date). Port features use 'nearest_port'
    as the port identifier; weather uses 'port'. We align these before merging.

    If port_features.parquet does not exist yet, return the weather-only
    table with a warning so the file can still be saved independently.

    Parameters
    ----------
    weather_daily : pd.DataFrame
        Daily weather + calendar feature table.
    processed_dir : Path
        Path to data/processed/ directory.

    Returns
    -------
    pd.DataFrame
        Merged table containing both port AIS features and external features,
        or weather-only table if port_features.parquet is not yet available.
    """
    port_features_path = processed_dir / "port_features.parquet"

    if not port_features_path.exists():
        log.warning(
            "port_features.parquet not found — saving external features standalone.\n"
            "Run port_features.py first, then re-run this script to get the merged table."
        )
        return weather_daily

    log.info(f"Merging with port_features.parquet ...")

    port_df = pd.read_parquet(port_features_path, engine="pyarrow")
    log.info(f"  Port features loaded: {len(port_df):,} rows")

    # Align port name column — port_features uses 'nearest_port', weather uses 'port'
    weather_daily = weather_daily.rename(columns={"port": "nearest_port"})

    # Ensure date types match before joining
    port_df["date"]        = pd.to_datetime(port_df["date"]).dt.normalize()
    weather_daily["date"]  = pd.to_datetime(weather_daily["date"]).dt.normalize()

    merged = port_df.merge(
        weather_daily,
        on=["nearest_port", "date"],
        how="left",
        suffixes=("", "_weather"),
    )

    # Log merge quality
    weather_cols = [c for c in weather_daily.columns
                    if c not in ("nearest_port", "date")]
    matched = merged[weather_cols[0]].notna().sum() if weather_cols else 0
    log.info(f"  Merged rows         : {len(merged):,}")
    log.info(f"  Rows with weather   : {matched:,} ({matched/len(merged)*100:.1f}%)")

    unmatched = len(merged) - matched
    if unmatched > 0:
        log.warning(
            f"  {unmatched:,} port-day rows have no weather data.\n"
            "  This usually means the port name in config.yaml doesn't match\n"
            "  the 'port' field in the weather CSV. Check for spelling differences."
        )

    return merged


# ---------------------------------------------------------------------------
# 8. Save Output
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the external feature DataFrame to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        External (weather + calendar) feature table.
    output_path : Path
        Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving {len(df):,} rows → {output_path} ...")
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"  ✓ Saved — file size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# 9. Summary Report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """
    Print a validation summary of the external feature table.
    """
    expected_cols = [
        "max_weather_severity_score", "mean_weather_severity_score",
        "max_windspeed", "mean_windspeed",
        "total_precipitation",
        "gale_warning_flag", "heavy_rain_flag", "storm_flag", "adverse_weather_day",
        "month", "quarter", "season", "is_weekend", "is_holiday",
    ]

    print("\n" + "=" * 60)
    print("  EXTERNAL FEATURES — ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"  Total rows    : {len(df):,}")

    port_col = "nearest_port" if "nearest_port" in df.columns else "port"
    if port_col in df.columns:
        print(f"  Ports covered : {df[port_col].nunique()}")

    print(f"  Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
    print()

    print("  Engineered feature status:")
    for col in expected_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            null_info  = f"  ({null_count:,} nulls)" if null_count > 0 else ""
            print(f"    ✓  {col:<35} dtype={df[col].dtype}{null_info}")
        else:
            print(f"    –  {col:<35} not present (column may be named differently)")

    print()
    print("  Adverse conditions summary (port-days):")

    for flag in ["gale_warning_flag", "heavy_rain_flag", "storm_flag", "adverse_weather_day"]:
        if flag in df.columns:
            count = df[flag].sum()
            pct   = df[flag].mean() * 100
            print(f"    {flag:<30} {count:>6,}  ({pct:.1f}%)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    raw_weather_dir = PROJECT_ROOT / config["paths"]["raw_weather"]
    processed_dir   = PROJECT_ROOT / config["paths"]["processed"]
    output_path     = processed_dir / "external_features.parquet"

    log.info("=" * 60)
    log.info("Phase 2 — External Feature Engineering")
    log.info("=" * 60)

    # Step 1: Load raw weather CSVs
    df = load_weather_data(raw_weather_dir)

    # Step 2: Parse timestamps
    df = parse_weather_timestamps(df)

    # Step 3: Hourly severity scores
    df = add_weather_severity_score(df)

    # Step 4: Hourly operational risk flags
    df = add_operational_risk_flags(df)

    # Step 5: Aggregate hourly → daily
    daily = aggregate_to_daily(df)

    # Step 6: Calendar features
    daily = add_calendar_features(daily)

    # Step 7: Merge with port features (if available)
    merged = merge_with_port_features(daily, processed_dir)

    # Step 8: Summary
    print_summary(merged)

    # Step 9: Save
    save_features(merged, output_path)

    log.info("Phase 2 Step 3 complete — external_features.parquet ready.")
    log.info("Next: run src/pipeline/build_master_dataset.py")


if __name__ == "__main__":
    main()
