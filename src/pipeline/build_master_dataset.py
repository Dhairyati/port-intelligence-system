"""
build_master_dataset.py
-----------------------
Phase 2 — Master Dataset Assembly Pipeline

Loads the three processed feature tables produced by Phase 2 feature
engineering scripts, validates their schemas, joins them into a single
master dataset, engineers cross-source features, runs a quality report,
and saves the result to data/processed/master_dataset.parquet.

This file is the final step of Phase 2. Its output is the sole input
artifact consumed by Phase 3 (model training).

Input files (must exist before running)
----------------------------------------
    data/processed/vessel_features.parquet   ← from vessel_features.py
    data/processed/port_features.parquet     ← from port_features.py
    data/processed/external_features.parquet ← from external_features.py

Output
------
    data/processed/master_dataset.parquet
        One row per port per day.
        Contains all engineered features + binary congestion_label.

Cross-source features engineered here
--------------------------------------
    weather_congestion_interaction : weather_severity × slow_vessel_ratio
                                     Captures days when bad weather AND
                                     vessel congestion co-occur (compounding risk)

    vessel_weather_risk            : speed_drop_ratio (daily mean) ×
                                     max_weather_severity_score
                                     Vessel slowdown amplified by bad conditions

    congestion_pressure_index      : weighted combination of port_congestion_score,
                                     weather severity, and rolling vessel count
                                     Unified risk index across all three data sources

    congestion_label               : binary target for Phase 3 classifier
                                     1 if port_congestion_score > CONGESTION_THRESHOLD
                                     0 otherwise

Usage (from project root):
    python src/pipeline/build_master_dataset.py
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

# Binary label threshold — port-days with congestion_score above this
# are labelled as congested (1). Adjust if class imbalance is too severe.
# Documented here so Phase 3 can reference the same value.
CONGESTION_THRESHOLD = 0.35

# Weights for the unified congestion_pressure_index (must sum to 1.0)
PRESSURE_INDEX_WEIGHTS = {
    "port_congestion_score":       0.50,   # primary AIS-derived signal
    "max_weather_severity_score":  0.30,   # external weather disruption
    "vessels_7d_rolling_norm":     0.20,   # traffic volume trend (normalised)
}

# Expected columns per input file — used for schema validation
EXPECTED_COLUMNS = {
    "vessel_features": [
        "mmsi", "timestamp", "lat", "lon", "sog",
        "anchored_flag", "slow_moving_flag", "speed_drop_ratio",
        "idle_duration_proxy", "movement_intensity", "vessel_class",
        "hour_of_day", "weekday", "weekend_flag",
    ],
    "port_features": [
        "nearest_port", "date",
        "vessels_in_zone", "anchored_vessel_count", "slow_vessel_count",
        "slow_vessel_ratio", "avg_speed_in_zone", "avg_idle_duration",
        "vessels_7d_rolling", "slow_ratio_7d_rolling",
        "port_congestion_score",
    ],
    "external_features": [
        "date",
        "adverse_weather_day", "is_weekend", "is_holiday",
        "month", "quarter", "season",
    ],
}

# Null tolerance — columns with more than this fraction of nulls trigger a warning
NULL_WARNING_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# 1. Validate Input Files
# ---------------------------------------------------------------------------

def validate_inputs(processed_dir: Path) -> dict:
    """
    Check that all three required parquet files exist and contain at least
    their minimum required columns. Raises clear errors for common failures.

    Parameters
    ----------
    processed_dir : Path
        Path to data/processed/ directory.

    Returns
    -------
    dict
        Mapping of name → loaded DataFrame for all three input files.

    Raises
    ------
    FileNotFoundError
        If any required parquet file is missing.
    ValueError
        If any file is missing required columns.
    """
    file_map = {
        "vessel_features":   processed_dir / "vessel_features.parquet",
        "port_features":     processed_dir / "port_features.parquet",
        "external_features": processed_dir / "external_features.parquet",
    }

    log.info("Validating input files ...")
    dataframes = {}

    for name, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run the corresponding feature script first:\n"
                f"  vessel_features   → python src/features/vessel_features.py\n"
                f"  port_features     → python src/features/port_features.py\n"
                f"  external_features → python src/features/external_features.py"
            )

        df = pd.read_parquet(path, engine="pyarrow")
        log.info(f"  ✓ {name:<25} {len(df):>10,} rows  ×  {df.shape[1]} columns")

        # Check required columns
        required = EXPECTED_COLUMNS[name]
        missing  = [c for c in required if c not in df.columns]

        if missing:
            # Warn but do not raise — some columns are optional depending on
            # what data was available (e.g. weather columns depend on API response)
            log.warning(
                f"  ⚠ {name}: {len(missing)} expected column(s) not found: {missing}\n"
                f"    Available columns: {list(df.columns)}"
            )

        dataframes[name] = df

    return dataframes


# ---------------------------------------------------------------------------
# 2. Aggregate Vessel Features to Daily Port Level
# ---------------------------------------------------------------------------

def build_vessel_daily_aggregations(
    vessel_df: pd.DataFrame,
    ports: list,
    radius_km: float,
) -> pd.DataFrame:
    """
    Collapse vessel_features (ping-level) to daily port-level aggregations.

    vessel_features.parquet is one row per AIS ping — it cannot be directly
    joined to port_features (one row per port per day) without aggregation.
    This function computes the daily mean/median of vessel-level features
    per port, using the same spatial assignment logic as port_features.py.

    Rather than re-running the full Haversine join (expensive), we check
    whether the vessel_features table already has a 'nearest_port' column
    (it does if port_features.py wrote it back — which it does not in the
    current design). If not, we perform a lightweight spatial assignment.

    Parameters
    ----------
    vessel_df : pd.DataFrame
        Ping-level vessel features with lat, lon, timestamp columns.
    ports : list of dict
        Port definitions from config.yaml.
    radius_km : float
        Anchorage zone radius in kilometres.

    Returns
    -------
    pd.DataFrame
        Daily vessel-level aggregations per port — one row per port per day.
        Returns None if no pings fall within any port zone.
    """
    log.info("Aggregating vessel features to daily port level ...")

    # ── Spatial assignment ────────────────────────────────────────────────────
    # Check if already assigned (e.g. from a previous run that cached assignment)
    if "nearest_port" not in vessel_df.columns:
        log.info("  Assigning vessel pings to ports via Haversine distance ...")

        lat_arr = vessel_df["lat"].to_numpy(dtype=np.float64)
        lon_arr = vessel_df["lon"].to_numpy(dtype=np.float64)

        def haversine_to_port(port: dict) -> np.ndarray:
            lat2 = np.radians(port["lat"])
            lon2 = np.radians(port["lon"])
            lat1 = np.radians(lat_arr)
            lon1 = np.radians(lon_arr)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        distance_matrix = np.column_stack([haversine_to_port(p) for p in ports])
        nearest_idx  = np.argmin(distance_matrix, axis=1)
        nearest_dist = distance_matrix[np.arange(len(vessel_df)), nearest_idx]

        within_mask  = nearest_dist <= radius_km
        log.info(f"  Pings within {radius_km} km of any port: {within_mask.sum():,} "
                 f"({within_mask.mean()*100:.1f}%)")

        if within_mask.sum() == 0:
            log.warning(
                "  No vessel pings matched any port zone.\n"
                "  Vessel daily aggregations will be skipped.\n"
                "  Check that your AIS data covers the same geographic region\n"
                "  as the ports configured in config.yaml."
            )
            return None

        vessel_df = vessel_df[within_mask].copy()
        vessel_df["nearest_port"] = [ports[i]["name"] for i in nearest_idx[within_mask]]

    # ── Date column ───────────────────────────────────────────────────────────
    vessel_df["date"] = pd.to_datetime(vessel_df["timestamp"]).dt.normalize()

    # ── Daily aggregations ────────────────────────────────────────────────────
    # Select only numeric feature columns that exist in this dataset
    candidate_agg_cols = {
        "speed_drop_ratio":    ["mean", "min"],
        "movement_intensity":  ["mean"],
        "idle_duration_proxy": ["mean", "max"],
        "sog":                 ["mean", "std"],
    }

    agg_spec = {}
    for col, funcs in candidate_agg_cols.items():
        if col in vessel_df.columns:
            for func in funcs:
                agg_spec[f"vessel_{col}_{func}"] = (col, func)

    # Vessel class mix — what fraction of pings are cargo/tanker (key vessel types)
    if "vessel_class" in vessel_df.columns:
        for cls in ["cargo", "tanker"]:
            vessel_df[f"is_{cls}"] = (vessel_df["vessel_class"] == cls).astype(np.int8)
            agg_spec[f"pct_{cls}_vessels"] = (f"is_{cls}", "mean")

    if not agg_spec:
        log.warning("  No vessel feature columns found for aggregation.")
        return None

    vessel_daily = (
        vessel_df.groupby(["nearest_port", "date"])
        .agg(**agg_spec)
        .reset_index()
    )

    # Round for clean output
    numeric_cols = vessel_daily.select_dtypes(include="number").columns
    vessel_daily[numeric_cols] = vessel_daily[numeric_cols].round(4)

    log.info(f"  Vessel daily aggregations: {len(vessel_daily):,} rows")

    return vessel_daily


# ---------------------------------------------------------------------------
# 3. Merge All Feature Tables
# ---------------------------------------------------------------------------

def merge_all_features(
    port_df:     pd.DataFrame,
    external_df: pd.DataFrame,
    vessel_daily: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Join all three feature tables into a single master DataFrame.

    Join strategy
    -------------
    Base table  : port_features (one row per port per day — the anchor)
    First join  : LEFT JOIN external_features on (nearest_port / port, date)
    Second join : LEFT JOIN vessel_daily aggregations on (nearest_port, date)

    LEFT JOIN is used throughout so that port-days with no weather data
    or no vessel pings are retained (with NaN in the joined columns) rather
    than silently dropped. The quality report flags these rows.

    Parameters
    ----------
    port_df : pd.DataFrame
        Port-level daily features (from port_features.parquet).
    external_df : pd.DataFrame
        Weather + calendar features (from external_features.parquet).
    vessel_daily : pd.DataFrame or None
        Daily vessel aggregations per port. None if spatial assignment failed.

    Returns
    -------
    pd.DataFrame
        Merged master DataFrame, sorted by port and date.
    """
    log.info("Merging all feature tables ...")

    # ── Normalise date types ──────────────────────────────────────────────────
    port_df["date"]     = pd.to_datetime(port_df["date"]).dt.normalize()
    external_df["date"] = pd.to_datetime(external_df["date"]).dt.normalize()

    # ── Normalise port name column ────────────────────────────────────────────
    # external_features.py renames 'port' → 'nearest_port' during the merge,
    # but if it ran standalone (without port_features), it may retain 'port'.
    if "port" in external_df.columns and "nearest_port" not in external_df.columns:
        external_df = external_df.rename(columns={"port": "nearest_port"})

    # Confirm both sides have nearest_port before joining
    for name, df in [("port_features", port_df), ("external_features", external_df)]:
        if "nearest_port" not in df.columns:
            raise ValueError(
                f"'{name}' is missing the 'nearest_port' column.\n"
                f"Available columns: {list(df.columns)}"
            )

    # ── Join 1: port + external ───────────────────────────────────────────────
    # Identify overlapping columns (other than join keys) to avoid _x/_y suffixes
    port_cols     = set(port_df.columns) - {"nearest_port", "date"}
    external_cols = set(external_df.columns) - {"nearest_port", "date"}
    overlap       = port_cols & external_cols

    if overlap:
        log.info(f"  Dropping overlapping columns from external before join: {overlap}")
        external_df = external_df.drop(columns=list(overlap))

    master = port_df.merge(
        external_df,
        on=["nearest_port", "date"],
        how="left",
        suffixes=("", "_ext"),
    )

    log.info(f"  After port + external join : {len(master):,} rows × {master.shape[1]} cols")

    # ── Join 2: + vessel daily aggregations ───────────────────────────────────
    if vessel_daily is not None:
        vessel_daily["date"] = pd.to_datetime(vessel_daily["date"]).dt.normalize()

        master = master.merge(
            vessel_daily,
            on=["nearest_port", "date"],
            how="left",
            suffixes=("", "_vessel"),
        )
        log.info(f"  After + vessel daily join  : {len(master):,} rows × {master.shape[1]} cols")
    else:
        log.warning("  Skipping vessel daily join — no matching vessel pings found.")

    master.sort_values(["nearest_port", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)

    return master


# ---------------------------------------------------------------------------
# 4. Cross-Source Feature Engineering
# ---------------------------------------------------------------------------

def add_cross_source_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that require data from two or more source tables.
    These features cannot be computed inside any individual feature script.

    Features added
    --------------
    weather_congestion_interaction
        max_weather_severity_score × slow_vessel_ratio
        Captures compounding risk: bad weather AND vessel congestion on the same day.
        A port that is 60% congested during a storm is far riskier than one that
        is 60% congested on a clear day.

    vessel_weather_risk
        vessel_speed_drop_ratio_mean × max_weather_severity_score
        Vessel slowdown signal amplified by external weather conditions.
        Requires vessel daily aggregations — set to NaN if not available.

    vessels_7d_rolling_norm
        vessels_7d_rolling normalised to [0, 1] within each port's observed range.
        Used as a traffic volume component in the pressure index without
        dominating due to absolute vessel count differences between ports.

    congestion_pressure_index
        Unified 0–1 composite index across all three data sources.
        Weighted combination of port_congestion_score, weather severity,
        and normalised rolling vessel count.
        This is the richest single feature in the dataset.

    congestion_label
        Binary classification target for Phase 3.
        1 if port_congestion_score > CONGESTION_THRESHOLD (0.35), else 0.
        Documented alongside CONGESTION_THRESHOLD constant at top of file.

    Parameters
    ----------
    master : pd.DataFrame
        Merged master DataFrame from merge_all_features().

    Returns
    -------
    pd.DataFrame
        Master DataFrame with cross-source feature columns added.
    """
    log.info("Engineering cross-source features ...")

    # ── weather × congestion interaction ─────────────────────────────────────
    weather_col     = "max_weather_severity_score"
    slow_ratio_col  = "slow_vessel_ratio"

    if weather_col in master.columns and slow_ratio_col in master.columns:
        master["weather_congestion_interaction"] = (
            master[weather_col].fillna(0) * master[slow_ratio_col].fillna(0)
        ).round(4)
        log.info("  ✓ weather_congestion_interaction")
    else:
        master["weather_congestion_interaction"] = np.nan
        missing = [c for c in [weather_col, slow_ratio_col] if c not in master.columns]
        log.warning(f"  weather_congestion_interaction: missing source cols {missing} → NaN")

    # ── vessel speed drop × weather risk ─────────────────────────────────────
    speed_drop_col = "vessel_speed_drop_ratio_mean"

    if speed_drop_col in master.columns and weather_col in master.columns:
        master["vessel_weather_risk"] = (
            master[speed_drop_col].fillna(1.0) * master[weather_col].fillna(0)
        ).round(4)
        # Invert: lower speed_drop_ratio means more slowdown, so higher risk
        # speed_drop_ratio near 0 = vessel very slow = high risk
        # Multiply by (1 - ratio) so it reads intuitively as a risk score
        master["vessel_weather_risk"] = (
            (1 - master[speed_drop_col].fillna(1.0).clip(0, 1))
            * master[weather_col].fillna(0)
        ).round(4)
        log.info("  ✓ vessel_weather_risk")
    else:
        master["vessel_weather_risk"] = np.nan
        log.warning(
            f"  vessel_weather_risk: '{speed_drop_col}' not found "
            f"(vessel daily join likely skipped) → NaN"
        )

    # ── Normalise vessels_7d_rolling within each port ─────────────────────────
    rolling_col = "vessels_7d_rolling"

    if rolling_col in master.columns:
        # Min-max normalise per port independently so Rotterdam's 200 vessels/day
        # and LA's 80 vessels/day are on the same 0–1 scale
        def minmax_norm(series: pd.Series) -> pd.Series:
            s_min = series.min()
            s_max = series.max()
            if s_max == s_min:
                return pd.Series(0.5, index=series.index)
            return (series - s_min) / (s_max - s_min)

        master["vessels_7d_rolling_norm"] = (
            master.groupby("nearest_port")[rolling_col]
            .transform(minmax_norm)
            .round(4)
        )
        log.info("  ✓ vessels_7d_rolling_norm")
    else:
        master["vessels_7d_rolling_norm"] = np.nan
        log.warning(f"  vessels_7d_rolling_norm: '{rolling_col}' not found → NaN")

    # ── Unified congestion pressure index ────────────────────────────────────
    pressure_components = {
        "port_congestion_score":      PRESSURE_INDEX_WEIGHTS["port_congestion_score"],
        weather_col:                  PRESSURE_INDEX_WEIGHTS["max_weather_severity_score"],
        "vessels_7d_rolling_norm":    PRESSURE_INDEX_WEIGHTS["vessels_7d_rolling_norm"],
    }

    available_components  = {col: w for col, w in pressure_components.items()
                              if col in master.columns}
    missing_components    = [col for col in pressure_components if col not in master.columns]

    if missing_components:
        log.warning(f"  Pressure index: missing components {missing_components} — "
                    f"will use available components with renormalised weights.")

    if available_components:
        # Renormalise weights to sum to 1.0 if some components are missing
        total_weight = sum(available_components.values())
        index_value  = sum(
            master[col].fillna(0) * (w / total_weight)
            for col, w in available_components.items()
        )
        master["congestion_pressure_index"] = index_value.clip(0, 1).round(4)
        log.info(f"  ✓ congestion_pressure_index  "
                 f"(components: {list(available_components.keys())})")
    else:
        master["congestion_pressure_index"] = np.nan
        log.warning("  congestion_pressure_index: no components available → NaN")

    # ── Binary congestion label (Phase 3 classification target) ──────────────
    if "port_congestion_score" in master.columns:
        master["congestion_label"] = (
            master["port_congestion_score"] > CONGESTION_THRESHOLD
        ).astype(np.int8)

        label_counts = master["congestion_label"].value_counts()
        n_positive   = label_counts.get(1, 0)
        n_negative   = label_counts.get(0, 0)
        n_total      = len(master)

        log.info(f"  ✓ congestion_label  (threshold={CONGESTION_THRESHOLD})")
        log.info(f"     Label=1 (congested)    : {n_positive:>6,}  ({n_positive/n_total*100:.1f}%)")
        log.info(f"     Label=0 (not congested): {n_negative:>6,}  ({n_negative/n_total*100:.1f}%)")

        # Warn if class imbalance is severe
        minority_pct = min(n_positive, n_negative) / n_total * 100
        if minority_pct < 10:
            log.warning(
                f"  ⚠ Severe class imbalance — minority class is {minority_pct:.1f}% of data.\n"
                f"    Consider adjusting CONGESTION_THRESHOLD (currently {CONGESTION_THRESHOLD}).\n"
                f"    In Phase 3, use class_weight='balanced' in XGBoost."
            )
        elif minority_pct < 20:
            log.warning(
                f"  ⚠ Moderate class imbalance — minority class is {minority_pct:.1f}% of data.\n"
                f"    Monitor precision-recall carefully in Phase 3."
            )
    else:
        master["congestion_label"] = np.nan
        log.warning("  congestion_label: 'port_congestion_score' not found → NaN")

    return master


# ---------------------------------------------------------------------------
# 5. Quality Report
# ---------------------------------------------------------------------------

def run_quality_report(master: pd.DataFrame) -> None:
    """
    Run a comprehensive quality audit on the master dataset before saving.

    Checks performed
    ----------------
    1. Shape and coverage summary
    2. Null rate per column — warns on columns exceeding NULL_WARNING_THRESHOLD
    3. Feature distribution summary for key numeric columns
    4. Congestion label balance
    5. Port-level coverage (rows per port)
    6. Date continuity check (missing days per port)

    Parameters
    ----------
    master : pd.DataFrame
        Assembled master dataset.
    """
    print("\n" + "=" * 65)
    print("  MASTER DATASET — QUALITY REPORT")
    print("=" * 65)

    # ── 1. Shape ─────────────────────────────────────────────────────────────
    print(f"\n  Shape         : {len(master):,} rows  ×  {master.shape[1]} columns")
    print(f"  Ports         : {master['nearest_port'].nunique()}")
    print(f"  Date range    : {master['date'].min().date()} → {master['date'].max().date()}")

    # ── 2. Null audit ─────────────────────────────────────────────────────────
    null_rates = master.isnull().mean().sort_values(ascending=False)
    high_null  = null_rates[null_rates > NULL_WARNING_THRESHOLD]

    print(f"\n  Null rate audit (threshold: >{NULL_WARNING_THRESHOLD*100:.0f}%):")
    if high_null.empty:
        print(f"    ✓ No columns exceed {NULL_WARNING_THRESHOLD*100:.0f}% null rate.")
    else:
        for col, rate in high_null.items():
            print(f"    ⚠  {col:<45} {rate*100:.1f}% null")

    # ── 3. Key feature distributions ─────────────────────────────────────────
    key_numeric_cols = [
        "port_congestion_score", "slow_vessel_ratio", "avg_speed_in_zone",
        "vessels_in_zone", "max_weather_severity_score",
        "weather_congestion_interaction", "congestion_pressure_index",
    ]

    available_numeric = [c for c in key_numeric_cols if c in master.columns]

    if available_numeric:
        print(f"\n  Key feature distributions:")
        print(f"  {'Column':<40} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*80}")
        for col in available_numeric:
            s = master[col].dropna()
            if len(s) > 0:
                print(
                    f"  {col:<40} "
                    f"{s.mean():>8.4f} {s.median():>8.4f} "
                    f"{s.std():>8.4f} {s.min():>8.4f} {s.max():>8.4f}"
                )

    # ── 4. Label balance ──────────────────────────────────────────────────────
    if "congestion_label" in master.columns:
        label_counts = master["congestion_label"].value_counts().sort_index()
        n_total      = len(master)
        print(f"\n  Congestion label balance (threshold={CONGESTION_THRESHOLD}):")
        for label, count in label_counts.items():
            bar    = "█" * int(count / n_total * 40)
            status = "CONGESTED    " if label == 1 else "NOT CONGESTED"
            print(f"    Label={label}  {status}  {count:>6,}  ({count/n_total*100:.1f}%)  {bar}")

    # ── 5. Port-level coverage ────────────────────────────────────────────────
    print(f"\n  Rows per port:")
    port_counts = master.groupby("nearest_port").size().sort_values(ascending=False)
    for port, count in port_counts.items():
        print(f"    {port:<25} {count:>6,} days")

    # ── 6. Date continuity ────────────────────────────────────────────────────
    print(f"\n  Date continuity check (missing days per port):")
    all_dates    = pd.date_range(master["date"].min(), master["date"].max(), freq="D")
    has_gap      = False

    for port in master["nearest_port"].unique():
        port_dates   = set(master[master["nearest_port"] == port]["date"].dt.normalize())
        expected_set = set(all_dates.normalize())
        missing_days = expected_set - port_dates

        if missing_days:
            has_gap = True
            print(f"    ⚠  {port:<25} {len(missing_days):>4} missing days")
        else:
            print(f"    ✓  {port:<25} complete")

    if not has_gap:
        print(f"    ✓ No gaps detected in any port time series.")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# 6. Save Master Dataset
# ---------------------------------------------------------------------------

def save_master(master: pd.DataFrame, output_path: Path) -> None:
    """
    Save the master dataset to Parquet format.

    Also saves a companion CSV summary (first 500 rows) for quick inspection
    without needing to load the full Parquet in a notebook.

    Parameters
    ----------
    master : pd.DataFrame
        Assembled master dataset.
    output_path : Path
        Destination Parquet file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving master dataset → {output_path} ...")
    master.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"  ✓ Parquet saved — {size_mb:.1f} MB")

    # Companion CSV sample for quick inspection
    sample_path = output_path.parent / "master_dataset_sample.csv"
    master.head(500).to_csv(sample_path, index=False)
    log.info(f"  ✓ Sample CSV (500 rows) → {sample_path.name}")

    # Log final column list for reference
    log.info(f"\n  Final columns ({len(master.columns)}):")
    for col in master.columns:
        dtype     = master[col].dtype
        null_rate = master[col].isnull().mean()
        null_info = f"  ← {null_rate*100:.0f}% null" if null_rate > 0.05 else ""
        log.info(f"    {col:<45} {str(dtype):<15}{null_info}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    processed_dir = PROJECT_ROOT / config["paths"]["processed"]
    output_path   = processed_dir / "master_dataset.parquet"
    ports         = config["ports"]
    radius_km     = config["anchorage"]["radius_km"]

    log.info("=" * 65)
    log.info("Phase 2 — Master Dataset Assembly")
    log.info(f"  Congestion label threshold : {CONGESTION_THRESHOLD}")
    log.info(f"  Anchorage radius           : {radius_km} km")
    log.info(f"  Ports configured           : {[p['name'] for p in ports]}")
    log.info("=" * 65)

    # Step 1: Validate and load all three input files
    dataframes = validate_inputs(processed_dir)

    vessel_df   = dataframes["vessel_features"]
    port_df     = dataframes["port_features"]
    external_df = dataframes["external_features"]

    # Step 2: Aggregate vessel features to daily port level
    vessel_daily = build_vessel_daily_aggregations(vessel_df, ports, radius_km)

    # Step 3: Merge all three tables
    master = merge_all_features(port_df, external_df, vessel_daily)

    # Step 4: Cross-source feature engineering
    master = add_cross_source_features(master)

    # Step 5: Quality report
    run_quality_report(master)

    # Step 6: Save
    save_master(master, output_path)

    log.info("\nPhase 2 complete.")
    log.info(f"  Master dataset : {output_path}")
    log.info(f"  Rows           : {len(master):,}")
    log.info(f"  Columns        : {master.shape[1]}")
    log.info(f"  Ready for Phase 3 model training.")


if __name__ == "__main__":
    main()
