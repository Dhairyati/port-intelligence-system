"""
vessel_features.py
------------------
Phase 2 — Vessel-Level Feature Engineering

Loads raw AIS data from data/raw/ais/, engineers behavioral features that
capture vessel movement patterns indicative of port congestion and delay,
and saves the result to data/processed/vessel_features.parquet.

Features engineered
-------------------
Time features:
    hour_of_day        : hour of the AIS ping (0–23)
    weekday            : day of week (0=Monday, 6=Sunday)
    weekend_flag       : 1 if Saturday or Sunday, else 0

Speed / movement features:
    anchored_flag      : 1 if SOG < 0.5 knots (vessel is stationary)
    slow_moving_flag   : 1 if SOG < 3.0 knots (very slow, likely waiting)
    speed_drop_ratio   : SOG / vessel's rolling median SOG; low ratio = slowdown
    idle_duration_proxy: consecutive slow-ping count per vessel (waiting proxy)
    movement_intensity : rolling std of SOG per vessel (high = erratic, low = steady)

Vessel classification:
    vessel_class       : human-readable category derived from NOAA VesselType code

Usage (from project root):
    python src/features/vessel_features.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — make src/ importable when running directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging — print timestamped progress to stdout
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

# NOAA AIS column names → internal standard names used throughout the project
COLUMN_MAP = {
    "MMSI":         "mmsi",
    "BaseDateTime": "timestamp",
    "LAT":          "lat",
    "LON":          "lon",
    "SOG":          "sog",          # Speed Over Ground (knots)
    "COG":          "cog",          # Course Over Ground (degrees)
    "Heading":      "heading",
    "VesselName":   "vessel_name",
    "IMO":          "imo",
    "VesselType":   "vessel_type",
    "Status":       "nav_status",
    "Length":       "length",
    "Width":        "width",
    "Draft":        "draft",
    "Cargo":        "cargo_type",
}

# Minimum columns required to proceed — if any are missing, we abort early
REQUIRED_COLUMNS = ["mmsi", "timestamp", "lat", "lon", "sog"]

# Speed thresholds (knots)
ANCHORED_THRESHOLD   = 0.5    # below this → vessel is stationary / anchored
SLOW_MOVING_THRESHOLD = 3.0   # below this → very slow, likely in anchorage queue

# Rolling window size (number of pings per vessel) for speed statistics
# AIS pings every ~2–10 min → window of 12 ≈ last 24–120 minutes of data
ROLLING_WINDOW = 12

# Minimum pings a vessel must have to compute rolling stats reliably
MIN_PINGS_FOR_ROLLING = 3

# NOAA VesselType integer codes → human-readable class labels
# Source: https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
VESSEL_TYPE_MAP = {
    # Cargo vessels
    70: "cargo", 71: "cargo", 72: "cargo", 73: "cargo",
    74: "cargo", 75: "cargo", 76: "cargo", 77: "cargo", 78: "cargo", 79: "cargo",
    # Tankers
    80: "tanker", 81: "tanker", 82: "tanker", 83: "tanker",
    84: "tanker", 85: "tanker", 86: "tanker", 87: "tanker", 88: "tanker", 89: "tanker",
    # Passenger
    60: "passenger", 61: "passenger", 62: "passenger", 63: "passenger",
    64: "passenger", 65: "passenger", 66: "passenger", 67: "passenger", 68: "passenger", 69: "passenger",
    # Fishing
    30: "fishing",
    # Tug / port service
    21: "tug_service", 22: "tug_service", 31: "tug_service", 32: "tug_service",
    # High speed craft
    40: "high_speed", 41: "high_speed", 42: "high_speed", 43: "high_speed",
    44: "high_speed", 45: "high_speed", 46: "high_speed", 47: "high_speed", 48: "high_speed", 49: "high_speed",
    # Pleasure / sailing
    36: "pleasure", 37: "pleasure",
    # Military / law enforcement
    35: "military", 55: "law_enforcement",
    # Search and rescue
    51: "sar",
    # Dredging / diving
    33: "dredging", 34: "diving",
}


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_ais_data(raw_ais_dir: Path) -> pd.DataFrame:
    """
    Load all AIS CSV files from raw_ais_dir into a single DataFrame.

    Reads files one at a time to keep peak memory usage bounded, then
    concatenates. Applies column renaming and drops rows with nulls in
    the required columns.

    Parameters
    ----------
    raw_ais_dir : Path
        Directory containing NOAA AIS CSV files (one or many).

    Returns
    -------
    pd.DataFrame
        Combined AIS data with standardised column names.
    """
    csv_files = sorted(raw_ais_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_ais_dir}.\n"
            "Run: python src/ingestion/fetch_ais.py"
        )

    log.info(f"Found {len(csv_files)} AIS file(s) in {raw_ais_dir}")

    frames = []
    total_raw_rows = 0

    for csv_path in csv_files:
        log.info(f"  Reading: {csv_path.name} ...")

        df = pd.read_csv(
            csv_path,
            low_memory=False,
            # Only read columns we know about; ignore the rest.
            # This cuts memory use on wide datasets.
            usecols=lambda col: col in COLUMN_MAP,
        )

        raw_rows = len(df)
        total_raw_rows += raw_rows
        log.info(f"    {raw_rows:,} rows, {df.shape[1]} relevant columns")

        frames.append(df)

    # Concatenate all files into one DataFrame
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total raw rows across all files: {total_raw_rows:,}")

    # Rename to internal standard names
    combined.rename(columns=COLUMN_MAP, inplace=True)

    # Check required columns are present
    missing = [c for c in REQUIRED_COLUMNS if c not in combined.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from AIS data after renaming: {missing}\n"
            f"Available columns: {list(combined.columns)}\n"
            f"Check that COLUMN_MAP matches your dataset's actual column names."
        )

    # Drop rows with nulls in critical columns only
    before = len(combined)
    combined.dropna(subset=REQUIRED_COLUMNS, inplace=True)
    dropped = before - len(combined)
    if dropped > 0:
        log.warning(f"Dropped {dropped:,} rows with nulls in required columns ({dropped/before*100:.1f}%)")

    log.info(f"Clean rows after null removal: {len(combined):,}")
    return combined


# ---------------------------------------------------------------------------
# 2. Timestamp Parsing
# ---------------------------------------------------------------------------

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the timestamp column to datetime and validate the result.

    NOAA AIS timestamps are ISO 8601 strings: "2023-01-01T00:03:00"
    We convert them to pandas datetime (UTC) and drop unparseable rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'timestamp' column (string).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 'timestamp' cast to datetime64[ns].
    """
    log.info("Parsing timestamps ...")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)

    failed = df["timestamp"].isnull().sum()
    if failed > 0:
        log.warning(f"  {failed:,} rows failed timestamp parsing — dropping them")
        df = df.dropna(subset=["timestamp"])

    log.info(f"  Timestamp range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ---------------------------------------------------------------------------
# 3. Time Features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar and time-of-day features from the parsed timestamp.

    These capture the cyclical operational patterns of ports:
    - Ports are slower on weekends and public holidays
    - Night shifts have lower throughput
    - Monthly patterns reflect shipping seasons

    Features added
    --------------
    hour_of_day  : int  0–23
    weekday      : int  0 (Monday) – 6 (Sunday)
    weekend_flag : int  1 if Saturday or Sunday, else 0

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed 'timestamp' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with time feature columns added.
    """
    log.info("Adding time features ...")

    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["weekday"]      = df["timestamp"].dt.dayofweek    # 0 = Monday
    df["weekend_flag"] = (df["weekday"] >= 5).astype(np.int8)

    log.info("  ✓ hour_of_day, weekday, weekend_flag")
    return df


# ---------------------------------------------------------------------------
# 4. Speed & Movement Features
# ---------------------------------------------------------------------------

def add_speed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer vessel speed and movement behavioral features.

    All operations are fully vectorized using pandas groupby + transform,
    which avoids Python-level loops and scales to millions of rows.

    The key insight: a vessel's absolute speed is less informative than
    its speed *relative to its own historical cruise speed*. A container
    ship doing 4 knots is slow; a tug doing 4 knots is normal. The
    speed_drop_ratio normalises for this.

    Features added
    --------------
    anchored_flag      : 1 if sog < ANCHORED_THRESHOLD (0.5 knots)
                         → vessel is stationary, likely at berth or anchor

    slow_moving_flag   : 1 if sog < SLOW_MOVING_THRESHOLD (3.0 knots)
                         → vessel is barely moving, likely in anchorage queue

    speed_drop_ratio   : sog / rolling_median_sog per vessel
                         → how much this ping's speed differs from vessel's
                            typical cruise speed over the last ROLLING_WINDOW pings
                         → values close to 0 = sudden significant slowdown
                         → values close to 1 = normal cruising
                         → NaN if vessel has too few pings for rolling stats

    idle_duration_proxy: count of consecutive slow pings per vessel
                         per slow-movement episode (resets when vessel speeds up)
                         → higher = longer waiting period (proxy for delay)

    movement_intensity : rolling standard deviation of sog per vessel
                         → high std = erratic speed changes (port approach)
                         → low std  = steady cruise (open sea)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by mmsi and timestamp, with 'sog' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with speed feature columns added.
    """
    log.info("Adding speed and movement features ...")

    # Coerce sog to numeric — some rows may have string artefacts
    df["sog"] = pd.to_numeric(df["sog"], errors="coerce")

    # Clip unrealistic SOG values (> 50 knots = sensor noise)
    outlier_count = (df["sog"] > 50).sum()
    if outlier_count > 0:
        log.warning(f"  Clipping {outlier_count:,} SOG values above 50 knots")
    df["sog"] = df["sog"].clip(lower=0, upper=50)

    # ── Binary speed flags ──────────────────────────────────────────────────
    df["anchored_flag"]    = (df["sog"] < ANCHORED_THRESHOLD).astype(np.int8)
    df["slow_moving_flag"] = (df["sog"] < SLOW_MOVING_THRESHOLD).astype(np.int8)

    log.info(f"  anchored_flag    : {df['anchored_flag'].sum():,} pings ({df['anchored_flag'].mean()*100:.1f}% of all pings)")
    log.info(f"  slow_moving_flag : {df['slow_moving_flag'].sum():,} pings ({df['slow_moving_flag'].mean()*100:.1f}% of all pings)")

    # ── Sort by vessel and time before rolling calculations ─────────────────
    # Rolling operations require data to be ordered chronologically per vessel.
    log.info("  Sorting by mmsi + timestamp for rolling calculations ...")
    df.sort_values(["mmsi", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Rolling median SOG per vessel ───────────────────────────────────────
    # groupby + transform applies the rolling window per vessel independently
    # min_periods=MIN_PINGS_FOR_ROLLING avoids stats on vessels with 1–2 pings
    log.info(f"  Computing rolling median SOG (window={ROLLING_WINDOW}) ...")

    rolling_median_sog = (
        df.groupby("mmsi", sort=False)["sog"]
        .transform(
            lambda x: x.rolling(
                window=ROLLING_WINDOW,
                min_periods=MIN_PINGS_FOR_ROLLING
            ).median()
        )
    )

    # speed_drop_ratio: current sog / rolling median sog
    # Replace 0 median with NaN to avoid division-by-zero
    safe_median = rolling_median_sog.replace(0, np.nan)
    df["speed_drop_ratio"] = (df["sog"] / safe_median).round(4)

    # Cap at 3.0 — ratios above this are artefacts (e.g. vessel was anchored
    # for a long time, then a single fast ping skews the ratio)
    df["speed_drop_ratio"] = df["speed_drop_ratio"].clip(upper=3.0)

    log.info(f"  speed_drop_ratio : median={df['speed_drop_ratio'].median():.3f}, "
             f"nulls={df['speed_drop_ratio'].isnull().sum():,}")

    # ── Movement intensity: rolling std of SOG per vessel ───────────────────
    log.info(f"  Computing movement_intensity (rolling std, window={ROLLING_WINDOW}) ...")

    df["movement_intensity"] = (
        df.groupby("mmsi", sort=False)["sog"]
        .transform(
            lambda x: x.rolling(
                window=ROLLING_WINDOW,
                min_periods=MIN_PINGS_FOR_ROLLING
            ).std()
        )
    ).round(4)

    log.info(f"  movement_intensity: median={df['movement_intensity'].median():.3f}")

    # ── Idle duration proxy ─────────────────────────────────────────────────
    # For each vessel, count consecutive slow pings within a slow episode.
    # When the vessel speeds up (slow_moving_flag = 0), the counter resets.
    #
    # Implementation: use cumsum trick — create a group ID that increments
    # each time slow_moving_flag transitions from 1→0, then cumcount within
    # slow episodes only.
    log.info("  Computing idle_duration_proxy ...")

    # Mark the start of each NEW non-slow episode per vessel
    # (i.e. where slow_moving_flag changes from 1 to 0)
    flag = df["slow_moving_flag"]
    # Group ID increments every time the vessel stops being slow
    # cumsum on (flag == 0) creates a counter that ticks up at each non-slow ping
    group_id = df.groupby("mmsi", sort=False)["slow_moving_flag"].transform(
        lambda x: (x == 0).cumsum()
    )

    # Within each slow episode (slow_moving_flag == 1), cumcount gives position
    # in the episode (0-indexed → add 1 for human readability)
    df["idle_duration_proxy"] = (
        df[flag == 1]
        .groupby(["mmsi", group_id[flag == 1]])
        .cumcount() + 1
    )

    # Non-slow pings get 0
    df["idle_duration_proxy"] = df["idle_duration_proxy"].fillna(0).astype(np.int32)

    log.info(f"  idle_duration_proxy: max={df['idle_duration_proxy'].max()}, "
             f"mean (slow pings only)={df[df['idle_duration_proxy']>0]['idle_duration_proxy'].mean():.1f}")

    return df


# ---------------------------------------------------------------------------
# 5. Vessel Classification
# ---------------------------------------------------------------------------

def add_vessel_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map numeric VesselType codes to human-readable vessel class labels.

    NOAA AIS uses integer codes defined in the ITU-R M.1371 standard.
    We map these to broad operational categories relevant to port congestion:
    cargo, tanker, passenger, fishing, tug_service, etc.

    Vessels with unknown or unmapped codes are labelled "other".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'vessel_type' column (may be float due to nulls).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'vessel_class' string column added.
    """
    log.info("Adding vessel_class ...")

    if "vessel_type" not in df.columns:
        log.warning("  'vessel_type' column not found — setting vessel_class to 'unknown'")
        df["vessel_class"] = "unknown"
        return df

    # Convert to int-compatible (nulls become -1 sentinel)
    vessel_type_int = pd.to_numeric(df["vessel_type"], errors="coerce").fillna(-1).astype(int)

    df["vessel_class"] = vessel_type_int.map(VESSEL_TYPE_MAP).fillna("other")

    # Log class distribution
    class_counts = df["vessel_class"].value_counts()
    log.info("  Vessel class distribution:")
    for cls, count in class_counts.items():
        log.info(f"    {cls:<20} {count:>10,}  ({count/len(df)*100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 6. Save Output
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the engineered feature DataFrame to Parquet format.

    Parquet is columnar, compressed, and ~5–10x faster to read than CSV
    for analytical workloads. It also preserves dtypes (datetime, int8, etc.)
    so downstream scripts don't need to re-parse.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to save.
    output_path : Path
        Full file path including filename, e.g. data/processed/vessel_features.parquet
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving {len(df):,} rows to {output_path} ...")

    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"  ✓ Saved — file size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# 7. Summary Report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """
    Print a concise feature summary to confirm engineering was successful.
    This is what you'd verify before moving to Phase 3.
    """
    engineered_cols = [
        "anchored_flag", "slow_moving_flag", "speed_drop_ratio",
        "idle_duration_proxy", "movement_intensity", "vessel_class",
        "hour_of_day", "weekday", "weekend_flag",
    ]

    print("\n" + "=" * 60)
    print("  VESSEL FEATURES — ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"  Total rows      : {len(df):,}")
    print(f"  Unique vessels  : {df['mmsi'].nunique():,}")
    print(f"  Date range      : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Output columns  : {len(df.columns)}")
    print()
    print("  Engineered features:")
    for col in engineered_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            null_info  = f"  ({null_count:,} nulls)" if null_count > 0 else ""
            print(f"    ✓  {col:<28} dtype={df[col].dtype}{null_info}")
        else:
            print(f"    ✗  {col:<28} MISSING")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    raw_ais_dir = PROJECT_ROOT / config["paths"]["raw_ais"]
    output_path = PROJECT_ROOT / config["paths"]["processed"] / "vessel_features.parquet"

    log.info("=" * 60)
    log.info("Phase 2 — Vessel Feature Engineering")
    log.info("=" * 60)

    # Step 1: Load
    df = load_ais_data(raw_ais_dir)

    # Step 2: Parse timestamps
    df = parse_timestamps(df)

    # Step 3: Time features
    df = add_time_features(df)

    # Step 4: Speed & movement features
    df = add_speed_features(df)

    # Step 5: Vessel classification
    df = add_vessel_class(df)

    # Step 6: Summary
    print_summary(df)

    # Step 7: Save
    save_features(df, output_path)

    log.info("Phase 2 Step 1 complete — vessel_features.parquet ready.")
    log.info(f"Next: run src/features/port_features.py")


if __name__ == "__main__":
    main()
