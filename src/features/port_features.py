"""
port_features.py
----------------
Phase 2 — Port-Level Congestion Feature Engineering

Reads vessel_features.parquet in streaming row-group batches using
PyArrow's native batch reader. Never loads the full dataset into RAM.
Each batch is spatially assigned to ports and aggregated to daily
port-level counts. Only the tiny daily aggregations accumulate in memory.

On 242M rows the peak RAM usage is ~200-400 MB (one row group at a time)
regardless of total dataset size.

Output
------
    data/processed/port_features.parquet
        One row per port per day with congestion signals and rolling features.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
EARTH_RADIUS_KM       = 6371.0
ROLLING_DAYS          = 7
ANCHORED_THRESHOLD    = 0.5
SLOW_THRESHOLD        = 3.0
IDLE_DURATION_CAP     = 50

CONGESTION_WEIGHTS = {
    "slow_vessel_ratio": 0.45,
    "anchored_ratio":    0.35,
    "idle_duration_norm":0.20,
}

# Columns we actually need from vessel_features.parquet
# Requesting only these prevents PyArrow from loading unused columns
REQUIRED_COLUMNS = [
    "mmsi", "timestamp", "lat", "lon", "sog",
    "anchored_flag", "slow_moving_flag", "idle_duration_proxy",
]


# ---------------------------------------------------------------------------
# Haversine (vectorised NumPy)
# ---------------------------------------------------------------------------

def haversine_vectorised(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    dlat   = np.radians(lat2) - lat1_r
    dlon   = np.radians(lon2) - lon1_r
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1_r) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Assign one batch to ports and aggregate to daily counts
# ---------------------------------------------------------------------------

def aggregate_batch(
    batch_df: pd.DataFrame,
    ports: list,
    radius_km: float,
) -> pd.DataFrame | None:
    """
    Given one batch of vessel pings as a DataFrame:
      1. Compute distance from each ping to every port (Haversine).
      2. Keep only pings within radius_km of at least one port.
      3. Assign each ping to its nearest port.
      4. Group by (nearest_port, date) and compute daily sums/counts.

    Returns a small daily aggregation DataFrame, or None if no pings
    in this batch fall within any port zone.

    The returned DataFrame uses SUM aggregations intentionally —
    sums from multiple batches covering the same (port, date) can be
    combined correctly in the final reduce step. Averaging across
    batches would produce wrong results (you can't average averages
    without knowing the denominator).
    """
    if len(batch_df) == 0:
        return None

    lat_arr = batch_df["lat"].to_numpy(dtype=np.float64)
    lon_arr = batch_df["lon"].to_numpy(dtype=np.float64)

    # Distance matrix: shape (n_pings, n_ports)
    dist_matrix    = np.column_stack([
        haversine_vectorised(lat_arr, lon_arr, p["lat"], p["lon"])
        for p in ports
    ])
    nearest_idx    = np.argmin(dist_matrix, axis=1)
    nearest_dist   = dist_matrix[np.arange(len(batch_df)), nearest_idx]
    within_mask    = nearest_dist <= radius_km

    if within_mask.sum() == 0:
        return None

    batch_df = batch_df[within_mask].copy()
    batch_df["nearest_port"] = [ports[i]["name"] for i in nearest_idx[within_mask]]
    batch_df["date"]         = batch_df["timestamp"].dt.normalize()

    # Aggregate to daily port level using SUMS and COUNTS
    # We avoid mean() here because means cannot be merged across batches
    daily = (
        batch_df.groupby(["nearest_port", "date"])
        .agg(
            total_pings           = ("mmsi",               "count"),
            unique_mmsi_approx    = ("mmsi",               "nunique"),
            anchored_pings        = ("anchored_flag",       "sum"),
            slow_pings            = ("slow_moving_flag",    "sum"),
            sog_sum               = ("sog",                 "sum"),
            idle_duration_sum     = ("idle_duration_proxy", "sum"),
        )
        .reset_index()
    )

    return daily


# ---------------------------------------------------------------------------
# Stream through vessel_features.parquet batch by batch
# ---------------------------------------------------------------------------

def stream_aggregate_all_batches(
    parquet_path: Path,
    ports: list,
    radius_km: float,
) -> pd.DataFrame:
    """
    Open vessel_features.parquet and iterate over its row groups one at a
    time using PyArrow's ParquetFile batch reader.

    Each row group is converted to a pandas DataFrame, aggregated to
    daily port-level sums, then immediately discarded. Only the tiny
    daily aggregation DataFrames accumulate in memory.

    Peak RAM = size of one row group (~50-200 MB) + accumulated daily
               aggregations (negligible — at most ~155 rows × n_cols).

    Parameters
    ----------
    parquet_path : Path
        Path to vessel_features.parquet.
    ports : list of dict
        Port definitions from config.yaml.
    radius_km : float
        Anchorage zone radius in kilometres.

    Returns
    -------
    pd.DataFrame
        Combined daily aggregations across all batches, ready for
        final reduce step.
    """
    pf       = pq.ParquetFile(parquet_path)
    n_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    log.info(f"Streaming vessel_features.parquet")
    log.info(f"  Row groups : {n_groups}")
    log.info(f"  Total rows : {total_rows:,}")
    log.info(f"  Reading columns: {REQUIRED_COLUMNS}")

    # Only read the columns we need — saves significant I/O and RAM
    # Check which required columns actually exist in this parquet
    file_columns    = [s.name for s in pf.schema_arrow]
    columns_to_read = [c for c in REQUIRED_COLUMNS if c in file_columns]
    missing_cols    = [c for c in REQUIRED_COLUMNS if c not in file_columns]

    if missing_cols:
        log.warning(f"  Columns not found in parquet (will be skipped): {missing_cols}")

    if "lat" not in columns_to_read or "lon" not in columns_to_read:
        raise ValueError(
            "vessel_features.parquet is missing 'lat' or 'lon' columns.\n"
            "Cannot perform spatial assignment without coordinates.\n"
            "Re-run vessel_features.py to regenerate the parquet."
        )

    batch_results = []
    total_matched = 0

    for group_idx in range(n_groups):
        # Read one row group — this is the only large object in RAM
        table    = pf.read_row_group(group_idx, columns=columns_to_read)
        batch_df = table.to_pandas()

        # Free the PyArrow table immediately — we only need the pandas df
        del table

        # Parse timestamp if it came in as string or object
        if not pd.api.types.is_datetime64_any_dtype(batch_df["timestamp"]):
            batch_df["timestamp"] = pd.to_datetime(
                batch_df["timestamp"], errors="coerce"
            )
        batch_df.dropna(subset=["timestamp", "lat", "lon"], inplace=True)

        # Coerce numeric columns
        for col in ["sog", "anchored_flag", "slow_moving_flag", "idle_duration_proxy"]:
            if col in batch_df.columns:
                batch_df[col] = pd.to_numeric(batch_df[col], errors="coerce").fillna(0)

        # Aggregate this batch
        daily_batch = aggregate_batch(batch_df, ports, radius_km)

        matched_in_batch = len(batch_df) if daily_batch is not None else 0
        total_matched   += matched_in_batch

        if daily_batch is not None:
            batch_results.append(daily_batch)

        # Log progress every 5 row groups
        if (group_idx + 1) % 5 == 0 or group_idx == n_groups - 1:
            pct = (group_idx + 1) / n_groups * 100
            log.info(
                f"  Row group {group_idx+1:>3}/{n_groups}  ({pct:.0f}%)  "
                f"batches with port matches: {len(batch_results)}"
            )

        # Release batch DataFrame immediately
        del batch_df
        del daily_batch

    log.info(f"Streaming complete.")

    if not batch_results:
        raise RuntimeError(
            "No vessel pings fell within the anchorage radius of any port.\n"
            "Possible causes:\n"
            "  1. config.yaml radius_km may be too small — try 50 instead of 25\n"
            "  2. AIS data geographic coverage may not overlap with configured ports\n"
            "  3. lat/lon values may be null or corrupted in vessel_features.parquet\n"
            "Check the AIS data covers the Port of Los Angeles region."
        )

    # Combine all batch aggregations — this is just n_groups × ~155 tiny rows
    log.info(f"Combining {len(batch_results)} batch aggregation results ...")
    combined = pd.concat(batch_results, ignore_index=True)
    del batch_results

    return combined


# ---------------------------------------------------------------------------
# Reduce: combine batch sums into correct daily totals
# ---------------------------------------------------------------------------

def reduce_to_daily(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Multiple batches may cover the same (nearest_port, date) combination
    because one day's pings may span multiple row groups. This step sums
    all batch contributions for each (port, date) pair into correct totals.

    Then derives ratio features from the sums:
        slow_vessel_ratio  = slow_pings / total_pings
        anchored_ratio     = anchored_pings / total_pings
        avg_speed_in_zone  = sog_sum / total_pings
        avg_idle_duration  = idle_duration_sum / total_pings

    Parameters
    ----------
    combined : pd.DataFrame
        Raw batch aggregations with sum columns.

    Returns
    -------
    pd.DataFrame
        One row per (nearest_port, date) with correct daily totals.
    """
    log.info("Reducing batch sums to daily port totals ...")

    daily = (
        combined.groupby(["nearest_port", "date"])
        .agg(
            total_pings        = ("total_pings",        "sum"),
            vessels_in_zone    = ("unique_mmsi_approx", "sum"),  # approx — sum of nunique
            anchored_pings     = ("anchored_pings",     "sum"),
            slow_pings         = ("slow_pings",         "sum"),
            sog_sum            = ("sog_sum",            "sum"),
            idle_duration_sum  = ("idle_duration_sum",  "sum"),
        )
        .reset_index()
    )

    # Derive ratio features from correct daily sums
    safe_pings = daily["total_pings"].replace(0, np.nan)

    daily["slow_vessel_ratio"]     = (daily["slow_pings"]    / safe_pings).round(4)
    daily["anchored_ratio"]        = (daily["anchored_pings"] / safe_pings).round(4)
    daily["avg_speed_in_zone"]     = (daily["sog_sum"]        / safe_pings).round(3)
    daily["avg_idle_duration"]     = (daily["idle_duration_sum"] / safe_pings).round(2)

    # Rename pings to cleaner names
    daily.rename(columns={
        "anchored_pings": "anchored_vessel_count",
        "slow_pings":     "slow_vessel_count",
    }, inplace=True)

    # Drop intermediate sum columns no longer needed
    daily.drop(columns=["sog_sum", "idle_duration_sum"], inplace=True)

    daily.sort_values(["nearest_port", "date"], inplace=True)
    daily.reset_index(drop=True, inplace=True)

    log.info(f"  Daily rows : {len(daily):,}")
    log.info(f"  Ports      : {daily['nearest_port'].unique().tolist()}")
    log.info(f"  Date range : {daily['date'].min().date()} → {daily['date'].max().date()}")

    return daily


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def add_rolling_port_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add 7-day rolling averages per port using groupby + transform."""
    log.info(f"Adding {ROLLING_DAYS}-day rolling features ...")

    def rolling_mean(s: pd.Series) -> pd.Series:
        return s.rolling(window=ROLLING_DAYS, min_periods=3).mean()

    rolling_map = {
        "vessels_7d_rolling":    "vessels_in_zone",
        "anchored_7d_rolling":   "anchored_vessel_count",
        "slow_ratio_7d_rolling": "slow_vessel_ratio",
        "speed_7d_rolling":      "avg_speed_in_zone",
    }

    for new_col, src_col in rolling_map.items():
        if src_col in daily.columns:
            daily[new_col] = (
                daily.groupby("nearest_port", sort=False)[src_col]
                .transform(rolling_mean)
                .round(4)
            )
            log.info(f"  ✓ {new_col}")

    return daily


# ---------------------------------------------------------------------------
# Composite congestion score
# ---------------------------------------------------------------------------

def add_congestion_score(daily: pd.DataFrame) -> pd.DataFrame:
    """Weighted composite port_congestion_score in [0, 1]."""
    log.info("Computing port_congestion_score ...")

    idle_norm = (daily["avg_idle_duration"] / IDLE_DURATION_CAP).clip(0, 1)

    daily["port_congestion_score"] = (
        CONGESTION_WEIGHTS["slow_vessel_ratio"]  * daily["slow_vessel_ratio"].fillna(0)
        + CONGESTION_WEIGHTS["anchored_ratio"]   * daily["anchored_ratio"].fillna(0)
        + CONGESTION_WEIGHTS["idle_duration_norm"] * idle_norm.fillna(0)
    ).round(4)

    score = daily["port_congestion_score"]
    log.info(f"  Score range  : {score.min():.4f} → {score.max():.4f}")
    log.info(f"  Score mean   : {score.mean():.4f}")
    log.info(f"  Score median : {score.median():.4f}")

    return daily


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving {len(df):,} rows → {output_path}")
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = output_path.stat().st_size / 1e6
    log.info(f"  ✓ Saved — {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(daily: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  PORT FEATURES — ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"  Total rows     : {len(daily):,}")
    print(f"  Ports covered  : {daily['nearest_port'].nunique()}")
    print(f"  Date range     : {daily['date'].min().date()} → {daily['date'].max().date()}")

    print("\n  Congestion score by port (mean):")
    score_by_port = (
        daily.groupby("nearest_port")["port_congestion_score"]
        .mean()
        .sort_values(ascending=False)
    )
    for port, score in score_by_port.items():
        bar = "█" * int(score * 30)
        print(f"    {port:<25} {score:.4f}  {bar}")

    print("\n  Feature null rates:")
    key_cols = [
        "vessels_in_zone", "slow_vessel_ratio", "anchored_ratio",
        "avg_speed_in_zone", "avg_idle_duration",
        "vessels_7d_rolling", "slow_ratio_7d_rolling",
        "port_congestion_score",
    ]
    for col in key_cols:
        if col in daily.columns:
            null_pct = daily[col].isnull().mean() * 100
            flag     = "  ⚠" if null_pct > 10 else "  ✓"
            print(f"  {flag}  {col:<30} {null_pct:.1f}% null")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    vessel_path   = PROJECT_ROOT / config["paths"]["processed"] / "vessel_features.parquet"
    output_path   = PROJECT_ROOT / config["paths"]["processed"] / "port_features.parquet"
    ports         = config["ports"]
    radius_km     = config["anchorage"]["radius_km"]

    log.info("=" * 60)
    log.info("Phase 2 — Port Feature Engineering (Streaming)")
    log.info(f"  Anchorage radius : {radius_km} km")
    log.info(f"  Ports            : {[p['name'] for p in ports]}")
    log.info("=" * 60)

    if not vessel_path.exists():
        raise FileNotFoundError(
            f"vessel_features.parquet not found at {vessel_path}.\n"
            "Run vessel_features.py first."
        )

    # Step 1: Stream through parquet, aggregate each row group
    combined = stream_aggregate_all_batches(vessel_path, ports, radius_km)

    # Step 2: Reduce batch sums to correct daily totals
    daily = reduce_to_daily(combined)
    del combined

    # Step 3: Rolling features
    daily = add_rolling_port_features(daily)

    # Step 4: Congestion score
    daily = add_congestion_score(daily)

    # Step 5: Summary
    print_summary(daily)

    # Step 6: Save
    save_features(daily, output_path)

    log.info("Phase 2 Step 2 complete — port_features.parquet ready.")
    log.info("Next: run src/features/external_features.py")


if __name__ == "__main__":
    main()