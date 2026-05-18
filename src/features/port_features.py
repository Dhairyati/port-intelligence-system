"""
port_features.py
----------------
Phase 2 — Port-Level Congestion Feature Engineering

Loads vessel-level features from data/processed/vessel_features.parquet,
spatially assigns each AIS ping to the nearest configured port (within the
anchorage radius defined in config.yaml), then aggregates vessel behavior
into daily port-level congestion signals.

No external geo libraries required — distance is calculated with the
Haversine formula using pure NumPy, which vectorises across millions of rows.

Features engineered (one row per port per day)
----------------------------------------------
Daily aggregations:
    vessels_in_zone       : unique vessels within anchorage radius that day
    anchored_vessel_count : unique vessels with anchored_flag=1 that day
    slow_vessel_count     : unique vessels with slow_moving_flag=1 that day
    slow_vessel_ratio     : slow_vessel_count / vessels_in_zone
    avg_speed_in_zone     : mean SOG of all pings near port that day
    avg_idle_duration     : mean idle_duration_proxy of slow vessels
    total_pings           : total AIS pings received near port that day

7-day rolling averages (smoothed trends):
    vessels_7d_rolling    : rolling mean of vessels_in_zone
    anchored_7d_rolling   : rolling mean of anchored_vessel_count
    slow_ratio_7d_rolling : rolling mean of slow_vessel_ratio

Composite score:
    port_congestion_score : 0–1 composite of slow ratio + anchored ratio
                            + normalised idle duration. Higher = more congested.

Usage (from project root):
    python src/features/port_features.py
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

# Radius of the Earth in kilometres — used in Haversine formula
EARTH_RADIUS_KM = 6371.0

# Rolling window in days for smoothed port-level trend features
ROLLING_DAYS = 7

# Weights for the composite port_congestion_score (must sum to 1.0)
# Slow ratio is weighted most heavily — it is the most direct congestion signal
CONGESTION_WEIGHTS = {
    "slow_vessel_ratio":   0.45,
    "anchored_ratio":      0.35,    # anchored_vessel_count / vessels_in_zone
    "idle_duration_norm":  0.20,    # normalised avg_idle_duration (0–1)
}

# idle_duration values above this are treated as maximum congestion (clipped to 1.0)
IDLE_DURATION_CAP = 50


# ---------------------------------------------------------------------------
# 1. Load Vessel Features
# ---------------------------------------------------------------------------

def load_vessel_features(processed_dir: Path) -> pd.DataFrame:
    """
    Load the vessel_features.parquet file produced by vessel_features.py.

    Parameters
    ----------
    processed_dir : Path
        Path to data/processed/ directory.

    Returns
    -------
    pd.DataFrame
        Vessel-level feature table with spatial and temporal columns.

    Raises
    ------
    FileNotFoundError
        If vessel_features.parquet does not exist.
    """
    parquet_path = processed_dir / "vessel_features.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"vessel_features.parquet not found at {parquet_path}.\n"
            "Run: python src/features/vessel_features.py first."
        )

    log.info(f"Loading vessel features from {parquet_path} ...")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    log.info(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")
    log.info(f"  Vessels (MMSI): {df['mmsi'].nunique():,}")
    log.info(f"  Date range    : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    return df


# ---------------------------------------------------------------------------
# 2. Haversine Distance (vectorised)
# ---------------------------------------------------------------------------

def haversine_vectorised(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """
    Compute great-circle distance between each (lat1, lon1) point and a
    single fixed point (lat2, lon2), using the Haversine formula.

    Fully vectorised — operates on NumPy arrays, no Python loops.
    At 1 million rows, this runs in ~100ms.

    Parameters
    ----------
    lat1, lon1 : np.ndarray
        Arrays of vessel latitudes and longitudes (degrees).
    lat2, lon2 : float
        Fixed port latitude and longitude (degrees).

    Returns
    -------
    np.ndarray
        Array of distances in kilometres, same length as lat1/lon1.
    """
    # Convert degrees → radians
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# 3. Assign Vessels to Ports
# ---------------------------------------------------------------------------

def assign_vessels_to_ports(df: pd.DataFrame, ports: list, radius_km: float) -> pd.DataFrame:
    """
    For each AIS ping, determine which configured port (if any) it is within
    radius_km of. Assigns the nearest port within that radius.

    A ping can only belong to ONE port — the nearest one within radius_km.
    Pings outside radius_km of all ports are dropped (open-ocean pings
    carry no port congestion signal).

    Parameters
    ----------
    df : pd.DataFrame
        Vessel feature DataFrame with 'lat' and 'lon' columns.
    ports : list of dict
        Port definitions from config.yaml (name, lat, lon).
    radius_km : float
        Maximum distance (km) for a vessel to be considered "near" a port.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame — only rows within radius_km of at least one port,
        with 'nearest_port' and 'dist_to_port_km' columns added.
    """
    log.info(f"Assigning {len(df):,} pings to {len(ports)} ports (radius={radius_km} km) ...")

    lat_arr = df["lat"].to_numpy(dtype=np.float64)
    lon_arr = df["lon"].to_numpy(dtype=np.float64)

    # For each ping, compute distance to every port
    # Shape: (n_pings, n_ports)
    distance_matrix = np.column_stack([
        haversine_vectorised(lat_arr, lon_arr, port["lat"], port["lon"])
        for port in ports
    ])

    # Find the nearest port index and distance for each ping
    nearest_port_idx = np.argmin(distance_matrix, axis=1)
    nearest_dist_km  = distance_matrix[np.arange(len(df)), nearest_port_idx]

    # Only keep pings within the radius
    within_radius_mask = nearest_dist_km <= radius_km

    log.info(f"  Pings within {radius_km} km of any port: {within_radius_mask.sum():,} "
             f"({within_radius_mask.mean()*100:.1f}% of total)")

    df = df[within_radius_mask].copy()
    df["nearest_port"]     = [ports[i]["name"] for i in nearest_port_idx[within_radius_mask]]
    df["dist_to_port_km"]  = nearest_dist_km[within_radius_mask].round(2)

    # Log how many pings were assigned to each port
    port_ping_counts = df["nearest_port"].value_counts()
    log.info("  Pings per port:")
    for port_name, count in port_ping_counts.items():
        log.info(f"    {port_name:<25} {count:>10,}")

    return df


# ---------------------------------------------------------------------------
# 4. Build Daily Port Time Series
# ---------------------------------------------------------------------------

def build_port_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vessel pings to daily port-level observations.

    Each row in the output represents one port on one day, summarising
    the collective behaviour of all vessels that were within the anchorage
    zone that day.

    Parameters
    ----------
    df : pd.DataFrame
        Ping-level DataFrame with 'nearest_port', 'timestamp', and all
        vessel feature columns.

    Returns
    -------
    pd.DataFrame
        Daily port-level DataFrame, sorted by port and date.
    """
    log.info("Building daily port-level time series ...")

    # Create a date column for grouping (date only, no time component)
    df["date"] = df["timestamp"].dt.normalize()   # midnight of each day

    # ── Daily aggregations ──────────────────────────────────────────────────
    # nunique(mmsi) = number of distinct vessels in the zone that day
    daily = df.groupby(["nearest_port", "date"]).agg(

        vessels_in_zone       = ("mmsi",               "nunique"),
        total_pings           = ("mmsi",               "count"),
        avg_speed_in_zone     = ("sog",                "mean"),

        # For anchored/slow counts: number of unique vessels that had at
        # least one anchored/slow ping near this port today.
        # We use max per vessel (1 if any ping was slow) then sum.
        anchored_vessel_count = ("anchored_flag",      "sum"),
        slow_vessel_count     = ("slow_moving_flag",   "sum"),
        avg_idle_duration     = ("idle_duration_proxy","mean"),

    ).reset_index()

    # ── Derived ratios ──────────────────────────────────────────────────────
    # Avoid division by zero with np.where
    daily["slow_vessel_ratio"] = np.where(
        daily["vessels_in_zone"] > 0,
        daily["slow_vessel_count"] / daily["total_pings"],
        0.0,
    ).round(4)

    daily["anchored_ratio"] = np.where(
        daily["vessels_in_zone"] > 0,
        daily["anchored_vessel_count"] / daily["total_pings"],
        0.0,
    ).round(4)

    daily["avg_speed_in_zone"]  = daily["avg_speed_in_zone"].round(3)
    daily["avg_idle_duration"]  = daily["avg_idle_duration"].round(2)

    log.info(f"  Daily observations: {len(daily):,} rows "
             f"({daily['nearest_port'].nunique()} ports × ~{len(daily)//max(daily['nearest_port'].nunique(),1)} days avg)")

    return daily


# ---------------------------------------------------------------------------
# 5. Add Rolling Port Features
# ---------------------------------------------------------------------------

def add_rolling_port_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7-day rolling averages for key port congestion metrics.

    Rolling features smooth out daily noise and capture the trending
    congestion pressure over the past week — much more useful for
    a classifier than a single noisy daily value.

    All rolling windows are computed per port independently using
    groupby + transform to avoid cross-port contamination.

    Parameters
    ----------
    daily : pd.DataFrame
        Daily port-level DataFrame sorted by port and date.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with rolling feature columns added.
    """
    log.info(f"Adding {ROLLING_DAYS}-day rolling port features ...")

    # Sort chronologically per port before rolling
    daily.sort_values(["nearest_port", "date"], inplace=True)
    daily.reset_index(drop=True, inplace=True)

    def rolling_mean(series: pd.Series, window: int) -> pd.Series:
        """Rolling mean with min 3 periods to avoid stats on sparse early days."""
        return series.rolling(window=window, min_periods=3).mean()

    rolling_cols = {
        "vessels_7d_rolling":    "vessels_in_zone",
        "anchored_7d_rolling":   "anchored_vessel_count",
        "slow_ratio_7d_rolling": "slow_vessel_ratio",
        "speed_7d_rolling":      "avg_speed_in_zone",
    }

    for new_col, source_col in rolling_cols.items():
        daily[new_col] = (
            daily.groupby("nearest_port", sort=False)[source_col]
            .transform(lambda x: rolling_mean(x, ROLLING_DAYS))
            .round(4)
        )
        log.info(f"  ✓ {new_col}")

    return daily


# ---------------------------------------------------------------------------
# 6. Compute Composite Congestion Score
# ---------------------------------------------------------------------------

def add_congestion_score(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite port_congestion_score (0–1) that combines the
    three most informative daily congestion signals into a single index.

    This score is what the downstream XGBoost classifier will predict,
    and what the dashboard will display as a port risk level.

    Formula
    -------
    port_congestion_score = (
        0.45 × slow_vessel_ratio
      + 0.35 × anchored_ratio
      + 0.20 × clip(avg_idle_duration / IDLE_DURATION_CAP, 0, 1)
    )

    All three inputs are already in [0, 1] range (ratios) or normalised
    to [0, 1] via the cap. The weighted sum is therefore also in [0, 1].

    Parameters
    ----------
    daily : pd.DataFrame
        Daily port-level DataFrame with slow_vessel_ratio, anchored_ratio,
        and avg_idle_duration columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 'port_congestion_score' column added.
    """
    log.info("Computing composite port_congestion_score ...")

    # Normalise idle duration to [0, 1]
    idle_norm = (daily["avg_idle_duration"] / IDLE_DURATION_CAP).clip(0, 1)

    daily["port_congestion_score"] = (
        CONGESTION_WEIGHTS["slow_vessel_ratio"]  * daily["slow_vessel_ratio"]
        + CONGESTION_WEIGHTS["anchored_ratio"]   * daily["anchored_ratio"]
        + CONGESTION_WEIGHTS["idle_duration_norm"] * idle_norm
    ).round(4)

    score = daily["port_congestion_score"]
    log.info(f"  Score range : {score.min():.4f} → {score.max():.4f}")
    log.info(f"  Score mean  : {score.mean():.4f}")
    log.info(f"  Score median: {score.median():.4f}")

    # Sanity check — should always be within [0, 1]
    out_of_range = ((score < 0) | (score > 1)).sum()
    if out_of_range > 0:
        log.warning(f"  ⚠ {out_of_range} scores outside [0,1] — check input columns")

    return daily


# ---------------------------------------------------------------------------
# 7. Save Output
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the port-level feature DataFrame to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Port-level daily feature table.
    output_path : Path
        Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving {len(df):,} rows → {output_path} ...")
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"  ✓ Saved — file size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# 8. Summary Report
# ---------------------------------------------------------------------------

def print_summary(daily: pd.DataFrame) -> None:
    """
    Print a concise validation summary of the port feature table.
    """
    engineered_cols = [
        "vessels_in_zone", "anchored_vessel_count", "slow_vessel_count",
        "slow_vessel_ratio", "anchored_ratio", "avg_speed_in_zone",
        "avg_idle_duration", "vessels_7d_rolling", "anchored_7d_rolling",
        "slow_ratio_7d_rolling", "speed_7d_rolling", "port_congestion_score",
    ]

    print("\n" + "=" * 60)
    print("  PORT FEATURES — ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"  Total rows     : {len(daily):,}")
    print(f"  Ports covered  : {daily['nearest_port'].nunique()}")
    print(f"  Date range     : {daily['date'].min().date()} → {daily['date'].max().date()}")
    print()

    # Per-port congestion score summary
    print("  Congestion score by port (mean):")
    score_by_port = (
        daily.groupby("nearest_port")["port_congestion_score"]
        .mean()
        .sort_values(ascending=False)
    )
    for port, score in score_by_port.items():
        bar = "█" * int(score * 30)
        print(f"    {port:<25} {score:.4f}  {bar}")

    print()
    print("  Engineered feature status:")
    for col in engineered_cols:
        if col in daily.columns:
            null_count = daily[col].isnull().sum()
            null_info  = f"  ({null_count:,} nulls)" if null_count > 0 else ""
            print(f"    ✓  {col:<30} dtype={daily[col].dtype}{null_info}")
        else:
            print(f"    ✗  {col:<30} MISSING")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    processed_dir = PROJECT_ROOT / config["paths"]["processed"]
    output_path   = processed_dir / "port_features.parquet"
    ports         = config["ports"]
    radius_km     = config["anchorage"]["radius_km"]

    log.info("=" * 60)
    log.info("Phase 2 — Port Feature Engineering")
    log.info("=" * 60)

    # Step 1: Load vessel features
    df = load_vessel_features(processed_dir)

    # Step 2: Assign each ping to nearest port within radius
    df = assign_vessels_to_ports(df, ports, radius_km)

    if len(df) == 0:
        log.error(
            "No vessel pings fell within the anchorage radius of any configured port.\n"
            "Possible causes:\n"
            "  1. Your AIS data covers a different geographic region than the configured ports.\n"
            "     The NOAA Jan 1-5 dataset covers US waters — ensure ports include US locations\n"
            "     (Los Angeles is configured and should capture vessel traffic).\n"
            "  2. The anchorage radius_km in config.yaml may be too small — try increasing to 200.\n"
            "  3. Verify lat/lon columns parsed correctly in vessel_features.parquet."
        )
        return

    # Step 3: Aggregate to daily port time series
    daily = build_port_time_series(df)

    # Step 4: Add rolling trend features
    daily = add_rolling_port_features(daily)

    # Step 5: Add composite congestion score
    daily = add_congestion_score(daily)

    # Step 6: Summary
    print_summary(daily)

    # Step 7: Save
    save_features(daily, output_path)

    log.info("Phase 2 Step 2 complete — port_features.parquet ready.")
    log.info("Next: run src/features/external_features.py")


if __name__ == "__main__":
    main()
