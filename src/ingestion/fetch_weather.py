"""
fetch_weather.py
----------------
Fetches historical hourly weather data from the Open-Meteo Archive API
(https://open-meteo.com) for every port defined in config.yaml.

No API key required. Data is free and publicly available.

For each port, this script fetches:
  - windspeed_10m      : wind speed at 10m height (km/h)
  - winddirection_10m  : wind direction in degrees
  - precipitation      : rainfall in mm
  - weathercode        : WMO weather interpretation code (0=clear, 95=storm, etc.)

Each port gets its own CSV file:
  data/raw/weather/weather_los_angeles.csv
  data/raw/weather/weather_rotterdam.csv
  ... etc.

Usage (from project root):
    python src/ingestion/fetch_weather.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

from src.config_loader import load_config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def port_to_filename(port_name: str) -> str:
    """
    Convert a human-readable port name to a safe filename.

    Example: "Los Angeles" → "weather_los_angeles.csv"
    """
    safe_name = port_name.lower().replace(" ", "_")
    return f"weather_{safe_name}.csv"


def fetch_weather_for_port(
    port: dict,
    start_date: str,
    end_date: str,
    config: dict,
) -> pd.DataFrame:
    """
    Call Open-Meteo Archive API for a single port and return a DataFrame.

    Parameters
    ----------
    port : dict
        Port entry from config.yaml with keys: name, lat, lon.
    start_date : str
        ISO date string, e.g. "2023-01-01".
    end_date : str
        ISO date string, e.g. "2023-12-31".
    config : dict
        Full config dictionary (used for API URL and variable list).

    Returns
    -------
    pd.DataFrame
        Hourly weather data for the port, with a 'port' column added.
    """
    weather_cfg = config["weather"]

    params = {
        "latitude":           port["lat"],
        "longitude":          port["lon"],
        "start_date":         start_date,
        "end_date":           end_date,
        "hourly":             ",".join(weather_cfg["hourly_variables"]),
        "timezone":           weather_cfg["timezone"],
    }

    print(f"  [Weather] Fetching {port['name']} ({start_date} → {end_date}) ...")

    # Retry logic: try up to 3 times with a pause between attempts.
    # Open-Meteo is reliable but occasional timeouts happen.
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                weather_cfg["base_url"],
                params=params,
                timeout=30
            )
            response.raise_for_status()   # raises on 4xx / 5xx
            break                         # success — exit retry loop

        except requests.exceptions.RequestException as e:
            print(f"  [Weather] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to fetch weather for {port['name']} after "
                    f"{max_retries} attempts."
                ) from e
            time.sleep(5 * attempt)       # back-off: 5s, 10s, 15s

    data = response.json()

    # Open-Meteo returns {"hourly": {"time": [...], "windspeed_10m": [...], ...}}
    # We convert this to a flat DataFrame.
    if "hourly" not in data:
        raise ValueError(
            f"Unexpected API response for {port['name']}: 'hourly' key missing. "
            f"Response: {data}"
        )

    df = pd.DataFrame(data["hourly"])
    df["port"] = port["name"]
    df["lat"]  = port["lat"]
    df["lon"]  = port["lon"]

    # Rename 'time' to 'timestamp' for consistency across the project
    df.rename(columns={"time": "timestamp"}, inplace=True)

    return df


def save_weather_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    """
    Save a weather DataFrame to CSV in the output directory.

    Parameters
    ----------
    df : pd.DataFrame
        Weather data for one port.
    output_dir : Path
        Target directory (data/raw/weather/).
    filename : str
        Output filename, e.g. "weather_los_angeles.csv".

    Returns
    -------
    Path
        Full path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    return filepath


def main():
    config = load_config()
    root = get_project_root()

    output_dir  = root / config["paths"]["raw_weather"]
    start_date  = config["data"]["start_date"]
    end_date    = config["data"]["end_date"]
    ports       = config["ports"]

    print(f"[Weather] Starting weather ingestion for {len(ports)} ports")
    print(f"[Weather] Date range: {start_date} → {end_date}")
    print(f"[Weather] Output: {output_dir}\n")

    results = []

    for port in ports:
        filename = port_to_filename(port["name"])
        filepath = output_dir / filename

        # Skip if already downloaded — no need to re-hit the API
        if filepath.exists():
            print(f"  [Weather] {port['name']} — already exists, skipping.")
            results.append({"port": port["name"], "status": "skipped", "file": filename})
            continue

        try:
            df = fetch_weather_for_port(port, start_date, end_date, config)
            saved_path = save_weather_csv(df, output_dir, filename)

            print(f"  [Weather] {port['name']} — ✓ {len(df):,} rows saved → {filename}")
            results.append({"port": port["name"], "status": "success", "file": filename, "rows": len(df)})

        except Exception as e:
            print(f"  [Weather] {port['name']} — ✗ FAILED: {e}")
            results.append({"port": port["name"], "status": "failed", "error": str(e)})

        # Be polite to the API: small pause between port requests
        time.sleep(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n[Weather] ── Ingestion Summary ─────────────────────────────────")
    for r in results:
        status_icon = "✓" if r["status"] == "success" else ("–" if r["status"] == "skipped" else "✗")
        row_info = f"  ({r.get('rows', '?'):,} rows)" if "rows" in r else ""
        print(f"  {status_icon}  {r['port']:<20} {r['status']}{row_info}")

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print(f"\n[Weather] WARNING: {len(failed)} port(s) failed. Check errors above.")
    else:
        print(f"\n[Weather] ✓ All ports complete. Raw files at: {output_dir}")


if __name__ == "__main__":
    main()
