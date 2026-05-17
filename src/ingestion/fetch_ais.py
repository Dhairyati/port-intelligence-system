"""
fetch_ais.py
------------
Downloads the AIS (Automatic Identification System) vessel tracking dataset
from Kaggle and saves it to data/raw/ais/.

AIS data contains one row per vessel position ping, with columns like:
  - MMSI         : unique vessel identifier
  - LAT / LON    : vessel coordinates at that moment
  - SPEED        : speed over ground in knots
  - COURSE       : heading in degrees
  - TIMESTAMP    : UTC datetime of the ping
  - SHIP_TYPE    : vessel category (cargo, tanker, etc.)

This is the raw movement data we will later use to engineer features like
anchorage waiting time and speed drop near ports.

Usage (from project root):
    python src/ingestion/fetch_ais.py
"""

import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file lives at src/ingestion/fetch_ais.py → root is 2 levels up
    return Path(__file__).resolve().parents[2]


def download_ais_dataset(output_dir: Path, dataset_slug: str) -> Path:
    """
    Download a Kaggle dataset using the Kaggle CLI.

    Parameters
    ----------
    output_dir : Path
        Where to save the downloaded zip and extracted files.
    dataset_slug : str
        Kaggle dataset identifier in the form "owner/dataset-name".

    Returns
    -------
    Path
        Path to the output directory containing extracted files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded to avoid re-downloading on repeat runs
    existing_csvs = list(output_dir.glob("*.csv"))
    if existing_csvs:
        print(f"[AIS] Data already exists at {output_dir}. Skipping download.")
        print(f"[AIS] Found: {[f.name for f in existing_csvs]}")
        return output_dir

    print(f"[AIS] Downloading dataset: {dataset_slug}")
    print(f"[AIS] Saving to: {output_dir}")

    # Kaggle CLI command: downloads zip into output_dir
    command = [
        "kaggle", "datasets", "download",
        dataset_slug,
        "--path", str(output_dir),
        "--unzip"          # automatically unzip after download
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("[AIS] ERROR: Kaggle download failed.")
        print("[AIS] STDOUT:", result.stdout)
        print("[AIS] STDERR:", result.stderr)
        print()
        print("[AIS] Troubleshooting checklist:")
        print("  1. Is kaggle installed?         pip install kaggle")
        print("  2. Is kaggle.json in place?     ~/.kaggle/kaggle.json")
        print("  3. Correct permissions?         chmod 600 ~/.kaggle/kaggle.json")
        print("  4. Have you accepted dataset terms on Kaggle website?")
        raise RuntimeError("Kaggle download failed. See errors above.")

    print(f"[AIS] Download complete.")
    return output_dir


def load_and_validate_ais(output_dir: Path, filename: str) -> pd.DataFrame:
    """
    Load the raw AIS CSV into a DataFrame and run basic sanity checks.

    Parameters
    ----------
    output_dir : Path
        Directory containing the downloaded CSV.
    filename : str
        Expected filename from config.yaml (ais.filename).

    Returns
    -------
    pd.DataFrame
        Raw AIS data, unchanged — we never modify raw data.
    """
    csv_path = output_dir / filename

    # Some Kaggle datasets name their file differently after extraction.
    # If the exact name isn't found, pick the first CSV in the folder.
    if not csv_path.exists():
        csvs = list(output_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in {output_dir}. "
                "Check if the Kaggle download completed successfully."
            )
        csv_path = csvs[0]
        print(f"[AIS] Expected '{filename}' not found. Using '{csv_path.name}' instead.")

    print(f"\n[AIS] Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    print(f"\n[AIS] ── Basic Info ──────────────────────────────────")
    print(f"       Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"       Columns    : {list(df.columns)}")
    print(f"       Memory     : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print(f"\n[AIS] ── Null Counts ─────────────────────────────────")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0].to_string() if null_counts.any() else "       No nulls found.")

    print(f"\n[AIS] ── Sample Rows ─────────────────────────────────")
    print(df.head(3).to_string())

    return df


def main():
    config = load_config()
    root = get_project_root()

    # Resolve output directory from config (relative → absolute)
    output_dir = root / config["paths"]["raw_ais"]
    dataset_slug = config["ais"]["kaggle_dataset"]
    filename = config["ais"]["filename"]

    # Step 1: Download
    download_ais_dataset(output_dir, dataset_slug)

    # Step 2: Load and validate
    df = load_and_validate_ais(output_dir, filename)

    print(f"\n[AIS] ✓ Phase 1 ingestion complete.")
    print(f"[AIS]   Raw data saved at: {output_dir}")
    print(f"[AIS]   Rows loaded: {len(df):,}")


if __name__ == "__main__":
    main()
