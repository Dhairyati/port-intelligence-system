"""
config_loader.py
----------------
Loads config.yaml from the project root and returns it as a dictionary.
Every script in the project imports this instead of re-implementing YAML loading.

Usage:
    from src.utils.config_loader import load_config
    config = load_config()
    ports = config["ports"]
"""

import yaml
from pathlib import Path


def load_config() -> dict:
    """
    Locate config.yaml relative to the project root and return its contents.

    Returns
    -------
    dict
        Parsed contents of config.yaml.

    Raises
    ------
    FileNotFoundError
        If config.yaml is not found at the expected location.
    """
    # Walk up from this file's location to find the project root
    # This file lives at: src/config_loader.py
    # Project root is two levels up.
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Make sure you are running from the project root."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
