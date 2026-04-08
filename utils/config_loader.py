from __future__ import annotations

"""
Config loader — reads config.yaml once and returns it as a dict.

All parameters (weights, thresholds, assumptions) live in config.yaml.
This module is the single entry point for configuration across the project.
"""

import os
from pathlib import Path

import yaml


_CONFIG_CACHE: dict | None = None


def load_config(path: str | None = None) -> dict:
    """Load config.yaml and return as a dict. Caches after first read.

    Parameters
    ----------
    path : str, optional
        Explicit path to config.yaml. If None, looks for config.yaml in the
        project root (two levels up from this file).

    Returns
    -------
    dict
        Full configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    if path is None:
        project_root = Path(__file__).resolve().parent.parent
        path = str(project_root / "config.yaml")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if path is None or Path(path).name == "config.yaml":
        _CONFIG_CACHE = config

    return config


def reset_cache() -> None:
    """Clear the cached config (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
