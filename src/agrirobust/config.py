"""AgriRobust configuration loader and validation utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Return the absolute path to the AgriRobust repository root."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Safely load and parse a YAML file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML content in {path} must be a dictionary mapping.")
    return data


def get_project_config() -> Dict[str, Any]:
    """Load configs/project.yaml."""
    return load_yaml(get_project_root() / "configs" / "project.yaml")


def get_datasets_config() -> Dict[str, Any]:
    """Load configs/datasets.yaml."""
    return load_yaml(get_project_root() / "configs" / "datasets.yaml")


def get_metrics_config() -> Dict[str, Any]:
    """Load configs/metrics.yaml."""
    return load_yaml(get_project_root() / "configs" / "metrics.yaml")
