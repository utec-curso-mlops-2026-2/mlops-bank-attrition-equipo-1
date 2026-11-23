"""Carga de configuración: YAML con sustitución de variables de entorno ${VAR}."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _substitute(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    if isinstance(value, str):
        whole = _ENV.fullmatch(value.strip())
        if whole:  # "${VAR}" solo -> valor del entorno o None si no existe
            return os.environ.get(whole.group(1)) or None
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), value)
    return value


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Lee el YAML, sustituye variables de entorno y aplica valores por defecto."""
    with open(path, encoding="utf-8") as handle:
        cfg = _substitute(yaml.safe_load(handle))
    cfg["project"]["author"] = cfg["project"].get("author") or os.environ.get("USER") or "unknown"
    mlflow_cfg = cfg["mlflow"]
    mlflow_cfg["experiment_name"] = mlflow_cfg.get("experiment_name") or "bank-attrition"
    mlflow_cfg["tracking_uri"] = mlflow_cfg.get("tracking_uri") or mlflow_cfg["fallback_tracking_uri"]
    return cfg
