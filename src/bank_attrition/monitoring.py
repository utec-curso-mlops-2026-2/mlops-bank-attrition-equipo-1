"""Data drift con PSI (R9): población de entrenamiento (referencia) vs. población de aplicación.

PSI = Σ (actual_i − ref_i) · ln(actual_i / ref_i) sobre bins de la variable:
- numéricas: bins por cuantiles de la referencia (los extremos se abren a ±∞) y un bin propio para nulos;
- categóricas: una categoría por bin y otro para nulos.
Semáforo (lineamientos del curso): PSI < 0,10 estable · 0,10–0,25 alerta · > 0,25 crítico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4  # evita log(0) cuando un bin queda vacío en una de las poblaciones
NULL_BIN = "__nulo__"


def _psi_from_shares(ref: pd.Series, cur: pd.Series) -> float:
    bins = ref.index.union(cur.index)
    ref = ref.reindex(bins, fill_value=0.0).clip(lower=EPS)
    cur = cur.reindex(bins, fill_value=0.0).clip(lower=EPS)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def numeric_edges(reference, bins: int = 10) -> np.ndarray:
    """Cortes por cuantiles de la referencia (sin duplicados), con extremos abiertos."""
    values = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return np.array([-np.inf, np.inf])
    inner = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def _numeric_shares(values, edges: np.ndarray) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    labels = pd.cut(series, edges, right=True, include_lowest=True).astype(str)
    labels = labels.where(series.notna(), NULL_BIN)
    return labels.value_counts(normalize=True)


def psi_numeric(reference, current, bins: int = 10) -> float:
    edges = numeric_edges(reference, bins)
    return _psi_from_shares(_numeric_shares(reference, edges), _numeric_shares(current, edges))


def _categorical_shares(values) -> pd.Series:
    series = pd.Series(values, dtype="object")
    return series.where(series.notna(), NULL_BIN).astype(str).value_counts(normalize=True)


def psi_categorical(reference, current) -> float:
    return _psi_from_shares(_categorical_shares(reference), _categorical_shares(current))


def traffic_light(psi: float, alert: float = 0.10, critical: float = 0.25) -> str:
    if psi > critical:
        return "critico"
    if psi >= alert:
        return "alerta"
    return "estable"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str],
                 categorical: list[str], reference_score, current_score, bins: int = 10,
                 alert: float = 0.10, critical: float = 0.25) -> pd.DataFrame:
    """Una fila por variable (y una para el score), ordenada de mayor a menor PSI."""
    rows = []
    for col in numeric:
        rows.append(("variable", col, "numerica", psi_numeric(reference[col], current[col], bins)))
    for col in categorical:
        rows.append(("variable", col, "categorica", psi_categorical(reference[col], current[col])))
    rows.append(("score", "probabilidad_fuga", "numerica",
                 psi_numeric(reference_score, current_score, bins)))
    report = pd.DataFrame(rows, columns=["nivel", "variable", "tipo", "psi"])
    report["estado"] = [traffic_light(p, alert, critical) for p in report["psi"]]
    return report.sort_values(["nivel", "psi"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def drift_summary(report: pd.DataFrame) -> dict:
    """Resumen para MLflow y para la decisión de reentrenar (insumo del plan de monitoreo)."""
    variables = report[report["nivel"] == "variable"]
    score = report[report["nivel"] == "score"].iloc[0]
    counts = variables["estado"].value_counts()
    return {
        "psi_score": float(score["psi"]),
        "estado_score": str(score["estado"]),
        "psi_max_variable": float(variables["psi"].max()) if len(variables) else 0.0,
        "variable_psi_max": str(variables.iloc[0]["variable"]) if len(variables) else "",
        "n_variables": int(len(variables)),
        "n_estable": int(counts.get("estable", 0)),
        "n_alerta": int(counts.get("alerta", 0)),
        "n_critico": int(counts.get("critico", 0)),
    }
