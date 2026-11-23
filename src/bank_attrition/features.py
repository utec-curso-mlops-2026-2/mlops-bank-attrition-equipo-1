"""Cruce clientes + requerimientos y construcción de features (R2, paso 1).

Regla clave: una fila por cliente y corte (CODMES). Los requerimientos se agregan en una
ventana de `window_months` meses que termina en el corte del cliente; nunca se usan
requerimientos posteriores al corte (evita fuga de información).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_index(yyyymm: pd.Series) -> pd.Series:
    """Convierte yyyymm a un entero de meses (permite restar meses de forma segura)."""
    values = pd.to_numeric(yyyymm, errors="coerce")
    return (values // 100) * 12 + (values % 100) - 1


def to_ordinal(series: pd.Series) -> pd.Series:
    """'Rango_SDO_03' -> 3.0, 'Cero' -> 0.0, nulo -> NaN (el orden de los rangos es informativo)."""
    text = series.astype("object")
    number = pd.to_numeric(text.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    number = number.where(~text.astype(str).str.lower().eq("cero"), 0.0)
    return number.where(series.notna())


def fit_feature_spec(clientes: pd.DataFrame, requerimientos: pd.DataFrame, cfg: dict) -> dict:
    """Define, SOLO con train, las columnas y categorías que generan features de requerimientos."""
    schema, feats = cfg["schema"], cfg["features"]
    skip = {schema["id_col"], schema["req_period_col"]}
    candidates = [c for c in requerimientos.columns if c not in skip]
    numeric = schema["req_numeric_cols"] or [
        c for c in candidates if pd.api.types.is_numeric_dtype(requerimientos[c])
    ]
    categorical = schema["req_categorical_cols"] or [c for c in candidates if c not in numeric]
    categories: dict[str, list[str]] = {}
    for col in categorical:
        counts = requerimientos[col].astype(str).value_counts()
        keep = counts[counts >= feats["min_category_count"]].index[: feats["max_categories_per_col"]]
        categories[col] = sorted(keep)
    return {
        "window_months": int(feats["window_months"]),
        "numeric": list(numeric),
        "categorical": list(categorical),
        "categories": categories,
    }


def _aggregate_requirements(
    clientes: pd.DataFrame, requerimientos: pd.DataFrame, spec: dict, cfg: dict
) -> pd.DataFrame:
    schema = cfg["schema"]
    id_col, period, req_period = schema["id_col"], schema["period_col"], schema["req_period_col"]
    cut = clientes[[id_col, period]].rename(columns={period: "_cut_period"})
    cut["_cut_idx"] = month_index(cut["_cut_period"])
    req = requerimientos.copy()
    req["_req_idx"] = month_index(req[req_period])
    merged = req.merge(cut, on=id_col, how="inner")
    window = int(spec["window_months"])
    in_window = (merged["_req_idx"] <= merged["_cut_idx"]) & (
        merged["_req_idx"] > merged["_cut_idx"] - window
    )
    merged = merged.loc[in_window]
    keys = [id_col, "_cut_period"]
    agg = merged.groupby(keys).size().rename("N_REQUERIMIENTOS_VENTANA").to_frame()
    for col in spec["numeric"]:
        grouped = merged.groupby(keys)[col]
        agg[f"REQ_{col}_SUM"] = grouped.sum()
        agg[f"REQ_{col}_MEAN"] = grouped.mean()
        agg[f"REQ_{col}_MAX"] = grouped.max()
    for col in spec["categorical"]:
        as_text = merged[col].astype(str)
        agg[f"REQ_{col}_NUNIQUE"] = as_text.groupby([merged[k] for k in keys]).nunique()
        for category in spec["categories"].get(col, []):
            hit = (as_text == category).groupby([merged[k] for k in keys]).sum()
            agg[f"REQ_{col}_{category}_N"] = hit
    return agg.reset_index().rename(columns={"_cut_period": period})


def build_features(
    clientes: pd.DataFrame, requerimientos: pd.DataFrame, spec: dict, cfg: dict
) -> pd.DataFrame:
    """Tabla modelable: una fila por (cliente, corte) con metadatos, predictores y, si existe, target."""
    schema = cfg["schema"]
    id_col, period = schema["id_col"], schema["period_col"]
    base = clientes.drop(columns=[c for c in schema["drop_cols"] if c in clientes.columns]).copy()
    if base.duplicated([id_col, period]).any():
        raise ValueError(f"clientes tiene más de una fila por ({id_col}, {period})")
    agg = _aggregate_requirements(clientes, requerimientos, spec, cfg)
    for col in schema.get("ordinal_cols", []):
        if col in base.columns:
            base[col + "_ORD"] = to_ordinal(base[col])
            base = base.drop(columns=[col])
    table = base.merge(agg, on=[id_col, period], how="left")
    count_like = [c for c in agg.columns if c.endswith(("_SUM", "_N", "_NUNIQUE", "VENTANA"))]
    table[count_like] = table[count_like].fillna(0)  # sin requerimientos = 0 eventos
    return table


def split_columns(table: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) de predictores: todo menos id, periodo y target."""
    schema = cfg["schema"]
    excluded = {schema["id_col"], schema["period_col"], schema["target_col"]}
    predictors = [c for c in table.columns if c not in excluded]
    numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(table[c])]
    categorical = [c for c in predictors if c not in numeric]
    return numeric, categorical


def prepare_input(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Tipos estables para la signature: numéricas float64, categóricas texto.

    Columnas ausentes se crean como NaN (el imputador del pipeline las trata).
    """
    out = pd.DataFrame(index=frame.index)
    for col in numeric:
        out[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64") if col in frame else np.nan
    for col in categorical:
        out[col] = frame[col].astype("object").where(frame[col].notna(), None) if col in frame else None
    return out
