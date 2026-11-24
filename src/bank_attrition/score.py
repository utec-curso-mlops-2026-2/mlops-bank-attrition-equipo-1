"""Scoreo batch del dataset de aplicación (OOT) con el modelo del Registry (R2 paso 6, núcleo de R7).

Flujo:
1. Carga el modelo **por alias** desde el Model Registry (nunca desde un archivo local) y obtiene
   su versión y el `feature_spec` con el que se entrenó.
2. Lee clientes y requerimientos del OOT (`raw/<yyyymm>/`, local o S3) y construye las features con
   ese mismo `feature_spec` (no lo recalcula: así el OOT se transforma igual que el train).
3. Predice `probabilidad_fuga` y asigna el decil relativo (1 = menor riesgo, 10 = mayor) y el segmento.
4. Escribe el CSV ordenado por probabilidad (local y, si corresponde, S3) y registra un run de
   scoreo en MLflow (insumo del monitoreo de drift, R9).

Uso:
    python -m bank_attrition.score --config config/config.yaml [--alias champion] [--no-upload]
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from . import data as data_mod
from .config import load_config
from .features import build_features, prepare_input
from .metrics import assign_decile
from .pyfunc_model import OUTPUT_COLUMN
from .tracking import log_score_distribution, setup_mlflow, standard_tags

OUTPUT_COLUMNS = [
    "ID_CORRELATIVO", "CODMES", OUTPUT_COLUMN, "decil_riesgo", "segmento_riesgo",
    "fecha_score", "modelo_nombre", "modelo_version", "modelo_alias",
]


def resolve_model(cfg: dict, alias: str) -> dict:
    """Versión, run de origen, columnas de entrada y feature_spec del modelo con `alias`."""
    name = cfg["mlflow"]["registered_model_name"]
    version = MlflowClient().get_model_version_by_alias(name, alias)
    return describe_model(cfg, str(version.version), alias, uri=f"models:/{name}@{alias}",
                          source_run_id=version.run_id)


def describe_model(cfg: dict, version: str, alias: str, uri: str | None = None,
                   source_run_id: str | None = None) -> dict:
    """Metadatos de una versión concreta (la que fijó el paso de preparación del batch)."""
    name = cfg["mlflow"]["registered_model_name"]
    uri = uri or f"models:/{name}/{version}"
    if source_run_id is None:
        source_run_id = MlflowClient().get_model_version(name, version).run_id
    info = mlflow.models.get_model_info(uri)
    inputs = [(c.name, c.type.name) for c in info.signature.inputs.inputs]
    return {
        "uri": uri,
        "name": name,
        "alias": alias,
        "version": str(version),
        "source_run_id": source_run_id,
        "feature_spec": info.metadata["feature_spec"],
        "numeric": [n for n, t in inputs if t == "double"],
        "categorical": [n for n, t in inputs if t != "double"],
    }


def assign_segment(decile, segments: dict[str, int]) -> np.ndarray:
    """Segmento según el decil mínimo de cada uno (p. ej. Alto: 9, Medio: 6, Bajo: 1)."""
    ordered = sorted(segments.items(), key=lambda kv: kv[1], reverse=True)
    decile = np.asarray(decile)
    out = np.full(decile.shape, ordered[-1][0], dtype=object)
    for label, minimum in reversed(ordered):
        out[decile >= minimum] = label
    return out


def build_model_input(table: pd.DataFrame, model_meta: dict) -> pd.DataFrame:
    """Columnas de entrada del modelo con tipos estables; falla si el OOT no trae alguna."""
    expected = model_meta["numeric"] + model_meta["categorical"]
    missing = [c for c in expected if c not in table.columns]
    if missing:
        raise ValueError(f"El dataset de aplicación no tiene columnas requeridas por el modelo: {missing}")
    return prepare_input(table, model_meta["numeric"], model_meta["categorical"])


def score_table(model, table: pd.DataFrame, model_meta: dict, cfg: dict, score_date: date) -> pd.DataFrame:
    """Salida de negocio: una fila por cliente, ordenada de mayor a menor probabilidad de fuga."""
    schema = cfg["schema"]
    id_col, period = schema["id_col"], schema["period_col"]
    if table.duplicated([id_col, period]).any():
        raise ValueError(f"El dataset de aplicación tiene más de una fila por ({id_col}, {period})")
    proba = model.predict(build_model_input(table, model_meta))[OUTPUT_COLUMN].to_numpy()
    return finalize_scores(table[id_col].to_numpy(), table[period].to_numpy(), proba,
                           model_meta, cfg, score_date)


def finalize_scores(ids, periods, proba, model_meta: dict, cfg: dict, score_date: date) -> pd.DataFrame:
    """Decil relativo, segmento y metadatos de trazabilidad (lo usan el scoreo local y el batch en AWS)."""
    proba = np.asarray(proba, dtype="float64")
    decile = assign_decile(proba)
    out = pd.DataFrame({
        "ID_CORRELATIVO": np.asarray(ids),
        "CODMES": np.asarray(periods),
        OUTPUT_COLUMN: proba,
        "decil_riesgo": decile,
        "segmento_riesgo": assign_segment(decile, cfg["scoring"]["segments"]),
        "fecha_score": score_date.isoformat(),
        "modelo_nombre": model_meta["name"],
        "modelo_version": model_meta["version"],
        "modelo_alias": model_meta["alias"],
    })
    return out.sort_values(OUTPUT_COLUMN, ascending=False, kind="mergesort").reset_index(drop=True)


def output_name(release: str, score_date: date, version: str) -> str:
    return f"scores_{release}_{score_date:%Y%m%d}_v{version}.csv"


def output_key(cfg: dict, release: str, score_date: date, version: str) -> str:
    """Clave S3: <prefix>/<yyyymm>/scores_<yyyymm>_<yyyymmdd>_v<version>.csv"""
    return f"{cfg['scoring']['s3_prefix']}/{release}/{output_name(release, score_date, version)}"


def write_outputs(scores: pd.DataFrame, cfg: dict, release: str, score_date: date,
                  version: str, upload: bool) -> dict[str, str]:
    """Guarda el CSV en local y, si `upload`, lo sube a S3. Devuelve las rutas escritas."""
    local_dir = Path(cfg["scoring"]["local_dir"]) / release
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / output_name(release, score_date, version)
    scores.to_csv(local_path, index=False)
    paths = {"local": str(local_path)}
    if upload:
        bucket = cfg["data"].get("s3_bucket")
        if not bucket:
            raise ValueError("data.s3_bucket vacío: define MLOPS_DATA_BUCKET o usa --no-upload")
        import boto3  # import diferido: solo se necesita al subir a S3

        key = output_key(cfg, release, score_date, version)
        boto3.client("s3").upload_file(str(local_path), bucket, key)
        paths["s3"] = f"s3://{bucket}/{key}"
    return paths


def decile_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Resumen por decil para el área de retención y para el monitoreo."""
    return (
        scores.groupby("decil_riesgo")
        .agg(clientes=(OUTPUT_COLUMN, "size"), prob_min=(OUTPUT_COLUMN, "min"),
             prob_media=(OUTPUT_COLUMN, "mean"), prob_max=(OUTPUT_COLUMN, "max"),
             segmento=("segmento_riesgo", "first"))
        .sort_index(ascending=False)
        .reset_index()
    )


def run_scoring(cfg: dict, alias: str | None = None, upload: bool | None = None,
                score_date: date | None = None) -> dict:
    sc = cfg["scoring"]
    alias = alias or sc["alias"]
    upload = sc["upload_to_s3"] if upload is None else upload
    score_date = score_date or datetime.now(timezone.utc).date()
    release = str(cfg["data"]["release"])
    setup_mlflow(cfg)

    model_meta = resolve_model(cfg, alias)
    model = mlflow.pyfunc.load_model(model_meta["uri"])
    clientes, requerimientos = data_mod.load_tables(cfg, "oot")
    table = build_features(clientes, requerimientos, model_meta["feature_spec"], cfg)
    scores = score_table(model, table, model_meta, cfg, score_date)
    paths = write_outputs(scores, cfg, release, score_date, model_meta["version"], upload)

    data_uri = data_mod.data_uri(cfg)
    tags = standard_tags(cfg, release, data_uri, "scoring")
    tags.update(model_name=model_meta["name"], model_alias=alias, model_version=model_meta["version"],
                model_source_run_id=model_meta["source_run_id"])
    proba = scores[OUTPUT_COLUMN]
    segment_share = scores["segmento_riesgo"].value_counts(normalize=True)
    with mlflow.start_run(run_name=f"score-{release}-{score_date:%Y%m%d}") as run:
        mlflow.set_tags(tags)
        mlflow.log_params({
            "data_release": release, "data_uri": data_uri, "model_uri": model_meta["uri"],
            "model_version": model_meta["version"], "score_date": score_date.isoformat(),
            "output_uri": paths.get("s3", paths["local"]),
        })
        mlflow.log_metrics({
            "score_n": float(len(scores)), "score_mean": float(proba.mean()),
            "score_p50": float(proba.quantile(0.5)), "score_p90": float(proba.quantile(0.9)),
            "score_p99": float(proba.quantile(0.99)),
            **{f"share_segmento_{k.lower()}": float(v) for k, v in segment_share.items()},
        })
        mlflow.log_table(decile_summary(scores), "scoring/resumen_deciles.json")
        mlflow.log_dict(model_meta["feature_spec"], "scoring/feature_spec.json")
        log_score_distribution(proba.to_numpy(), "oot")
    return {"run_id": run.info.run_id, "model_version": model_meta["version"], "rows": len(scores),
            "paths": paths, "summary": decile_summary(scores)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--alias", default=None, help="alias del Registry (por defecto scoring.alias)")
    parser.add_argument("--no-upload", action="store_true", help="no sube el resultado a S3")
    parser.add_argument("--output-json", help="guarda run, versión, rutas y resumen por decil en un JSON (sin datos de clientes)")
    args = parser.parse_args(argv)
    result = run_scoring(load_config(args.config), alias=args.alias,
                         upload=False if args.no_upload else None)
    if args.output_json:
        import json

        summary = {k: result[k] for k in ("run_id", "model_version", "rows", "paths")}
        summary["deciles"] = json.loads(result["summary"].round(6).to_json(orient="records"))
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
                                          encoding="utf-8")
    print(f"run_id={result['run_id']} modelo_version={result['model_version']} filas={result['rows']}")
    for kind, path in result["paths"].items():
        print(f"{kind}: {path}")
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
