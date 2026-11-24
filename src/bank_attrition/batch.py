"""Inferencia batch en AWS con SageMaker Batch Transform (R7) + data drift (R9).

El SageMaker Pipeline `equipo1-bank-attrition-batch` ejecuta tres pasos con la misma imagen:

1. `prepare` (Processing): resuelve el alias del Registry → versión, construye las features del dataset de
   aplicación con el `feature_spec` del modelo y deja en S3 un JSON Lines (una línea por cliente) más un
   manifiesto del lote.
2. Batch Transform: el contenedor (`bank_attrition.serve`) carga el modelo por alias y predice cada línea.
3. `publish` (Processing): valida la salida contra el manifiesto, asigna decil y segmento sobre TODO el mes,
   publica el CSV de scores en S3, calcula el PSI vs. entrenamiento y registra el lote en MLflow.

Estructura en S3 (`batch.s3_prefix`):
    batch/<yyyymm>/<ejecucion>/input/registros.jsonl
    batch/<yyyymm>/<ejecucion>/output/registros.jsonl.out     (lo escribe Batch Transform)
    batch/<yyyymm>/<ejecucion>/manifest.json
    scores/<yyyymm>/scores_<yyyymm>_<yyyymmdd>_v<version>.csv
    monitoring/<yyyymm>/drift_<yyyymm>_<yyyymmdd>_v<version>.{csv,json}

Uso (dentro del contenedor o en local con credenciales):
    python -m bank_attrition.batch prepare --release 202408 --alias champion --execution-id <id>
    python -m bank_attrition.batch publish --release 202408 --execution-id <id>
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timezone

import mlflow
import mlflow.pyfunc
import pandas as pd

from . import data as data_mod
from .config import load_config
from .features import build_features, prepare_input
from .monitoring import drift_report, drift_summary
from .pyfunc_model import OUTPUT_COLUMN
from .score import (
    build_model_input,
    decile_summary,
    describe_model,
    finalize_scores,
    output_name,
    resolve_model,
)
from .tracking import log_score_distribution, setup_mlflow, standard_tags

ID_COL, PERIOD_COL = "ID_CORRELATIVO", "CODMES"


# ---------------------------------------------------------------- rutas y E/S en S3
def batch_prefix(cfg: dict, release: str, execution_id: str) -> str:
    return f"{cfg['batch']['s3_prefix']}/{release}/{execution_id}"


def batch_keys(cfg: dict, release: str, execution_id: str) -> dict[str, str]:
    prefix = batch_prefix(cfg, release, execution_id)
    name = cfg["batch"]["input_file"]
    return {
        "input_prefix": f"{prefix}/input/",
        "input": f"{prefix}/input/{name}",
        "output_prefix": f"{prefix}/output/",
        "output": f"{prefix}/output/{name}.out",  # Batch Transform agrega ".out" al nombre de entrada
        "manifest": f"{prefix}/manifest.json",
    }


def drift_key(cfg: dict, release: str, score_date: date, version: str, ext: str) -> str:
    return f"{cfg['monitoring']['s3_prefix']}/{release}/drift_{release}_{score_date:%Y%m%d}_v{version}.{ext}"


def _bucket(cfg: dict) -> str:
    bucket = cfg["data"].get("s3_bucket")
    if not bucket:
        raise ValueError("data.s3_bucket vacío: define MLOPS_DATA_BUCKET")
    return bucket


def _s3():
    import boto3  # import diferido

    return boto3.client("s3")


def _put(s3, bucket: str, key: str, body: str, content_type: str) -> str:
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"), ContentType=content_type)
    return f"s3://{bucket}/{key}"


def _get(s3, bucket: str, key: str) -> str:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def to_jsonlines(frame: pd.DataFrame) -> str:
    """Una línea JSON por fila; NaN/None → null (el imputador del modelo los trata)."""
    return frame.to_json(orient="records", lines=True, force_ascii=False, double_precision=15)


def read_jsonlines(text: str) -> pd.DataFrame:
    return pd.DataFrame.from_records([json.loads(line) for line in text.splitlines() if line.strip()])


def _with_release(cfg: dict, release: str | None) -> tuple[dict, str]:
    release = str(release or cfg["data"]["release"])
    cfg["data"]["release"] = release
    return cfg, release


# ---------------------------------------------------------------- paso 1: preparar
def prepare(cfg: dict, execution_id: str, release: str | None = None, alias: str | None = None,
            s3=None) -> dict:
    """Snapshot de features del mes (una línea por cliente) + manifiesto, en S3."""
    cfg, release = _with_release(cfg, release)
    alias = alias or cfg["scoring"]["alias"]
    s3 = s3 or _s3()
    bucket = _bucket(cfg)
    keys = batch_keys(cfg, release, execution_id)
    setup_mlflow(cfg)

    model_meta = resolve_model(cfg, alias)
    clientes, requerimientos = data_mod.load_tables(cfg, "oot")
    table = build_features(clientes, requerimientos, model_meta["feature_spec"], cfg)
    if table.duplicated([ID_COL, PERIOD_COL]).any():
        raise ValueError(f"El dataset de aplicación tiene más de una fila por ({ID_COL}, {PERIOD_COL})")
    records = build_model_input(table, model_meta)
    records.insert(0, PERIOD_COL, table[PERIOD_COL].to_numpy())
    records.insert(0, ID_COL, table[ID_COL].to_numpy())

    input_uri = _put(s3, bucket, keys["input"], to_jsonlines(records), "application/jsonlines")
    manifest = {
        "execution_id": execution_id,
        "release": release,
        "data_uri": data_mod.data_uri(cfg),
        "model_name": model_meta["name"],
        "model_alias": alias,
        "model_version": model_meta["version"],
        "model_source_run_id": model_meta["source_run_id"],
        "rows": int(len(records)),
        "numeric_features": model_meta["numeric"],
        "categorical_features": model_meta["categorical"],
        "input_uri": input_uri,
        "output_uri": f"s3://{bucket}/{keys['output']}",
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _put(s3, bucket, keys["manifest"], json.dumps(manifest, indent=2, ensure_ascii=False), "application/json")
    return manifest


# ---------------------------------------------------------------- paso 3: publicar
def check_transform_output(predictions: pd.DataFrame, inputs: pd.DataFrame, manifest: dict) -> None:
    """La salida del Transform debe cubrir exactamente a los clientes del lote, con la versión fijada."""
    problems = []
    if len(predictions) != manifest["rows"]:
        problems.append(f"filas de salida {len(predictions)} != filas del lote {manifest['rows']}")
    if predictions[ID_COL].duplicated().any():
        problems.append("hay clientes repetidos en la salida")
    if set(predictions[ID_COL]) != set(inputs[ID_COL]):
        problems.append("los clientes de la salida no coinciden con los de la entrada")
    versions = set(predictions["modelo_version"].astype(str))
    if versions != {str(manifest["model_version"])}:
        problems.append(f"versiones del modelo en la salida {sorted(versions)} != fijada "
                        f"v{manifest['model_version']} (¿cambió el alias durante el lote?)")
    proba = pd.to_numeric(predictions[OUTPUT_COLUMN], errors="coerce")
    if proba.isna().any() or not proba.between(0, 1).all():
        problems.append("hay probabilidades nulas o fuera de [0, 1]")
    if problems:
        raise ValueError("Salida de Batch Transform inválida: " + "; ".join(problems))


def reference_population(cfg: dict, model, model_meta: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Población de entrenamiento con las mismas features y su score: referencia del PSI."""
    clientes, requerimientos = data_mod.load_tables(cfg, "train")
    table = build_features(clientes, requerimientos, model_meta["feature_spec"], cfg)
    model_input = build_model_input(table, model_meta)
    return model_input, model.predict(model_input)[OUTPUT_COLUMN]


def _metric_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-. /]", "_", text)


def publish(cfg: dict, execution_id: str, release: str | None = None, score_date: date | None = None,
            s3=None) -> dict:
    cfg, release = _with_release(cfg, release)
    score_date = score_date or datetime.now(timezone.utc).date()
    s3 = s3 or _s3()
    bucket = _bucket(cfg)
    keys = batch_keys(cfg, release, execution_id)
    mon = cfg["monitoring"]
    setup_mlflow(cfg)

    manifest = json.loads(_get(s3, bucket, keys["manifest"]))
    inputs = read_jsonlines(_get(s3, bucket, keys["input"]))
    predictions = read_jsonlines(_get(s3, bucket, keys["output"]))
    check_transform_output(predictions, inputs, manifest)

    # Decil y segmento se asignan sobre el mes completo (el Transform solo ve mini-lotes).
    model_meta = describe_model(cfg, manifest["model_version"], manifest["model_alias"])
    scores = finalize_scores(predictions[ID_COL], predictions[PERIOD_COL], predictions[OUTPUT_COLUMN],
                             model_meta, cfg, score_date)  # ya ordenado de mayor a menor probabilidad
    version = model_meta["version"]
    scores_uri = _put(s3, bucket, f"{cfg['scoring']['s3_prefix']}/{release}/{output_name(release, score_date, version)}",
                      scores.to_csv(index=False), "text/csv")

    # Data drift (R9): features y score del mes vs. población de entrenamiento.
    model = mlflow.pyfunc.load_model(f"models:/{model_meta['name']}/{version}")
    reference, reference_score = reference_population(cfg, model, model_meta)
    current = prepare_input(inputs, model_meta["numeric"], model_meta["categorical"])
    current_score = inputs[[ID_COL]].merge(predictions[[ID_COL, OUTPUT_COLUMN]], on=ID_COL)[OUTPUT_COLUMN]
    report = drift_report(reference, current, model_meta["numeric"], model_meta["categorical"],
                          reference_score, current_score, bins=int(mon["psi_bins"]),
                          alert=float(mon["psi_alert"]), critical=float(mon["psi_critical"]))
    summary = drift_summary(report)
    drift_csv_uri = _put(s3, bucket, drift_key(cfg, release, score_date, version, "csv"),
                         report.to_csv(index=False), "text/csv")
    drift_doc = {"release": release, "fecha_score": score_date.isoformat(), "execution_id": execution_id,
                 "modelo_version": version, "umbrales": {"alerta": mon["psi_alert"], "critico": mon["psi_critical"]},
                 "resumen": summary, "detalle": json.loads(report.round(6).to_json(orient="records"))}
    drift_json_uri = _put(s3, bucket, drift_key(cfg, release, score_date, version, "json"),
                          json.dumps(drift_doc, indent=2, ensure_ascii=False), "application/json")

    tags = standard_tags(cfg, release, manifest["data_uri"], "batch_inference")
    tags.update(model_name=model_meta["name"], model_alias=model_meta["alias"], model_version=version,
                model_source_run_id=model_meta["source_run_id"], batch_execution_id=execution_id,
                drift_score=summary["estado_score"])
    proba = scores[OUTPUT_COLUMN]
    shares = scores["segmento_riesgo"].value_counts(normalize=True)
    with mlflow.start_run(run_name=f"batch-{release}-{score_date:%Y%m%d}") as run:
        mlflow.set_tags(tags)
        mlflow.log_params({"data_release": release, "data_uri": manifest["data_uri"],
                           "model_uri": f"models:/{model_meta['name']}/{version}", "model_version": version,
                           "score_date": score_date.isoformat(), "batch_execution_id": execution_id,
                           "output_uri": scores_uri, "drift_report_uri": drift_json_uri})
        mlflow.log_metrics({
            "score_n": float(len(scores)), "score_mean": float(proba.mean()),
            "score_p50": float(proba.quantile(0.5)), "score_p90": float(proba.quantile(0.9)),
            **{f"share_segmento_{k.lower()}": float(v) for k, v in shares.items()},
            "psi_score": summary["psi_score"], "psi_max_variable": summary["psi_max_variable"],
            "drift_n_alerta": float(summary["n_alerta"]), "drift_n_critico": float(summary["n_critico"]),
            **{_metric_name(f"psi_{r.variable}"): float(r.psi)
               for r in report[report["nivel"] == "variable"].itertuples()},
        })
        mlflow.log_table(decile_summary(scores), "batch/resumen_deciles.json")
        mlflow.log_table(report, "monitoring/drift_report.json")
        mlflow.log_dict(manifest, "batch/manifest.json")
        log_score_distribution(proba.to_numpy(), "batch")

    result = {"run_id": run.info.run_id, "execution_id": execution_id, "release": release,
              "model_version": version, "rows": int(len(scores)), "scores_uri": scores_uri,
              "drift_csv_uri": drift_csv_uri, "drift_json_uri": drift_json_uri, "drift": summary}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("prepare", "publish"):
        p = sub.add_parser(step)
        p.add_argument("--config", default="config/config.yaml")
        p.add_argument("--release", default=None, help="yyyymm de los datos (por defecto data.release)")
        p.add_argument("--execution-id", required=True, help="id de la ejecución del pipeline (prefijo S3)")
        if step == "prepare":
            p.add_argument("--alias", default=None, help="alias del Registry (por defecto scoring.alias)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.step == "prepare":
        print(json.dumps(prepare(cfg, args.execution_id, args.release, args.alias), indent=2, ensure_ascii=False))
    else:
        publish(cfg, args.execution_id, args.release)


if __name__ == "__main__":
    main()
