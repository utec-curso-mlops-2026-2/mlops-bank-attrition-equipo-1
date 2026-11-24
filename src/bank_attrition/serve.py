"""Servidor de inferencia del contenedor SageMaker (R7 Batch Transform y R8 endpoint online).

Contrato SageMaker: escucha en el puerto 8080, `GET /ping` (salud) y `POST /invocations`.
El modelo se carga UNA vez al arrancar, **desde el Registry de MLflow por alias** (nunca desde un archivo).

Tipos de contenido de `/invocations`:
- `application/jsonlines` (Batch Transform): una línea JSON por cliente con `ID_CORRELATIVO`, `CODMES` y
  las features del modelo; responde una línea por cliente con la probabilidad y la versión del modelo.
- `application/json` (API online): `{"ID_CORRELATIVO": ..., "CODMES": ..., "features": {...}}` o
  `{"instances": [ ... ]}`; responde la probabilidad, el modelo y la versión.
Las entradas inválidas reciben 4xx con un mensaje claro (columnas faltantes, tipos no numéricos, JSON mal formado).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .features import prepare_input
from .pyfunc_model import OUTPUT_COLUMN

ID_COL, PERIOD_COL = "ID_CORRELATIVO", "CODMES"
JSONLINES = {"application/jsonlines", "application/jsonl", "application/x-jsonlines"}
MAX_ONLINE_INSTANCES = 100

log = logging.getLogger("bank_attrition.serve")


class InputError(ValueError):
    """Entrada que no cumple el contrato: se responde 422 con el detalle."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _is_number(value) -> bool:
    if value is None:
        return True  # nulo: lo imputa el pipeline del modelo
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isinf(value))
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def validate_records(records: list, model_meta: dict) -> pd.DataFrame:
    """Valida contra la signature del modelo y devuelve la entrada tipada. Reúne TODOS los problemas."""
    if not records:
        raise InputError(["no se recibió ningún cliente"])
    numeric, categorical = model_meta["numeric"], model_meta["categorical"]
    problems: list[str] = []
    for i, rec in enumerate(records):
        where = f"registro {i}" + (f" ({rec.get(ID_COL)})" if isinstance(rec, dict) and rec.get(ID_COL) else "")
        if not isinstance(rec, dict):
            problems.append(f"{where}: debe ser un objeto JSON")
            continue
        if rec.get(ID_COL) in (None, ""):
            problems.append(f"{where}: falta {ID_COL}")
        missing = [c for c in numeric + categorical if c not in rec]
        if missing:
            shown = ", ".join(missing[:10]) + (f" y {len(missing) - 10} más" if len(missing) > 10 else "")
            problems.append(f"{where}: faltan features requeridas: {shown}")
        bad = [c for c in numeric if c in rec and not _is_number(rec[c])]
        if bad:
            problems.append(f"{where}: deben ser numéricas: {', '.join(bad)}")
        if len(problems) >= 20:
            problems.append("... (se detuvo la validación tras 20 problemas)")
            break
    if problems:
        raise InputError(problems)
    frame = pd.DataFrame.from_records(records)
    return prepare_input(frame, numeric, categorical)


def parse_online(body: bytes) -> list[dict]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError([f"JSON mal formado: {exc}"]) from exc
    items = payload.get("instances") if isinstance(payload, dict) and "instances" in payload else [payload]
    if not isinstance(items, list):
        raise InputError(["'instances' debe ser una lista"])
    if len(items) > MAX_ONLINE_INSTANCES:
        raise InputError([f"máximo {MAX_ONLINE_INSTANCES} clientes por solicitud online"])
    records = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("features"), dict):
            raise InputError([f"instancia {i}: se espera {{'{ID_COL}', '{PERIOD_COL}', 'features': {{...}}}}"])
        records.append({**item["features"], ID_COL: item.get(ID_COL), PERIOD_COL: item.get(PERIOD_COL)})
    return records


def parse_jsonlines(body: bytes) -> list[dict]:
    records = []
    for n, line in enumerate(body.decode("utf-8").splitlines(), start=1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise InputError([f"línea {n}: JSON mal formado: {exc}"]) from exc
    return records


def predict_records(model, records: list[dict], model_meta: dict) -> list[dict]:
    proba = model.predict(validate_records(records, model_meta))[OUTPUT_COLUMN].to_numpy()
    return [
        {ID_COL: rec.get(ID_COL), PERIOD_COL: rec.get(PERIOD_COL), OUTPUT_COLUMN: float(p),
         "modelo_version": model_meta["version"]}
        for rec, p in zip(records, proba, strict=True)
    ]


def create_app(model, model_meta: dict):
    """App FastAPI con el modelo ya cargado (separado de `main` para poder probarla sin MLflow)."""
    app = FastAPI(title="bank-attrition", openapi_url=None)

    def error(status: int, code: str, detail) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": code, "detalle": detail})

    @app.get("/ping")
    def ping():
        return Response(status_code=200)

    @app.get("/execution-parameters")
    def execution_parameters():
        # Batch Transform consulta este endpoint para dimensionar los mini-lotes.
        return {"MaxConcurrentTransforms": 1, "BatchStrategy": "MULTI_RECORD", "MaxPayloadInMB": 6}

    @app.post("/invocations")
    async def invocations(request: Request):
        request_id = request.headers.get("x-amzn-requestid") or uuid.uuid4().hex
        started = time.perf_counter()
        content_type = (request.headers.get("content-type") or "application/json").split(";")[0].strip().lower()
        body = await request.body()
        status, rows = 200, 0
        try:
            if content_type in JSONLINES:
                results = predict_records(model, parse_jsonlines(body), model_meta)
                rows = len(results)
                text = "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n"
                return Response(content=text, media_type="application/jsonlines")
            if content_type != "application/json":
                status = 415
                return error(415, "tipo_no_soportado",
                             f"Content-Type {content_type!r}: use application/json o application/jsonlines")
            results = predict_records(model, parse_online(body), model_meta)
            rows = len(results)
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for r in results:
                r.update(modelo_nombre=model_meta["name"], modelo_alias=model_meta["alias"],
                         fecha_prediccion=stamp, request_id=request_id)
            return JSONResponse(results[0] if len(results) == 1 else {"predicciones": results})
        except InputError as exc:
            status = 422
            return error(422, "entrada_invalida", exc.problems)
        except Exception:  # noqa: BLE001 - nunca exponer trazas al consumidor
            status = 500
            log.exception("error de inferencia request_id=%s", request_id)
            return error(500, "error_interno", f"revise los logs con request_id={request_id}")
        finally:
            log.info(json.dumps({
                "request_id": request_id, "content_type": content_type, "status": status, "rows": rows,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "model_version": model_meta["version"], "model_alias": model_meta["alias"],
            }))

    return app


def main() -> None:
    import mlflow.pyfunc
    import uvicorn

    from .config import load_config
    from .score import resolve_model
    from .tracking import load_mlflow_password

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(os.environ.get("BANK_ATTRITION_CONFIG", "config/config.yaml"))
    load_mlflow_password()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])  # solo lee el Registry: no crea runs ni experimentos
    alias = os.environ.get("MODEL_ALIAS") or cfg["scoring"]["alias"]
    model_meta = resolve_model(cfg, alias)
    # El alias se resuelve una vez y se carga esa versión exacta: lo que se reporta es lo que predice.
    model = mlflow.pyfunc.load_model(f"models:/{model_meta['name']}/{model_meta['version']}")
    log.info("modelo cargado %s v%s (alias %s)", model_meta["name"], model_meta["version"], alias)
    uvicorn.run(create_app(model, model_meta), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                log_level="info", access_log=False)


if __name__ == "__main__":
    main()
