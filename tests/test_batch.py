"""Pipeline batch de SageMaker (R7) de punta a punta en local: preparar → Transform (simulado) → publicar.

S3 se reemplaza por un diccionario en memoria y Batch Transform por el mismo servidor (`serve`) que corre en
el contenedor, partiendo la entrada en mini-lotes como lo hace SageMaker con SplitType=Line.
"""
import io
import json
from datetime import date

import mlflow.pyfunc
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bank_attrition.batch import (
    batch_keys,
    check_transform_output,
    prepare,
    publish,
    read_jsonlines,
)
from bank_attrition.score import OUTPUT_COLUMNS, resolve_model
from bank_attrition.serve import create_app
from bank_attrition.train import run_training

SCORE_DATE = date(2026, 11, 24)
EXEC_ID = "test-exec-1"


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803 - firma de boto3
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


@pytest.fixture()
def batch_cfg(cfg):
    cfg["data"]["s3_bucket"] = "bucket-test"
    assert run_training(cfg, register=True)["registry"]["alias"] == "champion"
    return cfg


def fake_transform(cfg, s3, chunk=37):
    """Lo que hace Batch Transform: mini-lotes de líneas → POST /invocations → salida unida por líneas."""
    meta = resolve_model(cfg, "champion")
    client = TestClient(create_app(mlflow.pyfunc.load_model(meta["uri"]), meta))
    keys = batch_keys(cfg, cfg["data"]["release"], EXEC_ID)
    lines = s3.objects[("bucket-test", keys["input"])].decode().splitlines()
    out = []
    for start in range(0, len(lines), chunk):
        body = "\n".join(lines[start:start + chunk]).encode()
        resp = client.post("/invocations", content=body, headers={"Content-Type": "application/jsonlines"})
        assert resp.status_code == 200, resp.text
        out.append(resp.text.strip())
    s3.put_object(Bucket="bucket-test", Key=keys["output"], Body=("\n".join(out) + "\n").encode())


def test_batch_pipeline_end_to_end(batch_cfg):
    s3 = FakeS3()
    manifest = prepare(batch_cfg, EXEC_ID, alias="champion", s3=s3)
    assert manifest["rows"] == 200 and manifest["model_version"] == "1"
    keys = batch_keys(batch_cfg, batch_cfg["data"]["release"], EXEC_ID)
    first = json.loads(s3.objects[("bucket-test", keys["input"])].decode().splitlines()[0])
    assert {"ID_CORRELATIVO", "CODMES"} <= set(first)
    assert set(manifest["numeric_features"]) <= set(first)

    fake_transform(batch_cfg, s3)
    result = publish(batch_cfg, EXEC_ID, score_date=SCORE_DATE, s3=s3)

    scores_key = "scores/202408/scores_202408_20261124_v1.csv"
    assert result["scores_uri"] == f"s3://bucket-test/{scores_key}"
    scores = pd.read_csv(io.BytesIO(s3.objects[("bucket-test", scores_key)]))
    assert list(scores.columns) == OUTPUT_COLUMNS and len(scores) == 200
    assert scores["probabilidad_fuga"].is_monotonic_decreasing
    assert scores["decil_riesgo"].value_counts().tolist() == [20] * 10  # deciles sobre el mes completo

    drift = json.loads(s3.objects[("bucket-test", "monitoring/202408/drift_202408_20261124_v1.json")])
    assert drift["resumen"]["estado_score"] in {"estable", "alerta", "critico"}
    assert drift["detalle"][0]["variable"] == "probabilidad_fuga"

    run = mlflow.get_run(result["run_id"])
    assert run.data.tags["stage"] == "batch_inference"
    assert run.data.tags["batch_execution_id"] == EXEC_ID
    assert "psi_score" in run.data.metrics


def test_publish_rejects_incomplete_or_mixed_output():
    inputs = pd.DataFrame({"ID_CORRELATIVO": ["a", "b", "c"]})
    manifest = {"rows": 3, "model_version": "2"}
    good = pd.DataFrame({"ID_CORRELATIVO": ["a", "b", "c"], "probabilidad_fuga": [0.1, 0.5, 0.9],
                         "modelo_version": ["2", "2", "2"]})
    check_transform_output(good, inputs, manifest)
    with pytest.raises(ValueError, match="filas de salida 2"):
        check_transform_output(good.iloc[:2], inputs, manifest)
    with pytest.raises(ValueError, match="cambió el alias"):
        check_transform_output(good.assign(modelo_version=["2", "3", "2"]), inputs, manifest)
    with pytest.raises(ValueError, match=r"fuera de \[0, 1\]"):
        check_transform_output(good.assign(probabilidad_fuga=[0.1, 1.5, 0.2]), inputs, manifest)


def test_jsonlines_roundtrip_keeps_nulls():
    text = '{"ID_CORRELATIVO":"x","EDAD":null}\n\n{"ID_CORRELATIVO":"y","EDAD":3.5}\n'
    frame = read_jsonlines(text)
    assert frame["ID_CORRELATIVO"].tolist() == ["x", "y"] and pd.isna(frame.loc[0, "EDAD"])
