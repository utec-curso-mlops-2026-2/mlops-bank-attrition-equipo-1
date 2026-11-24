"""Definición del SageMaker Pipeline del batch (R7): pasos, dependencias, rutas S3 y ausencia de secretos."""
import json

import pytest

from bank_attrition.aws_batch import container_env, model_name_for, pipeline_definition, settings
from bank_attrition.batch import batch_keys

IMAGE = "123.dkr.ecr.us-east-1.amazonaws.com/repo@sha256:abc"


@pytest.fixture()
def s(monkeypatch):
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123:role/sm")
    monkeypatch.setenv("MLOPS_DATA_BUCKET", "bucket-x")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "no-debe-aparecer")
    return settings()


def _resolve(value, params, execution_id="EXEC"):
    """Evalúa Get/Std:Join como lo haría SageMaker, para comparar con las rutas de batch.py."""
    if isinstance(value, dict) and "Get" in value:
        ref = value["Get"]
        return execution_id if ref == "Execution.PipelineExecutionId" else params[ref.split(".", 1)[1]]
    if isinstance(value, dict) and "Std:Join" in value:
        join = value["Std:Join"]
        return join["On"].join(_resolve(v, params, execution_id) for v in join["Values"])
    return value


def test_three_steps_in_order(s):
    d = pipeline_definition(s, IMAGE, "modelo-x")
    steps = {st["Name"]: st for st in d["Steps"]}
    assert [st["Type"] for st in d["Steps"]] == ["Processing", "Transform", "Processing"]
    assert steps["ScorearTransform"]["DependsOn"] == ["PrepararFeatures"]
    assert steps["PublicarScoresDrift"]["DependsOn"] == ["ScorearTransform"]
    assert steps["ScorearTransform"]["Arguments"]["ModelName"] == "modelo-x"
    assert steps["PrepararFeatures"]["Arguments"]["AppSpecification"]["ImageUri"] == IMAGE


def test_transform_paths_match_batch_module(s, base_cfg):
    d = pipeline_definition(s, IMAGE, "m")
    params = {p["Name"]: p["DefaultValue"] for p in d["Parameters"]}
    transform = next(st for st in d["Steps"] if st["Type"] == "Transform")["Arguments"]
    keys = batch_keys(base_cfg, params["Release"], "EXEC")
    src = _resolve(transform["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"], params)
    dst = _resolve(transform["TransformOutput"]["S3OutputPath"], params)
    assert src == f"s3://bucket-x/{keys['input_prefix'].rstrip('/')}"
    assert dst == f"s3://bucket-x/{keys['output_prefix'].rstrip('/')}"
    assert transform["TransformInput"]["SplitType"] == "Line"
    assert transform["TransformInput"]["ContentType"] == "application/jsonlines"


def test_no_secrets_in_definition_or_container_env(s):
    text = json.dumps(pipeline_definition(s, IMAGE, "m"))
    assert "no-debe-aparecer" not in text
    env = container_env(s)
    assert "MLFLOW_TRACKING_PASSWORD" not in env
    assert env["MLFLOW_TRACKING_PASSWORD_SSM"].startswith("/equipo1/")
    assert env["MLOPS_DATA_SOURCE"] == "s3"


def test_model_name_is_valid_for_sagemaker(s):
    name = model_name_for(s, "0123456789abcdef0123")
    assert name == "equipo1-bank-attrition-0123456789ab" and len(name) <= 63
