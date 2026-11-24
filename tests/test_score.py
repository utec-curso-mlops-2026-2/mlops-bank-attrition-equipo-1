"""Scoreo batch del OOT (R2 paso 6 / núcleo de R7): salida, deciles, segmentos y uso del feature_spec del modelo."""
import copy
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from bank_attrition.config import load_config
from bank_attrition.score import OUTPUT_COLUMNS, assign_segment, output_key, run_scoring
from bank_attrition.train import run_training

SCORE_DATE = date(2026, 11, 24)
SEGMENTS = {"Alto": 9, "Medio": 6, "Bajo": 1}


@pytest.fixture()
def trained_cfg(cfg, tmp_path):
    """Config de prueba con un champion ya registrado y salida local en tmp."""
    cfg["scoring"]["local_dir"] = str(tmp_path / "scores")
    result = run_training(cfg, register=True)
    assert result["registry"]["alias"] == "champion"
    return cfg


def test_assign_segment_boundaries():
    deciles = np.arange(1, 11)
    labels = assign_segment(deciles, SEGMENTS)
    assert list(labels) == ["Bajo"] * 5 + ["Medio"] * 3 + ["Alto"] * 2


def test_assign_segment_order_in_config_does_not_matter():
    shuffled = {"Medio": 6, "Bajo": 1, "Alto": 9}
    assert list(assign_segment([1, 6, 9], shuffled)) == ["Bajo", "Medio", "Alto"]


def test_output_key_layout(base_cfg):
    key = output_key(base_cfg, "202408", SCORE_DATE, "3")
    assert key == "scores/202408/scores_202408_20261124_v3.csv"


def test_data_source_defaults_to_local(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("MLOPS_DATA_SOURCE", raising=False)
    assert load_config(root / "config" / "config.yaml")["data"]["source"] == "local"
    monkeypatch.setenv("MLOPS_DATA_SOURCE", "S3")
    assert load_config(root / "config" / "config.yaml")["data"]["source"] == "s3"
    monkeypatch.setenv("MLOPS_DATA_SOURCE", "ftp")
    with pytest.raises(ValueError, match="data.source"):
        load_config(root / "config" / "config.yaml")


def test_scoring_output_contract(trained_cfg):
    result = run_scoring(trained_cfg, upload=False, score_date=SCORE_DATE)
    scores = pd.read_csv(result["paths"]["local"])

    assert list(scores.columns) == OUTPUT_COLUMNS
    assert len(scores) == 200 and scores["ID_CORRELATIVO"].is_unique  # una fila por cliente del OOT
    proba = scores["probabilidad_fuga"]
    assert proba.between(0, 1).all()
    assert proba.is_monotonic_decreasing  # lista priorizada para retención
    assert set(scores["decil_riesgo"]) == set(range(1, 11))
    assert (scores["decil_riesgo"].value_counts() == 20).all()  # deciles relativos: 10 % cada uno
    expected_segment = assign_segment(scores["decil_riesgo"], SEGMENTS)
    assert (scores["segmento_riesgo"] == expected_segment).all()
    assert (scores["fecha_score"] == "2026-11-24").all()
    assert (scores["modelo_version"].astype(str) == "1").all()
    assert (scores["modelo_alias"] == "champion").all()
    assert Path(result["paths"]["local"]).name == "scores_202408_20261124_v1.csv"
    assert "s3" not in result["paths"]


def test_scoring_run_is_traced_in_mlflow(trained_cfg):
    result = run_scoring(trained_cfg, upload=False, score_date=SCORE_DATE)
    run = mlflow.MlflowClient().get_run(result["run_id"])
    assert run.data.tags["stage"] == "scoring"
    assert run.data.tags["model_version"] == "1"
    for tag in ("git_commit", "data_version", "data_uri", "model_source_run_id"):
        assert tag in run.data.tags
    for metric in ("score_n", "score_mean", "score_p90", "share_segmento_alto"):
        assert metric in run.data.metrics
    assert run.data.metrics["score_n"] == 200


def test_scoring_uses_feature_spec_saved_with_model(trained_cfg):
    """Cambiar la config de features después de entrenar no debe alterar el scoreo."""
    baseline = pd.read_csv(run_scoring(trained_cfg, upload=False, score_date=SCORE_DATE)["paths"]["local"])
    changed = copy.deepcopy(trained_cfg)
    changed["features"].update(window_months=1, min_category_count=10_000, max_categories_per_col=0)
    again = pd.read_csv(run_scoring(changed, upload=False, score_date=SCORE_DATE)["paths"]["local"])
    pd.testing.assert_series_equal(baseline["probabilidad_fuga"], again["probabilidad_fuga"])


def test_scoring_fails_clearly_when_columns_are_missing(trained_cfg):
    folder = Path(trained_cfg["data"]["local_dir"]) / str(trained_cfg["data"]["release"])
    path = folder / trained_cfg["data"]["files"]["oot_clientes"]
    pd.read_csv(path).drop(columns=["EDAD"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="EDAD"):
        run_scoring(trained_cfg, upload=False, score_date=SCORE_DATE)
