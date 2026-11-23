"""Corrida completa sobre datos sintéticos: entrena, registra, valida y carga por alias."""
import mlflow
import numpy as np

from bank_attrition.registry import describe_contract, load_model, promote_challenger
from bank_attrition.synthetic import make_synthetic
from bank_attrition.train import run_training


def test_train_register_and_load_by_alias(cfg):
    result = run_training(cfg, register=True)
    assert result["registry"]["validation"] == "passed"
    assert result["registry"]["alias"] == "champion"  # primer modelo: no había champion

    client = mlflow.MlflowClient()
    run = client.get_run(result["run_id"])
    for tag in ("git_commit", "data_version", "data_uri", "author"):
        assert tag in run.data.tags
    for metric in ("test_auc", "test_ks", "test_capture_top20", "oot_score_mean"):
        assert metric in run.data.metrics
    children = client.search_runs(
        [run.info.experiment_id], f"tags.mlflow.parentRunId = '{result['run_id']}'"
    )
    assert len(children) == len(cfg["models"])

    model = load_model(cfg)  # por alias, nunca desde archivo
    contract = describe_contract(cfg)
    assert contract["outputs"] == [("probabilidad_fuga", "double")]
    assert contract["metadata"]["data_version"] == str(cfg["data"]["release"])

    from bank_attrition.features import build_features, prepare_input, split_columns

    spec = contract["metadata"]["feature_spec"]
    clientes, req = make_synthetic(50, seed=9, with_target=False)
    table = build_features(clientes, req, spec, cfg)
    numeric = [n for n, t in contract["inputs"] if t == "double"]
    categorical = [n for n, t in contract["inputs"] if t == "string"]
    table["ATTRITION"] = 0
    assert set(numeric + categorical) <= set(split_columns(table, cfg)[0] + split_columns(table, cfg)[1])
    scores = model.predict(prepare_input(table, numeric, categorical))["probabilidad_fuga"]
    assert len(scores) == 50 and np.all((scores >= 0) & (scores <= 1))


def test_failed_gate_leaves_version_without_alias(cfg):
    cfg["registry"]["min_test_auc"] = 0.995
    result = run_training(cfg, register=True)
    assert result["registry"]["validation"] == "failed"
    assert result["registry"]["alias"] is None
    client = mlflow.MlflowClient()
    version = client.get_model_version(cfg["mlflow"]["registered_model_name"], "1")
    assert version.tags["validation"] == "failed"
    assert version.aliases == []


def test_second_model_becomes_challenger_when_not_better(cfg):
    first = run_training(cfg, register=True)
    assert first["registry"]["alias"] == "champion"
    cfg["models"] = {"logistic_regression": {"C": 0.0001, "max_iter": 200}}  # peor modelo
    second = run_training(cfg, register=True)
    assert second["registry"]["alias"] in {"challenger", "champion"}
    if second["registry"]["alias"] == "challenger":
        assert promote_challenger(cfg)["promoted"] is False


def test_missing_values_signal_is_preserved(cfg):
    """El pipeline conserva el indicador de nulos (ANTIGUEDAD nula = riesgo alto en los datos reales)."""
    from bank_attrition.models import build_pipeline

    pipeline = build_pipeline("logistic_regression", {"max_iter": 200}, ["EDAD"], ["SEGMENTO"], 0)
    frame = __import__("pandas").DataFrame({"EDAD": [30.0, None, 50.0, None], "SEGMENTO": ["A", None, "B", "A"]})
    pipeline.fit(frame, [0, 1, 0, 1])
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert any("missingindicator" in n for n in names)
    assert any("NULO" in n for n in names)
