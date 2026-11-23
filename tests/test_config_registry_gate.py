import os

from bank_attrition.config import load_config
from bank_attrition.registry import passes_gate


def test_env_substitution_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    assert cfg["mlflow"]["tracking_uri"] == "http://mlflow.test:5000"
    assert cfg["mlflow"]["experiment_name"] == "bank-attrition"


def test_fallback_tracking_uri_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    assert cfg["mlflow"]["tracking_uri"].startswith("sqlite:")


def test_gate_passes_and_fails():
    reg = {"min_test_auc": 0.7, "min_test_ks": 0.3, "min_capture_top20": 0.4}
    ok, failures = passes_gate({"test_auc": 0.8, "test_ks": 0.4, "test_capture_top20": 0.5}, reg)
    assert ok and not failures
    ok, failures = passes_gate({"test_auc": 0.65, "test_ks": 0.4, "test_capture_top20": 0.5}, reg)
    assert not ok and len(failures) == 1
    ok, failures = passes_gate({"test_auc": 0.9}, reg)  # métricas ausentes también fallan
    assert not ok and len(failures) == 2
