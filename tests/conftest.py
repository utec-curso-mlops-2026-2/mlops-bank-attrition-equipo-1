import copy
from pathlib import Path

import pytest

from bank_attrition.config import load_config
from bank_attrition.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_cfg():
    return load_config(ROOT / "config" / "config.yaml")


@pytest.fixture()
def cfg(base_cfg, tmp_path):
    """Config de prueba: datos sintéticos en tmp, MLflow local en tmp, modelos pequeños."""
    cfg = copy.deepcopy(base_cfg)
    raw = tmp_path / "raw"
    folder = raw / str(cfg["data"]["release"])
    folder.mkdir(parents=True)
    train_c, train_r = make_synthetic(700, seed=1, miscoded=True)
    oot_c, oot_r = make_synthetic(200, seed=2, with_target=False)
    files = cfg["data"]["files"]
    train_c.to_csv(folder / files["train_clientes"], index=False)
    train_r.to_csv(folder / files["train_requerimientos"], index=False)
    oot_c.to_csv(folder / files["oot_clientes"], index=False)
    oot_r.to_csv(folder / files["oot_requerimientos"], index=False)
    cfg["data"].update(source="local", local_dir=str(raw))
    cfg["selection"]["cv_folds"] = 3
    cfg["models"] = {}
    cfg["models"] = {
        "logistic_regression": {"C": 0.5, "max_iter": 300},
        "random_forest": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 20},
    }
    cfg["mlflow"].update(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", experiment_name="test-exp"
    )
    cfg["registry"].update(min_test_auc=0.60, min_test_ks=0.15, min_capture_top20=0.30)
    return cfg
