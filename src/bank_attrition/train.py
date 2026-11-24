"""Pipeline de entrenamiento (R2) con trazabilidad en MLflow (R4) y empaquetado (R5).

Uso:
    python -m bank_attrition.train --config config/config.yaml [--no-register]
"""
from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import data as data_mod
from .config import load_config
from .features import build_features, fit_feature_spec, prepare_input, split_columns
from .metrics import evaluate
from .models import build_pipeline
from .pyfunc_model import OUTPUT_COLUMN, ChurnModel
from .tracking import (
    log_confusion_top,
    log_feature_importance,
    log_roc,
    log_score_distribution,
    setup_mlflow,
    standard_tags,
)

warnings.filterwarnings("ignore", message=".*y_pred.*renamed.*", category=FutureWarning)


def _importances(pipeline, x_test: pd.DataFrame, y_test, seed: int) -> dict[str, float]:
    preprocess, estimator = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    names = list(preprocess.get_feature_names_out())
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        result = permutation_importance(
            estimator, preprocess.transform(x_test), y_test, scoring="roc_auc",
            n_repeats=3, random_state=seed,
        )
        values = result.importances_mean
    return dict(zip(names, map(float, values), strict=True))


def _train_one(name, params, cfg, x_train, y_train, x_test, y_test, numeric, categorical):
    seed, sel = cfg["project"]["random_seed"], cfg["selection"]
    pipeline = build_pipeline(name, params, numeric, categorical, seed)
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tag("model_name", name)
        mlflow.log_params({f"model.{k}": v for k, v in params.items()})
        folds = StratifiedKFold(sel["cv_folds"], shuffle=True, random_state=seed)
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=folds, scoring="roc_auc")
        mlflow.sklearn.autolog(log_models=False, log_datasets=False, silent=True)
        pipeline.fit(x_train, y_train)  # autologging registra parámetros y métricas de ajuste
        mlflow.sklearn.autolog(disable=True)
        fractions = sel["top_fractions"]
        score_train = pipeline.predict_proba(x_train)[:, 1]
        score_test = pipeline.predict_proba(x_test)[:, 1]
        metrics = {"cv_auc_mean": float(cv_scores.mean()), "cv_auc_std": float(cv_scores.std())}
        metrics.update({f"train_{k}": v for k, v in evaluate(y_train, score_train, fractions).items()})
        metrics.update({f"test_{k}": v for k, v in evaluate(y_test, score_test, fractions).items()})
        mlflow.log_metrics(metrics)
        log_roc(y_test, score_test, name)
        log_confusion_top(y_test, score_test, fractions[-1])
        log_feature_importance(_importances(pipeline, x_test, y_test, seed))
        return {"name": name, "pipeline": pipeline, "metrics": metrics}


def _log_pyfunc_model(pipeline, x_example, spec, cfg, tags):
    """Registra el modelo con entorno, signature e input example, y el preprocesamiento dentro."""
    package_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        sk_dir = str(Path(tmp) / "sklearn_model")
        mlflow.sklearn.save_model(
            pipeline, sk_dir, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        proba = pipeline.predict_proba(x_example)[:, 1]
        signature = infer_signature(x_example, pd.DataFrame({OUTPUT_COLUMN: proba}))
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=ChurnModel(),
            artifacts={"sklearn_model": sk_dir},
            signature=signature,
            input_example=x_example.head(5),
            code_paths=[str(package_dir)],
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                *_pinned(["scikit-learn", "pandas", "numpy", "scipy"]),
            ],
            metadata={"feature_spec": spec, **tags},
        )


def _pinned(packages: list[str]) -> list[str]:
    from importlib.metadata import version

    return [f"{p}=={version(p)}" for p in packages]


def run_training(cfg: dict, register: bool = True) -> dict:
    seed, sel = cfg["project"]["random_seed"], cfg["selection"]
    schema = cfg["schema"]
    setup_mlflow(cfg)

    clientes, requerimientos = data_mod.load_tables(cfg, "train")
    oot_clientes, oot_requerimientos = data_mod.load_tables(cfg, "oot")
    manifest = data_mod.data_manifest(cfg)
    spec = fit_feature_spec(clientes, requerimientos, cfg)
    table = build_features(clientes, requerimientos, spec, cfg)
    oot_table = build_features(oot_clientes, oot_requerimientos, spec, cfg)
    numeric, categorical = split_columns(table, cfg)

    y = table[schema["target_col"]].astype(int)
    x = prepare_input(table, numeric, categorical)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg["split"]["test_size"], random_state=seed,
        stratify=y if cfg["split"]["stratify"] else None,
    )

    tags = standard_tags(cfg, cfg["data"]["release"], manifest["uri"], "training")
    with mlflow.start_run(run_name=f"train-{cfg['data']['release']}") as parent:
        mlflow.set_tags(tags)
        mlflow.log_params({
            "seed": seed, "data_release": manifest["release"], "data_uri": manifest["uri"],
            "window_months": spec["window_months"], "n_train": len(x_train), "n_test": len(x_test),
            "n_features": len(numeric) + len(categorical),
            "positive_rate_train": round(float(y_train.mean()), 5),
        })
        mlflow.log_dict(manifest, "data_manifest.json")
        mlflow.log_dict(cfg, "config_used.json")
        mlflow.log_dict(spec, "feature_spec.json")
        try:  # linaje de datos en la UI de MLflow (no debe romper el entrenamiento)
            dataset = mlflow.data.from_pandas(
                table, source=manifest["uri"], name=f"clientes_{manifest['release']}",
                targets=schema["target_col"],
            )
            mlflow.log_input(dataset, context="training")
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("dataset_logging_warning", str(exc)[:200])

        results = [
            _train_one(name, params, cfg, x_train, y_train, x_test, y_test, numeric, categorical)
            for name, params in cfg["models"].items()
        ]
        best = max(results, key=lambda r: r["metrics"][sel["criterion"]])
        mlflow.log_param("selection_criterion", sel["criterion"])
        mlflow.set_tag("best_model", best["name"])
        mlflow.log_metrics(best["metrics"])
        mlflow.log_dict(
            {r["name"]: r["metrics"] for r in results}, "model_comparison.json"
        )

        # El OOT no tiene target: solo se registra la distribución del score (insumo de drift, R9).
        oot_x = prepare_input(oot_table, numeric, categorical)
        oot_score = best["pipeline"].predict_proba(oot_x)[:, 1]
        mlflow.log_metrics({
            "oot_n": float(len(oot_score)), "oot_score_mean": float(oot_score.mean()),
            "oot_score_p50": float(np.quantile(oot_score, 0.5)),
            "oot_score_p90": float(np.quantile(oot_score, 0.9)),
            "oot_score_p99": float(np.quantile(oot_score, 0.99)),
        })
        log_score_distribution(oot_score, "oot")

        model_info = _log_pyfunc_model(best["pipeline"], x_train.head(200), spec, cfg, tags)
        output = {
            "run_id": parent.info.run_id, "model_uri": model_info.model_uri,
            "best_model": best["name"], "metrics": best["metrics"],
            "all_metrics": {r["name"]: r["metrics"] for r in results},
        }

    if register:
        from .registry import register_candidate

        output["registry"] = register_candidate(cfg, output["model_uri"], output["run_id"],
                                                output["metrics"])
    return output


def write_result(result: dict, path: str | Path) -> dict:
    """Resumen serializable del entrenamiento (lo consume GitHub Actions como outputs y artefacto)."""
    import json

    registry = result.get("registry") or {}
    summary = {
        "run_id": result["run_id"],
        "model_uri": result["model_uri"],
        "best_model": result["best_model"],
        "metrics": {k: round(float(v), 6) for k, v in result["metrics"].items()},
        "all_metrics": {
            name: {k: round(float(v), 6) for k, v in metrics.items()}
            for name, metrics in result["all_metrics"].items()
        },
        "registered": bool(registry),
        "model_name": registry.get("model"),
        "model_version": registry.get("version"),
        "validation": registry.get("validation", "not_registered"),
        "validation_failures": registry.get("failures", []),
        "alias": registry.get("alias"),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-register", action="store_true", help="solo entrena y registra el run")
    parser.add_argument("--output-json", help="guarda el resultado (run, métricas, registro) en un JSON (para CI)")
    args = parser.parse_args(argv)
    result = run_training(load_config(args.config), register=not args.no_register)
    if args.output_json:
        write_result(result, args.output_json)
    print(f"run_id={result['run_id']} mejor_modelo={result['best_model']}")
    print({k: round(v, 4) for k, v in result["metrics"].items() if k.startswith("test_")})
    if "registry" in result:
        print(result["registry"])


if __name__ == "__main__":
    main()
