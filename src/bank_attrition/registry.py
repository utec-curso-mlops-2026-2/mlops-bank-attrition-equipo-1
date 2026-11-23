"""Model Registry (R5): versiones, alias champion/challenger y validación previa a la promoción.

Los pipelines de inferencia deben cargar SIEMPRE por alias (`models:/<nombre>@champion`),
nunca desde un archivo local.

Uso:
    python -m bank_attrition.registry promote            # challenger -> champion si cumple y mejora
    python -m bank_attrition.registry contract           # muestra signature e input example del champion
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from .config import load_config
from .tracking import setup_mlflow


def passes_gate(metrics: dict[str, float], reg_cfg: dict) -> tuple[bool, list[str]]:
    """Criterio de validación: el modelo solo se promueve si cumple TODOS los mínimos."""
    checks = {
        "test_auc": reg_cfg["min_test_auc"],
        "test_ks": reg_cfg["min_test_ks"],
        "test_capture_top20": reg_cfg["min_capture_top20"],
    }
    failures = [
        f"{name}={metrics.get(name, float('nan')):.3f} < {minimum}"
        for name, minimum in checks.items()
        if not metrics.get(name, float("nan")) >= minimum
    ]
    return not failures, failures


def _champion_auc(client: MlflowClient, name: str, alias: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return None
    run = client.get_run(version.run_id)
    return run.data.metrics.get("test_auc")


def register_candidate(cfg: dict, model_uri: str, run_id: str, metrics: dict[str, float]) -> dict:
    """Crea una versión del modelo, la valida y le asigna alias.

    - Falla el gate: queda registrada con tag validation=failed y SIN alias.
    - Pasa el gate: alias `challenger`; si no hay champion o supera su AUC en test, pasa a `champion`.
    """
    setup_mlflow(cfg)
    reg = cfg["registry"]
    name = cfg["mlflow"]["registered_model_name"]
    client = MlflowClient()
    version = mlflow.register_model(model_uri, name)
    ok, failures = passes_gate(metrics, reg)
    client.set_model_version_tag(name, version.version, "validation", "passed" if ok else "failed")
    client.set_model_version_tag(name, version.version, "test_auc", f"{metrics['test_auc']:.4f}")
    client.set_model_version_tag(name, version.version, "source_run_id", run_id)
    result = {"model": name, "version": int(version.version), "validation": "passed" if ok else "failed",
              "failures": failures, "alias": None}
    if not ok:
        return result
    client.set_registered_model_alias(name, reg["challenger_alias"], version.version)
    result["alias"] = reg["challenger_alias"]
    current = _champion_auc(client, name, reg["champion_alias"])
    if current is None or metrics["test_auc"] > current:
        client.set_registered_model_alias(name, reg["champion_alias"], version.version)
        result["alias"] = reg["champion_alias"]
    return result


def promote_challenger(cfg: dict) -> dict:
    """Promueve el challenger a champion si cumple el gate y supera al champion actual."""
    setup_mlflow(cfg)
    reg = cfg["registry"]
    name = cfg["mlflow"]["registered_model_name"]
    client = MlflowClient()
    challenger = client.get_model_version_by_alias(name, reg["challenger_alias"])
    run = client.get_run(challenger.run_id)
    ok, failures = passes_gate(run.data.metrics, reg)
    current = _champion_auc(client, name, reg["champion_alias"])
    new_auc = run.data.metrics["test_auc"]
    if ok and (current is None or new_auc > current):
        client.set_registered_model_alias(name, reg["champion_alias"], challenger.version)
        return {"promoted": True, "version": int(challenger.version), "test_auc": new_auc}
    return {"promoted": False, "failures": failures, "champion_test_auc": current, "challenger_test_auc": new_auc}


def model_uri(cfg: dict, alias: str | None = None) -> str:
    alias = alias or cfg["registry"]["champion_alias"]
    return f"models:/{cfg['mlflow']['registered_model_name']}@{alias}"


def load_model(cfg: dict, alias: str | None = None):
    """Carga el modelo pyfunc por alias desde el Registry (uso en batch y online)."""
    setup_mlflow(cfg)
    return mlflow.pyfunc.load_model(model_uri(cfg, alias))


def describe_contract(cfg: dict, alias: str | None = None) -> dict:
    """Contrato de entrada/salida del modelo para los equipos de batch y API."""
    setup_mlflow(cfg)
    info = mlflow.models.get_model_info(model_uri(cfg, alias))
    return {
        "model_uri": model_uri(cfg, alias),
        "inputs": [(c.name, c.type.name) for c in info.signature.inputs.inputs],
        "outputs": [(c.name, c.type.name) for c in info.signature.outputs.inputs],
        "metadata": info.metadata,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["promote", "contract"])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(promote_challenger(cfg) if args.action == "promote" else describe_contract(cfg))


if __name__ == "__main__":
    main()
