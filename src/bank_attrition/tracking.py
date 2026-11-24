"""Utilidades de MLflow: servidor remoto, tags de trazabilidad y artefactos gráficos (R4)."""
from __future__ import annotations

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402


def load_mlflow_password() -> None:
    """En AWS (SageMaker) la contraseña de MLflow se lee de SSM: nunca viaja en la imagen ni en el pipeline.

    Si existe MLFLOW_TRACKING_PASSWORD (local, GitHub Actions) no hace nada.
    """
    parameter = os.environ.get("MLFLOW_TRACKING_PASSWORD_SSM")
    if parameter and not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        import boto3  # import diferido: solo en AWS

        response = boto3.client("ssm").get_parameter(Name=parameter, WithDecryption=True)
        os.environ["MLFLOW_TRACKING_PASSWORD"] = response["Parameter"]["Value"]


def setup_mlflow(cfg: dict) -> None:
    load_mlflow_password()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):  # dentro de la imagen no hay .git: el CD lo pasa al construirla
        return os.environ["GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def standard_tags(cfg: dict, data_version: str, uri: str, stage: str) -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "data_version": str(data_version),
        "data_uri": uri,
        "author": cfg["project"]["author"],
        "stage": stage,
    }


def log_roc(y_true, score, name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="FPR", ylabel="TPR", title=f"Curva ROC (test) - {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/roc_test.png")
    plt.close(fig)


def log_confusion_top(y_true, score, fraction: float) -> None:
    """Matriz de confusión marcando como 'fuga' al top `fraction` de scores (regla de campaña)."""
    cutoff = np.quantile(score, 1 - fraction)
    predicted = (np.asarray(score) >= cutoff).astype(int)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, predicted, ax=ax, colorbar=False)
    ax.set_title(f"Confusión (test), top {int(fraction * 100)}% = fuga")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix_test.png")
    plt.close(fig)


def log_feature_importance(importances: dict[str, float], top: int = 25) -> None:
    ordered = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    mlflow.log_dict({k: float(v) for k, v in ordered}, "feature_importance.json")
    names, values = zip(*ordered[:top], strict=False) if ordered else ((), ())
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(names))))
    ax.barh(list(names)[::-1], list(values)[::-1])
    ax.set_title("Importancia de variables")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)


def log_score_distribution(score, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(score, bins=30)
    ax.set(title=f"Distribución del score {name}", xlabel="probabilidad de fuga")
    fig.tight_layout()
    mlflow.log_figure(fig, f"plots/score_distribution_{name}.png")
    plt.close(fig)
