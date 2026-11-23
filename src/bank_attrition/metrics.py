"""Métricas pertinentes al negocio: AUC/Gini, KS y priorización (lift, fugas capturadas)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true, score) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def _top_mask(score, fraction: float) -> np.ndarray:
    n_top = max(int(np.ceil(len(score) * fraction)), 1)
    order = np.argsort(-np.asarray(score), kind="stable")
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:n_top]] = True
    return mask


def capture_rate_at(y_true, score, fraction: float) -> float:
    """% de las fugas totales que caen en el top `fraction` de clientes con mayor score."""
    y = np.asarray(y_true)
    total = y.sum()
    return float(y[_top_mask(score, fraction)].sum() / total) if total else float("nan")


def lift_at(y_true, score, fraction: float) -> float:
    """Tasa de fuga del top `fraction` dividida por la tasa de fuga global."""
    y = np.asarray(y_true)
    base = y.mean()
    return float(y[_top_mask(score, fraction)].mean() / base) if base else float("nan")


def evaluate(y_true, score, top_fractions=(0.10, 0.20)) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, score))
    result = {"auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y_true, score)}
    for fraction in top_fractions:
        tag = f"top{int(round(fraction * 100))}"
        result[f"lift_{tag}"] = lift_at(y_true, score, fraction)
        result[f"capture_{tag}"] = capture_rate_at(y_true, score, fraction)
    return result


def assign_decile(score) -> np.ndarray:
    """Decil de riesgo 1 (menor score) a 10 (mayor score), por rango."""
    ranks = np.asarray(score).argsort().argsort()
    return (ranks * 10 // len(ranks) + 1).astype(int)
