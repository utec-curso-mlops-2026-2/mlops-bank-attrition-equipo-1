"""PSI y semáforo de data drift (R9)."""
import numpy as np
import pandas as pd
import pytest

from bank_attrition.monitoring import (
    drift_report,
    drift_summary,
    numeric_edges,
    psi_categorical,
    psi_numeric,
    traffic_light,
)


def test_psi_is_zero_for_identical_populations():
    values = np.random.default_rng(0).normal(size=5_000)
    assert psi_numeric(values, values) == pytest.approx(0.0, abs=1e-9)
    cats = ["a", "b", "c", None] * 100
    assert psi_categorical(cats, cats) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=20_000)
    small = psi_numeric(reference, rng.normal(0.1, 1, size=20_000))
    large = psi_numeric(reference, rng.normal(1.0, 1, size=20_000))
    assert small < 0.10 < 0.25 < large


def test_psi_handles_nulls_and_new_categories():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0] * 50)
    current = pd.Series([1.0, 2.0, np.nan, np.nan] * 50)  # aparecen nulos: debe detectarse
    assert psi_numeric(reference, current) > 0.25
    assert psi_categorical(["a"] * 100, ["a"] * 50 + ["nueva"] * 50) > 0.25


def test_edges_are_open_and_unique_for_constant_reference():
    edges = numeric_edges([5.0] * 100)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert len(edges) == len(np.unique(edges))
    assert psi_numeric([5.0] * 100, [5.0] * 100) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("psi", "state"), [(0.0, "estable"), (0.0999, "estable"), (0.10, "alerta"),
                                            (0.25, "alerta"), (0.2501, "critico")])
def test_traffic_light_thresholds(psi, state):
    assert traffic_light(psi) == state


def test_drift_report_has_one_row_per_variable_plus_score():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"x": rng.normal(size=1000), "c": rng.choice(["a", "b"], 1000)})
    cur = pd.DataFrame({"x": rng.normal(2, 1, size=1000), "c": rng.choice(["a", "b"], 1000)})
    report = drift_report(ref, cur, ["x"], ["c"], rng.uniform(size=1000), rng.uniform(size=1000))
    assert list(report["variable"]) == ["probabilidad_fuga", "x", "c"]  # score primero, luego por PSI
    assert report.set_index("variable").loc["x", "estado"] == "critico"
    summary = drift_summary(report)
    assert summary["variable_psi_max"] == "x" and summary["n_critico"] == 1 and summary["n_variables"] == 2
