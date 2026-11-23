import numpy as np
import pytest

from bank_attrition.metrics import assign_decile, capture_rate_at, evaluate, ks_statistic, lift_at


def test_perfect_ranking():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    score = np.linspace(0, 1, 10)
    result = evaluate(y, score)
    assert result["auc"] == pytest.approx(1.0)
    assert result["gini"] == pytest.approx(1.0)
    assert result["ks"] == pytest.approx(1.0)
    assert result["capture_top20"] == pytest.approx(1.0)


def test_lift_and_capture_values():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # tasa global 20 %
    score = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])
    assert lift_at(y, score, 0.10) == pytest.approx(1 / 0.2)  # el top 10 % es 1 cliente, fuga
    assert capture_rate_at(y, score, 0.20) == pytest.approx(0.5)


def test_ks_is_zero_for_constant_score():
    y = np.array([0, 1, 0, 1])
    assert ks_statistic(y, np.ones(4)) == pytest.approx(0.0)


def test_deciles_cover_one_to_ten():
    deciles = assign_decile(np.random.default_rng(0).random(1000))
    assert set(deciles) == set(range(1, 11))
    assert deciles.min() == 1 and deciles.max() == 10
