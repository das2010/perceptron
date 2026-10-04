"""Umbral de decisión por costo de los errores (ADR-0040, fase 2)."""

from __future__ import annotations

import numpy as np

from perceptron.evaluation.cost import CostSpec, cost_threshold


def _data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    score = np.clip(0.2 * y + rng.normal(0.35, 0.15, n), 0, 1)  # clases solapadas
    return y, score


def test_missing_a_failure_10x_worse_lowers_the_threshold() -> None:
    (yv, sv), (yt, st) = _data(400, 0), _data(400, 1)
    out = cost_threshold(yv, sv, yt, st, CostSpec(worse="false_negative", ratio=10), "falla")
    assert out["positive_class"] == "falla" and out["threshold"] < 0.5  # type: ignore[operator]
    test, at50 = out["test"], out["test_at_50"]
    assert test["recall"] > at50["recall"]  # type: ignore[index]
    assert test["cost"] < at50["cost"]  # type: ignore[index]


def test_false_alarms_expensive_raise_the_threshold() -> None:
    (yv, sv), (yt, st) = _data(400, 2), _data(400, 3)
    out = cost_threshold(yv, sv, yt, st, CostSpec(worse="false_positive", ratio=10), "1")
    assert out["threshold"] > 0.5  # type: ignore[operator]
    assert out["test"]["fp"] <= out["test_at_50"]["fp"]  # type: ignore[index]


def test_symmetric_costs_keep_the_threshold_near_50() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    out = cost_threshold(y, s, y, s, CostSpec(worse="false_negative", ratio=1), "1")
    assert out["test"]["fn"] == 0 and out["test"]["fp"] == 0  # type: ignore[index]
    assert abs(out["threshold"] - 0.5) <= 0.1  # type: ignore[operator]
