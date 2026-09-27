from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from perceptron.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    regression_metrics,
)


def test_binary_classification() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 300), 0, 1)
    proba = np.stack([1 - score, score], 1)
    m, curves = classification_metrics(y, proba, ["no", "si"])
    assert m.accuracy == pytest.approx(skm.accuracy_score(y, proba.argmax(1)))
    assert m.roc_auc == pytest.approx(skm.roc_auc_score(y, score), abs=1e-6)
    assert m.pr_auc is not None and 0 < m.pr_auc <= 1
    assert np.array(m.confusion_matrix).sum() == 300
    assert [c.label for c in m.per_class] == ["no", "si"]
    assert m.optimal_threshold is not None and 0 < m.optimal_threshold.threshold < 1
    assert 0 < len(curves.roc) <= 200 and len(curves.pr) > 0
    assert curves.reliability


def test_multiclass_classification() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    logits = rng.normal(0, 1, (200, 3))
    logits[np.arange(200), y] += 2
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    m, _ = classification_metrics(y, proba, ["a", "b", "c"])
    assert m.roc_auc is not None and m.roc_auc > 0.8
    assert m.f1_macro == pytest.approx(skm.f1_score(y, proba.argmax(1), average="macro"))
    assert m.optimal_threshold is None
    assert len(m.confusion_matrix) == 3


def test_classification_with_missing_class_in_test() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.05, 0.05], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
    m, _ = classification_metrics(y, proba, ["a", "b", "c"])
    assert m.roc_auc is None  # no está la clase "c"
    assert m.accuracy == 1.0


def test_ece_perfectly_calibrated_is_small() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0.5, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    proba = np.stack([1 - p, p], 1)
    ece, diagram = expected_calibration_error(np.where(y == 1, 1, 0), proba)
    assert ece < 0.03
    assert diagram


def test_regression() -> None:
    y = np.array([10.0, 20.0, 30.0, 0.0])
    pred = np.array([12.0, 18.0, 33.0, 1.0])
    m, curves = regression_metrics(y, pred)
    assert m.mae == pytest.approx(2.0)
    assert m.rmse == pytest.approx(np.sqrt((4 + 4 + 9 + 1) / 4))
    assert m.mape == pytest.approx((0.2 + 0.1 + 0.1) / 3)  # excluye y = 0
    assert m.r2 == pytest.approx(skm.r2_score(y, pred))
    assert len(curves.residuals) == 4
    assert sum(n for _, n in curves.error_histogram) == 4
