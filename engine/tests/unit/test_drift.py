"""Métricas de drift (RF-MON-02) con distribuciones sintéticas conocidas."""

from __future__ import annotations

import numpy as np
import polars as pl

from perceptron.domain.enums import Severity
from perceptron.monitoring.drift import (
    categorical_drift,
    data_drift,
    embedding_drift,
    js_distance,
    numeric_drift,
    psi,
)


def test_psi_and_js_basics() -> None:
    same = np.array([0.25, 0.25, 0.5])
    assert psi(same, same) == 0.0
    assert js_distance(same, same) < 1e-6
    assert js_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) > 0.99


def test_numeric_drift_detects_shift_only_when_present() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(40, 15, 2000)
    stable = numeric_drift("antiguedad", ref, rng.normal(40, 15, 400))
    shifted = numeric_drift("antiguedad", ref, rng.normal(12, 5, 400))
    assert stable.severity in (Severity.NONE, Severity.LOW)
    assert shifted.severity is Severity.HIGH and shifted.psi and shifted.psi > 0.3
    assert shifted.ks_pvalue is not None and shifted.ks_pvalue < 1e-6
    assert shifted.current_mean is not None and shifted.current_mean < 20


def test_categorical_drift_new_category_and_mix() -> None:
    ref = ["AMBA"] * 50 + ["Córdoba"] * 30 + ["Mendoza"] * 20
    stable = categorical_drift("region", ref, ["AMBA"] * 25 + ["Córdoba"] * 15 + ["Mendoza"] * 10)
    new = categorical_drift("region", ref, ["AMBA"] * 20 + ["Patagonia"] * 20 + ["Mendoza"] * 10)
    assert stable.severity is Severity.NONE
    assert new.severity.rank >= Severity.MEDIUM.rank
    assert new.unseen_fraction == 0.4 and "Patagonia" in new.top_changes


def test_data_drift_overall_share() -> None:
    rng = np.random.default_rng(1)
    ref = pl.DataFrame(
        {"a": rng.normal(0, 1, 1000), "b": rng.normal(5, 2, 1000), "c": ["x", "y"] * 500}
    )
    cur = pl.DataFrame(
        {"a": rng.normal(3, 1, 300), "b": rng.normal(5, 2, 300), "c": ["x", "y"] * 150}
    )
    report = data_drift(ref, cur, ["a", "b"], ["c"])
    assert [f.feature for f in report.drifted] == ["a"]
    assert report.severity.rank >= Severity.MEDIUM.rank
    assert report.n_reference == 1000 and report.n_current == 300


def test_embedding_drift() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, (300, 8))
    same = embedding_drift(ref, rng.normal(0, 1, (200, 8)), permutations=30)
    moved = embedding_drift(ref, rng.normal(1.5, 1, (200, 8)), permutations=30)
    assert same.severity.rank <= Severity.LOW.rank and 0.35 < same.domain_auc < 0.65
    assert moved.severity is Severity.HIGH and moved.domain_auc > 0.8
    assert moved.mmd > same.mmd and moved.mmd_pvalue is not None and moved.mmd_pvalue < 0.05
