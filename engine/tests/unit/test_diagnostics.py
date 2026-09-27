"""Diagnóstico por reglas sobre curvas sintéticas (evidencia del Diagnosticador)."""

from __future__ import annotations

import math

from perceptron.training.diagnostics import rules_diagnosis


def _hist(train: list[float], val: list[float]) -> list[dict[str, float]]:
    return [
        {"epoch": float(i), "train_loss": t, "val_loss": v}
        for i, (t, v) in enumerate(zip(train, val, strict=True))
    ]


def _kinds(history: list[dict[str, float]], **kw: object) -> set[str]:
    return {p.kind for p in rules_diagnosis(history, **kw).problems}  # type: ignore[arg-type]


def test_overfitting() -> None:
    train = [1.0 - 0.08 * i for i in range(12)]
    val = [0.9, 0.7, 0.6, 0.55, 0.56, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    d = rules_diagnosis(_hist(train, val), hyperparameters={"lr": 1e-3, "dropout": 0.1})
    [p] = [p for p in d.problems if p.kind == "overfitting"]
    assert "época 4" in p.evidence and p.severity == "high"
    assert {a.kind for a in d.actions} >= {"add_regularization", "fewer_epochs"}


def test_divergence_short_circuits() -> None:
    d = rules_diagnosis(
        _hist([1.0, 2.0, math.nan], [1.0, 3.0, math.nan]), hyperparameters={"lr": 0.1}
    )
    assert [p.kind for p in d.problems] == ["divergence"]
    assert d.actions[0].target == "lr" and d.actions[0].value == 0.1 * 0.1


def test_underfitting_and_plateau() -> None:
    flat = [1.0, 0.99, 0.98, 0.97, 0.97, 0.97, 0.97, 0.97]
    assert {"underfitting", "plateau"} <= _kinds(_hist(flat, flat))


def test_stopped_too_early_and_imbalance() -> None:
    down = [1.0 / (i + 1) for i in range(5)]
    kinds = _kinds(_hist(down, down), max_epochs=5, imbalance_ratio=0.1)
    assert {"stopped_too_early", "class_imbalance"} <= kinds


def test_healthy_run() -> None:
    train = [1.0, 0.6, 0.4, 0.3, 0.25, 0.22]
    val = [1.0, 0.65, 0.45, 0.36, 0.33, 0.325]
    d = rules_diagnosis(_hist(train, val), max_epochs=30)
    assert not d.problems and "No se detectaron" in d.summary
