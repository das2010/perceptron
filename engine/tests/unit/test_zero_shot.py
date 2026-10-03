"""Confianza del zero-shot normalizada entre las clases (RF-LBL-02)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from perceptron.data.labeling import zero_shot
from perceptron.data.labeling.zero_shot import classify, normalize_scores
from perceptron.domain.enums import Modality


def test_independent_scores_become_a_distribution() -> None:
    """SigLIP: puntajes por clase independientes y chicos aunque acierte (caso UC-11)."""
    raw = {"Cat": 0.035, "Dog": 0.002, "Rabbit": 0.001, "Cow": 0.0005}
    norm = normalize_scores(raw)
    assert math.isclose(sum(norm.values()), 1.0)
    assert max(norm, key=lambda k: norm[k]) == "Cat"  # el orden no cambia
    assert norm["Cat"] > 0.85  # y ahora puede pasar el umbral de «aceptar confiables»


def test_distributions_are_left_alone() -> None:
    """NLI y CLAP ya devuelven una distribución: no se tocan."""
    dist = {"a": 0.7, "b": 0.3}
    assert normalize_scores(dist) == dist
    assert normalize_scores({}) == {}


def test_tiny_scores_do_not_vanish() -> None:
    norm = normalize_scores({"x": 1e-9, "y": 1e-10})
    assert norm["x"] > 0.9 and math.isclose(sum(norm.values()), 1.0)


@pytest.mark.parametrize("multi", [False, True])
def test_classify_normalizes_only_single_label(multi: bool) -> None:
    def fake(model: str, modality: Any, inputs: Any, labels: list[str], m: bool) -> Any:
        return [{"Cat": 0.32, "Dog": 0.04} for _ in inputs]

    zero_shot.set_backend(fake)
    try:
        out = classify(Modality.IMAGE, ["a.jpg"], ["Cat", "Dog"], multi_label=multi)
    finally:
        zero_shot.set_backend(None)
    if multi:
        assert out == [{"Cat": 0.32, "Dog": 0.04}]  # cada clase se decide sola
    else:
        assert math.isclose(sum(out[0].values()), 1.0) and out[0]["Cat"] > 0.9
