"""Forma de los datos y requisitos que dispara (caso «Tabla X»)."""

from __future__ import annotations

import numpy as np

from perceptron.data.profiling.card import ProfileCard
from perceptron.domain.enums import Modality, TaskType
from perceptron.services.design import design_requirements
from perceptron.services.fit_checks import linearity_of

X = np.arange(1, 5001, 3, dtype=float).reshape(-1, 1)


def _card() -> ProfileCard:
    return ProfileCard(
        modality=Modality.TABULAR,
        num_samples=976,
        profiled_samples=976,
        split_counts={"train": 976},
        num_features=1,
        target=None,
        columns=[],
    )


def _codes(linearity: object, use_case: dict[str, object] | None = None) -> dict[str, str]:
    reqs = design_requirements(
        _card(),
        TaskType.REGRESSION,
        use_case,
        device="cpu",
        allow_pretrained=True,
        linearity=linearity,  # type: ignore[arg-type]
    )
    return {r.code: r.level for r in reqs.items}


def test_a_curve_needs_hidden_layers_tabla_x() -> None:
    y = X[:, 0] + np.sqrt(2 / 7) * X[:, 0] ** 1.5
    lin = linearity_of(X, y, ["entrada"])
    assert lin is not None and lin.evident_curvature and not lin.linear_explains
    assert 0.97 < lin.r2_linear < 0.99 and lin.r2_curved > 0.9999
    codes = _codes(lin)
    assert codes["nonlinear_capacity"] == "must" and "linear_option" not in codes
    # La ficha pedía una regla, pero los datos son curvos: la lineal queda como referencia.
    assert _codes(lin, {"problem": "rule"})["linear_option"] == "should"


def test_a_line_makes_linear_mandatory_tabla_3() -> None:
    lin = linearity_of(X, 3 * X[:, 0], ["numero"])
    assert lin is not None and lin.linear_explains and not lin.evident_curvature
    codes = _codes(lin)
    assert codes["linear_option"] == "must" and "nonlinear_capacity" not in codes


def test_noise_is_not_curvature() -> None:
    rng = np.random.default_rng(0)
    y = 2 * X[:, 0] + rng.normal(0, 800, len(X))
    lin = linearity_of(X, y, ["x"])
    assert lin is not None and not lin.evident_curvature and not lin.linear_explains
    assert linearity_of(X[:5], X[:5, 0], ["x"]) is None  # muy pocas filas
