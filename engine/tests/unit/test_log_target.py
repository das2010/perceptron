"""Objetivo en escala logarítmica y datos sin ruido (caso «Tabla X», salida de 13 a 194 000)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from perceptron.catalog.templates import tabular_template
from perceptron.data.pipeline.pipeline import (
    FittedPipeline,
    PipelineSpec,
    TargetSpec,
    _fit_target,
    decode_regression,
    encode_target,
)
from perceptron.domain.enums import Modality, TaskType
from perceptron.hpo.plan import plan_budget
from perceptron.hpo.strategy import default_search_space, without_dropout
from perceptron.services.fit_checks import Linearity, deterministic, log_target_reason

Y = pl.Series("salida", [13.0, 634.5, 6476.1, 69315.3, 193982.2])


def _fitted(log: bool, y: pl.Series = Y) -> FittedPipeline:
    target = TargetSpec(name="salida", task=TaskType.REGRESSION, standardize=True, log=log)
    spec = PipelineSpec(modality=Modality.TABULAR, target=target)
    return FittedPipeline(spec=spec, **_fit_target(target, y))


def test_log_target_round_trips_and_trains_on_relative_scale() -> None:
    fitted = _fitted(log=True)
    assert fitted.target_log
    encoded = encode_target(fitted, Y)
    assert abs(float(encoded.mean())) < 1e-5  # estandarizado sobre log(1 + y)
    assert decode_regression(fitted, encoded) == pytest.approx(Y.to_numpy(), rel=1e-4)
    # El mismo error en la escala del modelo es relativo: +10 % en 634 y en 193 982.
    up = decode_regression(fitted, encoded + np.float32(0.1 / fitted.target_std))  # type: ignore[operator]
    ratios = (up + 1) / (Y.to_numpy() + 1)
    assert ratios.max() / ratios.min() < 1.001


def test_negative_targets_never_use_log() -> None:
    fitted = _fitted(log=True, y=pl.Series("salida", [-5.0, 10.0, 1000.0]))
    assert not fitted.target_log


def _card(p05: float, p95: float) -> object:
    from perceptron.data.profiling.card import NumericStats, ProfileCard, TargetProfile
    from perceptron.data.schema import SemanticType

    stats = NumericStats(
        mean=1.0,
        std=1.0,
        quantiles={"p05": p05, "p25": None, "p50": None, "p75": None, "p95": p95},
        histogram=[],
        outlier_fraction=0.0,
    )
    target = TargetProfile(
        name="salida", semantic=SemanticType.NUMERIC, task_hint=TaskType.REGRESSION, numeric=stats
    )
    return ProfileCard(
        modality=Modality.TABULAR,
        num_samples=10,
        profiled_samples=10,
        split_counts={},
        num_features=1,
        target=target,
        columns=[],
    )


def test_log_is_proposed_only_for_wide_positive_targets() -> None:
    curved = Linearity(r2_linear=0.98, r2_curved=0.99998, n=900)
    line = Linearity(r2_linear=0.99999, r2_curved=0.99999, n=900)
    assert log_target_reason(_card(2363, 179_600), curved)  # Tabla X: 76 veces
    assert log_target_reason(_card(22, 427), curved) is None  # menos de 20 veces
    assert log_target_reason(_card(-10, 179_600), curved) is None  # negativos
    assert log_target_reason(_card(2363, 179_600), line) is None  # una recta: queda lineal
    assert deterministic(curved) and deterministic(line)
    assert not deterministic(Linearity(r2_linear=0.9, r2_curved=0.95, n=900))


def test_noise_free_data_fixes_dropout_at_zero() -> None:
    spec = tabular_template(
        "mlp", task=TaskType.REGRESSION, num_classes=None, num_numeric=1, cardinalities=[]
    )
    space = default_search_space(spec)
    assert any("dropout" in p.name for p in space)
    fixed = without_dropout(space)
    [drop] = [p for p in fixed if "dropout" in p.name]
    assert drop.type == "categorical" and drop.choices == [0.0] and drop.default == 0.0
    normal = plan_budget(spec, epoch_time_s=0.1)
    quiet = plan_budget(spec, epoch_time_s=0.1, noise_free=True)
    assert quiet.max_trials == normal.max_trials - 6
    assert not any("dropout" in p for p in quiet.tuned_params)
    assert any("sin ruido" in r for r in quiet.reasons)
