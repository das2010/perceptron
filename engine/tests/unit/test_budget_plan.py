"""Plan de intentos y épocas del HPO (ADR-0041)."""

from __future__ import annotations

from perceptron.catalog.templates import image_template, tabular_template
from perceptron.domain.enums import TaskType
from perceptron.hpo.plan import DEFAULT_TIME_S, MAX_TRIALS, plan_budget


def _mlp() -> object:
    return tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[]
    )


def test_linear_regression_needs_one_short_trial_tabla_x() -> None:
    spec = tabular_template(
        "linear", task=TaskType.REGRESSION, num_classes=None, num_numeric=1, cardinalities=[]
    )
    plan = plan_budget(spec, epoch_time_s=0.4)
    assert plan.max_trials == 1 and plan.max_epochs_per_trial == 10 and not plan.tuned_params
    assert "mínimos cuadrados" in plan.reasons[0]


def test_trials_grow_with_the_search_space() -> None:
    plan = plan_budget(_mlp(), epoch_time_s=0.1)  # type: ignore[arg-type]
    n = len(plan.tuned_params)
    assert n >= 3 and plan.max_trials == min(MAX_TRIALS, 4 + 6 * n)
    assert plan.max_epochs_per_trial == 40  # las de la plantilla
    assert plan.estimated_s is not None and plan.estimated_s <= DEFAULT_TIME_S
    assert plan.time_budget_s == DEFAULT_TIME_S


def test_slow_epochs_cut_trials_then_epochs() -> None:
    fast = plan_budget(_mlp(), epoch_time_s=0.1)  # type: ignore[arg-type]
    slow = plan_budget(_mlp(), epoch_time_s=5.0, time_budget_s=30 * 60)  # type: ignore[arg-type]
    assert slow.max_trials < fast.max_trials
    assert slow.estimated_s is not None and slow.estimated_s <= 30 * 60
    very_slow = plan_budget(_mlp(), epoch_time_s=60.0, time_budget_s=30 * 60)  # type: ignore[arg-type]
    assert very_slow.max_epochs_per_trial < 40 and very_slow.max_trials >= 5
    assert any("tiempo disponible" in r for r in very_slow.reasons)


def test_pretrained_backbones_need_fewer_epochs() -> None:
    spec = image_template(
        "mobilenetv3_small_100",
        task=TaskType.CLASSIFICATION,
        num_classes=3,
        image_size=96,
        pretrained=True,
        epochs=40,
    )
    plan = plan_budget(spec, epoch_time_s=2.0)
    assert plan.max_epochs_per_trial == 15
    assert any("preentrenado" in r for r in plan.reasons)


def test_without_an_epoch_estimate_nothing_is_fitted_to_time() -> None:
    plan = plan_budget(_mlp(), epoch_time_s=None)  # type: ignore[arg-type]
    assert plan.estimated_s is None and plan.max_trials > 1
    assert any("Sin estimación" in r for r in plan.reasons)
