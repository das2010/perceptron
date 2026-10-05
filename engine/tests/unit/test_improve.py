"""Del diagnóstico a la próxima corrida: acciones aplicadas a la ArchSpec (ADR-0041, it. 3)."""

from __future__ import annotations

from perceptron.archspec.schema import HP
from perceptron.catalog.templates import image_template, tabular_template
from perceptron.domain.enums import TaskType
from perceptron.llm.schemas import Diagnosis, SuggestedAction
from perceptron.services.improve import apply_action, improvement_options, with_defaults


def _mlp() -> object:
    return tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[]
    )


def _act(kind: str, target: str | None = None, value: object = None) -> SuggestedAction:
    return SuggestedAction(kind=kind, target=target, value=value, rationale="porque sí")  # type: ignore[arg-type]


def test_best_trial_values_become_the_defaults() -> None:
    spec = with_defaults(_mlp(), {"lr": 0.004, "inexistente": 1})  # type: ignore[arg-type]
    assert isinstance(spec.optimizer.lr, HP) and spec.optimizer.lr.default == 0.004


def test_change_hparam_lowers_learning_rate() -> None:
    base = with_defaults(_mlp(), {"lr": 0.01})  # type: ignore[arg-type]
    out = apply_action(base, _act("change_hparam", "lr", 0.001))
    assert out is not None
    spec, change = out
    assert isinstance(spec.optimizer.lr, HP) and spec.optimizer.lr.default == 0.001
    assert change == "lr: 0.01 → 0.001"
    # Sin valor: un tercio del actual.
    spec2, _ = apply_action(base, _act("change_hparam", "lr")) or (base, "")
    assert (
        isinstance(spec2.optimizer.lr, HP) and abs(float(spec2.optimizer.lr.default) - 0.003) < 1e-9
    )
    assert apply_action(base, _act("change_hparam", "no_existe", 3)) is None


def test_regularization_raises_dropout_then_weight_decay() -> None:
    img = image_template("small_cnn", task=TaskType.CLASSIFICATION, num_classes=3, image_size=64)
    spec, change = apply_action(img, _act("add_regularization")) or (img, "")
    assert spec.hyperparameters()["dropout"] == 0.3 and change.startswith("dropout: 0.2")
    linear = tabular_template(
        "linear", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=1, cardinalities=[]
    )
    spec, change = apply_action(linear, _act("add_regularization")) or (linear, "")
    assert change.startswith("weight_decay") and spec.hyperparameters()["weight_decay"] == 1e-3


def test_epochs_and_rebalance() -> None:
    base = _mlp()
    spec, change = apply_action(base, _act("more_epochs")) or (base, "")  # type: ignore[arg-type]
    assert change == "epochs: 40 → 80"
    spec, change = apply_action(base, _act("fewer_epochs", value=12)) or (base, "")  # type: ignore[arg-type]
    assert change == "epochs: 40 → 12"
    base.loss.class_weights = "none"  # type: ignore[attr-defined]
    spec, change = apply_action(base, _act("rebalance")) or (base, "")  # type: ignore[arg-type]
    assert spec.loss.class_weights == "auto" and change == "class_weights: none → auto"
    spec, change = apply_action(spec, _act("rebalance")) or (spec, "")
    assert spec.training.oversample and change == "oversample: no → sí"


def test_options_mark_what_is_not_architecture() -> None:
    diagnosis = Diagnosis(
        summary="sobreajuste",
        actions=[
            _act("add_regularization"),
            _act("add_augmentation"),
            _act("change_architecture"),
            _act("none"),
        ],
    )
    options = improvement_options(_mlp(), diagnosis, {"lr": 0.002})  # type: ignore[arg-type]
    assert [o.index for o in options] == [0, 1, 2]
    assert options[0].applicable and options[0].change
    assert not options[1].applicable and options[1].hint == "pipeline"
    assert options[2].hint == "design"
