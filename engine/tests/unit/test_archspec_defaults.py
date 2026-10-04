"""Regresión lineal sin weight decay (caso «Tabla 3»: 500 → 1489 en vez de 1500)."""

from __future__ import annotations

from perceptron.archspec.defaults import is_linear_regression, without_linear_shrinkage
from perceptron.archspec.schema import HP, ArchSpec, Node
from perceptron.archspec.validate import validate_archspec
from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType


def _mlp(task: TaskType = TaskType.REGRESSION) -> ArchSpec:
    return tabular_template(
        "mlp",
        task=task,
        num_classes=None if task is TaskType.REGRESSION else 2,
        num_numeric=1,
        cardinalities=[],
    )


def _linear(task: TaskType = TaskType.REGRESSION) -> ArchSpec:
    """Como la propone el arquitecto: entrada → cabeza lineal, con weight decay en el HPO."""
    spec = _mlp(task)
    return spec.model_copy(
        update={
            "name": "tabular-linear-reg",
            "nodes": [
                Node(id="features", block="input.tabular", params={"dropout": 0.0}),
                Node(id="head", block="head.linear"),
            ],
            "edges": [("input", "features"), ("features", "head")],
            "optimizer": spec.optimizer.model_copy(
                update={"weight_decay": HP(hp="weight_decay", default=0.01)}
            ),
        }
    )


def test_linear_regression_drops_weight_decay_and_keeps_it_out_of_hpo() -> None:
    spec = _linear()
    assert is_linear_regression(spec) and validate_archspec(spec).valid
    assert "weight_decay" in spec.hyperparameters()
    fixed = without_linear_shrinkage(spec)
    assert fixed.optimizer.weight_decay == 0.0
    assert "weight_decay" not in fixed.hyperparameters()  # el HPO ya no lo explora
    assert fixed.optimizer.lr == spec.optimizer.lr  # el resto no cambia


def test_other_architectures_keep_their_regularization() -> None:
    mlp = _mlp()
    assert not is_linear_regression(mlp)
    assert without_linear_shrinkage(mlp) == mlp
    logistic = _linear(TaskType.CLASSIFICATION)  # en clasificación regulariza con sentido
    assert not is_linear_regression(logistic)
    assert without_linear_shrinkage(logistic) == logistic
