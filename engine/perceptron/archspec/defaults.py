"""Ajustes de defaults que dependen de la forma de la ArchSpec (no del bloque suelto)."""

from __future__ import annotations

from perceptron.archspec.schema import HP, ArchSpec
from perceptron.domain.enums import TaskType

LINEAR_HEAD = "head.linear"


def is_linear_regression(spec: ArchSpec) -> bool:
    """Regresión lineal pura: la entrada va directo a una cabeza lineal, sin capas ocultas."""
    if spec.task.type is not TaskType.REGRESSION:
        return False
    blocks = [n.block for n in spec.nodes]
    return LINEAR_HEAD in blocks and all(b == LINEAR_HEAD or b.startswith("input.") for b in blocks)


def without_linear_shrinkage(spec: ArchSpec) -> ArchSpec:
    """En una regresión lineal el weight decay solo achica la pendiente (sesgo) y empeora la
    extrapolación: se fija en 0 y queda fuera del HPO. Caso «Tabla 3»: con 0,01 aprendía
    2,963 × numero + 7,5 en vez de 3 × numero, y 500 → 1489 en vez de 1500."""
    if not is_linear_regression(spec) or spec.optimizer.weight_decay == 0.0:
        return spec
    return spec.model_copy(
        update={"optimizer": spec.optimizer.model_copy(update={"weight_decay": 0.0})}
    )


def spec_epochs(spec: ArchSpec) -> int:
    """Épocas que propone la arquitectura (el default del HP si se ajusta; 30 si no se sabe)."""
    epochs = spec.training.epochs
    value = epochs.default if isinstance(epochs, HP) else epochs
    return int(value) if isinstance(value, int | float) else 30
