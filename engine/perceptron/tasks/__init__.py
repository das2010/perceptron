"""Adaptadores de tarea (ADR-0017). `get_adapter(task)` devuelve el de cada TaskType."""

from __future__ import annotations

from perceptron.domain.enums import TaskType
from perceptron.tasks.base import Predictions, StepOutput, TaskAdapter, TaskEvaluation

_REGISTRY: dict[TaskType, TaskAdapter] = {}


def register(adapter: TaskAdapter) -> TaskAdapter:
    _REGISTRY[adapter.task] = adapter
    return adapter


def get_adapter(task: TaskType) -> TaskAdapter:
    if not _REGISTRY:
        _load_builtin()
    try:
        return _REGISTRY[task]
    except KeyError:
        raise NotImplementedError(f"la tarea {task.value} todavía no tiene adaptador") from None


def _load_builtin() -> None:
    from perceptron.tasks.supervised import ClassificationAdapter, RegressionAdapter

    for adapter in (ClassificationAdapter(), RegressionAdapter()):
        register(adapter)


__all__ = ["Predictions", "StepOutput", "TaskAdapter", "TaskEvaluation", "get_adapter", "register"]
