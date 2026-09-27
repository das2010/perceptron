"""Contrato de los adaptadores de tarea (ADR-0017).

Cada tarea (clasificación, regresión, forecasting, anomalías, detección,
segmentación, OCR, eventos sonoros) define en un solo lugar: cuántas salidas
tiene la cabeza, su loss, sus métricas de validación, cómo se hace un paso de
entrenamiento, cómo se predice y cómo se evalúa sobre el test sellado.
`training.module`, `training.inference` y `evaluation.evaluate` solo delegan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torchmetrics
from torch import nn

from perceptron.domain.enums import TaskType

if TYPE_CHECKING:
    from perceptron.archspec.schema import ArchSpec
    from perceptron.data.pipeline.pipeline import FittedPipeline

Resolver = Callable[[object], Any]


@dataclass
class StepOutput:
    loss: torch.Tensor
    metric_args: tuple[Any, ...]
    batch_size: int


@dataclass
class Predictions:
    """Salidas del modelo sobre un split, en la escala original del problema."""

    y_true: np.ndarray | None
    y_pred: np.ndarray
    proba: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskEvaluation:
    metrics: dict[str, float]
    detail: dict[str, Any] = field(default_factory=dict)
    curves: dict[str, list[list[float]]] = field(default_factory=dict)


class TaskAdapter(ABC):
    task: ClassVar[TaskType]
    default_monitor: ClassVar[str] = "val_loss"

    @abstractmethod
    def num_outputs(self, spec: ArchSpec) -> int: ...

    def check_output(self, kind: str, shape: tuple[int, ...], spec: ArchSpec) -> str | None:
        """None si la salida del grafo sirve para la tarea; si no, el motivo (etapa 4 de §9.3).

        Por defecto: vector de `num_outputs` valores.
        """
        expected = self.num_outputs(spec)
        if kind != "features" or shape != (expected,):
            return f"la salida debe ser un vector de {expected} valores y es {kind} {list(shape)}"
        return None

    @abstractmethod
    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module: ...

    @abstractmethod
    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection: ...

    @abstractmethod
    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput: ...

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions: ...

    def calibrate(
        self, model: nn.Module, val_loader: torch.utils.data.DataLoader[Any], spec: ArchSpec
    ) -> dict[str, Any]:
        """Ajustes post-entrenamiento sobre val (p. ej. umbral de anomalías). Por defecto nada."""
        return {}

    @property
    def needs_calibration(self) -> bool:
        return type(self).calibrate is not TaskAdapter.calibrate

    @abstractmethod
    def evaluate(
        self,
        preds: Predictions,
        spec: ArchSpec,
        pipeline: FittedPipeline,
        calibration: dict[str, Any] | None = None,
    ) -> TaskEvaluation: ...


def pick_metrics(
    available: dict[str, Callable[[], torchmetrics.Metric]], names: list[str]
) -> torchmetrics.MetricCollection:
    chosen: dict[str, torchmetrics.Metric | torchmetrics.MetricCollection] = {
        n: available[n]() for n in (names or list(available)) if n in available
    }
    return torchmetrics.MetricCollection(chosen)


def flat_numbers(data: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in data.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
