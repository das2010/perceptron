"""Clasificación y regresión (lógica de la Capa 1a movida a adaptadores)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from perceptron.domain.enums import TaskType
from perceptron.tasks.base import (
    Predictions,
    Resolver,
    StepOutput,
    TaskAdapter,
    TaskEvaluation,
    flat_numbers,
    pick_metrics,
)

if TYPE_CHECKING:
    from perceptron.archspec.schema import ArchSpec
    from perceptron.data.pipeline.pipeline import FittedPipeline


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight: torch.Tensor | None
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _forward(model: nn.Module, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    *inputs, y = batch
    out: torch.Tensor = model(*inputs)
    return out, y


@torch.no_grad()
def collect(
    model: nn.Module, loader: torch.utils.data.DataLoader[Any]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    outs, ys = [], []
    for batch in loader:
        out, y = _forward(model, batch)
        outs.append(out)
        ys.append(y)
    out = torch.cat(outs) if outs else torch.zeros(0, 1)
    return out, (torch.cat(ys) if ys else None)


class ClassificationAdapter(TaskAdapter):
    task: ClassVar[TaskType] = TaskType.CLASSIFICATION

    def num_outputs(self, spec: ArchSpec) -> int:
        t = spec.task
        if t.num_classes is None:
            raise ValueError("task.num_classes es requerido para clasificación")
        if t.num_classes == 2 and spec.loss.type == "bce" and not t.multilabel:
            return 1
        return t.num_classes

    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module:
        loss = spec.loss
        weight = None
        if isinstance(loss.class_weights, list):
            weight = torch.tensor(loss.class_weights, dtype=torch.float32)
        elif loss.class_weights == "auto":
            weight = class_weights
        match loss.type:
            case "cross_entropy":
                return nn.CrossEntropyLoss(
                    weight=weight, label_smoothing=float(resolve(loss.label_smoothing))
                )
            case "focal":
                return FocalLoss(weight=weight)
            case "bce":
                return nn.BCEWithLogitsLoss()
        raise ValueError(f"loss {loss.type} no válida para clasificación")

    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection:
        k = spec.task.num_classes or 2
        return pick_metrics(
            {
                "accuracy": lambda: torchmetrics.Accuracy(task="multiclass", num_classes=k),
                "f1_macro": lambda: torchmetrics.F1Score(
                    task="multiclass", num_classes=k, average="macro"
                ),
                "auroc": lambda: torchmetrics.AUROC(task="multiclass", num_classes=k),
            },
            spec.metrics,
        )

    @staticmethod
    def _proba(out: torch.Tensor, bce: bool) -> torch.Tensor:
        if bce:
            p1 = torch.sigmoid(out.squeeze(-1))
            return torch.stack([1 - p1, p1], dim=1)
        return out.softmax(-1)

    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput:
        out, y = _forward(model, batch)
        bce = isinstance(loss_fn, nn.BCEWithLogitsLoss)
        loss = loss_fn(out.squeeze(-1), y.float()) if bce else loss_fn(out, y)
        return StepOutput(
            loss=loss, metric_args=(self._proba(out.detach(), bce), y), batch_size=len(y)
        )

    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions:
        out, y = collect(model, loader)
        proba = self._proba(out, spec.loss.type == "bce")
        return Predictions(
            y_true=y.numpy() if y is not None else None,
            y_pred=proba.argmax(-1).numpy(),
            proba=proba.numpy(),
        )

    def evaluate(
        self, preds: Predictions, spec: ArchSpec, pipeline: FittedPipeline
    ) -> TaskEvaluation:
        from perceptron.evaluation.metrics import classification_metrics

        if preds.y_true is None or preds.proba is None:
            raise ValueError("faltan etiquetas o probabilidades para evaluar")
        cls, curves = classification_metrics(
            preds.y_true.astype(int), preds.proba, pipeline.classes or []
        )
        return TaskEvaluation(
            metrics=flat_numbers(cls.model_dump()),
            detail={"classification": cls.model_dump(mode="json")},
            curves=curves.model_dump(),
        )


class RegressionAdapter(TaskAdapter):
    task: ClassVar[TaskType] = TaskType.REGRESSION

    def num_outputs(self, spec: ArchSpec) -> int:
        return spec.task.num_targets

    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module:
        match spec.loss.type:
            case "mse":
                return nn.MSELoss()
            case "mae":
                return nn.L1Loss()
            case "huber":
                return nn.HuberLoss()
        raise ValueError(f"loss {spec.loss.type} no válida para regresión")

    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection:
        return pick_metrics(
            {
                "mae": torchmetrics.MeanAbsoluteError,
                "rmse": lambda: torchmetrics.MeanSquaredError(squared=False),
                "r2": torchmetrics.R2Score,
            },
            spec.metrics,
        )

    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput:
        out, y = _forward(model, batch)
        pred = out.squeeze(-1)
        loss = loss_fn(pred, y.float())
        return StepOutput(loss=loss, metric_args=(pred.detach(), y.float()), batch_size=len(y))

    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions:
        from perceptron.data.pipeline.pipeline import decode_regression

        out, y = collect(model, loader)
        pred = decode_regression(pipeline, out.squeeze(-1).numpy())
        y_true = decode_regression(pipeline, y.numpy()) if y is not None else None
        return Predictions(y_true=y_true, y_pred=pred)

    def evaluate(
        self, preds: Predictions, spec: ArchSpec, pipeline: FittedPipeline
    ) -> TaskEvaluation:
        from perceptron.evaluation.metrics import regression_metrics

        if preds.y_true is None:
            raise ValueError("faltan valores reales para evaluar")
        reg, curves = regression_metrics(
            preds.y_true.astype(float), np.asarray(preds.y_pred, dtype=float)
        )
        return TaskEvaluation(
            metrics=flat_numbers(reg.model_dump()),
            detail={"regression": reg.model_dump(mode="json")},
            curves=curves.model_dump(),
        )
