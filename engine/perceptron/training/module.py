"""`LightningModule` construido desde una ArchSpec (RF-TRN-03).

Optimizador, scheduler, loss (pesos de clase, label smoothing, focal) y métricas
(torchmetrics) salen del documento; los `{hp, default}` se resuelven con los
overrides del trial de HPO.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from perceptron.archspec.builder import build_model, num_outputs
from perceptron.archspec.schema import ArchSpec, Scalar, resolve
from perceptron.domain.enums import TaskType

LOWER_IS_BETTER = ("loss", "mae", "rmse", "mse", "mape", "smape")


def monitor_mode(metric: str) -> str:
    return "min" if any(k in metric for k in LOWER_IS_BETTER) else "max"


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _metrics(
    task: TaskType, num_classes: int | None, names: list[str]
) -> torchmetrics.MetricCollection:
    ms: dict[str, torchmetrics.Metric] = {}
    if task is TaskType.REGRESSION:
        available: dict[str, Any] = {
            "mae": torchmetrics.MeanAbsoluteError,
            "rmse": lambda: torchmetrics.MeanSquaredError(squared=False),
            "r2": torchmetrics.R2Score,
        }
    else:
        k = num_classes or 2
        available = {
            "accuracy": lambda: torchmetrics.Accuracy(task="multiclass", num_classes=k),
            "f1_macro": lambda: torchmetrics.F1Score(
                task="multiclass", num_classes=k, average="macro"
            ),
            "auroc": lambda: torchmetrics.AUROC(task="multiclass", num_classes=k),
        }
    for name in names or list(available):
        if name in available:
            ms[name] = available[name]()
    return torchmetrics.MetricCollection(ms)


class PerceptronModule(L.LightningModule):
    def __init__(
        self,
        spec: ArchSpec,
        overrides: dict[str, Scalar] | None = None,
        *,
        class_weights: torch.Tensor | None = None,
        pretrained_allowed: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.overrides = dict(overrides or {})
        built = build_model(spec, self.overrides, pretrained_allowed=pretrained_allowed)
        self.model = built.model
        self.backbones = built.backbones
        self.task = spec.task.type
        self.num_outputs = num_outputs(spec)
        self.loss_fn = self._loss(class_weights)
        k = spec.task.num_classes
        self.train_metrics = _metrics(self.task, k, spec.metrics).clone(prefix="train_")
        self.val_metrics = _metrics(self.task, k, spec.metrics).clone(prefix="val_")

    # ---------------------------------------------------------------- loss

    def _r(self, v: object) -> Any:
        return resolve(v, self.overrides)

    def _loss(self, class_weights: torch.Tensor | None) -> nn.Module:
        loss = self.spec.loss
        weight = None
        if isinstance(loss.class_weights, list):
            weight = torch.tensor(loss.class_weights, dtype=torch.float32)
        elif loss.class_weights == "auto":
            weight = class_weights
        match loss.type:
            case "cross_entropy":
                return nn.CrossEntropyLoss(
                    weight=weight, label_smoothing=float(self._r(loss.label_smoothing))
                )
            case "focal":
                return FocalLoss(weight=weight)
            case "bce":
                return nn.BCEWithLogitsLoss()
            case "mse":
                return nn.MSELoss()
            case "mae":
                return nn.L1Loss()
            case "huber":
                return nn.HuberLoss()
        raise ValueError(f"loss no soportada: {loss.type}")

    # ---------------------------------------------------------------- forward

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.model(*inputs)

    def _step(self, batch: tuple[torch.Tensor, ...], stage: str) -> torch.Tensor:
        *inputs, y = batch
        out = self(*inputs)
        if self.task is TaskType.REGRESSION:
            pred = out.squeeze(-1)
            loss = self.loss_fn(pred, y.float())
            metric_input: tuple[torch.Tensor, torch.Tensor] = (pred.detach(), y.float())
        elif self.spec.loss.type == "bce":
            loss = self.loss_fn(out.squeeze(-1), y.float())
            p1 = torch.sigmoid(out.squeeze(-1)).detach()
            metric_input = (torch.stack([1 - p1, p1], dim=1), y)
        else:
            loss = self.loss_fn(out, y)
            metric_input = (out.detach().softmax(-1), y)
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(*metric_input)
        self.log(
            f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(y)
        )
        return loss

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    # ---------------------------------------------------------------- optim

    def configure_optimizers(self) -> Any:
        opt = self.spec.optimizer
        lr = float(self._r(opt.lr))
        wd = float(self._r(opt.weight_decay))
        params = [p for p in self.parameters() if p.requires_grad] or list(self.parameters())
        if opt.type == "sgd":
            momentum = float(self._r(opt.momentum) or 0.9)
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=wd
            )
        elif opt.type == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        sched = self.spec.scheduler
        p = sched.params
        match sched.type:
            case "none":
                return optimizer
            case "one_cycle":
                total = int(self.trainer.estimated_stepping_batches)
                s: Any = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr, total_steps=max(total, 1)
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": s, "interval": "step"},
                }
            case "cosine":
                t_max = int(p.get("t_max") or self.trainer.max_epochs or 10)
                s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(t_max, 1))
            case "step":
                s = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(p.get("step_size") or 10),
                    gamma=float(p.get("gamma") or 0.1),
                )
            case "plateau":
                s = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=int(p.get("patience") or 3)
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": s, "monitor": "val_loss"},
                }
        return {"optimizer": optimizer, "lr_scheduler": s}

    # ---------------------------------------------------------------- fine-tuning

    def set_backbone_trainable(self, trainable: bool) -> None:
        for node_id in self.backbones:
            block = self.model.blocks[node_id]  # type: ignore[index]
            for param in block.parameters():
                param.requires_grad = trainable
