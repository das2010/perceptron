"""`LightningModule` construido desde una ArchSpec (RF-TRN-03).

Optimizador, scheduler, loss (pesos de clase, label smoothing, focal) y métricas
(torchmetrics) salen del documento; los `{hp, default}` se resuelven con los
overrides del trial de HPO.
"""

from __future__ import annotations

from typing import Any, cast

import lightning as L
import torch

from perceptron.archspec.builder import ArchModel, build_model, num_outputs
from perceptron.archspec.schema import ArchSpec, Scalar, resolve
from perceptron.tasks import get_adapter, supervised

LOWER_IS_BETTER = ("loss", "mae", "rmse", "mse", "mape", "smape")


def monitor_mode(metric: str) -> str:
    return "min" if any(k in metric for k in LOWER_IS_BETTER) else "max"


FocalLoss = supervised.FocalLoss  # compatibilidad


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
        self.model = cast(ArchModel, built.model)
        self.backbones = built.backbones
        self.task = spec.task.type
        self.adapter = get_adapter(self.task)
        self.num_outputs = num_outputs(spec)
        self.loss_fn = self.adapter.build_loss(spec, self._r, class_weights)
        self.train_metrics = self.adapter.build_metrics(spec).clone(prefix="train_")
        self.val_metrics = self.adapter.build_metrics(spec).clone(prefix="val_")
        self.lr_override: float | None = None  # LR finder (RF-TRN-04)

    # ---------------------------------------------------------------- loss

    def _r(self, v: object) -> Any:
        return resolve(v, self.overrides)

    # ---------------------------------------------------------------- forward

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(*inputs)
        return out

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        res = self.adapter.step(self.model, self.loss_fn, batch)
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(*res.metric_args)
        self.log(
            f"{stage}_loss",
            res.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=res.batch_size,
        )
        return res.loss

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
        lr = self.lr_override or float(self._r(opt.lr))
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
            block = self.model.blocks[node_id]
            for param in block.parameters():
                param.requires_grad = trainable
