"""`LightningModule` construido desde una ArchSpec (RF-TRN-03).

Optimizador, scheduler, loss (pesos de clase, label smoothing, focal) y métricas
(torchmetrics) salen del documento; los `{hp, default}` se resuelven con los
overrides del trial de HPO.
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

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
        target_scale: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        # Regresión con target estandarizado: (media, desvío) para informar MAE/RMSE en las
        # unidades del dato, igual que la evaluación en test. La loss sigue estandarizada.
        self.target_scale = target_scale
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

    def _val_loss_fn(self) -> Any:
        """La `val_loss` se mide sin suavizado de etiquetas (el entrenamiento sí lo usa).

        Con suavizado ε la entropía cruzada mínima ya no es 0: comparar `val_loss` entre trials
        con distinto `label_smoothing` premiaba no suavizar y el análisis del estudio lo
        mostraba como el hiperparámetro más importante. Sin suavizado la comparación es justa
        (y también el early stopping). Es una función, no un submódulo: no cambia el
        `state_dict` de los checkpoints.
        """
        fn = self.loss_fn
        if isinstance(fn, nn.CrossEntropyLoss) and fn.label_smoothing > 0:
            return partial(
                F.cross_entropy,
                weight=fn.weight,
                ignore_index=fn.ignore_index,
                reduction=fn.reduction,
            )
        return fn

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        loss_fn = self.loss_fn if stage == "train" else self._val_loss_fn()
        res = self.adapter.step(self.model, loss_fn, batch)
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        args = res.metric_args
        if self.target_scale is not None:
            mean, std = self.target_scale
            args = tuple(a * std + mean for a in args)
        metrics.update(*args)
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
        params = self.param_groups(lr)
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
                    optimizer,
                    max_lr=[g["lr"] for g in optimizer.param_groups],
                    total_steps=max(total, 1),
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

    def param_groups(self, lr: float) -> list[dict[str, Any]]:
        """Un grupo con el LR base y, con LR discriminativo, otro para el backbone (RF-TRN-09).
        Todos los parámetros entran al optimizador: el congelado solo apaga el gradiente."""
        mult = float(self.spec.training.backbone_lr_mult)
        backbone = {
            id(p) for node_id in self.backbones for p in self.model.blocks[node_id].parameters()
        }
        if mult == 1.0 or not backbone:
            return [{"params": list(self.parameters()), "lr": lr}]
        rest = [p for p in self.parameters() if id(p) not in backbone]
        base = [p for p in self.parameters() if id(p) in backbone]
        return [{"params": rest, "lr": lr}, {"params": base, "lr": lr * mult}]

    def _tunable_backbones(self) -> list[Any]:
        """Backbones que se congelan/descongelan (con LoRA el modelo base queda congelado)."""
        blocks = [self.model.blocks[node_id] for node_id in self.backbones]
        return [b for b in blocks if not getattr(b, "uses_lora", False)]

    def backbone_layer_groups(self) -> list[list[torch.nn.Parameter]]:
        """Grupos de capas del backbone, de la entrada a la salida (para descongelar de a uno)."""
        groups: list[list[torch.nn.Parameter]] = []
        for block in self._tunable_backbones():
            body = getattr(block, "body", block)
            layers = getattr(getattr(body, "encoder", None), "layer", None)  # transformers
            parts = (
                [getattr(body, "embeddings", None), *layers]
                if layers is not None
                else list(body.children())
            )
            groups += [list(p.parameters()) for p in parts if p is not None]
        return [g for g in groups if g]

    def set_backbone_trainable(self, trainable: bool, top_groups: int | None = None) -> None:
        """`top_groups`: con descongelado progresivo, solo los últimos N grupos se entrenan."""
        for block in self._tunable_backbones():
            for param in block.parameters():
                param.requires_grad = trainable and top_groups is None
        if trainable and top_groups:
            groups = self.backbone_layer_groups()
            for group in groups[max(len(groups) - top_groups, 0) :]:
                for param in group:
                    param.requires_grad = True
