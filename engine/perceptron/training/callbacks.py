"""Callbacks del worker: eventos JSONL, control por stdin y congelado del backbone."""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, TextIO

import lightning as L
import psutil
import torch

from perceptron.training.config import RunEvent


def is_rank_zero() -> bool:
    """Proceso principal (en DDP, Lightning lanza los demás con LOCAL_RANK > 0)."""
    import os

    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"


class EventEmitter:
    """Escribe eventos como JSON en una línea (stdout del worker)."""

    def __init__(self, run_id: str, stream: TextIO | None = None) -> None:
        self.run_id = run_id
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()
        self.enabled = is_rank_zero()

    def emit(self, event: str, **kw: Any) -> None:
        if not self.enabled:
            return  # rank > 0 en DDP: solo el proceso principal informa (RF-TRN-08)
        ev = RunEvent(event=event, run_id=self.run_id, ts=time.time(), **kw)  # type: ignore[arg-type]
        with self._lock:
            self.stream.write(ev.model_dump_json() + "\n")
            self.stream.flush()


def resource_usage() -> dict[str, float]:
    out = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(psutil.Process().memory_info().rss / 2**30, 3),
    }
    if torch.cuda.is_available():
        out["vram_used_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
    return out


def _floats(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, int | float | torch.Tensor) and torch.as_tensor(v).numel() == 1
    }


class ProgressCallback(L.Callback):
    """Progreso en vivo (RF-TRN-06): época, batch, loss, métricas, LR, throughput, recursos, ETA.

    El evento de época se emite en `on_train_epoch_end`: la validación ya corrió y sus
    métricas están en `callback_metrics`; el `train_loss` se promedia acá mismo porque
    Lightning agrega las métricas de train recién después de los hooks de fin de época.
    """

    def __init__(self, emitter: EventEmitter, every_n_batches: int) -> None:
        self.emitter = emitter
        self.every = every_n_batches
        self.history: list[dict[str, float]] = []
        self._epoch_start = 0.0
        self._samples = 0
        self._loss_sum = 0.0
        self._batches = 0

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_start = time.time()
        self._samples = 0
        self._loss_sum = 0.0
        self._batches = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = float(outputs["loss"] if isinstance(outputs, dict) else outputs)
        self._samples += len(batch[-1])
        self._loss_sum += loss
        self._batches += 1
        if batch_idx % self.every == 0:
            elapsed = max(time.time() - self._epoch_start, 1e-6)
            self.emitter.emit(
                "batch",
                epoch=trainer.current_epoch,
                step=trainer.global_step,
                metrics={
                    "loss": loss,
                    "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
                    "samples_per_s": round(self._samples / elapsed, 2),
                },
                data={"batch": batch_idx, "batches": trainer.num_training_batches},
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = {
            k: v for k, v in _floats(trainer.callback_metrics).items() if k.startswith("val_")
        }
        if self._batches:
            metrics["train_loss"] = self._loss_sum / self._batches
        elapsed = time.time() - self._epoch_start
        metrics["epoch_time_s"] = round(elapsed, 3)
        metrics["samples_per_s"] = round(self._samples / max(elapsed, 1e-6), 2)
        metrics["lr"] = float(trainer.optimizers[0].param_groups[0]["lr"])
        self.history.append({"epoch": float(trainer.current_epoch), **metrics})
        done = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or done
        eta = metrics["epoch_time_s"] * max(max_epochs - done, 0)
        self.emitter.emit(
            "epoch",
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            metrics=metrics,
            data={"max_epochs": max_epochs, "eta_s": round(eta, 1), "resources": resource_usage()},
        )


class ControlCallback(L.Callback):
    """Órdenes del Engine por stdin (`{"cmd": "stop" | "pause"}`): corte al terminar el batch."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.command: str | None = None
        self._stream = stream or sys.stdin
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._stream:
            try:
                cmd = json.loads(line).get("cmd")
            except (json.JSONDecodeError, AttributeError):
                continue
            if cmd in ("stop", "pause"):
                self.command = cmd

    def on_train_batch_end(self, trainer: L.Trainer, *args: Any) -> None:
        if self.command:
            trainer.should_stop = True


class FreezeBackboneCallback(L.Callback):
    """Congela el backbone las primeras N épocas y luego lo descongela (RF-TRN-09): todo junto
    o, con `progressive`, un grupo de capas más por época empezando por las de salida."""

    def __init__(self, epochs: int, *, progressive: bool = False) -> None:
        self.epochs = epochs
        self.progressive = progressive

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        set_trainable = getattr(pl_module, "set_backbone_trainable", None)
        if set_trainable is None:
            return
        epoch = trainer.current_epoch
        if epoch < self.epochs:
            set_trainable(False)
        elif self.progressive:
            set_trainable(True, top_groups=epoch - self.epochs + 1)
        else:
            set_trainable(True)
