"""Configuración de un run y protocolo de eventos worker → Engine (ADR-0015)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from perceptron.archspec.schema import Scalar
from perceptron.domain.enums import Device

RUN_CONFIG_FILE = "run.json"
RESULT_FILE = "result.json"
ARCHSPEC_FILE = "archspec.json"
PIPELINE_FILE = "pipeline.json"
WORKER_LOG = "worker.log"
CHECKPOINTS_DIR = "checkpoints"
BEST_CKPT = "best.ckpt"
LAST_CKPT = "last.ckpt"


class RunConfig(BaseModel):
    """Todo lo que el worker necesita; se serializa a `<run_dir>/run.json`."""

    run_id: str
    run_dir: Path
    dataset_dir: Path
    archspec: dict[str, Any]
    pipeline: dict[str, Any] = Field(description="FittedPipeline serializado")
    overrides: dict[str, Scalar] = Field(default_factory=dict)
    device: Device = Device.CPU
    seed: int = 42
    max_epochs: int | None = Field(default=None, ge=1, description="Override de training.epochs")
    batch_size: int | Literal["auto"] | None = None
    num_workers: int | Literal["auto"] = "auto"
    max_time_s: float | None = Field(default=None, gt=0)
    resume_from: Path | None = None
    deterministic: bool = True
    pretrained_allowed: bool = True
    limit_train_batches: float | int | None = None
    emit_every_n_batches: int = Field(default=20, ge=1)

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / RUN_CONFIG_FILE
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


EventType = Literal["started", "epoch", "batch", "checkpoint", "paused", "finished", "error", "log"]


class RunEvent(BaseModel):
    event: EventType
    run_id: str
    ts: float
    epoch: int | None = None
    step: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


RunStatusName = Literal["succeeded", "failed", "cancelled", "paused", "pruned"]


class RunResult(BaseModel):
    run_id: str
    status: RunStatusName
    epochs: int = 0
    best_metrics: dict[str, float] = Field(default_factory=dict)
    last_metrics: dict[str, float] = Field(default_factory=dict)
    monitor: str | None = None
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    duration_s: float = 0.0
    environment: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    history: list[dict[str, float]] = Field(default_factory=list, description="Métricas por época")
