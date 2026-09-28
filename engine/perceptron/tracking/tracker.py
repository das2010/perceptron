"""Tracking de experimentos (RF-TRK-01).

`Tracker` abstrae el backend: `MlflowTracker` (local: SQLite + archivos en
`<workspace>/mlflow`; servidor: PostgreSQL + S3 en Capa 5) y `MemoryTracker`
para tests. `RunRecorder` traduce los eventos del worker en llamadas al tracker
mientras el run avanza.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from perceptron.training.config import (
    ARCHSPEC_FILE,
    PIPELINE_FILE,
    RESULT_FILE,
    RUN_CONFIG_FILE,
    RunConfig,
    RunEvent,
    RunResult,
)

MAX_PARAM_LEN = 500

_STATUS = {
    "succeeded": "FINISHED",
    "failed": "FAILED",
    "cancelled": "KILLED",
    "pruned": "KILLED",
    "paused": "KILLED",
}


class Tracker(Protocol):
    def start_run(self, experiment: str, name: str, tags: dict[str, str]) -> str: ...
    def log_params(self, run_id: str, params: dict[str, Any]) -> None: ...
    def log_metrics(
        self, run_id: str, metrics: dict[str, float], step: int | None = None
    ) -> None: ...
    def set_tags(self, run_id: str, tags: dict[str, str]) -> None: ...
    def log_artifact(self, run_id: str, path: Path, artifact_path: str | None = None) -> None: ...
    def end_run(self, run_id: str, status: str) -> None: ...


def _param_str(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s[:MAX_PARAM_LEN]


# ------------------------------------------------------------------ memoria


@dataclass
class MemoryRun:
    experiment: str
    name: str
    tags: dict[str, str]
    params: dict[str, str] = field(default_factory=dict)
    metrics: list[tuple[str, float, int | None]] = field(default_factory=list)
    artifacts: list[tuple[Path, str | None]] = field(default_factory=list)
    status: str | None = None


class MemoryTracker:
    def __init__(self) -> None:
        self.runs: dict[str, MemoryRun] = {}

    def start_run(self, experiment: str, name: str, tags: dict[str, str]) -> str:
        rid = f"mem-{len(self.runs) + 1}"
        self.runs[rid] = MemoryRun(experiment, name, dict(tags))
        return rid

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        self.runs[run_id].params.update({k: _param_str(v) for k, v in params.items()})

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None = None) -> None:
        self.runs[run_id].metrics += [(k, v, step) for k, v in metrics.items()]

    def set_tags(self, run_id: str, tags: dict[str, str]) -> None:
        self.runs[run_id].tags.update(tags)

    def log_artifact(self, run_id: str, path: Path, artifact_path: str | None = None) -> None:
        self.runs[run_id].artifacts.append((path, artifact_path))

    def end_run(self, run_id: str, status: str) -> None:
        self.runs[run_id].status = status


# ------------------------------------------------------------------ MLflow


class MlflowTracker:
    """Cliente MLflow: SQLite y artefactos en disco (desktop) o un MLflow server (Team Server)."""

    def __init__(self, root: Path, tracking_uri: str | None = None) -> None:
        from mlflow.tracking import MlflowClient

        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.artifacts_root = root / "artifacts"
        self.artifacts_root.mkdir(exist_ok=True)
        self.tracking_uri = tracking_uri or f"sqlite:///{(root / 'mlflow.db').resolve().as_posix()}"
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.remote = self.tracking_uri.startswith(("http://", "https://"))

    def experiment_id(self, name: str) -> str:
        exp = self.client.get_experiment_by_name(name)
        if exp is not None:
            return str(exp.experiment_id)
        if self.remote:  # MLflow server (Team Server): artefactos donde lo configure el server
            return str(self.client.create_experiment(name))
        location = (self.artifacts_root / name).resolve().as_uri()
        return str(self.client.create_experiment(name, artifact_location=location))

    def start_run(self, experiment: str, name: str, tags: dict[str, str]) -> str:
        run = self.client.create_run(self.experiment_id(experiment), run_name=name, tags=tags)
        return str(run.info.run_id)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        from mlflow.entities import Param

        items = [Param(k, _param_str(v)) for k, v in params.items()]
        for i in range(0, len(items), 100):
            self.client.log_batch(run_id, params=items[i : i + 100])

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None = None) -> None:
        from mlflow.entities import Metric

        ts = int(time.time() * 1000)
        items = [Metric(k, float(v), ts, step or 0) for k, v in metrics.items()]
        if items:
            self.client.log_batch(run_id, metrics=items)

    def set_tags(self, run_id: str, tags: dict[str, str]) -> None:
        for k, v in tags.items():
            self.client.set_tag(run_id, k, v)

    def log_artifact(self, run_id: str, path: Path, artifact_path: str | None = None) -> None:
        self.client.log_artifact(run_id, str(path), artifact_path)

    def end_run(self, run_id: str, status: str) -> None:
        self.client.set_terminated(run_id, status=status)


# ------------------------------------------------------------------ puente eventos → tracker


def run_params(cfg: RunConfig) -> dict[str, Any]:
    from perceptron.archspec.schema import ArchSpec

    spec = ArchSpec.model_validate(cfg.archspec)
    params: dict[str, Any] = {f"hp.{k}": v for k, v in spec.hyperparameters().items()}
    params.update({f"hp.{k}": v for k, v in cfg.overrides.items()})
    params.update(
        {
            "archspec.name": spec.name,
            "archspec.hash": spec.content_hash(),
            "archspec.template": spec.provenance.template,
            "device": cfg.device.value,
            "seed": cfg.seed,
            "max_epochs": cfg.max_epochs,
            "batch_size": cfg.batch_size,
        }
    )
    return params


class RunRecorder:
    """Callback `on_event` para `RunHandle.wait`: registra el run en vivo."""

    def __init__(
        self,
        tracker: Tracker,
        cfg: RunConfig,
        *,
        experiment: str,
        tags: dict[str, str] | None = None,
        log_checkpoint: bool = True,
    ) -> None:
        self.tracker = tracker
        self.cfg = cfg
        self.experiment = experiment
        self.tags = {"perceptron.run_id": cfg.run_id, **(tags or {})}
        self.log_checkpoint = log_checkpoint
        self.tracking_id: str | None = None

    def __call__(self, ev: RunEvent) -> None:
        if ev.event == "started":
            self.tracking_id = self.tracker.start_run(self.experiment, self.cfg.run_id, self.tags)
            self.tracker.log_params(self.tracking_id, run_params(self.cfg))
            started = {k: v for k, v in ev.data.items() if k not in ("environment",)}
            self.tracker.log_params(self.tracking_id, {f"run.{k}": v for k, v in started.items()})
            env = ev.data.get("environment", {})
            self.tracker.set_tags(self.tracking_id, {f"env.{k}": str(v) for k, v in env.items()})
        elif ev.event == "epoch" and self.tracking_id:
            self.tracker.log_metrics(self.tracking_id, ev.metrics, step=ev.epoch)

    def finish(self, result: RunResult) -> str | None:
        """Artefactos y estado final. Devuelve el id de tracking (o None si nunca arrancó)."""
        if self.tracking_id is None:
            self.tracking_id = self.tracker.start_run(self.experiment, self.cfg.run_id, self.tags)
            self.tracker.log_params(self.tracking_id, run_params(self.cfg))
        rid = self.tracking_id
        if result.best_metrics:
            self.tracker.log_metrics(rid, {f"best.{k}": v for k, v in result.best_metrics.items()})
        for name in (ARCHSPEC_FILE, PIPELINE_FILE, RUN_CONFIG_FILE, RESULT_FILE):
            path = self.cfg.run_dir / name
            if path.is_file():
                self.tracker.log_artifact(rid, path, "run")
        if self.log_checkpoint and result.best_checkpoint and result.best_checkpoint.is_file():
            self.tracker.log_artifact(rid, result.best_checkpoint, "checkpoints")
        if result.error:
            self.tracker.set_tags(rid, {"error.code": str(result.error.get("code"))})
        self.tracker.end_run(rid, _STATUS.get(result.status, "FAILED"))
        return rid
