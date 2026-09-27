"""Carga de un modelo entrenado desde su directorio de run y predicción por lotes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from perceptron.archspec.builder import build_model
from perceptron.archspec.schema import ArchSpec
from perceptron.core.errors import NotFoundError
from perceptron.data.pipeline.pipeline import FittedPipeline
from perceptron.domain.enums import TaskType
from perceptron.tasks import Predictions, get_adapter
from perceptron.training.config import (
    ARCHSPEC_FILE,
    BEST_CKPT,
    CHECKPOINTS_DIR,
    LAST_CKPT,
    PIPELINE_FILE,
    RUN_CONFIG_FILE,
    RunConfig,
)


@dataclass
class TrainedModel:
    spec: ArchSpec
    pipeline: FittedPipeline
    model: torch.nn.Module
    checkpoint: Path

    @property
    def task(self) -> TaskType:
        return self.spec.task.type


def load_trained(run_dir: Path, *, prefer: str = BEST_CKPT) -> TrainedModel:
    ckpt_dir = run_dir / CHECKPOINTS_DIR
    ckpt = ckpt_dir / prefer
    if not ckpt.is_file():
        ckpt = ckpt_dir / LAST_CKPT
    if not ckpt.is_file():
        raise NotFoundError(f"el run no tiene checkpoints: {run_dir}")
    spec = ArchSpec.model_validate_json((run_dir / ARCHSPEC_FILE).read_text(encoding="utf-8"))
    pipeline = FittedPipeline.model_validate_json(
        (run_dir / PIPELINE_FILE).read_text(encoding="utf-8")
    )
    cfg = RunConfig.model_validate_json((run_dir / RUN_CONFIG_FILE).read_text(encoding="utf-8"))
    model = build_model(spec, cfg.overrides, pretrained_allowed=False).model
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    model_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(model_state)
    model.eval()
    return TrainedModel(spec=spec, pipeline=pipeline, model=model, checkpoint=ckpt)


def predict(trained: TrainedModel, ds: Dataset[Any], batch_size: int = 256) -> Predictions:
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=getattr(ds, "collate_fn", None)
    )
    with torch.no_grad():
        return get_adapter(trained.task).predict(
            trained.model, loader, trained.spec, trained.pipeline
        )
