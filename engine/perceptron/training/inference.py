"""Carga de un modelo entrenado desde su directorio de run y predicción por lotes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from perceptron.archspec.builder import build_model
from perceptron.archspec.schema import ArchSpec
from perceptron.core.errors import NotFoundError
from perceptron.data.pipeline.pipeline import FittedPipeline, decode_regression
from perceptron.domain.enums import TaskType
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


@dataclass
class Predictions:
    y_true: np.ndarray | None
    y_pred: np.ndarray  # clase (índice) o valor en la escala original
    proba: np.ndarray | None  # [n, k] para clasificación


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


@torch.no_grad()
def predict(
    trained: TrainedModel, ds: Dataset[tuple[torch.Tensor, ...]], batch_size: int = 256
) -> Predictions:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs, ys = [], []
    for batch in loader:
        *inputs, y = batch
        outs.append(trained.model(*inputs))
        ys.append(y)
    out = torch.cat(outs) if outs else torch.zeros(0, 1)
    y_true = torch.cat(ys).numpy() if ys else None
    if trained.task is TaskType.REGRESSION:
        pred = decode_regression(trained.pipeline, out.squeeze(-1).numpy())
        if y_true is not None:
            y_true = decode_regression(trained.pipeline, y_true)
        return Predictions(y_true=y_true, y_pred=pred, proba=None)
    if trained.spec.loss.type == "bce":
        p1 = torch.sigmoid(out.squeeze(-1))
        proba = torch.stack([1 - p1, p1], dim=1)
    else:
        proba = out.softmax(-1)
    return Predictions(y_true=y_true, y_pred=proba.argmax(-1).numpy(), proba=proba.numpy())
