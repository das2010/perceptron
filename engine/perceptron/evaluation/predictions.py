"""Predicciones por muestra de la evaluación final (base del análisis de errores y fairness).

`evaluate_run` las guarda en `<run>/evaluation/predictions.parquet`, alineadas por posición
con las filas del split evaluado (`eval_frame`), para clasificación y regresión.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from perceptron.core.errors import ValidationError
from perceptron.data.pipeline.pipeline import FittedPipeline
from perceptron.data.view import DatasetView, Purpose
from perceptron.domain.enums import TaskType
from perceptron.tasks import Predictions

PREDICTIONS_FILE = "predictions.parquet"
ROW = "__row__"


def eval_frame(view: DatasetView, split: str = "test") -> pl.DataFrame:
    """Filas del split en el mismo orden (y con el mismo filtro) que el dataset evaluado."""
    df = view.read(split, purpose=Purpose.FINAL_EVALUATION)
    if "corrupt" in df.columns:
        df = df.filter(~pl.col("corrupt"))
    return df.with_row_index(ROW)


def save_predictions(
    out_dir: Path, preds: Predictions, pipeline: FittedPipeline, task: TaskType
) -> Path | None:
    if task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION) or preds.y_true is None:
        return None
    y_pred = np.asarray(preds.y_pred)
    if y_pred.ndim != 1:
        return None
    cols: dict[str, object] = {ROW: np.arange(len(y_pred), dtype=np.int64)}
    if task is TaskType.REGRESSION:  # el adaptador ya devuelve la escala original
        cols["y_true"] = np.asarray(preds.y_true, dtype=np.float64)
        cols["y_pred"] = y_pred.astype(np.float64)
    else:
        classes = pipeline.classes or [str(i) for i in range(int(y_pred.max()) + 1)]
        cols["y_true"] = [classes[int(i)] if int(i) >= 0 else None for i in preds.y_true]
        cols["y_pred"] = [classes[int(i)] for i in y_pred]
        if preds.proba is not None:
            proba = np.asarray(preds.proba, dtype=np.float64)
            cols["confidence"] = proba.max(axis=1)
            for k, c in enumerate(classes[: proba.shape[1]]):
                cols[f"p::{c}"] = proba[:, k]
    path = out_dir / PREDICTIONS_FILE
    pl.DataFrame(cols).write_parquet(path)
    return path


def load_predictions(eval_dir: Path) -> pl.DataFrame:
    path = eval_dir / PREDICTIONS_FILE
    if not path.is_file():
        raise ValidationError("evaluá el run en el test sellado primero (o re-evaluálo)")
    return pl.read_parquet(path)


def proba_columns(df: pl.DataFrame) -> dict[str, str]:
    """Clase → columna de probabilidad."""
    return {c.removeprefix("p::"): c for c in df.columns if c.startswith("p::")}
