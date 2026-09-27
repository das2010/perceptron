"""Datasets y DataLoaders a partir de una DatasetVersion + FittedPipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from perceptron.data.pipeline.pipeline import (
    FittedPipeline,
    encode_target,
    image_transforms,
    transform_tabular,
    transform_text,
)
from perceptron.data.view import DatasetView, Purpose
from perceptron.domain.enums import Modality, TaskType


class TabularDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, fitted: FittedPipeline, df: pl.DataFrame) -> None:
        arr = transform_tabular(fitted, df)
        self.x_num = torch.tensor(arr.x_num)  # copia: los arrays de Polars son de solo lectura
        self.x_cat = torch.tensor(arr.x_cat)
        y = arr.y if arr.y is not None else np.zeros(df.height, dtype=np.int64)
        self.y = torch.tensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        return self.x_num[i], self.x_cat[i], self.y[i]


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, fitted: FittedPipeline, df: pl.DataFrame) -> None:
        self.ids = torch.tensor(transform_text(fitted, df))
        target = fitted.spec.target
        if target and target.name in df.columns:
            self.y = torch.tensor(encode_target(fitted, df[target.name]))
        else:
            self.y = torch.zeros(len(self.ids), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ids[i], self.y[i]


class SeriesForecastDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Ventanas lookback → horizonte cuyo horizonte cae en `split`."""

    def __init__(
        self, fitted: FittedPipeline, df: pl.DataFrame, split: str, *, train: bool
    ) -> None:
        from perceptron.data.pipeline.series_windows import forecast_windows

        sspec = fitted.spec.series
        if sspec is None or fitted.series_state is None:
            raise ValueError("pipeline de series sin ajustar")
        w = forecast_windows(sspec.config, fitted.series_state, df, split, calendar=sspec.calendar)
        self.x = torch.tensor(w.x)
        self.y = torch.tensor(w.y)
        self.mean = torch.tensor(w.mean)
        self.std = torch.tensor(w.std)
        self.naive = torch.tensor(w.naive)
        self.scale = torch.tensor(w.mase_scale)
        self.series = w.series
        self.jitter = sspec.jitter if train else 0.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        x = self.x[i]
        if self.jitter:
            x = x + self.jitter * torch.randn_like(x)
        return x, self.y[i], self.mean[i], self.std[i], self.naive[i], self.scale[i]


class SeriesAnomalyDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Una ventana por punto; para entrenar se usan solo ventanas sin anomalías."""

    def __init__(
        self, fitted: FittedPipeline, df: pl.DataFrame, split: str, *, normal_only: bool
    ) -> None:
        from perceptron.data.pipeline.series_windows import anomaly_windows

        sspec = fitted.spec.series
        if sspec is None or fitted.series_state is None:
            raise ValueError("pipeline de series sin ajustar")
        w = anomaly_windows(sspec.config, fitted.series_state, df, split, calendar=sspec.calendar)
        keep = w.normal if normal_only else np.ones(len(w.label), dtype=bool)
        self.x = torch.tensor(w.x[keep])
        self.label = torch.tensor(w.label[keep])
        self.normal = torch.tensor(w.normal[keep])
        self.series = [s for s, k in zip(w.series, keep, strict=True) if k]

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        return self.x[i], self.label[i], self.normal[i]


class ImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lee las imágenes desde disco en cada acceso (no carga el dataset en memoria)."""

    def __init__(
        self, fitted: FittedPipeline, df: pl.DataFrame, files_dir: Path, *, train: bool
    ) -> None:
        df = df.filter(~pl.col("corrupt")) if "corrupt" in df.columns else df
        self.paths = [files_dir / p for p in df["path"].to_list()]
        target = fitted.spec.target
        if target and target.name in df.columns:
            self.y = torch.tensor(encode_target(fitted, df[target.name]))
        else:
            self.y = torch.zeros(len(self.paths), dtype=torch.long)
        self.transform = image_transforms(fitted, train=train)
        img = fitted.spec.image
        self.mode = "L" if img and img.channels == 1 else "RGB"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert(self.mode))
        return x, self.y[i]


def make_dataset(
    view: DatasetView,
    fitted: FittedPipeline,
    split: str,
    *,
    train: bool,
    purpose: Purpose = Purpose.TRAINING,
) -> Dataset[Any]:
    if view.modality is Modality.TIMESERIES:
        # Las ventanas necesitan la historia previa al split: se lee todo lo permitido.
        full = view.read(None, purpose=purpose)
        target = fitted.spec.target
        if target and target.task is TaskType.ANOMALY_DETECTION:
            return SeriesAnomalyDataset(
                fitted, full, split, normal_only=purpose is Purpose.TRAINING
            )
        return SeriesForecastDataset(fitted, full, split, train=train)
    df = view.read(split, purpose=purpose)
    if view.modality is Modality.TABULAR:
        return TabularDataset(fitted, df)
    if view.modality is Modality.IMAGE:
        return ImageDataset(fitted, df, view.files_dir, train=train)
    if view.modality is Modality.TEXT:
        return TextDataset(fitted, df)
    raise NotImplementedError(f"datasets para {view.modality} llegan en Capa 1b")


SMALL_DATASET = 2_000


def auto_num_workers(modality: Modality, n_train: int) -> int:
    """Tabular en memoria o datasets chicos → 0 (arrancar workers cuesta más que leer).

    Con más datos se usan pocos workers; siempre con `spawn` (ver `make_loader`).
    """
    if (
        modality in (Modality.TABULAR, Modality.TEXT, Modality.TIMESERIES)
        or n_train < SMALL_DATASET
    ):
        return 0
    cpus = os.cpu_count() or 1
    return 0 if sys.platform == "win32" and cpus <= 4 else min(4, max(cpus - 1, 0))


def auto_batch_size(modality: Modality, n_train: int) -> int:
    """Batch por defecto en CPU (en GPU se ajusta contra OOM en la Capa 3).

    Con pocos datos conviene un batch chico: más pasos de optimización por época y
    estadísticas de BatchNorm que llegan a converger (con 1 paso por época la
    inferencia queda sesgada aunque el ranking sea perfecto).
    """
    if modality is Modality.TABULAR:
        size = 256 if n_train >= 20_000 else 128 if n_train >= 2_000 else 32
    else:
        size = 32 if n_train >= 2_000 else 16 if n_train >= 500 else 8
    return max(2, min(size, n_train // 2 or 2))


def make_loader(
    ds: Dataset[Any], batch_size: int, *, shuffle: bool, num_workers: int, seed: int
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        # `spawn` y no `fork`: el worker de entrenamiento ya tiene hilos (lector de órdenes por
        # stdin) y hacer fork de un proceso con hilos puede dejar locks tomados (deadlock).
        multiprocessing_context="spawn" if num_workers > 0 else None,
        drop_last=shuffle and len(ds) > batch_size,  # type: ignore[arg-type]
        generator=generator,
        collate_fn=getattr(ds, "collate_fn", None),
    )


def class_weights(fitted: FittedPipeline, ds: Dataset[Any]) -> torch.Tensor | None:
    """Pesos inversos a la frecuencia de cada clase en train (`class_weights: auto`)."""
    target = fitted.spec.target
    if not target or target.task is TaskType.REGRESSION or not fitted.classes:
        return None
    y = ds.y if hasattr(ds, "y") else None
    if y is None:
        return None
    k = len(fitted.classes)
    counts = torch.bincount(y[y >= 0], minlength=k).float().clamp(min=1)
    return counts.sum() / (k * counts)
