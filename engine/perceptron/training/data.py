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
    df = view.read(split, purpose=purpose)
    if view.modality is Modality.TABULAR:
        return TabularDataset(fitted, df)
    if view.modality is Modality.IMAGE:
        return ImageDataset(fitted, df, view.files_dir, train=train)
    raise NotImplementedError(f"datasets para {view.modality} llegan en Capa 1b")


SMALL_DATASET = 2_000


def auto_num_workers(modality: Modality, n_train: int) -> int:
    """Tabular en memoria o datasets chicos → 0 (arrancar workers cuesta más que leer).

    Con más datos se usan pocos workers; siempre con `spawn` (ver `make_loader`).
    """
    if modality is Modality.TABULAR or n_train < SMALL_DATASET:
        return 0
    cpus = os.cpu_count() or 1
    return 0 if sys.platform == "win32" and cpus <= 4 else min(4, max(cpus - 1, 0))


def auto_batch_size(modality: Modality, n_train: int) -> int:
    if modality is Modality.TABULAR:
        size = 256 if n_train >= 20_000 else 128 if n_train >= 2_000 else 32
    else:
        size = 32
    return max(2, min(size, n_train))


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
