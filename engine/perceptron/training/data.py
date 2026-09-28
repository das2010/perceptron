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
from torch.utils.data import DataLoader, Dataset, IterableDataset, WeightedRandomSampler

from perceptron.data.pipeline.pipeline import (
    FittedPipeline,
    ImageSpec,
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


# Tabulares más grandes que esto (en memoria como tensores) se leen por lotes (RF-ING-09).
STREAMING_TABULAR_BYTES = int(
    float(os.environ.get("PERCEPTRON_STREAMING_THRESHOLD_MB", "2048")) * 2**20
)
STREAM_BATCH_ROWS = 65_536
STREAM_SHUFFLE_BATCHES = 4


class StreamingTabularDataset(IterableDataset[tuple[torch.Tensor, ...]]):
    """Tabular grande: lee el Parquet del split por lotes (pyarrow), transforma cada lote con
    el pipeline y mezcla con un buffer de varios lotes. Nunca tiene el split entero en memoria."""

    def __init__(
        self, fitted: FittedPipeline, data_file: Path, split: str, *, shuffle: bool, seed: int
    ) -> None:
        import pyarrow.compute as pc  # type: ignore[import-untyped]
        import pyarrow.dataset as pads  # type: ignore[import-untyped]

        from perceptron.data.splits import SPLIT_COLUMN

        self.fitted = fitted
        self.data_file = data_file
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._filter = pc.field(SPLIT_COLUMN) == split
        dataset = pads.dataset(str(data_file), format="parquet")
        self.n = int(dataset.count_rows(filter=self._filter))
        target = fitted.spec.target
        self.class_counts: torch.Tensor | None = None
        if target and fitted.classes and target.task is not TaskType.REGRESSION:
            col = dataset.to_table(columns=[target.name], filter=self._filter)[target.name]
            values = pl.from_arrow(col)
            series = values if isinstance(values, pl.Series) else values.to_series()
            y = encode_target(fitted, series)
            k = len(fitted.classes)
            yt = torch.tensor(y)
            self.class_counts = torch.bincount(yt[yt >= 0], minlength=k)

    def __len__(self) -> int:
        return self.n

    def _batches(self) -> Any:
        import pyarrow.dataset as pads

        dataset = pads.dataset(str(self.data_file), format="parquet")
        yield from dataset.to_batches(filter=self._filter, batch_size=STREAM_BATCH_ROWS)

    def _emit(self, frames: list[pl.DataFrame], gen: torch.Generator) -> Any:
        df = pl.concat(frames, how="vertical_relaxed")
        arr = transform_tabular(self.fitted, df)
        x_num, x_cat = torch.tensor(arr.x_num), torch.tensor(arr.x_cat)
        y = torch.tensor(arr.y if arr.y is not None else np.zeros(df.height, dtype=np.int64))
        order = torch.randperm(len(y), generator=gen) if self.shuffle else torch.arange(len(y))
        for i in order.tolist():
            yield x_num[i], x_cat[i], y[i]

    def __iter__(self) -> Any:
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        buffer: list[pl.DataFrame] = []
        for batch in self._batches():
            buffer.append(pl.from_arrow(batch))  # type: ignore[arg-type]
            if len(buffer) >= (STREAM_SHUFFLE_BATCHES if self.shuffle else 1):
                yield from self._emit(buffer, gen)
                buffer = []
        if buffer:
            yield from self._emit(buffer, gen)


def _tabular_bytes(view: DatasetView, split: str, fitted: FittedPipeline) -> int:
    rows = int(view.scan(split).select(pl.len()).collect().item())
    width = len(fitted.numeric_features) + len(fitted.categorical_features) + 1
    return rows * width * 8


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
        # Normalización por ventana (RevIN): se predice la desviación respecto del último
        # valor observado; el modelo parte del pronóstico naive y aprende la corrección.
        self.level = torch.tensor(
            w.level if w.level is not None else np.zeros(len(w.y), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        level = self.level[i]
        x = self.x[i].clone()
        x[:, 0] -= level
        if self.jitter:
            x = x + self.jitter * torch.randn_like(x)
        y = self.y[i] - level
        return x, y, self.mean[i], self.std[i], self.naive[i], self.scale[i], level


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


class AudioDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Features de audio calculadas al vuelo con caché en memoria por archivo."""

    def __init__(
        self, fitted: FittedPipeline, df: pl.DataFrame, files_dir: Path, *, train: bool
    ) -> None:
        df = df.filter(~pl.col("corrupt")) if "corrupt" in df.columns else df
        self.paths = [files_dir / p for p in df["path"].to_list()]
        spec = fitted.spec.audio
        if spec is None:
            raise ValueError("pipeline de audio sin `audio`")
        self.spec = spec
        self.train = train
        self.mean = fitted.audio_mean or 0.0
        self.std = fitted.audio_std or 1.0
        target = fitted.spec.target
        if target and target.name in df.columns:
            self.y = torch.tensor(encode_target(fitted, df[target.name]))
        else:
            self.y = torch.zeros(len(self.paths), dtype=torch.long)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def _wave(self, i: int) -> np.ndarray:
        from perceptron.data.audio import load

        if i not in self._cache:
            self._cache[i], _ = load(self.paths[i], sample_rate=self.spec.sample_rate)
        return self._cache[i]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        from perceptron.data.audio import spec_augment
        from perceptron.data.pipeline.pipeline import audio_features

        wav = self._wave(i)
        if self.train and self.spec.time_shift:
            shift = int(np.random.uniform(-1, 1) * self.spec.time_shift * wav.shape[-1])
            wav = np.roll(wav, shift, axis=-1)
        if self.train and self.spec.noise:
            wav = wav + np.random.normal(0, self.spec.noise, wav.shape).astype(np.float32)
        feats = (audio_features(self.spec, wav) - self.mean) / self.std
        if self.train and (self.spec.freq_mask or self.spec.time_mask):
            feats = spec_augment(feats, self.spec.freq_mask, self.spec.time_mask)
        return feats.float(), self.y[i]


def _to_tensor(img: Any, fitted: FittedPipeline, size: tuple[int, int], mode: str) -> torch.Tensor:
    im = img.convert(mode).resize((size[1], size[0]))
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    x = x.unsqueeze(0) if x.ndim == 2 else x.permute(2, 0, 1)
    c = x.shape[0]
    mean = torch.tensor((fitted.image_mean or [0.5] * c)[:c]).view(c, 1, 1)
    std = torch.tensor((fitted.image_std or [0.25] * c)[:c]).view(c, 1, 1)
    return (x - mean) / std


class DetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Imagen redimensionada a S×S + cajas xyxy escaladas; espejado horizontal en train."""

    def __init__(
        self, fitted: FittedPipeline, df: pl.DataFrame, files_dir: Path, *, train: bool
    ) -> None:
        from perceptron.tasks.vision import detection_collate

        df = df.filter(~pl.col("corrupt")) if "corrupt" in df.columns else df
        self.rows = df.select("path", "width", "height", "boxes", "box_labels").to_dicts()
        self.files_dir = files_dir
        self.size = (fitted.spec.image or ImageSpec()).size
        self.index = {c: i for i, c in enumerate(fitted.classes or [])}
        self.fitted = fitted
        self.train = train
        self.collate_fn = detection_collate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        r = self.rows[i]
        with Image.open(self.files_dir / r["path"]) as im:
            x = _to_tensor(im, self.fitted, (self.size, self.size), "RGB")
        sx, sy = self.size / max(r["width"] or 1, 1), self.size / max(r["height"] or 1, 1)
        boxes = torch.tensor(
            [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in r["boxes"] or []],
            dtype=torch.float32,
        ).reshape(-1, 4)
        labels = torch.tensor(
            [self.index.get(n, 0) for n in r["box_labels"] or []], dtype=torch.long
        )
        if self.train and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
            boxes = (
                torch.stack(
                    [self.size - boxes[:, 2], boxes[:, 1], self.size - boxes[:, 0], boxes[:, 3]], 1
                )
                if len(boxes)
                else boxes
            )
        return x, {"boxes": boxes, "labels": labels}


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, fitted: FittedPipeline, df: pl.DataFrame, files_dir: Path, *, train: bool
    ) -> None:
        df = df.filter(~pl.col("corrupt")) if "corrupt" in df.columns else df
        self.rows = df.select("path", "mask_path").rows()
        self.files_dir = files_dir
        self.size = (fitted.spec.image or ImageSpec()).size
        self.fitted = fitted
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.rows[i]
        with Image.open(self.files_dir / img_path) as im:
            x = _to_tensor(im, self.fitted, (self.size, self.size), "RGB")
        with Image.open(self.files_dir / mask_path) as m:
            arr = np.asarray(
                m.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
            ).astype(np.int64)
        mask = torch.from_numpy(np.where(arr == 255, 1, arr))
        if self.train and torch.rand(1).item() < 0.5:
            x, mask = x.flip(-1), mask.flip(-1)
        return x, mask


class OCRDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    MAX_LEN = 64

    def __init__(
        self, fitted: FittedPipeline, df: pl.DataFrame, files_dir: Path, *, train: bool
    ) -> None:
        df = df.filter(~pl.col("corrupt")) if "corrupt" in df.columns else df
        self.rows = df.select("path", "text").rows()
        self.files_dir = files_dir
        img = fitted.spec.image or ImageSpec(size=32)
        self.shape = (img.size, img.width or img.size * 4)
        self.index = {c: i + 1 for i, c in enumerate(fitted.classes or [])}  # 0 = blank
        self.fitted = fitted

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, text = self.rows[i]
        with Image.open(self.files_dir / path) as im:
            x = _to_tensor(im, self.fitted, self.shape, "L")
        ids = [self.index[c] for c in (text or "") if c in self.index][: self.MAX_LEN]
        padded = torch.zeros(self.MAX_LEN, dtype=torch.long)
        padded[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return x, padded, torch.tensor(len(ids), dtype=torch.long)


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


_VISION_DATASETS: dict[TaskType, Any] = {
    TaskType.OBJECT_DETECTION: DetectionDataset,
    TaskType.SEGMENTATION: SegmentationDataset,
    TaskType.OCR: OCRDataset,
}


def make_dataset_from_frame(
    view: DatasetView, fitted: FittedPipeline, df: pl.DataFrame
) -> Dataset[Any]:
    """Dataset de evaluación sobre filas arbitrarias de la versión (p. ej. el pre-etiquetado)."""
    if view.modality is Modality.TABULAR:
        return TabularDataset(fitted, df)
    if view.modality is Modality.IMAGE:
        return ImageDataset(fitted, df, view.files_dir, train=False)
    if view.modality is Modality.TEXT:
        return TextDataset(fitted, df)
    if view.modality is Modality.AUDIO:
        return AudioDataset(fitted, df, view.files_dir, train=False)
    raise NotImplementedError(f"pre-etiquetado para {view.modality} no disponible")


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
    if (
        view.modality is Modality.TABULAR
        and purpose is Purpose.TRAINING
        and _tabular_bytes(view, split, fitted) > STREAMING_TABULAR_BYTES
    ):
        return StreamingTabularDataset(fitted, view.data_file, split, shuffle=train, seed=42)
    df = view.read(split, purpose=purpose)
    if view.modality is Modality.TABULAR:
        return TabularDataset(fitted, df)
    task = fitted.spec.target.task if fitted.spec.target else None
    if view.modality is Modality.IMAGE and task in _VISION_DATASETS:
        vision_ds: Dataset[Any] = _VISION_DATASETS[task](fitted, df, view.files_dir, train=train)
        return vision_ds
    if view.modality is Modality.IMAGE:
        return ImageDataset(fitted, df, view.files_dir, train=train)
    if view.modality is Modality.TEXT:
        return TextDataset(fitted, df)
    if view.modality is Modality.AUDIO:
        return AudioDataset(fitted, df, view.files_dir, train=train)
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
    """Batch de partida (en GPU, `tuning.tune_batch_size` busca el mayor que entra).

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
    ds: Dataset[Any],
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    seed: int,
    oversample: bool = False,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    if isinstance(ds, IterableDataset):
        # Streaming (RF-ING-09): el dataset mezcla solo; sin sampler ni workers extra.
        return DataLoader(
            ds, batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available()
        )
    sampler = balanced_sampler(ds, generator) if oversample and shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
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


def balanced_sampler(ds: Dataset[Any], generator: torch.Generator) -> WeightedRandomSampler | None:
    """Sobremuestreo: cada clase se ve con la misma frecuencia esperada (RF-TRN-10)."""
    y = getattr(ds, "y", None)
    if y is None or y.dtype not in (torch.long, torch.int64) or len(y) == 0:
        return None
    counts = torch.bincount(y[y >= 0])
    weights = 1.0 / counts.clamp(min=1).float()[y.clamp(min=0)]
    return WeightedRandomSampler(
        weights.tolist(), num_samples=len(y), replacement=True, generator=generator
    )


def class_weights(fitted: FittedPipeline, ds: Dataset[Any]) -> torch.Tensor | None:
    """Pesos inversos a la frecuencia de cada clase en train (`class_weights: auto`)."""
    target = fitted.spec.target
    if not target or target.task is TaskType.REGRESSION or not fitted.classes:
        return None
    k = len(fitted.classes)
    streamed = getattr(ds, "class_counts", None)
    if streamed is not None:
        counts: torch.Tensor = streamed.float().clamp(min=1)
        weights: torch.Tensor = counts.sum() / (k * counts)
        return weights
    y = ds.y if hasattr(ds, "y") else None
    if y is None:
        return None
    counts = torch.bincount(y[y >= 0], minlength=k).float().clamp(min=1)
    return counts.sum() / (k * counts)
