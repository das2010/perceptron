"""Pipeline de preparación serializable (RF-PIP-01, RF-PIP-04).

`PipelineSpec` es el documento declarativo (lo que se guarda en `Pipeline.graph`
y edita el usuario o, en Capa 2, sugiere el LLM). `fit_pipeline` lo ajusta
**solo con train** y devuelve un `FittedPipeline` JSON que se empaqueta con el
modelo. En 1a el DAG es lineal: una secuencia de pasos sobre columnas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from perceptron.data.pipeline.steps import OrdinalEncode, State, StepSpec, build_step
from perceptron.data.splits import FOLD_COLUMN, SPLIT_COLUMN
from perceptron.domain.enums import Modality, TaskType

PIPELINE_VERSION = "1.0"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_INTERNAL = {SPLIT_COLUMN, FOLD_COLUMN}


class TargetSpec(BaseModel):
    name: str
    task: TaskType
    standardize: bool = Field(default=False, description="Solo regresión")


class AugmentSpec(BaseModel):
    kind: Literal[
        "hflip", "vflip", "rotation", "color_jitter", "random_resized_crop", "trivial_augment"
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class ImageSpec(BaseModel):
    size: int = Field(default=224, ge=16, le=1024)
    channels: Literal[1, 3] = 3
    normalize: Literal["imagenet", "dataset"] = "imagenet"
    augment: list[AugmentSpec] = Field(default_factory=list)


class TextSpec(BaseModel):
    column: str
    lowercase: bool = True
    accents: bool = True
    urls: bool = True
    numbers: bool = False
    tokenizer: Literal["word", "hf"] = "word"
    hf_model: str | None = None
    vocab_size: int = Field(default=20_000, ge=10)
    min_freq: int = Field(default=1, ge=1)
    max_length: int = Field(default=64, ge=4, le=4096)


class PipelineSpec(BaseModel):
    pipeline_version: str = PIPELINE_VERSION
    modality: Modality
    target: TargetSpec | None
    steps: list[StepSpec] = Field(default_factory=list)
    image: ImageSpec | None = None
    text: TextSpec | None = None
    rationale: list[str] = Field(default_factory=list, description="Por qué se eligió cada paso")


class FittedPipeline(BaseModel):
    spec: PipelineSpec
    states: dict[str, State] = Field(default_factory=dict)
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    cardinalities: dict[str, int] = Field(default_factory=dict)
    classes: list[str] | None = None
    target_mean: float | None = None
    target_std: float | None = None
    image_mean: list[float] | None = None
    image_std: list[float] | None = None
    vocab: list[str] | None = None
    pad_id: int = 0

    @property
    def num_classes(self) -> int | None:
        return len(self.classes) if self.classes is not None else None


@dataclass
class TabularArrays:
    x_num: np.ndarray  # float32 [n, n_num]
    x_cat: np.ndarray  # int64 [n, n_cat]
    y: np.ndarray | None  # int64 (clasificación) o float32 (regresión)


# ------------------------------------------------------------------ target


def _fit_target(spec: TargetSpec, s: pl.Series) -> dict[str, Any]:
    if spec.task is TaskType.REGRESSION:
        x = s.cast(pl.Float64).drop_nulls()
        mean, std = float(x.mean() or 0.0), float(x.std() or 1.0)  # type: ignore[arg-type]
        return {"target_mean": mean, "target_std": std or 1.0} if spec.standardize else {}
    return {"classes": sorted(s.drop_nulls().cast(pl.String).unique().to_list())}


def encode_target(fitted: FittedPipeline, s: pl.Series) -> np.ndarray:
    target = fitted.spec.target
    if target is None:
        raise ValueError("el pipeline no tiene target")
    if target.task is TaskType.REGRESSION:
        y = s.cast(pl.Float64).to_numpy().astype(np.float32)
        if fitted.target_mean is not None and fitted.target_std:
            y = (y - fitted.target_mean) / fitted.target_std
        return y.astype(np.float32)
    mapping = {c: i for i, c in enumerate(fitted.classes or [])}
    return s.cast(pl.String).replace_strict(mapping, default=-1, return_dtype=pl.Int64).to_numpy()


def decode_regression(fitted: FittedPipeline, y: np.ndarray) -> np.ndarray:
    if fitted.target_mean is not None and fitted.target_std:
        return y * fitted.target_std + fitted.target_mean
    return y


# ------------------------------------------------------------------ fit / transform


def _apply_steps(fitted: FittedPipeline, df: pl.DataFrame) -> pl.DataFrame:
    for step_spec in fitted.spec.steps:
        df = build_step(step_spec).transform(df, fitted.states.get(step_spec.id, {}))
    return df


def fit_pipeline(spec: PipelineSpec, train: pl.DataFrame) -> FittedPipeline:
    """Ajusta el pipeline con las filas de **train** únicamente."""
    fitted = FittedPipeline(spec=spec)
    if spec.target is not None and spec.target.name in train.columns:
        extra = _fit_target(spec.target, train[spec.target.name])
        fitted = fitted.model_copy(update=extra)

    if spec.modality is Modality.IMAGE:
        return fitted
    if spec.modality is Modality.TEXT:
        return _fit_text(fitted, train)

    target_name = spec.target.name if spec.target else None
    df = train.drop([c for c in (*_INTERNAL, target_name) if c and c in train.columns])
    kinds: dict[str, str] = {}
    cardinalities: dict[str, int] = {}
    for step_spec in spec.steps:
        step = build_step(step_spec)
        state = step.fit(df)
        fitted.states[step_spec.id] = state
        df = step.transform(df, state)
        kinds.update(step.output_columns(state))
        if isinstance(step, OrdinalEncode):
            cardinalities.update(OrdinalEncode.cardinalities(state))

    categorical = [c for c in df.columns if kinds.get(c) == "categorical"]
    numeric = [c for c in df.columns if c not in categorical]
    not_numeric = [c for c in numeric if not df[c].dtype.is_numeric() and df[c].dtype != pl.Boolean]
    if not_numeric:
        raise ValueError(f"columnas sin codificar al final del pipeline: {not_numeric}")
    return fitted.model_copy(
        update={
            "numeric_features": numeric,
            "categorical_features": categorical,
            "cardinalities": cardinalities,
        }
    )


def transform_tabular(fitted: FittedPipeline, df: pl.DataFrame) -> TabularArrays:
    target = fitted.spec.target
    y = encode_target(fitted, df[target.name]) if target and target.name in df.columns else None
    x = _apply_steps(fitted, df)
    x_num = (
        x.select(pl.col(c).cast(pl.Float32).fill_null(0.0) for c in fitted.numeric_features)
        .to_numpy()
        .astype(np.float32)
        if fitted.numeric_features
        else np.zeros((df.height, 0), dtype=np.float32)
    )
    x_cat = (
        x.select(pl.col(c).cast(pl.Int64).fill_null(0) for c in fitted.categorical_features)
        .to_numpy()
        .astype(np.int64)
        if fitted.categorical_features
        else np.zeros((df.height, 0), dtype=np.int64)
    )
    return TabularArrays(x_num=x_num, x_cat=x_cat, y=y)


# ------------------------------------------------------------------ imágenes


def fit_image_stats(fitted: FittedPipeline, images: list[np.ndarray]) -> FittedPipeline:
    """Media/desvío por canal (HWC en [0, 1]) sobre una muestra de train."""
    if not images:
        return fitted
    stacked = np.concatenate([im.reshape(-1, im.shape[-1]) for im in images])
    return fitted.model_copy(
        update={
            "image_mean": [float(v) for v in stacked.mean(0)],
            "image_std": [float(v) or 1.0 for v in stacked.std(0)],
        }
    )


def image_transforms(fitted: FittedPipeline, *, train: bool) -> Any:
    """Compose de torchvision v2 (augmentations solo en train)."""
    import torch
    from torchvision.transforms import v2

    img = fitted.spec.image or ImageSpec()
    if img.normalize == "dataset" and fitted.image_mean and fitted.image_std:
        mean, std = fitted.image_mean, fitted.image_std
    else:
        mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
    if img.channels == 1:
        mean, std = mean[:1], std[:1]

    ops: list[Any] = [v2.ToImage()]
    if img.channels == 1:
        ops.append(v2.Grayscale(1))
    ops.append(v2.Resize((img.size, img.size), antialias=True))
    if train:
        for aug in img.augment:
            p = aug.params
            match aug.kind:
                case "hflip":
                    ops.append(v2.RandomHorizontalFlip(p.get("p", 0.5)))
                case "vflip":
                    ops.append(v2.RandomVerticalFlip(p.get("p", 0.5)))
                case "rotation":
                    ops.append(v2.RandomRotation(p.get("degrees", 10)))
                case "color_jitter":
                    j = p.get("strength", 0.1)
                    ops.append(v2.ColorJitter(brightness=j, contrast=j, saturation=j))
                case "random_resized_crop":
                    scale = tuple(p.get("scale", (0.8, 1.0)))
                    ops.append(v2.RandomResizedCrop(img.size, scale=scale, antialias=True))
                case "trivial_augment":
                    ops.append(v2.TrivialAugmentWide())
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


# ------------------------------------------------------------------ texto


def _texts(spec: TextSpec, df: pl.DataFrame) -> list[str]:
    from perceptron.data.text import normalize

    raw = df[spec.column].cast(pl.String).fill_null("").to_list()
    return [
        normalize(
            t, lowercase=spec.lowercase, accents=spec.accents, urls=spec.urls, numbers=spec.numbers
        )
        for t in raw
    ]


_HF_TOKENIZERS: dict[str, Any] = {}


def hf_tokenizer(name: str) -> Any:
    if name not in _HF_TOKENIZERS:
        from transformers import AutoTokenizer

        _HF_TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _HF_TOKENIZERS[name]


def _fit_text(fitted: FittedPipeline, train: pl.DataFrame) -> FittedPipeline:
    from perceptron.data.text import PAD, build_vocab

    spec = fitted.spec.text
    if spec is None:
        raise ValueError("pipeline de texto sin `text`")
    if spec.tokenizer == "hf":
        if not spec.hf_model:
            raise ValueError("tokenizer 'hf' requiere hf_model")
        tok = hf_tokenizer(spec.hf_model)
        return fitted.model_copy(update={"pad_id": int(tok.pad_token_id or 0)})
    vocab = build_vocab(_texts(spec, train), max_size=spec.vocab_size, min_freq=spec.min_freq)
    return fitted.model_copy(update={"vocab": vocab, "pad_id": PAD})


def transform_text(fitted: FittedPipeline, df: pl.DataFrame) -> np.ndarray:
    """Textos → ids [n, max_length] (int64), con padding y truncado."""
    from perceptron.data.text import encode

    spec = fitted.spec.text
    if spec is None:
        raise ValueError("pipeline de texto sin `text`")
    texts = _texts(spec, df)
    if spec.tokenizer == "hf" and spec.hf_model:
        tok = hf_tokenizer(spec.hf_model)
        enc = tok(texts, padding="max_length", truncation=True, max_length=spec.max_length)
        return np.asarray(enc["input_ids"], dtype=np.int64)
    index = {w: i for i, w in enumerate(fitted.vocab or [])}
    return np.asarray([encode(t, index, spec.max_length) for t in texts], dtype=np.int64).reshape(
        len(texts), spec.max_length
    )
