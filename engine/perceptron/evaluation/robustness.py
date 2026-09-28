"""Robustez ante perturbaciones sobre el test sellado (RF-EVL-05).

Se mide la métrica principal (accuracy o MAE) con entradas perturbadas a tres niveles de
severidad y se informa la degradación respecto de las entradas limpias:

- tabular: ruido gaussiano en numéricas (en desvíos estándar), categorías cambiadas al azar y
  valores faltantes (numéricas a la mediana, categóricas a "desconocida");
- imagen: ruido gaussiano, desenfoque y compresión JPEG;
- texto: typos (caracteres cambiados, borrados o duplicados) y palabras eliminadas;
- audio: ruido de fondo (por relación señal/ruido en dB) y volumen bajo.

Determinístico (semilla fija). Los modelos de código experto no se evalúan en el Engine.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel

from perceptron.data.pipeline.pipeline import decode_regression, encode_target, image_transforms
from perceptron.data.view import DatasetView
from perceptron.evaluation.explain import load_analyzable
from perceptron.evaluation.predictions import eval_frame
from perceptron.training.data import TabularDataset
from perceptron.training.inference import TrainedModel

SEED = 0
MAX_ROWS = 2000
MAX_IMAGES = 300
TABULAR = {
    "ruido gaussiano": (0.1, 0.25, 0.5),
    "categorías cambiadas": (0.05, 0.1, 0.2),
    "valores faltantes": (0.1, 0.2, 0.4),
}
IMAGE = {
    "ruido gaussiano": (0.05, 0.1, 0.2),
    "desenfoque": (1.0, 2.0, 3.0),
    "compresión JPEG": (50.0, 20.0, 10.0),
}
MAX_TEXTS = 1000
TEXT = {
    "typos": (0.02, 0.05, 0.1),  # fracción de caracteres alterados
    "palabras eliminadas": (0.1, 0.2, 0.3),
}
MAX_AUDIO = 200
AUDIO = {
    "ruido de fondo": (20.0, 10.0, 5.0),  # SNR en dB: menos es más ruido
    "volumen bajo": (0.5, 0.25, 0.1),  # ganancia
}


class PerturbationResult(BaseModel):
    kind: str
    severity: float
    metric: float
    degradation: float


class RobustnessReport(BaseModel):
    metric: str
    higher_is_better: bool
    baseline: float
    samples: int
    results: list[PerturbationResult]


def _metric(trained: TrainedModel, out: torch.Tensor, y: torch.Tensor) -> float:
    if trained.task.value == "regression":
        pred = decode_regression(trained.pipeline, out.reshape(len(out), -1)[:, 0].numpy())
        true = decode_regression(trained.pipeline, y.float().numpy())
        return float(np.abs(pred - true).mean())
    labels = (out[:, 0] > 0).long() if out.shape[1] == 1 else out.argmax(1)
    return float((labels == y).float().mean())


@torch.no_grad()
def _tabular(
    trained: TrainedModel, df: pl.DataFrame
) -> tuple[float, list[tuple[str, float, float]]]:
    ds = TabularDataset(trained.pipeline, df.head(MAX_ROWS))
    x_num, x_cat, y = ds.x_num, ds.x_cat, ds.y
    model = trained.model
    base = _metric(trained, model(x_num, x_cat), y)
    rng = torch.Generator().manual_seed(SEED)
    std = x_num.std(0, keepdim=True).clamp(min=1e-6) if x_num.shape[1] else x_num
    median = x_num.median(0, keepdim=True).values if x_num.shape[1] else x_num
    cards = [
        trained.pipeline.cardinalities.get(c, 2) for c in trained.pipeline.categorical_features
    ]
    results = []
    for kind, levels in TABULAR.items():
        for level in levels:
            xn, xc = x_num.clone(), x_cat.clone()
            if kind == "ruido gaussiano" and xn.shape[1]:
                xn = xn + torch.randn(xn.shape, generator=rng) * level * std
            elif kind == "categorías cambiadas" and xc.shape[1]:
                mask = torch.rand(xc.shape, generator=rng) < level
                rand = torch.stack(
                    [torch.randint(0, max(c, 1), (len(xc),), generator=rng) for c in cards], dim=1
                )
                xc = torch.where(mask, rand, xc)
            elif kind == "valores faltantes":
                if xn.shape[1]:
                    m = torch.rand(xn.shape, generator=rng) < level
                    xn = torch.where(m, median.expand_as(xn), xn)
                if xc.shape[1]:
                    m = torch.rand(xc.shape, generator=rng) < level
                    xc = torch.where(m, torch.zeros_like(xc), xc)
            results.append((kind, level, _metric(trained, model(xn, xc), y)))
    return base, results


def _perturb_image(img: Any, kind: str, level: float, rng: np.random.Generator) -> Any:
    from PIL import Image, ImageFilter

    if kind == "ruido gaussiano":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, level, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "desenfoque":
        return img.filter(ImageFilter.GaussianBlur(level))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(level))
    return Image.open(io.BytesIO(buf.getvalue())).convert(img.mode)


@torch.no_grad()
def _images(
    trained: TrainedModel, df: pl.DataFrame, files_dir: Path
) -> tuple[float, list[tuple[str, float, float]]]:
    from PIL import Image

    df = df.head(MAX_IMAGES)
    target = trained.pipeline.spec.target
    y = torch.tensor(encode_target(trained.pipeline, df[target.name])) if target else None
    if y is None:
        raise ValueError("el dataset no tiene target")
    tf = image_transforms(trained.pipeline, train=False)
    spec = trained.pipeline.spec.image
    mode = "L" if spec is not None and spec.channels == 1 else "RGB"
    images = []
    for p in df["path"].to_list():
        with Image.open(files_dir / p) as im:
            images.append(im.convert(mode))

    def score(imgs: list[Any]) -> float:
        out = torch.cat(
            [
                trained.model(torch.stack([tf(i) for i in imgs[k : k + 64]]))
                for k in range(0, len(imgs), 64)
            ]
        )
        return _metric(trained, out, y)

    base = score(images)
    rng = np.random.default_rng(SEED)
    results = [
        (kind, level, score([_perturb_image(i, kind, level, rng) for i in images]))
        for kind, levels in IMAGE.items()
        for level in levels
    ]
    return base, results


def _typos(text: str, rate: float, rng: np.random.Generator) -> str:
    chars = list(text)
    out: list[str] = []
    for c in chars:
        r = rng.random()
        if c.isalpha() and r < rate / 3:
            continue  # borrado
        if c.isalpha() and r < 2 * rate / 3:
            out.append(chr(ord("a") + int(rng.integers(0, 26))))  # cambiado
            continue
        out.append(c)
        if c.isalpha() and r < rate:
            out.append(c)  # duplicado
    return "".join(out)


def _drop_words(text: str, rate: float, rng: np.random.Generator) -> str:
    words = text.split()
    kept = [w for w in words if rng.random() >= rate]
    return " ".join(kept or words[:1])


@torch.no_grad()
def _texts(trained: TrainedModel, df: pl.DataFrame) -> tuple[float, list[tuple[str, float, float]]]:
    from perceptron.data.pipeline.pipeline import transform_text

    spec = trained.pipeline.spec.text
    target = trained.pipeline.spec.target
    if spec is None or target is None:
        raise ValueError("el dataset no tiene texto o target")
    df = df.head(MAX_TEXTS)
    y = torch.tensor(encode_target(trained.pipeline, df[target.name]))
    texts = df[spec.column].cast(pl.String).fill_null("").to_list()

    def score(items: list[str]) -> float:
        ids = torch.tensor(transform_text(trained.pipeline, pl.DataFrame({spec.column: items})))
        out = torch.cat([trained.model(ids[k : k + 128]) for k in range(0, len(ids), 128)])
        return _metric(trained, out, y)

    rng = np.random.default_rng(SEED)
    perturb = {"typos": _typos, "palabras eliminadas": _drop_words}
    results = [
        (kind, level, score([perturb[kind](t, level, rng) for t in texts]))
        for kind, levels in TEXT.items()
        for level in levels
    ]
    return score(texts), results


@torch.no_grad()
def _audio(
    trained: TrainedModel, df: pl.DataFrame, files_dir: Path
) -> tuple[float, list[tuple[str, float, float]]]:
    from perceptron.data.audio import load
    from perceptron.data.pipeline.pipeline import audio_features

    spec = trained.pipeline.spec.audio
    target = trained.pipeline.spec.target
    if spec is None or target is None:
        raise ValueError("el dataset no tiene audio o target")
    df = df.filter(~pl.col("corrupt")) if "corrupt" in df.columns else df
    df = df.head(MAX_AUDIO)
    y = torch.tensor(encode_target(trained.pipeline, df[target.name]))
    waves = [load(files_dir / p, sample_rate=spec.sample_rate)[0] for p in df["path"].to_list()]
    mean = trained.pipeline.audio_mean or 0.0
    std = trained.pipeline.audio_std or 1.0

    def score(items: list[np.ndarray]) -> float:
        feats = torch.stack([(audio_features(spec, w) - mean) / std for w in items]).float()
        out = torch.cat([trained.model(feats[k : k + 64]) for k in range(0, len(feats), 64)])
        return _metric(trained, out, y)

    rng = np.random.default_rng(SEED)

    def noisy(w: np.ndarray, snr_db: float) -> np.ndarray:
        power = float(np.mean(w**2)) or 1e-8
        noise = rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), w.shape)
        return (w + noise).astype(np.float32)

    results = [
        (k, lvl, score([noisy(w, lvl) for w in waves]))
        for k, lvl in [("ruido de fondo", x) for x in AUDIO["ruido de fondo"]]
    ]
    results += [
        ("volumen bajo", g, score([(w * g).astype(np.float32) for w in waves]))
        for g in AUDIO["volumen bajo"]
    ]
    return score(waves), results


def robustness_report(run_dir: Path, dataset_dir: Path) -> RobustnessReport:
    trained = load_analyzable(run_dir)
    view = DatasetView(dataset_dir)
    df = eval_frame(view)
    kind = trained.spec.input.kind
    if kind == "tabular":
        base, results = _tabular(trained, df)
        samples = min(df.height, MAX_ROWS)
    elif kind == "image":
        base, results = _images(trained, df, view.files_dir)
        samples = min(df.height, MAX_IMAGES)
    elif kind == "tokens":
        base, results = _texts(trained, df)
        samples = min(df.height, MAX_TEXTS)
    elif kind == "spectrogram":
        base, results = _audio(trained, df, view.files_dir)
        samples = min(df.height, MAX_AUDIO)
    else:
        from perceptron.core.errors import ValidationError

        raise ValidationError(f"la prueba de robustez todavía no cubre entradas {kind}")
    regression = trained.task.value == "regression"
    return RobustnessReport(
        metric="mae" if regression else "accuracy",
        higher_is_better=not regression,
        baseline=base,
        samples=samples,
        results=[
            PerturbationResult(
                kind=k,
                severity=lvl,
                metric=m,
                degradation=(m - base) if regression else (base - m),
            )
            for k, lvl, m in results
        ],
    )
