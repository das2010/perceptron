"""Explicabilidad (RF-EVL-02; ADR-0028) con Captum.

- Tabular: valores de Shapley por muestreo (ShapleyValueSampling), una "feature" por
  columna del pipeline (numéricas y categóricas por separado), con una línea base típica
  (mediana/moda de validación). Global: |atribución| media sobre validación. Local: por fila.
- Imagen: Integrated Gradients respecto de la clase predicha, con base en cero (la media
  normalizada), como mapa de calor superpuesto a la imagen.
- Texto: oclusión por token (cada token se reemplaza por el de padding): cuánto baja la
  probabilidad de la clase predicha sin él. Sirve igual con vocabulario propio o de HF.
- Audio: Integrated Gradients sobre el espectrograma (la entrada real del modelo), como mapa
  de calor tiempo × frecuencia.

Los modelos de código experto no se explican en el proceso del Engine (ADR-0025).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel

from perceptron.catalog.registry import CODE_BLOCK
from perceptron.core.errors import ValidationError
from perceptron.data.pipeline.pipeline import (
    audio_features,
    hf_tokenizer,
    image_transforms,
    transform_tabular,
    transform_text,
)
from perceptron.data.view import DatasetView
from perceptron.training.data import make_dataset
from perceptron.training.inference import TrainedModel, load_trained

GLOBAL_SAMPLES = 200
GLOBAL_PERMUTATIONS = 25
LOCAL_PERMUTATIONS = 100
IG_STEPS = 32


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    mean_attribution: float


class GlobalExplanation(BaseModel):
    method: str
    samples: int
    target: str
    features: list[FeatureImportance]


class Contribution(BaseModel):
    feature: str
    value: Any = None
    attribution: float


class LocalExplanation(BaseModel):
    method: str
    prediction: str
    contributions: list[Contribution] = []
    heatmap_png: str | None = None


def load_analyzable(run_dir: Path) -> TrainedModel:
    """Modelo entrenado en el proceso del Engine (no admite código experto, ADR-0025)."""
    from perceptron.training.inference import load_run_artifacts

    spec, _, _ = load_run_artifacts(run_dir)
    if any(n.block == CODE_BLOCK for n in spec.nodes):
        raise ValidationError(
            "la explicación de modelos de código experto todavía no está disponible"
        )
    trained = load_trained(run_dir)
    trained.model.eval()
    return trained


def _classes(trained: TrainedModel) -> list[str]:
    return trained.pipeline.classes or []


def _target_and_label(trained: TrainedModel, out: torch.Tensor) -> tuple[Any, list[str]]:
    """Índice de salida a explicar por fila y la etiqueta predicha."""
    if trained.task.value == "regression":
        return 0, [f"{float(v):.4g}" for v in out.reshape(len(out), -1)[:, 0]]
    classes = _classes(trained)
    if out.shape[1] == 1:  # binaria con BCE: se explica el logit de la clase positiva
        labels = [classes[int(v > 0)] if classes else str(int(v > 0)) for v in out[:, 0]]
        return 0, labels
    idx = out.argmax(1)
    return idx, [classes[int(i)] if classes else str(int(i)) for i in idx]


def _tabular_attribute(
    trained: TrainedModel, x_num: torch.Tensor, x_cat: torch.Tensor, n_samples: int
) -> tuple[np.ndarray, list[str], list[str]]:
    """Shapley por columna; solo se perturban las entradas no vacías (numéricas/categóricas)."""
    from captum.attr import ShapleyValueSampling

    model = trained.model
    names = [*trained.pipeline.numeric_features, *trained.pipeline.categorical_features]
    with torch.no_grad():
        out = model(x_num, x_cat)
    target, labels = _target_and_label(trained, out)
    parts = [("num", x_num), ("cat", x_cat)]
    active = [(k, x) for k, x in parts if x.shape[1] > 0]
    if not active:
        raise ValidationError("el modelo no tiene variables de entrada que explicar")

    def forward(*xs: torch.Tensor) -> torch.Tensor:
        given = dict(zip([k for k, _ in active], xs, strict=True))
        result: torch.Tensor = model(
            given.get("num", x_num[: len(xs[0])]), given.get("cat", x_cat[: len(xs[0])])
        )
        return result

    baselines, masks, offset = [], [], 0
    for kind, x in active:
        base = x.median(0, keepdim=True).values if kind == "num" else x.mode(0, keepdim=True).values
        baselines.append(base.expand_as(x))
        masks.append((torch.arange(x.shape[1]) + offset).unsqueeze(0))
        offset += x.shape[1]
    with torch.no_grad():
        attrs = ShapleyValueSampling(forward).attribute(
            tuple(x for _, x in active),
            baselines=tuple(baselines),
            target=target,
            feature_mask=tuple(masks),
            n_samples=n_samples,
        )
    attr = torch.cat([a.float() for a in attrs], dim=1).numpy()
    return attr, names, labels


def global_explanation(run_dir: Path, dataset_dir: Path) -> GlobalExplanation:
    trained = load_analyzable(run_dir)
    if trained.spec.input.kind != "tabular":
        raise ValidationError("la explicación global está disponible para modelos tabulares")
    ds = make_dataset(DatasetView(dataset_dir), trained.pipeline, "val", train=False)
    n = min(GLOBAL_SAMPLES, len(ds))  # type: ignore[arg-type]
    x_num, x_cat = ds.x_num[:n], ds.x_cat[:n]  # type: ignore[attr-defined]
    attr, names, _ = _tabular_attribute(trained, x_num, x_cat, GLOBAL_PERMUTATIONS)
    feats = [
        FeatureImportance(
            feature=name,
            importance=float(np.abs(attr[:, i]).mean()),
            mean_attribution=float(attr[:, i].mean()),
        )
        for i, name in enumerate(names)
    ]
    feats.sort(key=lambda f: f.importance, reverse=True)
    target = "valor predicho" if trained.task.value == "regression" else "clase predicha"
    return GlobalExplanation(method="shapley_sampling", samples=n, target=target, features=feats)


def local_tabular(run_dir: Path, row: dict[str, Any]) -> LocalExplanation:
    trained = load_analyzable(run_dir)
    if trained.spec.input.kind != "tabular":
        raise ValidationError("este modelo no es tabular")
    arrays = transform_tabular(trained.pipeline, pl.DataFrame([row]))
    x_num, x_cat = torch.tensor(arrays.x_num), torch.tensor(arrays.x_cat)
    attr, names, labels = _tabular_attribute(trained, x_num, x_cat, LOCAL_PERMUTATIONS)
    contributions = [
        Contribution(feature=name, value=row.get(name), attribution=float(attr[0, i]))
        for i, name in enumerate(names)
    ]
    contributions.sort(key=lambda c: abs(c.attribution), reverse=True)
    return LocalExplanation(
        method="shapley_sampling", prediction=labels[0], contributions=contributions
    )


def local_image(run_dir: Path, image: Any) -> LocalExplanation:
    """Integrated Gradients de la clase predicha, como PNG superpuesto (base64)."""
    from captum.attr import IntegratedGradients
    from PIL import Image

    trained = load_analyzable(run_dir)
    if trained.spec.input.kind != "image":
        raise ValidationError("este modelo no es de imágenes")
    spec = trained.pipeline.spec.image
    mode = "L" if spec is not None and spec.channels == 1 else "RGB"
    base = image.convert(mode)
    x = image_transforms(trained.pipeline, train=False)(base).unsqueeze(0)
    with torch.no_grad():
        out = trained.model(x)
    target, labels = _target_and_label(trained, out)
    ig = IntegratedGradients(trained.model)
    attr = ig.attribute(x, baselines=torch.zeros_like(x), target=target, n_steps=IG_STEPS)
    heat = attr.detach().abs().sum(1)[0].numpy()
    heat = heat / (heat.max() or 1.0)
    overlay = _overlay(base.convert("RGB"), heat, Image)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return LocalExplanation(
        method="integrated_gradients",
        prediction=labels[0],
        heatmap_png=base64.b64encode(buf.getvalue()).decode("ascii"),
    )


def _overlay(img: Any, heat: np.ndarray, image_mod: Any) -> Any:
    """Mapa de calor lima (marca Preteco) sobre la imagen, con transparencia por intensidad."""
    h = image_mod.fromarray((heat * 255).astype(np.uint8)).resize(img.size)
    alpha = np.asarray(h, dtype=np.float32)[..., None] / 255.0 * 0.7
    lime = np.array([203, 255, 0], dtype=np.float32)
    base = np.asarray(img, dtype=np.float32)
    mixed = base * (1 - alpha) + lime * alpha
    return image_mod.fromarray(mixed.clip(0, 255).astype(np.uint8))


def _score(trained: TrainedModel, out: torch.Tensor, target: Any, sign: float) -> torch.Tensor:
    """Qué tan fuerte es la predicción explicada: probabilidad de la clase predicha; en
    regresión el valor y en binaria el logit orientado hacia la clase predicha (`sign`)."""
    if trained.task.value == "regression" or out.shape[1] == 1:
        return sign * out[:, 0]
    k = int(target[0]) if torch.is_tensor(target) else int(target)
    return torch.softmax(out.float(), dim=1)[:, k]


def _token_texts(trained: TrainedModel, ids: list[int]) -> list[str]:
    spec = trained.pipeline.spec.text
    if spec is not None and spec.tokenizer == "hf" and spec.hf_model:
        return [str(t) for t in hf_tokenizer(spec.hf_model).convert_ids_to_tokens(ids)]
    vocab = trained.pipeline.vocab or []
    return [vocab[i] if 0 <= i < len(vocab) else "<unk>" for i in ids]


def local_text(run_dir: Path, text: str) -> LocalExplanation:
    """Oclusión por token sobre la clase predicha (una sola pasada con un batch)."""
    trained = load_analyzable(run_dir)
    if trained.spec.input.kind != "tokens":
        raise ValidationError("este modelo no es de texto")
    spec = trained.pipeline.spec.text
    if spec is None:
        raise ValidationError("el modelo no tiene pipeline de texto")
    ids = torch.tensor(transform_text(trained.pipeline, pl.DataFrame({spec.column: [text]})))
    pad = int(trained.pipeline.pad_id or 0)
    positions = [i for i, v in enumerate(ids[0].tolist()) if v != pad]
    if not positions:
        raise ValidationError("el texto no tiene tokens conocidos por el modelo")
    with torch.no_grad():
        out = trained.model(ids)
        target, labels = _target_and_label(trained, out)
        binary = trained.task.value != "regression" and out.shape[1] == 1
        sign = -1.0 if binary and float(out[0, 0]) < 0 else 1.0
        base = float(_score(trained, out, target, sign)[0])
        occluded = ids.repeat(len(positions), 1)
        for row, pos in enumerate(positions):
            occluded[row, pos] = pad
        scores = _score(trained, trained.model(occluded), target, sign)
    tokens = _token_texts(trained, [int(ids[0, p]) for p in positions])
    contributions = [
        Contribution(feature=f"{i}:{tok}", value=tok, attribution=base - float(sc))
        for i, (tok, sc) in enumerate(zip(tokens, scores.tolist(), strict=True))
    ]
    return LocalExplanation(method="occlusion", prediction=labels[0], contributions=contributions)


def local_audio(run_dir: Path, path: Path) -> LocalExplanation:
    """Integrated Gradients sobre el espectrograma de la clase predicha, como PNG."""
    from captum.attr import IntegratedGradients
    from PIL import Image

    from perceptron.data.audio import load

    trained = load_analyzable(run_dir)
    if trained.spec.input.kind != "spectrogram":
        raise ValidationError("este modelo no es de audio")
    spec = trained.pipeline.spec.audio
    if spec is None:
        raise ValidationError("el modelo no tiene pipeline de audio")
    try:
        wav, _ = load(path, sample_rate=spec.sample_rate)
    except Exception as exc:
        raise ValidationError(f"no se pudo leer el audio: {exc}") from exc
    mean = trained.pipeline.audio_mean or 0.0
    std = trained.pipeline.audio_std or 1.0
    x = ((audio_features(spec, wav) - mean) / std).float().unsqueeze(0)
    with torch.no_grad():
        out = trained.model(x)
    target, labels = _target_and_label(trained, out)
    attr = IntegratedGradients(trained.model).attribute(
        x, baselines=torch.zeros_like(x), target=target, n_steps=IG_STEPS
    )
    heat = attr.detach().abs().reshape(attr.shape[-2], attr.shape[-1]).numpy()[::-1]
    heat = heat / (heat.max() or 1.0)
    feats = x.reshape(x.shape[-2], x.shape[-1]).numpy()[::-1]  # graves abajo
    gray = (feats - feats.min()) / ((feats.max() - feats.min()) or 1.0) * 255
    base = Image.fromarray(gray.astype(np.uint8)).convert("RGB")
    base = base.resize((max(base.width * 4, 256), max(base.height * 3, 128)))
    buf = io.BytesIO()
    _overlay(base, np.ascontiguousarray(heat), Image).save(buf, format="PNG")
    return LocalExplanation(
        method="integrated_gradients",
        prediction=labels[0],
        heatmap_png=base64.b64encode(buf.getvalue()).decode("ascii"),
    )
