"""Explicabilidad (RF-EVL-02; ADR-0028) con Captum.

- Tabular: valores de Shapley por muestreo (ShapleyValueSampling), una "feature" por
  columna del pipeline (numéricas y categóricas por separado), con una línea base típica
  (mediana/moda de validación). Global: |atribución| media sobre validación. Local: por fila.
- Imagen: Integrated Gradients respecto de la clase predicha, con base en cero (la media
  normalizada), como mapa de calor superpuesto a la imagen.

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
from perceptron.data.pipeline.pipeline import image_transforms, transform_tabular
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
