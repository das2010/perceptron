"""Recomendador de arquitectura por reglas (RF-ARC-04, SPEC §8).

Tabla de decisión por (modalidad, tarea, n.º de muestras, hardware). Es el
fallback cuando no hay LLM (privacidad L0) o su propuesta falla la validación.
"""

from __future__ import annotations

from dataclasses import dataclass

from perceptron.archspec.schema import ArchSpec
from perceptron.archspec.validate import offline_mode
from perceptron.catalog.templates import image_template, tabular_template
from perceptron.data.pipeline.pipeline import FittedPipeline
from perceptron.data.profiling.card import ProfileCard
from perceptron.domain.enums import Modality, TaskType
from perceptron.training.hardware import HardwareReport

SMALL_TABULAR = 2_000
LARGE_TABULAR = 50_000
SMALL_IMAGES = 5_000
GPU_MIN_GB = 6.0
TINY_IMAGE = 64


@dataclass(frozen=True)
class Recommendation:
    template: str
    spec: ArchSpec
    rationale: str


def _gpu_gb(hw: HardwareReport | None) -> float:
    return max((g.vram_total_gb for g in hw.gpus), default=0.0) if hw else 0.0


def _tabular(
    card: ProfileCard, fitted: FittedPipeline, task: TaskType, gpu: float
) -> Recommendation:
    n = card.num_samples

    def rec(template: str, why: str) -> Recommendation:
        spec = tabular_template(
            template,
            task=task,
            num_classes=fitted.num_classes if task is not TaskType.REGRESSION else None,
            num_numeric=len(fitted.numeric_features),
            cardinalities=[fitted.cardinalities[c] for c in fitted.categorical_features],
            rationale=why,
        )
        return Recommendation(template, spec, why)

    if n < SMALL_TABULAR:
        return rec("mlp", f"Pocas filas ({n}): un MLP chico con dropout generaliza mejor.")
    if n >= LARGE_TABULAR and gpu >= GPU_MIN_GB:
        return rec(
            "ft_transformer",
            f"{n} filas y GPU de {gpu:.0f} GB: FT-Transformer suele superar a los MLP.",
        )
    return rec("resnet_mlp", f"{n} filas: ResNet-MLP es un baseline fuerte y estable.")


def _image(
    card: ProfileCard, fitted: FittedPipeline, task: TaskType, gpu: float, can_download: bool
) -> Recommendation:
    n = card.num_samples
    img = fitted.spec.image
    size = img.size if img else 224
    channels = img.channels if img else 3

    def rec(
        backbone: str, why: str, *, pretrained: bool, freeze: int = 0, min_size: int = 0
    ) -> Recommendation:
        spec = image_template(
            backbone,
            task=task,
            num_classes=fitted.num_classes if task is not TaskType.REGRESSION else None,
            image_size=max(size, min_size),
            channels=channels,
            pretrained=pretrained,
            freeze_epochs=freeze,
            rationale=why,
        )
        return Recommendation(backbone, spec, why)

    if gpu >= GPU_MIN_GB and can_download:
        if n < SMALL_IMAGES:
            why = (
                f"{n} imágenes y GPU de {gpu:.0f} GB: EfficientNet-B0 preentrenado, "
                "backbone congelado 3 épocas y luego ajuste fino completo."
            )
            return rec("efficientnet_b0", why, pretrained=True, freeze=3, min_size=224)
        why = f"{n} imágenes y GPU de {gpu:.0f} GB: ConvNeXt-Tiny preentrenado."
        return rec("convnext_tiny", why, pretrained=True, freeze=1, min_size=224)
    if size <= TINY_IMAGE or not can_download:
        offline = " sin conexión" if not can_download else ""
        why = f"Imágenes de {size}px{offline} y sin GPU: una CNN compacta desde cero es rápida."
        return rec("small_cnn", why, pretrained=False)
    why = "Sin GPU: MobileNetV3-Small preentrenado es liviano y rinde bien en CPU."
    return rec("mobilenetv3_small_100", why, pretrained=True, freeze=3)


def recommend(
    card: ProfileCard,
    fitted: FittedPipeline,
    hardware: HardwareReport | None = None,
    *,
    allow_download: bool | None = None,
) -> Recommendation:
    if card.target is None or fitted.spec.target is None:
        raise ValueError("se necesita un target para recomendar una arquitectura")
    task = fitted.spec.target.task
    gpu = _gpu_gb(hardware)
    can_download = (not offline_mode()) if allow_download is None else allow_download
    if card.modality is Modality.TABULAR:
        return _tabular(card, fitted, task, gpu)
    if card.modality is Modality.IMAGE:
        return _image(card, fitted, task, gpu, can_download)
    raise NotImplementedError(f"reglas para {card.modality} llegan en Capa 1b")
