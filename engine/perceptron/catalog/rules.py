"""Recomendador de arquitectura por reglas (RF-ARC-04, SPEC §8).

Tabla de decisión por (modalidad, tarea, n.º de muestras, hardware). Es el
fallback cuando no hay LLM (privacidad L0) o su propuesta falla la validación.
"""

from __future__ import annotations

from dataclasses import dataclass

from perceptron.archspec.schema import ArchSpec
from perceptron.archspec.validate import offline_mode
from perceptron.catalog.templates import image_template, tabular_template, text_template
from perceptron.data.pipeline.pipeline import FittedPipeline
from perceptron.data.profiling.card import ProfileCard
from perceptron.domain.enums import Modality, TaskType
from perceptron.training.hardware import HardwareReport

_VISION_TASKS = (TaskType.OBJECT_DETECTION, TaskType.SEGMENTATION, TaskType.OCR)
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


def _text(card: ProfileCard, fitted: FittedPipeline, task: TaskType) -> Recommendation:
    tspec = fitted.spec.text
    if tspec is None:
        raise ValueError("pipeline de texto sin configuración de texto")
    num_classes = fitted.num_classes if task is not TaskType.REGRESSION else None
    if tspec.tokenizer == "hf":
        why = f"Encoder preentrenado {tspec.hf_model}: transfiere conocimiento del idioma."
        spec = text_template(
            "hf",
            task=task,
            num_classes=num_classes,
            max_length=tspec.max_length,
            pad_id=fitted.pad_id,
            hf_model=tspec.hf_model,
            pretrained=True,
            freeze_epochs=1,
            rationale=why,
        )
        return Recommendation(f"hf:{tspec.hf_model}", spec, why)
    why = (
        f"{card.num_samples} textos sin encoder preentrenado disponible: TextCNN sobre "
        "embeddings propios es rápido en CPU y fuerte para clasificar textos cortos."
    )
    spec = text_template(
        "textcnn",
        task=task,
        num_classes=num_classes,
        max_length=tspec.max_length,
        vocab_size=len(fitted.vocab or []),
        pad_id=fitted.pad_id,
        rationale=why,
    )
    return Recommendation("textcnn", spec, why)


def _vision_task(card: ProfileCard, fitted: FittedPipeline, task: TaskType) -> Recommendation:
    from perceptron.catalog.templates import vision_task_template

    img = fitted.spec.image
    if img is None:
        raise ValueError("pipeline de visión sin configuración de imagen")
    channels = img.channels
    shape = [channels, img.size, img.width or img.size]
    classes = fitted.classes or []
    if task is TaskType.OCR:
        n, why = (
            len(classes) + 1,
            "CRNN + CTC: lee la línea de izquierda a derecha sin segmentar caracteres.",
        )
    elif task is TaskType.SEGMENTATION:
        n, why = (
            len(classes),
            "U-Net compacta: codificador-decodificador con conexiones de salto por píxel.",
        )
    else:
        n, why = (
            len(classes),
            "CenterNet compacto: predice centros y tamaños; se entrena desde cero en CPU.",
        )
    spec = vision_task_template(task, num_classes=n, image_shape=shape, rationale=why)
    return Recommendation(spec.provenance.template or task.value, spec, why)


def _audio(
    card: ProfileCard, fitted: FittedPipeline, task: TaskType, gpu: float, can_download: bool
) -> Recommendation:
    from perceptron.catalog.templates import audio_template
    from perceptron.data.audio import frames_for

    aspec = fitted.spec.audio
    if aspec is None:
        raise ValueError("pipeline de audio sin configuración")
    frames = frames_for(aspec.duration_s, aspec.sample_rate)
    num_classes = fitted.num_classes if task is not TaskType.REGRESSION else None
    common = {"task": task, "num_classes": num_classes, "bins": aspec.bins, "frames": frames}
    if gpu >= GPU_MIN_GB and can_download:
        why = (
            "Espectrograma como imagen + EfficientNet-B0 preentrenado (transfer learning desde "
            "ImageNet), backbone congelado 3 épocas."
        )
        spec = audio_template("efficientnet_b0", pretrained=True, rationale=why, **common)  # type: ignore[arg-type]
        return Recommendation("efficientnet_b0", spec, why)
    why = f"{card.num_samples} clips en CPU: CNN compacta sobre el log-mel spectrogram."
    spec = audio_template("small_cnn", rationale=why, **common)  # type: ignore[arg-type]
    return Recommendation("small_cnn", spec, why)


LARGE_SERIES_WINDOWS = 20_000


def _series(card: ProfileCard, fitted: FittedPipeline, gpu: float) -> Recommendation:
    from perceptron.catalog.templates import series_template
    from perceptron.data.pipeline.series_windows import channels

    sspec = fitted.spec.series
    if sspec is None:
        raise ValueError("pipeline de series sin configuración")
    cfg = sspec.config
    c = len(channels(cfg, sspec.calendar))
    if cfg.task is TaskType.ANOMALY_DETECTION:
        why = (
            "Autoencoder de ventanas: aprende la forma normal de la señal; un error de "
            "reconstrucción alto marca una anomalía (umbral calibrado en validación)."
        )
        spec = series_template(
            "ae_conv", task=cfg.task, lookback=cfg.lookback, channels=c, rationale=why
        )
        return Recommendation("ae_conv", spec, why)
    windows = card.num_samples  # cota superior del n.º de ventanas de train
    if gpu >= GPU_MIN_GB and windows >= LARGE_SERIES_WINDOWS:
        why = "Muchas ventanas y GPU: PatchTST captura dependencias largas con atención."
        backbone = "patchtst"
    else:
        why = (
            "N-BEATS: MLP profundo con backcast residual, fuerte en forecasting univariado y "
            "rápido en CPU."
        )
        backbone = "nbeats"
    spec = series_template(
        backbone,
        task=cfg.task,
        lookback=cfg.lookback,
        channels=c,
        horizon=cfg.horizon,
        rationale=why,
    )
    return Recommendation(backbone, spec, why)


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
    if card.modality is Modality.IMAGE and task in _VISION_TASKS:
        return _vision_task(card, fitted, task)
    if card.modality is Modality.IMAGE:
        return _image(card, fitted, task, gpu, can_download)
    if card.modality is Modality.TEXT:
        return _text(card, fitted, task)
    if card.modality is Modality.TIMESERIES:
        return _series(card, fitted, gpu)
    if card.modality is Modality.AUDIO:
        return _audio(card, fitted, task, gpu, can_download)
    raise NotImplementedError(f"reglas para {card.modality} llegan en Capa 1b")
