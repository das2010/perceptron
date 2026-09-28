"""Sub-wizard de definición de arquitectura (SPEC §7.6, paso 6): familia → backbone → cabeza
→ regularización.

Cada paso ofrece opciones explicadas del catálogo (§8), marca la que recomiendan las reglas
(RF-ARC-04) y desactiva las que no aplican (p. ej. pesos preentrenados sin conexión). Con
todas las elecciones se arma una ArchSpec sobre las plantillas del catálogo; el validador
(§9.3) la revisa antes de guardarla. El copiloto ("guía de definición", §7.7) explica las
opciones del paso actual.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from perceptron.archspec.schema import HP, ArchSpec, LossSpec
from perceptron.archspec.validate import offline_mode
from perceptron.catalog.rules import SMALL_TABULAR, recommend
from perceptron.catalog.templates import image_template, tabular_template
from perceptron.data.pipeline.pipeline import FittedPipeline
from perceptron.data.profiling.card import ProfileCard
from perceptron.domain.enums import Modality, Origin, TaskType
from perceptron.training.hardware import HardwareReport

STEPS = ("family", "backbone", "head", "regularization")
_TASKS = (TaskType.CLASSIFICATION, TaskType.REGRESSION)
IMBALANCED = 0.2
FEW_IMAGES = 1_000


class DefineOption(BaseModel):
    id: str
    title: str
    description: str
    recommended: bool = False
    available: bool = True
    reason: str | None = Field(default=None, description="Por qué no está disponible")


class DefineStep(BaseModel):
    step: str
    title: str
    options: list[DefineOption]
    choice: str | None = None


class DefinePlan(BaseModel):
    modality: Modality
    task: TaskType
    steps: list[DefineStep]
    complete: bool


# ---------------------------------------------------------------- tabular

_TAB_FAMILIES = {
    "mlp": ("MLP", "Capas densas con dropout. Rápido y robusto con pocas filas."),
    "resnet_mlp": ("ResNet-MLP", "MLP con conexiones residuales: baseline fuerte y estable."),
    "ft_transformer": (
        "FT-Transformer",
        "Cada columna es un token con atención entre features. Rinde más con muchas filas y GPU.",
    ),
}
_TAB_SIZES: dict[str, dict[str, dict[str, int]]] = {
    "mlp": {
        "small": {"hidden": 64, "layers": 2},
        "medium": {"hidden": 128, "layers": 2},
        "large": {"hidden": 256, "layers": 3},
    },
    "resnet_mlp": {
        "small": {"d": 64, "blocks": 1},
        "medium": {"d": 128, "blocks": 2},
        "large": {"d": 256, "blocks": 4},
    },
    "ft_transformer": {
        "small": {"d_token": 32, "n_blocks": 2},
        "medium": {"d_token": 64, "n_blocks": 3},
        "large": {"d_token": 128, "n_blocks": 4},
    },
}
_MAIN_NODE = {"mlp": "mlp", "resnet_mlp": "resmlp", "ft_transformer": "encoder"}
_SIZE_TEXT = {
    "small": ("Chico", "Menos parámetros: entrena rápido y sobreajusta menos."),
    "medium": ("Mediano", "Equilibrio entre capacidad y velocidad."),
    "large": ("Grande", "Más capacidad para relaciones complejas; necesita más datos."),
}

# ---------------------------------------------------------------- imagen

_IMG_FAMILIES = {
    "scratch": ("CNN desde cero", "Red convolucional compacta sin pesos previos: rápida en CPU."),
    "light": (
        "Preentrenada liviana",
        "Aprovecha pesos entrenados en ImageNet con un modelo chico: buena en CPU.",
    ),
    "strong": (
        "Preentrenada de alta capacidad",
        "Modelos más grandes preentrenados: la mejor precisión, idealmente con GPU.",
    ),
}
_IMG_BACKBONES: dict[str, dict[str, tuple[str, str]]] = {
    "scratch": {
        "small_cnn:16": ("CNN compacta · angosta", "16 filtros base: mínima y muy rápida."),
        "small_cnn:32": ("CNN compacta · media", "32 filtros base: el punto de partida usual."),
        "small_cnn:64": ("CNN compacta · ancha", "64 filtros base: más capacidad, más lenta."),
    },
    "light": {
        "mobilenetv3_small_100": ("MobileNetV3-Small", "~1,5 M parámetros; la más liviana."),
        "mobilenetv3_large_100": ("MobileNetV3-Large", "~4 M parámetros; más precisa."),
        "efficientnet_b0": (
            "EfficientNet-B0",
            "~4 M parámetros; muy buena relación costo/precisión.",
        ),
    },
    "strong": {
        "resnet18": ("ResNet-18", "~11 M parámetros; clásica y estable."),
        "resnet50": ("ResNet-50", "~23 M parámetros; más capacidad."),
        "convnext_tiny": ("ConvNeXt-Tiny", "~28 M parámetros; estado del arte en CNN."),
    },
}

_LOSSES = {
    "ce": ("Entropía cruzada", "Estándar para clasificación, con pesos de clase automáticos."),
    "ce_smooth": (
        "Entropía cruzada suavizada",
        "Label smoothing 0,1: modelos menos sobreconfiados y mejor calibrados.",
    ),
    "focal": ("Focal loss", "Enfoca el aprendizaje en los casos difíciles; útil con desbalance."),
}
_REG_LOSSES = {
    "mse": ("Error cuadrático (MSE)", "Estándar; penaliza mucho los errores grandes."),
    "huber": ("Huber", "Como MSE cerca de cero y lineal lejos: robusta a outliers."),
    "mae": ("Error absoluto (MAE)", "Robusta a outliers; optimiza la mediana."),
}
_REGULARIZATION = {
    "low": ("Baja", "Poco dropout y weight decay: para datasets grandes."),
    "medium": ("Media", "Valores por defecto razonables."),
    "high": ("Alta", "Más dropout y weight decay: para pocos datos o si sobreajusta."),
}
_TAB_REG = {"low": (0.0, 1e-5), "medium": (0.1, 1e-4), "high": (0.3, 1e-3)}
_IMG_REG = {"low": (0.1, 1e-4, 0), "medium": (0.2, 1e-2, 0), "high": (0.4, 5e-2, 3)}


def _task(fitted: FittedPipeline) -> TaskType:
    if fitted.spec.target is None:
        raise ValueError("se necesita un target para definir la arquitectura")
    return fitted.spec.target.task


def _options(
    table: dict[str, tuple[str, str]], recommended: str | None, unavailable: dict[str, str]
) -> list[DefineOption]:
    return [
        DefineOption(
            id=k,
            title=title,
            description=desc,
            recommended=k == recommended,
            available=k not in unavailable,
            reason=unavailable.get(k),
        )
        for k, (title, desc) in table.items()
    ]


def _step(
    step: str, title: str, options: list[DefineOption], choices: dict[str, str]
) -> DefineStep:
    choice = choices.get(step)
    valid = {o.id for o in options if o.available}
    return DefineStep(
        step=step, title=title, options=options, choice=choice if choice in valid else None
    )


def _image_family(template: str) -> str:
    if template == "small_cnn":
        return "scratch"
    return next((f for f, models in _IMG_BACKBONES.items() if template in models), "light")


def _defaults(
    card: ProfileCard, fitted: FittedPipeline, hardware: HardwareReport | None, can_download: bool
) -> dict[str, str]:
    """Elecciones que corresponden a la recomendación por reglas."""
    rec = recommend(card, fitted, hardware, allow_download=can_download)
    task = _task(fitted)
    imbalance = card.target.imbalance_ratio if card.target else None
    head = (
        "mse"
        if task is TaskType.REGRESSION
        else ("focal" if imbalance is not None and imbalance < IMBALANCED else "ce")
    )
    if card.modality is Modality.TABULAR:
        few = card.num_samples < SMALL_TABULAR
        return {
            "family": rec.template,
            "backbone": "small" if few else "medium",
            "head": head,
            "regularization": "high" if few else "medium",
        }
    return {
        "family": _image_family(rec.template),
        "backbone": "small_cnn:32" if rec.template == "small_cnn" else rec.template,
        "head": head,
        "regularization": "high" if card.num_samples < FEW_IMAGES else "medium",
    }


def _check(card: ProfileCard, fitted: FittedPipeline) -> TaskType:
    task = _task(fitted)
    if card.modality not in (Modality.TABULAR, Modality.IMAGE) or task not in _TASKS:
        raise ValueError(
            "el asistente de definición cubre clasificación y regresión tabular o de imágenes; "
            "para esta tarea usá las propuestas o el editor visual"
        )
    return task


def plan(
    card: ProfileCard,
    fitted: FittedPipeline,
    hardware: HardwareReport | None,
    choices: dict[str, str],
    *,
    allow_download: bool | None = None,
) -> DefinePlan:
    """Opciones de cada paso dadas las elecciones hechas (las siguientes dependen de la familia)."""
    task = _check(card, fitted)
    can_download = (not offline_mode()) if allow_download is None else allow_download
    rec = _defaults(card, fitted, hardware, can_download)
    if card.modality is Modality.TABULAR:
        family = _step("family", "Familia", _options(_TAB_FAMILIES, rec["family"], {}), choices)
        backbone = _step("backbone", "Tamaño", _options(_SIZE_TEXT, rec["backbone"], {}), choices)
    else:
        offline = "Sin conexión: no se pueden descargar pesos preentrenados."
        unavailable = {} if can_download else {"light": offline, "strong": offline}
        family = _step(
            "family", "Familia", _options(_IMG_FAMILIES, rec["family"], unavailable), choices
        )
        # Los backbones dependen de la familia elegida (o de la recomendada, si falta elegir).
        models = _IMG_BACKBONES[family.choice or rec["family"]]
        suggested = rec["backbone"] if rec["backbone"] in models else next(iter(models))
        backbone = _step("backbone", "Backbone", _options(models, suggested, {}), choices)
    losses = _REG_LOSSES if task is TaskType.REGRESSION else _LOSSES
    head_rec = rec["head"] if rec["head"] in losses else next(iter(losses))
    steps = [
        family,
        backbone,
        _step("head", "Cabeza y pérdida", _options(losses, head_rec, {}), choices),
        _step(
            "regularization",
            "Regularización",
            _options(_REGULARIZATION, rec["regularization"], {}),
            choices,
        ),
    ]
    return DefinePlan(
        modality=card.modality,
        task=task,
        steps=steps,
        complete=all(s.choice is not None for s in steps),
    )


def _loss(task: TaskType, head: str) -> LossSpec:
    if task is TaskType.REGRESSION:
        return LossSpec(type=head)  # type: ignore[arg-type]
    if head == "focal":
        return LossSpec(type="focal", class_weights="none")
    smoothing = 0.1 if head == "ce_smooth" else 0.0
    return LossSpec(
        type="cross_entropy",
        class_weights="auto",
        label_smoothing=HP(hp="label_smoothing", default=smoothing),
    )


def build(
    card: ProfileCard,
    fitted: FittedPipeline,
    hardware: HardwareReport | None,
    choices: dict[str, str],
    *,
    allow_download: bool | None = None,
) -> ArchSpec:
    """ArchSpec con las elecciones de los 4 pasos (todas deben ser opciones disponibles)."""
    p = plan(card, fitted, hardware, choices, allow_download=allow_download)
    if not p.complete:
        missing = [s.step for s in p.steps if s.choice is None]
        raise ValueError(f"faltan elecciones válidas en: {', '.join(missing)}")
    c = {s.step: s.choice or "" for s in p.steps}
    task = p.task
    num_classes = fitted.num_classes if task is not TaskType.REGRESSION else None
    picked = [
        f"{s.title.lower()}: {next(o.title for o in s.options if o.id == s.choice)}"
        for s in p.steps
    ]
    rationale = "Definida con el asistente (" + "; ".join(picked) + ")."
    if card.modality is Modality.TABULAR:
        family = c["family"]
        spec = tabular_template(
            family,
            task=task,
            num_classes=num_classes,
            num_numeric=len(fitted.numeric_features),
            cardinalities=[fitted.cardinalities[col] for col in fitted.categorical_features],
            rationale=rationale,
        )
        main = next(n for n in spec.nodes if n.id == _MAIN_NODE[family])
        for name, value in _TAB_SIZES[family][c["backbone"]].items():
            main.params[name] = HP(hp=name, default=value)
        dropout, wd = _TAB_REG[c["regularization"]]
        main.params["dropout"] = HP(hp="dropout", default=dropout)
        imbalance = card.target.imbalance_ratio if card.target else None
        spec.training.oversample = bool(imbalance is not None and imbalance < IMBALANCED)
    else:
        family, backbone = c["family"], c["backbone"]
        img = fitted.spec.image
        dropout, wd, freeze = _IMG_REG[c["regularization"]]
        pretrained = family != "scratch"
        spec = image_template(
            backbone.split(":")[0],
            task=task,
            num_classes=num_classes,
            image_size=img.size if img else 224,
            channels=img.channels if img else 3,
            pretrained=pretrained,
            freeze_epochs=freeze if pretrained else 0,
            rationale=rationale,
        )
        if not pretrained:
            encoder = next(n for n in spec.nodes if n.id == "encoder")
            encoder.params["width"] = HP(hp="width", default=int(backbone.split(":")[1]))
        drop = next(n for n in spec.nodes if n.id == "drop")
        drop.params["p"] = HP(hp="dropout", default=dropout)
    spec.loss = _loss(task, c["head"])
    spec.optimizer.weight_decay = HP(hp="weight_decay", default=wd)
    spec.name = f"{spec.name}-asistente"
    spec.provenance = spec.provenance.model_copy(
        update={"origin": Origin.MANUAL, "template": f"define:{spec.provenance.template}"}
    )
    return spec
