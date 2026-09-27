"""Pipeline automático a partir del ProfileCard (RF-PIP-01).

Cada decisión queda explicada en `rationale` (la UI la muestra en "¿Por qué?").
"""

from __future__ import annotations

from perceptron.data.pipeline.pipeline import AugmentSpec, ImageSpec, PipelineSpec, TargetSpec
from perceptron.data.pipeline.steps import StepSpec
from perceptron.data.profiling.card import ColumnProfile, ProfileCard
from perceptron.data.schema import SemanticType
from perceptron.domain.enums import Modality, TaskType

SKEW_LOG = 2.0
OUTLIERS_ROBUST = 0.05
ONE_HOT_MAX = 10


def _needs_log(c: ColumnProfile) -> bool:
    n = c.numeric
    return bool(
        n and n.skew is not None and n.skew > SKEW_LOG and (n.quantiles.get("p05") or 0) >= 0
    )


def propose_pipeline(
    card: ProfileCard, *, image_size: int | None = None, pretrained: bool = False
) -> PipelineSpec:
    target = (
        TargetSpec(
            name=card.target.name,
            task=card.target.task_hint,
            standardize=card.target.task_hint is TaskType.REGRESSION,
        )
        if card.target
        else None
    )
    if card.modality is Modality.IMAGE:
        return _propose_image(card, target, image_size, pretrained)
    if card.modality is not Modality.TABULAR:
        raise NotImplementedError(f"pipeline para {card.modality} llega en Capa 1b")

    why: list[str] = []
    steps: list[StepSpec] = []

    def add(kind: str, cols: list[str], reason: str, **params: object) -> None:
        if cols:
            steps.append(
                StepSpec(id=f"s{len(steps) + 1}_{kind}", kind=kind, columns=cols, params=params)
            )
            why.append(reason)

    by_sem: dict[SemanticType, list[ColumnProfile]] = {}
    for c in card.columns:
        by_sem.setdefault(c.semantic, []).append(c)

    dropped = [
        c.name
        for s in (SemanticType.ID, SemanticType.TEXT, SemanticType.FILEPATH)
        for c in by_sem.get(s, [])
    ]
    add("drop", dropped, "Se descartan ids, texto libre y rutas (el texto se modela en Capa 1b).")
    constant = [c.name for c in card.columns if c.n_unique <= 1 and c.name not in dropped]
    add("drop", constant, "Se descartan columnas constantes.")

    dates = [c.name for c in by_sem.get(SemanticType.DATETIME, [])]
    add("date_features", dates, "Las fechas se convierten en año, mes, día de la semana y del año.")

    booleans = [c.name for c in by_sem.get(SemanticType.BOOLEAN, []) if c.name not in constant]
    add("to_numeric", booleans, "Las columnas binarias se codifican como 0/1.")

    numeric = [c for c in by_sem.get(SemanticType.NUMERIC, []) if c.name not in constant]
    num_names = (
        [c.name for c in numeric]
        + booleans
        + [f"{d}.{p}" for d in dates for p in ("year", "month", "weekday", "ordinal_day")]
    )
    with_nulls = [c.name for c in numeric if c.null_fraction > 0] + [
        c.name
        for c in by_sem.get(SemanticType.BOOLEAN, [])
        if c.null_fraction > 0 and c.name in booleans
    ]
    add(
        "impute_numeric",
        with_nulls,
        "Los faltantes numéricos se completan con la mediana de train.",
        strategy="median",
    )
    skewed = [c.name for c in numeric if _needs_log(c)]
    add("log1p", skewed, "Variables muy asimétricas y no negativas: se aplica log(1 + x).")
    robust = [c.name for c in numeric if c.numeric and c.numeric.outlier_fraction > OUTLIERS_ROBUST]
    standard = [n for n in num_names if n not in robust]
    add(
        "scale",
        robust,
        "Variables con muchos outliers: escalado robusto (mediana/IQR).",
        method="robust",
    )
    add(
        "scale",
        standard,
        "El resto de las numéricas se estandariza (media 0, desvío 1).",
        method="standard",
    )

    cats = [c for c in by_sem.get(SemanticType.CATEGORICAL, []) if c.name not in constant]
    add(
        "impute_categorical",
        [c.name for c in cats],
        "Los faltantes categóricos se tratan como una categoría propia.",
    )
    add(
        "ordinal",
        [c.name for c in cats],
        "Las categóricas se codifican como índices para embeddings aprendidas.",
    )
    return PipelineSpec(modality=Modality.TABULAR, target=target, steps=steps, rationale=why)


def _propose_image(
    card: ProfileCard, target: TargetSpec | None, image_size: int | None, pretrained: bool
) -> PipelineSpec:
    img = card.images
    p50 = (img.width_quantiles.get("p50") if img else None) or 224
    if image_size is None:
        image_size = 224 if pretrained else int(min(224, max(32, round(p50 / 32) * 32)))
    gray = bool(img and img.channels and img.channels[0].value == "1" and len(img.channels) == 1)
    why = [
        f"Las imágenes se redimensionan a {image_size}×{image_size}.",
        "Normalización ImageNet (compatible con backbones preentrenados)."
        if pretrained
        else "Normalización con la media y el desvío de train.",
        "Augmentations leves en train: espejado, rotación ±10°, variación de color y recorte.",
    ]
    return PipelineSpec(
        modality=Modality.IMAGE,
        target=target,
        image=ImageSpec(
            size=image_size,
            channels=1 if gray else 3,
            normalize="imagenet" if pretrained else "dataset",
            augment=[
                AugmentSpec(kind="hflip"),
                AugmentSpec(kind="rotation", params={"degrees": 10}),
                AugmentSpec(kind="color_jitter", params={"strength": 0.1}),
                AugmentSpec(kind="random_resized_crop", params={"scale": [0.85, 1.0]}),
            ],
        ),
        rationale=why,
    )
