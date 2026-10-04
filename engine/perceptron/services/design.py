"""Requisitos de diseño por escenario y evaluación de las arquitecturas propuestas (ADR-0041).

El arquitecto (LLM o reglas) propone; el sistema decide qué exige el escenario y cuál propuesta
lo cumple mejor. Los requisitos salen de hechos medibles (modalidad, tarea, cantidad de
ejemplos de entrenamiento, desbalance, hardware, conexión) y de la ficha del caso de uso
(extrapolar, regla, despliegue, explicabilidad, costo de los errores):

- `must`: si ninguna propuesta lo cumple, el arquitecto reintenta con el motivo y, si sigue sin
  cumplirse, el sistema agrega una propuesta de plantilla que sí lo cumple;
- `should`: suma o resta en el puntaje, no bloquea.

`scope` dice si lo tiene que cumplir cada propuesta (`each`, p. ej. el tope de parámetros de un
equipo embebido) o alcanza con una (`any`, p. ej. «una opción preentrenada»).

Caso «Tubos»: 3 clases con pocas imágenes; el arquitecto propuso redes preentrenadas, pero el
wizard no recomendó ninguna y se eligió una CNN desde cero, la peor opción para tan pocos datos.
Ahora la preentrenada queda marcada como recomendada y elegir otra muestra el motivo.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from perceptron.archspec.defaults import is_linear_regression
from perceptron.domain.enums import Modality, TaskType
from perceptron.services.brief import UseCaseBrief, prefers_linear

if TYPE_CHECKING:
    from perceptron.archspec.schema import ArchSpec
    from perceptron.data.profiling.card import ProfileCard

Level = Literal["must", "should"]
Scope = Literal["each", "any"]
RequirementCode = Literal[
    "pretrained_backbone",
    "no_pretrained_offline",
    "small_model",
    "linear_option",
    "edge_size",
    "low_latency",
    "imbalance_handling",
    "explainable",
    "epoch_time",
]

PRETRAINED_BLOCKS = frozenset({"vision.timm_backbone", "text.hf_encoder"})
FEW_IMAGES_MUST = 1_000  # menos ejemplos de entrenamiento: desde cero casi siempre sobreajusta
FEW_IMAGES = 5_000
FEW_TEXTS = 5_000
TINY_IMAGE = 64
SMALL_TABULAR = 2_000
PARAMS_PER_ROW = 20  # tope de parámetros por fila en tablas chicas
MIN_PARAM_CAP = 10_000
EDGE_PARAMS = 5_000_000
LOW_LATENCY_MS = 20.0
LOW_LATENCY_PARAMS = 2_000_000
EXPLAINABLE_PARAMS = 50_000
IMBALANCED = 0.3  # minoritaria / mayoritaria
EPOCH_BUDGET_S = {"cpu": 300.0, "gpu": 120.0}
MUST_PENALTY = 45.0
SHOULD_PENALTY = 15.0


class DesignRequirement(BaseModel):
    code: RequirementCode
    level: Level
    scope: Scope
    message: str = Field(description="Qué se exige y por qué, en castellano (la UI lo traduce)")
    params: dict[str, float | int | str] = Field(default_factory=dict)


class DesignRequirements(BaseModel):
    scenario: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict, description="Hechos del escenario que justifican los requisitos"
    )
    items: list[DesignRequirement] = Field(default_factory=list)

    def musts(self) -> list[DesignRequirement]:
        return [r for r in self.items if r.level == "must"]

    def for_llm(self) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in self.items]


class RequirementCheck(BaseModel):
    code: RequirementCode
    level: Level
    met: bool


class DesignAssessment(BaseModel):
    score: float = Field(ge=0, le=100)
    recommended: bool = False
    checks: list[RequirementCheck] = Field(default_factory=list)


# ---------------------------------------------------------------------- requisitos


def _brief(use_case: dict[str, Any] | None) -> UseCaseBrief:
    data = {k: v for k, v in (use_case or {}).items() if k in UseCaseBrief.model_fields}
    try:
        return UseCaseBrief.model_validate(data)
    except ValueError:
        return UseCaseBrief()


def design_requirements(
    card: ProfileCard,
    task: TaskType,
    use_case: dict[str, Any] | None,
    *,
    device: str,
    allow_pretrained: bool,
    pretrained_text: bool = False,
    image_size: int | None = None,
) -> DesignRequirements:
    """Qué exige el escenario. Puro: mismos hechos, mismos requisitos."""
    brief = _brief(use_case)
    n_train = int(card.split_counts.get("train") or card.num_samples)
    imbalance = card.target.imbalance_ratio if card.target else None
    items: list[DesignRequirement] = []

    def add(code: RequirementCode, level: Level, scope: Scope, message: str, **params: Any) -> None:
        items.append(
            DesignRequirement(code=code, level=level, scope=scope, message=message, params=params)
        )

    supervised = task in (TaskType.CLASSIFICATION, TaskType.REGRESSION)
    if card.modality in (Modality.IMAGE, Modality.AUDIO) and supervised and n_train < FEW_IMAGES:
        what = "imágenes" if card.modality is Modality.IMAGE else "audios"
        if allow_pretrained:
            tiny = card.modality is Modality.IMAGE and (image_size or 224) <= TINY_IMAGE
            level: Level = "must" if n_train < FEW_IMAGES_MUST and not tiny else "should"
            add(
                "pretrained_backbone",
                level,
                "any",
                f"Con {n_train} {what} de entrenamiento, una red preentrenada (transfer learning) "
                "generaliza mucho mejor que una entrenada desde cero, que tiende a memorizar.",
                n_train=n_train,
            )
        else:
            add(
                "no_pretrained_offline",
                "should",
                "each",
                f"Sin conexión no hay pesos preentrenados: con {n_train} {what} conviene una red "
                "chica desde cero y aumentar los datos; esperá menos precisión.",
                n_train=n_train,
                max_params=LOW_LATENCY_PARAMS,
            )
    if card.modality is Modality.TEXT and supervised and n_train < FEW_TEXTS and pretrained_text:
        add(
            "pretrained_backbone",
            "should",
            "any",
            f"Con {n_train} textos, un encoder preentrenado aporta el conocimiento del idioma.",
            n_train=n_train,
        )
    if card.modality is Modality.TABULAR and n_train < SMALL_TABULAR:
        cap = max(MIN_PARAM_CAP, PARAMS_PER_ROW * n_train)
        add(
            "small_model",
            "should",
            "each",
            f"Con {n_train} filas, un modelo de más de {cap:,} parámetros memoriza en vez de "
            "aprender: conviene uno chico.".replace(",", "."),
            n_train=n_train,
            max_params=cap,
        )
    if card.modality is Modality.TABULAR and task is TaskType.REGRESSION and prefers_linear(brief):
        why = "buscás una regla" if brief.problem == "rule" else "vas a predecir fuera del rango"
        add(
            "linear_option",
            "must",
            "any",
            f"La ficha dice que {why}: las redes solo interpolan dentro de lo visto, así que hace "
            "falta una opción lineal (y conviene probar la regresión simbólica).",
        )
    if brief.deployment == "edge":
        add(
            "edge_size",
            "must",
            "each",
            "Va a correr en un equipo embebido: el modelo no puede superar los 5 millones de "
            "parámetros.",
            max_params=EDGE_PARAMS,
        )
    if brief.max_latency_ms is not None and brief.max_latency_ms <= LOW_LATENCY_MS:
        add(
            "low_latency",
            "should",
            "each",
            f"La respuesta tiene que llegar en {brief.max_latency_ms:g} ms: conviene un modelo "
            "liviano (hasta 2 millones de parámetros).",
            max_ms=brief.max_latency_ms,
            max_params=LOW_LATENCY_PARAMS,
        )
    if task is TaskType.CLASSIFICATION and (
        (imbalance is not None and imbalance < IMBALANCED)
        or brief.error_costs in ("false_negative_worse", "false_positive_worse")
    ):
        why = (
            f"las clases están desbalanceadas (ratio {imbalance:.2f})"
            if imbalance is not None and imbalance < IMBALANCED
            else "un tipo de error cuesta más que el otro"
        )
        add(
            "imbalance_handling",
            "should",
            "each",
            f"Como {why}, el entrenamiento tiene que compensarlo: pesos por clase, pérdida focal "
            "o sobremuestreo.",
        )
    if brief.explainability:
        add(
            "explainable",
            "should",
            "any",
            "Hay que poder explicar las decisiones: conviene una opción lineal o muy chica.",
            max_params=EXPLAINABLE_PARAMS,
        )
    budget = EPOCH_BUDGET_S["cpu" if device == "cpu" else "gpu"]
    add(
        "epoch_time",
        "should",
        "each",
        f"Cada época debería tardar menos de {budget / 60:g} minutos en este equipo, para poder "
        "probar varias configuraciones.",
        max_s=budget,
    )
    scenario: dict[str, float | int | str | bool | None] = {
        "modality": card.modality.value,
        "task": task.value,
        "n_train": n_train,
        "imbalance_ratio": imbalance,
        "device": device,
        "allow_pretrained": allow_pretrained,
        "deployment": brief.deployment,
    }
    return DesignRequirements(scenario=scenario, items=items)


# ---------------------------------------------------------------------- evaluación


def uses_pretrained(spec: ArchSpec) -> bool:
    return any(
        n.block in PRETRAINED_BLOCKS and n.params.get("pretrained", True) is not False
        for n in spec.nodes
    )


def handles_imbalance(spec: ArchSpec) -> bool:
    return (
        spec.loss.type == "focal" or spec.loss.class_weights != "none" or spec.training.oversample
    )


def is_linear(spec: ArchSpec) -> bool:
    blocks = [n.block for n in spec.nodes]
    return is_linear_regression(spec) or all(
        b == "head.linear" or b.startswith("input.") for b in blocks
    )


def meets(
    req: DesignRequirement, spec: ArchSpec, num_params: float | None, epoch_s: float | None
) -> bool:
    """Si una propuesta cumple el requisito. Sin estimación disponible, se da por cumplido."""
    cap = req.params.get("max_params")
    small = num_params is None or not isinstance(cap, int | float) or num_params <= cap
    match req.code:
        case "pretrained_backbone":
            return uses_pretrained(spec)
        case "no_pretrained_offline" | "small_model" | "edge_size" | "low_latency":
            return small
        case "linear_option":
            return is_linear(spec)
        case "imbalance_handling":
            return handles_imbalance(spec)
        case "explainable":
            return is_linear(spec) or small
        case "epoch_time":
            limit = req.params.get("max_s")
            return epoch_s is None or not isinstance(limit, int | float) or epoch_s <= limit


def assess(
    reqs: DesignRequirements,
    spec: ArchSpec,
    num_params: float | None,
    epoch_s: float | None,
    confidence: float | None = None,
) -> DesignAssessment:
    checks = [
        RequirementCheck(code=r.code, level=r.level, met=meets(r, spec, num_params, epoch_s))
        for r in reqs.items
    ]
    score = 95.0 + 5.0 * (confidence if confidence is not None else 0.5)
    for c in checks:
        if not c.met:
            score -= MUST_PENALTY if c.level == "must" else SHOULD_PENALTY
    return DesignAssessment(score=round(max(0.0, min(100.0, score)), 1), checks=checks)


def unmet_musts(
    reqs: DesignRequirements, evaluated: Sequence[tuple[str, ArchSpec, float | None]]
) -> list[str]:
    """Mensajes para el arquitecto con los `must` que su set de propuestas no cumple.

    `evaluated` = (título, spec, parámetros) de cada propuesta válida. El tiempo por época no
    entra: medirlo es caro y no es un `must`.
    """
    errors: list[str] = []
    for req in reqs.musts():
        ok = [meets(req, spec, params, None) for _, spec, params in evaluated]
        if req.scope == "any" and evaluated and not any(ok):
            errors.append(
                f"Requisito obligatorio sin cubrir ({req.code}): {req.message} "
                "Incluí al menos una propuesta que lo cumpla y ponela primera."
            )
        if req.scope == "each":
            errors += [
                f"La propuesta «{title}» no cumple un requisito obligatorio ({req.code}): "
                f"{req.message}"
                for (title, _, _), passed in zip(evaluated, ok, strict=True)
                if not passed
            ]
    return errors


def missing_any_musts(
    reqs: DesignRequirements, specs: Sequence[ArchSpec]
) -> list[DesignRequirement]:
    """`must` de alcance `any` que ninguna propuesta cumple (se completan con una plantilla)."""
    return [
        r
        for r in reqs.musts()
        if r.scope == "any" and not any(meets(r, s, None, None) for s in specs)
    ]


def recommend_index(assessments: Sequence[DesignAssessment]) -> int | None:
    """La de mayor puntaje; ante empate, la primera (el orden del arquitecto)."""
    if not assessments:
        return None
    return max(range(len(assessments)), key=lambda i: (assessments[i].score, -i))


def supplement_spec(
    req: DesignRequirement, base: ArchSpec, device: str
) -> tuple[str, ArchSpec] | None:
    """Plantilla que cumple un `must` que el arquitecto no cubrió: (título, spec), o None si no
    hay plantilla para esa entrada. `input` y `task` salen de la propuesta base (los datos)."""
    from perceptron.archspec.defaults import without_linear_shrinkage
    from perceptron.catalog.templates import image_template, tabular_template

    task, inp = base.task, base.input
    if req.code == "linear_option" and inp.kind == "tabular":
        spec = tabular_template(
            "linear",
            task=task.type,
            num_classes=task.num_classes,
            num_numeric=inp.num_numeric or 0,
            cardinalities=list(inp.cardinalities or []),
            rationale=req.message,
        )
        return "Regresión lineal", without_linear_shrinkage(spec.model_copy(update={"input": inp}))
    if req.code == "pretrained_backbone" and inp.kind == "image" and inp.shape:
        backbone = "mobilenetv3_small_100" if device == "cpu" else "efficientnet_b0"
        spec = image_template(
            backbone,
            task=task.type,
            num_classes=task.num_classes,
            image_size=inp.shape[-1],
            channels=inp.shape[0],
            pretrained=True,
            freeze_epochs=3,
            rationale=req.message,
        )
        title = "MobileNetV3 preentrenada" if device == "cpu" else "EfficientNet-B0 preentrenada"
        return title, spec.model_copy(update={"input": inp})
    return None
