"""Wizard adaptativo (ADR-0040): ficha del caso, hechos de los datos y plan compilado.

Tres responsabilidades separadas:
- el LLM *entiende* el caso y propone cambios a la ficha (`BriefPatch`); nada entra sin que la
  persona lo acepte;
- `compile_plan` *decide* el plan con reglas puras sobre la ficha y hechos medibles de los
  datos (`DataFacts`), nunca con nombres de columnas ni de datasets;
- la persona *aprueba*: los defaults del plan se muestran como sugerencias con su «por qué».

Un caso que ninguna regla reconoce produce el plan estándar de 9 pasos (el comportamiento
anterior). Sin LLM (o en L0) la ficha se completa como formulario y corre el mismo compilador.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from perceptron.data.schema import SemanticType
from perceptron.data.view import Purpose
from perceptron.domain.enums import TaskType

if TYPE_CHECKING:
    from perceptron.data.profiling.card import ProfileCard
    from perceptron.data.view import DatasetView
    from perceptron.domain.models import DatasetVersion

Step = Literal[
    "goal",
    "data",
    "quality",
    "formula",
    "labeling",
    "task",
    "threshold",
    "architecture",
    "hpo",
    "budget",
    "review",
]
# Orden de todos los pasos. «formula» y «threshold» son condicionales (fase 2 del ADR-0040):
# solo aparecen si la ficha los justifica.
STEPS: tuple[Step, ...] = (
    "goal",
    "data",
    "quality",
    "formula",
    "labeling",
    "task",
    "threshold",
    "architecture",
    "hpo",
    "budget",
    "review",
)
CONDITIONAL: frozenset[Step] = frozenset({"formula", "threshold"})
PLAN_VERSION = 3  # sube cuando cambian las reglas: los planes guardados se recompilan
COLLINEAR = 0.98  # correlación de rangos entre entradas numéricas
FACTS_SAMPLE = 5000
FEW_ROWS = 100

Problem = Literal["value", "category", "anomaly", "forecast", "rule", "other"]
ErrorCosts = Literal["symmetric", "false_negative_worse", "false_positive_worse"]
Deployment = Literal["desktop", "server", "edge", "spreadsheet"]
Origin = Literal["user", "llm", "profile"]


class Assumption(BaseModel):
    text: str = Field(max_length=500)
    confidence: float = Field(ge=0, le=1)


class UseCaseBrief(BaseModel):
    """Lo que la persona sabe del caso, en términos que sirven para cualquier dataset."""

    model_config = ConfigDict(extra="forbid")

    problem: Problem | None = Field(
        default=None,
        description="value: predecir un número; category: una categoría; anomaly: detectar "
        "anomalías; forecast: pronosticar una serie; rule: descubrir una regla o fórmula; other",
    )
    problem_other: str | None = Field(default=None, max_length=500)
    prediction: str | None = Field(
        default=None, max_length=500, description="Qué se predice y en qué unidad"
    )
    error_costs: ErrorCosts | None = None
    error_cost_ratio: float | None = Field(
        default=None, ge=1, le=1000, description="Cuántas veces peor es el error más caro"
    )
    business_metric: str | None = Field(default=None, max_length=500)
    has_time: bool | None = Field(default=None, description="Los datos tienen fechas u orden")
    has_entities: bool | None = Field(
        default=None, description="Hay entidades repetidas (máquinas, clientes, pacientes)"
    )
    labels_available: bool | None = None
    independent_inputs: bool | None = Field(
        default=None, description="Las entradas varían por separado (no una función de otra)"
    )
    extrapolate: bool | None = Field(
        default=None, description="Se va a predecir fuera del rango de los datos"
    )
    explainability: bool | None = None
    deployment: Deployment | None = None
    max_latency_ms: float | None = Field(default=None, gt=0)
    assumptions: list[Assumption] = Field(default_factory=list, max_length=20)
    open_questions: list[str] = Field(default_factory=list, max_length=20)
    origins: dict[str, Origin] = Field(
        default_factory=dict, description="De dónde salió cada campo (historial)"
    )


BriefField = Literal[
    "problem",
    "problem_other",
    "prediction",
    "error_costs",
    "error_cost_ratio",
    "business_metric",
    "has_time",
    "has_entities",
    "labels_available",
    "independent_inputs",
    "extrapolate",
    "explainability",
    "deployment",
    "max_latency_ms",
]


# Guía corta de campos y valores para el LLM: más liviana que el JSON Schema completo (los
# modelos chicos locales se pierden con el schema y con el borrador entero).
BRIEF_GUIDE: dict[str, str] = {
    "problem": '"value" (predecir un número) | "category" (predecir una clase) | "anomaly" '
    '(detectar algo raro) | "forecast" (pronosticar una serie) | "rule" (descubrir la fórmula '
    'o regla que genera el dato) | "other" (nada de lo anterior)',
    "problem_other": "texto: qué quiere lograr, si problem es other",
    "prediction": "texto: qué se predice y en qué unidad",
    "error_costs": '"symmetric" | "false_negative_worse" (no detectar un caso es peor) | '
    '"false_positive_worse" (una falsa alarma es peor)',
    "error_cost_ratio": "número ≥ 1: cuántas veces peor es el error más caro",
    "business_metric": "texto: cómo mide el éxito el negocio",
    "has_time": "true | false: los datos tienen fechas u orden temporal",
    "has_entities": "true | false: hay entidades repetidas (máquinas, clientes, pacientes)",
    "labels_available": "true | false: ya tiene las etiquetas o el valor a predecir",
    "independent_inputs": "true | false: las entradas varían por separado",
    "extrapolate": "true | false: va a predecir fuera del rango de los datos",
    "explainability": "true | false: hay que explicar las predicciones",
    "deployment": '"desktop" | "server" | "edge" | "spreadsheet"',
    "max_latency_ms": "número: latencia máxima aceptable en milisegundos",
}


class BriefChange(BaseModel):
    field: BriefField
    value: Any
    rationale: str = Field(max_length=500)


class BriefPatch(BaseModel):
    """Lo que el LLM entendió de la conversación: cambios propuestos y la próxima pregunta."""

    changes: list[BriefChange] = Field(default_factory=list, max_length=14)
    assumptions: list[Assumption] = Field(default_factory=list, max_length=10)
    next_question: str | None = Field(
        default=None, max_length=500, description="La pregunta más útil que falta (o null)"
    )


def validate_patch(patch: BriefPatch) -> str | None:
    """Cada cambio tiene que ser un valor válido de la ficha (feedback para el reintento)."""
    errors = []
    for c in patch.changes:
        try:
            UseCaseBrief.model_validate({c.field: c.value})
        except ValueError as e:
            errors.append(f"{c.field}: {str(e).splitlines()[-1]}")
    return "\n".join(errors) or None


def normalize_patch(patch: BriefPatch) -> BriefPatch:
    """Cada valor con el tipo real del campo («true» → true, «5» → 5.0): los modelos chicos
    suelen devolver booleanos o números como texto; la validación los acepta, pero la UI y quien
    aplique el cambio necesitan el tipo correcto."""
    changes = []
    for c in patch.changes:
        typed = getattr(UseCaseBrief.model_validate({c.field: c.value}), c.field)
        changes.append(c.model_copy(update={"value": typed}))
    return patch.model_copy(update={"changes": changes})


def prefers_linear(brief: UseCaseBrief | dict[str, Any] | None) -> bool:
    """La ficha pide una regla o extrapolar (fuera de clasificación): lineal y fórmula primero,
    porque las redes aproximan dentro del rango visto. Lo usan el plan y el agente."""
    if brief is None:
        return False
    b = (
        brief
        if isinstance(brief, UseCaseBrief)
        else UseCaseBrief.model_validate(
            {k: v for k, v in brief.items() if k in UseCaseBrief.model_fields}
        )
    )
    return b.problem == "rule" or (
        bool(b.extrapolate) and b.problem not in ("category", "anomaly", "other")
    )


def contextualize_card(card: ProfileCard, use_case: dict[str, Any] | None) -> ProfileCard:
    """Ajusta las alertas del perfil a la ficha del proyecto (al leerlo, no se guarda).

    Si el objetivo es una regla o hay que extrapolar, una columna que determina el objetivo es
    lo esperable: «probable fuga» (alta) pasa a «relación determinística» (atención). Antes,
    en «Sensores», la alerta falsa distraía al agente y a la revisión de la ficha.
    """
    from perceptron.data.profiling.card import AlertCode, AlertSeverity

    if not prefers_linear(use_case):
        return card
    alerts = [
        a.model_copy(
            update={
                "code": AlertCode.DETERMINISTIC_RELATION,
                "severity": AlertSeverity.WARNING,
                "message": "La columna determina el objetivo casi por completo. La ficha dice "
                "que buscás una regla o que vas a extrapolar: una relación así es lo esperable. "
                "Solo sería una fuga si el valor se conoce recién después del resultado.",
            }
        )
        if a.code is AlertCode.TARGET_LEAKAGE
        else a
        for a in card.alerts
    ]
    return card.model_copy(update={"alerts": alerts})


def apply_patch(brief: UseCaseBrief, changes: list[BriefChange], origin: Origin) -> UseCaseBrief:
    data = brief.model_dump()
    for c in changes:
        data[c.field] = c.value
        data["origins"][c.field] = origin
    return UseCaseBrief.model_validate(data)


# ------------------------------------------------------------------ hechos de los datos


class DataFacts(BaseModel):
    """Hechos medibles del dataset elegido: lo único que el compilador mira de los datos."""

    dataset_version_id: str
    modality: str
    rows: int
    split_strategy: str | None = None
    target: str | None = None
    target_task: TaskType | None = None
    target_classes: int | None = None
    imbalance_ratio: float | None = None
    numeric_inputs: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    collinear_pairs: list[list[str]] = Field(
        default_factory=list, description="Pares de entradas que varían juntas"
    )


def _collinear_pairs(df: pl.DataFrame, cols: list[str]) -> list[list[str]]:
    if len(cols) < 2 or df.height < 3:
        return []
    ranks = df.select(pl.col(c).cast(pl.Float64).rank() for c in cols).drop_nulls()
    if ranks.height < 3:
        return []
    with np.errstate(all="ignore"):
        corr = np.corrcoef(ranks.to_numpy(), rowvar=False)
    return [
        [cols[i], cols[j]]
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= COLLINEAR
    ]


def data_facts(dv: DatasetVersion, view: DatasetView, card: ProfileCard | None) -> DataFacts:
    schema = view.schema
    numeric = [c.name for c in schema.feature_columns if c.semantic is SemanticType.NUMERIC]
    dates = [c.name for c in schema.columns if c.semantic is SemanticType.DATETIME]
    pairs: list[list[str]] = []
    if 2 <= len(numeric) <= 50:
        train = view.read("train", purpose=Purpose.TRAINING).select(numeric)
        if train.height > FACTS_SAMPLE:
            train = train.sample(FACTS_SAMPLE, seed=0)
        pairs = _collinear_pairs(train, numeric)
    target = card.target if card else None
    return DataFacts(
        dataset_version_id=dv.id,
        modality=dv.modality.value if dv.modality else "tabular",
        rows=dv.num_samples,
        split_strategy=dv.split.strategy.value if dv.split else None,
        target=dv.target,
        target_task=target.task_hint if target else None,
        target_classes=len(target.classes) if target and target.classes else None,
        imbalance_ratio=target.imbalance_ratio if target else None,
        numeric_inputs=numeric,
        datetime_columns=dates,
        collinear_pairs=pairs,
    )


# ------------------------------------------------------------------ plan


class PlanStep(BaseModel):
    id: Step
    reason: str | None = Field(default=None, description="Por qué el plan incluye o saltea el paso")


class PlanDefault(BaseModel):
    key: Literal["task", "target_metric", "architecture_hint"]
    value: str
    step: Step
    reason: str


class PlanCheck(BaseModel):
    code: str
    severity: Literal["info", "warning", "high"]
    step: Step
    message: str


class WizardPlan(BaseModel):
    steps: list[PlanStep]
    skipped: list[PlanStep] = Field(default_factory=list)
    defaults: list[PlanDefault] = Field(default_factory=list)
    checks: list[PlanCheck] = Field(default_factory=list)
    adapted: bool = Field(default=False, description="False: plan estándar (nada reconocido)")
    version: int = Field(default=1, description="Versión de las reglas que lo compilaron")


class PlanDiff(BaseModel):
    """Qué cambió del plan respecto del anterior (para mostrarlo en la UI)."""

    added_steps: list[str] = Field(default_factory=list)
    removed_steps: list[str] = Field(default_factory=list)
    changed_defaults: list[str] = Field(default_factory=list)
    new_checks: list[str] = Field(default_factory=list, description="Mensajes de avisos nuevos")
    resolved_checks: list[str] = Field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (
            self.added_steps
            or self.removed_steps
            or self.changed_defaults
            or self.new_checks
            or self.resolved_checks
        )


def diff_plans(old: WizardPlan | None, new: WizardPlan) -> PlanDiff:
    if old is None:
        return PlanDiff()
    old_steps, new_steps = [s.id for s in old.steps], [s.id for s in new.steps]
    old_defaults = {d.key: d.value for d in old.defaults}
    new_defaults = {d.key: d.value for d in new.defaults}
    old_checks = {c.code: c.message for c in old.checks}
    new_checks = {c.code: c.message for c in new.checks}
    return PlanDiff(
        added_steps=[s for s in new_steps if s not in old_steps],
        removed_steps=[s for s in old_steps if s not in new_steps],
        changed_defaults=sorted(
            k
            for k in set(old_defaults) | set(new_defaults)
            if old_defaults.get(k) != new_defaults.get(k)
        ),
        new_checks=[m for c, m in new_checks.items() if c not in old_checks],
        resolved_checks=[m for c, m in old_checks.items() if c not in new_checks],
    )


_TASK_OF: dict[str, TaskType] = {
    "value": TaskType.REGRESSION,
    "rule": TaskType.REGRESSION,
    "category": TaskType.CLASSIFICATION,
    "anomaly": TaskType.ANOMALY_DETECTION,
    "forecast": TaskType.FORECASTING,
}
_PROBLEM_LABEL = {
    "value": "predecir un valor",
    "rule": "descubrir una regla",
    "category": "predecir una categoría",
    "anomaly": "detectar anomalías",
    "forecast": "pronosticar",
}


def _pairs(pairs: list[list[str]]) -> str:
    return ", ".join(f"«{a}» y «{b}»" for a, b in pairs[:5])


def compile_plan(brief: UseCaseBrief | None, facts: DataFacts | None) -> WizardPlan:
    """Plan del wizard a partir de la ficha y los hechos medibles (puro y determinístico)."""
    b = brief or UseCaseBrief()
    defaults: list[PlanDefault] = []
    checks: list[PlanCheck] = []
    reasons: dict[str, str] = {}
    skipped: list[PlanStep] = []

    # --- etiquetado: solo si faltan etiquetas
    if (facts and facts.target) or b.labels_available:
        why = (
            f"Ya hay etiquetas: el objetivo es «{facts.target}»."
            if facts and facts.target
            else "En la ficha: las etiquetas ya están disponibles."
        )
        skipped.append(PlanStep(id="labeling", reason=why))

    # --- tarea y métrica
    if b.problem in _TASK_OF:
        task = _TASK_OF[b.problem]
        defaults.append(
            PlanDefault(
                key="task",
                value=task.value,
                step="task",
                reason=f"En la ficha: el problema es {_PROBLEM_LABEL[b.problem]}.",
            )
        )
        if facts and facts.target_task and facts.target_task is not task:
            checks.append(
                PlanCheck(
                    code="task_mismatch",
                    severity="warning",
                    step="task",
                    message=f"La ficha dice «{_PROBLEM_LABEL[b.problem]}», pero el objetivo "
                    f"«{facts.target}» parece de {facts.target_task.value}. Revisá cuál es el "
                    "objetivo o el tipo de problema.",
                )
            )
    if b.problem == "category" or (facts and facts.target_task is TaskType.CLASSIFICATION):
        if b.error_costs == "false_negative_worse":
            ratio = f" ({b.error_cost_ratio:g}× peor)" if b.error_cost_ratio else ""
            defaults.append(
                PlanDefault(
                    key="target_metric",
                    value="val_recall_macro",
                    step="task",
                    reason=f"En la ficha: no detectar un caso es peor que una falsa alarma{ratio}. "
                    "El recall mide cuántos casos reales se detectan.",
                )
            )
        elif b.error_costs == "false_positive_worse" or (
            facts and facts.imbalance_ratio is not None and facts.imbalance_ratio < 0.2
        ):
            why = (
                "En la ficha: una falsa alarma es peor que no detectar un caso."
                if b.error_costs == "false_positive_worse"
                else "Las clases están desbalanceadas: la exactitud engaña."
            )
            defaults.append(
                PlanDefault(
                    key="target_metric",
                    value="val_f1_macro",
                    step="task",
                    reason=why + " F1 equilibra precisión y recall.",
                )
            )
    if b.problem in ("value", "rule"):
        defaults.append(
            PlanDefault(
                key="target_metric",
                value="val_mae",
                step="task",
                reason="Error medio en las unidades del dato: se lee directo.",
            )
        )

    # --- reglas y extrapolación
    if prefers_linear(b):
        why = (
            "En la ficha: el objetivo es descubrir una regla."
            if b.problem == "rule"
            else "En la ficha: vas a predecir fuera del rango de los datos."
        )
        defaults.append(
            PlanDefault(
                key="architecture_hint",
                value="linear_first",
                step="architecture",
                reason=why + " Las redes aproximan dentro del rango visto y no extrapolan: "
                "elegí primero la propuesta lineal y probá la fórmula sugerida en Experimentos.",
            )
        )
        reasons["architecture"] = why
    if facts and facts.collinear_pairs and (b.problem == "rule" or b.independent_inputs):
        declared = b.independent_inputs is True
        checks.append(
            PlanCheck(
                code="collinear_inputs",
                severity="high" if declared else "warning",
                step="data",
                message=(
                    "Dijiste que las entradas son independientes, pero "
                    if declared
                    else "Hay entradas que varían juntas: "
                )
                + f"{_pairs(facts.collinear_pairs)} tienen una correlación de rangos ≥ "
                f"{COLLINEAR:g}. Con estos datos muchas fórmulas explican igual el objetivo y la "
                "regla real no se puede inferir: hacen falta datos donde varíen por separado.",
            )
        )
        reasons["data"] = "Hay que revisar las entradas antes de entrenar."

    # --- particiones
    if facts and b.has_time and facts.split_strategy not in ("temporal", None):
        checks.append(
            PlanCheck(
                code="split_not_temporal",
                severity="warning",
                step="data",
                message="En la ficha: los datos tienen orden temporal, pero la partición es "
                f"«{facts.split_strategy}». Para no evaluar con el futuro, volvé a cargar los "
                "datos con partición temporal.",
            )
        )
    if facts and b.has_entities and facts.split_strategy not in ("group", None):
        checks.append(
            PlanCheck(
                code="split_not_group",
                severity="warning",
                step="data",
                message="En la ficha: hay entidades repetidas, pero la partición es "
                f"«{facts.split_strategy}». La misma entidad puede quedar en entrenamiento y en "
                "test e inflar las métricas: usá partición por grupo.",
            )
        )
    if facts and facts.rows < FEW_ROWS and b.problem not in (None, "rule", "other"):
        checks.append(
            PlanCheck(
                code="few_rows",
                severity="info",
                step="data",
                message=f"Hay {facts.rows} filas: las métricas van a variar mucho entre corridas.",
            )
        )

    # --- fuera del catálogo y uso
    if b.problem == "other":
        checks.append(
            PlanCheck(
                code="out_of_catalog",
                severity="info",
                step="goal",
                message="Este tipo de problema no tiene pasos específicos en Perceptron: se usa "
                "el flujo estándar. Si no es predecir un valor o una categoría, detectar "
                "anomalías o pronosticar, puede que Perceptron no sea la herramienta indicada.",
            )
        )
    if b.explainability:
        reasons["review"] = (
            "En la ficha: hay que explicar el modelo. Después de entrenar, la evaluación "
            "avanzada muestra la contribución de cada variable."
        )
    if b.deployment == "spreadsheet":
        checks.append(
            PlanCheck(
                code="spreadsheet",
                severity="info",
                step="architecture",
                message="En la ficha: el resultado se usa en una planilla. Una red no se puede "
                "copiar a Excel; la fórmula sugerida sí (si existe una fórmula simple).",
            )
        )

    # --- reconciliación por reglas: lo que la ficha dice contra lo que se mide
    if facts and facts.datetime_columns:
        cols = ", ".join(f"«{c}»" for c in facts.datetime_columns[:3])
        if b.has_time is False:
            checks.append(
                PlanCheck(
                    code="dates_contradiction",
                    severity="warning",
                    step="data",
                    message=f"La ficha dice que no hay orden temporal, pero hay fechas ({cols}). "
                    "Si los datos se juntaron en el tiempo, evaluar con una partición aleatoria "
                    "puede mezclar el futuro con el pasado.",
                )
            )
        elif b.has_time is None:
            checks.append(
                PlanCheck(
                    code="dates_undeclared",
                    severity="info",
                    step="data",
                    message=f"Hay columnas de fecha ({cols}). ¿Los datos tienen orden temporal? "
                    "Si es así, indicalo en la ficha: cambia cómo conviene evaluar.",
                )
            )
    if facts and b.labels_available is False and facts.target:
        checks.append(
            PlanCheck(
                code="labels_contradiction",
                severity="info",
                step="data",
                message=f"La ficha dice que faltan etiquetas, pero los datos tienen el objetivo "
                f"«{facts.target}». Si esa columna es la etiqueta, corregí la ficha.",
            )
        )

    # --- pasos condicionales
    include: set[Step] = set()
    tabular_regression = facts is None or (
        facts.modality == "tabular"
        and facts.target_task in (None, TaskType.REGRESSION)
        and bool(facts.numeric_inputs)
    )
    if prefers_linear(b) and tabular_regression:
        include.add("formula")
        reasons["formula"] = (
            "En la ficha: el objetivo es descubrir una regla. "
            if b.problem == "rule"
            else "En la ficha: vas a predecir fuera del rango de los datos. "
        ) + (
            "Antes de entrenar redes, buscá una fórmula: si existe, la vas a ver escrita y "
            "extrapola; una red no."
        )
    asymmetric = b.error_costs in ("false_negative_worse", "false_positive_worse")
    regression = b.problem in ("value", "rule", "forecast") or (
        facts is not None and facts.target_task in (TaskType.REGRESSION, TaskType.FORECASTING)
    )
    if asymmetric and regression:
        checks.append(
            PlanCheck(
                code="costs_ignored",
                severity="info",
                step="task",
                message="En la ficha hay un error más caro que el otro, pero eso se usa para "
                "elegir un umbral de decisión en clasificación. En regresión no se aplica: el "
                "modelo minimiza el error medio en las dos direcciones.",
            )
        )
    classification = b.problem == "category" or (
        facts is not None and facts.target_task is TaskType.CLASSIFICATION
    )
    binary = facts is None or facts.target_classes in (None, 2)
    if b.error_costs in ("false_negative_worse", "false_positive_worse") and classification:
        if binary:
            include.add("threshold")
            reasons["threshold"] = (
                "En la ficha: un error es peor que el otro. El umbral de decisión se elige con "
                "validación para minimizar el costo esperado, no el 50 %."
            )
        else:
            checks.append(
                PlanCheck(
                    code="threshold_multiclass",
                    severity="info",
                    step="task",
                    message="Los costos asimétricos se calibran con un umbral solo en "
                    "clasificación binaria; con más clases se usa la métrica sugerida.",
                )
            )

    skip_ids = {s.id for s in skipped}
    steps = [
        PlanStep(id=s, reason=reasons.get(s))
        for s in STEPS
        if s not in skip_ids and (s not in CONDITIONAL or s in include)
    ]
    adapted = bool(defaults or checks or skipped or reasons)
    return WizardPlan(
        steps=steps,
        skipped=skipped,
        defaults=defaults,
        checks=checks,
        adapted=adapted,
        version=PLAN_VERSION,
    )


def project_use_case(ctx: Any, project_id: str) -> dict[str, Any] | None:
    """La ficha aceptada del proyecto, sin vacíos, para el contexto de los otros roles LLM."""
    from perceptron.domain.models import ProjectDraft

    found = list(ctx.repo(ProjectDraft).list(filters={"project_id": project_id}, limit=1))
    brief = found[0].values.get("brief") if found else None
    if not brief:
        return None
    data = {k: v for k, v in brief.items() if v not in (None, [], {}) and k != "origins"}
    return data or None
