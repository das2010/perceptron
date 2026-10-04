"""Wizard de creación de la red (SPEC §7.6): borrador versionado y copiloto (RF-WIZ-01..04).

El `ProjectDraft` guarda en qué paso está el usuario y los valores elegidos. El copiloto
conversa en streaming y, aparte, propone un `DraftPatch` estructurado que solo puede tocar
campos conocidos del borrador; la UI lo muestra en violeta y el usuario lo acepta o lo
rechaza (CLAUDE.md: toda salida del LLM que modifica el sistema es aceptable/rechazable).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from perceptron.core.errors import ConflictError, ValidationError
from perceptron.domain.enums import LLMPurpose, TaskType
from perceptron.domain.models import ProjectDraft
from perceptron.llm.privacy import LLMContext
from perceptron.llm.types import Message
from perceptron.services.brief import (
    BRIEF_GUIDE,
    PLAN_VERSION,
    STEPS,
    BriefPatch,
    DataFacts,
    PlanDiff,
    UseCaseBrief,
    WizardPlan,
    compile_plan,
    data_facts,
    diff_plans,
    normalize_patch,
    validate_patch,
)

if TYPE_CHECKING:
    from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 200
CHAT_TURNS = 8


class DraftValues(BaseModel):
    """Lo que el wizard va definiendo. Todo opcional: se completa paso a paso."""

    model_config = ConfigDict(extra="forbid")

    goal: str | None = Field(default=None, max_length=4000)
    dataset_version_id: str | None = None
    target: str | None = None
    task: TaskType | None = None
    target_metric: str | None = Field(default=None, description="p. ej. val_roc_auc, val_f1_macro")
    success_threshold: float | None = Field(
        default=None, description="Umbral de éxito del negocio traducido a la métrica técnica"
    )
    pipeline_id: str | None = None
    archspec_id: str | None = None
    strategy: dict[str, Any] | None = Field(default=None, description="HPOStrategy elegida")
    max_trials: int | None = Field(default=None, ge=1, le=500)
    max_epochs_per_trial: int | None = Field(default=None, ge=1, le=1000)
    max_time_s: float | None = Field(default=None, gt=0)
    device: Literal["cpu", "cuda", "rocm", "xpu", "mps"] | None = None
    autonomous: bool | None = Field(default=None, description="Lanzar con el agente autónomo")
    llm_budget_usd: float | None = Field(default=None, ge=0)
    # Wizard adaptativo (ADR-0040). La ficha la edita la persona (o acepta lo del LLM); los
    # hechos y el plan los calcula el sistema.
    brief: UseCaseBrief | None = None
    data_facts: DataFacts | None = None
    plan: WizardPlan | None = None
    plan_diff: PlanDiff | None = Field(
        default=None, description="Qué cambió en la última recompilación del plan"
    )


SYSTEM_FIELDS = ("data_facts", "plan", "plan_diff")


# Campos que el copiloto puede proponer cambiar (los ids los fija la UI al elegir).
COPILOT_FIELDS = (
    "goal",
    "target",
    "task",
    "target_metric",
    "success_threshold",
    "max_trials",
    "max_epochs_per_trial",
    "max_time_s",
    "autonomous",
)
CopilotField = Literal[
    "goal",
    "target",
    "task",
    "target_metric",
    "success_threshold",
    "max_trials",
    "max_epochs_per_trial",
    "max_time_s",
    "autonomous",
]


class DraftChange(BaseModel):
    field: CopilotField
    value: Any
    rationale: str


class DraftPatch(BaseModel):
    """Cambios sugeridos por el copiloto al borrador (SPEC §7.7.4, "Copiloto del wizard")."""

    changes: list[DraftChange] = Field(default_factory=list, max_length=6)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Wizard:
    def __init__(self, wf: Workflow) -> None:
        self.wf = wf
        self.ctx = wf.ctx
        self.repo = self.ctx.repo(ProjectDraft)

    # ------------------------------------------------------------------ borrador

    def get(self, project_id: str) -> ProjectDraft:
        project = self.wf.project(project_id)
        found = list(self.repo.list(filters={"project_id": project_id}, limit=1))
        if found:
            return found[0]
        values = DraftValues(goal=project.goal or None).model_dump(mode="json", exclude_none=True)
        # El plan nace con el borrador: así no figura como cambio en la primera edición.
        values["plan"] = compile_plan(None, None).model_dump(mode="json", exclude_none=True)
        draft = ProjectDraft(project_id=project_id, values=values)
        return self.repo.add(draft)

    def update(
        self,
        project_id: str,
        *,
        version: int,
        values: dict[str, Any] | None = None,
        step: str | None = None,
        origin: Literal["user", "copilot"] = "user",
    ) -> ProjectDraft:
        draft = self.get(project_id)
        if draft.version != version:
            raise ConflictError(
                "el borrador cambió; recargalo", details={"current_version": draft.version}
            )
        if step is not None and step not in STEPS:
            raise ValidationError(f"paso desconocido: {step}", details={"steps": list(STEPS)})
        if any(k in SYSTEM_FIELDS for k in values or {}):
            raise ValidationError(
                "los hechos de los datos y el plan los calcula el sistema",
                details={"fields": [k for k in values or {} if k in SYSTEM_FIELDS]},
            )
        merged = {**draft.values, **(values or {})}
        if "brief" in (values or {}) and values and values["brief"] is not None:
            merged["brief"] = self._with_origins(draft.values.get("brief"), values["brief"], origin)
        try:
            clean = DraftValues.model_validate(merged).model_dump(mode="json", exclude_none=True)
        except PydanticValidationError as e:
            raise ValidationError(
                "valores inválidos para el borrador", details={"errors": e.errors()}
            ) from e
        # Lo guardado pasa por la misma serialización (sin nulos anidados): si no, el plan
        # parecería distinto aunque sea igual y figuraría como cambio.
        before = DraftValues.model_validate(draft.values).model_dump(mode="json", exclude_none=True)
        clean = self._replan(before, clean)
        changed = sorted(k for k in clean if before.get(k) != clean.get(k))
        history = draft.history
        if changed or step:
            entry = {"ts": _now(), "origin": origin, "fields": changed, "step": step}
            history = [*history, entry][-HISTORY_LIMIT:]
        updated = draft.model_copy(
            update={"values": clean, "step": step or draft.step, "history": history}
        )
        result = self.repo.update(updated)
        if clean.get("goal") is not None:
            project = self.wf.project(project_id)
            if project.goal != clean["goal"]:
                self.ctx.projects.update(project.model_copy(update={"goal": clean["goal"]}))
        return result

    # ------------------------------------------------------------------ ficha y plan

    @staticmethod
    def _with_origins(
        old: dict[str, Any] | None, new: dict[str, Any], origin: str
    ) -> dict[str, Any]:
        """Marca de dónde salió cada campo que cambió (persona o LLM aceptado)."""
        before = old or {}
        origins = dict(before.get("origins") or {})
        for key, value in new.items():
            if key not in ("origins", "assumptions", "open_questions") and before.get(key) != value:
                origins[key] = "llm" if origin == "copilot" else "user"
        return {**new, "origins": origins}

    def _facts(self, dataset_version_id: str) -> dict[str, Any] | None:
        try:
            dv = self.wf.dataset(dataset_version_id)
            card = self.wf.profile_card(dataset_version_id)
            facts = data_facts(dv, self.wf.view(dv), card)
        except Exception:  # un dataset que no se puede perfilar no rompe el wizard
            logger.warning("sin hechos del dataset", exc_info=True)
            return None
        return facts.model_dump(mode="json")

    def _replan(self, old: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
        """Recalcula hechos y plan si cambió el dataset o la ficha (o faltan)."""
        out = dict(clean)
        dv = out.get("dataset_version_id")
        facts = out.get("data_facts")
        if dv and (dv != old.get("dataset_version_id") or not facts):
            facts = self._facts(str(dv))
        if not dv:
            facts = None
        if facts is None:
            out.pop("data_facts", None)
        else:
            out["data_facts"] = facts
        changed = out.get("brief") != old.get("brief") or facts != old.get("data_facts")
        stale = (old.get("plan") or {}).get("version", 1) < PLAN_VERSION
        if changed or stale or "plan" not in old:
            plan = compile_plan(
                UseCaseBrief.model_validate(out["brief"]) if out.get("brief") else None,
                DataFacts.model_validate(facts) if facts else None,
            )
            out["plan"] = plan.model_dump(mode="json", exclude_none=True)
            previous = WizardPlan.model_validate(old["plan"]) if old.get("plan") else None
            diff = diff_plans(previous, plan)
            if diff.empty:
                out.pop("plan_diff", None)
            else:
                out["plan_diff"] = diff.model_dump(mode="json")
        return out

    def plan(self, draft: ProjectDraft) -> WizardPlan:
        """El plan guardado o, en borradores anteriores al ADR-0040, el compilado al vuelo."""
        values = draft.values
        if values.get("plan"):
            stored = WizardPlan.model_validate(values["plan"])
            if stored.version >= PLAN_VERSION:
                return stored
            # Reglas nuevas: el plan guardado se recompila al vuelo (se persiste al editar).
        brief = values.get("brief")
        facts = values.get("data_facts")
        return compile_plan(
            UseCaseBrief.model_validate(brief) if brief else None,
            DataFacts.model_validate(facts) if facts else None,
        )

    def intake(
        self, project_id: str, message: str, history: list[Message]
    ) -> tuple[BriefPatch, str]:
        """Entrevista: qué entendió el LLM del caso (cambios a la ficha) y qué conviene preguntar.

        No modifica nada: la UI muestra los cambios y la persona los acepta o los descarta.
        """
        draft = self.get(project_id)
        project = self.wf.project(project_id)
        ctx = self._brief_context(draft, facts=False)
        turns = history[-CHAT_TURNS * 2 :]
        transcript = "\n".join(f"{m.role}: {m.content}" for m in turns)
        out = self.ctx.llm.structured(
            LLMPurpose.COPILOT,
            BriefPatch,
            ctx,
            project=project,
            validator=validate_patch,
            prompt="intake",
            prompt_vars={"message": message, "transcript": transcript or "(sin mensajes previos)"},
        )
        return normalize_patch(out.value), out.call_id

    def reconcile(self, project_id: str) -> tuple[BriefPatch, str]:
        """Compara la ficha con el perfil de los datos (fase 2 del ADR-0040).

        El LLM propone correcciones a la ficha o preguntas cuando lo declarado no coincide con
        lo que se ve en la Profile Card. No modifica nada: la persona acepta o descarta.
        """
        draft = self.get(project_id)
        if not draft.values.get("dataset_version_id"):
            raise ValidationError(
                "Primero elegí los datos: la ficha se compara con su perfil.",
                details={"reason": "no_dataset"},
            )
        project = self.wf.project(project_id)
        out = self.ctx.llm.structured(
            LLMPurpose.COPILOT,
            BriefPatch,
            self._brief_context(draft, facts=True),
            project=project,
            validator=validate_patch,
            prompt="reconcile",
        )
        return normalize_patch(out.value), out.call_id

    def _brief_context(self, draft: ProjectDraft, *, facts: bool) -> LLMContext:
        """Contexto liviano de la entrevista y la reconciliación: la ficha (sin vacíos), el
        perfil si hay datos y, para reconciliar, los hechos medidos. Sin el plan ni el borrador
        entero, y con una guía corta de campos en vez del JSON Schema: los modelos chicos
        locales se perdían en el contexto completo y no proponían el tipo de problema."""
        values = draft.values
        brief = {
            k: v
            for k, v in (values.get("brief") or {}).items()
            if v not in (None, [], {}) and k != "origins"
        }
        constraints: dict[str, Any] = {"brief": brief}
        if facts and values.get("data_facts"):
            constraints["data_facts"] = values["data_facts"]
        project = self.wf.project(draft.project_id)
        return LLMContext(
            goal=project.goal or values.get("goal"),
            card=self._card(values),
            constraints=constraints,
            system={"brief_fields": BRIEF_GUIDE},
        )

    def _card(self, values: dict[str, Any]) -> Any:
        if not values.get("dataset_version_id"):
            return None
        try:
            return self.wf.profile_card(str(values["dataset_version_id"]))
        except Exception:  # sin perfil todavía: el copiloto igual responde
            logger.debug("borrador sin perfil", exc_info=True)
            return None

    # ------------------------------------------------------------------ copiloto

    def _context(self, draft: ProjectDraft) -> LLMContext:
        values = draft.values
        card = None
        if values.get("dataset_version_id"):
            try:
                card = self.wf.profile_card(str(values["dataset_version_id"]))
            except Exception:  # sin perfil todavía: el copiloto igual responde
                logger.debug("borrador sin perfil", exc_info=True)
        project = self.wf.project(draft.project_id)
        return LLMContext(
            goal=project.goal or values.get("goal"),
            card=card,
            constraints={"step": draft.step, "steps": list(STEPS), "draft": values},
            system={"editable_fields": list(COPILOT_FIELDS)},
        )

    def chat(self, project_id: str, question: str, history: list[Message]) -> Iterator[str]:
        """Respuesta del copiloto en fragmentos (RF-LLM-05); pasa por el Gateway y su filtro."""
        draft = self.get(project_id)
        project = self.wf.project(project_id)
        yield from self.ctx.llm.stream_chat(
            LLMPurpose.COPILOT,
            self._context(draft),
            project=project,
            history=history[-CHAT_TURNS * 2 :],
            prompt_vars={"question": question},
        )

    def suggest_patch(self, project_id: str, question: str, answer: str) -> tuple[DraftPatch, str]:
        """Cambios estructurados al borrador que surgen de la conversación (o ninguno)."""
        draft = self.get(project_id)
        project = self.wf.project(project_id)

        def validator(p: DraftPatch) -> str | None:
            errors = []
            for c in p.changes:
                try:
                    DraftValues.model_validate({c.field: c.value})
                except PydanticValidationError as e:
                    errors.append(f"{c.field}: {e.errors()[0]['msg']}")
            return "\n".join(errors) or None

        out = self.ctx.llm.structured(
            LLMPurpose.COPILOT,
            DraftPatch,
            self._context(draft),
            project=project,
            validator=validator,
            prompt="patch",
            prompt_vars={"question": question, "answer": answer},
        )
        return out.value, out.call_id
