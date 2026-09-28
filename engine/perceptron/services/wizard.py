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

if TYPE_CHECKING:
    from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

STEPS = (
    "goal",
    "data",
    "quality",
    "labeling",
    "task",
    "architecture",
    "hpo",
    "budget",
    "review",
)
Step = Literal[
    "goal", "data", "quality", "labeling", "task", "architecture", "hpo", "budget", "review"
]
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
        draft = ProjectDraft(
            project_id=project_id,
            values=DraftValues(goal=project.goal or None).model_dump(
                mode="json", exclude_none=True
            ),
        )
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
        merged = {**draft.values, **(values or {})}
        try:
            clean = DraftValues.model_validate(merged).model_dump(mode="json", exclude_none=True)
        except PydanticValidationError as e:
            raise ValidationError(
                "valores inválidos para el borrador", details={"errors": e.errors()}
            ) from e
        changed = sorted(k for k in (values or {}) if draft.values.get(k) != clean.get(k))
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
