"""Asistente guiado de punta a punta (RF-WIZ-01/03/04, RF-ING-01/07/08, RF-EXP-02).

Una capa sobre el motor existente para personas sin conocimientos de machine learning:

1. **Entrevista:** el LLM pregunta tantas veces como haga falta y propone hechos del caso; cada
   hecho tiene que estar respaldado por una cita literal de la persona. Sin LLM, los controles de
   la próxima pregunta siguen funcionando. Al confirmar, los hechos pasan a la ficha del caso del
   wizard: el arquitecto, los requisitos de diseño y el plan los usan igual que en el modo avanzado.
2. **Datos:** guía con un ejemplo concreto (`journey_guide.data_guide`); la ingesta, el perfil,
   las alertas y el plan del wizard revisan lo que se subió y separan bloqueos de avisos.
3. **Entrenamiento:** el diseño guiado del wizard (requisitos, comparación corta, plan de intentos
   y épocas), el estudio por la cola del servidor, la evaluación en el test sellado, el registro y
   la exportación. Con datos casi determinísticos, también la fórmula sugerida.
4. **Validación y uso:** casos nuevos con su resultado real (de a uno o en lote) y predicciones con
   solo los campos que el modelo necesita.

El LLM no decide transiciones ni efectos: los dueños son reglas verificables. Los documentos se
actualizan con compare-and-swap por versión (también en SQLite) antes de lanzar trabajo.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal, cast

import psutil
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.engine import CursorResult

from perceptron.api.jobs import JobContext
from perceptron.core.errors import ConflictError, ValidationError
from perceptron.domain.enums import LLMPurpose, Modality, SplitStrategy, TaskType
from perceptron.domain.models import JourneyRecord, ModelVersion, utcnow
from perceptron.llm.privacy import LLMContext
from perceptron.services import journey_files as files_
from perceptron.services import journey_talk as talk
from perceptron.services.journey_files import FileInfo, GoalProposal
from perceptron.services.journey_guide import (
    SERIES,
    VALIDATION_MIN,
    VISION,
    DataGuide,
    FactField,
    Facts,
    InputForm,
    data_guide,
    input_form,
    missing,
    summarize,
    task_of,
    to_brief,
    unsupported_reason,
)
from perceptron.services.workflow import Workflow
from perceptron.storage.db import EntityRow

logger = logging.getLogger(__name__)

State = Literal[
    "interview",
    "unsupported",
    "awaiting_data",
    "reviewing_data",
    "ready_to_train",
    "training",
    "ready_to_validate",
    "needs_improvement",
    "ready_to_use",
    "interrupted",
    "failed",
]
ACTIVE = {"reviewing_data", "training"}
UNMARKED = "anomalia"  # columna neutra para casos raros sin marcar
DATA_STATES = {
    "awaiting_data",
    "ready_to_train",
    "needs_improvement",
    "ready_to_validate",
    "ready_to_use",
    "failed",
    "interrupted",
}
TRAIN_STATES = {"ready_to_train", "failed", "interrupted", "needs_improvement"}
MODEL_STATES = {"ready_to_validate", "needs_improvement", "ready_to_use"}
DEFAULT_WAIT_S = 20 * 60
FORMULA_TIME_S = 90
PROCESS_STARTED = psutil.Process().create_time()
_KIND_MODALITY = {
    "table": Modality.TABULAR,
    "images": Modality.IMAGE,
    "audio": Modality.AUDIO,
    "text": Modality.TEXT,
}


class FactChange(BaseModel):
    field: FactField
    value: Any = None
    evidence: str = Field(default="", max_length=400, description="Cita literal de la persona")


class InterviewTurn(BaseModel):
    """Salida del LLM en cada turno de la entrevista."""

    reply: str = Field(
        default="",
        max_length=1500,
        description="Acuse de lo entendido y respuesta a lo que preguntó la persona",
    )
    changes: list[FactChange] = Field(default_factory=list, max_length=12)
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    question: str | None = Field(default=None, max_length=1000)
    question_field: FactField | None = None
    options: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Respuestas cortas posibles, como las diría la persona",
    )
    goal: str | None = Field(
        default=None,
        max_length=20,
        description="Id del objetivo propuesto (de `goals`) que la persona parece elegir",
    )


class GoalAdvice(BaseModel):
    """Salida del LLM al leer los archivos: qué objetivos mostrar primero y cómo contarlo."""

    reply: str = Field(default="", max_length=1200)
    picks: list[str] = Field(default_factory=list, max_length=4, description="Ids de `goals`")


ChatAction = Literal["confirm", "train", "revise", "stop"]
# Lo que el asistente puede ofrecer en cada estado (la persona lo acepta con un clic).
CHAT_ACTIONS: dict[str, set[str]] = {
    "awaiting_data": {"revise"},
    "reviewing_data": set(),
    "ready_to_train": {"train", "revise"},
    "training": {"stop"},
    "ready_to_validate": {"revise"},
    "needs_improvement": {"train", "revise"},
    "ready_to_use": {"revise"},
    "interrupted": {"train", "revise"},
    "failed": {"train", "revise"},
}


class AssistantReply(BaseModel):
    """Salida del LLM cuando la persona escribe fuera de la entrevista."""

    reply: str = Field(max_length=2000)
    action: ChatAction | None = None


Step = Literal["case", "data", "review", "training", "validate", "use"]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    step: Step | None = Field(default=None, description="Etapa del recorrido en que se dijo")
    field: FactField | None = Field(default=None, description="Dato que pregunta el asistente")
    options: list[str] = Field(default_factory=list, description="Respuestas rápidas sugeridas")
    action: str | None = Field(
        default=None, description="Acción que ofrece el asistente (se acepta con un clic)"
    )
    chat: bool = Field(
        default=False, description="Pregunta libre y su respuesta, fuera de los pasos del recorrido"
    )
    goals: list[str] = Field(
        default_factory=list, description="Objetivos propuestos (ids de `Journey.goals`)"
    )


class Journey(BaseModel):
    schema_version: int = 2
    version: int = 1
    state: State = "interview"
    facts: Facts = Field(default_factory=Facts)
    language: Literal["es", "en"] = "es"
    messages: list[Turn] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    unsupported: str | None = None
    llm_failed: bool = False
    confirmed: bool = False
    guide: DataGuide | None = None
    source_id: str | None = Field(default=None, description="Última fuente de datos subida")
    files: list[FileInfo] = Field(
        default_factory=list, description="Lo que se leyó de los archivos, sin valores"
    )
    goals: list[GoalProposal] = Field(
        default_factory=list, description="Objetivos posibles según los archivos"
    )
    goal_id: str | None = Field(default=None, description="Objetivo elegido")
    data_ready: bool = Field(
        default=False, description="Los datos llegaron antes de confirmar: se revisan al confirmar"
    )
    dataset_id: str | None = None
    columns: list[str] = Field(default_factory=list, description="Columnas de la tabla subida")
    target: str | None = Field(default=None, description="Columna con el resultado")
    data_summary: dict[str, Any] | None = None
    job_id: str | None = None
    child_job_id: str | None = None
    study_id: str | None = None
    owner_pid: int | None = None
    owner_started: float | None = None
    stage: str = "interview"
    progress: dict[str, int] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_id: str | None = None
    model_id: str | None = None
    contract: dict[str, Any] | None = None
    form: InputForm | None = None
    design_reason: str | None = None
    result: dict[str, Any] | None = None
    validations: list[dict[str, Any]] = Field(default_factory=list)
    notification: str | None = None
    error: str | None = None
    stop_requested: bool = False


def _cited(evidence: str, said: str) -> bool:
    quote = " ".join(evidence.casefold().strip(" \"'«»“”.").split())
    return len(quote) >= 2 and quote in " ".join(said.casefold().split())


def apply_changes(facts: Facts, changes: list[FactChange], said: str) -> Facts:
    """Aplica solo los hechos respaldados por una cita de lo que dijo la persona y que validan."""
    values = facts.model_dump()
    for change in changes:
        if not _cited(change.evidence, said):
            continue
        candidate = {**values, change.field: change.value}
        try:
            Facts.model_validate(candidate)
        except ValueError:
            continue
        values = candidate
    return Facts.model_validate(values)


class Coordinator:
    def __init__(self, wf: Workflow):
        self.wf, self.ctx = wf, wf.ctx
        self.repo = self.ctx.repo(JourneyRecord)

    # ------------------------------------------------------------------ persistencia

    def get(self, pid: str) -> Journey:
        self.wf.project(pid)
        record = self.repo.find(pid)
        if record is None:
            return Journey(pending=list(missing(Facts())))
        payload = dict(record.payload)
        if payload.get("schema_version", 1) < 2:
            # Recorridos de la primera versión (solo imágenes): se retoman desde la entrevista.
            payload = {"messages": payload.get("messages", [])}
        j = Journey.model_validate(payload).model_copy(update={"version": record.version})
        if j.state in ACTIVE and j.owner_pid:
            return self._recover(pid, j)
        return j

    def _recover(self, pid: str, j: Journey) -> Journey:
        """Si el proceso dueño del trabajo murió: interrumpido, conservando datos y contexto."""
        try:
            alive = psutil.Process(j.owner_pid or 0).create_time() == j.owner_started
        except psutil.Error:
            alive = False
        if alive and j.owner_pid == os.getpid() and j.job_id:
            alive = self.ctx.jobs.get(j.job_id) is not None
        if alive:
            return j
        child = self.ctx.jobs.get(j.child_job_id) if j.child_job_id else None
        if child is None and j.study_id:
            child = next(
                (job for job in self.ctx.jobs.list() if job.refs.get("study_id") == j.study_id),
                None,
            )
        if child and child.finished_at is None:
            # Un worker remoto restaurado sigue entrenando: se cancela antes de reintentar.
            self.ctx.jobs.cancel(child.id)
            j.stop_requested, j.stage = True, "stopping"
            return j
        if j.study_id:
            from perceptron.services.study_control import mark_interrupted

            mark_interrupted(self.ctx, j.study_id)
        j.state, j.stage, j.notification = "interrupted", "interrupted", "interrupted"
        j.error = "El motor se interrumpió. Podés reintentar: tus datos se conservan."
        return self.save(pid, j)

    def save(self, pid: str, j: Journey) -> Journey:
        record = self.repo.find(pid)
        if record is None:
            self.repo.add(JourneyRecord(id=pid, project_id=pid, payload=j.model_dump(mode="json")))
            return j
        next_j = j.model_copy(update={"version": j.version + 1})
        entity = record.model_copy(
            update={
                "version": next_j.version,
                "updated_at": utcnow(),
                "payload": next_j.model_dump(mode="json"),
            }
        )
        with self.ctx.db.session() as session:
            result = session.execute(
                update(EntityRow)
                .where(
                    EntityRow.kind == "JourneyRecord",
                    EntityRow.id == pid,
                    EntityRow.version == j.version,
                )
                .values(
                    version=entity.version,
                    updated_at=entity.updated_at,
                    data=entity.model_dump(mode="json"),
                )
            )
            if not isinstance(result, CursorResult) or result.rowcount != 1:
                raise ConflictError("El recorrido cambió. Recargá para continuar.")
        return next_j

    def checked(self, pid: str, version: int, states: set[str]) -> Journey:
        j = self.get(pid)
        if j.version != version:
            raise ConflictError("El recorrido cambió. Recargá para continuar.")
        if j.state not in states:
            raise ValidationError("Esta acción no está disponible en la etapa actual.")
        return j

    def _change(self, pid: str, say: Turn | None = None, **values: Any) -> Journey:
        """Cambia campos con reintentos; `say` agrega un mensaje del asistente al hilo."""
        for _ in range(8):
            j = self.get(pid).model_copy(update=values)
            if say is not None:
                j.messages = [*j.messages, say]
            try:
                return self.save(pid, j)
            except ConflictError:
                continue
        raise ConflictError("No se pudo actualizar el recorrido.")

    # ------------------------------------------------------------------ entrevista

    def interview(
        self,
        pid: str,
        version: int,
        message: str,
        facts: Facts | None = None,
        language: str = "es",
    ) -> Journey:
        j = self.checked(pid, version, {"interview", "unsupported"})
        j.language = "en" if language == "en" else "es"
        if message:
            j.messages.append(Turn(role="user", text=message, step="case"))
        if facts is not None:
            # Lo que la persona completa en los controles es un dato confirmado.
            merged = {**j.facts.model_dump(), **facts.model_dump(exclude_none=True)}
            j.facts = Facts.model_validate(merged)
        j.confirmed = False
        named = files_.mentioned(message, j.goals) if message and not j.goal_id else None
        if named is not None:
            # Nombró la columna (o el objetivo) de una sola propuesta: es su elección.
            self._apply_goal(pid, j, named)
        j.pending = list(missing(j.facts))
        j = self.save(pid, j)  # lo escrito queda guardado aunque falle el LLM
        # Siempre hay respuesta: del LLM si está, si no, redactada para el caso (journey_talk).
        out = self._ask_llm(pid, j, language) if message else None
        if out is None and message and facts is None:
            # Sin LLM: la respuesta a la pregunta que se acaba de hacer, si se entiende sola.
            asked = next(
                (m.field for m in reversed(j.messages[:-1]) if m.role == "assistant"), None
            )
            read = talk.understand(asked, message)
            if read:
                with contextlib.suppress(ValueError):
                    j.facts = Facts.model_validate({**j.facts.model_dump(), **read})
        if j.goals and not j.goal_id and j.facts.problem:
            # Lo que contó coincide con una sola de las propuestas: se trabaja con esos datos.
            fits = files_.consistent(j.facts, j.goals)
            if len(fits) == 1:
                self._apply_goal(pid, j, fits[0])
        j.pending = list(missing(j.facts))
        j.unsupported = unsupported_reason(j.facts)
        j.state = "unsupported" if j.unsupported else "interview"
        j.messages.append(self._next_turn(j, out))
        return self.save(pid, j)

    def _ask_llm(self, pid: str, j: Journey, language: str) -> InterviewTurn | None:
        """El LLM entiende el mensaje: hechos con cita, acuse y la próxima pregunta."""
        project = self.wf.project(pid)
        said = "\n".join(m.text for m in j.messages if m.role == "user")
        try:
            out = self.ctx.llm.structured(
                LLMPurpose.COPILOT,
                InterviewTurn,
                LLMContext(
                    goal=project.goal or None,
                    constraints={
                        "facts": j.facts.model_dump(),
                        "pending": list(missing(j.facts)),
                        "files": _files_context(j),
                        "goals": None if j.goal_id else _goals_context(j.goals),
                        "chosen_goal": _chosen(j),
                        "language": language,
                        "conversation": [
                            {"role": m.role, "text": m.text} for m in j.messages[-24:]
                        ],
                    },
                ),
                project=project,
                prompt="journey",
            ).value
        except Exception:  # sin proveedor, L0, presupuesto o salida inválida: preguntas redactadas
            logger.info("entrevista guiada sin LLM", exc_info=True)
            j.llm_failed = True
            return None
        j.facts = apply_changes(j.facts, out.changes, said)
        j.assumptions = out.assumptions
        j.llm_failed = False
        if out.goal not in {g.id for g in j.goals}:
            out.goal = None
        return out

    def _next_turn(self, j: Journey, out: InterviewTurn | None) -> Turn:
        """La respuesta del asistente: acuse, respuesta a dudas y la próxima pregunta."""
        lang = j.language
        reply = (out.reply.strip() if out else "") or ""
        if j.unsupported:
            return Turn(
                role="assistant",
                text=" ".join(x for x in (reply, j.unsupported) if x),
                step="case",
                action="advanced",
            )
        if not j.pending:
            return Turn(
                role="assistant",
                text=" ".join(x for x in (reply, talk.summary(j.facts, lang)) if x),
                step="case",
                action="confirm",
            )
        fits = files_.consistent(j.facts, j.goals) if not j.goal_id else []
        if fits:
            # Hay archivos leídos y todavía no eligió qué aprender: se ofrecen los que encajan.
            ids = [g.id for g in fits]
            first = out.goal if out and out.goal in ids else None
            ordered = ([first] if first else []) + [i for i in ids if i != first]
            return Turn(
                role="assistant",
                text=" ".join(x for x in (reply, talk.pick_goal(lang)) if x),
                step="case",
                goals=ordered[:4],
            )
        # La pregunta del LLM vale si es sobre algo que todavía falta; si no, la redactada.
        llm_field = out.question_field if out else None
        if out and out.question and llm_field in j.pending:
            field, question, options = llm_field, out.question.strip(), out.options
        else:
            first = cast(FactField, j.pending[0])
            field, question, options = first, talk.question(first, j.facts, lang), []
        if not reply:
            said = sum(1 for m in j.messages if m.role == "user")
            reply = talk.greeting(lang) if said <= 1 else talk.acknowledge(said, lang)
        return Turn(
            role="assistant",
            text=f"{reply} {question}".strip(),
            step="case",
            field=field,
            options=[o.strip() for o in options if o.strip()][:4],
        )

    # ------------------------------------------------------------------ conversación libre

    def ask(self, pid: str, version: int, message: str, language: str = "es") -> Journey:
        """La persona escribe en cualquier etapa después de la entrevista: el asistente responde
        con lo que sabe del recorrido y, si corresponde, ofrece una acción para aceptar con un
        clic. Nunca cambia el estado por su cuenta."""
        j = self.checked(pid, version, set(CHAT_ACTIONS))
        j.language = "en" if language == "en" else "es"
        j.messages.append(Turn(role="user", text=message, step=_step(j), chat=True))
        j = self.save(pid, j)
        reply = self._chat_llm(pid, j)
        allowed = CHAT_ACTIONS.get(j.state, set())
        action = reply.action if reply and reply.action in allowed else None
        text = reply.reply.strip() if reply else talk.step_help(j.state, j.language)
        return self._change(
            pid,
            say=Turn(role="assistant", text=text, step=_step(j), action=action, chat=True),
        )

    def _chat_llm(self, pid: str, j: Journey) -> AssistantReply | None:
        project = self.wf.project(pid)
        result = {
            k: v
            for k, v in (j.result or {}).items()
            if k
            in (
                "kind",
                "meets_target",
                "percent",
                "typical_error",
                "typical_percent",
                "r2",
                "explanation",
                "next_steps",
                "per_class",
            )
        }
        try:
            return self.ctx.llm.structured(
                LLMPurpose.COPILOT,
                AssistantReply,
                LLMContext(
                    goal=project.goal or None,
                    constraints={
                        "state": j.state,
                        "step": _step(j),
                        "stage": j.stage,
                        "facts": j.facts.model_dump(exclude_none=True),
                        "data_guide": j.guide.model_dump() if j.guide else None,
                        "data_summary": j.data_summary,
                        "blockers": j.blockers,
                        "warnings": j.warnings,
                        "progress": j.progress,
                        "result": result or None,
                        "design_reason": j.design_reason,
                        "validations": {
                            "total": len(j.validations),
                            "correct": sum(1 for v in j.validations if v.get("correct")),
                            "needed": validation_needed(j),
                        },
                        "input_fields": [f.name for f in j.form.fields] if j.form else [],
                        "error": j.error,
                        "allowed_actions": sorted(CHAT_ACTIONS.get(j.state, set())),
                        "language": j.language,
                        "conversation": [
                            {"role": m.role, "text": m.text} for m in j.messages[-16:]
                        ],
                    },
                ),
                project=project,
                prompt="journey_chat",
            ).value
        except Exception:
            logger.info("conversación guiada sin LLM", exc_info=True)
            return None

    def confirm(self, pid: str, version: int) -> Journey:
        from perceptron.services.wizard import Wizard

        j = self.checked(pid, version, {"interview"})
        if missing(j.facts):
            raise ValidationError("Completá los datos esenciales antes de confirmar.")
        if unsupported_reason(j.facts):
            raise ValidationError(unsupported_reason(j.facts) or "")
        # La ficha del caso del wizard: la usan el arquitecto, los requisitos y el plan.
        wizard = Wizard(self.wf)
        draft = wizard.get(pid)
        values: dict[str, Any] = {"brief": to_brief(j.facts)}
        first = next(
            (m.text for m in j.messages if m.role == "user" and m.action not in USER_EVENTS),
            None,
        )
        chosen = next((g.title for g in j.goals if g.id == j.goal_id), None)
        if (first or chosen) and not self.wf.project(pid).goal:
            values["goal"] = (first or chosen or "")[:4000]
        wizard.update(pid, version=draft.version, values=values)
        j.confirmed, j.state, j.stage = True, "awaiting_data", "awaiting_data"
        j.guide = data_guide(j.facts)
        source_id = j.source_id
        if j.data_ready and source_id:
            # Los datos ya están: se revisan directamente para el objetivo elegido.
            j.messages.append(
                Turn(role="assistant", text=talk.after_confirm_with_data(j.language), step="data")
            )
            j = self.save(pid, j)
            return self.prepare(pid, j.version, source_id, j.target)
        j.messages.append(Turn(role="assistant", text=talk.after_confirm(j.language), step="data"))
        return self.save(pid, j)

    def revise(self, pid: str, version: int) -> Journey:
        j = self.get(pid)
        if j.version != version or j.state in ACTIVE:
            raise ConflictError("Esperá a que termine el trabajo antes de corregir el caso.")
        return self.save(
            pid,
            Journey(
                version=j.version,
                facts=j.facts,
                messages=j.messages,
                assumptions=j.assumptions,
                language=j.language,
                pending=list(missing(j.facts)),
                unsupported=unsupported_reason(j.facts),
                # Lo leído de los archivos sigue valiendo para replantear el caso.
                files=j.files,
                goals=j.goals,
                goal_id=j.goal_id,
                data_ready=j.data_ready,
                source_id=j.source_id if j.data_ready else None,
                target=j.target if j.data_ready else None,
            ),
        )

    # ------------------------------------------------------------------ archivos y objetivos

    def read_files(
        self, pid: str, version: int, source_id: str, message: str = "", language: str = "es"
    ) -> Journey:
        """Lee lo que subió la persona y le propone qué puede aprender la red con eso.

        Las propuestas salen de reglas sobre lo leído (`journey_files`); el LLM, si está, solo
        elige cuáles mostrar primero y lo cuenta con las palabras del caso. La persona elige con
        un clic o con sus palabras: nada se aplica sin que lo elija."""
        from perceptron.api.routers.data import _source_path
        from perceptron.domain.models import DataSource

        j = self.checked(pid, version, {"interview", "unsupported"})
        j.language = "en" if language == "en" else "es"
        source = self.ctx.repo(DataSource).get(source_id)
        if source.project_id != pid:
            raise ValidationError("La fuente pertenece a otro proyecto.")
        if message.strip():
            j.messages.append(Turn(role="user", text=message.strip(), step="case"))
        j.messages.append(
            Turn(
                role="user",
                text=talk.shared(_upload_names(source), j.language),
                step="case",
                action="files",
            )
        )
        j = self.save(pid, j)
        reading = files_.read(_source_path(self.ctx, source), j.language)
        j.files, j.goals, j.goal_id, j.data_ready = reading.files, reading.goals, None, False
        j.source_id, j.target = source_id, None
        fits = files_.consistent(j.facts, j.goals) or j.goals
        advice = self._goals_llm(pid, j) if fits else None
        ids = [g.id for g in fits]
        picks = [p for p in dict.fromkeys(advice.picks if advice else []) if p in ids]
        ordered = picks + [i for i in ids if i not in picks]
        text = (advice.reply.strip() if advice else "") or talk.files_read(
            reading.files, reading.combined, bool(fits), j.language
        )
        if reading.note:
            text = f"{text} {reading.note}".strip()
        j.messages.append(Turn(role="assistant", text=text, step="case", goals=ordered[:4]))
        j.pending = list(missing(j.facts))
        return self.save(pid, j)

    def _goals_llm(self, pid: str, j: Journey) -> GoalAdvice | None:
        project = self.wf.project(pid)
        try:
            return self.ctx.llm.structured(
                LLMPurpose.COPILOT,
                GoalAdvice,
                LLMContext(
                    goal=project.goal or None,
                    constraints={
                        "files": _files_context(j),
                        "goals": _goals_context(j.goals),
                        "facts": j.facts.model_dump(exclude_none=True),
                        "language": j.language,
                        "conversation": [
                            {"role": m.role, "text": m.text} for m in j.messages[-12:]
                        ],
                    },
                ),
                project=project,
                prompt="journey_goals",
            ).value
        except Exception:  # sin proveedor, L0 o salida inválida: el orden y el texto de reglas
            logger.info("propuesta de objetivos sin LLM", exc_info=True)
            return None

    def choose(self, pid: str, version: int, goal_id: str, language: str = "es") -> Journey:
        """La persona eligió uno de los objetivos propuestos: sus hechos pasan al caso."""
        j = self.checked(pid, version, {"interview", "unsupported"})
        j.language = "en" if language == "en" else "es"
        goal = next((g for g in j.goals if g.id == goal_id), None)
        if goal is None:
            raise ValidationError("Ese objetivo ya no está disponible: volvé a subir los datos.")
        j.messages.append(Turn(role="user", text=goal.title, step="case", action="goal"))
        self._apply_goal(pid, j, goal)
        j.pending = list(missing(j.facts))
        j.unsupported = unsupported_reason(j.facts)
        j.state = "unsupported" if j.unsupported else "interview"
        # El acuse es el objetivo elegido; después, lo próximo que falta saber.
        turn = self._next_turn(j, InterviewTurn(reply=talk.goal_chosen(goal, j.language)))
        j.messages.append(turn)
        return self.save(pid, j)

    def _apply_goal(self, pid: str, j: Journey, goal: GoalProposal) -> None:
        """Los hechos del objetivo (salen de los datos) y la fuente con la que se entrena."""
        from perceptron.api.routers.data import _source_path
        from perceptron.domain.enums import DataSourceType
        from perceptron.domain.models import DataSource

        replaced = ("data_kind", "problem", "target_name", "input_names", "classes")
        keep = {k: v for k, v in j.facts.model_dump().items() if k not in replaced}
        j.facts = Facts.model_validate({**keep, **goal.facts.model_dump(exclude_none=True)})
        j.goal_id, j.target, j.data_ready = goal.id, goal.target, True
        if not goal.parts or not j.source_id:
            return  # fotos, grabaciones o textos: la carpeta subida es la fuente
        repo = self.ctx.repo(DataSource)
        base = _source_path(self.ctx, repo.get(j.source_id))
        if base.is_file() and goal.parts == [base.name]:
            return  # la planilla subida es la del objetivo
        root = base if base.is_dir() else base.parent
        dest = self.ctx.settings.paths.project(pid).root / "journey" / f"objetivo-{goal.id}.parquet"
        path = files_.materialize(root, goal, dest)
        j.source_id = repo.add(
            DataSource(
                project_id=pid,
                name=Path(goal.parts[0]).name if len(goal.parts) == 1 else dest.name,
                type=DataSourceType.FILE,
                config={"path": str(path), "uploaded": True, "files": len(goal.parts)},
            )
        ).id

    # ------------------------------------------------------------------ trabajo en segundo plano

    def _launch(self, pid: str, j: Journey, work: Any) -> Journey:
        j.owner_pid, j.owner_started = os.getpid(), PROCESS_STARTED
        j.job_id, j.child_job_id, j.study_id = None, None, None
        j.error, j.notification, j.stop_requested = None, None, False
        j = self.save(pid, j)  # reserva durable ANTES de lanzar: el doble clic recibe 409
        gate = threading.Event()

        def run(job: JobContext) -> None:
            gate.wait()

            def cancel() -> None:
                self._change(pid, stop_requested=True)

            job.cancel_callback = cancel
            try:
                work(job)
            except Exception as exc:
                logger.warning("recorrido guiado con error", exc_info=True)
                error = _plain_error(exc)
                self._change(
                    pid,
                    state="failed",
                    stage="failed",
                    notification="failed",
                    error=error,
                    say=Turn(
                        role="assistant",
                        text=talk.failed(error, self.get(pid).language),
                        step=_step(self.get(pid)),
                        action="train" if self.get(pid).dataset_id else "revise",
                    ),
                )
                raise

        try:
            job = self.ctx.jobs.submit("journey", run, refs={"project_id": pid}, dedicated=True)
            j = self._change(pid, job_id=job.id)
        except Exception:
            self._change(pid, state="failed", notification="failed")
            raise
        finally:
            gate.set()
        return j

    def _stopped(self, pid: str) -> bool:
        j = self.get(pid)
        if j.stop_requested:
            self._change(
                pid,
                state="interrupted",
                stage="interrupted",
                notification="interrupted",
                say=Turn(
                    role="assistant",
                    text=talk.interrupted(j.language),
                    step=_step(j),
                    action="train" if j.dataset_id else None,
                ),
            )
            return True
        return False

    def stop(self, pid: str, version: int) -> Journey:
        j = self.checked(pid, version, ACTIVE)
        j.stop_requested = True
        if j.child_job_id:
            self.ctx.jobs.cancel(j.child_job_id)
        return self.save(pid, j)

    # ------------------------------------------------------------------ datos

    def prepare(self, pid: str, version: int, source_id: str, target: str | None = None) -> Journey:
        from perceptron.api.routers.data import _source_path
        from perceptron.domain.models import DataSource

        j = self.checked(pid, version, DATA_STATES)
        if not j.confirmed:
            raise ValidationError("Confirmá primero el resumen del caso.")
        source = self.ctx.repo(DataSource).get(source_id)
        if source.project_id != pid:
            raise ValidationError("La fuente pertenece a otro proyecto.")
        path = _source_path(self.ctx, source)
        facts = j.facts
        j.state, j.stage = "reviewing_data", "reviewing_data"
        j.dataset_id = j.model_id = j.run_id = None
        j.contract = j.form = j.result = j.data_summary = None
        j.validations, j.blockers, j.warnings, j.columns, j.progress = [], [], [], [], {}
        j.target = target
        if source_id != j.source_id:
            j.messages.append(
                Turn(role="user", text=talk.uploaded(source.name, j.language), step="data")
            )
        j.source_id = source_id

        def work(job: JobContext) -> None:
            self._review(pid, facts, path, source, target)

        return self._launch(pid, j, work)

    def _ingest(self, pid: str, facts: Facts, path: Any, source: Any, target: str | None) -> Any:
        from perceptron.data.splits import SplitRequest

        kind = facts.data_kind or "table"
        task = TaskType(task_of(facts))
        if facts.problem in VISION:
            # Ubicar objetos necesita la tarea explícita; zonas y texto se reconocen solos.
            return self.wf.ingest(
                pid,
                path,
                source_record=source,
                task=TaskType.OBJECT_DETECTION if facts.problem == "locate" else None,
            )
        split = None
        if kind in ("images", "audio") and facts.grouped:
            split = SplitRequest(strategy=SplitStrategy.GROUP, group_column="__image_group__")
        if kind in ("images", "audio"):
            return self.wf.ingest(
                pid, path, split=split, source_record=source, modality=_KIND_MODALITY[kind]
            )
        if facts.problem in SERIES:
            return self._ingest_series(pid, facts, path, source, target)
        wanted = target or facts.target_name
        joined = self._joined_key(pid, source)
        grouped: dict[str, Any] = {}
        if joined:
            # Planillas unidas: cada cliente (o producto) tiene varias filas. Se separan por esa
            # columna para que el modelo no se pruebe con alguien que ya vio, y no es un dato.
            from perceptron.data.schema import SemanticType
            from perceptron.data.splits import SplitRequest

            grouped = {
                "split": SplitRequest(strategy=SplitStrategy.GROUP, group_column=joined),
                "overrides": {joined: SemanticType.ID},
            }
        try:
            return self.wf.ingest(
                pid,
                path,
                target=wanted,
                source_record=source,
                modality=_KIND_MODALITY[kind],
                task=task,
                **grouped,
            )
        except (ValidationError, KeyError):
            if not wanted:
                raise
        # La columna nombrada no está (o no sirve): se carga igual y se pide elegirla.
        return self.wf.ingest(
            pid, path, source_record=source, modality=_KIND_MODALITY[kind], **grouped
        )

    def _joined_key(self, pid: str, source: Any) -> str | None:
        """La columna por la que se unieron las planillas del objetivo elegido, si esta fuente
        es esa tabla unida."""
        j = self.get(pid)
        goal = next((g for g in j.goals if g.id == j.goal_id), None)
        if goal is None or goal.combine != "join" or not goal.keys or j.source_id != source.id:
            return None
        return goal.keys[0]

    def _ingest_series(
        self, pid: str, facts: Facts, path: Any, source: Any, target: str | None
    ) -> Any:
        """Mediciones en el tiempo: pronóstico, o casos raros (marcados o no).

        Sin casos raros marcados se agrega una columna `anomalia` en 0: el motor aprende lo
        normal y calibra el umbral con el cuantil alto de los datos normales."""
        from pathlib import Path

        import polars as pl

        from perceptron.data.sources.files import scan_table

        overrides: dict[str, Any] | None = (
            {"horizon": facts.horizon} if facts.problem == "forecast" and facts.horizon else None
        )
        if facts.problem == "anomaly":
            # La serie se arma como de casos raros aunque la marca no tenga positivos.
            overrides = {"task": TaskType.ANOMALY_DETECTION.value, "horizon": 1, "lookback": 20}
        wanted = target or facts.target_name
        if facts.problem == "anomaly" and facts.labels_available is False:
            df = scan_table(Path(path)).collect()
            df = df.with_columns(pl.lit(0).alias(UNMARKED))
            copy = self.ctx.settings.paths.project(pid).root / "journey" / f"{Path(path).stem}.csv"
            copy.parent.mkdir(parents=True, exist_ok=True)
            df.write_csv(copy)
            path, wanted = copy, UNMARKED
        if facts.problem == "anomaly" and not wanted:
            columns = scan_table(Path(path)).collect_schema().names()
            wanted = next((c for c in columns if "anomal" in c.lower()), None)
        try:
            return self.wf.ingest(
                pid,
                path,
                target=wanted,
                source_record=source,
                modality=Modality.TIMESERIES,
                series_overrides=overrides,
            )
        except (ValidationError, KeyError):
            # No se reconoce como serie: se carga como tabla y la revisión explica qué falta.
            return self.wf.ingest(pid, path, source_record=source)

    def _review(self, pid: str, facts: Facts, path: Any, source: Any, target: str | None) -> None:
        from perceptron.services.wizard import Wizard

        dv = self._ingest(pid, facts, path, source, target)
        card = self.wf.profile(dv.id)
        if self._stopped(pid):
            return
        blockers, warnings = self._checks(facts, dv, card, target)
        columns: list[str] = []
        if facts.data_kind in ("table", "text"):
            columns = [c.name for c in card.columns]
            if card.target and card.target.name not in columns:
                columns.append(card.target.name)
        if not blockers:
            # El borrador del wizard sigue al recorrido: datos, objetivo y plan del caso.
            wizard = Wizard(self.wf)
            for _ in range(3):
                try:
                    draft = wizard.get(pid)
                    draft = wizard.update(
                        pid,
                        version=draft.version,
                        values={
                            "dataset_version_id": dv.id,
                            "target": dv.target,
                            "task": card.target.task_hint.value if card.target else None,
                        },
                    )
                    break
                except ConflictError:
                    continue
            plan = wizard.plan(wizard.get(pid))
            warnings += [c.message for c in plan.checks if c.severity in ("high", "warning")]
        self._change(
            pid,
            dataset_id=dv.id,
            target=dv.target,
            columns=columns,
            data_summary=_data_summary(dv, card),
            blockers=blockers,
            warnings=list(dict.fromkeys(warnings)),
            state="awaiting_data" if blockers else "ready_to_train",
            stage="awaiting_data" if blockers else "ready_to_train",
            say=self._review_turn(pid, dv, card, blockers, warnings, columns),
        )

    def _review_turn(
        self,
        pid: str,
        dv: Any,
        card: Any,
        blockers: list[str],
        warnings: list[str],
        columns: list[str],
    ) -> Turn:
        lang = self.get(pid).language
        if blockers:
            return Turn(
                role="assistant",
                text=talk.blocked(blockers, bool(columns), lang),
                step="data",
            )
        return Turn(
            role="assistant",
            text=talk.reviewed(_data_summary(dv, card), warnings, lang),
            step="review",
            action="train",
        )

    def _checks(
        self, facts: Facts, dv: Any, card: Any, target: str | None
    ) -> tuple[list[str], list[str]]:
        """Bloqueos (no se puede entrenar) y avisos (se puede, con cuidado) de los datos."""
        from perceptron.data.profiling.card import AlertSeverity

        blockers: list[str] = []
        warnings: list[str] = []
        kind = facts.data_kind
        if kind in ("table", "text") and not dv.target:
            blockers.append(
                "Elegí cuál columna tiene el resultado que querés predecir."
                + (f" No encontré «{facts.target_name}»." if facts.target_name else "")
            )
            return blockers, warnings
        expected = TaskType(task_of(facts))
        if facts.problem in SERIES and (card.series is None or card.series.task is not expected):
            blockers.append(
                "No pude leerlo como mediciones en el tiempo. Necesito una columna de fecha a "
                "intervalos regulares, al menos 30 fechas por serie y "
                + (
                    "el valor a pronosticar como número."
                    if facts.problem == "forecast"
                    else "las mediciones como números (y, si marcaste casos raros, una columna "
                    "«anomalia» con sí/no)."
                )
            )
            return blockers, warnings
        if facts.problem in VISION and (
            card.vision_task is None or card.vision_task.task is not expected
        ):
            blockers.append(
                {
                    "locate": "No encontré las marcas de los objetos: el ZIP tiene que traer "
                    "las fotos y el archivo de anotaciones (COCO .json, VOC .xml o YOLO .txt).",
                    "segment": "No encontré las máscaras: el ZIP tiene que traer «images/» y "
                    "«masks/» con los mismos nombres de archivo.",
                    "read_text": "No encontré los textos: el ZIP tiene que traer «images/» y una "
                    "planilla «labels.csv» con el nombre de cada imagen y su texto.",
                }[facts.problem or "locate"]
            )
            return blockers, warnings
        if card.num_samples < 20:
            blockers.append(
                f"Hay {card.num_samples} ejemplos: hacen falta al menos 20 para aprender algo."
            )
        classes = card.target.classes if card.target else None
        if task_of(facts) == "classification":
            counts = {str(c.value): c.count for c in classes or []}
            if len(counts) < 2:
                blockers.append(
                    "El resultado tiene una sola categoría: hacen falta ejemplos de al menos dos."
                )
            few = [k for k, n in counts.items() if n < 5]
            if few:
                blockers.append(
                    "Agregá al menos 5 ejemplos de cada categoría (faltan de: "
                    + ", ".join(few[:10])
                    + ")."
                )
            wanted = set(facts.classes or [])
            if kind in ("images", "audio") and wanted and set(counts) != wanted:
                blockers.append(
                    "Las carpetas tienen que coincidir con las categorías del caso ("
                    + ", ".join(sorted(wanted))
                    + "); encontré: "
                    + ", ".join(sorted(counts))
                    + "."
                )
            if facts.positive_class and counts and facts.positive_class not in counts:
                warnings.append(
                    f"No encontré la categoría «{facts.positive_class}» en los datos: los "
                    "resultados no van a poder contar sus omisiones y falsas alarmas."
                )
        if any(not card.split_counts.get(s) for s in ("train", "val", "test")):
            blockers.append(
                "Hacen falta más ejemplos para separar los de aprendizaje de los de evaluación."
            )
        if card.images and card.images.corrupt:
            blockers.append(f"Reemplazá {card.images.corrupt} imágenes que no se pueden leer.")
        if card.audio and card.audio.corrupt:
            blockers.append(f"Reemplazá {card.audio.corrupt} audios que no se pueden leer.")
        if kind == "table" and not blockers:
            try:
                pipeline = self.wf.propose_pipeline(dv.id)
                self.wf.fitted_pipeline(pipeline.id, dv.id)
            except ValidationError:
                blockers.append(
                    "Ninguna columna sirve como dato de entrada (son identificadores o están "
                    "vacías). Agregá columnas con datos que influyan en el resultado."
                )
        for alert in card.alerts:
            if alert.severity in (AlertSeverity.HIGH, AlertSeverity.WARNING):
                warnings.append(alert.message)
        if card.num_samples < 200:
            warnings.append(
                f"Hay pocos ejemplos ({card.num_samples}): el resultado es orientativo. Con más "
                "casos variados mejora y es más confiable."
            )
        return blockers, warnings

    # ------------------------------------------------------------------ entrenamiento

    def train(self, pid: str, version: int) -> Journey:
        j = self.checked(pid, version, TRAIN_STATES)
        if not j.dataset_id or j.blockers or not j.confirmed:
            raise ValidationError("Revisá los datos antes de entrenar.")
        dataset_id, facts = j.dataset_id, j.facts
        j.state, j.stage, j.progress = "training", "designing", {}
        j.result = j.contract = j.form = j.model_id = j.run_id = None
        j.validations = []
        j.messages.append(
            Turn(role="assistant", text=talk.training_started(j.language), step="training")
        )

        def work(job: JobContext) -> None:
            self._train(pid, dataset_id, facts)

        return self._launch(pid, j, work)

    def _train(self, pid: str, dataset_id: str, facts: Facts) -> None:
        from perceptron.api.jobs import TERMINAL
        from perceptron.domain.models import Run
        from perceptron.export.formats import ExportRequest
        from perceptron.hpo.strategy import HPOStrategy
        from perceptron.hpo.study import StudyResult
        from perceptron.services.studies import new_study
        from perceptron.services.wizard import Wizard

        wizard = Wizard(self.wf)

        # 1. Diseño guiado: requisitos del caso, propuestas evaluadas, comparación corta y plan.
        def progress(stage: str) -> None:
            self._change(pid, stage=f"design_{stage}")

        outcome = wizard.design(pid, tournament="auto", progress=progress)
        if self._stopped(pid):
            return
        if outcome.pick is None or outcome.strategy is None:
            raise ValidationError("No se encontró una arquitectura para estos datos.")
        strategy = HPOStrategy.model_validate(outcome.strategy)
        if strategy.budget.max_time_s is None:
            strategy = strategy.model_copy(
                update={"budget": strategy.budget.model_copy(update={"max_time_s": DEFAULT_WAIT_S})}
            )
        self._accept_design(wizard, pid, outcome, strategy)
        # 2. Entrenamiento por la cola del servidor (cuotas y workers como en el modo avanzado).
        study = new_study(self.ctx, pid, dataset_id, outcome.pipeline_id, outcome.pick, strategy)
        self._change(
            pid,
            study_id=study.id,
            stage="queued",
            design_reason=outcome.pick_reason,
            progress={"done": 0, "total": strategy.budget.max_trials},
        )
        if self._stopped(pid):
            return
        child = self.ctx.launch_study(study)
        self._change(pid, child_job_id=child.id)
        deadline = time.monotonic() + (strategy.budget.max_time_s or DEFAULT_WAIT_S) * 1.5 + 600
        timed_out, last_done = False, -1
        while True:
            current = self.ctx.jobs.get(child.id)
            if current is None:
                raise ValidationError("Se perdió el trabajo de entrenamiento. Reintentá.")
            if current.status == "running":
                runs = self.ctx.repo(Run).list(filters={"study_id": study.id}, limit=1000)
                done = sum(1 for r in runs if r.status.value not in ("queued", "running"))
                if done != last_done:
                    last_done = done
                    self._change(
                        pid,
                        stage="learning",
                        progress={"done": done, "total": strategy.budget.max_trials},
                    )
            timed_out = timed_out or time.monotonic() > deadline
            if self.get(pid).stop_requested or timed_out:
                self.ctx.jobs.cancel(child.id)
            if current.status in TERMINAL and (
                current.status != "cancelled" or current.finished_at is not None
            ):
                break
            time.sleep(1.0)
        if self._stopped(pid):
            return
        if timed_out or current.status != "succeeded":
            raise ValidationError(
                "El entrenamiento no terminó. Revisá los datos o los recursos disponibles y "
                "reintentá."
            )
        best = StudyResult.model_validate(current.result).best_trial
        if best is None:
            raise ValidationError("No se obtuvo un modelo válido. Revisá los datos y reintentá.")
        rid = best.run_id
        # 3. Evaluación en casos que el modelo nunca vio, registro y preparación para predecir.
        self._change(pid, stage="evaluating", run_id=rid)
        _, report = self.wf.evaluate(rid)
        if self._stopped(pid):
            return
        model = self.wf.register(rid)
        self._change(pid, stage="preparing_prediction")
        exported = self.wf.export(rid, ExportRequest())
        if not exported.ok:
            raise ValidationError("El modelo entrenó, pero no pudo prepararse para predecir.")
        dv = self.wf.dataset(dataset_id)
        card = self.wf.profile_card(dataset_id)
        typical = None
        if card.target and card.target.numeric:
            typical = card.target.numeric.quantiles.get("p50") or card.target.numeric.mean
        result = summarize(report, facts, typical)
        formula = self._formula(pid, dataset_id, facts, result)
        if formula:
            result["formula"] = formula
        form = input_form(model.signature, card, _example_row(self.wf, dv), series=card.series)
        if self._stopped(pid):
            return
        self._change(
            pid,
            state="ready_to_validate"
            if result["meets_target"] or result.get("measurable") is False
            else "needs_improvement",
            stage="complete",
            model_id=model.id,
            contract=model.signature,
            form=form,
            notification="training_finished",
            result=result,
            say=Turn(
                role="assistant",
                text=talk.trained(result, self.get(pid).language),
                step="validate",
                action=None if result["meets_target"] else "train",
            ),
        )

    def _accept_design(self, wizard: Any, pid: str, outcome: Any, strategy: Any) -> None:
        """El diseño elegido queda en el borrador del wizard (se ve en las opciones avanzadas)."""
        for _ in range(3):
            try:
                draft = wizard.get(pid)
                wizard.update(
                    pid,
                    version=draft.version,
                    values={
                        "archspec_id": outcome.pick,
                        "strategy": strategy.model_dump(mode="json"),
                        "max_trials": strategy.budget.max_trials,
                        "max_epochs_per_trial": strategy.budget.max_epochs_per_trial,
                    },
                )
                return
            except ConflictError:
                continue

    def _formula(
        self, pid: str, dataset_id: str, facts: Facts, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Con datos casi determinísticos, la fórmula exacta suele ganarle a la red y extrapola."""
        if facts.data_kind != "table" or facts.problem not in ("value", "rule"):
            return None
        from perceptron.evaluation.symbolic import SymbolicConfig
        from perceptron.services.symbolic import run_symbolic

        if not self._near_deterministic(dataset_id):
            return None
        self._change(pid, stage="searching_formula")
        try:
            fit = run_symbolic(
                self.ctx, pid, dataset_id, SymbolicConfig(time_limit_s=FORMULA_TIME_S)
            )
        except Exception:
            logger.info("sin fórmula sugerida", exc_info=True)
            return None
        val = fit.metrics.get("val") or {}
        r2 = val.get("r2")
        better = r2 is not None and r2 >= (result.get("r2") or 0) + 0.0005
        if better and r2 is not None:
            result["explanation"].append(
                f"Además encontré una fórmula exacta que explica el {100 * r2:.3f} % y sirve "
                "también fuera del rango de los datos: " + fit.formula
            )
        return {
            "id": fit.id,
            "formula": fit.formula,
            "excel": fit.excel_es,
            "python": fit.python,
            "r2": r2,
            "better": better,
        }

    def _near_deterministic(self, dataset_id: str) -> bool:
        """Una recta o una curva de las entradas explica prácticamente todo el resultado."""
        from perceptron.data.schema import SemanticType
        from perceptron.services.fit_checks import LINEAR_EXACT, measure_linearity

        card = self.wf.profile_card(dataset_id)
        if card.target is None:
            return False
        features = [
            c.name
            for c in card.columns
            if c.semantic is SemanticType.NUMERIC and c.name != card.target.name
        ]
        if not features:
            return False
        view = self.wf.view(self.wf.dataset(dataset_id))
        lin = measure_linearity(view, card.target.name, features)
        return lin is not None and max(lin.r2_linear, lin.r2_curved) >= LINEAR_EXACT

    # ------------------------------------------------------------------ validación y uso

    def contract(self, pid: str) -> Journey:
        j = self.get(pid)
        if j.state not in MODEL_STATES or not j.model_id:
            raise ValidationError("Todavía no hay un modelo preparado para probar.")
        model = self.ctx.repo(ModelVersion).get(j.model_id)
        if model.project_id != pid or model.run_id != j.run_id or model.signature != j.contract:
            raise ValidationError("El contrato no corresponde al modelo de este recorrido.")
        return j

    def predict(
        self,
        j: Journey,
        *,
        row: dict[str, Any] | None = None,
        text: str | None = None,
        file: tuple[bytes, str] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Predicción del modelo del recorrido con lo que pide su contrato (nunca el real). En
        series devuelve una predicción por serie."""
        import io
        import tempfile
        from pathlib import Path

        from perceptron.serving.runtime import InputError

        model = self.wf.inference_model(j.run_id or "")
        try:
            if model.kind == "sequence":
                if not rows:
                    raise ValidationError("Subí la historia reciente (una tabla con las fechas).")
                return model.predict_series(rows)
            if model.kind == "tabular":
                if row is None:
                    raise ValidationError("Completá los datos del caso.")
                return model.predict_rows([row])[0]
            if model.kind == "tokens":
                if not (text or "").strip():
                    raise ValidationError("Escribí el texto del caso.")
                return model.predict_texts([text or ""])[0]
            if file is None:
                raise ValidationError("Subí el archivo del caso.")
            data, suffix = file
            if model.kind == "spectrogram":
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / f"audio{suffix or '.wav'}"
                    path.write_bytes(data)
                    return model.predict_audio([path])[0]
            from PIL import Image, UnidentifiedImageError

            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    return model.predict_images([image])[0]
            except (UnidentifiedImageError, OSError) as exc:
                raise ValidationError("No se pudo leer la imagen. Probá un PNG o JPG.") from exc
        except InputError as exc:
            raise ValidationError(str(exc)) from exc

    def series_config(self, j: Journey) -> Any:
        """Columnas de tiempo, serie y valor, y el horizonte con que se entrenó."""
        cfg = self.wf.view(self.wf.dataset(j.dataset_id or "")).series if j.dataset_id else None
        if cfg is None:
            raise ValidationError("Este modelo no trabaja con mediciones en el tiempo.")
        return cfg

    def record_validation(
        self, pid: str, key: str, actual: Any, prediction: Any, model_id: str | None
    ) -> Journey:
        """Una comparación por caso nuevo (repetir un caso la reemplaza) y el veredicto."""
        for _ in range(8):
            j = self.contract(pid)
            if j.model_id != model_id:
                raise ConflictError("El modelo cambió durante la validación. Probá de nuevo.")
            entry = _compare(j, key, actual, prediction)
            j.validations = [v for v in j.validations if v["key"] != key] + [entry]
            before, j.state = j.state, _validated_state(j)
            if j.state != before and j.state in ("ready_to_use", "needs_improvement"):
                j.messages.append(
                    Turn(
                        role="assistant",
                        text=talk.validated(j.validations, j.state == "ready_to_use", j.language),
                        step="use" if j.state == "ready_to_use" else "validate",
                        action="train" if j.state == "needs_improvement" else None,
                    )
                )
            try:
                return self.save(pid, j)
            except ConflictError:
                continue
        raise ConflictError("No se pudo registrar la validación.")

    def seen_case(self, j: Journey, *, row: dict[str, Any] | None, digest: str | None) -> bool:
        """El caso ya estaba en los datos de entrenamiento (no sirve para validar)."""
        if not j.dataset_id:
            return False
        dv = self.wf.dataset(j.dataset_id)
        view = self.wf.view(dv)
        if digest is not None:
            from perceptron.data.view import MANIFEST_FILE

            manifest = json.loads((view.root / MANIFEST_FILE).read_text(encoding="utf-8"))
            return digest in {e["sha256"] for e in manifest["entries"]}
        if row is None or not j.form:
            return False
        names = [f.name for f in j.form.fields]
        df = view.read()
        if df.height > 200_000 or not names or any(n not in df.columns for n in names):
            return False
        wanted = tuple(_norm(row.get(n)) for n in names)
        return any(
            tuple(_norm(v) for v in values) == wanted for values in df.select(names).iter_rows()
        )


# ---------------------------------------------------------------------- utilidades


def _norm(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError):
        return str(value).strip().casefold()


USER_EVENTS = {"files", "goal"}  # turnos de la persona que no son algo que escribió


def _upload_names(source: Any) -> list[str]:
    """Los nombres de lo subido, para contarlo en el hilo."""
    path = Path(str(source.config.get("path", "")))
    entries = [p for p in path.iterdir() if not p.name.startswith(".")] if path.is_dir() else []
    if 1 < len(entries) <= 6 and all(p.is_file() for p in entries):
        return sorted(p.name for p in entries)  # varias planillas sueltas
    return [source.name]


def _files_context(j: Journey) -> list[dict[str, Any]] | None:
    """Lo leído, para el LLM: nombres, tipos y cantidades; nunca valores de las filas."""
    if not j.files:
        return None
    return [f.model_dump(exclude_none=True) for f in j.files]


def _goals_context(goals: list[GoalProposal]) -> list[dict[str, Any]] | None:
    if not goals:
        return None
    return [
        {
            "id": g.id,
            "title": g.title,
            "detail": g.detail,
            "problem": g.facts.problem,
            "data_kind": g.facts.data_kind,
            "target": g.target,
        }
        for g in goals
    ]


def _chosen(j: Journey) -> dict[str, Any] | None:
    goal = next((g for g in j.goals if g.id == j.goal_id), None)
    return {"title": goal.title, "detail": goal.detail} if goal else None


def _plain_error(exc: Exception) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return str(message)[:1000]


def _step(j: Journey) -> Step:
    """La etapa que ve la persona (la misma que el stepper de la UI)."""
    if j.state in ("interview", "unsupported"):
        return "case"
    if j.state == "awaiting_data":
        return "data"
    if j.state in ("reviewing_data", "ready_to_train"):
        return "review"
    if j.state == "training":
        return "training"
    if j.state == "ready_to_use":
        return "use"
    if j.state in MODEL_STATES:
        return "validate"
    return "review" if j.dataset_id else "data"


def _data_summary(dv: Any, card: Any) -> dict[str, Any]:
    target = card.target
    out: dict[str, Any] = {
        "rows": card.num_samples,
        "target": dv.target,
        # Los identificadores no son datos para aprender: no se muestran como entradas.
        "inputs": [
            c.name
            for c in card.columns
            if (not target or c.name != target.name) and c.semantic != "id"
        ],
        "splits": card.split_counts,
    }
    if target and target.classes:
        out["classes"] = [{"value": str(c.value), "count": c.count} for c in target.classes]
    if target and target.numeric:
        q = target.numeric.quantiles
        out["range"] = {"low": q.get("p05"), "typical": q.get("p50"), "high": q.get("p95")}
    if card.series is not None:
        out["series"] = {
            "count": card.series.num_series,
            "points": card.series.points_min,
            "horizon": card.series.horizon,
            "history": card.series.lookback,
        }
    if card.vision_task is not None and card.vision_task.classes:
        out["objects"] = list(card.vision_task.classes)
    return out


def _example_row(wf: Workflow, dv: Any) -> dict[str, Any]:
    """Un ejemplo de train para mostrar el formato esperado de cada campo (nunca test)."""
    try:
        df = wf.view(dv).read("train")
    except Exception:
        return {}
    return df.row(0, named=True) if df.height else {}


JUDGED = {"object_detection", "segmentation", "anomaly_detection"}
YES = {"si", "sí", "yes", "bien", "true", "1", "correcto"}


def _task(j: Journey) -> str:
    return str((j.contract or {}).get("outputs", {}).get("task") or "")


def _compare(j: Journey, key: str, actual: Any, prediction: Any) -> dict[str, Any]:
    task = _task(j)
    if task == "forecasting":
        # Pronóstico: los valores reales que vinieron después, contra lo pronosticado.
        try:
            later = [float(v) for v in actual]
            ahead = [float(v) for v in prediction][: len(later)]
        except (TypeError, ValueError) as exc:
            raise ValidationError("Los valores reales tienen que ser números.") from exc
        scale = sum(abs(v) for v in later[: len(ahead)]) or 1.0
        error = 100 * sum(abs(p - r) for p, r in zip(ahead, later, strict=False)) / scale
        return {
            "key": key,
            "actual": later,
            "prediction": ahead,
            "error_percent": round(error, 2),
            "correct": error <= (j.facts.success_target or 0),
        }
    if task == "ocr":

        def norm(v: Any) -> str:
            return " ".join(str(v).split()).casefold()

        return {
            "key": key,
            "actual": str(actual),
            "prediction": str(prediction),
            "correct": norm(actual) == norm(prediction),
        }
    if task in JUDGED:
        # Lo juzga la persona: vio lo que marcó el modelo y dice si está bien.
        verdict = str(actual).strip().casefold() in YES
        return {
            "key": key,
            "actual": "bien" if verdict else "mal",
            "prediction": prediction,
            "correct": verdict,
        }
    if task == "regression":
        try:
            real, pred = float(actual), float(prediction)
        except (TypeError, ValueError) as exc:
            raise ValidationError("El resultado real tiene que ser un número.") from exc
        tolerance = (j.facts.success_target or 0) / 100 * abs(real)
        error = abs(pred - real)
        return {
            "key": key,
            "actual": real,
            "prediction": pred,
            "error": error,
            "correct": error <= tolerance,
        }
    classes = [str(c) for c in (j.contract or {}).get("outputs", {}).get("classes", [])]
    if classes and str(actual) not in classes:
        raise ValidationError("El resultado real tiene que ser una de las categorías del modelo.")
    return {
        "key": key,
        "actual": str(actual),
        "prediction": str(prediction),
        "correct": str(actual) == str(prediction),
    }


def validation_needed(j: Journey) -> int:
    """Casos nuevos para dar el modelo por validado (en series, cada serie de un archivo)."""
    return 3 if _task(j) in ("forecasting", "anomaly_detection") else VALIDATION_MIN


def _validated_state(j: Journey) -> State:
    total = len(j.validations)
    if total < validation_needed(j):
        return j.state if j.state in MODEL_STATES else "ready_to_validate"
    classes = (
        set((j.contract or {}).get("outputs", {}).get("classes", []) or [])
        if _task(j) == "classification"
        else set()
    )
    covered = not classes or {v["actual"] for v in j.validations} >= classes
    rate = 100 * sum(v["correct"] for v in j.validations) / total
    meets = rate >= (j.facts.success_target or 100) and bool((j.result or {}).get("meets_target"))
    if j.facts.problem != "category":
        trained_ok = bool((j.result or {}).get("meets_target"))
        meets = trained_ok and rate >= (50 if _task(j) in ("regression", "forecasting") else 70)
    if not covered:
        return j.state if j.state in MODEL_STATES else "ready_to_validate"
    return "ready_to_use" if meets else "needs_improvement"


def validation_key(*, row: dict[str, Any] | None, text: str | None, data: bytes | None) -> str:
    if data is not None:
        return hashlib.sha256(data).hexdigest()
    payload = json.dumps(row, sort_keys=True, default=str) if row is not None else (text or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
