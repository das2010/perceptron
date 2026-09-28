"""Sugerencias de cambios al pipeline (RF-PIP-05): el LLM (o las reglas) propone cambios
tipados con justificación y el usuario acepta o descarta cada uno viendo el diff.

Un cambio es agregar, quitar o modificar un paso. Toda sugerencia se valida aplicándola: el
paso tiene que existir en el catálogo y sus columnas en el dataset. Sin LLM (o si falla), las
sugerencias salen de comparar el pipeline actual con el que propondrían las reglas.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from perceptron.core.errors import ConflictError, ValidationError
from perceptron.data.pipeline.pipeline import PipelineSpec
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.pipeline.steps import STEPS, StepSpec
from perceptron.domain.enums import LLMPurpose, Origin
from perceptron.domain.models import Pipeline

if TYPE_CHECKING:
    from perceptron.services.llm_roles import LLMRoles, Mode

logger = logging.getLogger(__name__)

MAX_SUGGESTIONS = 8


class PipelineChange(BaseModel):
    op: Literal["add", "remove", "update"]
    step_id: str = Field(description="Paso afectado (el nuevo, en `add`)")
    step: StepSpec | None = Field(default=None, description="Paso completo en `add` y `update`")
    after: str | None = Field(
        default=None, description="`add`: id del paso después del cual va (None = al final)"
    )
    rationale: str = Field(min_length=1, max_length=600)


class PipelineSuggestions(BaseModel):
    suggestions: list[PipelineChange] = Field(default_factory=list, max_length=MAX_SUGGESTIONS)


class SuggestionItem(BaseModel):
    """Una sugerencia con su diff: el paso antes y después (None si no existe)."""

    index: int
    change: PipelineChange
    before: StepSpec | None
    after: StepSpec | None


class SuggestionsResult(BaseModel):
    pipeline_id: str
    pipeline_version: int
    origin: Origin
    llm_call_id: str | None = None
    fallback_reason: str | None = None
    items: list[SuggestionItem]


def apply_change(spec: PipelineSpec, change: PipelineChange) -> PipelineSpec:
    steps = list(spec.steps)
    ids = [s.id for s in steps]
    if change.op == "remove":
        if change.step_id not in ids:
            raise ValidationError(f"no existe el paso {change.step_id}")
        steps = [s for s in steps if s.id != change.step_id]
    else:
        if change.step is None:
            raise ValidationError(f"`{change.op}` necesita el paso completo")
        step = change.step.model_copy(update={"id": change.step_id})
        if change.op == "update":
            if change.step_id not in ids:
                raise ValidationError(f"no existe el paso {change.step_id}")
            steps = [step if s.id == change.step_id else s for s in steps]
        else:
            if change.step_id in ids:
                raise ValidationError(f"ya existe un paso {change.step_id}")
            pos = ids.index(change.after) + 1 if change.after in ids else len(steps)
            steps.insert(pos, step)
    return spec.model_copy(update={"steps": steps})


def check_change(spec: PipelineSpec, change: PipelineChange, columns: set[str]) -> str | None:
    """Motivo por el que la sugerencia no sirve, o None."""
    try:
        apply_change(spec, change)
    except ValidationError as e:
        return e.message
    if change.step is not None:
        if change.step.kind not in STEPS:
            return f"paso desconocido {change.step.kind} (válidos: {sorted(STEPS)})"
        missing = [c for c in change.step.columns if c not in columns]
        if missing:
            return f"columnas inexistentes en el dataset: {missing}"
    return None


def _diff_item(index: int, spec: PipelineSpec, change: PipelineChange) -> SuggestionItem:
    before = next((s for s in spec.steps if s.id == change.step_id), None)
    after = None if change.op == "remove" else change.step
    if after is not None:
        after = after.model_copy(update={"id": change.step_id})
    return SuggestionItem(index=index, change=change, before=before, after=after)


def rules_suggestions(current: PipelineSpec, proposed: PipelineSpec) -> list[PipelineChange]:
    """Lo que cambiaría para llegar al pipeline que proponen las reglas."""
    have = {s.id: s for s in current.steps}
    want = {s.id: s for s in proposed.steps}
    changes: list[PipelineChange] = []
    order = [s.id for s in proposed.steps]
    for sid in order:
        if sid not in have:
            prev = order[: order.index(sid)]
            anchor = next((p for p in reversed(prev) if p in have), None)
            changes.append(
                PipelineChange(
                    op="add",
                    step_id=sid,
                    step=want[sid],
                    after=anchor,
                    rationale="Lo propone el perfil de los datos (reglas).",
                )
            )
        elif have[sid] != want[sid]:
            changes.append(
                PipelineChange(
                    op="update",
                    step_id=sid,
                    step=want[sid],
                    rationale="Las reglas ajustarían columnas o parámetros de este paso.",
                )
            )
    for sid in have:
        if sid not in want:
            changes.append(
                PipelineChange(
                    op="remove",
                    step_id=sid,
                    rationale="Las reglas no incluyen este paso para estos datos.",
                )
            )
    return changes[:MAX_SUGGESTIONS]


def suggest(
    roles: LLMRoles, pipeline_id: str, dataset_version_id: str, *, mode: Mode = "auto"
) -> SuggestionsResult:
    from perceptron.llm.privacy.context import LLMContext
    from perceptron.services.llm_roles import FALLBACK, _reason

    ctx, wf = roles.ctx, roles.wf
    pipeline = ctx.repo(Pipeline).get(pipeline_id)
    spec = PipelineSpec.model_validate(pipeline.graph)
    card = wf.profile_card(dataset_version_id)
    columns = {c.name for c in card.columns}
    project = wf.project(pipeline.project_id)
    reason = roles._skip(mode, LLMPurpose.ARCHITECT, project)
    if reason is None:

        def validator(out: PipelineSuggestions) -> str | None:
            problems = [
                f"sugerencia {i}: {why}"
                for i, c in enumerate(out.suggestions)
                if (why := check_change(spec, c, columns))
            ]
            return "; ".join(problems) or None

        llm_ctx = LLMContext(
            goal=project.goal or None,
            card=card,
            pipeline=spec.model_dump(mode="json", exclude={"rationale"}),
            constraints={
                "step_kinds": sorted(STEPS),
                "columns": sorted(columns),
                "max_suggestions": MAX_SUGGESTIONS,
            },
        )
        try:
            out = roles.gateway.structured(
                LLMPurpose.ARCHITECT,
                PipelineSuggestions,
                llm_ctx,
                project=project,
                validator=validator,
                prompt="pipeline",
            )
            items = [_diff_item(i, spec, c) for i, c in enumerate(out.value.suggestions)]
            return SuggestionsResult(
                pipeline_id=pipeline.id,
                pipeline_version=pipeline.version,
                origin=Origin.LLM,
                llm_call_id=out.call_id,
                items=items,
            )
        except FALLBACK as e:
            reason = _reason(e)
            logger.info("sugerencias de pipeline por reglas", extra={"reason": reason})
    hf = spec.text.hf_model if spec.text and spec.text.tokenizer == "hf" else None
    proposed = propose_pipeline(card, pretrained=hf is not None, hf_model=hf)
    changes = [c for c in rules_suggestions(spec, proposed) if not check_change(spec, c, columns)]
    return SuggestionsResult(
        pipeline_id=pipeline.id,
        pipeline_version=pipeline.version,
        origin=Origin.RULES,
        fallback_reason=reason,
        items=[_diff_item(i, spec, c) for i, c in enumerate(changes)],
    )


def apply_accepted(
    roles: LLMRoles, pipeline_id: str, changes: list[PipelineChange], version: int
) -> Pipeline:
    """Aplica solo los cambios que el usuario aceptó (en orden); bloqueo optimista."""
    ctx = roles.ctx
    pipeline = ctx.repo(Pipeline).get(pipeline_id)
    if pipeline.version != version:
        raise ConflictError(
            "el pipeline cambió desde que se pidieron las sugerencias",
            details={"expected": pipeline.version, "got": version},
        )
    spec = PipelineSpec.model_validate(pipeline.graph)
    for change in changes:
        if change.step is not None and change.step.kind not in STEPS:
            raise ValidationError(f"paso desconocido: {change.step.kind}")
        spec = apply_change(spec, change)
    updated: Pipeline = roles.wf.update_pipeline(pipeline.id, spec, pipeline.version)
    return updated
