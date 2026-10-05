"""Capa LLM (SPEC §10): proveedores, perfiles, auditoría, prueba de conexión y roles.

Las claves se escriben (write-only) y nunca se devuelven: la respuesta solo dice si hay una
guardada (RF-LLM-08). La auditoría muestra exactamente qué salió en cada llamada (RF-PRV-03).
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from perceptron.api.context import EngineContext, get_context
from perceptron.archspec.defaults import spec_epochs
from perceptron.archspec.schema import ArchSpec, Provenance
from perceptron.core.errors import NotFoundError, ValidationError
from perceptron.domain.enums import LLMPurpose, Origin, PrivacyLevel
from perceptron.domain.models import ArchSpecRecord, LabelSet, LLMCall, Project
from perceptron.llm.config import LLMProfile, ModelInfo, ProviderInfo, ProviderKind
from perceptron.llm.errors import LLMBudgetExceededError, LLMProviderError, LLMUnavailableError
from perceptron.llm.privacy import LLMContext
from perceptron.llm.schemas import Diagnosis, LabelingGuide, Report
from perceptron.services.improve import (
    ImprovementOption,
    apply_action,
    improvement_options,
    with_defaults,
)
from perceptron.services.workflow import Workflow

router = APIRouter()
Ctx = Annotated[EngineContext, Depends(get_context)]
Mode = Literal["auto", "llm", "rules"]


# ------------------------------------------------------------------ proveedores y perfiles


class ProviderView(BaseModel):
    name: str
    kind: ProviderKind
    base_url: str | None
    api_key_ref: str | None
    local: bool
    has_key: bool
    models: dict[str, ModelInfo]


class ProviderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind
    base_url: str | None = None
    api_key_ref: str | None = None
    local: bool | None = None
    timeout_s: float | None = Field(default=None, gt=0, description="None = conservar")
    models: dict[str, ModelInfo] | None = None
    api_key: SecretStr | None = Field(default=None, description="Solo escritura: va al keychain")


class ProfilesView(BaseModel):
    active: str
    profiles: dict[str, LLMProfile]


class ProfilesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: str | None = None
    profiles: dict[str, LLMProfile] = Field(default_factory=dict)


def provider_views(ctx: EngineContext) -> list[ProviderView]:
    gw = ctx.llm
    return [
        ProviderView(
            name=name,
            kind=info.kind,
            base_url=info.base_url,
            api_key_ref=info.api_key_ref,
            local=info.is_local,
            has_key=bool(info.api_key_ref and gw.secrets.get(info.api_key_ref)),
            models=info.models,
        )
        for name, info in sorted(gw.config.providers().items())
    ]


@router.get("/llm/providers", tags=["llm"], operation_id="listLlmProviders")
def list_providers(ctx: Ctx) -> list[ProviderView]:
    return provider_views(ctx)


@router.put("/llm/providers/{name}", tags=["llm"], operation_id="putLlmProvider")
def put_provider(name: str, body: ProviderUpdate, ctx: Ctx) -> list[ProviderView]:
    gw = ctx.llm
    current = gw.config.providers().get(name)
    ref = body.api_key_ref or (current.api_key_ref if current else None)
    if body.api_key is not None:
        ref = ref or f"PERCEPTRON_LLM_{name.upper()}_KEY"
        gw.secrets.set(ref, body.api_key.get_secret_value())
    models = body.models if body.models is not None else (current.models if current else {})
    cfg = gw.config.workspace()
    cfg.providers[name] = ProviderInfo(
        kind=body.kind,
        base_url=body.base_url,
        api_key_ref=ref,
        local=body.local,
        timeout_s=body.timeout_s or (current.timeout_s if current else 120.0),
        models=models,
    )
    gw.config.save_workspace(cfg)
    return provider_views(ctx)


@router.get("/llm/profiles", tags=["llm"], operation_id="getLlmProfiles")
def get_profiles(ctx: Ctx) -> ProfilesView:
    cfg = ctx.llm.config
    return ProfilesView(active=cfg.active_profile(), profiles=cfg.profiles())


@router.put("/llm/profiles", tags=["llm"], operation_id="putLlmProfiles")
def put_profiles(body: ProfilesUpdate, ctx: Ctx) -> ProfilesView:
    cfg = ctx.llm.config
    ws = cfg.workspace()
    ws.profiles.update(body.profiles)
    if body.active is not None:
        ws.active_profile = body.active
    cfg.save_workspace(ws)
    return get_profiles(ctx)


# ------------------------------------------------------------------ auditoría y prueba


@router.get("/llm/audit", tags=["llm"], operation_id="listLlmAudit")
def audit(
    ctx: Ctx,
    project: str,
    purpose: LLMPurpose | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[LLMCall]:
    """Cada llamada con el payload tal como salió (post-filtro), proveedor y modelo."""
    filters: dict[str, str] = {"project_id": project}
    if purpose is not None:
        filters["purpose"] = purpose.value
    return list(ctx.repo(LLMCall).list(filters=filters, limit=limit, offset=offset))


class LLMTestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str | None = None
    purpose: LLMPurpose = LLMPurpose.COPILOT


class Ping(BaseModel):
    ok: bool
    message: str = ""


class LLMTestResult(BaseModel):
    ok: bool
    provider: str | None = None
    model: str | None = None
    latency_s: float | None = None
    error: str | None = None


@router.post("/llm/test", tags=["llm"], operation_id="testLlm")
def test_llm(body: LLMTestBody, ctx: Ctx) -> LLMTestResult:
    project = (
        ctx.projects.get(body.project_id)
        if body.project_id
        else Project(name="llm-test", privacy_level=PrivacyLevel.L1)
    )
    started = time.perf_counter()
    try:
        out = ctx.llm.structured(
            body.purpose, Ping, LLMContext(), project=project, prompt="ping", scope="llm-test"
        )
    except (LLMUnavailableError, LLMProviderError, LLMBudgetExceededError) as e:
        return LLMTestResult(ok=False, error=f"{e.code}: {e.message}")
    return LLMTestResult(
        ok=out.value.ok,
        provider=out.provider,
        model=out.model,
        latency_s=round(time.perf_counter() - started, 3),
    )


# ------------------------------------------------------------------ diagnóstico e informe


@router.get("/runs/{run_id}/diagnosis", tags=["runs"], operation_id="getRunDiagnosis")
def get_diagnosis(run_id: str, ctx: Ctx, mode: Mode = "auto", refresh: bool = False) -> Diagnosis:
    """Diagnóstico guardado; si no hay (o `refresh`), se calcula (reglas + LLM)."""
    from perceptron.domain.models import Run

    run = ctx.repo(Run).get(run_id)
    if run.diagnosis and "summary" in run.diagnosis and not refresh:
        return Diagnosis.model_validate(run.diagnosis)
    return Workflow(ctx).roles.diagnose(run_id, mode=mode)


def _diagnosis(ctx: EngineContext, run_id: str) -> Diagnosis:
    from perceptron.domain.models import Run

    run = ctx.repo(Run).get(run_id)
    if run.diagnosis and "summary" in run.diagnosis:
        return Diagnosis.model_validate(run.diagnosis)
    return Workflow(ctx).roles.diagnose(run_id)


class ImprovementApplied(BaseModel):
    archspec: ArchSpecRecord
    change: str
    budget: dict[str, Any] = Field(
        default_factory=dict, description="Presupuesto sugerido (el del estudio del run)"
    )


@router.get(
    "/runs/{run_id}/improvements",
    tags=["runs"],
    operation_id="listRunImprovements",
)
def list_improvements(run_id: str, ctx: Ctx) -> list[ImprovementOption]:
    """Acciones del diagnóstico traducidas a cambios concretos de la arquitectura (ADR-0041)."""
    from perceptron.domain.models import Run

    run = ctx.repo(Run).get(run_id)
    spec = ArchSpec.model_validate(ctx.repo(ArchSpecRecord).get(run.archspec_id).spec)
    return improvement_options(spec, _diagnosis(ctx, run_id), run.hyperparams)


@router.post(
    "/runs/{run_id}/improvements/{index}",
    status_code=201,
    tags=["runs"],
    operation_id="applyRunImprovement",
)
def apply_improvement(run_id: str, index: int, ctx: Ctx) -> ImprovementApplied:
    """Guarda la ArchSpec mejorada (desde el mejor punto del run). No entrena: eso lo decide la
    persona con el presupuesto sugerido."""
    from perceptron.domain.models import Run, Study

    run = ctx.repo(Run).get(run_id)
    diagnosis = _diagnosis(ctx, run_id)
    if not 0 <= index < len(diagnosis.actions):
        raise NotFoundError(f"el diagnóstico no tiene la acción {index}")
    record = ctx.repo(ArchSpecRecord).get(run.archspec_id)
    base = with_defaults(ArchSpec.model_validate(record.spec), run.hyperparams)
    action = diagnosis.actions[index]
    applied = apply_action(base, action)
    if applied is None:
        raise ValidationError(
            "esa acción no se aplica sobre la arquitectura", details={"kind": action.kind}
        )
    spec, change = applied
    origin = Origin.LLM if diagnosis.origin == "llm" else Origin.RULES
    spec = spec.model_copy(
        update={
            "name": f"{base.name[:110]}-mejora",
            "provenance": Provenance(
                origin=origin,
                llm_call_id=diagnosis.llm_call_id,
                rationale=f"Mejora del run {run_id}: {change}. {action.rationale}",
            ),
        }
    )
    saved = Workflow(ctx).save_archspec(run.project_id, spec, origin=origin)
    budget: dict[str, Any] = {}
    if run.study_id:
        study = ctx.repo(Study).find(run.study_id)
        budget = dict(study.budget) if study else {}
    if action.kind in ("more_epochs", "fewer_epochs"):
        budget["max_epochs_per_trial"] = int(spec_epochs(spec))
    return ImprovementApplied(archspec=saved, change=change, budget=budget)


class ReportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode = "auto"
    language: Literal["español", "english"] = "español"


@router.post("/runs/{run_id}/report", tags=["runs"], operation_id="createRunReport")
def create_report(run_id: str, body: ReportBody, ctx: Ctx) -> Report:
    return Workflow(ctx).roles.report(run_id, mode=body.mode, language=body.language)


# ------------------------------------------------------------------ etiquetado asistido


class GuideBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: dict[str, str] = Field(min_length=1, description="Clase → descripción del usuario")
    mode: Mode = "auto"


@router.post(
    "/projects/{project_id}/labels/guide", tags=["labels"], operation_id="createLabelingGuide"
)
def labeling_guide(project_id: str, body: GuideBody, ctx: Ctx) -> LabelingGuide:
    return Workflow(ctx).roles.labeling_guide(project_id, body.classes, mode=body.mode)


class PrelabelBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    guide: LabelingGuide
    column: str | None = None
    limit: int = Field(default=100, ge=1, le=2000)


@router.post("/labels/prelabel", status_code=201, tags=["labels"], operation_id="prelabel")
def prelabel(body: PrelabelBody, ctx: Ctx) -> LabelSet:
    """Pre-etiquetado de texto con el LLM (requiere privacidad L2+, RF-LBL-02)."""
    return Workflow(ctx).roles.prelabel(
        body.dataset_version_id, body.guide, column=body.column, limit=body.limit
    )
