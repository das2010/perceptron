"""Pipelines, arquitectura, catálogo, HPO, estudios, runs, evaluación y registro (SPEC §10)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, WebSocket, status
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.jobs import JOB_TOPIC, TERMINAL, Job, JobContext
from perceptron.api.streaming import stream_events
from perceptron.archspec.schema import ArchSpec
from perceptron.archspec.to_code import archspec_to_code
from perceptron.archspec.validate import ValidationReport
from perceptron.catalog.registry import BLOCKS, blocks_for
from perceptron.core.errors import NotFoundError
from perceptron.data.pipeline.pipeline import PipelineSpec, transform_tabular
from perceptron.domain.enums import Device, Modality, Origin, TaskType
from perceptron.domain.models import ArchSpecRecord, Evaluation, ModelVersion, Pipeline, Run, Study
from perceptron.evaluation.evaluate import EvaluationReport
from perceptron.hpo.strategy import Budget, HPOStrategy
from perceptron.hpo.study import StudyControl
from perceptron.services.workflow import Workflow

router = APIRouter()
Ctx = Annotated[EngineContext, Depends(get_context)]
Mode = Literal["auto", "llm", "rules"]


# ------------------------------------------------------------------ pipelines


class ProposePipelineBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version_id: str
    pretrained: bool | None = None


class PipelineUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1)
    graph: PipelineSpec


class PipelinePreviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version_id: str
    rows: int = Field(default=10, ge=1, le=100)


class PipelinePreview(BaseModel):
    numeric_features: list[str]
    categorical_features: list[str]
    x_num: list[list[float]]
    x_cat: list[list[int]]
    classes: list[str] | None


@router.post(
    "/projects/{project_id}/pipelines/propose",
    status_code=201,
    tags=["pipelines"],
    operation_id="proposePipeline",
)
def propose_pipeline(project_id: str, body: ProposePipelineBody, ctx: Ctx) -> Pipeline:
    ctx.projects.get(project_id)
    return Workflow(ctx).propose_pipeline(body.dataset_version_id, pretrained=body.pretrained)


@router.put("/pipelines/{pipeline_id}", tags=["pipelines"], operation_id="updatePipeline")
def update_pipeline(pipeline_id: str, body: PipelineUpdate, ctx: Ctx) -> Pipeline:
    return Workflow(ctx).update_pipeline(pipeline_id, body.graph, body.version)


@router.post("/pipelines/{pipeline_id}/preview", tags=["pipelines"], operation_id="previewPipeline")
def preview_pipeline(pipeline_id: str, body: PipelinePreviewBody, ctx: Ctx) -> PipelinePreview:
    """Pipeline ajustado aplicado a las primeras filas de train (vista previa, RF-PIP-02)."""
    wf = Workflow(ctx)
    fitted = wf.fitted_pipeline(pipeline_id, body.dataset_version_id)
    view = wf.view(wf.dataset(body.dataset_version_id))
    if view.modality is not Modality.TABULAR:
        return PipelinePreview(
            numeric_features=[], categorical_features=[], x_num=[], x_cat=[], classes=fitted.classes
        )
    arr = transform_tabular(fitted, view.read("train").head(body.rows))
    return PipelinePreview(
        numeric_features=fitted.numeric_features,
        categorical_features=fitted.categorical_features,
        x_num=arr.x_num.round(6).tolist(),
        x_cat=arr.x_cat.tolist(),
        classes=fitted.classes,
    )


# ------------------------------------------------------------------ arquitectura


class ProposeArchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version_id: str
    pipeline_id: str
    mode: Mode = Field(
        default="auto", description="auto: LLM si está disponible; si no, reglas (RF-ARC-04)"
    )
    n: int = Field(default=3, ge=2, le=4, description="Propuestas pedidas al LLM (RF-ARC-01)")
    device: Device | None = None


class ArchEstimates(BaseModel):
    num_params: float | None = None
    memory_mb: float | None = None
    epoch_time_s: float | None = None


class ArchProposal(BaseModel):
    archspec: ArchSpecRecord
    title: str
    rationale: str
    validation: ValidationReport
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float | None = None
    estimates: ArchEstimates = Field(default_factory=ArchEstimates)


class ArchProposals(BaseModel):
    proposals: list[ArchProposal]
    origin: Origin
    llm_call_id: str | None = None
    fallback_reason: str | None = Field(
        default=None, description="Por qué se usaron reglas o se descartaron propuestas"
    )


class CodeResponse(BaseModel):
    code: str


@router.post(
    "/projects/{project_id}/arch/propose",
    status_code=201,
    tags=["arch"],
    operation_id="proposeArchitecture",
)
def propose_architecture(project_id: str, body: ProposeArchBody, ctx: Ctx) -> ArchProposals:
    """2–4 propuestas del LLM validadas, o la de reglas como fallback (RF-ARC-01..04).

    Cada propuesta ya queda guardada como ArchSpec (origin llm/rules): el usuario elige una
    por `archspec.id`, la edita o las descarta.
    """
    ctx.projects.get(project_id)
    out = Workflow(ctx).roles.propose_architectures(
        body.dataset_version_id,
        body.pipeline_id,
        mode=body.mode,
        n=body.n,
        device=body.device.value if body.device else None,
    )
    return ArchProposals(
        proposals=[
            ArchProposal(
                archspec=o.record,
                title=o.title,
                rationale=o.rationale,
                validation=o.validation,
                pros=o.pros,
                cons=o.cons,
                risks=o.risks,
                confidence=o.confidence,
                estimates=ArchEstimates(**o.estimates),
            )
            for o in out.options
        ],
        origin=out.origin,
        llm_call_id=out.llm_call_id,
        fallback_reason=out.fallback_reason,
    )


@router.post("/arch/validate", tags=["arch"], operation_id="validateArchitecture")
def validate_architecture(body: dict[str, Any]) -> ValidationReport:
    return Workflow.validate(body)


@router.post("/arch/to-code", tags=["arch"], operation_id="archToCode")
def arch_to_code(spec: ArchSpec) -> CodeResponse:
    return CodeResponse(code=archspec_to_code(spec))


@router.get("/catalog/blocks", tags=["arch"], operation_id="listCatalogBlocks")
def catalog_blocks(
    modality: Modality | None = None, task: TaskType | None = None
) -> list[dict[str, Any]]:
    blocks = blocks_for(modality, task) if modality else list(BLOCKS.values())
    return [b.public() for b in blocks]


# ------------------------------------------------------------------ HPO y estudios


class StrategyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    archspec_id: str
    budget: Budget = Field(default_factory=Budget)
    mode: Mode = "auto"
    dataset_version_id: str | None = Field(
        default=None, description="Dataset para darle al estratega el perfil (opcional)"
    )


class StudyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version_id: str
    pipeline_id: str
    archspec_id: str
    strategy: HPOStrategy | None = None
    budget: Budget = Field(default_factory=Budget)
    device: Device | None = None


class StudyLaunch(BaseModel):
    study: Study
    job: Job


@router.post(
    "/projects/{project_id}/hpo/strategy", tags=["hpo"], operation_id="recommendHpoStrategy"
)
def hpo_strategy(project_id: str, body: StrategyBody, ctx: Ctx) -> HPOStrategy:
    ctx.projects.get(project_id)
    return Workflow(ctx).hpo_strategy(
        body.archspec_id,
        body.budget,
        mode=body.mode,
        dataset_version_id=body.dataset_version_id,
    )


def _launch_study(
    ctx: EngineContext, study: Study, body: StudyCreate, strategy: HPOStrategy
) -> Job:
    control = StudyControl()

    def work(job: JobContext) -> Any:
        job.cancel_callback = control.cancel

        def on_event(ev: Any) -> None:
            if ev.event == "epoch":
                job.emit("epoch", run_id=ev.run_id, epoch=ev.epoch, metrics=ev.metrics)

        _, result = Workflow(ctx).run_study(
            study.project_id,
            body.dataset_version_id,
            body.pipeline_id,
            body.archspec_id,
            strategy,
            device=body.device,
            control=control,
            on_event=on_event,
            study=study,
        )
        return result

    return ctx.jobs.submit(
        "study", work, refs={"study_id": study.id, "project_id": study.project_id}
    )


@router.post(
    "/projects/{project_id}/studies",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["hpo"],
    operation_id="createStudy",
)
def create_study(project_id: str, body: StudyCreate, ctx: Ctx) -> StudyLaunch:
    ctx.projects.get(project_id)
    wf = Workflow(ctx)
    strategy = body.strategy or wf.hpo_strategy(body.archspec_id, body.budget)
    record = ctx.repo(ArchSpecRecord).get(body.archspec_id)
    study = ctx.repo(Study).add(
        Study(
            project_id=project_id,
            name=f"hpo-{record.name}",
            strategy=strategy.model_dump(mode="json"),
            budget={
                **strategy.budget.model_dump(mode="json"),
                "request": body.model_dump(mode="json"),
            },
            objectives=[o.metric for o in strategy.objectives],
            origin=strategy.origin,
        )
    )
    return StudyLaunch(study=study, job=_launch_study(ctx, study, body, strategy))


@router.get("/studies/{study_id}", tags=["hpo"], operation_id="getStudy")
def get_study(study_id: str, ctx: Ctx) -> Study:
    return ctx.repo(Study).get(study_id)


def _study_job(ctx: EngineContext, study_id: str) -> Job | None:
    return next(
        (
            j
            for j in ctx.jobs.list()
            if j.refs.get("study_id") == study_id and j.status not in TERMINAL
        ),
        None,
    )


@router.post("/studies/{study_id}/cancel", tags=["hpo"], operation_id="cancelStudy")
def cancel_study(study_id: str, ctx: Ctx) -> Job:
    job = _study_job(ctx, study_id)
    if job is None:
        raise NotFoundError(f"el estudio {study_id} no está en ejecución")
    return ctx.jobs.cancel(job.id) or job


@router.post("/studies/{study_id}/pause", tags=["hpo"], operation_id="pauseStudy")
def pause_study(study_id: str, ctx: Ctx) -> Job:
    """Detiene el estudio; los trials terminados quedan en el storage y se puede reanudar."""
    return cancel_study(study_id, ctx)


@router.post(
    "/studies/{study_id}/resume",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["hpo"],
    operation_id="resumeStudy",
)
def resume_study(study_id: str, ctx: Ctx) -> StudyLaunch:
    study = ctx.repo(Study).get(study_id)
    body = StudyCreate.model_validate(study.budget.get("request", {}))
    strategy = HPOStrategy.model_validate(study.strategy)
    return StudyLaunch(study=study, job=_launch_study(ctx, study, body, strategy))


# ------------------------------------------------------------------ runs


class CompareBody(BaseModel):
    run_ids: list[str] = Field(min_length=1, max_length=50)


@router.get("/projects/{project_id}/runs", tags=["runs"], operation_id="listRuns")
def list_runs(project_id: str, ctx: Ctx, study_id: str | None = None) -> list[Run]:
    filters = {"project_id": project_id, **({"study_id": study_id} if study_id else {})}
    return list(ctx.repo(Run).list(filters=filters, limit=500))


@router.get("/runs/{run_id}", tags=["runs"], operation_id="getRun")
def get_run(run_id: str, ctx: Ctx) -> Run:
    return ctx.repo(Run).get(run_id)


@router.post("/runs/compare", tags=["runs"], operation_id="compareRuns")
def compare_runs(body: CompareBody, ctx: Ctx) -> list[dict[str, Any]]:
    runs = [ctx.repo(Run).get(r) for r in body.run_ids]
    return [
        {
            "run_id": r.id,
            "status": r.status.value,
            "hyperparams": r.hyperparams,
            "metrics": r.metrics,
        }
        for r in runs
    ]


@router.websocket("/runs/{run_id}/live")
async def run_live(ws: WebSocket, run_id: str) -> None:
    ctx: EngineContext = ws.app.state.ctx
    await stream_events(
        ws,
        ctx.events,
        "run.event",
        lambda ev: ev.payload.get("run_id") == run_id,
        until=lambda msg: msg.get("event", {}).get("event") in ("finished", "error", "paused"),
    )


@router.post("/runs/{run_id}/evaluate", tags=["runs"], operation_id="evaluateRun")
def evaluate_run(run_id: str, ctx: Ctx) -> EvaluationReport:
    _, report = Workflow(ctx).evaluate(run_id)
    return report


@router.get("/runs/{run_id}/evaluation", tags=["runs"], operation_id="getEvaluation")
def get_evaluation(run_id: str, ctx: Ctx) -> EvaluationReport:
    return Workflow(ctx).evaluation_report(run_id)


@router.get("/runs/{run_id}/evaluations", tags=["runs"], operation_id="listEvaluations")
def list_evaluations(run_id: str, ctx: Ctx) -> list[Evaluation]:
    return list(ctx.repo(Evaluation).list(filters={"run_id": run_id}))


@router.post(
    "/runs/{run_id}/register", status_code=201, tags=["models"], operation_id="registerModel"
)
def register_model(run_id: str, ctx: Ctx) -> ModelVersion:
    return Workflow(ctx).register(run_id)


@router.get("/projects/{project_id}/models", tags=["models"], operation_id="listModels")
def list_models(project_id: str, ctx: Ctx) -> list[ModelVersion]:
    return list(ctx.repo(ModelVersion).list(filters={"project_id": project_id}))


# ------------------------------------------------------------------ jobs


@router.get("/jobs", tags=["jobs"], operation_id="listJobs")
def list_jobs(ctx: Ctx) -> list[Job]:
    return ctx.jobs.list()


@router.get("/jobs/{job_id}", tags=["jobs"], operation_id="getJob")
def get_job(job_id: str, ctx: Ctx) -> Job:
    job = ctx.jobs.get(job_id)
    if job is None:
        raise NotFoundError(f"job {job_id} no existe")
    return job


@router.websocket("/jobs/{job_id}")
async def job_stream(ws: WebSocket, job_id: str) -> None:
    ctx: EngineContext = ws.app.state.ctx
    job = ctx.jobs.get(job_id)
    initial = [
        {"job_id": job_id, "kind": "status", "data": {"status": job.status if job else "unknown"}}
    ]
    done = job is None or job.status in TERMINAL
    await stream_events(
        ws,
        ctx.events,
        JOB_TOPIC,
        lambda ev: ev.payload.get("job_id") == job_id,
        initial=initial,
        until=lambda msg: done or msg.get("kind") == "finished",
    )
