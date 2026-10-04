"""Fórmula sugerida (ADR-0039): regresión simbólica como modelo de referencia."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.jobs import Job, JobContext
from perceptron.domain.models import SymbolicFit
from perceptron.evaluation.symbolic import SymbolicConfig
from perceptron.services.symbolic import predict_symbolic, run_symbolic, symbolic_view

router = APIRouter(tags=["symbolic"])
Ctx = Annotated[EngineContext, Depends(get_context)]
MAX_PREDICT_ROWS = 1000


class SymbolicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    time_limit_s: int = Field(default=60, ge=5, le=1800, description="Tope total de búsqueda")


class SymbolicLaunch(BaseModel):
    job: Job


class SymbolicPredictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, float | None]] = Field(min_length=1, max_length=MAX_PREDICT_ROWS)


class SymbolicPredictions(BaseModel):
    predictions: list[float | None]


@router.post(
    "/projects/{project_id}/symbolic",
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="startSymbolicFit",
)
def start_symbolic(project_id: str, body: SymbolicRequest, ctx: Ctx) -> SymbolicLaunch:
    """Busca una fórmula cerrada para el target (job). 422 si el dataset no la admite."""
    ctx.projects.get(project_id)
    symbolic_view(ctx, project_id, body.dataset_version_id)  # valida antes de encolar
    config = SymbolicConfig(time_limit_s=body.time_limit_s)

    def work(job: JobContext) -> Any:
        def progress(done: int, total: int) -> None:
            job.emit("progress", done=done, total=total)

        fit = run_symbolic(ctx, project_id, body.dataset_version_id, config, progress)
        return {"symbolic_fit_id": fit.id}

    refs = {"project_id": project_id, "dataset_version_id": body.dataset_version_id}
    return SymbolicLaunch(job=ctx.jobs.submit("symbolic", work, refs=refs))


@router.get("/projects/{project_id}/symbolic", operation_id="listSymbolicFits")
def list_symbolic(project_id: str, ctx: Ctx) -> list[SymbolicFit]:
    """Fórmulas buscadas en el proyecto, de la más nueva a la más vieja."""
    ctx.projects.get(project_id)
    return list(ctx.repo(SymbolicFit).list(filters={"project_id": project_id}, limit=100))


@router.post("/symbolic/{symbolic_fit_id}/predict", operation_id="predictSymbolic")
def predict(symbolic_fit_id: str, body: SymbolicPredictBody, ctx: Ctx) -> SymbolicPredictions:
    """Valores de la fórmula para filas nuevas (como el playground de un modelo)."""
    fit = ctx.repo(SymbolicFit).get(symbolic_fit_id)
    return SymbolicPredictions(predictions=predict_symbolic(fit, body.rows))
