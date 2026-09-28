"""Export de modelos entrenados (RF-EXP-01, RF-EXP-05; ADR-0027)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from perceptron.api.context import EngineContext, get_context
from perceptron.api.jobs import Job, JobContext
from perceptron.domain.models import Run
from perceptron.export.formats import ExportReport, ExportRequest
from perceptron.services.workflow import Workflow

router = APIRouter(tags=["export"])
Ctx = Annotated[EngineContext, Depends(get_context)]


class ExportLaunch(BaseModel):
    job: Job


@router.post(
    "/runs/{run_id}/export",
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="exportRun",
)
def export_run(run_id: str, body: ExportRequest, ctx: Ctx) -> ExportLaunch:
    """Exporta el modelo del run (job): ONNX, torch.export y/o TorchScript, verificados."""
    run = ctx.repo(Run).get(run_id)

    def work(_: JobContext) -> Any:
        return Workflow(ctx).export(run_id, body).model_dump(mode="json")

    job = ctx.jobs.submit("export", work, refs={"run_id": run_id, "project_id": run.project_id})
    return ExportLaunch(job=job)


@router.get("/runs/{run_id}/export", operation_id="getRunExport")
def get_run_export(run_id: str, ctx: Ctx) -> ExportReport:
    return Workflow(ctx).export_report(run_id)


@router.get("/runs/{run_id}/export/files/{name}", operation_id="downloadRunExport")
def download_run_export(run_id: str, name: str, ctx: Ctx) -> FileResponse:
    path = Workflow(ctx).export_file(run_id, name)
    return FileResponse(path, filename=name, media_type="application/octet-stream")
