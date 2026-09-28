"""Export de modelos entrenados (RF-EXP-01, RF-EXP-05; ADR-0027)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.jobs import Job, JobContext
from perceptron.api.security import MB, read_limited
from perceptron.core.errors import ValidationError
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


@router.get("/runs/{run_id}/export/serving.zip", operation_id="downloadServingBundle")
def download_serving_bundle(run_id: str, ctx: Ctx) -> FileResponse:
    """Servidor de inferencia listo para Docker (RF-EXP-03): FastAPI + ONNX Runtime."""
    path = Workflow(ctx).serving_bundle(run_id)
    return FileResponse(path, filename=f"{run_id}-serving.zip", media_type="application/zip")


@router.get("/runs/{run_id}/export/project.zip", operation_id="downloadExportProject")
def download_export_project(run_id: str, ctx: Ctx) -> FileResponse:
    """Proyecto de código autónomo (RF-EXP-04): uv, modelo generado, train/infer/serve y datos."""
    path = Workflow(ctx).project_zip(run_id)
    return FileResponse(path, filename=f"{run_id}-proyecto.zip", media_type="application/zip")


class PlaygroundRows(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class PlaygroundTexts(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=1000)


class PlaygroundResult(BaseModel):
    predictions: list[dict[str, Any]]
    task: str
    input_kind: str


def _predict(run_id: str, ctx: EngineContext, call: Any) -> PlaygroundResult:
    from perceptron.serving.runtime import InputError

    model = Workflow(ctx).inference_model(run_id)
    try:
        preds = call(model)
    except InputError as exc:
        raise ValidationError(str(exc)) from exc
    return PlaygroundResult(
        predictions=[p.to_dict() for p in preds], task=model.task, input_kind=model.kind
    )


@router.post("/runs/{run_id}/predict", operation_id="predictRows")
def predict_rows(run_id: str, body: PlaygroundRows, ctx: Ctx) -> PlaygroundResult:
    """Playground (RF-EXP-02): filas de tabla con las columnas originales."""
    return _predict(run_id, ctx, lambda m: m.predict_rows(body.rows))


@router.post("/runs/{run_id}/predict/text", operation_id="predictTexts")
def predict_texts(run_id: str, body: PlaygroundTexts, ctx: Ctx) -> PlaygroundResult:
    """Playground (RF-EXP-02): textos para modelos de texto."""
    return _predict(run_id, ctx, lambda m: m.predict_texts(body.texts))


AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")


@router.post("/runs/{run_id}/predict/file", operation_id="predictFile")
async def predict_file(
    run_id: str, ctx: Ctx, file: Annotated[UploadFile, File()]
) -> PlaygroundResult:
    """Playground: una imagen, un audio, un texto (.txt: una muestra por línea) o un CSV."""
    import csv
    import io
    import tempfile
    from pathlib import Path

    data = await read_limited(file, 256 * MB)
    suffix = Path(file.filename or "").suffix.lower()

    def call(model: Any) -> Any:
        if model.kind == "tabular":
            rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
            return model.predict_rows(rows[:1000])
        if model.kind == "tokens":
            text = data.decode("utf-8-sig", errors="replace")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return model.predict_texts(lines[:1000])
        if model.kind == "spectrogram":
            if suffix not in AUDIO_SUFFIXES:
                raise ValidationError(
                    f"formato de audio no soportado: {suffix or '(sin extensión)'}"
                )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"audio{suffix}"
                path.write_bytes(data)
                return model.predict_audio([path])
        from PIL import Image, UnidentifiedImageError

        try:
            return model.predict_images([Image.open(io.BytesIO(data))])
        except UnidentifiedImageError as exc:
            raise ValidationError("el archivo no es una imagen válida") from exc

    return _predict(run_id, ctx, call)
