"""Evaluación avanzada de un run (RF-EVL-02..05): errores, fairness, explicaciones, robustez."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.security import MB, read_limited
from perceptron.core.errors import ValidationError
from perceptron.evaluation.errors import ErrorAnalysis
from perceptron.evaluation.explain import GlobalExplanation, LocalExplanation
from perceptron.evaluation.fairness import DEFAULT_THRESHOLD, FairnessReport
from perceptron.evaluation.robustness import RobustnessReport
from perceptron.services.workflow import Workflow

router = APIRouter(tags=["analysis"])
Ctx = Annotated[EngineContext, Depends(get_context)]


@router.get("/runs/{run_id}/errors", operation_id="getErrorAnalysis")
def error_analysis(run_id: str, ctx: Ctx) -> ErrorAnalysis:
    """Slices de bajo rendimiento, confusiones, posibles errores de etiqueta y mal predichos."""
    result: ErrorAnalysis = Workflow(ctx).error_analysis(run_id)
    return result


class FairnessBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attributes: list[str] = Field(min_length=1, max_length=10)
    positive_class: str | None = None
    threshold: float = Field(default=DEFAULT_THRESHOLD, gt=0, lt=1)


@router.post("/runs/{run_id}/fairness", operation_id="computeFairness")
def compute_fairness(run_id: str, body: FairnessBody, ctx: Ctx) -> list[FairnessReport]:
    """Métricas por subgrupo de los atributos sensibles que marca el usuario (RF-EVL-04)."""
    result: list[FairnessReport] = Workflow(ctx).fairness(
        run_id, body.attributes, positive_class=body.positive_class, threshold=body.threshold
    )
    return result


@router.get("/runs/{run_id}/explain", operation_id="getExplanation")
def explanation(run_id: str, ctx: Ctx) -> GlobalExplanation:
    """Importancia global de las features (Shapley por muestreo sobre validación, RF-EVL-02)."""
    result: GlobalExplanation = Workflow(ctx).explanation(run_id)
    return result


class ExplainRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: dict[str, Any]


@router.post("/runs/{run_id}/explain/row", operation_id="explainRow")
def explain_row(run_id: str, body: ExplainRow, ctx: Ctx) -> LocalExplanation:
    """Explicación local de una fila: contribución de cada feature a la predicción."""
    result: LocalExplanation = Workflow(ctx).explain_row(run_id, body.row)
    return result


@router.post("/runs/{run_id}/explain/image", operation_id="explainImage")
async def explain_image(
    run_id: str, ctx: Ctx, file: Annotated[UploadFile, File()]
) -> LocalExplanation:
    """Integrated Gradients de una imagen: mapa de calor sobre la clase predicha."""
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(await read_limited(file, 64 * MB)))
    except UnidentifiedImageError as exc:
        raise ValidationError("el archivo no es una imagen válida") from exc
    result: LocalExplanation = Workflow(ctx).explain_image(run_id, image)
    return result


@router.get("/runs/{run_id}/robustness", operation_id="getRobustness")
def robustness(run_id: str, ctx: Ctx) -> RobustnessReport:
    """Métrica del test con entradas perturbadas a tres severidades (RF-EVL-05)."""
    result: RobustnessReport = Workflow(ctx).robustness(run_id)
    return result


@router.get("/runs/{run_id}/report/document", operation_id="getReportDocument")
def report_document(
    run_id: str,
    ctx: Ctx,
    format: Annotated[Literal["html", "pdf", "md"], Query()] = "html",
) -> Response:
    """Informe con marca Preteco (RF-EVL-06): HTML autocontenido, PDF o Markdown."""
    body, media = Workflow(ctx).report_document(run_id, format)
    ext = {"html": "html", "pdf": "pdf", "md": "md"}[format]
    headers = {"Content-Disposition": f'attachment; filename="informe-{run_id}.{ext}"'}
    return Response(content=body, media_type=media, headers=headers)
