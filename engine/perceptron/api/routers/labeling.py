"""Etiquetado asistido (SPEC §7.5, RF-LBL-01..06; ADR-0029)."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.security import MB, read_limited
from perceptron.core.errors import ValidationError
from perceptron.domain.enums import LabelKind
from perceptron.domain.models import DatasetVersion, LabelSet
from perceptron.llm.schemas import LabelingGuide
from perceptron.services.labeling import (
    Labeling,
    LabelingSummary,
    LabelQuality,
    LabelUpdate,
    QueueStrategy,
    Sample,
)
from perceptron.services.workflow import Workflow

router = APIRouter(tags=["labeling"])
Ctx = Annotated[EngineContext, Depends(get_context)]
LabelFormat = Literal["csv", "jsonl", "coco", "yolo", "voc"]


def _labeling(ctx: EngineContext) -> Labeling:
    return Labeling(Workflow(ctx))


class LabelSetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: LabelKind = LabelKind.CLASS
    classes: list[str] = Field(default_factory=list)
    name: str | None = None
    target: str | None = Field(default=None, description="Columna objetivo (tabular/texto)")


@router.post(
    "/datasets/{dataset_version_id}/labelsets",
    status_code=status.HTTP_201_CREATED,
    operation_id="createLabelSet",
)
def create_labelset(dataset_version_id: str, body: LabelSetCreate, ctx: Ctx) -> LabelSet:
    return _labeling(ctx).create(
        dataset_version_id, kind=body.kind, classes=body.classes, name=body.name, target=body.target
    )


@router.get("/datasets/{dataset_version_id}/labelsets", operation_id="listLabelSets")
def list_labelsets(dataset_version_id: str, ctx: Ctx) -> list[LabelSet]:
    return _labeling(ctx).list_sets(dataset_version_id)


@router.get("/labelsets/{labelset_id}", operation_id="getLabelSet")
def get_labelset(labelset_id: str, ctx: Ctx) -> LabelingSummary:
    return _labeling(ctx).summary(labelset_id)


@router.get("/labelsets/{labelset_id}/queue", operation_id="labelQueue")
def label_queue(
    labelset_id: str,
    ctx: Ctx,
    strategy: Annotated[QueueStrategy, Query()] = "uncertainty",
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[Sample]:
    """Próximas muestras a revisar (active learning, RF-LBL-03)."""
    return _labeling(ctx).queue(labelset_id, strategy=strategy, limit=limit)


@router.get("/labelsets/{labelset_id}/samples/{sample_id}/file", operation_id="labelSampleFile")
def sample_file(labelset_id: str, sample_id: str, ctx: Ctx) -> FileResponse:
    return FileResponse(_labeling(ctx).sample_file(labelset_id, sample_id))


class LabelsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    updates: list[LabelUpdate] = Field(min_length=1, max_length=5000)


class Count(BaseModel):
    count: int


@router.put("/labelsets/{labelset_id}/labels", operation_id="setLabels")
def set_labels(labelset_id: str, body: LabelsBody, ctx: Ctx) -> Count:
    return Count(count=_labeling(ctx).set_labels(labelset_id, body.updates))


class AcceptBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_confidence: float = Field(default=0.9, ge=0, le=1)


@router.post("/labelsets/{labelset_id}/accept", operation_id="acceptSuggestions")
def accept_suggestions(labelset_id: str, body: AcceptBody, ctx: Ctx) -> Count:
    """Acepta en lote las sugerencias con confianza ≥ umbral."""
    return Count(
        count=_labeling(ctx).accept_suggestions(labelset_id, min_confidence=body.min_confidence)
    )


class ClassesBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classes: list[str] = Field(min_length=1)


@router.post("/labelsets/{labelset_id}/classes", operation_id="addLabelClasses")
def add_classes(labelset_id: str, body: ClassesBody, ctx: Ctx) -> LabelSet:
    return _labeling(ctx).add_classes(labelset_id, body.classes)


class PrelabelBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["model", "llm"]
    run_id: str | None = Field(default=None, description="Modelo del proyecto (method=model)")
    guide: LabelingGuide | None = Field(default=None, description="Guía (method=llm)")
    limit: int = Field(default=200, ge=1, le=2000)


@router.post("/labelsets/{labelset_id}/prelabel", operation_id="prelabelSet")
def prelabel(labelset_id: str, body: PrelabelBody, ctx: Ctx) -> Count:
    """Sugerencias del modelo del proyecto o del LLM para lo que falta (RF-LBL-02)."""
    lab = _labeling(ctx)
    if body.method == "model":
        if not body.run_id:
            raise ValidationError("indicá el run del modelo (run_id)")
        return Count(count=lab.prelabel_with_model(labelset_id, body.run_id))
    if body.guide is None:
        raise ValidationError("el pre-etiquetado con LLM necesita la guía de etiquetado")
    return Count(count=lab.prelabel_with_llm(labelset_id, body.guide, limit=body.limit))


@router.get("/labelsets/{labelset_id}/quality", operation_id="labelQuality")
def quality(labelset_id: str, ctx: Ctx) -> LabelQuality:
    return _labeling(ctx).quality(labelset_id)


@router.get("/labelsets/{labelset_id}/export", operation_id="exportLabels")
def export_labels(
    labelset_id: str, ctx: Ctx, format: Annotated[LabelFormat, Query()] = "csv"
) -> Response:
    body, media = _labeling(ctx).export(labelset_id, format)
    ext = {"csv": "csv", "jsonl": "jsonl", "coco": "json", "yolo": "zip", "voc": "zip"}[format]
    headers = {"Content-Disposition": f'attachment; filename="{labelset_id}-{format}.{ext}"'}
    return Response(content=body, media_type=media, headers=headers)


@router.post("/labelsets/{labelset_id}/import", operation_id="importLabels")
async def import_labels(
    labelset_id: str,
    ctx: Ctx,
    file: Annotated[UploadFile, File()],
    format: Annotated[LabelFormat, Query()] = "csv",
) -> Count:
    data = await read_limited(file, 512 * MB)
    return Count(count=_labeling(ctx).import_labels(labelset_id, data, format))


@router.post(
    "/labelsets/{labelset_id}/apply",
    status_code=status.HTTP_201_CREATED,
    operation_id="applyLabels",
)
def apply_labels(labelset_id: str, ctx: Ctx) -> DatasetVersion:
    """Crea una versión nueva del dataset con las etiquetas aceptadas (para reentrenar)."""
    return _labeling(ctx).apply(labelset_id)
