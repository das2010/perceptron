"""Fuentes, datasets y profiling (SPEC §10; RF-ING-01, 06, 07, 08; RF-PRF-07)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.core.errors import ValidationError
from perceptron.data.profiling.card import ProfileCard
from perceptron.data.schema import SemanticType, TableSchema, infer_schema
from perceptron.data.sources.files import SourceKind, open_source, scan_table
from perceptron.data.splits import SPLIT_COLUMN, SplitRequest
from perceptron.domain.enums import DataSourceType
from perceptron.domain.models import DatasetVersion, DataSource
from perceptron.services.workflow import Workflow

router = APIRouter(tags=["data"])
Ctx = Annotated[EngineContext, Depends(get_context)]


class SourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    type: DataSourceType = DataSourceType.FILE
    path: str = Field(description="Archivo, carpeta o ZIP accesible por el Engine")


class SourcePreview(BaseModel):
    kind: SourceKind
    columns: list[str]
    rows: list[dict[str, Any]]
    schema_: TableSchema | None = Field(default=None, alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class IngestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str | None = None
    split: SplitRequest | None = None
    overrides: dict[str, SemanticType] | None = None


@router.post(
    "/projects/{project_id}/sources",
    status_code=status.HTTP_201_CREATED,
    operation_id="createSource",
)
def create_source(project_id: str, body: SourceCreate, ctx: Ctx) -> DataSource:
    ctx.projects.get(project_id)
    path = Path(body.path)
    if not path.exists():
        raise ValidationError(f"la ruta no existe: {path}", details={"path": body.path})
    src = DataSource(
        project_id=project_id,
        name=body.name or path.name,
        type=body.type,
        config={"path": str(path)},
    )
    return ctx.repo(DataSource).add(src)


@router.post("/sources/{source_id}/preview", operation_id="previewSource")
def preview_source(
    source_id: str, ctx: Ctx, limit: Annotated[int, Query(ge=1, le=200)] = 20
) -> SourcePreview:
    src = ctx.repo(DataSource).get(source_id)
    with open_source(Path(src.config["path"])) as detected:
        if detected.kind is SourceKind.TABLE:
            df = scan_table(detected.path).head(max(limit, 500)).collect()
            return SourcePreview(
                kind=detected.kind,
                columns=df.columns,
                rows=df.head(limit).to_dicts(),
                schema=infer_schema(df),
            )
        return SourcePreview(kind=detected.kind, columns=["path", "label"], rows=[])


@router.post(
    "/sources/{source_id}/ingest", status_code=status.HTTP_201_CREATED, operation_id="ingestSource"
)
def ingest_source(source_id: str, body: IngestBody, ctx: Ctx) -> DatasetVersion:
    src = ctx.repo(DataSource).get(source_id)
    return Workflow(ctx).ingest(
        src.project_id,
        Path(src.config["path"]),
        target=body.target,
        split=body.split,
        overrides=body.overrides,
        source_record=src,
    )


@router.get("/projects/{project_id}/datasets", operation_id="listDatasets")
def list_datasets(project_id: str, ctx: Ctx) -> list[DatasetVersion]:
    return list(ctx.repo(DatasetVersion).list(filters={"project_id": project_id}))


@router.get("/datasets/{dataset_version_id}", operation_id="getDataset")
def get_dataset(dataset_version_id: str, ctx: Ctx) -> DatasetVersion:
    return ctx.repo(DatasetVersion).get(dataset_version_id)


@router.get("/datasets/{dataset_version_id}/samples", operation_id="getDatasetSamples")
def dataset_samples(
    dataset_version_id: str,
    ctx: Ctx,
    split: str = "train",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict[str, Any]]:
    """Muestras para la UI (el test sellado no se expone)."""
    wf = Workflow(ctx)
    view = wf.view(wf.dataset(dataset_version_id))
    df = view.scan(split).slice(offset, limit).collect()
    return df.drop(SPLIT_COLUMN).to_dicts()


@router.post("/datasets/{dataset_version_id}/profile", operation_id="profileDataset")
def profile(dataset_version_id: str, ctx: Ctx) -> ProfileCard:
    return Workflow(ctx).profile(dataset_version_id)


@router.get("/datasets/{dataset_version_id}/profile", operation_id="getProfile")
def get_profile(dataset_version_id: str, ctx: Ctx) -> ProfileCard:
    return Workflow(ctx).profile_card(dataset_version_id)
