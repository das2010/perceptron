"""Fuentes, datasets y profiling (SPEC §10; RF-ING-01, 06, 07, 08; RF-PRF-07)."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from perceptron.api.context import EngineContext, get_context
from perceptron.core.errors import ForbiddenError, ValidationError
from perceptron.core.ids import IdPrefix, new_id
from perceptron.core.netguard import check_host
from perceptron.core.paths import ensure_within, safe_parts, within_roots
from perceptron.data.profiling.card import ProfileCard
from perceptron.data.schema import SemanticType, TableSchema, infer_schema
from perceptron.data.sources.files import SourceKind, folder_preview, open_source, scan_table
from perceptron.data.sources.remote import DbConfig, download_hf, download_kaggle, materialize_db
from perceptron.data.splits import SPLIT_COLUMN, SplitRequest
from perceptron.data.versioning.diff import DatasetDiff, LineageNode, dataset_diff, lineage
from perceptron.data.versioning.retention import RetentionReport, apply_retention, export_dvc
from perceptron.domain.enums import DataSourceType, Modality
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
    total: int | None = Field(default=None, description="Carpetas: cantidad de archivos")
    classes: dict[str, int] | None = Field(
        default=None, description="Carpetas: archivos por clase (primer nivel de carpeta)"
    )

    model_config = ConfigDict(populate_by_name=True)


class IngestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str | None = None
    split: SplitRequest | None = None
    overrides: dict[str, SemanticType] | None = None
    modality: Modality | None = None


@router.post(
    "/projects/{project_id}/sources",
    status_code=status.HTTP_201_CREATED,
    operation_id="createSource",
)
def create_source(project_id: str, body: SourceCreate, ctx: Ctx) -> DataSource:
    ctx.projects.get(project_id)
    path = Path(body.path)
    _check_roots(ctx, path, project_id)
    if not path.exists():
        raise ValidationError(f"la ruta no existe: {path}", details={"path": body.path})
    src = DataSource(
        project_id=project_id,
        name=body.name or path.name,
        type=body.type,
        config={"path": str(path)},
    )
    return ctx.repo(DataSource).add(src)


def _check_roots(ctx: EngineContext, path: Path, project_id: str | None = None) -> None:
    """En el Team Server solo se leen rutas debajo de las «fuentes del servidor» (RF-SRV-05) o
    de la carpeta del propio proyecto (subidas y fuentes materializadas)."""
    if project_id is not None and within_roots(path, [ctx.settings.paths.project(project_id).root]):
        return
    if not within_roots(path, ctx.settings.source_roots):
        raise ForbiddenError(
            "la ruta no está dentro de una fuente habilitada por el administrador",
            details={"path": str(path)},
        )


def _source_path(ctx: EngineContext, src: DataSource) -> Path:
    """Ruta de la fuente, verificada otra vez: las raíces pudieron cambiar desde que se creó."""
    path = Path(src.config["path"])
    _check_roots(ctx, path, src.project_id)
    return path


MAX_UPLOAD_BYTES = 10 * 1024**3  # ~10 GB por proyecto (SPEC §2)


def _safe_relative(name: str) -> Path:
    """Ruta relativa segura: `..` y `/` iniciales se descartan; letras de unidad, NUL y nombres
    reservados de Windows se rechazan (`safe_parts`)."""
    return Path(*safe_parts(name, drop_parent=True))


@router.post(
    "/projects/{project_id}/uploads",
    status_code=status.HTTP_201_CREATED,
    operation_id="uploadSource",
)
async def upload_source(
    project_id: str, ctx: Ctx, files: Annotated[list[UploadFile], File()]
) -> DataSource:
    """Sube un archivo (CSV, Parquet, ZIP…) o una carpeta (nombres con ruta relativa, p. ej.
    `clase/img.png`) al proyecto y la registra como fuente (RF-ING-01, UI web)."""
    ctx.projects.get(project_id)
    if not files:
        raise ValidationError("no se recibió ningún archivo")
    upload_id = new_id(IdPrefix.DATASOURCE).split("_", 1)[1].lower()
    root = ctx.settings.paths.project(project_id).root / "uploads" / upload_id
    total = 0
    written: list[Path] = []
    try:
        for f in files:
            rel = _safe_relative(f.filename or "archivo")
            dest = ensure_within(root / rel, root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as out:
                while chunk := await f.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValidationError("la subida supera el límite de 10 GB")
                    out.write(chunk)
            written.append(rel)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)  # sin subidas a medias en el proyecto
        raise
    single = len(written) == 1 and len(written[0].parts) == 1
    path = root / written[0] if single else root
    if (
        not single
        and len({r.parts[0] for r in written}) == 1
        and all(len(r.parts) > 1 for r in written)
    ):
        # Carpeta raíz elegida en el navegador (webkitdirectory antepone su nombre); puede
        # traer archivos sueltos en la raíz además de las subcarpetas (p. ej. anotaciones COCO).
        path = root / written[0].parts[0]
    src = DataSource(
        project_id=project_id,
        name=written[0].name if single else path.name,
        type=DataSourceType.FILE,
        config={"path": str(path), "uploaded": True, "files": len(written)},
    )
    return ctx.repo(DataSource).add(src)


# ------------------------------------------------------------------ fuentes remotas (RF-ING-03/04)


class DbSourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    config: DbConfig
    password: str | None = Field(default=None, description="Solo de escritura: va al keychain")


class HubSourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["huggingface", "kaggle"]
    dataset: str = Field(min_length=3, description="repo_id (HF) u owner/slug (Kaggle)")
    split: str | None = Field(default=None, description="Split de HF (train, test…)")
    token: str | None = Field(
        default=None, description="Solo de escritura: token de HF o «usuario:clave» de Kaggle"
    )
    name: str | None = None


def _source_dir(ctx: EngineContext, src: DataSource) -> Path:
    path: Path = ctx.settings.paths.project(src.project_id).root / "sources" / src.id
    return path


def _secret_name(src: DataSource) -> str:
    return f"source/{src.id}/credentials"


def _materialize(ctx: EngineContext, src: DataSource, secret: str | None) -> dict[str, Any]:
    folder = _source_dir(ctx, src)
    folder.mkdir(parents=True, exist_ok=True)
    if src.type is DataSourceType.DB:
        cfg = DbConfig.model_validate(src.config["db"])
        if cfg.dialect == "sqlite":
            _check_roots(ctx, Path(cfg.database))
        else:
            default_ports = {"postgresql": 5432, "mysql": 3306, "mssql": 1433}
            check_host(
                cfg.host or "localhost",
                cfg.port or default_ports[cfg.dialect],
                ctx.settings.net_policy(),
            )
        dest = folder / "data.parquet"
        rows = materialize_db(cfg, secret, dest)
        return {"path": str(dest), "rows": rows}
    if src.type is DataSourceType.HF:
        path = download_hf(src.config["dataset"], src.config.get("split"), secret, folder)
        return {"path": str(path)}
    path = download_kaggle(src.config["dataset"], secret, folder)
    return {"path": str(path)}


@router.post(
    "/projects/{project_id}/sources/db",
    status_code=status.HTTP_201_CREATED,
    operation_id="createDbSource",
)
def create_db_source(project_id: str, body: DbSourceCreate, ctx: Ctx) -> DataSource:
    """Consulta SQL (SQL Server, PostgreSQL, MySQL/MariaDB, SQLite) materializada en Parquet."""
    ctx.projects.get(project_id)
    src = DataSource(
        project_id=project_id,
        name=body.name,
        type=DataSourceType.DB,
        config={"db": body.config.model_dump(mode="json")},
    )
    if body.password:
        ctx.llm.secrets.set(_secret_name(src), body.password)
        src.secret_refs = {"password": _secret_name(src)}
    src.config.update(_materialize(ctx, src, body.password))
    return ctx.repo(DataSource).add(src)


@router.post(
    "/projects/{project_id}/sources/hub",
    status_code=status.HTTP_201_CREATED,
    operation_id="createHubSource",
)
def create_hub_source(project_id: str, body: HubSourceCreate, ctx: Ctx) -> DataSource:
    """Dataset público de Hugging Face o Kaggle, descargado a la caché del proyecto."""
    ctx.projects.get(project_id)
    kind = DataSourceType.HF if body.provider == "huggingface" else DataSourceType.KAGGLE
    src = DataSource(
        project_id=project_id,
        name=body.name or body.dataset,
        type=kind,
        config={"dataset": body.dataset, "split": body.split},
    )
    if body.token:
        ctx.llm.secrets.set(_secret_name(src), body.token)
        src.secret_refs = {"token": _secret_name(src)}
    src.config.update(_materialize(ctx, src, body.token))
    return ctx.repo(DataSource).add(src)


@router.post("/sources/{source_id}/refresh", operation_id="refreshSource")
def refresh_source(source_id: str, ctx: Ctx) -> DataSource:
    """Vuelve a ejecutar la consulta o la descarga; la próxima ingesta crea otra versión."""
    src = ctx.repo(DataSource).get(source_id)
    if src.type not in (DataSourceType.DB, DataSourceType.HF, DataSourceType.KAGGLE):
        raise ValidationError("solo las fuentes remotas se refrescan")
    secret = ctx.llm.secrets.get(_secret_name(src)) if src.secret_refs else None
    config = {**src.config, **_materialize(ctx, src, secret)}
    updated: DataSource = ctx.repo(DataSource).update(src.model_copy(update={"config": config}))
    return updated


@router.post("/sources/{source_id}/preview", operation_id="previewSource")
def preview_source(
    source_id: str, ctx: Ctx, limit: Annotated[int, Query(ge=1, le=200)] = 20
) -> SourcePreview:
    src = ctx.repo(DataSource).get(source_id)
    path = _source_path(ctx, src)
    # Carpetas y zips de imágenes/audio: se previsualiza con la lista de archivos (un zip de
    # cientos de MB no se extrae para esto).
    fast = None
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            fast = folder_preview(zf.namelist(), limit, zipped=True)
    elif path.is_dir():
        fast = folder_preview(
            (p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()), limit
        )
    if fast is not None:
        return SourcePreview(
            kind=fast.kind,
            columns=["path", "label"],
            rows=list(fast.samples),
            total=fast.total,
            classes=fast.classes,
        )
    with open_source(path) as detected:
        if detected.kind is SourceKind.TABLE:
            df = scan_table(detected.path).head(max(limit, 500)).collect()
            return SourcePreview(
                kind=detected.kind,
                columns=df.columns,
                rows=df.head(limit).to_dicts(),
                schema_=infer_schema(df),
            )
        return SourcePreview(kind=detected.kind, columns=["path", "label"], rows=[])


@router.post(
    "/sources/{source_id}/ingest", status_code=status.HTTP_201_CREATED, operation_id="ingestSource"
)
def ingest_source(source_id: str, body: IngestBody, ctx: Ctx) -> DatasetVersion:
    src = ctx.repo(DataSource).get(source_id)
    return Workflow(ctx).ingest(
        src.project_id,
        _source_path(ctx, src),
        target=body.target,
        split=body.split,
        overrides=body.overrides,
        source_record=src,
        modality=body.modality,
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


# ------------------------------------------------------------------ diff y linaje (RF-MON-07)


@router.get("/datasets/{dataset_version_id}/diff/{other_id}", operation_id="diffDatasets")
def diff_datasets(dataset_version_id: str, other_id: str, ctx: Ctx) -> DatasetDiff:
    """Qué cambió de la versión A a la B: filas, esquema, distribución y archivos."""
    wf = Workflow(ctx)
    a, b = wf.dataset(dataset_version_id), wf.dataset(other_id)
    if a.project_id != b.project_id:
        raise ValidationError("las dos versiones tienen que ser del mismo proyecto")
    return dataset_diff(a, wf.view(a), b, wf.view(b))


@router.get("/datasets/{dataset_version_id}/lineage", operation_id="getLineage")
def dataset_lineage(dataset_version_id: str, ctx: Ctx) -> list[LineageNode]:
    dv = ctx.repo(DatasetVersion).get(dataset_version_id)
    versions = {
        v.id: v
        for v in ctx.repo(DatasetVersion).list(filters={"project_id": dv.project_id}, limit=10_000)
    }
    return lineage(versions, dv.id)


# ------------------------------------------------------------------ retención y DVC (RF-MON-07)


class RetentionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep_last: int = Field(ge=1, le=10_000)
    dry_run: bool = True


@router.post("/projects/{project_id}/datasets/retention", operation_id="applyDatasetRetention")
def dataset_retention(project_id: str, body: RetentionBody, ctx: Ctx) -> RetentionReport:
    """Conserva las últimas N versiones (y las que están en uso); con `dry_run` solo informa."""
    return apply_retention(ctx, project_id, keep_last=body.keep_last, dry_run=body.dry_run)


@router.get("/datasets/{dataset_version_id}/dvc.zip", operation_id="downloadDatasetDvc")
def download_dvc(dataset_version_id: str, ctx: Ctx) -> FileResponse:
    """Versión de datos con su `.dvc` (DVC 3, md5) para agregarla a un repositorio DVC."""
    dv = ctx.repo(DatasetVersion).get(dataset_version_id)
    tmp = Path(tempfile.mkdtemp(prefix="perceptron-dvc-"))
    # El nombre sale del hash guardado, no del id del pedido.
    out = export_dvc(ctx, dv.id, tmp / "dataset-dvc.zip")
    return FileResponse(
        out,
        filename=f"dataset-{dv.content_hash[:12]}-dvc.zip",
        media_type="application/zip",
        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True),
    )
