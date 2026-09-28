"""`/sync` — sincronización desktop ↔ servidor para proyectos de equipo (RF-SRV-03).

El servidor es la fuente de verdad y el desktop mantiene una copia de trabajo:
- **Entidades** (proyecto, versiones de datos, perfiles, pipelines, arquitecturas, estudios,
  runs) con los mismos IDs en ambos lados (ULID globales) y bloqueo optimista: el cliente
  manda la versión que conoce del servidor (`base_version`) y un desfasaje da 409.
- **Archivos** del proyecto (rutas relativas al proyecto, como en las entidades) con subida
  **resumible por chunks** y verificación SHA-256; si el archivo ya está igual, no se sube.

Las versiones de datos son inmutables: volver a subir una igual no cambia nada.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from perceptron.core.errors import ConflictError, ForbiddenError, NotFoundError, ValidationError
from perceptron.core.ids import IdPrefix
from perceptron.core.paths import ensure_within, safe_parts
from perceptron.domain.enums import Role
from perceptron.domain.models import (
    ArchSpecRecord,
    DatasetVersion,
    Entity,
    Evaluation,
    Pipeline,
    Profile,
    Project,
    Run,
    Study,
)
from perceptron_server.accounts import ROLE_RANK, Principal
from perceptron_server.routers.auth import Who
from perceptron_server.state import ServerState, client_ip, server_state

router = APIRouter(prefix="/sync", tags=["sync"])
State = Annotated[ServerState, Depends(server_state)]

SYNC_KINDS: dict[str, type[Entity]] = {
    m.__name__: m
    for m in (DatasetVersion, Profile, Pipeline, ArchSpecRecord, Study, Run, Evaluation)
}
# Lo que el desktop puede subir; bajar se puede también runs/ y exports/.
UPLOAD_PREFIXES = ("datasets", "pipelines", "archspecs", "code", "labels")
DOWNLOAD_PREFIXES = (*UPLOAD_PREFIXES, "runs", "exports")
MAX_CHUNK = 16 * 1024 * 1024
MAX_FILE = 10 * 1024**3


class ProjectPush(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: dict[str, Any]
    base_version: int | None = Field(default=None, description="Versión del servidor conocida")


class EntityPush(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any]
    base_version: int | None = None


class SyncResult(BaseModel):
    id: str
    version: int
    created: bool = False
    unchanged: bool = False


class UploadStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0, le=MAX_FILE)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class UploadState(BaseModel):
    upload_id: str | None
    path: str
    offset: int
    size: int
    complete: bool


class RemoteFile(BaseModel):
    path: str
    size: int


# ------------------------------------------------------------------ helpers


def _role(who: Principal, project: Project) -> Role | None:
    return who.project_role(project)


def _require(who: Principal, project: Project, need: Role) -> None:
    role = _role(who, project)
    if role is None or ROLE_RANK[role] < ROLE_RANK[need]:
        raise ForbiddenError(
            "tu rol en el proyecto no permite sincronizarlo", details={"project": project.id}
        )


def _project(state: ServerState, who: Principal, project_id: str, need: Role) -> Project:
    project = state.ctx.projects.find(project_id)
    if project is None:
        raise NotFoundError(f"el proyecto {project_id} no está en el servidor")
    _require(who, project, need)
    return project


def _safe_rel(path: str, prefixes: tuple[str, ...]) -> PurePosixPath:
    """Relativa al proyecto, bajo un prefijo permitido y segura también en Windows."""
    try:
        parts = safe_parts(path)
    except ValidationError:
        raise ValidationError(f"ruta no permitida para sincronizar: {path!r}") from None
    if parts[0] not in prefixes:
        raise ValidationError(f"ruta no permitida para sincronizar: {path!r}")
    return PurePosixPath(*parts)


def _target(state: ServerState, project: Project, rel: PurePosixPath) -> Path:
    """Archivo del proyecto; un symlink intermedio no puede sacarlo de la carpeta."""
    root = state.ctx.settings.paths.project(project.id).root
    return ensure_within(root / Path(*rel.parts), root)


_ID = re.compile(r"[a-z]{2,4}_[0-9A-Za-z]{1,64}")


def _check_id(value: str, prefix: str | None = None) -> None:
    """Forma de id de Perceptron (`prj_01…`); nunca separadores ni otra cosa que arme rutas."""
    if not _ID.fullmatch(value) or (prefix and not value.startswith(f"{prefix}_")):
        raise ValidationError(f"id inválido: {value!r}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _uploads_dir(state: ServerState) -> Path:
    path = state.settings.workspace_dir / ".uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _entity_project(state: ServerState, kind: str, entity: Entity) -> str | None:
    if isinstance(entity, Profile):
        dv = state.ctx.repo(DatasetVersion).find(entity.dataset_version_id)
        return dv.project_id if dv else None
    if isinstance(entity, Evaluation):
        run = state.ctx.repo(Run).find(entity.run_id)
        return run.project_id if run else None
    return getattr(entity, "project_id", None)


# ------------------------------------------------------------------ entidades


@router.put("/projects/{project_id}", operation_id="syncProject")
def push_project(
    project_id: str, body: ProjectPush, who: Who, request: Request, state: State
) -> SyncResult:
    """Crea el proyecto de equipo (mismo id que en el desktop) o lo actualiza."""
    _check_id(project_id, IdPrefix.PROJECT.value)
    try:
        incoming = Project.model_validate({**body.project, "id": project_id})
    except PydanticValidationError as exc:
        raise ValidationError("proyecto inválido", details={"errors": exc.errors()}) from None
    existing = state.ctx.projects.find(project_id)
    if existing is None:
        draft = state.access.prepare_project(request, incoming.model_copy(update={"version": 1}))
        project = state.ctx.projects.add(draft)
        state.ctx.files.init_project(project)
        state.audit.record(
            "sync.project_created",
            user_id=who.user.id,
            project_id=project.id,
            ip=client_ip(request),
        )
        return SyncResult(id=project.id, version=project.version, created=True)
    _require(who, existing, Role.EDITOR)
    if body.base_version != existing.version:
        raise ConflictError(
            "el proyecto cambió en el servidor",
            details={"server_version": existing.version, "base_version": body.base_version},
        )
    keep = {"workspace_id", "scope", "id", "created_at", "version"}
    fields = incoming.model_dump(exclude=keep)
    updated = state.ctx.projects.update(existing.model_copy(update=fields))
    state.ctx.files.write_project(updated)
    return SyncResult(id=updated.id, version=updated.version)


@router.put("/projects/{project_id}/entities/{kind}/{entity_id}", operation_id="syncEntity")
def push_entity(
    project_id: str,
    kind: str,
    entity_id: str,
    body: EntityPush,
    who: Who,
    request: Request,
    state: State,
) -> SyncResult:
    project = _project(state, who, project_id, Role.EDITOR)
    model = SYNC_KINDS.get(kind)
    if model is None:
        raise ValidationError(f"no se sincroniza {kind}", details={"kinds": sorted(SYNC_KINDS)})
    _check_id(entity_id)
    data = {**body.data, "id": entity_id}
    if "project_id" in model.model_fields:
        data["project_id"] = project.id
    try:
        entity = model.model_validate(data)
    except PydanticValidationError as exc:
        raise ValidationError(f"{kind} inválido", details={"errors": exc.errors()}) from None
    if _entity_project(state, kind, entity) != project.id:
        raise ForbiddenError(f"{kind} {entity_id} no pertenece al proyecto")
    repo = state.ctx.repo(model)
    current = repo.find(entity_id)
    if current is None:
        # Un id ajeno (de otro tipo o de otro proyecto) no se puede pisar.
        other = state.access.project_of(entity_id)
        if other is not None and other != project.id:
            raise ForbiddenError(f"{entity_id} pertenece a otro proyecto")
        created = repo.add(entity.model_copy(update={"version": 1}))
        state.audit.record(
            "sync.entity",
            user_id=who.user.id,
            project_id=project.id,
            resource=f"{kind}/{entity_id}",
            ip=client_ip(request),
        )
        return SyncResult(id=entity_id, version=created.version, created=True)
    if _entity_project(state, kind, current) != project.id:
        raise ForbiddenError(f"{entity_id} pertenece a otro proyecto")
    if isinstance(current, DatasetVersion) and current.content_hash == getattr(
        entity, "content_hash", None
    ):
        return SyncResult(id=entity_id, version=current.version, unchanged=True)  # inmutable
    if body.base_version != current.version:
        raise ConflictError(
            f"{kind} {entity_id} cambió en el servidor",
            details={"server_version": current.version, "base_version": body.base_version},
        )
    updated = repo.update(entity.model_copy(update={"version": current.version}))
    return SyncResult(id=entity_id, version=updated.version)


@router.get("/projects/{project_id}/entities/{kind}", operation_id="pullEntities")
def pull_entities(
    project_id: str,
    kind: str,
    who: Who,
    state: State,
    ids: Annotated[list[str] | None, Query()] = None,
    study_id: str | None = None,
) -> list[dict[str, Any]]:
    """Entidades del proyecto para bajar al desktop (p. ej. los runs de un estudio remoto)."""
    project = _project(state, who, project_id, Role.VIEWER)
    model = SYNC_KINDS.get(kind)
    if model is None:
        raise ValidationError(f"no se sincroniza {kind}")
    filters: dict[str, Any] = {}
    if "project_id" in model.model_fields:
        filters["project_id"] = project.id
    if study_id is not None and "study_id" in model.model_fields:
        filters["study_id"] = study_id
    items = state.ctx.repo(model).list(filters=filters, ids=ids, limit=1000)
    return [
        e.model_dump(mode="json") for e in items if _entity_project(state, kind, e) == project.id
    ]


# ------------------------------------------------------------------ archivos


@router.post("/projects/{project_id}/uploads", operation_id="startUpload")
def start_upload(project_id: str, body: UploadStart, who: Who, state: State) -> UploadState:
    """Empieza (o retoma) la subida de un archivo del proyecto."""
    project = _project(state, who, project_id, Role.EDITOR)
    rel = _safe_rel(body.path, UPLOAD_PREFIXES)
    target = _target(state, project, rel)
    if target.is_file() and target.stat().st_size == body.size and _sha256(target) == body.sha256:
        return UploadState(
            upload_id=None, path=str(rel), offset=body.size, size=body.size, complete=True
        )
    key = hashlib.sha256(f"{who.user.id}:{project.id}:{rel}:{body.sha256}".encode()).hexdigest()
    folder = _uploads_dir(state) / key[:32]
    meta = folder / "meta.json"
    if not meta.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "data.part").touch()
        info = {
            "upload_id": key[:32],
            "user_id": who.user.id,
            "project_id": project.id,
            "path": str(rel),
            "size": body.size,
            "sha256": body.sha256,
        }
        meta.write_text(json.dumps(info), encoding="utf-8")
    offset = (folder / "data.part").stat().st_size
    return UploadState(
        upload_id=key[:32], path=str(rel), offset=offset, size=body.size, complete=False
    )


def _upload(state: ServerState, who: Principal, upload_id: str) -> tuple[Path, dict[str, Any]]:
    if not upload_id.isalnum() or len(upload_id) != 32:
        raise NotFoundError("subida inexistente")
    folder = _uploads_dir(state) / upload_id
    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        raise NotFoundError("subida inexistente o vencida")
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["user_id"] != who.user.id:
        raise ForbiddenError("la subida es de otro usuario")
    return folder, meta


@router.put("/uploads/{upload_id}", operation_id="uploadChunk")
async def upload_chunk(
    upload_id: str,
    request: Request,
    who: Who,
    state: State,
    offset: Annotated[int, Query(ge=0)],
) -> UploadState:
    """Agrega un chunk en `offset` (debe coincidir con lo recibido: permite retomar)."""
    folder, meta = _upload(state, who, upload_id)
    part = folder / "data.part"
    have = part.stat().st_size
    if offset != have:
        raise ConflictError("offset desfasado", details={"offset": have})
    limit = min(MAX_CHUNK, int(meta["size"]) - have)
    chunk = bytearray()
    async for piece in request.stream():  # sin cargar en memoria más que el tope
        chunk += piece
        if len(chunk) > limit:
            raise ValidationError("chunk demasiado grande")
    with part.open("ab") as f:
        f.write(chunk)
    return UploadState(
        upload_id=upload_id,
        path=meta["path"],
        offset=have + len(chunk),
        size=int(meta["size"]),
        complete=False,
    )


@router.post("/uploads/{upload_id}/complete", operation_id="completeUpload")
def complete_upload(upload_id: str, who: Who, request: Request, state: State) -> UploadState:
    folder, meta = _upload(state, who, upload_id)
    project = _project(state, who, meta["project_id"], Role.EDITOR)
    part = folder / "data.part"
    if part.stat().st_size != int(meta["size"]) or _sha256(part) != meta["sha256"]:
        raise ValidationError("el archivo recibido no coincide (tamaño o SHA-256)")
    rel = _safe_rel(meta["path"], UPLOAD_PREFIXES)
    target = _target(state, project, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(part), target)
    shutil.rmtree(folder, ignore_errors=True)
    state.audit.record(
        "sync.upload",
        user_id=who.user.id,
        project_id=project.id,
        resource=str(rel),
        ip=client_ip(request),
        details={"size": meta["size"]},
    )
    return UploadState(
        upload_id=None,
        path=str(rel),
        offset=int(meta["size"]),
        size=int(meta["size"]),
        complete=True,
    )


@router.get("/projects/{project_id}/files", operation_id="listProjectFiles")
def list_files(
    project_id: str,
    who: Who,
    state: State,
    prefix: Annotated[str, Query(min_length=1, max_length=1024)],
) -> list[RemoteFile]:
    project = _project(state, who, project_id, Role.VIEWER)
    rel = _safe_rel(prefix, DOWNLOAD_PREFIXES)
    root = state.ctx.settings.paths.project(project.id).root.resolve()
    base = _target(state, project, rel)
    if base.is_file():
        return [RemoteFile(path=str(rel), size=base.stat().st_size)]
    if not base.is_dir():
        return []
    return [
        RemoteFile(path=p.relative_to(root).as_posix(), size=p.stat().st_size)
        for p in sorted(base.rglob("*"))
        if p.is_file() and not p.is_symlink()
    ]


@router.get("/projects/{project_id}/file", operation_id="downloadProjectFile")
def download_file(
    project_id: str,
    who: Who,
    request: Request,
    state: State,
    path: Annotated[str, Query(min_length=1, max_length=1024)],
) -> FileResponse:
    project = _project(state, who, project_id, Role.VIEWER)
    rel = _safe_rel(path, DOWNLOAD_PREFIXES)
    target = state.ctx.settings.paths.project(project.id).root / Path(*rel.parts)
    if not target.is_file() or target.is_symlink():
        raise NotFoundError(f"no existe {path}")
    target = _target(state, project, rel)
    if not target.is_file():
        raise NotFoundError(f"no existe {path}")
    state.audit.record(
        "sync.download",
        user_id=who.user.id,
        project_id=project.id,
        resource=str(rel),
        ip=client_ip(request),
    )
    return FileResponse(target, filename=target.name)
