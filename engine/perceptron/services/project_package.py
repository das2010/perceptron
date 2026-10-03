"""Paquete `.perceptron` (RF-PRJ-03): exportar e importar un proyecto completo.

Un zip con `manifest.json` y la carpeta del proyecto:
- el manifiesto trae el proyecto y todas sus entidades (también perfiles y evaluaciones,
  que cuelgan de datasets y runs), la versión de Perceptron y la ruta original del proyecto;
- los datos (versiones de datasets, subidas, fuentes materializadas y buffers) son opcionales:
  sin ellos el paquete es chico y sirve para compartir configuración, runs y modelos.

Al importar se conservan los ids (ULID globales): si el proyecto o alguna entidad ya existe se
rechaza. Las rutas absolutas que apuntaban a la carpeta original se reescriben a la nueva.
El tracking de MLflow no viaja (los runs y métricas sí, en las entidades).

El paquete no es de confianza (en el Team Server lo sube un usuario):
- usuarios, workspaces y membresías nunca se importan (los permisos los da el servidor);
- cada entidad tiene que ser del proyecto importado: su `project_id` es el del proyecto o su
  padre (perfil → versión de datos, evaluación → run, …) viene en el mismo paquete;
- ninguna referencia (`*_id`) puede apuntar a una entidad de otro proyecto del servidor;
- todo entra en una sola transacción (o el proyecto entero, o nada).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from perceptron import __version__
from perceptron.core.errors import ConflictError, ValidationError
from perceptron.core.paths import ensure_within, safe_parts
from perceptron.data.sources.files import check_zip_limits
from perceptron.domain.models import (
    ALL_ENTITIES,
    IDENTITY_ENTITIES,
    PROJECT_CHILDREN,
    Entity,
    Project,
)
from perceptron.storage.db import EntityRow
from perceptron.storage.repositories import add_entities

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

FORMAT = "perceptron-project"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
FILES_PREFIX = "project/"
DATA_DIRS = ("datasets", "uploads", "sources", "streams")
SUFFIX = ".perceptron"


def _entities(ctx: EngineContext, project_id: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with ctx.db.session() as s:
        rows = list(s.scalars(select(EntityRow).where(EntityRow.project_id == project_id)))
        identity = {m.__name__ for m in IDENTITY_ENTITIES}
        for row in rows:
            if row.kind not in identity:  # las membresías del proyecto no viajan
                out.setdefault(row.kind, []).append(row.data)
        for child, field, parent in PROJECT_CHILDREN:
            parents = [d["id"] for d in out.get(parent.__name__, [])]
            if parents:
                extra = s.scalars(
                    select(EntityRow).where(
                        EntityRow.kind == child.__name__,
                        EntityRow.data[field].as_string().in_(parents),
                    )
                )
                out.setdefault(child.__name__, []).extend(r.data for r in extra)
    return out


def export_project(ctx: EngineContext, project_id: str, out: Path, *, include_data: bool) -> Path:
    project = ctx.projects.get(project_id)
    root = ctx.settings.paths.project(project.id).root
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "perceptron_version": __version__,
        "include_data": include_data,
        # Tal como quedó en las entidades y resuelta (symlinks, nombres cortos de Windows).
        "original_roots": sorted({str(root), str(root.resolve())}),
        "project": project.model_dump(mode="json"),
        "entities": _entities(ctx, project.id),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = path.relative_to(root)
                if not include_data and rel.parts[0] in DATA_DIRS:
                    continue
                z.write(path, FILES_PREFIX + rel.as_posix())
    return out


def _rewrite(value: Any, old: str, new: Path) -> Any:
    r"""Rutas absolutas dentro de la carpeta original → la carpeta nueva (con `\` o `/`)."""
    if isinstance(value, str):
        norm, base = value.replace("\\", "/"), old.replace("\\", "/").rstrip("/")
        if base and (norm == base or norm.startswith(base + "/")):
            rest = norm[len(base) :].strip("/")
            return str(new / Path(*rest.split("/"))) if rest else str(new)
        return value
    if isinstance(value, dict):
        return {k: _rewrite(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite(v, old, new) for v in value]
    return value


def read_manifest(z: zipfile.ZipFile) -> dict[str, Any]:
    try:
        manifest: dict[str, Any] = json.loads(z.read(MANIFEST))
    except KeyError:
        raise ValidationError(
            "el archivo no es un paquete .perceptron (falta el manifiesto)"
        ) from None
    if manifest.get("format") != FORMAT:
        raise ValidationError("el archivo no es un paquete .perceptron")
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise ValidationError(
            "el paquete es de una versión más nueva de Perceptron: actualizá la app",
            details={"perceptron_version": manifest.get("perceptron_version")},
        )
    return manifest


# Campos `*_id` que no son entidades de Perceptron (otros sistemas o datos libres).
_EXTERNAL_IDS = {"mlflow_run_id", "client_id", "group_id"}


def _check_ownership(ctx: EngineContext, project: Project, entities: list[Entity]) -> None:
    """Cada entidad tiene que ser del proyecto importado y no referenciar otros proyectos."""
    ids = {e.id for e in entities} | {project.id}
    by_kind: dict[str, set[str]] = {}
    for e in entities:
        by_kind.setdefault(type(e).__name__, set()).add(e.id)
    children = {
        child.__name__: (field, parent.__name__) for child, field, parent in PROJECT_CHILDREN
    }
    seen: set[tuple[str, str]] = set()
    for e in entities:
        kind = type(e).__name__
        if (kind, e.id) in seen:
            raise ValidationError(f"{kind} {e.id} aparece dos veces en el paquete")
        seen.add((kind, e.id))
        if "project_id" in type(e).model_fields:
            if getattr(e, "project_id", None) != project.id:
                raise ValidationError(
                    f"{kind} {e.id} es de otro proyecto", details={"kind": kind, "id": e.id}
                )
        elif kind in children:
            field, parent = children[kind]
            if getattr(e, field, None) not in by_kind.get(parent, set()):
                raise ValidationError(
                    f"{kind} {e.id} cuelga de un {parent} que no está en el paquete",
                    details={"kind": kind, "id": e.id},
                )
        else:
            raise ValidationError(
                f"{kind} no se puede asociar al proyecto importado", details={"kind": kind}
            )
    # Referencias a entidades que no vienen en el paquete: se aceptan si no existen (quedaron
    # colgadas en el origen), nunca si existen en otro proyecto de este servidor.
    refs: set[str] = set()
    for e in entities:
        for name, value in e.model_dump().items():
            if (
                name.endswith("_id")
                and name not in _EXTERNAL_IDS
                and isinstance(value, str)
                and value not in ids
            ):
                refs.add(value)
    if refs:
        with ctx.db.session() as s:
            foreign = list(
                s.execute(select(EntityRow.kind, EntityRow.id).where(EntityRow.id.in_(refs)))
            )
        if foreign:
            kind, eid = foreign[0]
            raise ValidationError(
                "el paquete referencia entidades de otro proyecto",
                details={"kind": kind, "id": eid},
            )
    with ctx.db.session() as s:
        existing = list(s.scalars(select(EntityRow.id).where(EntityRow.id.in_(ids))))
    if existing:
        raise ConflictError(
            "el paquete trae entidades que ya existen en este servidor",
            details={"ids": sorted(existing)[:5]},
        )


def import_project(
    ctx: EngineContext, package: Path, prepare: Callable[[Project], Project] | None = None
) -> Project:
    """`prepare`: en el Team Server asigna workspace y alcance según quien importa."""
    kinds = {m.__name__: m for m in ALL_ENTITIES if m not in IDENTITY_ENTITIES}
    with zipfile.ZipFile(package) as z:
        check_zip_limits(z)
        manifest = read_manifest(z)
        project = Project.model_validate(manifest["project"])
        if prepare is not None:
            project = prepare(project)
        if ctx.projects.find(project.id) is not None:
            raise ConflictError(
                "el proyecto ya existe en este workspace",
                details={"project_id": project.id, "name": project.name},
            )
        root = ctx.settings.paths.project(project.id).root
        entities: list[Any] = []
        olds = sorted(map(str, manifest.get("original_roots", [])), key=len, reverse=True)
        new = root.resolve()
        for kind, items in dict(manifest.get("entities", {})).items():
            model = kinds.get(kind)
            if model is None or model is Project:
                continue  # identidad/permisos o entidad de una versión futura: no se importa
            for item in items:
                data = item
                for old in olds:
                    data = _rewrite(data, old, new)
                entities.append(model.model_validate(data))
        _check_ownership(ctx, project, entities)
        root.mkdir(parents=True, exist_ok=True)
        try:
            for info in z.infolist():
                if info.is_dir() or not info.filename.startswith(FILES_PREFIX):
                    continue
                rel = PurePosixPath(info.filename[len(FILES_PREFIX) :])
                dest = ensure_within(root / Path(*safe_parts(str(rel))), root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
            add_entities(ctx.db, [project, *entities])  # una transacción: todo o nada
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
    ctx.files.write_project(project)
    return project
