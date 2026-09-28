"""Paquete `.perceptron` (RF-PRJ-03): exportar e importar un proyecto completo.

Un zip con `manifest.json` y la carpeta del proyecto:
- el manifiesto trae el proyecto y todas sus entidades (también perfiles y evaluaciones,
  que cuelgan de datasets y runs), la versión de Perceptron y la ruta original del proyecto;
- los datos (versiones de datasets, subidas, fuentes materializadas y buffers) son opcionales:
  sin ellos el paquete es chico y sirve para compartir configuración, runs y modelos.

Al importar se conservan los ids (ULID globales): si el proyecto ya existe en este workspace
se rechaza. Las rutas absolutas que apuntaban a la carpeta original se reescriben a la nueva.
El tracking de MLflow no viaja (los runs y métricas sí, en las entidades).
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
from perceptron.domain.models import ALL_ENTITIES, DatasetVersion, Evaluation, Profile, Project, Run
from perceptron.storage.db import EntityRow

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
        for row in rows:
            out.setdefault(row.kind, []).append(row.data)
        linked = [
            (Profile.__name__, "dataset_version_id", DatasetVersion.__name__),
            (Evaluation.__name__, "run_id", Run.__name__),
        ]
        for kind, field, parent in linked:
            parents = [d["id"] for d in out.get(parent, [])]
            if parents:
                extra = s.scalars(
                    select(EntityRow).where(
                        EntityRow.kind == kind, EntityRow.data[field].as_string().in_(parents)
                    )
                )
                out.setdefault(kind, []).extend(r.data for r in extra)
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


def import_project(
    ctx: EngineContext, package: Path, prepare: Callable[[Project], Project] | None = None
) -> Project:
    """`prepare`: en el Team Server asigna workspace y alcance según quien importa."""
    kinds = {m.__name__: m for m in ALL_ENTITIES}
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
                continue  # entidad de una versión futura: se ignora
            for item in items:
                data = item
                for old in olds:
                    data = _rewrite(data, old, new)
                entities.append(model.model_validate(data))
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
            ctx.projects.add(project)
            for entity in entities:
                ctx.repo(type(entity)).add(entity)
        except BaseException:
            # Sin importaciones a medias: se deshace lo que haya entrado.
            if ctx.projects.find(project.id) is not None:
                from perceptron.services.projects import purge_project

                purge_project(ctx, project.id)
            shutil.rmtree(root, ignore_errors=True)
            raise
    ctx.files.write_project(project)
    return project
