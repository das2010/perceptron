"""Ciclo de vida de proyectos (RF-PRJ-01): duplicar y eliminar de verdad."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from perceptron.core.errors import ConflictError
from perceptron.core.paths import ensure_within
from perceptron.domain.models import DatasetVersion, Evaluation, Profile, Project, Run
from perceptron.storage.db import EntityRow

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

# Lo que se copia al duplicar: la configuración, no los datos ni los resultados.
DUPLICATED_FIELDS = (
    "description",
    "goal",
    "modalities",
    "task",
    "target_metric",
    "privacy_level",
    "llm_profile_id",
    "template",
    "workspace_id",
)


def duplicate_project(ctx: EngineContext, project_id: str, name: str | None = None) -> Project:
    """Proyecto nuevo con la misma configuración (objetivo, tarea, métrica, privacidad,
    plantilla), en borrador y sin datos ni runs."""
    source = ctx.projects.get(project_id)
    fields = source.model_dump(include=set(DUPLICATED_FIELDS))
    return Project(**fields, name=(name or f"{source.name} (copia)")[:200])


def _running_jobs(ctx: EngineContext, project_id: str) -> bool:
    return any(
        job.status in ("queued", "running") and job.refs.get("project_id") == project_id
        for job in ctx.jobs.list()
    )


def purge_project(ctx: EngineContext, project_id: str) -> dict[str, int]:
    """Borra el proyecto, todas sus entidades (también perfiles y evaluaciones, que se
    vinculan por dataset y por run) y su carpeta. Devuelve cuántas entidades borró."""
    project = ctx.projects.get(project_id)
    if _running_jobs(ctx, project.id):
        raise ConflictError("hay tareas en curso en el proyecto: cancelalas antes de eliminarlo")

    def ids_of(model: type[DatasetVersion] | type[Run]) -> list[str]:
        with ctx.db.session() as s:
            return list(
                s.scalars(
                    select(EntityRow.id).where(
                        EntityRow.kind == model.__name__, EntityRow.project_id == project.id
                    )
                )
            )

    # Perfiles y evaluaciones no tienen project_id: cuelgan de un dataset o de un run.
    linked = [
        (Profile.__name__, "dataset_version_id", ids_of(DatasetVersion)),
        (Evaluation.__name__, "run_id", ids_of(Run)),
    ]
    removed = 0
    with ctx.db.session() as s:
        for kind, field, parents in linked:
            if parents:
                result = s.execute(
                    delete(EntityRow).where(
                        EntityRow.kind == kind, EntityRow.data[field].as_string().in_(parents)
                    )
                )
                removed += int(getattr(result, "rowcount", 0) or 0)
        result = s.execute(delete(EntityRow).where(EntityRow.project_id == project.id))
        removed += int(getattr(result, "rowcount", 0) or 0)
    ctx.projects.delete(project.id)
    root = ctx.settings.paths.project(project.id).root
    if root.exists():
        shutil.rmtree(ensure_within(root, ctx.settings.paths.projects_dir), ignore_errors=True)
    return {"entities": removed}
