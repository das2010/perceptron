"""`/projects` — CRUD mínimo de metadata (RF-PRJ-01, parcial en Capa 0)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.access import get_access
from perceptron.api.context import EngineContext, get_context
from perceptron.domain.enums import Modality, PrivacyLevel, ProjectStatus, TaskType
from perceptron.domain.models import Project

router = APIRouter(prefix="/projects", tags=["projects"])

Ctx = Annotated[EngineContext, Depends(get_context)]


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    goal: str = ""
    modalities: list[Modality] = Field(default_factory=list)
    task: TaskType | None = None
    target_metric: str | None = None
    privacy_level: PrivacyLevel = PrivacyLevel.L1
    llm_profile_id: str | None = None
    template: str | None = None
    workspace_id: str | None = Field(
        default=None, description="Team Server: workspace del proyecto (default: el del usuario)"
    )


class ProjectPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1, description="Versión conocida por el cliente (bloqueo optimista)")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    goal: str | None = None
    modalities: list[Modality] | None = None
    task: TaskType | None = None
    target_metric: str | None = None
    privacy_level: PrivacyLevel | None = None
    llm_profile_id: str | None = None
    status: ProjectStatus | None = None


@router.get("", operation_id="listProjects")
def list_projects(
    ctx: Ctx,
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Project]:
    visible = get_access(request).visible_projects(request)
    return list(ctx.projects.list(ids=visible, limit=limit, offset=offset))


@router.post("", status_code=status.HTTP_201_CREATED, operation_id="createProject")
def create_project(body: ProjectCreate, ctx: Ctx, request: Request) -> Project:
    draft = get_access(request).prepare_project(request, Project(**body.model_dump()))
    project = ctx.projects.add(draft)
    ctx.files.init_project(project)
    ctx.events.publish("project.created", project_id=project.id)
    return project


@router.get("/{project_id}", operation_id="getProject")
def get_project(project_id: str, ctx: Ctx) -> Project:
    return ctx.projects.get(project_id)


@router.patch("/{project_id}", operation_id="updateProject")
def update_project(project_id: str, body: ProjectPatch, ctx: Ctx) -> Project:
    current = ctx.projects.get(project_id)
    changes = body.model_dump(exclude_unset=True, exclude={"version"})
    candidate = current.model_copy(update={**changes, "version": body.version})
    project = Project.model_validate(candidate.model_dump())
    updated = ctx.projects.update(project)
    ctx.files.write_project(updated)
    ctx.events.publish("project.updated", project_id=updated.id)
    return updated


@router.delete(
    "/{project_id}", status_code=status.HTTP_204_NO_CONTENT, operation_id="deleteProject"
)
def delete_project(project_id: str, ctx: Ctx) -> None:
    # Borra la metadata; los archivos se conservan (borrado físico con confirmación: Capa 3).
    ctx.projects.delete(project_id)
    ctx.events.publish("project.deleted", project_id=project_id)
