"""`/remote` — servidores de equipo en el desktop (RF-SRV-03/04, Capa 5c).

Conectarse a un Team Server (login una vez; el refresco queda en el keychain), subir un
proyecto y lanzar estudios que corren en los workers del servidor con el progreso en vivo
en el desktop.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict, Field

from perceptron.api.context import EngineContext, get_context
from perceptron.api.jobs import Job
from perceptron.core.config import RuntimeMode
from perceptron.core.errors import ForbiddenError
from perceptron.domain.enums import Device
from perceptron.domain.models import ArchSpecRecord
from perceptron.hpo.strategy import Budget, HPOStrategy
from perceptron.remote.client import RemoteClient, RemoteServer
from perceptron.remote.jobs import launch_remote_study
from perceptron.services.workflow import Workflow

router = APIRouter(prefix="/remote", tags=["remote"])
project_router = APIRouter(tags=["remote"])
Ctx = Annotated[EngineContext, Depends(get_context)]


def desktop_only(ctx: Ctx) -> None:
    """Conectarse a un Team Server es cosa del desktop: en el servidor, un usuario no puede
    hacer que el servidor inicie sesión contra una URL arbitraria (SSRF)."""
    if ctx.settings.mode is RuntimeMode.SERVER:
        raise ForbiddenError("los servidores remotos se configuran desde el desktop")


DesktopOnly = [Depends(desktop_only)]


class ServerConnect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    url: str = Field(pattern=r"^https?://", max_length=500)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256, description="Solo para el login")


class RemoteStudyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: str
    dataset_version_id: str
    pipeline_id: str
    archspec_id: str
    strategy: HPOStrategy | None = None
    budget: Budget = Field(default_factory=Budget)
    device: Device | None = Field(default=None, description="cuda/rocm/xpu → cola GPU")
    workspace_id: str | None = Field(default=None, description="Workspace del servidor")


def client_for(ctx: EngineContext, name: str) -> RemoteClient:
    return RemoteClient(ctx.remotes.get(name), ctx.remotes, **ctx.remote_http_kwargs())


@router.get("/servers", operation_id="listRemoteServers")
def list_servers(ctx: Ctx) -> list[RemoteServer]:
    return ctx.remotes.list()


@router.post(
    "/servers",
    status_code=status.HTTP_201_CREATED,
    operation_id="connectRemoteServer",
    dependencies=DesktopOnly,
)
def connect(body: ServerConnect, ctx: Ctx) -> RemoteServer:
    """Login en el servidor; la contraseña no se guarda (solo el token de refresco)."""
    tokens, me = RemoteClient.login(body.url, body.email, body.password, **ctx.remote_http_kwargs())
    server = RemoteServer(
        name=body.name,
        url=body.url.rstrip("/"),
        email=me["user"]["email"],
        user_id=me["user"]["id"],
        display_name=me["user"].get("display_name"),
    )
    ctx.remotes.save(server, tokens["refresh_token"])
    return server


@router.delete(
    "/servers/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="removeRemoteServer",
    dependencies=DesktopOnly,
)
def remove(name: str, ctx: Ctx) -> Response:
    ctx.remotes.remove(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@project_router.post(
    "/projects/{project_id}/remote/studies",
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="createRemoteStudy",
    dependencies=DesktopOnly,
)
def create_remote_study(project_id: str, body: RemoteStudyCreate, ctx: Ctx) -> Job:
    """Entrena en el servidor: sube datos y arquitectura, encola y sigue el progreso."""
    ctx.projects.get(project_id)
    ctx.repo(ArchSpecRecord).get(body.archspec_id)
    strategy = body.strategy or Workflow(ctx).hpo_strategy(body.archspec_id, body.budget)
    payload = {
        "dataset_version_id": body.dataset_version_id,
        "pipeline_id": body.pipeline_id,
        "archspec_id": body.archspec_id,
        "strategy": strategy.model_dump(mode="json"),
        "budget": body.budget.model_dump(mode="json"),
        "device": body.device.value if body.device else None,
    }
    return launch_remote_study(
        ctx, client_for(ctx, body.server), project_id, payload, workspace_id=body.workspace_id
    )


class PromoteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: str
    workspace_id: str | None = Field(default=None, description="Workspace del servidor")
    dataset_version_ids: list[str] = Field(default_factory=list, max_length=200)
    run_ids: list[str] = Field(default_factory=list, max_length=500)


@project_router.post(
    "/projects/{project_id}/remote/promote",
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="promoteProject",
    dependencies=DesktopOnly,
)
def promote_project(project_id: str, body: PromoteBody, ctx: Ctx) -> Job:
    """Convierte el proyecto local en proyecto de equipo (RF-PRJ-04): sube la metadata, los
    datasets y los runs elegidos al Team Server."""
    from perceptron.remote.sync import ProjectSync

    ctx.projects.get(project_id)
    client = client_for(ctx, body.server)

    def work(job: Any) -> dict[str, int]:
        sync = ProjectSync(ctx, client, project_id)
        return sync.promote(
            workspace_id=body.workspace_id,
            dataset_version_ids=body.dataset_version_ids,
            run_ids=body.run_ids,
            progress=lambda step, **d: job.emit("sync", step=step, **d),
        )

    return ctx.jobs.submit("promote", work, refs={"project_id": project_id, "server": body.server})
