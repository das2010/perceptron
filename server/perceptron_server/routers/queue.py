"""`/server/queue` — vista de la cola y de los workers (RF-SRV-04)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from perceptron.api.jobs import TERMINAL, Job
from perceptron_server.routers.auth import Who
from perceptron_server.state import ServerState, server_state

router = APIRouter(prefix="/server/queue", tags=["server"])
State = Annotated[ServerState, Depends(server_state)]


class WorkerInfo(BaseModel):
    worker: str
    queues: list[str] = Field(default_factory=list)
    busy_job: str | None = None
    device: str | None = None
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    ram_available_gb: float | None = None
    seen_s_ago: float


class QueueView(BaseModel):
    mode: Literal["local", "queue"]
    workers: list[WorkerInfo]
    jobs: list[Job] = Field(description="Estudios en cola o en curso que podés ver")
    quota_per_user: int
    quota_per_workspace: int


@router.get("", operation_id="getQueue")
def get_queue(who: Who, request: Request, state: State) -> QueueView:
    visible = state.access.visible_projects(request)
    jobs = [
        j
        for j in state.ctx.jobs.list()
        if j.kind == "study"
        and j.status not in TERMINAL
        and (visible is None or j.refs.get("project_id") in visible)
    ]
    return QueueView(
        mode="queue" if state.queue_mode == "queue" else "local",
        workers=[WorkerInfo.model_validate(w) for w in state.workers.list()],
        jobs=jobs,
        quota_per_user=state.server.max_running_studies_per_user,
        quota_per_workspace=state.server.max_running_studies_per_workspace,
    )
