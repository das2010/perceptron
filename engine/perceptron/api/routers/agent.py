"""Agente autónomo y mini-torneo (SPEC §10: `/agent/runs`, RF-AGT-01..05, RF-ARC-03)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket, status
from pydantic import BaseModel, ConfigDict, Field

from perceptron.agent.loop import AGENT_TOPIC, AgentRunner
from perceptron.agent.models import AgentLimits, ApprovalPolicy
from perceptron.api.context import EngineContext, get_context
from perceptron.api.jobs import Job, JobContext
from perceptron.api.streaming import stream_events
from perceptron.domain.enums import Device
from perceptron.domain.models import AgentRun
from perceptron.services.workflow import Workflow

router = APIRouter()
Ctx = Annotated[EngineContext, Depends(get_context)]


class AgentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    pipeline_id: str
    limits: AgentLimits = Field(default_factory=AgentLimits)
    approval: ApprovalPolicy = Field(default_factory=ApprovalPolicy)


class AgentLaunch(BaseModel):
    agent_run: AgentRun
    job: Job


class ApprovalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment: str | None = None


def _continue(ctx: EngineContext, agent_id: str, first: Any = None) -> Job:
    """Corre el ciclo en segundo plano (202 + job, SPEC §10)."""

    def work(job: JobContext) -> Any:
        runner = AgentRunner(Workflow(ctx))

        def cancel() -> None:
            runner.stop(agent_id)

        job.cancel_callback = cancel
        if first is not None:
            first(runner)
        ar = runner.run(agent_id)
        return {"state": ar.state.value, "best_run_id": ar.best_run_id}

    project_id = ctx.repo(AgentRun).get(agent_id).project_id
    return ctx.jobs.submit("agent", work, refs={"agent_run_id": agent_id, "project_id": project_id})


@router.post(
    "/projects/{project_id}/agent/runs",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["agent"],
    operation_id="createAgentRun",
)
def create_agent_run(project_id: str, body: AgentCreate, ctx: Ctx) -> AgentLaunch:
    ctx.projects.get(project_id)
    ar = AgentRunner(Workflow(ctx)).start(
        project_id,
        body.dataset_version_id,
        body.pipeline_id,
        limits=body.limits,
        approval=body.approval,
    )
    return AgentLaunch(agent_run=ar, job=_continue(ctx, ar.id))


@router.get("/agent/runs/{agent_id}", tags=["agent"], operation_id="getAgentRun")
def get_agent_run(agent_id: str, ctx: Ctx) -> AgentRun:
    return ctx.repo(AgentRun).get(agent_id)


@router.post(
    "/agent/runs/{agent_id}/approve",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["agent"],
    operation_id="approveAgentRun",
)
def approve_agent_run(agent_id: str, body: ApprovalBody, ctx: Ctx) -> Job:
    ctx.repo(AgentRun).get(agent_id)
    return _continue(
        ctx, agent_id, lambda r: r.approve(agent_id, approved=True, comment=body.comment)
    )


@router.post(
    "/agent/runs/{agent_id}/reject",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["agent"],
    operation_id="rejectAgentRun",
)
def reject_agent_run(agent_id: str, body: ApprovalBody, ctx: Ctx) -> Job:
    ctx.repo(AgentRun).get(agent_id)
    return _continue(
        ctx, agent_id, lambda r: r.approve(agent_id, approved=False, comment=body.comment)
    )


@router.post("/agent/runs/{agent_id}/stop", tags=["agent"], operation_id="stopAgentRun")
def stop_agent_run(agent_id: str, ctx: Ctx) -> AgentRun:
    return AgentRunner(Workflow(ctx)).stop(agent_id)


@router.websocket("/agent/runs/{agent_id}/log")
async def agent_log(ws: WebSocket, agent_id: str) -> None:
    """Bitácora en vivo (RF-AGT-04): primero lo ya registrado, luego cada entrada nueva."""
    ctx: EngineContext = ws.app.state.ctx
    ar = ctx.repo(AgentRun).find(agent_id)
    initial = [{"agent_id": agent_id, "entry": e} for e in (ar.log if ar else [])]
    if ar is None:
        initial = [
            {
                "agent_id": agent_id,
                "entry": {"kind": "system", "message": "Estado final: desconocido"},
            }
        ]

    def final(msg: dict[str, Any]) -> bool:
        # Los agentes terminados ya tienen la entrada final en la bitácora inicial.
        entry = msg.get("entry") or {}
        return str(entry.get("message", "")).startswith("Estado final")

    await stream_events(
        ws,
        ctx.events,
        AGENT_TOPIC,
        lambda ev: ev.payload.get("agent_id") == agent_id,
        initial=initial,
        until=final,
    )


# ------------------------------------------------------------------ mini-torneo


class TournamentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    pipeline_id: str
    archspec_ids: list[str] = Field(min_length=2, max_length=4)
    fraction: float = Field(default=0.1, gt=0, le=1, description="Fracción de las épocas")
    subset: float = Field(default=0.3, gt=0, le=1, description="Fracción de train por época")
    metric: str = "val_loss"
    device: Device | None = None


@router.post(
    "/projects/{project_id}/arch/tournament",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["arch"],
    operation_id="runTournament",
)
def run_tournament(project_id: str, body: TournamentBody, ctx: Ctx) -> Job:
    """Entrena cada propuesta con presupuesto corto; la mejor en validación gana (RF-ARC-03)."""
    from perceptron.services.tournament import mini_tournament

    ctx.projects.get(project_id)

    def work(_: JobContext) -> Any:
        res = mini_tournament(
            Workflow(ctx),
            project_id,
            body.dataset_version_id,
            body.pipeline_id,
            body.archspec_ids,
            fraction=body.fraction,
            subset=body.subset,
            metric=body.metric,
            device=body.device,
        )
        return {
            "id": res.id,
            "metric": res.metric,
            "direction": res.direction,
            "winner": res.winner,
            "entries": [e.__dict__ for e in res.entries],
        }

    return ctx.jobs.submit("tournament", work, refs={"project_id": project_id})
