"""`perceptron agent …`: ciclo autónomo por CLI (RF-AGT-01..05, SPEC §7.18)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from perceptron.domain.enums import Modality, PrivacyLevel, TaskType

if TYPE_CHECKING:
    from perceptron.domain.models import AgentRun

agent_app = typer.Typer(help="Agente autónomo (LLM como ML engineer)", no_args_is_help=True)

WorkspaceOpt = Annotated[
    Path | None, typer.Option("--workspace", "-w", help="Directorio del workspace")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Salida JSON")]


def summary(ar: AgentRun) -> dict[str, Any]:
    return {
        "agent_run_id": ar.id,
        "project_id": ar.project_id,
        "state": ar.state.value,
        "stop_reason": ar.stop_reason,
        "iterations": ar.iterations,
        "steps": ar.steps,
        "trials": ar.trials,
        "llm_cost_usd": round(ar.cost_usd, 6),
        "fallback": ar.fallback,
        "best_run_id": ar.best_run_id,
        "model_version_id": ar.model_version_id,
        "test_metrics": ar.test_metrics,
        "archspecs": ar.archspecs,
        "studies": ar.studies,
        "log": [f"[{e['kind']}] {e['message']}" for e in ar.log if e["kind"] != "observation"],
    }


def _print(ar: AgentRun, as_json: bool) -> None:
    s = summary(ar)
    if as_json:
        typer.echo(json.dumps(s, ensure_ascii=False, indent=2, default=str))
        return
    typer.echo("\n".join(s["log"]))
    typer.echo(
        f"Estado: {s['state']} ({s['stop_reason']}) · {s['iterations']} iteraciones · "
        f"{s['trials']} trials · LLM ${s['llm_cost_usd']:.4f}"
    )
    if ar.test_metrics:
        shown = ", ".join(f"{k}={v:.4f}" for k, v in list(ar.test_metrics.items())[:6])
        typer.echo(f"Test: {shown} · modelo {ar.model_version_id}")


@agent_app.command("run")
def agent_run(
    source: Annotated[Path, typer.Argument(help="Archivo o carpeta con los datos")],
    goal: Annotated[str, typer.Option(help="Objetivo en palabras")] = "",
    target: Annotated[str | None, typer.Option(help="Columna objetivo")] = None,
    name: Annotated[str | None, typer.Option(help="Nombre del proyecto")] = None,
    modality: Annotated[Modality | None, typer.Option()] = None,
    task: Annotated[TaskType | None, typer.Option()] = None,
    privacy: Annotated[PrivacyLevel, typer.Option(help="Nivel de privacidad")] = PrivacyLevel.L1,
    profile: Annotated[str | None, typer.Option(help="Perfil LLM (catálogo)")] = None,
    max_time: Annotated[float, typer.Option(help="Tiempo máximo (s)")] = 3600,
    max_iterations: Annotated[int, typer.Option(help="Estudios máximos")] = 4,
    max_steps: Annotated[int, typer.Option(help="Decisiones máximas del LLM")] = 30,
    max_trials: Annotated[int, typer.Option(help="Trials máximos en total")] = 30,
    max_epochs: Annotated[int | None, typer.Option(help="Épocas máximas por trial")] = None,
    max_cost: Annotated[float, typer.Option(help="Costo máximo de LLM (USD)")] = 2.0,
    approval: Annotated[
        str, typer.Option(help="never | each_iteration | family_change | budget_pct")
    ] = "never",
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """De los datos a un modelo, con el LLM decidiendo arquitectura, HPO e iteraciones."""
    from perceptron.agent.loop import AgentRunner
    from perceptron.agent.models import AgentLimits, ApprovalPolicy
    from perceptron.cli.ml import _ctx
    from perceptron.domain.models import Project
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        wf = Workflow(ctx)
        project = ctx.projects.add(
            Project(
                name=name or source.stem, goal=goal, privacy_level=privacy, llm_profile_id=profile
            )
        )
        ctx.files.init_project(project)
        dv = wf.ingest(project.id, source, target=target, modality=modality, task=task)
        wf.profile(dv.id)
        pipeline = wf.propose_pipeline(dv.id)
        runner = AgentRunner(wf)
        ar = runner.start(
            project.id,
            dv.id,
            pipeline.id,
            limits=AgentLimits(
                max_time_s=max_time,
                max_iterations=max_iterations,
                max_steps=max_steps,
                max_trials=max_trials,
                max_epochs_per_trial=max_epochs,
                max_llm_cost_usd=max_cost,
            ),
            approval=ApprovalPolicy(mode=approval),  # type: ignore[arg-type]
        )
        ar = runner.run(ar.id)
        _print(ar, as_json)


@agent_app.command("show")
def agent_show(
    agent_id: Annotated[str, typer.Argument()],
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Estado y bitácora de un agente."""
    from perceptron.cli.ml import _ctx
    from perceptron.domain.models import AgentRun

    with _ctx(workspace) as ctx:
        _print(ctx.repo(AgentRun).get(agent_id), as_json)


def _answer(
    agent_id: str, approved: bool, comment: str | None, workspace: Path | None, as_json: bool
) -> None:
    from perceptron.agent.loop import AgentRunner
    from perceptron.cli.ml import _ctx
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        runner = AgentRunner(Workflow(ctx))
        runner.approve(agent_id, approved=approved, comment=comment)
        _print(runner.run(agent_id), as_json)


@agent_app.command("approve")
def agent_approve(
    agent_id: Annotated[str, typer.Argument()],
    comment: Annotated[str | None, typer.Option()] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Aprueba el paso pendiente y continúa."""
    _answer(agent_id, True, comment, workspace, as_json)


@agent_app.command("reject")
def agent_reject(
    agent_id: Annotated[str, typer.Argument()],
    comment: Annotated[str | None, typer.Option()] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Rechaza el paso pendiente (con motivo) y continúa."""
    _answer(agent_id, False, comment, workspace, as_json)
