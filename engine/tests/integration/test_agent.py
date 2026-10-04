"""Agente autónomo con guiones de FakeLLMProvider sobre UC-01 (RF-AGT-01..05, RF-ARC-03)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from perceptron.agent.loop import AgentRunner, control_for
from perceptron.agent.models import AgentLimits, ApprovalPolicy
from perceptron.api.context import EngineContext
from perceptron.archspec.schema import ArchSpec
from perceptron.domain.enums import AgentState, Origin
from perceptron.domain.models import ArchSpecRecord, LLMCall, Project
from perceptron.llm.errors import LLMProviderError
from perceptron.llm.privacy.audit import find_leaks, individual_values
from perceptron.llm.providers.fake import FakeLLMProvider
from perceptron.llm.types import LLMRequest
from perceptron.services.tournament import mini_tournament, spec_epochs
from perceptron.services.workflow import Workflow
from perceptron.tracking.tracker import MemoryTracker

LIMITS = AgentLimits(max_trials=4, max_epochs_per_trial=1, max_iterations=3, max_steps=8)


def datos(request: LLMRequest) -> dict[str, Any]:
    text = request.messages[0].content
    return json.loads(text.split("<datos>", 1)[1].split("</datos>", 1)[0])


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


@pytest.fixture
def wf(ctx: EngineContext) -> Workflow:
    return Workflow(ctx, MemoryTracker())


def _setup(wf: Workflow, fixtures_dir: Path) -> tuple[str, str, str]:
    p = wf.ctx.projects.add(Project(name="churn", goal="anticipar la baja de clientes"))
    wf.ctx.files.init_project(p)
    dv = wf.ingest(p.id, fixtures_dir / "uc01_churn" / "churn.csv")
    wf.profile(dv.id)
    return p.id, dv.id, wf.propose_pipeline(dv.id).id


def launch_base(request: LLMRequest) -> dict[str, Any]:
    base = datos(request)["constraints"]["base_archspec_id"]
    return {
        "log_entry": "Iteración 1: entreno la arquitectura base como referencia.",
        "action": {"tool": "launch_study", "archspec_id": base, "max_trials": 1},
    }


def propose_variant(request: LLMRequest) -> dict[str, Any]:
    spec = datos(request)["constraints"]["base_archspec"]
    return {
        "log_entry": "Pruebo una variante de la base.",
        "action": {
            "tool": "propose_archspec",
            "title": "MLP del agente",
            "archspec": {**spec, "name": "mlp-agente"},
            "rationale": "Misma familia, para comparar.",
        },
    }


def curves(request: LLMRequest) -> dict[str, Any]:
    best = datos(request)["system"]["best_run_id"]
    return {
        "log_entry": "Reviso las curvas del mejor run.",
        "action": {"tool": "get_run_curves", "run_id": best},
    }


FINISH = {
    "log_entry": "No espero mejoras: termino.",
    "action": {"tool": "finish", "summary": "Listo."},
}


def test_agent_happy_path_and_privacy(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    pid, dv, pipe = _setup(wf, fixtures_dir)
    fake_llm.script("agent", launch_base, propose_variant, curves, FINISH)
    runner = AgentRunner(wf)
    ar = runner.run(runner.start(pid, dv, pipe, limits=LIMITS).id)
    assert ar.state is AgentState.FINISHED and ar.stop_reason == "finish", ar.log[-3:]
    assert ar.iterations == 1 and ar.steps == 4 and not ar.fallback
    assert ar.model_version_id and "accuracy" in ar.test_metrics
    assert len(ar.archspecs) == 2
    assert wf.ctx.repo(ArchSpecRecord).get(ar.archspecs[1]).origin is Origin.AGENT
    kinds = [e["kind"] for e in ar.log]
    assert kinds.count("decision") == 4 and "finish" in kinds
    assert ar.log[kinds.index("decision")]["message"].startswith("Iteración 1")
    # El agente nunca ve el test: sus payloads solo tienen métricas de validación.
    calls = [
        c
        for c in wf.ctx.repo(LLMCall).list(filters={"project_id": pid}, limit=500)
        if c.purpose.value == "agent"
    ]
    assert len(calls) == 4 and all(c.scope == f"agent:{ar.id}" for c in calls)
    assert all("test_" not in json.dumps(c.payload) for c in calls)
    view = wf.view(wf.dataset(dv))
    assert not find_leaks([(c.id, c.payload) for c in calls], individual_values(view))


def test_agent_validator_feedback_and_iteration_limit(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    pid, dv, pipe = _setup(wf, fixtures_dir)
    fake_llm.script("agent", FINISH, launch_base)
    runner = AgentRunner(wf)
    limits = LIMITS.model_copy(update={"max_iterations": 1})
    ar = runner.run(runner.start(pid, dv, pipe, limits=limits).id)
    retry = fake_llm.calls("agent")[1]
    assert "todavía no hay runs" in retry.messages[-1].content
    assert ar.state is AgentState.FINISHED and ar.stop_reason == "limit:iteraciones"
    assert ar.model_version_id


def test_agent_approval_each_iteration(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    pid, dv, pipe = _setup(wf, fixtures_dir)
    fake_llm.script("agent", launch_base, FINISH)
    runner = AgentRunner(wf)
    ar = runner.start(pid, dv, pipe, limits=LIMITS, approval=ApprovalPolicy(mode="each_iteration"))
    ar = runner.run(ar.id)
    assert ar.state is AgentState.AWAITING_APPROVAL and ar.pending_action
    assert ar.iterations == 0
    ar = runner.approve(ar.id, approved=True, comment="dale")
    assert ar.iterations == 1 and ar.state is AgentState.RUNNING
    ar = runner.run(ar.id)
    assert ar.state is AgentState.FINISHED


def test_agent_falls_back_to_rules(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    pid, dv, pipe = _setup(wf, fixtures_dir)
    fake_llm.script("agent", LLMProviderError("sin conexión"))
    runner = AgentRunner(wf)
    ar = runner.run(runner.start(pid, dv, pipe, limits=LIMITS).id)
    assert ar.fallback and ar.state is AgentState.FINISHED
    assert ar.stop_reason and ar.stop_reason.startswith("fallback:")
    assert ar.iterations == 1 and ar.model_version_id


def test_agent_stop(wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path) -> None:
    pid, dv, pipe = _setup(wf, fixtures_dir)
    runner = AgentRunner(wf)
    ar = runner.start(pid, dv, pipe, limits=LIMITS)
    control_for(ar.id).request_stop()
    ar = runner.run(ar.id)
    assert ar.state is AgentState.STOPPED and not fake_llm.calls()


def test_mini_tournament(wf: Workflow, fixtures_dir: Path) -> None:
    pid, dv, pipe = _setup(wf, fixtures_dir)
    base, _ = wf.propose_architecture(dv, pipe)
    spec = ArchSpec.model_validate(base.spec)
    other = wf.save_archspec(pid, spec.model_copy(update={"name": "variante"}))
    res = mini_tournament(wf, pid, dv, pipe, [base.id, other.id], fraction=0.05, subset=0.5)
    assert [e.status for e in res.entries] == ["complete", "complete"]
    assert res.winner in {base.id, other.id} and res.direction == "minimize"
    expected = max(1, round(spec_epochs(spec) * 0.05))
    assert all(e.epochs == expected for e in res.entries)
    assert res.winner_entry is not None and res.winner_entry.metric is not None


def test_agent_repeated_actions_are_not_reexecuted(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    pid, dv, pipe = _setup(wf, fixtures_dir)
    fake_llm.script("agent", launch_base, curves)  # luego repite `curves` para siempre
    runner = AgentRunner(wf)
    ar = runner.run(runner.start(pid, dv, pipe, limits=LIMITS).id)
    assert ar.state is AgentState.FINISHED and ar.stop_reason == "limit:acciones repetidas"
    assert ar.iterations == 1 and ar.steps == 5  # launch + curves + 3 repeticiones
    errors = [e for e in ar.log if e["kind"] == "observation" and "error" in (e["data"] or {})]
    assert len(errors) == 2 and "repetida" in errors[0]["data"]["error"]


def test_agent_starts_linear_when_the_brief_asks_to_extrapolate(
    wf: Workflow, fake_llm: FakeLLMProvider, tmp_path: Path
) -> None:
    """Caso «Sensores» (ADR-0040): la ficha pide extrapolar → base lineal, la red de alternativa,
    y el agente recibe la ficha y la guía."""
    import numpy as np
    import polars as pl

    from perceptron.services.wizard import Wizard

    rng = np.random.default_rng(0)
    s1, s2 = rng.uniform(0, 1.5, 300), rng.uniform(1.2, 2.3, 300)
    path = tmp_path / "sensores.csv"
    pl.DataFrame({"sensor1": s1, "sensor2": s2, "resultado": s1 * (1 + np.sqrt(s2))}).write_csv(
        path
    )
    p = wf.ctx.projects.add(Project(name="sensores"))
    wf.ctx.files.init_project(p)
    dv = wf.ingest(p.id, path)
    wizard = Wizard(wf)
    draft = wizard.get(p.id)
    wizard.update(
        p.id,
        version=draft.version,
        values={"brief": {"problem": "value", "extrapolate": True}},
    )
    pipe = wf.propose_pipeline(dv.id).id
    runner = AgentRunner(wf)
    ar = runner.start(p.id, dv.id, pipe, limits=LIMITS)
    assert len(ar.archspecs) == 2
    linear = ArchSpec.model_validate(wf.ctx.repo(ArchSpecRecord).get(ar.archspecs[0]).spec)
    assert linear.name == "tabular-linear" and linear.optimizer.weight_decay == 0.0
    assert "lineal" in ar.log[0]["message"]

    seen: list[dict[str, Any]] = []

    def finish_now(request: LLMRequest) -> dict[str, Any]:
        seen.append(datos(request)["constraints"])
        return {
            "log_entry": "Cierro sin entrenar.",
            "action": {"tool": "finish", "summary": "fin"},
        }

    # Primero captura el contexto (cierre rechazado: no hay runs), después entrena la base y cierra.
    fake_llm.script("agent", finish_now, launch_base, FINISH)
    runner.run(ar.id)
    assert seen and seen[0]["use_case"]["extrapolate"] is True
    assert "extrapolar" in seen[0]["guidance"][0]
