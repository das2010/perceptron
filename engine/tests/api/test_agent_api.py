"""API del agente: 202 + job, estado, bitácora por WebSocket y aprobación (SPEC §10)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import Project
from perceptron.llm.providers.fake import FakeLLMProvider
from perceptron.llm.types import LLMRequest
from perceptron.services.workflow import Workflow
from perceptron.tracking.tracker import MemoryTracker

API = "/api/v1"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _wait(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("el job no terminó")


def _launch(request: LLMRequest) -> dict[str, Any]:
    text = request.messages[0].content
    data = json.loads(text.split("<datos>", 1)[1].split("</datos>", 1)[0])
    return {
        "log_entry": "Entreno la base.",
        "action": {
            "tool": "launch_study",
            "archspec_id": data["constraints"]["base_archspec_id"],
            "max_trials": 1,
        },
    }


def test_agent_endpoints(
    ctx: EngineContext, client: TestClient, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    wf = Workflow(ctx, MemoryTracker())
    p = ctx.projects.add(Project(name="churn"))
    ctx.files.init_project(p)
    dv = wf.ingest(p.id, fixtures_dir / "uc01_churn" / "churn.csv")
    wf.profile(dv.id)
    pipe = wf.propose_pipeline(dv.id)
    fake_llm.script(
        "agent",
        _launch,
        {"log_entry": "Termino.", "action": {"tool": "finish", "summary": "ok"}},
    )
    launch = _ok(
        client.post(
            f"{API}/projects/{p.id}/agent/runs",
            json={
                "dataset_version_id": dv.id,
                "pipeline_id": pipe.id,
                "limits": {"max_trials": 2, "max_epochs_per_trial": 1},
                "approval": {"mode": "each_iteration"},
            },
        ),
        202,
    )
    aid = launch["agent_run"]["id"]
    assert _wait(client, launch["job"]["id"])["status"] == "succeeded"
    ar = _ok(client.get(f"{API}/agent/runs/{aid}"))
    assert ar["state"] == "awaiting_approval" and ar["pending_action"]["tool"] == "launch_study"

    job = _ok(client.post(f"{API}/agent/runs/{aid}/approve", json={"comment": "ok"}), 202)
    assert _wait(client, job["id"])["status"] == "succeeded"
    ar = _ok(client.get(f"{API}/agent/runs/{aid}"))
    assert ar["state"] == "finished" and ar["model_version_id"] and ar["test_metrics"]

    with client.websocket_connect(f"{API}/agent/runs/{aid}/log") as ws:
        entries = []
        while True:
            msg = ws.receive_json()
            entries.append(msg["entry"])
            if msg["entry"]["message"].startswith("Estado final"):
                break
    assert any(e["kind"] == "approval" for e in entries)
    assert _ok(client.post(f"{API}/agent/runs/{aid}/stop"))["state"] == "finished"
