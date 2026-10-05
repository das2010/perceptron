"""Estado y control de estudios: listar, interrumpidos y reanudar (caso «Tubos»)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.api.jobs import Job
from perceptron.domain.enums import RunStatus
from perceptron.domain.models import Run, Study, utcnow

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _study(ctx: EngineContext, pid: str, statuses: list[RunStatus]) -> Study:
    study = ctx.repo(Study).add(
        Study(
            project_id=pid, name="hpo-cnn", budget={"max_trials": 5}, cancel_requested_at=utcnow()
        )
    )
    for i, status in enumerate(statuses):
        ctx.repo(Run).add(
            Run(
                id=f"{study.id}-t00{i}",
                project_id=pid,
                study_id=study.id,
                archspec_id="arc_x",
                pipeline_id="pip_x",
                dataset_version_id="dsv_x",
                status=status,
                metrics={"val_loss": 1.0 - i / 10} if status is RunStatus.SUCCEEDED else {},
            )
        )
    return study


def test_interrupted_study_is_listed_and_resumes_cleanly(
    client: TestClient, ctx: EngineContext
) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Tubos"}), 201)["id"]
    ok, running = RunStatus.SUCCEEDED, RunStatus.RUNNING
    study = _study(ctx, pid, [ok, ok, running])
    [view] = _ok(client.get(f"{API}/projects/{pid}/studies"))
    assert view["status"] == "interrupted" and view["resumable"]
    assert view["trials_done"] == 2 and view["trials_total"] == 5
    assert view["best_run_id"] == f"{study.id}-t001"

    launched: list[Study] = []

    def launcher(c: EngineContext, s: Study) -> Job:
        launched.append(s)
        return c.jobs.track("study", refs={"study_id": s.id, "project_id": pid})

    ctx.study_launcher = launcher
    _ok(client.post(f"{API}/studies/{study.id}/resume"), 202)
    # Sin la cancelación persistida (el worker lo descartaría) y sin el run colgado.
    assert launched and launched[0].cancel_requested_at is None
    assert ctx.repo(Study).get(study.id).cancel_requested_at is None
    assert ctx.repo(Run).get(f"{study.id}-t002").status is RunStatus.FAILED
    [view] = _ok(client.get(f"{API}/projects/{pid}/studies"))
    assert view["status"] == "queued" and view["job_id"] and not view["resumable"]
    again = client.post(f"{API}/studies/{study.id}/resume")
    assert again.status_code == 409  # ya está en ejecución


def test_finished_and_stopped_studies(client: TestClient, ctx: EngineContext) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "otro"}), 201)["id"]
    done = _study(ctx, pid, [RunStatus.SUCCEEDED] * 5)
    stopped = _study(ctx, pid, [RunStatus.SUCCEEDED, RunStatus.CANCELLED])
    views = {v["study"]["id"]: v for v in _ok(client.get(f"{API}/projects/{pid}/studies"))}
    assert views[done.id]["status"] == "finished" and not views[done.id]["resumable"]
    assert views[stopped.id]["status"] == "stopped" and views[stopped.id]["resumable"]
