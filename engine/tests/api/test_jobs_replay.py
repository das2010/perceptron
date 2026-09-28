"""WS de jobs con historial: quien se conecta tarde (o reconecta) recibe lo que ya pasó."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext


def test_late_websocket_gets_job_history(client: TestClient, ctx: EngineContext) -> None:
    def work(job: object) -> dict[str, int]:
        for epoch in range(3):
            job.emit("epoch", run_id="r1", epoch=epoch, metrics={"loss": 1.0 / (epoch + 1)})  # type: ignore[attr-defined]
        return {"ok": 1}

    job = ctx.jobs.submit("demo", work)
    deadline = time.time() + 10
    while ctx.jobs.get(job.id).status != "succeeded":  # type: ignore[union-attr]
        assert time.time() < deadline
        time.sleep(0.05)

    with client.websocket_connect(f"/api/v1/jobs/{job.id}") as ws:
        kinds = []
        while True:
            msg = ws.receive_json()
            kinds.append(msg["kind"])
            if msg["kind"] == "finished":
                break
    assert kinds == ["started", "epoch", "epoch", "epoch", "finished"]


def test_remote_job_track_and_apply(ctx: EngineContext) -> None:
    cancelled: list[str] = []
    job = ctx.jobs.track(
        "study", refs={"project_id": "prj_1"}, cancel=lambda: cancelled.append("x")
    )
    assert job.status == "queued" and job.runner == "remote"
    ctx.jobs.apply(job.id, "started", {"worker": "gpu-1"})
    ctx.jobs.apply(job.id, "epoch", {"run_id": "r", "epoch": 0, "metrics": {}})
    assert job.status == "running" and job.worker == "gpu-1"
    ctx.jobs.apply(job.id, "finished", {"status": "succeeded", "result": {"best": 1}})
    assert job.status == "succeeded" and job.result == {"best": 1}
    assert [e["kind"] for e in ctx.jobs.history(job.id)] == ["started", "epoch", "finished"]
    ctx.jobs.apply("job_inexistente", "started", {})  # ignorado
    # Cancelar un job remoto llama al callback (el servidor avisa al worker).
    other = ctx.jobs.track("study", cancel=lambda: cancelled.append("y"))
    ctx.jobs.cancel(other.id)
    assert cancelled == ["y"] and other.status == "cancelled"
    ctx.jobs.shutdown()  # no cancela jobs remotos al apagar el Engine
    assert job.status == "succeeded"
