"""Cola de jobs y workers (RF-SRV-04, Capa 5b): el estudio corre en otro EngineContext
(como un worker remoto) y el progreso vuelve al servidor por el relay."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from srv_helpers import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    API,
    PASSWORD,
    login,
    ok,
    prepare_study,
    wait_job,
)

from perceptron.api.jobs import JOB_TOPIC
from perceptron.core.config import Settings
from perceptron.core.events import Event
from perceptron_server.app import create_server_app
from perceptron_server.queue.dispatch import Dispatcher, ThreadDispatcher
from perceptron_server.queue.relay import MemoryRelay, Relay
from perceptron_server.queue.worker import WorkerRuntime
from perceptron_server.settings import ServerSettings


class Cluster:
    """Servidor + un worker en memoria (otro EngineContext sobre la misma base y workspace)."""

    def __init__(self, settings: Settings, server: ServerSettings) -> None:
        self.relay = MemoryRelay()
        self.runtime: WorkerRuntime | None = None

        def factory(s: Settings) -> tuple[Relay, Dispatcher]:
            self.runtime = WorkerRuntime.create(s, self.relay, ["cpu", "gpu"], heartbeat_s=0.2)
            self.runtime.name = "worker-de-prueba"
            self.runtime.start_heartbeat()
            return self.relay, ThreadDispatcher(self.runtime)

        self.app: FastAPI = create_server_app(settings, server, queue_factory=factory)


@pytest.fixture
def cluster(settings: Settings, server_settings: ServerSettings) -> Iterator[Cluster]:
    c = Cluster(settings, server_settings.model_copy(update={"max_running_studies_per_user": 1}))
    with TestClient(c.app):
        yield c
    if c.runtime is not None:
        c.runtime.stop()


def _editor(c: Cluster) -> TestClient:
    admin = login(c.app, ADMIN_EMAIL, ADMIN_PASSWORD)
    ws = ok(admin.get(f"{API}/auth/me"))["workspaces"][0]["id"]
    body = {"email": "cola@preteco.test", "password": PASSWORD}
    ok(admin.post(f"{API}/admin/users", json={**body, "workspace_id": ws, "role": "editor"}), 201)
    return login(c.app, body["email"], PASSWORD)


def test_study_runs_on_worker_with_live_progress(cluster: Cluster, fixtures_dir: Path) -> None:
    editor = _editor(cluster)
    pid, body = prepare_study(editor, fixtures_dir, trials=1, epochs=2)
    relayed: list[Event] = []
    cluster.app.state.ctx.events.subscribe(JOB_TOPIC, relayed.append)

    launch = ok(editor.post(f"{API}/projects/{pid}/studies", json=body), 202)
    job = launch["job"]
    assert job["runner"] == "queue:cpu" and job["refs"]["queue"] == "cpu"
    assert job["refs"]["user_id"] and job["refs"]["workspace_id"]
    done = wait_job(editor, job["id"])
    assert done["status"] == "succeeded", done["error"]
    assert done["worker"] == "worker-de-prueba"
    assert done["result"]["best_trial"]["run_id"]
    # El progreso llegó al bus del servidor (lo que ven los WebSocket).
    kinds = [e.payload["kind"] for e in relayed if e.payload.get("job_id") == job["id"]]
    assert kinds[0] == "started" and "epoch" in kinds and kinds[-1] == "finished"
    run = ok(editor.get(f"{API}/runs/{done['result']['best_trial']['run_id']}"))
    assert run["status"] == "succeeded"

    view = ok(editor.get(f"{API}/server/queue"))
    assert view["mode"] == "queue" and view["jobs"] == []
    assert [w["worker"] for w in view["workers"]] == ["worker-de-prueba"]
    assert view["workers"][0]["busy_job"] is None


def test_cancel_and_quota(cluster: Cluster, fixtures_dir: Path) -> None:
    editor = _editor(cluster)
    pid, body = prepare_study(editor, fixtures_dir, trials=30, epochs=50)
    first = ok(editor.post(f"{API}/projects/{pid}/studies", json=body), 202)
    # Cuota por usuario = 1: el segundo espera a que termine el primero.
    second = editor.post(f"{API}/projects/{pid}/studies", json=body)
    assert second.status_code == 429 and second.json()["details"]["scope"] == "user"

    deadline = time.time() + 120
    while ok(editor.get(f"{API}/jobs/{first['job']['id']}"))["status"] == "queued":
        assert time.time() < deadline, "el worker no tomó el estudio"
        time.sleep(0.2)
    queue = ok(editor.get(f"{API}/server/queue"))
    assert [j["id"] for j in queue["jobs"]] == [first["job"]["id"]]
    ok(editor.post(f"{API}/studies/{first['study']['id']}/cancel"))
    done = wait_job(editor, first["job"]["id"], timeout=300)
    assert done["status"] == "cancelled"
    assert ok(editor.post(f"{API}/projects/{pid}/studies", json=body), 202)  # hay cupo otra vez
    last = ok(editor.get(f"{API}/jobs"))[0]
    ok(editor.post(f"{API}/studies/{last['refs']['study_id']}/cancel"))
    wait_job(editor, last["id"], timeout=300)


def test_gpu_studies_go_to_the_gpu_queue(cluster: Cluster, fixtures_dir: Path) -> None:
    editor = _editor(cluster)
    pid, body = prepare_study(editor, fixtures_dir)
    launch = ok(editor.post(f"{API}/projects/{pid}/studies", json={**body, "device": "cuda"}), 202)
    assert launch["job"]["refs"]["queue"] == "gpu"
    wait_job(editor, launch["job"]["id"])  # sin GPU real puede fallar: solo importa la cola


# ---------------------------------------------------------------- Celery + Valkey/Redis reales

REDIS_URL = os.environ.get("PERCEPTRON_TEST_REDIS_URL")


@pytest.mark.skipif(not REDIS_URL, reason="requiere PERCEPTRON_TEST_REDIS_URL (CI)")
def test_celery_worker_process(
    settings: Settings, server_settings: ServerSettings, fixtures_dir: Path, tmp_path: Path
) -> None:
    assert REDIS_URL
    server = server_settings.model_copy(update={"redis_url": SecretStr(REDIS_URL)})
    assert settings.database_url is not None
    env: dict[str, Any] = {
        **os.environ,
        "PERCEPTRON_DATABASE_URL": settings.database_url.get_secret_value(),
        "PERCEPTRON_WORKSPACE_DIR": str(settings.workspace_dir),
        "PERCEPTRON_SERVER__REDIS_URL": REDIS_URL,
        "PERCEPTRON_SERVER__SECRET_KEY": server.secret_key.get_secret_value(),
        "PERCEPTRON_SERVER__WORKER_HEARTBEAT_S": "1",
        "PERCEPTRON_LOGGING__TO_FILE": "false",
    }
    app = create_server_app(settings, server)
    log = (tmp_path / "worker.log").open("w", encoding="utf-8")
    with TestClient(app):
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "perceptron_server",
                "worker",
                "--queues",
                "cpu",
                "--name",
                "ci-worker",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            admin = login(app, ADMIN_EMAIL, ADMIN_PASSWORD)
            pid, body = prepare_study(admin, fixtures_dir)
            launch = ok(admin.post(f"{API}/projects/{pid}/studies", json=body), 202)
            done = wait_job(admin, launch["job"]["id"], timeout=600)
            assert done["status"] == "succeeded", done["error"]
            assert done["worker"] == "ci-worker"
            deadline = time.time() + 30
            while not ok(admin.get(f"{API}/server/queue"))["workers"]:
                assert time.time() < deadline, "sin latido del worker"
                time.sleep(0.5)
        finally:
            worker.terminate()
            worker.wait(timeout=30)
            log.close()
            print((tmp_path / "worker.log").read_text(encoding="utf-8")[-5000:])
