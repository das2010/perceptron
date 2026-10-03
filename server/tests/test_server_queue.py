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
        # Un job cancelado figura terminado enseguida, pero el worker sigue hasta el próximo
        # punto de control: si el test siguiente prepara la base mientras tanto, choca con él.
        # Libre durante 1 s seguido: puede haber otro estudio esperando el turno del worker.
        deadline, idle_since = time.time() + 300, time.time()
        while c.runtime is not None and time.time() < deadline:
            if c.runtime.busy is not None:
                idle_since = time.time()
            elif time.time() - idle_since >= 1.0:
                break
            time.sleep(0.1)
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


def test_cancel_while_queued_never_trains(cluster: Cluster, fixtures_dir: Path) -> None:
    """Cancelar un estudio que sigue en la cola: el worker que lo toma después no lo entrena
    (la cancelación queda persistida en el estudio, no solo en el canal de control)."""
    from perceptron.domain.models import Run, Study

    editor = _editor(cluster)
    admin = login(cluster.app, ADMIN_EMAIL, ADMIN_PASSWORD)
    pid, body = prepare_study(editor, fixtures_dir, trials=30, epochs=50)
    busy = ok(editor.post(f"{API}/projects/{pid}/studies", json=body), 202)
    deadline = time.time() + 120
    while ok(editor.get(f"{API}/jobs/{busy['job']['id']}"))["status"] == "queued":
        assert time.time() < deadline, "el worker no tomó el estudio"
        time.sleep(0.2)
    # El worker (un estudio a la vez) está ocupado: el segundo queda esperando.
    waiting = ok(admin.post(f"{API}/projects/{pid}/studies", json=body), 202)
    assert ok(admin.get(f"{API}/jobs/{waiting['job']['id']}"))["status"] == "queued"
    ok(admin.post(f"{API}/studies/{waiting['study']['id']}/cancel"))
    ctx = cluster.app.state.server.ctx
    assert ctx.repo(Study).get(waiting["study"]["id"]).cancel_requested_at is not None
    # Se libera el worker: toma el estudio cancelado y lo descarta sin entrenar.
    ok(editor.post(f"{API}/studies/{busy['study']['id']}/cancel"))
    wait_job(editor, busy["job"]["id"], timeout=300)
    runtime = cluster.runtime
    assert runtime is not None
    deadline = time.time() + 120
    while time.time() < deadline and (
        runtime.last_finished is None or runtime.last_finished[0] != waiting["job"]["id"]
    ):
        time.sleep(0.2)
    assert runtime.last_finished == (waiting["job"]["id"], "cancelled")
    assert not ctx.repo(Run).list(filters={"study_id": waiting["study"]["id"]})


class _SharedRelay:
    """El relay del cluster visto por un segundo servidor: cerrarlo no lo apaga."""

    def __init__(self, inner: Relay) -> None:
        self.inner = inner

    def publish(self, channel: str, message: dict[str, Any]) -> None:
        self.inner.publish(channel, message)

    def listen(self, channel: str, listener: Any) -> Any:
        return self.inner.listen(channel, listener)

    def close(self) -> None:
        pass


def test_restart_keeps_tracking_running_studies(
    cluster: Cluster, fixtures_dir: Path, settings: Settings, server_settings: ServerSettings
) -> None:
    """Reiniciar el servidor no olvida los estudios que siguen en los workers: se ven, cuentan
    para las cuotas y se pueden cancelar."""
    editor = _editor(cluster)
    pid, body = prepare_study(editor, fixtures_dir, trials=30, epochs=50)
    first = ok(editor.post(f"{API}/projects/{pid}/studies", json=body), 202)
    job_id = first["job"]["id"]
    deadline = time.time() + 120
    while ok(editor.get(f"{API}/jobs/{job_id}"))["status"] == "queued":
        assert time.time() < deadline, "el worker no tomó el estudio"
        time.sleep(0.2)
    runtime = cluster.runtime
    assert runtime is not None

    def same_queue(_s: Settings) -> tuple[Any, Dispatcher]:
        return _SharedRelay(cluster.relay), ThreadDispatcher(runtime)

    # "Reinicio": un servidor nuevo, sin nada en memoria, sobre la misma base y los mismos workers.
    quota = server_settings.model_copy(update={"max_running_studies_per_user": 1})
    restarted = create_server_app(settings, quota, queue_factory=same_queue)
    with TestClient(restarted):
        again = login(restarted, "cola@preteco.test", PASSWORD)
        job = ok(again.get(f"{API}/jobs/{job_id}"))
        assert job["status"] == "running" and job["worker"] == "worker-de-prueba"
        assert job["refs"]["study_id"] == first["study"]["id"]
        blocked = again.post(f"{API}/projects/{pid}/studies", json=body)
        assert blocked.status_code == 429 and blocked.json()["details"]["scope"] == "user"
        ok(again.post(f"{API}/studies/{first['study']['id']}/cancel"))
        deadline = time.time() + 300
        while runtime.last_finished is None or runtime.last_finished[0] != job_id:
            assert time.time() < deadline, "el worker no recibió la cancelación"
            time.sleep(0.2)
        assert runtime.last_finished == (job_id, "cancelled")
        assert ok(again.get(f"{API}/jobs/{job_id}"))["status"] == "cancelled"


def test_job_store_never_goes_back(settings: Settings) -> None:
    from perceptron.api.context import EngineContext
    from perceptron.api.jobs import Job
    from perceptron_server.queue.jobstore import DbJobStore

    ctx = EngineContext.create(settings)
    try:
        store = DbJobStore(ctx.db)
        job = Job(id="job_x", kind="study", runner="queue:cpu", refs={"study_id": "std_x"})
        store.save(job)
        store.mark("job_x", status="running", worker="w1")
        store.save(job.model_copy(update={"status": "cancelled"}))  # se canceló en el servidor
        store.mark("job_x", status="succeeded", result={"best": 1})  # el worker terminó después
        store.mark("job_x", status="running")  # un "started" tardío no la revive
        store.mark("job_otro", status="running")  # de otro servidor: no se inventa
        [loaded] = store.load()
        assert loaded.id == "job_x" and loaded.status == "cancelled"
        assert loaded.result == {"best": 1} and loaded.finished_at is not None
    finally:
        ctx.close()


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
