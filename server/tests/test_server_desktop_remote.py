"""Aceptación de la Capa 5 (SPEC §14): un desktop sin GPU lanza un run en un worker del
servidor y ve el progreso en vivo (RF-SRV-03/04).

El "desktop" es otro Engine con su propio workspace (SQLite local); el servidor corre en
uvicorn real (HTTP + WebSocket) con la cola y un worker. En CI el worker es CPU: la cola
`gpu` se prueba por ruteo en `test_server_queue.py`; con un worker GPU real (perfil `gpu`
del Compose) el camino es el mismo.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from srv_helpers import ADMIN_EMAIL, ADMIN_PASSWORD, API, PASSWORD, LiveServer, login, ok

from perceptron.api.app import create_app
from perceptron.api.context import EngineContext
from perceptron.api.jobs import JOB_TOPIC
from perceptron.core.config import LLMSettings, LoggingSettings, Settings
from perceptron.core.events import Event
from perceptron.domain.models import Run, Study
from perceptron.llm.config import LLMConfig
from perceptron.llm.gateway import Gateway
from perceptron.llm.providers.fake import FakeLLMProvider
from perceptron.llm.secrets import MemorySecrets
from perceptron_server.app import create_server_app
from perceptron_server.queue.dispatch import Dispatcher, ThreadDispatcher
from perceptron_server.queue.relay import MemoryRelay, Relay
from perceptron_server.queue.worker import WorkerRuntime
from perceptron_server.settings import ServerSettings


@pytest.fixture
def team_server(settings: Settings, server_settings: ServerSettings) -> Iterator[LiveServer]:
    relay = MemoryRelay()
    runtimes: list[WorkerRuntime] = []

    def factory(s: Settings) -> tuple[Relay, Dispatcher]:
        runtime = WorkerRuntime.create(s, relay, ["cpu", "gpu"], heartbeat_s=0.5)
        runtime.name = "worker-del-servidor"
        runtimes.append(runtime)
        return relay, ThreadDispatcher(runtime)

    app = create_server_app(settings, server_settings, queue_factory=factory)
    with LiveServer(app) as live:
        live.app = app  # type: ignore[attr-defined]
        yield live
    for r in runtimes:
        r.stop()


@pytest.fixture
def desktop(tmp_path: Path) -> Iterator[tuple[TestClient, EngineContext]]:
    settings = Settings(
        workspace_dir=tmp_path / "desktop sin GPU ñ",
        logging=LoggingSettings(to_file=False),
    )
    ctx = EngineContext.create(settings)
    # Keychain en memoria (el runner no tiene llavero del sistema).
    config = LLMConfig(LLMSettings(enabled=False), settings.workspace_dir)
    fake = FakeLLMProvider()
    ctx.use_llm(Gateway(ctx.db, config, MemorySecrets(), provider_factory=lambda *_: fake))
    with TestClient(create_app(ctx=ctx)) as client:
        yield client, ctx
    ctx.close()


def _wait(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job: dict[str, Any] = ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.3)
    raise AssertionError("el estudio remoto no terminó")


def test_desktop_trains_on_server_worker_with_live_progress(
    team_server: LiveServer,
    desktop: tuple[TestClient, EngineContext],
    fixtures_dir: Path,
) -> None:
    app = team_server.app  # type: ignore[attr-defined]
    admin = login(app, ADMIN_EMAIL, ADMIN_PASSWORD)
    ws = ok(admin.get(f"{API}/auth/me"))["workspaces"][0]["id"]
    body = {"email": "analista@preteco.test", "password": PASSWORD}
    ok(admin.post(f"{API}/admin/users", json={**body, "workspace_id": ws, "role": "editor"}), 201)

    dclient, dctx = desktop
    wrong = dclient.post(
        f"{API}/remote/servers",
        json={"name": "equipo", "url": team_server.url, "email": body["email"], "password": "x"},
    )
    assert wrong.status_code == 401
    server = ok(
        dclient.post(
            f"{API}/remote/servers",
            json={"name": "equipo", "url": team_server.url, **body},
        ),
        201,
    )
    assert server["email"] == body["email"] and "password" not in server
    assert [s["name"] for s in ok(dclient.get(f"{API}/remote/servers"))] == ["equipo"]

    # Proyecto armado en el desktop (datos, pipeline y arquitectura locales).
    pid = ok(dclient.post(f"{API}/projects", json={"name": "Churn remoto"}), 201)["id"]
    src = ok(
        dclient.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = ok(dclient.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    ok(dclient.post(f"{API}/datasets/{dv['id']}/profile"))
    pipe = ok(
        dclient.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    arch = ok(
        dclient.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]["archspec"]

    events: list[Event] = []
    dctx.events.subscribe(JOB_TOPIC, events.append)
    study_body = {
        "server": "equipo",
        "dataset_version_id": dv["id"],
        "pipeline_id": pipe["id"],
        "archspec_id": arch["id"],
        "budget": {"max_trials": 1, "max_epochs_per_trial": 3},
    }
    job = ok(dclient.post(f"{API}/projects/{pid}/remote/studies", json=study_body), 202)
    assert job["runner"] == "remote:equipo"
    done = _wait(dclient, job["id"])
    assert done["status"] == "succeeded", done["error"]
    assert done["worker"] == "worker-del-servidor"

    # Progreso en vivo en el desktop: los eventos del worker llegaron por el WS del servidor.
    kinds = [e.payload["kind"] for e in events if e.payload.get("job_id") == job["id"]]
    assert "started" in kinds and "epoch" in kinds and kinds[-1] == "finished"
    epochs = [e.payload["data"] for e in events if e.payload.get("kind") == "epoch"]
    assert epochs and "metrics" in epochs[0]

    # Resultados bajados al desktop: estudio, runs y archivos del run.
    study_id = done["refs"]["study_id"]
    assert dctx.repo(Study).get(study_id).project_id == pid
    runs = dctx.repo(Run).list(filters={"study_id": study_id})
    assert runs and all(r.status.value == "succeeded" for r in runs)
    run_dir = dctx.settings.paths.project(pid).root / "runs" / runs[0].id
    assert run_dir.is_dir() and any(run_dir.iterdir())
    assert ok(dclient.get(f"{API}/runs/{runs[0].id}"))["metrics"]

    # En el servidor: proyecto de equipo en el workspace, visible para el equipo.
    team = [p for p in ok(admin.get(f"{API}/projects")) if p["id"] == pid]
    assert team and team[0]["scope"] == "team" and team[0]["workspace_id"] == ws
    uploads = ok(admin.get(f"{API}/admin/audit", params={"action": "sync.upload"}))
    assert uploads

    # Segundo estudio: los archivos ya están en el servidor (no se suben de nuevo).
    events.clear()
    again = ok(dclient.post(f"{API}/projects/{pid}/remote/studies", json=study_body), 202)
    assert _wait(dclient, again["id"])["status"] == "succeeded"
    pushed = [
        e.payload["data"]
        for e in events
        if e.payload.get("kind") == "sync" and e.payload["data"].get("step") == "pushed"
    ]
    assert pushed and pushed[0]["uploaded"] == 0 and pushed[0]["skipped"] > 0


def test_sync_rejects_foreign_ids_and_stale_versions(
    team_server: LiveServer, fixtures_dir: Path
) -> None:
    app = team_server.app  # type: ignore[attr-defined]
    admin = login(app, ADMIN_EMAIL, ADMIN_PASSWORD)
    ws = ok(admin.get(f"{API}/auth/me"))["workspaces"][0]["id"]
    for email in ("a@preteco.test", "b@preteco.test"):
        user = {"email": email, "password": PASSWORD, "workspace_id": ws, "role": "editor"}
        ok(admin.post(f"{API}/admin/users", json=user), 201)
    a = login(app, "a@preteco.test", PASSWORD)
    pa = ok(a.post(f"{API}/projects", json={"name": "A"}), 201)
    pb = ok(a.post(f"{API}/projects", json={"name": "B"}), 201)
    pipeline = {"name": "p", "graph": {}}
    first = ok(
        a.put(
            f"{API}/sync/projects/{pa['id']}/entities/Pipeline/pip_01TEST",
            json={"data": pipeline, "base_version": None},
        )
    )
    assert first["created"] and first["version"] == 1
    # Mismo id desde otro proyecto: no se puede pisar.
    foreign = a.put(
        f"{API}/sync/projects/{pb['id']}/entities/Pipeline/pip_01TEST",
        json={"data": pipeline, "base_version": 1},
    )
    assert foreign.status_code == 403
    # Versión vieja: conflicto (alguien lo cambió en el servidor).
    ok(
        a.put(
            f"{API}/sync/projects/{pa['id']}/entities/Pipeline/pip_01TEST",
            json={"data": {**pipeline, "name": "v2"}, "base_version": 1},
        )
    )
    stale = a.put(
        f"{API}/sync/projects/{pa['id']}/entities/Pipeline/pip_01TEST",
        json={"data": {**pipeline, "name": "v3"}, "base_version": 1},
    )
    assert stale.status_code == 409 and stale.json()["details"]["server_version"] == 2
    # Ids que no son de Perceptron no llegan a rutas ni al repositorio.
    for bad_id in ("..", "pip_01..x", "C:evil"):
        bad = a.put(
            f"{API}/sync/projects/{pa['id']}/entities/Pipeline/{bad_id}",
            json={"data": pipeline, "base_version": None},
        )
        assert bad.status_code in (404, 422), bad_id
    bad_project = a.put(f"{API}/sync/projects/..%5Cotro", json={"project": {"name": "x"}})
    assert bad_project.status_code in (404, 422)
    # Rutas fuera de lo sincronizable.
    for path in (
        "../secreto",
        "runs/x",
        "/etc/passwd",
        "datasets/../../x",
        "datasets/C:evil",  # letra de unidad: en Windows escaparía de la carpeta del proyecto
        "datasets/con.txt",
        "datasets/a.csv:ads",
    ):
        bad = a.post(
            f"{API}/sync/projects/{pa['id']}/uploads",
            json={"path": path, "size": 1, "sha256": "0" * 64},
        )
        assert bad.status_code == 422, path
    # Un Viewer no sube nada.
    viewer = {"email": "v@preteco.test", "password": PASSWORD, "workspace_id": ws, "role": "viewer"}
    ok(admin.post(f"{API}/admin/users", json=viewer), 201)
    v = login(app, "v@preteco.test", PASSWORD)
    denied = v.post(
        f"{API}/sync/projects/{pa['id']}/uploads",
        json={"path": "datasets/x.csv", "size": 1, "sha256": "0" * 64},
    )
    assert denied.status_code == 403


def test_resumable_chunked_upload(team_server: LiveServer) -> None:
    import hashlib

    app = team_server.app  # type: ignore[attr-defined]
    admin = login(app, ADMIN_EMAIL, ADMIN_PASSWORD)
    pid = ok(admin.post(f"{API}/projects", json={"name": "Subidas"}), 201)["id"]
    payload = b"fecha,valor\n2026-01-01,1\n2026-01-02,2\n"
    meta = {
        "path": "datasets/dsv_x/data ñ.csv",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    start = ok(admin.post(f"{API}/sync/projects/{pid}/uploads", json=meta))
    assert start["offset"] == 0 and not start["complete"]
    uid = start["upload_id"]
    ok(admin.put(f"{API}/sync/uploads/{uid}", params={"offset": 0}, content=payload[:10]))
    # Se cortó la conexión: retomar informa lo recibido y no acepta offsets viejos.
    again = ok(admin.post(f"{API}/sync/projects/{pid}/uploads", json=meta))
    assert again["upload_id"] == uid and again["offset"] == 10
    stale = admin.put(f"{API}/sync/uploads/{uid}", params={"offset": 0}, content=payload[:10])
    assert stale.status_code == 409 and stale.json()["details"]["offset"] == 10
    ok(admin.put(f"{API}/sync/uploads/{uid}", params={"offset": 10}, content=payload[10:]))
    done = ok(admin.post(f"{API}/sync/uploads/{uid}/complete"))
    assert done["complete"] and done["path"] == meta["path"]
    # Ya está igual en el servidor: no se vuelve a subir.
    assert ok(admin.post(f"{API}/sync/projects/{pid}/uploads", json=meta))["complete"]
    files = ok(admin.get(f"{API}/sync/projects/{pid}/files", params={"prefix": "datasets"}))
    assert files == [{"path": meta["path"], "size": len(payload)}]
    got = admin.get(f"{API}/sync/projects/{pid}/file", params={"path": meta["path"]})
    assert got.content == payload
    # Otro usuario no puede continuar la subida de alguien más.
    other = {"email": "o@preteco.test", "password": PASSWORD}
    ws = ok(admin.get(f"{API}/auth/me"))["workspaces"][0]["id"]
    ok(admin.post(f"{API}/admin/users", json={**other, "workspace_id": ws, "role": "editor"}), 201)
    meta2 = {**meta, "path": "datasets/dsv_x/otro.csv"}
    uid2 = ok(admin.post(f"{API}/sync/projects/{pid}/uploads", json=meta2))["upload_id"]
    intruder = login(app, other["email"], other["password"])
    r = intruder.put(f"{API}/sync/uploads/{uid2}", params={"offset": 0}, content=b"x")
    assert r.status_code == 403
