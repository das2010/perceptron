"""Cabeceras de seguridad, errores sin eco de datos y fuentes con destinos internos (Capa 7)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.core.config import RuntimeMode

API = "/api/v1"


def test_security_headers_and_no_store_on_api(client: TestClient) -> None:
    r = client.get(f"{API}/system/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "same-origin"
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert "camera=()" in r.headers["permissions-policy"]
    assert r.headers["cache-control"] == "no-store"


def test_validation_errors_do_not_echo_input(client: TestClient) -> None:
    secret = "valor-que-no-debe-volver-9f3a"
    r = client.post(f"{API}/projects", json={"name": "x", "clave": secret})
    assert r.status_code == 422
    assert secret not in r.text
    errors = r.json()["detail"]
    assert errors and set(errors[0]) <= {"type", "loc", "msg"}


def test_failed_job_reports_message_without_traceback(ctx: EngineContext) -> None:
    def boom(_: object) -> None:
        raise RuntimeError("se rompió")

    job = ctx.jobs.submit("prueba", boom)
    deadline = time.monotonic() + 10
    while job.finished_at is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert job.status == "failed"
    assert job.error == {"type": "RuntimeError", "message": "se rompió"}


def test_uploaded_file_names_cannot_escape_on_windows(client: TestClient) -> None:
    pid = client.post(f"{API}/projects", json={"name": "subidas"}).json()["id"]
    for name in ("C:evil.csv", "datos/con.csv", "a.csv:flujo"):
        r = client.post(
            f"{API}/projects/{pid}/uploads", files=[("files", (name, b"a,b\n1,2\n", "text/csv"))]
        )
        assert r.status_code == 422, name


def test_stream_sources_to_internal_hosts_blocked_in_server_mode(ctx: EngineContext) -> None:
    from perceptron.api.app import create_app

    ctx.settings.mode = RuntimeMode.SERVER
    with TestClient(create_app(ctx=ctx)) as c:
        pid = c.post(f"{API}/projects", json={"name": "streams"}).json()["id"]
        for kind, url in (
            ("rest", "http://169.254.169.254/latest/meta-data/"),
            ("rest", "http://127.0.0.1:6379/"),
            ("websocket", "ws://10.0.0.1/stream"),
        ):
            r = c.post(
                f"{API}/projects/{pid}/sources/stream",
                json={"name": "s", "kind": kind, "config": {"url": url}},
            )
            assert r.status_code == 422, url
        remote = c.post(
            f"{API}/remote/servers",
            json={"name": "x", "url": "http://127.0.0.1:1", "email": "a@b.c", "password": "p"},
        )
        assert remote.status_code == 403
