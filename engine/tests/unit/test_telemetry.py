"""Telemetría opt-in (D7): nada sale sin consentimiento ni endpoint, y nunca datos."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from perceptron.core.telemetry import Telemetry


def test_nothing_is_sent_without_consent_and_endpoint(tmp_path: Path) -> None:
    sent: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(204)

    client = lambda: httpx.Client(transport=httpx.MockTransport(handler))  # noqa: E731
    off = Telemetry(tmp_path / "t.json", endpoint="https://telemetry.example.test/v1")
    off.count("createProject")
    assert not off.enabled and not off.flush(client())
    no_endpoint = Telemetry(tmp_path / "t2.json", endpoint=None)
    no_endpoint.consent(True)
    assert not no_endpoint.enabled and not no_endpoint.flush(client())

    on = Telemetry(tmp_path / "t3.json", endpoint="https://telemetry.example.test/v1")
    on.consent(True)
    on.count("createProject")
    on.count("createProject")
    on.error("StorageError")
    assert on.flush(client())
    body = sent[0]
    assert set(body) == {
        "installation_id",
        "version",
        "os",
        "os_release",
        "python",
        "cpu_count",
        "ram_gb",
        "features",
        "errors",
    }
    assert body["features"] == {"createProject": 2} and body["errors"] == {"StorageError": 1}
    assert on.payload()["features"] == {}  # se reinicia al enviar
    on.consent(False)
    assert not on.enabled


def test_telemetry_api_shows_preview_and_stores_consent(client: TestClient) -> None:
    client.get("/api/v1/projects")
    status = client.get("/api/v1/system/telemetry").json()
    assert status["asked"] is False and status["opt_in"] is False
    assert status["endpoint_configured"] is False
    assert status["preview"]["features"].get("listProjects", 0) >= 1
    after = client.put("/api/v1/system/telemetry", json={"opt_in": True}).json()
    assert after["asked"] and after["opt_in"]
