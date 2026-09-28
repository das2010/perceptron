"""Historial de actividad del proyecto (RF-PRJ-05)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from perceptron.api.activity import LOCAL_USER

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def test_activity_records_writes_with_actor_and_skips_read_only_ops(
    client: TestClient, fixtures_dir: Path
) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Historial"}), 201)["id"]
    data = (fixtures_dir / "uc01_churn" / "churn.csv").read_bytes()
    files = [("files", ("churn.csv", data, "text/csv"))]
    src = _ok(client.post(f"{API}/projects/{pid}/uploads", files=files), 201)
    _ok(client.post(f"{API}/sources/{src['id']}/preview"))  # no cambia nada: no se anota
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    project = _ok(client.get(f"{API}/projects/{pid}"))
    _ok(client.patch(f"{API}/projects/{pid}", json={"version": project["version"], "goal": "x"}))
    stale = client.patch(f"{API}/projects/{pid}", json={"version": 1, "goal": "viejo"})
    assert stale.status_code == 409  # los fallos no se anotan

    entries = _ok(client.get(f"{API}/projects/{pid}/activity"))
    ops = [e["operation"] for e in entries]
    assert ops == ["updateProject", "ingestSource", "uploadSource", "createProject"]  # reciente 1.º
    assert all(e["actor"] == LOCAL_USER and e["status"] < 300 for e in entries)
    assert entries[1]["path"] == f"{API}/sources/{src['id']}/ingest"
    assert dv["id"]
    copy = _ok(client.post(f"{API}/projects/{pid}/duplicate", json={}), 201)
    assert [e["operation"] for e in _ok(client.get(f"{API}/projects/{copy['id']}/activity"))] == [
        "duplicateProject"
    ]
