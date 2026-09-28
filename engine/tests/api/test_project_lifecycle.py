"""Duplicar, archivar y eliminar proyectos (RF-PRJ-01)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import DatasetVersion, Profile
from perceptron.storage.db import EntityRow

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _project_with_data(client: TestClient, fixtures_dir: Path, name: str) -> tuple[str, str]:
    body = {"name": name, "goal": "bajas", "template": "churn", "privacy_level": "L2"}
    pid = _ok(client.post(f"{API}/projects", json=body), 201)["id"]
    data = (fixtures_dir / "uc01_churn" / "churn.csv").read_bytes()
    files = [("files", ("churn.csv", data, "text/csv"))]
    src = _ok(client.post(f"{API}/projects/{pid}/uploads", files=files), 201)
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    _ok(client.post(f"{API}/datasets/{dv['id']}/profile"))
    return pid, dv["id"]


def test_duplicate_copies_configuration_but_not_data(
    client: TestClient, fixtures_dir: Path
) -> None:
    pid, _ = _project_with_data(client, fixtures_dir, "Churn Q3")
    copy = _ok(client.post(f"{API}/projects/{pid}/duplicate", json={}), 201)
    assert copy["id"] != pid and copy["name"] == "Churn Q3 (copia)"
    for field in ("goal", "template", "task", "target_metric", "modalities", "privacy_level"):
        assert copy[field] == _ok(client.get(f"{API}/projects/{pid}"))[field], field
    assert copy["status"] == "draft"
    assert _ok(client.get(f"{API}/projects/{copy['id']}/datasets")) == []
    named = _ok(client.post(f"{API}/projects/{pid}/duplicate", json={"name": "Churn Q4"}), 201)
    assert named["name"] == "Churn Q4"


def test_archive_keeps_everything_and_restore(client: TestClient, fixtures_dir: Path) -> None:
    pid, dv_id = _project_with_data(client, fixtures_dir, "Para archivar")
    project = _ok(client.get(f"{API}/projects/{pid}"))
    archived = _ok(
        client.patch(
            f"{API}/projects/{pid}", json={"version": project["version"], "status": "archived"}
        )
    )
    assert archived["status"] == "archived"
    assert _ok(client.get(f"{API}/datasets/{dv_id}"))["id"] == dv_id
    restored = _ok(
        client.patch(
            f"{API}/projects/{pid}", json={"version": archived["version"], "status": "active"}
        )
    )
    assert restored["status"] == "active"


def test_delete_removes_entities_and_folder(
    client: TestClient, ctx: EngineContext, fixtures_dir: Path
) -> None:
    pid, dv_id = _project_with_data(client, fixtures_dir, "Para borrar")
    keep, keep_dv = _project_with_data(client, fixtures_dir, "Se queda")
    folder = ctx.settings.paths.project(pid).root
    assert folder.is_dir()
    assert ctx.repo(Profile).list(filters={}, limit=10)  # el ingest perfiló los datos
    r = client.delete(f"{API}/projects/{pid}")
    assert r.status_code == 204
    assert client.get(f"{API}/projects/{pid}").status_code == 404
    assert client.get(f"{API}/datasets/{dv_id}").status_code == 404
    assert not folder.exists()
    with ctx.db.session() as s:
        left = [row.data for row in s.query(EntityRow).filter(EntityRow.kind == "Profile")]
    assert all(p["dataset_version_id"] != dv_id for p in left)
    # El otro proyecto no se toca.
    assert ctx.repo(DatasetVersion).get(keep_dv).project_id == keep
    assert any(p["dataset_version_id"] == keep_dv for p in left)
