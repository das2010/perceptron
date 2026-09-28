"""Paquete .perceptron: exportar e importar un proyecto (RF-PRJ-03)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import DataSource

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _project(client: TestClient, fixtures_dir: Path) -> tuple[str, str, str]:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Bajas Q3", "template": "churn"}), 201)[
        "id"
    ]
    data = (fixtures_dir / "uc01_churn" / "churn.csv").read_bytes()
    files = [("files", ("churn.csv", data, "text/csv"))]
    src = _ok(client.post(f"{API}/projects/{pid}/uploads", files=files), 201)
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    _ok(client.post(f"{API}/datasets/{dv['id']}/profile"))
    return pid, src["id"], dv["id"]


def test_roundtrip_with_data_restores_everything(
    client: TestClient, ctx: EngineContext, fixtures_dir: Path
) -> None:
    pid, src_id, dv_id = _project(client, fixtures_dir)
    r = client.get(f"{API}/projects/{pid}/package", params={"include_data": True})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    package = r.content
    with zipfile.ZipFile(io.BytesIO(package)) as z:
        names = z.namelist()
    assert "manifest.json" in names and any(n.startswith("project/datasets/") for n in names)

    assert client.delete(f"{API}/projects/{pid}").status_code == 204
    files = [("file", ("Bajas Q3.perceptron", package, "application/zip"))]
    imported = _ok(client.post(f"{API}/projects/import", files=files), 201)
    assert imported["id"] == pid and imported["template"] == "churn"
    assert _ok(client.get(f"{API}/datasets/{dv_id}"))["id"] == dv_id
    assert _ok(client.get(f"{API}/datasets/{dv_id}/profile"))
    # La fuente subida apunta a la carpeta nueva del proyecto y se puede previsualizar.
    preview = _ok(client.post(f"{API}/sources/{src_id}/preview"))
    assert "churn" in preview["columns"]
    root = ctx.settings.paths.project(pid).root.resolve()
    source = ctx.repo(DataSource)
    assert Path(source.get(src_id).config["path"]).resolve().is_relative_to(root)

    again = client.post(f"{API}/projects/import", files=files)
    assert again.status_code == 409  # ya existe


def test_package_without_data_is_small_and_bad_files_are_rejected(
    client: TestClient, fixtures_dir: Path
) -> None:
    pid, _, _ = _project(client, fixtures_dir)
    r = client.get(f"{API}/projects/{pid}/package")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        manifest = z.read("manifest.json").decode("utf-8")
    assert not any(n.startswith(("project/datasets/", "project/uploads/")) for n in names)
    assert '"include_data": false' in manifest and "DatasetVersion" in manifest
    bad = [("file", ("x.perceptron", b"no es un zip", "application/zip"))]
    assert client.post(f"{API}/projects/import", files=bad).status_code == 422
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("otra_cosa.txt", "hola")
    other = [("file", ("x.perceptron", buf.getvalue(), "application/zip"))]
    assert client.post(f"{API}/projects/import", files=other).status_code == 422


def test_import_rejects_escaping_paths(client: TestClient, ctx: EngineContext) -> None:
    buf = io.BytesIO()
    project = {"id": "prj_01JMALICIOSO0000000000000", "name": "x"}
    manifest = {"format": "perceptron-project", "format_version": 1, "project": project}
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("project/../../fuera.txt", "escape")
    files = [("file", ("x.perceptron", buf.getvalue(), "application/zip"))]
    assert client.post(f"{API}/projects/import", files=files).status_code == 422
    assert ctx.projects.find(project["id"]) is None  # sin importaciones a medias
    assert not (ctx.settings.paths.projects_dir.parent / "fuera.txt").exists()
