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


def _rebuild(package: bytes, mutate: Any) -> bytes:
    """Mismo paquete con el manifiesto modificado (simula un .perceptron adulterado)."""
    src, out = zipfile.ZipFile(io.BytesIO(package)), io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "manifest.json":
                manifest = json.loads(data)
                mutate(manifest)
                data = json.dumps(manifest).encode("utf-8")
            z.writestr(info.filename, data)
    return out.getvalue()


def _import(client: TestClient, package: bytes) -> Any:
    files = [("file", ("p.perceptron", package, "application/zip"))]
    return client.post(f"{API}/projects/import", files=files)


def test_import_never_brings_identity_or_permissions(
    client: TestClient, ctx: EngineContext, fixtures_dir: Path
) -> None:
    """Un paquete con usuarios, workspaces o membresías no crea permisos (hallazgo crítico)."""
    from perceptron.domain.models import Membership, User, Workspace

    other = _ok(client.post(f"{API}/projects", json={"name": "Ajeno"}), 201)["id"]
    pid, _, _ = _project(client, fixtures_dir)
    package = client.get(f"{API}/projects/{pid}/package").content
    assert client.delete(f"{API}/projects/{pid}").status_code == 204

    def inject(m: dict[str, Any]) -> None:
        m["entities"]["User"] = [{"id": "usr_01INTRUSO", "email": "x@y.z", "name": "x"}]
        m["entities"]["Workspace"] = [{"id": "wsp_01INTRUSO", "name": "x"}]
        m["entities"]["Membership"] = [
            {
                "id": "mbr_01INTRUSO",
                "user_id": "usr_01INTRUSO",
                "workspace_id": "wsp_01INTRUSO",
                "project_id": other,
                "role": "admin",
            },
        ]

    _ok(_import(client, _rebuild(package, inject)), 201)
    assert ctx.repo(Membership).find("mbr_01INTRUSO") is None
    assert ctx.repo(User).find("usr_01INTRUSO") is None
    assert ctx.repo(Workspace).find("wsp_01INTRUSO") is None


def test_import_rejects_entities_of_other_projects_atomically(
    client: TestClient, ctx: EngineContext, fixtures_dir: Path
) -> None:
    from perceptron.domain.models import Pipeline

    other_pid, _, other_dv = _project(client, fixtures_dir)
    pid, _, _ = _project(client, fixtures_dir)
    package = client.get(f"{API}/projects/{pid}/package").content
    assert client.delete(f"{API}/projects/{pid}").status_code == 204

    # 1) Una entidad con el project_id de otro proyecto.
    def foreign_owner(m: dict[str, Any]) -> None:
        m["entities"]["Pipeline"] = [
            {"id": "pip_01AJENO", "project_id": other_pid, "name": "x", "graph": {}}
        ]

    assert _import(client, _rebuild(package, foreign_owner)).status_code == 422

    # 2) Una referencia a la versión de datos de otro proyecto (leería sus datos).
    def foreign_ref(m: dict[str, Any]) -> None:
        for dv in m["entities"]["DatasetVersion"]:
            dv["parent_id"] = other_dv

    r = _import(client, _rebuild(package, foreign_ref))
    assert r.status_code == 422 and r.json()["details"]["id"] == other_dv

    # 3) Un id que ya existe en el servidor: 409 y no queda nada a medias.
    def clash(m: dict[str, Any]) -> None:
        m["entities"]["Pipeline"] = [{"id": other_dv, "project_id": pid, "name": "x", "graph": {}}]

    assert _import(client, _rebuild(package, clash)).status_code in (409, 422)
    assert ctx.projects.find(pid) is None
    assert not ctx.repo(Pipeline).list(filters={"project_id": pid})
    # El paquete original sigue importándose bien.
    _ok(_import(client, package), 201)


def test_labelsets_travel_with_the_project(client: TestClient, fixtures_dir: Path) -> None:
    pid, _, dv_id = _project(client, fixtures_dir)
    ls = _ok(
        client.post(
            f"{API}/datasets/{dv_id}/labelsets", json={"kind": "class", "classes": ["0", "1"]}
        ),
        201,
    )
    package = client.get(f"{API}/projects/{pid}/package").content
    assert client.delete(f"{API}/projects/{pid}").status_code == 204
    assert client.get(f"{API}/labelsets/{ls['id']}").status_code == 404  # se borró con el proyecto
    _ok(_import(client, package), 201)
    assert _ok(client.get(f"{API}/labelsets/{ls['id']}"))["labelset"]["id"] == ls["id"]
