"""RBAC y auditoría (RF-SRV-02, RF-SRV-07). Aceptación de la Capa 5: dos usuarios con roles
distintos colaboran en un proyecto del servidor."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from srv_helpers import API, UserFactory, ok
from starlette.websockets import WebSocketDisconnect


def _wait_job(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("el job no terminó a tiempo")


def _dataset(editor: TestClient, fixtures_dir: Path) -> tuple[str, str]:
    pid = ok(editor.post(f"{API}/projects", json={"name": "Churn del equipo"}), 201)["id"]
    src = ok(
        editor.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = ok(editor.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    return pid, dv["id"]


def test_editor_and_viewer_collaborate(
    admin: TestClient, make_user: UserFactory, fixtures_dir: Path
) -> None:
    _, editor = make_user("editora@preteco.test", "editor")
    _, viewer = make_user("visor@preteco.test", "viewer")
    _, outsider = make_user("ajeno@preteco.test")  # sin rol en ningún workspace

    pid, dv = _dataset(editor, fixtures_dir)
    pipe = ok(
        editor.post(f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv}),
        201,
    )
    prop = ok(
        editor.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv, "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]
    study = {
        "dataset_version_id": dv,
        "pipeline_id": pipe["id"],
        "archspec_id": prop["archspec"]["id"],
        "budget": {"max_trials": 1, "max_epochs_per_trial": 2},
    }

    # El Viewer ve el proyecto y sus datos, pero no puede entrenar ni cambiar nada.
    assert [p["id"] for p in ok(viewer.get(f"{API}/projects"))] == [pid]
    assert ok(viewer.get(f"{API}/datasets/{dv}"))["id"] == dv
    denied = viewer.post(f"{API}/projects/{pid}/studies", json=study)
    assert denied.status_code == 403 and denied.json()["code"] == "forbidden"
    assert viewer.patch(f"{API}/projects/{pid}", json={"goal": "otro"}).status_code == 403

    launch = ok(editor.post(f"{API}/projects/{pid}/studies", json=study), 202)
    job = _wait_job(editor, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id = job["result"]["best_trial"]["run_id"]

    # El Viewer sigue el progreso, ve el run y usa el playground (§3.2).
    assert ok(viewer.get(f"{API}/jobs/{launch['job']['id']}"))["status"] == "succeeded"
    assert launch["job"]["id"] in [j["id"] for j in ok(viewer.get(f"{API}/jobs"))]
    assert ok(viewer.get(f"{API}/runs/{run_id}"))["id"] == run_id
    assert viewer.post(f"{API}/runs/{run_id}/export", json={"formats": ["onnx"]}).status_code == 403
    export = ok(editor.post(f"{API}/runs/{run_id}/export", json={"formats": ["onnx"]}), 202)
    assert _wait_job(editor, export["job"]["id"])["status"] == "succeeded"
    assert viewer.get(f"{API}/runs/{run_id}/export/files/model.onnx").status_code == 403
    assert editor.get(f"{API}/runs/{run_id}/export/files/model.onnx").status_code == 200
    rows = [{k: v for k, v in r.items() if k != "churn"} for r in _churn_rows(fixtures_dir, 2)]
    preds = ok(viewer.post(f"{API}/runs/{run_id}/predict", json={"rows": rows}))
    assert len(preds["predictions"]) == 2
    assert viewer.post(f"{API}/runs/{run_id}/evaluate").status_code == 403
    assert viewer.post(f"{API}/runs/{run_id}/register").status_code == 403
    ok(editor.post(f"{API}/runs/{run_id}/evaluate"))

    # Alguien sin rol no ve nada del proyecto.
    assert ok(outsider.get(f"{API}/projects")) == []
    assert outsider.get(f"{API}/projects/{pid}").status_code == 403
    assert outsider.get(f"{API}/runs/{run_id}").status_code == 403
    assert outsider.get(f"{API}/jobs/{launch['job']['id']}").status_code == 403
    assert ok(outsider.get(f"{API}/jobs")) == []
    compare = outsider.post(f"{API}/runs/compare", json={"run_ids": [run_id]})
    assert compare.status_code == 403
    assert outsider.post(f"{API}/projects", json={"name": "x"}).status_code == 403

    # La auditoría registra quién hizo qué, incluidas las denegaciones.
    events = ok(admin.get(f"{API}/admin/audit", params={"project_id": pid, "limit": 500}))
    actions = {(e["action"], e["status"]) for e in events}
    assert ("api.createStudy", 202) in actions
    assert ("api.createStudy", 403) in actions  # el intento del Viewer
    assert ("api.ingestSource", 201) in actions
    assert ("api.downloadRunExport", 200) in actions  # descarga auditada
    assert ("api.downloadRunExport", 403) in actions
    assert all(e["user_id"] for e in events)


def _churn_rows(fixtures_dir: Path, n: int) -> list[dict[str, str]]:
    import csv

    with (fixtures_dir / "uc01_churn" / "churn.csv").open(encoding="utf-8") as f:
        return [r for _, r in zip(range(n), csv.DictReader(f), strict=False)]


def test_project_role_overrides_workspace_role(
    admin: TestClient, make_user: UserFactory, fixtures_dir: Path, default_workspace: str
) -> None:
    _, editor = make_user("edi@preteco.test", "editor")
    viewer_id, viewer = make_user("vio@preteco.test", "viewer")
    pid, dv = _dataset(editor, fixtures_dir)
    other = ok(editor.post(f"{API}/projects", json={"name": "Otro"}), 201)["id"]
    assert viewer.post(f"{API}/datasets/{dv}/profile").status_code == 403

    grant = {
        "user_id": viewer_id,
        "workspace_id": default_workspace,
        "project_id": pid,
        "role": "editor",
    }
    m = ok(admin.post(f"{API}/admin/memberships", json=grant), 201)
    assert viewer.post(f"{API}/datasets/{dv}/profile").status_code in (200, 201, 202)
    # En el resto del workspace sigue siendo Viewer.
    assert viewer.patch(f"{API}/projects/{other}", json={"goal": "x"}).status_code == 403

    ok(admin.delete(f"{API}/admin/memberships/{m['id']}"), 204)
    assert viewer.patch(f"{API}/projects/{pid}", json={"goal": "x"}).status_code == 403
    events = ok(admin.get(f"{API}/admin/audit", params={"action": "admin.role_"}))
    assert {e["action"] for e in events} >= {"admin.role_granted", "admin.role_revoked"}


def test_admin_only_operations(admin: TestClient, make_user: UserFactory) -> None:
    _, editor = make_user("e2@preteco.test", "editor")
    pid = ok(editor.post(f"{API}/projects", json={"name": "Borrable"}), 201)["id"]
    provider = {"kind": "ollama", "base_url": "http://localhost:11434"}
    assert editor.put(f"{API}/llm/providers/local", json=provider).status_code == 403
    assert editor.get(f"{API}/llm/providers").status_code == 200
    assert editor.delete(f"{API}/projects/{pid}").status_code == 403
    assert editor.get(f"{API}/admin/users").status_code == 403
    assert editor.get(f"{API}/admin/audit").status_code == 403
    assert admin.delete(f"{API}/projects/{pid}").status_code in (200, 204)


def test_workspace_admin_manages_only_their_workspace(
    admin: TestClient, make_user: UserFactory, default_workspace: str
) -> None:
    lead_id, lead = make_user("lider@preteco.test", "admin")
    member_id, _ = make_user("miembro@preteco.test")
    other_ws = ok(admin.post(f"{API}/admin/workspaces", json={"name": "Otro equipo"}), 201)["id"]
    grant = {"user_id": member_id, "workspace_id": default_workspace, "role": "editor"}
    assert ok(lead.post(f"{API}/admin/memberships", json=grant), 201)["role"] == "editor"
    listed = ok(lead.get(f"{API}/admin/memberships", params={"workspace_id": default_workspace}))
    assert {m["user_id"] for m in listed} >= {lead_id, member_id}
    elsewhere = {**grant, "workspace_id": other_ws}
    assert lead.post(f"{API}/admin/memberships", json=elsewhere).status_code == 403
    assert lead.get(f"{API}/admin/users").status_code == 200  # directorio para asignar roles
    new_user = {"email": "x@preteco.test", "password": "clave-de-prueba-123"}
    assert lead.post(f"{API}/admin/users", json=new_user).status_code == 403
    assert lead.get(f"{API}/admin/audit").status_code == 403


@pytest.mark.parametrize("who", ["anon", "outsider", "viewer"])
def test_websocket_authorization(
    who: str,
    anon: TestClient,
    admin: TestClient,
    make_user: UserFactory,
    fixtures_dir: Path,
) -> None:
    _, editor = make_user("ws-edit@preteco.test", "editor")
    pid, _ = _dataset(editor, fixtures_dir)
    clients = {
        "anon": anon,
        "outsider": make_user("ws-out@preteco.test")[1],
        "viewer": make_user("ws-view@preteco.test", "viewer")[1],
    }
    client = clients[who]
    path = f"{API}/projects/{pid}/copilot"
    if who == "anon":
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(path):
            pass
        assert exc.value.code == 4401
        return
    # El copiloto escribe el borrador: requiere Editor.
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(path):
        pass
    assert exc.value.code == 4403
    with editor.websocket_connect(path) as ws:
        assert ws is not None


def test_workspace_privacy_policy(
    admin: TestClient, make_user: UserFactory, default_workspace: str
) -> None:
    _, lead = make_user("pol-lead@preteco.test", "admin")
    _, editor = make_user("pol-ed@preteco.test", "editor")
    body = {"max_privacy_level": "L1", "allowed_llm_providers": ["ollama"]}
    assert editor.patch(f"{API}/admin/workspaces/{default_workspace}", json=body).status_code == 403
    ws = ok(lead.patch(f"{API}/admin/workspaces/{default_workspace}", json=body))
    assert ws["max_privacy_level"] == "L1" and ws["allowed_llm_providers"] == ["ollama"]
    cleared = ok(
        admin.patch(
            f"{API}/admin/workspaces/{default_workspace}", json={"clear_allowed_providers": True}
        )
    )
    assert cleared["allowed_llm_providers"] is None and cleared["max_privacy_level"] == "L1"
    events = ok(admin.get(f"{API}/admin/audit", params={"action": "admin.workspace_policy"}))
    assert len(events) == 2


def test_license_usage_and_install_is_admin_only(
    admin: TestClient, make_user: UserFactory, app: FastAPI
) -> None:
    from perceptron.licensing.signed import keygen, sign

    _, editor = make_user("lic-ed@preteco.test", "editor")
    usage = ok(admin.get(f"{API}/admin/license"))
    assert usage["status"] == "missing" and usage["seats_used"] == 2 and usage["over_limit"] == []
    assert editor.get(f"{API}/admin/license").status_code == 403
    private, public = keygen()
    app.state.ctx.settings.license.public_keys = {"k": public}
    doc = sign(
        {
            "id": "l",
            "licensee": "Acme",
            "seats": 1,
            "features": ["*"],
            "issued_at": "2026-01-01T00:00:00+00:00",
        },
        private,
        "k",
    )
    assert editor.put(f"{API}/system/license", json={"content": doc}).status_code == 403
    assert ok(admin.put(f"{API}/system/license", json={"content": doc}))["status"] == "valid"
    usage = ok(admin.get(f"{API}/admin/license"))
    assert usage["licensee"] == "Acme" and usage["over_limit"] == ["seats"]  # 2 usuarios, tope 1
