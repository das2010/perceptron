from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import SecretStr

from perceptron import __version__
from perceptron.api.app import TOKEN_HEADER, create_app
from perceptron.api.context import EngineContext


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/system/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_version(client: TestClient) -> None:
    body = client.get("/api/v1/system/version").json()
    assert body["version"] == __version__
    assert body["mode"] == "desktop"


def test_project_crud(client: TestClient, ctx: EngineContext) -> None:
    r = client.post(
        "/api/v1/projects",
        json={"name": "Churn clientes", "modalities": ["tabular"], "task": "classification"},
    )
    assert r.status_code == 201, r.text
    project = r.json()
    pid = project["id"]
    assert project["privacy_level"] == "L1"
    assert ctx.settings.paths.project(pid).project_file.is_file()

    assert [p["id"] for p in client.get("/api/v1/projects").json()] == [pid]

    r = client.patch(f"/api/v1/projects/{pid}", json={"version": 1, "goal": "reducir churn"})
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 2
    assert r.json()["goal"] == "reducir churn"

    # versión obsoleta → 409
    r = client.patch(f"/api/v1/projects/{pid}", json={"version": 1, "goal": "x"})
    assert r.status_code == 409
    assert r.json()["code"] == "conflict"

    # valor inválido → 422
    r = client.patch(f"/api/v1/projects/{pid}", json={"version": 2, "privacy_level": "L9"})
    assert r.status_code == 422

    assert client.delete(f"/api/v1/projects/{pid}").status_code == 204
    r = client.get(f"/api/v1/projects/{pid}")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


def test_create_rejects_unknown_fields(client: TestClient) -> None:
    r = client.post("/api/v1/projects", json={"name": "x", "secret": "y"})
    assert r.status_code == 422


def test_token_auth(ctx: EngineContext) -> None:
    ctx.settings.api.token = SecretStr("t0k3n")
    with TestClient(create_app(ctx=ctx)) as c:
        assert c.get("/api/v1/system/health").status_code == 200  # público
        r = c.get("/api/v1/projects")
        assert r.status_code == 401
        assert r.json()["code"] == "unauthorized"
        assert c.get("/api/v1/projects", headers={TOKEN_HEADER: "mal"}).status_code == 401
        assert c.get("/api/v1/projects", headers={TOKEN_HEADER: "t0k3n"}).status_code == 200


def test_openapi_has_stable_operation_ids(client: TestClient) -> None:
    schema = client.get("/api/v1/openapi.json").json()
    op_ids = {op["operationId"] for path in schema["paths"].values() for op in path.values()}
    assert {"getHealth", "getVersion", "listProjects", "createProject"} <= op_ids
