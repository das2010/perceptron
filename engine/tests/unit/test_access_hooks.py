"""Ganchos del Engine para el Team Server: raíces de fuentes, filtro por IDs y política abierta."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from perceptron.api.access import OpenAccess
from perceptron.api.context import EngineContext
from perceptron.core.config import Settings
from perceptron.core.paths import within_roots
from perceptron.domain.models import Project


def test_within_roots(tmp_path: Path) -> None:
    root = tmp_path / "datos con ñ"
    (root / "sub").mkdir(parents=True)
    assert within_roots(root / "sub" / "a.csv", None)  # desktop: sin límite
    assert within_roots(root / "sub" / "a.csv", [root])
    assert within_roots(root, [root])
    assert not within_roots(root / ".." / "otro.csv", [root])
    assert not within_roots(tmp_path / "datos con ñ 2" / "x.csv", [root])  # prefijo engañoso


def test_repository_list_by_ids(ctx: EngineContext) -> None:
    a, b, _ = (ctx.projects.add(Project(name=n)) for n in ("a", "b", "c"))
    assert {p.id for p in ctx.projects.list(ids={a.id, b.id})} == {a.id, b.id}
    assert ctx.projects.list(ids=set()) == []


def test_open_access_is_default(client: TestClient, settings: Settings) -> None:
    assert isinstance(client.app.state.access, OpenAccess)  # type: ignore[attr-defined]
    r = client.post("/api/v1/projects", json={"name": "Local", "workspace_id": None})
    assert r.status_code == 201 and r.json()["scope"] == "local"
    assert len(client.get("/api/v1/projects").json()) == 1
    assert settings.source_roots is None and settings.database_url is None
