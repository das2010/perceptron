"""Migraciones, UI web servida por el servidor y «fuentes del servidor» (RF-SRV-05)."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, inspect
from srv_helpers import ADMIN_EMAIL, ADMIN_PASSWORD, API, UserFactory, login, ok

from perceptron.core.config import Settings
from perceptron_server.app import create_server_app
from perceptron_server.migrate import current_revision, downgrade, target_metadata, upgrade
from perceptron_server.settings import ServerSettings


def test_migrations_match_models_and_roundtrip(db_url: str) -> None:
    upgrade(db_url)
    assert current_revision(db_url) == "0001"
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), target_metadata())
        assert diff == [], f"el modelo difiere de las migraciones: {diff}"
        downgrade(db_url, "base")
        assert not {"entities", "server_accounts"} & set(inspect(engine).get_table_names())
        upgrade(db_url)
        assert {"entities", "server_audit"} <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_bootstrap_is_idempotent(settings: Settings, server_settings: ServerSettings) -> None:
    for _ in range(2):  # dos arranques sobre la misma base
        app = create_server_app(settings, server_settings)
        with TestClient(app):
            c = login(app, ADMIN_EMAIL, ADMIN_PASSWORD)
            users = ok(c.get(f"{API}/admin/users"))
            assert [u["user"]["email"] for u in users] == [ADMIN_EMAIL]
            assert len(ok(c.get(f"{API}/admin/workspaces"))) == 1


def test_weak_secret_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="32"):
        ServerSettings(secret_key=SecretStr("corta"))


def test_spa_served_with_client_routes_and_headers(
    tmp_path: Path, settings: Settings, server_settings: ServerSettings
) -> None:
    spa = tmp_path / "ui dist"
    (spa / "assets").mkdir(parents=True)
    (spa / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (spa / "assets" / "app-123.js").write_text("console.log(1)", encoding="utf-8")
    app = create_server_app(settings, server_settings.model_copy(update={"spa_dir": spa}))
    with TestClient(app) as c:
        index = c.get("/")
        assert index.status_code == 200 and "id=root" in index.text
        assert "frame-ancestors 'none'" in index.headers["content-security-policy"]
        assert index.headers["cache-control"] == "no-cache"
        deep = c.get("/projects/prj_1/data")  # ruta del cliente
        assert deep.status_code == 200 and "id=root" in deep.text
        asset = c.get("/assets/app-123.js")
        assert "immutable" in asset.headers["cache-control"]
        missing = c.get(f"{API}/no-existe")
        assert missing.status_code in (401, 404) and "id=root" not in missing.text
        assert c.get(f"{API}/system/health").headers["x-content-type-options"] == "nosniff"


def test_server_sources_confine_paths(
    make_user: UserFactory, source_root: Path, fixtures_dir: Path, tmp_path: Path
) -> None:
    (source_root / "ventas 2026").mkdir()
    csv = source_root / "ventas 2026" / "demanda ñ.csv"
    csv.write_bytes((fixtures_dir / "uc07_demand" / "demanda.csv").read_bytes())
    (source_root / ".oculto").write_text("x", encoding="utf-8")
    secret = tmp_path / "fuera.csv"
    secret.write_text("a,b\n1,2\n", encoding="utf-8")

    _, editor = make_user("fuentes@preteco.test", "editor")
    _, viewer = make_user("fuentes-v@preteco.test", "viewer")
    roots = ok(editor.get(f"{API}/server/sources"))
    assert [r["path"] for r in roots] == [str(source_root), str(fixtures_dir)]
    assert viewer.get(f"{API}/server/sources").status_code == 403

    top = ok(editor.get(f"{API}/server/sources/0/browse"))
    assert [e["name"] for e in top["entries"]] == ["ventas 2026"]  # sin ocultos
    inner = ok(editor.get(f"{API}/server/sources/0/browse", params={"path": "ventas 2026"}))
    assert inner["entries"][0]["kind"] == "file" and inner["path"] == "ventas 2026"
    escape = editor.get(f"{API}/server/sources/0/browse", params={"path": "../"})
    assert escape.status_code == 403

    pid = ok(editor.post(f"{API}/projects", json={"name": "Demanda"}), 201)["id"]
    src = editor.post(f"{API}/projects/{pid}/sources", json={"path": inner["entries"][0]["path"]})
    assert src.status_code == 201, src.text
    outside = editor.post(f"{API}/projects/{pid}/sources", json={"path": str(secret)})
    assert outside.status_code == 403 and outside.json()["code"] == "forbidden"
    traversal = editor.post(
        f"{API}/projects/{pid}/sources",
        json={"path": str(source_root / ".." / "fuera.csv")},
    )
    assert traversal.status_code == 403
    sqlite = editor.post(
        f"{API}/projects/{pid}/sources/db",
        json={
            "name": "x",
            "config": {"dialect": "sqlite", "database": str(secret), "query": "SELECT 1"},
        },
    )
    assert sqlite.status_code == 403


def test_hardened_defaults_without_explicit_configuration(
    settings: Settings, server_settings: ServerSettings, fixtures_dir: Path
) -> None:
    """Capa 7: sin Swagger público, sin rutas del servidor como fuentes si el admin no las
    habilitó, y cabeceras de seguridad también en la API."""
    bare = settings.model_copy(update={"source_roots": None})
    app = create_server_app(bare, server_settings)
    with TestClient(app) as c:
        assert c.get(f"{API}/docs").status_code != 200
        assert c.get(f"{API}/openapi.json").status_code != 200
        admin = login(app, ADMIN_EMAIL, ADMIN_PASSWORD)
        health = admin.get(f"{API}/system/health")
        assert health.headers["cache-control"] == "no-store"
        assert health.headers["x-frame-options"] == "DENY"
        pid = ok(admin.post(f"{API}/projects", json={"name": "Raíces"}), 201)["id"]
        csv = next(fixtures_dir.rglob("*.csv"))
        denied = admin.post(f"{API}/projects/{pid}/sources", json={"path": str(csv)})
        assert denied.status_code == 403
        # Las subidas (carpeta del proyecto) siguen funcionando.
        up = admin.post(
            f"{API}/projects/{pid}/uploads",
            files=[("files", ("datos.csv", b"a,b\n1,2\n", "text/csv"))],
        )
        src = ok(up, 201)
        assert ok(admin.post(f"{API}/sources/{src['id']}/preview"))["columns"] == ["a", "b"]
    assert app.openapi()["paths"]  # el contrato se sigue generando desde el código
