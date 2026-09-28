"""Fixtures del Team Server. Base: SQLite en tmp; en CI también PostgreSQL
(`PERCEPTRON_TEST_DATABASE_URL`, esquema limpio por test)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from srv_helpers import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    API,
    AWKWARD_DIR,
    FIXTURES_DIR,
    PASSWORD,
    UserFactory,
    login,
    ok,
)

from perceptron.core.config import LoggingSettings, Settings
from perceptron.storage.db import sqlite_url
from perceptron_server.app import create_server_app
from perceptron_server.settings import ServerSettings


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_LLM__ENABLED", "false")
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    url = os.environ.get("PERCEPTRON_TEST_DATABASE_URL")
    if not url:
        path = tmp_path / AWKWARD_DIR / "server.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite_url(path)
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    return url


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "datos del servidor ñ"
    root.mkdir()
    return root


@pytest.fixture
def settings(tmp_path: Path, db_url: str, source_root: Path, fixtures_dir: Path) -> Settings:
    return Settings(
        workspace_dir=tmp_path / AWKWARD_DIR / "workspace",
        logging=LoggingSettings(to_file=False),
        database_url=SecretStr(db_url),
        source_roots=[source_root, fixtures_dir],
    )


@pytest.fixture
def server_settings() -> ServerSettings:
    return ServerSettings(
        secret_key=SecretStr("k" * 48),
        cookie_secure=False,  # TestClient habla HTTP
        bootstrap_admin_email=ADMIN_EMAIL,
        bootstrap_admin_password=SecretStr(ADMIN_PASSWORD),
        auth_rate_per_minute=10_000,
    )


@pytest.fixture
def app(settings: Settings, server_settings: ServerSettings) -> FastAPI:
    return create_server_app(settings, server_settings)


@pytest.fixture
def anon(app: FastAPI) -> Iterator[TestClient]:
    """Cliente sin sesión; su `with` levanta el servidor (migraciones + bootstrap)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin(app: FastAPI, anon: TestClient) -> TestClient:
    return login(app, ADMIN_EMAIL, ADMIN_PASSWORD)


@pytest.fixture
def default_workspace(admin: TestClient) -> str:
    me = ok(admin.get(f"{API}/auth/me"))
    return str(me["workspaces"][0]["id"])


@pytest.fixture
def make_user(app: FastAPI, admin: TestClient, default_workspace: str) -> UserFactory:
    """Crea un usuario (opcionalmente con rol en el workspace por defecto) y lo loguea."""

    def make(email: str, role: str | None = None) -> tuple[str, TestClient]:
        body: dict[str, Any] = {"email": email, "display_name": email.split("@", maxsplit=1)[0]}
        body["password"] = PASSWORD
        if role:
            body |= {"workspace_id": default_workspace, "role": role}
        user = ok(admin.post(f"{API}/admin/users", json=body), 201)
        return user["user"]["id"], login(app, email, PASSWORD)

    return make
