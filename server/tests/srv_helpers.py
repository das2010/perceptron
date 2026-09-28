"""Constantes y helpers de los tests del Team Server (importables desde cada módulo)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

API = "/api/v1"
ADMIN_EMAIL = "admin@preteco.test"
ADMIN_PASSWORD = "clave-admin-muy-segura"
PASSWORD = "clave-de-prueba-123"
# Rutas con espacios y caracteres no ASCII (SPEC §13.5).
AWKWARD_DIR = "Servidor con ñ, acentos (áéí) y espacios"
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


def ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json() if r.content else None


def login(app: FastAPI, email: str, password: str) -> TestClient:
    """Cliente con la sesión del navegador (cookies + cabecera CSRF). Requiere `anon` activo."""
    c = TestClient(app)
    ok(c.post(f"{API}/auth/login", json={"email": email, "password": password}))
    c.headers["X-CSRF-Token"] = c.cookies["pt_csrf"]
    return c


UserFactory = Callable[..., tuple[str, TestClient]]
