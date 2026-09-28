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


def wait_job(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        job: dict[str, Any] = ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.3)
    raise AssertionError(f"el job {job_id} no terminó a tiempo")


def prepare_study(
    editor: TestClient, fixtures_dir: Path, *, trials: int = 1, epochs: int = 2
) -> tuple[str, dict[str, Any]]:
    """Proyecto UC-01 con datos, pipeline y arquitectura; devuelve (project_id, cuerpo)."""
    pid: str = ok(editor.post(f"{API}/projects", json={"name": "Cola"}), 201)["id"]
    src = ok(
        editor.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = ok(editor.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    pipe = ok(
        editor.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    prop = ok(
        editor.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]
    body = {
        "dataset_version_id": dv["id"],
        "pipeline_id": pipe["id"],
        "archspec_id": prop["archspec"]["id"],
        "budget": {"max_trials": trials, "max_epochs_per_trial": epochs},
    }
    return pid, body
