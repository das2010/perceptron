"""Fórmula sugerida desde la API (ADR-0039): job, listado y predicción."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl
from fastapi.testclient import TestClient

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _wait(client: TestClient, job_id: str, timeout: float = 300) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job: dict[str, Any] = _ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError("el job no terminó")


def _dataset(client: TestClient, path: Path, target: str) -> tuple[str, str]:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Fórmulas"}), 201)["id"]
    src = _ok(client.post(f"{API}/projects/{pid}/sources", json={"path": str(path)}), 201)
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": target}), 201)
    return pid, dv["id"]


def test_formula_for_tabla_3_extrapolates(client: TestClient, tmp_path: Path) -> None:
    """Caso «Tabla 3»: `multiplo = 3 × numero`, y 500 → 1500 (fuera del rango 1–397)."""
    path = tmp_path / "tabla 3.csv"
    pl.DataFrame({"numero": range(1, 398), "multiplo": [3 * i for i in range(1, 398)]}).write_csv(
        path, separator=";"
    )
    pid, dv = _dataset(client, path, "multiplo")
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/symbolic", json={"dataset_version_id": dv, "time_limit_s": 9}
        ),
        202,
    )
    job = _wait(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    [fit] = _ok(client.get(f"{API}/projects/{pid}/symbolic"))
    assert fit["id"] == job["result"]["symbolic_fit_id"]
    assert fit["formula"] == "3·numero" and fit["features"] == ["numero"]
    assert fit["metrics"]["test"]["r2"] > 0.999999 and fit["excel_es"] == "=(3*A2)"
    out = _ok(client.post(f"{API}/symbolic/{fit['id']}/predict", json={"rows": [{"numero": 500}]}))
    assert out["predictions"] == [1500.0]
    missing = client.post(f"{API}/symbolic/{fit['id']}/predict", json={"rows": [{"otro": 1}]})
    assert missing.status_code == 422


def test_classification_datasets_are_rejected_before_queuing(
    client: TestClient, fixtures_dir: Path
) -> None:
    pid, dv = _dataset(client, fixtures_dir / "uc01_churn" / "churn.csv", "churn")
    r = client.post(f"{API}/projects/{pid}/symbolic", json={"dataset_version_id": dv})
    assert r.status_code == 422 and r.json()["details"]["reason"] == "not_regression"
