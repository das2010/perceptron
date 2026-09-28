"""Modo experto por la API (RF-ARC-06, ADR-0025): código → sandbox → entrenar → evaluar."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

API = "/api/v1"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _wait_job(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("el job no terminó a tiempo")


def test_expert_code_trains_in_sandbox(client: TestClient, fixtures_dir: Path) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Experto"}), 201)["id"]
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    prop = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )
    base_id = prop["proposals"][0]["archspec"]["id"]

    starter = _ok(client.get(f"{API}/archspecs/{base_id}/code/starter"))["code"]
    assert _ok(client.post(f"{API}/arch/code/lint", json={"source": starter}))["valid"]
    lint = _ok(client.post(f"{API}/arch/code/lint", json={"source": "import os\n"}))
    assert not lint["valid"]

    body = {"base_archspec_id": base_id, "source": starter, "name": "mlp-a-mano"}
    # Sin confirmación explícita, no.
    assert client.post(f"{API}/projects/{pid}/archspecs/code", json=body).status_code == 422
    broken = starter.replace("return Model(", "raise ValueError('roto') or Model(")
    r = client.post(
        f"{API}/projects/{pid}/archspecs/code",
        json={**body, "source": broken, "acknowledge_risk": True},
    )
    assert r.status_code == 422 and "roto" in r.text

    created = _ok(
        client.post(
            f"{API}/projects/{pid}/archspecs/code", json={**body, "acknowledge_risk": True}
        ),
        201,
    )
    record, check = created["record"], created["check"]
    assert check["ok"] and check["num_params"] > 0 and len(check["output_shape"]) == 1
    assert record["code_path"] and record["origin"] == "manual"
    assert _ok(client.get(f"{API}/archspecs/{record['id']}/code"))["code"] == starter
    # Una ArchSpec de código no entra por el editor manual (sin sandbox ni confirmación).
    manual = client.post(f"{API}/projects/{pid}/archspecs", json={"spec": record["spec"]})
    assert manual.status_code == 422
    # "Ver como código" de una ArchSpec de código: su fuente está en /code.
    assert client.post(f"{API}/arch/to-code", json=record["spec"]).status_code == 422

    budget = {"max_trials": 1, "max_epochs_per_trial": 2}
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": record["id"],
                "budget": budget,
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id = job["result"]["best_trial"]["run_id"]
    run = _ok(client.get(f"{API}/runs/{run_id}"))
    assert run["status"] == "succeeded", run

    report = _ok(client.post(f"{API}/runs/{run_id}/evaluate"))
    assert report["split"] == "test" and "accuracy" in report["metrics"]
    mv = _ok(client.post(f"{API}/runs/{run_id}/register"), 201)
    assert mv["model_card"]["declarative"] is False
