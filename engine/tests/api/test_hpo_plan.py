"""Plan de intentos y épocas del HPO por API (ADR-0041)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def test_plan_proposes_trials_and_epochs_for_the_time_available(
    client: TestClient, fixtures_dir: Path
) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "plan"}), 201)["id"]
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
    arch = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"], "mode": "rules"},
        ),
        201,
    )["proposals"][0]["archspec"]
    body = {"archspec_id": arch["id"], "dataset_version_id": dv["id"]}
    plan = _ok(client.post(f"{API}/projects/{pid}/hpo/plan", json=body))
    assert plan["max_trials"] > 1 and plan["max_epochs_per_trial"] >= 1
    assert plan["tuned_params"] and plan["reasons"] and plan["epoch_time_s"]
    assert plan["time_budget_s"] == 20 * 60
    # Con muy poco tiempo, el plan se achica para entrar (o lo avisa).
    tight = _ok(client.post(f"{API}/projects/{pid}/hpo/plan", json={**body, "time_budget_s": 5}))
    assert tight["max_trials"] <= plan["max_trials"]
    assert tight["max_epochs_per_trial"] <= plan["max_epochs_per_trial"]
    assert tight["estimated_s"] <= 5 or any("más largo" in r for r in tight["reasons"])
