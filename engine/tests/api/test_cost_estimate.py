"""Estimación de complejidad y costo por dispositivo (RF-PRF-08)."""

from __future__ import annotations

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


def test_estimate_reports_effective_size_memory_and_time_per_device(
    client: TestClient, fixtures_dir: Path
) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Costos"}), 201)["id"]
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
    )["proposals"][0]
    est = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/estimate",
            json={"archspec_id": prop["archspec"]["id"], "dataset_version_id": dv["id"]},
        )
    )
    assert 0 < est["n_train"] < est["num_samples"]
    assert est["size_bytes"] > 0 and est["batch_size"] >= 2
    assert est["num_params"] > 0 and est["memory_mb"] > 0
    cpu = next(d for d in est["devices"] if d["device"] == "cpu")
    assert cpu["epoch_time_s"] is not None and cpu["epoch_time_s"] > 0
    assert cpu["fits"] is True
