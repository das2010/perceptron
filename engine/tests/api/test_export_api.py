"""Export verificado del modelo (RF-EXP-01, RF-EXP-05; ADR-0027) por la API."""

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


def _trained_run(client: TestClient, fixtures_dir: Path) -> str:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Export"}), 201)["id"]
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
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": prop["archspec"]["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": 2},
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id: str = job["result"]["best_trial"]["run_id"]
    return run_id


def test_export_all_formats_verified(client: TestClient, fixtures_dir: Path) -> None:
    run_id = _trained_run(client, fixtures_dir)
    assert client.get(f"{API}/runs/{run_id}/export").status_code == 404

    body = {"formats": ["onnx", "torch_export", "torchscript"], "fp16": True, "int8": True}
    launch = _ok(client.post(f"{API}/runs/{run_id}/export", json=body), 202)
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]

    report = _ok(client.get(f"{API}/runs/{run_id}/export"))
    by_format = {a["format"]: a for a in report["artifacts"]}
    assert set(by_format) >= {"onnx", "onnx_fp16", "onnx_int8", "torch_export", "torchscript"}
    for fmt in ("onnx", "onnx_fp16", "torch_export", "torchscript"):
        art = by_format[fmt]
        assert art["error"] is None, (fmt, art["error"])
        assert art["verification"]["passed"], (fmt, art["verification"])
    # ONNX coincide con PyTorch (aceptación §14): 1e-4 en fp32 y 1e-2 en fp16.
    assert by_format["onnx"]["verification"]["tolerance"] == 1e-4
    assert by_format["onnx_fp16"]["verification"]["tolerance"] == 1e-2
    assert by_format["onnx_int8"]["verification"]["tolerance"] is None  # solo se informa
    assert by_format["torchscript"]["legacy"] is True
    assert [i["name"] for i in report["inputs"]] == ["x_num", "x_cat"]
    assert report["signature"]["inputs"]["kind"] == "tabular"
    assert report["signature"]["run_id"] == run_id and report["signature"]["perceptron_version"]

    onnx = client.get(f"{API}/runs/{run_id}/export/files/model.onnx")
    assert onnx.status_code == 200 and len(onnx.content) == by_format["onnx"]["size_bytes"]
    assert _ok(client.get(f"{API}/runs/{run_id}/export/files/signature.json"))["inputs"]
    # Solo se sirven los archivos del reporte: nada de rutas arbitrarias del run.
    bad = client.get(f"{API}/runs/{run_id}/export/files/run.json")
    assert bad.status_code == 404
    assert client.get(f"{API}/runs/{run_id}/export/files/..%2Frun.json").status_code == 404


def test_export_unknown_run(client: TestClient) -> None:
    assert client.post(f"{API}/runs/run_nope/export", json={}).status_code == 404
