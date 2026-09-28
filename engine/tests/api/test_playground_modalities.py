"""Playground de texto y audio sobre el modelo exportado a ONNX (RF-EXP-02)."""

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


def _wait_job(client: TestClient, job_id: str, timeout: float = 900) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("el job no terminó a tiempo")


def _exported_run(client: TestClient, source: Path, target: str | None) -> str:
    pid = _ok(client.post(f"{API}/projects", json={"name": source.name}), 201)["id"]
    src = _ok(client.post(f"{API}/projects/{pid}/sources", json={"path": str(source)}), 201)
    body = {"target": target} if target else {}
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json=body), 201)
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
    export = _ok(client.post(f"{API}/runs/{run_id}/export", json={"formats": ["onnx"]}), 202)
    done = _wait_job(client, export["job"]["id"])
    assert done["status"] == "succeeded", done["error"]
    return run_id


def test_text_playground(client: TestClient, fixtures_dir: Path) -> None:
    run_id = _exported_run(client, fixtures_dir / "uc03_tickets_es" / "tickets.jsonl", "categoria")
    texts = ["No puedo entrar a mi cuenta, la contraseña no funciona", "Quiero exportar a Excel"]
    out = _ok(client.post(f"{API}/runs/{run_id}/predict/text", json={"texts": texts}))
    assert out["input_kind"] == "tokens" and len(out["predictions"]) == 2
    first = out["predictions"][0]
    assert 0 <= first["confidence"] <= 1 and first["prediction"] in first["probabilities"]
    lines = "\n".join(texts).encode("utf-8")
    files = [("file", ("tickets.txt", lines, "text/plain"))]
    by_file = _ok(client.post(f"{API}/runs/{run_id}/predict/file", files=files))
    assert [p["prediction"] for p in by_file["predictions"]] == [
        p["prediction"] for p in out["predictions"]
    ]
    rows = client.post(f"{API}/runs/{run_id}/predict", json={"rows": [{"texto": "x"}]})
    assert rows.status_code == 422  # un modelo de texto no recibe filas de tabla
    why = _ok(client.post(f"{API}/runs/{run_id}/explain/text", json={"text": texts[0]}))
    assert why["method"] == "occlusion" and why["prediction"] == first["prediction"]
    assert why["contributions"] and all("attribution" in c for c in why["contributions"])
    # RF-EVL-05: robustez ante typos y palabras eliminadas.
    rob = _ok(client.get(f"{API}/runs/{run_id}/robustness"))
    assert {r["kind"] for r in rob["results"]} == {"typos", "palabras eliminadas"}
    assert rob["samples"] > 0 and len(rob["results"]) == 6


def test_audio_playground(client: TestClient, fixtures_dir: Path) -> None:
    root = fixtures_dir / "uc09_motor_audio"
    run_id = _exported_run(client, root, None)
    clip = next((root / "rodamiento").glob("*.wav"))
    files = [("file", (clip.name, clip.read_bytes(), "audio/wav"))]
    out = _ok(client.post(f"{API}/runs/{run_id}/predict/file", files=files))
    assert out["input_kind"] == "spectrogram"
    pred = out["predictions"][0]
    assert pred["prediction"] is not None and pred.get("probabilities")
    bad = [("file", ("clip.xyz", b"no es audio", "application/octet-stream"))]
    assert client.post(f"{API}/runs/{run_id}/predict/file", files=bad).status_code == 422
    import base64

    why = _ok(client.post(f"{API}/runs/{run_id}/explain/audio", files=files))
    assert why["method"] == "integrated_gradients"
    rob = _ok(client.get(f"{API}/runs/{run_id}/robustness"))
    assert {r["kind"] for r in rob["results"]} == {"ruido de fondo", "volumen bajo"}
    assert base64.b64decode(why["heatmap_png"])[1:4] == b"PNG"
