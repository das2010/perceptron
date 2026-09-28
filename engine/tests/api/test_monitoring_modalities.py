"""Monitoreo de modelos de texto e imagen con embeddings internos (RF-MON-02)."""

from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

API = "/api/v1"
RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


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
        time.sleep(0.3)
    raise AssertionError("el job no terminó a tiempo")


def _deployment(client: TestClient, source: Path, target: str | None, epochs: int) -> str:
    pid = _ok(client.post(f"{API}/projects", json={"name": source.name}), 201)["id"]
    src = _ok(client.post(f"{API}/projects/{pid}/sources", json={"path": str(source)}), 201)
    dv = _ok(
        client.post(f"{API}/sources/{src['id']}/ingest", json={"target": target} if target else {}),
        201,
    )
    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    arch = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]["archspec"]
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": arch["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": epochs},
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id = job["result"]["best_trial"]["run_id"]
    _ok(client.post(f"{API}/runs/{run_id}/evaluate"))
    export = _ok(client.post(f"{API}/runs/{run_id}/export", json={"formats": ["onnx"]}), 202)
    assert _wait_job(client, export["job"]["id"])["status"] == "succeeded"
    mv = _ok(client.post(f"{API}/runs/{run_id}/register"), 201)["id"]
    dep = _ok(
        client.post(
            f"{API}/models/{mv}/deployments",
            json={"name": "prod", "monitoring": {"window": 10_000}},
        ),
        201,
    )
    dep_id: str = dep["id"]
    return dep_id


def test_text_deployment_embedding_drift(client: TestClient, fixtures_dir: Path) -> None:
    path = fixtures_dir / "uc03_tickets_es" / "tickets.jsonl"
    dep = _deployment(client, path, "categoria", epochs=3)
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    base = [{"texto": r["texto"]} for r in rows[:80]]
    out = _ok(client.post(f"{API}/deployments/{dep}/predict", json={"rows": base}))
    assert len(out["predictions"]) == 80
    missing = client.post(f"{API}/deployments/{dep}/predict", json={"rows": [{"otro": "x"}]})
    assert missing.status_code == 422
    stable = _ok(client.post(f"{API}/deployments/{dep}/check", params={"last": 80}))
    emb = stable["metrics"]["embedding"]
    assert emb is not None and emb["n_current"] == 80, stable["metrics"]
    recent = _ok(client.get(f"{API}/deployments/{dep}/predictions", params={"limit": 3}))
    assert recent[0]["embedding"] and "x:texto" in recent[0]

    # Otro dominio: texto en otro idioma y sin el vocabulario de los tickets.
    rng = random.Random(0)  # noqa: S311 - datos de prueba
    words = ["quantum", "harbor", "violin", "glacier", "orbit", "saffron", "tundra", "ledger"]
    odd = [{"texto": " ".join(rng.choice(words) for _ in range(12))} for _ in range(80)]
    _ok(client.post(f"{API}/deployments/{dep}/predict", json={"rows": odd}))
    drifted = _ok(client.post(f"{API}/deployments/{dep}/check", params={"last": 80}))
    emb = drifted["metrics"]["embedding"]
    assert RANK[emb["severity"]] >= RANK["medium"], emb
    assert emb["domain_auc"] > stable["metrics"]["embedding"]["domain_auc"]
    assert drifted["action"] and drifted["action"].startswith("alert:")


def test_image_deployment_files_and_embedding_drift(client: TestClient, fixtures_dir: Path) -> None:
    from PIL import Image

    root = fixtures_dir / "uc04_defects"
    dep = _deployment(client, root, None, epochs=2)
    images = sorted(root.rglob("*.png"))[:40]
    files = [("files", (p.name, p.read_bytes(), "image/png")) for p in images]
    out = _ok(client.post(f"{API}/deployments/{dep}/predict/file", files=files))
    assert len(out["predictions"]) == 40 and out["predictions"][0]["prediction"] is not None
    rows = client.post(f"{API}/deployments/{dep}/predict", json={"rows": [{"a": 1}]})
    assert rows.status_code == 422  # el modelo de imagen recibe archivos
    stable = _ok(client.post(f"{API}/deployments/{dep}/check", params={"last": 40}))
    assert stable["metrics"]["embedding"]["n_current"] == 40, stable["metrics"]
    recent = _ok(client.get(f"{API}/deployments/{dep}/predictions", params={"limit": 3}))
    # De la imagen solo queda el embedding: ni el archivo ni su nombre.
    assert recent[0]["embedding"] and not any(k.startswith("x:") for k in recent[0])

    rng = random.Random(1)  # noqa: S311 - datos de prueba
    noise = []
    for i in range(40):
        size = Image.open(images[0]).size
        im = Image.effect_noise(size, 60 + rng.random() * 40).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        noise.append(("files", (f"ruido_{i}.png", buf.getvalue(), "image/png")))
    _ok(client.post(f"{API}/deployments/{dep}/predict/file", files=noise))
    drifted = _ok(client.post(f"{API}/deployments/{dep}/check", params={"last": 40}))
    emb = drifted["metrics"]["embedding"]
    assert RANK[emb["severity"]] >= RANK["medium"], emb
    bad = [("files", ("roto.png", b"no es una imagen", "image/png"))]
    assert client.post(f"{API}/deployments/{dep}/predict/file", files=bad).status_code == 422
