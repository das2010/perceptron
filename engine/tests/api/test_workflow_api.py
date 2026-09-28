"""Flujo completo por la API REST + WebSocket (UC-01, presupuesto mínimo)."""

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


def test_full_flow_uc01(client: TestClient, fixtures_dir: Path) -> None:
    project = _ok(
        client.post(f"{API}/projects", json={"name": "Churn API", "template": "UC-01"}), 201
    )
    pid = project["id"]

    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    preview = _ok(client.post(f"{API}/sources/{src['id']}/preview?limit=5"))
    assert preview["kind"] == "table" and len(preview["rows"]) == 5
    assert "churn" in preview["schema"]["target_candidates"]

    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    assert dv["num_samples"] == 400 and dv["target"] == "churn"
    assert _ok(client.get(f"{API}/projects/{pid}"))["status"] == "active"
    samples = _ok(client.get(f"{API}/datasets/{dv['id']}/samples?limit=3"))
    assert len(samples) == 3 and "__split__" not in samples[0]
    assert (
        client.get(f"{API}/datasets/{dv['id']}/samples?split=test").status_code == 403
    )  # test sellado

    card = _ok(client.post(f"{API}/datasets/{dv['id']}/profile"))
    assert card["target"]["name"] == "churn"
    assert _ok(client.get(f"{API}/datasets/{dv['id']}/profile"))["num_samples"] == 400

    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    prev = _ok(
        client.post(
            f"{API}/pipelines/{pipe['id']}/preview",
            json={"dataset_version_id": dv["id"], "rows": 4},
        )
    )
    assert len(prev["x_num"]) == 4 and prev["classes"] == ["0", "1"]
    # Vista previa por paso de un grafo sin guardar (RF-PIP-02): la salida intermedia.
    graph = pipe["graph"]
    first = graph["steps"][0]["id"]
    step = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/preview-steps",
            json={"dataset_version_id": dv["id"], "graph": graph, "upto_step": first, "rows": 3},
        )
    )
    assert step["step_id"] == first and len(step["rows"]) == 3
    assert "churn" not in step["columns"] and len(step["dtypes"]) == len(step["columns"])
    broken = {**graph, "steps": [{**graph["steps"][0], "kind": "no_existe"}]}
    r = client.post(
        f"{API}/projects/{pid}/pipelines/preview-steps",
        json={"dataset_version_id": dv["id"], "graph": broken},
    )
    assert r.status_code == 422, r.text

    proposals = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )
    # Sin LLM configurado (tests herméticos) cae a reglas y lo informa (RF-ARC-04).
    assert proposals["origin"] == "rules" and proposals["fallback_reason"]
    prop = proposals["proposals"][0]
    assert prop["validation"]["valid"] and prop["rationale"]
    assert prop["estimates"]["num_params"] > 0
    spec = prop["archspec"]["spec"]
    assert _ok(client.post(f"{API}/arch/validate", json=spec))["valid"]
    # Editor visual (RF-ARC-05): listar, leer y guardar una ArchSpec editada.
    assert any(
        a["id"] == prop["archspec"]["id"]
        for a in _ok(client.get(f"{API}/projects/{pid}/archspecs"))
    )
    assert _ok(client.get(f"{API}/archspecs/{prop['archspec']['id']}"))["name"] == spec["name"]
    edited = {**spec, "name": "mlp-editada"}
    saved = _ok(client.post(f"{API}/projects/{pid}/archspecs", json={"spec": edited}), 201)
    assert saved["origin"] == "manual" and saved["name"] == "mlp-editada"
    broken = {**spec, "nodes": [{**spec["nodes"][0], "block": "no.existe"}, *spec["nodes"][1:]]}
    assert client.post(f"{API}/projects/{pid}/archspecs", json={"spec": broken}).status_code == 422
    assert any(p["id"] == pipe["id"] for p in _ok(client.get(f"{API}/projects/{pid}/pipelines")))
    assert _ok(client.get(f"{API}/pipelines/{pipe['id']}"))["id"] == pipe["id"]
    code = _ok(client.post(f"{API}/arch/to-code", json=spec))["code"]
    assert "class Model(nn.Module)" in code
    blocks = _ok(client.get(f"{API}/catalog/blocks?modality=tabular"))
    assert any(b["key"] == "input.tabular" for b in blocks)

    budget = {"max_trials": 2, "max_epochs_per_trial": 1}
    strategy = _ok(
        client.post(
            f"{API}/projects/{pid}/hpo/strategy",
            json={"archspec_id": prop["archspec"]["id"], "budget": budget},
        )
    )
    assert strategy["budget"]["max_trials"] == 2

    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": prop["archspec"]["id"],
                "budget": budget,
            },
        ),
        202,
    )
    job_id = launch["job"]["id"]
    with client.websocket_connect(f"{API}/jobs/{job_id}") as ws:
        kinds = []
        while True:
            msg = ws.receive_json()
            kinds.append(msg["kind"])
            if msg["kind"] == "finished":
                break
    assert "epoch" in kinds

    job = _wait_job(client, job_id)
    assert job["status"] == "succeeded", job["error"]
    best = job["result"]["best_trial"]
    assert best is not None

    runs = _ok(client.get(f"{API}/projects/{pid}/runs?study_id={launch['study']['id']}"))
    assert len(runs) == 2
    assert all(r["mlflow_run_id"] for r in runs)
    run_id = best["run_id"]
    assert _ok(client.get(f"{API}/runs/{run_id}"))["status"] == "succeeded"
    history = _ok(client.get(f"{API}/runs/{run_id}/history"))
    assert history and "val_loss" in history[0]

    report = _ok(client.post(f"{API}/runs/{run_id}/evaluate"))
    assert report["split"] == "test" and "accuracy" in report["metrics"]
    assert _ok(client.get(f"{API}/runs/{run_id}/evaluation"))["run_id"] == run_id
    mv = _ok(client.post(f"{API}/runs/{run_id}/register"), 201)
    assert mv["stage"] == "candidate"
    assert [m["id"] for m in _ok(client.get(f"{API}/projects/{pid}/models"))] == [mv["id"]]
    cmp = _ok(client.post(f"{API}/runs/compare", json={"run_ids": [r["id"] for r in runs]}))
    assert len(cmp) == 2


def test_unknown_job_and_study(client: TestClient) -> None:
    assert client.get(f"{API}/jobs/job_nope").status_code == 404
    assert client.post(f"{API}/studies/std_nope/cancel").status_code == 404
