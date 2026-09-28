"""Etiquetado asistido por la API (RF-LBL-01..06; aceptación UC-04 de la Capa 4)."""

from __future__ import annotations

import csv
import io
import json
import random
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

API = "/api/v1"
THRESHOLD = 0.8


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


def _defects(client: TestClient, fixtures_dir: Path) -> tuple[str, dict[str, Any]]:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Etiquetado UC-04"}), 201)["id"]
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources", json={"path": str(fixtures_dir / "uc04_defects")}
        ),
        201,
    )
    return pid, _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={}), 201)


def _train(client: TestClient, pid: str, dv_id: str, epochs: int) -> str:
    pipe = _ok(
        client.post(f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv_id}),
        201,
    )
    prop = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv_id, "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv_id,
                "pipeline_id": pipe["id"],
                "archspec_id": prop["archspec"]["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": epochs},
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id: str = job["result"]["best_trial"]["run_id"]
    return run_id


def _truth(client: TestClient, ls_id: str) -> dict[str, str]:
    resp = client.get(f"{API}/labelsets/{ls_id}/export?format=csv")
    assert resp.status_code == 200
    return {r["sample_id"]: r["label"] for r in csv.DictReader(io.StringIO(resp.text))}


def test_uc04_prelabels_halve_manual_actions(client: TestClient, fixtures_dir: Path) -> None:
    pid, dv0 = _defects(client, fixtures_dir)
    ls = _ok(client.post(f"{API}/datasets/{dv0['id']}/labelsets", json={"name": "piezas"}), 201)
    assert sorted(ls["classes"]) == ["defect", "ok"]
    summary = _ok(client.get(f"{API}/labelsets/{ls['id']}"))
    assert summary["accepted"] == summary["total"] == dv0["num_samples"]  # sembradas del dataset
    truth = _truth(client, ls["id"])

    # Se ocultan la mitad de las etiquetas (quedan por etiquetar) y se entrena con el resto.
    rng = random.Random(7)  # noqa: S311 - partición reproducible del test
    hidden = sorted(rng.sample(sorted(truth), len(truth) // 2))
    _ok(
        client.put(
            f"{API}/labelsets/{ls['id']}/labels",
            json={"updates": [{"sample_id": s, "label": None} for s in hidden]},
        )
    )
    assert _ok(client.get(f"{API}/labelsets/{ls['id']}"))["unlabeled"] == len(hidden)
    dv1 = _ok(client.post(f"{API}/labelsets/{ls['id']}/apply"), 201)
    assert dv1["parent_id"] == dv0["id"] and dv1["num_samples"] == len(truth) - len(hidden)
    run_id = _train(client, pid, dv1["id"], epochs=20)

    # El modelo pre-etiqueta lo que falta; la cola prioriza lo más dudoso.
    n = _ok(
        client.post(
            f"{API}/labelsets/{ls['id']}/prelabel", json={"method": "model", "run_id": run_id}
        )
    )["count"]
    assert n == len(hidden)
    queue = _ok(client.get(f"{API}/labelsets/{ls['id']}/queue?strategy=uncertainty&limit=200"))
    confs = [s["item"]["confidence"] for s in queue]
    assert confs == sorted(confs) and all(s["item"]["status"] == "suggested" for s in queue)
    assert queue[0]["path"] and queue[0]["split"] in {"train", "val", "test"}
    img = client.get(f"{API}/labelsets/{ls['id']}/samples/{queue[0]['sample_id']}/file")
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"

    # Anotador simulado: acepta en lote lo confiado (1 acción) y revisa lo dudoso uno por uno.
    low = [s["sample_id"] for s in queue if s["item"]["confidence"] < THRESHOLD]
    bulk = _ok(
        client.post(f"{API}/labelsets/{ls['id']}/accept", json={"min_confidence": THRESHOLD})
    )["count"]
    if low:
        _ok(
            client.put(
                f"{API}/labelsets/{ls['id']}/labels",
                json={"updates": [{"sample_id": s, "label": truth[s]} for s in low]},
            )
        )
    manual_actions = len(hidden)  # sin pre-etiquetas: una acción por muestra
    assisted_actions = (1 if bulk else 0) + len(low)
    reduction = 1 - assisted_actions / manual_actions
    final = _truth(client, ls["id"])
    accuracy = sum(final[s] == truth[s] for s in hidden) / len(hidden)
    print(json.dumps({"reduction": reduction, "accuracy": accuracy, "bulk": bulk, "low": len(low)}))
    assert reduction >= 0.5, (reduction, bulk, len(low))
    assert accuracy >= 0.85, accuracy

    quality = _ok(client.get(f"{API}/labelsets/{ls['id']}/quality"))
    assert quality["compared"] == len(low)  # humanas con opinión previa del modelo


def test_class_labels_roundtrip_and_validation(client: TestClient, fixtures_dir: Path) -> None:
    _, dv0 = _defects(client, fixtures_dir)
    ls = _ok(client.post(f"{API}/datasets/{dv0['id']}/labelsets", json={}), 201)
    bad = client.put(
        f"{API}/labelsets/{ls['id']}/labels",
        json={"updates": [{"sample_id": "row:0", "label": "inexistente"}]},
    )
    assert bad.status_code == 422
    assert (
        client.put(
            f"{API}/labelsets/{ls['id']}/labels",
            json={"updates": [{"sample_id": "row:99999", "label": "ok"}]},
        ).status_code
        == 422
    )
    for fmt in ("csv", "jsonl"):
        data = client.get(f"{API}/labelsets/{ls['id']}/export?format={fmt}").content
        other = _ok(client.post(f"{API}/datasets/{dv0['id']}/labelsets", json={}), 201)
        # Vacío y reimportado: mismas etiquetas.
        all_ids = list(_truth(client, other["id"]))
        _ok(
            client.put(
                f"{API}/labelsets/{other['id']}/labels",
                json={"updates": [{"sample_id": s, "label": None} for s in all_ids]},
            )
        )
        count = _ok(
            client.post(
                f"{API}/labelsets/{other['id']}/import?format={fmt}",
                files={"file": (f"labels.{fmt}", data, "text/plain")},
            )
        )["count"]
        assert count == dv0["num_samples"]
        assert _truth(client, other["id"]) == _truth(client, ls["id"])


def test_boxes_export_import(client: TestClient, fixtures_dir: Path) -> None:
    _, dv0 = _defects(client, fixtures_dir)
    ls = _ok(
        client.post(
            f"{API}/datasets/{dv0['id']}/labelsets", json={"kind": "box", "classes": ["rayón"]}
        ),
        201,
    )
    boxes = [{"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6, "label": "rayón"}]
    _ok(
        client.put(
            f"{API}/labelsets/{ls['id']}/labels",
            json={
                "updates": [
                    {"sample_id": "row:0", "boxes": boxes},
                    {"sample_id": "row:1", "boxes": boxes},
                ]
            },
        )
    )
    for fmt in ("coco", "yolo", "voc"):
        data = client.get(f"{API}/labelsets/{ls['id']}/export?format={fmt}").content
        if fmt != "coco":
            assert zipfile.ZipFile(io.BytesIO(data)).namelist()
        other = _ok(
            client.post(
                f"{API}/datasets/{dv0['id']}/labelsets", json={"kind": "box", "classes": ["rayón"]}
            ),
            201,
        )
        count = _ok(
            client.post(
                f"{API}/labelsets/{other['id']}/import?format={fmt}",
                files={"file": (f"labels.{fmt}", data, "application/octet-stream")},
            )
        )["count"]
        assert count == 2, fmt
        queue = _ok(client.get(f"{API}/labelsets/{other['id']}/queue?strategy=random&limit=200"))
        got = [s["item"] for s in queue if s["item"]]
        assert all(len(i["boxes"]) == 1 for i in got) and len(got) == 2
        b = got[0]["boxes"][0]
        assert b["label"] == "rayón" and b["x1"] == pytest.approx(0.1, abs=1e-4)
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("pieza_000.xml", '<!DOCTYPE a [<!ENTITY x "y">]><annotation>&x;</annotation>')
    r = client.post(
        f"{API}/labelsets/{ls['id']}/import?format=voc",
        files={"file": ("evil.zip", evil.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
