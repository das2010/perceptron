"""Monitoreo de punta a punta por la API (RF-MON-01..04, 06, 07): deployment con registro,
feedback, drift con alerta por webhook, champion/challenger con rollback, diff y linaje."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import DatasetVersion
from perceptron.monitoring import alerts

API = "/api/v1"
pytestmark = pytest.mark.usefixtures("fake_llm")  # keychain en memoria (webhook)


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
        time.sleep(0.3)
    raise AssertionError("el job no terminó a tiempo")


def _churn_rows(fixtures_dir: Path) -> list[dict[str, Any]]:
    with (fixtures_dir / "uc01_churn" / "churn.csv").open(encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _label(row: dict[str, Any]) -> str:
    """La regla de UC-01 sin ruido (etiqueta "real" del feedback)."""
    logit = (
        -1.2
        - 0.05 * float(row["antiguedad_meses"])
        + 0.35 * float(row["tickets_90d"])
        + (0.8 if row["plan"] == "básico" else 0)
    )
    return str(int(1 / (1 + math.exp(-logit)) > 0.5))


def _model(client: TestClient, pid: str, dv: str, pipe: str, arch: str, epochs: int) -> str:
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv,
                "pipeline_id": pipe,
                "archspec_id": arch,
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
    mv: str = _ok(client.post(f"{API}/runs/{run_id}/register"), 201)["id"]
    return mv


@pytest.fixture
def setup(client: TestClient, fixtures_dir: Path) -> dict[str, str]:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Churn en producción"}), 201)["id"]
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
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]["archspec"]
    return {"project": pid, "dv": dv["id"], "pipeline": pipe["id"], "arch": arch["id"]}


def test_deployment_logging_drift_alert_and_performance(
    client: TestClient,
    setup: dict[str, str],
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        posted.append(json.loads(req.content))
        return httpx.Response(200)

    monkeypatch.setattr(
        alerts, "default_http", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    pid = setup["project"]
    mv = _model(client, pid, setup["dv"], setup["pipeline"], setup["arch"], epochs=6)
    dep = _ok(
        client.post(
            f"{API}/models/{mv}/deployments",
            json={
                "name": "churn-api",
                "key_column": "customer_id",
                "monitoring": {"window": 1000, "min_labels": 20},  # sin chequeos automáticos
                "webhook_url": "https://hooks.example.test/perceptron",
            },
        ),
        201,
    )
    assert dep["endpoint"] == f"/api/v1/deployments/{dep['id']}/predict"
    assert "hooks.example" not in str(dep)  # el webhook va al keychain
    champion = _ok(client.get(f"{API}/projects/{pid}/models"))[0]
    assert champion["stage"] == "production"  # el primer deployment fija el champion

    rows = _churn_rows(fixtures_dir)
    base = [{k: v for k, v in r.items() if k != "churn"} for r in rows[:300]]
    out = _ok(client.post(f"{API}/deployments/{dep['id']}/predict", json={"rows": base}))
    assert len(out["predictions"]) == 300 and out["predictions"][0]["prediction_id"].startswith(
        "prd_"
    )
    stable = _ok(client.post(f"{API}/deployments/{dep['id']}/check", params={"last": 300}))
    assert stable["severity"] in ("none", "low"), stable["metrics"]["data"]
    assert posted == []

    # Drift sintético: clientes nuevos, con muchos reclamos y cargos altos.
    drifted = [
        {
            **r,
            "antiguedad_meses": str(1 + i % 6),
            "tickets_90d": str(7 + i % 3),
            "cargo_mensual": str(float(r["cargo_mensual"]) * 1.8),
        }
        for i, r in enumerate(base[:100])
    ]
    out = _ok(client.post(f"{API}/deployments/{dep['id']}/predict", json={"rows": drifted}))
    feedback = [
        {"prediction_id": p["prediction_id"], "label": _label(r)}
        for p, r in zip(out["predictions"], drifted, strict=True)
    ]
    # Parte del feedback llega por clave de negocio (sin prediction_id).
    feedback[:10] = [{"key": r["customer_id"], "label": _label(r)} for r in drifted[:10]]
    assert (
        _ok(client.post(f"{API}/deployments/{dep['id']}/feedback", json={"items": feedback}))[
            "received"
        ]
        == 100
    )
    report = _ok(client.post(f"{API}/deployments/{dep['id']}/check", params={"last": 100}))
    data = report["metrics"]["data"]
    by_feature = {f["feature"]: f for f in data["features"]}
    assert by_feature["antiguedad_meses"]["severity"] == "high"
    assert report["severity"] in ("medium", "high") and report["action"].startswith("alert:")
    assert report["metrics"]["performance"]["n_labeled"] >= 20

    alerts_list = _ok(client.get(f"{API}/projects/{pid}/alerts"))
    drift_alert = next(a for a in alerts_list if a["kind"] == "data_drift")
    assert drift_alert["channels"] == ["app", "webhook"]
    assert posted and "Drift de datos" in posted[0]["text"]
    assert "antiguedad_meses" in drift_alert["details"]["features"]
    # Otra ventana con drift no repite la alerta abierta (cooldown).
    _ok(client.post(f"{API}/deployments/{dep['id']}/check", params={"last": 100}))
    assert (
        len(
            [
                a
                for a in _ok(client.get(f"{API}/projects/{pid}/alerts"))
                if a["kind"] == "data_drift"
            ]
        )
        == 1
    )
    ack = _ok(client.post(f"{API}/alerts/{drift_alert['id']}/acknowledge"))
    assert ack["status"] == "acknowledged"
    assert len(_ok(client.get(f"{API}/deployments/{dep['id']}/drift"))) == 3
    recent = _ok(client.get(f"{API}/deployments/{dep['id']}/predictions", params={"limit": 5}))
    assert len(recent) == 5 and "x:antiguedad_meses" in recent[0]

    stopped = _ok(client.patch(f"{API}/deployments/{dep['id']}", json={"status": "stopped"}))
    assert stopped["status"] == "stopped"
    blocked = client.post(f"{API}/deployments/{dep['id']}/predict", json={"rows": base[:1]})
    assert blocked.status_code == 409


def test_champion_challenger_and_rollback(
    client: TestClient, setup: dict[str, str], fixtures_dir: Path
) -> None:
    pid = setup["project"]
    first = _model(client, pid, setup["dv"], setup["pipeline"], setup["arch"], epochs=2)
    dep = _ok(client.post(f"{API}/models/{first}/deployments", json={}), 201)
    rows = _churn_rows(fixtures_dir)[:80]
    out = _ok(
        client.post(
            f"{API}/deployments/{dep['id']}/predict",
            json={"rows": [{k: v for k, v in r.items() if k != "churn"} for r in rows]},
        )
    )
    _ok(
        client.post(
            f"{API}/deployments/{dep['id']}/feedback",
            json={
                "items": [
                    {"prediction_id": p["prediction_id"], "label": r["churn"]}
                    for p, r in zip(out["predictions"], rows, strict=True)
                ]
            },
        )
    )
    second = _model(client, pid, setup["dv"], setup["pipeline"], setup["arch"], epochs=4)

    # Exigir una mejora imposible: se evalúan los dos y no se promueve.
    held = _ok(client.post(f"{API}/models/{second}/challenge", json={"min_improvement": 10.0}))
    assert not held["promoted"] and held["champion_id"] == first
    assert held["n_samples"] == 80 and held["metric"] == "roc_auc"
    assert held["champion_value"] is not None and "accuracy" in held["challenger_metrics"]
    models = {m["id"]: m["stage"] for m in _ok(client.get(f"{API}/projects/{pid}/models"))}
    assert models == {first: "production", second: "candidate"}

    # Holdout de otra versión de datos (su test) y promoción explícita.
    on_dataset = _ok(
        client.post(
            f"{API}/models/{second}/challenge",
            json={"holdout": setup["dv"], "promote": False},
        )
    )
    assert on_dataset["holdout"] == f"dataset:{setup['dv']}" and not on_dataset["promoted"]
    _ok(client.post(f"{API}/models/{second}/promote"))
    assert _ok(client.get(f"{API}/deployments/{dep['id']}"))["model_version_id"] == second
    models = {m["id"]: m["stage"] for m in _ok(client.get(f"{API}/projects/{pid}/models"))}
    assert models == {first: "archived", second: "production"}

    back = _ok(client.post(f"{API}/projects/{pid}/models/rollback"))
    assert back["id"] == first and back["stage"] == "production"
    assert _ok(client.get(f"{API}/deployments/{dep['id']}"))["model_version_id"] == first
    again = _ok(client.post(f"{API}/projects/{pid}/models/rollback"))
    assert again["id"] == second


def test_dataset_diff_and_lineage(
    client: TestClient,
    ctx: EngineContext,
    setup: dict[str, str],
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    pid = setup["project"]
    rows = _churn_rows(fixtures_dir)
    fields = list(rows[0])
    newer = tmp_path / "churn nuevo ñ.csv"
    with newer.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*fields, "canal"])
        w.writeheader()
        for r in rows[50:]:
            w.writerow({**r, "canal": "web"})
        for i in range(60):
            w.writerow(
                {**rows[i], "customer_id": f"X{i:03d}", "antiguedad_meses": "1", "canal": "app"}
            )
    src = _ok(client.post(f"{API}/projects/{pid}/sources", json={"path": str(newer)}), 201)
    dv2 = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    repo = ctx.repo(DatasetVersion)
    repo.update(
        repo.get(dv2["id"]).model_copy(
            update={"parent_id": setup["dv"], "transformation": "append:lote-1"}
        )
    )

    diff = _ok(client.get(f"{API}/datasets/{setup['dv']}/diff/{dv2['id']}"))
    assert diff["schema_change"]["added"] == ["canal"]
    assert diff["rows_a"] == len(rows) and diff["rows_b"] == len(rows) - 50 + 60
    # Las filas se comparan por las columnas comunes: 60 nuevas y 50 que ya no están.
    assert diff["rows_added"] == 60 and diff["rows_removed"] == 50
    assert diff["sample_added"][0]["customer_id"].startswith("X")
    tenure = {f["feature"]: f for f in diff["distribution"]["features"]}["antiguedad_meses"]
    assert tenure["current_mean"] < tenure["reference_mean"]

    chain = _ok(client.get(f"{API}/datasets/{dv2['id']}/lineage"))
    assert [n["id"] for n in chain] == [dv2["id"], setup["dv"]]
    assert chain[0]["transformation"] == "append:lote-1"
