"""Aceptación de la Capa 6 (SPEC §14, UC-10): al inyectar drift sintético en el stream del
fixture se genera una alerta, se reentrena, el challenger se evalúa y se promueve solo si
mejora; el rollback funciona. También: disparadores por volumen y cron del scheduler."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import DatasetVersion, RetrainPolicy
from perceptron.monitoring.scheduler import Scheduler

API = "/api/v1"
pytestmark = pytest.mark.usefixtures("fake_llm")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _wait(client: TestClient, url: str, done: set[str], timeout: float = 900) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        item = _ok(client.get(url))
        if item["status"] in done:
            return item
        time.sleep(0.5)
    raise AssertionError(f"{url} no terminó a tiempo")


def _champion(client: TestClient, fixtures_dir: Path) -> dict[str, str]:
    pid = _ok(client.post(f"{API}/projects", json={"name": "UC-10 churn"}), 201)["id"]
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
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": arch["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": 8},
            },
        ),
        202,
    )
    job = _wait(client, f"{API}/jobs/{launch['job']['id']}", {"succeeded", "failed"})
    assert job["status"] == "succeeded", job["error"]
    run_id = job["result"]["best_trial"]["run_id"]
    _ok(client.post(f"{API}/runs/{run_id}/evaluate"))
    export = _ok(client.post(f"{API}/runs/{run_id}/export", json={"formats": ["onnx"]}), 202)
    assert (
        _wait(client, f"{API}/jobs/{export['job']['id']}", {"succeeded", "failed"})["status"]
        == "succeeded"
    )
    mv = _ok(client.post(f"{API}/runs/{run_id}/register"), 201)["id"]
    return {"project": pid, "dv": dv["id"], "mv": mv}


def _stream(fixtures_dir: Path) -> list[dict[str, Any]]:
    lines = (fixtures_dir / "uc10_stream" / "stream.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_uc10_drift_alert_retrain_challenger_and_rollback(
    client: TestClient, ctx: EngineContext, fixtures_dir: Path, tmp_path: Path
) -> None:
    base = _champion(client, fixtures_dir)
    pid, champion = base["project"], base["mv"]
    dep = _ok(
        client.post(
            f"{API}/models/{champion}/deployments",
            json={"name": "churn", "key_column": "customer_id", "monitoring": {"window": 5000}},
        ),
        201,
    )
    feed = tmp_path / "stream de clientes ñ.jsonl"
    feed.write_text("", encoding="utf-8")
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources/stream",
            json={"name": "altas", "kind": "file", "config": {"path": str(feed)}},
        ),
        201,
    )
    policy = _ok(
        client.put(
            f"{API}/projects/{pid}/retrain-policy",
            json={
                "deployment_id": dep["id"],
                "source_ids": [src["id"]],
                "triggers": [{"type": "drift", "min_severity": "medium"}],
                "require_approval": False,
                "use_feedback": False,  # el stream ya trae la etiqueta real
                "budget": {"max_trials": 1, "max_epochs_per_trial": 8},
                "cooldown_s": 0,
            },
        )
    )
    assert policy["enabled"] and policy["triggers"] == [{"type": "drift", "min_severity": "medium"}]

    records = _stream(fixtures_dir)
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_batch.setdefault(int(r["batch"]), []).append(r)

    def deliver(batch: int) -> None:
        rows = by_batch[batch]
        with feed.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        pulled = _ok(client.post(f"{API}/sources/{src['id']}/pull"))
        assert pulled["added"] == len(rows)
        features = [{k: v for k, v in r.items() if k not in ("churn", "batch")} for r in rows]
        _ok(client.post(f"{API}/deployments/{dep['id']}/predict", json={"rows": features}))

    for batch in range(7):  # sin drift
        deliver(batch)
    stable = _ok(client.post(f"{API}/deployments/{dep['id']}/check", params={"last": 7 * 60}))
    assert stable["severity"] in ("none", "low"), stable["metrics"]["data"]
    assert _ok(client.get(f"{API}/projects/{pid}/retrain-runs")) == []

    for batch in (7, 8, 9):  # drift sintético del fixture
        deliver(batch)
    drifted = _ok(client.post(f"{API}/deployments/{dep['id']}/check", params={"last": 3 * 60}))
    assert drifted["severity"] in ("medium", "high")
    alerts = _ok(client.get(f"{API}/projects/{pid}/alerts"))
    assert any(a["kind"] == "data_drift" for a in alerts)

    runs = _ok(client.get(f"{API}/projects/{pid}/retrain-runs"))
    assert len(runs) == 1 and runs[0]["trigger"]["type"] == "drift"
    done = _wait(
        client,
        f"{API}/retrain-runs/{runs[0]['id']}",
        {"promoted", "not_improved", "failed", "skipped", "awaiting_approval"},
    )
    assert done["status"] in ("promoted", "not_improved"), done
    assert done["new_rows"] == len(records)

    # Versión nueva con linaje y el test del champion intacto dentro del test nuevo.
    dv1 = ctx.repo(DatasetVersion).get(done["dataset_version_id"])
    assert dv1.parent_id == base["dv"] and dv1.transformation == f"append:{len(records)}"
    dv0 = ctx.repo(DatasetVersion).get(base["dv"])
    assert dv1.split is not None and dv0.split is not None
    assert dv1.split.test == dv0.split.test + round(len(records) * 0.3)

    result = done["challenge"]
    assert result["holdout"] == f"dataset:{dv1.id}" and result["champion_id"] == champion
    models = {m["id"]: m["stage"] for m in _ok(client.get(f"{API}/projects/{pid}/models"))}
    challenger = done["model_version_id"]
    # Se promueve solo si mejora.
    if result["improvement"] > 0:
        assert done["status"] == "promoted" and result["promoted"]
        assert models[challenger] == "production" and models[champion] == "archived"
        assert _ok(client.get(f"{API}/deployments/{dep['id']}"))["model_version_id"] == challenger
    else:
        assert done["status"] == "not_improved" and not result["promoted"]
        assert models[champion] == "production" and models[challenger] == "candidate"
        _ok(client.post(f"{API}/models/{challenger}/promote"))
    retrain_alerts = [
        a for a in _ok(client.get(f"{API}/projects/{pid}/alerts")) if a["kind"] == "retrain"
    ]
    assert retrain_alerts

    # Rollback funcional: vuelve el champion original y el deployment lo sigue.
    back = _ok(client.post(f"{API}/projects/{pid}/models/rollback"))
    assert back["id"] == champion and back["stage"] == "production"
    assert _ok(client.get(f"{API}/deployments/{dep['id']}"))["model_version_id"] == champion
    lineage = _ok(client.get(f"{API}/datasets/{dv1.id}/lineage"))
    assert [n["id"] for n in lineage] == [dv1.id, base["dv"]]


def test_scheduler_volume_and_cron_triggers(
    client: TestClient, ctx: EngineContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Disparadores"}), 201)["id"]
    feed = tmp_path / "altas.jsonl"
    feed.write_text(
        "".join(json.dumps({"x": i, "churn": i % 2}) + "\n" for i in range(30)), encoding="utf-8"
    )
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources/stream",
            json={
                "name": "f",
                "kind": "file",
                "config": {"path": str(feed)},
                "poll_interval_s": 10,
            },
        ),
        201,
    )
    _ok(
        client.put(
            f"{API}/projects/{pid}/retrain-policy",
            json={
                "source_ids": [src["id"]],
                "triggers": [
                    {"type": "volume", "min_rows": 25},
                    {"type": "cron", "expr": "0 3 * * *"},
                ],
                "cooldown_s": 0,
            },
        )
    )
    assert (
        client.put(
            f"{API}/projects/{pid}/retrain-policy",
            json={"triggers": [{"type": "cron", "expr": "nunca"}]},
        ).status_code
        == 422
    )
    fired: list[dict[str, Any]] = []
    monkeypatch.setattr(
        Scheduler, "fire", lambda self, policy, trigger, now=None: fired.append(trigger)
    )
    scheduler = Scheduler(ctx, interval_s=3600)
    try:
        scheduler.tick(datetime(2026, 9, 28, 2, 59, tzinfo=UTC))  # sondea la fuente y ve 30 filas
        assert _ok(client.get(f"{API}/sources/{src['id']}/buffer"))["rows"] == 30
        assert fired == [{"type": "volume", "rows": 30}]
        fired.clear()
        policy = ctx.repo(RetrainPolicy).list(filters={"project_id": pid})[0]
        ctx.repo(RetrainPolicy).update(
            policy.model_copy(update={"last_cron_at": datetime(2026, 9, 28, 2, 59, tzinfo=UTC)})
        )
        scheduler.tick(datetime(2026, 9, 28, 3, 1, tzinfo=UTC))
        assert {"type": "cron", "expr": "0 3 * * *"} in fired
    finally:
        scheduler.stop()
