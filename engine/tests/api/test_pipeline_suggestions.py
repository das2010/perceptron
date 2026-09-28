"""Sugerencias de cambios al pipeline con aceptación por cambio (RF-PIP-05)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from perceptron.llm.providers.fake import FakeLLMProvider

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _pipeline(client: TestClient, fixtures_dir: Path) -> tuple[str, str, dict[str, Any]]:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Pipelines"}), 201)["id"]
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    _ok(client.post(f"{API}/datasets/{dv['id']}/profile"))
    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    return dv["id"], pipe["id"], pipe


def test_llm_suggestions_are_validated_and_applied_one_by_one(
    client: TestClient, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    dv_id, pipe_id, pipe = _pipeline(client, fixtures_dir)
    first = pipe["graph"]["steps"][0]
    column = first["columns"][0]
    updated = {**first, "params": {**first["params"], "revisado": True}}
    fake_llm.script(
        "architect",
        # 1.er intento inválido (paso inexistente): el gateway reintenta con el error.
        {
            "suggestions": [
                {
                    "op": "add",
                    "step_id": "x",
                    "step": {"id": "x", "kind": "magia", "columns": [column]},
                    "rationale": "no existe",
                }
            ]
        },
        {
            "suggestions": [
                {"op": "update", "step_id": first["id"], "step": updated, "rationale": "a"},
                {"op": "remove", "step_id": first["id"], "rationale": "b"},
            ]
        },
    )
    res = _ok(
        client.post(
            f"{API}/pipelines/{pipe_id}/suggestions",
            json={"dataset_version_id": dv_id, "mode": "llm"},
        )
    )
    assert res["origin"] == "llm" and res["llm_call_id"]
    assert len(fake_llm.calls("architect")) == 2  # reintento con el error de validación
    update, remove = res["items"]
    assert update["before"]["params"] != update["after"]["params"]  # diff visual
    assert remove["after"] is None and remove["before"]["id"] == first["id"]
    # El usuario acepta solo la primera.
    applied = _ok(
        client.post(
            f"{API}/pipelines/{pipe_id}/suggestions/apply",
            json={"version": res["pipeline_version"], "changes": [update["change"]]},
        )
    )
    steps = {s["id"]: s for s in applied["graph"]["steps"]}
    assert steps[first["id"]]["params"]["revisado"] is True
    assert applied["version"] == res["pipeline_version"] + 1
    stale = client.post(
        f"{API}/pipelines/{pipe_id}/suggestions/apply",
        json={"version": res["pipeline_version"], "changes": [remove["change"]]},
    )
    assert stale.status_code == 409  # el pipeline cambió desde la sugerencia


def test_rules_suggest_what_the_profile_needs(client: TestClient, fixtures_dir: Path) -> None:
    dv_id, pipe_id, pipe = _pipeline(client, fixtures_dir)
    removed = pipe["graph"]["steps"][0]
    graph = {**pipe["graph"], "steps": pipe["graph"]["steps"][1:]}
    edited = _ok(
        client.put(f"{API}/pipelines/{pipe_id}", json={"graph": graph, "version": pipe["version"]})
    )
    res = _ok(
        client.post(
            f"{API}/pipelines/{pipe_id}/suggestions",
            json={"dataset_version_id": dv_id, "mode": "rules"},
        )
    )
    assert res["origin"] == "rules" and res["pipeline_version"] == edited["version"]
    adds = [i for i in res["items"] if i["change"]["op"] == "add"]
    assert [a["change"]["step_id"] for a in adds] == [removed["id"]]
    assert adds[0]["before"] is None and adds[0]["after"]["kind"] == removed["kind"]
