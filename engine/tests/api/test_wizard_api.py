"""Wizard: borrador versionado (RF-WIZ-04) y copiloto por WebSocket (RF-LLM-05)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from perceptron.llm.providers.fake import FakeLLMProvider

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _project(client: TestClient) -> str:
    return _ok(
        client.post(f"{API}/projects", json={"name": "motores", "goal": "detectar fallas"}), 201
    )["id"]


def test_draft_is_created_and_versioned(client: TestClient) -> None:
    pid = _project(client)
    view = _ok(client.get(f"{API}/projects/{pid}/draft"))
    assert view["steps"][0] == "goal" and view["values"]["goal"] == "detectar fallas"
    v = view["draft"]["version"]
    view = _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={
                "version": v,
                "step": "task",
                "values": {"task": "classification", "max_trials": 10},
            },
        )
    )
    assert view["draft"]["step"] == "task" and view["values"]["max_trials"] == 10
    assert view["draft"]["history"][-1]["fields"] == ["max_trials", "task"]
    stale = client.patch(f"{API}/projects/{pid}/draft", json={"version": v, "values": {}})
    assert stale.status_code == 409
    bad = client.patch(
        f"{API}/projects/{pid}/draft",
        json={"version": view["draft"]["version"], "values": {"max_trials": 0}},
    )
    assert bad.status_code == 422
    # el objetivo del borrador se refleja en el proyecto
    _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={"version": view["draft"]["version"], "values": {"goal": "anticipar fallas"}},
        )
    )
    assert _ok(client.get(f"{API}/projects/{pid}"))["goal"] == "anticipar fallas"


def test_copilot_streams_and_suggests_patch(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    pid = _project(client)
    fake_llm.script(
        "copilot",
        "Para no perder fallas conviene optimizar el recall de la clase falla.",
        {
            "changes": [
                {
                    "field": "target_metric",
                    "value": "val_recall_macro",
                    "rationale": "Perder una falla cuesta más que una falsa alarma.",
                }
            ]
        },
    )
    with client.websocket_connect(f"{API}/projects/{pid}/copilot") as ws:
        ws.send_json({"type": "message", "text": "¿Qué métrica uso?"})
        events = []
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] in ("patch", "error"):
                break
    kinds = [e["type"] for e in events]
    assert kinds.count("token") > 1 and "done" in kinds and kinds[-1] == "patch"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text.startswith("Para no perder fallas")
    change = events[-1]["patch"]["changes"][0]
    assert change["field"] == "target_metric" and events[-1]["llm_call_id"]


def test_copilot_reports_unavailable_llm(client: TestClient) -> None:
    pid = _project(client)
    with client.websocket_connect(f"{API}/projects/{pid}/copilot") as ws:
        ws.send_json({"type": "message", "text": "hola"})
        ev = ws.receive_json()
    assert ev["type"] == "error" and ev["code"] == "llm_unavailable"
