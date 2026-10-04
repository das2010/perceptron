"""Wizard: borrador versionado (RF-WIZ-04) y copiloto por WebSocket (RF-LLM-05)."""

from __future__ import annotations

import json
import time
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


# ------------------------------------------------------------------ wizard adaptativo (ADR-0040)


def test_brief_compiles_the_plan_and_system_fields_are_protected(
    client: TestClient, fixtures_dir: Any
) -> None:
    pid = _project(client)
    view = _ok(client.get(f"{API}/projects/{pid}/draft"))
    base = [s for s in view["steps"] if s not in ("formula", "threshold")]  # condicionales
    assert [s["id"] for s in view["plan"]["steps"]] == base
    assert not view["plan"]["adapted"]
    brief = {"problem": "category", "error_costs": "false_negative_worse", "error_cost_ratio": 10}
    view = _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={"version": view["draft"]["version"], "values": {"brief": brief}},
        )
    )
    plan = view["plan"]
    assert plan["adapted"] and view["values"]["brief"]["origins"]["problem"] == "user"
    metric = next(d for d in plan["defaults"] if d["key"] == "target_metric")
    assert metric["value"] == "val_recall_macro"
    assert {"brief", "plan"} <= set(view["draft"]["history"][-1]["fields"])
    forged = client.patch(
        f"{API}/projects/{pid}/draft",
        json={"version": view["draft"]["version"], "values": {"plan": {"steps": []}}},
    )
    assert forged.status_code == 422
    # Con datos: los hechos se miden y el etiquetado se saltea (ya hay target).
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    view = _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={"version": view["draft"]["version"], "values": {"dataset_version_id": dv["id"]}},
        )
    )
    facts = view["values"]["data_facts"]
    assert facts["dataset_version_id"] == dv["id"] and facts["target"] == "churn"
    assert "labeling" not in [s["id"] for s in view["plan"]["steps"]]


def test_intake_proposes_without_applying(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    pid = _project(client)
    fake_llm.script(
        "copilot",
        {
            "changes": [
                {"field": "problem", "value": "category", "rationale": "Quiere detectar fallas."},
                {
                    "field": "error_costs",
                    "value": "false_negative_worse",
                    "rationale": "Dijo que perder una falla es lo peor.",
                },
            ],
            "assumptions": [{"text": "Las fallas son raras", "confidence": 0.6}],
            "next_question": "¿Los datos vienen de varias máquinas?",
        },
    )
    before = _ok(client.get(f"{API}/projects/{pid}/draft"))
    reply = _ok(
        client.post(
            f"{API}/projects/{pid}/draft/intake",
            json={
                "message": "Quiero saber cuándo falla un rodamiento; perder una falla es lo peor."
            },
        )
    )
    assert [c["field"] for c in reply["patch"]["changes"]] == ["problem", "error_costs"]
    assert reply["patch"]["next_question"] and reply["llm_call_id"]
    after = _ok(client.get(f"{API}/projects/{pid}/draft"))
    assert after["draft"]["version"] == before["draft"]["version"]  # nada se aplicó
    assert after["values"]["brief"] is None


def test_intake_without_llm_is_unavailable(client: TestClient) -> None:
    pid = _project(client)
    r = client.post(f"{API}/projects/{pid}/draft/intake", json={"message": "hola"})
    assert r.status_code == 503 and r.json()["code"] == "llm_unavailable"


def test_plan_diff_shows_what_changed(client: TestClient) -> None:
    pid = _project(client)
    view = _ok(client.get(f"{API}/projects/{pid}/draft"))
    for brief in ({"problem": "value"}, {"problem": "rule", "extrapolate": True}):
        view = _ok(
            client.patch(
                f"{API}/projects/{pid}/draft",
                json={"version": view["draft"]["version"], "values": {"brief": brief}},
            )
        )
    diff = view["values"]["plan_diff"]
    assert diff["added_steps"] == ["formula"] and "architecture_hint" in diff["changed_defaults"]
    assert "formula" in [s["id"] for s in view["plan"]["steps"]]


def test_reconcile_compares_the_brief_with_the_data(
    client: TestClient, fake_llm: FakeLLMProvider, fixtures_dir: Any
) -> None:
    pid = _project(client)
    r = client.post(f"{API}/projects/{pid}/draft/reconcile")
    assert r.status_code == 422 and r.json()["details"]["reason"] == "no_dataset"
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    view = _ok(client.get(f"{API}/projects/{pid}/draft"))
    view = _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={
                "version": view["draft"]["version"],
                "values": {"dataset_version_id": dv["id"], "brief": {"problem": "value"}},
            },
        )
    )
    assert "task_mismatch" in [c["code"] for c in view["plan"]["checks"]]  # también sin LLM
    fake_llm.script(
        "copilot",
        {
            "changes": [
                {
                    "field": "problem",
                    "value": "category",
                    "rationale": "El objetivo «churn» tiene dos clases: es una categoría.",
                }
            ],
            "next_question": None,
        },
    )
    reply = _ok(client.post(f"{API}/projects/{pid}/draft/reconcile"))
    assert reply["patch"]["changes"][0]["value"] == "category"
    after = _ok(client.get(f"{API}/projects/{pid}/draft"))
    assert after["draft"]["version"] == view["draft"]["version"]  # nada se aplicó


def _wait_job(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("el job no terminó a tiempo")


def _two_mlps(request: Any) -> dict[str, Any]:
    text = request.messages[0].content
    data = json.loads(text.split("<datos>", 1)[1].split("</datos>", 1)[0])
    base = data["constraints"]["base_archspec"]
    return {
        "proposals": [
            {"title": "MLP", "archspec": {**base, "name": "mlp"}, "rationale": "base"},
            {"title": "MLP bis", "archspec": {**base, "name": "mlp-bis"}, "rationale": "otra"},
        ]
    }


def test_guided_design_runs_the_chain_and_waits_for_acceptance(
    client: TestClient, fake_llm: FakeLLMProvider, fixtures_dir: Any
) -> None:
    """ADR-0041 it. 2: propuestas evaluadas → torneo → elegida con evidencia → estrategia."""
    pid = _project(client)
    r = client.post(f"{API}/projects/{pid}/draft/design", json={})
    assert r.status_code == 422  # sin datos
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    view = _ok(client.get(f"{API}/projects/{pid}/draft"))
    view = _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={
                "version": view["draft"]["version"],
                "values": {"dataset_version_id": dv["id"], "brief": {"problem": "category"}},
            },
        )
    )
    fake_llm.script("architect", _two_mlps)
    job = _ok(client.post(f"{API}/projects/{pid}/draft/design", json={"tournament": "always"}), 202)
    done = _wait_job(client, job["id"])
    assert done["status"] == "succeeded", done.get("error")
    after = _ok(client.get(f"{API}/projects/{pid}/draft"))
    values = after["values"]
    design = values["design"]
    assert values["pipeline_id"] == design["pipeline_id"]
    assert not values.get("archspec_id")  # no se aplica solo
    ids = {c["archspec_id"] for c in design["candidates"]}
    assert len(ids) == 2 and design["requirements"]["items"]
    t = design["tournament"]
    assert t["metric"] == "val_f1_macro" and t["subset"] == 1.0 and t["winner"] in ids
    assert design["pick"] == t["winner"] and design["pick_reason"].startswith("Ganó")
    assert all(c["tournament_status"] == "complete" for c in design["candidates"])
    assert design["strategy"]["strategy"] and design["max_epochs_per_trial"] >= 1
    # El diseño lo calcula el sistema: no se edita por PATCH.
    version = after["draft"]["version"]
    bad = client.patch(
        f"{API}/projects/{pid}/draft", json={"version": version, "values": {"design": None}}
    )
    assert bad.status_code == 422
    # Aceptar = fijar la arquitectura y la estrategia propuestas.
    accepted = _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={
                "version": version,
                "values": {"archspec_id": design["pick"], "strategy": design["strategy"]},
            },
        )
    )
    assert accepted["values"]["archspec_id"] == design["pick"] and accepted["values"]["design"]
    # Otra ficha → el diseño ya no corresponde.
    changed = _ok(
        client.patch(
            f"{API}/projects/{pid}/draft",
            json={
                "version": accepted["draft"]["version"],
                "values": {"brief": {"problem": "category", "deployment": "edge"}},
            },
        )
    )
    assert "design" not in changed["values"] or changed["values"]["design"] is None
