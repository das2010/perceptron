"""API de la capa LLM: proveedores (clave write-only), perfiles, prueba, auditoría, roles."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from perceptron.llm.providers.fake import FakeLLMProvider

API = "/api/v1"


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def test_providers_key_is_write_only(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    providers = {p["name"]: p for p in _ok(client.get(f"{API}/llm/providers"))}
    assert {"anthropic", "openai", "gemini", "moonshot", "openai_compat", "ollama"} <= set(
        providers
    )
    assert providers["anthropic"]["has_key"] is False and providers["ollama"]["local"] is True
    r = client.put(
        f"{API}/llm/providers/anthropic",
        json={"kind": "anthropic", "api_key": "sk-ant-secreto"},
    )
    assert "sk-ant-secreto" not in r.text
    updated = {p["name"]: p for p in _ok(r)}
    assert updated["anthropic"]["has_key"] is True
    assert updated["anthropic"]["api_key_ref"] == "ANTHROPIC_API_KEY"
    assert updated["anthropic"]["models"], "conserva los modelos del catálogo"


def test_profiles_get_put(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    view = _ok(client.get(f"{API}/llm/profiles"))
    assert view["active"] == "anthropic" and "ollama" in view["profiles"]
    custom = {
        "provider": "ollama",
        "purposes": {"copilot": {"model": "qwen3:4b"}, "architect": {"model": "qwen3:4b"}},
    }
    view = _ok(
        client.put(f"{API}/llm/profiles", json={"active": "mio", "profiles": {"mio": custom}})
    )
    assert view["active"] == "mio" and view["profiles"]["mio"]["provider"] == "ollama"


def test_llm_test_and_audit(client: TestClient, fake_llm: FakeLLMProvider) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "p"}), 201)["id"]
    fake_llm.script("copilot", {"ok": True, "message": "hola"})
    res = _ok(client.post(f"{API}/llm/test", json={"project_id": pid}))
    assert res["ok"] and res["provider"] == "anthropic" and res["model"]
    [call] = _ok(client.get(f"{API}/llm/audit", params={"project": pid}))
    assert call["purpose"] == "copilot" and call["privacy_level"] == "L1"
    assert call["payload"]["system"] and call["prompt_version"] == "copilot/ping@v1"
    assert (
        _ok(client.get(f"{API}/llm/audit", params={"project": pid, "purpose": "architect"})) == []
    )


def test_llm_test_reports_unavailable(client: TestClient) -> None:
    res = _ok(client.post(f"{API}/llm/test", json={}))
    assert res["ok"] is False and "llm_unavailable" in res["error"]


def test_labeling_guide_rules_mode(client: TestClient) -> None:
    pid = _ok(client.post(f"{API}/projects", json={"name": "p"}), 201)["id"]
    guide = _ok(
        client.post(
            f"{API}/projects/{pid}/labels/guide",
            json={"classes": {"falla": "ruido anómalo"}, "mode": "rules"},
        )
    )
    assert guide["classes"][0] == {
        "name": "falla",
        "definition": "ruido anómalo",
        "positive_examples": [],
        "edge_cases": [],
    }
