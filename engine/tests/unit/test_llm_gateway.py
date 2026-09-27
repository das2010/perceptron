"""LLM Gateway: política, salida estructurada con reintento, caché, presupuesto y auditoría."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from perceptron.api.context import EngineContext
from perceptron.core.config import LLMProviderSettings, LLMSettings
from perceptron.domain.enums import LLMPurpose, PrivacyLevel
from perceptron.domain.models import LLMCall, Project, Workspace
from perceptron.llm.config import LLMConfig, load_catalog
from perceptron.llm.errors import (
    LLMBudgetExceededError,
    LLMOutputInvalidError,
    LLMProviderError,
    LLMUnavailableError,
)
from perceptron.llm.gateway import Gateway
from perceptron.llm.jsonparse import extract_json
from perceptron.llm.privacy import LLMContext
from perceptron.llm.prompts.registry import PromptRegistry
from perceptron.llm.providers.fake import FakeLLMProvider
from perceptron.llm.secrets import EncryptedFileSecrets, MemorySecrets
from perceptron.llm.types import LLMRequest, Usage


class Answer(BaseModel):
    value: int
    label: str = ""


def datos(request: LLMRequest) -> dict[str, Any]:
    """El payload filtrado que viajó dentro del prompt."""
    text = request.messages[0].content
    body = text.split("<datos>", 1)[1].split("</datos>", 1)[0]
    return json.loads(body)


def _gateway(
    ctx: EngineContext,
    fake: FakeLLMProvider,
    *,
    settings: LLMSettings | None = None,
    secrets: MemorySecrets | None = None,
    factory: bool = True,
) -> Gateway:
    config = LLMConfig(settings or LLMSettings(enabled=True), ctx.settings.workspace_dir)
    return Gateway(
        ctx.db,
        config,
        secrets or MemorySecrets(),
        provider_factory=(lambda *_: fake) if factory else None,
    )


@pytest.fixture
def project(ctx: EngineContext) -> Project:
    return ctx.projects.add(Project(name="p", goal="predecir la baja de clientes"))


def test_catalog_and_profiles_have_no_code_ids() -> None:
    cat = load_catalog()
    assert cat.default_profile in cat.profiles
    for name, prof in cat.profiles.items():
        assert prof.provider in cat.providers, name
        assert set(prof.purposes) == set(LLMPurpose), name
    assert cat.providers["ollama"].is_local and not cat.providers["anthropic"].is_local


def test_prompts_all_render() -> None:
    reg = PromptRegistry()
    templates = reg.all()
    names = {(t.purpose, t.name) for t in templates}
    assert (LLMPurpose.ARCHITECT, "main") in names and (LLMPurpose.LABELER, "prelabel") in names
    for t in templates:
        variables = dict.fromkeys(t.meta.get("variables") or [], "x")
        system, user = t.render({"a": 1}, **variables)
        assert "<datos>" in user and "instrucciones" in system, t.ref


def test_structured_success_is_audited(ctx: EngineContext, project: Project) -> None:
    fake = FakeLLMProvider().script("architect", {"value": 3, "label": "ok"})
    gw = _gateway(ctx, fake)
    out = gw.structured(
        LLMPurpose.ARCHITECT,
        Answer,
        LLMContext(goal=project.goal, constraints={"k": 1}),
        project=project,
        prompt_vars={"n_min": 2, "n_max": 3},
    )
    assert out.value == Answer(value=3, label="ok") and out.attempts == 1
    assert out.level is PrivacyLevel.L1 and out.provider == "anthropic"
    [call] = ctx.repo(LLMCall).list(filters={"project_id": project.id})
    assert call.id == out.call_id and call.status == "ok" and call.prompt_version
    assert call.payload["data"]["goal"] == project.goal
    assert call.cost_usd == pytest.approx(out.cost_usd) and call.cost_usd > 0
    sent = fake.calls("architect")[0]
    assert sent.output_schema is not None and datos(sent)["constraints"] == {"k": 1}


def test_retry_with_validation_feedback(ctx: EngineContext, project: Project) -> None:
    fake = FakeLLMProvider().script("architect", {"value": "no-es-int"}, {"value": 1}, {"value": 2})
    gw = _gateway(ctx, fake)

    def validator(a: Answer) -> str | None:
        return None if a.value % 2 == 0 else "value debe ser par"

    out = gw.structured(
        LLMPurpose.ARCHITECT,
        Answer,
        LLMContext(),
        project=project,
        validator=validator,
        prompt_vars={"n_min": 2, "n_max": 3},
    )
    assert out.value.value == 2 and out.attempts == 3
    third = fake.calls("architect")[2]
    assert "value debe ser par" in third.messages[-1].content
    calls = list(ctx.repo(LLMCall).list(filters={"project_id": project.id}))
    assert [c.status for c in sorted(calls, key=lambda c: c.attempt)] == [
        "invalid",
        "invalid",
        "ok",
    ]


def test_invalid_after_max_attempts_keeps_last_value(ctx: EngineContext, project: Project) -> None:
    fake = FakeLLMProvider().script("architect", {"value": 1})
    gw = _gateway(ctx, fake)
    with pytest.raises(LLMOutputInvalidError) as exc:
        gw.structured(
            LLMPurpose.ARCHITECT,
            Answer,
            LLMContext(),
            project=project,
            validator=lambda a: "nunca",
            prompt_vars={"n_min": 2, "n_max": 3},
        )
    assert exc.value.last_value == Answer(value=1) and exc.value.details["call_id"]
    assert len(fake.calls()) == 3


def test_cache_makes_repeated_call_free(ctx: EngineContext, project: Project) -> None:
    fake = FakeLLMProvider().script("reporter", {"value": 7})
    gw = _gateway(ctx, fake)

    def call() -> Any:
        return gw.structured(
            LLMPurpose.REPORTER,
            Answer,
            LLMContext(goal="x"),
            project=project,
            prompt_vars={"language": "español"},
        )

    first, second = call(), call()
    assert second.value == first.value and second.cache_hit and second.cost_usd == 0
    assert len(fake.calls()) == 1


def test_budget_cuts_before_calling(ctx: EngineContext, project: Project) -> None:
    fake = FakeLLMProvider().script("architect", {"value": 1})
    gw = _gateway(ctx, fake, settings=LLMSettings(enabled=True, project_budget_usd=0.0))
    with pytest.raises(LLMBudgetExceededError):
        gw.structured(
            LLMPurpose.ARCHITECT,
            Answer,
            LLMContext(),
            project=project,
            prompt_vars={"n_min": 2, "n_max": 3},
        )
    assert not fake.calls()


def test_scope_budget(ctx: EngineContext, project: Project) -> None:
    # 1M tokens de entrada al precio del catálogo: la primera llamada gasta ~1 USD.
    fake = FakeLLMProvider(usage=Usage(input_tokens=1_000_000, output_tokens=0))
    fake.script("reporter", {"value": 1}, {"value": 2})
    gw = _gateway(ctx, fake, settings=LLMSettings(enabled=True, cache=False))
    kwargs: dict[str, Any] = {
        "project": project,
        "prompt_vars": {"language": "español"},
        "scope": "agent:1",
        "scope_budget_usd": 0.5,
    }
    gw.structured(LLMPurpose.REPORTER, Answer, LLMContext(), **kwargs)
    with pytest.raises(LLMBudgetExceededError):
        gw.structured(LLMPurpose.REPORTER, Answer, LLMContext(goal="otra"), **kwargs)


def test_policy_unavailable_cases(ctx: EngineContext) -> None:
    fake = FakeLLMProvider()
    l0 = Project(name="l0", privacy_level=PrivacyLevel.L0)
    assert not _gateway(ctx, fake).available(LLMPurpose.ARCHITECT, l0)
    ok = Project(name="ok")
    off = _gateway(ctx, fake, settings=LLMSettings(enabled=False))
    with pytest.raises(LLMUnavailableError) as exc:
        off.resolve(LLMPurpose.ARCHITECT, ok)
    assert exc.value.details["reason"] == "disabled"
    # Sin fábrica falsa hace falta la clave real del proveedor.
    real = _gateway(ctx, fake, factory=False)
    with pytest.raises(LLMUnavailableError) as exc:
        real.resolve(LLMPurpose.ARCHITECT, ok)
    assert exc.value.details["reason"] == "missing_key"
    keyed = _gateway(ctx, fake, secrets=MemorySecrets({"ANTHROPIC_API_KEY": "k"}), factory=False)
    assert keyed.available(LLMPurpose.ARCHITECT, ok)
    ctx.repo(Workspace).add(Workspace(name="ws", allowed_llm_providers=["ollama"]))
    with pytest.raises(LLMUnavailableError) as exc:
        keyed.resolve(LLMPurpose.ARCHITECT, ok)
    assert exc.value.details["reason"] == "provider_not_allowed"


def test_workspace_caps_privacy_and_local_override(ctx: EngineContext) -> None:
    gw = _gateway(ctx, FakeLLMProvider())
    ctx.repo(Workspace).add(
        Workspace(
            name="ws", max_privacy_level=PrivacyLevel.L1, local_llm_max_privacy=PrivacyLevel.L3
        )
    )
    p = Project(name="p", privacy_level=PrivacyLevel.L3)
    assert gw.effective_level(p, local=False) is PrivacyLevel.L1
    assert gw.effective_level(p, local=True) is PrivacyLevel.L3
    assert gw.effective_level(
        p.model_copy(update={"privacy_level": PrivacyLevel.L2}), local=True
    ) is (PrivacyLevel.L2)


def test_text_mode_for_models_without_structured_output(
    ctx: EngineContext, project: Project
) -> None:
    fake = FakeLLMProvider().script(
        "architect", 'Claro, acá va:\n```json\n{"value": 5, "label": "texto"}\n```'
    )
    settings = LLMSettings(
        enabled=True,
        profile="compat",
        providers={"openai_compat": LLMProviderSettings(kind="openai_compat")},
    )
    gw = _gateway(ctx, fake, settings=settings)
    cfg = gw.config.workspace()
    cfg.profiles["compat"] = gw.config.profiles()["ollama"].model_copy(
        update={"provider": "openai_compat"}, deep=True
    )
    for ref in cfg.profiles["compat"].purposes.values():
        ref.model = "local-model"
    gw.config.save_workspace(cfg)
    out = gw.structured(
        LLMPurpose.ARCHITECT,
        Answer,
        LLMContext(),
        project=project,
        prompt_vars={"n_min": 2, "n_max": 3},
    )
    assert out.value.value == 5 and out.provider == "openai_compat"
    assert "JSON Schema" in fake.calls()[0].system


def test_provider_error_is_audited_and_raised(ctx: EngineContext, project: Project) -> None:
    fake = FakeLLMProvider().script("architect", LLMProviderError("caído"))
    gw = _gateway(ctx, fake)
    with pytest.raises(LLMProviderError):
        gw.structured(
            LLMPurpose.ARCHITECT,
            Answer,
            LLMContext(),
            project=project,
            prompt_vars={"n_min": 2, "n_max": 3},
        )
    [call] = ctx.repo(LLMCall).list(filters={"project_id": project.id})
    assert call.status == "error" and call.error == "caído"


def test_stream_chat_yields_and_audits(ctx: EngineContext, project: Project) -> None:
    fake = FakeLLMProvider().script("copilot", "Hola, te ayudo con el objetivo.")
    gw = _gateway(ctx, fake)
    pieces = list(
        gw.stream_chat(
            LLMPurpose.COPILOT, LLMContext(), project=project, prompt_vars={"question": "¿qué?"}
        )
    )
    assert "".join(pieces) == "Hola, te ayudo con el objetivo." and len(pieces) > 1
    [call] = ctx.repo(LLMCall).list(filters={"project_id": project.id})
    assert call.purpose is LLMPurpose.COPILOT


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('texto {"a": {"b": "}"}} más') == {"a": {"b": "}"}}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    with pytest.raises(ValueError, match="JSON"):
        extract_json("sin json")


def test_encrypted_secrets_roundtrip(tmp_path: Path) -> None:
    key = EncryptedFileSecrets.generate_key()
    import base64

    store = EncryptedFileSecrets(tmp_path / "s.json", base64.b64decode(key))
    store.set("ANTHROPIC_API_KEY", "sk-secreto")
    assert "sk-secreto" not in (tmp_path / "s.json").read_text(encoding="utf-8")
    again = EncryptedFileSecrets(tmp_path / "s.json", base64.b64decode(key))
    assert again.get("ANTHROPIC_API_KEY") == "sk-secreto"
    again.delete("ANTHROPIC_API_KEY")
    assert again.get("ANTHROPIC_API_KEY") is None


def test_compact_payload_for_local_models(ctx: EngineContext, project: Project) -> None:
    from perceptron.llm.compact import compact_payload

    data = {
        "card": {
            "columns": [{"numeric": {"histogram": [1, 2], "quantiles": {"p05": 1, "p50": 2}}}]
        },
        "runs": [{"history": [{"epoch": float(i)} for i in range(30)]} for _ in range(6)],
        "catalog": [{"key": f"b{i}", "description": "larga"} for i in range(15)],
    }
    out = compact_payload(data)
    col = out["card"]["columns"][0]["numeric"]
    assert "histogram" not in col and col["quantiles"] == {"p50": 2}
    assert len(out["runs"]) == 4 and len(out["runs"][0]["history"]) == 8
    assert len(out["catalog"]) == 15 and "description" not in out["catalog"][0]

    fake = FakeLLMProvider().script("architect", {"value": 1})
    gw = _gateway(ctx, fake, settings=LLMSettings(enabled=True, profile="ollama"))
    assert gw.compact(LLMPurpose.ARCHITECT, project)
    card_ctx = LLMContext(evidence=[{"i": i} for i in range(10)])
    out_call = gw.structured(
        LLMPurpose.ARCHITECT,
        Answer,
        card_ctx,
        project=project,
        prompt_vars={"n_min": 2, "n_max": 2},
    )
    assert len(datos(fake.calls()[0])["evidence"]) == 4 and "compacto" in out_call.redactions
    assert not _gateway(ctx, fake).compact(LLMPurpose.ARCHITECT, project)  # Claude: completo
