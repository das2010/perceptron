from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from perceptron.api.app import create_app
from perceptron.api.context import EngineContext
from perceptron.core.config import LLMSettings, LoggingSettings, Settings
from perceptron.llm.config import LLMConfig
from perceptron.llm.gateway import Gateway
from perceptron.llm.providers.fake import FakeLLMProvider
from perceptron.llm.secrets import MemorySecrets

# Rutas con espacios y caracteres no ASCII (SPEC §13.5).
AWKWARD_DIR = "Carpeta con ñ, acentos (áéí) y espacios"

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


LLM_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")


@pytest.fixture(autouse=True)
def _hermetic_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ningún test llama a un LLM real: capa apagada y sin claves del entorno."""
    monkeypatch.setenv("PERCEPTRON_LLM__ENABLED", "false")
    for key in LLM_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    return tmp_path / AWKWARD_DIR / "workspace"


@pytest.fixture
def settings(workspace_dir: Path) -> Settings:
    return Settings(workspace_dir=workspace_dir, logging=LoggingSettings(to_file=False))


@pytest.fixture
def ctx(settings: Settings) -> Iterator[EngineContext]:
    c = EngineContext.create(settings)
    yield c
    c.close()


@pytest.fixture
def client(ctx: EngineContext) -> Iterator[TestClient]:
    with TestClient(create_app(ctx=ctx)) as c:
        yield c


@pytest.fixture
def fake_llm(ctx: EngineContext) -> FakeLLMProvider:
    """Gateway real (privacidad, prompts, reintentos, caché, auditoría) con proveedor falso."""
    fake = FakeLLMProvider()
    config = LLMConfig(LLMSettings(enabled=True), ctx.settings.workspace_dir)
    ctx.use_llm(Gateway(ctx.db, config, MemorySecrets(), provider_factory=lambda *_: fake))
    return fake
