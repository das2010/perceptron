from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from perceptron.api.app import create_app
from perceptron.api.context import EngineContext
from perceptron.core.config import LoggingSettings, Settings

# Rutas con espacios y caracteres no ASCII (SPEC §13.5).
AWKWARD_DIR = "Carpeta con ñ, acentos (áéí) y espacios"

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


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
