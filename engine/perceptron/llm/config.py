"""Catálogo de modelos y perfiles por propósito (RF-LLM-02, RF-LLM-03).

Tres capas, de menor a mayor prioridad:
1. `catalog.yaml` empaquetado (o `PERCEPTRON_LLM__CATALOG_FILE`);
2. `LLMSettings.providers` (variables de entorno / `.env`);
3. `<workspace>/llm.json`, editable por la API (`PUT /llm/providers|profiles`).

Ningún ID de modelo está en el código: todos salen de aquí. Las claves nunca se
guardan en estos archivos, solo su referencia (RF-LLM-08).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field

from perceptron.core.config import LLMProviderSettings, LLMSettings
from perceptron.domain.enums import LLMPurpose
from perceptron.storage.filesystem import atomic_write_text

CATALOG_FILE = Path(__file__).with_name("catalog.yaml")
WORKSPACE_FILE = "llm.json"

ProviderKind = Literal[
    "anthropic", "openai", "gemini", "moonshot", "openai_compat", "ollama", "fake"
]
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})  # noqa: S104


class ModelInfo(BaseModel):
    """Capacidades declaradas de un modelo: el sistema degrada funciones según ellas."""

    model_config = ConfigDict(extra="forbid")

    structured_output: bool = True
    tool_use: bool = False
    vision: bool = False
    context_window: int = Field(default=32_768, ge=1)
    input_per_mtok: float = Field(default=0.0, ge=0)
    output_per_mtok: float = Field(default=0.0, ge=0)
    temperature: bool = Field(default=True, description="Acepta temperatura explícita")
    license: str | None = None
    verified: bool = False

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class ProviderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind
    base_url: str | None = None
    api_key_ref: str | None = None
    local: bool | None = None
    timeout_s: float = 120.0
    models: dict[str, ModelInfo] = Field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        if self.local is not None:
            return self.local
        if self.kind == "fake":
            return True
        host = urlparse(self.base_url or "").hostname
        return host in LOCAL_HOSTS

    def model(self, model_id: str) -> ModelInfo:
        """Modelos fuera del catálogo se aceptan con capacidades conservadoras."""
        return self.models.get(model_id) or ModelInfo(structured_output=False)


class PurposeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str | None = Field(default=None, description="Otro proveedor para este propósito")
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=64, le=64_000)
    prompt_version: str | None = None


class LLMProfile(BaseModel):
    """Qué modelo atiende cada propósito (RF-LLM-03)."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    purposes: dict[LLMPurpose, PurposeRef] = Field(default_factory=dict)


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    default_profile: str
    providers: dict[str, ProviderInfo]
    profiles: dict[str, LLMProfile] = Field(default_factory=dict)


class WorkspaceLLMConfig(BaseModel):
    """Lo que el usuario/Admin editó (`<workspace>/llm.json`); sin secretos."""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderInfo] = Field(default_factory=dict)
    profiles: dict[str, LLMProfile] = Field(default_factory=dict)
    active_profile: str | None = None


class Resolved(BaseModel):
    """Proveedor + modelo concretos para un propósito."""

    purpose: LLMPurpose
    profile: str
    provider_name: str
    provider: ProviderInfo
    model_id: str
    model: ModelInfo
    ref: PurposeRef


def load_catalog(path: Path | None = None) -> Catalog:
    data = yaml.safe_load((path or CATALOG_FILE).read_text(encoding="utf-8"))
    return Catalog.model_validate(data)


def _merge_provider(base: ProviderInfo | None, override: LLMProviderSettings) -> ProviderInfo:
    data: dict[str, Any] = base.model_dump() if base else {"models": {}}
    data.update({k: v for k, v in override.model_dump().items() if v is not None})
    return ProviderInfo.model_validate(data)


class LLMConfig:
    """Vista combinada de catálogo + settings + workspace."""

    def __init__(self, settings: LLMSettings, workspace_dir: Path | None = None) -> None:
        self.settings = settings
        self.workspace_file = workspace_dir / WORKSPACE_FILE if workspace_dir else None
        self.catalog = load_catalog(settings.catalog_file)

    # ------------------------------------------------------------------ persistencia

    def workspace(self) -> WorkspaceLLMConfig:
        if self.workspace_file is None or not self.workspace_file.is_file():
            return WorkspaceLLMConfig()
        return WorkspaceLLMConfig.model_validate_json(
            self.workspace_file.read_text(encoding="utf-8")
        )

    def save_workspace(self, cfg: WorkspaceLLMConfig) -> None:
        if self.workspace_file is None:
            raise RuntimeError("sin workspace para guardar la configuración LLM")
        self.workspace_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.workspace_file, json.dumps(cfg.model_dump(mode="json"), indent=2))

    # ------------------------------------------------------------------ vistas

    def providers(self) -> dict[str, ProviderInfo]:
        out = dict(self.catalog.providers)
        for name, ov in self.settings.providers.items():
            out[name] = _merge_provider(out.get(name), ov)
        out.update(self.workspace().providers)
        return out

    def profiles(self) -> dict[str, LLMProfile]:
        return {**self.catalog.profiles, **self.workspace().profiles}

    def active_profile(self) -> str:
        return (
            self.settings.profile
            or self.workspace().active_profile
            or (self.catalog.default_profile)
        )

    def resolve(self, purpose: LLMPurpose, profile: str | None = None) -> Resolved | None:
        """Proveedor y modelo para el propósito; None si el perfil no lo cubre."""
        name = profile or self.active_profile()
        prof = self.profiles().get(name)
        if prof is None:
            return None
        ref = prof.purposes.get(purpose) or prof.purposes.get(LLMPurpose.COPILOT)
        if ref is None:
            return None
        provider_name = ref.provider or prof.provider
        provider = self.providers().get(provider_name)
        if provider is None:
            return None
        return Resolved(
            purpose=purpose,
            profile=name,
            provider_name=provider_name,
            provider=provider,
            model_id=ref.model,
            model=provider.model(ref.model),
            ref=ref,
        )
