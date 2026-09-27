"""Adaptadores de proveedores LLM (RF-LLM-01) y su fábrica."""

from __future__ import annotations

import httpx

from perceptron.llm.config import ProviderInfo
from perceptron.llm.providers.anthropic import AnthropicProvider
from perceptron.llm.providers.base import LLMProvider
from perceptron.llm.providers.fake import FakeLLMProvider
from perceptron.llm.providers.gemini import GeminiProvider
from perceptron.llm.providers.ollama import OllamaProvider
from perceptron.llm.providers.openai import MoonshotProvider, OpenAICompatProvider, OpenAIProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "moonshot": MoonshotProvider,
    "openai_compat": OpenAICompatProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}

# Proveedores que requieren clave (los locales no).
NEEDS_KEY = frozenset({"anthropic", "openai", "moonshot", "gemini"})


def build_provider(
    info: ProviderInfo, api_key: str | None, *, http_client: httpx.Client | None = None
) -> LLMProvider:
    cls = PROVIDERS.get(info.kind)
    if cls is None:
        raise ValueError(f"proveedor desconocido: {info.kind}")
    return cls(
        api_key=api_key, base_url=info.base_url, timeout_s=info.timeout_s, http_client=http_client
    )


__all__ = [
    "NEEDS_KEY",
    "PROVIDERS",
    "AnthropicProvider",
    "FakeLLMProvider",
    "GeminiProvider",
    "LLMProvider",
    "MoonshotProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "OpenAIProvider",
    "build_provider",
]
