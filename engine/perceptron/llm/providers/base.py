"""Interfaz única de proveedor (RF-LLM-01)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import httpx

from perceptron.llm.config import ModelInfo
from perceptron.llm.types import LLMRequest, LLMResponse


class LLMProvider(ABC):
    """Un proveedor sabe hablar con su API; no conoce propósitos ni privacidad."""

    kind: str = "base"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.http_client = http_client

    @abstractmethod
    def complete(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        """Una respuesta completa; estructurada si hay `output_schema` y el modelo lo soporta."""

    def stream(self, request: LLMRequest, model: ModelInfo) -> Iterator[str]:
        """Fragmentos de texto (RF-LLM-05). Por defecto, la respuesta completa de una vez."""
        yield self.complete(request, model).text

    def _client(self) -> httpx.Client:
        return self.http_client or httpx.Client(timeout=self.timeout_s)
