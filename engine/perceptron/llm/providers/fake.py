"""Proveedor falso determinístico para tests unitarios, integración y CI (SPEC §15.3).

Responde por propósito con una cola de respuestas grabadas. Cada respuesta es un
dict (JSON estructurado), un str (texto) o un callable `request -> dict | str`
(útil para armar la respuesta a partir de lo que el prompt pidió). Si la cola de un
propósito se vacía, repite la última. Registra cada request para las aserciones.

Cassette JSON: `{"<propósito>": [<respuesta>, ...], "_usage": {"input": 100, "output": 50}}`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from perceptron.llm.config import ModelInfo
from perceptron.llm.errors import LLMProviderError
from perceptron.llm.providers.base import LLMProvider
from perceptron.llm.types import LLMRequest, LLMResponse, Usage

Scripted = dict[str, Any] | str | Callable[[LLMRequest], dict[str, Any] | str] | Exception


class FakeLLMProvider(LLMProvider):
    kind = "fake"

    def __init__(
        self,
        responses: dict[str, list[Scripted]] | None = None,
        *,
        usage: Usage | None = None,
    ) -> None:
        super().__init__()
        self.responses: dict[str, list[Scripted]] = {
            k: list(v) for k, v in (responses or {}).items()
        }
        self.usage = usage or Usage(input_tokens=100, output_tokens=50)
        self.requests: list[LLMRequest] = []
        self._last: dict[str, Scripted] = {}

    @classmethod
    def from_cassette(cls, path: Path) -> FakeLLMProvider:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        u = data.pop("_usage", {})
        usage = Usage(input_tokens=int(u.get("input", 100)), output_tokens=int(u.get("output", 50)))
        return cls({k: list(v) for k, v in data.items()}, usage=usage)

    def script(self, purpose: str, *responses: Scripted) -> FakeLLMProvider:
        self.responses.setdefault(purpose, []).extend(responses)
        return self

    def calls(self, purpose: str | None = None) -> list[LLMRequest]:
        return [r for r in self.requests if purpose is None or r.purpose == purpose]

    def _next(self, request: LLMRequest) -> Scripted:
        key = request.purpose or "default"
        queue = self.responses.get(key) or self.responses.get("default")
        if queue:
            item = queue.pop(0)
            self._last[key] = item
            return item
        if key in self._last:
            return self._last[key]
        raise LLMProviderError(f"FakeLLMProvider: sin respuesta grabada para '{key}'")

    def complete(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        self.requests.append(request)
        item = self._next(request)
        if isinstance(item, Exception):
            raise item
        value = item(request) if callable(item) else item
        if isinstance(value, dict):
            structured = request.output_schema is not None and model.structured_output
            return LLMResponse(
                text="" if structured else json.dumps(value, ensure_ascii=False),
                data=value if structured else None,
                usage=self.usage,
                model=request.model,
                stop_reason="end_turn",
            )
        return LLMResponse(
            text=value, usage=self.usage, model=request.model, stop_reason="end_turn"
        )

    def stream(self, request: LLMRequest, model: ModelInfo) -> Iterator[str]:
        text = self.complete(request, model).text
        for i in range(0, len(text), 16):
            yield text[i : i + 16]
