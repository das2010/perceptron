"""Ollama (LLM local). API nativa `/api/chat`: `format` acepta un JSON Schema."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from perceptron.llm.config import ModelInfo
from perceptron.llm.errors import LLMProviderError
from perceptron.llm.providers.base import LLMProvider
from perceptron.llm.types import LLMRequest, LLMResponse, Usage

DEFAULT_URL = "http://127.0.0.1:11434"
MAX_CTX = 32_768


class OllamaProvider(LLMProvider):
    kind = "ollama"

    def _body(self, request: LLMRequest, model: ModelInfo, *, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": request.system}]
        for m in request.messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.images:
                msg["images"] = [i.data_b64 for i in m.images]
            messages.append(msg)
        # Ollama usa un contexto chico por defecto y trunca en silencio: se fija el del modelo.
        options: dict[str, Any] = {
            "num_predict": request.max_tokens,
            "num_ctx": min(model.context_window, MAX_CTX),
        }
        if request.temperature is not None and model.temperature:
            options["temperature"] = request.temperature
        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if request.output_schema is not None and not stream:
            body["format"] = request.output_schema if model.structured_output else "json"
        body.update(model.request_extra)
        return body

    def _url(self) -> str:
        return f"{(self.base_url or DEFAULT_URL).rstrip('/')}/api/chat"

    def complete(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        try:
            r = self._client().post(self._url(), json=self._body(request, model, stream=False))
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMProviderError(f"Ollama: {e}", details={"provider": self.kind}) from e
        payload: dict[str, Any] = r.json()
        text = str(payload.get("message", {}).get("content", ""))
        data = None
        if request.output_schema is not None:
            try:
                parsed = json.loads(text)
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = None
        return LLMResponse(
            text=text,
            data=data,
            usage=Usage(
                input_tokens=int(payload.get("prompt_eval_count") or 0),
                output_tokens=int(payload.get("eval_count") or 0),
            ),
            model=str(payload.get("model", request.model)),
            stop_reason=payload.get("done_reason"),
        )

    def stream(self, request: LLMRequest, model: ModelInfo) -> Iterator[str]:
        try:
            with self._client().stream(
                "POST", self._url(), json=self._body(request, model, stream=True)
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    piece = json.loads(line).get("message", {}).get("content", "")
                    if piece:
                        yield str(piece)
        except httpx.HTTPError as e:
            raise LLMProviderError(f"Ollama: {e}", details={"provider": self.kind}) from e
