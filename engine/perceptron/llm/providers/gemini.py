"""Google Gemini vía REST (`generateContent`); sin SDK para mantener liviana la instalación."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from perceptron.llm.config import ModelInfo
from perceptron.llm.errors import LLMProviderError
from perceptron.llm.providers.base import LLMProvider
from perceptron.llm.types import LLMRequest, LLMResponse, Usage

DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    kind = "gemini"

    def _body(self, request: LLMRequest, model: ModelInfo) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for m in request.messages:
            parts: list[dict[str, Any]] = [{"text": m.content}]
            parts += [
                {"inlineData": {"mimeType": i.media_type, "data": i.data_b64}} for i in m.images
            ]
            contents.append({"role": "model" if m.role == "assistant" else "user", "parts": parts})
        config: dict[str, Any] = {"maxOutputTokens": request.max_tokens}
        if request.temperature is not None and model.temperature:
            config["temperature"] = request.temperature
        if request.output_schema is not None:
            config["responseMimeType"] = "application/json"
            if model.structured_output:
                config["responseJsonSchema"] = request.output_schema
        return {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": contents,
            "generationConfig": config,
        }

    def _url(self, model_id: str, method: str) -> str:
        return f"{(self.base_url or DEFAULT_URL).rstrip('/')}/models/{model_id}:{method}"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key or ""}

    @staticmethod
    def _text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(str(p.get("text", "")) for p in parts)

    def complete(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        try:
            r = self._client().post(
                self._url(request.model, "generateContent"),
                json=self._body(request, model),
                headers=self._headers(),
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMProviderError(f"Gemini: {e}", details={"provider": self.kind}) from e
        payload: dict[str, Any] = r.json()
        text = self._text(payload)
        data = None
        if request.output_schema is not None and text:
            try:
                parsed = json.loads(text)
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = None
        usage = payload.get("usageMetadata", {})
        candidates = payload.get("candidates") or [{}]
        return LLMResponse(
            text=text,
            data=data,
            usage=Usage(
                input_tokens=int(usage.get("promptTokenCount") or 0),
                output_tokens=int(usage.get("candidatesTokenCount") or 0),
            ),
            model=request.model,
            stop_reason=candidates[0].get("finishReason"),
        )

    def stream(self, request: LLMRequest, model: ModelInfo) -> Iterator[str]:
        body = self._body(request.model_copy(update={"output_schema": None}), model)
        try:
            with self._client().stream(
                "POST",
                self._url(request.model, "streamGenerateContent") + "?alt=sse",
                json=body,
                headers=self._headers(),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        piece = self._text(json.loads(line[5:]))
                        if piece:
                            yield piece
        except httpx.HTTPError as e:
            raise LLMProviderError(f"Gemini: {e}", details={"provider": self.kind}) from e
