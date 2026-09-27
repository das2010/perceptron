"""Anthropic Claude (proveedor por defecto, SPEC §2). Salida estructurada vía tool forzado."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from perceptron.llm.config import ModelInfo
from perceptron.llm.errors import LLMProviderError
from perceptron.llm.providers.base import LLMProvider
from perceptron.llm.types import LLMRequest, LLMResponse, Message, Usage


def _content(msg: Message) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": img.media_type, "data": img.data_b64},
        }
        for img in msg.images
    ]
    parts.append({"type": "text", "text": msg.content})
    return parts


class AnthropicProvider(LLMProvider):
    kind = "anthropic"

    def _sdk(self) -> Any:
        import anthropic

        return anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            http_client=self.sdk_http_client,
            max_retries=2,
        )

    def _kwargs(self, request: LLMRequest, model: ModelInfo) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "system": request.system,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": _content(m)} for m in request.messages],
        }
        # La API de mensajes actual de Anthropic no acepta `temperature`: no se envía.
        if request.output_schema is not None and model.structured_output:
            kwargs["tools"] = [
                {
                    "name": request.schema_name,
                    "description": "Devolvé la respuesta completa con este schema.",
                    "input_schema": request.output_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": request.schema_name}
        return kwargs

    def complete(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        import anthropic

        try:
            msg = self._sdk().messages.create(**self._kwargs(request, model))
        except anthropic.APIError as e:
            raise LLMProviderError(f"Anthropic: {e}", details={"provider": self.kind}) from e
        text, data = "", None
        for block in msg.content:
            if block.type == "tool_use":
                data = dict(block.input)
            elif block.type == "text":
                text += block.text
        return LLMResponse(
            text=text,
            data=data,
            usage=Usage(
                input_tokens=int(msg.usage.input_tokens), output_tokens=int(msg.usage.output_tokens)
            ),
            model=str(msg.model),
            stop_reason=msg.stop_reason,
        )

    def stream(self, request: LLMRequest, model: ModelInfo) -> Iterator[str]:
        import anthropic

        kwargs = self._kwargs(request.model_copy(update={"output_schema": None}), model)
        try:
            with self._sdk().messages.stream(**kwargs) as s:
                yield from s.text_stream
        except anthropic.APIError as e:
            raise LLMProviderError(f"Anthropic: {e}", details={"provider": self.kind}) from e
