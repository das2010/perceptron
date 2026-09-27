"""OpenAI, Kimi/Moonshot y servidores compatibles (LM Studio, vLLM, llama.cpp).

Moonshot y los compatibles hablan la misma API de chat: cambian `base_url` y a veces
el soporte de `json_schema` (se declara en el catálogo como `structured_output`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from perceptron.llm.config import ModelInfo
from perceptron.llm.errors import LLMProviderError
from perceptron.llm.providers.base import LLMProvider
from perceptron.llm.types import LLMRequest, LLMResponse, Message, Usage


def _content(msg: Message) -> str | list[dict[str, Any]]:
    if not msg.images:
        return msg.content
    parts: list[dict[str, Any]] = [{"type": "text", "text": msg.content}]
    parts += [
        {"type": "image_url", "image_url": {"url": f"data:{i.media_type};base64,{i.data_b64}"}}
        for i in msg.images
    ]
    return parts


class OpenAIProvider(LLMProvider):
    kind = "openai"
    # La API de OpenAI renombró `max_tokens`; los compatibles siguen con el nombre viejo.
    max_tokens_field = "max_completion_tokens"

    def _sdk(self) -> Any:
        import openai

        return openai.OpenAI(
            api_key=self.api_key or "sin-clave",
            base_url=self.base_url,
            timeout=self.timeout_s,
            http_client=self.http_client,
            max_retries=2,
        )

    def _kwargs(self, request: LLMRequest, model: ModelInfo) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": request.system}]
        messages += [{"role": m.role, "content": _content(m)} for m in request.messages]
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            self.max_tokens_field: request.max_tokens,
        }
        if request.temperature is not None and model.temperature:
            kwargs["temperature"] = request.temperature
        if request.output_schema is not None:
            if model.structured_output:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.schema_name,
                        "schema": request.output_schema,
                        "strict": False,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def complete(self, request: LLMRequest, model: ModelInfo) -> LLMResponse:
        import openai

        try:
            resp = self._sdk().chat.completions.create(**self._kwargs(request, model))
        except openai.OpenAIError as e:
            raise LLMProviderError(f"{self.kind}: {e}", details={"provider": self.kind}) from e
        if not resp.choices:
            raise LLMProviderError(f"{self.kind}: respuesta sin opciones")
        choice = resp.choices[0]
        text = choice.message.content or ""
        data = None
        if request.output_schema is not None and text:
            try:
                parsed = json.loads(text)
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = None  # el gateway intenta extraerlo del texto
        usage = resp.usage
        return LLMResponse(
            text=text,
            data=data,
            usage=Usage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            model=str(resp.model),
            stop_reason=choice.finish_reason,
        )

    def stream(self, request: LLMRequest, model: ModelInfo) -> Iterator[str]:
        import openai

        kwargs = self._kwargs(request.model_copy(update={"output_schema": None}), model)
        try:
            for chunk in self._sdk().chat.completions.create(**kwargs, stream=True):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as e:
            raise LLMProviderError(f"{self.kind}: {e}", details={"provider": self.kind}) from e


class MoonshotProvider(OpenAIProvider):
    kind = "moonshot"
    max_tokens_field = "max_tokens"


class OpenAICompatProvider(OpenAIProvider):
    kind = "openai_compat"
    max_tokens_field = "max_tokens"
