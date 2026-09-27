"""Adaptadores de proveedores contra servidores HTTP simulados (sin red, RF-LLM-01/02/05)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from perceptron.llm.config import ModelInfo, ProviderInfo
from perceptron.llm.errors import LLMProviderError
from perceptron.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    MoonshotProvider,
    OllamaProvider,
    OpenAIProvider,
    build_provider,
)
from perceptron.llm.types import ImagePart, LLMRequest, Message

SCHEMA = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
STRUCT = ModelInfo(structured_output=True, vision=True)
PLAIN = ModelInfo(structured_output=False)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, seen: list[httpx.Request]) -> httpx.Client:
    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(wrapped))


def _req(schema: dict[str, Any] | None = SCHEMA, *, images: bool = False) -> LLMRequest:
    img = [ImagePart(data_b64="aGVsbG8=")] if images else []
    return LLMRequest(
        model="m-1",
        system="sistema",
        messages=[Message(role="user", content="hola", images=img)],
        output_schema=schema,
        schema_name="architect",
        max_tokens=256,
    )


def _body(r: httpx.Request) -> dict[str, Any]:
    return json.loads(r.content)


def test_anthropic_forced_tool() -> None:
    seen: list[httpx.Request] = []

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m-1",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "architect", "input": {"value": 4}}
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    p = AnthropicProvider(
        api_key="k", base_url="http://anthropic.test", http_client=_client(handler, seen)
    )
    out = p.complete(_req(images=True), STRUCT)
    assert out.data == {"value": 4} and out.usage.input_tokens == 11
    body = _body(seen[0])
    assert body["tool_choice"] == {"type": "tool", "name": "architect"}
    assert body["tools"][0]["input_schema"] == SCHEMA and body["system"] == "sistema"
    assert body["messages"][0]["content"][0]["type"] == "image"
    assert seen[0].headers["x-api-key"] == "k"


def test_anthropic_error_is_mapped() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "mal"}},
        )

    p = AnthropicProvider(
        api_key="k", base_url="http://anthropic.test", http_client=_client(handler, [])
    )
    with pytest.raises(LLMProviderError):
        p.complete(_req(), STRUCT)


def _chat_response(content: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }


@pytest.mark.parametrize(
    ("cls", "tokens_field"),
    [(OpenAIProvider, "max_completion_tokens"), (MoonshotProvider, "max_tokens")],
)
def test_openai_family_json_schema(cls: type[OpenAIProvider], tokens_field: str) -> None:
    seen: list[httpx.Request] = []
    p = cls(
        api_key="k",
        base_url="http://openai.test/v1",
        http_client=_client(
            lambda _: httpx.Response(200, json=_chat_response('{"value": 2}')), seen
        ),
    )
    out = p.complete(_req(images=True), STRUCT)
    assert out.data == {"value": 2} and out.usage.output_tokens == 3
    body = _body(seen[0])
    assert body[tokens_field] == 256
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["messages"][0] == {"role": "system", "content": "sistema"}
    assert body["messages"][1]["content"][1]["type"] == "image_url"
    assert seen[0].url.path.endswith("/chat/completions")


def test_openai_without_structured_output_uses_json_mode() -> None:
    seen: list[httpx.Request] = []
    p = OpenAIProvider(
        base_url="http://compat.test/v1",
        http_client=_client(lambda _: httpx.Response(200, json=_chat_response("texto")), seen),
    )
    out = p.complete(_req(), PLAIN)
    assert out.data is None and out.text == "texto"
    assert _body(seen[0])["response_format"] == {"type": "json_object"}


def test_ollama_format_schema_and_stream() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        if _body(r)["stream"]:
            lines = [
                json.dumps({"message": {"content": "Ho"}, "done": False}),
                json.dumps({"message": {"content": "la"}, "done": True}),
            ]
            return httpx.Response(200, content="\n".join(lines).encode())
        return httpx.Response(
            200,
            json={
                "model": "m-1",
                "message": {"role": "assistant", "content": '{"value": 9}'},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        )

    p = OllamaProvider(base_url="http://127.0.0.1:11434", http_client=_client(handler, seen))
    out = p.complete(_req(images=True), STRUCT)
    assert out.data == {"value": 9} and out.usage.input_tokens == 12
    body = _body(seen[0])
    assert body["format"] == SCHEMA and body["options"]["num_predict"] == 256
    assert body["messages"][1]["images"] == ["aGVsbG8="]
    assert "".join(p.stream(_req(None), STRUCT)) == "Hola"


def test_gemini_rest_and_sse() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        if "streamGenerateContent" in r.url.path:
            chunk = {"candidates": [{"content": {"parts": [{"text": "Hola"}]}}]}
            return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\n".encode())
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"value": 5}'}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2},
            },
        )

    p = GeminiProvider(
        api_key="g", base_url="http://gemini.test/v1beta", http_client=_client(handler, seen)
    )
    out = p.complete(_req(), STRUCT)
    assert out.data == {"value": 5} and out.usage.input_tokens == 8
    body = _body(seen[0])
    assert body["generationConfig"]["responseJsonSchema"] == SCHEMA
    assert seen[0].headers["x-goog-api-key"] == "g"
    assert seen[0].url.path.endswith("/models/m-1:generateContent")
    assert "".join(p.stream(_req(None), STRUCT)) == "Hola"


def test_http_errors_are_mapped() -> None:
    p = OllamaProvider(http_client=_client(lambda _: httpx.Response(500, text="boom"), []))
    with pytest.raises(LLMProviderError):
        p.complete(_req(), STRUCT)


def test_factory_builds_every_kind() -> None:
    for kind in ("anthropic", "openai", "moonshot", "openai_compat", "gemini", "ollama"):
        p = build_provider(ProviderInfo(kind=kind), "k")  # type: ignore[arg-type]
        assert p.kind == kind
