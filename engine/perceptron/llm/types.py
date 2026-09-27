"""Tipos comunes a todos los proveedores (RF-LLM-01)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ImagePart(BaseModel):
    media_type: str = "image/png"
    data_b64: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    images: list[ImagePart] = Field(default_factory=list)


class LLMRequest(BaseModel):
    model: str
    system: str
    messages: list[Message]
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema de la salida estructurada (RF-LLM-04)"
    )
    schema_name: str = "respuesta"
    temperature: float | None = 0.2
    max_tokens: int = 4096
    purpose: str | None = Field(default=None, description="Solo metadatos: no se envía")


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str = ""
    data: dict[str, Any] | None = Field(
        default=None, description="JSON ya decodificado si el proveedor lo devolvió estructurado"
    )
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    stop_reason: str | None = None
