"""Errores de la capa LLM. Quien llama cae a reglas ante `LLMUnavailableError` (RF-WIZ-03)."""

from __future__ import annotations

from typing import Any

from perceptron.core.errors import PerceptronError


class LLMUnavailableError(PerceptronError):
    """No hay LLM utilizable: L0, proveedor no permitido, sin credenciales o desactivado."""

    code = "llm_unavailable"
    http_status = 503


class LLMProviderError(PerceptronError):
    """El proveedor falló (red, cuota, respuesta vacía)."""

    code = "llm_provider_error"
    http_status = 502


class LLMBudgetExceededError(PerceptronError):
    """Se alcanzó el presupuesto de LLM del ámbito (RF-LLM-06)."""

    code = "llm_budget_exceeded"
    http_status = 402


class LLMOutputInvalidError(PerceptronError):
    """La salida no validó contra el schema tras los reintentos (RF-LLM-04)."""

    code = "llm_output_invalid"
    http_status = 422

    def __init__(
        self, message: str, *, last_value: Any = None, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details=details)
        self.last_value = last_value
