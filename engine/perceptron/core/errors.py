"""Jerarquía de errores del Engine.

Cada error tiene un `code` estable (usado por la API y la UI para i18n) y un
`details` serializable. La API los traduce a respuestas JSON uniformes.
"""

from __future__ import annotations

from typing import Any, ClassVar


class PerceptronError(Exception):
    code: ClassVar[str] = "internal_error"
    http_status: ClassVar[int] = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class NotFoundError(PerceptronError):
    code = "not_found"
    http_status = 404


class ValidationError(PerceptronError):
    code = "validation_error"
    http_status = 422


class ConflictError(PerceptronError):
    """Conflicto de versión (bloqueo optimista) o recurso duplicado."""

    code = "conflict"
    http_status = 409


class AuthError(PerceptronError):
    code = "unauthorized"
    http_status = 401


class ForbiddenError(PerceptronError):
    """Autenticado pero sin permiso para la operación (RBAC del Team Server, RF-SRV-02)."""

    code = "forbidden"
    http_status = 403


class RateLimitedError(PerceptronError):
    code = "rate_limited"
    http_status = 429


class StorageError(PerceptronError):
    code = "storage_error"
    http_status = 500


class FeatureDisabledError(PerceptronError):
    code = "feature_disabled"
    http_status = 403
