"""Cabeceras de seguridad y errores sin eco de datos (Capa 7, ASVS V7.4 y V14.4).

Las usan el Engine del desktop y el Team Server. El servidor agrega la CSP de la SPA y HSTS.
"""

from __future__ import annotations

from typing import Any

from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from perceptron.core.errors import ValidationError

PERMISSIONS_POLICY = "camera=(), microphone=(), geolocation=(), payment=(), usb=()"


class SecurityHeaders:
    """`api_prefix`: respuestas de la API sin caché. `csp` y `hsts`: solo el Team Server."""

    def __init__(
        self, app: ASGIApp, *, api_prefix: str, csp: str | None = None, hsts: bool = False
    ) -> None:
        self.app = app
        self.api_prefix = api_prefix
        self.csp = csp
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        api = scope["path"].startswith(self.api_prefix)

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("x-content-type-options", "nosniff")
                headers.setdefault("referrer-policy", "same-origin")
                headers.setdefault("x-frame-options", "DENY")
                headers.setdefault("cross-origin-opener-policy", "same-origin")
                headers.setdefault("permissions-policy", PERMISSIONS_POLICY)
                if api:
                    # Tokens, datos de proyectos y descargas: ni el navegador ni un proxy los
                    # guardan.
                    headers.setdefault("cache-control", "no-store")
                elif self.csp:
                    headers.setdefault("content-security-policy", self.csp)
                if self.hsts:
                    headers.setdefault(
                        "strict-transport-security", "max-age=31536000; includeSubDomains"
                    )
            await send(message)

        await self.app(scope, receive, wrapped)


def _clean(error: dict[str, Any]) -> dict[str, Any]:
    """Error de validación sin el valor recibido (`input`), que puede ser una contraseña o un
    dato del dataset, ni el contexto interno."""
    return {k: v for k, v in error.items() if k in ("type", "loc", "msg")}


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return JSONResponse(
        {"detail": jsonable_encoder([_clean(dict(e)) for e in errors])}, status_code=422
    )


MB = 1024**2


async def read_limited(file: UploadFile, limit: int) -> bytes:
    """Lee un archivo subido sin pasar de `limit` bytes (nunca todo a memoria sin tope)."""
    data = bytearray()
    while chunk := await file.read(MB):
        data += chunk
        if len(data) > limit:
            raise ValidationError(f"el archivo supera el límite de {limit // MB} MB")
    return bytes(data)
