"""Factory de la aplicación FastAPI."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from perceptron import __version__
from perceptron.api.access import AccessPolicy, OpenAccess, authorized
from perceptron.api.activity import activity_middleware
from perceptron.api.context import EngineContext
from perceptron.api.routers import (
    agent,
    analysis,
    data,
    export,
    labeling,
    llm,
    modeling,
    monitoring,
    projects,
    remote,
    system,
    wizard,
)
from perceptron.api.security import SecurityHeaders, validation_error_handler
from perceptron.core.config import Settings, get_settings
from perceptron.core.errors import AuthError, PerceptronError

API_PREFIX = "/api/v1"
TOKEN_HEADER = "X-Perceptron-Token"  # noqa: S105 - nombre de cabecera, no un secreto
_PUBLIC_PATHS = {f"{API_PREFIX}/system/health"}


def create_app(
    settings: Settings | None = None,
    ctx: EngineContext | None = None,
    access: AccessPolicy | None = None,
    ctx_factory: Callable[[Settings], EngineContext] | None = None,
) -> FastAPI:
    """App del Engine. `access` es la política de acceso: abierta en desktop; el Team Server
    instala la suya (autenticación + RBAC) sin cambiar los routers. `ctx_factory` crea el
    contexto al arrancar (el servidor migra la base antes y arma sus servicios después)."""
    settings = settings or (ctx.settings if ctx else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = ctx is None
        app.state.ctx = ctx or (ctx_factory or EngineContext.create)(settings)
        app.state.ctx.start_scheduler()
        try:
            yield
        finally:
            if owned:
                app.state.ctx.close()

    app = FastAPI(
        title="Perceptron Engine API",
        version=__version__,
        openapi_url=f"{API_PREFIX}/openapi.json" if settings.api.docs else None,
        docs_url=f"{API_PREFIX}/docs" if settings.api.docs else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.access = access or OpenAccess()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeaders, api_prefix=API_PREFIX)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    expected = settings.api.token.get_secret_value() if settings.api.token else None

    @app.middleware("http")
    async def token_auth(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Token efímero del sidecar (SPEC §13.2). CORS preflight y health son públicos.
        if (
            expected
            and request.method != "OPTIONS"
            and request.url.path.startswith(API_PREFIX)
            and request.url.path not in _PUBLIC_PATHS
        ):
            got = request.headers.get(TOKEN_HEADER, "")
            if not secrets.compare_digest(got, expected):
                err = AuthError("Token inválido o ausente")
                return JSONResponse(err.to_dict(), status_code=err.http_status)
        return await call_next(request)

    @app.exception_handler(PerceptronError)
    async def perceptron_error_handler(request: Request, exc: PerceptronError) -> JSONResponse:
        if exc.http_status >= 500:  # solo el tipo, nunca el mensaje (telemetría opt-in)
            request.app.state.ctx.telemetry.error(type(exc).__name__)
        return JSONResponse(exc.to_dict(), status_code=exc.http_status)

    app.middleware("http")(activity_middleware)

    @app.middleware("http")
    async def usage_counter(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        route = request.scope.get("route")
        op = getattr(route, "operation_id", None)
        ctx = getattr(request.app.state, "ctx", None)
        if op and ctx is not None:
            ctx.telemetry.count(op)  # contador en memoria; sale solo con consentimiento
        return response

    for module in (
        system,
        projects,
        data,
        modeling,
        llm,
        agent,
        wizard,
        export,
        analysis,
        labeling,
        remote,
        monitoring,
    ):
        app.include_router(module.router, prefix=API_PREFIX, dependencies=[authorized])
    app.include_router(remote.project_router, prefix=API_PREFIX, dependencies=[authorized])
    return app
