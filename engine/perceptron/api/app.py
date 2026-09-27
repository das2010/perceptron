"""Factory de la aplicación FastAPI."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from perceptron import __version__
from perceptron.api.context import EngineContext
from perceptron.api.routers import agent, data, llm, modeling, projects, system
from perceptron.core.config import Settings, get_settings
from perceptron.core.errors import AuthError, PerceptronError

API_PREFIX = "/api/v1"
TOKEN_HEADER = "X-Perceptron-Token"  # noqa: S105 - nombre de cabecera, no un secreto
_PUBLIC_PATHS = {f"{API_PREFIX}/system/health"}


def create_app(settings: Settings | None = None, ctx: EngineContext | None = None) -> FastAPI:
    settings = settings or (ctx.settings if ctx else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = ctx is None
        app.state.ctx = ctx or EngineContext.create(settings)
        try:
            yield
        finally:
            if owned:
                app.state.ctx.close()

    app = FastAPI(
        title="Perceptron Engine API",
        version=__version__,
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    async def perceptron_error_handler(_: Request, exc: PerceptronError) -> JSONResponse:
        return JSONResponse(exc.to_dict(), status_code=exc.http_status)

    app.include_router(system.router, prefix=API_PREFIX)
    app.include_router(projects.router, prefix=API_PREFIX)
    app.include_router(data.router, prefix=API_PREFIX)
    app.include_router(modeling.router, prefix=API_PREFIX)
    app.include_router(llm.router, prefix=API_PREFIX)
    app.include_router(agent.router, prefix=API_PREFIX)
    return app
