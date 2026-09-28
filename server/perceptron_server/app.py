"""Factory del Team Server: el mismo Engine + auth, RBAC, auditoría y la UI web (SPEC §4.2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from perceptron.api.app import API_PREFIX, create_app
from perceptron.api.context import EngineContext
from perceptron.core.config import RuntimeMode, Settings, get_settings
from perceptron.storage.db import sqlite_url
from perceptron_server import __version__
from perceptron_server.migrate import upgrade
from perceptron_server.queue.dispatch import CeleryDispatcher, Dispatcher
from perceptron_server.queue.launcher import (
    LocalLauncher,
    QueueLauncher,
    QuotaGuard,
    start_event_bridge,
)
from perceptron_server.queue.relay import RedisRelay, Relay
from perceptron_server.routers import admin, auth, oidc, queue, sources, sync
from perceptron_server.session import SessionMiddleware
from perceptron_server.settings import ServerSettings
from perceptron_server.state import ServerState

# La SPA no usa scripts inline ni recursos externos (fuentes y Monaco empaquetados).
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
    "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


QueueFactory = Callable[[Settings], tuple[Relay, Dispatcher]]


def redis_queue(url: str) -> QueueFactory:
    def factory(_: Settings) -> tuple[Relay, Dispatcher]:
        return RedisRelay(url), CeleryDispatcher(url)

    return factory


def database_url(settings: Settings) -> str:
    if settings.database_url:
        return settings.database_url.get_secret_value()
    return sqlite_url(settings.paths.db_file)


class SecurityHeaders:
    def __init__(self, app: ASGIApp, hsts: bool) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        api = scope["path"].startswith(API_PREFIX)

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("x-content-type-options", "nosniff")
                headers.setdefault("referrer-policy", "same-origin")
                headers.setdefault("x-frame-options", "DENY")
                if not api:
                    headers.setdefault("content-security-policy", CSP)
                if self.hsts:
                    headers.setdefault("strict-transport-security", "max-age=31536000")
            await send(message)

        await self.app(scope, receive, wrapped)


class SpaFiles(StaticFiles):
    """UI compilada: rutas del cliente (TanStack Router) caen en `index.html`."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or path.startswith("api"):
                raise
            response = await super().get_response("index.html", scope)
        # En Windows Starlette pasa la ruta con "\\" (os.path.normpath).
        immutable = path.replace("\\", "/").startswith("assets/") and response.status_code == 200
        response.headers["cache-control"] = (
            "public, max-age=31536000, immutable" if immutable else "no-cache"
        )
        return response


def create_server_app(
    settings: Settings | None = None,
    server: ServerSettings | None = None,
    *,
    migrate: bool = True,
    queue_factory: QueueFactory | None = None,
) -> FastAPI:
    """`queue_factory` (relay + dispatcher) activa la cola de workers; por defecto, la de
    `PERCEPTRON_SERVER__REDIS_URL` si está configurada."""
    base = settings or get_settings()
    api = base.api
    if "cors_origins" not in api.model_fields_set:
        api = api.model_copy(update={"cors_origins": []})  # misma origen: sin CORS por defecto
    settings = base.model_copy(update={"mode": RuntimeMode.SERVER, "api": api})
    server = server or ServerSettings()  # secret_key por entorno
    state = ServerState(settings, server)

    def ctx_factory(s: Settings) -> EngineContext:
        s.paths.ensure()
        if migrate:
            upgrade(database_url(s))
        ctx = EngineContext.create(s)
        state.bind(ctx)
        state.accounts.bootstrap()
        quota = QuotaGuard(server)
        make_queue = queue_factory or (
            redis_queue(server.redis_url.get_secret_value()) if server.redis_url else None
        )
        if make_queue is None:
            ctx.study_launcher = LocalLauncher(quota)
        else:
            relay, dispatcher = make_queue(s)
            ctx.study_launcher = QueueLauncher(quota, relay, dispatcher)
            ctx.on_close(relay.close)
            ctx.on_close(start_event_bridge(ctx, relay, state.workers))
            state.queue_mode = "queue"
        return ctx

    app = create_app(settings, access=state.access, ctx_factory=ctx_factory)
    app.title = "Perceptron Team Server API"
    app.version = __version__
    app.state.server = state
    for module in (auth, oidc, admin, sources, queue, sync):
        app.include_router(module.router, prefix=API_PREFIX)
    app.add_middleware(SessionMiddleware)
    app.add_middleware(SecurityHeaders, hsts=server.cookie_secure)
    if server.spa_dir:
        spa = Path(server.spa_dir)
        if not (spa / "index.html").is_file():
            raise RuntimeError(f"no está la UI compilada en {spa} (pnpm -C ui build)")
        app.mount("/", SpaFiles(directory=spa, html=True), name="spa")
    return app
