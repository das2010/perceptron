"""Factory del Team Server: el mismo Engine + auth, RBAC, auditoría y la UI web (SPEC §4.2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from perceptron.api.app import API_PREFIX, create_app
from perceptron.api.context import EngineContext
from perceptron.api.security import SecurityHeaders
from perceptron.core.config import RuntimeMode, Settings, get_settings
from perceptron.storage.db import sqlite_url
from perceptron_server import __version__
from perceptron_server.migrate import upgrade
from perceptron_server.queue.dispatch import CeleryDispatcher, Dispatcher
from perceptron_server.queue.jobstore import DbJobStore
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
    if "docs" not in api.model_fields_set:
        api = api.model_copy(update={"docs": False})  # sin Swagger público en el servidor
    update: dict[str, object] = {"mode": RuntimeMode.SERVER, "api": api}
    if base.source_roots is None:
        # Sin PERCEPTRON_SOURCE_ROOTS no se lee ninguna ruta del servidor (solo subidas).
        update["source_roots"] = []
    settings = base.model_copy(update=update)
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
            launcher = QueueLauncher(quota, relay, dispatcher)
            ctx.study_launcher = launcher
            ctx.on_close(relay.close)
            # Primero se retoman los jobs guardados y después llegan los eventos de los workers.
            launcher.restore(ctx, DbJobStore(ctx.db))
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
    # El Engine ya pone las cabeceras comunes; el servidor suma la CSP de la SPA y HSTS.
    app.add_middleware(SecurityHeaders, api_prefix=API_PREFIX, csp=CSP, hsts=server.cookie_secure)
    if server.spa_dir:
        spa = Path(server.spa_dir)
        if not (spa / "index.html").is_file():
            raise RuntimeError(f"no está la UI compilada en {spa} (pnpm -C ui build)")
        app.mount("/", SpaFiles(directory=spa, html=True), name="spa")
    return app
