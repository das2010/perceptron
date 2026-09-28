"""Estado compartido del Team Server (se completa al arrancar, cuando existe el EngineContext)."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import cast

from starlette.requests import HTTPConnection

from perceptron.api.context import EngineContext
from perceptron.core.config import Settings
from perceptron_server.accounts import Accounts
from perceptron_server.audit import AuditLog
from perceptron_server.oidc import OIDCClient
from perceptron_server.policy import ServerAccess
from perceptron_server.queue.launcher import WorkerRegistry
from perceptron_server.settings import ServerSettings


class RateLimiter:
    """Ventana deslizante en memoria por clave (IP). Con varias réplicas: Redis (Capa 5b)."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > 10_000:  # acota memoria ante barridos de IPs
                self._hits = {k: v for k, v in self._hits.items() if v}
            return True


@dataclass
class ServerState:
    settings: Settings
    server: ServerSettings
    access: ServerAccess = field(default_factory=ServerAccess)
    _ctx: EngineContext | None = None
    _accounts: Accounts | None = None
    _audit: AuditLog | None = None
    limiter: RateLimiter = field(init=False)
    workers: WorkerRegistry = field(init=False)
    queue_mode: str = "local"
    oidc_clients: dict[str, OIDCClient] = field(init=False)

    def __post_init__(self) -> None:
        self.limiter = RateLimiter(self.server.auth_rate_per_minute)
        self.workers = WorkerRegistry(stale_after_s=3 * self.server.worker_heartbeat_s)
        self.oidc_clients = {
            name: OIDCClient(name, provider) for name, provider in self.server.oidc.items()
        }

    def bind(self, ctx: EngineContext) -> None:
        self._ctx = ctx
        self._accounts = Accounts(ctx, self.server)
        self._audit = AuditLog(ctx.db)
        self.access.bind(ctx, self._accounts)

    @property
    def ctx(self) -> EngineContext:
        if self._ctx is None:
            raise RuntimeError("Team Server sin inicializar")
        return self._ctx

    @property
    def accounts(self) -> Accounts:
        if self._accounts is None:
            raise RuntimeError("Team Server sin inicializar")
        return self._accounts

    @property
    def audit(self) -> AuditLog:
        if self._audit is None:
            raise RuntimeError("Team Server sin inicializar")
        return self._audit


def server_state(conn: HTTPConnection) -> ServerState:
    return cast(ServerState, conn.app.state.server)


def client_ip(conn: HTTPConnection) -> str | None:
    return conn.client.host if conn.client else None
