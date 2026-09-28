"""Sesiones del Team Server: quién hace cada request, CSRF y auditoría (RF-SRV-01, RF-SRV-07).

- **Navegador**: cookies `HttpOnly` + `SameSite=Strict` con un JWT corto y un token de refresco
  rotativo. Si el JWT venció, el middleware rota el refresco y renueva las cookies en la misma
  respuesta (la UI nunca ve vencimientos). Las escrituras exigen la cabecera `X-CSRF-Token`
  igual a la cookie `pt_csrf` (double submit).
- **CLI / desktop**: `Authorization: Bearer <jwt>` (sin CSRF: no hay cookies ambientales).

ASGI puro: cubre HTTP y WebSocket y, al terminar, ve la operación que resolvió el ruteo.
"""

from __future__ import annotations

import secrets
from http.cookies import SimpleCookie
from typing import Any

import anyio
from starlette.datastructures import MutableHeaders
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from perceptron_server.accounts import Principal, TokenPair
from perceptron_server.audit import AUDITED_READS
from perceptron_server.queue.launcher import CURRENT_PRINCIPAL
from perceptron_server.security import decode_access
from perceptron_server.state import ServerState, client_ip

ACCESS_COOKIE = "pt_access"
REFRESH_COOKIE = "pt_refresh"
CSRF_COOKIE = "pt_csrf"
CSRF_HEADER = "x-csrf-token"
API_PREFIX = "/api/v1"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
SSO_PREFIX = f"{API_PREFIX}/auth/oidc/"
# Login y emisión de tokens no llevan CSRF (no dependen de una sesión previa).
CSRF_EXEMPT = frozenset(
    {f"{API_PREFIX}/auth/login", f"{API_PREFIX}/auth/token", f"{API_PREFIX}/auth/refresh"}
)


def session_cookies(
    state: ServerState, pair: TokenPair | None, csrf: str | None = None
) -> list[str]:
    """Cabeceras `Set-Cookie`; sin `pair` las borra (logout)."""
    secure = "; Secure" if state.server.cookie_secure else ""
    base = f"; Path=/; SameSite=Strict{secure}"
    if pair is None:
        return [
            f"{name}=; Max-Age=0{base}{'; HttpOnly' if name != CSRF_COOKIE else ''}"
            for name in (ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE)
        ]
    access_age, refresh_age = state.server.access_ttl_s, state.server.refresh_ttl_s
    cookies = [
        f"{ACCESS_COOKIE}={pair.access_token}; Max-Age={access_age}{base}; HttpOnly",
        f"{REFRESH_COOKIE}={pair.refresh_token}; Max-Age={refresh_age}{base}; HttpOnly",
    ]
    if csrf:
        cookies.append(f"{CSRF_COOKIE}={csrf}; Max-Age={state.server.refresh_ttl_s}{base}")
    return cookies


def new_csrf() -> str:
    return secrets.token_urlsafe(24)


def _cookies(conn: HTTPConnection) -> dict[str, str]:
    raw = conn.headers.get("cookie")
    if not raw:
        return {}
    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(raw)
    except Exception:  # cookie malformada de otro sitio del mismo host: se ignora
        return {}
    return {k: v.value for k, v in jar.items()}


class SessionMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Solo la API necesita sesión: los archivos de la SPA se sirven sin consultar la base.
        if scope["type"] not in ("http", "websocket") or not scope["path"].startswith(API_PREFIX):
            await self.app(scope, receive, send)
            return
        state: ServerState = scope["app"].state.server
        conn = HTTPConnection(scope)
        websocket = scope["type"] == "websocket"
        # Login, tokens y el retorno del SSO no dependen de una sesión previa.
        skip = scope["path"] in CSRF_EXEMPT or scope["path"].startswith(SSO_PREFIX)
        principal, pair = (
            (None, None)
            if skip
            else await anyio.to_thread.run_sync(self._authenticate, state, conn, not websocket)
        )
        scope.setdefault("state", {})["principal"] = principal
        principal_token = CURRENT_PRINCIPAL.set(principal)
        method = scope.get("method", "GET")
        path: str = scope["path"]

        if (
            not websocket
            and principal is not None
            and principal.via == "cookie"
            and method not in SAFE_METHODS
            and path.startswith(API_PREFIX)
            and path not in CSRF_EXEMPT
        ):
            cookie = _cookies(conn).get(CSRF_COOKIE, "")
            header = conn.headers.get(CSRF_HEADER, "")
            if not cookie or not secrets.compare_digest(cookie, header):
                response = JSONResponse(
                    {"code": "csrf_failed", "message": "falta o no coincide el token CSRF"},
                    status_code=403,
                )
                await response(scope, receive, send)
                return

        status: dict[str, int] = {}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                if pair is not None:
                    headers = MutableHeaders(scope=message)
                    for cookie in session_cookies(state, pair):
                        headers.append("set-cookie", cookie)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            CURRENT_PRINCIPAL.reset(principal_token)
            if not websocket:
                await anyio.to_thread.run_sync(
                    self._audit, state, scope, principal, status.get("code")
                )

    # ------------------------------------------------------------------ autenticación

    @staticmethod
    def _authenticate(
        state: ServerState, conn: HTTPConnection, can_rotate: bool
    ) -> tuple[Principal | None, TokenPair | None]:
        secret = state.server.secret_key.get_secret_value()
        auth = conn.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            claims = decode_access(auth[7:].strip(), secret)
            if claims is None or claims == "expired":
                return None, None
            if not state.accounts.session_active(claims.session_id):
                return None, None
            return state.accounts.principal(claims.user_id, claims.session_id, "bearer"), None

        cookies = _cookies(conn)
        token = cookies.get(ACCESS_COOKIE)
        if token:
            claims = decode_access(token, secret)
            if claims is not None and claims != "expired":
                if state.accounts.session_active(claims.session_id):
                    return state.accounts.principal(claims.user_id, claims.session_id), None
                return None, None
        refresh = cookies.get(REFRESH_COOKIE)
        if not refresh:
            return None, None
        resolved = state.accounts.resolve_refresh(
            refresh,
            rotate=can_rotate,
            user_agent=conn.headers.get("user-agent"),
            ip=client_ip(conn),
        )
        if resolved is None:
            return None, None
        user_id, session_id, pair = resolved
        return state.accounts.principal(user_id, session_id), pair

    # ------------------------------------------------------------------ auditoría

    @staticmethod
    def _audit(
        state: ServerState, scope: Scope, principal: Principal | None, status: int | None
    ) -> None:
        st: dict[str, Any] = scope.get("state", {})
        op = st.get("operation")
        method = scope.get("method", "GET")
        if not op or (method in SAFE_METHODS and op not in AUDITED_READS and status != 403):
            return
        client = scope.get("client")
        state.audit.record(
            f"api.{op}",
            user_id=principal.user.id if principal else None,
            project_id=st.get("audit_project"),
            resource=f"{method} {scope['path']}",
            status=status,
            ip=client[0] if client else None,
        )
