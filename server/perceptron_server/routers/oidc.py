"""`/auth/oidc/{provider}` — login SSO (RF-SRV-01): Entra ID, Google Workspace, genérico.

`/login` redirige al IdP con PKCE, `state` y `nonce`, guardados en una cookie firmada de corta
vida (`SameSite=Lax`: vuelve en la navegación de retorno desde el IdP). `/callback` canjea el
código, valida el ID token, vincula o crea el usuario, aplica los roles por grupo y abre la
misma sesión que el login local.
"""

from __future__ import annotations

import time
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

from perceptron.core.errors import AuthError, NotFoundError
from perceptron_server.accounts import sso_login
from perceptron_server.oidc import OIDCClient
from perceptron_server.session import new_csrf, session_cookies
from perceptron_server.state import ServerState, client_ip, server_state

router = APIRouter(prefix="/auth/oidc", tags=["auth"])
State = Annotated[ServerState, Depends(server_state)]
FLOW_COOKIE = "pt_oidc"
FLOW_TTL_S = 600
_ISSUER = "perceptron-oidc-flow"


def _client(state: ServerState, provider: str) -> OIDCClient:
    client = state.oidc_clients.get(provider)
    if client is None:
        raise NotFoundError(f"no hay un proveedor SSO {provider}")
    return client


def _redirect_uri(state: ServerState, request: Request, provider: str) -> str:
    base = (state.server.public_url or str(request.base_url)).rstrip("/")
    return f"{base}/api/v1/auth/oidc/{provider}/callback"


def _safe_next(value: str | None) -> str:
    # Solo rutas propias: evita redirecciones abiertas (//otro-sitio, https://…).
    if value and value.startswith("/") and not value.startswith("//") and "\\" not in value:
        return value
    return "/"


def _flow_cookie(state: ServerState, value: str, max_age: int) -> str:
    secure = "; Secure" if state.server.cookie_secure else ""
    return (
        f"{FLOW_COOKIE}={value}; Max-Age={max_age}; Path=/api/v1/auth/oidc; "
        f"HttpOnly; SameSite=Lax{secure}"
    )


@router.get("/{provider}/login", operation_id="ssoLogin", response_class=RedirectResponse)
def sso_start(
    provider: str, request: Request, state: State, next: str | None = None
) -> RedirectResponse:
    client = _client(state, provider)
    login = client.login_request(_redirect_uri(state, request, provider))
    flow = jwt.encode(
        {
            "iss": _ISSUER,
            "exp": int(time.time()) + FLOW_TTL_S,
            "provider": provider,
            "state": login.state,
            "nonce": login.nonce,
            "verifier": login.code_verifier,
            "next": _safe_next(next),
        },
        state.server.secret_key.get_secret_value(),
        algorithm="HS256",
    )
    response = RedirectResponse(login.url, status_code=302)
    response.headers.append("set-cookie", _flow_cookie(state, flow, FLOW_TTL_S))
    return response


@router.get("/{provider}/callback", operation_id="ssoCallback", response_class=RedirectResponse)
def sso_callback(
    provider: str,
    request: Request,
    state: State,
    code: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> RedirectResponse:
    ip = client_ip(request)

    def fail(reason: str) -> RedirectResponse:
        state.audit.record(
            "auth.sso_failed", ip=ip, details={"provider": provider, "reason": reason[:300]}
        )
        response = RedirectResponse("/?sso_error=1", status_code=302)
        response.headers.append("set-cookie", _flow_cookie(state, "", 0))
        return response

    if error:
        return fail(f"{error}: {error_description or ''}")
    raw = request.cookies.get(FLOW_COOKIE)
    try:
        flow = jwt.decode(
            raw or "",
            state.server.secret_key.get_secret_value(),
            algorithms=["HS256"],
            issuer=_ISSUER,
        )
    except jwt.PyJWTError:
        return fail("flujo SSO vencido o ausente")
    if flow.get("provider") != provider or request.query_params.get("state") != flow.get("state"):
        return fail("state no coincide")
    if not code:
        return fail("sin código de autorización")
    client = _client(state, provider)
    try:
        tokens = client.exchange(code, _redirect_uri(state, request, provider), flow["verifier"])
        identity = client.validate(str(tokens["id_token"]), str(flow["nonce"]))
        user = sso_login(state.accounts, provider, client.provider, identity)
    except AuthError as exc:
        return fail(exc.message)
    pair = state.accounts.start_session(user.id, request.headers.get("user-agent"), ip)
    state.audit.record("auth.login_sso", user_id=user.id, ip=ip, details={"provider": provider})
    response = RedirectResponse(_safe_next(flow.get("next")), status_code=302)
    for cookie in session_cookies(state, pair, csrf=new_csrf()):
        response.headers.append("set-cookie", cookie)
    response.headers.append("set-cookie", _flow_cookie(state, "", 0))
    return response
