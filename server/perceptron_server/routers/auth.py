"""`/auth` — login local, sesiones y tokens (RF-SRV-01). OIDC (Entra ID, Google) en la 5c."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from perceptron.core.errors import AuthError, RateLimitedError
from perceptron.domain.models import Membership, User, Workspace
from perceptron_server.accounts import Principal
from perceptron_server.policy import principal_of
from perceptron_server.session import REFRESH_COOKIE, new_csrf, session_cookies
from perceptron_server.state import ServerState, client_ip, server_state

router = APIRouter(prefix="/auth", tags=["auth"])
State = Annotated[ServerState, Depends(server_state)]


def current_principal(request: Request) -> Principal:
    who = principal_of(request)
    if who is None:
        raise AuthError("iniciá sesión para continuar")
    return who


Who = Annotated[Principal, Depends(current_principal)]


def rate_limited(request: Request, state: State) -> None:
    if not state.limiter.allow(client_ip(request) or "?"):
        raise RateLimitedError("demasiados intentos; esperá un minuto")


class AuthConfig(BaseModel):
    mode: Literal["server"] = "server"
    password_login: bool = True
    oidc_providers: list[str] = Field(default_factory=list, description="Capa 5c")


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class Me(BaseModel):
    user: User
    is_server_admin: bool
    memberships: list[Membership]
    workspaces: list[Workspace]


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - tipo OAuth, no un secreto
    expires_at: datetime
    refresh_expires_at: datetime


class RefreshBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refresh_token: str = Field(min_length=10, max_length=200)


class PasswordChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


def _me(state: ServerState, who: Principal) -> Me:
    workspaces = [
        w
        for w in state.accounts.list_workspaces()
        if who.is_server_admin or w.id in {m.workspace_id for m in who.memberships}
    ]
    return Me(
        user=who.user,
        is_server_admin=who.is_server_admin,
        memberships=who.memberships,
        workspaces=workspaces,
    )


@router.get("/config", operation_id="getAuthConfig")
def auth_config() -> AuthConfig:
    """Público: la UI web lo usa para saber que habla con un Team Server y cómo loguearse."""
    return AuthConfig()


@router.post("/login", operation_id="login", dependencies=[Depends(rate_limited)])
def login(body: Credentials, request: Request, response: Response, state: State) -> Me:
    ip = client_ip(request)
    try:
        user = state.accounts.authenticate(body.email, body.password)
    except (AuthError, RateLimitedError) as exc:
        state.audit.record("auth.login_failed", ip=ip, details={"email": body.email[:320]})
        raise exc from None
    pair = state.accounts.start_session(user.id, request.headers.get("user-agent"), ip)
    for cookie in session_cookies(state, pair, csrf=new_csrf()):
        response.headers.append("set-cookie", cookie)
    state.audit.record("auth.login", user_id=user.id, ip=ip)
    who = state.accounts.principal(user.id, pair.session_id)
    if who is None:
        raise AuthError("usuario inactivo")
    return _me(state, who)


@router.post("/logout", status_code=204, operation_id="logout")
def logout(request: Request, response: Response, state: State) -> Response:
    who = principal_of(request)
    if who is not None and who.session_id:
        state.accounts.revoke_session(who.session_id)
        state.audit.record("auth.logout", user_id=who.user.id, ip=client_ip(request))
    elif refresh := request.cookies.get(REFRESH_COOKIE):
        resolved = state.accounts.resolve_refresh(refresh, rotate=False)
        if resolved:
            state.accounts.revoke_session(resolved[1])
    response.status_code = 204
    for cookie in session_cookies(state, None):
        response.headers.append("set-cookie", cookie)
    return response


@router.get("/me", operation_id="getMe")
def me(who: Who, state: State) -> Me:
    return _me(state, who)


@router.post("/token", operation_id="issueToken", dependencies=[Depends(rate_limited)])
def token(body: Credentials, request: Request, state: State) -> TokenResponse:
    """Tokens bearer para la CLI y el desktop (sync con el servidor, Capa 5c)."""
    ip = client_ip(request)
    try:
        user = state.accounts.authenticate(body.email, body.password)
    except (AuthError, RateLimitedError) as exc:
        state.audit.record("auth.login_failed", ip=ip, details={"email": body.email[:320]})
        raise exc from None
    pair = state.accounts.start_session(user.id, request.headers.get("user-agent"), ip)
    state.audit.record("auth.token", user_id=user.id, ip=ip)
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_at=pair.access_expires_at,
        refresh_expires_at=pair.refresh_expires_at,
    )


@router.post("/refresh", operation_id="refreshToken", dependencies=[Depends(rate_limited)])
def refresh(body: RefreshBody, request: Request, state: State) -> TokenResponse:
    resolved = state.accounts.resolve_refresh(
        body.refresh_token,
        rotate=True,
        user_agent=request.headers.get("user-agent"),
        ip=client_ip(request),
    )
    if resolved is None or resolved[2] is None:
        raise AuthError("sesión vencida o revocada")
    pair = resolved[2]
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_at=pair.access_expires_at,
        refresh_expires_at=pair.refresh_expires_at,
    )


@router.post("/password", status_code=204, operation_id="changePassword")
def change_password(body: PasswordChange, who: Who, request: Request, state: State) -> Response:
    """Cambia la contraseña propia; cierra todas las sesiones (incluida esta)."""
    state.accounts.change_password(who.user.id, body.current_password, body.new_password)
    state.audit.record("auth.password_changed", user_id=who.user.id, ip=client_ip(request))
    response = Response(status_code=204)
    for cookie in session_cookies(state, None):
        response.headers.append("set-cookie", cookie)
    return response
