"""SSO OIDC (RF-SRV-01) contra un IdP simulado con claves RSA reales: discovery, JWKS, PKCE,
state, nonce, validación del ID token y mapeo de grupos a roles (estilo Entra ID)."""

from __future__ import annotations

import base64
import hashlib
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from joserfc import jwt
from joserfc.jwk import RSAKey
from pydantic import SecretStr
from srv_helpers import ADMIN_EMAIL, ADMIN_PASSWORD, API, login, ok

from perceptron.core.config import Settings
from perceptron_server.app import create_server_app
from perceptron_server.settings import GroupRole, OIDCProvider, ServerSettings

TENANT = "11111111-2222-3333-4444-555555555555"
ISSUER = f"https://login.microsoftonline.com/{TENANT}/v2.0"
CLIENT_ID = "perceptron-app"
DATA_GROUP = "grp-data-science"


class FakeIdP:
    """Emisor OIDC mínimo: discovery, JWKS y endpoint de tokens con PKCE."""

    def __init__(self) -> None:
        self.key = RSAKey.generate_key(2048, parameters={"kid": "k1"}, private=True)
        self.codes: dict[str, dict[str, Any]] = {}
        self.claims: dict[str, Any] = {}
        self.token_requests: list[dict[str, str]] = []

    def issue_code(self, challenge: str, nonce: str, **overrides: Any) -> str:
        code = f"code-{len(self.codes)}"
        self.codes[code] = {"challenge": challenge, "nonce": nonce, **overrides}
        return code

    def id_token(self, nonce: str, **overrides: Any) -> str:
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "sub": "oid-123",
            "iat": now,
            "exp": now + 600,
            "nonce": nonce,
            "preferred_username": "Laura.Diaz@Preteco.com",
            "name": "Laura Díaz",
            "groups": [DATA_GROUP],
            **self.claims,
            **overrides,
        }
        return jwt.encode({"alg": "RS256", "kid": "k1"}, claims, self.key)

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/.well-known/openid-configuration"):
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/authorize",
                    "token_endpoint": f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token",
                    "jwks_uri": f"https://login.microsoftonline.com/{TENANT}/discovery/v2.0/keys",
                },
            )
        if path.endswith("/keys"):
            return httpx.Response(200, json={"keys": [self.key.as_dict(private=False)]})
        if path.endswith("/token"):
            form = dict(parse_qs(request.content.decode()))
            body = {k: v[0] for k, v in form.items()}
            self.token_requests.append(body)
            grant = self.codes.pop(body.get("code", ""), None)
            if grant is None:
                return httpx.Response(400, json={"error": "invalid_grant"})
            digest = hashlib.sha256(body["code_verifier"].encode()).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != grant["challenge"]:
                return httpx.Response(
                    400, json={"error": "invalid_grant", "error_description": "pkce"}
                )
            extra = {k: v for k, v in grant.items() if k not in ("challenge", "nonce")}
            return httpx.Response(
                200,
                json={"id_token": self.id_token(grant["nonce"], **extra), "token_type": "Bearer"},
            )
        return httpx.Response(404)


@pytest.fixture
def idp() -> FakeIdP:
    return FakeIdP()


@pytest.fixture
def sso_app(settings: Settings, server_settings: ServerSettings, idp: FakeIdP) -> Iterator[FastAPI]:
    provider = OIDCProvider(
        kind="entra",
        display_name="Microsoft (Preteco)",
        client_id=CLIENT_ID,
        client_secret=SecretStr("secreto-del-cliente"),
        tenant_id=TENANT,
        allowed_domains=["preteco.com"],
        role_mapping=[GroupRole(group=DATA_GROUP, workspace="Ciencia de datos", role="editor")],
    )
    server = server_settings.model_copy(
        update={"oidc": {"entra": provider}, "public_url": "http://testserver"}
    )
    app = create_server_app(settings, server)
    for client in app.state.server.oidc_clients.values():
        client.http = lambda: httpx.Client(transport=httpx.MockTransport(idp.handler))
    with TestClient(app):
        yield app


def _start(app: FastAPI) -> tuple[TestClient, dict[str, str]]:
    browser = TestClient(app, follow_redirects=False)
    res = browser.get(f"{API}/auth/oidc/entra/login", params={"next": "/projects"})
    assert res.status_code == 302
    url = urlparse(res.headers["location"])
    assert url.netloc == "login.microsoftonline.com"
    params = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert params["code_challenge_method"] == "S256" and params["client_id"] == CLIENT_ID
    assert params["redirect_uri"] == "http://testserver/api/v1/auth/oidc/entra/callback"
    flow = next(h for h in res.headers.get_list("set-cookie") if h.startswith("pt_oidc="))
    assert "HttpOnly" in flow and "SameSite=Lax" in flow
    return browser, params


def test_config_lists_sso_providers(sso_app: FastAPI) -> None:
    config = ok(TestClient(sso_app).get(f"{API}/auth/config"))
    assert config["oidc_providers"] == [
        {"id": "entra", "display_name": "Microsoft (Preteco)", "kind": "entra"}
    ]


def test_sso_login_creates_user_with_group_roles(sso_app: FastAPI, idp: FakeIdP) -> None:
    browser, params = _start(sso_app)
    code = idp.issue_code(params["code_challenge"], params["nonce"])
    res = browser.get(
        f"{API}/auth/oidc/entra/callback", params={"code": code, "state": params["state"]}
    )
    assert res.status_code == 302 and res.headers["location"] == "/projects"
    names = {h.split("=", 1)[0] for h in res.headers.get_list("set-cookie")}
    assert {"pt_access", "pt_refresh", "pt_csrf"} <= names
    assert idp.token_requests[0]["client_secret"] == "secreto-del-cliente"

    me = ok(browser.get(f"{API}/auth/me"))
    assert me["user"]["email"] == "laura.diaz@preteco.com"
    assert me["user"]["auth_provider"] == "oidc:entra"
    ws = {w["id"]: w["name"] for w in me["workspaces"]}
    roles = {ws[m["workspace_id"]]: m["role"] for m in me["memberships"]}
    assert roles == {"Ciencia de datos": "editor"}  # por grupo; sin rol por defecto extra
    # Sin contraseña local: el login por contraseña no sirve para este usuario.
    r = TestClient(sso_app).post(
        f"{API}/auth/login", json={"email": "laura.diaz@preteco.com", "password": "x" * 12}
    )
    assert r.status_code == 401

    admin = login(sso_app, ADMIN_EMAIL, ADMIN_PASSWORD)
    events = ok(admin.get(f"{API}/admin/audit", params={"action": "auth.login_sso"}))
    assert events and events[0]["details"]["provider"] == "entra"


def test_sso_rejects_tampering(sso_app: FastAPI, idp: FakeIdP) -> None:
    def callback(browser: TestClient, params: dict[str, str], **query: str) -> httpx.Response:
        return browser.get(f"{API}/auth/oidc/entra/callback", params=query)

    # state distinto al de la cookie
    browser, params = _start(sso_app)
    code = idp.issue_code(params["code_challenge"], params["nonce"])
    res = callback(browser, params, code=code, state="otro")
    assert res.headers["location"] == "/?sso_error=1"

    # nonce distinto (ID token repetido de otro flujo)
    browser, params = _start(sso_app)
    code = idp.issue_code(params["code_challenge"], "nonce-viejo")
    assert (
        "sso_error"
        in callback(browser, params, code=code, state=params["state"]).headers["location"]
    )

    # audiencia de otra aplicación
    browser, params = _start(sso_app)
    code = idp.issue_code(params["code_challenge"], params["nonce"], aud="otra-app")
    assert (
        "sso_error"
        in callback(browser, params, code=code, state=params["state"]).headers["location"]
    )

    # dominio no habilitado
    browser, params = _start(sso_app)
    code = idp.issue_code(
        params["code_challenge"], params["nonce"], preferred_username="x@gmail.com"
    )
    assert (
        "sso_error"
        in callback(browser, params, code=code, state=params["state"]).headers["location"]
    )

    # PKCE: otro navegador sin la cookie del flujo no puede usar el código
    _, params = _start(sso_app)
    code = idp.issue_code(params["code_challenge"], params["nonce"])
    stranger = TestClient(sso_app, follow_redirects=False)
    assert (
        "sso_error"
        in callback(stranger, params, code=code, state=params["state"]).headers["location"]
    )

    # error del IdP (p. ej. el usuario canceló)
    browser, _ = _start(sso_app)
    res = browser.get(
        f"{API}/auth/oidc/entra/callback", params={"error": "access_denied", "state": "x"}
    )
    assert res.headers["location"] == "/?sso_error=1"

    admin = login(sso_app, ADMIN_EMAIL, ADMIN_PASSWORD)
    failures = ok(admin.get(f"{API}/admin/audit", params={"action": "auth.sso_failed"}))
    assert len(failures) == 6
    assert ok(admin.get(f"{API}/admin/users"))[0]["user"]["email"] == ADMIN_EMAIL  # nadie más


def test_open_redirect_is_blocked(sso_app: FastAPI, idp: FakeIdP) -> None:
    browser = TestClient(sso_app, follow_redirects=False)
    res = browser.get(f"{API}/auth/oidc/entra/login", params={"next": "//evil.example/x"})
    params = {k: v[0] for k, v in parse_qs(urlparse(res.headers["location"]).query).items()}
    code = idp.issue_code(params["code_challenge"], params["nonce"])
    done = browser.get(
        f"{API}/auth/oidc/entra/callback", params={"code": code, "state": params["state"]}
    )
    assert done.headers["location"] == "/"


def test_unknown_provider_and_disabled_password_login(
    settings: Settings, server_settings: ServerSettings
) -> None:
    app = create_server_app(settings, server_settings.model_copy(update={"password_login": False}))
    with TestClient(app) as c:
        assert c.get(f"{API}/auth/oidc/nada/login").status_code == 404
        r = c.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
        assert r.status_code == 403
        assert ok(c.get(f"{API}/auth/config"))["password_login"] is False
