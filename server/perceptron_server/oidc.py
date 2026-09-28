"""SSO OIDC (RF-SRV-01): Entra ID, Google Workspace o cualquier proveedor OIDC.

Flujo *authorization code* con PKCE (S256), `state` y `nonce`. Authlib (BSD-3) genera los
tokens y el desafío PKCE, y joserfc (BSD-3, del mismo autor) valida el ID token contra el JWKS
publicado por el emisor: firma, emisor, audiencia, vencimiento y nonce. La MFA la resuelve el
IdP; el servidor solo confía en el ID token validado.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from perceptron.core.errors import AuthError
from perceptron_server.settings import OIDCProvider

HttpFactory = Callable[[], httpx.Client]
_CACHE_TTL_S = 3600.0
_SIGNING_ALGS = ["RS256", "RS384", "RS512", "PS256", "ES256", "ES384"]


def default_http() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(10.0), follow_redirects=False)


@dataclass(frozen=True, slots=True)
class LoginRequest:
    url: str
    state: str
    nonce: str
    code_verifier: str


@dataclass(frozen=True, slots=True)
class Identity:
    subject: str
    email: str
    name: str
    groups: list[str]
    claims: dict[str, Any]


class OIDCClient:
    def __init__(self, name: str, provider: OIDCProvider, http: HttpFactory = default_http):
        self.name = name
        self.provider = provider
        self.http = http
        self._discovery: tuple[float, dict[str, Any]] | None = None
        self._jwks: tuple[float, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ metadata del emisor

    def _get_json(self, url: str) -> dict[str, Any]:
        try:
            with self.http() as client:
                res = client.get(url)
                res.raise_for_status()
                data = res.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthError(f"el proveedor SSO {self.name} no responde: {exc}") from None
        if not isinstance(data, dict):
            raise AuthError(f"respuesta inválida del proveedor SSO {self.name}")
        return data

    def discovery(self) -> dict[str, Any]:
        with self._lock:
            if self._discovery and time.monotonic() - self._discovery[0] < _CACHE_TTL_S:
                return self._discovery[1]
        issuer = self.provider.issuer_url()
        data = self._get_json(f"{issuer}/.well-known/openid-configuration")
        if str(data.get("issuer", "")).rstrip("/") != issuer.rstrip("/"):
            raise AuthError("el emisor del proveedor SSO no coincide con la configuración")
        with self._lock:
            self._discovery = (time.monotonic(), data)
        return data

    def _keys(self, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            cached = self._jwks
        if cached and not refresh and time.monotonic() - cached[0] < _CACHE_TTL_S:
            return cached[1]
        data = self._get_json(str(self.discovery()["jwks_uri"]))
        with self._lock:
            self._jwks = (time.monotonic(), data)
        return data

    # ------------------------------------------------------------------ flujo

    def login_request(self, redirect_uri: str) -> LoginRequest:
        from authlib.common.security import generate_token
        from authlib.oauth2.rfc7636 import create_s256_code_challenge

        state, nonce, verifier = generate_token(32), generate_token(32), generate_token(64)
        params = {
            "response_type": "code",
            "client_id": self.provider.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(self.provider.scopes),
            "state": state,
            "nonce": nonce,
            "code_challenge": create_s256_code_challenge(verifier),
            "code_challenge_method": "S256",
        }
        if self.provider.kind == "google" and self.provider.allowed_domains:
            params["hd"] = self.provider.allowed_domains[0]
        endpoint = str(self.discovery()["authorization_endpoint"])
        sep = "&" if "?" in endpoint else "?"
        return LoginRequest(f"{endpoint}{sep}{urlencode(params)}", state, nonce, verifier)

    def exchange(self, code: str, redirect_uri: str, code_verifier: str) -> dict[str, Any]:
        body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.provider.client_id,
            "code_verifier": code_verifier,
        }
        if self.provider.client_secret is not None:
            body["client_secret"] = self.provider.client_secret.get_secret_value()
        try:
            with self.http() as client:
                res = client.post(
                    str(self.discovery()["token_endpoint"]),
                    data=body,
                    headers={"Accept": "application/json"},
                )
                data = res.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthError(f"no se pudo completar el SSO con {self.name}: {exc}") from None
        if res.status_code != 200 or not isinstance(data, dict) or "id_token" not in data:
            detail = (
                data.get("error_description") or data.get("error")
                if isinstance(data, dict)
                else None
            )
            raise AuthError(f"el proveedor SSO rechazó el código: {detail or res.status_code}")
        return data

    def validate(self, id_token: str, nonce: str) -> Identity:
        from joserfc import jwt
        from joserfc.errors import JoseError
        from joserfc.jwk import KeySet

        claims_registry = jwt.JWTClaimsRegistry(
            leeway=60,
            iss={"essential": True, "value": self.discovery()["issuer"]},
            aud={"essential": True, "value": self.provider.client_id},
            exp={"essential": True},
            sub={"essential": True},
            nonce={"essential": True, "value": nonce},
        )
        token = None
        for refresh in (False, True):  # rotación de claves: reintenta con el JWKS nuevo
            try:
                keys = KeySet.import_key_set(self._keys(refresh))  # type: ignore[arg-type]
                token = jwt.decode(id_token, keys, algorithms=_SIGNING_ALGS)
                break
            except JoseError as exc:
                if refresh:
                    raise AuthError(f"ID token inválido: {exc}") from None
        if token is None:
            raise AuthError("ID token inválido")
        claims: dict[str, Any] = dict(token.claims)
        try:
            claims_registry.validate(claims)
        except JoseError as exc:
            raise AuthError(f"ID token inválido: {exc}") from None
        return self._identity(claims)

    def _identity(self, claims: dict[str, Any]) -> Identity:
        email = claims.get("email")
        if not email and self.provider.kind == "entra":
            email = claims.get("preferred_username") or claims.get("upn")
        if not isinstance(email, str) or "@" not in email:
            raise AuthError("el proveedor SSO no informó un email")
        if claims.get("email_verified") is False:
            raise AuthError("el email del proveedor SSO no está verificado")
        domain = email.rsplit("@", 1)[1].lower()
        allowed = [d.lower() for d in self.provider.allowed_domains]
        if allowed and domain not in allowed:
            raise AuthError(f"el dominio {domain} no está habilitado para este SSO")
        raw_groups = claims.get(self.provider.groups_claim) or []
        groups = [str(g) for g in raw_groups] if isinstance(raw_groups, list) else []
        name = claims.get("name") or email
        return Identity(str(claims["sub"]), email.lower(), str(name), groups, claims)
