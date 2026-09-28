"""Cliente del Team Server para el desktop (RF-SRV-03/04): login, tokens y llamadas.

El desktop guarda los servidores conocidos en `remotes.json` del workspace (sin secretos) y
el token de refresco en el almacén de secretos (keychain). El token de acceso vive en memoria
y se renueva solo cuando vence (rotación del refresco en el servidor).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from perceptron.core.errors import AuthError, NotFoundError, PerceptronError, ValidationError
from perceptron.llm.secrets import SecretStore

HttpFactory = Callable[[], httpx.Client]


def default_http() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(30.0, read=300.0), follow_redirects=False)


class RemoteError(PerceptronError):
    """El servidor respondió con error (se propaga su código y mensaje)."""

    code = "remote_error"
    http_status = 502


class RemoteServer(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    url: str
    email: str
    user_id: str | None = None
    display_name: str | None = None


class RemoteRegistry:
    """Servidores de equipo configurados en este desktop."""

    def __init__(self, workspace_dir: Path, secrets: SecretStore) -> None:
        self.file = workspace_dir / "remotes.json"
        self.secrets = secrets
        self._lock = threading.Lock()

    def _read(self) -> dict[str, RemoteServer]:
        if not self.file.is_file():
            return {}
        raw = json.loads(self.file.read_text(encoding="utf-8"))
        return {k: RemoteServer.model_validate(v) for k, v in raw.items()}

    def list(self) -> list[RemoteServer]:
        with self._lock:
            return sorted(self._read().values(), key=lambda s: s.name)

    def get(self, name: str) -> RemoteServer:
        with self._lock:
            server = self._read().get(name)
        if server is None:
            raise NotFoundError(f"no hay un servidor de equipo {name!r}")
        return server

    def save(self, server: RemoteServer, refresh_token: str) -> None:
        with self._lock:
            items = self._read()
            items[server.name] = server
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(
                json.dumps({k: v.model_dump() for k, v in items.items()}, indent=2),
                encoding="utf-8",
            )
        self.secrets.set(self.secret_name(server.name), refresh_token)

    def remove(self, name: str) -> None:
        with self._lock:
            items = self._read()
            if items.pop(name, None) is None:
                raise NotFoundError(f"no hay un servidor de equipo {name!r}")
            self.file.write_text(
                json.dumps({k: v.model_dump() for k, v in items.items()}, indent=2),
                encoding="utf-8",
            )
        self.secrets.delete(self.secret_name(name))

    @staticmethod
    def secret_name(name: str) -> str:
        return f"remote/{name}/refresh"


def _raise_for(res: httpx.Response) -> None:
    if res.status_code < 400:
        return
    try:
        body = res.json()
    except ValueError:
        body = {}
    message = body.get("message") if isinstance(body, dict) else None
    code = body.get("code") if isinstance(body, dict) else None
    if res.status_code == 401:
        raise AuthError(message or "el servidor rechazó la sesión: volvé a conectarte")
    err = RemoteError(
        message or f"el servidor respondió {res.status_code}",
        details={"status": res.status_code, "code": code, "details": body.get("details")}
        if isinstance(body, dict)
        else {"status": res.status_code},
    )
    raise err


class RemoteClient:
    def __init__(
        self, server: RemoteServer, registry: RemoteRegistry, http: HttpFactory = default_http
    ) -> None:
        self.server = server
        self.registry = registry
        self.http = http
        self.base = server.url.rstrip("/") + "/api/v1"
        self._access: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def login(
        url: str, email: str, password: str, http: HttpFactory = default_http
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Tokens y usuario (`/auth/token` + `/auth/me`) de un servidor nuevo."""
        base = url.rstrip("/") + "/api/v1"
        try:
            with http() as client:
                res = client.post(f"{base}/auth/token", json={"email": email, "password": password})
                _raise_for(res)
                tokens: dict[str, Any] = res.json()
                me = client.get(
                    f"{base}/auth/me",
                    headers={"Authorization": f"Bearer {tokens['access_token']}"},
                )
                _raise_for(me)
        except httpx.HTTPError as exc:
            raise ValidationError(f"no se pudo conectar con {url}: {exc}") from None
        return tokens, me.json()

    # ------------------------------------------------------------------ tokens

    def _refresh(self) -> str:
        refresh = self.registry.secrets.get(RemoteRegistry.secret_name(self.server.name))
        if not refresh:
            raise AuthError(f"sin sesión en {self.server.name}: volvé a conectarte")
        with self.http() as client:
            res = client.post(f"{self.base}/auth/refresh", json={"refresh_token": refresh})
        _raise_for(res)
        data = res.json()
        self.registry.secrets.set(
            RemoteRegistry.secret_name(self.server.name), data["refresh_token"]
        )
        self._access = str(data["access_token"])
        return self._access

    def access_token(self) -> str:
        with self._lock:
            return self._access or self._refresh()

    # ------------------------------------------------------------------ llamadas

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in (0, 1):
            with self._lock:
                token = self._access or self._refresh()
            headers = {**kwargs.pop("headers", {}), "Authorization": f"Bearer {token}"}
            try:
                with self.http() as client:
                    res = client.request(method, f"{self.base}{path}", headers=headers, **kwargs)
            except httpx.HTTPError as exc:
                raise RemoteError(f"sin conexión con {self.server.name}: {exc}") from None
            if res.status_code == 401 and attempt == 0:
                with self._lock:
                    self._access = None  # vencido: renueva y reintenta una vez
                continue
            return res
        return res

    def json(self, method: str, path: str, **kwargs: Any) -> Any:
        res = self.request(method, path, **kwargs)
        _raise_for(res)
        return res.json() if res.content else None

    def ws_url(self, path: str) -> str:
        url = f"{self.base}{path}"
        return (
            "wss://" + url[len("https://") :]
            if url.startswith("https://")
            else "ws://" + url[len("http://") :]
        )
