"""Fuentes de datos que llegan de a poco (RF-ING-05): APIs REST, WebSocket y archivos que
crecen. Alimentan el reentrenamiento automático (§7.15).

Cada `StreamSource` trae un **lote** a partir de un estado (cursor, página, offset) y devuelve
el estado nuevo. Los lotes se acumulan en un **buffer** del proyecto (Parquet por lote); el
reentrenamiento consume las filas nuevas desde la última vez. Las credenciales van al almacén
de secretos, nunca a la configuración.

Para sumar Kafka, MQTT u otro broker alcanza con implementar `StreamSource.fetch`.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import polars as pl
from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError
from perceptron.core.netguard import NetPolicy, check_url, same_origin
from perceptron.domain.models import utcnow

HttpFactory = Callable[[], httpx.Client]
MAX_BATCH_ROWS = 100_000


def default_http() -> httpx.Client:
    # Sin redirects: cada URL pasa por la política de red antes de conectar (SSRF).
    return httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=False)


class Batch(BaseModel):
    rows: list[dict[str, Any]]
    state: dict[str, Any] = Field(default_factory=dict)
    exhausted: bool = Field(default=False, description="No hay más por ahora")


class StreamSource(Protocol):
    def fetch(self, state: dict[str, Any], secret: str | None) -> Batch: ...


def dig(data: Any, path: str | None) -> Any:
    """`a.b.0.c` sobre dicts/listas (mapeo JSON → tabla)."""
    if not path:
        return data
    for part in path.split("."):
        if isinstance(data, list) and part.isdigit():
            data = data[int(part)] if int(part) < len(data) else None
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            return None
    return data


def _flatten(record: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(record, dict):
        return {prefix or "value": record}
    out: dict[str, Any] = {}
    for k, v in record.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


# ------------------------------------------------------------------ REST


class RestConfig(BaseModel):
    url: str = Field(pattern=r"^https?://")
    method: Literal["GET", "POST"] = "GET"
    params: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict, description="Sin secretos")
    auth: Literal["none", "bearer", "header"] = "none"
    auth_header: str = Field(default="Authorization", description="auth=header: nombre")
    records_path: str | None = Field(
        default=None, description="Dónde está la lista, p. ej. data.items"
    )
    pagination: Literal["none", "offset", "page", "cursor", "link"] = "none"
    page_size: int = Field(default=100, ge=1, le=10_000)
    offset_param: str = "offset"
    limit_param: str = "limit"
    page_param: str = "page"
    cursor_param: str = "cursor"
    cursor_path: str | None = Field(default=None, description="cursor: dónde viene el próximo")
    max_pages: int = Field(default=50, ge=1, le=10_000)


class RestSource:
    def __init__(
        self, config: RestConfig, http: HttpFactory = default_http, net: NetPolicy | None = None
    ) -> None:
        self.config = config
        self.http = http
        self.net = net or NetPolicy()

    def _headers(self, secret: str | None) -> dict[str, str]:
        headers = dict(self.config.headers)
        if self.config.auth != "none":
            if not secret:
                raise ValidationError("la fuente requiere un token")
            value = f"Bearer {secret}" if self.config.auth == "bearer" else secret
            headers[self.config.auth_header] = value
        return headers

    def fetch(self, state: dict[str, Any], secret: str | None) -> Batch:
        cfg = self.config
        rows: list[dict[str, Any]] = []
        state = dict(state)
        url: str | None = state.get("next_url") or cfg.url
        headers = self._headers(secret)
        exhausted = False
        with self.http() as client:
            for _ in range(cfg.max_pages):
                params = dict(cfg.params)
                if cfg.pagination == "offset":
                    params |= {
                        cfg.offset_param: str(state.get("offset", 0)),
                        cfg.limit_param: str(cfg.page_size),
                    }
                elif cfg.pagination == "page":
                    params |= {
                        cfg.page_param: str(state.get("page", 1)),
                        cfg.limit_param: str(cfg.page_size),
                    }
                elif cfg.pagination == "cursor" and state.get("cursor"):
                    params[cfg.cursor_param] = str(state["cursor"])
                if url is None:
                    exhausted = True
                    break
                if not same_origin(url, cfg.url):
                    # `Link: next` a otro host: no se sigue ni se le mandan las credenciales.
                    raise ValidationError("la paginación apunta a otro host")
                check_url(url, self.net)
                res = client.request(
                    cfg.method,
                    url,
                    params=params if cfg.pagination != "link" or url == cfg.url else None,
                    headers=headers,
                )
                if res.status_code in (401, 403):
                    raise ValidationError("la API rechazó las credenciales")
                res.raise_for_status()
                body = res.json()
                records = dig(body, cfg.records_path)
                if not isinstance(records, list):
                    raise ValidationError(
                        "la respuesta no tiene una lista de registros en records_path"
                    )
                rows.extend(_flatten(r) for r in records)
                if cfg.pagination == "none":
                    exhausted = True
                    break
                if cfg.pagination == "offset":
                    state["offset"] = int(state.get("offset", 0)) + len(records)
                    if len(records) < cfg.page_size:
                        exhausted = True
                        break
                elif cfg.pagination == "page":
                    if not records:
                        exhausted = True
                        break
                    state["page"] = int(state.get("page", 1)) + 1
                elif cfg.pagination == "cursor":
                    nxt = dig(body, cfg.cursor_path)
                    if not nxt:
                        exhausted = True
                        break
                    state["cursor"] = nxt
                else:  # link: cabecera Link rel="next"
                    nxt_url = res.links.get("next", {}).get("url")
                    if not nxt_url:
                        state.pop("next_url", None)
                        exhausted = True
                        break
                    url = str(res.url.join(nxt_url))
                    state["next_url"] = url
                if len(rows) >= MAX_BATCH_ROWS:
                    break
        return Batch(rows=rows, state=state, exhausted=exhausted)


# ------------------------------------------------------------------ WebSocket


class WebSocketConfig(BaseModel):
    url: str = Field(pattern=r"^wss?://")
    auth: Literal["none", "bearer", "header"] = "none"
    auth_header: str = "Authorization"
    records_path: str | None = None
    max_messages: int = Field(default=1000, ge=1, le=MAX_BATCH_ROWS)
    idle_timeout_s: float = Field(default=5.0, gt=0, le=300)


class WebSocketSource:
    """Lee mensajes JSON hasta `max_messages` o `idle_timeout_s` sin mensajes."""

    def __init__(self, config: WebSocketConfig, net: NetPolicy | None = None) -> None:
        self.config = config
        self.net = net or NetPolicy()

    def fetch(self, state: dict[str, Any], secret: str | None) -> Batch:
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        cfg = self.config
        headers: dict[str, str] = {}
        if cfg.auth != "none":
            if not secret:
                raise ValidationError("la fuente requiere un token")
            headers[cfg.auth_header] = f"Bearer {secret}" if cfg.auth == "bearer" else secret
        rows: list[dict[str, Any]] = []
        check_url(cfg.url, self.net, schemes=("ws", "wss"))
        with connect(cfg.url, additional_headers=headers, open_timeout=15) as ws:
            try:
                while len(rows) < cfg.max_messages:
                    raw = ws.recv(timeout=cfg.idle_timeout_s)
                    msg = json.loads(raw)
                    data = dig(msg, cfg.records_path)
                    items = data if isinstance(data, list) else [data]
                    rows.extend(_flatten(i) for i in items if i is not None)
            except (TimeoutError, ConnectionClosed):
                pass
        return Batch(
            rows=rows, state={**state, "messages": int(state.get("messages", 0)) + len(rows)}
        )


# ------------------------------------------------------------------ archivo que crece


class FileConfig(BaseModel):
    path: str = Field(min_length=1)
    max_lines: int = Field(default=10_000, ge=1, le=MAX_BATCH_ROWS)


class FileTailSource:
    """JSON Lines que crece (logs, exportaciones periódicas): lee desde el último offset."""

    def __init__(self, config: FileConfig) -> None:
        self.config = config

    def fetch(self, state: dict[str, Any], secret: str | None) -> Batch:
        path = Path(self.config.path)
        if not path.is_file():
            raise ValidationError(f"no existe {path}")
        offset = int(state.get("offset", 0))
        if path.stat().st_size < offset:  # rotado o truncado: vuelve a empezar
            offset = 0
        rows: list[dict[str, Any]] = []
        with path.open("rb") as f:
            f.seek(offset)
            while len(rows) < self.config.max_lines:
                line = f.readline()
                if not line or not line.endswith(b"\n"):
                    break  # línea incompleta: se lee la próxima vez
                offset = f.tell()
                if line.strip():
                    rows.append(_flatten(json.loads(line)))
        exhausted = len(rows) < self.config.max_lines
        return Batch(rows=rows, state={**state, "offset": offset}, exhausted=exhausted)


# ------------------------------------------------------------------ buffer


class StreamBuffer:
    """Lotes acumulados de una fuente (`projects/<id>/streams/<source>/`)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state_file = root / "state.json"

    def state(self) -> dict[str, Any]:
        if not self.state_file.is_file():
            return {}
        data: dict[str, Any] = json.loads(self.state_file.read_text(encoding="utf-8"))
        return data

    def save_state(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")

    def append(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = utcnow()
        df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
            pl.lit(stamp).alias("__received_at")
        )
        name = f"batch-{stamp.strftime('%Y%m%dT%H%M%S%f')}-{secrets.token_hex(3)}.parquet"
        df.write_parquet(self.root / name)
        return df.height

    def batches(self) -> list[Path]:
        return sorted(self.root.glob("batch-*.parquet")) if self.root.is_dir() else []

    def read(self, *, after: str | None = None) -> pl.DataFrame:
        """Filas de los lotes posteriores a `after` (nombre de lote); todas si es None."""
        parts = [p for p in self.batches() if after is None or p.name > after]
        if not parts:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")

    def stats(self) -> dict[str, Any]:
        parts = self.batches()
        rows = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in parts)
        return {
            "batches": len(parts),
            "rows": int(rows),
            "last_batch": parts[-1].name if parts else None,
            "state": self.state(),
        }


def build_source(
    kind: str,
    config: dict[str, Any],
    http: HttpFactory | None = None,
    *,
    net: NetPolicy | None = None,
) -> StreamSource:
    """`net`: política de destinos de red (`Settings.net_policy()`); None = sin restricción."""
    if kind == "rest":
        return RestSource(RestConfig.model_validate(config), http or default_http, net)
    if kind == "websocket":
        return WebSocketSource(WebSocketConfig.model_validate(config), net)
    if kind == "file":
        return FileTailSource(FileConfig.model_validate(config))
    raise ValidationError(f"tipo de fuente streaming desconocido: {kind}")
