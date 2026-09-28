"""Logging estructurado JSON con contexto (SPEC §13.4).

Los campos `project_id`, `run_id`, `job_id` y `llm_call_id` se propagan vía
`contextvars` y se agregan automáticamente a cada registro. Las claves `extra`
que parecen secretos se enmascaran (SPEC §13.2: secretos nunca en logs).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perceptron.core.config import LoggingSettings

CONTEXT_FIELDS = ("project_id", "run_id", "job_id", "llm_call_id")

_context: ContextVar[dict[str, str] | None] = ContextVar("perceptron_log_context", default=None)

# Atributos estándar de LogRecord: todo lo demás se considera `extra`.
_RESERVED = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

_SECRET_HINTS = ("token", "secret", "password", "api_key", "apikey", "authorization")
_MARKER = "_perceptron_handler"


def current_context() -> dict[str, str]:
    return dict(_context.get() or {})


@contextmanager
def log_context(**fields: str | None) -> Iterator[None]:
    """Agrega campos de contexto dentro del bloque (se anidan)."""
    unknown = set(fields) - set(CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"campos de contexto desconocidos: {sorted(unknown)}")
    merged = {**current_context(), **{k: v for k, v in fields.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


# Secretos dentro de texto: `?token=…` en URLs (access log de WebSockets), `Bearer …`.
_SECRET_IN_TEXT = re.compile(
    r"(?i)((?:[?&;]|\b)(?:token|access_token|refresh_token|api_key|apikey|password|secret)=)"
    r"[^&\s\"']+"
)
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def scrub(text: str) -> str:
    """Enmascara secretos embebidos en un mensaje o una URL."""
    return _BEARER.sub(r"\1***", _SECRET_IN_TEXT.sub(r"\1***", text))


def redact(key: str, value: Any) -> Any:
    if any(h in key.lower() for h in _SECRET_HINTS):
        return "***"
    if isinstance(value, dict):
        return {str(k): redact(str(k), v) for k, v in value.items()}
    if isinstance(value, str):
        return scrub(value)
    return value


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in current_context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class ScrubbingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": scrub(record.getMessage()),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                data[key] = redact(key, value)
        if record.exc_info:
            data["exc"] = scrub(self.formatException(record.exc_info))
        return json.dumps(data, ensure_ascii=False, default=str)


def configure_logging(settings: LoggingSettings, logs_dir: Path | None = None) -> None:
    """Configura el logger raíz. Idempotente: reemplaza los handlers propios previos."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter: logging.Formatter = (
        JsonFormatter()
        if settings.json_output
        else ScrubbingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.to_file and logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                logs_dir / "engine.log",
                maxBytes=settings.max_bytes,
                backupCount=settings.backup_count,
                encoding="utf-8",
            )
        )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(ContextFilter())
        setattr(handler, _MARKER, True)
        root.addHandler(handler)
    root.setLevel(settings.level)
