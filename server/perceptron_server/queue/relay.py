"""Relay de eventos entre workers y el servidor (Valkey/Redis pub/sub, Capa 5b).

Los workers publican los eventos de su `EventBus` (progreso del job, épocas, trials, latidos)
y el servidor los re-publica en el suyo: los WebSocket del Engine (`/jobs/{id}`,
`/runs/{id}/live`) funcionan igual que con un job local. El canal de control lleva las
cancelaciones en sentido inverso.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

logger = logging.getLogger(__name__)

EVENTS_CHANNEL = "perceptron:events"
CONTROL_CHANNEL = "perceptron:control"

Listener = Callable[[dict[str, Any]], None]


def _default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    return str(value)


def encode(message: dict[str, Any]) -> str:
    return json.dumps(message, default=_default, ensure_ascii=False)


class Relay(Protocol):
    def publish(self, channel: str, message: dict[str, Any]) -> None: ...

    def listen(self, channel: str, listener: Listener) -> Callable[[], None]:
        """Suscribe `listener` (en un hilo propio); devuelve la función de baja."""
        ...

    def close(self) -> None: ...


class MemoryRelay:
    """Relay en memoria para tests y para correr worker y servidor en el mismo proceso."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = {}
        self._lock = threading.Lock()

    def publish(self, channel: str, message: dict[str, Any]) -> None:
        decoded: dict[str, Any] = json.loads(encode(message))  # mismo contrato que Redis
        with self._lock:
            targets = list(self._listeners.get(channel, ()))
        for listener in targets:
            try:
                listener(decoded)
            except Exception:
                logger.exception("listener del relay falló", extra={"channel": channel})

    def listen(self, channel: str, listener: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.setdefault(channel, []).append(listener)

        def stop() -> None:
            with self._lock:
                if listener in self._listeners.get(channel, []):
                    self._listeners[channel].remove(listener)

        return stop

    def close(self) -> None:
        with self._lock:
            self._listeners.clear()


class RedisRelay:
    """Pub/sub de Valkey/Redis (redis-py, MIT). Un hilo por suscripción."""

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, health_check_interval=30)
        self._threads: list[Any] = []

    def publish(self, channel: str, message: dict[str, Any]) -> None:
        try:
            self._client.publish(channel, encode(message))
        except Exception:  # el progreso no debe tumbar un entrenamiento
            logger.exception("no se pudo publicar en el relay", extra={"channel": channel})

    def listen(self, channel: str, listener: Listener) -> Callable[[], None]:
        pubsub = self._client.pubsub(ignore_subscribe_messages=True)  # type: ignore[no-untyped-call]

        def handle(raw: dict[str, Any]) -> None:
            try:
                listener(json.loads(raw["data"]))
            except Exception:
                logger.exception("listener del relay falló", extra={"channel": channel})

        pubsub.subscribe(**{channel: handle})
        thread = pubsub.run_in_thread(sleep_time=0.2, daemon=True)
        self._threads.append(thread)

        def stop() -> None:
            thread.stop()
            pubsub.close()

        return stop

    def close(self) -> None:
        for thread in self._threads:
            thread.stop()
        self._client.close()
