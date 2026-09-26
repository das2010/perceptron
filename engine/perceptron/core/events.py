"""Bus de eventos in-process mínimo.

Desacopla productores (p. ej. el runner de entrenamiento) de consumidores
(WebSocket, tracking, auditoría). En el servidor se podrá puentear a Redis (Capa 5).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

WILDCARD = "*"


@dataclass(frozen=True, slots=True)
class Event:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


Handler = Callable[[Event], None]


class EventBus:
    """Publicación síncrona; un handler que falla no afecta a los demás."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Suscribe `handler` a `topic` (`"*"` recibe todo). Devuelve la función de baja."""
        self._handlers[topic].append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers[topic]:
                self._handlers[topic].remove(handler)

        return unsubscribe

    def publish(self, topic: str, **payload: Any) -> Event:
        event = Event(topic=topic, payload=payload)
        targets = [*self._handlers.get(topic, ()), *self._handlers.get(WILDCARD, ())]
        for handler in targets:
            try:
                handler(event)
            except Exception:
                logger.exception("event handler failed", extra={"topic": topic})
        return event
