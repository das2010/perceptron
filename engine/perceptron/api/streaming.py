"""Puente EventBus (hilos) → WebSocket (asyncio)."""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from perceptron.core.events import Event, EventBus

TOKEN_HEADER = "X-Perceptron-Token"  # noqa: S105 - nombre de cabecera
WS_UNAUTHORIZED = 4401


def ws_authorized(ws: WebSocket, expected: str | None) -> bool:
    """Los WebSocket no pasan por el middleware HTTP: el token va en cabecera o `?token=`."""
    if not expected:
        return True
    got = ws.headers.get(TOKEN_HEADER) or ws.query_params.get("token") or ""
    return secrets.compare_digest(got, expected)


async def stream_events(
    ws: WebSocket,
    bus: EventBus,
    topic: str,
    match: Callable[[Event], bool],
    *,
    initial: list[dict[str, Any]] | None = None,
    until: Callable[[dict[str, Any]], bool] | None = None,
) -> None:
    """Reenvía al cliente los eventos de `topic` que cumplen `match` hasta `until` o desconexión."""
    token = ws.app.state.ctx.settings.api.token
    if not ws_authorized(ws, token.get_secret_value() if token else None):
        await ws.close(code=WS_UNAUTHORIZED)
        return
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def handler(ev: Event) -> None:
        if match(ev):
            loop.call_soon_threadsafe(queue.put_nowait, ev.payload)

    unsubscribe = bus.subscribe(topic, handler)
    try:
        for msg in initial or []:
            await ws.send_json(msg)
            if until and until(msg):
                return
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
            if until and until(msg):
                return
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
        with contextlib.suppress(RuntimeError):
            await ws.close()
