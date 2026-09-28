"""Telemetría de producto opcional y opt-in (D7, ADR-0036).

Apagada por defecto y sin endpoint de fábrica: solo se envía si la persona la activó y hay un
endpoint configurado (`PERCEPTRON_TELEMETRY__ENDPOINT`). Nunca incluye datos, nombres de
columnas, rutas, textos ni métricas de modelos: versión, SO, hardware agregado, contadores de
uso por feature y tipos de error (sin mensajes ni trazas). El contenido exacto queda a la
vista en `GET /system/telemetry` antes de aceptar.
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from perceptron import __version__

logger = logging.getLogger(__name__)


class Telemetry:
    def __init__(self, state_file: Path, endpoint: str | None) -> None:
        self.state_file = state_file
        self.endpoint = endpoint
        self._counts: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ consentimiento

    def _state(self) -> dict[str, Any]:
        if not self.state_file.is_file():
            return {}
        data: dict[str, Any] = json.loads(self.state_file.read_text(encoding="utf-8"))
        return data

    @property
    def enabled(self) -> bool:
        return bool(self._state().get("opt_in")) and bool(self.endpoint)

    def consent(self, opt_in: bool) -> dict[str, Any]:
        state = self._state()
        state["opt_in"] = opt_in
        state.setdefault("installation_id", uuid.uuid4().hex)  # aleatorio, no identifica a nadie
        state["asked"] = True
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state), encoding="utf-8")
        return state

    def status(self) -> dict[str, Any]:
        state = self._state()
        return {
            "asked": bool(state.get("asked")),
            "opt_in": bool(state.get("opt_in")),
            "endpoint_configured": bool(self.endpoint),
            "preview": self.payload(),
        }

    # ------------------------------------------------------------------ recolección

    def count(self, feature: str) -> None:
        with self._lock:
            self._counts[feature] += 1

    def error(self, kind: str) -> None:
        with self._lock:
            self._errors[kind] += 1

    def payload(self) -> dict[str, Any]:
        import psutil

        with self._lock:
            counts, errors = dict(self._counts), dict(self._errors)
        return {
            "installation_id": self._state().get("installation_id"),
            "version": __version__,
            "os": platform.system(),
            "os_release": platform.release(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1024**3),
            "features": counts,
            "errors": errors,
        }

    def flush(self, http: httpx.Client | None = None) -> bool:
        """Envía y reinicia los contadores. Sin consentimiento o sin endpoint, no hace nada."""
        if not self.enabled or not self.endpoint:
            return False
        body = self.payload()
        try:
            client = http or httpx.Client(timeout=10.0)
            with client:
                client.post(self.endpoint, json=body).raise_for_status()
        except httpx.HTTPError:
            logger.info("no se pudo enviar la telemetría (se reintenta más tarde)")
            return False
        with self._lock:
            self._counts.clear()
            self._errors.clear()
        return True
