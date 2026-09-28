"""UI de MLflow opcional (RF-TRK-02): la UI de Perceptron es la principal; esto es un enlace.

- Desktop: se levanta `mlflow ui` a pedido sobre el SQLite del workspace, solo en 127.0.0.1 y
  en un puerto libre; vive mientras vive el Engine.
- Team Server: la URL pública del MLflow server (`PERCEPTRON_MLFLOW_UI_URL`); sin ella no hay
  enlace (la URL interna de Docker no sirve desde el navegador).
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from perceptron.core.errors import NotFoundError, PerceptronError

logger = logging.getLogger(__name__)

READY_TIMEOUT_S = 60.0


class MlflowUiError(PerceptronError):
    code = "mlflow_ui_unavailable"
    http_status = 503


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class MlflowUi:
    def __init__(
        self,
        tracking_uri: str,
        artifacts_root: Path,
        *,
        public_url: str | None = None,
        can_launch: bool = True,
    ) -> None:
        """`can_launch`: solo el desktop levanta una UI propia (en el servidor, 127.0.0.1 no le
        sirve al navegador del usuario)."""
        self.can_launch = can_launch
        self.tracking_uri = tracking_uri
        self.artifacts_root = artifacts_root
        self.public_url = public_url.rstrip("/") if public_url else None
        self._proc: subprocess.Popen[bytes] | None = None
        self._port: int | None = None
        self._lock = threading.Lock()

    @property
    def remote(self) -> bool:
        return self.tracking_uri.startswith(("http://", "https://"))

    def base_url(self) -> str:
        """URL de la UI; en el desktop la levanta si hace falta."""
        if self.public_url:
            return self.public_url
        if self.remote or not self.can_launch:
            raise NotFoundError("el administrador no configuró la URL pública de MLflow")
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._port:
                return f"http://127.0.0.1:{self._port}"
            self._start()
            return f"http://127.0.0.1:{self._port}"

    def _start(self) -> None:
        port = _free_port()
        cmd = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            self.tracking_uri,
            "--default-artifact-root",
            self.artifacts_root.resolve().as_uri(),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # sin consola en Windows
        self._proc = subprocess.Popen(  # noqa: S603 - argumentos fijos, sin shell
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags
        )
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise MlflowUiError("la UI de MLflow no pudo arrancar")
            if _listening(port):
                self._port = port
                logger.info("UI de MLflow en http://127.0.0.1:%d", port)
                return
            time.sleep(0.25)
        self.close()
        raise MlflowUiError("la UI de MLflow no respondió a tiempo")

    def run_url(self, experiment_id: str, mlflow_run_id: str) -> str:
        return f"{self.base_url()}/#/experiments/{experiment_id}/runs/{mlflow_run_id}"

    def close(self) -> None:
        proc, self._proc, self._port = self._proc, None, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
