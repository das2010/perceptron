"""Supervisor de runs (RF-TRN-05, RF-TRN-07, ADR-0015).

Lanza el worker en un subproceso, lee sus eventos JSONL, los publica en el
`EventBus` (→ WebSocket, tracking, HPO) y traduce la salida en un `RunResult`.
Permite cancelar y pausar (corte limpio con checkpoint) y reanudar desde el
último checkpoint.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from perceptron.core.events import EventBus
from perceptron.sandbox.process import python_args, sandbox_env
from perceptron.training.config import (
    RESULT_FILE,
    WORKER_LOG,
    RunConfig,
    RunEvent,
    RunResult,
    RunStatusName,
)

logger = logging.getLogger(__name__)

EVENT_TOPIC = "run.event"
STOP_GRACE_S = 60.0


class RunHandle:
    def __init__(self, config: RunConfig, bus: EventBus | None = None) -> None:
        self.config = config
        self.bus = bus
        self.events: list[RunEvent] = []
        self._requested: str | None = None
        path = config.save()
        if config.code is not None:
            # Código experto: proceso aislado con entorno mínimo (ADR-0025).
            args = python_args("perceptron.training.worker", str(path))
            env = sandbox_env(config.run_dir)
        else:
            args = [sys.executable, "-m", "perceptron.training.worker", str(path)]
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self._log = (config.run_dir / WORKER_LOG).open("a", encoding="utf-8")
        self.started_at = time.time()
        self.process = subprocess.Popen(  # noqa: S603 - intérprete actual y módulo propio
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
            cwd=config.run_dir,
        )

    @property
    def run_id(self) -> str:
        return self.config.run_id

    # --------------------------------------------------------------- control

    def _send(self, cmd: str) -> None:
        self._requested = cmd
        if self.process.stdin and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"cmd": cmd}) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def stop(self) -> None:
        """Cancelación limpia: el worker termina el batch, guarda checkpoint y sale."""
        self._send("stop")

    def pause(self) -> None:
        self._send("pause")

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()

    # --------------------------------------------------------------- eventos

    def iter_events(self) -> Iterator[RunEvent]:
        assert self.process.stdout is not None  # noqa: S101 - creado con PIPE
        for raw in self.process.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                ev = RunEvent.model_validate_json(line)
            except ValueError:
                logger.warning(
                    "línea no JSON del worker", extra={"run_id": self.run_id, "line": line[:200]}
                )
                continue
            self.events.append(ev)
            if self.bus is not None:
                self.bus.publish(EVENT_TOPIC, run_id=self.run_id, event=ev.model_dump(mode="json"))
            yield ev

    def wait(
        self, on_event: Callable[[RunEvent], None] | None = None, timeout: float | None = None
    ) -> RunResult:
        """Consume eventos hasta que el worker termina y devuelve el resultado."""
        if self.config.code is not None and self.config.sandbox.wall_seconds:
            limit = self.config.sandbox.wall_seconds
            timeout = min(timeout, limit) if timeout else limit
        timer = None
        if timeout:
            timer = threading.Timer(timeout, self._timeout)
            timer.daemon = True
            timer.start()
        try:
            for ev in self.iter_events():
                if on_event is not None:
                    on_event(ev)
            code = self.process.wait()
        finally:
            if timer:
                timer.cancel()
            if self.process.stdin:
                with contextlib.suppress(OSError):
                    self.process.stdin.close()
            self._log.close()
        return self._result(code)

    def _timeout(self) -> None:
        logger.warning("run excedió el tiempo, se detiene", extra={"run_id": self.run_id})
        self.stop()
        threading.Timer(STOP_GRACE_S, self.kill).start()

    def _result(self, code: int) -> RunResult:
        result_file = self.config.run_dir / RESULT_FILE
        duration = round(time.time() - self.started_at, 3)
        if code == 0 and result_file.is_file():
            result = RunResult.model_validate_json(result_file.read_text(encoding="utf-8"))
            if self._requested == "prune":
                result.status = "pruned"
            return result
        error = next((e.data for e in reversed(self.events) if e.event == "error"), None)
        if error is None:
            tail = _tail(self.config.run_dir / WORKER_LOG)
            error = {
                "code": "crash",
                "message": f"el worker terminó con código {code}",
                "log_tail": tail,
            }
        status: RunStatusName = "cancelled" if self._requested == "stop" else "failed"
        if self._requested == "prune":
            status = "pruned"
        return RunResult(run_id=self.run_id, status=status, duration_s=duration, error=error)

    def prune(self) -> None:
        """Poda de HPO: se corta como una cancelación pero se informa `pruned`."""
        self._send("stop")
        self._requested = "prune"


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def start_run(config: RunConfig, bus: EventBus | None = None) -> RunHandle:
    return RunHandle(config, bus)


def run_sync(
    config: RunConfig,
    bus: EventBus | None = None,
    on_event: Callable[[RunEvent], None] | None = None,
    timeout: float | None = None,
) -> RunResult:
    return start_run(config, bus).wait(on_event, timeout)
