"""Workers del Team Server (RF-SRV-04): ejecutan estudios que el servidor encoló.

Cada worker es un proceso con su propio `EngineContext` sobre la misma base (PostgreSQL) y el
mismo workspace (volumen compartido). Recibe solo `(job_id, study_id)`: el estudio guarda todo
lo necesario para ejecutarse (ver `perceptron.services.studies`). El progreso vuelve al
servidor por el relay, y las cancelaciones llegan por el canal de control.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Any

from perceptron.api.context import EngineContext
from perceptron.api.jobs import JOB_TOPIC
from perceptron.core.config import Settings
from perceptron.core.events import Event
from perceptron.domain.models import Study
from perceptron.hpo.study import StudyControl
from perceptron.services.studies import execute_study
from perceptron_server.queue.relay import CONTROL_CHANNEL, EVENTS_CHANNEL, Relay

logger = logging.getLogger(__name__)
HEARTBEAT_TOPIC = "worker.heartbeat"


def worker_name() -> str:
    return os.environ.get("PERCEPTRON_WORKER_NAME") or socket.gethostname()


@dataclass
class WorkerRuntime:
    """Contexto de un proceso worker: Engine, relay y latido con su hardware."""

    ctx: EngineContext
    relay: Relay
    queues: list[str]
    name: str = field(default_factory=worker_name)
    heartbeat_s: float = 15.0
    busy: str | None = None
    last_finished: tuple[str, str] | None = None  # (job, estado) del último estudio
    _stop: threading.Event = field(default_factory=threading.Event)

    @classmethod
    def create(
        cls, settings: Settings, relay: Relay, queues: list[str], heartbeat_s: float = 15.0
    ) -> WorkerRuntime:
        return cls(EngineContext.create(settings), relay, queues, heartbeat_s=heartbeat_s)

    # ------------------------------------------------------------------ latido

    def heartbeat(self) -> dict[str, Any]:
        from perceptron.training.hardware import detect_hardware

        hw = detect_hardware(self.ctx.settings.workspace_dir)
        message = {
            "worker": self.name,
            "queues": self.queues,
            "busy_job": self.busy,
            "device": hw.recommended_device,
            "gpus": [g.model_dump(mode="json") for g in hw.gpus],
            "ram_available_gb": hw.ram_available_gb,
        }
        self.relay.publish(EVENTS_CHANNEL, {"topic": HEARTBEAT_TOPIC, "payload": message})
        return message

    def start_heartbeat(self) -> None:
        def loop() -> None:
            while True:
                try:
                    self.heartbeat()
                except Exception:
                    logger.exception("latido del worker falló")
                if self._stop.wait(self.heartbeat_s):
                    return

        threading.Thread(target=loop, name="perceptron-heartbeat", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self.ctx.close()

    # ------------------------------------------------------------------ estudios

    def run_study(self, job_id: str, study_id: str) -> str:
        """Ejecuta el estudio y devuelve el estado final (succeeded, failed o cancelled)."""
        control = StudyControl()

        def on_control(msg: dict[str, Any]) -> None:
            if msg.get("cancel") == job_id:
                control.cancel()

        def forward(ev: Event) -> None:
            self.relay.publish(EVENTS_CHANNEL, {"topic": ev.topic, "payload": ev.payload})

        def emit(kind: str, **data: Any) -> None:
            self.ctx.events.publish(JOB_TOPIC, job_id=job_id, kind=kind, data=data)

        stop_control = self.relay.listen(CONTROL_CHANNEL, on_control)
        unsubscribe = self.ctx.events.subscribe("*", forward)
        self.busy = job_id
        status, result, error = "failed", None, None
        try:
            emit("started", worker=self.name)
            # Ya suscripto al canal de control: si la cancelación se pidió mientras esperaba en
            # la cola, está persistida en el estudio y no se entrena.
            study = self.ctx.repo(Study).get(study_id)
            if study.cancel_requested_at is not None:
                control.cancel()
                status = "cancelled"
            else:
                outcome = execute_study(self.ctx, study, control=control, emit=emit)
                result = outcome.model_dump(mode="json")
                status = "cancelled" if control.cancelled else "succeeded"
        except Exception as exc:
            logger.exception("el estudio falló en el worker", extra={"job_id": job_id})
            # El traceback queda en el log del worker, no en el job que ve el cliente.
            error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        finally:
            # Primero el latido "libre" y después el fin: quien vea el job terminado ya ve
            # el worker disponible en la cola.
            self.busy = None
            self.last_finished = (job_id, status)
            try:
                self.heartbeat()
            except Exception:
                logger.exception("latido del worker falló")
            emit("finished", status=status, result=result, error=error)
            unsubscribe()
            stop_control()
        return status
