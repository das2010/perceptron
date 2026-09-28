"""Envío de estudios a los workers: Celery (producción) o un hilo (tests, un solo nodo)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from perceptron_server.queue.worker import WorkerRuntime

TASK_NAME = "perceptron.run_study"
QUEUES = ("gpu", "cpu")


class Dispatcher(Protocol):
    def dispatch(self, job_id: str, study_id: str, queue: str) -> None: ...


def make_celery(broker_url: str) -> Any:
    """App Celery (BSD) sobre Valkey/Redis; tareas largas: ack tardío y sin prefetch."""
    from celery import Celery

    app = Celery("perceptron", broker=broker_url, set_as_current=False)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        task_ignore_result=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_default_queue="cpu",
        broker_connection_retry_on_startup=True,
        worker_hijack_root_logger=False,
        # Un estudio puede durar horas: que el broker no lo reentregue antes de tiempo.
        broker_transport_options={"visibility_timeout": 7 * 24 * 3600},
    )
    return app


class CeleryDispatcher:
    def __init__(self, broker_url: str) -> None:
        self.app = make_celery(broker_url)

    def dispatch(self, job_id: str, study_id: str, queue: str) -> None:
        self.app.send_task(TASK_NAME, args=[job_id, study_id], queue=queue)


class ThreadDispatcher:
    """Ejecuta en un hilo con un `WorkerRuntime` propio (otro EngineContext, mismo storage)."""

    def __init__(self, runtime: WorkerRuntime) -> None:
        self.runtime = runtime
        self._lock = threading.Lock()  # un estudio a la vez, como un worker `solo`

    def dispatch(self, job_id: str, study_id: str, queue: str) -> None:
        def run() -> None:
            with self._lock:
                self.runtime.run_study(job_id, study_id)

        threading.Thread(target=run, name=f"worker-{job_id}", daemon=True).start()
