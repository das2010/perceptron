"""Jobs largos en segundo plano (SPEC §10): `202 Accepted` + `job_id`, progreso por
`WS /jobs/{job_id}`. Los jobs locales corren en un pool de hilos; los del Team Server corren
en workers (Capa 5b) y acá solo se siguen (`track` + `apply`) con la misma API y el mismo WS."""

from __future__ import annotations

import logging
import threading
import traceback
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from perceptron.core.events import EventBus
from perceptron.core.ids import IdPrefix, new_id
from perceptron.core.logging import log_context
from perceptron.domain.models import utcnow

logger = logging.getLogger(__name__)

JOB_TOPIC = "job.event"
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL = {"succeeded", "failed", "cancelled"}
HISTORY_EVENTS = 2000  # por job: alcanza para las curvas de un estudio típico


class Job(BaseModel):
    id: str
    kind: str
    status: JobStatus = "queued"
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: dict[str, Any] | None = None
    refs: dict[str, str] = Field(default_factory=dict, description="Ids relacionados (study_id, …)")
    runner: str = Field(default="local", description="Dónde corre: local o la cola del servidor")
    worker: str | None = Field(default=None, description="Worker remoto que lo ejecuta")


class JobContext:
    def __init__(self, manager: JobManager, job: Job) -> None:
        self._manager = manager
        self.job = job
        self.cancel_callback: Callable[[], None] | None = None

    @property
    def cancelled(self) -> bool:
        return self.job.status == "cancelled"

    def emit(self, kind: str, **data: Any) -> None:
        self.job.progress = {"last": kind, **data}
        self._manager.record(self.job.id, kind, data)
        self._manager.bus.publish(JOB_TOPIC, job_id=self.job.id, kind=kind, data=data)


class JobManager:
    def __init__(self, bus: EventBus, max_workers: int = 2) -> None:
        self.bus = bus
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="perceptron-job"
        )
        self._jobs: dict[str, Job] = {}
        self._contexts: dict[str, JobContext] = {}
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def submit(
        self, kind: str, fn: Callable[[JobContext], Any], *, refs: dict[str, str] | None = None
    ) -> Job:
        job = Job(id=new_id(IdPrefix.JOB), kind=kind, refs=refs or {})
        ctx = JobContext(self, job)
        with self._lock:
            self._jobs[job.id] = job
            self._contexts[job.id] = ctx
        self._executor.submit(self._run, ctx, fn)
        return job

    def _run(self, ctx: JobContext, fn: Callable[[JobContext], Any]) -> None:
        job = ctx.job
        if job.status == "cancelled":
            return
        job.status = "running"
        job.started_at = utcnow()
        ctx.emit("started")
        try:
            with log_context(job_id=job.id):
                result = fn(ctx)
            job.result = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
            if not ctx.cancelled:
                job.status = "succeeded"
        except Exception as e:
            logger.exception("job falló", extra={"job_id": job.id})
            job.status = "failed"
            job.error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()[-3000:],
            }
        finally:
            job.finished_at = utcnow()
            ctx.emit("finished", status=job.status)

    def track(
        self,
        kind: str,
        *,
        refs: dict[str, str] | None = None,
        runner: str = "remote",
        cancel: Callable[[], None] | None = None,
    ) -> Job:
        """Job que corre en otro proceso (worker del Team Server); su progreso llega por `apply`."""
        job = Job(id=new_id(IdPrefix.JOB), kind=kind, refs=refs or {}, runner=runner)
        ctx = JobContext(self, job)
        ctx.cancel_callback = cancel
        with self._lock:
            self._jobs[job.id] = job
            self._contexts[job.id] = ctx
        return job

    def apply(self, job_id: str, kind: str, data: dict[str, Any]) -> None:
        """Evento de un job remoto: actualiza el estado y lo publica como si fuera local."""
        ctx = self._contexts.get(job_id)
        if ctx is None:
            return
        job = ctx.job
        if kind == "started":
            if job.status != "cancelled":
                job.status = "running"
            job.started_at = utcnow()
            job.worker = data.get("worker")
        elif kind == "finished":
            if job.status != "cancelled":
                job.status = data.get("status", "failed")
            job.result = data.get("result")
            job.error = data.get("error")
            job.finished_at = utcnow()
            ctx.emit("finished", status=job.status)
            return
        ctx.emit(kind, **data)

    def record(self, job_id: str, kind: str, data: dict[str, Any]) -> None:
        """Últimos eventos del job: quien se conecta tarde al WS (o reconecta) los recibe."""
        with self._lock:
            history = self._history.setdefault(job_id, deque(maxlen=HISTORY_EVENTS))
            history.append({"job_id": job_id, "kind": kind, "data": data})

    def history(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history.get(job_id, ()))

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        ctx = self._contexts.get(job_id)
        if ctx is None:
            return None
        if ctx.job.status not in TERMINAL:
            ctx.job.status = "cancelled"
            if ctx.cancel_callback is not None:
                ctx.cancel_callback()
        return ctx.job

    def shutdown(self) -> None:
        # Los jobs remotos siguen en sus workers aunque este proceso se detenga.
        for job_id, job in list(self._jobs.items()):
            if job.runner == "local":
                self.cancel(job_id)
        self._executor.shutdown(wait=False, cancel_futures=True)
