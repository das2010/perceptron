"""Lanzamiento de estudios en el Team Server: cuotas, cola por recurso y seguimiento (RF-SRV-04).

El Engine delega en `ctx.launch_study`. El servidor instala `QueueLauncher`, que:
1. aplica las cuotas de estudios en curso por usuario y por workspace;
2. elige la cola (`gpu` si se pidió CUDA/ROCm/XPU; si no, `cpu`);
3. registra un job remoto (`JobManager.track`) y lo encola.

`start_event_bridge` trae los eventos de los workers al `EventBus` del servidor.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from perceptron.api.jobs import JOB_TOPIC, TERMINAL, Job
from perceptron.core.errors import RateLimitedError
from perceptron.domain.models import Study, utcnow
from perceptron.services.studies import launch_local, needs_gpu
from perceptron_server.queue.dispatch import Dispatcher
from perceptron_server.queue.jobstore import DbJobStore
from perceptron_server.queue.relay import CONTROL_CHANNEL, EVENTS_CHANNEL, Relay
from perceptron_server.queue.worker import HEARTBEAT_TOPIC

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron_server.accounts import Principal
    from perceptron_server.settings import ServerSettings

# Quién hace el request en curso (lo fija el middleware de sesión).
logger = logging.getLogger(__name__)

CURRENT_PRINCIPAL: ContextVar[Principal | None] = ContextVar("perceptron_principal", default=None)


class WorkerRegistry:
    """Últimos latidos de los workers (hardware, colas y job en curso)."""

    def __init__(self, stale_after_s: float) -> None:
        self.stale_after_s = stale_after_s
        self._seen: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def update(self, heartbeat: dict[str, Any]) -> None:
        name = str(heartbeat.get("worker", "?"))
        with self._lock:
            self._seen[name] = (time.monotonic(), heartbeat)

    def list(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            return [
                {**hb, "seen_s_ago": round(now - at, 1)}
                for at, hb in sorted(self._seen.values(), key=lambda p: str(p[1].get("worker")))
                if now - at <= self.stale_after_s
            ]


class QuotaGuard:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings

    def check(self, ctx: EngineContext, study: Study, who: Principal | None) -> None:
        running = [j for j in ctx.jobs.list() if j.kind == "study" and j.status not in TERMINAL]
        if who is not None and not who.is_server_admin:
            mine = sum(1 for j in running if j.refs.get("user_id") == who.user.id)
            if mine >= self.settings.max_running_studies_per_user:
                raise RateLimitedError(
                    "llegaste a tu cuota de estudios en curso; esperá a que termine alguno",
                    details={"quota": self.settings.max_running_studies_per_user, "scope": "user"},
                )
        workspace = ctx.projects.get(study.project_id).workspace_id
        if workspace:
            in_ws = sum(1 for j in running if j.refs.get("workspace_id") == workspace)
            if in_ws >= self.settings.max_running_studies_per_workspace:
                raise RateLimitedError(
                    "el workspace llegó a su cuota de estudios en curso",
                    details={
                        "quota": self.settings.max_running_studies_per_workspace,
                        "scope": "workspace",
                    },
                )

    def refs(self, ctx: EngineContext, study: Study, who: Principal | None) -> dict[str, str]:
        refs = {"study_id": study.id, "project_id": study.project_id}
        workspace = ctx.projects.get(study.project_id).workspace_id
        if workspace:
            refs["workspace_id"] = workspace
        if who is not None:
            refs["user_id"] = who.user.id
        return refs


class LocalLauncher:
    """Sin cola configurada: el estudio corre en el proceso del servidor, con cuotas."""

    def __init__(self, quota: QuotaGuard) -> None:
        self.quota = quota

    def __call__(self, ctx: EngineContext, study: Study) -> Job:
        who = CURRENT_PRINCIPAL.get()
        self.quota.check(ctx, study, who)
        job = launch_local(ctx, study)
        job.refs.update(self.quota.refs(ctx, study, who))
        return job


class QueueLauncher:
    def __init__(self, quota: QuotaGuard, relay: Relay, dispatcher: Dispatcher) -> None:
        self.quota = quota
        self.relay = relay
        self.dispatcher = dispatcher

    def __call__(self, ctx: EngineContext, study: Study) -> Job:
        who = CURRENT_PRINCIPAL.get()
        self.quota.check(ctx, study, who)
        queue = "gpu" if needs_gpu(study) else "cpu"
        holder: dict[str, str] = {}

        def cancel() -> None:  # el id del job se conoce recién después de `track`
            self.canceller(ctx, study.id, holder["id"])()

        refs = {**self.quota.refs(ctx, study, who), "queue": queue}
        job = ctx.jobs.track("study", refs=refs, runner=f"queue:{queue}", cancel=cancel)
        holder["id"] = job.id
        self.dispatcher.dispatch(job.id, study.id, queue)
        return job

    def canceller(self, ctx: EngineContext, study_id: str, job_id: str) -> Callable[[], None]:
        def cancel() -> None:
            # Persistida: si el estudio sigue en la cola, el worker que lo tome no lo entrena
            # (el mensaje del canal solo llega a un worker que ya lo está ejecutando).
            repo = ctx.repo(Study)
            current = repo.find(study_id)
            if current is not None and current.cancel_requested_at is None:
                repo.update(current.model_copy(update={"cancel_requested_at": utcnow()}))
            self.relay.publish(CONTROL_CHANNEL, {"cancel": job_id})

        return cancel

    def restore(self, ctx: EngineContext, store: DbJobStore) -> int:
        """Al arrancar: vuelve a seguir los jobs que quedaron en la cola o en los workers
        (se ven, se pueden cancelar y cuentan para las cuotas). Devuelve cuántos retomó."""
        ctx.jobs.store = store
        jobs = store.load()
        for job in jobs:
            study_id = job.refs.get("study_id")
            cancel = None
            if job.status not in TERMINAL and study_id is not None:
                cancel = self.canceller(ctx, study_id, job.id)
            ctx.jobs.restore(job, cancel=cancel)
        return len(jobs)


LOST_MESSAGE = (
    "El worker se reinició o se cayó durante el entrenamiento y el estudio quedó interrumpido. "
    "Los trials terminados se conservan: podés reanudarlo."
)
LOST_MISSES = 2  # latidos seguidos del worker sin el job para darlo por perdido


class LostJobDetector:
    """Jobs «en curso» que su worker ya no está entrenando (se reinició o se cayó).

    El mensaje de Celery queda sin confirmar y la cola recién lo devuelve tras el visibility
    timeout (días): sin esto el estudio figuraba en curso para siempre y ocupaba la cuota.
    Se da por perdido cuando, pasado el margen desde que empezó, dos latidos seguidos del mismo
    worker no lo informan como su job.
    """

    def __init__(self, ctx: EngineContext, grace_s: float) -> None:
        self.ctx = ctx
        self.grace_s = grace_s
        self._misses: dict[str, int] = {}

    def on_heartbeat(self, heartbeat: dict[str, Any]) -> list[str]:
        worker, busy = heartbeat.get("worker"), heartbeat.get("busy_job")
        now = utcnow()
        lost: list[str] = []
        for job in self.ctx.jobs.list():
            if job.status != "running" or job.worker != worker or job.id == busy:
                self._misses.pop(job.id, None)
                continue
            if job.started_at and (now - job.started_at).total_seconds() < self.grace_s:
                continue
            self._misses[job.id] = self._misses.get(job.id, 0) + 1
            if self._misses[job.id] >= LOST_MISSES:
                self._misses.pop(job.id, None)
                self._lose(job)
                lost.append(job.id)
        return lost

    def _lose(self, job: Job) -> None:
        from perceptron.services.study_control import WORKER_LOST, mark_interrupted

        logger.warning("job perdido: su worker ya no lo entrena", extra={"job_id": job.id})
        error = {"type": WORKER_LOST, "message": LOST_MESSAGE}
        self.ctx.jobs.apply(job.id, "finished", {"status": "failed", "error": error})
        study_id = job.refs.get("study_id")
        if study_id:
            mark_interrupted(self.ctx, study_id)


def start_event_bridge(
    ctx: EngineContext, relay: Relay, workers: WorkerRegistry
) -> Callable[[], None]:
    """Re-publica en el servidor lo que emiten los workers (progreso, épocas, latidos)."""
    detector = LostJobDetector(ctx, grace_s=workers.stale_after_s)

    def on_event(message: dict[str, Any]) -> None:
        topic = message.get("topic")
        payload = message.get("payload") or {}
        if not isinstance(topic, str) or not isinstance(payload, dict):
            return
        if topic == JOB_TOPIC:
            ctx.jobs.apply(
                str(payload.get("job_id")), str(payload.get("kind")), payload.get("data") or {}
            )
        elif topic == HEARTBEAT_TOPIC:
            workers.update(payload)
            detector.on_heartbeat(payload)
        else:
            ctx.events.publish(topic, **payload)

    return relay.listen(EVENTS_CHANNEL, on_event)
