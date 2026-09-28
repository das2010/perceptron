"""Lanzamiento de estudios en el Team Server: cuotas, cola por recurso y seguimiento (RF-SRV-04).

El Engine delega en `ctx.launch_study`. El servidor instala `QueueLauncher`, que:
1. aplica las cuotas de estudios en curso por usuario y por workspace;
2. elige la cola (`gpu` si se pidió CUDA/ROCm/XPU; si no, `cpu`);
3. registra un job remoto (`JobManager.track`) y lo encola.

`start_event_bridge` trae los eventos de los workers al `EventBus` del servidor.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from perceptron.api.jobs import JOB_TOPIC, TERMINAL, Job
from perceptron.core.errors import RateLimitedError
from perceptron.domain.models import Study
from perceptron.services.studies import launch_local, needs_gpu
from perceptron_server.queue.dispatch import Dispatcher
from perceptron_server.queue.relay import CONTROL_CHANNEL, EVENTS_CHANNEL, Relay
from perceptron_server.queue.worker import HEARTBEAT_TOPIC

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron_server.accounts import Principal
    from perceptron_server.settings import ServerSettings

# Quién hace el request en curso (lo fija el middleware de sesión).
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

        def cancel() -> None:
            self.relay.publish(CONTROL_CHANNEL, {"cancel": holder["id"]})

        refs = {**self.quota.refs(ctx, study, who), "queue": queue}
        job = ctx.jobs.track("study", refs=refs, runner=f"queue:{queue}", cancel=cancel)
        holder["id"] = job.id
        self.dispatcher.dispatch(job.id, study.id, queue)
        return job


def start_event_bridge(
    ctx: EngineContext, relay: Relay, workers: WorkerRegistry
) -> Callable[[], None]:
    """Re-publica en el servidor lo que emiten los workers (progreso, épocas, latidos)."""

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
        else:
            ctx.events.publish(topic, **payload)

    return relay.listen(EVENTS_CHANNEL, on_event)
