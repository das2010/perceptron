"""Estudios en el Team Server lanzados desde el desktop (aceptación de la Capa 5).

El desktop sube lo necesario, crea el estudio en el servidor (que lo manda a la cola de
workers GPU/CPU) y sigue su progreso por el WebSocket del servidor. Cada evento se re-publica
en un job local (`JobManager.track/apply`), así que la UI del desktop muestra el entrenamiento
en vivo igual que uno local. Al terminar, baja el estudio y sus runs.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from perceptron.api.jobs import TERMINAL, Job
from perceptron.remote.client import RemoteClient
from perceptron.remote.sync import ProjectSync

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

logger = logging.getLogger(__name__)


def _follow(client: RemoteClient, remote_job_id: str, on_event: Any) -> None:
    """Reenvía los eventos del job remoto hasta `finished` (o hasta que se corte el WS)."""
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    url = client.ws_url(f"/jobs/{remote_job_id}")
    try:
        with connect(
            url,
            additional_headers={"Authorization": f"Bearer {client.access_token()}"},
            open_timeout=30,
        ) as ws:
            for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("kind")
                if kind == "status":
                    if msg.get("data", {}).get("status") in TERMINAL:
                        return
                    continue
                if kind == "finished":
                    return
                on_event(str(kind), msg.get("data") or {})
    except (OSError, WebSocketException) as exc:
        # Sin WS (proxy que no lo deja pasar, corte): el estado final se consulta igual.
        logger.warning("se cortó el seguimiento en vivo del job remoto: %s", exc)


def launch_remote_study(
    ctx: EngineContext,
    client: RemoteClient,
    project_id: str,
    body: dict[str, Any],
    *,
    workspace_id: str | None = None,
) -> Job:
    """Job local que representa el estudio remoto (runner `remote:<servidor>`)."""
    name = client.server.name
    remote: dict[str, str] = {}

    def cancel() -> None:
        if "study_id" in remote:
            try:
                client.json("POST", f"/studies/{remote['study_id']}/cancel")
            except Exception:
                logger.exception("no se pudo cancelar el estudio remoto")

    job = ctx.jobs.track(
        "study",
        refs={"project_id": project_id, "server": name},
        runner=f"remote:{name}",
        cancel=cancel,
    )

    def run() -> None:
        def emit(kind: str, **data: Any) -> None:
            ctx.jobs.apply(job.id, kind, data)

        try:
            sync = ProjectSync(ctx, client, project_id)
            emit("sync", step="push")
            pushed = sync.push_study_inputs(
                body["dataset_version_id"],
                body["pipeline_id"],
                body["archspec_id"],
                workspace_id=workspace_id,
                progress=lambda step, **d: emit("sync", step=step, **d),
            )
            launch = client.json("POST", f"/projects/{project_id}/studies", json=body)
            remote["study_id"] = launch["study"]["id"]
            remote["job_id"] = launch["job"]["id"]
            job.refs.update({"study_id": remote["study_id"], "remote_job_id": remote["job_id"]})
            emit("sync", step="launched", **pushed, remote_job_id=remote["job_id"])
            if job.status == "cancelled":
                cancel()
            _follow(client, remote["job_id"], lambda kind, data: ctx.jobs.apply(job.id, kind, data))
            final = client.json("GET", f"/jobs/{remote['job_id']}")
            while final["status"] not in TERMINAL:  # WS cortado: se espera por polling
                threading.Event().wait(3)
                final = client.json("GET", f"/jobs/{remote['job_id']}")
            if final.get("worker"):
                job.worker = final["worker"]
            emit("sync", step="pull")
            sync.pull_study(remote["study_id"])
            ctx.jobs.apply(
                job.id,
                "finished",
                {
                    "status": final["status"],
                    "result": final.get("result"),
                    "error": final.get("error"),
                },
            )
        except Exception as exc:
            logger.exception("falló el estudio remoto")
            ctx.jobs.apply(
                job.id,
                "finished",
                {"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}},
            )

    threading.Thread(target=run, name=f"remote-{job.id}", daemon=True).start()
    return job
