"""Jobs de la cola persistidos (RF-SRV-04): sobreviven a un reinicio del servidor.

Los estudios encolados siguen en los workers aunque el servidor se reinicie. Sin persistencia
el servidor los olvidaba: no se veían, no se podían cancelar y las cuotas de estudios en curso
quedaban libres. Ahora el servidor guarda cada job remoto en la base, el worker anota inicio
y fin (por si el servidor no estaba para recibir el evento) y al arrancar se retoman.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from perceptron.api.jobs import TERMINAL, Job
from perceptron.core.errors import ConflictError
from perceptron.domain.models import Entity, JsonDict, utcnow
from perceptron.storage.db import Database
from perceptron.storage.repositories import SqlRepository

logger = logging.getLogger(__name__)
RECENT = timedelta(hours=24)  # los terminados recientes también se retoman (GET /jobs/{id})
_RETRIES = 5


class TrackedJob(Entity):
    """Un job de la cola del Team Server tal como lo ve la API (`Job` serializado)."""

    id: str
    status: str
    job: JsonDict


class DbJobStore:
    def __init__(self, db: Database) -> None:
        self.repo: SqlRepository[TrackedJob] = SqlRepository(db, TrackedJob)

    def save(self, job: Job) -> None:
        self._write(job.id, lambda _current: job.model_dump(mode="json"))

    def mark(self, job_id: str, **changes: Any) -> None:
        """Lo anota el worker: `status`, `worker`, `result`, `error`…"""

        def merge(current: dict[str, Any] | None) -> dict[str, Any] | None:
            if current is None:
                return None  # job de otro servidor o ya purgado: nada que anotar
            stamp = "finished_at" if changes.get("status") in TERMINAL else "started_at"
            return {**current, **changes, stamp: utcnow().isoformat()}

        self._write(job_id, merge)

    def _write(
        self, job_id: str, build: Callable[[dict[str, Any] | None], dict[str, Any] | None]
    ) -> None:
        for _ in range(_RETRIES):
            current = self.repo.find(job_id)
            data = build(current.job if current else None)
            if data is None:
                return
            # Un job terminado no vuelve atrás (p. ej. un "started" que llega tarde).
            if current is not None and current.status in TERMINAL:
                if data.get("status") not in TERMINAL:
                    return
                if current.status == "cancelled":
                    data = {**data, "status": "cancelled"}
            try:
                if current is None:
                    self.repo.add(TrackedJob(id=job_id, status=data["status"], job=data))
                else:
                    self.repo.update(
                        current.model_copy(update={"status": data["status"], "job": data})
                    )
                return
            except ConflictError:
                continue  # el servidor y el worker escribieron a la vez: se reintenta
        logger.warning("no se pudo guardar el job tras reintentos", extra={"job_id": job_id})

    def load(self) -> list[Job]:
        """Jobs a retomar: los que siguen en curso y los terminados recientes.

        Los terminados viejos se borran: el resultado del estudio ya está en sus runs.
        """
        since = utcnow() - RECENT
        jobs: list[Job] = []
        old: list[str] = []
        offset = 0
        while True:
            page = list(self.repo.list(limit=500, offset=offset))
            for row in page:
                job = Job.model_validate(row.job)
                if job.status not in TERMINAL or row.updated_at >= since:
                    jobs.append(job)
                else:
                    old.append(row.id)
            if len(page) < 500:
                break
            offset += 500
        for job_id in old:
            self.repo.delete(job_id)
        return jobs
