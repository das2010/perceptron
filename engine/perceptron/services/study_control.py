"""Estado y control de estudios: listar, detener, reanudar e interrumpidos (RF-HPO, RF-SRV-04).

El estado de un estudio se deriva de su último job y de sus runs:
- en curso o en cola: hay un job activo;
- interrumpido: el worker se perdió (o el proceso se reinició) con trials a medio entrenar;
- detenido: se pausó o canceló; si quedan trials, se puede reanudar;
- terminado / con error: el job terminó.

Reanudar relanza el mismo estudio: Optuna conserva los trials terminados (`load_if_exists`) y
sigue con los que faltan. Antes se limpia la cancelación persistida (si no, el worker lo
descartaría) y se cierran los runs que quedaron «en curso» de la vez anterior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from perceptron.api.jobs import TERMINAL, Job
from perceptron.domain.enums import RunStatus
from perceptron.domain.models import Run, Study, utcnow

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

WORKER_LOST = "WorkerLost"
StudyStatus = Literal["queued", "running", "stopped", "interrupted", "finished", "failed"]
_DONE = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
_OPEN = {RunStatus.RUNNING, RunStatus.QUEUED, RunStatus.PAUSED}


class StudyView(BaseModel):
    study: Study
    status: StudyStatus
    job_id: str | None = None
    trials_done: int = 0
    trials_total: int | None = None
    best_run_id: str | None = None
    reason: str | None = Field(default=None, description="Por qué se interrumpió o falló")
    resumable: bool = False


def _latest_job(ctx: EngineContext, study_id: str) -> Job | None:
    jobs = [j for j in ctx.jobs.list() if j.refs.get("study_id") == study_id]
    return max(jobs, key=lambda j: j.created_at) if jobs else None


def study_view(ctx: EngineContext, study: Study, runs: list[Run]) -> StudyView:
    job = _latest_job(ctx, study.id)
    total = study.budget.get("max_trials")
    done = sum(1 for r in runs if r.status in _DONE)
    succeeded = [r for r in runs if r.status is RunStatus.SUCCEEDED and "val_loss" in r.metrics]
    best = min(succeeded, key=lambda r: r.metrics["val_loss"]).id if succeeded else None
    status: StudyStatus
    reason: str | None = None
    if job is not None and job.status not in TERMINAL:
        status = "running" if job.status == "running" else "queued"
    elif job is not None and job.error and job.error.get("type") == WORKER_LOST:
        status, reason = "interrupted", str(job.error.get("message", ""))
    elif job is not None and job.status == "cancelled":
        status = "stopped"
    elif job is not None and job.status == "succeeded":
        status = "finished"
    elif job is not None:
        status, reason = "failed", str((job.error or {}).get("message", "")) or None
    elif any(r.status in _OPEN for r in runs):
        # Sin job vivo pero con trials «en curso»: el proceso se reinició a mitad (desktop).
        status = "interrupted"
        reason = "El proceso se reinició durante el entrenamiento."
    elif total is not None and done >= int(total):
        status = "finished"
    else:
        status = "stopped"
    pending = total is None or done < int(total)
    return StudyView(
        study=study,
        status=status,
        job_id=job.id if job is not None and job.status not in TERMINAL else None,
        trials_done=done,
        trials_total=int(total) if total is not None else None,
        best_run_id=best,
        reason=reason,
        resumable=status in ("stopped", "interrupted", "failed") and pending,
    )


def list_study_views(ctx: EngineContext, project_id: str) -> list[StudyView]:
    studies = list(ctx.repo(Study).list(filters={"project_id": project_id}, limit=200))
    runs = list(ctx.repo(Run).list(filters={"project_id": project_id}, limit=5000))
    by_study: dict[str, list[Run]] = {}
    for r in runs:
        if r.study_id:
            by_study.setdefault(r.study_id, []).append(r)
    return [study_view(ctx, s, by_study.get(s.id, [])) for s in studies]


def close_open_runs(ctx: EngineContext, study_id: str) -> int:
    """Runs que quedaron «en curso» sin proceso que los entrene → con error. Devuelve cuántos."""
    repo = ctx.repo(Run)
    closed = 0
    for run in repo.list(filters={"study_id": study_id}, limit=5000):
        if run.status in _OPEN:
            repo.update(
                run.model_copy(update={"status": RunStatus.FAILED, "finished_at": utcnow()})
            )
            closed += 1
    return closed


def mark_interrupted(ctx: EngineContext, study_id: str) -> None:
    """Tras perder el worker: cierra sus runs y persiste la cancelación, así el mensaje que
    quedó sin confirmar en la cola (vuelve tras el visibility timeout) no lo entrena de nuevo."""
    close_open_runs(ctx, study_id)
    repo = ctx.repo(Study)
    study = repo.find(study_id)
    if study is not None and study.cancel_requested_at is None:
        repo.update(study.model_copy(update={"cancel_requested_at": utcnow()}))


def prepare_resume(ctx: EngineContext, study: Study) -> Study:
    """Antes de reanudar: sin cancelación persistida y sin runs colgados."""
    close_open_runs(ctx, study.id)
    if study.cancel_requested_at is None:
        return study
    return ctx.repo(Study).update(study.model_copy(update={"cancel_requested_at": None}))
