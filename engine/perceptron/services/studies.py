"""Ejecución de estudios de HPO, separada de dónde corren (RF-SRV-04).

Un estudio queda completamente descripto por su entidad `Study` (estrategia + el pedido de
lanzamiento en `budget.request`). Por eso puede ejecutarse en un hilo del Engine (desktop,
`launch_local`) o en un worker del Team Server que solo recibe el `study_id` (Capa 5b): el
servidor instala su propio `StudyLauncher` en el `EngineContext`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from perceptron.core.errors import ValidationError
from perceptron.domain.enums import Device, Modality
from perceptron.domain.models import Study
from perceptron.hpo.strategy import HPOStrategy
from perceptron.hpo.study import StudyControl, StudyResult

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron.api.jobs import Job, JobContext
    from perceptron.services.workflow import Workflow

Emit = Callable[..., None]
StudyLauncher = Callable[["EngineContext", Study], "Job"]
GPU_DEVICES = frozenset({Device.CUDA, Device.ROCM, Device.XPU})


class StudyRequest(BaseModel):
    """Lo que se pidió al lanzar (guardado en `Study.budget["request"]`)."""

    model_config = ConfigDict(extra="ignore")

    dataset_version_id: str
    pipeline_id: str
    archspec_id: str
    device: Device | None = None
    limit_train_batches: float | None = None  # mini-torneo: fracción de train por época


def study_request(study: Study) -> StudyRequest:
    raw = study.budget.get("request")
    if not isinstance(raw, dict):
        raise ValidationError(f"el estudio {study.id} no tiene pedido de lanzamiento")
    return StudyRequest.model_validate(raw)


def needs_gpu(study: Study) -> bool:
    return study_request(study).device in GPU_DEVICES


def execute_study(
    ctx: EngineContext, study: Study, *, control: StudyControl, emit: Emit
) -> StudyResult:
    """Corre el estudio en este proceso; `emit(kind, **data)` informa el progreso."""
    from perceptron.services.workflow import Workflow

    req = study_request(study)

    def on_event(ev: Any) -> None:
        if ev.event == "epoch":
            emit("epoch", run_id=ev.run_id, epoch=ev.epoch, metrics=ev.metrics)

    _, result = Workflow(ctx).run_study(
        study.project_id,
        req.dataset_version_id,
        req.pipeline_id,
        req.archspec_id,
        HPOStrategy.model_validate(study.strategy),
        device=req.device,
        control=control,
        on_event=on_event,
        study=study,
        limit_train_batches=req.limit_train_batches,
    )
    return result


def new_study(
    ctx: EngineContext,
    project_id: str,
    dataset_version_id: str,
    pipeline_id: str,
    archspec_id: str,
    strategy: HPOStrategy,
    *,
    device: Device | None = None,
    limit_train_batches: float | None = None,
) -> Study:
    """Crea el estudio con todo lo necesario para ejecutarse en cualquier lado."""
    from perceptron.domain.models import ArchSpecRecord, DatasetVersion
    from perceptron.services.workflow import Workflow

    record = ctx.repo(ArchSpecRecord).get(archspec_id)
    dv = ctx.repo(DatasetVersion).find(dataset_version_id)
    if dv is not None and dv.modality is Modality.TABULAR:
        # Antes de encolar: un pipeline sin columnas de entrada no puede aprender nada (422).
        Workflow(ctx).fitted_pipeline(pipeline_id, dataset_version_id)
    request = {
        "dataset_version_id": dataset_version_id,
        "pipeline_id": pipeline_id,
        "archspec_id": archspec_id,
        "strategy": strategy.model_dump(mode="json"),
        "budget": strategy.budget.model_dump(mode="json"),
        "device": device.value if device else None,
        "limit_train_batches": limit_train_batches,
    }
    return ctx.repo(Study).add(
        Study(
            project_id=project_id,
            name=f"hpo-{record.name}",
            strategy=strategy.model_dump(mode="json"),
            budget={**strategy.budget.model_dump(mode="json"), "request": request},
            objectives=[o.metric for o in strategy.objectives],
            origin=strategy.origin,
        )
    )


def launch_local(ctx: EngineContext, study: Study) -> Job:
    """Desktop: el estudio corre en un hilo del Engine (job 202 + WS, SPEC §10)."""
    control = StudyControl()

    def work(job: JobContext) -> StudyResult:
        job.cancel_callback = control.cancel
        return execute_study(ctx, study, control=control, emit=job.emit)

    return ctx.jobs.submit(
        "study", work, refs={"study_id": study.id, "project_id": study.project_id}
    )


STUDY_WAIT_S = 24 * 3600  # plazo para un estudio lanzado desde el agente o un mini-torneo


def run_study_managed(
    wf: Workflow,
    project_id: str,
    dataset_version_id: str,
    pipeline_id: str,
    archspec_id: str,
    strategy: HPOStrategy,
    *,
    device: Device | None = None,
    control: StudyControl | None = None,
    limit_train_batches: float | None = None,
    timeout_s: float = STUDY_WAIT_S,
) -> tuple[Study, StudyResult]:
    """Estudio del agente o de un mini-torneo por el mismo camino que el lanzamiento normal.

    - Desktop (sin `study_launcher`): corre en este hilo con el `Workflow` de quien llama
      (su tracker y su configuración), como siempre.
    - Team Server: pasa por `ctx.launch_study` (cuotas, cola y workers) y se espera el
      resultado; cancelar `control` cancela el job.
    """
    import time

    from perceptron.api.jobs import TERMINAL

    ctx = wf.ctx
    study = new_study(
        ctx,
        project_id,
        dataset_version_id,
        pipeline_id,
        archspec_id,
        strategy,
        device=device,
        limit_train_batches=limit_train_batches,
    )
    control = control or StudyControl()
    if ctx.study_launcher is None:
        return wf.run_study(
            project_id,
            dataset_version_id,
            pipeline_id,
            archspec_id,
            strategy,
            device=device,
            control=control,
            study=study,
            limit_train_batches=limit_train_batches,
        )
    job = ctx.launch_study(study)  # las cuotas del servidor rechazan acá (429)
    deadline = time.monotonic() + timeout_s
    while (current := ctx.jobs.get(job.id)) is not None and current.status not in TERMINAL:
        if control.cancelled or time.monotonic() > deadline:
            ctx.jobs.cancel(job.id)
            if not control.cancelled:
                raise ValidationError(f"el estudio {study.id} no terminó en {timeout_s:.0f} s")
        time.sleep(1.0)
    if control.cancelled:  # como en el desktop: un estudio detenido, no un error
        return study, StudyResult(study_name=study.name, strategy=strategy, stop_reason="cancelled")
    if current is None or current.status != "succeeded" or not current.result:
        state = current.status if current else "desconocido"
        detail = (current.error or {}).get("message", "") if current else ""
        raise ValidationError(f"el estudio {study.id} terminó {state} {detail}".strip())
    return study, StudyResult.model_validate(current.result)
