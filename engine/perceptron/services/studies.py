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
from perceptron.domain.enums import Device
from perceptron.domain.models import Study
from perceptron.hpo.strategy import HPOStrategy
from perceptron.hpo.study import StudyControl, StudyResult

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron.api.jobs import Job, JobContext

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
) -> Study:
    """Crea el estudio con todo lo necesario para ejecutarse en cualquier lado."""
    from perceptron.domain.models import ArchSpecRecord

    record = ctx.repo(ArchSpecRecord).get(archspec_id)
    request = {
        "dataset_version_id": dataset_version_id,
        "pipeline_id": pipeline_id,
        "archspec_id": archspec_id,
        "strategy": strategy.model_dump(mode="json"),
        "budget": strategy.budget.model_dump(mode="json"),
        "device": device.value if device else None,
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
