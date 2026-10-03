"""Agente y mini-torneos lanzan sus estudios por `ctx.launch_study` (cola y cuotas)."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from perceptron.api.context import EngineContext
from perceptron.api.jobs import Job
from perceptron.core.errors import RateLimitedError
from perceptron.domain.models import ArchSpecRecord, Study
from perceptron.hpo.strategy import Budget, HPOStrategy
from perceptron.hpo.study import StudyControl, StudyResult
from perceptron.services.studies import run_study_managed, study_request
from perceptron.services.workflow import Workflow

STRATEGY = HPOStrategy(strategy="single", pruner="none", budget=Budget(max_trials=1))


def _arch(ctx: EngineContext) -> str:
    return (
        ctx.repo(ArchSpecRecord)
        .add(ArchSpecRecord(project_id="prj_01X", name="mlp", spec={}, content_hash="0" * 64))
        .id
    )


def test_managed_study_goes_through_the_launcher(ctx: EngineContext) -> None:
    launched: list[Study] = []

    def launcher(c: EngineContext, study: Study) -> Job:
        launched.append(study)
        job = c.jobs.track("study", refs={"study_id": study.id})
        result = StudyResult(study_name=study.name, strategy=STRATEGY, stop_reason="max_trials")
        data: dict[str, Any] = {"status": "succeeded", "result": result.model_dump(mode="json")}
        threading.Timer(0.2, c.jobs.apply, args=(job.id, "finished", data)).start()
        return job

    ctx.study_launcher = launcher
    study, result = run_study_managed(
        Workflow(ctx),
        "prj_01X",
        "dsv_01X",
        "pip_01X",
        _arch(ctx),
        STRATEGY,
        limit_train_batches=0.25,
    )
    assert [s.id for s in launched] == [study.id]
    assert study_request(launched[0]).limit_train_batches == 0.25  # viaja al worker
    assert result.stop_reason == "max_trials"


def test_server_quotas_apply_to_agent_and_tournament(ctx: EngineContext) -> None:
    def full(_c: EngineContext, _s: Study) -> Job:
        raise RateLimitedError("cuota llena")

    ctx.study_launcher = full
    with pytest.raises(RateLimitedError):
        run_study_managed(Workflow(ctx), "prj_01X", "dsv_01X", "pip_01X", _arch(ctx), STRATEGY)


def test_stopping_the_agent_cancels_the_queued_job(ctx: EngineContext) -> None:
    jobs: list[str] = []

    def launcher(c: EngineContext, study: Study) -> Job:
        job = c.jobs.track("study", refs={"study_id": study.id})
        jobs.append(job.id)
        return job  # nunca termina solo

    ctx.study_launcher = launcher
    control = StudyControl()
    threading.Timer(0.3, control.cancel).start()
    _, result = run_study_managed(
        Workflow(ctx), "prj_01X", "dsv_01X", "pip_01X", _arch(ctx), STRATEGY, control=control
    )
    assert result.stop_reason == "cancelled"
    job = ctx.jobs.get(jobs[0])
    assert job is not None and job.status == "cancelled"
