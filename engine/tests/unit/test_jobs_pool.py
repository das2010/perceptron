"""Jobs que esperan a otros jobs no bloquean el pool (reentrenamiento → estudio)."""

from __future__ import annotations

import time

from perceptron.api.jobs import TERMINAL, Job, JobContext, JobManager
from perceptron.core.events import EventBus


def test_waiting_jobs_do_not_starve_the_pool() -> None:
    """Con 2 hilos en el pool, 3 «reentrenamientos» que esperan a su estudio terminan todos
    (antes, dos esperando ocupaban el pool y sus estudios nunca arrancaban)."""
    jobs = JobManager(EventBus(), max_workers=2)

    def retrain(_: JobContext) -> str:
        study = jobs.submit("study", lambda _: "modelo")
        while (current := jobs.get(study.id)) is not None and current.status not in TERMINAL:
            time.sleep(0.01)
        return str(current.status if current else "?")

    parents = [jobs.submit("retrain", retrain, dedicated=True) for _ in range(3)]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(
        (j := jobs.get(p.id)) is not None and j.status not in TERMINAL for p in parents
    ):
        time.sleep(0.01)
    try:
        assert [jobs.get(p.id).status for p in parents] == ["succeeded"] * 3  # type: ignore[union-attr]
        assert [jobs.get(p.id).result for p in parents] == ["succeeded"] * 3  # type: ignore[union-attr]
    finally:
        jobs.shutdown()


def test_remote_jobs_are_persisted_and_restored() -> None:
    """Team Server: los jobs remotos se guardan en cada cambio y se retoman tras reiniciar."""
    saved: list[tuple[str, str]] = []

    class Store:
        def save(self, job: Job) -> None:
            saved.append((job.id, job.status))

    jobs = JobManager(EventBus())
    jobs.store = Store()
    try:
        local = jobs.submit("study", lambda _: "ok")
        remote = jobs.track("study", runner="queue:cpu", refs={"study_id": "std_1"})
        jobs.apply(remote.id, "started", {"worker": "w1"})
        jobs.apply(remote.id, "epoch", {"epoch": 1})  # el progreso no se guarda
        jobs.apply(remote.id, "finished", {"status": "succeeded", "result": {"a": 1}})
        assert saved == [(remote.id, "queued"), (remote.id, "running"), (remote.id, "succeeded")]
        assert all(job_id != local.id for job_id, _ in saved)  # los locales mueren con el proceso

        cancelled: list[str] = []
        fresh = JobManager(EventBus())
        fresh.restore(
            remote.model_copy(update={"status": "running"}), cancel=lambda: cancelled.append("x")
        )
        assert fresh.get(remote.id) is not None and fresh.list()[0].refs == {"study_id": "std_1"}
        assert fresh.cancel(remote.id) is not None and cancelled == ["x"]
        fresh.shutdown()
    finally:
        jobs.shutdown()
