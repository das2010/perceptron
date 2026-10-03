"""Jobs que esperan a otros jobs no bloquean el pool (reentrenamiento → estudio)."""

from __future__ import annotations

import time

from perceptron.api.jobs import TERMINAL, JobContext, JobManager
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
