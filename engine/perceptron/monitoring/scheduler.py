"""Disparadores de reentrenamiento (RF-MON-05) y sondeo de fuentes streaming (RF-ING-05).

- **drift** `{min_severity}` y **degradation** `{max_drop}`: al llegar un `DriftReport` del
  deployment vigilado (evento del bus).
- **cron** `{expr}`: expresión de 5 campos en UTC.
- **volume** `{min_rows}`: filas nuevas etiquetadas desde el último reentrenamiento.

Además sondea las fuentes con `poll_interval_s`. Corre en un hilo del Engine (desktop y
servidor); `tick()` se puede llamar directo (tests, CLI).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from perceptron.core.errors import ConflictError
from perceptron.core.events import Event
from perceptron.domain.enums import DataSourceType, Severity
from perceptron.domain.models import DataSource, DriftReport, RetrainPolicy, utcnow
from perceptron.monitoring.cron import Cron
from perceptron.monitoring.service import DRIFT_TOPIC

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron.domain.models import RetrainRun

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, ctx: EngineContext, interval_s: float = 30.0) -> None:
        self.ctx = ctx
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unsubscribe = ctx.events.subscribe(DRIFT_TOPIC, self.on_drift)
        self._last_flush: datetime | None = None

    # ------------------------------------------------------------------ ciclo

    def start(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                try:
                    self.tick()
                except Exception:
                    logger.exception("falló el ciclo del scheduler de monitoreo")

        self._thread = threading.Thread(target=loop, name="perceptron-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._unsubscribe()

    def _policies(self) -> list[RetrainPolicy]:
        return [p for p in self.ctx.repo(RetrainPolicy).list(limit=1000) if p.enabled]

    def fire(
        self, policy: RetrainPolicy, trigger: dict[str, Any], now: datetime | None = None
    ) -> RetrainRun | None:
        from perceptron.monitoring.retrain import Retrainer

        now = now or utcnow()
        if policy.last_run_at and now - policy.last_run_at < timedelta(seconds=policy.cooldown_s):
            return None
        try:
            return Retrainer(self.ctx).start(policy.id, trigger)
        except ConflictError:
            return None  # ya hay uno en curso

    def tick(self, now: datetime | None = None) -> list[RetrainRun]:
        from perceptron.services.streams import pull_source

        now = now or utcnow()
        interval = timedelta(seconds=self.ctx.settings.telemetry.interval_s)
        if self._last_flush is None or now - self._last_flush >= interval:
            self._last_flush = now
            self.ctx.telemetry.flush()
        for src in self.ctx.repo(DataSource).list(limit=1000):
            interval = src.config.get("poll_interval_s")
            if src.type in (DataSourceType.STREAM, DataSourceType.API) and interval:
                last = src.config.get("last_poll")
                if last is None or (
                    now - datetime.fromisoformat(str(last))
                ).total_seconds() >= float(interval):
                    try:
                        pull_source(self.ctx, src.id)
                    except Exception:
                        logger.exception("no se pudo leer la fuente", extra={"source_id": src.id})
        fired: list[RetrainRun] = []
        for policy in self._policies():
            for trigger in policy.triggers:
                kind = trigger.get("type")
                if kind == "cron":
                    cron = Cron(str(trigger.get("expr", "")))
                    if cron.due(policy.last_cron_at, now):
                        repo = self.ctx.repo(RetrainPolicy)
                        stamped = repo.update(
                            repo.get(policy.id).model_copy(update={"last_cron_at": now})
                        )
                        run = self.fire(stamped, {"type": "cron", "expr": trigger.get("expr")}, now)
                        if run:
                            fired.append(run)
                elif kind == "volume":
                    pending = self.pending_rows(policy)
                    if pending >= int(trigger.get("min_rows", 500)):
                        run = self.fire(policy, {"type": "volume", "rows": pending}, now)
                        if run:
                            fired.append(run)
        return fired

    def pending_rows(self, policy: RetrainPolicy) -> int:
        from perceptron.data.sources.stream import StreamBuffer
        from perceptron.domain.models import Deployment
        from perceptron.monitoring.service import Monitoring

        total = 0
        root = self.ctx.settings.paths.project(policy.project_id).root / "streams"
        for sid in policy.source_ids:
            df = StreamBuffer(root / sid).read(after=policy.consumed.get(sid))
            total += df.height
        if policy.deployment_id and policy.use_feedback:
            dep = self.ctx.repo(Deployment).find(policy.deployment_id)
            if dep is not None:
                labeled = Monitoring(self.ctx).store(dep).labeled()
                if labeled.height and policy.last_run_at is not None:
                    import polars as pl

                    labeled = labeled.filter(pl.col("ts") > policy.last_run_at)
                total += labeled.height
        return total

    # ------------------------------------------------------------------ eventos

    def on_drift(self, event: Event) -> None:
        report_id = event.payload.get("report_id")
        if not report_id:
            return
        report = self.ctx.repo(DriftReport).find(str(report_id))
        if report is None:
            return
        for policy in self._policies():
            if policy.deployment_id != report.deployment_id:
                continue
            for trigger in policy.triggers:
                kind = trigger.get("type")
                if kind == "drift":
                    need = Severity(trigger.get("min_severity", Severity.MEDIUM.value))
                    if report.severity.rank >= need.rank:
                        self.fire(
                            policy,
                            {
                                "type": "drift",
                                "report_id": report.id,
                                "severity": report.severity.value,
                            },
                        )
                        return
                elif kind == "degradation":
                    perf = report.metrics.get("performance") or {}
                    drop = perf.get("relative_drop")
                    if drop is not None and float(drop) >= float(trigger.get("max_drop", 0.05)):
                        self.fire(
                            policy, {"type": "degradation", "report_id": report.id, "drop": drop}
                        )
                        return
