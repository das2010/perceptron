"""Reentrenamiento automático (RF-MON-05) y su comparación champion/challenger (RF-MON-06).

Una ejecución:
1. **Datos nuevos etiquetados**: lotes de las fuentes streaming/API posteriores al último
   reentrenamiento más el feedback con etiqueta del deployment.
2. **Versión de datos nueva** = la del champion + las filas nuevas, con split predefinido: las
   filas viejas conservan su partición (el test del champion no se mueve) y de las nuevas, una
   fracción al azar (`holdout_fraction`, con semilla) va a test y el resto a train/val. Linaje:
   `parent_id` = versión del champion, `transformation = append:<n>`.
3. **Challenger**: misma arquitectura y pipeline que el champion, HPO reducido (presupuesto
   de la política), por la cola del servidor si existe (`ctx.launch_study`).
4. **Comparación** sobre el test de la versión nueva (test viejo + filas nuevas reservadas:
   ninguno de los dos modelos las vio) y **promoción** solo si mejora; con aprobación, queda
   esperando.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from perceptron.api.jobs import TERMINAL
from perceptron.core.errors import ConflictError, ValidationError
from perceptron.data.splits import SPLIT_COLUMN, SplitRequest
from perceptron.data.view import Purpose
from perceptron.domain.enums import AlertKind, RetrainStatus, Severity, SplitStrategy
from perceptron.domain.models import (
    DatasetVersion,
    Deployment,
    RetrainPolicy,
    RetrainRun,
    Run,
    Study,
    utcnow,
)
from perceptron.hpo.strategy import Budget, HPOStrategy
from perceptron.monitoring.service import Monitoring
from perceptron.services.studies import new_study

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

logger = logging.getLogger(__name__)
RETRAIN_TOPIC = "retrain.event"
SPLIT_TMP = "__particion"
MIN_NEW_ROWS = 20
INTERNAL = (
    "__received_at",
    "prediction_id",
    "ts",
    "model_version_id",
    "key",
    "prediction",
    "confidence",
    "probabilities",
)


class Retrainer:
    def __init__(self, ctx: EngineContext) -> None:
        self.ctx = ctx
        self.mon = Monitoring(ctx)
        self.policies = ctx.repo(RetrainPolicy)
        self.runs = ctx.repo(RetrainRun)

    # ------------------------------------------------------------------ lanzamiento

    def start(self, policy_id: str, trigger: dict[str, Any]) -> RetrainRun:
        """Crea la ejecución y la corre como job (202 + WS, SPEC §10)."""
        policy = self.policies.get(policy_id)
        busy = self.runs.list(
            filters={"policy_id": policy.id, "status": RetrainStatus.RUNNING.value}
        )
        if busy:
            raise ConflictError("ya hay un reentrenamiento en curso para esta política")
        run = self.runs.add(
            RetrainRun(project_id=policy.project_id, policy_id=policy.id, trigger=trigger)
        )
        self.ctx.jobs.submit(
            "retrain",
            lambda job: self.execute(run.id).model_dump(mode="json"),
            refs={"retrain_run_id": run.id, "project_id": policy.project_id},
        )
        return run

    def _log(self, run: RetrainRun, step: str, **data: Any) -> RetrainRun:
        entry = {"at": utcnow().isoformat(), "step": step, **data}
        fresh = self.runs.get(run.id)
        updated = self.runs.update(fresh.model_copy(update={"log": [*fresh.log, entry]}))
        self.ctx.events.publish(RETRAIN_TOPIC, retrain_run_id=run.id, step=step, data=data)
        return updated

    def _finish(self, run: RetrainRun, status: RetrainStatus, **changes: Any) -> RetrainRun:
        fresh = self.runs.get(run.id)
        done = self.runs.update(
            fresh.model_copy(update={"status": status, "finished_at": utcnow(), **changes})
        )
        self.ctx.events.publish(
            RETRAIN_TOPIC, retrain_run_id=run.id, step="finished", data={"status": status.value}
        )
        return done

    # ------------------------------------------------------------------ datos

    def _new_rows(
        self, policy: RetrainPolicy, target: str, columns: list[str]
    ) -> tuple[pl.DataFrame, dict[str, str]]:
        from perceptron.data.sources.stream import StreamBuffer

        frames: list[pl.DataFrame] = []
        consumed = dict(policy.consumed)
        for sid in policy.source_ids:
            buffer = StreamBuffer(
                self.ctx.settings.paths.project(policy.project_id).root / "streams" / sid
            )
            df = buffer.read(after=consumed.get(sid))
            if df.height:
                frames.append(
                    df.rename({"__received_at": "__at"}) if "__received_at" in df.columns else df
                )
                last = buffer.batches()[-1].name
                consumed[sid] = last
        if policy.deployment_id and policy.use_feedback:
            dep = self.ctx.repo(Deployment).get(policy.deployment_id)
            store = self.mon.store(dep)
            labeled = store.labeled()
            if labeled.height and policy.last_run_at is not None:
                labeled = labeled.filter(pl.col("ts") > policy.last_run_at)
            if labeled.height:
                feats = store.features(labeled).with_columns(
                    labeled["label"].alias(target), labeled["ts"].alias("__at")
                )
                frames.append(feats)
        if not frames:
            return pl.DataFrame(), consumed
        df = pl.concat(frames, how="diagonal_relaxed")
        keep = [c for c in columns if c in df.columns]
        if target not in df.columns:
            return pl.DataFrame(), consumed
        extra = ["__at"] if "__at" in df.columns else []
        df = df.select([*keep, *extra]).filter(pl.col(target).is_not_null())
        df = df.filter(pl.col(target).cast(pl.Utf8) != "")
        return df.unique(subset=keep, keep="last", maintain_order=True), consumed

    def _combined(
        self, old: pl.DataFrame, new: pl.DataFrame, holdout_fraction: float
    ) -> pl.DataFrame:
        base = old.rename({SPLIT_COLUMN: SPLIT_TMP})
        n = new.height
        # Holdout al azar entre las filas nuevas (con semilla): así cubre también la parte con
        # drift, que el challenger necesita ver en train para poder mejorar.
        rng = np.random.default_rng(0)
        order = rng.permutation(n)
        n_hold = max(1, round(n * holdout_fraction))
        labels = np.full(n, "train", dtype=object)
        labels[order[:n_hold]] = "test"
        rest = order[n_hold:]
        if len(rest) > 6:
            val = rng.choice(rest, size=max(1, round(len(rest) * 0.15)), replace=False)
            labels[val] = "val"
        added = new.drop("__at", strict=False).with_columns(pl.Series(SPLIT_TMP, labels.tolist()))
        # Tipos de la versión vieja: las fuentes pueden traer números como texto.
        casted = added.with_columns(
            [
                pl.col(c).cast(base.schema[c], strict=False)
                for c in added.columns
                if c in base.columns and c != SPLIT_TMP
            ]
        )
        return pl.concat([base, casted.select(base.columns)], how="vertical_relaxed")

    # ------------------------------------------------------------------ ejecución

    def execute(self, retrain_run_id: str) -> RetrainRun:
        run = self.runs.get(retrain_run_id)
        policy = self.policies.get(run.policy_id)
        try:
            return self._execute(run, policy)
        except Exception as exc:
            logger.exception("falló el reentrenamiento", extra={"retrain_run_id": run.id})
            self.mon.alerts.raise_alert(
                project_id=policy.project_id,
                kind=AlertKind.RETRAIN,
                severity=Severity.HIGH,
                title="Falló el reentrenamiento automático",
                message=str(exc)[:500],
                details={"retrain_run_id": run.id},
            )
            return self._finish(run, RetrainStatus.FAILED, error=str(exc)[:2000])

    def _execute(self, run: RetrainRun, policy: RetrainPolicy) -> RetrainRun:
        champion = self.mon.champion(policy.project_id)
        if champion is None:
            raise ValidationError("no hay champion en producción para reentrenar")
        champ_run = self.ctx.repo(Run).get(champion.run_id)
        dv0 = self.ctx.repo(DatasetVersion).get(champ_run.dataset_version_id)
        if not dv0.target:
            raise ValidationError("la versión de datos del champion no tiene columna objetivo")
        old = self.mon.wf.view(dv0).read(purpose=Purpose.VERSIONING)
        columns = [c for c in old.columns if c != SPLIT_COLUMN]
        new, consumed = self._new_rows(policy, dv0.target, columns)
        run = self.runs.update(
            self.runs.get(run.id).model_copy(
                update={"champion_id": champion.id, "new_rows": new.height}
            )
        )
        self._log(run, "datos", nuevas=new.height, champion=champion.id)
        if new.height < MIN_NEW_ROWS:
            return self._finish(
                run,
                RetrainStatus.SKIPPED,
                error=f"hay {new.height} filas nuevas etiquetadas (mínimo {MIN_NEW_ROWS})",
            )

        combined = self._combined(old, new, policy.holdout_fraction)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "datos.parquet"
            combined.write_parquet(path)
            dv1 = self.mon.wf.ingest(
                policy.project_id,
                path,
                target=dv0.target,
                split=SplitRequest(strategy=SplitStrategy.PREDEFINED, split_column=SPLIT_TMP),
            )
        repo = self.ctx.repo(DatasetVersion)
        dv1 = repo.update(
            repo.get(dv1.id).model_copy(
                update={"parent_id": dv0.id, "transformation": f"append:{new.height}"}
            )
        )
        run = self.runs.update(
            self.runs.get(run.id).model_copy(update={"dataset_version_id": dv1.id})
        )
        self._log(run, "version", dataset_version_id=dv1.id, filas=dv1.num_samples)

        base = (
            HPOStrategy.model_validate(self.ctx.repo(Study).get(champ_run.study_id).strategy)
            if champ_run.study_id
            else None
        )
        if base is None:
            base = self.mon.wf.hpo_strategy(champ_run.archspec_id, Budget())
        budget = base.budget.model_copy(update={"max_trials": 3, **policy.budget})
        strategy = base.model_copy(update={"budget": budget})
        study = new_study(
            self.ctx,
            policy.project_id,
            dv1.id,
            champ_run.pipeline_id,
            champ_run.archspec_id,
            strategy,
        )
        job = self.ctx.launch_study(study)
        run = self.runs.update(self.runs.get(run.id).model_copy(update={"study_id": study.id}))
        self._log(run, "entrenamiento", study_id=study.id, job_id=job.id)
        while (current := self.ctx.jobs.get(job.id)) is not None and current.status not in TERMINAL:
            time.sleep(1.0)
        if current is None or current.status != "succeeded" or not current.result:
            state = current.status if current else "?"
            raise ValidationError(f"el entrenamiento del challenger no terminó bien ({state})")
        best = (current.result.get("best_trial") or {}).get("run_id")
        if not best:
            errors = [t.get("error") for t in current.result.get("trials") or [] if t.get("error")]
            detail = "; ".join(str(e)[:300] for e in errors[:3]) or "sin trials completos"
            raise ValidationError(f"el estudio no produjo un modelo: {detail}")
        from perceptron.export.formats import ExportRequest

        self.mon.wf.evaluate(best)
        self.mon.wf.export(best, ExportRequest())
        challenger = self.mon.wf.register(best)
        run = self.runs.update(
            self.runs.get(run.id).model_copy(
                update={"run_id": best, "model_version_id": challenger.id}
            )
        )
        self._log(run, "challenger", model_version_id=challenger.id)

        result = self.mon.challenge(
            challenger.id,
            holdout=dv1.id,
            min_improvement=policy.min_improvement,
            promote=not policy.require_approval,
        )
        improved = result.champion_value is None or result.improvement > policy.min_improvement
        status = (
            RetrainStatus.PROMOTED
            if result.promoted
            else RetrainStatus.AWAITING_APPROVAL
            if improved and policy.require_approval
            else RetrainStatus.NOT_IMPROVED
        )
        self._log(run, "comparacion", **result.model_dump(mode="json"))
        fresh_policy = self.policies.get(policy.id)
        self.policies.update(
            fresh_policy.model_copy(update={"consumed": consumed, "last_run_at": utcnow()})
        )
        text = {
            RetrainStatus.PROMOTED: "El challenger mejoró y es el nuevo champion.",
            RetrainStatus.AWAITING_APPROVAL: "El challenger mejoró: falta aprobar la promoción.",
            RetrainStatus.NOT_IMPROVED: "El challenger no mejoró: sigue el champion.",
        }[status]
        self.mon.alerts.raise_alert(
            project_id=policy.project_id,
            kind=AlertKind.RETRAIN,
            severity=Severity.LOW
            if status is not RetrainStatus.AWAITING_APPROVAL
            else Severity.MEDIUM,
            title=f"Reentrenamiento: {text}",
            message=(
                f"{result.metric}: challenger {result.challenger_value:.4f}, "
                f"champion {result.champion_value}"
            ),
            details={"retrain_run_id": run.id, "status": status.value},
        )
        return self._finish(run, status, challenge=result.model_dump(mode="json"))

    # ------------------------------------------------------------------ aprobación

    def approve(self, retrain_run_id: str) -> RetrainRun:
        run = self.runs.get(retrain_run_id)
        if run.status is not RetrainStatus.AWAITING_APPROVAL or not run.model_version_id:
            raise ConflictError("la ejecución no espera aprobación")
        self.mon.promote(run.model_version_id, reason="reentrenamiento aprobado")
        return self._finish(run, RetrainStatus.PROMOTED)

    def reject(self, retrain_run_id: str) -> RetrainRun:
        run = self.runs.get(retrain_run_id)
        if run.status is not RetrainStatus.AWAITING_APPROVAL:
            raise ConflictError("la ejecución no espera aprobación")
        return self._finish(run, RetrainStatus.REJECTED)
