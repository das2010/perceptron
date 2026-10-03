"""Mini-torneo de arquitecturas (RF-ARC-03).

Cada propuesta se entrena una vez (estrategia `single`, defaults de la plantilla) con un
presupuesto corto: una fracción de las épocas y de los batches de train por época. La de
mejor métrica de validación avanza al HPO completo. El test sellado no se toca.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from perceptron.archspec.schema import HP, ArchSpec
from perceptron.core.ids import new_id
from perceptron.domain.enums import Device, Origin
from perceptron.domain.models import ArchSpecRecord, Run
from perceptron.hpo.strategy import Budget, HPOStrategy, Objective
from perceptron.services.studies import run_study_managed
from perceptron.training.module import monitor_mode

if TYPE_CHECKING:
    from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

DEFAULT_FRACTION = 0.1
DEFAULT_SUBSET = 0.3


def spec_epochs(spec: ArchSpec) -> int:
    epochs = spec.training.epochs
    value = epochs.default if isinstance(epochs, HP) else epochs
    return int(value) if isinstance(value, int | float) else 30


@dataclass
class TournamentEntry:
    archspec_id: str
    name: str
    epochs: int
    run_id: str | None = None
    metric: float | None = None
    status: str = "pending"


@dataclass
class TournamentResult:
    id: str
    metric: str
    direction: Literal["minimize", "maximize"]
    entries: list[TournamentEntry] = field(default_factory=list)
    winner: str | None = None

    @property
    def winner_entry(self) -> TournamentEntry | None:
        return next((e for e in self.entries if e.archspec_id == self.winner), None)


def mini_tournament(
    wf: Workflow,
    project_id: str,
    dataset_version_id: str,
    pipeline_id: str,
    archspec_ids: list[str],
    *,
    fraction: float = DEFAULT_FRACTION,
    subset: float = DEFAULT_SUBSET,
    metric: str = "val_loss",
    device: Device | None = None,
) -> TournamentResult:
    direction: Literal["minimize", "maximize"] = (
        "minimize" if monitor_mode(metric) == "min" else "maximize"
    )
    result = TournamentResult(id=new_id_tournament(), metric=metric, direction=direction)
    for archspec_id in archspec_ids:
        record = wf.ctx.repo(ArchSpecRecord).get(archspec_id)
        spec = ArchSpec.model_validate(record.spec)
        epochs = max(1, round(spec_epochs(spec) * fraction))
        entry = TournamentEntry(archspec_id=archspec_id, name=spec.name, epochs=epochs)
        result.entries.append(entry)
        strategy = HPOStrategy(
            strategy="single",
            pruner="none",
            objectives=[Objective(metric=metric, direction=direction)],
            budget=Budget(max_trials=1, max_epochs_per_trial=epochs),
            origin=Origin.AGENT if record.origin is Origin.AGENT else Origin.RULES,
            rationale=f"Mini-torneo {result.id}: {epochs} épocas, {subset:.0%} de train por época",
        )
        # Por el mismo camino que el lanzamiento normal (en el servidor: cola y cuotas).
        _, res = run_study_managed(
            wf.ctx,
            project_id,
            dataset_version_id,
            pipeline_id,
            archspec_id,
            strategy,
            device=device,
            limit_train_batches=subset,
        )
        best = res.best_trial
        if best is None or not best.values:
            entry.status = "failed"
            continue
        entry.run_id, entry.metric, entry.status = best.run_id, float(best.values[0]), "complete"
        run = wf.ctx.repo(Run).get(best.run_id)
        if run.mlflow_run_id:
            wf.tracker.set_tags(run.mlflow_run_id, {"perceptron.tournament": result.id})
    scored = [e for e in result.entries if e.metric is not None]
    if scored:
        if direction == "minimize":
            result.winner = min(scored, key=lambda e: e.metric or 0.0).archspec_id
        else:
            result.winner = max(scored, key=lambda e: e.metric or 0.0).archspec_id
    logger.info(
        "mini-torneo",
        extra={"tournament": result.id, "winner": result.winner, "entries": len(result.entries)},
    )
    return result


def new_id_tournament() -> str:
    # Los torneos no son entidades: el id solo agrupa sus runs en MLflow.
    return "trn_" + new_id_suffix()


def new_id_suffix() -> str:
    from perceptron.core.ids import IdPrefix

    return new_id(IdPrefix.STUDY).split("_", 1)[1]
