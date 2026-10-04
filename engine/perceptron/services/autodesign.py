"""Diseño guiado de punta a punta (ADR-0041, iteración 2).

Un solo pedido arma el diseño completo que el wizard le muestra a la persona para aceptar:

1. preparación (la del borrador o la propuesta por reglas);
2. propuestas de arquitectura evaluadas contra los requisitos del escenario;
3. mini-torneo de las mejores que cumplen los obligatorios (RF-ARC-03), con una métrica que
   todas registran y un presupuesto adaptado al tamaño de los datos;
4. elección con evidencia: la ganadora del torneo o, sin torneo, la recomendada;
5. épocas por trial según el tiempo estimado y estrategia de HPO para la elegida.

No cambia la arquitectura ni la estrategia elegidas del borrador: queda como `design` y la
persona la acepta (o elige otra). Nada del LLM se aplica solo.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from perceptron.archspec.defaults import spec_epochs
from perceptron.archspec.schema import ArchSpec
from perceptron.domain.enums import Device, Origin
from perceptron.hpo.strategy import Budget
from perceptron.services.design import DesignRequirements, RequirementCheck

if TYPE_CHECKING:
    from perceptron.services.llm_roles import ArchOption
    from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

TournamentMode = Literal["auto", "always", "never"]
TOP_K = 3
TOURNAMENT_FRACTION = 0.15
TOURNAMENT_MIN_EPOCHS = 2
SMALL_TRAIN = 5_000  # con menos ejemplos el torneo usa todo train en cada época
LARGE_SUBSET = 0.3
TOURNAMENT_BUDGET_S = 15 * 60  # en modo auto, más que esto estimado → sin torneo
STUDY_BUDGET_S = 20 * 60  # como `suggestEpochs` de la UI
DEFAULT_EPOCHS = 15
DEFAULT_TRIALS = 10
MAX_EPOCHS = 500
PREFERRED_METRICS = {
    "classification": ("f1_macro", "accuracy", "auroc"),
    "regression": ("mae", "rmse", "r2"),
}

Progress = Callable[[str], None]


class Candidate(BaseModel):
    archspec_id: str
    title: str
    rationale: str
    origin: Origin
    score: float | None = None
    recommended: bool = False
    eligible: bool = Field(default=True, description="Cumple todos los requisitos obligatorios")
    checks: list[RequirementCheck] = Field(default_factory=list)
    estimates: dict[str, float | None] = Field(default_factory=dict)
    tournament_metric: float | None = None
    tournament_status: str | None = None
    run_id: str | None = None


class TournamentSummary(BaseModel):
    metric: str
    direction: Literal["minimize", "maximize"]
    epochs: dict[str, int] = Field(default_factory=dict, description="Épocas por archspec_id")
    subset: float
    winner: str | None = None


class DesignOutcome(BaseModel):
    pipeline_id: str
    requirements: DesignRequirements | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    origin: Origin
    fallback_reason: str | None = None
    tournament: TournamentSummary | None = None
    tournament_skipped: str | None = Field(
        default=None, description="Por qué no hubo torneo (pocas candidatas, demasiado largo…)"
    )
    pick: str | None = None
    pick_reason: str = ""
    max_epochs_per_trial: int = DEFAULT_EPOCHS
    strategy: dict[str, Any] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------- piezas puras


def common_metric(specs: list[ArchSpec], preferred: str | None = None) -> str:
    """Métrica de validación que registran todas las candidatas (las pérdidas pueden no ser
    comparables: focal vs entropía cruzada, pesos por clase). La del borrador si todas la
    tienen; si no, la primera preferida de la tarea; si no, `val_loss`."""
    if not specs:
        return "val_loss"
    shared = set.intersection(*(set(s.metrics) for s in specs))
    if preferred and preferred.removeprefix("val_") in shared:
        return "val_" + preferred.removeprefix("val_")
    task = specs[0].task.type.value
    for name in PREFERRED_METRICS.get(task, ()):
        if name in shared:
            return f"val_{name}"
    return "val_loss"


def tournament_epochs(spec: ArchSpec) -> int:
    return max(TOURNAMENT_MIN_EPOCHS, round(spec_epochs(spec) * TOURNAMENT_FRACTION))


def tournament_subset(n_train: int) -> float:
    return 1.0 if n_train < SMALL_TRAIN else LARGE_SUBSET


def estimated_tournament_s(
    specs: list[ArchSpec], epoch_times: list[float | None], subset: float
) -> float | None:
    """Segundos estimados del torneo; None si falta alguna estimación."""
    total = 0.0
    for spec, t in zip(specs, epoch_times, strict=True):
        if t is None:
            return None
        total += t * tournament_epochs(spec) * subset
    return total


def suggest_epochs(spec: ArchSpec, epoch_time_s: float | None, trials: int) -> int:
    """Épocas por trial: las que propone la arquitectura si el estudio entra en el presupuesto
    de tiempo; si no, las que entren (nunca menos que el default). Igual que la UI."""
    proposed = spec_epochs(spec)
    if not epoch_time_s or epoch_time_s <= 0:
        return DEFAULT_EPOCHS
    affordable = math.floor(STUDY_BUDGET_S / (max(trials, 1) * epoch_time_s))
    floor = min(proposed, DEFAULT_EPOCHS)
    return min(MAX_EPOCHS, max(floor, min(proposed, affordable)))


def _fmt(value: float) -> str:
    return f"{value:.4g}"


# ---------------------------------------------------------------------- orquestación


def _candidate(option: ArchOption) -> Candidate:
    a = option.assessment
    checks = list(a.checks) if a else []
    return Candidate(
        archspec_id=option.record.id,
        title=option.title,
        rationale=option.rationale,
        origin=option.record.origin,
        score=a.score if a else None,
        recommended=bool(a and a.recommended),
        eligible=all(c.met for c in checks if c.level == "must"),
        checks=checks,
        estimates=dict(option.estimates),
    )


def auto_design(
    wf: Workflow,
    values: dict[str, Any],
    *,
    tournament: TournamentMode = "auto",
    device: Device | None = None,
    progress: Progress | None = None,
) -> DesignOutcome:
    """Diseño completo para el borrador (`values` = DraftValues serializados)."""
    say = progress or (lambda _msg: None)
    dv_id = values.get("dataset_version_id")
    if not dv_id:
        from perceptron.core.errors import ValidationError

        raise ValidationError("primero elegí los datos del proyecto")
    pipeline_id = values.get("pipeline_id")
    if not pipeline_id:
        say("pipeline")
        pipeline_id = wf.propose_pipeline(str(dv_id)).id
    say("architectures")
    dev = device.value if device else values.get("device")
    proposals = wf.roles.propose_architectures(
        str(dv_id), str(pipeline_id), mode="auto", device=dev
    )
    options = {o.record.id: o for o in proposals.options}
    specs = {i: ArchSpec.model_validate(o.record.spec) for i, o in options.items()}
    out = DesignOutcome(
        pipeline_id=str(pipeline_id),
        requirements=proposals.requirements,
        candidates=[_candidate(o) for o in proposals.options],
        origin=proposals.origin,
        fallback_reason=proposals.fallback_reason,
    )
    eligible = [c for c in out.candidates if c.eligible] or list(out.candidates)
    top = sorted(eligible, key=lambda c: -(c.score or 0.0))[:TOP_K]

    card = wf.profile_card(str(dv_id))
    n_train = int(card.split_counts.get("train") or card.num_samples)
    subset = tournament_subset(n_train)
    top_specs = [specs[c.archspec_id] for c in top]
    estimate = estimated_tournament_s(
        top_specs, [c.estimates.get("epoch_time_s") for c in top], subset
    )
    if tournament == "never":
        out.tournament_skipped = "disabled"
    elif len(top) < 2:
        out.tournament_skipped = "single_candidate"
    elif tournament == "auto" and (estimate is None or estimate > TOURNAMENT_BUDGET_S):
        out.tournament_skipped = "too_long"
    else:
        say("tournament")
        from perceptron.services.tournament import mini_tournament

        metric = common_metric(top_specs, values.get("target_metric"))
        res = mini_tournament(
            wf,
            wf.dataset(str(dv_id)).project_id,
            str(dv_id),
            str(pipeline_id),
            [c.archspec_id for c in top],
            fraction=TOURNAMENT_FRACTION,
            subset=subset,
            metric=metric,
            device=device or (Device(dev) if dev else None),
            min_epochs=TOURNAMENT_MIN_EPOCHS,
        )
        by_id = {e.archspec_id: e for e in res.entries}
        for c in out.candidates:
            entry = by_id.get(c.archspec_id)
            if entry is not None:
                c.tournament_metric, c.tournament_status = entry.metric, entry.status
                c.run_id = entry.run_id
        out.tournament = TournamentSummary(
            metric=res.metric,
            direction=res.direction,
            epochs={e.archspec_id: e.epochs for e in res.entries},
            subset=subset,
            winner=res.winner,
        )

    pick = _pick(out, top)
    if pick is not None:
        say("strategy")
        spec = specs[pick.archspec_id]
        trials = int(values.get("max_trials") or DEFAULT_TRIALS)
        epochs = int(
            values.get("max_epochs_per_trial")
            or suggest_epochs(spec, pick.estimates.get("epoch_time_s"), trials)
        )
        out.max_epochs_per_trial = epochs
        strategy = wf.roles.hpo_strategy(
            pick.archspec_id,
            Budget(max_trials=trials, max_epochs_per_trial=epochs),
            mode="auto",
            dataset_version_id=str(dv_id),
        )
        out.strategy = strategy.model_dump(mode="json")
    logger.info(
        "diseño guiado",
        extra={
            "candidates": len(out.candidates),
            "tournament": out.tournament.winner if out.tournament else out.tournament_skipped,
            "pick": out.pick,
        },
    )
    return out


def _pick(out: DesignOutcome, top: list[Candidate]) -> Candidate | None:
    """Ganadora del torneo con la evidencia, o la recomendada por requisitos."""
    winner_id = out.tournament.winner if out.tournament else None
    winner = next((c for c in out.candidates if c.archspec_id == winner_id), None)
    if winner is not None and out.tournament is not None:
        rivals = [c for c in top if c is not winner and c.tournament_metric is not None]
        vs = "; ".join(f"{c.title}: {_fmt(c.tournament_metric or 0.0)}" for c in rivals)
        out.pick, out.pick_reason = (
            winner.archspec_id,
            f"Ganó el entrenamiento corto de comparación con {out.tournament.metric} = "
            f"{_fmt(winner.tournament_metric or 0.0)}" + (f" (frente a {vs})." if vs else "."),
        )
        return winner
    chosen = next((c for c in out.candidates if c.recommended), None) or (top[0] if top else None)
    if chosen is not None:
        out.pick = chosen.archspec_id
        out.pick_reason = "Es la que mejor cumple los requisitos de diseño del escenario."
    return chosen
