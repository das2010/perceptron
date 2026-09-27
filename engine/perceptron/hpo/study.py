"""Ejecución de estudios de HPO con Optuna (RF-HPO-01, 03, 05).

Interfaz ask/tell: cada trial es un run del supervisor (un subproceso). Los
eventos de época alimentan `trial.report` para el pruning. El presupuesto corta
por n.º de trials, tiempo total o métrica objetivo (lo que ocurra primero). El
storage SQLite permite reanudar un estudio interrumpido.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import optuna
from pydantic import BaseModel, Field

from perceptron.archspec.schema import Scalar
from perceptron.core.events import EventBus
from perceptron.hpo.strategy import HPOStrategy, SearchParam
from perceptron.training.config import RunConfig, RunEvent, RunResult
from perceptron.training.runner import start_run

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_TOPIC = "study.event"
StopReason = Literal["max_trials", "max_time", "target_reached", "cancelled", "exhausted"]


class Handle(Protocol):
    """Lo que el estudio necesita de un run (el real es `RunHandle`)."""

    def wait(
        self, on_event: Callable[[RunEvent], None] | None = None, timeout: float | None = None
    ) -> RunResult: ...
    def prune(self) -> None: ...
    def stop(self) -> None: ...


Launcher = Callable[[RunConfig], Handle]


class TrialRecord(BaseModel):
    number: int
    run_id: str
    state: Literal["complete", "pruned", "fail"]
    params: dict[str, Any]
    values: list[float] | None = None
    duration_s: float = 0.0
    epochs: int = 0
    best_checkpoint: Path | None = None
    error: str | None = None


class StudyResult(BaseModel):
    study_name: str
    strategy: HPOStrategy
    stop_reason: StopReason
    trials: list[TrialRecord] = Field(default_factory=list)
    best_trial: TrialRecord | None = None
    pareto_front: list[TrialRecord] = Field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class StudyControl:
    """Cancelación desde afuera (API / UI): corta el trial en curso y no lanza más."""

    cancelled: bool = False
    current: Handle | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            if self.current is not None:
                self.current.stop()


def _sampler(s: HPOStrategy) -> optuna.samplers.BaseSampler:
    match s.strategy:
        case "random" | "single":
            return optuna.samplers.RandomSampler(seed=s.seed)
        case "grid":
            grid = {p.name: p.grid_values() for p in s.search_space}
            return optuna.samplers.GridSampler(grid, seed=s.seed)
        case "cmaes":
            try:
                import cmaes  # noqa: F401
            except ImportError:
                logger.warning("cmaes no instalado: se usa TPE")
                return optuna.samplers.TPESampler(seed=s.seed)
            return optuna.samplers.CmaEsSampler(seed=s.seed)
        case "nsga2":
            return optuna.samplers.NSGAIISampler(seed=s.seed)
    return optuna.samplers.TPESampler(seed=s.seed, multivariate=True)


def _pruner(s: HPOStrategy) -> optuna.pruners.BasePruner:
    max_epochs = s.budget.max_epochs_per_trial or 30
    match s.pruner:
        case "median":
            return optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
        case "asha":
            return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
        case "hyperband":
            return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max_epochs)
    return optuna.pruners.NopPruner()


def _suggest(trial: optuna.Trial, space: list[SearchParam]) -> dict[str, Scalar]:
    params: dict[str, Scalar] = {}
    for p in space:
        if p.condition is not None and params.get(p.condition.param) != p.condition.equals:
            continue
        if p.type == "categorical":
            params[p.name] = trial.suggest_categorical(p.name, p.choices or [])
        elif p.type == "int":
            params[p.name] = trial.suggest_int(p.name, int(p.low or 0), int(p.high or 0), log=p.log)
        else:
            step = None if p.log else p.step
            params[p.name] = trial.suggest_float(
                p.name, float(p.low or 0), float(p.high or 0), log=p.log, step=step
            )
    return params


class _TrialMonitor:
    """Recibe los eventos de un trial: reporta a Optuna y decide el pruning."""

    def __init__(
        self,
        trial: optuna.Trial,
        strategy: HPOStrategy,
        metric: str,
        forward: Callable[[RunEvent], None] | None,
    ) -> None:
        self.trial = trial
        self.strategy = strategy
        self.metric = metric
        self.forward = forward
        self.handle: Handle | None = None
        self.pruned = False
        self.extra: dict[str, float] = {}

    def __call__(self, ev: RunEvent) -> None:
        if self.forward is not None:
            self.forward(ev)
        if ev.event == "started":
            self.extra["num_params"] = float(ev.data.get("num_params", 0))
        if (
            ev.event == "epoch"
            and not self.strategy.multi_objective
            and self.metric in ev.metrics
            and ev.epoch is not None
        ):
            self.trial.report(ev.metrics[self.metric], step=ev.epoch)
            if self.trial.should_prune() and not self.pruned and self.handle is not None:
                self.pruned = True
                self.handle.prune()


def _reached(value: float, target: float | None, direction: str) -> bool:
    if target is None:
        return False
    return value <= target if direction == "minimize" else value >= target


def run_study(
    strategy: HPOStrategy,
    base: RunConfig,
    *,
    storage: Path,
    study_name: str,
    launcher: Launcher | None = None,
    bus: EventBus | None = None,
    control: StudyControl | None = None,
    on_trial_end: Callable[[TrialRecord, RunConfig, RunResult], None] | None = None,
    on_run_event: Callable[[RunConfig], Callable[[RunEvent], None] | None] | None = None,
) -> StudyResult:
    """Ejecuta (o reanuda) el estudio. `launcher` permite inyectar runs simulados en tests."""
    launch: Launcher = launcher or (lambda cfg: start_run(cfg, bus))
    control = control or StudyControl()
    storage.parent.mkdir(parents=True, exist_ok=True)
    directions = [o.direction for o in strategy.objectives]
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage.resolve().as_posix()}",
        load_if_exists=True,
        directions=directions,
        sampler=_sampler(strategy),
        pruner=_pruner(strategy) if not strategy.multi_objective else optuna.pruners.NopPruner(),
    )
    budget = strategy.budget
    metric = strategy.objectives[0].metric
    start = time.time()
    finished_states = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    )
    records: list[TrialRecord] = []
    stop_reason: StopReason = "max_trials"
    grid_total = None
    if strategy.strategy == "grid":
        grid_total = 1
        for p in strategy.search_space:
            grid_total *= len(p.grid_values())

    while True:
        done = [t for t in study.get_trials(deepcopy=False) if t.state in finished_states]
        if len(done) >= budget.max_trials:
            stop_reason = "max_trials"
            break
        if grid_total is not None and len(done) >= grid_total:
            stop_reason = "exhausted"
            break
        if budget.max_time_s and time.time() - start >= budget.max_time_s:
            stop_reason = "max_time"
            break
        if control.cancelled:
            stop_reason = "cancelled"
            break
        best_done = [t for t in done if t.state == optuna.trial.TrialState.COMPLETE]
        if (
            not strategy.multi_objective
            and best_done
            and _reached(study.best_value, budget.target_value, directions[0])
        ):
            stop_reason = "target_reached"
            break

        trial = study.ask()
        overrides = _suggest(trial, strategy.search_space) if strategy.strategy != "single" else {}
        run_id = f"{base.run_id}-t{trial.number:03d}"
        cfg = base.model_copy(
            update={
                "run_id": run_id,
                "run_dir": base.run_dir.parent / run_id,
                "overrides": {**base.overrides, **overrides},
                "max_epochs": budget.max_epochs_per_trial or base.max_epochs,
            }
        )
        trial.set_user_attr("run_id", run_id)
        monitor = _TrialMonitor(
            trial, strategy, metric, on_run_event(cfg) if on_run_event else None
        )
        with control._lock:
            handle = launch(cfg)
            control.current = handle
            monitor.handle = handle
        result = handle.wait(monitor, timeout=budget.trial_timeout_s)
        control.current = None
        pruned = {"flag": monitor.pruned}
        num_params = monitor.extra

        record = TrialRecord(
            number=trial.number,
            run_id=run_id,
            state="fail",
            params=dict(overrides),
            duration_s=result.duration_s,
            epochs=result.epochs,
            best_checkpoint=result.best_checkpoint,
        )
        if result.status == "pruned" or pruned["flag"]:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            record.state = "pruned"
        elif result.status == "succeeded":
            values = []
            for obj in strategy.objectives:
                if obj.metric in result.best_metrics:
                    values.append(result.best_metrics[obj.metric])
                elif obj.metric in num_params:
                    values.append(num_params[obj.metric])
                elif obj.metric == "duration_s":
                    values.append(result.duration_s)
            if len(values) == len(strategy.objectives):
                study.tell(trial, values if strategy.multi_objective else values[0])
                record.state = "complete"
                record.values = values
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                record.error = f"el run no reportó {metric}"
        else:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            record.error = (result.error or {}).get("message", result.status)
            if result.status == "cancelled" and control.cancelled:
                record.error = "cancelado"
        records.append(record)
        if bus is not None:
            bus.publish(STUDY_TOPIC, study=study_name, trial=record.model_dump(mode="json"))
        if on_trial_end is not None:
            on_trial_end(record, cfg, result)

    return _summarize(study, strategy, study_name, stop_reason, records, time.time() - start)


def _summarize(
    study: optuna.Study,
    strategy: HPOStrategy,
    name: str,
    reason: StopReason,
    records: list[TrialRecord],
    duration: float,
) -> StudyResult:
    by_number = {r.number: r for r in records}

    def rec(t: optuna.trial.FrozenTrial) -> TrialRecord:
        if t.number in by_number:
            return by_number[t.number]
        state = {"COMPLETE": "complete", "PRUNED": "pruned"}.get(t.state.name, "fail")
        return TrialRecord(
            number=t.number,
            run_id=str(t.user_attrs.get("run_id", t.number)),
            state=state,  # type: ignore[arg-type]
            params=dict(t.params),
            values=list(t.values) if t.values else None,
        )

    all_trials = [rec(t) for t in study.get_trials(deepcopy=False) if t.state.is_finished()]
    complete = [
        t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best = None
    pareto: list[TrialRecord] = []
    if complete:
        if strategy.multi_objective:
            pareto = [rec(t) for t in study.best_trials]
        else:
            best = rec(study.best_trial)
    return StudyResult(
        study_name=name,
        strategy=strategy,
        stop_reason=reason,
        trials=all_trials,
        best_trial=best,
        pareto_front=pareto,
        duration_s=round(duration, 3),
    )
