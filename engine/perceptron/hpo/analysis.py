"""Visualizaciones de un estudio de HPO (RF-HPO-06): historia, importancia de
hiperparámetros, coordenadas paralelas y frente de Pareto.

Los trials se reconstruyen desde los runs del estudio (cada run guarda sus hiperparámetros y
sus mejores métricas de validación). La importancia usa PED-ANOVA de Optuna, sin
dependencias extra; con pocos trials no se estima.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel

from perceptron.hpo.strategy import HPOStrategy, SearchParam

logger = logging.getLogger(__name__)

MIN_TRIALS_FOR_IMPORTANCE = 4


class TrialPoint(BaseModel):
    number: int
    run_id: str
    status: str
    params: dict[str, Any]
    values: list[float | None]
    best_so_far: float | None = None
    pareto: bool = False


class ObjectiveInfo(BaseModel):
    metric: str
    direction: Literal["minimize", "maximize"]


class StudyAnalysis(BaseModel):
    objectives: list[ObjectiveInfo]
    params: list[str]
    trials: list[TrialPoint]
    importance: dict[str, float]


def _dominates(a: list[float], b: list[float], signs: list[float]) -> bool:
    better_or_equal = all(s * x <= s * y for x, y, s in zip(a, b, signs, strict=True))
    strictly = any(s * x < s * y for x, y, s in zip(a, b, signs, strict=True))
    return better_or_equal and strictly


def _distribution(p: SearchParam) -> Any:
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    if p.type == "categorical":
        return CategoricalDistribution(p.choices or [])
    if p.type == "int":
        return IntDistribution(int(p.low or 0), int(p.high or 0), log=p.log)
    return FloatDistribution(float(p.low or 0), float(p.high or 0), log=p.log)


def _importance(
    strategy: HPOStrategy, trials: list[TrialPoint], direction: str
) -> dict[str, float]:
    import optuna
    from optuna.importance import get_param_importances

    complete = [t for t in trials if t.values and t.values[0] is not None]
    if len(complete) < MIN_TRIALS_FOR_IMPORTANCE:
        return {}
    space = {p.name: p for p in strategy.search_space}
    dists = {name: _distribution(p) for name, p in space.items()}
    study = optuna.create_study(direction=direction)
    for t in complete:
        params = {k: v for k, v in t.params.items() if k in dists}
        # Valores fuera del espacio (p. ej. un trial manual) no aportan a la importancia.
        if not params or not all(space[k].accepts(v) for k, v in params.items()):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions={k: dists[k] for k in params},
                value=float(t.values[0]),  # type: ignore[arg-type]
            )
        )
    try:
        try:
            from optuna.importance import PedAnovaImportanceEvaluator

            evaluator: Any = PedAnovaImportanceEvaluator()
        except ImportError:  # Optuna viejo: el evaluador por defecto
            evaluator = None
        scores = get_param_importances(study, evaluator=evaluator)
    except Exception:  # pocos trials, parámetros constantes…
        logger.info("no se pudo estimar la importancia de hiperparámetros", exc_info=True)
        return {}
    return {k: float(v) for k, v in scores.items()}


def analyze(strategy: HPOStrategy, runs: list[Any]) -> StudyAnalysis:
    """`runs`: los `Run` del estudio, en orden de lanzamiento."""
    objectives = [
        ObjectiveInfo(metric=o.metric, direction=o.direction) for o in strategy.objectives
    ]
    signs = [1.0 if o.direction == "minimize" else -1.0 for o in objectives]
    trials: list[TrialPoint] = []
    best: float | None = None
    for i, run in enumerate(runs):
        values: list[float | None] = []
        for o in objectives:
            v = run.metrics.get(o.metric)
            values.append(float(v) if v is not None else None)
        first = values[0]
        if first is not None and (best is None or signs[0] * first < signs[0] * best):
            best = first
        trials.append(
            TrialPoint(
                number=i,
                run_id=run.id,
                status=str(getattr(run.status, "value", run.status)),
                params=dict(run.hyperparams),
                values=values,
                best_so_far=best,
            )
        )
    if len(objectives) > 1:
        full = [t for t in trials if all(v is not None for v in t.values)]
        for t in full:
            vt = [float(v) for v in t.values if v is not None]
            t.pareto = not any(
                _dominates([float(v) for v in o.values if v is not None], vt, signs)
                for o in full
                if o is not t
            )
    params = sorted({k for t in trials for k in t.params})
    importance = _importance(strategy, trials, objectives[0].direction) if trials else {}
    return StudyAnalysis(objectives=objectives, params=params, trials=trials, importance=importance)
