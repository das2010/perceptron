"""Estrategia de HPO por reglas (RF-HPO-02 sin LLM; el estratega LLM llega en Capa 2).

Heurísticas de la tabla de SPEC §7.9.
"""

from __future__ import annotations

from typing import Literal

from perceptron.archspec.schema import ArchSpec, resolve
from perceptron.hpo.strategy import Budget, HPOStrategy, Objective, default_search_space
from perceptron.training.module import monitor_mode

PRUNE_MIN_EPOCHS = 10
CMAES_MIN_TRIALS = 20


def recommend_strategy(
    spec: ArchSpec,
    budget: Budget,
    *,
    metric: str | None = None,
    extra_objectives: list[Objective] | None = None,
) -> HPOStrategy:
    space = default_search_space(spec)
    es = spec.training.early_stopping
    metric = metric or (es.monitor if es else "val_loss")
    direction: Literal["minimize", "maximize"] = (
        "minimize" if monitor_mode(metric) == "min" else "maximize"
    )
    objectives = [Objective(metric=metric, direction=direction), *(extra_objectives or [])]
    epochs = budget.max_epochs_per_trial or int(resolve(spec.training.epochs))
    trials = budget.max_trials

    if extra_objectives:
        return HPOStrategy(
            strategy="nsga2",
            pruner="none",
            search_space=space,
            objectives=objectives,
            budget=budget,
            rationale="Hay más de un objetivo: NSGA-II busca el frente de Pareto.",
        )
    if trials <= 2 or not space:
        return HPOStrategy(
            strategy="single",
            pruner="none",
            search_space=[],
            objectives=objectives,
            budget=budget.model_copy(update={"max_trials": 1}),
            rationale="Presupuesto mínimo: se entrena una vez con defaults sólidos.",
        )
    discrete = [p for p in space if p.type != "float"]
    grid_size = 1
    for p in space:
        grid_size *= len(p.grid_values())
    pruner = (
        "asha"
        if epochs >= PRUNE_MIN_EPOCHS and trials >= 5
        else ("median" if trials >= 5 else "none")
    )
    if len(space) <= 3 and len(discrete) == len(space) and grid_size <= trials:
        strategy, why = (
            "grid",
            "Pocos hiperparámetros discretos: grid search cubre todo el espacio.",
        )
    elif not discrete and trials >= CMAES_MIN_TRIALS:
        strategy, why = "cmaes", "Espacio continuo y presupuesto medio/alto: CMA-ES."
    else:
        strategy, why = "tpe", "Caso general con espacio mixto: Optuna TPE."
    if pruner != "none":
        why += (
            " Con pruning ASHA se cortan temprano los trials con curvas malas."
            if pruner == "asha"
            else " Pruning por mediana para cortar trials claramente peores."
        )
    return HPOStrategy(
        strategy=strategy,  # type: ignore[arg-type]
        pruner=pruner,  # type: ignore[arg-type]
        search_space=space,
        objectives=objectives,
        budget=budget,
        rationale=why,
    )
