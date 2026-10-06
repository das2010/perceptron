"""Estrategia de HPO por reglas (RF-HPO-02 sin LLM; el estratega LLM llega en Capa 2).

Heurísticas de la tabla de SPEC §7.9.
"""

from __future__ import annotations

from typing import Literal

from perceptron.archspec.defaults import is_linear_regression
from perceptron.archspec.schema import ArchSpec, resolve
from perceptron.domain.enums import Origin
from perceptron.hpo.strategy import Budget, HPOStrategy, Objective, default_search_space
from perceptron.training.module import monitor_mode

PRUNE_MIN_EPOCHS = 10
ASHA_MIN_TRIALS = 30
MEDIAN_MIN_TRIALS = 5
CMAES_MIN_TRIALS = 20


def training_metrics(spec: ArchSpec) -> set[str]:
    """Métricas de validación que registra el entrenamiento de esta arquitectura."""
    from perceptron.tasks import get_adapter

    names = get_adapter(spec.task.type).build_metrics(spec).keys()
    return {"val_loss", *(f"val_{n}" for n in names)}


def objective_metric(spec: ArchSpec, preferred: str | None) -> str | None:
    """`preferred` si el entrenamiento la registra; si no, None (default de la arquitectura)."""
    return preferred if preferred and preferred in training_metrics(spec) else None


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
            origin=Origin.RULES,
            rationale="Hay más de un objetivo: NSGA-II busca el frente de Pareto.",
        )
    if is_linear_regression(spec) and not extra_objectives:
        # El entrenamiento termina con el ajuste exacto por mínimos cuadrados
        # (`training.exact`): el resultado no depende de los hiperparámetros.
        return HPOStrategy(
            strategy="single",
            pruner="none",
            search_space=[],
            objectives=objectives,
            budget=budget.model_copy(update={"max_trials": 1}),
            origin=Origin.RULES,
            rationale=(
                "Regresión lineal: el entrenamiento termina con el ajuste exacto por mínimos "
                "cuadrados, que no depende de los hiperparámetros. Alcanza con un intento."
            ),
        )
    if trials <= 1 or not space:
        return HPOStrategy(
            strategy="single",
            pruner="none",
            search_space=[],
            objectives=objectives,
            budget=budget.model_copy(update={"max_trials": 1}),
            origin=Origin.RULES,
            rationale="Presupuesto mínimo: se entrena una vez con defaults sólidos.",
        )
    discrete = [p for p in space if p.type != "float"]
    grid_size = 1
    for p in space:
        grid_size *= len(p.grid_values())
    # ASHA reduce ×3 en cada peldaño: con pocos trials casi ninguno llega al final (en la
    # práctica, un solo entrenamiento). Con presupuestos chicos se poda por mediana y con
    # un calentamiento del 25 % de las épocas, para no cortar por el ruido del arranque.
    if trials >= ASHA_MIN_TRIALS and epochs >= PRUNE_MIN_EPOCHS:
        pruner = "asha"
    elif trials >= MEDIAN_MIN_TRIALS:
        pruner = "median"
    else:
        pruner = "none"
    # Un tercio de las épocas sin poda: los modelos suelen pasar por mesetas antes de
    # aprender los rasgos sutiles (visto en UC-09: meseta en 0,72 hasta la época ~15).
    warmup = max(2, epochs // 3)
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
        pruner_warmup_epochs=warmup,
        search_space=space,
        objectives=objectives,
        budget=budget,
        origin=Origin.RULES,
        rationale=why,
    )
