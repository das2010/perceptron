"""Plan de presupuesto del HPO: cuántos intentos y cuántas épocas por intento (ADR-0041).

Antes los números los ponía la persona y funcionaban como techo; nadie calculaba cuántos
convenía (caso «Tabla X»: 10 intentos para buscar un solo hiperparámetro de una regresión
lineal). Ahora el sistema los propone a partir de hechos medibles y del tiempo que la persona
quiere esperar:

- **épocas:** las que pide la arquitectura; menos con backbone preentrenado (converge rápido) o
  con ajuste exacto (la regresión lineal termina por mínimos cuadrados);
- **intentos:** según cuántos hiperparámetros se buscan (TPE necesita unos 10 para arrancar y
  ~6 más por dimensión); 1 si no hay nada que buscar;
- **tiempo:** intentos × épocas × segundos por época (medidos) entran en el tiempo disponible;
  si no entra, se recortan primero intentos (hasta un mínimo útil) y después épocas.

Es puro y determinístico. El estratega LLM recibe el plan como techo y puede ajustarlo con su
justificación; la persona lo acepta o lo cambia.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from perceptron.archspec.defaults import is_linear_regression, spec_epochs
from perceptron.archspec.schema import ArchSpec
from perceptron.hpo.strategy import default_search_space

DEFAULT_TIME_S = 20 * 60
BASE_TRIALS = 4
TRIALS_PER_PARAM = 6
MAX_TRIALS = 60
MIN_TRIALS_PER_PARAM = 2  # por debajo de esto la búsqueda casi no explora
MIN_EPOCHS = 5
PRETRAINED_EPOCHS = 15
EXACT_FIT_EPOCHS = 10
PRETRAINED_BLOCKS = frozenset({"vision.timm_backbone", "text.hf_encoder"})


class BudgetPlan(BaseModel):
    max_trials: int = Field(ge=1)
    max_epochs_per_trial: int = Field(ge=1)
    tuned_params: list[str] = Field(default_factory=list)
    epoch_time_s: float | None = Field(default=None, description="Medido o estimado")
    time_budget_s: float
    estimated_s: float | None = Field(
        default=None, description="Tope: intentos × épocas × s/época (el early stopping lo baja)"
    )
    reasons: list[str] = Field(default_factory=list, description="Por qué cada número")


def _pretrained(spec: ArchSpec) -> bool:
    return any(
        n.block in PRETRAINED_BLOCKS and n.params.get("pretrained", True) is not False
        for n in spec.nodes
    )


def plan_budget(
    spec: ArchSpec,
    *,
    epoch_time_s: float | None,
    time_budget_s: float | None = None,
    noise_free: bool = False,
) -> BudgetPlan:
    budget = float(time_budget_s or DEFAULT_TIME_S)
    reasons: list[str] = []
    params = [p.name for p in default_search_space(spec)]
    if noise_free and any("dropout" in p for p in params):
        params = [p for p in params if "dropout" not in p]
        reasons.append(
            "Datos casi sin ruido: el dropout queda en 0 y no se busca (solo empeoraría el ajuste)."
        )

    epochs = spec_epochs(spec)
    if is_linear_regression(spec):
        trials, epochs = 1, min(epochs, EXACT_FIT_EPOCHS)
        reasons.append(
            "Regresión lineal: termina con el ajuste exacto por mínimos cuadrados, así que no hay "
            f"hiperparámetros que buscar (1 intento) y alcanzan {epochs} épocas."
        )
        params = []
    elif not params:
        trials = 1
        reasons.append("La arquitectura no tiene hiperparámetros para buscar: 1 intento.")
    else:
        trials = min(MAX_TRIALS, BASE_TRIALS + TRIALS_PER_PARAM * len(params))
        reasons.append(
            f"{len(params)} hiperparámetro(s) a buscar ({', '.join(params)}): {trials} intentos "
            "le dan a la búsqueda margen para explorar y después afinar."
        )
    if _pretrained(spec) and epochs > PRETRAINED_EPOCHS:
        epochs = PRETRAINED_EPOCHS
        reasons.append(
            f"Backbone preentrenado: converge rápido, {epochs} épocas por intento alcanzan."
        )
    elif not is_linear_regression(spec):
        reasons.append(
            f"{epochs} épocas por intento, las que pide la arquitectura (el early stopping corta "
            "antes si deja de mejorar)."
        )

    if epoch_time_s and epoch_time_s > 0:
        min_trials = max(1, min(trials, MIN_TRIALS_PER_PARAM * max(len(params), 1) + 3))
        fit_trials = math.floor(budget / (epochs * epoch_time_s))
        if fit_trials < trials:
            if fit_trials >= min_trials:
                trials = fit_trials
                reasons.append(f"Para entrar en el tiempo disponible: {trials} intentos.")
            else:
                trials = min_trials
                fit_epochs = math.floor(budget / (trials * epoch_time_s))
                epochs = max(min(epochs, MIN_EPOCHS), min(epochs, fit_epochs))
                reasons.append(
                    f"Para entrar en el tiempo disponible: {trials} intentos de {epochs} épocas."
                )
        estimated = trials * epochs * epoch_time_s
        if estimated > budget:
            reasons.append(
                "Aun así se estima más largo que el tiempo disponible: conviene más tiempo, un "
                "modelo más liviano o una GPU."
            )
    else:
        estimated = None
        reasons.append("Sin estimación del tiempo por época: no se ajustó al tiempo disponible.")
    return BudgetPlan(
        max_trials=trials,
        max_epochs_per_trial=max(1, epochs),
        tuned_params=params,
        epoch_time_s=epoch_time_s,
        time_budget_s=budget,
        estimated_s=round(estimated, 1) if estimated is not None else None,
        reasons=reasons,
    )
