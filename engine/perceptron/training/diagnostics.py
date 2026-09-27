"""Diagnóstico por reglas de un entrenamiento (evidencia para el LLM y fallback).

Trabaja sobre `RunResult.history` (una fila por época: train_loss, val_loss, val_<métrica>,
lr, epoch_time_s, samples_per_s). Cada detector devuelve evidencia con números concretos
para que el Diagnosticador (LLM) la explique o, sin LLM, se muestre tal cual.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from perceptron.llm.schemas import Diagnosis, Problem, SuggestedAction

IMBALANCE_RATIO = 0.2
OVERFIT_GAP = 1.10
PLATEAU_WINDOW = 5
PLATEAU_REL = 0.01


def _series(history: list[dict[str, float]], key: str) -> list[float]:
    return [float(h[key]) for h in history if key in h and h[key] is not None]


def _bad(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)


def detect(
    history: list[dict[str, float]],
    *,
    max_epochs: int | None = None,
    imbalance_ratio: float | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> list[tuple[Problem, list[SuggestedAction]]]:
    found: list[tuple[Problem, list[SuggestedAction]]] = []
    train = _series(history, "train_loss")
    val = _series(history, "val_loss")
    hps = hyperparameters or {}
    lr = hps.get("lr")

    if any(_bad(x) for x in train + val) or (train and train[-1] > 3 * min(train)):
        found.append(
            (
                Problem(
                    kind="divergence",
                    severity="high",
                    evidence=f"train_loss: mínimo {min(train, default=float('nan')):.4g}, "
                    f"final {train[-1] if train else float('nan'):.4g} (o valores no finitos)",
                    explanation="El entrenamiento divergió: la pérdida explota o no es finita.",
                ),
                [
                    SuggestedAction(
                        kind="change_hparam",
                        target="lr",
                        value=float(lr) * 0.1 if isinstance(lr, int | float) else None,
                        rationale="Bajar el learning rate un orden de magnitud.",
                    )
                ],
            )
        )
        return found

    if len(val) >= 4 and len(train) == len(val):
        best = min(range(len(val)), key=val.__getitem__)
        after = len(val) - 1 - best
        if after >= 3 and val[-1] > val[best] * OVERFIT_GAP and train[-1] < train[best]:
            gap = val[-1] / max(val[best], 1e-12)
            found.append(
                (
                    Problem(
                        kind="overfitting",
                        severity="high" if gap > 1.5 else "medium",
                        evidence=f"val_loss mínima {val[best]:.4g} en la época {best + 1}; "
                        f"luego sube a {val[-1]:.4g} mientras train_loss baja a {train[-1]:.4g}",
                        explanation="El modelo memoriza train y generaliza peor desde "
                        f"la época {best + 1}.",
                    ),
                    [
                        SuggestedAction(
                            kind="add_regularization",
                            target="dropout" if "dropout" in hps else None,
                            rationale="Más dropout o weight decay.",
                        ),
                        SuggestedAction(
                            kind="add_augmentation", rationale="Más variedad en los datos de train."
                        ),
                        SuggestedAction(
                            kind="fewer_epochs",
                            value=best + 1,
                            rationale="El mejor punto está antes; el early stopping lo toma.",
                        ),
                    ],
                )
            )

    if len(train) >= 5 and train[-1] > 0.9 * train[0]:
        found.append(
            (
                Problem(
                    kind="underfitting",
                    severity="medium",
                    evidence=f"train_loss pasa de {train[0]:.4g} a {train[-1]:.4g} "
                    f"en {len(train)} épocas (baja menos del 10 %)",
                    explanation="El modelo casi no aprende: falta capacidad o el LR no sirve.",
                ),
                [
                    SuggestedAction(
                        kind="change_architecture", rationale="Un modelo con más capacidad."
                    ),
                    SuggestedAction(
                        kind="change_hparam",
                        target="lr",
                        value=float(lr) * 3 if isinstance(lr, int | float) else None,
                        rationale="Probar un learning rate mayor.",
                    ),
                ],
            )
        )

    if len(train) >= 6:
        diffs = [b - a for a, b in itertools.pairwise(train)]
        flips = sum(1 for a, b in itertools.pairwise(diffs) if a * b < 0)
        if flips / max(len(diffs) - 1, 1) > 0.6 and train[-1] > 0.8 * train[0]:
            found.append(
                (
                    Problem(
                        kind="lr_too_high",
                        severity="medium",
                        evidence=f"train_loss oscila: {flips} cambios de dirección en "
                        f"{len(diffs)} épocas sin tendencia clara",
                        explanation="El learning rate parece demasiado alto para converger.",
                    ),
                    [
                        SuggestedAction(
                            kind="change_hparam",
                            target="lr",
                            value=float(lr) * 0.3 if isinstance(lr, int | float) else None,
                            rationale="Bajar el learning rate.",
                        )
                    ],
                )
            )

    if len(val) >= PLATEAU_WINDOW + 1 and not any(p.kind == "overfitting" for p, _ in found):
        window = val[-PLATEAU_WINDOW:]
        rel = (max(window) - min(window)) / max(abs(min(window)), 1e-12)
        if rel < PLATEAU_REL:
            found.append(
                (
                    Problem(
                        kind="plateau",
                        severity="low",
                        evidence=f"val_loss varía {rel:.2%} en las últimas {PLATEAU_WINDOW} épocas",
                        explanation="La validación se estancó.",
                    ),
                    [
                        SuggestedAction(
                            kind="change_hparam",
                            target="lr",
                            rationale="Reducir el LR (scheduler) o probar otra arquitectura.",
                        )
                    ],
                )
            )

    if max_epochs and len(history) >= max_epochs and val and val[-1] <= min(val) * 1.001:
        found.append(
            (
                Problem(
                    kind="stopped_too_early",
                    severity="low",
                    evidence=f"val_loss sigue mejorando en la última época ({len(history)})",
                    explanation="El presupuesto de épocas cortó un modelo que seguía mejorando.",
                ),
                [SuggestedAction(kind="more_epochs", rationale="Más épocas por trial.")],
            )
        )

    if imbalance_ratio is not None and imbalance_ratio < IMBALANCE_RATIO:
        found.append(
            (
                Problem(
                    kind="class_imbalance",
                    severity="medium",
                    evidence=f"ratio minoritaria/mayoritaria = {imbalance_ratio:.2f}",
                    explanation="Las clases están desbalanceadas; la accuracy puede engañar.",
                ),
                [SuggestedAction(kind="rebalance", rationale="Pesos de clase u oversampling.")],
            )
        )
    return found


def rules_diagnosis(
    history: list[dict[str, float]],
    *,
    max_epochs: int | None = None,
    imbalance_ratio: float | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> Diagnosis:
    found = detect(
        history,
        max_epochs=max_epochs,
        imbalance_ratio=imbalance_ratio,
        hyperparameters=hyperparameters,
    )
    if not found:
        return Diagnosis(summary="No se detectaron problemas en las curvas de entrenamiento.")
    problems = [p for p, _ in found]
    actions = [a for _, acts in found for a in acts]
    kinds = ", ".join(p.kind for p in problems)
    return Diagnosis(summary=f"Problemas detectados: {kinds}.", problems=problems, actions=actions)
