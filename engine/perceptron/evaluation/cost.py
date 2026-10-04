"""Umbral de decisión por costo de los errores (ADR-0040, fase 2: paso «Umbral de decisión»).

En clasificación binaria con costos asimétricos («no detectar una falla es 10× peor que una
falsa alarma») el umbral del 50 % no es el mejor. Se elige en **validación** el umbral que
minimiza el costo esperado y recién después se informa el test con ese umbral, junto al del
50 % para comparar. El test sellado no participa de la elección.

La clase positiva («el caso») es la segunda del modelo (índice 1), igual que en las curvas
ROC/PR y el umbral óptimo por F1.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class CostSpec(BaseModel):
    worse: Literal["false_negative", "false_positive"]
    ratio: float = Field(ge=1, le=1000, description="Cuántas veces peor es el error más caro")


def _counts(y: np.ndarray, score: np.ndarray, t: float) -> dict[str, int]:
    pred = score >= t
    pos = y == 1
    return {
        "tp": int(np.sum(pos & pred)),
        "fp": int(np.sum(~pos & pred)),
        "fn": int(np.sum(pos & ~pred)),
        "tn": int(np.sum(~pos & ~pred)),
    }


def _cost(c: dict[str, int], spec: CostSpec) -> float:
    if spec.worse == "false_negative":
        return spec.ratio * c["fn"] + c["fp"]
    return c["fn"] + spec.ratio * c["fp"]


def _summary(y: np.ndarray, score: np.ndarray, t: float, spec: CostSpec) -> dict[str, float]:
    c = _counts(y, score, t)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return {
        **{k: float(v) for k, v in c.items()},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "cost": _cost(c, spec),
        "cost_per_case": round(_cost(c, spec) / max(len(y), 1), 4),
    }


def cost_threshold(
    y_val: np.ndarray,
    score_val: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
    spec: CostSpec,
    positive: str,
) -> dict[str, object]:
    """Umbral que minimiza el costo en validación y su efecto en test (vs. el 50 %)."""
    # Incluye «ningún positivo» (por encima de todo puntaje): puede ser lo óptimo si una falsa
    # alarma es carísima.
    candidates = np.unique(np.concatenate([score_val, [0.5, 1.0 + 1e-9]]))
    costs = [_cost(_counts(y_val, score_val, t), spec) for t in candidates]
    best = min(costs)
    # Empates: el más cercano al 50 % (no se mueve el umbral sin motivo).
    t = float(
        min(
            (c for c, k in zip(candidates, costs, strict=True) if k == best),
            key=lambda c: abs(c - 0.5),
        )
    )
    return {
        "positive_class": positive,
        "worse": spec.worse,
        "ratio": spec.ratio,
        "threshold": round(t, 4),
        "validation": _summary(y_val, score_val, t, spec),
        "test": _summary(y_test, score_test, t, spec),
        "test_at_50": _summary(y_test, score_test, 0.5, spec),
    }
