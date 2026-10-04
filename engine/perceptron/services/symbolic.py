"""Fórmula sugerida (ADR-0039): busca, guarda y evalúa fórmulas de regresión simbólica."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from perceptron.core.errors import NotFoundError, ValidationError
from perceptron.domain.enums import Modality
from perceptron.domain.models import DatasetVersion, SymbolicFit
from perceptron.evaluation.symbolic import (
    SymbolicConfig,
    check_symbolic,
    evaluate_expression,
    fit_symbolic,
)

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext
    from perceptron.data.view import DatasetView


def symbolic_view(ctx: EngineContext, project_id: str, dataset_version_id: str) -> DatasetView:
    """Vista del dataset si admite fórmula sugerida (422 con el motivo si no)."""
    from perceptron.services.workflow import Workflow

    dv = ctx.repo(DatasetVersion).get(dataset_version_id)
    if dv.project_id != project_id:
        raise NotFoundError(f"el dataset {dataset_version_id} no es de este proyecto")
    if dv.modality is not Modality.TABULAR:
        raise ValidationError(
            "La fórmula sugerida es para datos tabulares.", details={"reason": "not_tabular"}
        )
    view = Workflow(ctx).view(dv)
    check_symbolic(view)
    return view


def run_symbolic(
    ctx: EngineContext,
    project_id: str,
    dataset_version_id: str,
    config: SymbolicConfig,
    progress: Callable[[int, int], None] | None = None,
) -> SymbolicFit:
    view = symbolic_view(ctx, project_id, dataset_version_id)
    try:
        out = fit_symbolic(view, config, progress)
    except ImportError as e:  # instalación sin el extra `ml`
        raise ValidationError(
            "La regresión simbólica no está instalada en este Engine.",
            details={"reason": "not_installed"},
        ) from e
    return ctx.repo(SymbolicFit).add(
        SymbolicFit(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            time_limit_s=config.time_limit_s,
            **out.model_dump(exclude={"candidates"}),
            candidates=[c.model_dump() for c in out.candidates],
        )
    )


def predict_symbolic(
    fit: SymbolicFit, rows: list[Mapping[str, float | None]]
) -> list[float | None]:
    """Predicciones de la fórmula para filas nuevas (`None` si no se puede calcular)."""
    missing = sorted({f for r in rows for f in fit.features if r.get(f) is None})
    if missing:
        raise ValidationError("Faltan valores de entrada.", details={"missing": missing[:20]})
    x = np.array([[r.get(f) for f in fit.features] for r in rows], dtype=float).reshape(
        len(rows), len(fit.features)
    )
    pred = evaluate_expression(fit.expression, x)
    return [float(v) if np.isfinite(v) else None for v in pred]
