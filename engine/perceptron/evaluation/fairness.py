"""Fairness por atributos sensibles sobre el test sellado (RF-EVL-04; ADR-0028).

Métricas por subgrupo y las diferencias de Fairlearn, con sus mismas definiciones:
- demographic parity difference: máx − mín de la tasa de selección, P(ŷ = positivo);
- equalized odds difference: máx(diferencia de TPR, diferencia de FPR) entre grupos.

En regresión se compara el MAE por grupo. Los atributos numéricos continuos se agrupan
por cuartiles. Un atributo puede no ser entrada del modelo: igual se mide sobre él.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError
from perceptron.data.view import DatasetView
from perceptron.evaluation.errors import group_values
from perceptron.evaluation.predictions import ROW, eval_frame, load_predictions, mean

DEFAULT_THRESHOLD = 0.1
MIN_GROUP = 5


class GroupMetrics(BaseModel):
    group: str
    support: int
    selection_rate: float | None = None
    accuracy: float | None = None
    tpr: float | None = None
    fpr: float | None = None
    mae: float | None = None


class FairnessReport(BaseModel):
    attribute: str
    task: str
    positive_class: str | None = None
    groups: list[GroupMetrics]
    demographic_parity_difference: float | None = None
    equalized_odds_difference: float | None = None
    mae_difference: float | None = None
    threshold: float
    alerts: list[str] = Field(default_factory=list)


def _group_column(df: pl.DataFrame, attribute: str) -> pl.Series:
    if attribute not in df.columns:
        raise ValidationError(f"el atributo {attribute!r} no está en el dataset")
    groups = group_values(df[attribute])
    if groups is None:
        raise ValidationError(f"{attribute!r} tiene demasiados valores distintos para agrupar")
    return groups


def _rate(mask: pl.Series) -> float | None:
    return float(mask.mean()) if mask.len() else None  # type: ignore[arg-type]


def fairness_report(
    eval_dir: Path,
    dataset_dir: Path,
    attribute: str,
    *,
    positive_class: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> FairnessReport:
    df = eval_frame(DatasetView(dataset_dir)).join(load_predictions(eval_dir), on=ROW, how="inner")
    groups = _group_column(df, attribute)
    df = df.with_columns(groups.alias("__g__"))
    regression = df["y_true"].dtype.is_float()
    alerts: list[str] = []

    if regression:
        rows = []
        for g, part in df.group_by("__g__"):
            if part.height < MIN_GROUP:
                continue
            mae = mean((part["y_pred"] - part["y_true"]).abs())
            rows.append(GroupMetrics(group=str(g[0]), support=part.height, mae=mae))
        maes = [r.mae for r in rows if r.mae is not None]
        overall = mean((df["y_pred"] - df["y_true"]).abs())
        diff = (max(maes) - min(maes)) if len(maes) > 1 else 0.0
        if overall and diff / overall > threshold:
            alerts.append(
                f"El MAE varía {diff:.4g} entre grupos de {attribute} (> {threshold:.0%} del total)"
            )
        return FairnessReport(
            attribute=attribute,
            task="regression",
            groups=sorted(rows, key=lambda r: r.group),
            mae_difference=diff,
            threshold=threshold,
            alerts=alerts,
        )

    classes = sorted({str(c) for c in df["y_true"].drop_nulls().to_list()})
    if positive_class is None:
        if len(classes) != 2:
            raise ValidationError("en multiclase indicá la clase positiva (positive_class)")
        positive_class = classes[1]
    if positive_class not in classes:
        raise ValidationError(f"la clase {positive_class!r} no aparece en el test")
    rows = []
    for g, part in df.group_by("__g__"):
        if part.height < MIN_GROUP:
            continue
        pred_pos = part["y_pred"].cast(pl.String) == positive_class
        true_pos = part["y_true"].cast(pl.String) == positive_class
        rows.append(
            GroupMetrics(
                group=str(g[0]),
                support=part.height,
                selection_rate=_rate(pred_pos),
                accuracy=_rate(part["y_true"] == part["y_pred"]),
                tpr=_rate(pred_pos.filter(true_pos)),
                fpr=_rate(pred_pos.filter(~true_pos)),
            )
        )

    def spread(values: list[float | None]) -> float:
        vals = [v for v in values if v is not None]
        return (max(vals) - min(vals)) if len(vals) > 1 else 0.0

    dp = spread([r.selection_rate for r in rows])
    eo = max(spread([r.tpr for r in rows]), spread([r.fpr for r in rows]))
    if dp > threshold:
        alerts.append(
            f"Paridad demográfica: la tasa de «{positive_class}» difiere {dp:.2f} entre grupos de "
            f"{attribute} (umbral {threshold:.2f})"
        )
    if eo > threshold:
        alerts.append(
            f"Igualdad de oportunidades: TPR/FPR difieren {eo:.2f} entre grupos de {attribute} "
            f"(umbral {threshold:.2f})"
        )
    return FairnessReport(
        attribute=attribute,
        task="classification",
        positive_class=positive_class,
        groups=sorted(rows, key=lambda r: r.group),
        demographic_parity_difference=dp,
        equalized_odds_difference=eo,
        threshold=threshold,
        alerts=alerts,
    )
