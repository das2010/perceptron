"""Análisis de errores sobre el test sellado (RF-EVL-03).

- Slices automáticos de bajo rendimiento: por valor de las columnas categóricas y por
  cuartiles de las numéricas (tabular), o por clase real (resto de modalidades).
- Confusiones más frecuentes.
- Posibles errores de etiqueta (confident learning simplificado: la etiqueta recibe menos
  probabilidad que el umbral de su clase y otra clase supera el suyo).
- Explorador de mal predichos: las filas con su predicción, confianza y columnas originales.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from perceptron.data.splits import FOLD_COLUMN, SPLIT_COLUMN
from perceptron.data.view import DatasetView
from perceptron.evaluation.predictions import ROW, eval_frame, load_predictions, proba_columns

MIN_SUPPORT = 10
GAP = 0.05
MAX_SAMPLES = 200
MAX_CATEGORIES = 30
SAMPLE_COLUMNS = 12
_INTERNAL = {SPLIT_COLUMN, FOLD_COLUMN, ROW, "corrupt"}


class Slice(BaseModel):
    column: str
    value: str
    support: int
    metric: float
    gap: float = Field(description="Cuánto peor que el total (en la dirección del error)")


class Confusion(BaseModel):
    actual: str
    predicted: str
    count: int


class ErrorSample(BaseModel):
    row: int
    actual: Any
    predicted: Any
    confidence: float | None = None
    label_issue: bool = False
    features: dict[str, Any] = Field(default_factory=dict)


class ErrorAnalysis(BaseModel):
    task: str
    metric: str
    overall: float
    num_samples: int
    num_errors: int
    slices: list[Slice]
    confusions: list[Confusion]
    label_issues: int
    samples: list[ErrorSample]


def _json_value(v: Any) -> Any:
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _slice_columns(df: pl.DataFrame, target: str | None) -> list[str]:
    skip = _INTERNAL | {target, "path"}
    return [c for c in df.columns if c not in skip]


def group_values(s: pl.Series) -> pl.Series | None:
    """Grupos de una columna: sus valores (≤ 30) o cuartiles si es numérica continua."""
    if s.dtype.is_numeric() and s.n_unique() > MAX_CATEGORIES:
        edges = sorted({float(s.quantile(q) or 0.0) for q in (0.25, 0.5, 0.75)})
        labels = [f"≤ {edges[0]:.4g}"]
        labels += [f"{lo:.4g} – {hi:.4g}" for lo, hi in pairwise(edges)]
        labels.append(f"> {edges[-1]:.4g}")
        return s.cut(edges, labels=labels).cast(pl.String)
    if s.n_unique() <= MAX_CATEGORIES:
        return s.cast(pl.String).fill_null("∅")
    return None


def analyze_errors(eval_dir: Path, dataset_dir: Path, target: str | None) -> ErrorAnalysis:
    preds = load_predictions(eval_dir)
    df = eval_frame(DatasetView(dataset_dir)).join(preds, on=ROW, how="inner")
    regression = df["y_true"].dtype.is_float()
    if regression:
        err = (df["y_pred"] - df["y_true"]).abs()
        df = df.with_columns(err.alias("__err__"))
        metric, overall = "mae", float(err.mean() or 0.0)
        wrong = df.filter(pl.col("__err__") > overall)
    else:
        df = df.with_columns((pl.col("y_true") == pl.col("y_pred")).alias("__ok__"))
        metric, overall = "accuracy", float(df["__ok__"].mean() or 0.0)
        wrong = df.filter(~pl.col("__ok__"))

    # Slices de bajo rendimiento.
    min_support = max(MIN_SUPPORT, int(0.02 * df.height))
    columns = _slice_columns(df, target) if "path" not in df.columns else []
    candidates: list[tuple[str, pl.Series]] = []
    for col in columns:
        g = group_values(df[col])
        if g is not None:
            candidates.append((col, g))
    if not regression:
        candidates.append(("clase real", df["y_true"].cast(pl.String)))
    slices: list[Slice] = []
    for col, groups in candidates:
        agg = (
            df.with_columns(groups.alias("__g__"))
            .group_by("__g__")
            .agg(
                pl.len().alias("n"),
                (pl.col("__err__").mean() if regression else pl.col("__ok__").mean()).alias("m"),
            )
            .filter(pl.col("n") >= min_support)
        )
        for g, n, m in agg.iter_rows():
            gap = (m - overall) if regression else (overall - m)
            if gap > GAP * (overall if regression else 1.0):
                slices.append(
                    Slice(column=col, value=str(g), support=n, metric=float(m), gap=float(gap))
                )
    slices.sort(key=lambda s: s.gap, reverse=True)

    # Confusiones y posibles errores de etiqueta.
    confusions: list[Confusion] = []
    issues = np.zeros(df.height, dtype=bool)
    probs = proba_columns(df)
    if not regression:
        pairs = wrong.group_by("y_true", "y_pred").len().sort("len", descending=True).head(10)
        confusions = [
            Confusion(actual=str(a), predicted=str(p), count=n) for a, p, n in pairs.iter_rows()
        ]
        if probs:
            p = df.select(list(probs.values())).to_numpy()
            classes = list(probs)
            labels = np.array(
                [classes.index(y) if y in classes else -1 for y in df["y_true"].to_list()]
            )
            thresholds = np.array(
                [
                    p[labels == k, k].mean() if (labels == k).any() else 1.0
                    for k in range(len(classes))
                ]
            )
            own = np.where(labels >= 0, p[np.arange(len(p)), np.clip(labels, 0, None)], 1.0)
            other = (p >= thresholds) & (np.arange(len(classes)) != labels[:, None])
            issues = (
                (labels >= 0) & (own < thresholds[np.clip(labels, 0, None)]) & other.any(axis=1)
            )

    # Explorador de mal predichos (los más confiados primero: los más llamativos).
    df = df.with_columns(pl.Series("__issue__", issues))
    wrong_rows = (
        df.filter(pl.col("__err__") > overall) if regression else df.filter(~pl.col("__ok__"))
    )
    order = "__err__" if regression else ("confidence" if "confidence" in df.columns else ROW)
    wrong_rows = wrong_rows.sort(order, descending=True).head(MAX_SAMPLES)
    feature_cols = ["path"] if "path" in df.columns else _slice_columns(df, target)[:SAMPLE_COLUMNS]
    samples = [
        ErrorSample(
            row=int(r[ROW]),
            actual=_json_value(r["y_true"]),
            predicted=_json_value(r["y_pred"]),
            confidence=float(r["confidence"]) if r.get("confidence") is not None else None,
            label_issue=bool(r["__issue__"]),
            features={c: _json_value(r[c]) for c in feature_cols},
        )
        for r in wrong_rows.iter_rows(named=True)
    ]
    return ErrorAnalysis(
        task="regression" if regression else "classification",
        metric=metric,
        overall=overall,
        num_samples=df.height,
        num_errors=wrong.height,
        slices=slices[:10],
        confusions=confusions,
        label_issues=int(issues.sum()),
        samples=samples,
    )
