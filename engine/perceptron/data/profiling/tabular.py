"""Estadísticas por columna para datos tabulares (RF-PRF-01)."""

from __future__ import annotations

import numpy as np
import polars as pl

from perceptron.data.profiling.card import (
    K_ANONYMITY,
    MAX_CATEGORIES,
    CategoricalStats,
    CategoryCount,
    ColumnProfile,
    NumericStats,
    TargetProfile,
    TextStats,
    sig,
)
from perceptron.data.schema import ColumnSchema, SemanticType, TableSchema, series_mean
from perceptron.domain.enums import TaskType

HIST_BINS = 20
_QUANTILES = {"p05": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95}


def numeric_stats(s: pl.Series) -> NumericStats | None:
    x = s.drop_nulls().cast(pl.Float64)
    x = x.filter(x.is_finite())
    if x.is_empty():
        return None
    arr = x.to_numpy()
    q = {k: float(np.quantile(arr, v)) for k, v in _QUANTILES.items()}
    lo, hi = np.quantile(arr, [0.01, 0.99])
    hist = np.histogram(np.clip(arr, lo, hi), bins=HIST_BINS, range=(lo, hi) if hi > lo else None)[
        0
    ]
    iqr = q["p75"] - q["p25"]
    outliers = (
        ((arr < q["p25"] - 1.5 * iqr) | (arr > q["p75"] + 1.5 * iqr)).mean() if iqr > 0 else 0.0
    )
    std = float(arr.std(ddof=1)) if len(arr) > 1 else None
    skew = float(((arr - arr.mean()) ** 3).mean() / std**3) if std else None
    return NumericStats(
        mean=sig(float(arr.mean())),
        std=sig(std),
        quantiles={k: sig(v) for k, v in q.items()},
        histogram=[int(c) for c in hist],
        outlier_fraction=round(float(outliers), 4),
        skew=sig(skew, 3),
    )


def categorical_stats(s: pl.Series) -> CategoricalStats:
    counts = s.drop_nulls().cast(pl.String).value_counts(sort=True, name="n")
    total = int(counts["n"].sum())
    top = [
        CategoryCount(value=v, count=int(n))
        for v, n in counts.head(MAX_CATEGORIES).iter_rows()
        if n >= K_ANONYMITY
    ]
    return CategoricalStats(top=top, other_count=total - sum(c.count for c in top))


def text_stats(s: pl.Series) -> TextStats:
    t = s.drop_nulls().cast(pl.String)
    if t.is_empty():
        return TextStats(mean_length=None, p95_length=None, mean_words=None)
    lengths = t.str.len_chars()
    words = t.str.split(" ").list.len()
    return TextStats(
        mean_length=sig(series_mean(lengths)),
        p95_length=sig(lengths.quantile(0.95)),
        mean_words=sig(series_mean(words)),
    )


def _cramers_v(a: pl.Series, b: pl.Series) -> float | None:
    df = pl.DataFrame({"a": a.cast(pl.String), "b": b.cast(pl.String)}).drop_nulls()
    if df.height < 2:
        return None
    table = df.group_by("a", "b").len().pivot(on="b", index="a", values="len").fill_null(0)
    obs = table.drop("a").to_numpy().astype(float)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return None
    n = obs.sum()
    expected = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / n
    chi2 = ((obs - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    k = min(obs.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else None


def _pearson(a: pl.Series, b: pl.Series) -> float | None:
    df = pl.DataFrame({"a": a.cast(pl.Float64), "b": b.cast(pl.Float64)}).drop_nulls()
    if df.height < 3 or df["a"].std() == 0 or df["b"].std() == 0:
        return None
    return float(abs(np.corrcoef(df["a"].to_numpy(), df["b"].to_numpy())[0, 1]))


def association(
    col: ColumnSchema, s: pl.Series, target: pl.Series, target_sem: SemanticType
) -> float | None:
    numeric_like = (SemanticType.NUMERIC, SemanticType.BOOLEAN)
    if (
        col.semantic in numeric_like
        and target_sem in numeric_like
        and s.dtype.is_numeric()
        and target.dtype.is_numeric()
    ):
        return _pearson(s, target)
    if col.semantic in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN) or target_sem in (
        SemanticType.CATEGORICAL,
        SemanticType.BOOLEAN,
    ):
        if col.semantic is SemanticType.NUMERIC:
            s = s.qcut(10, allow_duplicates=True)
        tgt = (
            target.qcut(10, allow_duplicates=True) if target_sem is SemanticType.NUMERIC else target
        )
        return _cramers_v(s, tgt)
    return None


def task_hint(target_sem: SemanticType, n_unique: int) -> TaskType:
    if target_sem is SemanticType.NUMERIC and n_unique > 20:
        return TaskType.REGRESSION
    return TaskType.CLASSIFICATION


def profile_target(schema: TableSchema, df: pl.DataFrame) -> TargetProfile | None:
    if not schema.target:
        return None
    col = schema.column(schema.target)
    s = df[schema.target]
    hint = task_hint(col.semantic, s.n_unique())
    if hint is TaskType.CLASSIFICATION:
        counts = s.drop_nulls().cast(pl.String).value_counts(sort=True, name="n")
        classes = [CategoryCount(value=v, count=int(n)) for v, n in counts.iter_rows()]
        ratio = classes[-1].count / classes[0].count if classes else None
        return TargetProfile(
            name=col.name,
            semantic=col.semantic,
            task_hint=hint,
            classes=classes,
            imbalance_ratio=round(ratio, 4) if ratio is not None else None,
        )
    return TargetProfile(
        name=col.name, semantic=col.semantic, task_hint=hint, numeric=numeric_stats(s)
    )


def profile_columns(schema: TableSchema, df: pl.DataFrame) -> list[ColumnProfile]:
    target = df[schema.target] if schema.target else None
    target_sem = schema.column(schema.target).semantic if schema.target else None
    out: list[ColumnProfile] = []
    for col in schema.columns:
        if col.name == schema.target:
            continue
        s = df[col.name]
        prof = ColumnProfile(
            name=col.name,
            semantic=col.semantic,
            dtype=col.dtype,
            null_fraction=round(s.null_count() / max(df.height, 1), 4),
            n_unique=s.n_unique(),
        )
        if col.semantic is SemanticType.NUMERIC:
            prof.numeric = numeric_stats(s)
        elif col.semantic in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN):
            prof.categorical = categorical_stats(s)
        elif col.semantic is SemanticType.TEXT:
            prof.text = text_stats(s)
        if (
            target is not None
            and target_sem is not None
            and col.semantic
            not in (
                SemanticType.ID,
                SemanticType.TEXT,
                SemanticType.FILEPATH,
                SemanticType.DATETIME,
            )
        ):
            assoc = association(col, s, target, target_sem)
            prof.target_association = sig(assoc, 3) if assoc is not None else None
        out.append(prof)
    return out
