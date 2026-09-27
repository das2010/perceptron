"""Pasos de preparación tabular con `fit` (solo train) / `transform` (RF-PIP-03, RF-PIP-04).

Cada paso es declarativo (`kind`, `columns`, `params`) y su estado ajustado es
JSON serializable, para empaquetarlo junto al modelo en los exports.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import polars as pl
from pydantic import BaseModel, Field

State = dict[str, Any]

UNKNOWN = "__unknown__"
MISSING = "__missing__"


class StepSpec(BaseModel):
    id: str
    kind: str
    columns: list[str]
    params: dict[str, Any] = Field(default_factory=dict)


class Step:
    kind: ClassVar[str]

    def __init__(self, spec: StepSpec) -> None:
        self.spec = spec

    @property
    def columns(self) -> list[str]:
        return self.spec.columns

    def fit(self, df: pl.DataFrame) -> State:
        return {}

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        raise NotImplementedError

    def output_columns(self, state: State) -> dict[str, str]:
        """Columnas nuevas o modificadas → 'numeric' | 'categorical'. Vacío = sin cambio de tipo."""
        return {}


STEPS: dict[str, type[Step]] = {}


def register(cls: type[Step]) -> type[Step]:
    STEPS[cls.kind] = cls
    return cls


def build_step(spec: StepSpec) -> Step:
    try:
        return STEPS[spec.kind](spec)
    except KeyError:
        raise ValueError(f"paso desconocido: {spec.kind}") from None


def _to_float(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns(pl.col(c).cast(pl.Float64) for c in cols)


@register
class Drop(Step):
    kind = "drop"

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        return df.drop([c for c in self.columns if c in df.columns])


@register
class ImputeNumeric(Step):
    """strategy: mean | median | constant (params.value)."""

    kind = "impute_numeric"

    def fit(self, df: pl.DataFrame) -> State:
        strategy = self.spec.params.get("strategy", "median")
        fill: dict[str, float] = {}
        for c in self.columns:
            s = df[c].cast(pl.Float64).drop_nulls()
            if strategy == "constant":
                v = float(self.spec.params.get("value", 0.0))
            elif strategy == "mean":
                v = float(s.mean()) if s.len() else 0.0  # type: ignore[arg-type]
            else:
                v = float(s.median()) if s.len() else 0.0  # type: ignore[arg-type]
            fill[c] = 0.0 if math.isnan(v) else v
        return {"fill": fill}

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        df = _to_float(df, self.columns)
        return df.with_columns(pl.col(c).fill_null(v).fill_nan(v) for c, v in state["fill"].items())


@register
class ImputeCategorical(Step):
    """strategy: mode | constant (usa `__missing__`)."""

    kind = "impute_categorical"

    def fit(self, df: pl.DataFrame) -> State:
        fill: dict[str, str] = {}
        for c in self.columns:
            if self.spec.params.get("strategy", "constant") == "mode":
                mode = df[c].cast(pl.String).drop_nulls().mode()
                fill[c] = str(mode.sort()[0]) if mode.len() else MISSING
            else:
                fill[c] = MISSING
        return {"fill": fill}

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        return df.with_columns(
            pl.col(c).cast(pl.String).fill_null(v) for c, v in state["fill"].items()
        )


@register
class OrdinalEncode(Step):
    """Categoría → índice entero (0 = desconocida). Entrada de las embeddings categóricas."""

    kind = "ordinal"

    def fit(self, df: pl.DataFrame) -> State:
        min_freq = int(self.spec.params.get("min_frequency", 1))
        vocab: dict[str, list[str]] = {}
        for c in self.columns:
            vc = df[c].cast(pl.String).drop_nulls().value_counts(name="n")
            vocab[c] = sorted(v for v, n in vc.iter_rows() if n >= min_freq)
        return {"vocab": vocab}

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        exprs = []
        for c, values in state["vocab"].items():
            mapping = {v: i + 1 for i, v in enumerate(values)}
            exprs.append(
                pl.col(c).cast(pl.String).replace_strict(mapping, default=0, return_dtype=pl.Int64)
            )
        return df.with_columns(exprs)

    def output_columns(self, state: State) -> dict[str, str]:
        return dict.fromkeys(state["vocab"], "categorical")

    @staticmethod
    def cardinalities(state: State) -> dict[str, int]:
        return {c: len(v) + 1 for c, v in state["vocab"].items()}


@register
class OneHot(Step):
    kind = "one_hot"

    def fit(self, df: pl.DataFrame) -> State:
        max_cats = int(self.spec.params.get("max_categories", 30))
        cats: dict[str, list[str]] = {}
        for c in self.columns:
            vc = df[c].cast(pl.String).drop_nulls().value_counts(sort=True, name="n")
            cats[c] = sorted(v for v, _ in vc.head(max_cats).iter_rows())
        return {"categories": cats}

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        exprs = [
            (pl.col(c).cast(pl.String) == v).fill_null(False).cast(pl.Float64).alias(f"{c}={v}")
            for c, values in state["categories"].items()
            for v in values
        ]
        return df.with_columns(exprs).drop(list(state["categories"]))

    def output_columns(self, state: State) -> dict[str, str]:
        return {f"{c}={v}": "numeric" for c, values in state["categories"].items() for v in values}


@register
class Scale(Step):
    """method: standard | robust | minmax | quantile (aprox. por rangos percentiles)."""

    kind = "scale"

    def fit(self, df: pl.DataFrame) -> State:
        method = self.spec.params.get("method", "standard")
        stats: dict[str, list[float]] = {}
        for c in self.columns:
            s = df[c].cast(pl.Float64).drop_nulls()
            if s.is_empty():
                stats[c] = [0.0, 1.0]
                continue
            if method == "robust":
                q1, med, q3 = (float(s.quantile(q) or 0.0) for q in (0.25, 0.5, 0.75))
                center, scale = med, q3 - q1
            elif method == "minmax":
                center, scale = float(s.min()), float(s.max()) - float(s.min())  # type: ignore[arg-type]
            elif method == "quantile":
                qs = [float(s.quantile(i / 100) or 0.0) for i in range(0, 101)]
                stats[c] = qs
                continue
            else:
                center, scale = float(s.mean()), float(s.std() or 0.0)  # type: ignore[arg-type]
            stats[c] = [center, scale if scale and math.isfinite(scale) else 1.0]
        return {"method": method, "stats": stats}

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        df = _to_float(df, self.columns)
        if state["method"] == "quantile":
            exprs = [
                pl.col(c).map_batches(_quantile_mapper(qs), return_dtype=pl.Float64).alias(c)
                for c, qs in state["stats"].items()
            ]
            return df.with_columns(exprs)
        return df.with_columns(
            ((pl.col(c) - m) / s).alias(c) for c, (m, s) in state["stats"].items()
        )


def _quantile_mapper(qs: list[float]) -> Callable[[pl.Series], pl.Series]:
    import numpy as np

    grid = np.linspace(0, 1, len(qs))

    def f(s: pl.Series) -> pl.Series:
        return pl.Series(np.interp(s.to_numpy(), qs, grid))

    return f


@register
class Log1p(Step):
    """log(1 + x) para variables sesgadas no negativas (se recortan negativos a 0)."""

    kind = "log1p"

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        df = _to_float(df, self.columns)
        return df.with_columns(pl.col(c).clip(lower_bound=0).log1p() for c in self.columns)


@register
class DateFeatures(Step):
    """Fecha → año, mes, día de la semana, día del año (numéricas). Se elimina la original."""

    kind = "date_features"
    PARTS: ClassVar[tuple[str, ...]] = ("year", "month", "weekday", "ordinal_day")

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        exprs = []
        for c in self.columns:
            d = pl.col(c).cast(pl.Datetime)
            exprs += [
                d.dt.year().cast(pl.Float64).alias(f"{c}.year"),
                d.dt.month().cast(pl.Float64).alias(f"{c}.month"),
                d.dt.weekday().cast(pl.Float64).alias(f"{c}.weekday"),
                d.dt.ordinal_day().cast(pl.Float64).alias(f"{c}.ordinal_day"),
            ]
        return df.with_columns(exprs).drop(self.columns)

    def output_columns(self, state: State) -> dict[str, str]:
        return {f"{c}.{p}": "numeric" for c in self.columns for p in self.PARTS}


@register
class ToNumeric(Step):
    """Booleanos/strings "0"/"1"/"true" → 0.0/1.0."""

    kind = "to_numeric"

    def fit(self, df: pl.DataFrame) -> State:
        mapping: dict[str, dict[str, float]] = {}
        for c in self.columns:
            values = sorted(df[c].cast(pl.String).drop_nulls().unique().to_list())
            truthy = {"1", "true", "si", "sí", "yes", "y", "t"}
            mapping[c] = {
                v: (1.0 if v.lower() in truthy else float(i if len(values) == 2 else 0))
                for i, v in enumerate(values)
            }
        return {"mapping": mapping}

    def transform(self, df: pl.DataFrame, state: State) -> pl.DataFrame:
        return df.with_columns(
            pl.col(c).cast(pl.String).replace_strict(m, default=None, return_dtype=pl.Float64)
            for c, m in state["mapping"].items()
        )

    def output_columns(self, state: State) -> dict[str, str]:
        return dict.fromkeys(state["mapping"], "numeric")
