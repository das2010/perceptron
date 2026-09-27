"""Series temporales (RF-ING-08 temporal, RF-PRF-04).

Una tabla de series tiene una columna de tiempo, opcionalmente un identificador de
serie, un target numérico (forecasting) o una etiqueta de anomalía (anomalías) y
covariables numéricas. La configuración se guarda en `dataset.json`.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from perceptron.data.schema import SemanticType, TableSchema
from perceptron.data.splits import SPLIT_COLUMN, TEST, TRAIN, VAL
from perceptron.domain.enums import TaskType

SERIES_KEY = "__series__"
MIN_POINTS_PER_SERIES = 30
ANOMALY_MAX_POSITIVE_RATE = 0.2
_ANOMALY_NAMES = ("anomal", "falla", "fault", "outlier", "evento")

# Estacionalidades candidatas según la frecuencia (en segundos).
_SEASON_CANDIDATES = {
    60: [60, 1440],  # minutos: hora, día
    3600: [24, 168],  # horas: día, semana
    86400: [7, 30, 365],  # días
    604800: [52, 4, 13],  # semanas
    2592000: [12, 3],  # meses aprox.
}
_DEFAULT_HORIZON = {60: 60, 3600: 24, 86400: 14, 604800: 8, 2592000: 6}


class SeriesConfig(BaseModel):
    time_column: str
    series_id: str | None = None
    target: str
    task: TaskType = TaskType.FORECASTING
    covariates: list[str] = Field(default_factory=list)
    horizon: int = Field(default=8, ge=1)
    lookback: int = Field(default=32, ge=2)
    freq_seconds: float | None = None
    season: int | None = None


def _series_key(df: pl.DataFrame, cfg: SeriesConfig) -> pl.Expr:
    return pl.col(cfg.series_id).cast(pl.String) if cfg.series_id else pl.lit("_")


def with_series_key(df: pl.DataFrame, cfg: SeriesConfig) -> pl.DataFrame:
    return df.with_columns(_series_key(df, cfg).alias(SERIES_KEY)).sort(SERIES_KEY, cfg.time_column)


def _freq_seconds(times: pl.Series) -> float | None:
    t = times.cast(pl.Datetime).sort().drop_nulls()
    if t.len() < 3:
        return None
    diffs = t.diff().drop_nulls().dt.total_seconds()
    med = diffs.median()
    return float(med) if med else None


def _closest_freq(freq: float | None) -> int | None:
    if not freq:
        return None
    return min(_SEASON_CANDIDATES, key=lambda f: abs(math.log(freq / f)))


def autocorr(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(x) <= lag + 2:
        return 0.0
    a, b = x[:-lag] - x[:-lag].mean(), x[lag:] - x[lag:].mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den else 0.0


def detect_season(x: np.ndarray, freq: float | None, max_fraction: float = 0.5) -> int | None:
    """Mejor estacionalidad candidata con autocorrelación > 0,3 que entre en la serie."""
    base = _closest_freq(freq)
    candidates = _SEASON_CANDIDATES.get(base, []) if base else []
    best, best_ac = None, 0.3
    for lag in candidates:
        if lag <= len(x) * max_fraction:
            ac = autocorr(x - np.linspace(x[0], x[-1], len(x)), lag)  # sin tendencia
            if ac > best_ac:
                best, best_ac = lag, ac
    return best


def detect_series(schema: TableSchema, df: pl.DataFrame) -> SeriesConfig | None:
    """¿Es una tabla de series? (fecha regular + target numérico o etiqueta de anomalía)."""
    dates = [c.name for c in schema.columns if c.semantic is SemanticType.DATETIME]
    if not dates or not schema.target:
        return None
    time_col = dates[0]
    target = schema.column(schema.target)
    ids = [
        c.name
        for c in schema.columns
        if c.semantic is SemanticType.CATEGORICAL
        and c.name != schema.target
        and 1 < c.n_unique <= 1000
    ]
    series_id = ids[0] if ids else None
    groups = df.group_by(series_id).len()["len"] if series_id else pl.Series([df.height])
    if groups.min() < MIN_POINTS_PER_SERIES:  # type: ignore[operator]
        return None
    positive_rate = (
        float(df[schema.target].cast(pl.Float64).mean() or 0)
        if target.semantic is SemanticType.BOOLEAN
        else None
    )
    is_anomaly = (
        positive_rate is not None
        and positive_rate < ANOMALY_MAX_POSITIVE_RATE
        and any(k in schema.target.lower() for k in _ANOMALY_NAMES)
    )
    if not is_anomaly and target.semantic is not SemanticType.NUMERIC:
        return None
    freq = _freq_seconds(df[time_col])
    covariates = [
        c.name
        for c in schema.columns
        if c.semantic is SemanticType.NUMERIC and c.name not in (schema.target,)
    ]
    base = _closest_freq(freq)
    if is_anomaly:
        return SeriesConfig(
            time_column=time_col,
            series_id=series_id,
            target=schema.target,
            task=TaskType.ANOMALY_DETECTION,
            covariates=covariates,
            horizon=1,
            lookback=20,
            freq_seconds=freq,
        )
    horizon = _DEFAULT_HORIZON.get(base, 8) if base else 8
    shortest = int(groups.min())  # type: ignore[arg-type]
    horizon = max(1, min(horizon, shortest // 8))
    return SeriesConfig(
        time_column=time_col,
        series_id=series_id,
        target=schema.target,
        task=TaskType.FORECASTING,
        covariates=covariates,
        horizon=horizon,
        lookback=max(2 * horizon, min(4 * horizon, shortest // 3)),
        freq_seconds=freq,
    )


def temporal_split_per_series(
    df: pl.DataFrame, cfg: SeriesConfig, fraction: float = 0.15
) -> pl.DataFrame:
    """Split temporal dentro de cada serie: el final va a test y lo anterior a val.

    Forecasting: cada bloque tiene al menos 2·horizonte puntos (backtesting con varias
    ventanas). Anomalías: 70/15/15 contiguo.
    """
    df = with_series_key(df, cfg)
    labels: list[str] = []
    for _, part in df.group_by(SERIES_KEY, maintain_order=True):
        n = part.height
        block = max(round(n * fraction), 2 * cfg.horizon if cfg.task is TaskType.FORECASTING else 1)
        block = min(block, max((n - cfg.lookback) // 3, 1))
        labels += [TRAIN] * (n - 2 * block) + [VAL] * block + [TEST] * block
    return df.with_columns(pl.Series(SPLIT_COLUMN, labels, dtype=pl.String))


class SeriesProfile(BaseModel):
    """RF-PRF-04 (solo agregados)."""

    num_series: int
    points_min: int
    points_median: float
    freq_seconds: float | None
    gaps: int = Field(description="Saltos mayores a 1,5× la frecuencia")
    season: int | None
    season_strength: float | None
    trend_slope_per_step: float | None
    feasible_horizon: int
    task: TaskType
    horizon: int
    lookback: int
    config: SeriesConfig = Field(description="Columnas y parámetros (sin valores)")


def profile_series(df: pl.DataFrame, cfg: SeriesConfig) -> SeriesProfile:
    from perceptron.data.profiling.card import sig

    df = with_series_key(df, cfg)
    sizes = df.group_by(SERIES_KEY).len()["len"]
    gaps = 0
    seasons: list[int] = []
    strengths: list[float] = []
    slopes: list[float] = []
    for _, part in df.group_by(SERIES_KEY, maintain_order=True):
        t = part[cfg.time_column].cast(pl.Datetime)
        if cfg.freq_seconds:
            d = t.diff().drop_nulls().dt.total_seconds()
            gaps += int((d > 1.5 * cfg.freq_seconds).sum())
        y = part[cfg.target].cast(pl.Float64).fill_null(strategy="forward").fill_null(0).to_numpy()
        s = detect_season(y, cfg.freq_seconds)
        if s:
            seasons.append(s)
            strengths.append(autocorr(y - np.linspace(y[0], y[-1], len(y)), s))
        if len(y) > 2:
            slopes.append(float(np.polyfit(np.arange(len(y)), y, 1)[0]))
    season = max(set(seasons), key=seasons.count) if seasons else None
    cfg = cfg.model_copy(update={"season": season})
    return SeriesProfile(
        num_series=int(sizes.len()),
        points_min=int(sizes.min() or 0),  # type: ignore[arg-type]
        points_median=float(sizes.median() or 0),  # type: ignore[arg-type]
        freq_seconds=cfg.freq_seconds,
        gaps=gaps,
        season=season,
        season_strength=sig(float(np.mean(strengths)), 3) if strengths else None,
        trend_slope_per_step=sig(float(np.median(slopes)), 3) if slopes else None,
        feasible_horizon=max(1, int(sizes.min() or 0) // 5),  # type: ignore[arg-type]
        task=cfg.task,
        horizon=cfg.horizon,
        lookback=cfg.lookback,
        config=cfg,
    )
