"""Preparación de series temporales (RF-PIP-03 series): escalado por serie ajustado
con train, features de calendario, ventaneo lookback → horizonte y baseline
seasonal-naive para comparar (y escala de MASE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from perceptron.data.series import SERIES_KEY, SeriesConfig, with_series_key
from perceptron.data.splits import SPLIT_COLUMN, TRAIN
from perceptron.domain.enums import TaskType

DAY, WEEK = 86_400, 604_800


def calendar_columns(cfg: SeriesConfig) -> list[str]:
    f = cfg.freq_seconds or DAY
    cols: list[str] = []
    if f < DAY:
        cols += ["cal.hour_sin", "cal.hour_cos"]
    if f <= DAY:
        cols += ["cal.dow_sin", "cal.dow_cos"]
    if f >= DAY:
        cols += ["cal.year_sin", "cal.year_cos"]
    return cols


def _calendar(df: pl.DataFrame, cfg: SeriesConfig) -> pl.DataFrame:
    t = pl.col(cfg.time_column).cast(pl.Datetime)
    two_pi = 2 * math.pi
    exprs = {
        "cal.hour_sin": (two_pi * (t.dt.hour() * 60 + t.dt.minute()) / 1440).sin(),
        "cal.hour_cos": (two_pi * (t.dt.hour() * 60 + t.dt.minute()) / 1440).cos(),
        "cal.dow_sin": (two_pi * t.dt.weekday() / 7).sin(),
        "cal.dow_cos": (two_pi * t.dt.weekday() / 7).cos(),
        "cal.year_sin": (two_pi * t.dt.ordinal_day() / 365.25).sin(),
        "cal.year_cos": (two_pi * t.dt.ordinal_day() / 365.25).cos(),
    }
    return df.with_columns(exprs[c].alias(c) for c in calendar_columns(cfg))


def value_channels(cfg: SeriesConfig) -> list[str]:
    """Canales escalados: target (solo forecasting) + covariables."""
    base = [cfg.target] if cfg.task is TaskType.FORECASTING else []
    return base + [c for c in cfg.covariates if c != cfg.target]


def channels(cfg: SeriesConfig, calendar: bool) -> list[str]:
    return value_channels(cfg) + (calendar_columns(cfg) if calendar else [])


def fit_series(cfg: SeriesConfig, train: pl.DataFrame) -> dict[str, Any]:
    """Media/desvío por serie y canal (solo train) + escala de MASE por serie."""
    df = with_series_key(train, cfg)
    cols = value_channels(cfg)
    stats: dict[str, dict[str, list[float]]] = {}
    mase: dict[str, float] = {}
    m = cfg.season or 1
    for (key,), part in df.group_by(SERIES_KEY, maintain_order=True):
        values = part.select(
            pl.col(c).cast(pl.Float64).forward_fill().fill_null(0.0) for c in cols
        ).to_numpy()
        mean = values.mean(0) if len(values) else np.zeros(len(cols))
        std = values.std(0) if len(values) > 1 else np.ones(len(cols))
        stats[str(key)] = {"mean": mean.tolist(), "std": np.where(std > 1e-8, std, 1.0).tolist()}
        if cfg.task is TaskType.FORECASTING:
            y = values[:, 0]
            lag = m if len(y) > 2 * m else 1
            diffs = np.abs(y[lag:] - y[:-lag]) if len(y) > lag else np.array([1.0])
            mase[str(key)] = float(diffs.mean()) or 1.0
    global_mean = np.mean([s["mean"] for s in stats.values()], axis=0).tolist() if stats else []
    global_std = np.mean([s["std"] for s in stats.values()], axis=0).tolist() if stats else []
    return {"stats": stats, "mase_scale": mase, "global": {"mean": global_mean, "std": global_std}}


@dataclass
class ForecastWindows:
    x: np.ndarray  # [N, L, C] float32 (escalado)
    y: np.ndarray  # [N, H] float32 (escalado)
    mean: np.ndarray  # [N] media del target de la serie (para desescalar)
    std: np.ndarray  # [N]
    naive: np.ndarray  # [N, H] seasonal-naive en unidades originales
    mase_scale: np.ndarray  # [N]
    series: list[str]


@dataclass
class AnomalyWindows:
    x: np.ndarray  # [N, L, C]
    label: np.ndarray  # [N] etiqueta del último punto de la ventana
    normal: np.ndarray  # [N] bool: ninguna anomalía dentro de la ventana
    series: list[str]


def _prepared(cfg: SeriesConfig, state: dict[str, Any], df: pl.DataFrame, calendar: bool):  # type: ignore[no-untyped-def]
    df = _calendar(with_series_key(df, cfg), cfg) if calendar else with_series_key(df, cfg)
    vcols = value_channels(cfg)
    ccols = calendar_columns(cfg) if calendar else []
    for (key,), part in df.group_by(SERIES_KEY, maintain_order=True):
        st = state["stats"].get(str(key), state["global"])
        mean, std = np.asarray(st["mean"]), np.asarray(st["std"])
        values = part.select(
            pl.col(c).cast(pl.Float64).forward_fill().fill_null(0.0) for c in vcols
        ).to_numpy()
        scaled = (values - mean) / std if len(vcols) else np.zeros((part.height, 0))
        cal = part.select(ccols).to_numpy() if ccols else np.zeros((part.height, 0))
        feats = np.concatenate([scaled, cal], axis=1).astype(np.float32)
        yield str(key), part, values, feats, mean, std


def forecast_windows(
    cfg: SeriesConfig, state: dict[str, Any], df: pl.DataFrame, split: str, calendar: bool = True
) -> ForecastWindows:
    """Ventanas cuyo horizonte cae entero en `split` (backtesting con paso 1).

    La historia de la ventana puede venir de splits anteriores (es información del pasado).
    """
    L, H = cfg.lookback, cfg.horizon  # noqa: N806 - notación usual (lookback, horizonte)
    m = cfg.season or 1
    xs, ys, means, stds, naives, scales, keys = [], [], [], [], [], [], []
    for key, part, values, feats, mean, std in _prepared(cfg, state, df, calendar):
        splits = part[SPLIT_COLUMN].to_numpy()
        y_raw = values[:, 0]
        for start in range(L, part.height - H + 1):
            if not (splits[start : start + H] == split).all():
                continue
            xs.append(feats[start - L : start])
            ys.append(((y_raw[start : start + H] - mean[0]) / std[0]).astype(np.float32))
            hist = y_raw[start - L : start]
            naive = np.array([hist[-m + (h % m)] if m <= L else hist[-1] for h in range(H)])
            naives.append(naive)
            means.append(mean[0])
            stds.append(std[0])
            scales.append(state["mase_scale"].get(key, 1.0))
            keys.append(key)
    c = len(channels(cfg, calendar))
    return ForecastWindows(
        x=np.asarray(xs, dtype=np.float32).reshape(-1, L, c),
        y=np.asarray(ys, dtype=np.float32).reshape(-1, H),
        mean=np.asarray(means, dtype=np.float32),
        std=np.asarray(stds, dtype=np.float32),
        naive=np.asarray(naives, dtype=np.float64).reshape(-1, H),
        mase_scale=np.asarray(scales, dtype=np.float64),
        series=keys,
    )


def anomaly_windows(
    cfg: SeriesConfig, state: dict[str, Any], df: pl.DataFrame, split: str, calendar: bool = False
) -> AnomalyWindows:
    """Una ventana por punto de `split` (la ventana termina en ese punto)."""
    L = cfg.lookback  # noqa: N806 - notación usual
    xs, labels, normal, keys = [], [], [], []
    for key, part, _values, feats, _mean, _std in _prepared(cfg, state, df, calendar):
        splits = part[SPLIT_COLUMN].to_numpy()
        lab = part[cfg.target].cast(pl.Int64).fill_null(0).to_numpy()
        for end in range(L - 1, part.height):
            if splits[end] != split:
                continue
            xs.append(feats[end - L + 1 : end + 1])
            labels.append(int(lab[end]))
            normal.append(bool(lab[end - L + 1 : end + 1].max() == 0))
            keys.append(key)
    c = len(channels(cfg, calendar))
    return AnomalyWindows(
        x=np.asarray(xs, dtype=np.float32).reshape(-1, L, c),
        label=np.asarray(labels, dtype=np.int64),
        normal=np.asarray(normal, dtype=bool),
        series=keys,
    )


def train_rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col(SPLIT_COLUMN) == TRAIN)
