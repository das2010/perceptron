"""Particionado train/val/test (RF-ING-08).

El resultado es una columna `__split__` alineada con las filas. El test queda
sellado: solo la evaluación final puede leerlo (ver `data.view`).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, model_validator
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold, train_test_split

from perceptron.core.errors import ValidationError
from perceptron.domain.enums import SplitStrategy

SPLIT_COLUMN = "__split__"
FOLD_COLUMN = "__fold__"
TRAIN, VAL, TEST = "train", "val", "test"


class SplitRequest(BaseModel):
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    val_fraction: float = Field(default=0.15, ge=0, lt=1)
    test_fraction: float = Field(default=0.15, ge=0, lt=1)
    seed: int = 42
    folds: int | None = Field(default=None, ge=2, le=20)
    group_column: str | None = None
    time_column: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SplitRequest:
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction debe ser < 1")
        if self.strategy is SplitStrategy.GROUP and not self.group_column:
            raise ValueError("el split por grupo requiere group_column")
        if self.strategy is SplitStrategy.TEMPORAL and not self.time_column:
            raise ValueError("el split temporal requiere time_column")
        if self.strategy is SplitStrategy.KFOLD and not self.folds:
            raise ValueError("k-fold requiere folds")
        return self


def _labels_for(n: int, test_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    out = np.full(n, TRAIN, dtype=object)
    out[val_idx] = VAL
    out[test_idx] = TEST
    return out


def _can_stratify(y: np.ndarray, fraction: float) -> bool:
    if fraction <= 0:
        return False
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2 and round(len(y) * fraction) >= len(counts)


def _random_split(
    idx: np.ndarray, y: np.ndarray | None, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (resto, separado)."""
    if frac <= 0 or len(idx) < 2:
        return idx, np.array([], dtype=idx.dtype)
    strat = y if y is not None and _can_stratify(y, frac) else None
    rest, held = train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    return np.sort(rest), np.sort(held)


def assign_splits(df: pl.DataFrame, req: SplitRequest, target: str | None) -> pl.DataFrame:
    """Devuelve `df` con las columnas `__split__` (y `__fold__` si es k-fold)."""
    n = df.height
    if n < 3:
        raise ValidationError("se necesitan al menos 3 filas para particionar")
    idx = np.arange(n)
    stratified = req.strategy in (SplitStrategy.STRATIFIED, SplitStrategy.KFOLD)
    y = df[target].to_numpy() if target and stratified else None
    val_rel = req.val_fraction / (1 - req.test_fraction) if req.test_fraction < 1 else 0

    if req.strategy in (SplitStrategy.RANDOM, SplitStrategy.STRATIFIED, SplitStrategy.KFOLD):
        rest, test = _random_split(idx, y, req.test_fraction, req.seed)
        if req.strategy is SplitStrategy.KFOLD:
            val = np.array([], dtype=int)
        else:
            y_rest = y[rest] if y is not None else None
            _, val = _random_split(rest, y_rest, val_rel, req.seed + 1)
    elif req.strategy is SplitStrategy.GROUP and req.group_column:
        groups = df[req.group_column].to_numpy()
        test = val = np.array([], dtype=int)
        rest = idx
        if req.test_fraction > 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=req.test_fraction, random_state=req.seed)
            r, t = next(gss.split(idx, groups=groups))
            rest, test = idx[r], idx[t]
        if val_rel > 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=req.seed + 1)
            r, v = next(gss.split(rest, groups=groups[rest]))
            val = rest[v]
    elif req.strategy is SplitStrategy.TEMPORAL and req.time_column:
        order = df[req.time_column].arg_sort().to_numpy()
        n_test = round(n * req.test_fraction)
        n_val = round(n * req.val_fraction)
        test = order[n - n_test :] if n_test else np.array([], dtype=int)
        val = order[n - n_test - n_val : n - n_test] if n_val else np.array([], dtype=int)
    else:  # pragma: no cover
        raise ValidationError(f"estrategia no soportada: {req.strategy}")

    out = df.with_columns(pl.Series(SPLIT_COLUMN, _labels_for(n, test, val), dtype=pl.String))
    if req.strategy is SplitStrategy.KFOLD and req.folds:
        folds = np.full(n, -1, dtype=np.int16)
        rest = np.setdiff1d(idx, test)
        y_rest = y[rest] if y is not None else None
        splitter = (
            StratifiedKFold(req.folds, shuffle=True, random_state=req.seed)
            if y_rest is not None and _can_stratify(y_rest, 1 / req.folds)
            else KFold(req.folds, shuffle=True, random_state=req.seed)
        )
        for k, (_, fold_idx) in enumerate(splitter.split(rest, y_rest)):
            folds[rest[fold_idx]] = k
        out = out.with_columns(pl.Series(FOLD_COLUMN, folds))
    return out


def split_counts(df: pl.DataFrame) -> dict[str, int]:
    counts = dict(df.group_by(SPLIT_COLUMN).len().iter_rows())
    return {k: int(counts.get(k, 0)) for k in (TRAIN, VAL, TEST)}
