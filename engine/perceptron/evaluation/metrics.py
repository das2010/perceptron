"""Métricas por tarea (RF-EVL-01): clasificación y regresión.

Los gráficos se devuelven como datos (series de puntos) para que la UI los
dibuje con ECharts; las curvas se submuestrean a `MAX_POINTS`.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from pydantic import BaseModel, Field
from sklearn import metrics as skm

MAX_POINTS = 200
CALIBRATION_BINS = 15


def _downsample(*arrays: np.ndarray) -> list[list[float]]:
    n = len(arrays[0])
    idx = (
        np.unique(np.linspace(0, n - 1, min(n, MAX_POINTS)).round().astype(int))
        if n
        else np.array([], int)
    )
    return [[float(a[i]) for a in arrays] for i in idx]


def _f(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 6)


class PerClass(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class ThresholdInfo(BaseModel):
    threshold: float
    f1: float
    precision: float
    recall: float
    criterion: str = "max_f1"


class ClassificationMetrics(BaseModel):
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    roc_auc: float | None
    pr_auc: float | None
    log_loss: float | None
    ece: float
    per_class: list[PerClass]
    confusion_matrix: list[list[int]]
    labels: list[str]
    optimal_threshold: ThresholdInfo | None = None


class RegressionMetrics(BaseModel):
    mae: float
    rmse: float
    mape: float | None
    smape: float
    r2: float
    median_abs_error: float


class Curves(BaseModel):
    roc: list[list[float]] = Field(default_factory=list, description="[fpr, tpr]")
    pr: list[list[float]] = Field(default_factory=list, description="[recall, precision]")
    reliability: list[list[float]] = Field(
        default_factory=list, description="[confianza, acierto, n]"
    )
    residuals: list[list[float]] = Field(default_factory=list, description="[predicción, residuo]")
    error_histogram: list[list[float]] = Field(default_factory=list, description="[centro, n]")


def expected_calibration_error(
    y: np.ndarray, proba: np.ndarray, bins: int = CALIBRATION_BINS
) -> tuple[float, list[list[float]]]:
    conf = proba.max(1)
    correct = (proba.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    diagram: list[list[float]] = []
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc, c = correct[mask].mean(), conf[mask].mean()
            ece += mask.mean() * abs(acc - c)
            diagram.append([float(c), float(acc), float(mask.sum())])
    return float(ece), diagram


def classification_metrics(
    y: np.ndarray, proba: np.ndarray, labels: list[str]
) -> tuple[ClassificationMetrics, Curves]:
    k = len(labels)
    pred = proba.argmax(1)
    idx = list(range(k))
    p, r, f, s = skm.precision_recall_fscore_support(y, pred, labels=idx, zero_division=0)
    curves = Curves()
    roc_auc = pr_auc = None
    threshold = None
    present = np.unique(y)
    if k == 2 and len(present) == 2:
        score = proba[:, 1]
        roc_auc = skm.roc_auc_score(y, score)
        pr_auc = skm.average_precision_score(y, score)
        fpr, tpr, _ = skm.roc_curve(y, score)
        prec, rec, thr = skm.precision_recall_curve(y, score)
        curves.roc = _downsample(fpr, tpr)
        curves.pr = _downsample(rec, prec)
        f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        if len(f1s):
            best = int(f1s.argmax())
            threshold = ThresholdInfo(
                threshold=float(thr[best]),
                f1=float(f1s[best]),
                precision=float(prec[best]),
                recall=float(rec[best]),
            )
    elif k > 2 and len(present) == k:
        roc_auc = skm.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=idx)
        onehot = np.eye(k)[y]
        pr_auc = skm.average_precision_score(onehot, proba, average="macro")
    ece, curves.reliability = expected_calibration_error(y, proba)
    try:
        ll: float | None = skm.log_loss(y, np.clip(proba, 1e-12, 1), labels=idx)
    except ValueError:
        ll = None
    m = ClassificationMetrics(
        accuracy=float(skm.accuracy_score(y, pred)),
        balanced_accuracy=float(skm.balanced_accuracy_score(y, pred))
        if len(present) > 1
        else float(skm.accuracy_score(y, pred)),
        precision_macro=float(p.mean()),
        recall_macro=float(r.mean()),
        f1_macro=float(f.mean()),
        f1_micro=float(skm.f1_score(y, pred, labels=idx, average="micro", zero_division=0)),
        f1_weighted=float(skm.f1_score(y, pred, labels=idx, average="weighted", zero_division=0)),
        roc_auc=_f(roc_auc) if roc_auc is not None else None,
        pr_auc=_f(pr_auc) if pr_auc is not None else None,
        log_loss=_f(ll) if ll is not None else None,
        ece=round(ece, 6),
        per_class=[
            PerClass(
                label=labels[i],
                precision=float(p[i]),
                recall=float(r[i]),
                f1=float(f[i]),
                support=int(s[i]),
            )
            for i in idx
        ],
        confusion_matrix=skm.confusion_matrix(y, pred, labels=idx).tolist(),
        labels=labels,
        optimal_threshold=threshold,
    )
    return m, curves


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[RegressionMetrics, Curves]:
    err = pred - y
    nonzero = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs(err[nonzero] / y[nonzero]))) if nonzero.any() else None
    denom = np.abs(y) + np.abs(pred)
    smape = float(
        np.mean(np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0))
    )
    hist, edges = np.histogram(err, bins=30) if len(err) else (np.array([]), np.array([0.0]))
    centers = (edges[:-1] + edges[1:]) / 2 if len(edges) > 1 else np.array([])
    order = np.argsort(pred)
    m = RegressionMetrics(
        mae=float(skm.mean_absolute_error(y, pred)),
        rmse=float(math.sqrt(skm.mean_squared_error(y, pred))),
        mape=mape,
        smape=smape,
        r2=float(skm.r2_score(y, pred)) if len(y) > 1 else 0.0,
        median_abs_error=float(skm.median_absolute_error(y, pred)),
    )
    curves = Curves(
        residuals=_downsample(pred[order], err[order]),
        error_histogram=[[float(c), float(n)] for c, n in zip(centers, hist, strict=True)],
    )
    return m, curves
