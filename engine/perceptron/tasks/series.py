"""Forecasting y detección de anomalías en series temporales (RF-EVL-01 series/anomalías)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torchmetrics
from sklearn import metrics as skm
from torch import nn

from perceptron.domain.enums import TaskType
from perceptron.tasks.base import (
    Predictions,
    Resolver,
    StepOutput,
    TaskAdapter,
    TaskEvaluation,
    pick_metrics,
)

if TYPE_CHECKING:
    from perceptron.archspec.schema import ArchSpec
    from perceptron.data.pipeline.pipeline import FittedPipeline

ANOMALY_QUANTILE = 0.99


def _smape(y: np.ndarray, p: np.ndarray) -> float:
    den = np.abs(y) + np.abs(p)
    return float(
        np.mean(np.where(den > 0, 2 * np.abs(p - y) / np.where(den > 0, den, 1), 0.0)) * 100
    )


def forecast_report(
    y: np.ndarray, pred: np.ndarray, naive: np.ndarray, scale: np.ndarray, series: list[str]
) -> TaskEvaluation:
    """Métricas en unidades originales; `naive` es el baseline seasonal-naive."""
    err = pred - y
    mae = float(np.abs(err).mean())
    naive_mae = float(np.abs(naive - y).mean())
    per_h = np.abs(err).mean(0)
    mase = float((np.abs(err).mean(1) / np.clip(scale, 1e-12, None)).mean())
    naive_mase = float((np.abs(naive - y).mean(1) / np.clip(scale, 1e-12, None)).mean())
    per_series: dict[str, float] = {}
    for key in sorted(set(series)):
        idx = [i for i, s in enumerate(series) if s == key]
        per_series[key] = round(_smape(y[idx], pred[idx]), 4)
    metrics = {
        "mae": mae,
        "rmse": float(np.sqrt((err**2).mean())),
        "smape": _smape(y, pred),
        "mase": mase,
        "naive_mae": naive_mae,
        "naive_smape": _smape(y, naive),
        "naive_mase": naive_mase,
        "skill_vs_naive": 1 - mae / naive_mae if naive_mae > 0 else 0.0,
        "windows": float(len(y)),
    }
    return TaskEvaluation(
        metrics=metrics,
        detail={
            "mae_per_horizon": [round(float(v), 6) for v in per_h],
            "smape_per_series": per_series,
        },
        curves={"mae_per_horizon": [[float(h + 1), float(v)] for h, v in enumerate(per_h)]},
    )


class ForecastingAdapter(TaskAdapter):
    """Batch: (x [B, L, C], y [B, H], mean, std, naive, scale, level) — x e y relativos a level."""

    task: ClassVar[TaskType] = TaskType.FORECASTING

    def num_outputs(self, spec: ArchSpec) -> int:
        if not spec.task.horizon:
            raise ValueError("task.horizon es requerido para forecasting")
        return spec.task.horizon * spec.task.num_targets

    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module:
        return {"mse": nn.MSELoss(), "huber": nn.HuberLoss()}.get(spec.loss.type, nn.L1Loss())

    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection:
        return pick_metrics(
            {
                "mae": torchmetrics.MeanAbsoluteError,
                "rmse": lambda: torchmetrics.MeanSquaredError(squared=False),
            },
            spec.metrics,
        )

    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput:
        x, y = batch[0], batch[1]
        out: torch.Tensor = model(x)
        loss = loss_fn(out, y)
        return StepOutput(
            loss=loss, metric_args=(out.detach().flatten(), y.flatten()), batch_size=len(y)
        )

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions:
        preds, ys, naive, scale = [], [], [], []
        for x, y, mean, std, nv, sc, level in loader:
            out = model(x) + level[:, None]
            preds.append((out * std[:, None] + mean[:, None]).numpy())
            ys.append(((y + level[:, None]) * std[:, None] + mean[:, None]).numpy())
            naive.append(nv.numpy())
            scale.append(sc.numpy())
        dataset = loader.dataset
        cat = np.concatenate
        return Predictions(
            y_true=cat(ys) if ys else np.zeros((0, 1)),
            y_pred=cat(preds) if preds else np.zeros((0, 1)),
            extra={
                "naive": cat(naive) if naive else np.zeros((0, 1)),
                "scale": cat(scale) if scale else np.zeros(0),
                "series": list(getattr(dataset, "series", [])),
            },
        )

    def evaluate(
        self,
        preds: Predictions,
        spec: ArchSpec,
        pipeline: FittedPipeline,
        calibration: dict[str, Any] | None = None,
    ) -> TaskEvaluation:
        if preds.y_true is None or len(preds.y_true) == 0:
            raise ValueError("no hay ventanas de test para evaluar el pronóstico")
        return forecast_report(
            preds.y_true,
            preds.y_pred,
            preds.extra["naive"],
            preds.extra["scale"],
            preds.extra["series"],
        )


class AnomalyAdapter(TaskAdapter):
    """Autoencoder: el score es el error de reconstrucción. Batch: (x, label, normal)."""

    task: ClassVar[TaskType] = TaskType.ANOMALY_DETECTION

    def num_outputs(self, spec: ArchSpec) -> int:
        return 0

    def check_output(self, kind: str, shape: tuple[int, ...], spec: ArchSpec) -> str | None:
        expected = tuple(spec.input.shape or ())
        if kind != "sequence" or shape != expected:
            return (
                f"la salida debe reconstruir la entrada {list(expected)} y es {kind} {list(shape)}"
            )
        return None

    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module:
        return nn.MSELoss()

    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection({})

    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput:
        x = batch[0]
        out: torch.Tensor = model(x)
        return StepOutput(loss=loss_fn(out, x), metric_args=(), batch_size=len(x))

    @staticmethod
    @torch.no_grad()
    def _scores(
        model: nn.Module, loader: torch.utils.data.DataLoader[Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        scores, labels = [], []
        for x, label, _normal in loader:
            rec = model(x)
            # Score del punto etiquetado (el último de la ventana): si se usara la ventana
            # entera, cada anomalía contaminaría las `lookback` ventanas siguientes.
            scores.append(((rec[:, -1] - x[:, -1]) ** 2).mean(dim=1).numpy())
            labels.append(label.numpy())
        if not scores:
            return np.zeros(0), np.zeros(0, dtype=int)
        return np.concatenate(scores), np.concatenate(labels)

    def calibrate(
        self, model: nn.Module, val_loader: torch.utils.data.DataLoader[Any], spec: ArchSpec
    ) -> dict[str, Any]:
        """Umbral: mejor F1 en val si hay anomalías etiquetadas; si no, cuantil 0,99."""
        scores, labels = self._scores(model, val_loader)
        if len(scores) == 0:
            return {"threshold": float("inf"), "method": "none"}
        if labels.any() and not labels.all():
            prec, rec, thr = skm.precision_recall_curve(labels, scores)
            f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
            best = int(f1.argmax())
            return {
                "threshold": float(thr[best]),
                "method": "best_f1_val",
                "val_f1": float(f1[best]),
            }
        normal = scores[labels == 0] if (labels == 0).any() else scores
        return {
            "threshold": float(np.quantile(normal, ANOMALY_QUANTILE)),
            "method": f"q{ANOMALY_QUANTILE}_val",
        }

    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions:
        scores, labels = self._scores(model, loader)
        return Predictions(y_true=labels, y_pred=scores, extra={"scores": scores})

    def evaluate(
        self,
        preds: Predictions,
        spec: ArchSpec,
        pipeline: FittedPipeline,
        calibration: dict[str, Any] | None = None,
    ) -> TaskEvaluation:
        scores = np.asarray(preds.y_pred, dtype=float)
        threshold = float(
            (calibration or {}).get("threshold", np.quantile(scores, ANOMALY_QUANTILE))
        )
        flagged = scores >= threshold
        metrics: dict[str, float] = {
            "threshold": threshold,
            "anomaly_rate": float(flagged.mean()) if len(flagged) else 0.0,
            "windows": float(len(scores)),
        }
        detail: dict[str, Any] = {"calibration": calibration or {}}
        curves: dict[str, list[list[float]]] = {}
        y = preds.y_true
        if y is not None and len(y) and y.any():
            p, r, f, _ = skm.precision_recall_fscore_support(
                y, flagged, average="binary", zero_division=0
            )
            metrics.update(precision=float(p), recall=float(r), f1=float(f))
            if not y.all():
                metrics["roc_auc"] = float(skm.roc_auc_score(y, scores))
                metrics["pr_auc"] = float(skm.average_precision_score(y, scores))
            qs = np.quantile(scores, np.linspace(0.5, 0.999, 40))
            curves["f1_vs_threshold"] = [
                [float(t), float(skm.f1_score(y, scores >= t, zero_division=0))] for t in qs
            ]
        hist, edges = (
            np.histogram(scores, bins=30) if len(scores) else (np.array([]), np.array([0.0]))
        )
        curves["score_histogram"] = [
            [float((a + b) / 2), float(n)]
            for a, b, n in zip(edges[:-1], edges[1:], hist, strict=True)
        ]
        return TaskEvaluation(metrics=metrics, detail=detail, curves=curves)
