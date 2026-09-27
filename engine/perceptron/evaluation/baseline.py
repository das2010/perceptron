"""Baseline de referencia LightGBM para tabular (SPEC §5.1).

No compite en el catálogo: se entrena con el mismo pipeline ajustado y se evalúa
sobre el mismo test sellado para medir la brecha de la red elegida.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from perceptron.data.pipeline.pipeline import FittedPipeline, decode_regression, transform_tabular
from perceptron.data.view import DatasetView, Purpose
from perceptron.domain.enums import TaskType
from perceptron.evaluation.metrics import classification_metrics, regression_metrics
from perceptron.tasks.base import flat_numbers

N_ESTIMATORS = 400
EARLY_STOPPING = 30


def _matrix(
    fitted: FittedPipeline, view: DatasetView, split: str
) -> tuple[np.ndarray, np.ndarray | None]:
    purpose = Purpose.FINAL_EVALUATION if split == "test" else Purpose.TRAINING
    arr = transform_tabular(fitted, view.read(split, purpose=purpose))
    return np.concatenate([arr.x_num, arr.x_cat.astype(np.float32)], axis=1), arr.y


def lightgbm_baseline(view: DatasetView, fitted: FittedPipeline, seed: int = 42) -> dict[str, Any]:
    import lightgbm as lgb

    target = fitted.spec.target
    if target is None:
        raise ValueError("el baseline necesita un target")
    x_tr, y_tr = _matrix(fitted, view, "train")
    x_va, y_va = _matrix(fitted, view, "val")
    x_te, y_te = _matrix(fitted, view, "test")
    if y_tr is None or y_te is None:
        raise ValueError("faltan etiquetas para el baseline")
    n_num = len(fitted.numeric_features)
    cat_idx = list(range(n_num, x_tr.shape[1]))
    common = {
        "n_estimators": N_ESTIMATORS,
        "random_state": seed,
        "verbose": -1,
        "min_child_samples": 5,
    }
    callbacks = (
        [lgb.early_stopping(EARLY_STOPPING, verbose=False)]
        if y_va is not None and len(y_va)
        else []
    )
    eval_set = [(x_va, y_va)] if callbacks else None
    if target.task is TaskType.REGRESSION:
        model = lgb.LGBMRegressor(**common)
        model.fit(
            x_tr,
            y_tr,
            eval_set=eval_set,
            categorical_feature=cat_idx or "auto",
            callbacks=callbacks,
        )
        pred = decode_regression(fitted, np.asarray(model.predict(x_te)))
        reg, _ = regression_metrics(
            decode_regression(fitted, y_te).astype(float), pred.astype(float)
        )
        metrics = flat_numbers(reg.model_dump())
    else:
        model = lgb.LGBMClassifier(**common)
        model.fit(
            x_tr,
            y_tr,
            eval_set=eval_set,
            categorical_feature=cat_idx or "auto",
            callbacks=callbacks,
        )
        proba = np.asarray(model.predict_proba(x_te))
        k = len(fitted.classes or [])
        if proba.shape[1] < k:  # clases ausentes en train
            full = np.zeros((len(proba), k))
            full[:, model.classes_.astype(int)] = proba
            proba = full
        cls, _ = classification_metrics(y_te.astype(int), proba, fitted.classes or [])
        metrics = flat_numbers(cls.model_dump())
    return {
        "model": "lightgbm",
        "metrics": metrics,
        "best_iteration": int(getattr(model, "best_iteration_", 0) or N_ESTIMATORS),
    }
