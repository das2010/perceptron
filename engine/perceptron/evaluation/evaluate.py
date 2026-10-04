"""Evaluación final sobre el test sellado (RF-EVL-01) y registro del modelo.

Es el único lugar que abre el test con `Purpose.FINAL_EVALUATION`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from perceptron.archspec.schema import ArchSpec
from perceptron.catalog.registry import CODE_BLOCK
from perceptron.data.pipeline.pipeline import FittedPipeline
from perceptron.data.view import DatasetView, Purpose
from perceptron.domain.enums import ModelStage, TaskType
from perceptron.domain.models import ModelVersion
from perceptron.evaluation.cost import CostSpec, cost_threshold
from perceptron.evaluation.metrics import ClassificationMetrics, RegressionMetrics
from perceptron.storage.filesystem import write_json
from perceptron.tasks import get_adapter
from perceptron.training.config import RESULT_FILE, RunResult
from perceptron.training.data import make_dataset
from perceptron.training.inference import load_run_artifacts, load_trained, predict

EVALUATION_DIR = "evaluation"
EVALUATION_FILE = "evaluation.json"


class EvaluationReport(BaseModel):
    run_id: str
    split: str = "test"
    task: TaskType
    num_samples: int
    metrics: dict[str, float] = Field(description="Resumen plano (tracking, comparación de runs)")
    classification: ClassificationMetrics | None = None
    regression: RegressionMetrics | None = None
    details: dict[str, Any] = Field(
        default_factory=dict, description="Detalle propio de cada tarea"
    )
    curves: dict[str, list[list[float]]] = Field(default_factory=dict)
    checkpoint: str


def evaluate_run(
    run_dir: Path, dataset_dir: Path, *, split: str = "test", cost: CostSpec | None = None
) -> EvaluationReport:
    trained = load_trained(run_dir)
    view = DatasetView(dataset_dir)
    ds = make_dataset(view, trained.pipeline, split, train=False, purpose=Purpose.FINAL_EVALUATION)
    preds = predict(trained, ds)
    run_id = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_id"]
    adapter = get_adapter(trained.task)
    calibration: dict[str, Any] = {}
    if adapter.needs_calibration:
        val_ds = make_dataset(
            view, trained.pipeline, "val", train=False, purpose=Purpose.FINAL_EVALUATION
        )
        loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        with torch.no_grad():
            calibration = adapter.calibrate(trained.model, loader, trained.spec)
    result = adapter.evaluate(preds, trained.spec, trained.pipeline, calibration)
    detail = dict(result.detail)
    classes = trained.pipeline.classes or []
    if cost is not None and trained.task is TaskType.CLASSIFICATION and len(classes) == 2:
        # Umbral por costo (ADR-0040): se elige en validación, se informa en test.
        val_ds = make_dataset(
            view, trained.pipeline, "val", train=False, purpose=Purpose.FINAL_EVALUATION
        )
        val = predict(trained, val_ds)
        if val.proba is not None and preds.proba is not None and val.y_true is not None:
            detail["cost_threshold"] = cost_threshold(
                val.y_true.astype(int),
                val.proba[:, 1],
                np.asarray(preds.y_true).astype(int),
                preds.proba[:, 1],
                cost,
                positive=str(classes[1]),
            )
    report = EvaluationReport(
        run_id=run_id,
        split=split,
        task=trained.task,
        num_samples=len(preds.y_pred),
        metrics=result.metrics,
        classification=ClassificationMetrics.model_validate(detail.pop("classification"))
        if "classification" in detail
        else None,
        regression=RegressionMetrics.model_validate(detail.pop("regression"))
        if "regression" in detail
        else None,
        details=detail,
        curves=result.curves,
        checkpoint=trained.checkpoint.name,
    )
    write_json(run_dir / EVALUATION_DIR / EVALUATION_FILE, report.model_dump(mode="json"))
    # Por muestra, para el análisis de errores y de fairness (RF-EVL-03/04).
    from perceptron.evaluation.predictions import save_predictions

    save_predictions(run_dir / EVALUATION_DIR, preds, trained.pipeline, trained.task)
    return report


def signature(spec: ArchSpec, fp: FittedPipeline) -> dict[str, Any]:
    """Firma de entrada/salida del modelo (RF-EXP-05)."""
    inp: dict[str, Any] = {"kind": spec.input.kind}
    if spec.input.kind == "tabular":
        steps_cols = sorted({c for s in fp.spec.steps for c in s.columns})
        inp.update(
            columns=steps_cols,
            numeric_features=fp.numeric_features,
            categorical_features=fp.categorical_features,
        )
    else:
        inp.update(
            shape=spec.input.shape, channels=spec.input.shape[0] if spec.input.shape else None
        )
    out: dict[str, Any] = {"task": spec.task.type.value}
    if fp.classes is not None:
        out["classes"] = fp.classes
    return {"inputs": inp, "outputs": out, "archspec_hash": spec.content_hash()}


def build_model_version(
    project_id: str,
    run_id: str,
    run_dir: Path,
    report: EvaluationReport,
    *,
    dataset_hash: str | None = None,
) -> ModelVersion:
    """`ModelVersion` en stage `candidate` con firma y model card básica (RF-TRK-03 en Capa 4)."""
    # Sin construir el modelo: el de código experto solo se construye en el sandbox.
    spec, pipeline, _ = load_run_artifacts(run_dir)
    result_file = run_dir / RESULT_FILE
    result = (
        RunResult.model_validate_json(result_file.read_text(encoding="utf-8"))
        if result_file.is_file()
        else None
    )
    card = {
        "name": spec.name,
        "declarative": not any(n.block == CODE_BLOCK for n in spec.nodes),
        "template": spec.provenance.template,
        "rationale": spec.provenance.rationale,
        "task": report.task.value,
        "test_metrics": report.metrics,
        "validation_metrics": result.best_metrics if result else {},
        "dataset_hash": dataset_hash,
        "epochs": result.epochs if result else None,
        "environment": result.environment if result else {},
        "limitations": [
            "Evaluado solo sobre el test de este dataset; puede no generalizar a otros datos.",
        ],
    }
    return ModelVersion(
        project_id=project_id,
        run_id=run_id,
        stage=ModelStage.CANDIDATE,
        signature=signature(spec, pipeline),
        model_card=card,
    )
