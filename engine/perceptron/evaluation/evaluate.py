"""Evaluación final sobre el test sellado (RF-EVL-01) y registro del modelo.

Es el único lugar que abre el test con `Purpose.FINAL_EVALUATION`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from perceptron.data.view import DatasetView, Purpose
from perceptron.domain.enums import ModelStage, TaskType
from perceptron.domain.models import ModelVersion
from perceptron.evaluation.metrics import ClassificationMetrics, RegressionMetrics
from perceptron.storage.filesystem import write_json
from perceptron.tasks import get_adapter
from perceptron.training.config import RESULT_FILE, RunResult
from perceptron.training.data import make_dataset
from perceptron.training.inference import TrainedModel, load_trained, predict

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


def evaluate_run(run_dir: Path, dataset_dir: Path, *, split: str = "test") -> EvaluationReport:
    trained = load_trained(run_dir)
    view = DatasetView(dataset_dir)
    ds = make_dataset(view, trained.pipeline, split, train=False, purpose=Purpose.FINAL_EVALUATION)
    preds = predict(trained, ds)
    run_id = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_id"]
    result = get_adapter(trained.task).evaluate(preds, trained.spec, trained.pipeline)
    detail = dict(result.detail)
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
    return report


def signature(trained: TrainedModel) -> dict[str, Any]:
    """Firma de entrada/salida del modelo (RF-EXP-05)."""
    fp = trained.pipeline
    spec = trained.spec
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
    trained = load_trained(run_dir)
    result_file = run_dir / RESULT_FILE
    result = (
        RunResult.model_validate_json(result_file.read_text(encoding="utf-8"))
        if result_file.is_file()
        else None
    )
    card = {
        "name": trained.spec.name,
        "template": trained.spec.provenance.template,
        "rationale": trained.spec.provenance.rationale,
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
        signature=signature(trained),
        model_card=card,
    )
