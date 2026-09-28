"""Entrenamientos reales cortos de series (UC-07 forecasting, UC-08 anomalías) + evaluación."""

from __future__ import annotations

from pathlib import Path

import pytest

from perceptron.catalog.rules import recommend
from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import TaskType
from perceptron.evaluation.evaluate import evaluate_run
from perceptron.training.config import RunConfig
from perceptron.training.runner import run_sync


@pytest.mark.parametrize(
    ("uc", "task"),
    [
        ("uc07_demand/demanda.csv", TaskType.FORECASTING),
        ("uc08_telemetry/telemetria.csv", TaskType.ANOMALY_DETECTION),
    ],
)
def test_train_and_evaluate_series(
    uc: str,
    task: TaskType,
    workspace_dir: Path,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    paths = ProjectPaths(workspace_dir / "projects" / "prj_st").ensure()
    v = ingest(paths, IngestRequest(project_id="prj_st", source=fixtures_dir / uc))
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    fitted = fit_pipeline(propose_pipeline(card), view.read("train"))
    rec = recommend(card, fitted)
    cfg = RunConfig(
        run_id="run_ts",
        run_dir=paths.run(f"run_{task.value}"),
        dataset_dir=view.root,
        archspec=rec.spec.model_dump(mode="json"),
        pipeline=fitted.model_dump(mode="json"),
        pretrained_allowed=False,
        max_epochs=5,
    )
    result = run_sync(cfg, timeout=300)
    assert result.status == "succeeded", result.error
    report = evaluate_run(cfg.run_dir, view.root)
    assert report.task is task
    if task is TaskType.FORECASTING:
        assert {"mae", "smape", "mase", "naive_mae", "skill_vs_naive"} <= set(report.metrics)
        assert report.details["mae_per_horizon"]
    else:
        assert {"precision", "recall", "f1", "threshold"} <= set(report.metrics)
        assert report.details["calibration"]["method"].startswith(("best_f1", "q"))

    # RF-EVL-02: importancia por variable y por rezago sobre validación.
    from perceptron.evaluation.explain import global_explanation

    glob = global_explanation(cfg.run_dir, view.root)
    assert glob.method == "integrated_gradients" and glob.samples > 0
    variables = [f for f in glob.features if f.feature.startswith("variable: ")]
    lags = [f for f in glob.features if f.feature.startswith("rezago t-")]
    assert variables and lags and all(f.importance >= 0 for f in glob.features)
