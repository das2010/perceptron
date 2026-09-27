"""Entrenamientos reales cortos: detección (UC-04), segmentación (UC-05), OCR (UC-06)."""

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

CASES = [
    ("uc04_defects", TaskType.OBJECT_DETECTION, {"map", "map_50"}),
    ("uc05_masks", None, {"mean_iou", "iou_foreground", "pixel_accuracy"}),
    ("uc06_ocr", None, {"cer", "wer", "exact_match"}),
]


@pytest.mark.parametrize(("uc", "task", "keys"), CASES)
def test_train_vision_task(
    uc: str,
    task: TaskType | None,
    keys: set[str],
    workspace_dir: Path,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    paths = ProjectPaths(workspace_dir / "projects" / "prj_vtt").ensure()
    v = ingest(paths, IngestRequest(project_id="prj_vtt", source=fixtures_dir / uc, task=task))
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    fitted = fit_pipeline(propose_pipeline(card), view.read("train"), view.files_dir)
    rec = recommend(card, fitted)
    cfg = RunConfig(
        run_id=f"run_{uc}",
        run_dir=paths.run(f"run_{uc}"),
        dataset_dir=view.root,
        archspec=rec.spec.model_dump(mode="json"),
        pipeline=fitted.model_dump(mode="json"),
        pretrained_allowed=False,
        max_epochs=2,
    )
    result = run_sync(cfg, timeout=300)
    assert result.status == "succeeded", result.error
    report = evaluate_run(cfg.run_dir, view.root)
    assert keys <= set(report.metrics)
