"""Entrenamiento real corto de texto (UC-03) en CPU + evaluación en test sellado."""

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
from perceptron.evaluation.evaluate import evaluate_run
from perceptron.training.config import RunConfig
from perceptron.training.runner import run_sync


def test_train_and_evaluate_uc03(
    workspace_dir: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    paths = ProjectPaths(workspace_dir / "projects" / "prj_t3").ensure()
    v = ingest(
        paths,
        IngestRequest(
            project_id="prj_t3", source=fixtures_dir / "uc03_tickets_es" / "tickets.jsonl"
        ),
    )
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    fitted = fit_pipeline(propose_pipeline(card), view.read("train"))
    rec = recommend(card, fitted)
    cfg = RunConfig(
        run_id="run_txt",
        run_dir=paths.run("run_txt"),
        dataset_dir=view.root,
        archspec=rec.spec.model_dump(mode="json"),
        pipeline=fitted.model_dump(mode="json"),
        pretrained_allowed=False,
        max_epochs=8,
    )
    result = run_sync(cfg, timeout=300)
    assert result.status == "succeeded", result.error
    report = evaluate_run(cfg.run_dir, view.root)
    assert report.classification is not None
    assert report.metrics["accuracy"] > 0.6  # 8 épocas; la aceptación (e2e) exige > 0,9
