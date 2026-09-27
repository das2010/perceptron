"""Entrenar → evaluar en test sellado → registrar ModelVersion (UC-01 y UC-04)."""

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
from perceptron.domain.enums import ModelStage, TaskType
from perceptron.domain.models import ModelVersion
from perceptron.evaluation.evaluate import (
    EVALUATION_DIR,
    EVALUATION_FILE,
    build_model_version,
    evaluate_run,
)
from perceptron.storage.db import Database
from perceptron.storage.repositories import SqlRepository
from perceptron.training.config import RunConfig
from perceptron.training.runner import run_sync


@pytest.mark.parametrize("uc", ["uc01_churn/churn.csv", "uc04_defects"])
def test_train_evaluate_register(
    uc: str, workspace_dir: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    paths = ProjectPaths(workspace_dir / "projects" / "prj_e").ensure()
    v = ingest(paths, IngestRequest(project_id="prj_e", source=fixtures_dir / uc))
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    fitted = fit_pipeline(propose_pipeline(card), view.read("train"))
    rec = recommend(card, fitted)
    cfg = RunConfig(
        run_id="run_ev",
        run_dir=paths.run("run_ev"),
        dataset_dir=view.root,
        archspec=rec.spec.model_dump(mode="json"),
        pipeline=fitted.model_dump(mode="json"),
        pretrained_allowed=False,
        max_epochs=3,
        num_workers=0,
    )
    assert run_sync(cfg, timeout=300).status == "succeeded"

    report = evaluate_run(cfg.run_dir, view.root)
    assert report.task is TaskType.CLASSIFICATION
    assert v.split is not None and report.num_samples == v.split.test
    assert report.classification is not None
    assert {"accuracy", "f1_macro", "ece"} <= set(report.metrics)
    assert (cfg.run_dir / EVALUATION_DIR / EVALUATION_FILE).is_file()

    mv = build_model_version("prj_e", "run_ev", cfg.run_dir, report, dataset_hash=v.content_hash)
    assert mv.stage is ModelStage.CANDIDATE
    assert set(mv.signature["outputs"]["classes"]) == set(fitted.classes or [])
    assert mv.model_card["test_metrics"]["accuracy"] == report.metrics["accuracy"]

    db = Database.for_file(workspace_dir / "perceptron.db")
    db.create_all()
    repo = SqlRepository(db, ModelVersion)
    repo.add(mv)
    assert repo.get(mv.id).run_id == "run_ev"
    db.dispose()
