"""HPO real: 3 trials de 2 épocas sobre UC-01 en CPU."""

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
from perceptron.hpo.recommend import recommend_strategy
from perceptron.hpo.strategy import Budget
from perceptron.hpo.study import run_study
from perceptron.tracking.tracker import MemoryTracker, RunRecorder
from perceptron.training.config import RunConfig


def test_real_study_uc01(
    workspace_dir: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    paths = ProjectPaths(workspace_dir / "projects" / "prj_h").ensure()
    v = ingest(
        paths, IngestRequest(project_id="prj_h", source=fixtures_dir / "uc01_churn" / "churn.csv")
    )
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    fitted = fit_pipeline(propose_pipeline(card), view.read("train"))
    rec = recommend(card, fitted)
    base = RunConfig(
        run_id="std_real",
        run_dir=paths.run("std_real"),
        dataset_dir=view.root,
        archspec=rec.spec.model_dump(mode="json"),
        pipeline=fitted.model_dump(mode="json"),
        pretrained_allowed=False,
    )
    strategy = recommend_strategy(rec.spec, Budget(max_trials=3, max_epochs_per_trial=2))
    tracker = MemoryTracker()
    recorders: dict[str, RunRecorder] = {}

    def on_run_event(cfg: RunConfig) -> RunRecorder:
        recorders[cfg.run_id] = RunRecorder(
            tracker, cfg, experiment="prj_h", tags={"study": "std_real"}
        )
        return recorders[cfg.run_id]

    res = run_study(
        strategy,
        base,
        storage=paths.root / "hpo" / "optuna.db",
        study_name="std_real",
        on_run_event=on_run_event,
        on_trial_end=lambda rec_, cfg, result: recorders[cfg.run_id].finish(result),
    )
    assert len(res.trials) == 3
    assert res.best_trial is not None
    assert res.best_trial.best_checkpoint and res.best_trial.best_checkpoint.is_file()
    assert len(tracker.runs) == 3
    assert all(r.status in {"FINISHED", "KILLED"} for r in tracker.runs.values())
