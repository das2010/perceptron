from __future__ import annotations

from pathlib import Path

from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.evaluation.baseline import lightgbm_baseline


def test_lightgbm_baseline_uc01(workspace_dir: Path, fixtures_dir: Path) -> None:
    paths = ProjectPaths(workspace_dir / "projects" / "prj_b").ensure()
    v = ingest(
        paths, IngestRequest(project_id="prj_b", source=fixtures_dir / "uc01_churn" / "churn.csv")
    )
    view = DatasetView(paths.dataset(v.content_hash))
    fitted = fit_pipeline(propose_pipeline(profile_dataset(view)), view.read("train"))
    result = lightgbm_baseline(view, fitted)
    assert result["model"] == "lightgbm"
    assert result["metrics"]["roc_auc"] > 0.7
    assert result["best_iteration"] > 0
