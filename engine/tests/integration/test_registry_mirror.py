"""Espejo del registro de modelos en MLflow (RF-TRK-03)."""

from __future__ import annotations

from pathlib import Path

from perceptron.domain.enums import ModelStage
from perceptron.domain.models import ModelVersion
from perceptron.tracking.registry import CHAMPION_ALIAS, RegistryMirror, registered_name
from perceptron.tracking.tracker import MlflowTracker


def test_versions_stages_and_champion_alias(tmp_path: Path) -> None:
    tracker = MlflowTracker(tmp_path / "mlflow con espacio")
    mirror = RegistryMirror(tracker.client)
    first_run = tracker.start_run("prj_1", "t1", {})
    second_run = tracker.start_run("prj_1", "t2", {})

    a = ModelVersion(project_id="prj_1", run_id="run_a")
    b = ModelVersion(project_id="prj_1", run_id="run_b")
    va, vb = mirror.register(a, first_run), mirror.register(b, second_run)
    assert (va, vb) == ("1", "2")
    assert mirror.register(ModelVersion(project_id="prj_1", run_id="x"), None) is None

    name = registered_name("prj_1")
    a = a.model_copy(update={"mlflow_version": va, "stage": ModelStage.PRODUCTION})
    mirror.sync_stage(a)
    assert tracker.client.get_model_version_by_alias(name, CHAMPION_ALIAS).version == "1"
    # Promover b: a se archiva y el alias pasa a b.
    mirror.sync_stage(a.model_copy(update={"stage": ModelStage.ARCHIVED}))
    b = b.model_copy(update={"mlflow_version": vb, "stage": ModelStage.PRODUCTION})
    mirror.sync_stage(b)
    assert tracker.client.get_model_version_by_alias(name, CHAMPION_ALIAS).version == "2"
    tags = tracker.client.get_model_version(name, "1").tags
    assert tags["perceptron.stage"] == "archived"
    assert tags["perceptron.model_version"] == a.id
