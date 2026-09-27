from __future__ import annotations

import time
from pathlib import Path

import pytest

from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType
from perceptron.tracking.tracker import MemoryTracker, MlflowTracker, RunRecorder, Tracker
from perceptron.training.config import RunConfig, RunEvent, RunResult


def _cfg(tmp: Path) -> RunConfig:
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[4]
    )
    run_dir = tmp / "run ñ"
    run_dir.mkdir(parents=True)
    (run_dir / "archspec.json").write_text(spec.model_dump_json(), encoding="utf-8")
    ckpt = run_dir / "best.ckpt"
    ckpt.write_bytes(b"pesos")
    return RunConfig(
        run_id="run_x",
        run_dir=run_dir,
        dataset_dir=tmp,
        archspec=spec.model_dump(mode="json"),
        pipeline={},
        overrides={"lr": 0.01},
    )


def _events() -> list[RunEvent]:
    now = time.time()
    return [
        RunEvent(
            event="started",
            run_id="run_x",
            ts=now,
            data={"batch_size": 32, "environment": {"torch": "2.x"}},
        ),
        RunEvent(
            event="epoch",
            run_id="run_x",
            ts=now,
            epoch=0,
            metrics={"val_loss": 0.7, "train_loss": 0.8},
        ),
        RunEvent(
            event="epoch",
            run_id="run_x",
            ts=now,
            epoch=1,
            metrics={"val_loss": 0.5, "train_loss": 0.6},
        ),
    ]


def _replay(tracker: Tracker, cfg: RunConfig) -> str | None:
    rec = RunRecorder(tracker, cfg, experiment="prj_demo", tags={"perceptron.origin": "rules"})
    for ev in _events():
        rec(ev)
    result = RunResult(
        run_id="run_x",
        status="succeeded",
        epochs=2,
        best_metrics={"val_loss": 0.5},
        best_checkpoint=cfg.run_dir / "best.ckpt",
    )
    return rec.finish(result)


def test_memory_tracker(tmp_path: Path) -> None:
    tracker = MemoryTracker()
    rid = _replay(tracker, _cfg(tmp_path))
    assert rid is not None
    run = tracker.runs[rid]
    assert run.status == "FINISHED"
    assert run.params["hp.lr"] == "0.01"  # el override gana al default
    assert run.params["run.batch_size"] == "32"
    assert ("val_loss", 0.5, 1) in run.metrics
    assert run.tags["env.torch"] == "2.x"
    assert any(p.name == "best.ckpt" for p, _ in run.artifacts)


@pytest.mark.slow
def test_mlflow_tracker_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "Carpeta con espacios y ñ" / "mlflow"
    tracker = MlflowTracker(root)
    rid = _replay(tracker, _cfg(tmp_path))
    assert rid is not None
    run = tracker.client.get_run(rid)
    assert run.info.status == "FINISHED"
    assert run.data.params["hp.lr"] == "0.01"
    assert run.data.tags["perceptron.origin"] == "rules"
    history = tracker.client.get_metric_history(rid, "val_loss")
    assert [(m.step, m.value) for m in history] == [(0, 0.7), (1, 0.5)]
    artifacts = {a.path for a in tracker.client.list_artifacts(rid, "run")}
    assert "run/archspec.json" in artifacts
    assert {a.path for a in tracker.client.list_artifacts(rid, "checkpoints")} == {
        "checkpoints/best.ckpt"
    }
    # El mismo experimento se reutiliza
    assert tracker.experiment_id("prj_demo") == run.info.experiment_id
