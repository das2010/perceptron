"""Entrenamientos reales cortos en CPU (subproceso worker + supervisor)."""

from __future__ import annotations

from pathlib import Path

import pytest

from perceptron.catalog.rules import recommend
from perceptron.core.events import EventBus
from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView, Purpose
from perceptron.training.config import RunConfig, RunEvent
from perceptron.training.data import make_dataset
from perceptron.training.inference import load_trained, predict
from perceptron.training.runner import EVENT_TOPIC, run_sync, start_run


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _config(workspace_dir: Path, src: Path, run_id: str, **kw: object) -> RunConfig:
    paths = ProjectPaths(workspace_dir / "projects" / "prj_t").ensure()
    v = ingest(paths, IngestRequest(project_id="prj_t", source=src))
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    fitted = fit_pipeline(propose_pipeline(card), view.read("train"))
    rec = recommend(card, fitted)
    return RunConfig(
        run_id=run_id,
        run_dir=paths.run(run_id),
        dataset_dir=view.root,
        archspec=rec.spec.model_dump(mode="json"),
        pipeline=fitted.model_dump(mode="json"),
        pretrained_allowed=False,
        emit_every_n_batches=1,
        **kw,  # type: ignore[arg-type]
    )


def test_train_tabular_churn(workspace_dir: Path, fixtures_dir: Path) -> None:
    cfg = _config(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "run_tab", max_epochs=3)
    bus = EventBus()
    published: list[dict[str, object]] = []
    bus.subscribe(EVENT_TOPIC, lambda e: published.append(e.payload))
    seen: list[RunEvent] = []
    result = run_sync(cfg, bus, on_event=seen.append, timeout=300)

    assert result.status == "succeeded", result.error
    kinds = [e.event for e in seen]
    assert kinds[0] == "started" and kinds[-1] == "finished"
    assert kinds.count("epoch") == 3
    assert "batch" in kinds
    epoch = next(e for e in seen if e.event == "epoch")
    assert {"val_loss", "val_accuracy", "train_loss"} <= set(epoch.metrics)
    assert "eta_s" in epoch.data and "resources" in epoch.data
    assert len(published) == len(seen)
    assert result.best_checkpoint and result.best_checkpoint.is_file()
    assert result.last_checkpoint and result.last_checkpoint.is_file()
    assert result.environment["seed"] == 42
    assert len(result.history) == 3

    # El modelo guardado se carga y predice sobre el test sellado
    trained = load_trained(cfg.run_dir)
    view = DatasetView(cfg.dataset_dir)
    ds = make_dataset(view, trained.pipeline, "test", train=False, purpose=Purpose.FINAL_EVALUATION)
    preds = predict(trained, ds)
    assert preds.proba is not None and preds.proba.shape[1] == 2
    assert preds.y_true is not None and len(preds.y_true) == len(preds.y_pred)


def test_train_images_defects(workspace_dir: Path, fixtures_dir: Path) -> None:
    cfg = _config(
        workspace_dir, fixtures_dir / "uc04_defects", "run_img", max_epochs=2, num_workers=0
    )
    result = run_sync(cfg, timeout=300)
    assert result.status == "succeeded", result.error
    assert result.epochs == 2
    assert "val_accuracy" in result.best_metrics


def test_reproducible_with_same_seed(workspace_dir: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "uc01_churn" / "churn.csv"
    a = run_sync(_config(workspace_dir, src, "run_a", max_epochs=2), timeout=300)
    b = run_sync(_config(workspace_dir, src, "run_b", max_epochs=2), timeout=300)
    assert a.status == b.status == "succeeded"
    assert a.history[-1]["val_loss"] == pytest.approx(b.history[-1]["val_loss"], rel=5e-3)  # O3


def test_cancel_then_resume(workspace_dir: Path, fixtures_dir: Path) -> None:
    cfg = _config(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "run_c", max_epochs=200)
    cfg.archspec["training"]["early_stopping"] = None
    handle = start_run(cfg)

    def on_event(ev: RunEvent) -> None:
        if ev.event == "epoch" and ev.epoch == 1:
            handle.stop()

    result = handle.wait(on_event, timeout=300)
    assert result.status == "cancelled"
    assert result.last_checkpoint and result.last_checkpoint.is_file()
    assert result.epochs < 200

    resumed = cfg.model_copy(
        update={
            "run_id": "run_c2",
            "run_dir": cfg.run_dir.parent / "run_c2",
            "resume_from": result.last_checkpoint,
            "max_epochs": result.epochs + 2,
        }
    )
    again = run_sync(resumed, timeout=300)
    assert again.status == "succeeded", again.error
    assert again.epochs == result.epochs + 2


def test_worker_error_is_reported(workspace_dir: Path, fixtures_dir: Path) -> None:
    cfg = _config(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "run_err", max_epochs=1)
    cfg.archspec["nodes"][1]["params"]["hidden"] = 10_000_000  # fuera de rango → falla al construir
    result = run_sync(cfg, timeout=300)
    assert result.status == "failed"
    assert result.error is not None
    assert result.error["code"] == "exception"
    assert "hidden" in result.error["message"]


def test_lr_finder_suggests_the_learning_rate(workspace_dir: Path, fixtures_dir: Path) -> None:
    """RF-TRN-04: con `training.lr_finder` el run usa el LR sugerido y lo informa."""
    cfg = _config(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "run_lr", max_epochs=1)
    spec = dict(cfg.archspec)
    spec["training"] = {**spec["training"], "lr_finder": True}
    cfg = cfg.model_copy(update={"archspec": spec})
    seen: list[RunEvent] = []
    result = run_sync(cfg, EventBus(), on_event=seen.append, timeout=300)
    assert result.status == "succeeded", result.error
    started = next(e for e in seen if e.event == "started")
    lr = started.data["tuned"]["lr"]
    assert isinstance(lr, float) and 0 < lr < 1
    assert [e.event for e in seen].count("epoch") == 1  # el barrido no emite épocas del run


def test_batch_size_search_grows_until_the_data_cap(
    workspace_dir: Path, fixtures_dir: Path
) -> None:
    """La búsqueda binaria contra OOM (en GPU) se prueba en CPU: sin OOM crece hasta el techo."""
    from perceptron.archspec.schema import ArchSpec
    from perceptron.data.pipeline.pipeline import FittedPipeline
    from perceptron.training.data import make_loader
    from perceptron.training.module import PerceptronModule
    from perceptron.training.tuning import batch_size_cap, tune_batch_size

    cfg = _config(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "run_bs")
    view = DatasetView(cfg.dataset_dir)
    fitted = FittedPipeline.model_validate(cfg.pipeline)
    train_ds = make_dataset(view, fitted, "train", train=True)
    val_ds = make_dataset(view, fitted, "val", train=False)
    n_train = len(train_ds)  # type: ignore[arg-type]
    module = PerceptronModule(ArchSpec.model_validate(cfg.archspec), pretrained_allowed=False)

    def loader(size: int, train: bool) -> object:
        return make_loader(
            train_ds if train else val_ds, size, shuffle=train, num_workers=0, seed=1
        )

    found = tune_batch_size(
        module,
        loader,  # type: ignore[arg-type]
        start=8,
        n_train=n_train,
        trainer_kwargs={"accelerator": "cpu", "precision": "32-true"},
        root=workspace_dir / "tuning",
        max_trials=3,
    )
    assert 8 < found <= batch_size_cap(8, n_train)
    assert batch_size_cap(32, 100) == 32  # con pocos datos no se busca más grande


def test_streaming_tabular_dataset_reads_in_batches(
    workspace_dir: Path, fixtures_dir: Path
) -> None:
    """RF-ING-09: el split grande se lee por lotes, cada fila una vez por época."""
    from perceptron.data.pipeline.pipeline import FittedPipeline
    from perceptron.training import data as tdata

    cfg = _config(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "run_stream")
    view = DatasetView(cfg.dataset_dir)
    fitted = FittedPipeline.model_validate(cfg.pipeline)
    in_memory = make_dataset(view, fitted, "train", train=False)
    stream = tdata.StreamingTabularDataset(fitted, view.data_file, "train", shuffle=True, seed=3)
    assert len(stream) == len(in_memory)  # type: ignore[arg-type]
    first = [int(y) for *_, y in stream]
    second = [int(y) for *_, y in stream]
    assert len(first) == len(stream) and sorted(first) == sorted(second)
    assert first != second  # mezcla distinta en cada época
    assert stream.class_counts is not None and int(stream.class_counts.sum()) == len(stream)
    loader = tdata.make_loader(stream, 32, shuffle=True, num_workers=0, seed=1)
    x_num, _, y = next(iter(loader))
    assert x_num.shape[0] == 32 and y.shape[0] == 32


def test_training_uses_streaming_for_large_tabular(
    workspace_dir: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Umbral en 0: cualquier tabular va por el camino streaming (el worker hereda el entorno).
    monkeypatch.setenv("PERCEPTRON_STREAMING_THRESHOLD_MB", "0")
    cfg = _config(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "run_big", max_epochs=2)
    result = run_sync(cfg, EventBus(), timeout=300)
    assert result.status == "succeeded", result.error
    assert len(result.history) == 2 and "val_loss" in result.best_metrics
