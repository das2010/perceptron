"""Series temporales (Capa 1b, sub-hito 2): detección, split, ventanas, bloques y métricas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

from perceptron.archspec.builder import build_model
from perceptron.archspec.to_code import archspec_to_code
from perceptron.archspec.validate import validate_archspec
from perceptron.catalog.rules import recommend
from perceptron.catalog.templates import series_template
from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.pipeline.series_windows import anomaly_windows, forecast_windows
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.series import SERIES_KEY, autocorr, detect_season
from perceptron.data.splits import SPLIT_COLUMN
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView, Purpose
from perceptron.domain.enums import Modality, TaskType
from perceptron.tasks.series import AnomalyAdapter, forecast_report
from perceptron.training.data import make_dataset


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_ts").ensure()


def _view(paths: ProjectPaths, src: Path, **kw: Any) -> tuple[Any, DatasetView]:
    v = ingest(paths, IngestRequest(project_id="prj_ts", source=src, **kw))
    return v, DatasetView(paths.dataset(v.content_hash))


def test_autocorr_and_season() -> None:
    t = np.arange(400)
    x = np.sin(2 * np.pi * t / 7) + 0.01 * t
    assert autocorr(x, 7) > 0.9
    assert detect_season(x, 86_400) == 7  # diaria → semanal


def test_detect_forecasting_uc07(paths: ProjectPaths, fixtures_dir: Path) -> None:
    v, view = _view(paths, fixtures_dir / "uc07_demand" / "demanda.csv")
    assert v.modality is Modality.TIMESERIES
    cfg = view.series
    assert cfg is not None
    assert (cfg.time_column, cfg.series_id, cfg.target) == ("semana", "sku", "unidades")
    assert cfg.task is TaskType.FORECASTING
    assert cfg.horizon == 8 and cfg.lookback >= 16
    df = view.read(purpose=Purpose.FINAL_EVALUATION)
    # Split temporal dentro de cada serie: test al final
    for _, group in df.group_by(SERIES_KEY):
        splits = group.sort("semana")[SPLIT_COLUMN].to_list()
        assert splits[-1] == "test" and splits[0] == "train"
        assert splits.index("val") < splits.index("test")
        assert splits.count("test") >= 2 * cfg.horizon


def test_detect_anomaly_uc08(paths: ProjectPaths, fixtures_dir: Path) -> None:
    v, view = _view(paths, fixtures_dir / "uc08_telemetry" / "telemetria.csv")
    cfg = view.series
    assert v.modality is Modality.TIMESERIES and cfg is not None
    assert cfg.task is TaskType.ANOMALY_DETECTION
    assert cfg.target == "anomalia"
    assert set(cfg.covariates) == {"temperatura", "vibracion"}
    test = view.read("test", purpose=Purpose.FINAL_EVALUATION)
    assert test["anomalia"].sum() > 0  # el test tiene anomalías para evaluar


def test_series_overrides(paths: ProjectPaths, fixtures_dir: Path) -> None:
    _, view = _view(
        paths, fixtures_dir / "uc07_demand" / "demanda.csv", series_overrides={"horizon": 4}
    )
    assert view.series is not None and view.series.horizon == 4


def test_profile_pipeline_and_windows(paths: ProjectPaths, fixtures_dir: Path) -> None:
    _, view = _view(paths, fixtures_dir / "uc07_demand" / "demanda.csv")
    card = profile_dataset(view)
    assert card.series is not None
    assert card.series.num_series == 5
    assert card.series.freq_seconds == pytest.approx(604_800)
    assert card.series.gaps == 0
    spec = propose_pipeline(card)
    assert spec.series is not None and spec.series.calendar
    fitted = fit_pipeline(spec, view.read("train"))
    state = fitted.series_state
    assert state is not None and len(state["stats"]) == 5 and len(state["mase_scale"]) == 5

    cfg = spec.series.config
    full = view.read(purpose=Purpose.FINAL_EVALUATION)
    w = forecast_windows(cfg, state, full, "test")
    assert w.x.shape[1:] == (cfg.lookback, 1 + 2)  # target + calendario (año sin/cos)
    assert w.y.shape[1] == cfg.horizon
    assert len(set(w.series)) == 5
    # Desescalar recupera los valores reales
    y_real = w.y * w.std[:, None] + w.mean[:, None]
    assert y_real.min() > 0
    # Train no usa filas de test
    train_w = forecast_windows(cfg, state, full, "train")
    assert len(train_w.y) > len(w.y)

    rec = recommend(card, fitted)
    assert rec.template == "nbeats"
    assert validate_archspec(rec.spec).valid

    ds = make_dataset(view, fitted, "val", train=False)
    assert len(ds) > 0  # type: ignore[arg-type]
    x, y, *_ = ds[0]
    assert x.shape == (cfg.lookback, 3) and y.shape == (cfg.horizon,)


def test_anomaly_windows_and_datasets(paths: ProjectPaths, fixtures_dir: Path) -> None:
    _, view = _view(paths, fixtures_dir / "uc08_telemetry" / "telemetria.csv")
    card = profile_dataset(view)
    spec = propose_pipeline(card)
    fitted = fit_pipeline(spec, view.read("train"))
    cfg = spec.series.config  # type: ignore[union-attr]
    w = anomaly_windows(
        cfg, fitted.series_state or {}, view.read(purpose=Purpose.FINAL_EVALUATION), "test"
    )
    assert w.x.shape[1:] == (cfg.lookback, 2)
    assert w.label.sum() > 0 and not w.normal.all()
    train = make_dataset(view, fitted, "train", train=True)
    assert all(bool(train[i][2]) for i in range(0, len(train), 50))  # type: ignore[arg-type]  # solo normales
    evalset = make_dataset(view, fitted, "test", train=False, purpose=Purpose.FINAL_EVALUATION)
    assert len(evalset) == len(w.label)  # type: ignore[arg-type]
    rec = recommend(card, fitted)
    assert rec.template == "ae_conv"
    assert validate_archspec(rec.spec).valid


@pytest.mark.parametrize("backbone", ["nbeats", "lstm", "gru", "tcn", "patchtst"])
def test_forecast_templates_forward_and_code(backbone: str) -> None:
    spec = series_template(backbone, task=TaskType.FORECASTING, lookback=24, channels=3, horizon=6)
    assert validate_archspec(spec).valid, validate_archspec(spec).feedback()
    built = build_model(spec)
    x = torch.randn(4, 24, 3)
    assert built.model(x).shape == (4, 6)
    ns: dict[str, Any] = {"__name__": "gen"}
    exec(compile(archspec_to_code(spec), "<gen>", "exec"), ns)  # noqa: S102 - código propio
    gen = ns["Model"]()
    gen.load_state_dict({k.removeprefix("blocks."): v for k, v in built.model.state_dict().items()})
    built.model.eval()
    gen.eval()
    with torch.no_grad():
        assert torch.allclose(built.model(x), gen(x), atol=1e-5)


@pytest.mark.parametrize("backbone", ["ae_conv", "ae_lstm"])
def test_autoencoder_templates(backbone: str) -> None:
    spec = series_template(backbone, task=TaskType.ANOMALY_DETECTION, lookback=30, channels=2)
    assert validate_archspec(spec).valid
    assert build_model(spec).model(torch.randn(3, 30, 2)).shape == (3, 30, 2)
    bad = spec.model_dump(mode="json")
    bad["input"]["shape"] = [30, 3]
    bad["nodes"][0]["block"] = "seq.rnn"
    bad["nodes"][0]["params"] = {}
    assert not validate_archspec(bad).valid  # la salida no reconstruye la entrada


def test_forecast_report_and_naive() -> None:
    y = np.array([[10.0, 12.0], [20.0, 22.0]])
    pred = y + 1
    naive = y + 4
    rep = forecast_report(y, pred, naive, np.array([2.0, 2.0]), ["a", "b"])
    assert rep.metrics["mae"] == pytest.approx(1.0)
    assert rep.metrics["mase"] == pytest.approx(0.5)
    assert rep.metrics["naive_mae"] == pytest.approx(4.0)
    assert rep.metrics["skill_vs_naive"] == pytest.approx(0.75)
    assert set(rep.detail["smape_per_series"]) == {"a", "b"}
    assert len(rep.curves["mae_per_horizon"]) == 2


def test_anomaly_calibration_and_metrics() -> None:
    class Identityish(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)  # score = energía de la ventana

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, (200, 10, 2)).astype(np.float32)
    labels = np.zeros(200, dtype=np.int64)
    x[-20:] += 3.0
    labels[-20:] = 1
    ds = torch.utils.data.TensorDataset(
        torch.tensor(x), torch.tensor(labels), torch.tensor(labels == 0)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    adapter = AnomalyAdapter()
    spec = series_template("ae_conv", task=TaskType.ANOMALY_DETECTION, lookback=10, channels=2)
    calib = adapter.calibrate(Identityish(), loader, spec)
    assert calib["method"] == "best_f1_val"
    preds = adapter.predict(Identityish(), loader, spec, None)  # type: ignore[arg-type]
    rep = adapter.evaluate(preds, spec, None, calib)  # type: ignore[arg-type]
    assert rep.metrics["f1"] == pytest.approx(1.0)
    assert rep.metrics["roc_auc"] == pytest.approx(1.0)
    assert rep.curves["score_histogram"]


def test_non_series_tables_stay_tabular(paths: ProjectPaths) -> None:
    df = pl.DataFrame(
        {
            "fecha": pl.date_range(pl.date(2026, 1, 1), pl.date(2026, 1, 10), eager=True),
            "y": list(range(10)),
        }
    )
    src = paths.root / "corta.csv"
    df.write_csv(src)
    v, _ = _view(paths, src)
    assert v.modality is Modality.TABULAR  # muy pocos puntos para una serie
