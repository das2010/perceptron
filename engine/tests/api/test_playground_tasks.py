"""Serving de series y visión avanzada (RF-EXP-01, RF-EXP-02): entrenar poco, exportar a
ONNX, cargar el `InferenceModel` y predecir; también por el playground de la API.

Solo se verifica la forma del contrato (no la calidad: 1 trial de 1 época).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from perceptron.api.context import EngineContext
from perceptron.domain.enums import TaskType
from perceptron.export.formats import ExportRequest
from perceptron.services.workflow import Workflow, quickstart
from perceptron.serving.runtime import InferenceModel, InputError, Prediction

API = "/api/v1"
pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _exported(ctx: EngineContext, source: Path, task: TaskType | None = None) -> str:
    result = quickstart(ctx, source, task=task, trials=1, max_epochs=1)
    assert result.study_result.best_trial is not None
    run_id = result.study_result.best_trial.run_id
    report = Workflow(ctx).export(run_id, ExportRequest())
    onnx = next(a for a in report.artifacts if a.format == "onnx")
    assert onnx.error is None, onnx.error
    assert report.ok, [a.model_dump() for a in report.artifacts]
    return run_id


def _model(ctx: EngineContext, run_id: str) -> InferenceModel:
    model: InferenceModel = Workflow(ctx).inference_model(run_id)
    return model


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _image(root: Path) -> Path:
    return next(p for p in sorted(root.rglob("*.png")) if "mask" not in p.parent.name)


def _history(csv: Path) -> list[dict[str, Any]]:
    # Como llegan de un CSV o un formulario: todo texto.
    return pl.read_csv(csv, infer_schema=False).to_dicts()


# ------------------------------------------------------------------ series


def test_forecasting_serving(client: TestClient, ctx: EngineContext, fixtures_dir: Path) -> None:
    csv = fixtures_dir / "uc07_demand" / "demanda.csv"
    run_id = _exported(ctx, csv)
    model = _model(ctx, run_id)
    assert model.kind == "sequence" and model.task == "forecasting"
    assert model.required_columns == ["semana", "sku", "unidades"]
    cfg = model.series_config
    assert cfg is not None
    rows = _history(csv)
    preds = model.predict_series(rows)
    series = {r["sku"] for r in rows}
    assert len(preds) == len(series)
    for p in preds:
        assert isinstance(p.prediction, list) and len(p.prediction) == cfg.horizon
        assert all(isinstance(v, float) for v in p.prediction)
        assert p.details["series"] in series
        assert len(p.details["times"]) == cfg.horizon
        assert p.details["times"][0] > p.details["last_time"]
    first = preds[0].details
    last = max(r["semana"] for r in rows if r["sku"] == first["series"])
    assert first["last_time"] == last  # fechas sin hora: se devuelven como fechas
    # Faltan columnas o historia: errores claros.
    with pytest.raises(InputError, match="faltan columnas: unidades"):
        model.predict_series([{"semana": "2024-01-01", "sku": "A"}])
    one = [r for r in rows if r["sku"] == first["series"]][: cfg.lookback - 1]
    with pytest.raises(InputError, match=f"Hacen falta al menos {cfg.lookback} valores"):
        model.predict_series(one)

    # Playground de la API (predictSeries) y por archivo CSV.
    recent = [r for r in rows if r["sku"] == first["series"]][-cfg.lookback :]
    out = _ok(client.post(f"{API}/runs/{run_id}/predict/series", json={"rows": recent}))
    assert out["input_kind"] == "sequence" and out["task"] == "forecasting"
    assert len(out["predictions"]) == 1
    api_pred = out["predictions"][0]
    assert api_pred["prediction"] == pytest.approx(preds[0].prediction, rel=1e-5)
    assert api_pred["details"]["times"] == first["times"]
    bad = client.post(f"{API}/runs/{run_id}/predict/series", json={"rows": recent[:3]})
    assert bad.status_code == 422 and "Hacen falta" in bad.text
    files = [("file", ("demanda.csv", csv.read_bytes(), "text/csv"))]
    by_file = _ok(client.post(f"{API}/runs/{run_id}/predict/file", files=files))
    assert len(by_file["predictions"]) == len(series)
    # Un modelo de series no recibe filas de tabla.
    assert client.post(f"{API}/runs/{run_id}/predict", json={"rows": recent}).status_code == 422


def test_anomaly_serving(client: TestClient, ctx: EngineContext, fixtures_dir: Path) -> None:
    csv = fixtures_dir / "uc08_telemetry" / "telemetria.csv"
    run_id = _exported(ctx, csv)
    model = _model(ctx, run_id)
    assert model.kind == "sequence" and model.task == "anomaly_detection"
    # La etiqueta de anomalía no hace falta para predecir.
    assert "anomalia" not in model.required_columns
    assert model.required_columns == ["timestamp", "temperatura", "vibracion"]  # un solo sensor
    threshold = model.signature["outputs"]["threshold"]
    assert isinstance(threshold, float)
    cfg = model.series_config
    assert cfg is not None
    rows = [{k: v for k, v in r.items() if k != "anomalia"} for r in _history(csv)][-400:]
    preds = model.predict_series(rows)
    assert preds
    for p in preds:
        d = p.details
        assert isinstance(p.prediction, int) and p.prediction == len(d["anomalies"])
        assert d["threshold"] == threshold
        assert len(d["times"]) == len(d["scores"]) > 0
        assert set(d["anomalies"]) <= set(d["times"])
        assert all(s >= 0 for s in d["scores"])
        assert "T" in d["times"][0]  # datos por minuto: fecha y hora
    out = _ok(client.post(f"{API}/runs/{run_id}/predict/series", json={"rows": rows[-200:]}))
    assert out["predictions"] and "anomalies" in out["predictions"][0]["details"]


# ------------------------------------------------------------------ visión


def test_detection_serving(client: TestClient, ctx: EngineContext, fixtures_dir: Path) -> None:
    root = fixtures_dir / "uc04_defects"
    run_id = _exported(ctx, root, TaskType.OBJECT_DETECTION)
    model = _model(ctx, run_id)
    assert model.kind == "image" and model.task == "object_detection"
    assert model.signature["outputs"]["score_threshold"] == pytest.approx(0.3)
    image = _image(root)
    with Image.open(image) as im:
        (pred,) = model.predict_images([im.copy()])
    _check_detection(pred.to_dict(), model.pipeline.classes or [])
    files = [("file", (image.name, image.read_bytes(), "image/png"))]
    out = _ok(client.post(f"{API}/runs/{run_id}/predict/file", files=files))
    assert out["input_kind"] == "image" and out["task"] == "object_detection"
    _check_detection(out["predictions"][0], model.pipeline.classes or [])
    assert client.post(f"{API}/runs/{run_id}/predict/series", json={"rows": [{}]}).status_code == (
        422
    )


def _check_detection(pred: dict[str, Any], classes: list[str]) -> None:
    objects = pred["details"]["objects"]
    assert isinstance(pred["prediction"], int) and pred["prediction"] == len(objects)
    for obj in objects:
        assert obj["label"] in classes and obj["score"] >= 0.3
        x0, y0, x1, y1 = obj["box"]
        assert 0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1


def test_segmentation_serving(ctx: EngineContext, fixtures_dir: Path) -> None:
    root = fixtures_dir / "uc05_masks"
    run_id = _exported(ctx, root)
    model = _model(ctx, run_id)
    assert model.task == "segmentation"
    with Image.open(_image(root / "images")) as im:
        original = im.copy()
    big = original.resize((original.width * 2, original.height + 7))
    preds = model.predict_images([original, big])
    for p, img in zip(preds, [original, big], strict=True):
        _check_segmentation(p, img.size, len(model.pipeline.classes or []))


def _check_segmentation(p: Prediction, size: tuple[int, int], k: int) -> None:
    assert isinstance(p.prediction, float) and 0 <= p.prediction <= 100
    fraction = p.details["fraction"]
    assert len(fraction) == max(k, 2) and sum(fraction.values()) == pytest.approx(1, abs=1e-4)
    background = next(iter(fraction.values()))
    assert p.prediction == pytest.approx(100 * (1 - background), abs=1e-2)
    mask = Image.open(io.BytesIO(base64.b64decode(p.details["mask_png"])))
    assert mask.size == size and mask.mode == "L"
    assert max(mask.getextrema()) < len(fraction)


def test_ocr_serving(ctx: EngineContext, fixtures_dir: Path) -> None:
    root = fixtures_dir / "uc06_ocr"
    run_id = _exported(ctx, root)
    model = _model(ctx, run_id)
    assert model.task == "ocr"
    alphabet = set(model.pipeline.classes or [])
    with Image.open(_image(root / "images")) as im:
        (pred,) = model.predict_images([im.copy()])
    assert isinstance(pred.prediction, str) and set(pred.prediction) <= alphabet
    assert pred.to_dict() == {"prediction": pred.prediction}
