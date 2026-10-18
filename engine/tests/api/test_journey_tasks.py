"""El asistente guiado para todas las tareas del motor: pronóstico, casos raros (marcados o
no), ubicar objetos, marcar zonas y leer texto (ADR-0042)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from perceptron.domain.enums import TaskType
from perceptron.domain.models import DataSource, Project
from perceptron.evaluation.evaluate import EvaluationReport
from perceptron.services.journey import Coordinator
from perceptron.services.journey_guide import Facts, data_guide, missing, summarize
from perceptron.services.journey_talk import question, summary, understand
from perceptron.services.workflow import Workflow

FORECAST = Facts(
    data_kind="table",
    problem="forecast",
    prediction="las unidades semanales de cada producto",
    horizon=4,
    success_target=20,
    target_name="unidades",
)
ANOMALY = Facts(
    data_kind="table",
    problem="anomaly",
    prediction="fallas de los sensores",
    labels_available=True,
    success_target=70,
)
LOCATE = Facts(
    data_kind="images",
    problem="locate",
    prediction="dónde están los defectos",
    labels_available=True,
    success_target=50,
)
SEGMENT = LOCATE.model_copy(update={"problem": "segment", "prediction": "la zona del defecto"})
READ = LOCATE.model_copy(update={"problem": "read_text", "prediction": "el texto de la etiqueta"})


def test_each_task_asks_only_what_it_needs() -> None:
    assert missing(Facts(data_kind="table", problem="forecast")) == [
        "prediction",
        "horizon",
        "success_target",
    ]
    assert missing(Facts(data_kind="table", problem="anomaly")) == [
        "prediction",
        "labels_available",
        "success_target",
    ]
    assert missing(Facts(data_kind="images", problem="locate")) == [
        "prediction",
        "labels_available",
        "success_target",
    ]
    for facts in (FORECAST, ANOMALY, LOCATE, SEGMENT, READ):
        assert not missing(facts), facts


def test_questions_and_summaries_speak_the_case() -> None:
    table = Facts(data_kind="table")
    assert "pronosticar los próximos valores" in question("problem", table)
    assert "ubicar dónde están los objetos" in question("problem", Facts(data_kind="images"))
    assert "No es obligatorio" in question("labels_available", ANOMALY)
    assert "valores hacia adelante" in question("horizon", FORECAST)
    assert "querés pronosticar esto" in summary(FORECAST) and "próximos 4 valores" in summary(
        FORECAST
    )
    unmarked = ANOMALY.model_copy(update={"labels_available": False})
    assert "aprendo cómo es lo normal" in summary(unmarked)
    assert understand("horizon", "las próximas 4") == {"horizon": 4}


def test_data_guides_explain_each_format() -> None:
    assert data_guide(FORECAST).example_columns[0] == "fecha"
    assert "anomalia" in data_guide(ANOMALY).example_columns
    unmarked = data_guide(ANOMALY.model_copy(update={"labels_available": False}))
    assert "anomalia" not in unmarked.example_columns
    assert any("No hace falta marcar" in t for t in unmarked.tips)
    assert "anotaciones.json" in data_guide(LOCATE).example_tree
    assert "masks/foto_001.png" in data_guide(SEGMENT).example_tree
    assert data_guide(READ).example_columns == ["file_name", "texto"]


def _report(task: TaskType, metrics: dict[str, float], **details: Any) -> EvaluationReport:
    return EvaluationReport(
        run_id="r", task=task, num_samples=40, metrics=metrics, details=details, checkpoint="c"
    )


def test_results_are_explained_in_plain_words() -> None:
    r = summarize(_report(TaskType.FORECASTING, {"smape": 12.0, "skill_vs_naive": 0.3}), FORECAST)
    assert r["meets_target"] and "12.0 %" in r["explanation"][0] and "30 %" in r["explanation"][1]
    r = summarize(
        _report(
            TaskType.ANOMALY_DETECTION, {"recall": 0.8, "precision": 0.6, "anomaly_rate": 0.02}
        ),
        ANOMALY,
    )
    assert r["meets_target"] and "80 %" in r["explanation"][0]
    r = summarize(_report(TaskType.ANOMALY_DETECTION, {"anomaly_rate": 0.01}), ANOMALY)
    assert r["measurable"] is False and "no puedo medir" in r["explanation"][1]
    r = summarize(_report(TaskType.OBJECT_DETECTION, {"map_50": 0.62}), LOCATE)
    assert r["meets_target"] and "62 %" in r["explanation"][0]
    r = summarize(_report(TaskType.SEGMENTATION, {"iou_foreground": 0.4}), SEGMENT)
    assert not r["meets_target"] and "40 %" in r["explanation"][0]
    r = summarize(_report(TaskType.OCR, {"cer": 0.05, "exact_match": 0.7}), READ)
    assert r["meets_target"] and "95 %" in r["explanation"][0] and "70 %" in r["explanation"][1]


# ---------------------------------------------------------------------- hasta la revisión


def _wait(c: Coordinator, pid: str, timeout: float = 300) -> Any:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        j = c.get(pid)
        if j.state not in {"reviewing_data", "training"}:
            return j
        time.sleep(0.2)
    raise AssertionError("el recorrido no terminó")


def _reviewed(ctx: Any, facts: Facts, path: Path) -> Any:
    p = ctx.projects.add(Project(name=f"caso {facts.problem}"))
    ctx.files.init_project(p)
    c = Coordinator(Workflow(ctx))
    j = c.interview(p.id, 1, "Quiero armar un modelo", facts)
    j = c.confirm(p.id, j.version)
    src = ctx.repo(DataSource).add(
        DataSource(
            type="folder" if path.is_dir() else "file",
            project_id=p.id,
            name=path.name,
            config={"path": str(path)},
        )
    )
    c.prepare(p.id, j.version, src.id)
    return _wait(c, p.id)


@pytest.mark.parametrize(
    ("facts", "fixture"),
    [
        (FORECAST, "uc07_demand/demanda.csv"),
        (ANOMALY, "uc08_telemetry/telemetria.csv"),
        (LOCATE, "uc04_defects"),
        (SEGMENT, "uc05_masks"),
        (READ, "uc06_ocr"),
    ],
    ids=["pronostico", "casos-raros", "ubicar", "zonas", "texto"],
)
def test_each_task_reads_its_data_and_is_ready_to_train(
    ctx: Any, fixtures_dir: Path, facts: Facts, fixture: str
) -> None:
    j = _reviewed(ctx, facts, fixtures_dir / fixture)
    assert j.state == "ready_to_train", j.blockers
    assert j.messages[-1].action == "train" and "Revisé tus datos" in j.messages[-1].text
    if facts.problem in ("forecast", "anomaly"):
        assert j.data_summary["series"]["count"] >= 1


def test_unmarked_anomalies_learn_what_normal_looks_like(
    ctx: Any, fixtures_dir: Path, tmp_path: Path
) -> None:
    data = pl.read_csv(fixtures_dir / "uc08_telemetry" / "telemetria.csv").drop("anomalia")
    path = tmp_path / "telemetría sin marcas.csv"
    data.write_csv(path)
    j = _reviewed(ctx, ANOMALY.model_copy(update={"labels_available": False}), path)
    assert j.state == "ready_to_train", j.blockers
    assert j.data_summary["series"]["count"] >= 1


def test_wrong_format_is_explained(ctx: Any, fixtures_dir: Path) -> None:
    # Fotos sin marcas cuando se quiere ubicar objetos: se explica qué falta.
    j = _reviewed(ctx, LOCATE, fixtures_dir / "uc05_masks")
    assert j.state == "awaiting_data" and "marcas de los objetos" in j.blockers[0]
    # Una tabla sin fechas cuando se quiere pronosticar.
    j = _reviewed(ctx, FORECAST, fixtures_dir / "uc01_churn" / "churn.csv")
    assert j.state == "awaiting_data" and "mediciones en el tiempo" in j.blockers[0]


# ---------------------------------------------------------------------- de punta a punta


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presupuesto mínimo para el test: el recorrido es el mismo."""
    from perceptron.hpo.plan import plan_budget
    from perceptron.services import autodesign

    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    monkeypatch.setattr(autodesign, "TOURNAMENT_BUDGET_S", 0)

    def small(spec: Any, **kw: Any) -> Any:
        return plan_budget(spec, **kw).model_copy(
            update={"max_trials": 1, "max_epochs_per_trial": 2}
        )

    monkeypatch.setattr(autodesign, "plan_budget", small)


def _trained(ctx: Any, facts: Facts, path: Path) -> tuple[Coordinator, Any]:
    j = _reviewed(ctx, facts, path)
    assert j.state == "ready_to_train", j.blockers
    c = Coordinator(Workflow(ctx))
    c.train(_pid(ctx, facts), j.version)
    return c, _wait(c, _pid(ctx, facts), 900)


def test_forecast_journey_trains_validates_and_forecasts(
    client: Any, ctx: Any, fixtures_dir: Path, tiny: None
) -> None:
    c, j = _trained(ctx, FORECAST, fixtures_dir / "uc07_demand" / "demanda.csv")
    assert j.state in {"ready_to_validate", "needs_improvement"}, (j.error, j.stage)
    assert j.result["kind"] == "forecasting" and j.form.kind == "series"
    assert j.form.horizon and j.form.history
    rows = pl.read_csv(fixtures_dir / "uc07_demand" / "demanda.csv").to_dicts()
    forecast = c.predict(c.contract(_pid(ctx, FORECAST)), rows=rows)
    assert forecast and len(forecast[0].prediction) == j.form.horizon
    # Por la API: historia en CSV → próximos valores; datos nuevos → comparación automática.
    csv = (fixtures_dir / "uc07_demand" / "demanda.csv").read_bytes()
    url = f"/api/v1/projects/{_pid(ctx, FORECAST)}/journey"
    r = client.post(f"{url}/predict/series", files={"file": ("historia.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    first = r.json()["predictions"][0]
    assert len(first["prediction"]) == j.form.horizon and first["details"]["times"]
    r = client.post(f"{url}/validate/file", files={"file": ("nuevos.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["checked"] >= 1 and "error_percent" in body["journey"]["validations"][-1]


def _pid(ctx: Any, facts: Facts) -> str:
    return next(p.id for p in ctx.projects.list() if p.name == f"caso {facts.problem}")


def test_unmarked_anomaly_journey_trains_and_flags_moments(
    ctx: Any, fixtures_dir: Path, tmp_path: Path, tiny: None
) -> None:
    data = pl.read_csv(fixtures_dir / "uc08_telemetry" / "telemetria.csv").drop("anomalia")
    path = tmp_path / "telemetria.csv"
    data.write_csv(path)
    facts = ANOMALY.model_copy(update={"labels_available": False})
    c, j = _trained(ctx, facts, path)
    # Sin marcas no se puede medir cuántos detecta: se valida igual, con casos conocidos.
    assert j.state == "ready_to_validate", (j.error, j.stage)
    assert j.result["measurable"] is False and j.form.kind == "series"
    found = c.predict(c.contract(_pid(ctx, facts)), rows=data.tail(400).to_dicts())
    assert found and isinstance(found[0].prediction, int)
    j = c.record_validation(_pid(ctx, facts), "periodo-1", "sí", found[0].prediction, j.model_id)
    assert j.validations[-1]["correct"] is True and j.validations[-1]["actual"] == "bien"


@pytest.mark.parametrize(
    ("facts", "fixture", "task"),
    [
        (LOCATE, "uc04_defects", "object_detection"),
        (SEGMENT, "uc05_masks", "segmentation"),
        (READ, "uc06_ocr", "ocr"),
    ],
    ids=["ubicar", "zonas", "texto"],
)
def test_photo_journeys_train_and_predict(
    ctx: Any, fixtures_dir: Path, tiny: None, facts: Facts, fixture: str, task: str
) -> None:
    c, j = _trained(ctx, facts, fixtures_dir / fixture)
    assert j.state in {"ready_to_validate", "needs_improvement"}, (j.error, j.stage)
    assert j.result["kind"] == task and j.form.kind == "image" and j.form.task == task
    photo = next(
        p for p in (fixtures_dir / fixture).rglob("*") if p.suffix.lower() in (".png", ".jpg")
    )
    pred = c.predict(c.contract(_pid(ctx, facts)), file=(photo.read_bytes(), photo.suffix))
    if task == "object_detection":
        assert isinstance(pred.prediction, int) and "objects" in pred.details
    elif task == "segmentation":
        assert 0 <= pred.prediction <= 100 and pred.details["mask_png"]
    else:
        assert isinstance(pred.prediction, str)
