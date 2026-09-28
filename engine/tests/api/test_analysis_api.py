"""Evaluación avanzada por la API: análisis de errores (RF-EVL-03) y fairness (RF-EVL-04)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

API = "/api/v1"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _ok(r: Any, code: int = 200) -> Any:
    assert r.status_code == code, r.text
    return r.json()


def _wait_job(client: TestClient, job_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _ok(client.get(f"{API}/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("el job no terminó a tiempo")


@pytest.fixture
def evaluated_run(client: TestClient, fixtures_dir: Path) -> str:
    pid = _ok(client.post(f"{API}/projects", json={"name": "Análisis"}), 201)["id"]
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources",
            json={"path": str(fixtures_dir / "uc01_churn" / "churn.csv")},
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "churn"}), 201)
    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    prop = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": prop["archspec"]["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": 3},
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id: str = job["result"]["best_trial"]["run_id"]
    # Sin evaluación no hay predicciones por muestra.
    assert client.get(f"{API}/runs/{run_id}/errors").status_code == 422
    _ok(client.post(f"{API}/runs/{run_id}/evaluate"))
    return run_id


def test_error_analysis(client: TestClient, evaluated_run: str) -> None:
    report = _ok(client.get(f"{API}/runs/{evaluated_run}/evaluation"))
    errors = _ok(client.get(f"{API}/runs/{evaluated_run}/errors"))
    assert errors["metric"] == "accuracy"
    assert errors["num_samples"] == report["num_samples"]
    assert abs(errors["overall"] - report["metrics"]["accuracy"]) < 1e-6
    assert errors["num_errors"] == round((1 - errors["overall"]) * errors["num_samples"])
    assert len(errors["samples"]) == min(errors["num_errors"], 200)
    for s in errors["samples"]:
        assert s["actual"] != s["predicted"]
        assert "customer_id" not in s["features"] or s["features"]["customer_id"].startswith("C")
        assert "churn" not in s["features"]
    for sl in errors["slices"]:
        assert sl["metric"] < errors["overall"] and sl["support"] >= 10
    assert sum(c["count"] for c in errors["confusions"]) == errors["num_errors"]


def test_fairness_by_region_and_age(client: TestClient, evaluated_run: str) -> None:
    reports = _ok(
        client.post(
            f"{API}/runs/{evaluated_run}/fairness",
            json={"attributes": ["region", "edad"], "threshold": 0.1},
        )
    )
    by_attr = {r["attribute"]: r for r in reports}
    region = by_attr["region"]
    assert region["positive_class"] == "1"
    rates = [g["selection_rate"] for g in region["groups"]]
    assert region["demographic_parity_difference"] == pytest.approx(max(rates) - min(rates))
    assert len(by_attr["edad"]["groups"]) >= 2  # numérica continua: cuartiles
    if region["demographic_parity_difference"] > 0.1:
        assert any("Paridad demográfica" in a for a in region["alerts"])
    bad = client.post(f"{API}/runs/{evaluated_run}/fairness", json={"attributes": ["nope"]})
    assert bad.status_code == 422


def test_tabular_explanations(client: TestClient, evaluated_run: str, fixtures_dir: Path) -> None:
    import csv

    glob = _ok(client.get(f"{API}/runs/{evaluated_run}/explain"))
    assert glob["method"] == "shapley_sampling" and glob["samples"] > 0
    imps = [f["importance"] for f in glob["features"]]
    assert imps == sorted(imps, reverse=True) and imps[0] > 0 and min(imps) >= 0
    # Cacheada: la segunda lectura devuelve lo mismo.
    assert _ok(client.get(f"{API}/runs/{evaluated_run}/explain")) == glob

    with (fixtures_dir / "uc01_churn" / "churn.csv").open(encoding="utf-8") as f:
        row = dict(next(csv.DictReader(f)))
    row.pop("churn")
    local = _ok(client.post(f"{API}/runs/{evaluated_run}/explain/row", json={"row": row}))
    assert local["prediction"] in {"0", "1"}
    assert {c["feature"] for c in local["contributions"]} == {
        f["feature"] for f in glob["features"]
    }
    attrs = [abs(c["attribution"]) for c in local["contributions"]]
    assert attrs == sorted(attrs, reverse=True)


def test_image_explanation(client: TestClient, fixtures_dir: Path) -> None:
    import base64
    import io

    from PIL import Image

    pid = _ok(client.post(f"{API}/projects", json={"name": "Defectos"}), 201)["id"]
    src = _ok(
        client.post(
            f"{API}/projects/{pid}/sources", json={"path": str(fixtures_dir / "uc04_defects")}
        ),
        201,
    )
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={}), 201)
    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    prop = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"]},
        ),
        201,
    )["proposals"][0]
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": prop["archspec"]["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": 2},
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id = job["result"]["best_trial"]["run_id"]
    image = next((fixtures_dir / "uc04_defects" / "defect").glob("*.png"))
    res = _ok(
        client.post(
            f"{API}/runs/{run_id}/explain/image",
            files={"file": (image.name, image.read_bytes(), "image/png")},
        )
    )
    assert res["method"] == "integrated_gradients" and res["prediction"] in {"defect", "ok"}
    heat = Image.open(io.BytesIO(base64.b64decode(res["heatmap_png"])))
    assert heat.size == Image.open(image).size
    # Global solo para tabular (por ahora).
    assert client.get(f"{API}/runs/{run_id}/explain").status_code == 422


def test_robustness(client: TestClient, evaluated_run: str) -> None:
    rob = _ok(client.get(f"{API}/runs/{evaluated_run}/robustness"))
    assert rob["metric"] == "accuracy" and rob["higher_is_better"]
    kinds = {r["kind"] for r in rob["results"]}
    assert kinds == {"ruido gaussiano", "categorías cambiadas", "valores faltantes"}
    assert len(rob["results"]) == 9
    for r in rob["results"]:
        assert 0.0 <= r["metric"] <= 1.0
        assert r["degradation"] == pytest.approx(rob["baseline"] - r["metric"])
    # Determinístico y cacheado.
    assert _ok(client.get(f"{API}/runs/{evaluated_run}/robustness")) == rob


def test_report_documents(client: TestClient, evaluated_run: str) -> None:
    # Con fairness ya calculado, el informe lo incluye.
    _ok(client.post(f"{API}/runs/{evaluated_run}/fairness", json={"attributes": ["region"]}))
    _ok(client.get(f"{API}/runs/{evaluated_run}/explain"))
    html = client.get(f"{API}/runs/{evaluated_run}/report/document?format=html")
    assert html.status_code == 200 and html.headers["content-type"].startswith("text/html")
    text = html.text
    assert "Titillium Web" in text and "data:font/woff2;base64," in text
    assert "Equidad entre grupos" in text and "Qué variables pesan más" in text
    assert "<script" not in text.lower()
    pdf = client.get(f"{API}/runs/{evaluated_run}/report/document?format=pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert "attachment" in pdf.headers["content-disposition"]
    md = client.get(f"{API}/runs/{evaluated_run}/report/document?format=md")
    assert md.status_code == 200 and md.text.strip()
    bad = client.get(f"{API}/runs/{evaluated_run}/report/document?format=docx")
    assert bad.status_code == 422
