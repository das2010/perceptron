"""Caso «Tabla X» (salida = entrada + √(2/7)·entrada^1,5) de punta a punta por API.

La forma curva de los datos exige capas ocultas; una lineal entrenada igual queda con errores
con patrón, el diagnóstico lo dice y registrarla avisa que la fórmula es mucho mejor.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.archspec.defaults import without_linear_shrinkage
from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType
from perceptron.domain.models import SymbolicFit

API = "/api/v1"


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


def test_curved_data_linear_model_is_flagged(
    client: TestClient, ctx: EngineContext, tmp_path: Path
) -> None:
    csv = tmp_path / "tabla x.csv"
    rows = "".join(f"{i},{i + (2 / 7) ** 0.5 * i**1.5!r}\n" for i in range(1, 5001, 4))
    csv.write_text("entrada,salida\n" + rows, encoding="utf-8")
    pid = _ok(client.post(f"{API}/projects", json={"name": "Tabla X"}), 201)["id"]
    src = _ok(client.post(f"{API}/projects/{pid}/sources", json={"path": str(csv)}), 201)
    dv = _ok(client.post(f"{API}/sources/{src['id']}/ingest", json={"target": "salida"}), 201)
    pipe = _ok(
        client.post(
            f"{API}/projects/{pid}/pipelines/propose", json={"dataset_version_id": dv["id"]}
        ),
        201,
    )
    # Objetivo de 1,5 a 194 000: se entrena con log(1 + y) (error relativo).
    assert pipe["graph"]["target"]["log"] is True
    assert any("órdenes de magnitud" in r for r in pipe["graph"]["rationale"])
    props = _ok(
        client.post(
            f"{API}/projects/{pid}/arch/propose",
            json={"dataset_version_id": dv["id"], "pipeline_id": pipe["id"], "mode": "rules"},
        ),
        201,
    )
    reqs = {r["code"]: r["level"] for r in props["requirements"]["items"]}
    assert reqs.get("nonlinear_capacity") == "must", reqs
    assert props["requirements"]["scenario"]["r2_curved"] > 0.9999
    assert props["requirements"]["scenario"]["deterministic"] is True
    # Datos sin ruido: la búsqueda no prueba dropout (queda en 0).
    strategy = _ok(
        client.post(
            f"{API}/projects/{pid}/hpo/strategy",
            json={
                "archspec_id": props["proposals"][0]["archspec"]["id"],
                "budget": {"max_trials": 20, "max_epochs_per_trial": 10},
                "mode": "rules",
                "dataset_version_id": dv["id"],
            },
        )
    )
    drops = [p for p in strategy["search_space"] if "dropout" in p["name"]]
    assert drops and all(p["choices"] == [0.0] for p in drops)
    best = props["proposals"][0]
    assert best["assessment"]["recommended"]
    assert [n["block"] for n in best["archspec"]["spec"]["nodes"]] != [
        "input.tabular",
        "head.linear",
    ]

    # Igual se entrena una lineal (como en el caso real).
    linear = without_linear_shrinkage(
        tabular_template(
            "linear", task=TaskType.REGRESSION, num_classes=None, num_numeric=1, cardinalities=[]
        )
    )
    arc = _ok(
        client.post(
            f"{API}/projects/{pid}/archspecs", json={"spec": linear.model_dump(mode="json")}
        ),
        201,
    )
    launch = _ok(
        client.post(
            f"{API}/projects/{pid}/studies",
            json={
                "dataset_version_id": dv["id"],
                "pipeline_id": pipe["id"],
                "archspec_id": arc["id"],
                "budget": {"max_trials": 1, "max_epochs_per_trial": 4},
            },
        ),
        202,
    )
    job = _wait_job(client, launch["job"]["id"])
    assert job["status"] == "succeeded", job["error"]
    run_id = job["result"]["best_trial"]["run_id"]

    # Sin fórmula todavía: el diagnóstico sugiere buscarla (datos casi determinísticos).
    first = _ok(client.get(f"{API}/runs/{run_id}/diagnosis", params={"refresh": True}))
    assert "try_formula" in [a["kind"] for a in first["actions"]]
    options = _ok(client.get(f"{API}/runs/{run_id}/improvements"))
    assert any(o["kind"] == "try_formula" and o["hint"] == "formula" for o in options)

    # La fórmula sugerida del proyecto (como la que encontró la búsqueda real).
    ctx.repo(SymbolicFit).add(
        SymbolicFit(
            project_id=pid,
            dataset_version_id=dv["id"],
            target="salida",
            features=["entrada"],
            expression="x1 + sqrt(14)*x1**1.5/7",
            formula="entrada + sqrt(14)·entrada^1.5/7",
            python="",
            excel_es="",
            excel_en="",
            metrics={"val": {"r2": 1.0, "rmse": 0.0009}},
        )
    )
    diagnosis = _ok(client.get(f"{API}/runs/{run_id}/diagnosis", params={"refresh": True}))
    evidence = " ".join(p["evidence"] for p in diagnosis["problems"])
    kinds = [p["kind"] for p in diagnosis["problems"]]
    assert "underfitting" in kinds and "siguen a las entradas" in evidence, diagnosis
    assert "La fórmula sugerida explica" in evidence
    assert diagnosis["actions"][0]["kind"] == "change_architecture"

    warnings = _ok(client.get(f"{API}/runs/{run_id}/registration-check"))
    assert [w["code"] for w in warnings] == ["formula_better"]
