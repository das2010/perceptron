"""Plantillas de proyecto por caso de uso (RF-PRJ-02)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from perceptron.catalog.project_templates import PROJECT_TEMPLATES
from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType
from perceptron.hpo.recommend import objective_metric, recommend_strategy, training_metrics
from perceptron.hpo.strategy import Budget

API = "/api/v1"


def test_lists_one_template_per_use_case(client: TestClient) -> None:
    templates = client.get(f"{API}/projects/templates").json()
    assert [t["use_case"] for t in templates] == [f"UC-0{i}" for i in range(1, 10)]
    assert all(t["modalities"] and t["task"] and t["target_metric"] for t in templates)


def test_template_presets_what_the_user_did_not_choose(client: TestClient) -> None:
    churn = client.post(f"{API}/projects", json={"name": "Bajas", "template": "churn"}).json()
    assert churn["template"] == "churn"
    assert churn["modalities"] == ["tabular"]
    assert churn["task"] == "classification"
    assert churn["target_metric"] == "val_auroc"
    mine = client.post(
        f"{API}/projects",
        json={"name": "Propia", "template": "churn", "target_metric": "val_f1_macro"},
    ).json()
    assert mine["target_metric"] == "val_f1_macro"  # lo elegido se respeta
    bad = client.post(f"{API}/projects", json={"name": "x", "template": "no-existe"})
    assert bad.status_code == 422


def test_template_metrics_are_logged_by_training() -> None:
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[4]
    )
    logged = training_metrics(spec)
    assert {"val_loss", "val_auroc", "val_f1_macro"} <= logged
    churn = next(t for t in PROJECT_TEMPLATES if t.id == "churn")
    assert objective_metric(spec, churn.target_metric) == "val_auroc"
    assert objective_metric(spec, "val_no_existe") is None  # cae al default de la arquitectura
    strategy = recommend_strategy(spec, Budget(max_trials=5), metric="val_auroc")
    assert strategy.objectives[0].metric == "val_auroc"
    assert strategy.objectives[0].direction == "maximize"
