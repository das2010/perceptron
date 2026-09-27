"""Aceptación de la Capa 1a (SPEC §14): UC-01 y UC-04 de punta a punta por CLI.

ingesta → profiling → propuesta por reglas → HPO 10 trials → evaluación en test
sellado → registro. Corre en el job `e2e` de CI (`pytest -m e2e`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perceptron.cli.main import app

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def _quickstart(workspace: Path, source: Path, *extra: str) -> dict:  # type: ignore[type-arg]
    r = CliRunner().invoke(
        app, ["quickstart", str(source), "--trials", "10", *extra, "-w", str(workspace), "--json"]
    )
    assert r.exit_code == 0, r.stdout + str(r.exception)
    summary = json.loads(r.stdout)
    print(json.dumps(summary, indent=2, ensure_ascii=False))  # queda en el log de CI
    return summary


def test_uc01_churn(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(workspace_dir, fixtures_dir / "uc01_churn" / "churn.csv", "--max-epochs", "15")
    assert s["target"] == "churn"
    assert s["trials"] == 10
    assert s["model_version_id"]
    assert s["test_metrics"]["roc_auc"] > 0.75


def test_uc04_defects(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(workspace_dir, fixtures_dir / "uc04_defects", "--max-epochs", "20")
    assert s["modality"] == "image"
    assert s["trials"] == 10
    assert s["model_version_id"]
    assert s["test_metrics"]["accuracy"] > 0.8


def test_uc03_text(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(
        workspace_dir, fixtures_dir / "uc03_tickets_es" / "tickets.jsonl", "--max-epochs", "20"
    )
    assert s["modality"] == "text"
    assert s["architecture"] == "text-textcnn"
    assert s["trials"] == 10
    assert s["model_version_id"]
    assert s["test_metrics"]["accuracy"] > 0.9


def test_uc07_forecasting(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(
        workspace_dir, fixtures_dir / "uc07_demand" / "demanda.csv", "--max-epochs", "60"
    )
    assert s["modality"] == "timeseries"
    assert s["architecture"] == "series-nbeats"
    assert s["model_version_id"]
    assert s["test_metrics"]["smape"] < 15
    assert s["test_metrics"]["mase"] < 1


def test_uc08_anomalies(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(
        workspace_dir, fixtures_dir / "uc08_telemetry" / "telemetria.csv", "--max-epochs", "30"
    )
    assert s["modality"] == "timeseries"
    assert s["architecture"] == "series-ae_conv"
    assert s["model_version_id"]
    assert s["test_metrics"]["f1"] > 0.7
