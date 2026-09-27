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

# El límite de pytest-timeout (600 s) detecta cuelgues en tests normales; un e2e con
# 10 trials puede tardar más. El job de CI tiene su propio timeout-minutes.
pytestmark = [pytest.mark.e2e, pytest.mark.timeout(2400)]


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
    assert s["baseline"] and "roc_auc" in s["baseline"]["metrics"]  # referencia LightGBM


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


def test_uc09_audio(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(workspace_dir, fixtures_dir / "uc09_motor_audio", "--max-epochs", "30")
    assert s["modality"] == "audio"
    assert s["architecture"] == "audio-crnn"
    assert s["model_version_id"]
    assert s["test_metrics"]["accuracy"] > 0.8


def test_uc04_detection(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(
        workspace_dir,
        fixtures_dir / "uc04_defects",
        "--task",
        "object_detection",
        "--max-epochs",
        "60",
    )
    assert s["architecture"] == "vision-centernet_small"
    assert s["model_version_id"]
    assert s["test_metrics"]["map_50"] > 0.5


def test_uc05_segmentation(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(workspace_dir, fixtures_dir / "uc05_masks", "--max-epochs", "40")
    assert s["architecture"] == "vision-unet_small"
    assert s["model_version_id"]
    assert s["test_metrics"]["iou_foreground"] > 0.5


def test_uc06_ocr(workspace_dir: Path, fixtures_dir: Path) -> None:
    s = _quickstart(workspace_dir, fixtures_dir / "uc06_ocr", "--max-epochs", "60")
    assert s["architecture"] == "vision-crnn"
    assert s["model_version_id"]
    assert s["test_metrics"]["cer"] < 0.1
