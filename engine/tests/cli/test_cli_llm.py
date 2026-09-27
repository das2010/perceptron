"""CLI de la capa LLM: comandos `llm` y `quickstart --llm` con un cassette falso."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perceptron.cli.main import app
from perceptron.services.llm_roles import REPORT_SECTIONS

runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def test_llm_providers_and_profiles(workspace_dir: Path) -> None:
    r = runner.invoke(app, ["llm", "providers", "-w", str(workspace_dir), "--json"])
    assert r.exit_code == 0, r.stdout
    names = {p["name"] for p in json.loads(r.stdout)}
    assert {"anthropic", "ollama"} <= names
    r = runner.invoke(app, ["llm", "profiles", "-w", str(workspace_dir)])
    assert r.exit_code == 0 and "Activo: anthropic" in r.stdout
    r = runner.invoke(app, ["llm", "test", "-w", str(workspace_dir)])
    assert r.exit_code == 1 and "llm_unavailable" in r.stdout  # capa apagada en tests


def test_quickstart_llm_falls_back_and_reports(
    workspace_dir: Path, fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arquitecto y estratega devuelven basura → reglas; el informe sí sale del LLM."""
    cassette = tmp_path / "cassette.json"
    cassette.write_text(
        json.dumps(
            {
                "architect": [{"proposals": []}],
                "hpo_strategist": [{"strategy": "tpe"}],
                "reporter": [
                    {
                        "title": "Informe",
                        "summary": "ok",
                        "markdown": "# Informe\n\n"
                        + "\n\n".join(f"## {s}\n\nx" for s in REPORT_SECTIONS["español"]),
                        "model_card": {"intended_use": "x", "data": "y", "training": "z"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PERCEPTRON_LLM__ENABLED", "true")
    monkeypatch.setenv("PERCEPTRON_LLM__FAKE_CASSETTE", str(cassette))
    r = runner.invoke(
        app,
        [
            "quickstart",
            str(fixtures_dir / "uc01_churn" / "churn.csv"),
            "--llm",
            "--goal",
            "anticipar bajas",
            "--trials",
            "1",
            "--max-epochs",
            "1",
            "-w",
            str(workspace_dir),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout + str(r.exception)
    s = json.loads(r.stdout)
    assert s["architecture_origin"] == "rules" and s["llm_fallback"]
    assert s["hpo_origin"] == "rules" and s["report_origin"] == "llm"
    assert s["llm_calls"] == 3 + 3 + 1  # 3 intentos × 2 roles + informe
    assert s["model_version_id"]
