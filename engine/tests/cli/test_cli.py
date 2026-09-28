from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perceptron import __version__
from perceptron.cli.main import app

runner = CliRunner()


def test_version() -> None:
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert r.stdout.strip() == f"perceptron {__version__}"


def test_project_create_and_list(workspace_dir: Path) -> None:
    ws = str(workspace_dir)
    r = runner.invoke(
        app,
        [
            "project",
            "create",
            "Demanda semanal",
            "-w",
            ws,
            "--modality",
            "timeseries",
            "--task",
            "forecasting",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout
    created = json.loads(r.stdout)
    assert created["modalities"] == ["timeseries"]

    r = runner.invoke(app, ["project", "list", "-w", ws, "--json"])
    assert r.exit_code == 0
    assert [p["id"] for p in json.loads(r.stdout)] == [created["id"]]


def test_openapi_export(tmp_path: Path) -> None:
    out = tmp_path / "api con espacio" / "openapi.json"
    r = runner.invoke(app, ["openapi", "-o", str(out)])
    assert r.exit_code == 0
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert schema["info"]["title"] == "Perceptron Engine API"
    assert "/api/v1/system/health" in schema["paths"]


def test_legacy_windows_console_does_not_abort_on_unicode(monkeypatch: pytest.MonkeyPatch) -> None:
    """cp1252 (consola clásica de Windows): «→» no entra y antes abortaba el comando."""
    import io
    import sys

    from perceptron.cli.main import _tolerant_console

    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    _tolerant_console()
    print("sobrevivir N → N+1 con acentos: ñ")
    console.flush()
    assert raw.getvalue().decode("cp1252") == "sobrevivir N ? N+1 con acentos: ñ\n"
