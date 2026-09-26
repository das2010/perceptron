from __future__ import annotations

import json
from pathlib import Path

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
