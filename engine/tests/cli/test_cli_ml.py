from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perceptron.catalog.templates import tabular_template
from perceptron.cli.main import app
from perceptron.domain.enums import TaskType

runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


def test_arch_validate_and_to_code(tmp_path: Path) -> None:
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[5]
    )
    f = tmp_path / "arq ñ.json"
    f.write_text(spec.model_dump_json(), encoding="utf-8")
    r = runner.invoke(app, ["arch", "validate", str(f), "--json"])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["valid"]

    bad = spec.model_dump(mode="json")
    bad["nodes"][1]["block"] = "no.existe"
    f.write_text(json.dumps(bad), encoding="utf-8")
    r = runner.invoke(app, ["arch", "validate", str(f)])
    assert r.exit_code == 1
    assert "bloque desconocido" in r.stdout

    f.write_text(spec.model_dump_json(), encoding="utf-8")
    out = tmp_path / "modelo.py"
    assert runner.invoke(app, ["arch", "to-code", str(f), "-o", str(out)]).exit_code == 0
    assert "class Model(nn.Module)" in out.read_text(encoding="utf-8")


def test_step_by_step_commands(workspace_dir: Path, fixtures_dir: Path) -> None:
    ws = ["-w", str(workspace_dir)]

    def run(*args: str) -> dict:  # type: ignore[type-arg]
        r = runner.invoke(app, [*args, *ws, "--json"])
        assert r.exit_code == 0, r.stdout + str(r.exception)
        return json.loads(r.stdout)

    pid = run("project", "create", "Paso a paso")["id"]
    dv = run("data", "ingest", str(fixtures_dir / "uc01_churn" / "churn.csv"), "-p", pid)
    card = run("data", "profile", dv["id"])
    assert card["target"]["name"] == "churn"
    pipe = run("pipeline", "propose", dv["id"])
    arch = run("arch", "propose", dv["id"], "--pipeline", pipe["id"])
    strategy = run("hpo", "strategy", arch["archspec"]["id"], "--trials", "2")
    assert strategy["budget"]["max_trials"] == 2
    trained = run(
        "train",
        "-p",
        pid,
        "--dataset",
        dv["id"],
        "--pipeline",
        pipe["id"],
        "--archspec",
        arch["archspec"]["id"],
        "--max-epochs",
        "1",
    )
    assert trained["state"] == "complete"
    report = run("eval", trained["run_id"])
    assert "accuracy" in report["metrics"]
    mv = run("model", "register", trained["run_id"])
    assert mv["stage"] == "candidate"


def test_quickstart_images(workspace_dir: Path, fixtures_dir: Path) -> None:
    r = runner.invoke(
        app,
        [
            "quickstart",
            str(fixtures_dir / "uc04_defects"),
            "--trials",
            "2",
            "--max-epochs",
            "1",
            "-w",
            str(workspace_dir),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout + str(r.exception)
    s = json.loads(r.stdout)
    assert s["modality"] == "image"
    assert s["architecture"] == "image-small_cnn"
    assert s["trials"] == 2
    assert s["model_version_id"]
    assert "accuracy" in s["test_metrics"]
