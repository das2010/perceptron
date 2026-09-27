"""Aceptación de la Capa 2 con proveedores reales (SPEC §14): workflow `llm.yml`.

Con Claude y con un modelo local (Ollama) el agente completa UC-01, UC-04 y UC-09 dentro
del presupuesto, y en L1 la auditoría demuestra que no salió ningún valor individual.
El perfil sale de `PERCEPTRON_LLM__PROFILE` (anthropic | ollama); los límites, de
`PERCEPTRON_AGENT_MAX_COST` / `PERCEPTRON_AGENT_MAX_TIME`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from perceptron.api.context import EngineContext
from perceptron.cli.main import app
from perceptron.core.config import Settings
from perceptron.domain.enums import PrivacyLevel
from perceptron.domain.models import DatasetVersion, LLMCall
from perceptron.llm.privacy.audit import find_leaks, individual_values
from perceptron.services.workflow import Workflow

pytestmark = [pytest.mark.llm_live, pytest.mark.timeout(5400)]
runner = CliRunner()

CASES: dict[str, tuple[str, list[str], str, float]] = {
    "uc01": ("uc01_churn/churn.csv", ["--max-epochs", "15"], "roc_auc", 0.75),
    "uc04": ("uc04_defects", ["--max-epochs", "20"], "accuracy", 0.8),
    "uc09": ("uc09_motor_audio", ["--max-epochs", "30"], "accuracy", 0.8),
}
GOALS = {
    "uc01": "Anticipar qué clientes se van a dar de baja para priorizar la retención.",
    "uc04": "Detectar piezas defectuosas en fotos de la línea de producción.",
    "uc09": "Detectar fallas de motor (rodamiento, desbalance, cavitación) por el sonido.",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    monkeypatch.setenv("PERCEPTRON_LLM__ENABLED", "true")


@pytest.mark.parametrize("uc", sorted(CASES))
def test_agent_completes_use_case(uc: str, workspace_dir: Path, fixtures_dir: Path) -> None:
    path, extra, metric, threshold = CASES[uc]
    max_cost = float(os.environ.get("PERCEPTRON_AGENT_MAX_COST", "1.5"))
    max_time = float(os.environ.get("PERCEPTRON_AGENT_MAX_TIME", "2400"))
    r = runner.invoke(
        app,
        [
            "agent",
            "run",
            str(fixtures_dir / path),
            "--goal",
            GOALS[uc],
            "--privacy",
            "L1",
            "--max-cost",
            str(max_cost),
            "--max-time",
            str(max_time),
            "--max-trials",
            "12",
            "--max-iterations",
            "3",
            "--max-steps",
            os.environ.get("PERCEPTRON_AGENT_MAX_STEPS", "10"),
            "-w",
            str(workspace_dir),
            "--json",
            *extra,
        ],
    )
    assert r.exit_code == 0, r.stdout + str(r.exception)
    s: dict[str, Any] = json.loads(r.stdout)
    print(json.dumps(s, ensure_ascii=False, indent=2))  # queda en el log del job

    ctx = EngineContext.create(Settings(workspace_dir=workspace_dir))
    try:
        calls = list(ctx.repo(LLMCall).list(filters={"project_id": s["project_id"]}, limit=2000))
        [dv] = ctx.repo(DatasetVersion).list(filters={"project_id": s["project_id"]})
        view = Workflow(ctx).view(dv)
        leaks = find_leaks([(c.id, c.payload) for c in calls], individual_values(view))
    finally:
        ctx.close()
    report = {
        **{k: s[k] for k in ("state", "stop_reason", "iterations", "trials", "fallback")},
        "uc": uc,
        "profile": os.environ.get("PERCEPTRON_LLM__PROFILE"),
        "llm_calls": len(calls),
        "llm_cost_usd": s["llm_cost_usd"],
        "test_metrics": s["test_metrics"],
        "leaks": len(leaks),
    }
    out = os.environ.get("PERCEPTRON_REPORT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"agent-{uc}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert s["state"] == "finished", s["log"][-5:]
    assert s["test_metrics"][metric] > threshold
    assert s["llm_cost_usd"] <= max_cost
    assert calls and all(c.privacy_level is PrivacyLevel.L1 for c in calls)
    assert not leaks, leaks[:5]
