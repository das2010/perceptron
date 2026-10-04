"""Golden tests del wizard adaptativo con proveedores reales (ADR-0040, `llm.yml`).

La entrevista entiende casos distintos y la reconciliación detecta contradicciones entre la
ficha y los datos. Las aserciones son sobre la estructura, no sobre la redacción.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from perceptron.api.context import EngineContext
from perceptron.domain.models import Project
from perceptron.llm.gateway import Gateway
from perceptron.services.wizard import Wizard
from perceptron.services.workflow import Workflow
from perceptron.tracking.tracker import MemoryTracker

pytestmark = [pytest.mark.golden, pytest.mark.timeout(1800)]


@pytest.fixture
def wizard(ctx: EngineContext, monkeypatch: pytest.MonkeyPatch) -> Wizard:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    settings = ctx.settings.llm.model_copy(update={"enabled": True, "cache": False})
    ctx.use_llm(Gateway.from_settings(ctx.settings.model_copy(update={"llm": settings}), ctx.db))
    return Wizard(Workflow(ctx, MemoryTracker()))


def _changes(wizard: Wizard, text: str) -> tuple[dict[str, object], str]:
    """Cambios propuestos por campo y, para el mensaje de error, todo lo que devolvió el LLM."""
    project = wizard.ctx.projects.add(Project(name="caso"))
    patch, _ = wizard.intake(project.id, text, [])
    return {c.field: c.value for c in patch.changes}, patch.model_dump_json()


def test_intake_understands_a_failure_detection_case(wizard: Wizard) -> None:
    got, raw = _changes(
        wizard,
        "Quiero saber a partir del sonido si un rodamiento va a fallar. Tengo grabaciones "
        "etiquetadas como 'bien' o 'falla'. No detectar una falla es mucho peor que una falsa "
        "alarma, como diez veces peor.",
    )
    assert got.get("problem") in ("category", "anomaly"), raw
    assert got.get("error_costs") == "false_negative_worse", raw


def test_intake_understands_a_rule_to_extrapolate(wizard: Wizard) -> None:
    got, raw = _changes(
        wizard,
        "Tengo una tabla con un número y su múltiplo de 3. Quiero que el sistema descubra la "
        "regla y después usarla con números mucho más grandes que los de la tabla.",
    )
    assert got.get("problem") == "rule" and got.get("extrapolate") is True, raw


def test_intake_flags_out_of_catalog_problems(wizard: Wizard) -> None:
    got, raw = _changes(
        wizard,
        "Quiero recomendarle películas a cada usuario según lo que vio antes, como Netflix.",
    )
    assert got.get("problem") == "other", raw


def test_reconcile_detects_dependent_inputs(wizard: Wizard, tmp_path: Path) -> None:
    """Caso «Sensores»: la ficha dice entradas independientes, los datos dicen lo contrario."""
    s1 = np.linspace(0.01, 1, 120)
    path = tmp_path / "sensores.csv"
    pl.DataFrame(
        {"S1": s1, "S2": np.sqrt(s1), "Salida": s1 * (1 + np.sqrt(np.sqrt(s1)))}
    ).write_csv(path)
    project = wizard.ctx.projects.add(Project(name="sensores"))
    wizard.ctx.files.init_project(project)
    dv = wizard.wf.ingest(project.id, path)
    draft = wizard.get(project.id)
    wizard.update(
        project.id,
        version=draft.version,
        values={
            "dataset_version_id": dv.id,
            "brief": {"problem": "rule", "independent_inputs": True},
        },
    )
    patch, _ = wizard.reconcile(project.id)
    fields = {c.field: c.value for c in patch.changes}
    assert fields.get("independent_inputs") is False or (
        patch.next_question and "S2" in patch.next_question
    ), patch.model_dump_json()
