"""Golden tests de prompts contra proveedores reales (SPEC §15.3): workflow `llm.yml`.

Para cada propósito, una entrada fija y aserciones sobre la salida: schema válido, solo
bloques del catálogo, rangos razonables, el diagnóstico detecta el sobreajuste sembrado.
Ninguno cae a reglas: si el LLM no valida tras los reintentos, el test falla.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from perceptron.api.context import EngineContext
from perceptron.archspec.schema import ArchSpec
from perceptron.catalog.registry import blocks_for
from perceptron.core.config import LLMSettings
from perceptron.domain.enums import Modality, Origin, RunStatus, TaskType
from perceptron.domain.models import LLMCall, Project, Run
from perceptron.evaluation.evaluate import EVALUATION_DIR, EVALUATION_FILE, EvaluationReport
from perceptron.hpo.strategy import Budget, default_search_space
from perceptron.llm.config import LLMConfig
from perceptron.llm.gateway import Gateway
from perceptron.llm.privacy.audit import find_leaks, individual_values
from perceptron.services.workflow import Workflow
from perceptron.tracking.tracker import MemoryTracker
from perceptron.training.config import RESULT_FILE, RunResult

pytestmark = [pytest.mark.golden, pytest.mark.timeout(5400)]  # Ollama en CPU es lento


@pytest.fixture
def wf(ctx: EngineContext, monkeypatch: pytest.MonkeyPatch) -> Workflow:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    settings = ctx.settings.llm.model_copy(update={"enabled": True, "cache": False})
    ctx.use_llm(Gateway.from_settings(ctx.settings.model_copy(update={"llm": settings}), ctx.db))
    return Workflow(ctx, MemoryTracker())


@pytest.fixture
def uc01(wf: Workflow, fixtures_dir: Path) -> tuple[str, str, str]:
    p = wf.ctx.projects.add(
        Project(name="churn", goal="Anticipar bajas de clientes para priorizar retención")
    )
    wf.ctx.files.init_project(p)
    dv = wf.ingest(p.id, fixtures_dir / "uc01_churn" / "churn.csv")
    wf.profile(dv.id)
    return p.id, dv.id, wf.propose_pipeline(dv.id).id


def _no_leaks(wf: Workflow, project_id: str, dataset_version_id: str) -> None:
    calls = list(wf.ctx.repo(LLMCall).list(filters={"project_id": project_id}, limit=500))
    view = wf.view(wf.dataset(dataset_version_id))
    assert calls and not find_leaks([(c.id, c.payload) for c in calls], individual_values(view))


def test_architect(wf: Workflow, uc01: tuple[str, str, str]) -> None:
    pid, dv, pipe = uc01
    out = wf.roles.propose_architectures(dv, pipe, mode="llm", n=3)
    assert out.origin is Origin.LLM and out.fallback_reason is None
    assert 2 <= len(out.options) <= 4
    allowed = {b.key for b in blocks_for(wf.dataset(dv).modality or Modality.TABULAR)}
    for o in out.options:
        spec = ArchSpec.model_validate(o.record.spec)
        assert {n.block for n in spec.nodes} <= allowed and o.validation.valid
        assert o.rationale and 0 <= (o.confidence or 0) <= 1
    _no_leaks(wf, pid, dv)


def test_hpo_strategist(wf: Workflow, uc01: tuple[str, str, str]) -> None:
    _, dv, pipe = uc01
    record, _ = wf.propose_architecture(dv, pipe)
    spec = ArchSpec.model_validate(record.spec)
    budget = Budget(max_trials=15, max_epochs_per_trial=20)
    s = wf.hpo_strategy(record.id, budget, mode="llm", dataset_version_id=dv)
    assert s.origin is Origin.LLM and s.llm_call_id and s.rationale
    tunable = {p.name for p in default_search_space(spec)}
    assert {p.name for p in s.search_space} <= tunable and s.search_space
    assert 1 <= s.budget.max_trials <= 15
    assert (s.budget.max_epochs_per_trial or 0) <= 20


def test_diagnostician_detects_seeded_overfitting(wf: Workflow, uc01: tuple[str, str, str]) -> None:
    pid, dv, pipe = uc01
    record, _ = wf.propose_architecture(dv, pipe)
    run = wf.ctx.repo(Run).add(
        Run(
            project_id=pid,
            archspec_id=record.id,
            pipeline_id=pipe,
            dataset_version_id=dv,
            status=RunStatus.SUCCEEDED,
        )
    )
    train = [0.69 - 0.05 * i for i in range(14)]
    val = [0.68, 0.6, 0.52, 0.48, 0.47, 0.49, 0.53, 0.58, 0.63, 0.69, 0.75, 0.8, 0.86, 0.9]
    history = [
        {
            "epoch": float(i),
            "train_loss": t,
            "val_loss": v,
            "val_accuracy": 0.8 - 0.01 * max(i - 4, 0),
        }
        for i, (t, v) in enumerate(zip(train, val, strict=True))
    ]
    run_dir = wf.ctx.settings.paths.project(pid).run(run.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    result = RunResult(run_id=run.id, status="succeeded", epochs=14, history=history)
    (run_dir / RESULT_FILE).write_text(result.model_dump_json(), encoding="utf-8")
    d = wf.roles.diagnose(run.id, mode="llm")
    assert d.origin == "llm"
    assert "overfitting" in {p.kind for p in d.problems}
    assert d.actions


def test_reporter_and_labeler(wf: Workflow, uc01: tuple[str, str, str]) -> None:
    pid, dv, pipe = uc01
    record, _ = wf.propose_architecture(dv, pipe)
    run = wf.ctx.repo(Run).add(
        Run(
            project_id=pid,
            archspec_id=record.id,
            pipeline_id=pipe,
            dataset_version_id=dv,
            status=RunStatus.SUCCEEDED,
        )
    )
    folder = wf.ctx.settings.paths.project(pid).run(run.id) / EVALUATION_DIR
    folder.mkdir(parents=True, exist_ok=True)
    report = EvaluationReport(
        run_id=run.id,
        split="test",
        task=TaskType.CLASSIFICATION,
        num_samples=150,
        metrics={"accuracy": 0.86, "roc_auc": 0.91, "f1_macro": 0.8},
        checkpoint="best.ckpt",
    )
    (folder / EVALUATION_FILE).write_text(report.model_dump_json(), encoding="utf-8")
    out = wf.roles.report(run.id, mode="llm")
    assert out.origin == "llm" and "#" in out.markdown and out.model_card.metrics == report.metrics
    guide = wf.roles.labeling_guide(
        pid, {"baja": "el cliente cancela en 90 días", "retenido": "sigue activo"}, mode="llm"
    )
    assert {c.name for c in guide.classes} == {"baja", "retenido"}
    assert all(c.definition for c in guide.classes)
    _no_leaks(wf, pid, dv)


def test_catalog_models_match_profile() -> None:
    """El perfil activo apunta a un modelo del catálogo (IDs en configuración, D4)."""
    cfg = LLMConfig(LLMSettings())
    from perceptron.domain.enums import LLMPurpose

    res = cfg.resolve(LLMPurpose.AGENT)
    assert res is not None and res.model_id in res.provider.models
