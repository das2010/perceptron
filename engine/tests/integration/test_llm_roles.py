"""Roles del LLM sobre el flujo real (UC-01, UC-03) con FakeLLMProvider (SPEC §15.3).

El proveedor falso arma sus respuestas a partir de lo que el prompt le mandó (el payload
filtrado entre <datos>), como lo haría un LLM real; así se ejercitan la privacidad, los
reintentos con feedback, la validación de dominio, el fallback y la auditoría.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from perceptron.api.context import EngineContext
from perceptron.domain.enums import Origin, PrivacyLevel
from perceptron.domain.models import ArchSpecRecord, LLMCall, ModelVersion, Project
from perceptron.hpo.strategy import Budget
from perceptron.llm.errors import LLMUnavailableError
from perceptron.llm.providers.fake import FakeLLMProvider
from perceptron.llm.schemas import ClassGuide, LabelingGuide
from perceptron.llm.types import LLMRequest
from perceptron.services.llm_roles import REPORT_SECTIONS
from perceptron.services.workflow import Workflow
from perceptron.tracking.tracker import MemoryTracker


def datos(request: LLMRequest) -> dict[str, Any]:
    text = request.messages[0].content
    return json.loads(text.split("<datos>", 1)[1].split("</datos>", 1)[0])


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")


@pytest.fixture
def wf(ctx: EngineContext) -> Workflow:
    return Workflow(ctx, MemoryTracker())


def _uc01(wf: Workflow, fixtures_dir: Path, **project: Any) -> tuple[str, str, str]:
    p = wf.ctx.projects.add(Project(name="churn", goal="anticipar la baja de clientes", **project))
    wf.ctx.files.init_project(p)
    dv = wf.ingest(p.id, fixtures_dir / "uc01_churn" / "churn.csv")
    wf.profile(dv.id)
    pipe = wf.propose_pipeline(dv.id)
    return p.id, dv.id, pipe.id


def _architect(request: LLMRequest) -> dict[str, Any]:
    base = datos(request)["constraints"]["base_archspec"]
    wide = json.loads(json.dumps(base))
    wide["name"] = "mlp-ancho"
    return {
        "proposals": [
            {
                "title": "MLP de reglas",
                "archspec": {**base, "name": "mlp-base"},
                "rationale": "Punto de partida sólido para tabular chico.",
                "pros": ["rápido"],
                "cons": ["poca capacidad"],
                "confidence": 0.7,
            },
            {
                "title": "MLP ancho",
                "archspec": wide,
                "rationale": "Más capacidad por si hay interacciones.",
                "confidence": 0.5,
            },
        ]
    }


def test_architect_retries_then_saves_llm_proposals(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    pid, dv, pipe = _uc01(wf, fixtures_dir)

    def invalid(request: LLMRequest) -> dict[str, Any]:
        out = _architect(request)
        out["proposals"][1]["archspec"]["nodes"][0]["block"] = "bloque.inexistente"
        return out

    fake_llm.script("architect", invalid, _architect)
    res = wf.roles.propose_architectures(dv, pipe, mode="llm", n=2)
    assert res.origin is Origin.LLM and res.fallback_reason is None
    assert [o.title for o in res.options] == ["MLP de reglas", "MLP ancho"]
    second = fake_llm.calls("architect")[1]
    assert "bloque.inexistente" in second.messages[-1].content  # feedback del validador
    for o in res.options:
        record = wf.ctx.repo(ArchSpecRecord).get(o.record.id)
        assert record.origin is Origin.LLM and o.validation.valid
        assert record.spec and record.spec["provenance"]["llm_call_id"] == res.llm_call_id
        assert o.estimates["num_params"] and o.estimates["epoch_time_s"] is not None
    sent = datos(fake_llm.calls("architect")[0])
    assert sent["goal"] == "anticipar la baja de clientes" and sent["catalog"]
    assert "samples" not in sent  # L1
    calls = list(wf.ctx.repo(LLMCall).list(filters={"project_id": pid}))
    assert {c.status for c in calls} == {"invalid", "ok"}


def test_architect_falls_back_to_rules(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    _, dv, pipe = _uc01(wf, fixtures_dir, privacy_level=PrivacyLevel.L0)
    res = wf.roles.propose_architectures(dv, pipe)
    assert res.origin is Origin.RULES and "L0" in (res.fallback_reason or "")
    assert not fake_llm.calls()
    with pytest.raises(LLMUnavailableError):
        wf.roles.propose_architectures(dv, pipe, mode="llm")


def _strategy(request: LLMRequest) -> dict[str, Any]:
    tunable = datos(request)["constraints"]["tunable"][:2]
    space = [
        {k: p[k] for k in ("name", "type", "low", "high", "log", "choices") if p.get(k) is not None}
        for p in tunable
    ]
    return {
        "strategy": "tpe",
        "pruner": "median",
        "search_space": space,
        "objectives": [{"metric": "val_loss", "direction": "minimize"}],
        "max_trials": 2,
        "max_epochs_per_trial": 1,
        "pruner_warmup_epochs": 1,
        "rationale": "Pocos trials: TPE con dos hiperparámetros clave.",
    }


def test_hpo_strategist_diagnostician_and_reporter(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    pid, dv, pipe = _uc01(wf, fixtures_dir)
    record, _ = wf.propose_architecture(dv, pipe)

    def no_objectives(request: LLMRequest) -> dict[str, Any]:
        return {**_strategy(request), "objectives": []}  # no se puede reparar: reintento

    def unknown_param(request: LLMRequest) -> dict[str, Any]:
        out = _strategy(request)
        out["search_space"].append({"name": "inventado", "type": "float", "low": 0, "high": 1})
        return out  # se repara: el parámetro inexistente se descarta

    fake_llm.script("hpo_strategist", no_objectives, unknown_param)
    budget = Budget(max_trials=3, max_epochs_per_trial=2)
    strategy = wf.hpo_strategy(record.id, budget, mode="auto", dataset_version_id=dv)
    assert strategy.origin is Origin.LLM and strategy.llm_call_id
    assert strategy.budget.max_trials == 2 and strategy.budget.max_epochs_per_trial == 1
    assert all(p.default is not None for p in strategy.search_space)  # trial 0 = plantilla
    assert "objectives" in fake_llm.calls("hpo_strategist")[1].messages[-1].content
    assert "inventado" not in {p.name for p in strategy.search_space}
    assert strategy.rationale and "[Sistema:" in strategy.rationale

    study, result = wf.run_study(pid, dv, pipe, record.id, strategy)
    assert study.origin is Origin.LLM and result.best_trial is not None
    run_id = result.best_trial.run_id

    fake_llm.script(
        "diagnostician",
        {
            "summary": "Converge bien.",
            "actions": [{"kind": "change_hparam", "target": "inventado", "rationale": "x"}],
        },
        {
            "summary": "Converge bien; pocas épocas.",
            "problems": [
                {
                    "kind": "stopped_too_early",
                    "severity": "low",
                    "evidence": "val_loss baja en la última época",
                    "explanation": "Faltan épocas.",
                }
            ],
            "actions": [{"kind": "more_epochs", "rationale": "Más épocas."}],
        },
    )
    diagnosis = wf.roles.diagnose(run_id)
    assert diagnosis.origin == "llm" and diagnosis.actions[0].kind == "more_epochs"
    sent = datos(fake_llm.calls("diagnostician")[0])
    assert sent["runs"][0]["history"] and "evidence" in sent
    from perceptron.domain.models import Run

    assert wf.ctx.repo(Run).get(run_id).diagnosis == diagnosis.model_dump(mode="json")

    wf.evaluate(run_id)
    mv = wf.register(run_id)
    fake_llm.script(
        "reporter",
        {
            "title": "Informe churn",
            "summary": "El modelo anticipa bajas.",
            "markdown": "# Informe\n\n"
            + "\n\n".join(f"## {s}\n\nx" for s in REPORT_SECTIONS["español"]),
            "model_card": {
                "intended_use": "priorizar retención",
                "data": "clientes (agregados)",
                "training": "MLP",
                "metrics": {"accuracy": 99.0},
            },
        },
    )
    report = wf.roles.report(run_id)
    evaluation = wf.evaluation_report(run_id)
    assert report.origin == "llm" and report.model_card.metrics == evaluation.metrics
    run_dir = wf.ctx.settings.paths.project(pid).run(run_id)
    assert (run_dir / "report" / "report.md").read_text(encoding="utf-8").startswith("# Informe")
    assert wf.ctx.repo(ModelVersion).get(mv.id).model_card["intended_use"] == "priorizar retención"


def test_rules_report_and_diagnosis_without_llm(wf: Workflow, fixtures_dir: Path) -> None:
    pid, dv, pipe = _uc01(wf, fixtures_dir)
    record, _ = wf.propose_architecture(dv, pipe)
    strategy = wf.hpo_strategy(record.id, Budget(max_trials=1, max_epochs_per_trial=1))
    _, result = wf.run_study(pid, dv, pipe, record.id, strategy)
    assert result.best_trial is not None
    run_id = result.best_trial.run_id
    assert wf.roles.diagnose(run_id).origin == "rules"
    wf.evaluate(run_id)
    report = wf.roles.report(run_id)
    assert report.origin == "rules" and "| accuracy |" in report.markdown


def test_labeler_guide_and_prelabel_uc03(
    wf: Workflow, fake_llm: FakeLLMProvider, fixtures_dir: Path
) -> None:
    p = wf.ctx.projects.add(Project(name="tickets", privacy_level=PrivacyLevel.L3))
    wf.ctx.files.init_project(p)
    dv = wf.ingest(p.id, fixtures_dir / "uc03_tickets_es" / "tickets.jsonl")
    classes = {"funcionalidad": "pedidos de funciones", "facturacion": "cobros y facturas"}
    fake_llm.script(
        "labeler",
        {
            "classes": [
                {"name": k, "definition": v, "edge_cases": ["dudoso"]} for k, v in classes.items()
            ]
        },
    )
    guide = wf.roles.labeling_guide(p.id, classes)
    assert {c.name for c in guide.classes} == set(classes)

    def label_all(request: LLMRequest) -> dict[str, Any]:
        rows = datos(request)["samples"]["filas"]
        return {
            "labels": [
                {"sample_id": r["sample_id"], "label": "funcionalidad", "confidence": 0.8}
                for r in rows
            ]
        }

    fake_llm.script("labeler", label_all)
    labelset = wf.roles.prelabel(dv.id, guide, limit=25)
    assert labelset.path and labelset.classes == list(classes)
    lines = (wf.ctx.settings.paths.project(p.id).root / labelset.path).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in lines.splitlines()]
    assert len(rows) == 25 and all(r["origin"] == "llm" for r in rows)
    assert len(fake_llm.calls("labeler")) == 1 + 2  # guía + dos lotes de 20

    l1 = wf.ctx.projects.update(
        wf.ctx.projects.get(p.id).model_copy(update={"privacy_level": PrivacyLevel.L1})
    )
    assert l1.privacy_level is PrivacyLevel.L1
    with pytest.raises(LLMUnavailableError):
        wf.roles.prelabel(dv.id, LabelingGuide(classes=[ClassGuide(name="a", definition="b")]))
