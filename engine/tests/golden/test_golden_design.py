"""Golden tests del wizard que diseña con proveedores reales (ADR-0041, `llm.yml`).

Escenarios con una respuesta de diseño conocida: el arquitecto real recibe los requisitos del
escenario y la propuesta recomendada tiene que ser la correcta para cada caso.
- «Tabla 3»: una regla para extrapolar → la recomendada es lineal.
- «Tubos»: pocas imágenes con conexión → el LLM incluye una preentrenada y queda recomendada.
- Equipo embebido: ninguna propuesta del LLM supera el tope de parámetros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from perceptron.api.context import EngineContext
from perceptron.archspec.schema import ArchSpec
from perceptron.domain.enums import Origin
from perceptron.domain.models import LLMCall, Project, ProjectDraft
from perceptron.llm.gateway import Gateway
from perceptron.services.design import EDGE_PARAMS, is_linear, uses_pretrained
from perceptron.services.llm_roles import ArchProposals
from perceptron.services.workflow import Workflow
from perceptron.tracking.tracker import MemoryTracker

pytestmark = [pytest.mark.golden, pytest.mark.timeout(1800)]


@pytest.fixture
def wf(ctx: EngineContext) -> Workflow:
    settings = ctx.settings.llm.model_copy(update={"enabled": True, "cache": False})
    ctx.use_llm(Gateway.from_settings(ctx.settings.model_copy(update={"llm": settings}), ctx.db))
    return Workflow(ctx, MemoryTracker())


def _errors(wf: Workflow, project_id: str) -> list[str]:
    calls = wf.ctx.repo(LLMCall).list(filters={"project_id": project_id}, limit=500)
    return [f"{c.purpose.value}#{c.attempt}: {c.error[:300]}" for c in calls if c.error]


def _project(wf: Workflow, name: str, goal: str, brief: dict[str, object]) -> str:
    p = wf.ctx.projects.add(Project(name=name, goal=goal))
    wf.ctx.files.init_project(p)
    wf.ctx.repo(ProjectDraft).add(ProjectDraft(project_id=p.id, values={"brief": brief}))
    return p.id


def _propose(wf: Workflow, pid: str, source: Path, **ingest: object) -> ArchProposals:
    dv = wf.ingest(pid, source, **ingest)  # type: ignore[arg-type]
    wf.profile(dv.id)
    pipe = wf.propose_pipeline(dv.id)
    out = wf.roles.propose_architectures(dv.id, pipe.id, mode="llm", n=3)
    assert out.origin is Origin.LLM, _errors(wf, pid)
    assert out.requirements is not None and out.options[0].assessment is not None
    assert out.options[0].assessment.recommended
    return out


def test_rule_to_extrapolate_recommends_linear(
    wf: Workflow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    csv = tmp_path / "tabla 3.csv"
    csv.write_text(
        "numero,multiplo\n" + "".join(f"{i},{3 * i}\n" for i in range(1, 151)), encoding="utf-8"
    )
    pid = _project(
        wf,
        "Tabla 3",
        "Descubrir la regla del múltiplo de 3",
        {"problem": "rule", "extrapolate": True},
    )
    out = _propose(wf, pid, csv, target="multiplo")
    best = ArchSpec.model_validate(out.options[0].record.spec)
    assert is_linear(best), [o.title for o in out.options]
    assert "linear_option" in {r.code for r in out.requirements.items}  # type: ignore[union-attr]


def test_few_images_recommend_a_pretrained_backbone(
    wf: Workflow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PERCEPTRON_OFFLINE", raising=False)
    rng = np.random.default_rng(0)
    root = tmp_path / "tubos"
    for k, name in enumerate(["sano", "fisura", "corrosion"]):
        (root / name).mkdir(parents=True)
        for i in range(40):
            img = rng.integers(0, 60, size=(96, 96, 3), dtype=np.uint8)
            img[:, :, k] += np.uint8(120)  # cada clase con su tono
            Image.fromarray(img).save(root / name / f"{name}_{i:03d}.png")
    pid = _project(wf, "Tubos", "Clasificar defectos en fotos de tubos", {"problem": "category"})
    out = _propose(wf, pid, root)
    codes = {r.code: r.level for r in out.requirements.items}  # type: ignore[union-attr]
    assert codes.get("pretrained_backbone") == "must", codes
    best = ArchSpec.model_validate(out.options[0].record.spec)
    assert uses_pretrained(best), [o.title for o in out.options]
    # La propuso el LLM (no hizo falta completarla con una plantilla).
    assert any(
        o.record.origin is Origin.LLM and uses_pretrained(ArchSpec.model_validate(o.record.spec))
        for o in out.options
    ), _errors(wf, pid)


def test_edge_deployment_keeps_every_proposal_small(
    wf: Workflow, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    pid = _project(
        wf,
        "churn edge",
        "Anticipar bajas en un equipo embebido",
        {"problem": "category", "deployment": "edge"},
    )
    out = _propose(wf, pid, fixtures_dir / "uc01_churn" / "churn.csv", target="churn")
    for o in out.options:
        assert (o.estimates.get("num_params") or 0) <= EDGE_PARAMS, o.title
