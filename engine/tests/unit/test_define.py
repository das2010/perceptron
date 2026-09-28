"""Sub-wizard de definición de arquitectura (SPEC §7.6, paso 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from perceptron.archspec.schema import HP
from perceptron.archspec.validate import validate_archspec
from perceptron.catalog import define
from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import Origin


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_d").ensure()


def _fitted(paths: ProjectPaths, src: Path):  # type: ignore[no-untyped-def]
    v = ingest(paths, IngestRequest(project_id="prj_d", source=src))
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    return card, fit_pipeline(propose_pipeline(card), view.read("train"))


def _recommended(plan: define.DefinePlan) -> dict[str, str]:
    return {s.step: next(o.id for o in s.options if o.recommended) for s in plan.steps}


def test_image_offline_only_scratch_and_build(paths: ProjectPaths, fixtures_dir: Path) -> None:
    card, fitted = _fitted(paths, fixtures_dir / "uc04_defects")
    plan = define.plan(card, fitted, None, {}, allow_download=False)
    family = plan.steps[0]
    assert [o.id for o in family.options if o.available] == ["scratch"]
    assert all(o.reason for o in family.options if not o.available)
    assert _recommended(plan)["family"] == "scratch"
    # Una familia no disponible no cuenta como elegida.
    assert (
        define.plan(card, fitted, None, {"family": "strong"}, allow_download=False).steps[0].choice
        is None
    )

    choices = {**_recommended(plan), "backbone": "small_cnn:64", "regularization": "high"}
    spec = define.build(card, fitted, None, choices, allow_download=False)
    assert validate_archspec(spec).valid
    encoder = next(n for n in spec.nodes if n.id == "encoder")
    assert encoder.params["width"] == HP(hp="width", default=64)
    assert next(n for n in spec.nodes if n.id == "drop").params["p"] == HP(
        hp="dropout", default=0.4
    )
    assert spec.provenance.origin is Origin.MANUAL


def test_image_pretrained_backbones_follow_family(paths: ProjectPaths, fixtures_dir: Path) -> None:
    card, fitted = _fitted(paths, fixtures_dir / "uc04_defects")
    plan = define.plan(card, fitted, None, {"family": "strong"}, allow_download=True)
    assert [o.id for o in plan.steps[1].options] == ["resnet18", "resnet50", "convnext_tiny"]
    spec = define.build(
        card,
        fitted,
        None,
        {"family": "strong", "backbone": "resnet18", "head": "focal", "regularization": "high"},
        allow_download=True,
    )
    encoder = next(n for n in spec.nodes if n.id == "encoder")
    assert encoder.params["model"] == "resnet18" and encoder.params["pretrained"] is True
    assert spec.training.freeze_backbone_epochs == 3 and spec.loss.type == "focal"


def test_tabular_plan_and_incomplete_build(paths: ProjectPaths, fixtures_dir: Path) -> None:
    card, fitted = _fitted(paths, fixtures_dir / "uc01_churn" / "churn.csv")
    plan = define.plan(card, fitted, None, {})
    rec = _recommended(plan)
    assert rec["family"] == "mlp" and rec["backbone"] == "small"  # 400 filas
    spec = define.build(card, fitted, None, {**rec, "family": "ft_transformer"})
    assert validate_archspec(spec).valid
    assert next(n for n in spec.nodes if n.id == "encoder").params["d_token"] == HP(
        hp="d_token", default=32
    )
    with pytest.raises(ValueError, match="faltan elecciones"):
        define.build(card, fitted, None, {"family": "mlp"})
