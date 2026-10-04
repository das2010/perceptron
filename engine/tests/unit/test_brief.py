"""Wizard adaptativo (ADR-0040): ficha del caso, hechos de los datos y plan compilado."""

from __future__ import annotations

import numpy as np
import polars as pl

from perceptron.domain.enums import TaskType
from perceptron.services.brief import (
    CONDITIONAL,
    PLAN_VERSION,
    STEPS,
    BriefChange,
    BriefPatch,
    DataFacts,
    UseCaseBrief,
    _collinear_pairs,
    apply_patch,
    compile_plan,
    diff_plans,
    validate_patch,
)

BASE = [s for s in STEPS if s not in CONDITIONAL]


def _facts(**kw: object) -> DataFacts:
    base: dict[str, object] = {
        "dataset_version_id": "dsv_x",
        "modality": "tabular",
        "rows": 397,
        "split_strategy": "stratified",
        "target": "y",
        "target_task": TaskType.REGRESSION,
    }
    return DataFacts.model_validate({**base, **kw})


def _defaults(plan: object) -> dict[str, str]:
    return {d.key: d.value for d in plan.defaults}  # type: ignore[attr-defined]


def _checks(plan: object) -> dict[str, str]:
    return {c.code: c.severity for c in plan.checks}  # type: ignore[attr-defined]


def test_unknown_case_keeps_the_standard_plan() -> None:
    plan = compile_plan(None, None)
    assert [s.id for s in plan.steps] == BASE and not plan.adapted
    assert plan.version == PLAN_VERSION
    assert not plan.defaults and not plan.checks


def test_labeling_is_skipped_when_labels_exist() -> None:
    plan = compile_plan(None, _facts())
    assert "labeling" not in [s.id for s in plan.steps]
    assert plan.skipped[0].id == "labeling" and "«y»" in (plan.skipped[0].reason or "")


def test_tabla_3_rule_to_extrapolate() -> None:
    brief = UseCaseBrief(problem="rule", extrapolate=True, prediction="el múltiplo de 3")
    plan = compile_plan(brief, _facts(numeric_inputs=["numero"]))
    assert _defaults(plan) == {
        "task": "regression",
        "target_metric": "val_mae",
        "architecture_hint": "linear_first",
    }
    arch = next(s for s in plan.steps if s.id == "architecture")
    assert arch.reason and "regla" in arch.reason
    assert "collinear_inputs" not in _checks(plan)


def test_sensores_collinear_inputs_are_flagged_before_training() -> None:
    brief = UseCaseBrief(problem="rule", independent_inputs=True)
    facts = _facts(numeric_inputs=["Sensor1", "Sensor2"], collinear_pairs=[["Sensor1", "Sensor2"]])
    plan = compile_plan(brief, facts)
    check = next(c for c in plan.checks if c.code == "collinear_inputs")
    assert check.severity == "high" and check.step == "data"
    assert "Dijiste que las entradas son independientes" in check.message
    # Sin declararlas independientes, sigue siendo un aviso (la regla igual no se infiere).
    assert _checks(compile_plan(UseCaseBrief(problem="rule"), facts))["collinear_inputs"] == (
        "warning"
    )


def test_bearing_failures_optimize_recall() -> None:
    brief = UseCaseBrief(
        problem="category", error_costs="false_negative_worse", error_cost_ratio=10
    )
    plan = compile_plan(brief, _facts(target_task=TaskType.CLASSIFICATION, target_classes=2))
    metric = next(d for d in plan.defaults if d.key == "target_metric")
    assert metric.value == "val_recall_macro" and "10× peor" in metric.reason


def test_imbalance_without_declared_costs_uses_f1() -> None:
    facts = _facts(target_task=TaskType.CLASSIFICATION, imbalance_ratio=0.05)
    assert _defaults(compile_plan(None, facts))["target_metric"] == "val_f1_macro"


def test_split_checks_follow_the_brief() -> None:
    brief = UseCaseBrief(problem="value", has_time=True, has_entities=True)
    checks = _checks(compile_plan(brief, _facts(split_strategy="random")))
    assert checks["split_not_temporal"] == "warning" and checks["split_not_group"] == "warning"
    assert "split_not_temporal" not in _checks(
        compile_plan(brief, _facts(split_strategy="temporal"))
    )


def test_out_of_catalog_and_mismatch() -> None:
    other = compile_plan(UseCaseBrief(problem="other", problem_other="recomendar películas"), None)
    assert _checks(other) == {"out_of_catalog": "info"}
    assert [s.id for s in other.steps] == BASE
    mismatch = compile_plan(
        UseCaseBrief(problem="value"), _facts(target_task=TaskType.CLASSIFICATION)
    )
    assert _checks(mismatch)["task_mismatch"] == "warning"


def test_patch_validation_and_origins() -> None:
    bad = BriefPatch(changes=[BriefChange(field="problem", value="magia", rationale="x")])
    assert validate_patch(bad) and "problem" in (validate_patch(bad) or "")
    ok = BriefPatch(changes=[BriefChange(field="extrapolate", value=True, rationale="lo dijo")])
    assert validate_patch(ok) is None
    brief = apply_patch(UseCaseBrief(problem="rule"), ok.changes, "llm")
    assert brief.extrapolate is True and brief.origins == {"extrapolate": "llm"}


def test_collinearity_is_measured_on_ranks() -> None:
    s1 = np.linspace(0.01, 1, 60)
    df = pl.DataFrame({"S1": s1, "S2": np.sqrt(s1), "ruido": np.random.default_rng(0).random(60)})
    assert _collinear_pairs(df, ["S1", "S2", "ruido"]) == [["S1", "S2"]]


# ------------------------------------------------------------------ fase 2


def test_rule_adds_the_formula_step_after_quality() -> None:
    plan = compile_plan(UseCaseBrief(problem="rule"), _facts(numeric_inputs=["numero"]))
    ids = [s.id for s in plan.steps]
    assert ids.index("formula") == ids.index("quality") + 1
    assert "fórmula" in (plan.steps[ids.index("formula")].reason or "")
    # Sin entradas numéricas (p. ej. imágenes) no hay fórmula que buscar.
    images = _facts(modality="image", numeric_inputs=[], target_task=TaskType.CLASSIFICATION)
    assert "formula" not in [s.id for s in compile_plan(UseCaseBrief(problem="rule"), images).steps]


def test_asymmetric_binary_costs_add_the_threshold_step() -> None:
    brief = UseCaseBrief(
        problem="category", error_costs="false_negative_worse", error_cost_ratio=10
    )
    binary = _facts(target_task=TaskType.CLASSIFICATION, target_classes=2)
    ids = [s.id for s in compile_plan(brief, binary).steps]
    assert ids.index("threshold") == ids.index("task") + 1
    multi = compile_plan(brief, _facts(target_task=TaskType.CLASSIFICATION, target_classes=4))
    assert "threshold" not in [s.id for s in multi.steps]
    assert _checks(multi)["threshold_multiclass"] == "info"
    assert "threshold" not in [
        s.id for s in compile_plan(UseCaseBrief(problem="category"), binary).steps
    ]


def test_dates_are_reconciled_with_the_brief() -> None:
    facts = _facts(datetime_columns=["fecha"])
    assert _checks(compile_plan(None, facts))["dates_undeclared"] == "info"
    assert _checks(compile_plan(UseCaseBrief(has_time=False), facts))["dates_contradiction"] == (
        "warning"
    )
    assert not {"dates_undeclared", "dates_contradiction"} & set(
        _checks(compile_plan(UseCaseBrief(has_time=True), facts))
    )
    assert (
        _checks(compile_plan(UseCaseBrief(labels_available=False), _facts()))[
            "labels_contradiction"
        ]
        == "info"
    )


def test_plan_diff_lists_what_changed() -> None:
    before = compile_plan(UseCaseBrief(problem="value"), _facts(numeric_inputs=["x"]))
    after = compile_plan(
        UseCaseBrief(problem="rule", extrapolate=True), _facts(numeric_inputs=["x"])
    )
    diff = diff_plans(before, after)
    assert diff.added_steps == ["formula"] and "architecture_hint" in diff.changed_defaults
    assert diff_plans(after, after).empty and diff_plans(None, after).empty


def test_llm_field_guide_covers_every_brief_field() -> None:
    """La guía corta del prompt (modelos chicos) describe exactamente los campos editables."""
    from typing import get_args

    from perceptron.services.brief import BRIEF_GUIDE, BriefField

    assert set(BRIEF_GUIDE) == set(get_args(BriefField))


def test_patch_values_are_normalized_to_field_types() -> None:
    """Los modelos chicos devuelven «true» o «5» como texto: se normalizan al tipo del campo."""
    from perceptron.services.brief import normalize_patch

    patch = BriefPatch(
        changes=[
            BriefChange(field="extrapolate", value="true", rationale="x"),
            BriefChange(field="error_cost_ratio", value="5", rationale="x"),
            BriefChange(field="problem", value="rule", rationale="x"),
        ]
    )
    assert [c.value for c in normalize_patch(patch).changes] == [True, 5.0, "rule"]
