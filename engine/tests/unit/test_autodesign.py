"""Piezas puras del diseño guiado (ADR-0041, iteración 2)."""

from __future__ import annotations

from perceptron.catalog.templates import image_template, tabular_template
from perceptron.domain.enums import Origin, TaskType
from perceptron.services.autodesign import (
    Candidate,
    DesignOutcome,
    TournamentSummary,
    _pick,
    common_metric,
    estimated_tournament_s,
    tournament_epochs,
    tournament_subset,
)


def _tab(template: str, task: TaskType = TaskType.CLASSIFICATION) -> object:
    n = None if task is TaskType.REGRESSION else 2
    return tabular_template(template, task=task, num_classes=n, num_numeric=3, cardinalities=[])


def test_common_metric_is_shared_and_comparable() -> None:
    a = _tab("mlp")
    b = _tab("resnet_mlp")
    assert common_metric([a, b]) == "val_f1_macro"  # type: ignore[list-item]
    assert common_metric([a, b], "val_auroc") == "val_auroc"  # type: ignore[list-item]
    b.metrics = ["accuracy"]  # type: ignore[attr-defined]
    assert common_metric([a, b]) == "val_accuracy"  # type: ignore[list-item]
    b.metrics = []  # type: ignore[attr-defined]
    assert common_metric([a, b]) == "val_loss"  # type: ignore[list-item]
    r = [_tab("mlp", TaskType.REGRESSION), _tab("linear", TaskType.REGRESSION)]
    assert common_metric(r) == "val_mae"  # type: ignore[arg-type]
    assert common_metric([]) == "val_loss"


def test_tournament_budget_adapts_to_data_size() -> None:
    assert tournament_subset(300) == 1.0  # pocos datos: todo train en cada época
    assert tournament_subset(50_000) == 0.3
    spec = image_template("small_cnn", task=TaskType.CLASSIFICATION, num_classes=3, image_size=64)
    assert tournament_epochs(spec) >= 2
    assert estimated_tournament_s([spec, spec], [10.0, None], 1.0) is None
    assert estimated_tournament_s([spec], [10.0], 0.5) == 10.0 * tournament_epochs(spec) * 0.5


def _c(
    aid: str, score: float, *, recommended: bool = False, metric: float | None = None
) -> Candidate:
    return Candidate(
        archspec_id=aid,
        title=aid.upper(),
        rationale="r",
        origin=Origin.LLM,
        score=score,
        recommended=recommended,
        tournament_metric=metric,
    )


def test_pick_prefers_tournament_winner_with_evidence() -> None:
    a, b = _c("a", 97, recommended=True, metric=0.71), _c("b", 80, metric=0.9)
    out = DesignOutcome(pipeline_id="p", origin=Origin.LLM, candidates=[a, b])
    out.tournament = TournamentSummary(
        metric="val_f1_macro", direction="maximize", subset=1.0, winner="b"
    )
    assert _pick(out, [a, b]) is b
    assert (
        out.pick == "b" and "val_f1_macro = 0.9" in out.pick_reason and "A: 0.71" in out.pick_reason
    )


def test_pick_without_tournament_is_the_recommended() -> None:
    a, b = _c("a", 60), _c("b", 97, recommended=True)
    out = DesignOutcome(pipeline_id="p", origin=Origin.RULES, candidates=[a, b])
    assert _pick(out, [b, a]) is b and out.pick == "b" and "requisitos" in out.pick_reason
    empty = DesignOutcome(pipeline_id="p", origin=Origin.RULES)
    assert _pick(empty, []) is None and empty.pick is None
