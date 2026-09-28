"""Reparación de propuestas de HPO: el sistema acota lo que puede sin decidir por el LLM."""

from __future__ import annotations

from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType
from perceptron.hpo.strategy import Budget, Objective
from perceptron.llm.schemas import HPOProposal, ProposedParam
from perceptron.services.llm_roles import HPOSpace


def _space() -> HPOSpace:
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[4]
    )
    return HPOSpace.of(spec)


def test_repair_clamps_drops_and_bounds() -> None:
    space = _space()
    lr = space.tunable["lr"]
    proposal = HPOProposal(
        strategy="tpe",
        pruner="median",
        search_space=[
            ProposedParam(name="lr", type="float", low=1e-9, high=10.0, log=True),
            ProposedParam(name="inventado", type="int", low=1, high=3),
            ProposedParam(name="weight_decay", low=1.0),  # sin high ni tipo: se completa
        ],
        objectives=[Objective(metric="val_f1_inventada", direction="maximize")],
        max_trials=500,
        rationale="x",
    )
    budget = Budget(max_trials=10)
    assert space.errors(proposal, budget)  # sin reparar: inválida
    fixed, notes = space.repair(proposal, budget)
    assert not space.errors(fixed, budget)
    p = fixed.search_space[0]
    assert p.name == "lr" and p.low == lr.low and p.high == lr.high
    assert fixed.max_trials == 10 and fixed.objectives[0].metric == "val_loss"
    assert len(notes) == 5 and "[Sistema:" in fixed.rationale
    wd = next(p for p in fixed.search_space if p.name == "weight_decay")
    ref = space.tunable["weight_decay"]
    assert wd.type == ref.type and wd.high == ref.high and wd.low == ref.high


def test_repair_keeps_valid_proposals_untouched() -> None:
    space = _space()
    lr = space.tunable["lr"]
    proposal = HPOProposal(
        strategy="random",
        pruner="none",
        search_space=[ProposedParam.model_validate({**lr.model_dump(), "default": None})],
        objectives=[Objective()],
        max_trials=5,
        rationale="ok",
    )
    fixed, notes = space.repair(proposal, Budget(max_trials=10))
    assert not notes and fixed.rationale == "ok"
