"""Visualizaciones del HPO (RF-HPO-06): historia, importancia, coordenadas y Pareto."""

from __future__ import annotations

import itertools
import math

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import Project, Run, Study
from perceptron.hpo.strategy import HPOStrategy, Objective, SearchParam

SPACE = [
    SearchParam(name="lr", type="float", low=1e-4, high=1e-1, log=True),
    SearchParam(name="hidden", type="int", low=16, high=256),
]


def _study(ctx: EngineContext, strategy: HPOStrategy, trials: list[tuple[dict, dict]]) -> str:
    project = ctx.projects.add(Project(name="HPO"))
    study = ctx.repo(Study).add(
        Study(project_id=project.id, name="s", strategy=strategy.model_dump(mode="json"))
    )
    for params, metrics in trials:
        ctx.repo(Run).add(
            Run(
                project_id=project.id,
                study_id=study.id,
                archspec_id="arc_x",
                pipeline_id="pip_x",
                dataset_version_id="dsv_x",
                status="succeeded",
                hyperparams=params,
                metrics=metrics,
            )
        )
    return study.id


def test_history_best_so_far_and_importance(client: TestClient, ctx: EngineContext) -> None:
    strategy = HPOStrategy(search_space=SPACE, objectives=[Objective(metric="val_loss")])
    trials = []
    for i in range(20):
        lr = 10 ** (-4 + 3 * i / 19)
        hidden = 16 + (i * 37) % 240
        # La pérdida depende del LR (mínimo cerca de 1e-2) y casi nada de `hidden`.
        loss = (math.log10(lr) + 2) ** 2 + 0.001 * (hidden % 7)
        trials.append(({"lr": lr, "hidden": hidden}, {"val_loss": loss}))
    sid = _study(ctx, strategy, trials)
    data = client.get(f"/api/v1/studies/{sid}/analysis").json()
    assert data["params"] == ["hidden", "lr"]
    history = [t["best_so_far"] for t in data["trials"]]
    assert len(history) == 20
    assert all(b <= a for a, b in itertools.pairwise(history))  # nunca empeora
    importance = data["importance"]
    assert set(importance) <= {"lr", "hidden"}
    if importance:  # PED-ANOVA necesita suficientes trials buenos
        assert importance["lr"] >= importance.get("hidden", 0)


def test_pareto_front_for_two_objectives(client: TestClient, ctx: EngineContext) -> None:
    strategy = HPOStrategy(
        strategy="nsga2",
        pruner="none",
        search_space=SPACE,
        objectives=[
            Objective(metric="val_loss", direction="minimize"),
            Objective(metric="val_accuracy", direction="maximize"),
        ],
    )
    points = [(0.2, 0.90), (0.3, 0.95), (0.4, 0.80), (0.25, 0.85)]
    trials = [({"lr": 0.01, "hidden": 64}, {"val_loss": a, "val_accuracy": b}) for a, b in points]
    sid = _study(ctx, strategy, trials)
    data = client.get(f"/api/v1/studies/{sid}/analysis").json()
    front = [t["values"] for t in data["trials"] if t["pareto"]]
    assert sorted(front) == [[0.2, 0.9], [0.3, 0.95]]  # (0.4, 0.8) y (0.25, 0.85) dominados
