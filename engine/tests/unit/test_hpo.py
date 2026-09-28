from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from perceptron.catalog.templates import audio_template, tabular_template
from perceptron.domain.enums import TaskType
from perceptron.hpo.recommend import recommend_strategy
from perceptron.hpo.strategy import (
    Budget,
    HPOStrategy,
    Objective,
    SearchParam,
    default_search_space,
)
from perceptron.hpo.study import StudyControl, run_study
from perceptron.training.config import RunConfig, RunEvent, RunResult


class FakeHandle:
    """Run simulado: val_loss depende de lr y hidden de forma conocida (óptimo lr=0.01)."""

    def __init__(self, cfg: RunConfig, *, fail: bool = False, sleep: float = 0.0) -> None:
        self.cfg = cfg
        self.fail = fail
        self.sleep = sleep
        self._prune = False
        self._stop = False

    def prune(self) -> None:
        self._prune = True

    def stop(self) -> None:
        self._stop = True

    def wait(
        self, on_event: Callable[[RunEvent], None] | None = None, timeout: float | None = None
    ) -> RunResult:
        emit = on_event or (lambda e: None)
        o = self.cfg.overrides
        hidden = float(o.get("hidden", 64))  # type: ignore[arg-type]
        emit(
            RunEvent(
                event="started",
                run_id=self.cfg.run_id,
                ts=time.time(),
                data={"num_params": hidden * 100},
            )
        )
        if self.fail:
            return RunResult(run_id=self.cfg.run_id, status="failed", error={"message": "boom"})
        lr = float(o.get("lr", 1e-3))  # type: ignore[arg-type]
        base = (math.log10(lr) + 2) ** 2 + 0.1 + 10 / hidden
        history = []
        epochs = self.cfg.max_epochs or 5
        for e in range(epochs):
            if self.sleep:
                time.sleep(self.sleep)
            loss = base * (1 + 1 / (e + 1))
            history.append(loss)
            emit(
                RunEvent(
                    event="epoch",
                    run_id=self.cfg.run_id,
                    ts=time.time(),
                    epoch=e,
                    metrics={"val_loss": loss},
                )
            )
            if self._prune:
                return RunResult(run_id=self.cfg.run_id, status="pruned", epochs=e + 1)
            if self._stop:
                return RunResult(run_id=self.cfg.run_id, status="cancelled", epochs=e + 1)
        return RunResult(
            run_id=self.cfg.run_id,
            status="succeeded",
            epochs=epochs,
            best_metrics={"val_loss": min(history)},
        )


def _base(tmp: Path) -> RunConfig:
    return RunConfig(
        run_id="std",
        run_dir=tmp / "runs" / "std",
        dataset_dir=tmp,
        archspec={},
        pipeline={},
        max_epochs=5,
    )


SPACE = [
    SearchParam(name="lr", type="float", low=1e-4, high=1e-1, log=True),
    SearchParam(name="hidden", type="int", low=16, high=256, log=True),
]


def _run(tmp: Path, strategy: HPOStrategy, launcher: Any = FakeHandle, **kw: Any):  # type: ignore[no-untyped-def]
    return run_study(
        strategy,
        _base(tmp),
        storage=tmp / "hpo" / "optuna.db",
        study_name="s",
        launcher=launcher,
        **kw,
    )


def test_tpe_finds_good_region(tmp_path: Path) -> None:
    s = HPOStrategy(strategy="tpe", pruner="none", search_space=SPACE, budget=Budget(max_trials=15))
    res = _run(tmp_path, s)
    assert res.stop_reason == "max_trials"
    assert len(res.trials) == 15
    assert res.best_trial is not None and res.best_trial.values
    assert 1e-3 < float(res.best_trial.params["lr"]) < 1e-1


def test_first_trial_is_template_default(tmp_path: Path) -> None:
    space = [
        SearchParam(name="lr", type="float", low=1e-4, high=1e-1, log=True, default=2e-3),
        SearchParam(name="hidden", type="int", low=16, high=256, default=999),
    ]
    s = HPOStrategy(strategy="tpe", pruner="none", search_space=space, budget=Budget(max_trials=3))
    res = _run(tmp_path, s)
    first = min(res.trials, key=lambda t: t.number)
    assert first.params["lr"] == pytest.approx(2e-3)
    assert 16 <= int(first.params["hidden"]) <= 256  # default fuera de rango: lo sugiere TPE
    again = _run(tmp_path, s.model_copy(update={"budget": Budget(max_trials=4)}))
    assert sum(t.params["lr"] == pytest.approx(2e-3) for t in again.trials) >= 1
    assert len(again.trials) == 4  # al reanudar no se vuelve a encolar


def test_default_search_space_keeps_template_defaults() -> None:
    spec = audio_template("crnn", task=TaskType.CLASSIFICATION, num_classes=4, bins=64, frames=101)
    defaults = {p.name: p.default for p in default_search_space(spec)}
    assert defaults["lr"] == pytest.approx(2e-3) and defaults["width"] == 16


def test_target_value_stops_early(tmp_path: Path) -> None:
    s = HPOStrategy(
        strategy="random",
        pruner="none",
        search_space=SPACE,
        budget=Budget(max_trials=50, target_value=5.0),
    )
    res = _run(tmp_path, s)
    assert res.stop_reason == "target_reached"
    assert len(res.trials) < 50


def test_time_budget(tmp_path: Path) -> None:
    s = HPOStrategy(
        strategy="random",
        pruner="none",
        search_space=SPACE,
        budget=Budget(max_trials=100, max_time_s=0.3),
    )
    res = _run(tmp_path, s, launcher=lambda cfg: FakeHandle(cfg, sleep=0.03))
    assert res.stop_reason == "max_time"
    assert 1 <= len(res.trials) < 100


def test_median_pruning_cuts_bad_trials(tmp_path: Path) -> None:
    s = HPOStrategy(
        strategy="random", pruner="median", search_space=SPACE, budget=Budget(max_trials=20)
    )
    res = _run(tmp_path, s)
    states = [t.state for t in res.trials]
    assert "pruned" in states and "complete" in states


def test_resume_counts_previous_trials(tmp_path: Path) -> None:
    s = HPOStrategy(strategy="tpe", pruner="none", search_space=SPACE, budget=Budget(max_trials=3))
    first = _run(tmp_path, s)
    assert len(first.trials) == 3
    calls: list[str] = []

    def launcher(cfg: RunConfig) -> FakeHandle:
        calls.append(cfg.run_id)
        return FakeHandle(cfg)

    again = _run(tmp_path, s.model_copy(update={"budget": Budget(max_trials=5)}), launcher=launcher)
    assert len(again.trials) == 5
    assert len(calls) == 2  # solo los faltantes
    assert calls[0].endswith("t003")


def test_grid_is_exhaustive(tmp_path: Path) -> None:
    space = [
        SearchParam(name="hidden", type="categorical", choices=[32, 128]),
        SearchParam(name="lr", type="categorical", choices=[0.001, 0.01]),
    ]
    s = HPOStrategy(
        strategy="grid", pruner="none", search_space=space, budget=Budget(max_trials=10)
    )
    res = _run(tmp_path, s)
    assert res.stop_reason == "exhausted"
    assert sorted((t.params["hidden"], t.params["lr"]) for t in res.trials) == [
        (32, 0.001),
        (32, 0.01),
        (128, 0.001),
        (128, 0.01),
    ]


def test_multi_objective_pareto(tmp_path: Path) -> None:
    s = HPOStrategy(
        strategy="nsga2",
        pruner="none",
        search_space=SPACE,
        objectives=[Objective(metric="val_loss"), Objective(metric="num_params")],
        budget=Budget(max_trials=12),
    )
    res = _run(tmp_path, s)
    assert res.pareto_front
    assert all(t.values and len(t.values) == 2 for t in res.pareto_front)


def test_failed_trials_do_not_stop_the_study(tmp_path: Path) -> None:
    n = {"i": 0}

    def launcher(cfg: RunConfig) -> FakeHandle:
        n["i"] += 1
        return FakeHandle(cfg, fail=n["i"] == 2)

    s = HPOStrategy(
        strategy="random", pruner="none", search_space=SPACE, budget=Budget(max_trials=4)
    )
    res = _run(tmp_path, s, launcher=launcher)
    assert [t.state for t in res.trials].count("fail") == 1
    assert res.best_trial is not None


def test_cancel(tmp_path: Path) -> None:
    control = StudyControl()

    def on_trial_end(record: Any, cfg: Any, result: Any) -> None:
        if record.number == 1:
            control.cancel()

    s = HPOStrategy(
        strategy="random", pruner="none", search_space=SPACE, budget=Budget(max_trials=10)
    )
    res = _run(tmp_path, s, control=control, on_trial_end=on_trial_end)
    assert res.stop_reason == "cancelled"
    assert len(res.trials) == 2


def test_strategy_validation() -> None:
    with pytest.raises(ValueError, match="multi-objetivo"):
        HPOStrategy(
            strategy="grid", pruner="none", objectives=[Objective(), Objective(metric="num_params")]
        )
    with pytest.raises(ValueError, match="log"):
        SearchParam(name="x", type="float", low=0, high=1, log=True)


def test_default_space_and_recommendation() -> None:
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[4]
    )
    space = {p.name: p for p in default_search_space(spec)}
    assert {"hidden", "layers", "dropout", "lr", "weight_decay", "label_smoothing"} <= set(space)
    assert "epochs" not in space
    assert space["hidden"].log and space["lr"].log
    assert space["lr"].low == pytest.approx(1e-4) and space["lr"].high == pytest.approx(1e-2)

    assert recommend_strategy(spec, Budget(max_trials=1)).strategy == "single"
    rec = recommend_strategy(spec, Budget(max_trials=20, max_epochs_per_trial=30))
    assert rec.strategy == "tpe" and rec.pruner == "median"  # 20 trials: ASHA recién con 30
    assert rec.pruner_warmup_epochs == 10
    big = recommend_strategy(spec, Budget(max_trials=40, max_epochs_per_trial=30))
    assert big.pruner == "asha"
    assert rec.objectives[0].metric == "val_loss" and rec.objectives[0].direction == "minimize"
    assert rec.rationale
    multi = recommend_strategy(
        spec, Budget(max_trials=20), extra_objectives=[Objective(metric="num_params")]
    )
    assert multi.strategy == "nsga2"


def test_balanced_sampler_equalizes_classes() -> None:
    import torch

    from perceptron.training.data import balanced_sampler

    class DS(torch.utils.data.Dataset):  # type: ignore[type-arg]
        y = torch.tensor([0] * 90 + [1] * 10)

        def __len__(self) -> int:
            return 100

    sampler = balanced_sampler(DS(), torch.Generator().manual_seed(0))
    assert sampler is not None
    drawn = torch.tensor(list(iter(sampler)))
    minority_share = (DS.y[drawn] == 1).float().mean().item()
    assert 0.35 < minority_share < 0.65  # ≈ 50 % en vez de 10 %


def test_parallel_trials_one_per_gpu(tmp_path: Path) -> None:
    """RF-HPO-04: con varias GPUs corre un trial por GPU a la vez, cada uno en la suya."""
    import threading

    lock = threading.Lock()
    state = {"now": 0, "peak": 0}
    seen_gpus: list[int | None] = []

    class Counting(FakeHandle):
        def wait(self, on_event=None, timeout=None):  # type: ignore[no-untyped-def]
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
                seen_gpus.append(self.cfg.gpu_index)
            try:
                return super().wait(on_event, timeout)
            finally:
                with lock:
                    state["now"] -= 1

    s = HPOStrategy(
        strategy="random", pruner="none", search_space=SPACE, budget=Budget(max_trials=6)
    )
    res = _run(tmp_path, s, launcher=lambda cfg: Counting(cfg, sleep=0.05), gpus=[0, 1])
    assert len(res.trials) == 6 and res.stop_reason == "max_trials"
    assert state["peak"] == 2  # nunca más trials que GPUs
    assert set(seen_gpus) == {0, 1} and seen_gpus.count(0) == seen_gpus.count(1)
    assert res.best_trial is not None


def test_parallel_cancel_stops_every_running_trial(tmp_path: Path) -> None:
    import threading

    control = StudyControl()
    started: list[str] = []

    def launcher(cfg: RunConfig) -> FakeHandle:
        started.append(cfg.run_id)
        return FakeHandle(cfg, sleep=0.1)

    s = HPOStrategy(
        strategy="random",
        pruner="none",
        search_space=SPACE,
        budget=Budget(max_trials=10),
        parallelism=3,
    )
    threading.Timer(0.15, control.cancel).start()  # con los tres primeros en curso
    res = _run(tmp_path, s, launcher=launcher, control=control)
    assert res.stop_reason == "cancelled"
    assert len(started) == 3  # no se lanzaron más
    assert all(t.state == "fail" and t.error == "cancelado" for t in res.trials)
