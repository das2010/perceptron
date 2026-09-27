"""`HPOStrategy` (SPEC §7.9) y espacio de búsqueda por defecto desde la ArchSpec."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from perceptron.archspec.schema import HP, ArchSpec

StrategyName = Literal["single", "random", "grid", "tpe", "cmaes", "nsga2"]
PrunerName = Literal["none", "median", "asha", "hyperband"]


class Condition(BaseModel):
    """El parámetro solo se sugiere si `param` vale `equals` (espacios condicionales)."""

    param: str
    equals: Any


class SearchParam(BaseModel):
    name: str
    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: list[Any] | None = None
    condition: Condition | None = None

    @model_validator(mode="after")
    def _check(self) -> SearchParam:
        if self.type == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name}: categorical requiere choices")
        elif self.low is None or self.high is None or self.low > self.high:
            raise ValueError(f"{self.name}: rango inválido [{self.low}, {self.high}]")
        elif self.log and self.low <= 0:
            raise ValueError(f"{self.name}: escala log requiere low > 0")
        return self

    def grid_values(self, points: int = 3) -> list[Any]:
        if self.type == "categorical":
            return list(self.choices or [])
        lo, hi = float(self.low or 0), float(self.high or 0)
        if self.type == "int":
            vals = sorted({round(lo + (hi - lo) * i / (points - 1)) for i in range(points)})
            return [int(v) for v in vals]
        if self.log:
            return [
                math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (points - 1))
                for i in range(points)
            ]
        return [lo + (hi - lo) * i / (points - 1) for i in range(points)]


class Objective(BaseModel):
    metric: str = "val_loss"
    direction: Literal["minimize", "maximize"] = "minimize"


class Budget(BaseModel):
    max_trials: int = Field(default=20, ge=1)
    max_time_s: float | None = Field(default=None, gt=0)
    target_value: float | None = Field(
        default=None, description="Corta al alcanzarlo (1er objetivo)"
    )
    max_epochs_per_trial: int | None = Field(default=None, ge=1)
    trial_timeout_s: float | None = Field(default=None, gt=0)


class HPOStrategy(BaseModel):
    strategy: StrategyName = "tpe"
    pruner: PrunerName = "median"
    search_space: list[SearchParam] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=lambda: [Objective()], min_length=1)
    budget: Budget = Field(default_factory=Budget)
    parallelism: int = Field(default=1, ge=1)
    seed: int = 42
    rationale: str | None = None

    @property
    def multi_objective(self) -> bool:
        return len(self.objectives) > 1

    @model_validator(mode="after")
    def _check(self) -> HPOStrategy:
        if self.multi_objective and self.strategy not in ("nsga2", "random", "tpe"):
            raise ValueError("multi-objetivo requiere nsga2, tpe o random")
        if self.multi_objective and self.pruner != "none":
            raise ValueError("el pruning no se aplica a estudios multi-objetivo")
        names = [p.name for p in self.search_space]
        if len(names) != len(set(names)):
            raise ValueError("parámetros repetidos en el espacio de búsqueda")
        return self


# Rangos para hiperparámetros que no pertenecen a un bloque del catálogo.
_GLOBAL_RANGES: dict[str, dict[str, Any]] = {
    "lr": {"type": "float", "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
    "label_smoothing": {"type": "float", "low": 0.0, "high": 0.2},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
}
NOT_TUNED = {"epochs"}


def default_search_space(spec: ArchSpec) -> list[SearchParam]:
    """Espacio de búsqueda a partir de los `{hp}` de la ArchSpec y los rangos del catálogo."""
    from perceptron.catalog.registry import BLOCKS

    found: dict[str, SearchParam] = {}
    for node in spec.nodes:
        block = BLOCKS.get(node.block)
        if block is None:
            continue
        for key, value in node.params.items():
            if not isinstance(value, HP) or value.hp in found or value.hp in NOT_TUNED:
                continue
            p = block.params.get(key)
            if p is None:
                continue
            if p.choices and p.type in ("int", "str"):
                found[value.hp] = SearchParam(name=value.hp, type="categorical", choices=p.choices)
            elif p.type in ("int", "float") and p.low is not None and p.high is not None:
                found[value.hp] = SearchParam(
                    name=value.hp, type=p.type, low=p.low, high=p.high, log=p.log
                )
    for name, default in spec.hyperparameters().items():
        if name in found or name in NOT_TUNED or name not in _GLOBAL_RANGES:
            continue
        cfg = dict(_GLOBAL_RANGES[name])
        if name == "lr":
            base = float(default or 1e-3)
            cfg.update(low=base / 10, high=min(base * 10, 1.0))
        found[name] = SearchParam(name=name, **cfg)
    return list(found.values())
