"""Del diagnóstico a la próxima corrida (ADR-0041, iteración 3).

El diagnosticador (reglas o LLM) sugiere acciones tipadas (`SuggestedAction`). Acá cada acción
se traduce, de forma determinística, en un cambio concreto de la ArchSpec del run:

- se parte del **mejor punto conocido**: los hiperparámetros del run pasan a ser los defaults;
- se aplica la acción (bajar el learning rate, más regularización, más épocas, compensar el
  desbalance…) y se describe el cambio («lr: 0,001 → 0,0003»);
- lo que no es de la arquitectura (aumentar datos, cambiar de familia, conseguir más datos)
  queda como sugerencia con el lugar donde hacerlo.

Nada se aplica solo: la persona elige la mejora, se guarda como ArchSpec nueva y decide si
entrenarla.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from perceptron.archspec.schema import ArchSpec, Scalar
from perceptron.llm.schemas import Diagnosis, SuggestedAction

Hint = Literal["design", "pipeline", "data", "formula"]
MAX_EPOCHS = 500
MAX_DROPOUT = 0.5
MIN_WEIGHT_DECAY = 1e-4
_NOT_ARCH: dict[str, Hint] = {
    "try_formula": "formula",
    "change_architecture": "design",
    "add_augmentation": "pipeline",
    "more_data": "data",
}


class ImprovementOption(BaseModel):
    index: int = Field(description="Posición de la acción en el diagnóstico")
    kind: str
    rationale: str
    applicable: bool
    change: str | None = Field(default=None, description="Qué cambia en la arquitectura")
    hint: Hint | None = Field(default=None, description="Dónde se hace si no es de la arquitectura")


# ---------------------------------------------------------------------- utilidades


def _fmt(value: object) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _walk_hps(node: Any, fn: Any) -> None:
    """Llama `fn(dict_hp)` por cada `{hp, default}` del JSON de la spec."""
    if isinstance(node, dict):
        if "hp" in node and "default" in node and isinstance(node["hp"], str):
            fn(node)
            return
        for v in node.values():
            _walk_hps(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk_hps(v, fn)


def with_defaults(spec: ArchSpec, values: dict[str, Any]) -> ArchSpec:
    """Los valores (p. ej. los del mejor trial) pasan a ser los defaults de sus HP."""
    data = spec.model_dump(mode="json")

    def set_default(hp: dict[str, Any]) -> None:
        if hp["hp"] in values and isinstance(values[hp["hp"]], int | float | str | bool):
            hp["default"] = values[hp["hp"]]

    _walk_hps(data, set_default)
    return ArchSpec.model_validate(data)


def _current(spec: ArchSpec, name: str) -> Scalar | None:
    hps = spec.hyperparameters()
    if name in hps:
        return hps[name]
    fixed = {
        "lr": spec.optimizer.lr,
        "weight_decay": spec.optimizer.weight_decay,
        "epochs": spec.training.epochs,
        "batch_size": spec.training.batch_size,
    }.get(name)
    return fixed if isinstance(fixed, int | float) else None


def _set(spec: ArchSpec, name: str, value: Scalar) -> ArchSpec | None:
    """Fija `name` (HP declarado o campo de entrenamiento conocido). None si no existe."""
    data = spec.model_dump(mode="json")
    found = False

    def set_default(hp: dict[str, Any]) -> None:
        nonlocal found
        if hp["hp"] == name:
            hp["default"] = value
            found = True

    _walk_hps(data, set_default)
    if not found:
        if name in ("lr", "weight_decay") and not isinstance(data["optimizer"][name], dict):
            data["optimizer"][name] = value
            found = True
        elif name in ("epochs", "batch_size") and not isinstance(data["training"][name], dict):
            data["training"][name] = value
            found = True
        else:
            for node in data["nodes"]:
                if name in node["params"] and not isinstance(node["params"][name], dict):
                    node["params"][name] = value
                    found = True
    return ArchSpec.model_validate(data) if found else None


# ---------------------------------------------------------------------- acciones


def apply_action(spec: ArchSpec, action: SuggestedAction) -> tuple[ArchSpec, str] | None:
    """La spec con la acción aplicada y la descripción del cambio; None si no aplica acá."""
    kind = action.kind
    if kind in _NOT_ARCH or kind == "none":
        return None
    if kind == "change_hparam" and action.target:
        before = _current(spec, action.target)
        value = action.value
        if not isinstance(value, int | float) and action.target == "lr" and before:
            value = float(before) * 0.3
        if not isinstance(value, int | float | str | bool):
            return None
        out = _set(spec, action.target, value)
        return (out, f"{action.target}: {_fmt(before)} → {_fmt(value)}") if out else None
    if kind == "add_regularization":
        return _regularize(spec, action)
    if kind in ("more_epochs", "fewer_epochs"):
        before = _current(spec, "epochs")
        if not isinstance(before, int | float):
            return None
        if isinstance(action.value, int | float) and action.value >= 1:
            after = int(action.value)
        elif kind == "more_epochs":
            after = min(MAX_EPOCHS, int(before) * 2)
        else:
            after = max(1, int(before) // 2)
        out = _set(spec, "epochs", after)
        return (out, f"epochs: {int(before)} → {after}") if out else None
    if kind == "rebalance":
        if spec.task.type.value != "classification":
            return None
        if spec.loss.class_weights == "none":
            loss = spec.loss.model_copy(update={"class_weights": "auto"})
            return spec.model_copy(update={"loss": loss}), "class_weights: none → auto"
        if not spec.training.oversample:
            training = spec.training.model_copy(update={"oversample": True})
            return spec.model_copy(update={"training": training}), "oversample: no → sí"
        return None
    return None


def _regularize(spec: ArchSpec, action: SuggestedAction) -> tuple[ArchSpec, str] | None:
    if action.target and isinstance(action.value, int | float):
        before = _current(spec, action.target)
        out = _set(spec, action.target, action.value)
        if out is not None:
            return out, f"{action.target}: {_fmt(before)} → {_fmt(action.value)}"
    dropout = _current(spec, "dropout")
    if isinstance(dropout, int | float) and dropout < MAX_DROPOUT:
        after = round(min(MAX_DROPOUT, max(0.1, float(dropout) * 1.5)), 3)
        out = _set(spec, "dropout", after)
        if out is not None:
            return out, f"dropout: {_fmt(float(dropout))} → {_fmt(after)}"
    wd = _current(spec, "weight_decay")
    if isinstance(wd, int | float):
        after_wd = max(MIN_WEIGHT_DECAY, float(wd) * 10)
        out = _set(spec, "weight_decay", after_wd)
        if out is not None:
            return out, f"weight_decay: {_fmt(float(wd))} → {_fmt(after_wd)}"
    return None


def improvement_options(
    spec: ArchSpec, diagnosis: Diagnosis, hyperparams: dict[str, Any] | None = None
) -> list[ImprovementOption]:
    base = with_defaults(spec, hyperparams or {})
    options: list[ImprovementOption] = []
    for i, action in enumerate(diagnosis.actions):
        if action.kind == "none":
            continue
        applied = apply_action(base, action)
        options.append(
            ImprovementOption(
                index=i,
                kind=action.kind,
                rationale=action.rationale,
                applicable=applied is not None,
                change=applied[1] if applied else None,
                hint=_NOT_ARCH.get(action.kind),
            )
        )
    return options
