"""ArchSpec: lenguaje declarativo de arquitecturas (SPEC §9).

Un grafo acíclico de nodos (bloques del catálogo) con aristas explícitas. Los
valores `{"hp": "<nombre>", "default": x}` declaran hiperparámetros ajustables
que el `HPOStrategy` referencia por nombre.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from perceptron.domain.enums import Modality, Origin, TaskType

ARCHSPEC_VERSION = "1.0"
INPUT_NODE = "input"

Scalar = int | float | str | bool | None


class HP(BaseModel):
    """Referencia a un hiperparámetro ajustable con su valor por defecto."""

    model_config = ConfigDict(extra="forbid")

    hp: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    default: Scalar


ParamValue = HP | Scalar | list[Any] | dict[str, Any]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TaskType
    num_classes: int | None = Field(default=None, ge=2)
    multilabel: bool = False
    num_targets: int = Field(default=1, ge=1, description="Regresión multi-salida")


class InputSpec(BaseModel):
    """Entrada del modelo.

    - `tabular`: `num_numeric` columnas numéricas + categóricas con `cardinalities`.
    - `image` / `spectrogram`: `shape` = [C, H, W].
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tabular", "image", "spectrogram", "sequence", "tokens"]
    shape: list[int] | None = None
    num_numeric: int | None = Field(default=None, ge=0)
    cardinalities: list[int] | None = None
    vocab_size: int | None = Field(default=None, ge=2, description="Tokens: tamaño del vocabulario")
    pad_id: int = Field(default=0, ge=0, description="Tokens: id de padding")
    from_pipeline: str | None = None


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]*$")
    block: str
    params: dict[str, ParamValue] = Field(default_factory=dict)


class LossSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cross_entropy", "bce", "focal", "mse", "mae", "huber"]
    class_weights: Literal["auto", "none"] | list[float] = "none"
    label_smoothing: HP | float = 0.0


class OptimizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: HP | float = 1e-3
    weight_decay: HP | float = 0.0
    momentum: HP | float | None = None


class SchedulerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "one_cycle", "cosine", "step", "plateau"] = "none"
    params: dict[str, Scalar] = Field(default_factory=dict)


class EarlyStopping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monitor: str = "val_loss"
    patience: int = Field(default=5, ge=1)
    mode: Literal["min", "max"] | None = None


class TrainingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: HP | int = Field(default=30)
    batch_size: HP | int | Literal["auto"] = "auto"
    precision: Literal["auto", "32", "16-mixed", "bf16-mixed"] = "auto"
    gradient_clip: float | None = 1.0
    early_stopping: EarlyStopping | None = Field(default_factory=EarlyStopping)
    freeze_backbone_epochs: int = Field(default=0, ge=0)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: Origin = Origin.MANUAL
    llm_call_id: str | None = None
    template: str | None = None
    rationale: str | None = None


class ArchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    archspec_version: str = ARCHSPEC_VERSION
    name: str = Field(min_length=1, max_length=120)
    modality: Modality
    task: TaskSpec
    input: InputSpec
    nodes: list[Node] = Field(min_length=1)
    edges: list[tuple[str, str]] = Field(min_length=1)
    loss: LossSpec
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    metrics: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _unique_ids(self) -> ArchSpec:
        ids = [n.id for n in self.nodes]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"ids de nodo duplicados: {sorted(dup)}")
        if INPUT_NODE in ids:
            raise ValueError(f"'{INPUT_NODE}' está reservado para la entrada")
        return self

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def content_hash(self) -> str:
        """Hash estable de la arquitectura (sin `provenance`)."""
        data = self.model_dump(mode="json", exclude={"provenance"})
        raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def hyperparameters(self) -> dict[str, Scalar]:
        """Todos los `{hp, default}` declarados, por nombre."""
        found: dict[str, Scalar] = {}

        def walk(value: object) -> None:
            if isinstance(value, HP):
                found.setdefault(value.hp, value.default)
            elif isinstance(value, BaseModel):
                for field_name in type(value).model_fields:
                    walk(getattr(value, field_name))
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list | tuple):
                for v in value:
                    walk(v)

        walk(self)
        return found


def resolve(value: object, overrides: dict[str, Scalar] | None = None) -> Any:
    """Reemplaza `HP` por su valor (override del HPO o default), recursivamente."""
    if isinstance(value, HP):
        return (overrides or {}).get(value.hp, value.default)
    if isinstance(value, dict):
        return {k: resolve(v, overrides) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, overrides) for v in value]
    return value


def json_schema() -> dict[str, Any]:
    return ArchSpec.model_json_schema()
