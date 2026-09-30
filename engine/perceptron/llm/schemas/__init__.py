"""Salidas estructuradas de los roles del LLM (SPEC §7.7.4).

Todo lo que el LLM devuelve y modifica el sistema pasa por uno de estos modelos y un
validador de dominio; después el usuario lo acepta o lo edita (CLAUDE.md).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from perceptron.archspec.schema import ArchSpec
from perceptron.hpo.strategy import Condition, Objective, PrunerName, StrategyName

# ------------------------------------------------------------------ arquitecto


class ArchCandidate(BaseModel):
    title: str = Field(max_length=120)
    archspec: ArchSpec
    rationale: str = Field(description="Por qué esta arquitectura para estos datos y objetivo")
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ArchProposalSet(BaseModel):
    proposals: list[ArchCandidate] = Field(min_length=1, max_length=4)


# ------------------------------------------------------------------ estratega de HPO


class ProposedParam(BaseModel):
    """Hiperparámetro tal como lo propone el LLM: puede venir incompleto.

    El sistema lo completa y acota con los límites del catálogo (`HPOSpace.repair`) antes de
    validarlo como `SearchParam`; así un límite omitido no descarta toda la propuesta.
    """

    name: str
    type: Literal["int", "float", "categorical"] | None = None
    low: float | None = None
    high: float | None = None
    log: bool | None = None
    step: float | None = None
    choices: list[Any] | None = None
    condition: Condition | None = None
    default: Any = None


class HPOProposal(BaseModel):
    strategy: StrategyName
    pruner: PrunerName
    search_space: list[ProposedParam]
    objectives: list[Objective] = Field(min_length=1)
    max_trials: int = Field(ge=1)
    max_epochs_per_trial: int | None = Field(default=None, ge=1)
    pruner_warmup_epochs: int = Field(default=1, ge=0)
    rationale: str


# ------------------------------------------------------------------ diagnosticador

ProblemKind = Literal[
    "overfitting",
    "underfitting",
    "divergence",
    "lr_too_high",
    "lr_too_low",
    "plateau",
    "class_imbalance",
    "data_bottleneck",
    "stopped_too_early",
    "other",
]
ActionKind = Literal[
    "change_hparam",
    "add_regularization",
    "add_augmentation",
    "change_architecture",
    "more_epochs",
    "fewer_epochs",
    "rebalance",
    "more_data",
    "none",
]


class Problem(BaseModel):
    kind: ProblemKind
    severity: Literal["low", "medium", "high"]
    evidence: str
    explanation: str


class SuggestedAction(BaseModel):
    kind: ActionKind
    target: str | None = Field(default=None, description="Hiperparámetro o bloque afectado")
    value: Any = None
    rationale: str


class Diagnosis(BaseModel):
    summary: str
    problems: list[Problem] = Field(default_factory=list)
    actions: list[SuggestedAction] = Field(default_factory=list)
    origin: Literal["rules", "llm"] = "rules"
    llm_call_id: str | None = None


# ------------------------------------------------------------------ informante


class ModelCard(BaseModel):
    intended_use: str
    data: str
    training: str
    # Forma libre: el sistema la reemplaza por las métricas reales de la evaluación. Con muchas
    # clases el LLM agrega métricas por clase (dict o lista) y un schema solo numérico hacía
    # fallar el informe tres veces seguidas.
    metrics: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    ethical_considerations: list[str] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    summary: str
    markdown: str
    model_card: ModelCard
    origin: Literal["rules", "llm"] = "rules"
    llm_call_id: str | None = None


# ------------------------------------------------------------------ etiquetador


class ClassGuide(BaseModel):
    name: str
    definition: str
    positive_examples: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)


class LabelingGuide(BaseModel):
    classes: list[ClassGuide] = Field(min_length=1)
    general_rules: list[str] = Field(default_factory=list)


class Prelabel(BaseModel):
    sample_id: str
    label: str
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None


class PrelabelBatch(BaseModel):
    labels: list[Prelabel]


__all__ = [
    "ArchCandidate",
    "ArchProposalSet",
    "ClassGuide",
    "Diagnosis",
    "HPOProposal",
    "LabelingGuide",
    "ModelCard",
    "Prelabel",
    "PrelabelBatch",
    "Problem",
    "ProposedParam",
    "Report",
    "SuggestedAction",
]
