"""Contexto tipado que los roles entregan al Gateway.

Ningún llamador arma el payload a mano: construye un `LLMContext` y el Gateway le
aplica el `PrivacyFilter` según el nivel del proyecto (RF-PRV-01, ADR-0007).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from perceptron.data.profiling.card import ProfileCard
from perceptron.llm.types import ImagePart


class RunSummary(BaseModel):
    """Resumen de un run: hiperparámetros, métricas agregadas y curvas por época."""

    run_id: str
    status: str
    architecture: str | None = None
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, float]] = Field(default_factory=list)


class LLMContext(BaseModel):
    goal: str | None = Field(default=None, description="Objetivo en palabras del usuario")
    constraints: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] | None = None
    card: ProfileCard | None = None
    archspec: dict[str, Any] | None = None
    pipeline: dict[str, Any] | None = Field(
        default=None, description="PipelineSpec: pasos, nunca estadísticas ajustadas"
    )
    catalog: list[dict[str, Any]] | None = None
    runs: list[RunSummary] = Field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    # Datos del usuario: solo desde L2 (muestras) o L3 (crudas, imágenes).
    samples: list[dict[str, Any]] = Field(default_factory=list)
    text_fields: list[str] = Field(
        default_factory=list, description="Campos de `samples` con texto libre"
    )
    images: list[ImagePart] = Field(default_factory=list)
    # Metadata del sistema (acciones disponibles, presupuesto, instrucciones del rol).
    system: dict[str, Any] = Field(default_factory=dict)
