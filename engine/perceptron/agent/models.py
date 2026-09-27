"""Límites, aprobaciones y acciones del agente autónomo (SPEC §7.11, ADR-0021).

El LLM decide una acción por paso devolviendo `AgentStep` (salida estructurada): la acción
es una unión discriminada por `tool` con los argumentos validados por schema. Así el ciclo
funciona igual con Claude y con modelos locales sin tool-calling nativo.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from perceptron.archspec.schema import ArchSpec
from perceptron.llm.schemas import HPOProposal

ApprovalMode = Literal["never", "each_iteration", "family_change", "budget_pct"]


class AgentLimits(BaseModel):
    """Límites duros que aplica el sistema, no el LLM (RF-AGT-02)."""

    max_time_s: float = Field(default=3600, gt=0)
    max_iterations: int = Field(default=4, ge=1, description="Estudios que puede lanzar")
    max_steps: int = Field(default=30, ge=1, description="Decisiones del LLM")
    max_trials: int = Field(default=30, ge=1)
    max_epochs_per_trial: int | None = Field(default=None, ge=1)
    max_llm_cost_usd: float = Field(default=2.0, ge=0)
    max_disk_mb: float = Field(default=5000, gt=0)
    selection_metric: str = Field(default="val_loss", description="Métrica de validación")


class ApprovalPolicy(BaseModel):
    """Cuándo pedir aprobación humana antes de lanzar un estudio (RF-AGT-03)."""

    mode: ApprovalMode = "never"
    budget_pct: float = Field(default=50, gt=0, le=100)


# ---------------------------------------------------------------------- acciones (RF-AGT-01)


class GetProfile(BaseModel):
    tool: Literal["get_profile"]


class GetProjectGoal(BaseModel):
    tool: Literal["get_project_goal"]


class ListCatalogBlocks(BaseModel):
    tool: Literal["list_catalog_blocks"]


class ProposeArchspec(BaseModel):
    tool: Literal["propose_archspec"]
    title: str = Field(max_length=120)
    archspec: ArchSpec
    rationale: str


class ValidateArchspec(BaseModel):
    tool: Literal["validate_archspec"]
    archspec: ArchSpec


class ProposeHpoStrategy(BaseModel):
    tool: Literal["propose_hpo_strategy"]
    archspec_id: str
    strategy: HPOProposal


class LaunchStudy(BaseModel):
    tool: Literal["launch_study"]
    archspec_id: str
    strategy_id: str | None = Field(default=None, description="De propose_hpo_strategy")
    max_trials: int = Field(ge=1)
    max_epochs_per_trial: int | None = Field(default=None, ge=1)


class GetStudyStatus(BaseModel):
    tool: Literal["get_study_status"]
    study_id: str


class GetRunMetrics(BaseModel):
    tool: Literal["get_run_metrics"]
    run_id: str


class GetRunCurves(BaseModel):
    tool: Literal["get_run_curves"]
    run_id: str


class CompareRuns(BaseModel):
    tool: Literal["compare_runs"]
    run_ids: list[str] = Field(min_length=1)


class SuggestPipelineChange(BaseModel):
    tool: Literal["suggest_pipeline_change"]
    description: str
    rationale: str


class RequestHumanApproval(BaseModel):
    tool: Literal["request_human_approval"]
    reason: str


class Finish(BaseModel):
    tool: Literal["finish"]
    run_id: str | None = Field(default=None, description="Run elegido (por defecto, el mejor)")
    summary: str


AgentAction = Annotated[
    GetProfile
    | GetProjectGoal
    | ListCatalogBlocks
    | ProposeArchspec
    | ValidateArchspec
    | ProposeHpoStrategy
    | LaunchStudy
    | GetStudyStatus
    | GetRunMetrics
    | GetRunCurves
    | CompareRuns
    | SuggestPipelineChange
    | RequestHumanApproval
    | Finish,
    Field(discriminator="tool"),
]


class AgentStep(BaseModel):
    log_entry: str = Field(
        description="Una línea legible para la bitácora: qué observaste y qué hacés ahora"
    )
    action: AgentAction


TOOLS: dict[str, str] = {
    "get_profile": "Resumen del perfil del dataset (ya está en el contexto).",
    "get_project_goal": "Objetivo del usuario.",
    "list_catalog_blocks": "Bloques del catálogo permitidos (ya están en el contexto).",
    "propose_archspec": "Guarda una ArchSpec nueva si valida; devuelve archspec_id o el error.",
    "validate_archspec": "Valida una ArchSpec sin guardarla.",
    "propose_hpo_strategy": "Valida una estrategia de HPO para un archspec_id -> strategy_id.",
    "launch_study": "Entrena: lanza un estudio (usa el presupuesto; puede pedir aprobación).",
    "get_study_status": "Estado y mejor trial de un estudio.",
    "get_run_metrics": "Métricas de validación de un run (el test está sellado).",
    "get_run_curves": "Curvas por época de un run + diagnóstico por reglas.",
    "compare_runs": "Tabla de métricas e hiperparámetros de varios runs.",
    "suggest_pipeline_change": "Registra una sugerencia sobre los datos (no se aplica sola).",
    "request_human_approval": "Pausa y pide aprobación al usuario.",
    "finish": "Termina: el sistema evalúa el mejor run en test, lo registra e informa.",
}
