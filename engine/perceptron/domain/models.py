"""Entidades del dominio (SPEC §6).

Son modelos Pydantic v2 puros (sin persistencia). Los documentos complejos que
se definen en capas posteriores (ArchSpec §9, Pipeline §7.4, HPOStrategy §7.9)
se guardan aquí como `dict` opaco validado por su propio schema en esa capa.

Toda entidad tiene `version` para bloqueo optimista (RF-SRV-03).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from perceptron.core.ids import IdPrefix, new_id
from perceptron.domain.enums import (
    AgentState,
    AlertKind,
    AlertStatus,
    DataSourceType,
    DeploymentHost,
    DeploymentStatus,
    Device,
    ExportFormat,
    LabelKind,
    LabelOrigin,
    LLMPurpose,
    Modality,
    ModelStage,
    Origin,
    PrivacyLevel,
    ProjectScope,
    ProjectStatus,
    RetrainStatus,
    Role,
    RunStatus,
    Severity,
    SplitStrategy,
    TaskType,
)

JsonDict = dict[str, Any]


def utcnow() -> datetime:
    return datetime.now(UTC)


def _id_factory(prefix: IdPrefix) -> Any:
    return lambda: new_id(prefix)


class Entity(BaseModel):
    """Base de todas las entidades persistibles."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)

    id: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    version: int = Field(default=1, ge=1, description="Versión para bloqueo optimista")


# --------------------------------------------------------------------- organización


class Workspace(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.WORKSPACE))
    name: str = Field(min_length=1, max_length=200)
    max_privacy_level: PrivacyLevel = PrivacyLevel.L3
    local_llm_max_privacy: PrivacyLevel | None = Field(
        default=None, description="Tope con LLM local; puede superar el general (RF-PRV-02)"
    )
    allowed_llm_providers: list[str] | None = None


class User(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.USER))
    email: str
    display_name: str
    is_active: bool = True
    auth_provider: str = "local"


class Membership(Entity):
    """Rol de un usuario en un workspace o, si `project_id` está, en un proyecto."""

    id: str = Field(default_factory=_id_factory(IdPrefix.MEMBERSHIP))
    user_id: str
    workspace_id: str
    project_id: str | None = None
    role: Role


# --------------------------------------------------------------------- proyecto y datos


class Project(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.PROJECT))
    workspace_id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    goal: str = Field(default="", description="Objetivo en lenguaje natural del usuario")
    modalities: list[Modality] = Field(default_factory=list)
    task: TaskType | None = None
    target_metric: str | None = None
    privacy_level: PrivacyLevel = PrivacyLevel.L1
    llm_profile_id: str | None = None
    scope: ProjectScope = ProjectScope.LOCAL
    status: ProjectStatus = ProjectStatus.DRAFT
    template: str | None = Field(default=None, description="Plantilla UC-xx de origen")


class DataSource(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.DATASOURCE))
    project_id: str
    name: str
    type: DataSourceType
    config: JsonDict = Field(
        default_factory=dict, description="Sin secretos: usar `secret_refs` (keychain/vault)"
    )
    secret_refs: dict[str, str] = Field(default_factory=dict)
    inferred_schema: JsonDict | None = None


class Split(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: SplitStrategy
    train: int = Field(ge=0)
    val: int = Field(ge=0)
    test: int = Field(ge=0)
    folds: int | None = Field(default=None, ge=2)
    seed: int = 42
    group_column: str | None = None
    time_column: str | None = None
    test_sealed: bool = Field(default=True, description="El test no se usa en HPO (RF-ING-08)")


class DatasetVersion(Entity):
    """Snapshot inmutable identificado por el hash de su manifiesto (RF-ING-07)."""

    id: str = Field(default_factory=_id_factory(IdPrefix.DATASET_VERSION))
    project_id: str
    source_id: str | None = None
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    modality: Modality | None = None
    target: str | None = None
    path: str | None = Field(default=None, description="Directorio relativo al proyecto")
    num_samples: int = Field(ge=0)
    size_bytes: int = Field(default=0, ge=0)
    split: Split | None = None
    parent_id: str | None = None
    transformation: str | None = None


class Profile(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.PROFILE))
    dataset_version_id: str
    modality: Modality
    stats: JsonDict = Field(default_factory=dict)
    alerts: list[JsonDict] = Field(default_factory=list)
    complexity: JsonDict = Field(default_factory=dict)


class LabelSet(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.LABEL_SET))
    dataset_version_id: str
    kind: LabelKind
    classes: list[str] = Field(default_factory=list)
    path: str | None = Field(default=None, description="Ruta relativa al proyecto")
    name: str | None = None
    origin: LabelOrigin = LabelOrigin.HUMAN
    target: str = Field(default="label", description="Columna objetivo al aplicar (tabular)")
    applied_version_id: str | None = Field(
        default=None, description="DatasetVersion creada al aplicar las etiquetas (RF-LBL)"
    )


class Pipeline(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.PIPELINE))
    project_id: str
    name: str
    graph: JsonDict = Field(default_factory=dict, description="DAG de pasos (§7.4)")
    origin: Origin = Origin.RULES


class ArchSpecRecord(Entity):
    """Referencia persistida a una ArchSpec (§9) o a código experto (RF-ARC-06)."""

    id: str = Field(default_factory=_id_factory(IdPrefix.ARCHSPEC))
    project_id: str
    name: str
    spec: JsonDict | None = None
    code_path: str | None = None
    content_hash: str
    origin: Origin = Origin.MANUAL

    @property
    def is_declarative(self) -> bool:
        return self.spec is not None


# --------------------------------------------------------------------- experimentos


class Study(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.STUDY))
    project_id: str
    name: str
    strategy: JsonDict = Field(default_factory=dict, description="HPOStrategy (§7.9)")
    budget: JsonDict = Field(default_factory=dict)
    objectives: list[str] = Field(default_factory=list)
    origin: Origin = Origin.MANUAL


class Run(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.RUN))
    project_id: str
    study_id: str | None = None
    archspec_id: str
    pipeline_id: str
    dataset_version_id: str
    status: RunStatus = RunStatus.QUEUED
    device: Device = Device.CPU
    hyperparams: JsonDict = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    seed: int = 42
    environment: JsonDict = Field(default_factory=dict)
    mlflow_run_id: str | None = None
    diagnosis: JsonDict | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Evaluation(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.EVALUATION))
    run_id: str
    split: str = "test"
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    dataset_version_id: str | None = Field(
        default=None, description="Otra versión de datos (holdout de champion/challenger)"
    )


class ModelVersion(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.MODEL_VERSION))
    project_id: str
    run_id: str
    stage: ModelStage = ModelStage.CANDIDATE
    signature: JsonDict = Field(default_factory=dict)
    model_card: JsonDict = Field(default_factory=dict)
    promoted_at: datetime | None = Field(default=None, description="Última vez que fue champion")
    retired_at: datetime | None = Field(default=None, description="Cuándo dejó de ser champion")
    mlflow_version: str | None = Field(
        default=None, description="Versión espejo en el Model Registry de MLflow (RF-TRK-03)"
    )


class Export(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.EXPORT))
    model_version_id: str
    format: ExportFormat
    path: str
    checksums: dict[str, str] = Field(default_factory=dict)


class Deployment(Entity):
    """Modelo en uso con monitoreo (RF-MON-01..04). Sigue al champion del proyecto."""

    id: str = Field(default_factory=_id_factory(IdPrefix.DEPLOYMENT))
    project_id: str = ""
    name: str = "default"
    model_version_id: str
    endpoint: str
    host: DeploymentHost = DeploymentHost.DESKTOP
    status: DeploymentStatus = DeploymentStatus.ACTIVE
    follow_champion: bool = Field(default=True, description="Al promover, pasa al champion")
    sample_rate: float = Field(
        default=1.0, ge=0, le=1, description="Fracción de predicciones registradas"
    )
    key_column: str | None = Field(default=None, description="Columna para asociar el feedback")
    monitoring: JsonDict = Field(
        default_factory=dict, description="Ventana, umbrales y canales de alerta"
    )


class DriftReport(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.DRIFT_REPORT))
    deployment_id: str
    window_start: datetime
    window_end: datetime
    metrics: JsonDict = Field(default_factory=dict)
    severity: Severity = Severity.NONE
    action: str | None = None


class ActivityEntry(Entity):
    """Qué se hizo en el proyecto, quién y cuándo (RF-PRJ-05)."""

    id: str = Field(default_factory=_id_factory(IdPrefix.ACTIVITY))
    project_id: str
    operation: str = Field(description="operation_id de la API (p. ej. createSource)")
    method: str
    path: str
    actor: str | None = Field(default=None, description="Usuario (Team Server) o local")
    status: int


class Alert(Entity):
    """Alerta del monitoreo (RF-MON-04): en la app y, si hay canales, por email o webhook."""

    id: str = Field(default_factory=_id_factory(IdPrefix.ALERT))
    project_id: str
    deployment_id: str | None = None
    kind: AlertKind
    severity: Severity
    title: str
    message: str = ""
    details: JsonDict = Field(default_factory=dict)
    status: AlertStatus = AlertStatus.OPEN
    channels: list[str] = Field(default_factory=list, description="Por dónde salió")


class RetrainPolicy(Entity):
    """Cuándo y cómo reentrenar el modelo en uso (RF-MON-05)."""

    id: str = Field(default_factory=_id_factory(IdPrefix.RETRAIN_POLICY))
    project_id: str
    deployment_id: str | None = Field(default=None, description="Deployment que vigila")
    source_ids: list[str] = Field(
        default_factory=list, description="Fuentes streaming/API con datos nuevos"
    )
    triggers: list[JsonDict] = Field(
        default_factory=list,
        description="drift {min_severity}, cron {expr}, volume {min_rows}, degradation {max_drop}",
    )
    require_approval: bool = True
    budget: JsonDict = Field(default_factory=dict, description="max_trials, max_epochs_per_trial")
    min_improvement: float = Field(default=0.0, ge=0)
    holdout_fraction: float = Field(
        default=0.3, gt=0, lt=1, description="Filas nuevas reservadas para comparar (no entrenan)"
    )
    use_feedback: bool = Field(
        default=True, description="Suma el feedback etiquetado del deployment a los datos nuevos"
    )
    cooldown_s: int = Field(default=3600, ge=0)
    enabled: bool = False
    last_run_at: datetime | None = None
    last_cron_at: datetime | None = None
    consumed: JsonDict = Field(
        default_factory=dict, description="Último lote de cada fuente ya usado para reentrenar"
    )


class RetrainRun(Entity):
    """Una ejecución de la política: datos nuevos → challenger → comparación → promoción."""

    id: str = Field(default_factory=_id_factory(IdPrefix.RETRAIN_RUN))
    project_id: str
    policy_id: str
    trigger: JsonDict = Field(default_factory=dict)
    status: RetrainStatus = RetrainStatus.RUNNING
    champion_id: str | None = None
    dataset_version_id: str | None = None
    study_id: str | None = None
    run_id: str | None = None
    model_version_id: str | None = None
    challenge: JsonDict | None = None
    new_rows: int = 0
    log: list[JsonDict] = Field(default_factory=list)
    error: str | None = None
    finished_at: datetime | None = None


# --------------------------------------------------------------------- LLM y auditoría


class LLMSession(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.LLM_SESSION))
    project_id: str
    purpose: LLMPurpose


class LLMCall(Entity):
    """Registro de auditoría de una llamada al LLM (RF-PRV-03)."""

    id: str = Field(default_factory=_id_factory(IdPrefix.LLM_CALL))
    session_id: str | None = None
    project_id: str
    purpose: LLMPurpose
    provider: str
    model: str
    prompt_version: str | None = None
    privacy_level: PrivacyLevel
    payload: JsonDict = Field(description="Payload enviado, ya filtrado por PrivacyFilter")
    response: JsonDict | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    status: Literal["ok", "invalid", "error"] = "ok"
    error: str | None = None
    attempt: int = Field(default=1, ge=1, description="Intento dentro de la llamada (RF-LLM-04)")
    cache_hit: bool = False
    redactions: list[str] = Field(
        default_factory=list, description="Qué quitó o enmascaró el PrivacyFilter"
    )
    scope: str | None = Field(default=None, description="Ámbito del presupuesto: run, agente…")
    latency_s: float = Field(default=0.0, ge=0)


class ProjectDraft(Entity):
    """Estado del wizard de un proyecto (RF-WIZ-04): versionado, se retoma donde quedó."""

    id: str = Field(default_factory=_id_factory(IdPrefix.PROJECT_DRAFT))
    project_id: str
    step: str = "goal"
    values: JsonDict = Field(default_factory=dict, description="DraftValues (services.wizard)")
    history: list[JsonDict] = Field(
        default_factory=list, description="Cambios con su origen (usuario o copiloto)"
    )


class AgentRun(Entity):
    """Una ejecución del agente autónomo (RF-AGT-01..05): límites, bitácora y resultado."""

    id: str = Field(default_factory=_id_factory(IdPrefix.AGENT_RUN))
    project_id: str
    dataset_version_id: str
    pipeline_id: str
    state: AgentState = AgentState.RUNNING
    limits: JsonDict = Field(default_factory=dict, description="AgentLimits")
    approval: JsonDict = Field(default_factory=dict, description="ApprovalPolicy")
    iterations: int = Field(default=0, ge=0, description="Estudios lanzados")
    steps: int = Field(default=0, ge=0, description="Decisiones del LLM")
    trials: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    studies: list[str] = Field(default_factory=list)
    archspecs: list[str] = Field(default_factory=list)
    strategies: JsonDict = Field(default_factory=dict, description="Estrategias propuestas")
    best_run_id: str | None = None
    model_version_id: str | None = None
    test_metrics: dict[str, float] = Field(default_factory=dict)
    log: list[JsonDict] = Field(default_factory=list, description="Bitácora (RF-AGT-04)")
    pending_action: JsonDict | None = None
    last_family: str | None = None
    fallback: bool = False
    stop_reason: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


ALL_ENTITIES: tuple[type[Entity], ...] = (
    Workspace,
    AgentRun,
    ProjectDraft,
    User,
    Membership,
    Project,
    DataSource,
    DatasetVersion,
    Profile,
    LabelSet,
    Pipeline,
    ArchSpecRecord,
    Study,
    Run,
    Evaluation,
    ModelVersion,
    Export,
    Deployment,
    DriftReport,
    Alert,
    RetrainPolicy,
    RetrainRun,
    LLMSession,
    LLMCall,
    ActivityEntry,
)
