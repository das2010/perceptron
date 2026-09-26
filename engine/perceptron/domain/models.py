"""Entidades del dominio (SPEC §6).

Son modelos Pydantic v2 puros (sin persistencia). Los documentos complejos que
se definen en capas posteriores (ArchSpec §9, Pipeline §7.4, HPOStrategy §7.9)
se guardan aquí como `dict` opaco validado por su propio schema en esa capa.

Toda entidad tiene `version` para bloqueo optimista (RF-SRV-03).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from perceptron.core.ids import IdPrefix, new_id
from perceptron.domain.enums import (
    DataSourceType,
    DeploymentHost,
    Device,
    ExportFormat,
    LabelKind,
    LLMPurpose,
    Modality,
    ModelStage,
    Origin,
    PrivacyLevel,
    ProjectScope,
    ProjectStatus,
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


class ModelVersion(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.MODEL_VERSION))
    project_id: str
    run_id: str
    stage: ModelStage = ModelStage.CANDIDATE
    signature: JsonDict = Field(default_factory=dict)
    model_card: JsonDict = Field(default_factory=dict)


class Export(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.EXPORT))
    model_version_id: str
    format: ExportFormat
    path: str
    checksums: dict[str, str] = Field(default_factory=dict)


class Deployment(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.DEPLOYMENT))
    model_version_id: str
    endpoint: str
    host: DeploymentHost = DeploymentHost.DESKTOP
    monitoring: JsonDict = Field(default_factory=dict)


class DriftReport(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.DRIFT_REPORT))
    deployment_id: str
    window_start: datetime
    window_end: datetime
    metrics: JsonDict = Field(default_factory=dict)
    severity: Severity = Severity.NONE
    action: str | None = None


class RetrainPolicy(Entity):
    id: str = Field(default_factory=_id_factory(IdPrefix.RETRAIN_POLICY))
    project_id: str
    triggers: list[JsonDict] = Field(default_factory=list)
    require_approval: bool = True
    budget: JsonDict = Field(default_factory=dict)
    enabled: bool = False


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


ALL_ENTITIES: tuple[type[Entity], ...] = (
    Workspace,
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
    RetrainPolicy,
    LLMSession,
    LLMCall,
)
