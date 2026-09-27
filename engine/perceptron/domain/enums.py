"""Enumeraciones del dominio (SPEC §2, §6.1, §7)."""

from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIMESERIES = "timeseries"
    AUDIO = "audio"


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    OCR = "ocr"
    SOUND_EVENT_DETECTION = "sound_event_detection"


class PrivacyLevel(StrEnum):
    """Qué puede recibir el LLM (SPEC §7.7.3)."""

    L0 = "L0"  # sin LLM
    L1 = "L1"  # solo metadatos agregados (default)
    L2 = "L2"  # + muestras anonimizadas
    L3 = "L3"  # + muestras crudas

    @property
    def rank(self) -> int:
        return int(self.value[1])


class ProjectScope(StrEnum):
    LOCAL = "local"
    TEAM = "team"


class ProjectStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class Role(StrEnum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class DataSourceType(StrEnum):
    FILE = "file"
    FOLDER = "folder"
    DB = "db"
    HF = "hf"
    KAGGLE = "kaggle"
    API = "api"
    STREAM = "stream"


class SplitStrategy(StrEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    TEMPORAL = "temporal"
    KFOLD = "kfold"


class LabelKind(StrEnum):
    CLASS = "class"
    MULTILABEL = "multilabel"
    BOX = "box"
    MASK = "mask"
    TEMPORAL_EVENT = "temporal_event"


class LabelOrigin(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    LLM = "llm"


class Origin(StrEnum):
    """Quién originó una configuración (ArchSpec, Study)."""

    MANUAL = "manual"
    LLM = "llm"
    AGENT = "agent"
    RULES = "rules"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentState(StrEnum):
    """Estados del ciclo autónomo (SPEC §7.11)."""

    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    STOPPED = "stopped"
    FINISHED = "finished"
    FAILED = "failed"


class Device(StrEnum):
    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"
    CPU = "cpu"


class ModelStage(StrEnum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ExportFormat(StrEnum):
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TORCH_EXPORT = "torch_export"
    REST_API = "rest_api"
    CODE_PROJECT = "code_project"


class DeploymentHost(StrEnum):
    DESKTOP = "desktop"
    SERVER = "server"


class Severity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LLMPurpose(StrEnum):
    """Propósitos del LLM, cada uno con su perfil de modelo (RF-LLM-03)."""

    COPILOT = "copilot"
    ARCHITECT = "architect"
    HPO_STRATEGIST = "hpo_strategist"
    DIAGNOSTICIAN = "diagnostician"
    AGENT = "agent"
    REPORTER = "reporter"
    LABELER = "labeler"
