"""Configuración del Engine.

Fuentes (de mayor a menor prioridad): argumentos explícitos, variables de entorno
`PERCEPTRON_*` (anidadas con `__`, p. ej. `PERCEPTRON_API__PORT`), archivo `.env`.
Nunca se guardan secretos persistentes aquí: las claves van al keychain / vault
(SPEC §7.7.1, RF-LLM-08). El token del sidecar es efímero y solo vive en memoria.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from perceptron.core.paths import WorkspacePaths, default_workspace_dir


class RuntimeMode(StrEnum):
    """Dónde corre este Engine (SPEC §4.3)."""

    DESKTOP = "desktop"
    SERVER = "server"


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=0, ge=0, le=65535, description="0 = puerto aleatorio")
    token: SecretStr | None = Field(
        default=None, description="Token efímero exigido en cada request (sidecar desktop)"
    )
    cors_origins: list[str] = Field(
        # Vite en desarrollo y los orígenes de Tauri 2 (Linux/macOS y Windows).
        default_factory=lambda: [
            "http://localhost:5173",
            "tauri://localhost",
            "http://tauri.localhost",
        ]
    )


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_output: bool = True
    to_file: bool = True
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


class LLMProviderSettings(BaseModel):
    """Proveedor configurado por el usuario o el Admin; la clave va por referencia (RF-LLM-08)."""

    kind: Literal["anthropic", "openai", "gemini", "moonshot", "openai_compat", "ollama", "fake"]
    base_url: str | None = None
    api_key_ref: str | None = Field(
        default=None, description="Nombre del secreto en keychain/vault/entorno; nunca la clave"
    )
    local: bool | None = Field(
        default=None, description="Corre en la máquina o red propia (RF-PRV-02); None = según URL"
    )
    timeout_s: float = Field(default=120.0, gt=0)


class LLMSettings(BaseModel):
    """Capa LLM (SPEC §7.7). Los IDs de modelo viven en el catálogo o en perfiles, no en código."""

    enabled: bool = True
    catalog_file: Path | None = Field(
        default=None, description="Catálogo de modelos alternativo al empaquetado"
    )
    providers: dict[str, LLMProviderSettings] = Field(default_factory=dict)
    profile: str | None = Field(
        default=None, description="Perfil activo (default: el del catálogo)"
    )
    max_attempts: int = Field(
        default=3, ge=1, le=5, description="Intentos con feedback (RF-LLM-04)"
    )
    cache: bool = True
    project_budget_usd: float | None = Field(default=5.0, ge=0)
    l2_samples: int = Field(default=5, ge=0, le=100)
    fake_cassette: Path | None = Field(
        default=None, description="Solo tests/CI: responde con FakeLLMProvider desde un cassette"
    )


class AlertSettings(BaseModel):
    """Canales de alertas del monitoreo (RF-MON-04). La clave SMTP va al almacén de secretos
    (`alerts/smtp/password`), nunca acá."""

    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True
    public_url: str | None = Field(
        default=None, description="URL de la UI para los links de las alertas"
    )
    cooldown_s: int = Field(
        default=3600, ge=0, description="No repite la misma alerta abierta antes de este plazo"
    )


class MonitoringSettings(BaseModel):
    """Scheduler del monitoreo: disparadores de reentrenamiento y sondeo de fuentes."""

    scheduler: bool = True
    interval_s: float = Field(default=30.0, gt=0)


class LicenseSettings(BaseModel):
    """Licencia firmada (Ed25519, offline; ADR-0035). Sin enforcement en v1 (RF-LIC-03)."""

    path: Path | None = Field(default=None, description="Default: <workspace>/license.json")
    public_keys: dict[str, str] = Field(
        default_factory=dict, description="key_id → clave pública (además de las empaquetadas)"
    )
    enforce: bool = False


class TelemetrySettings(BaseModel):
    """Telemetría opt-in (D7): sin endpoint no se envía nada aunque se acepte."""

    endpoint: str | None = Field(default=None, pattern=r"^https://")
    interval_s: float = Field(default=6 * 3600, ge=60)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERCEPTRON_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: RuntimeMode = RuntimeMode.DESKTOP
    workspace_dir: Path = Field(default_factory=default_workspace_dir)
    locale: Literal["es", "en"] = "es"
    api: ApiSettings = Field(default_factory=ApiSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    license: LicenseSettings = Field(default_factory=LicenseSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    database_url: SecretStr | None = Field(
        default=None,
        description="URL SQLAlchemy de la metadata (Team Server: PostgreSQL); None = SQLite local",
    )
    tracking_uri: str | None = Field(
        default=None,
        description="MLflow server (Team Server: http://mlflow:5000); None = SQLite del workspace",
    )
    source_roots: list[Path] | None = Field(
        default=None,
        description=(
            "Si está, las fuentes por ruta solo pueden leer debajo de estas carpetas "
            "(Team Server: «fuentes del servidor», RF-SRV-05); None = cualquier ruta (desktop)"
        ),
    )

    @field_validator("workspace_dir")
    @classmethod
    def _absolute_workspace(cls, value: Path) -> Path:
        # Los workers corren como subprocesos con otro cwd: las rutas deben ser absolutas.
        return value.expanduser().absolute()

    @property
    def paths(self) -> WorkspacePaths:
        return WorkspacePaths(self.workspace_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
