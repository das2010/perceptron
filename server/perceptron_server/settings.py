"""Configuración del Team Server (`PERCEPTRON_SERVER__*`), además de la del Engine.

El Engine se configura como siempre (`PERCEPTRON_*`), en modo `server`, con
`PERCEPTRON_DATABASE_URL` (PostgreSQL) y, si se habilitan, `PERCEPTRON_SOURCE_ROOTS`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERCEPTRON_SERVER__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret_key: SecretStr = Field(
        description="Clave HMAC de los JWT (≥ 32 caracteres aleatorios; rotarla cierra sesiones)"
    )
    access_ttl_s: int = Field(default=15 * 60, ge=60, le=24 * 3600)
    refresh_ttl_s: int = Field(default=7 * 24 * 3600, ge=3600)
    cookie_secure: bool = Field(
        default=True, description="Cookies solo por HTTPS; apagar solo en desarrollo/CI sin TLS"
    )
    spa_dir: Path | None = Field(
        default=None, description="UI compilada (`ui/dist`) servida en `/` (RF-SRV-05)"
    )
    default_workspace: str = "Equipo"
    bootstrap_admin_email: str | None = Field(
        default=None, description="Primer arranque sin usuarios: crea este Admin"
    )
    bootstrap_admin_password: SecretStr | None = None
    login_max_failures: int = Field(
        default=5, ge=1, description="Fallos seguidos antes de bloquear"
    )
    login_lockout_s: int = Field(default=5 * 60, ge=1)
    auth_rate_per_minute: int = Field(
        default=30, ge=1, description="Intentos de login/refresh por IP y minuto"
    )
    min_password_length: int = Field(default=12, ge=8)
    redis_url: SecretStr | None = Field(
        default=None,
        description=(
            "Cola de jobs (Valkey/Redis, Capa 5b): los estudios van a los workers. "
            "Sin cola, corren en el proceso del servidor"
        ),
    )
    max_running_studies_per_user: int = Field(default=2, ge=1)
    max_running_studies_per_workspace: int = Field(default=6, ge=1)
    worker_heartbeat_s: float = Field(default=15.0, gt=0)

    @field_validator("secret_key")
    @classmethod
    def _strong_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("PERCEPTRON_SERVER__SECRET_KEY debe tener al menos 32 caracteres")
        return value
