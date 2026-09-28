"""Configuración del Team Server (`PERCEPTRON_SERVER__*`), además de la del Engine.

El Engine se configura como siempre (`PERCEPTRON_*`), en modo `server`, con
`PERCEPTRON_DATABASE_URL` (PostgreSQL) y, si se habilitan, `PERCEPTRON_SOURCE_ROOTS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from perceptron.domain.enums import Role


class GroupRole(BaseModel):
    """Un grupo del IdP da un rol en un workspace (o en un proyecto) (RF-SRV-01)."""

    group: str = Field(description="Id o nombre del grupo tal como viene en el token")
    workspace: str = Field(description="Nombre del workspace (se crea si no existe)")
    role: Role
    project_id: str | None = None


class OIDCProvider(BaseModel):
    """Proveedor SSO OIDC (Entra ID, Google Workspace o genérico). MFA la resuelve el IdP."""

    kind: Literal["entra", "google", "generic"] = "generic"
    display_name: str
    client_id: str
    client_secret: SecretStr | None = Field(default=None, description="Cliente confidencial")
    tenant_id: str | None = Field(default=None, description="Entra ID: id del tenant")
    issuer: str | None = Field(default=None, description="Genérico: URL del emisor")
    scopes: list[str] = Field(default_factory=lambda: ["openid", "email", "profile"])
    groups_claim: str = "groups"
    allowed_domains: list[str] = Field(
        default_factory=list, description="Dominios de email aceptados (vacío = cualquiera)"
    )
    auto_create: bool = Field(default=True, description="Crea el usuario en el primer login")
    default_role: Role | None = Field(
        default=Role.VIEWER, description="Rol en el workspace por defecto si ningún grupo aplica"
    )
    role_mapping: list[GroupRole] = Field(default_factory=list)

    def issuer_url(self) -> str:
        if self.kind == "entra":
            if not self.tenant_id:
                raise ValueError("Entra ID requiere tenant_id")
            return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"
        if self.kind == "google":
            return self.issuer or "https://accounts.google.com"
        if not self.issuer:
            raise ValueError("el proveedor OIDC genérico requiere issuer")
        return self.issuer.rstrip("/")


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
    password_login: bool = Field(
        default=True, description="Login local; se puede apagar si todo pasa por SSO"
    )
    oidc: dict[str, OIDCProvider] = Field(
        default_factory=dict,
        description='SSO: {"entra": {...}} (JSON en PERCEPTRON_SERVER__OIDC)',
    )
    public_url: str | None = Field(
        default=None,
        description="URL pública (https://perceptron.empresa.com) para el redirect de SSO",
    )
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
