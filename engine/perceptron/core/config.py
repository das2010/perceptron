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

from pydantic import BaseModel, Field, SecretStr
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
        default_factory=lambda: ["http://localhost:5173", "tauri://localhost"]
    )


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_output: bool = True
    to_file: bool = True
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


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

    @property
    def paths(self) -> WorkspacePaths:
        return WorkspacePaths(self.workspace_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
