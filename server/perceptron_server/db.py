"""Tablas propias del Team Server (ADR-0030).

Usuarios, workspaces y membresías son entidades del dominio (`entities`, como el resto del
Engine). Acá solo va lo que no debe viajar con los documentos: hashes de contraseña, tokens
de refresco y la auditoría (append-only, consultable por fecha, usuario y proyecto).
El esquema se crea y migra con Alembic (`perceptron_server/migrations`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ServerBase(DeclarativeBase):
    pass


class AccountRow(ServerBase):
    """Credenciales y estado de login de un `User` (RF-SRV-01)."""

    __tablename__ = "server_accounts"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    is_server_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failed_logins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RefreshTokenRow(ServerBase):
    """Token de refresco rotativo; solo se guarda su hash. Reusar uno revocado revoca la familia."""

    __tablename__ = "server_refresh_tokens"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    ip: Mapped[str | None] = mapped_column(String(64))


class AuditRow(ServerBase):
    """Evento de auditoría (RF-SRV-07): login, acceso a datos, exportaciones, LLM, permisos."""

    __tablename__ = "server_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    project_id: Mapped[str | None] = mapped_column(String(64))
    resource: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[int | None] = mapped_column(Integer)
    ip: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_server_audit_at", "at"),
        Index("ix_server_audit_user_at", "user_id", "at"),
        Index("ix_server_audit_project_at", "project_id", "at"),
        Index("ix_server_audit_action_at", "action", "at"),
    )
