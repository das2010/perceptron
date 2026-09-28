"""Esquema inicial del Team Server: documentos del Engine, caché LLM, cuentas, sesiones y auditoría.

Revision ID: 0001
Revises:
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ts(name: str, *, nullable: bool = False) -> sa.Column[object]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    # --- Engine (ADR-0013): documentos por entidad y caché de respuestas del LLM
    op.create_table(
        "entities",
        sa.Column("kind", sa.String(64), primary_key=True),
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("project_id", sa.String(64), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_entities_project_id", "entities", ["project_id"])
    op.create_index("ix_entities_kind_created", "entities", ["kind", "created_at"])
    op.create_table(
        "llm_cache",
        sa.Column("key", sa.String(64), primary_key=True),
        _ts("created_at"),
        sa.Column("data", sa.JSON(), nullable=False),
    )

    # --- Team Server (ADR-0030)
    op.create_table(
        "server_accounts",
        sa.Column("user_id", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("is_server_admin", sa.Boolean(), nullable=False),
        sa.Column("failed_logins", sa.Integer(), nullable=False),
        _ts("locked_until", nullable=True),
        _ts("password_changed_at", nullable=True),
        _ts("last_login_at", nullable=True),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "server_refresh_tokens",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("family_id", sa.String(64), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        _ts("created_at"),
        _ts("expires_at"),
        _ts("revoked_at", nullable=True),
        sa.Column("user_agent", sa.String(255), nullable=True),
        sa.Column("ip", sa.String(64), nullable=True),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_server_refresh_tokens_user_id", "server_refresh_tokens", ["user_id"])
    op.create_index("ix_server_refresh_tokens_family_id", "server_refresh_tokens", ["family_id"])
    op.create_table(
        "server_audit",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        _ts("at"),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=True),
        sa.Column("resource", sa.String(255), nullable=True),
        sa.Column("status", sa.Integer(), nullable=True),
        sa.Column("ip", sa.String(64), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
    )
    op.create_index("ix_server_audit_at", "server_audit", ["at"])
    op.create_index("ix_server_audit_user_at", "server_audit", ["user_id", "at"])
    op.create_index("ix_server_audit_project_at", "server_audit", ["project_id", "at"])
    op.create_index("ix_server_audit_action_at", "server_audit", ["action", "at"])


def downgrade() -> None:
    op.drop_table("server_audit")
    op.drop_table("server_refresh_tokens")
    op.drop_table("server_accounts")
    op.drop_table("llm_cache")
    op.drop_table("entities")
