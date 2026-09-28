"""Auditoría del Team Server (RF-SRV-07): append-only, consultable por la consola de admin."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from perceptron.domain.models import utcnow
from perceptron.storage.db import Database
from perceptron_server.db import AuditRow

# Operaciones del Engine que se auditan aunque sean lecturas: acceso a datos y descargas.
AUDITED_READS = frozenset(
    {
        "getDatasetSamples",
        "labelSampleFile",
        "exportLabels",
        "downloadRunExport",
        "downloadExportProject",
        "downloadServingBundle",
        "getReportDocument",
        "listLlmAudit",
    }
)


class AuditEvent(BaseModel):
    id: int
    at: datetime
    user_id: str | None
    action: str
    project_id: str | None
    resource: str | None
    status: int | None
    ip: str | None
    details: dict[str, Any]


class AuditLog:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        action: str,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        resource: str | None = None,
        status: int | None = None,
        ip: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.db.session() as s:
            s.add(
                AuditRow(
                    at=utcnow(),
                    user_id=user_id,
                    action=action[:64],
                    project_id=project_id,
                    resource=(resource or "")[:255] or None,
                    status=status,
                    ip=ip,
                    details=details or {},
                )
            )

    def query(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        action: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[AuditEvent]:
        stmt = select(AuditRow)
        if user_id:
            stmt = stmt.where(AuditRow.user_id == user_id)
        if project_id:
            stmt = stmt.where(AuditRow.project_id == project_id)
        if action:
            stmt = stmt.where(AuditRow.action.startswith(action))
        if since:
            stmt = stmt.where(AuditRow.at >= since)
        stmt = stmt.order_by(AuditRow.id.desc()).limit(limit).offset(offset)
        with self.db.session() as s:
            return [AuditEvent.model_validate(r, from_attributes=True) for r in s.scalars(stmt)]
