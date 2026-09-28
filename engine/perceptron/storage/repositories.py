"""Repositorio genérico SQL sobre la tabla de documentos `entities`."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from sqlalchemy import select

from perceptron.core.errors import ConflictError, NotFoundError, ValidationError
from perceptron.domain.models import Entity, utcnow
from perceptron.storage.db import Database, EntityRow

_INDEXED_FILTERS = {"project_id"}


class SqlRepository[E: Entity]:
    def __init__(self, db: Database, model: type[E]) -> None:
        self.db = db
        self.model = model
        self.kind = model.__name__

    # -- helpers ---------------------------------------------------------------

    def _to_row(self, entity: E) -> EntityRow:
        return EntityRow(
            kind=self.kind,
            id=entity.id,
            project_id=getattr(entity, "project_id", None),
            version=entity.version,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            data=entity.model_dump(mode="json"),
        )

    def _from_row(self, row: EntityRow) -> E:
        return self.model.model_validate(row.data)

    # -- API -------------------------------------------------------------------

    def add(self, entity: E) -> E:
        with self.db.session() as s:
            if s.get(EntityRow, (self.kind, entity.id)) is not None:
                raise ConflictError(f"{self.kind} {entity.id} ya existe")
            s.add(self._to_row(entity))
        return entity

    def find(self, entity_id: str) -> E | None:
        with self.db.session() as s:
            row = s.get(EntityRow, (self.kind, entity_id))
            return self._from_row(row) if row is not None else None

    def get(self, entity_id: str) -> E:
        entity = self.find(entity_id)
        if entity is None:
            raise NotFoundError(
                f"{self.kind} {entity_id} no existe", details={"kind": self.kind, "id": entity_id}
            )
        return entity

    def list(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        ids: Collection[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[E]:
        stmt = select(EntityRow).where(EntityRow.kind == self.kind)
        if ids is not None:
            stmt = stmt.where(EntityRow.id.in_(list(ids)))
        for key, value in (filters or {}).items():
            if key in _INDEXED_FILTERS:
                stmt = stmt.where(getattr(EntityRow, key) == value)
            elif key in self.model.model_fields:
                stmt = stmt.where(EntityRow.data[key].as_string() == str(value))
            else:
                raise ValidationError(f"filtro desconocido para {self.kind}: {key}")
        stmt = (
            stmt.order_by(EntityRow.created_at.desc(), EntityRow.id.desc())
            .limit(limit)
            .offset(offset)
        )
        with self.db.session() as s:
            return [self._from_row(r) for r in s.scalars(stmt)]

    def update(self, entity: E) -> E:
        with self.db.session() as s:
            row = s.get(EntityRow, (self.kind, entity.id), with_for_update=True)
            if row is None:
                raise NotFoundError(f"{self.kind} {entity.id} no existe")
            if row.version != entity.version:
                raise ConflictError(
                    f"{self.kind} {entity.id}: versión {entity.version} obsoleta",
                    details={"expected": row.version, "got": entity.version},
                )
            updated = entity.model_copy(
                update={"version": entity.version + 1, "updated_at": utcnow()}
            )
            new_row = self._to_row(updated)
            row.version = new_row.version
            row.updated_at = new_row.updated_at
            row.project_id = new_row.project_id
            row.data = new_row.data
        return updated

    def delete(self, entity_id: str) -> None:
        with self.db.session() as s:
            row = s.get(EntityRow, (self.kind, entity_id))
            if row is None:
                raise NotFoundError(f"{self.kind} {entity_id} no existe")
            s.delete(row)
