"""Contratos de repositorio. Implementaciones: `perceptron.storage` (SQLite local,
PostgreSQL en el servidor en Capa 5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

from perceptron.domain.models import Entity

E = TypeVar("E", bound=Entity)


class Repository(Protocol[E]):
    def add(self, entity: E) -> E: ...

    def get(self, entity_id: str) -> E:
        """Lanza `NotFoundError` si no existe."""
        ...

    def find(self, entity_id: str) -> E | None: ...

    def list(
        self, *, filters: Mapping[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> Sequence[E]: ...

    def update(self, entity: E) -> E:
        """Guarda si `entity.version` coincide con la persistida (si no, `ConflictError`)
        e incrementa la versión."""
        ...

    def delete(self, entity_id: str) -> None: ...
