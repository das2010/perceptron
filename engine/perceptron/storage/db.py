"""Base de datos de metadata (SQLAlchemy 2).

Local: SQLite en `<workspace>/perceptron.db` con WAL. Servidor (Capa 5): PostgreSQL
con la misma capa de repositorios.

Esquema de Capa 0: una tabla de documentos por entidad (`entities`) con columnas
indexadas para las consultas frecuentes y el cuerpo en JSON. Ver ADR-0013.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Engine,
    Index,
    Integer,
    String,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class EntityRow(Base):
    __tablename__ = "entities"

    kind: Mapped[str] = mapped_column(String(64), primary_key=True)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    __table_args__ = (Index("ix_entities_kind_created", "kind", "created_at"),)


class LLMCacheRow(Base):
    """Caché de respuestas del LLM por hash de (prompt, modelo, parámetros) (RF-LLM-07)."""

    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


def sqlite_url(db_file: Path) -> str:
    # as_posix(): SQLAlchemy acepta "/" en Windows y evita escapes con espacios/acentos.
    return f"sqlite:///{db_file.resolve().as_posix()}"


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


class Database:
    def __init__(self, url: str) -> None:
        self.engine: Engine = create_engine(url, future=True)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _sqlite_pragmas)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @classmethod
    def for_file(cls, db_file: Path) -> Database:
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite_url(db_file))

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Unidad de trabajo transaccional (SPEC §13.3)."""
        with self._sessions() as s, s.begin():
            yield s

    def dispose(self) -> None:
        self.engine.dispose()
