"""Migraciones del esquema (Alembic, SPEC §5.4).

Corren al arrancar el servidor y con `perceptron-server migrate`.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import MetaData, create_engine, pool

from perceptron.storage.db import Base
from perceptron_server.db import ServerBase

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def target_metadata() -> MetaData:
    """Engine + servidor en una sola MetaData (autogenerate y chequeo de deriva)."""
    combined = MetaData()
    for meta in (Base.metadata, ServerBase.metadata):
        for table in meta.tables.values():
            table.to_metadata(combined)
    return combined


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # Por atributos y no por opción: configparser interpretaría los `%` de la contraseña.
    cfg.attributes["url"] = url
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str) -> None:
    command.downgrade(alembic_config(url), revision)


def current_revision(url: str) -> str | None:
    engine = create_engine(url, poolclass=pool.NullPool)
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()
