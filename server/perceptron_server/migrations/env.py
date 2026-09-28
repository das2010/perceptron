"""Entorno de Alembic del Team Server: esquema del Engine + tablas propias del servidor."""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection, create_engine, pool

from perceptron_server.migrate import target_metadata

config = context.config


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata(),
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(
        url=config.attributes["url"],
        target_metadata=target_metadata(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_engine(config.attributes["url"], poolclass=pool.NullPool)
    try:
        with engine.connect() as conn:
            _run(conn)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
