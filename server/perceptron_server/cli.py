"""CLI del Team Server: `perceptron-server serve|migrate|create-user|openapi`."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import SecretStr

from perceptron.domain.enums import Role

app = typer.Typer(name="perceptron-server", help="Perceptron Team Server", no_args_is_help=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interfaz de escucha")] = "0.0.0.0",  # noqa: S104
    port: Annotated[int, typer.Option(help="Puerto")] = 8080,
    forwarded_allow_ips: Annotated[
        str,
        typer.Option(
            envvar="FORWARDED_ALLOW_IPS", help="Proxies confiables para X-Forwarded-For/Proto"
        ),
    ] = "127.0.0.1",
) -> None:
    """Migra la base, crea el primer Admin (si se configuró) y levanta API + UI web."""
    import uvicorn

    from perceptron.core.config import get_settings
    from perceptron.core.logging import configure_logging
    from perceptron_server.app import create_server_app

    settings = get_settings()
    configure_logging(settings.logging, settings.paths.logs_dir)
    uvicorn.run(
        create_server_app(settings),
        host=host,
        port=port,
        log_config=None,
        proxy_headers=True,
        forwarded_allow_ips=forwarded_allow_ips,
        # Jobs y eventos viven en memoria del proceso: un solo worker hasta la cola (5b).
        workers=1,
    )


@app.command()
def worker(
    queues: Annotated[str, typer.Option(help="Colas que atiende (gpu, cpu)")] = "cpu",
    name: Annotated[
        str | None, typer.Option(envvar="PERCEPTRON_WORKER_NAME", help="Nombre del worker")
    ] = None,
) -> None:
    """Worker de la cola (Capa 5b): ejecuta estudios en esta máquina (CPU o GPU)."""
    from perceptron.core.config import get_settings
    from perceptron.core.logging import configure_logging
    from perceptron_server.queue.dispatch import QUEUES, TASK_NAME, make_celery
    from perceptron_server.queue.relay import RedisRelay
    from perceptron_server.queue.worker import WorkerRuntime
    from perceptron_server.settings import ServerSettings

    wanted = [q.strip() for q in queues.split(",") if q.strip()]
    if not wanted or set(wanted) - set(QUEUES):
        raise typer.BadParameter("colas válidas: " + ", ".join(QUEUES))
    server = ServerSettings()
    if server.redis_url is None:
        raise typer.BadParameter("definí PERCEPTRON_SERVER__REDIS_URL")
    url = server.redis_url.get_secret_value()
    if name:
        os.environ["PERCEPTRON_WORKER_NAME"] = name
    settings = get_settings()
    configure_logging(settings.logging, settings.paths.logs_dir)
    runtime = WorkerRuntime.create(
        settings, RedisRelay(url), wanted, heartbeat_s=server.worker_heartbeat_s
    )
    runtime.start_heartbeat()
    celery = make_celery(url)
    celery.task(name=TASK_NAME)(runtime.run_study)
    try:
        # Pool "solo": el entrenamiento lanza sus propios subprocesos (no daemonic).
        celery.worker_main(
            [
                "worker",
                "--queues",
                ",".join(wanted),
                "--pool",
                "solo",
                "--hostname",
                f"{runtime.name}@%h",
                "--without-gossip",
                "--without-mingle",
                "--loglevel",
                settings.logging.level,
            ]
        )
    finally:
        runtime.stop()


@app.command()
def migrate() -> None:
    """Aplica las migraciones pendientes (Alembic)."""
    from perceptron.core.config import get_settings
    from perceptron_server.app import database_url
    from perceptron_server.migrate import current_revision, upgrade

    url = database_url(get_settings())
    upgrade(url)
    typer.echo(f"esquema en la revisión {current_revision(url)}")


@app.command("create-user")
def create_user(
    email: str,
    name: Annotated[str, typer.Option(help="Nombre visible")] = "",
    admin: Annotated[bool, typer.Option(help="Administrador del servidor")] = False,
    role: Annotated[Role, typer.Option(help="Rol en el workspace por defecto")] = Role.VIEWER,
) -> None:
    """Crea un usuario local. La contraseña se pide por consola o por PERCEPTRON_NEW_PASSWORD."""
    from perceptron.api.context import EngineContext
    from perceptron.core.config import get_settings
    from perceptron_server.app import database_url
    from perceptron_server.migrate import upgrade
    from perceptron_server.settings import ServerSettings
    from perceptron_server.state import ServerState

    password = os.environ.get("PERCEPTRON_NEW_PASSWORD") or typer.prompt(
        "Contraseña", hide_input=True, confirmation_prompt=True
    )
    settings = get_settings()
    upgrade(database_url(settings))
    state = ServerState(settings, ServerSettings())
    ctx = EngineContext.create(settings)
    try:
        state.bind(ctx)
        workspace = state.accounts.ensure_default_workspace()
        user = state.accounts.create_user(email, name, password, is_server_admin=admin)
        state.accounts.grant(user.id, workspace.id, Role.ADMIN if admin else role)
        state.audit.record("admin.user_created", resource=user.id, details={"via": "cli"})
        typer.echo(f"usuario {user.email} creado ({user.id})")
    finally:
        ctx.close()


@app.command()
def openapi(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Archivo destino")] = None,
) -> None:
    """Schema OpenAPI del Team Server (Engine + auth/admin) para `pnpm gen:api`."""
    from perceptron.core.config import Settings
    from perceptron_server.app import create_server_app
    from perceptron_server.settings import ServerSettings

    server = ServerSettings(secret_key=SecretStr("x" * 32))
    schema = create_server_app(Settings(), server, migrate=False).openapi()
    text = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
