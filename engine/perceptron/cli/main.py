"""Punto de entrada de la CLI `perceptron`.

Capa 0: `--version`, `serve`, `openapi`, `project list|create`. El resto de los
subcomandos de §7.18 se agregan en sus capas.
"""

from __future__ import annotations

import json
import secrets
import socket
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import SecretStr

from perceptron import __version__
from perceptron.core.config import Settings
from perceptron.domain.enums import Modality, PrivacyLevel, TaskType

app = typer.Typer(name="perceptron", help="Perceptron Engine CLI", no_args_is_help=True)
project_app = typer.Typer(help="Gestión de proyectos", no_args_is_help=True)
app.add_typer(project_app, name="project")

WorkspaceOpt = Annotated[
    Path | None, typer.Option("--workspace", "-w", help="Directorio del workspace")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Salida JSON")]


def _settings(workspace: Path | None) -> Settings:
    return Settings(workspace_dir=workspace) if workspace else Settings()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"perceptron {__version__}")
        raise typer.Exit


@app.callback()
def main(
    _version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Muestra la versión"
        ),
    ] = False,
) -> None:
    """Perceptron: entrenar redes neuronales localmente, guiado por un LLM."""


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


@app.command()
def serve(
    workspace: WorkspaceOpt = None,
    host: Annotated[str, typer.Option(help="Interfaz de escucha")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="0 = puerto aleatorio")] = 0,
    token: Annotated[
        str | None, typer.Option(envvar="PERCEPTRON_API__TOKEN", help="Token exigido por la API")
    ] = None,
    new_token: Annotated[bool, typer.Option(help="Genera un token efímero aleatorio")] = False,
) -> None:
    """Levanta el Engine (API REST + WS). Emite una línea JSON `ready` por stdout."""
    import uvicorn

    from perceptron.api.app import create_app
    from perceptron.core.logging import configure_logging

    settings = _settings(workspace)
    if new_token:
        token = secrets.token_urlsafe(32)
    port = port or _free_port(host)
    settings.api.host, settings.api.port = host, port
    settings.api.token = SecretStr(token) if token else None
    configure_logging(settings.logging, settings.paths.logs_dir)

    # Handshake con el proceso padre (Tauri sidecar): puerto y token por stdout.
    ready = {"event": "ready", "host": host, "port": port, "token": token}
    sys.stdout.write(json.dumps(ready) + "\n")
    sys.stdout.flush()
    uvicorn.run(create_app(settings), host=host, port=port, log_config=None)


@app.command()
def openapi(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Archivo destino")] = None,
) -> None:
    """Exporta el schema OpenAPI del Engine (para `pnpm gen:api`)."""
    from perceptron.api.app import create_app

    schema = create_app(Settings()).openapi()
    text = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)


@project_app.command("list")
def project_list(workspace: WorkspaceOpt = None, as_json: JsonOpt = False) -> None:
    from perceptron.api.context import EngineContext

    ctx = EngineContext.create(_settings(workspace))
    try:
        items = ctx.projects.list()
        if as_json:
            typer.echo(json.dumps([p.model_dump(mode="json") for p in items], ensure_ascii=False))
            return
        if not items:
            typer.echo("(sin proyectos)")
        for p in items:
            typer.echo(f"{p.id}  {p.name}  [{p.status.value}]")
    finally:
        ctx.close()


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="Nombre del proyecto")],
    workspace: WorkspaceOpt = None,
    goal: Annotated[str, typer.Option(help="Objetivo en lenguaje natural")] = "",
    modality: Annotated[list[Modality] | None, typer.Option(help="Modalidad (repetible)")] = None,
    task: Annotated[TaskType | None, typer.Option(help="Tipo de tarea")] = None,
    privacy: Annotated[
        PrivacyLevel, typer.Option(help="Nivel de privacidad LLM")
    ] = PrivacyLevel.L1,
    as_json: JsonOpt = False,
) -> None:
    from perceptron.api.context import EngineContext
    from perceptron.domain.models import Project

    ctx = EngineContext.create(_settings(workspace))
    try:
        project = ctx.projects.add(
            Project(
                name=name, goal=goal, modalities=modality or [], task=task, privacy_level=privacy
            )
        )
        ctx.files.init_project(project)
        if as_json:
            typer.echo(project.model_dump_json())
        else:
            typer.echo(f"Proyecto creado: {project.id}")
    finally:
        ctx.close()


if __name__ == "__main__":  # pragma: no cover
    app()
