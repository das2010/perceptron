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
from perceptron.cli import ml
from perceptron.core.config import Settings
from perceptron.domain.enums import Modality, PrivacyLevel, TaskType

app = typer.Typer(name="perceptron", help="Perceptron Engine CLI", no_args_is_help=True)
project_app = typer.Typer(help="Gestión de proyectos", no_args_is_help=True)
app.add_typer(project_app, name="project")
system_app = typer.Typer(help="Información del sistema", no_args_is_help=True)
app.add_typer(system_app, name="system")
ml.register(app)

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
    _tolerant_console()


def _tolerant_console() -> None:
    """Consola de Windows en cp1252: un carácter que no entra (→, emojis) se reemplaza en vez de
    abortar el comando. Para scripts, `PYTHONUTF8=1` da UTF-8 exacto."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


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


@system_app.command("hardware")
def system_hardware(workspace: WorkspaceOpt = None, as_json: JsonOpt = False) -> None:
    """Hardware detectado y dispositivo recomendado (RF-TRN-01)."""
    from perceptron.training.hardware import detect_hardware

    report = detect_hardware(_settings(workspace).workspace_dir)
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(f"SO:      {report.os} · Python {report.python}")
    typer.echo(f"CPU:     {report.cpu.model} ({report.cpu.logical_cores} hilos)")
    typer.echo(f"RAM:     {report.ram_available_gb} / {report.ram_total_gb} GB libres")
    typer.echo(f"Disco:   {report.disk_free_gb} GB libres")
    for g in report.gpus:
        typer.echo(f"GPU {g.index}:   {g.name} [{g.backend.value}] {g.vram_total_gb} GB")
    torch_desc = (
        f"{report.torch.version} ({report.torch.variant})"
        if report.torch.installed
        else "no instalado"
    )
    typer.echo(f"PyTorch: {torch_desc}")
    typer.echo(
        f"Recomendado: dispositivo {report.recommended_device.value}, "
        f"variante de PyTorch {report.recommended_torch_variant}"
    )
    for note in report.notes:
        typer.echo(f"Nota: {note}")


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
    template: Annotated[
        str | None, typer.Option(help="Plantilla de caso de uso (churn, visual-inspection…)")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    from perceptron.api.context import EngineContext
    from perceptron.api.routers.projects import apply_template
    from perceptron.domain.models import Project

    ctx = EngineContext.create(_settings(workspace))
    try:
        values = apply_template(
            {
                "name": name,
                "goal": goal,
                "modalities": modality or [],
                "task": task,
                "privacy_level": privacy,
                "template": template,
            }
        )
        project = ctx.projects.add(Project.model_validate(values))
        ctx.files.init_project(project)
        if as_json:
            typer.echo(project.model_dump_json())
        else:
            typer.echo(f"Proyecto creado: {project.id}")
    finally:
        ctx.close()


@project_app.command("duplicate")
def project_duplicate(
    project_id: Annotated[str, typer.Argument(help="Proyecto a duplicar")],
    name: Annotated[str | None, typer.Option(help="Nombre del nuevo proyecto")] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Proyecto nuevo con la misma configuración, sin datos ni runs (RF-PRJ-01)."""
    from perceptron.api.context import EngineContext
    from perceptron.services.projects import duplicate_project

    ctx = EngineContext.create(_settings(workspace))
    try:
        project = ctx.projects.add(duplicate_project(ctx, project_id, name))
        ctx.files.init_project(project)
        typer.echo(project.model_dump_json() if as_json else f"Proyecto creado: {project.id}")
    finally:
        ctx.close()


@project_app.command("archive")
def project_archive(
    project_id: Annotated[str, typer.Argument(help="Proyecto")],
    restore: Annotated[bool, typer.Option("--restore", help="Sacarlo del archivo")] = False,
    workspace: WorkspaceOpt = None,
) -> None:
    """Archiva (o restaura) un proyecto: se conserva todo y deja de aparecer en la lista."""
    from perceptron.api.context import EngineContext
    from perceptron.domain.enums import ProjectStatus

    ctx = EngineContext.create(_settings(workspace))
    try:
        current = ctx.projects.get(project_id)
        status = ProjectStatus.ACTIVE if restore else ProjectStatus.ARCHIVED
        updated = ctx.projects.update(current.model_copy(update={"status": status}))
        ctx.files.write_project(updated)
        typer.echo(f"{updated.id}  [{updated.status.value}]")
    finally:
        ctx.close()


@project_app.command("export")
def project_export(
    project_id: Annotated[str, typer.Argument(help="Proyecto")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Archivo .perceptron a generar")],
    with_data: Annotated[bool, typer.Option("--with-data", help="Incluir los datos")] = False,
    workspace: WorkspaceOpt = None,
) -> None:
    """Exporta el proyecto como paquete .perceptron (RF-PRJ-03)."""
    from perceptron.api.context import EngineContext
    from perceptron.services.project_package import export_project

    ctx = EngineContext.create(_settings(workspace))
    try:
        path = export_project(ctx, project_id, out, include_data=with_data)
        typer.echo(f"Paquete: {path} ({path.stat().st_size / 1024**2:.1f} MB)")
    finally:
        ctx.close()


@project_app.command("import")
def project_import(
    package: Annotated[Path, typer.Argument(help="Archivo .perceptron")],
    workspace: WorkspaceOpt = None,
) -> None:
    """Importa un paquete .perceptron (conserva los ids; falla si el proyecto ya existe)."""
    from perceptron.api.context import EngineContext
    from perceptron.services.project_package import import_project

    ctx = EngineContext.create(_settings(workspace))
    try:
        project = import_project(ctx, package)
        typer.echo(f"Proyecto importado: {project.id}  {project.name}")
    finally:
        ctx.close()


@project_app.command("delete")
def project_delete(
    project_id: Annotated[str, typer.Argument(help="Proyecto")],
    yes: Annotated[bool, typer.Option("--yes", help="Confirmar el borrado (irreversible)")] = False,
    workspace: WorkspaceOpt = None,
) -> None:
    """Elimina el proyecto con sus datos, runs, modelos y carpeta. Irreversible."""
    from perceptron.api.context import EngineContext
    from perceptron.services.projects import purge_project

    if not yes:
        typer.echo("El borrado es irreversible: repetí el comando con --yes para confirmar.")
        raise typer.Exit(code=2)
    ctx = EngineContext.create(_settings(workspace))
    try:
        removed = purge_project(ctx, project_id)
        typer.echo(f"Proyecto eliminado ({removed['entities']} entidades)")
    finally:
        ctx.close()


if __name__ == "__main__":  # pragma: no cover
    app()


# ------------------------------------------------------------------ licencias (ADR-0035)

license_app = typer.Typer(help="Licencias firmadas (Ed25519, offline)", no_args_is_help=True)
app.add_typer(license_app, name="license")


@license_app.command("keygen")
def license_keygen(
    out: Annotated[Path, typer.Option(help="Carpeta segura donde guardar el par")],
    key_id: Annotated[str, typer.Option(help="Identificador de la clave")] = "preteco",
) -> None:
    """Genera el par de claves de Preteco (la privada no se comparte nunca)."""
    from perceptron.licensing.signed import keygen

    private, public = keygen()
    out.mkdir(parents=True, exist_ok=True)
    key_file, pub_file = out / f"{key_id}.key", out / f"{key_id}.pub"
    if key_file.exists():
        raise typer.BadParameter(f"ya existe {key_file}: no se sobrescribe")
    key_file.write_bytes(private)
    pub_file.write_text(public, encoding="utf-8")
    typer.echo(f"privada: {key_file}\npública: {pub_file} (copiala a perceptron/licensing/keys/)")


@license_app.command("issue")
def license_issue(
    key: Annotated[Path, typer.Option(help="Clave privada PEM")],
    key_id: Annotated[str, typer.Option(help="Identificador de la clave")],
    licensee: Annotated[str, typer.Option(help="Cliente")],
    out: Annotated[Path, typer.Option(help="Archivo license.json a generar")],
    edition: Annotated[str, typer.Option()] = "team",
    seats: Annotated[int | None, typer.Option(help="Usuarios del Team Server")] = None,
    servers: Annotated[int | None, typer.Option(help="Instalaciones del Team Server")] = None,
    gpus: Annotated[int | None, typer.Option(help="Workers GPU")] = None,
    features: Annotated[str, typer.Option(help="Claves separadas por coma o *")] = "*",
    days: Annotated[int | None, typer.Option(help="Vigencia en días (sin = perpetua)")] = None,
) -> None:
    """Emite una licencia firmada."""
    import uuid
    from datetime import UTC, datetime, timedelta

    from perceptron.licensing.signed import sign

    now = datetime.now(UTC).replace(microsecond=0)
    terms = {
        "id": str(uuid.uuid4()),
        "licensee": licensee,
        "edition": edition,
        "seats": seats,
        "servers": servers,
        "gpus": gpus,
        "features": [f.strip() for f in features.split(",") if f.strip()],
        "issued_at": now.isoformat(),
        "not_before": now.isoformat(),
        "expires_at": (now + timedelta(days=days)).isoformat() if days else None,
    }
    out.write_text(sign(terms, key.read_bytes(), key_id), encoding="utf-8")
    typer.echo(f"licencia {terms['id']} para {licensee} → {out}")


@license_app.command("verify")
def license_verify(
    file: Annotated[Path, typer.Argument(help="license.json")],
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Verifica firma y vigencia con las claves públicas configuradas."""
    from perceptron.licensing.signed import bundled_keys, verify

    settings = _settings(workspace)
    check = verify(file.read_bytes(), {**bundled_keys(), **settings.license.public_keys})
    if as_json:
        typer.echo(json.dumps({"status": check.status.value, "message": check.message}))
    else:
        who = check.terms.licensee if check.terms else ""
        typer.echo(f"{check.status.value}: {check.message or who}")
    raise typer.Exit(0 if check.valid else 1)
