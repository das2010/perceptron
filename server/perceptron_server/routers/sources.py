"""`/server/sources` — «fuentes del servidor» (RF-SRV-05).

En el navegador no hay selector de carpetas locales: además de subir archivos, un Editor
puede elegir datos de carpetas montadas en el servidor que el Admin habilitó
(`PERCEPTRON_SOURCE_ROOTS`). El Engine rechaza cualquier ruta fuera de esas raíces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from perceptron.core.errors import ForbiddenError, NotFoundError, ValidationError
from perceptron.core.paths import within_roots
from perceptron.domain.enums import Role
from perceptron_server.accounts import Principal
from perceptron_server.routers.auth import Who
from perceptron_server.state import ServerState, server_state

router = APIRouter(prefix="/server/sources", tags=["server"])
State = Annotated[ServerState, Depends(server_state)]
MAX_ENTRIES = 1000


def editor(who: Who) -> Principal:
    if not who.has_role_anywhere(Role.EDITOR):
        raise ForbiddenError("requiere rol Editor")
    return who


Editor = Annotated[Principal, Depends(editor)]


class ServerSourceRoot(BaseModel):
    index: int
    name: str
    path: str


class ServerEntry(BaseModel):
    name: str
    path: str
    kind: Literal["dir", "file"]
    size: int | None = None


class ServerListing(BaseModel):
    root: ServerSourceRoot
    path: str
    entries: list[ServerEntry]
    truncated: bool


def _roots(state: ServerState) -> list[Path]:
    return list(state.settings.source_roots or [])


def _root(state: ServerState, index: int) -> ServerSourceRoot:
    roots = _roots(state)
    if not 0 <= index < len(roots):
        raise NotFoundError(f"no existe la fuente del servidor {index}")
    root = roots[index]
    return ServerSourceRoot(index=index, name=root.name or str(root), path=str(root))


@router.get("", operation_id="listServerSources")
def list_roots(_: Editor, state: State) -> list[ServerSourceRoot]:
    return [_root(state, i) for i in range(len(_roots(state)))]


@router.get("/{index}/browse", operation_id="browseServerSource")
def browse(
    index: int,
    _: Editor,
    state: State,
    path: Annotated[str, Query(max_length=1024)] = "",
) -> ServerListing:
    root = _root(state, index)
    base = Path(root.path)
    target = (base / path) if path else base
    if not within_roots(target, [base]):
        raise ForbiddenError("la ruta sale de la fuente habilitada")
    target = target.resolve()
    if not target.is_dir():
        raise ValidationError(f"no es una carpeta: {path}")
    entries: list[ServerEntry] = []
    children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for child in children[:MAX_ENTRIES]:
        if child.name.startswith(".") or not within_roots(child, [base]):
            continue  # ocultos y symlinks que apuntan afuera
        is_dir = child.is_dir()
        entries.append(
            ServerEntry(
                name=child.name,
                path=str(child),
                kind="dir" if is_dir else "file",
                size=None if is_dir else child.stat().st_size,
            )
        )
    rel = target.relative_to(base.resolve()).as_posix()
    return ServerListing(
        root=root,
        path="" if rel == "." else rel,
        entries=entries,
        truncated=len(children) > MAX_ENTRIES,
    )
