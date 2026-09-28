"""Control de acceso enchufable del Engine (SPEC §4.2: un solo Engine en desktop y servidor).

En el desktop no hay usuarios: la política es abierta y la única barrera es el token efímero
del sidecar. El Team Server (Capa 5) instala su propia política con autenticación y RBAC
(RF-SRV-01/02) sin que los routers del Engine sepan nada de usuarios ni roles.

La política se consulta en tres puntos:
- `authorize`: dependencia de cada router (HTTP y WebSocket), después del ruteo, así que ve
  la operación y los parámetros de ruta.
- `visible_projects`: filtra listados que cruzan proyectos (proyectos, jobs).
- `prepare_project`: completa un proyecto nuevo (p. ej. su workspace) antes de guardarlo.
"""

from __future__ import annotations

from typing import Protocol, cast

from fastapi import Depends
from starlette.requests import HTTPConnection

from perceptron.domain.models import Project


class AccessPolicy(Protocol):
    async def authorize(self, conn: HTTPConnection) -> None:
        """Lanza `AuthError`/`ForbiddenError` (HTTP) o `WebSocketException` si no hay acceso."""
        ...

    def visible_projects(self, conn: HTTPConnection) -> set[str] | None:
        """IDs de proyectos visibles para quien llama; `None` = todos."""
        ...

    def prepare_project(self, conn: HTTPConnection, project: Project) -> Project: ...


class OpenAccess:
    """Desktop y CLI: un solo usuario local, sin restricciones por proyecto."""

    async def authorize(self, conn: HTTPConnection) -> None:
        return None

    def visible_projects(self, conn: HTTPConnection) -> set[str] | None:
        return None

    def prepare_project(self, conn: HTTPConnection, project: Project) -> Project:
        return project


def get_access(conn: HTTPConnection) -> AccessPolicy:
    return cast(AccessPolicy, conn.app.state.access)


async def _authorize(conn: HTTPConnection) -> None:
    await get_access(conn).authorize(conn)


# Dependencia que el factory agrega a todos los routers del Engine.
authorized = Depends(_authorize)
