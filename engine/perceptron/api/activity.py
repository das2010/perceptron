"""Historial de actividad del proyecto (RF-PRJ-05).

Cada operación de escritura exitosa de la API queda anotada como `ActivityEntry`: qué
(operation_id), quién (el usuario del Team Server o el usuario local del desktop) y cuándo.
El proyecto sale de `project_id` o de cualquier `*_id` de la ruta. Las escrituras que no
cambian nada (previsualizar, predecir, comparar, explicar, estimar) no se anotan.
"""

from __future__ import annotations

import getpass
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import Response

from perceptron.domain.models import ActivityEntry
from perceptron.storage.db import EntityRow

logger = logging.getLogger(__name__)

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
QUIET_OPS = frozenset(
    {
        "previewSource",
        "previewPipeline",
        "compareRuns",
        "compareRunConfigs",
        "predictRows",
        "predictFile",
        "predictTexts",
        "predictDeployment",
        "explainRow",
        "explainImage",
        "explainText",
        "explainAudio",
        "validateArchitecture",
        "lintArchCode",
        "archToCode",
        "estimateArchitecture",
        "openRunInMlflow",
        "testLlm",
        "profileDataset",
        "suggestPipelineChanges",
        "recommendHpoStrategy",
        "previewPipelineSteps",
        "planArchDefinition",
    }
)


def _local_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # sin usuario del SO (contenedores)
        return None


LOCAL_USER = _local_user()


def _actor(request: Request) -> str | None:
    principal = request.scope.get("state", {}).get("principal")
    user = getattr(principal, "user", None)
    if user is not None:
        return str(getattr(user, "email", None) or getattr(user, "id", ""))
    return LOCAL_USER


def _project_of(ctx: Any, params: dict[str, Any]) -> str | None:
    if pid := params.get("project_id"):
        return str(pid)
    ids = [str(v) for k, v in params.items() if k.endswith("_id") and isinstance(v, str)]
    if not ids:
        return None
    with ctx.db.session() as s:
        for entity_id in ids:
            row = s.scalars(select(EntityRow).where(EntityRow.id == entity_id).limit(1)).first()
            if row is not None and row.project_id:
                return str(row.project_id)
    return None


def note(request: Request, project_id: str, operation: str, status: int = 201) -> None:
    """Anota una operación cuyo proyecto recién existe al responder (crear, duplicar, importar)."""
    ctx = request.app.state.ctx
    try:
        ctx.repo(ActivityEntry).add(
            ActivityEntry(
                project_id=project_id,
                operation=operation,
                method=request.method,
                path=request.url.path,
                actor=_actor(request),
                status=status,
            )
        )
    except Exception:
        logger.warning("no se pudo anotar la actividad", exc_info=True)


def record(request: Request, response: Response) -> None:
    if request.method not in WRITE_METHODS or not 200 <= response.status_code < 300:
        return
    route = request.scope.get("route")
    op = getattr(route, "operation_id", None)
    ctx = getattr(request.app.state, "ctx", None)
    if not op or op in QUIET_OPS or ctx is None:
        return
    params = dict(request.scope.get("path_params") or {})
    try:
        project_id = _project_of(ctx, params)
        if project_id is None or op in ("deleteProject", "duplicateProject"):
            return  # sin proyecto, o anotado por el endpoint en el proyecto nuevo
        ctx.repo(ActivityEntry).add(
            ActivityEntry(
                project_id=project_id,
                operation=op,
                method=request.method,
                path=request.url.path,
                actor=_actor(request),
                status=response.status_code,
            )
        )
    except Exception:  # el historial nunca rompe una operación
        logger.warning("no se pudo anotar la actividad", exc_info=True)


async def activity_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    if request.method in WRITE_METHODS:
        import anyio

        await anyio.to_thread.run_sync(record, request, response)
    return response
