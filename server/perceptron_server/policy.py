"""RBAC del Team Server sobre la API del Engine (RF-SRV-02, SPEC §3.2).

Cada operación del Engine requiere un permiso sobre los proyectos que toca:
- **view** (Viewer): leer proyectos, runs e informes, previsualizar y usar el playground.
- **edit** (Editor): crear y modificar, entrenar, etiquetar, exportar y descargar.
- **admin** (Admin del proyecto o del workspace): borrar el proyecto.
- **server_admin**: configuración global (proveedores y perfiles LLM).

El proyecto se deduce de los parámetros de ruta (cualquier `*_id` es una entidad con
`project_id`), del query `project` o, en pocas operaciones, del cuerpo JSON. Por defecto
GET = view y el resto = edit; las excepciones están en las tablas de abajo.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from fastapi import WebSocketException
from sqlalchemy import select
from starlette.requests import HTTPConnection, Request

from perceptron.api.context import EngineContext
from perceptron.core.errors import AuthError, ForbiddenError, NotFoundError
from perceptron.domain.enums import ProjectScope, Role
from perceptron.domain.models import Project, Workspace
from perceptron.storage.db import EntityRow
from perceptron_server.accounts import ROLE_RANK, Accounts, Principal

WS_UNAUTHORIZED = 4401
WS_FORBIDDEN = 4403


class Perm(StrEnum):
    PUBLIC = "public"
    AUTH = "auth"
    VIEW = "view"
    EDIT = "edit"
    ADMIN = "admin"
    SERVER_ADMIN = "server_admin"


PUBLIC_OPS = frozenset({"getHealth", "getVersion"})
SERVER_ADMIN_OPS = frozenset(
    {"putLlmProfiles", "putLlmProvider", "testLlm", "putLicense", "putTelemetry"}
)
PROJECT_ADMIN_OPS = frozenset({"deleteProject"})
# Escrituras que un Viewer puede hacer: no modifican nada (previsualizar, comparar, playground).
VIEW_WRITES = frozenset(
    {
        "previewSource",
        "previewPipeline",
        "compareRuns",
        "predictRows",
        "predictFile",
        "explainRow",
        "explainImage",
        "validateArchitecture",
        "lintArchCode",
        "archToCode",
        "predictDeployment",  # usar el modelo en uso (como el playground)
        "openRunInMlflow",  # solo arma el enlace (en el servidor, la URL configurada)
    }
)
# Lecturas reservadas a Editor: descargas de exportaciones y la auditoría LLM del proyecto.
EDIT_READS = frozenset(
    {
        "downloadRunExport",
        "downloadExportProject",
        "downloadServingBundle",
        "exportLabels",
        "listLlmAudit",
    }
)
# WebSockets (no tienen operation_id): el copiloto escribe el borrador; el resto es lectura.
WS_EDIT_SUFFIXES = ("/copilot",)
# Cuerpos que referencian entidades de otro proyecto (comparar runs, pre-etiquetar).
BODY_ID_OPS = {"compareRuns": ("run_ids",), "prelabel": ("dataset_version_id",)}
_SKIP_PARAMS = frozenset({"name", "sample_id"})
_NEED_RANK = {Perm.VIEW: Role.VIEWER, Perm.EDIT: Role.EDITOR, Perm.ADMIN: Role.ADMIN}


def operation_of(conn: HTTPConnection) -> str:
    route = conn.scope.get("route")
    op = getattr(route, "operation_id", None)
    if op:
        return str(op)
    return f"ws:{getattr(route, 'path', conn.url.path)}"


def required_permission(op: str, method: str) -> Perm:
    if op in PUBLIC_OPS:
        return Perm.PUBLIC
    if op in SERVER_ADMIN_OPS:
        return Perm.SERVER_ADMIN
    if op in PROJECT_ADMIN_OPS:
        return Perm.ADMIN
    if op.startswith("ws:"):
        return Perm.EDIT if op.endswith(WS_EDIT_SUFFIXES) else Perm.VIEW
    if op in EDIT_READS:
        return Perm.EDIT
    if op in VIEW_WRITES or method in {"GET", "HEAD"}:
        return Perm.VIEW
    return Perm.EDIT


def principal_of(conn: HTTPConnection) -> Principal | None:
    value = getattr(conn.state, "principal", None)
    return value if isinstance(value, Principal) else None


class ServerAccess:
    """`AccessPolicy` del Engine para el Team Server."""

    def __init__(self) -> None:
        self._ctx: EngineContext | None = None
        self._accounts: Accounts | None = None

    def bind(self, ctx: EngineContext, accounts: Accounts) -> None:
        self._ctx, self._accounts = ctx, accounts

    @property
    def ctx(self) -> EngineContext:
        if self._ctx is None:
            raise RuntimeError("ServerAccess sin inicializar")
        return self._ctx

    # ------------------------------------------------------------------ resolución

    def project_of(self, entity_id: str) -> str | None:
        with self.ctx.db.session() as s:
            row = s.execute(
                select(EntityRow.kind, EntityRow.project_id).where(EntityRow.id == entity_id)
            ).first()
        if row is None:
            return None
        kind, project_id = row
        return entity_id if kind == "Project" else project_id

    async def _projects(self, conn: HTTPConnection, op: str) -> tuple[set[str], bool]:
        """Proyectos que toca el request y si hay un job sin proyecto (solo lo ve el admin).

        Una entidad inexistente no suma proyecto: el Engine responde 404."""
        ids: set[str] = set()
        orphan_job = False

        def add(entity_id: object) -> None:
            if isinstance(entity_id, str) and (pid := self.project_of(entity_id)):
                ids.add(pid)

        for name, value in conn.path_params.items():
            if name in _SKIP_PARAMS:
                continue
            if name == "project_id":
                ids.add(str(value))
            elif name == "job_id":
                job = self.ctx.jobs.get(str(value))
                if job is not None and (pid := job.refs.get("project_id")):
                    ids.add(pid)
                elif job is not None:
                    orphan_job = True
            else:
                add(value)
        if project := conn.query_params.get("project"):
            ids.add(project)
        keys = BODY_ID_OPS.get(op)
        if keys and isinstance(conn, Request):
            try:
                body: Any = json.loads(await conn.body() or b"{}")
            except ValueError:
                body = {}
            for key in keys:
                value = body.get(key) if isinstance(body, dict) else None
                for item in value if isinstance(value, list) else [value]:
                    add(item)
        return ids, orphan_job

    # ------------------------------------------------------------------ AccessPolicy

    async def authorize(self, conn: HTTPConnection) -> None:
        op = operation_of(conn)
        perm = required_permission(op, conn.scope.get("method", "GET"))
        conn.state.operation = op
        if perm is Perm.PUBLIC:
            return
        websocket = conn.scope["type"] == "websocket"
        principal = principal_of(conn)
        if principal is None:
            if websocket:
                raise WebSocketException(code=WS_UNAUTHORIZED, reason="no autenticado")
            raise AuthError("iniciá sesión para continuar")
        try:
            await self._check(conn, op, perm, principal)
        except ForbiddenError as exc:
            if websocket:
                raise WebSocketException(code=WS_FORBIDDEN, reason=exc.message) from None
            raise

    async def _check(self, conn: HTTPConnection, op: str, perm: Perm, who: Principal) -> None:
        if perm is Perm.SERVER_ADMIN:
            if not who.is_server_admin:
                raise ForbiddenError("requiere ser administrador del servidor")
            return
        projects, orphan_job = await self._projects(conn, op)
        conn.state.audit_project = min(projects) if projects else None
        if orphan_job and not who.is_server_admin:
            raise ForbiddenError("sin acceso a este job")
        if not projects:
            if perm is not Perm.VIEW and not who.has_role_anywhere(Role.EDITOR):
                raise ForbiddenError("tu rol no permite esta operación")
            return
        need = _NEED_RANK.get(perm, Role.VIEWER)
        for pid in projects:
            project = self.ctx.projects.find(pid)
            if project is None:
                continue  # el Engine responde 404
            role = who.project_role(project)
            if role is None:
                raise ForbiddenError("no tenés acceso a este proyecto", details={"project": pid})
            if ROLE_RANK[role] < ROLE_RANK[need]:
                raise ForbiddenError(
                    f"tu rol en el proyecto ({role.value}) no permite esta operación",
                    details={"project": pid, "required": need.value},
                )

    def visible_projects(self, conn: HTTPConnection) -> set[str] | None:
        who = principal_of(conn)
        if who is None:
            return set()
        if who.is_server_admin:
            return None
        workspaces = who.workspaces()
        visible = {m.project_id for m in who.memberships if m.project_id}
        if workspaces:
            with self.ctx.db.session() as s:
                visible.update(
                    s.scalars(
                        select(EntityRow.id).where(
                            EntityRow.kind == "Project",
                            EntityRow.data["workspace_id"].as_string().in_(sorted(workspaces)),
                        )
                    )
                )
        return visible

    def prepare_project(self, conn: HTTPConnection, project: Project) -> Project:
        who = principal_of(conn)
        if who is None:
            raise AuthError("iniciá sesión para continuar")
        workspace_id = project.workspace_id or self._default_workspace(who)
        if workspace_id is None:
            raise ForbiddenError("no tenés un workspace donde crear proyectos")
        if self.ctx.repo(Workspace).find(workspace_id) is None:
            raise NotFoundError(f"workspace {workspace_id} no existe")
        role = who.workspace_role(workspace_id)
        if role is None or ROLE_RANK[role] < ROLE_RANK[Role.EDITOR]:
            raise ForbiddenError("tu rol en el workspace no permite crear proyectos")
        return project.model_copy(update={"workspace_id": workspace_id, "scope": ProjectScope.TEAM})

    def _default_workspace(self, who: Principal) -> str | None:
        if self._accounts is None:
            raise RuntimeError("ServerAccess sin inicializar")
        editable = who.workspaces(Role.EDITOR)
        default = self._accounts.ensure_default_workspace()
        if who.is_server_admin or default.id in editable:
            return default.id
        return min(editable) if editable else None
