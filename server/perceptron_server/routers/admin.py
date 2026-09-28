"""`/admin` — usuarios, workspaces, membresías y auditoría (RF-SRV-06, parcial en la 5a).

Usuarios, workspaces y auditoría: administrador del servidor. Membresías y el directorio de
usuarios: también el Admin del workspace (gestiona su equipo sin ser admin global).
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from perceptron.core.errors import ForbiddenError
from perceptron.domain.enums import PrivacyLevel, Role
from perceptron.domain.models import Membership, User, Workspace, utcnow
from perceptron_server.accounts import Principal, aware
from perceptron_server.audit import AuditEvent
from perceptron_server.db import AccountRow
from perceptron_server.routers.auth import Who
from perceptron_server.state import ServerState, client_ip, server_state

router = APIRouter(prefix="/admin", tags=["admin"])
State = Annotated[ServerState, Depends(server_state)]


def server_admin(who: Who) -> Principal:
    if not who.is_server_admin:
        raise ForbiddenError("requiere ser administrador del servidor")
    return who


Admin = Annotated[Principal, Depends(server_admin)]


def _require_workspace_admin(who: Principal, workspace_id: str) -> None:
    if who.workspace_role(workspace_id) is not Role.ADMIN:
        raise ForbiddenError("requiere ser Admin del workspace")


class UserAccount(BaseModel):
    user: User
    is_server_admin: bool
    last_login_at: datetime | None
    locked: bool
    has_password: bool


def _account(user: User, row: AccountRow) -> UserAccount:
    return UserAccount(
        user=user,
        is_server_admin=row.is_server_admin,
        last_login_at=aware(row.last_login_at) if row.last_login_at else None,
        locked=bool(row.locked_until and aware(row.locked_until) > utcnow()),
        has_password=row.password_hash is not None,
    )


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=320)
    display_name: str = Field(default="", max_length=200)
    password: str = Field(min_length=1, max_length=256)
    is_server_admin: bool = False
    workspace_id: str | None = Field(default=None, description="Si está, se suma con `role`")
    role: Role = Role.VIEWER


class UserPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = Field(default=None, max_length=200)
    is_active: bool | None = None
    is_server_admin: bool | None = None
    password: str | None = Field(default=None, min_length=1, max_length=256)


class WorkspaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)


class WorkspacePolicy(BaseModel):
    """Políticas del workspace frente al LLM (RF-SRV-06, RF-PRV-02)."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    max_privacy_level: PrivacyLevel | None = None
    local_llm_max_privacy: PrivacyLevel | None = None
    allowed_llm_providers: list[str] | None = Field(
        default=None, description="Proveedores permitidos; lista vacía = ninguno"
    )
    clear_allowed_providers: bool = Field(
        default=False, description="Quita la restricción de proveedores (todos permitidos)"
    )
    llm_monthly_budget_usd: float | None = Field(
        default=None, ge=0, description="Cuota mensual de gasto LLM (RF-LLM-06)"
    )
    clear_llm_budget: bool = Field(default=False, description="Quita la cuota mensual")


class MembershipCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str
    workspace_id: str
    project_id: str | None = None
    role: Role


# ------------------------------------------------------------------ usuarios


def any_admin(who: Who) -> Principal:
    """Admin del servidor o de algún workspace (necesita el directorio para asignar roles)."""
    if not (who.is_server_admin or who.workspaces(Role.ADMIN)):
        raise ForbiddenError("requiere ser Admin del servidor o de un workspace")
    return who


@router.get("/users", operation_id="listUsers")
def list_users(_: Annotated[Principal, Depends(any_admin)], state: State) -> list[UserAccount]:
    return [_account(u, a) for u, a in state.accounts.list_users()]


@router.post("/users", status_code=201, operation_id="createUser")
def create_user(body: UserCreate, who: Admin, request: Request, state: State) -> UserAccount:
    user = state.accounts.create_user(
        body.email, body.display_name, body.password, is_server_admin=body.is_server_admin
    )
    if body.workspace_id:
        state.accounts.grant(user.id, body.workspace_id, body.role)
    state.audit.record(
        "admin.user_created",
        user_id=who.user.id,
        resource=user.id,
        ip=client_ip(request),
        details={
            "email": user.email,
            "is_server_admin": body.is_server_admin,
            "workspace_id": body.workspace_id,
            "role": body.role.value if body.workspace_id else None,
        },
    )
    return _account(*state.accounts.get_user(user.id))


@router.patch("/users/{user_id}", operation_id="updateUser")
def update_user(
    user_id: str, body: UserPatch, who: Admin, request: Request, state: State
) -> UserAccount:
    state.accounts.update_user(
        user_id,
        display_name=body.display_name,
        is_active=body.is_active,
        is_server_admin=body.is_server_admin,
        password=body.password,
    )
    changes = body.model_dump(exclude_none=True, exclude={"password"})
    if body.password is not None:
        changes["password_reset"] = True
    state.audit.record(
        "admin.user_updated",
        user_id=who.user.id,
        resource=user_id,
        ip=client_ip(request),
        details=changes,
    )
    return _account(*state.accounts.get_user(user_id))


# ------------------------------------------------------------------ workspaces


@router.get("/workspaces", operation_id="listWorkspaces")
def list_workspaces(_: Admin, state: State) -> list[Workspace]:
    return list(state.accounts.list_workspaces())


@router.post("/workspaces", status_code=201, operation_id="createWorkspace")
def create_workspace(
    body: WorkspaceCreate, who: Admin, request: Request, state: State
) -> Workspace:
    ws = state.accounts.create_workspace(body.name)
    state.audit.record(
        "admin.workspace_created", user_id=who.user.id, resource=ws.id, ip=client_ip(request)
    )
    return ws


@router.patch("/workspaces/{workspace_id}", operation_id="updateWorkspacePolicy")
def update_workspace(
    workspace_id: str, body: WorkspacePolicy, who: Who, request: Request, state: State
) -> Workspace:
    """Admin del servidor o del workspace: nombre y topes de privacidad y proveedores LLM."""
    _require_workspace_admin(who, workspace_id)
    repo = state.accounts.workspaces
    ws = repo.get(workspace_id)
    changes = body.model_dump(
        exclude_none=True, exclude={"clear_allowed_providers", "clear_llm_budget"}
    )
    if body.clear_allowed_providers:
        changes["allowed_llm_providers"] = None
    if body.clear_llm_budget:
        changes["llm_monthly_budget_usd"] = None
    updated = repo.update(ws.model_copy(update=changes))
    state.audit.record(
        "admin.workspace_policy",
        user_id=who.user.id,
        resource=workspace_id,
        ip=client_ip(request),
        details=body.model_dump(mode="json", exclude_none=True),
    )
    return updated


# ------------------------------------------------------------------ membresías


@router.get("/memberships", operation_id="listMemberships")
def list_memberships(
    who: Who,
    state: State,
    workspace_id: str | None = None,
    project_id: str | None = None,
) -> list[Membership]:
    if not who.is_server_admin:
        if workspace_id is None:
            raise ForbiddenError("indicá el workspace")
        _require_workspace_admin(who, workspace_id)
    return state.accounts.list_memberships(workspace_id=workspace_id, project_id=project_id)


@router.post("/memberships", status_code=201, operation_id="grantMembership")
def grant(body: MembershipCreate, who: Who, request: Request, state: State) -> Membership:
    _require_workspace_admin(who, body.workspace_id)
    m = state.accounts.grant(body.user_id, body.workspace_id, body.role, body.project_id)
    state.audit.record(
        "admin.role_granted",
        user_id=who.user.id,
        project_id=body.project_id,
        resource=m.id,
        ip=client_ip(request),
        details={"target": body.user_id, "workspace_id": body.workspace_id, "role": body.role},
    )
    return m


@router.delete("/memberships/{membership_id}", status_code=204, operation_id="revokeMembership")
def revoke(membership_id: str, who: Who, request: Request, state: State) -> Response:
    m = state.accounts.memberships.get(membership_id)
    _require_workspace_admin(who, m.workspace_id)
    state.accounts.revoke(membership_id)
    state.audit.record(
        "admin.role_revoked",
        user_id=who.user.id,
        project_id=m.project_id,
        resource=membership_id,
        ip=client_ip(request),
        details={"target": m.user_id, "workspace_id": m.workspace_id, "role": m.role},
    )
    return Response(status_code=204)


# ------------------------------------------------------------------ auditoría


@router.get("/audit", operation_id="listAuditEvents")
def audit(
    _: Admin,
    state: State,
    user_id: str | None = None,
    project_id: str | None = None,
    action: str | None = None,
    since: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AuditEvent]:
    return state.audit.query(
        user_id=user_id,
        project_id=project_id,
        action=action,
        since=since,
        limit=limit,
        offset=offset,
    )


# ------------------------------------------------------------------ licencia (ADR-0035)


class LicenseUsage(BaseModel):
    status: str
    message: str = ""
    licensee: str | None = None
    edition: str | None = None
    expires_at: datetime | None = None
    seats: int | None = None
    seats_used: int
    gpus: int | None = None
    gpus_used: int
    servers: int | None = None
    servers_used: int = 1
    over_limit: list[str] = Field(default_factory=list, description="Solo se informa (v1)")


@router.get("/license", operation_id="getLicenseUsage")
def license_usage(_: Admin, state: State) -> LicenseUsage:
    """Uso real contra los topes de la licencia. Sin enforcement en v1 (RF-LIC-03)."""
    from perceptron.licensing.signed import license_provider

    check = license_provider(state.settings).check()
    t = check.terms
    seats_used = sum(1 for u, _ in state.accounts.list_users() if u.is_active)
    gpus_used = sum(len(w.get("gpus") or []) for w in state.workers.list())
    usage = LicenseUsage(
        status=check.status.value,
        message=check.message,
        licensee=t.licensee if t else None,
        edition=t.edition if t else None,
        expires_at=t.expires_at if t else None,
        seats=t.seats if t else None,
        seats_used=seats_used,
        gpus=t.gpus if t else None,
        gpus_used=gpus_used,
        servers=t.servers if t else None,
    )
    for name, used, cap in (
        ("seats", seats_used, usage.seats),
        ("gpus", gpus_used, usage.gpus),
        ("servers", usage.servers_used, usage.servers),
    ):
        if cap is not None and used > cap:
            usage.over_limit.append(name)
    return usage


# ------------------------------------------------------------------ sistema (RF-SRV-06)


class ProjectStorage(BaseModel):
    project_id: str
    name: str
    bytes: int


class SystemStatus(BaseModel):
    workspace_path: str
    used_bytes: int
    free_bytes: int
    projects: list[ProjectStorage]
    quotas: dict[str, int]
    queue_mode: str
    workers: list[dict[str, Any]]


def _dir_size(path: Path) -> int:
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir():
                        stack.append(Path(entry.path))
                    elif entry.is_file():
                        total += entry.stat().st_size
        except OSError:
            continue
    return total


@router.get("/system", operation_id="getSystemStatus")
def system_status(_: Admin, state: State) -> SystemStatus:
    """Almacenamiento, cuotas y estado de los workers para la consola (RF-SRV-06)."""
    paths = state.settings.paths
    names = {p.id: p.name for p in state.ctx.projects.list(limit=10_000)}
    projects = [
        ProjectStorage(project_id=pid, name=names[pid], bytes=_dir_size(paths.project(pid).root))
        for pid in names
    ]
    projects.sort(key=lambda p: p.bytes, reverse=True)
    usage = shutil.disk_usage(paths.root)
    return SystemStatus(
        workspace_path=str(paths.root),
        used_bytes=_dir_size(paths.root),
        free_bytes=usage.free,
        projects=projects[:50],
        quotas={
            "max_running_studies_per_user": state.server.max_running_studies_per_user,
            "max_running_studies_per_workspace": state.server.max_running_studies_per_workspace,
        },
        queue_mode=state.queue_mode,
        workers=state.workers.list(),
    )
