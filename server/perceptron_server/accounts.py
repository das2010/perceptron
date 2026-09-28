"""Cuentas, sesiones, workspaces y membresías del Team Server (RF-SRV-01, RF-SRV-02)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import func, select, update

from perceptron.api.context import EngineContext
from perceptron.core.errors import (
    AuthError,
    ConflictError,
    NotFoundError,
    RateLimitedError,
    ValidationError,
)
from perceptron.core.ids import IdPrefix, new_id
from perceptron.domain.enums import Role
from perceptron.domain.models import Membership, Project, User, Workspace, utcnow
from perceptron_server.db import AccountRow, RefreshTokenRow
from perceptron_server.oidc import Identity
from perceptron_server.security import (
    hash_password,
    issue_access,
    needs_rehash,
    new_token,
    token_hash,
    verify_password,
)
from perceptron_server.settings import OIDCProvider, ServerSettings

ROLE_RANK: dict[Role, int] = {Role.VIEWER: 1, Role.EDITOR: 2, Role.ADMIN: 3}
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_LIST = 10_000
REFRESH_GRACE = timedelta(seconds=30)


def aware(value: datetime) -> datetime:
    """SQLite devuelve fechas sin zona: se interpretan como UTC (así se guardaron)."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _required(row: AccountRow | None) -> AccountRow:
    if row is None:
        raise NotFoundError("la cuenta ya no existe")
    return row


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if not _EMAIL.match(value) or len(value) > 320:
        raise ValidationError("email inválido", details={"email": email})
    return value


@dataclass
class Principal:
    """Quién hace el request: usuario, rol global y membresías (cargadas por request)."""

    user: User
    is_server_admin: bool
    memberships: list[Membership] = field(default_factory=list)
    session_id: str | None = None
    via: Literal["cookie", "bearer"] = "cookie"

    def workspace_role(self, workspace_id: str) -> Role | None:
        if self.is_server_admin:
            return Role.ADMIN
        for m in self.memberships:
            if m.workspace_id == workspace_id and m.project_id is None:
                return m.role
        return None

    def project_role(self, project: Project) -> Role | None:
        """La membresía del proyecto manda sobre la del workspace (§3.2: Editor en uno, Viewer
        en otro); sin ninguna de las dos, no hay acceso."""
        if self.is_server_admin:
            return Role.ADMIN
        for m in self.memberships:
            if m.project_id == project.id:
                return m.role
        return self.workspace_role(project.workspace_id) if project.workspace_id else None

    def workspaces(self, min_role: Role = Role.VIEWER) -> set[str]:
        return {
            m.workspace_id
            for m in self.memberships
            if m.project_id is None and ROLE_RANK[m.role] >= ROLE_RANK[min_role]
        }

    def has_role_anywhere(self, min_role: Role) -> bool:
        return self.is_server_admin or any(
            ROLE_RANK[m.role] >= ROLE_RANK[min_role] for m in self.memberships
        )


@dataclass(frozen=True, slots=True)
class TokenPair:
    access_token: str
    access_expires_at: datetime
    refresh_token: str
    refresh_expires_at: datetime
    session_id: str


class Accounts:
    def __init__(self, ctx: EngineContext, settings: ServerSettings) -> None:
        self.ctx = ctx
        self.settings = settings
        self.users = ctx.repo(User)
        self.workspaces = ctx.repo(Workspace)
        self.memberships = ctx.repo(Membership)

    # ------------------------------------------------------------------ usuarios

    def _account(self, user_id: str) -> AccountRow | None:
        with self.ctx.db.session() as s:
            return s.get(AccountRow, user_id)

    def account_by_email(self, email: str) -> AccountRow | None:
        with self.ctx.db.session() as s:
            return s.scalar(select(AccountRow).where(AccountRow.email == normalize_email(email)))

    def count_accounts(self) -> int:
        with self.ctx.db.session() as s:
            return int(s.scalar(select(func.count()).select_from(AccountRow)) or 0)

    def check_password_policy(self, password: str) -> None:
        if len(password) < self.settings.min_password_length:
            raise ValidationError(
                f"la contraseña debe tener al menos {self.settings.min_password_length} caracteres"
            )
        if len(password) > 256:
            raise ValidationError("la contraseña es demasiado larga")

    def create_user(
        self,
        email: str,
        display_name: str,
        password: str | None,
        *,
        is_server_admin: bool = False,
    ) -> User:
        email = normalize_email(email)
        if self.account_by_email(email) is not None:
            raise ConflictError(f"ya existe un usuario con el email {email}")
        if password is not None:
            self.check_password_policy(password)
        user = self.users.add(User(email=email, display_name=display_name.strip() or email))
        with self.ctx.db.session() as s:
            s.add(
                AccountRow(
                    user_id=user.id,
                    email=email,
                    password_hash=hash_password(password) if password else None,
                    is_server_admin=is_server_admin,
                    failed_logins=0,
                    password_changed_at=utcnow() if password else None,
                )
            )
        return user

    def list_users(self) -> list[tuple[User, AccountRow]]:
        with self.ctx.db.session() as s:
            accounts = {a.user_id: a for a in s.scalars(select(AccountRow))}
        users = self.users.list(ids=list(accounts), limit=_MAX_LIST)
        return sorted(((u, accounts[u.id]) for u in users), key=lambda p: p[0].email)

    def get_user(self, user_id: str) -> tuple[User, AccountRow]:
        account = self._account(user_id)
        if account is None:
            raise NotFoundError(f"usuario {user_id} no existe")
        return self.users.get(user_id), account

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        is_active: bool | None = None,
        is_server_admin: bool | None = None,
        password: str | None = None,
    ) -> User:
        user, account = self.get_user(user_id)
        losing_admin = account.is_server_admin and (is_server_admin is False or is_active is False)
        if losing_admin and self._active_admins() <= 1:
            raise ConflictError("no se puede quitar el último administrador del servidor")
        if display_name is not None or is_active is not None:
            user = self.users.update(
                user.model_copy(
                    update={
                        "display_name": display_name or user.display_name,
                        "is_active": user.is_active if is_active is None else is_active,
                    }
                )
            )
        if password is not None:
            self.set_password(user_id, password)
        with self.ctx.db.session() as s:
            row = _required(s.get(AccountRow, user_id))
            if is_server_admin is not None:
                row.is_server_admin = is_server_admin
            if is_active is True:
                row.failed_logins, row.locked_until = 0, None
        if is_active is False:
            self.revoke_user_sessions(user_id)
        return user

    def _active_admins(self) -> int:
        with self.ctx.db.session() as s:
            ids = list(s.scalars(select(AccountRow.user_id).where(AccountRow.is_server_admin)))
        return sum(1 for u in self.users.list(ids=ids, limit=_MAX_LIST) if u.is_active)

    def set_password(self, user_id: str, password: str) -> None:
        self.check_password_policy(password)
        with self.ctx.db.session() as s:
            row = s.get(AccountRow, user_id)
            if row is None:
                raise NotFoundError(f"usuario {user_id} no existe")
            row.password_hash = hash_password(password)
            row.password_changed_at = utcnow()
            row.failed_logins, row.locked_until = 0, None
        self.revoke_user_sessions(user_id)

    def change_password(self, user_id: str, current: str, new: str) -> None:
        account = self._account(user_id)
        if account is None or not verify_password(account.password_hash, current):
            raise AuthError("la contraseña actual no es correcta")
        self.set_password(user_id, new)

    def authenticate(self, email: str, password: str) -> User:
        """Login local. Mensaje genérico ante cualquier fallo (no revela si el email existe)."""
        invalid = AuthError("email o contraseña incorrectos")
        try:
            account = self.account_by_email(email)
        except ValidationError:
            raise invalid from None
        now = utcnow()
        if account is not None and account.locked_until and aware(account.locked_until) > now:
            raise RateLimitedError(
                "cuenta bloqueada temporalmente por intentos fallidos; probá más tarde"
            )
        ok = verify_password(account.password_hash if account else None, password)
        if account is None:
            raise invalid
        user = self.users.get(account.user_id)
        failed = not ok or not user.is_active
        # El contador se guarda antes de responder (el raise dentro de la sesión haría rollback).
        with self.ctx.db.session() as s:
            row = _required(s.get(AccountRow, account.user_id))
            if failed:
                row.failed_logins += 1
                if row.failed_logins >= self.settings.login_max_failures:
                    row.locked_until = now + timedelta(seconds=self.settings.login_lockout_s)
                    row.failed_logins = 0
            else:
                row.failed_logins, row.locked_until, row.last_login_at = 0, None, now
                if row.password_hash and needs_rehash(row.password_hash):
                    row.password_hash = hash_password(password)
        if failed:
            raise invalid
        return user

    def principal(
        self,
        user_id: str,
        session_id: str | None = None,
        via: Literal["cookie", "bearer"] = "cookie",
    ) -> Principal | None:
        account = self._account(user_id)
        user = self.users.find(user_id)
        if account is None or user is None or not user.is_active:
            return None
        return Principal(
            user=user,
            is_server_admin=account.is_server_admin,
            memberships=list(self.memberships.list(filters={"user_id": user_id}, limit=_MAX_LIST)),
            session_id=session_id,
            via=via,
        )

    # ------------------------------------------------------------------ sesiones

    def _issue(
        self, user_id: str, family_id: str, user_agent: str | None, ip: str | None
    ) -> TokenPair:
        refresh = new_token()
        now = utcnow()
        refresh_exp = now + timedelta(seconds=self.settings.refresh_ttl_s)
        with self.ctx.db.session() as s:
            s.add(
                RefreshTokenRow(
                    id=new_id(IdPrefix.REFRESH_TOKEN),
                    user_id=user_id,
                    family_id=family_id,
                    token_hash=token_hash(refresh),
                    created_at=now,
                    expires_at=refresh_exp,
                    user_agent=(user_agent or "")[:255] or None,
                    ip=ip,
                )
            )
        access, access_exp = issue_access(
            user_id,
            family_id,
            self.settings.secret_key.get_secret_value(),
            self.settings.access_ttl_s,
        )
        return TokenPair(access, access_exp, refresh, refresh_exp, family_id)

    def start_session(self, user_id: str, user_agent: str | None, ip: str | None) -> TokenPair:
        return self._issue(user_id, new_id(IdPrefix.SESSION), user_agent, ip)

    def resolve_refresh(
        self,
        refresh_token: str,
        *,
        rotate: bool,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> tuple[str, str, TokenPair | None] | None:
        """(usuario, sesión, tokens nuevos) de un token de refresco válido; `None` si no sirve.

        Rotación: el token usado se revoca. Un token recién rotado sigue valiendo unos segundos
        (requests en paralelo del navegador); reusarlo después revoca toda la sesión.
        """
        now = utcnow()
        with self.ctx.db.session() as s:
            row = s.scalar(
                select(RefreshTokenRow).where(
                    RefreshTokenRow.token_hash == token_hash(refresh_token)
                )
            )
            if row is None or aware(row.expires_at) <= now:
                return None
            user_id, family_id, revoked_at = row.user_id, row.family_id, row.revoked_at
            if revoked_at is None and rotate:
                row.revoked_at = now
        user = self.users.find(user_id)
        if user is None or not user.is_active:
            return None
        if revoked_at is not None:
            if now - aware(revoked_at) <= REFRESH_GRACE and self.session_active(family_id):
                return user_id, family_id, None
            self.revoke_session(family_id)  # reuso tardío: posible robo del token
            return None
        pair = self._issue(user_id, family_id, user_agent, ip) if rotate else None
        return user_id, family_id, pair

    def session_active(self, session_id: str) -> bool:
        now = utcnow()
        with self.ctx.db.session() as s:
            rows = s.scalars(
                select(RefreshTokenRow.expires_at).where(
                    RefreshTokenRow.family_id == session_id,
                    RefreshTokenRow.revoked_at.is_(None),
                )
            ).all()
        return any(aware(exp) > now for exp in rows)

    def revoke_session(self, session_id: str) -> None:
        with self.ctx.db.session() as s:
            s.execute(
                update(RefreshTokenRow)
                .where(
                    RefreshTokenRow.family_id == session_id, RefreshTokenRow.revoked_at.is_(None)
                )
                .values(revoked_at=utcnow())
            )

    def revoke_user_sessions(self, user_id: str) -> None:
        with self.ctx.db.session() as s:
            s.execute(
                update(RefreshTokenRow)
                .where(RefreshTokenRow.user_id == user_id, RefreshTokenRow.revoked_at.is_(None))
                .values(revoked_at=utcnow())
            )

    # ------------------------------------------------------------------ workspaces y roles

    def ensure_default_workspace(self) -> Workspace:
        existing = self.workspaces.list(filters={"name": self.settings.default_workspace}, limit=1)
        if existing:
            return existing[0]
        return self.workspaces.add(Workspace(name=self.settings.default_workspace))

    def workspace_named(self, name: str) -> Workspace:
        found = self.workspaces.list(filters={"name": name}, limit=1)
        return found[0] if found else self.workspaces.add(Workspace(name=name))

    def create_workspace(self, name: str) -> Workspace:
        if self.workspaces.list(filters={"name": name}, limit=1):
            raise ConflictError(f"ya existe el workspace {name}")
        return self.workspaces.add(Workspace(name=name))

    def list_workspaces(self) -> Sequence[Workspace]:
        return self.workspaces.list(limit=_MAX_LIST)

    def grant(
        self, user_id: str, workspace_id: str, role: Role, project_id: str | None = None
    ) -> Membership:
        """Una membresía por (usuario, workspace, proyecto): si existe, cambia el rol."""
        self.get_user(user_id)
        self.workspaces.get(workspace_id)
        if project_id is not None:
            project = self.ctx.projects.get(project_id)
            if project.workspace_id != workspace_id:
                raise ValidationError("el proyecto no pertenece a ese workspace")
        for m in self.memberships.list(filters={"user_id": user_id}, limit=_MAX_LIST):
            if m.workspace_id == workspace_id and m.project_id == project_id:
                if m.role is role:
                    return m
                return self.memberships.update(m.model_copy(update={"role": role}))
        return self.memberships.add(
            Membership(user_id=user_id, workspace_id=workspace_id, project_id=project_id, role=role)
        )

    def revoke(self, membership_id: str) -> Membership:
        m = self.memberships.get(membership_id)
        self.memberships.delete(membership_id)
        return m

    def list_memberships(
        self, *, workspace_id: str | None = None, project_id: str | None = None
    ) -> list[Membership]:
        filters = {"workspace_id": workspace_id} if workspace_id else {}
        items = self.memberships.list(filters=filters, limit=_MAX_LIST)
        return [m for m in items if project_id is None or m.project_id == project_id]

    # ------------------------------------------------------------------ arranque

    def bootstrap(self) -> User | None:
        """Primer arranque: workspace por defecto y, si se configuró, el primer Admin."""
        workspace = self.ensure_default_workspace()
        email = self.settings.bootstrap_admin_email
        password = self.settings.bootstrap_admin_password
        if self.count_accounts() or not email or not password:
            return None
        user = self.create_user(
            email, "Administrador", password.get_secret_value(), is_server_admin=True
        )
        self.grant(user.id, workspace.id, Role.ADMIN)
        return user


def sso_login(
    accounts: Accounts, provider_name: str, provider: OIDCProvider, identity: Identity
) -> User:
    """Usuario del SSO: se vincula por email (verificado por el IdP) o se crea.

    Los grupos del token dan roles según `role_mapping` en cada login (no se quitan roles
    asignados a mano). Si ningún grupo aplica, un usuario nuevo recibe `default_role` en el
    workspace por defecto.
    """
    account = accounts.account_by_email(identity.email)
    if account is None:
        if not provider.auto_create:
            raise AuthError("tu usuario no está habilitado en Perceptron; pedíselo al Admin")
        user = accounts.create_user(identity.email, identity.name, None)
        user = accounts.users.update(
            user.model_copy(update={"auth_provider": f"oidc:{provider_name}"})
        )
        is_new = True
    else:
        user = accounts.users.get(account.user_id)
        if not user.is_active:
            raise AuthError("tu usuario está desactivado")
        is_new = False
    granted = False
    for rule in provider.role_mapping:
        if rule.group in identity.groups:
            ws = accounts.workspace_named(rule.workspace)
            accounts.grant(user.id, ws.id, rule.role, rule.project_id)
            granted = True
    if is_new and not granted and provider.default_role is not None:
        ws = accounts.ensure_default_workspace()
        accounts.grant(user.id, ws.id, provider.default_role)
    with accounts.ctx.db.session() as s:
        row = _required(s.get(AccountRow, user.id))
        row.last_login_at = utcnow()
    return user
