"""Control de costos del LLM (RF-LLM-06): se suma lo auditado y se corta antes de pasarse.

Cada llamada reserva su costo máximo antes de salir (`BudgetLedger.reserve`): la reserva se
graba primero y se verifica después, así de dos llamadas concurrentes al menos una ve a la otra
y entre las dos no superan el límite (en el peor caso se rechazan las dos). La reserva se
libera recién cuando la llamada quedó auditada con su costo real.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta

from perceptron.core.errors import NotFoundError
from perceptron.domain.models import LLMCall, LLMReservation, Project, Workspace, utcnow
from perceptron.llm.errors import LLMBudgetExceededError
from perceptron.storage.repositories import SqlRepository

PAGE = 500
# Estimación previa conservadora: ~3 caracteres por token (los textos en español y el JSON
# rinden menos que el inglés), cada imagen como una grande, y la salida al máximo pedido.
CHARS_PER_TOKEN = 3.0
IMAGE_TOKENS = 1_600
# Una reserva abandonada (el proceso murió a mitad de la llamada) deja de contar.
RESERVATION_TTL = timedelta(hours=1)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_input_tokens(texts: list[str], images: int = 0) -> int:
    return sum(estimate_tokens(t) for t in texts) + images * IMAGE_TOKENS


class BudgetLedger:
    def __init__(
        self,
        calls: SqlRepository[LLMCall],
        projects: SqlRepository[Project] | None = None,
        reservations: SqlRepository[LLMReservation] | None = None,
    ) -> None:
        self.calls = calls
        self.projects = projects
        self.reservations = reservations

    # ------------------------------------------------------------------ reservas

    def reserved(
        self,
        *,
        project_id: str | None = None,
        workspace_id: str | None = None,
        scope: str | None = None,
        exclude: str | None = None,
    ) -> float:
        """Costo reservado por llamadas en curso (sin contar `exclude` ni las vencidas)."""
        if self.reservations is None:
            return 0.0
        filters: dict[str, object] = {}
        if project_id is not None:
            filters["for_project_id"] = project_id
        if workspace_id is not None:
            filters["workspace_id"] = workspace_id
        if scope is not None:
            filters["scope"] = scope
        cutoff = utcnow() - RESERVATION_TTL
        return sum(
            r.amount_usd
            for r in self.reservations.list(filters=filters, limit=10_000)
            if r.id != exclude and r.created_at >= cutoff
        )

    @contextmanager
    def reserve(
        self,
        project_id: str,
        workspace: Workspace | None,
        amount_usd: float,
        *,
        project_limit: float | None,
        scope: str | None = None,
        scope_limit: float | None = None,
    ) -> Iterator[None]:
        """Reserva `amount_usd` mientras dura el bloque; `LLMBudgetExceededError` si no entra."""

        def verify(exclude: str | None) -> None:
            self.check(
                project_id,
                amount_usd,
                project_limit=project_limit,
                scope=scope,
                scope_limit=scope_limit,
                exclude=exclude,
            )
            if workspace is not None:
                self.check_workspace(workspace, amount_usd, exclude=exclude)

        verify(None)  # corte barato, sin escribir
        limited = (
            project_limit is not None
            or (scope is not None and scope_limit is not None)
            or (workspace is not None and workspace.llm_monthly_budget_usd is not None)
        )
        if self.reservations is None or not limited:
            yield
            return
        mine = self.reservations.add(
            LLMReservation(
                for_project_id=project_id,
                workspace_id=workspace.id if workspace is not None else None,
                scope=scope,
                amount_usd=amount_usd,
            )
        )
        try:
            verify(mine.id)  # ya con la reserva visible para las demás llamadas
            yield
        finally:
            with suppress(NotFoundError):
                self.reservations.delete(mine.id)

    # ------------------------------------------------------------------ gasto

    def spent_this_month(
        self, workspace: Workspace, now: datetime | None = None, *, exclude: str | None = None
    ) -> float:
        """Gasto LLM del mes en curso de todos los proyectos del workspace (más lo reservado)."""
        if self.projects is None:
            return 0.0
        now = now or datetime.now(UTC)
        month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        projects = self.projects.list(filters={"workspace_id": workspace.id}, limit=10_000)
        total = self.reserved(workspace_id=workspace.id, exclude=exclude)
        for project in projects:
            offset = 0
            while True:
                page = list(
                    self.calls.list(filters={"project_id": project.id}, limit=PAGE, offset=offset)
                )
                total += sum(c.cost_usd for c in page if c.created_at >= month)
                # Ordenado del más nuevo al más viejo: al salir del mes no hay más que sumar.
                if len(page) < PAGE or (page and page[-1].created_at < month):
                    break
                offset += PAGE
        return total

    def check_workspace(
        self, workspace: Workspace, estimate_usd: float, *, exclude: str | None = None
    ) -> None:
        """Cuota mensual del workspace en el Team Server (RF-LLM-06)."""
        limit = workspace.llm_monthly_budget_usd
        if limit is None:
            return
        spent = self.spent_this_month(workspace, exclude=exclude)
        if spent + estimate_usd > limit:
            raise LLMBudgetExceededError(
                "Cuota mensual de LLM del workspace agotada",
                details={
                    "workspace_id": workspace.id,
                    "spent_usd": round(spent, 4),
                    "limit_usd": limit,
                },
            )

    def spent(self, project_id: str, scope: str | None = None) -> float:
        """Gasto auditado del proyecto (o de un ámbito: run, agente…)."""
        total, offset = 0.0, 0
        filters: dict[str, object] = {"project_id": project_id}
        if scope is not None:
            filters["scope"] = scope
        while True:
            page = list(self.calls.list(filters=filters, limit=PAGE, offset=offset))
            total += sum(c.cost_usd for c in page)
            if len(page) < PAGE:
                return total
            offset += PAGE

    def check(
        self,
        project_id: str,
        estimate_usd: float,
        *,
        project_limit: float | None,
        scope: str | None = None,
        scope_limit: float | None = None,
        exclude: str | None = None,
    ) -> None:
        if project_limit is not None:
            spent = self.spent(project_id) + self.reserved(project_id=project_id, exclude=exclude)
            if spent + estimate_usd > project_limit:
                raise LLMBudgetExceededError(
                    "Presupuesto de LLM del proyecto agotado",
                    details={"spent_usd": round(spent, 4), "limit_usd": project_limit},
                )
        if scope is not None and scope_limit is not None:
            spent = self.spent(project_id, scope) + self.reserved(
                project_id=project_id, scope=scope, exclude=exclude
            )
            if spent + estimate_usd > scope_limit:
                raise LLMBudgetExceededError(
                    "Presupuesto de LLM agotado",
                    details={
                        "scope": scope,
                        "spent_usd": round(spent, 4),
                        "limit_usd": scope_limit,
                    },
                )
