"""Control de costos del LLM (RF-LLM-06): se suma lo auditado y se corta antes de pasarse."""

from __future__ import annotations

from perceptron.domain.models import LLMCall
from perceptron.llm.errors import LLMBudgetExceededError
from perceptron.storage.repositories import SqlRepository

PAGE = 500
# Estimación previa: ~3,5 caracteres por token; la salida se estima al 25 % del máximo.
CHARS_PER_TOKEN = 3.5
OUTPUT_FRACTION = 0.25


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


class BudgetLedger:
    def __init__(self, calls: SqlRepository[LLMCall]) -> None:
        self.calls = calls

    def spent(self, project_id: str, scope: str | None = None) -> float:
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
    ) -> None:
        if project_limit is not None:
            spent = self.spent(project_id)
            if spent + estimate_usd > project_limit:
                raise LLMBudgetExceededError(
                    "Presupuesto de LLM del proyecto agotado",
                    details={"spent_usd": round(spent, 4), "limit_usd": project_limit},
                )
        if scope is not None and scope_limit is not None:
            spent = self.spent(project_id, scope)
            if spent + estimate_usd > scope_limit:
                raise LLMBudgetExceededError(
                    "Presupuesto de LLM agotado",
                    details={
                        "scope": scope,
                        "spent_usd": round(spent, 4),
                        "limit_usd": scope_limit,
                    },
                )
