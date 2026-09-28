"""Política y cuota LLM por workspace (RF-PRV-04, RF-LLM-06)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from perceptron.api.context import EngineContext
from perceptron.domain.enums import LLMPurpose, PrivacyLevel
from perceptron.domain.models import LLMCall, Project, Workspace
from perceptron.llm.errors import LLMBudgetExceededError, LLMUnavailableError
from perceptron.llm.providers.fake import FakeLLMProvider


def _setup(ctx: EngineContext) -> tuple[Workspace, Workspace, Project, Project]:
    open_ws = ctx.repo(Workspace).add(Workspace(name="Abierto"))
    strict = ctx.repo(Workspace).add(
        Workspace(name="Estricto", allowed_llm_providers=["ollama"], llm_monthly_budget_usd=1.0)
    )
    p_open = ctx.projects.add(Project(name="a", workspace_id=open_ws.id))
    p_strict = ctx.projects.add(Project(name="b", workspace_id=strict.id))
    return open_ws, strict, p_open, p_strict


def test_each_project_gets_its_own_workspace_policy(
    ctx: EngineContext, fake_llm: FakeLLMProvider
) -> None:
    """Con varios workspaces (Team Server), manda el del proyecto y no el primero."""
    _, _, p_open, p_strict = _setup(ctx)
    gw = ctx.llm
    assert gw.resolve(LLMPurpose.COPILOT, p_open)
    with pytest.raises(LLMUnavailableError, match="no permite"):
        gw.resolve(LLMPurpose.COPILOT, p_strict)


def test_monthly_workspace_quota(ctx: EngineContext, fake_llm: FakeLLMProvider) -> None:
    _, strict, _, p_strict = _setup(ctx)
    ledger = ctx.llm.ledger
    now = datetime.now(UTC)

    def call(cost: float, when: datetime) -> None:
        ctx.repo(LLMCall).add(
            LLMCall(
                project_id=p_strict.id,
                purpose=LLMPurpose.COPILOT,
                provider="ollama",
                model="m",
                privacy_level=PrivacyLevel.L1,
                payload={},
                cost_usd=cost,
                created_at=when,
            )
        )

    call(0.6, now)
    call(5.0, now.replace(day=1) - timedelta(days=2))  # del mes pasado: no cuenta
    assert ledger.spent_this_month(strict) == pytest.approx(0.6)
    ledger.check_workspace(strict, 0.3)
    with pytest.raises(LLMBudgetExceededError, match="mensual"):
        ledger.check_workspace(strict, 0.5)
    ledger.check_workspace(strict.model_copy(update={"llm_monthly_budget_usd": None}), 100)
