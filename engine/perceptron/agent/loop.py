"""Ciclo iterativo autónomo (SPEC §7.11, RF-AGT-01..05).

    perfil + objetivo + presupuesto → [LLM: acción] → validación → (¿aprobación?) →
    ejecutar → observar → … → finish: evaluación en test sellado, registro e informe.

El LLM propone; el sistema valida y ejecuta. Los límites (tiempo, decisiones, estudios,
trials, costo de LLM, disco) los aplica el sistema antes de cada paso. Las herramientas
solo ven métricas de validación: el test sellado lo abre `finish`, que ejecuta el sistema.
Si el LLM falla, el ciclo cae a la política por reglas o se detiene conservando el mejor
modelo (RF-AGT-05).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from perceptron.agent.models import (
    TOOLS,
    AgentAction,
    AgentLimits,
    AgentStep,
    ApprovalPolicy,
    CompareRuns,
    Finish,
    GetProfile,
    GetProjectGoal,
    GetRunCurves,
    GetRunMetrics,
    GetStudyStatus,
    LaunchStudy,
    ListCatalogBlocks,
    ProposeArchspec,
    ProposeHpoStrategy,
    RequestHumanApproval,
    SuggestPipelineChange,
    ValidateArchspec,
)
from perceptron.archspec.schema import ArchSpec, Provenance
from perceptron.archspec.validate import validate_archspec
from perceptron.core.errors import ConflictError
from perceptron.domain.enums import AgentState, LLMPurpose, Origin, RunStatus
from perceptron.domain.models import AgentRun, ArchSpecRecord, Run, Study, utcnow
from perceptron.hpo.recommend import recommend_strategy
from perceptron.hpo.strategy import Budget, HPOStrategy
from perceptron.hpo.study import StudyControl
from perceptron.licensing.features import features
from perceptron.llm.errors import (
    LLMBudgetExceededError,
    LLMOutputInvalidError,
    LLMProviderError,
    LLMUnavailableError,
)
from perceptron.llm.privacy import LLMContext, RunSummary
from perceptron.training.config import RESULT_FILE, RunResult
from perceptron.training.diagnostics import detect
from perceptron.training.module import monitor_mode

if TYPE_CHECKING:
    from perceptron.services.workflow import Workflow

logger = logging.getLogger(__name__)

AGENT_TOPIC = "agent.event"
FALLBACK = (LLMUnavailableError, LLMBudgetExceededError, LLMProviderError, LLMOutputInvalidError)
OBSERVATIONS_IN_CONTEXT = 6
RUNS_IN_CONTEXT = 8
HISTORY_POINTS = 20
MAX_REPEATS = 3


@dataclass
class AgentControl:
    """Canal en memoria para detener un agente en curso (y su estudio)."""

    stop: bool = False
    study: StudyControl | None = None

    def request_stop(self) -> None:
        self.stop = True
        if self.study is not None:
            self.study.cancel()


_CONTROLS: dict[str, AgentControl] = {}
_LOCK = threading.Lock()


def control_for(agent_id: str) -> AgentControl:
    with _LOCK:
        return _CONTROLS.setdefault(agent_id, AgentControl())


def _signature(action: AgentAction) -> str:
    return json.dumps(action.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def family(spec: ArchSpec) -> str:
    """Familia de arquitectura para la política de aprobación `family_change`."""
    if spec.provenance.template:
        return spec.provenance.template
    return "+".join(sorted({n.block for n in spec.nodes if not n.block.startswith("head.")}))


def _downsample(history: list[dict[str, float]], n: int = HISTORY_POINTS) -> list[dict[str, float]]:
    if len(history) <= n:
        return history
    step = len(history) / n
    return [*(history[int(i * step)] for i in range(n - 1)), history[-1]]


def _val_only(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if not k.startswith("test")}


class AgentRunner:
    def __init__(self, wf: Workflow) -> None:
        self.wf = wf
        self.ctx = wf.ctx
        self.repo = self.ctx.repo(AgentRun)

    # ------------------------------------------------------------------ ciclo de vida

    def start(
        self,
        project_id: str,
        dataset_version_id: str,
        pipeline_id: str,
        *,
        limits: AgentLimits | None = None,
        approval: ApprovalPolicy | None = None,
    ) -> AgentRun:
        """Crea el AgentRun; la propuesta por reglas queda como archspec base (a0)."""
        features.require("llm.agent")
        base, why = self.wf.propose_architecture(dataset_version_id, pipeline_id)
        ar = AgentRun(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            pipeline_id=pipeline_id,
            limits=(limits or AgentLimits()).model_dump(mode="json"),
            approval=(approval or ApprovalPolicy()).model_dump(mode="json"),
            archspecs=[base.id],
            started_at=utcnow(),
        )
        self._log(ar, "system", f"Agente iniciado. Arquitectura base por reglas: {base.name}.")
        self._observe(ar, "system", {"base_archspec_id": base.id, "name": base.name, "why": why})
        return self.repo.add(ar)

    def run(self, agent_id: str) -> AgentRun:
        """Avanza hasta terminar, pedir aprobación, detenerse o fallar."""
        ar = self.repo.get(agent_id)
        control = control_for(agent_id)
        while ar.state is AgentState.RUNNING:
            if control.stop:
                ar = self._end(ar, AgentState.STOPPED, "detenido por el usuario")
                break
            reason = self._limit_reason(ar)
            if reason:
                ar = self._conclude(ar, reason)
                break
            try:
                step, call_id = self._decide(ar)
            except FALLBACK as e:
                ar = self._fallback(ar, e)
                continue
            ar.steps += 1
            ar.cost_usd = self.ctx.llm.ledger.spent(ar.project_id, self._scope(ar))
            sig = _signature(step.action)
            repeats = self._repeats(ar, sig)
            self._log(
                ar,
                "decision",
                step.log_entry,
                tool=step.action.tool,
                llm_call_id=call_id,
                data={"signature": sig},
            )
            if repeats >= MAX_REPEATS:
                ar = self._conclude(ar, "acciones repetidas")
                break
            if repeats:
                # Modelos chicos tienden a repetir la misma acción: no se re-ejecuta.
                self._observe(
                    ar,
                    step.action.tool,
                    {
                        "error": "acción repetida con los mismos argumentos: su resultado ya "
                        "está en evidence. Elegí otra acción o usá finish."
                    },
                )
                ar = self.repo.update(ar)
                continue
            try:
                ar = self._execute(ar, step.action)
            except Exception as e:  # una herramienta falló: se informa al LLM y sigue
                logger.exception("herramienta del agente falló")
                self._observe(ar, step.action.tool, {"error": f"{type(e).__name__}: {e}"})
            ar = self.repo.update(ar)
        return ar

    @staticmethod
    def _repeats(ar: AgentRun, sig: str) -> int:
        """Cuántas decisiones seguidas, al final de la bitácora, tienen esta misma acción."""
        count = 0
        for entry in reversed(ar.log):
            if entry.get("kind") != "decision":
                continue
            if (entry.get("data") or {}).get("signature") != sig:
                break
            count += 1
        return count

    def approve(self, agent_id: str, *, approved: bool, comment: str | None = None) -> AgentRun:
        """Respuesta humana a un punto de aprobación (RF-AGT-03). Luego se llama a `run`."""
        ar = self.repo.get(agent_id)
        if ar.state is not AgentState.AWAITING_APPROVAL:
            raise ConflictError("el agente no está esperando aprobación")
        pending, ar.pending_action = ar.pending_action, None
        ar.state = AgentState.RUNNING
        if not approved:
            self._log(ar, "approval", f"Rechazado por el usuario. {comment or ''}".strip())
            self._observe(
                ar, "approval", {"rechazado": True, "comentario": comment, "accion": pending}
            )
            return self.repo.update(ar)
        self._log(ar, "approval", f"Aprobado por el usuario. {comment or ''}".strip())
        if self._approval(ar).mode == "budget_pct":
            ar.approval = {**ar.approval, "budget_approved": True}
        if pending is not None and pending.get("tool") == "launch_study":
            ar = self._launch(ar, LaunchStudy.model_validate(pending), approved=True)
        return self.repo.update(ar)

    def stop(self, agent_id: str) -> AgentRun:
        control_for(agent_id).request_stop()
        ar = self.repo.get(agent_id)
        if ar.state is AgentState.AWAITING_APPROVAL:
            ar = self._end(ar, AgentState.STOPPED, "detenido por el usuario")
        return ar

    # ------------------------------------------------------------------ decisión

    def _scope(self, ar: AgentRun) -> str:
        return f"agent:{ar.id}"

    def _limits(self, ar: AgentRun) -> AgentLimits:
        return AgentLimits.model_validate(ar.limits)

    def _approval(self, ar: AgentRun) -> ApprovalPolicy:
        return ApprovalPolicy.model_validate(
            {k: v for k, v in ar.approval.items() if k != "budget_approved"}
        )

    def _decide(self, ar: AgentRun) -> tuple[AgentStep, str]:
        project = self.wf.project(ar.project_id)
        res = self.ctx.llm.structured(
            LLMPurpose.AGENT,
            AgentStep,
            self._context(ar),
            project=project,
            validator=self._validator(ar),
            scope=self._scope(ar),
            scope_budget_usd=self._limits(ar).max_llm_cost_usd,
        )
        return res.value, res.call_id

    def _runs(self, ar: AgentRun) -> list[Run]:
        runs: list[Run] = []
        for sid in ar.studies:
            runs += self.ctx.repo(Run).list(filters={"study_id": sid}, limit=500)
        return runs

    def _completed(self, ar: AgentRun) -> list[Run]:
        metric = self._limits(ar).selection_metric
        return [
            r for r in self._runs(ar) if r.status is RunStatus.SUCCEEDED and metric in r.metrics
        ]

    def _best(self, ar: AgentRun) -> str | None:
        runs = self._completed(ar)
        if not runs:
            return None
        metric = self._limits(ar).selection_metric
        if monitor_mode(metric) == "min":
            return min(runs, key=lambda r: r.metrics[metric]).id
        return max(runs, key=lambda r: r.metrics[metric]).id

    def _validator(self, ar: AgentRun) -> Callable[[AgentStep], str | None]:
        run_ids = {r.id for r in self._runs(ar)}
        completed = {r.id for r in self._completed(ar)}
        archspecs, strategies, studies = set(ar.archspecs), set(ar.strategies), set(ar.studies)

        def check(step: AgentStep) -> str | None:
            a = step.action
            if isinstance(a, ProposeHpoStrategy | LaunchStudy) and a.archspec_id not in archspecs:
                return f"archspec_id '{a.archspec_id}' desconocido; válidos: {sorted(archspecs)}"
            if isinstance(a, LaunchStudy) and a.strategy_id and a.strategy_id not in strategies:
                return f"strategy_id '{a.strategy_id}' desconocido; válidos: {sorted(strategies)}"
            if isinstance(a, GetStudyStatus) and a.study_id not in studies:
                return f"study_id '{a.study_id}' desconocido; válidos: {sorted(studies)}"
            if isinstance(a, GetRunMetrics | GetRunCurves) and a.run_id not in run_ids:
                return f"run_id '{a.run_id}' desconocido; válidos: {sorted(run_ids)}"
            if isinstance(a, CompareRuns) and not set(a.run_ids) <= run_ids:
                return f"run_ids desconocidos; válidos: {sorted(run_ids)}"
            if isinstance(a, Finish):
                if not completed:
                    return "todavía no hay runs completos: lanzá un estudio (launch_study) antes"
                if a.run_id and a.run_id not in completed:
                    return (
                        f"run_id '{a.run_id}' no es un run completo; válidos: {sorted(completed)}"
                    )
            return None

        return check

    def _context(self, ar: AgentRun) -> LLMContext:
        from perceptron.hpo.strategy import default_search_space
        from perceptron.services.llm_roles import catalog_for

        wf = self.wf
        project = wf.project(ar.project_id)
        card = wf.profile_card(ar.dataset_version_id)
        limits = self._limits(ar)
        base = self._spec(ar.archspecs[0])
        archspecs: dict[str, Any] = {}
        for aid in ar.archspecs:
            record = self.ctx.repo(ArchSpecRecord).get(aid)
            spec = ArchSpec.model_validate(record.spec)
            archspecs[aid] = {
                "name": spec.name,
                "origin": record.origin.value,
                "family": family(spec),
                "tunable": [p.name for p in default_search_space(spec)],
            }
        runs = [self._summary(r) for r in self._runs(ar)[-RUNS_IN_CONTEXT:]]
        observations = [e["data"] for e in ar.log if e.get("kind") == "observation"]
        elapsed = self._elapsed(ar)
        return LLMContext(
            goal=project.goal or None,
            card=card,
            catalog=catalog_for(card, base.task.type, compact=True),
            constraints={
                "base_archspec_id": ar.archspecs[0],
                "base_archspec": base.model_dump(mode="json", exclude={"provenance"}),
                "limits": limits.model_dump(mode="json"),
                "remaining": {
                    "time_s": round(max(limits.max_time_s - elapsed, 0), 1),
                    "iterations": limits.max_iterations - ar.iterations,
                    "steps": limits.max_steps - ar.steps,
                    "trials": limits.max_trials - ar.trials,
                    "llm_cost_usd": round(max(limits.max_llm_cost_usd - ar.cost_usd, 0), 4),
                },
                "approval": self._approval(ar).model_dump(mode="json"),
                "selection_metric": limits.selection_metric,
                "direction": "min" if monitor_mode(limits.selection_metric) == "min" else "max",
            },
            runs=runs,
            evidence=observations[-OBSERVATIONS_IN_CONTEXT:],
            system={
                "tools": TOOLS,
                "archspecs": archspecs,
                "strategies": {
                    sid: {"archspec_id": s["archspec_id"], "strategy": s["strategy"]["strategy"]}
                    for sid, s in ar.strategies.items()
                },
                "studies": ar.studies,
                "best_run_id": ar.best_run_id,
                "iteration": ar.iterations,
                "step": ar.steps,
            },
        )

    def _history(self, run: Run) -> list[dict[str, float]]:
        path = self.ctx.settings.paths.project(run.project_id).run(run.id) / RESULT_FILE
        if not path.is_file():
            return []
        return RunResult.model_validate_json(path.read_text(encoding="utf-8")).history

    def _summary(self, run: Run) -> RunSummary:
        return RunSummary(
            run_id=run.id,
            status=run.status.value,
            architecture=run.archspec_id,
            hyperparams=run.hyperparams,
            metrics=_val_only(run.metrics),
            history=_downsample(self._history(run)),
        )

    # ------------------------------------------------------------------ herramientas

    def _spec(self, archspec_id: str) -> ArchSpec:
        return ArchSpec.model_validate(self.ctx.repo(ArchSpecRecord).get(archspec_id).spec)

    def _budget(self, ar: AgentRun) -> Budget:
        limits = self._limits(ar)
        return Budget(
            max_trials=max(limits.max_trials - ar.trials, 1),
            max_epochs_per_trial=limits.max_epochs_per_trial,
            max_time_s=max(limits.max_time_s - self._elapsed(ar), 1.0),
        )

    def _execute(self, ar: AgentRun, action: AgentAction) -> AgentRun:
        wf = self.wf
        if isinstance(action, GetProfile):
            card = wf.profile_card(ar.dataset_version_id)
            self._observe(
                ar,
                action.tool,
                {
                    "modality": card.modality.value,
                    "num_samples": card.num_samples,
                    "target": card.target.name if card.target else None,
                    "alerts": [a.code.value for a in card.alerts],
                },
            )
        elif isinstance(action, GetProjectGoal):
            self._observe(ar, action.tool, {"goal": wf.project(ar.project_id).goal})
        elif isinstance(action, ListCatalogBlocks):
            card = wf.profile_card(ar.dataset_version_id)
            from perceptron.catalog.registry import blocks_for

            self._observe(ar, action.tool, {"blocks": [b.key for b in blocks_for(card.modality)]})
        elif isinstance(action, ProposeArchspec | ValidateArchspec):
            self._archspec(ar, action)
        elif isinstance(action, ProposeHpoStrategy):
            self._hpo(ar, action)
        elif isinstance(action, LaunchStudy):
            ar = self._launch(ar, action)
        elif isinstance(action, GetStudyStatus):
            study = self.ctx.repo(Study).get(action.study_id)
            runs = self.ctx.repo(Run).list(filters={"study_id": study.id}, limit=500)
            self._observe(
                ar,
                action.tool,
                {
                    "study_id": study.id,
                    "result": study.budget.get("result"),
                    "trials": [
                        {"run_id": r.id, "status": r.status.value, "metrics": _val_only(r.metrics)}
                        for r in runs
                    ],
                },
            )
        elif isinstance(action, GetRunMetrics):
            run = self.ctx.repo(Run).get(action.run_id)
            self._observe(
                ar,
                action.tool,
                {
                    "run_id": run.id,
                    "status": run.status.value,
                    "metrics": _val_only(run.metrics),
                    "hyperparams": run.hyperparams,
                },
            )
        elif isinstance(action, GetRunCurves):
            run = self.ctx.repo(Run).get(action.run_id)
            history = self._history(run)
            spec = ArchSpec.model_validate(self.ctx.repo(ArchSpecRecord).get(run.archspec_id).spec)
            found = detect(history, hyperparameters={**spec.hyperparameters(), **run.hyperparams})
            self._observe(
                ar,
                action.tool,
                {
                    "run_id": run.id,
                    "history": _downsample(history, 30),
                    "rules_diagnosis": [p.model_dump(mode="json") for p, _ in found],
                },
            )
        elif isinstance(action, CompareRuns):
            table: dict[str, Any] = {}
            for rid in action.run_ids:
                run = self.ctx.repo(Run).get(rid)
                table[rid] = {
                    "archspec_id": run.archspec_id,
                    "metrics": _val_only(run.metrics),
                    "hyperparams": run.hyperparams,
                }
            self._observe(ar, action.tool, {"runs": table})
        elif isinstance(action, SuggestPipelineChange):
            self._log(ar, "suggestion", f"{action.description} — {action.rationale}")
            self._observe(
                ar,
                action.tool,
                {"registrado": True, "aplicado": False, "nota": "requiere revisión humana"},
            )
        elif isinstance(action, RequestHumanApproval):
            ar.state = AgentState.AWAITING_APPROVAL
            ar.pending_action = None
            self._log(ar, "approval", f"Pide aprobación: {action.reason}")
        elif isinstance(action, Finish):
            run_id = action.run_id or self._best(ar)
            if run_id is None:
                self._observe(ar, action.tool, {"error": "no hay runs completos"})
            else:
                ar = self._finalize(ar, run_id, action.summary, reason="finish")
        return ar

    def _archspec(self, ar: AgentRun, action: ProposeArchspec | ValidateArchspec) -> None:
        base = self._spec(ar.archspecs[0])
        rationale = action.rationale if isinstance(action, ProposeArchspec) else None
        spec = action.archspec.model_copy(
            update={
                "input": base.input,
                "task": base.task,
                "provenance": Provenance(origin=Origin.AGENT, rationale=rationale),
            }
        )
        report = validate_archspec(spec)
        if not report.valid:
            self._observe(ar, action.tool, {"valid": False, "feedback": report.feedback()})
            return
        data: dict[str, Any] = {
            "valid": True,
            "num_params": report.num_params,
            "memory_mb": report.estimated_memory_mb,
        }
        if isinstance(action, ProposeArchspec):
            from perceptron.hpo.strategy import default_search_space

            record = self.wf.save_archspec(ar.project_id, spec, origin=Origin.AGENT)
            ar.archspecs.append(record.id)
            data |= {
                "archspec_id": record.id,
                "name": spec.name,
                "tunable": [p.name for p in default_search_space(spec)],
            }
        self._observe(ar, action.tool, data)

    def _hpo(self, ar: AgentRun, action: ProposeHpoStrategy) -> None:
        from perceptron.services.llm_roles import HPOSpace

        space = HPOSpace.of(self._spec(action.archspec_id))
        budget = self._budget(ar)
        proposal, notes = space.repair(action.strategy, budget)
        errors = space.errors(proposal, budget)
        if errors:
            self._observe(ar, action.tool, {"valid": False, "errors": errors})
            return
        strategy = space.build(proposal, budget, origin=Origin.AGENT)
        sid = f"s{len(ar.strategies) + 1}"
        ar.strategies = {
            **ar.strategies,
            sid: {"archspec_id": action.archspec_id, "strategy": strategy.model_dump(mode="json")},
        }
        self._observe(
            ar,
            action.tool,
            {
                "strategy_id": sid,
                "strategy": strategy.strategy,
                "pruner": strategy.pruner,
                "params": [p.name for p in strategy.search_space],
                "max_trials": strategy.budget.max_trials,
                "system_notes": notes,
            },
        )

    def _approval_needed(self, ar: AgentRun, spec: ArchSpec) -> str | None:
        policy = self._approval(ar)
        if policy.mode == "each_iteration":
            return "antes de cada iteración"
        if policy.mode == "family_change" and ar.last_family and family(spec) != ar.last_family:
            return f"cambio de familia: {ar.last_family} → {family(spec)}"
        if policy.mode == "budget_pct" and not ar.approval.get("budget_approved"):
            limits = self._limits(ar)
            used = max(
                ar.trials / limits.max_trials,
                ar.cost_usd / limits.max_llm_cost_usd if limits.max_llm_cost_usd else 0.0,
                self._elapsed(ar) / limits.max_time_s,
            )
            if used * 100 >= policy.budget_pct:
                return f"se usó el {used:.0%} del presupuesto"
        return None

    def _launch(self, ar: AgentRun, action: LaunchStudy, *, approved: bool = False) -> AgentRun:
        spec = self._spec(action.archspec_id)
        if not approved:
            reason = self._approval_needed(ar, spec)
            if reason:
                ar.pending_action = action.model_dump(mode="json")
                ar.state = AgentState.AWAITING_APPROVAL
                self._log(
                    ar, "approval", f"Aprobación requerida ({reason}) para entrenar {spec.name}"
                )
                return ar
        limits = self._limits(ar)
        budget = self._budget(ar)
        if action.strategy_id:
            strategy = HPOStrategy.model_validate(ar.strategies[action.strategy_id]["strategy"])
        else:
            strategy = recommend_strategy(spec, budget)
        trials = min(action.max_trials, budget.max_trials, strategy.budget.max_trials)
        epochs = action.max_epochs_per_trial or strategy.budget.max_epochs_per_trial
        if limits.max_epochs_per_trial:
            epochs = min(epochs or limits.max_epochs_per_trial, limits.max_epochs_per_trial)
        strategy = strategy.model_copy(
            update={
                "budget": strategy.budget.model_copy(
                    update={
                        "max_trials": trials,
                        "max_epochs_per_trial": epochs,
                        "max_time_s": budget.max_time_s,
                    }
                ),
                "origin": Origin.AGENT,
            }
        )
        return self._study(ar, action.archspec_id, spec, strategy)

    def _study(
        self, ar: AgentRun, archspec_id: str, spec: ArchSpec, strategy: HPOStrategy
    ) -> AgentRun:
        control = control_for(ar.id)
        control.study = StudyControl()
        try:
            study, result = self.wf.run_study(
                ar.project_id,
                ar.dataset_version_id,
                ar.pipeline_id,
                archspec_id,
                strategy,
                control=control.study,
            )
        finally:
            control.study = None
        ar.iterations += 1
        ar.trials += len(result.trials)
        ar.studies = [*ar.studies, study.id]
        ar.last_family = family(spec)
        ar.best_run_id = self._best(ar)
        best = result.best_trial
        best_metrics: dict[str, float] = {}
        if best is not None:
            best_metrics = _val_only(self.ctx.repo(Run).get(best.run_id).metrics)
        self._observe(
            ar,
            "launch_study",
            {
                "study_id": study.id,
                "archspec_id": archspec_id,
                "strategy": strategy.strategy,
                "trials": {
                    s: sum(t.state == s for t in result.trials)
                    for s in ("complete", "pruned", "fail")
                },
                "stop_reason": result.stop_reason,
                "best_run_id": best.run_id if best else None,
                "best_metrics": best_metrics,
                "agent_best_run_id": ar.best_run_id,
            },
        )
        self._log(
            ar,
            "system",
            f"Iteración {ar.iterations}: {spec.name} con {strategy.strategy}, "
            f"{len(result.trials)} trials; mejor run {ar.best_run_id}.",
        )
        return ar

    # ------------------------------------------------------------------ cierre

    def _finalize(self, ar: AgentRun, run_id: str, summary: str, *, reason: str) -> AgentRun:
        """Único punto que abre el test sellado: evaluación, registro e informe."""
        _, report = self.wf.evaluate(run_id)
        mv = self.wf.register(run_id)
        try:
            self.wf.roles.report(run_id, mode="auto")
        except Exception:
            logger.warning("el informe final falló", exc_info=True)
        ar.best_run_id = run_id
        ar.model_version_id = mv.id
        ar.test_metrics = {k: float(v) for k, v in report.metrics.items()}
        self._log(ar, "finish", summary, data={"run_id": run_id, "test_metrics": ar.test_metrics})
        return self._end(ar, AgentState.FINISHED, reason, save=False)

    def _conclude(self, ar: AgentRun, reason: str) -> AgentRun:
        """Límite alcanzado: se cierra con el mejor modelo, si hay (RF-AGT-02)."""
        self._log(ar, "limit", f"Límite alcanzado: {reason}.")
        best = self._best(ar)
        if best is None:
            return self._end(ar, AgentState.FAILED, f"límite ({reason}) sin modelo entrenado")
        ar = self._finalize(ar, best, f"Cierre por límite ({reason}).", reason=f"limit:{reason}")
        return self.repo.update(ar)

    def _fallback(self, ar: AgentRun, error: Exception) -> AgentRun:
        """RF-AGT-05: con modelo, se cierra con el mejor; sin modelo, una iteración por reglas."""
        code = getattr(error, "code", type(error).__name__)
        self._log(
            ar, "fallback", f"El LLM no respondió de forma válida ({code}); política por reglas."
        )
        ar.fallback = True
        if self._best(ar) is None:
            aid = ar.archspecs[0]
            spec = self._spec(aid)
            strategy = recommend_strategy(spec, self._budget(ar))
            ar = self._study(ar, aid, spec, strategy)
        best = self._best(ar)
        if best is None:
            return self._end(ar, AgentState.FAILED, f"fallback:{code} sin modelo entrenado")
        ar = self._finalize(ar, best, "Cierre por fallback a reglas.", reason=f"fallback:{code}")
        return self.repo.update(ar)

    def _end(self, ar: AgentRun, state: AgentState, reason: str, *, save: bool = True) -> AgentRun:
        ar.state = state
        ar.stop_reason = reason
        ar.finished_at = utcnow()
        self._log(ar, "system", f"Estado final: {state.value} ({reason}).")
        return self.repo.update(ar) if save else ar

    # ------------------------------------------------------------------ límites y bitácora

    def _elapsed(self, ar: AgentRun) -> float:
        started = ar.started_at or ar.created_at
        return (utcnow() - started).total_seconds()

    def _disk_mb(self, ar: AgentRun) -> float:
        root: Path = self.ctx.settings.paths.project(ar.project_id).root
        total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        return total / 1e6

    def _limit_reason(self, ar: AgentRun) -> str | None:
        limits = self._limits(ar)
        checks = [
            (self._elapsed(ar) >= limits.max_time_s, "tiempo"),
            (ar.steps >= limits.max_steps, "decisiones"),
            (ar.iterations >= limits.max_iterations, "iteraciones"),
            (ar.trials >= limits.max_trials, "trials"),
            (ar.cost_usd >= limits.max_llm_cost_usd, "costo de LLM"),
        ]
        for hit, name in checks:
            if hit:
                return name
        if self._disk_mb(ar) >= limits.max_disk_mb:
            return "disco"
        return None

    def _log(
        self,
        ar: AgentRun,
        kind: str,
        message: str,
        *,
        tool: str | None = None,
        data: dict[str, Any] | None = None,
        llm_call_id: str | None = None,
    ) -> None:
        """Bitácora legible (RF-AGT-04), persistida en el AgentRun y publicada en vivo."""
        entry = {
            "ts": time.time(),
            "step": ar.steps,
            "iteration": ar.iterations,
            "kind": kind,
            "message": message,
            "tool": tool,
            "llm_call_id": llm_call_id,
            "data": data,
        }
        ar.log = [*ar.log, entry]
        self.ctx.events.publish(AGENT_TOPIC, agent_id=ar.id, entry=entry)

    def _observe(self, ar: AgentRun, tool: str, data: dict[str, Any]) -> None:
        self._log(ar, "observation", f"Resultado de {tool}", tool=tool, data={"tool": tool, **data})
