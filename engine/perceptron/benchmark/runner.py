"""Benchmark O2 (SPEC §1.3, §15.4): el agente vs. una configuración manual de referencia.

Por dataset: (1) referencia = arquitectura por reglas con hiperparámetros manuales fijos,
entrenada una vez; (2) el agente con presupuesto fijo. Se informa la brecha en la métrica
de test; O2 se cumple si el agente queda dentro del 5 % de la referencia.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perceptron.archspec.schema import HP, ArchSpec
from perceptron.benchmark.datasets import DATASETS, BenchDataset
from perceptron.domain.enums import Origin
from perceptron.domain.models import Project
from perceptron.hpo.strategy import Budget, HPOStrategy
from perceptron.training.module import monitor_mode

logger = logging.getLogger(__name__)

O2_GAP = 0.05


def with_defaults(spec: ArchSpec, values: dict[str, Any]) -> ArchSpec:
    """Copia de la ArchSpec con los `{hp, default}` indicados fijados a mano."""

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            if set(v) >= {"hp", "default"} and v["hp"] in values:
                return {**v, "default": values[v["hp"]]}
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    data = walk(spec.model_dump(mode="json"))
    return ArchSpec.model_validate(data)


def gap(reference: float, agent: float, metric: str) -> float:
    """Brecha relativa (positivo = el agente es peor).

    Con referencia 0 no hay escala relativa: se usa la diferencia absoluta (antes devolvía 0
    y un agente peor que una referencia perfecta de error 0 «aprobaba» O2).
    """
    worse = agent - reference if monitor_mode(metric) == "min" else reference - agent
    if reference == 0:
        return worse
    if monitor_mode(metric) == "min":
        return (agent - reference) / abs(reference)
    return (reference - agent) / abs(reference)


@dataclass
class BenchResult:
    dataset: str
    description: str
    license: str
    metric: str
    reference: float | None
    agent: float | None
    gap: float | None
    o2: bool
    agent_state: str | None = None
    agent_cost_usd: float = 0.0
    agent_trials: int = 0
    fallback: bool = False
    error: str | None = None


def _epochs_spec(spec: ArchSpec, epochs: int) -> ArchSpec:
    training = spec.training.model_copy(update={"epochs": HP(hp="epochs", default=epochs)})
    return spec.model_copy(update={"training": training})


def run_dataset(ctx: Any, ds: BenchDataset, cache: Path, *, max_cost: float) -> BenchResult:
    from perceptron.agent.loop import AgentRunner
    from perceptron.agent.models import AgentLimits
    from perceptron.services.workflow import Workflow

    wf = Workflow(ctx)
    source = ds.prepare(cache)
    metric = ds.metric
    try:
        # (1) referencia manual
        ref_project = ctx.projects.add(Project(name=f"bench-{ds.key}-ref"))
        ctx.files.init_project(ref_project)
        dv = wf.ingest(ref_project.id, source)
        wf.profile(dv.id)
        pipe = wf.propose_pipeline(dv.id)
        rules, _ = wf.propose_architecture(dv.id, pipe.id)
        spec = _epochs_spec(
            with_defaults(ArchSpec.model_validate(rules.spec), ds.reference), ds.epochs
        )
        ref_record = wf.save_archspec(ref_project.id, spec, origin=Origin.MANUAL)
        strategy = HPOStrategy(
            strategy="single",
            pruner="none",
            budget=Budget(max_trials=1, max_epochs_per_trial=ds.epochs),
            origin=Origin.MANUAL,
            rationale="Referencia manual del benchmark O2",
        )
        _, res = wf.run_study(ref_project.id, dv.id, pipe.id, ref_record.id, strategy)
        if res.best_trial is None:
            raise RuntimeError("la referencia no entrenó")
        _, ref_eval = wf.evaluate(res.best_trial.run_id)
        reference = float(ref_eval.metrics[metric])

        # (2) agente con presupuesto fijo
        project = ctx.projects.add(
            Project(name=f"bench-{ds.key}-agente", goal=f"Maximizar {metric}: {ds.description}")
        )
        ctx.files.init_project(project)
        dv2 = wf.ingest(project.id, source)
        wf.profile(dv2.id)
        pipe2 = wf.propose_pipeline(dv2.id)
        runner = AgentRunner(wf)
        ar = runner.start(
            project.id,
            dv2.id,
            pipe2.id,
            limits=AgentLimits(
                max_trials=10,
                max_iterations=3,
                max_steps=12,
                max_epochs_per_trial=ds.epochs,
                max_llm_cost_usd=max_cost,
                max_time_s=3600,
            ),
        )
        ar = runner.run(ar.id)
        agent = ar.test_metrics.get(metric)
        g = gap(reference, agent, metric) if agent is not None else None
        return BenchResult(
            dataset=ds.key,
            description=ds.description,
            license=ds.license,
            metric=metric,
            reference=reference,
            agent=agent,
            gap=g,
            o2=g is not None and g <= O2_GAP,
            agent_state=ar.state.value,
            agent_cost_usd=ar.cost_usd,
            agent_trials=ar.trials,
            fallback=ar.fallback,
        )
    except Exception as e:
        logger.exception("benchmark falló", extra={"dataset": ds.key})
        return BenchResult(
            dataset=ds.key,
            description=ds.description,
            license=ds.license,
            metric=metric,
            reference=None,
            agent=None,
            gap=None,
            o2=False,
            error=f"{type(e).__name__}: {e}",
        )


def markdown(results: list[BenchResult], when: str) -> str:
    lines = [
        f"# Benchmark O2 — {when}",
        "",
        "Referencia: arquitectura por reglas con hiperparámetros manuales fijos. Agente: "
        "presupuesto de 10 trials / 3 iteraciones. O2: brecha ≤ 5 %.",
        "",
        "| Dataset | Licencia | Métrica | Referencia | Agente | Brecha | O2 | Costo LLM |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{x:.4f}"

    for r in results:
        brecha = "—" if r.gap is None else f"{r.gap:+.1%}"
        estado = "✅" if r.o2 else "❌"
        if r.error:
            estado += f" ({r.error[:60]})"
        lines.append(
            f"| {r.description} | {r.license} | {r.metric} | {fmt(r.reference)} | "
            f"{fmt(r.agent)} | {brecha} | {estado} | ${r.agent_cost_usd:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run_benchmark(ctx: Any, keys: list[str], out: Path, *, max_cost: float) -> list[BenchResult]:
    cache = ctx.settings.paths.root / "cache" / "bench"
    results = [run_dataset(ctx, DATASETS[k], cache, max_cost=max_cost) for k in keys]
    when = datetime.now(UTC).strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / f"benchmark-{when}.md").write_text(markdown(results, when), encoding="utf-8")
    return results
