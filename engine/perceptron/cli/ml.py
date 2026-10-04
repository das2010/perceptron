"""Subcomandos de ML de la CLI (SPEC §7.18).

data, pipeline, arch, hpo, train, eval, model y quickstart.

Todos aceptan `--json` para salida estructurada. Los imports pesados (torch,
polars, optuna, mlflow) se hacen dentro de cada comando.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from perceptron.core.config import Settings
from perceptron.domain.enums import Device, Modality, TaskType

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

WorkspaceOpt = Annotated[
    Path | None, typer.Option("--workspace", "-w", help="Directorio del workspace")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Salida JSON")]
LlmOpt = Annotated[
    bool,
    typer.Option("--llm/--no-llm", help="Proponer con el LLM (cae a reglas si no está disponible)"),
]
ProjectOpt = Annotated[str, typer.Option("--project", "-p", help="Id del proyecto")]

data_app = typer.Typer(help="Ingesta y profiling de datos", no_args_is_help=True)
pipeline_app = typer.Typer(help="Pipelines de preparación", no_args_is_help=True)
arch_app = typer.Typer(help="Arquitecturas (ArchSpec)", no_args_is_help=True)
hpo_app = typer.Typer(help="Optimización de hiperparámetros", no_args_is_help=True)
model_app = typer.Typer(help="Modelos registrados", no_args_is_help=True)
llm_app = typer.Typer(
    help="Capa LLM: proveedores, perfiles, auditoría y roles", no_args_is_help=True
)


def register(app: typer.Typer) -> None:
    app.add_typer(data_app, name="data")
    app.add_typer(pipeline_app, name="pipeline")
    app.add_typer(arch_app, name="arch")
    app.add_typer(hpo_app, name="hpo")
    app.add_typer(model_app, name="model")
    app.add_typer(llm_app, name="llm")
    from perceptron.cli.agent import agent_app
    from perceptron.cli.bench import bench_app

    app.add_typer(agent_app, name="agent")
    app.add_typer(bench_app, name="bench")
    app.command("train")(train)
    app.command("eval")(evaluate)
    app.command("quickstart")(quickstart)
    app.command("export")(export)


@contextmanager
def _ctx(workspace: Path | None) -> Iterator[EngineContext]:
    from perceptron.api.context import EngineContext

    ctx = EngineContext.create(Settings(workspace_dir=workspace) if workspace else Settings())
    try:
        yield ctx
    finally:
        ctx.close()


def _out(data: Any, as_json: bool, human: str | None = None) -> None:
    from pydantic import BaseModel

    if as_json:
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        typer.echo(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
    elif human is not None:
        typer.echo(human)


def _progress(ev: Any) -> None:
    if ev.event == "epoch":
        m = ev.metrics
        shown = " ".join(
            f"{k}={v:.4g}" for k, v in m.items() if k.startswith("val_") or k == "train_loss"
        )
        typer.echo(f"  [{ev.run_id}] época {ev.epoch}: {shown}", err=True)
    elif ev.event == "error":
        typer.echo(f"  [{ev.run_id}] ERROR: {ev.data.get('message')}", err=True)


# ------------------------------------------------------------------ data


@data_app.command("ingest")
def data_ingest(
    source: Annotated[Path, typer.Argument(help="Archivo, carpeta de imágenes o ZIP")],
    project: ProjectOpt,
    target: Annotated[str | None, typer.Option(help="Columna objetivo")] = None,
    modality: Annotated[Modality | None, typer.Option(help="Forzar la modalidad")] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Crea una DatasetVersion inmutable (RF-ING-01, 07, 08)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        dv = Workflow(ctx).ingest(project, source, target=target, modality=modality)
        split = dv.split
        _out(
            dv,
            as_json,
            f"{dv.id}  {dv.num_samples} muestras · target={dv.target} · "
            f"split train/val/test={split.train}/{split.val}/{split.test}"
            if split
            else dv.id,
        )


@data_app.command("profile")
def data_profile(
    dataset: Annotated[str, typer.Argument(help="Id de la DatasetVersion")],
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Dataset Profile Card con alertas de calidad (RF-PRF-07)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        card = Workflow(ctx).profile(dataset)
        lines = [
            f"{card.modality.value} · {card.num_samples} muestras · {card.num_features} features"
        ]
        if card.target:
            lines.append(f"target: {card.target.name} ({card.target.task_hint.value})")
        lines += [
            f"[{a.severity.value}] {a.code.value}{' · ' + a.column if a.column else ''}: "
            f"{a.message}"
            for a in card.alerts
        ]
        _out(card, as_json, "\n".join(lines))


# ------------------------------------------------------------------ pipeline / arch


@pipeline_app.command("propose")
def pipeline_propose(
    dataset: Annotated[str, typer.Argument()],
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Pipeline automático desde el profiling (RF-PIP-01)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        p = Workflow(ctx).propose_pipeline(dataset)
        why = "\n".join(f"  · {r}" for r in p.graph.get("rationale", []))
        _out(p, as_json, f"{p.id}\n{why}")


@arch_app.command("propose")
def arch_propose(
    dataset: Annotated[str, typer.Argument()],
    pipeline: Annotated[str, typer.Option("--pipeline", help="Id del pipeline")],
    llm: LlmOpt = False,
    n: Annotated[int, typer.Option(min=2, max=4, help="Propuestas pedidas al LLM")] = 3,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Propuestas de arquitectura: del LLM (RF-ARC-01) o por reglas (RF-ARC-04)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        out = Workflow(ctx).roles.propose_architectures(
            dataset, pipeline, mode="auto" if llm else "rules", n=n
        )
        data = {
            "origin": out.origin.value,
            "llm_call_id": out.llm_call_id,
            "fallback_reason": out.fallback_reason,
            "requirements": out.requirements.model_dump(mode="json") if out.requirements else None,
            "proposals": [
                {
                    "archspec": o.record.model_dump(mode="json"),
                    "title": o.title,
                    "rationale": o.rationale,
                    "pros": o.pros,
                    "cons": o.cons,
                    "risks": o.risks,
                    "confidence": o.confidence,
                    "estimates": o.estimates,
                    "assessment": o.assessment.model_dump(mode="json") if o.assessment else None,
                }
                for o in out.options
            ],
        }
        head = f"Origen: {out.origin.value}"
        if out.fallback_reason:
            head += f" ({out.fallback_reason})"

        def mark(o: Any) -> str:
            a = o.assessment
            return f"  [recomendada · {a.score:g}]" if a is not None and a.recommended else ""

        lines = [head] + [
            f"{o.record.id}  {o.title}{mark(o)}\n  {o.rationale}" for o in out.options
        ]
        _out(data, as_json, "\n".join(lines))


@arch_app.command("validate")
def arch_validate(
    spec_file: Annotated[Path, typer.Argument(help="ArchSpec JSON")], as_json: JsonOpt = False
) -> None:
    """Valida una ArchSpec (§9.3). Código de salida 1 si es inválida."""
    from perceptron.archspec.validate import validate_archspec

    report = validate_archspec(json.loads(spec_file.read_text(encoding="utf-8")))
    human = "válida" if report.valid else "inválida"
    human += (
        f" · {report.num_params:,} parámetros · ~{report.estimated_memory_mb} MB"
        if report.num_params
        else ""
    )
    human += "".join(
        f"\n  [{i.severity.value}] {i.stage.value} {i.path}: {i.message}" for i in report.issues
    )
    _out(report.model_dump(mode="json", exclude={"spec"}), as_json, human)
    if not report.valid:
        raise typer.Exit(1)


@arch_app.command("to-code")
def arch_to_code(
    spec_file: Annotated[Path, typer.Argument(help="ArchSpec JSON")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Código PyTorch legible equivalente a la ArchSpec (RF-ARC-07)."""
    from perceptron.archspec.schema import ArchSpec
    from perceptron.archspec.to_code import archspec_to_code

    code = archspec_to_code(ArchSpec.model_validate_json(spec_file.read_text(encoding="utf-8")))
    if output:
        output.write_text(code, encoding="utf-8")
    else:
        typer.echo(code)


# ------------------------------------------------------------------ HPO / train / eval


def _budget(trials: int, max_epochs: int | None, max_time: float | None) -> Any:
    from perceptron.hpo.strategy import Budget

    return Budget(max_trials=trials, max_epochs_per_trial=max_epochs, max_time_s=max_time)


@hpo_app.command("strategy")
def hpo_strategy(
    archspec: Annotated[str, typer.Argument(help="Id de la ArchSpec")],
    trials: Annotated[int, typer.Option(help="Máximo de trials")] = 20,
    max_epochs: Annotated[int | None, typer.Option(help="Épocas máximas por trial")] = None,
    llm: LlmOpt = False,
    dataset: Annotated[str | None, typer.Option(help="Dataset (perfil para el LLM)")] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Estrategia de HPO recomendada (RF-HPO-02): LLM estratega o reglas."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        s = Workflow(ctx).hpo_strategy(
            archspec,
            _budget(trials, max_epochs, None),
            mode="auto" if llm else "rules",
            dataset_version_id=dataset,
        )
        _out(
            s,
            as_json,
            f"{s.strategy} + pruner {s.pruner} · {len(s.search_space)} hiperparámetros"
            f"\n  {s.rationale}",
        )


@hpo_app.command("run")
def hpo_run(
    project: ProjectOpt,
    dataset: Annotated[str, typer.Option("--dataset")],
    pipeline: Annotated[str, typer.Option("--pipeline")],
    archspec: Annotated[str, typer.Option("--archspec")],
    trials: Annotated[int, typer.Option()] = 10,
    max_epochs: Annotated[int | None, typer.Option()] = None,
    max_time: Annotated[float | None, typer.Option(help="Segundos totales")] = None,
    device: Annotated[Device | None, typer.Option()] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Estudio de HPO completo (RF-HPO-01, 03)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        wf = Workflow(ctx)
        strategy = wf.hpo_strategy(archspec, _budget(trials, max_epochs, max_time))
        study, result = wf.run_study(
            project, dataset, pipeline, archspec, strategy, device=device, on_event=_progress
        )
        best = result.best_trial
        _out(
            {"study_id": study.id, **result.model_dump(mode="json")},
            as_json,
            f"{study.id}: {len(result.trials)} trials ({result.stop_reason}) · mejor: "
            f"{best.run_id if best else '-'} {best.values if best else ''}",
        )


def train(
    project: ProjectOpt,
    dataset: Annotated[str, typer.Option("--dataset")],
    pipeline: Annotated[str, typer.Option("--pipeline")],
    archspec: Annotated[str, typer.Option("--archspec")],
    max_epochs: Annotated[int | None, typer.Option()] = None,
    device: Annotated[Device | None, typer.Option()] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Un entrenamiento con los hiperparámetros por defecto de la ArchSpec."""
    from perceptron.hpo.strategy import HPOStrategy
    from perceptron.services.workflow import Workflow

    strategy = HPOStrategy(strategy="single", pruner="none", budget=_budget(1, max_epochs, None))
    with _ctx(workspace) as ctx:
        _, result = Workflow(ctx).run_study(
            project, dataset, pipeline, archspec, strategy, device=device, on_event=_progress
        )
        best = result.trials[0] if result.trials else None
        _out(
            best.model_dump(mode="json") if best else {},
            as_json,
            f"{best.run_id}: {best.state}" if best else "sin resultado",
        )


def evaluate(
    run: Annotated[str, typer.Argument(help="Id del run")],
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Evaluación final sobre el test sellado (RF-EVL-01)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        _, report = Workflow(ctx).evaluate(run)
        _out(report, as_json, "\n".join(f"{k}: {v:.4f}" for k, v in report.metrics.items()))


@model_app.command("register")
def model_register(
    run: Annotated[str, typer.Argument()], workspace: WorkspaceOpt = None, as_json: JsonOpt = False
) -> None:
    """Registra el run evaluado como ModelVersion `candidate`."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        mv = Workflow(ctx).register(run)
        _out(mv, as_json, f"{mv.id} ({mv.stage.value})")


def _series_overrides(
    task: TaskType | None, time_column: str | None, series_id: str | None, horizon: int | None
) -> dict[str, Any] | None:
    raw = {"task": task, "time_column": time_column, "series_id": series_id, "horizon": horizon}
    given = {k: v for k, v in raw.items() if v is not None}
    return given or None


def quickstart(
    source: Annotated[Path, typer.Argument(help="Archivo tabular o carpeta de imágenes")],
    target: Annotated[
        str | None, typer.Option(help="Columna objetivo (se infiere si falta)")
    ] = None,
    name: Annotated[str | None, typer.Option(help="Nombre del proyecto")] = None,
    modality: Annotated[Modality | None, typer.Option(help="Forzar la modalidad")] = None,
    task: Annotated[
        TaskType | None, typer.Option(help="Series: forecasting | anomaly_detection")
    ] = None,
    time_column: Annotated[str | None, typer.Option(help="Series: columna de tiempo")] = None,
    series_id: Annotated[str | None, typer.Option(help="Series: columna identificadora")] = None,
    horizon: Annotated[int | None, typer.Option(help="Series: pasos a pronosticar")] = None,
    trials: Annotated[int, typer.Option(help="Trials de HPO")] = 10,
    max_epochs: Annotated[int | None, typer.Option(help="Épocas máximas por trial")] = None,
    max_time: Annotated[float | None, typer.Option(help="Tiempo máximo total (s)")] = None,
    device: Annotated[Device | None, typer.Option()] = None,
    llm: LlmOpt = False,
    goal: Annotated[str | None, typer.Option(help="Objetivo en palabras (para el LLM)")] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """De los datos a un modelo evaluado y registrado, sin escribir código."""
    from perceptron.services.workflow import quickstart as run_quickstart

    with _ctx(workspace) as ctx:
        res = run_quickstart(
            ctx,
            source,
            name=name,
            target=target,
            modality=modality,
            series_overrides=_series_overrides(task, time_column, series_id, horizon),
            task=task,
            trials=trials,
            max_epochs=max_epochs,
            max_time_s=max_time,
            device=device,
            on_event=_progress,
            llm=llm,
            goal=goal,
        )
        s = res.summary()
        human = [
            f"Proyecto {s['project_id']} · {s['modality']} · {s['samples']} muestras · "
            f"target {s['target']}",
            f"Arquitectura: {s['architecture']} — {s['architecture_rationale']}",
            f"HPO: {s['hpo_strategy']} + {s['hpo_pruner']} · {s['trials']} trials "
            f"{s['trial_states']} ({s['stop_reason']})",
        ]
        if s["test_metrics"]:
            human.append(
                "Test: "
                + ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in s["test_metrics"].items()
                    if k in ("accuracy", "f1_macro", "roc_auc", "mae", "rmse", "r2")
                )
            )
        human.append(f"Modelo registrado: {s['model_version_id']}")
        _out(s, as_json, "\n".join(human))


# ------------------------------------------------------------------ capa LLM


@llm_app.command("providers")
def llm_providers(workspace: WorkspaceOpt = None, as_json: JsonOpt = False) -> None:
    """Proveedores configurados (sin mostrar claves)."""
    from perceptron.api.routers.llm import provider_views

    with _ctx(workspace) as ctx:
        views = provider_views(ctx)
        human = "\n".join(
            f"{v.name:14} {v.kind:13} {'local' if v.local else 'nube':5} "
            f"clave: {'sí' if v.has_key else 'no'}  modelos: {', '.join(v.models)}"
            for v in views
        )
        _out([v.model_dump(mode="json") for v in views], as_json, human)


@llm_app.command("profiles")
def llm_profiles(workspace: WorkspaceOpt = None, as_json: JsonOpt = False) -> None:
    """Perfiles por propósito y el activo (RF-LLM-03)."""
    with _ctx(workspace) as ctx:
        cfg = ctx.llm.config
        profiles = cfg.profiles()
        human = [f"Activo: {cfg.active_profile()}"]
        for name, prof in profiles.items():
            human.append(f"{name} ({prof.provider})")
            human += [f"  {p.value:15} {ref.model}" for p, ref in prof.purposes.items()]
        data = {
            "active": cfg.active_profile(),
            "profiles": {k: v.model_dump(mode="json") for k, v in profiles.items()},
        }
        _out(data, as_json, "\n".join(human))


@llm_app.command("test")
def llm_test(
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Prueba de conexión con el perfil activo (o el del proyecto)."""
    from perceptron.api.routers.llm import LLMTestBody, test_llm

    with _ctx(workspace) as ctx:
        res = test_llm(LLMTestBody(project_id=project), ctx)
        if res.ok:
            human = f"ok · {res.provider}/{res.model} · {res.latency_s}s"
        else:
            human = f"ERROR: {res.error}"
        _out(res, as_json, human)
        if not res.ok:
            raise typer.Exit(1)


@llm_app.command("audit")
def llm_audit(
    project: ProjectOpt,
    limit: Annotated[int, typer.Option()] = 20,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Qué se envió al LLM en cada llamada (RF-PRV-03)."""
    from perceptron.domain.models import LLMCall

    with _ctx(workspace) as ctx:
        calls = list(ctx.repo(LLMCall).list(filters={"project_id": project}, limit=limit))
        lines = []
        for c in calls:
            cached = " (caché)" if c.cache_hit else ""
            lines.append(
                f"{c.created_at:%Y-%m-%d %H:%M:%S} {c.purpose.value:14} {c.provider}/{c.model} "
                f"{c.privacy_level.value} {c.status} intento {c.attempt} "
                f"${c.cost_usd:.4f}{cached} {' '.join(c.redactions)}"
            )
        _out(
            [c.model_dump(mode="json") for c in calls], as_json, "\n".join(lines) or "sin llamadas"
        )


@llm_app.command("diagnose")
def llm_diagnose(
    run: Annotated[str, typer.Argument(help="Id del run")],
    llm: LlmOpt = True,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Diagnóstico del entrenamiento: reglas + LLM (Diagnosticador)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        d = Workflow(ctx).roles.diagnose(run, mode="auto" if llm else "rules")
        lines = [f"{d.summary} [{d.origin}]"]
        lines += [f"  - {p.kind} ({p.severity}): {p.evidence}" for p in d.problems]
        for a in d.actions:
            value = "" if a.value is None else f" = {a.value}"
            lines.append(f"  → {a.kind} {a.target or ''}{value}: {a.rationale}")
        _out(d, as_json, "\n".join(lines))


@llm_app.command("report")
def llm_report(
    run: Annotated[str, typer.Argument(help="Id del run evaluado")],
    llm: LlmOpt = True,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Informe final en Markdown + model card (Informante)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        r = Workflow(ctx).roles.report(run, mode="auto" if llm else "rules")
        _out(r, as_json, r.markdown)


def export(
    run_id: Annotated[str, typer.Argument(help="Run a exportar")],
    formats: Annotated[
        str, typer.Option(help="Formatos separados por coma: onnx, torch_export, torchscript")
    ] = "onnx",
    fp16: Annotated[bool, typer.Option(help="ONNX fp16")] = False,
    int8: Annotated[bool, typer.Option(help="ONNX INT8 dinámico")] = False,
    serving: Annotated[
        Path | None, typer.Option(help="Guardar el servidor de inferencia (zip) en esta ruta")
    ] = None,
    project: Annotated[
        Path | None, typer.Option(help="Guardar el proyecto de código (zip) en esta ruta")
    ] = None,
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Exporta un run verificado (ONNX/torch.export/TorchScript) y, opcionalmente, el servidor
    de inferencia y el proyecto de código (RF-EXP-01, 03, 04)."""
    import shutil

    from perceptron.export.formats import ExportFormat, ExportRequest
    from perceptron.services.workflow import Workflow

    request = ExportRequest(
        formats=[ExportFormat(f.strip()) for f in formats.split(",") if f.strip()],
        fp16=fp16,
        int8=int8,
    )
    with _ctx(workspace) as ctx:
        wf = Workflow(ctx)
        report = wf.export(run_id, request)
        out: dict[str, Any] = {"report": report.model_dump(mode="json")}
        if serving is not None:
            shutil.copyfile(wf.serving_bundle(run_id), serving)
            out["serving"] = str(serving)
        if project is not None:
            shutil.copyfile(wf.project_zip(run_id), project)
            out["project"] = str(project)
    if as_json:
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return
    for a in report.artifacts:
        v = a.verification
        state = a.error or (f"máx. dif. {v.max_abs_diff:.2e}" if v else "ok")
        typer.echo(f"{a.format:14} {a.file or '—':28} {state}")
    for key in ("serving", "project"):
        if key in out:
            typer.echo(f"{key}: {out[key]}")
    if not report.ok:
        raise typer.Exit(1)
