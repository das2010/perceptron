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
from perceptron.domain.enums import Device

if TYPE_CHECKING:
    from perceptron.api.context import EngineContext

WorkspaceOpt = Annotated[
    Path | None, typer.Option("--workspace", "-w", help="Directorio del workspace")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Salida JSON")]
ProjectOpt = Annotated[str, typer.Option("--project", "-p", help="Id del proyecto")]

data_app = typer.Typer(help="Ingesta y profiling de datos", no_args_is_help=True)
pipeline_app = typer.Typer(help="Pipelines de preparación", no_args_is_help=True)
arch_app = typer.Typer(help="Arquitecturas (ArchSpec)", no_args_is_help=True)
hpo_app = typer.Typer(help="Optimización de hiperparámetros", no_args_is_help=True)
model_app = typer.Typer(help="Modelos registrados", no_args_is_help=True)


def register(app: typer.Typer) -> None:
    app.add_typer(data_app, name="data")
    app.add_typer(pipeline_app, name="pipeline")
    app.add_typer(arch_app, name="arch")
    app.add_typer(hpo_app, name="hpo")
    app.add_typer(model_app, name="model")
    app.command("train")(train)
    app.command("eval")(evaluate)
    app.command("quickstart")(quickstart)


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
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Crea una DatasetVersion inmutable (RF-ING-01, 07, 08)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        dv = Workflow(ctx).ingest(project, source, target=target)
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
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Arquitectura recomendada por reglas (RF-ARC-04)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        record, why = Workflow(ctx).propose_architecture(dataset, pipeline)
        _out(
            {"archspec": record.model_dump(mode="json"), "rationale": why},
            as_json,
            f"{record.id}  {record.name}\n  {why}",
        )


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
    workspace: WorkspaceOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Estrategia de HPO recomendada (RF-HPO-02, por reglas)."""
    from perceptron.services.workflow import Workflow

    with _ctx(workspace) as ctx:
        s = Workflow(ctx).hpo_strategy(archspec, _budget(trials, max_epochs, None))
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


def quickstart(
    source: Annotated[Path, typer.Argument(help="Archivo tabular o carpeta de imágenes")],
    target: Annotated[
        str | None, typer.Option(help="Columna objetivo (se infiere si falta)")
    ] = None,
    name: Annotated[str | None, typer.Option(help="Nombre del proyecto")] = None,
    trials: Annotated[int, typer.Option(help="Trials de HPO")] = 10,
    max_epochs: Annotated[int | None, typer.Option(help="Épocas máximas por trial")] = None,
    max_time: Annotated[float | None, typer.Option(help="Tiempo máximo total (s)")] = None,
    device: Annotated[Device | None, typer.Option()] = None,
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
            trials=trials,
            max_epochs=max_epochs,
            max_time_s=max_time,
            device=device,
            on_event=_progress,
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
