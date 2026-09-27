"""`perceptron bench run`: benchmark O2 (SPEC §15.4). Descarga datasets públicos."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

bench_app = typer.Typer(
    help="Benchmark de calidad O2 (agente vs. referencia)", no_args_is_help=True
)


@bench_app.command("run")
def bench_run(
    datasets: Annotated[str, typer.Option(help="Separados por coma")] = "adult,fashion,fsdd",
    out: Annotated[Path, typer.Option(help="Carpeta del reporte")] = Path("bench-report"),
    max_cost: Annotated[float, typer.Option(help="Costo máximo de LLM por dataset (USD)")] = 2.0,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w")] = None,
) -> None:
    """Corre la referencia manual y el agente en cada dataset e informa la brecha (O2)."""
    from perceptron.benchmark.datasets import DATASETS
    from perceptron.benchmark.runner import markdown, run_benchmark
    from perceptron.cli.ml import _ctx

    keys = [k.strip() for k in datasets.split(",") if k.strip()]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise typer.BadParameter(f"datasets desconocidos: {unknown} (hay {sorted(DATASETS)})")
    with _ctx(workspace) as ctx:
        results = run_benchmark(ctx, keys, out, max_cost=max_cost)
    typer.echo(markdown(results, "hoy"))
    if not all(r.error is None for r in results):
        raise typer.Exit(1)
