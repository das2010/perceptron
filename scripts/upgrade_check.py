"""Actualización N → N+1 sin pérdida de proyectos (aceptación de la Capa 7).

Abre con la versión actual un workspace creado por la versión anterior y verifica que:
- todas las entidades guardadas se leen con los modelos actuales (compatibilidad de datos);
- cada proyecto tiene su carpeta y cada versión de datos se puede leer;
- los runs conservan sus artefactos y los modelos registrados siguen ahí.

    uv run python scripts/upgrade_check.py <workspace>
"""

from __future__ import annotations

import sys
from pathlib import Path

from perceptron.api.context import EngineContext
from perceptron.core.config import LoggingSettings, Settings
from perceptron.domain.models import ALL_ENTITIES, DatasetVersion, ModelVersion, Project, Run
from perceptron.services.workflow import Workflow


def main(workspace: Path) -> int:
    settings = Settings(workspace_dir=workspace, logging=LoggingSettings(to_file=False))
    ctx = EngineContext.create(settings)
    problems: list[str] = []
    try:
        counts = {}
        for model in ALL_ENTITIES:
            try:
                counts[model.__name__] = len(ctx.repo(model).list(limit=1_000_000))
            except Exception as exc:  # documento que el modelo nuevo no acepta
                problems.append(f"{model.__name__}: {type(exc).__name__}: {exc}"[:500])
        wf = Workflow(ctx)
        projects = ctx.repo(Project).list(limit=100_000)
        if not projects:
            problems.append("el workspace no tiene proyectos")
        for p in projects:
            if not settings.paths.project(p.id).root.is_dir():
                problems.append(f"falta la carpeta del proyecto {p.id}")
        for dv in ctx.repo(DatasetVersion).list(limit=100_000):
            try:
                wf.view(dv).scan().head(5).collect()
            except Exception as exc:
                problems.append(f"no se puede leer la versión {dv.id}: {exc}"[:300])
        for run in ctx.repo(Run).list(limit=100_000):
            if run.status.value == "succeeded" and not any(wf._run_dir(run).glob("*.json")):
                problems.append(f"el run {run.id} perdió sus artefactos")
        if not ctx.repo(ModelVersion).list(limit=1):
            problems.append("no quedó ningún modelo registrado")
        summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        print(f"entidades leídas: {summary}")
    finally:
        ctx.close()
    for p in problems:
        print(f"PROBLEMA  {p}")
    print("actualización OK" if not problems else f"{len(problems)} problemas")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
