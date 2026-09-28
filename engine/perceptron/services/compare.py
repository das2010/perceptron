"""Diff de configuración entre runs (RF-TRK-04): ArchSpec y pipeline, además de los
hiperparámetros que ya muestra la comparación.

Cada configuración se aplana a rutas (`nodes[enc].params.hidden`); en listas de objetos con
`id` (nodos, pasos) se usa el id y no la posición, así reordenar no aparece como cambio.
Solo se devuelven las rutas que difieren entre los runs.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

MISSING = "—"


class ConfigDiffRow(BaseModel):
    path: str
    values: list[Any]


class ConfigDiff(BaseModel):
    run_ids: list[str]
    same_archspec: bool
    same_pipeline: bool
    archspec: list[ConfigDiffRow]
    pipeline: list[ConfigDiffRow]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Hojas del JSON por ruta. Listas de objetos con `id` se indexan por id."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value and prefix:
            out[prefix] = {}
        for key in sorted(value):
            out.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        keyed = value and all(isinstance(v, dict) and "id" in v for v in value)
        if not value and prefix:
            out[prefix] = []
        for i, item in enumerate(value):
            key = str(item["id"]) if keyed else str(i)
            out.update(flatten(item, f"{prefix}[{key}]"))
    else:
        out[prefix] = value
    return out


def diff_rows(configs: list[Any]) -> list[ConfigDiffRow]:
    flat = [flatten(c or {}) for c in configs]
    paths = sorted({p for f in flat for p in f})
    rows = []
    for path in paths:
        values = [f.get(path, MISSING) for f in flat]
        if len({json.dumps(v, sort_keys=True, default=str) for v in values}) > 1:
            rows.append(ConfigDiffRow(path=path, values=values))
    return rows


def config_diff(run_ids: list[str], archspecs: list[Any], pipelines: list[Any]) -> ConfigDiff:
    arch = diff_rows(archspecs)
    pipe = diff_rows(pipelines)
    return ConfigDiff(
        run_ids=run_ids,
        same_archspec=not arch,
        same_pipeline=not pipe,
        archspec=arch,
        pipeline=pipe,
    )
