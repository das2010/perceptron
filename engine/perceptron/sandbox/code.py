"""Carga y construcción del modelo de código experto (RF-ARC-06; ADR-0025).

El código solo se ejecuta dentro de un proceso sandbox (`guard.install` ya aplicado): fuera
de él, `load` y `build` fallan. Interfaz fija del código del usuario:

    def build_model(config: dict) -> torch.nn.Module

`config` trae `input` (kind, shape, num_numeric, cardinalities, vocab_size, pad_id),
`num_outputs`, `task` y `hparams`. El módulo recibe `(x_num, x_cat)` en tabular y `x` en
el resto, y devuelve `[B, num_outputs]`. Opcionalmente, `input_spec = {"kind": ...}` y
`output_spec = {"task": ...}` se contrastan con los datos.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from torch import nn

from perceptron.catalog.registry import CODE_BLOCK
from perceptron.sandbox.guard import SANDBOX_ENV
from perceptron.sandbox.static import ENTRYPOINT, check_source

__all__ = ["CODE_BLOCK", "CODE_FILE", "build", "in_sandbox", "load", "source_sha256"]

CODE_FILE = "model_code.py"

_loaded: dict[str, Any] | None = None
_sha: str | None = None


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def in_sandbox() -> bool:
    return os.environ.get(SANDBOX_ENV) == "1"


def load(source: str) -> None:
    """Ejecuta el módulo del usuario (solo dentro del sandbox)."""
    global _loaded, _sha
    if not in_sandbox():
        raise RuntimeError("el código experto solo se ejecuta dentro del sandbox")
    report = check_source(source)
    if not report.valid:
        raise ValueError("; ".join(f"línea {i.line}: {i.message}" for i in report.issues))
    namespace: dict[str, Any] = {"__name__": "perceptron_user_model"}
    exec(compile(source, CODE_FILE, "exec"), namespace)  # noqa: S102 - dentro del sandbox
    if not callable(namespace.get(ENTRYPOINT)):
        raise ValueError(f"el código no define `{ENTRYPOINT}(config)`")
    _loaded, _sha = namespace, source_sha256(source)


def build(params: dict[str, Any], input_spec: dict[str, Any], num_outputs: int) -> nn.Module:
    """Llama a `build_model(config)` del código cargado y verifica el contrato."""
    if _loaded is None or not in_sandbox():
        raise ValueError("el código experto solo se construye dentro del sandbox")
    expected = params.get("code_sha256")
    if expected and expected != _sha:
        raise ValueError("el código cargado no coincide con el de la ArchSpec (code_sha256)")
    declared = _loaded.get("input_spec")
    if isinstance(declared, dict) and declared.get("kind") not in (None, input_spec["kind"]):
        raise ValueError(
            f"input_spec.kind = {declared.get('kind')!r} y los datos son {input_spec['kind']!r}"
        )
    task = params.get("task")
    out_decl = _loaded.get("output_spec")
    if isinstance(out_decl, dict) and out_decl.get("task") not in (None, task):
        raise ValueError(f"output_spec.task = {out_decl.get('task')!r} y la tarea es {task!r}")
    hparams = {k: v for k, v in params.items() if k not in {"code_sha256", "task"}}
    config = {"input": input_spec, "num_outputs": num_outputs, "task": task, "hparams": hparams}
    model = _loaded[ENTRYPOINT](config)
    if not isinstance(model, nn.Module):
        raise TypeError(f"`{ENTRYPOINT}` devolvió {type(model).__name__}, no un nn.Module")
    return model
