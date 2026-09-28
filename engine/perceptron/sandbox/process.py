"""Lanzamiento de procesos sandbox desde el Engine (SPEC §13.2; ADR-0025).

El proceso hijo corre con `python -I` (sin variables `PYTHON*` ni site del usuario), un
entorno mínimo (sin proxies, tokens ni claves) y el directorio del run como `cwd`, `HOME` y
temporal. Las guardas internas las instala el propio hijo (`guard.install`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, Field

from perceptron.core.errors import ValidationError

# Variables que el runtime necesita (DLLs de Windows, hilos, GPU, locale); nada más.
_PASSTHROUGH = (
    "CUDA_VISIBLE_DEVICES",
    "LANG",
    "LC_ALL",
    "MKL_NUM_THREADS",
    "NUMBER_OF_PROCESSORS",
    "OMP_NUM_THREADS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TZ",
    "WINDIR",
)
CHECK_TIMEOUT_S = 180.0


class SandboxLimits(BaseModel):
    memory_mb: int | None = Field(default=None, ge=256, description="RAM máxima del proceso")
    cpu_seconds: int | None = Field(default=None, ge=1, description="Tiempo de CPU máximo")
    wall_seconds: float | None = Field(default=None, gt=0, description="Tiempo real máximo")


def default_limits(max_time_s: float | None = None) -> SandboxLimits:
    """RAM: 75 % de la física; CPU: proporcional al tiempo real pedido (todos los núcleos)."""
    memory = int(psutil.virtual_memory().total / 2**20 * 0.75)
    cpu = int(max_time_s * (os.cpu_count() or 1) * 1.5) if max_time_s else None
    wall = max_time_s * 1.5 + 120 if max_time_s else None
    return SandboxLimits(memory_mb=memory, cpu_seconds=cpu, wall_seconds=wall)


def python_args(module: str, *args: str) -> list[str]:
    # -I: aislado; -u: stdout sin buffer (eventos JSONL); -X utf8: sin depender del locale.
    return [sys.executable, "-I", "-u", "-X", "utf8", "-m", module, *args]


def sandbox_env(work_dir: Path) -> dict[str, str]:
    tmp = work_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
    env.update(
        HOME=str(work_dir),
        USERPROFILE=str(work_dir),
        TMP=str(tmp),
        TEMP=str(tmp),
        TMPDIR=str(tmp),
        PERCEPTRON_OFFLINE="1",
        HF_HUB_OFFLINE="1",
    )
    return env


class CodeCheck(BaseModel):
    """Resultado de construir el modelo de código y probarlo con un batch sintético."""

    ok: bool
    num_params: int | None = None
    trainable_params: int | None = None
    output_shape: list[int] | None = None
    error: str | None = None
    violations: list[str] = Field(default_factory=list)


def run_json(
    module: str, payload: dict[str, Any], work_dir: Path, *, timeout: float = CHECK_TIMEOUT_S
) -> dict[str, Any]:
    """Corre `python -m <module> <payload.json>` en el sandbox y devuelve su última línea JSON."""
    work_dir.mkdir(parents=True, exist_ok=True)
    request = work_dir / "request.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    try:
        proc = subprocess.run(  # noqa: S603 - intérprete actual, módulo propio, sin shell
            python_args(module, str(request)),
            cwd=work_dir,
            env=sandbox_env(work_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(f"el sandbox superó el tiempo máximo ({timeout:.0f} s)") from exc
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        tail = (proc.stderr or "").strip()[-2000:]
        raise ValidationError(
            f"el sandbox terminó sin respuesta (código {proc.returncode})", details={"log": tail}
        )
    result: dict[str, Any] = json.loads(lines[-1])
    return result


def check_code(spec: dict[str, Any], source: str, work_dir: Path) -> CodeCheck:
    """Fase dinámica de validación (ADR-0025): construir + forward, dentro del sandbox."""
    return CodeCheck.model_validate(
        run_json("perceptron.sandbox.check", {"spec": spec, "source": source}, work_dir)
    )


def evaluate_in_sandbox(run_dir: Path, dataset_dir: Path, *, timeout: float = 1800.0) -> None:
    """Evalúa un run de código experto en el sandbox (escribe el reporte en el run)."""
    out = run_json(
        "perceptron.sandbox.evaluate",
        {"run_dir": str(run_dir), "dataset_dir": str(dataset_dir)},
        run_dir / "sandbox-eval",
        timeout=timeout,
    )
    if not out.get("ok"):
        raise ValidationError(
            f"la evaluación en el sandbox falló: {out.get('error')}",
            details={"violations": out.get("violations", [])},
        )


def export_in_sandbox(
    run_dir: Path, dataset_dir: Path, request: dict[str, Any], *, timeout: float = 1800.0
) -> None:
    """Exporta un run de código experto en el sandbox (ADR-0027): el export ejecuta el modelo."""
    out = run_json(
        "perceptron.sandbox.export",
        {"run_dir": str(run_dir), "dataset_dir": str(dataset_dir), "request": request},
        run_dir / "sandbox-export",
        timeout=timeout,
    )
    if not out.get("ok"):
        raise ValidationError(
            f"el export en el sandbox falló: {out.get('error')}",
            details={"violations": out.get("violations", [])},
        )
