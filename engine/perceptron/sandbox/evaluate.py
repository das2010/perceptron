"""Evaluación en el test sellado de un run de código experto, dentro del sandbox (ADR-0025).

Uso: `python -I -m perceptron.sandbox.evaluate <request.json>` con `{"run_dir", "dataset_dir"}`.
Escribe el reporte en el run (como `evaluate_run`) e imprime una línea JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    payload = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    run_dir, dataset_dir = Path(payload["run_dir"]), Path(payload["dataset_dir"])
    import torch  # noqa: F401

    from perceptron.evaluation.evaluate import evaluate_run
    from perceptron.sandbox import code
    from perceptron.sandbox.guard import Policy, install, runtime_roots

    source = (run_dir / code.CODE_FILE).read_text(encoding="utf-8")
    policy = Policy(write_roots=[run_dir], read_roots=[dataset_dir, *runtime_roots()])
    install(policy)
    try:
        code.load(source)
        evaluate_run(run_dir, dataset_dir)
        out: dict[str, object] = {"ok": True}
    except Exception as e:  # se informa al Engine
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
    out["violations"] = policy.violations
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
