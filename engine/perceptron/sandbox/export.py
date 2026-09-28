"""Export de un run de código experto, dentro del sandbox (ADR-0025, ADR-0027).

Uso: `python -I -m perceptron.sandbox.export <request.json>` con `{"run_dir", "dataset_dir",
"request"}`. Escribe los artefactos y el reporte en `<run>/export` e imprime una línea JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    payload = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    run_dir, dataset_dir = Path(payload["run_dir"]), Path(payload["dataset_dir"])
    import onnxruntime  # noqa: F401 - todo lo pesado antes de las guardas
    import torch  # noqa: F401

    import perceptron.tasks  # noqa: F401
    from perceptron.export.formats import ExportRequest, export_run
    from perceptron.sandbox import code
    from perceptron.sandbox.guard import Policy, install, runtime_roots

    request = ExportRequest.model_validate(payload["request"])
    source = (run_dir / code.CODE_FILE).read_text(encoding="utf-8")
    policy = Policy(write_roots=[run_dir], read_roots=[dataset_dir, *runtime_roots()])
    install(policy)
    try:
        code.load(source)
        export_run(run_dir, dataset_dir, request)
        out: dict[str, object] = {"ok": True}
    except Exception as e:  # se informa al Engine
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
    out["violations"] = policy.violations
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
