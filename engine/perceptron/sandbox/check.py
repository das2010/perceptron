"""Fase dinámica de validación del código experto (ADR-0025), dentro del sandbox.

Uso: `python -I -m perceptron.sandbox.check <request.json>` con `{"spec": ..., "source": ...}`.
Construye el modelo, hace un forward sobre un batch sintético e imprime una línea JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    request = Path(argv[0])
    work = request.parent
    payload = json.loads(request.read_text(encoding="utf-8"))
    # Todo lo pesado se importa antes de las guardas; después solo corre el código del usuario.
    import torch  # noqa: F401

    from perceptron.archspec.builder import build_model
    from perceptron.archspec.schema import ArchSpec
    from perceptron.sandbox import code
    from perceptron.sandbox.guard import Policy, install, runtime_roots

    spec = ArchSpec.model_validate(payload["spec"])
    policy = Policy(write_roots=[work], read_roots=runtime_roots(), cpu_seconds=120)
    install(policy)
    try:
        code.load(payload["source"])
        built = build_model(spec, pretrained_allowed=False)
        out = {
            "ok": True,
            "num_params": built.num_params,
            "trainable_params": built.trainable_params,
            "output_shape": list(built.output.shape),
        }
    except Exception as e:  # el reporte es la salida: cualquier error del código del usuario
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
    out["violations"] = policy.violations
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
