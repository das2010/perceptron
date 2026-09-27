"""Estimaciones del sistema para cada propuesta (RF-ARC-01): no las inventa el LLM.

El tiempo por época se mide con un micro-benchmark forward + backward sobre un batch
sintético en el dispositivo elegido, y se escala al número de batches de train.
"""

from __future__ import annotations

import logging
import math
import time

from perceptron.archspec.schema import ArchSpec

logger = logging.getLogger(__name__)

BENCH_BATCH = 16
BENCH_STEPS = 3


def estimate_epoch_time(
    spec: ArchSpec, n_train: int, *, batch_size: int = 32, device: str = "cpu"
) -> float | None:
    """Segundos por época estimados; None si no se pudo medir (se informa sin estimación)."""
    try:
        import torch

        from perceptron.archspec.builder import _dummy, build_model, input_tensor_spec

        dev = torch.device(device)
        built = build_model(spec, pretrained_allowed=False)
        model = built.model.to(dev).train()
        inputs = _dummy(input_tensor_spec(spec), dev, batch=BENCH_BATCH)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=1e-3) if params else None

        def step() -> None:
            out = model(*inputs)
            loss = out.float().mean()
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

        step()  # calentamiento
        start = time.perf_counter()
        for _ in range(BENCH_STEPS):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        per_sample = (time.perf_counter() - start) / (BENCH_STEPS * BENCH_BATCH)
    except Exception:
        logger.warning("no se pudo estimar el tiempo por época", exc_info=True)
        return None
    batches = math.ceil(max(n_train, 1) / batch_size)
    return round(per_sample * batches * batch_size, 2)
