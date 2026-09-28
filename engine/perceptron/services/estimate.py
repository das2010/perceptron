"""Estimaciones del sistema para cada propuesta (RF-ARC-01): no las inventa el LLM.

El tiempo por época se mide con un micro-benchmark forward + backward sobre un batch
sintético en el dispositivo elegido, y se escala al número de batches de train.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from pydantic import BaseModel, Field

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


# ------------------------------------------------------ costo por dispositivo (RF-PRF-08)


class DeviceEstimate(BaseModel):
    device: str
    name: str
    memory_gb: float | None = None
    fits: bool | None = None
    epoch_time_s: float | None = None


class CostEstimate(BaseModel):
    n_train: int
    num_samples: int
    size_bytes: int
    input_shape: list[int] | None
    sample_mb: float | None = Field(default=None, description="Tensor de entrada por ejemplo")
    train_tensor_mb: float | None = Field(
        default=None, description="Tamaño efectivo de train como tensores (lo que ve el modelo)"
    )
    batch_size: int
    num_params: int | None
    memory_mb: float | None = Field(default=None, description="Memoria estimada por batch")
    devices: list[DeviceEstimate]


def estimate_cost(spec: ArchSpec, dv: Any, hardware: Any, *, batch_size: int) -> CostEstimate:
    """Tamaño efectivo, memoria por batch y tiempo por época en cada dispositivo disponible."""
    from perceptron.archspec.validate import validate_archspec

    split = getattr(dv, "split", None)
    n_train = int(split.train) if split is not None else int(dv.num_samples)
    shape = [int(x) for x in spec.input.shape] if spec.input.shape else None
    per_sample = float(math.prod(shape)) * 4 / 2**20 if shape else None
    report = validate_archspec(spec, batch_size=batch_size)
    devices = [
        DeviceEstimate(
            device="cpu",
            name=hardware.cpu.model or "CPU",
            memory_gb=hardware.ram_available_gb,
        )
    ]
    for gpu in hardware.gpus:
        backend = gpu.backend.value if hasattr(gpu.backend, "value") else str(gpu.backend)
        devices.append(
            DeviceEstimate(
                device=backend if backend != "rocm" else "cuda",
                name=gpu.name,
                memory_gb=gpu.vram_free_gb or gpu.vram_total_gb,
            )
        )
    seen: set[str] = set()
    for d in devices:
        if report.estimated_memory_mb is not None and d.memory_gb:
            d.fits = report.estimated_memory_mb <= d.memory_gb * 1024 * 0.9
        if d.device not in seen:  # se mide una vez por tipo de dispositivo
            seen.add(d.device)
            d.epoch_time_s = estimate_epoch_time(
                spec, n_train, batch_size=batch_size, device=d.device
            )
    return CostEstimate(
        n_train=n_train,
        num_samples=int(dv.num_samples),
        size_bytes=int(dv.size_bytes),
        input_shape=shape,
        sample_mb=round(per_sample, 4) if per_sample is not None else None,
        train_tensor_mb=round(per_sample * n_train, 1) if per_sample is not None else None,
        batch_size=batch_size,
        num_params=report.num_params,
        memory_mb=report.estimated_memory_mb,
        devices=devices,
    )
