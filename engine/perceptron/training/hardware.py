"""Detección de hardware y recomendación de dispositivo (RF-TRN-01, RF-TRN-02).

Funciona aunque torch no esté instalado (reporta solo CPU/RAM/disco). La
instalación de la variante de wheel recomendada en el runtime embebido es Capa 3.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, Field

from perceptron.domain.enums import Device

logger = logging.getLogger(__name__)

GiB = 1024**3


class GPUInfo(BaseModel):
    index: int
    backend: Device
    name: str
    vram_total_gb: float
    vram_free_gb: float | None = None
    compute_capability: str | None = None
    driver: str | None = None
    supports_bf16: bool = False


class CPUInfo(BaseModel):
    model: str
    physical_cores: int | None
    logical_cores: int
    arch: str
    flags: list[str] = Field(
        default_factory=list, description="Instrucciones relevantes (avx2, avx512…)"
    )


class TorchInfo(BaseModel):
    installed: bool
    version: str | None = None
    variant: str | None = Field(default=None, description="cpu | cuXYZ | rocmX.Y | xpu")


class HardwareReport(BaseModel):
    os: str
    python: str
    cpu: CPUInfo
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: list[GPUInfo] = Field(default_factory=list)
    torch: TorchInfo
    recommended_device: Device
    recommended_torch_variant: str
    notes: list[str] = Field(default_factory=list)


def _cpu_flags() -> list[str]:
    interesting = ("avx", "avx2", "avx512f", "fma", "sse4_2", "amx_bf16")
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if line.startswith("flags"):
            present = set(line.split(":", 1)[1].split())
            return [f for f in interesting if f in present]
    return []


def _cpu_info() -> CPUInfo:
    return CPUInfo(
        model=platform.processor() or platform.machine(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        arch=platform.machine(),
        flags=_cpu_flags(),
    )


def _torch_variant(torch: Any) -> str:
    if getattr(torch.version, "hip", None):
        return f"rocm{'.'.join(str(torch.version.hip).split('.')[:2])}"
    if getattr(torch.version, "cuda", None):
        return "cu" + str(torch.version.cuda).replace(".", "")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _nvidia_driver() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - ruta resuelta con which, sin shell
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (
        out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    )


def _torch_gpus(torch: Any) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    if torch.cuda.is_available():
        backend = Device.ROCM if getattr(torch.version, "hip", None) else Device.CUDA
        driver = _nvidia_driver() if backend is Device.CUDA else None
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free: float | None = None
            try:
                free_b, _ = torch.cuda.mem_get_info(i)
                free = round(free_b / GiB, 2)
            except RuntimeError:
                pass
            gpus.append(
                GPUInfo(
                    index=i,
                    backend=backend,
                    name=props.name,
                    vram_total_gb=round(props.total_memory / GiB, 2),
                    vram_free_gb=free,
                    compute_capability=f"{props.major}.{props.minor}",
                    driver=driver,
                    supports_bf16=bool(torch.cuda.is_bf16_supported()),
                )
            )
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            gpus.append(
                GPUInfo(
                    index=i,
                    backend=Device.XPU,
                    name=props.name,
                    vram_total_gb=round(props.total_memory / GiB, 2),
                    supports_bf16=True,
                )
            )
    return gpus


def recommend_torch_variant(os_name: str, gpus: list[GPUInfo], nvidia_driver: str | None) -> str:
    """Variante de wheel de PyTorch sugerida para este hardware (RF-TRN-02)."""
    if any(g.backend is Device.ROCM for g in gpus) and os_name == "Linux":
        return "rocm"
    if any(g.backend is Device.CUDA for g in gpus) or nvidia_driver:
        return "cuda"
    if any(g.backend is Device.XPU for g in gpus):
        return "xpu"
    return "cpu"


def recommend_device(gpus: list[GPUInfo]) -> Device:
    """GPU con más VRAM si existe; si no, CPU."""
    if not gpus:
        return Device.CPU
    return max(gpus, key=lambda g: g.vram_total_gb).backend


def detect_hardware(workspace_dir: Path | None = None) -> HardwareReport:
    notes: list[str] = []
    try:
        import torch

        torch_info = TorchInfo(
            installed=True, version=torch.__version__, variant=_torch_variant(torch)
        )
        gpus = _torch_gpus(torch)
    except ImportError:
        torch_info = TorchInfo(installed=False)
        gpus = []
        notes.append("PyTorch no está instalado: solo se reporta CPU.")

    driver = _nvidia_driver()
    if driver and not any(g.backend is Device.CUDA for g in gpus):
        notes.append(f"Hay una GPU NVIDIA (driver {driver}) pero el PyTorch instalado no usa CUDA.")

    disk_path = workspace_dir if workspace_dir and workspace_dir.exists() else Path.home()
    vm = psutil.virtual_memory()
    os_name = platform.system()
    return HardwareReport(
        os=f"{os_name} {platform.release()}",
        python=platform.python_version(),
        cpu=_cpu_info(),
        ram_total_gb=round(vm.total / GiB, 2),
        ram_available_gb=round(vm.available / GiB, 2),
        disk_free_gb=round(shutil.disk_usage(disk_path).free / GiB, 2),
        gpus=gpus,
        torch=torch_info,
        recommended_device=recommend_device(gpus),
        recommended_torch_variant=recommend_torch_variant(os_name, gpus, driver),
        notes=notes,
    )
