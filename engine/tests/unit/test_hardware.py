from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from perceptron.cli.main import app
from perceptron.domain.enums import Device
from perceptron.training.hardware import (
    GPUInfo,
    detect_hardware,
    recommend_device,
    recommend_torch_variant,
)


def _gpu(backend: Device, vram: float, index: int = 0) -> GPUInfo:
    return GPUInfo(index=index, backend=backend, name="x", vram_total_gb=vram)


def test_detect_hardware_on_ci_machine(workspace_dir: Path) -> None:
    report = detect_hardware(workspace_dir)
    assert report.cpu.logical_cores >= 1
    assert report.ram_total_gb > 0
    assert report.disk_free_gb >= 0
    assert report.torch.installed  # el extra `ml` está en el grupo dev
    if not report.gpus:
        assert report.recommended_device is Device.CPU


def test_recommend_device_picks_largest_gpu() -> None:
    assert recommend_device([]) is Device.CPU
    gpus = [_gpu(Device.XPU, 8, 0), _gpu(Device.CUDA, 24, 1)]
    assert recommend_device(gpus) is Device.CUDA


def test_recommend_torch_variant() -> None:
    assert recommend_torch_variant("Windows", [], None) == "cpu"
    assert recommend_torch_variant("Windows", [], "560.1") == "cuda"  # driver sin torch CUDA
    assert recommend_torch_variant("Linux", [_gpu(Device.ROCM, 16)], None) == "rocm"
    assert recommend_torch_variant("Linux", [_gpu(Device.XPU, 16)], None) == "xpu"


def test_hardware_endpoint(client: TestClient) -> None:
    body = client.get("/api/v1/system/hardware").json()
    assert body["recommended_device"] in {"cpu", "cuda", "rocm", "xpu"}
    assert body["cpu"]["logical_cores"] >= 1


def test_hardware_cli(workspace_dir: Path) -> None:
    r = CliRunner().invoke(app, ["system", "hardware", "-w", str(workspace_dir), "--json"])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["torch"]["installed"]
    r = CliRunner().invoke(app, ["system", "hardware", "-w", str(workspace_dir)])
    assert "Recomendado" in r.stdout
