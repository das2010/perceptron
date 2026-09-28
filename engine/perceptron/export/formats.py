"""Export de un modelo entrenado (RF-EXP-01, RF-EXP-05; ADR-0027).

Formatos: ONNX (verificado contra PyTorch en ONNX Runtime; opcional fp16 e INT8 dinámico),
`torch.export` (ExportedProgram) y TorchScript (legacy). Cada artefacto lleva su
verificación numérica sobre un batch real de validación y la firma del modelo.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from perceptron.data.view import DatasetView
from perceptron.domain.models import utcnow
from perceptron.evaluation.evaluate import signature
from perceptron.training.data import make_dataset
from perceptron.training.inference import TrainedModel, load_trained

EXPORT_DIR = "export"
REPORT_FILE = "export.json"
ONNX_OPSET = 18
TOL_FP32 = 1e-4
TOL_FP16 = 1e-2
SAMPLE_BATCH = 16


class ExportFormat(StrEnum):
    ONNX = "onnx"
    TORCH_EXPORT = "torch_export"
    TORCHSCRIPT = "torchscript"


class ExportRequest(BaseModel):
    formats: list[ExportFormat] = Field(default_factory=lambda: [ExportFormat.ONNX])
    fp16: bool = Field(default=False, description="ONNX en fp16 (verificado con tolerancia 1e-2)")
    int8: bool = Field(default=False, description="ONNX con cuantización dinámica INT8 (CPU)")


class Verification(BaseModel):
    samples: int
    max_abs_diff: float
    tolerance: float | None = Field(description="None: solo se informa (INT8)")
    passed: bool
    argmax_agreement: float | None = None


class ExportArtifact(BaseModel):
    format: str
    file: str
    size_bytes: int
    sha256: str
    legacy: bool = False
    verification: Verification | None = None
    error: str | None = None


class ExportReport(BaseModel):
    run_id: str
    created_at: str
    signature: dict[str, Any]
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    artifacts: list[ExportArtifact]

    @property
    def ok(self) -> bool:
        return all(
            a.error is None and (a.verification is None or a.verification.passed)
            for a in self.artifacts
        )


def _version() -> str:
    try:
        return metadata.version("perceptron-engine")
    except metadata.PackageNotFoundError:  # pragma: no cover - checkout sin instalar
        return "dev"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_numpy(out: Any) -> np.ndarray:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, torch.Tensor):
        return out.detach().float().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


def _verify(ref: np.ndarray, got: np.ndarray, tolerance: float | None) -> Verification:
    diff = float(np.max(np.abs(ref - got))) if ref.size else 0.0
    agreement = None
    if ref.ndim == 2 and ref.shape[1] > 1:
        agreement = float(np.mean(ref.argmax(1) == got.argmax(1)))
    return Verification(
        samples=int(ref.shape[0]) if ref.ndim else 1,
        max_abs_diff=diff,
        tolerance=tolerance,
        passed=tolerance is None or diff <= tolerance,
        argmax_agreement=agreement,
    )


class _Tabular(torch.nn.Module):
    """Firma explícita para los exportadores (el `forward(*inputs)` del grafo no se traza bien)."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.inner(x_num, x_cat)
        return out


class _Single(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.inner(x)
        return out


def _explicit(model: torch.nn.Module, n_inputs: int) -> torch.nn.Module:
    if n_inputs == 2:
        return _Tabular(model).eval()
    if n_inputs == 1:
        return _Single(model).eval()
    raise ValueError(f"export con {n_inputs} entradas no soportado")


def _batch_dynamic(n_inputs: int) -> tuple[dict[int, Any], ...]:
    dim = getattr(torch.export.Dim, "DYNAMIC", None) or torch.export.Dim("batch")
    return tuple({0: dim} for _ in range(n_inputs))


def sample_inputs(
    trained: TrainedModel, dataset_dir: Path, n: int = SAMPLE_BATCH
) -> tuple[torch.Tensor, ...]:
    """Un batch real de validación (sin el target) para trazar y verificar."""
    ds = make_dataset(DatasetView(dataset_dir), trained.pipeline, "val", train=False)
    loader = DataLoader(ds, batch_size=n, shuffle=False, collate_fn=getattr(ds, "collate_fn", None))
    *inputs, _ = next(iter(loader))
    return tuple(t for t in inputs if isinstance(t, torch.Tensor))


def _input_names(trained: TrainedModel, count: int) -> list[str]:
    if trained.spec.input.kind == "tabular" and count == 2:
        return ["x_num", "x_cat"]
    return ["x"] if count == 1 else [f"x{i}" for i in range(count)]


def _io_schema(names: list[str], tensors: tuple[torch.Tensor, ...]) -> list[dict[str, Any]]:
    return [
        {"name": n, "dtype": str(t.dtype).removeprefix("torch."), "shape": [-1, *t.shape[1:]]}
        for n, t in zip(names, tensors, strict=True)
    ]


def _artifact(path: Path, fmt: str, verification: Verification | None, **kw: Any) -> ExportArtifact:
    return ExportArtifact(
        format=fmt,
        file=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        verification=verification,
        **kw,
    )


def _failed(fmt: str, exc: BaseException) -> ExportArtifact:
    detail = f"{type(exc).__name__}: {exc}"[:1500]
    return ExportArtifact(format=fmt, file="", size_bytes=0, sha256="", error=detail)


def _onnx(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    names: list[str],
    ref: np.ndarray,
    out_dir: Path,
    request: ExportRequest,
) -> list[ExportArtifact]:
    import onnxruntime as ort

    path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=names,
        output_names=["output"],
        dynamic_shapes=_batch_dynamic(len(inputs)),
        opset_version=ONNX_OPSET,
        # Un solo archivo (sin `model.onnx.data`): el paquete de serving y las conversiones
        # fp16/INT8 lo copian y cargan entero. Los modelos del catálogo quedan lejos de 2 GB.
        external_data=False,
    )

    def run(p: Path, feed_dtype: Any = None) -> np.ndarray:
        session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        feed = {
            n: (t.numpy().astype(feed_dtype) if feed_dtype and t.is_floating_point() else t.numpy())
            for n, t in zip(names, inputs, strict=True)
        }
        return np.asarray(session.run(None, feed)[0], dtype=np.float32)

    artifacts = [_artifact(path, "onnx", _verify(ref, run(path), TOL_FP32))]
    if request.fp16:
        try:
            import onnx
            from onnxconverter_common import float16

            half = out_dir / "model.fp16.onnx"
            onnx.save(
                float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True), half
            )
            artifacts.append(_artifact(half, "onnx_fp16", _verify(ref, run(half), TOL_FP16)))
        except Exception as e:  # se informa por artefacto
            artifacts.append(_failed("onnx_fp16", e))
    if request.int8:
        try:
            import onnx
            from onnxruntime.quantization import QuantType, quantize_dynamic

            # El exportador dynamo deja anotaciones de formas (value_info) que la inferencia
            # de ONNX contradice al cuantizar: se borran y el cuantizador las vuelve a inferir.
            pre = out_dir / "model.int8-pre.onnx"
            graph = onnx.load(str(path))
            del graph.graph.value_info[:]
            onnx.save(graph, str(pre))
            q = out_dir / "model.int8.onnx"
            quantize_dynamic(str(pre), str(q), weight_type=QuantType.QInt8)
            pre.unlink(missing_ok=True)
            artifacts.append(_artifact(q, "onnx_int8", _verify(ref, run(q), None)))
        except Exception as e:
            artifacts.append(_failed("onnx_int8", e))
    return artifacts


def _torch_export(
    model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], ref: np.ndarray, out_dir: Path
) -> ExportArtifact:
    program = torch.export.export(model, inputs, dynamic_shapes=_batch_dynamic(len(inputs)))
    path = out_dir / "model.pt2"
    # Por archivo abierto desde Python: el escritor C++ de PyTorch no resuelve en Windows las
    # rutas con acentos (workspaces en OneDrive corporativo, §13.5).
    with path.open("wb") as f:
        torch.export.save(program, f)
    with path.open("rb") as f:
        loaded = torch.export.load(f)
    got = _as_numpy(loaded.module()(*inputs))
    return _artifact(path, "torch_export", _verify(ref, got, TOL_FP32))


def _torchscript(
    model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], ref: np.ndarray, out_dir: Path
) -> ExportArtifact:
    traced = torch.jit.trace(model, inputs, check_trace=False)  # type: ignore[no-untyped-call]
    path = out_dir / "model.torchscript.pt"
    with path.open("wb") as f:
        torch.jit.save(traced, f)
    with path.open("rb") as f:
        got = _as_numpy(torch.jit.load(f)(*inputs))  # type: ignore[no-untyped-call]
    return _artifact(path, "torchscript", _verify(ref, got, TOL_FP32), legacy=True)


def export_run(run_dir: Path, dataset_dir: Path, request: ExportRequest) -> ExportReport:
    """Exporta el modelo del run en los formatos pedidos y verifica cada uno."""
    trained = load_trained(run_dir)
    inputs = sample_inputs(trained, dataset_dir)
    model = _explicit(trained.model.eval(), len(inputs))
    names = _input_names(trained, len(inputs))
    with torch.no_grad():
        ref = _as_numpy(model(*inputs))
    out_dir = run_dir / EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[ExportArtifact] = []
    for fmt in dict.fromkeys(request.formats):
        try:
            with torch.no_grad():
                if fmt is ExportFormat.ONNX:
                    artifacts += _onnx(model, inputs, names, ref, out_dir, request)
                elif fmt is ExportFormat.TORCH_EXPORT:
                    artifacts.append(_torch_export(model, inputs, ref, out_dir))
                else:
                    artifacts.append(_torchscript(model, inputs, ref, out_dir))
        except Exception as e:  # un formato que falla no impide los demás
            artifacts.append(_failed(fmt.value, e))
            traceback.print_exc()

    # Firma (RF-EXP-05) y pipeline junto a los artefactos: todo lo que necesita la inferencia.
    run_id = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_id"]
    sig = {
        **signature(trained.spec, trained.pipeline),
        "run_id": run_id,
        "perceptron_version": _version(),
    }
    (out_dir / "pipeline.json").write_text(trained.pipeline.model_dump_json(indent=2), "utf-8")
    report = ExportReport(
        run_id=run_id,
        created_at=utcnow().isoformat(),
        signature=sig,
        inputs=_io_schema(names, inputs),
        outputs=[{"name": "output", "dtype": "float32", "shape": [-1, *ref.shape[1:]]}],
        artifacts=artifacts,
    )
    (out_dir / "signature.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    (out_dir / REPORT_FILE).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
