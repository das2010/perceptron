"""Paquete de serving de un modelo exportado (RF-EXP-03; ADR-0027).

Un zip listo para `docker build`: servidor FastAPI (ONNX Runtime) con el pipeline
embebido, Dockerfiles CPU y CUDA, `requirements.txt` con las versiones exactas del Engine y
el código mínimo del Engine necesario para preprocesar (`vendor/perceptron`).
"""

from __future__ import annotations

import re
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import perceptron
from perceptron.core.errors import ValidationError
from perceptron.domain.models import utcnow
from perceptron.export.formats import EXPORT_DIR, REPORT_FILE, ExportReport

TEMPLATES = Path(__file__).parent / "templates"
BUNDLE_FILE = "serving-bundle.zip"
SUPPORTED_KINDS = ("tabular", "image")

# Módulos del Engine que usa el runtime de inferencia (sin torch, Lightning ni el resto).
VENDORED = (
    "__init__.py",
    "core/__init__.py",
    "core/errors.py",
    "domain/__init__.py",
    "domain/enums.py",
    "data/__init__.py",
    "data/schema.py",
    "data/series.py",
    "data/splits.py",
    "data/pipeline/__init__.py",
    "data/pipeline/pipeline.py",
    "data/pipeline/steps.py",
    "serving/__init__.py",
    "serving/runtime.py",
)
REQUIREMENTS = (
    "numpy",
    "polars",
    "pydantic",
    "scikit-learn",
    "onnxruntime",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "prometheus-client",
    "pillow",
)


def _version(pkg: str) -> str:
    return metadata.version(pkg)


def requirements() -> str:
    return "".join(f"{pkg}=={_version(pkg)}\n" for pkg in REQUIREMENTS)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "modelo"


def build_bundle(run_dir: Path, name: str) -> Path:
    """Arma `<run>/export/serving-bundle.zip` a partir del export ONNX verificado."""
    export_dir = run_dir / EXPORT_DIR
    report_path = export_dir / REPORT_FILE
    if not report_path.is_file():
        raise ValidationError("exportá el modelo a ONNX antes de generar el servidor")
    report = ExportReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    onnx = next((a for a in report.artifacts if a.format == "onnx"), None)
    if onnx is None or onnx.error or not (onnx.verification and onnx.verification.passed):
        raise ValidationError("el servidor necesita un export ONNX verificado")
    kind = str(report.signature["inputs"]["kind"])
    if kind not in SUPPORTED_KINDS:
        raise ValidationError(f"el servidor de inferencia admite {SUPPORTED_KINDS}; no {kind}")

    needs_torch = kind == "image"  # las transformaciones de evaluación usan torchvision
    context: dict[str, Any] = {
        "name": name,
        "slug": _slug(name),
        "version": perceptron.__version__,
        "model_version": report.signature.get("archspec_hash", "")[:12],
        "description": f"Modelo {report.signature['outputs']['task']} sobre {kind}",
        "created": utcnow().date().isoformat(),
        "run_id": report.run_id,
        "kind": kind,
        "task": report.signature["outputs"]["task"],
        "classes": report.signature["outputs"].get("classes") or [],
        "columns": report.signature["inputs"].get("columns") or [],
        "max_diff": f"{onnx.verification.max_abs_diff:.2e}",
        "needs_torch": needs_torch,
        "torch_version": _version("torch").split("+")[0] if needs_torch else "",
        "torchvision_version": _version("torchvision").split("+")[0] if needs_torch else "",
    }
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,  # noqa: S701 - genera Python, Dockerfiles y Markdown, no HTML
    )

    def render(template: str) -> str:
        return env.get_template(template).render(**context)

    package_root = Path(perceptron.__file__).parent
    out = export_dir / BUNDLE_FILE
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("app/__init__.py", "")
        z.writestr("app/app.py", render("app.py.j2"))
        z.writestr("Dockerfile.cpu", render("Dockerfile.cpu.j2"))
        z.writestr("Dockerfile.cuda", render("Dockerfile.cuda.j2"))
        z.writestr("README.md", render("README.md.j2"))
        z.writestr("requirements.txt", requirements())
        for file in ("model.onnx", "pipeline.json", "signature.json"):
            z.write(export_dir / file, f"app/model/{file}")
        for rel in VENDORED:
            z.write(package_root / rel, f"vendor/perceptron/{rel}")
    return out
