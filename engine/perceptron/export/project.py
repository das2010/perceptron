"""Proyecto de código exportable (RF-EXP-04, objetivo O5; ADR-0027).

Repositorio Python autónomo (uv) con el modelo como código PyTorch generado desde la ArchSpec,
el pipeline de Perceptron (vendorizado), `train`, `infer`, `serve`, `config.yaml`, datos de
train/val, los pesos entrenados y una prueba de humo. Reproduce el entrenamiento en otra
máquina con `uv sync && uv run <paquete>-train`.
"""

from __future__ import annotations

import io
import re
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import perceptron
from perceptron.archspec.schema import resolve
from perceptron.archspec.to_code import archspec_to_code, identifier
from perceptron.catalog.registry import CODE_BLOCK, TIMM_WEIGHTS
from perceptron.core.errors import ValidationError
from perceptron.data.view import DatasetView
from perceptron.domain.models import utcnow
from perceptron.serving.bundle import VENDORED
from perceptron.training.config import RUN_CONFIG_FILE, RunConfig
from perceptron.training.data import auto_batch_size
from perceptron.training.inference import load_run_artifacts

TEMPLATES = Path(__file__).parent / "templates" / "project"
PROJECT_FILE = "project.zip"
MAX_DATA_BYTES = 1 << 30  # 1 GiB de datos de train/val dentro del proyecto
SUPPORTED = {("tabular", "classification"), ("tabular", "regression")}
SUPPORTED |= {("image", "classification"), ("image", "regression")}
BASE_REQUIREMENTS = ("torch", "numpy", "polars", "pydantic", "scikit-learn", "pyyaml")


def _package(name: str) -> str:
    ident = re.sub(r"\W+", "_", name.lower()).strip("_") or "modelo"
    return f"m_{ident}" if ident[0].isdigit() else ident


def _pin(pkg: str) -> str:
    return f"{pkg}=={metadata.version(pkg).split('+')[0]}"


def _state_for_generated_code(ckpt: Path) -> dict[str, torch.Tensor]:
    """Pesos del checkpoint con los nombres del código generado (`blocks.<nodo>.` → `<nodo>.`)."""
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not key.startswith("model."):
            continue
        k = key.removeprefix("model.")
        if k.startswith("blocks."):
            node, _, rest = k.removeprefix("blocks.").partition(".")
            k = f"{identifier(node)}.{rest}"
        out[k] = value
    return out


def _config(spec: Any, cfg: RunConfig, n_train: int, modality: Any) -> dict[str, Any]:
    ov = cfg.overrides
    es = spec.training.early_stopping
    epochs = cfg.max_epochs or int(resolve(spec.training.epochs, ov))
    bs = cfg.batch_size or resolve(spec.training.batch_size, ov)
    return {
        "seed": cfg.seed,
        "epochs": epochs,
        "batch_size": auto_batch_size(modality, n_train) if bs in (None, "auto") else int(bs),
        "lr": float(resolve(spec.optimizer.lr, ov)),
        "weight_decay": float(resolve(spec.optimizer.weight_decay, ov)),
        "scheduler": spec.scheduler.type,
        "patience": es.patience if es else epochs,
        "loss_type": spec.loss.type,
        "class_weights": spec.loss.class_weights
        if isinstance(spec.loss.class_weights, str)
        else "none",
        "label_smoothing": float(resolve(spec.loss.label_smoothing, ov) or 0.0),
        "gradient_clip": spec.training.gradient_clip,
    }


def build_project(run_dir: Path, dataset_dir: Path, name: str) -> Path:
    """Arma `<run>/export/project.zip`."""
    spec, fitted, ckpt = load_run_artifacts(run_dir)
    if any(n.block == CODE_BLOCK for n in spec.nodes):
        raise ValidationError(
            "los modelos de código experto se exportan como código desde su fuente"
        )
    kind, task = spec.input.kind, spec.task.type.value
    if (kind, task) not in SUPPORTED:
        raise ValidationError(
            f"el proyecto exportable admite {sorted(SUPPORTED)}; no ({kind}, {task})"
        )
    cfg = RunConfig.model_validate_json((run_dir / RUN_CONFIG_FILE).read_text(encoding="utf-8"))
    view = DatasetView(dataset_dir)
    train_df, val_df = view.read("train"), view.read("val")
    image = kind == "image"

    files: list[tuple[Path, str]] = []
    if image:
        rels = sorted(set(train_df["path"].to_list()) | set(val_df["path"].to_list()))
        files = [(view.files_dir / r, f"data/files/{r}") for r in rels]
    size = (
        sum(p.stat().st_size for p, _ in files)
        + train_df.estimated_size()
        + val_df.estimated_size()
    )
    if size > MAX_DATA_BYTES:
        raise ValidationError(
            f"los datos de train/val ({size / 2**20:.0f} MB) superan el máximo de 1 GB"
        )

    package = _package(name)
    uses_timm = any(n.block == "vision.timm_backbone" for n in spec.nodes)
    weights = [
        TIMM_WEIGHTS[str(resolve(n.params.get("model"), cfg.overrides))]
        for n in spec.nodes
        if n.block == "vision.timm_backbone"
        and resolve(n.params.get("pretrained", True), cfg.overrides)
        and str(resolve(n.params.get("model"), cfg.overrides)) in TIMM_WEIGHTS
    ]
    reqs = [_pin(p) for p in BASE_REQUIREMENTS]
    if image:
        reqs += [_pin("torchvision"), _pin("pillow")]
    if uses_timm:
        reqs.append(_pin("timm"))
    target = fitted.spec.target
    context: dict[str, Any] = {
        "name": name,
        "slug": package.replace("_", "-"),
        "package": package,
        "version": perceptron.__version__,
        "run_id": cfg.run_id,
        "created": utcnow().date().isoformat(),
        "kind": kind,
        "task": task,
        "image": image,
        "target": target.name if target else "",
        "num_classes": spec.task.num_classes,
        "classes": fitted.classes or [],
        "columns": sorted({c for s in fitted.spec.steps for c in s.columns}),
        "archspec_hash": spec.content_hash(),
        "requirements": reqs,
        "weights": [w.model_dump() for w in weights],
        **_config(spec, cfg, train_df.height, view.modality),
    }
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,  # noqa: S701 - genera Python, TOML, YAML y Markdown
    )

    def render(template: str) -> str:
        return env.get_template(template).render(**context)

    buf = io.BytesIO()
    torch.save(_state_for_generated_code(ckpt), buf)
    root = context["slug"]
    out = run_dir / "export" / PROJECT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    package_root = Path(perceptron.__file__).parent
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for tpl, dest in (
            ("pyproject.toml.j2", "pyproject.toml"),
            ("README.md.j2", "README.md"),
            ("config.yaml.j2", "config.yaml"),
            ("LICENSES.md.j2", "LICENSES.md"),
            ("src/data.py.j2", f"src/{package}/data.py"),
            ("src/train.py.j2", f"src/{package}/train.py"),
            ("src/infer.py.j2", f"src/{package}/infer.py"),
            ("src/serve.py.j2", f"src/{package}/serve.py"),
            ("tests/test_smoke.py.j2", "tests/test_smoke.py"),
        ):
            z.writestr(f"{root}/{dest}", render(tpl))
        z.writestr(
            f"{root}/src/{package}/__init__.py", f'"""{name} (exportado desde Perceptron)."""\n'
        )
        z.writestr(f"{root}/src/{package}/model.py", archspec_to_code(spec, cfg.overrides))
        z.writestr(f"{root}/artifacts/model.pt", buf.getvalue())
        z.writestr(f"{root}/artifacts/pipeline.json", fitted.model_dump_json(indent=2))
        z.writestr(f"{root}/artifacts/pipeline_spec.json", fitted.spec.model_dump_json(indent=2))
        for split, df in (("train", train_df), ("val", val_df)):
            data = io.BytesIO()
            df.write_parquet(data)
            z.writestr(f"{root}/data/{split}.parquet", data.getvalue())
        for src, dest in files:
            z.write(src, f"{root}/{dest}")
        for rel in VENDORED:
            z.write(package_root / rel, f"{root}/vendor/perceptron/{rel}")
        z.writestr(f"{root}/.gitignore", ".venv/\n__pycache__/\npredicciones.csv\n")
    return out
