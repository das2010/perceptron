"""Validación de ArchSpec en las 6 etapas de SPEC §9.3.

1. JSON Schema / Pydantic
2. Bloques existentes y permitidos para la modalidad/tarea; parámetros en rango
3. Grafo acíclico, conexo, una sola salida
4. Inferencia de shapes (forward en `meta`)
5. Parámetros y memoria estimada vs. dispositivo
6. Disponibilidad y licencia de pesos preentrenados

Cada problema lleva ruta JSON y mensaje legible (se reenvía al LLM en Capa 2).
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import pydantic
from pydantic import BaseModel, Field

from perceptron.archspec.builder import ArchBuildError, build_model, topological_order
from perceptron.archspec.schema import ArchSpec, Scalar, resolve
from perceptron.catalog.registry import BLOCKS, TIMM_WEIGHTS

BYTES_PER_FLOAT = 4
# pesos + gradientes + 2 momentos de Adam
PARAM_MEMORY_FACTOR = 4
# activaciones guardadas para backward (aprox.)
ACTIVATION_FACTOR = 2


class Stage(StrEnum):
    SCHEMA = "schema"
    BLOCKS = "blocks"
    GRAPH = "graph"
    SHAPES = "shapes"
    RESOURCES = "resources"
    WEIGHTS = "weights"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Issue(BaseModel):
    stage: Stage
    severity: Severity
    path: str
    message: str
    suggestion: str | None = None


class ValidationReport(BaseModel):
    valid: bool
    issues: list[Issue] = Field(default_factory=list)
    num_params: int | None = None
    trainable_params: int | None = None
    estimated_memory_mb: float | None = None
    output_shape: list[int] | None = None
    spec: ArchSpec | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    def feedback(self) -> str:
        """Texto compacto para reenviar al LLM en el reintento (RF-LLM-04)."""
        return "\n".join(f"[{i.stage.value}] {i.path}: {i.message}" for i in self.errors)


def offline_mode() -> bool:
    return os.environ.get("PERCEPTRON_OFFLINE", "").lower() in {
        "1",
        "true",
        "yes",
    } or os.environ.get("HF_HUB_OFFLINE", "") in {"1", "true"}


def weights_cached(model: str, tag: str, cache_dir: Path | None = None) -> bool:
    """¿Están los pesos en la caché de Hugging Face (timm los publica ahí)?"""
    root = cache_dir or Path(
        os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
    )
    return (root / f"models--timm--{model}.{tag}").is_dir()


def hf_cached(repo: str, cache_dir: Path | None = None) -> bool:
    default = Path.home() / ".cache" / "huggingface" / "hub"
    root = cache_dir or Path(os.environ.get("HF_HUB_CACHE", default))
    return (root / f"models--{repo.replace('/', '--')}").is_dir()


def _hf_text_issues(
    i: int, params: dict[str, Any], commercial: bool, cache_dir: Path | None
) -> list[Issue]:
    from perceptron.catalog.registry import HF_TEXT_MODELS

    model = params.get("model", "")
    info = HF_TEXT_MODELS.get(model)
    if info is None:
        return []
    if commercial and not info.commercial_ok:
        return [
            Issue(
                stage=Stage.WEIGHTS,
                severity=Severity.ERROR,
                path=f"nodes[{i}].params.model",
                message=f"{model} ({info.license}) no permite uso comercial",
            )
        ]
    if params.get("pretrained", True) and offline_mode() and not hf_cached(model, cache_dir):
        return [
            Issue(
                stage=Stage.WEIGHTS,
                severity=Severity.WARNING,
                path=f"nodes[{i}].params.model",
                message=f"sin conexión y {model} no está en la caché de Hugging Face",
                suggestion="usar una plantilla sin preentrenado (TextCNN) o descargar el modelo",
            )
        ]
    return []


def validate_archspec(
    data: dict[str, Any] | ArchSpec,
    *,
    overrides: dict[str, Scalar] | None = None,
    batch_size: int = 32,
    device_memory_gb: float | None = None,
    commercial_use: bool = True,
    cache_dir: Path | None = None,
) -> ValidationReport:
    issues: list[Issue] = []

    # 1. schema
    if isinstance(data, ArchSpec):
        spec = data
    else:
        try:
            spec = ArchSpec.model_validate(data)
        except pydantic.ValidationError as e:
            for err in e.errors():
                path = ".".join(str(p) for p in err["loc"]) or "$"
                issues.append(
                    Issue(
                        stage=Stage.SCHEMA, severity=Severity.ERROR, path=path, message=err["msg"]
                    )
                )
            return ValidationReport(valid=False, issues=issues)

    # 2. bloques y modalidad/tarea (los rangos de parámetros se validan al construir)
    for i, node in enumerate(spec.nodes):
        block = BLOCKS.get(node.block)
        if block is None:
            issues.append(
                Issue(
                    stage=Stage.BLOCKS,
                    severity=Severity.ERROR,
                    path=f"nodes[{i}].block",
                    message=f"bloque desconocido '{node.block}'",
                    suggestion=f"bloques disponibles: {', '.join(sorted(BLOCKS))}",
                )
            )
            continue
        if block.modalities and spec.modality not in block.modalities:
            issues.append(
                Issue(
                    stage=Stage.BLOCKS,
                    severity=Severity.ERROR,
                    path=f"nodes[{i}].block",
                    message=f"{block.key} no se permite para la modalidad {spec.modality.value}",
                )
            )
        if block.tasks and spec.task.type not in block.tasks:
            issues.append(
                Issue(
                    stage=Stage.BLOCKS,
                    severity=Severity.ERROR,
                    path=f"nodes[{i}].block",
                    message=f"{block.key} no se permite para la tarea {spec.task.type.value}",
                )
            )
    if issues:
        return ValidationReport(valid=False, issues=issues, spec=spec)

    # 3. grafo
    try:
        topological_order(spec)
    except ArchBuildError as e:
        issues.append(
            Issue(stage=Stage.GRAPH, severity=Severity.ERROR, path=e.path, message=e.message)
        )
        return ValidationReport(valid=False, issues=issues, spec=spec)

    # 4. shapes (y parámetros en rango)
    try:
        built = build_model(spec, overrides, pretrained_allowed=False, materialize=False)
    except ArchBuildError as e:
        stage = Stage.BLOCKS if ".params" in e.path else Stage.SHAPES
        issues.append(Issue(stage=stage, severity=Severity.ERROR, path=e.path, message=e.message))
        return ValidationReport(valid=False, issues=issues, spec=spec)

    # 5. recursos
    param_bytes = built.num_params * BYTES_PER_FLOAT * PARAM_MEMORY_FACTOR
    act_bytes = built.activations_per_sample * batch_size * BYTES_PER_FLOAT * ACTIVATION_FACTOR
    memory_mb = round((param_bytes + act_bytes) / 2**20, 1)
    if device_memory_gb is not None and memory_mb > device_memory_gb * 1024 * 0.9:
        issues.append(
            Issue(
                stage=Stage.RESOURCES,
                severity=Severity.ERROR,
                path="training.batch_size",
                message=(
                    f"memoria estimada {memory_mb} MB supera la disponible ({device_memory_gb} GB)"
                ),
                suggestion="reducí batch_size, la resolución o usá un backbone más chico",
            )
        )

    # 6. pesos preentrenados
    for i, node in enumerate(spec.nodes):
        if node.block == "text.hf_encoder":
            issues += _hf_text_issues(i, resolve(node.params, overrides), commercial_use, cache_dir)
            continue
        if node.block != "vision.timm_backbone":
            continue
        params = resolve(node.params, overrides)
        if not params.get("pretrained", True):
            continue
        model = params.get("model", "efficientnet_b0")
        info = TIMM_WEIGHTS.get(model)
        if info is None:
            continue
        if commercial_use and not info.commercial_ok:
            issues.append(
                Issue(
                    stage=Stage.WEIGHTS,
                    severity=Severity.ERROR,
                    path=f"nodes[{i}].params.model",
                    message=f"los pesos de {model} ({info.license}) no permiten uso comercial",
                )
            )
        elif offline_mode() and not weights_cached(model, info.pretrained_tag, cache_dir):
            issues.append(
                Issue(
                    stage=Stage.WEIGHTS,
                    severity=Severity.WARNING,
                    path=f"nodes[{i}].params.pretrained",
                    message=f"sin conexión y los pesos de {model} no están en la caché",
                    suggestion="usar pretrained: false o descargar los pesos antes",
                )
            )

    return ValidationReport(
        valid=not any(i.severity is Severity.ERROR for i in issues),
        issues=issues,
        num_params=built.num_params,
        trainable_params=built.trainable_params,
        estimated_memory_mb=memory_mb,
        output_shape=list(built.output.shape),
        spec=spec,
    )
