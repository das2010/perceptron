"""ArchSpec → `nn.Module` (SPEC §9.2).

La construcción se hace en dos pasadas:
1. **meta**: se construye cada bloque en el dispositivo `meta` y se propaga un
   tensor de prueba para inferir shapes, contar parámetros y estimar memoria de
   activaciones, sin reservar memoria real ni descargar pesos (§9.3, etapa 4).
2. **real**: se construyen los bloques en CPU con las shapes ya conocidas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from perceptron.archspec.schema import INPUT_NODE, ArchSpec, Scalar, resolve
from perceptron.catalog.registry import (
    SHAPE_PRESERVING,
    BlockSpec,
    BuildContext,
    TensorKind,
    TensorSpec,
    get_block,
)


class ArchBuildError(ValueError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def input_tensor_spec(spec: ArchSpec) -> TensorSpec:
    inp = spec.input
    if inp.kind == "tabular":
        return TensorSpec(
            TensorKind.TABULAR,
            (inp.num_numeric or 0,),
            num_numeric=inp.num_numeric or 0,
            cardinalities=tuple(inp.cardinalities or ()),
        )
    if inp.kind in ("image", "spectrogram"):
        if not inp.shape or len(inp.shape) != 3:
            raise ArchBuildError("input.shape", "se espera [C, H, W]")
        return TensorSpec(TensorKind.IMAGE, tuple(inp.shape))
    if inp.kind == "tokens":
        if not inp.shape or len(inp.shape) != 1:
            raise ArchBuildError("input.shape", "se espera [L] (largo máximo de la secuencia)")
        return TensorSpec(
            TensorKind.TOKENS, tuple(inp.shape), vocab_size=inp.vocab_size or 0, pad_id=inp.pad_id
        )
    raise ArchBuildError("input.kind", f"'{inp.kind}' todavía no está soportado")


def num_outputs(spec: ArchSpec) -> int:
    """Tamaño de la salida que espera la tarea (lo define su adaptador, ADR-0017)."""
    from perceptron.tasks import get_adapter

    try:
        return get_adapter(spec.task.type).num_outputs(spec)
    except ValueError as e:
        raise ArchBuildError("task", str(e)) from None


def topological_order(spec: ArchSpec) -> tuple[list[str], dict[str, list[str]]]:
    """Orden topológico estable (por orden de declaración) y predecesores de cada nodo."""
    ids = [n.id for n in spec.nodes]
    known = {*ids, INPUT_NODE}
    preds: dict[str, list[str]] = {i: [] for i in ids}
    for k, (a, b) in enumerate(spec.edges):
        if a not in known:
            raise ArchBuildError(f"edges[{k}][0]", f"nodo inexistente '{a}'")
        if b not in preds:
            raise ArchBuildError(f"edges[{k}][1]", f"nodo inexistente '{b}'")
        preds[b].append(a)
    order: list[str] = []
    done = {INPUT_NODE}
    pending = list(ids)
    while pending:
        ready = [i for i in pending if all(p in done for p in preds[i])]
        if not ready:
            raise ArchBuildError("edges", f"el grafo tiene ciclos entre {sorted(pending)}")
        for i in ready:
            order.append(i)
            done.add(i)
            pending.remove(i)
    for i in ids:
        if not preds[i]:
            raise ArchBuildError(f"nodes[{ids.index(i)}]", f"el nodo '{i}' no tiene entradas")
    used = {a for a, _ in spec.edges}
    sinks = [i for i in ids if i not in used]
    if len(sinks) != 1:
        raise ArchBuildError(
            "edges", f"debe haber exactamente una salida; hay {sinks or 'ninguna'}"
        )
    if order[-1] != sinks[0]:
        order.remove(sinks[0])
        order.append(sinks[0])
    return order, preds


def resolve_params(
    block: BlockSpec, raw: dict[str, Any], overrides: dict[str, Scalar] | None, path: str
) -> dict[str, Any]:
    unknown = set(raw) - set(block.params)
    if unknown:
        raise ArchBuildError(
            f"{path}.params", f"parámetros desconocidos para {block.key}: {sorted(unknown)}"
        )
    params = {k: p.default for k, p in block.params.items()}
    params.update(resolve(raw, overrides))
    for k, p in block.params.items():
        v = params[k]
        if v is None:
            continue
        if p.choices is not None and v not in p.choices and p.type != "int":
            raise ArchBuildError(f"{path}.params.{k}", f"valor {v!r} fuera de {p.choices}")
        numeric = isinstance(v, int | float) and not isinstance(v, bool)
        if numeric and ((p.low is not None and v < p.low) or (p.high is not None and v > p.high)):
            raise ArchBuildError(f"{path}.params.{k}", f"{v} fuera del rango [{p.low}, {p.high}]")
    return params


class ArchModel(nn.Module):
    """Ejecuta los nodos en orden topológico (secuencial, ramas, skips y fusiones)."""

    def __init__(
        self,
        order: list[str],
        preds: dict[str, list[str]],
        blocks: dict[str, nn.Module],
        tabular: bool,
    ) -> None:
        super().__init__()
        self.order = order
        self.preds = preds
        self.blocks = nn.ModuleDict(blocks)
        self.tabular = tabular

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        values: dict[str, Any] = {INPUT_NODE: tuple(inputs) if self.tabular else inputs[0]}
        for node_id in self.order:
            args = [values[p] for p in self.preds[node_id]]
            block = self.blocks[node_id]
            if len(args) == 1 and isinstance(args[0], tuple):
                values[node_id] = block(*args[0])
            else:
                values[node_id] = block(*args)
        out: torch.Tensor = values[self.order[-1]]
        return out


@dataclass
class BuildResult:
    model: nn.Module
    specs: dict[str, TensorSpec]
    output: TensorSpec
    num_params: int
    trainable_params: int
    activations_per_sample: int  # elementos (floats) de todas las salidas intermedias
    backbones: list[str] = field(default_factory=list)


def _dummy(t: TensorSpec, device: torch.device, batch: int = 2) -> tuple[torch.Tensor, ...]:
    if t.kind is TensorKind.TABULAR:
        return (
            torch.zeros(batch, t.num_numeric, device=device),
            torch.zeros(batch, len(t.cardinalities), dtype=torch.long, device=device),
        )
    if t.kind is TensorKind.TOKENS:
        return (torch.zeros(batch, *t.shape, dtype=torch.long, device=device),)
    return (torch.zeros(batch, *t.shape, device=device),)


def _out_spec(block: BlockSpec, inputs: list[TensorSpec], out: torch.Tensor) -> TensorSpec:
    shape = tuple(int(s) for s in out.shape[1:])
    kind: TensorKind
    if block.key in SHAPE_PRESERVING:
        kind = inputs[0].kind if inputs[0].kind is not TensorKind.IMAGE else TensorKind.FEATURE_MAP
    else:
        kind = block.produces
    if kind is TensorKind.FEATURES and len(shape) != 1:
        raise ArchBuildError(
            block.key, f"se esperaba un vector de features y la salida es {list(shape)}"
        )
    return TensorSpec(kind, shape)


def build_model(
    spec: ArchSpec,
    overrides: dict[str, Scalar] | None = None,
    *,
    pretrained_allowed: bool = True,
    materialize: bool = True,
) -> BuildResult:
    """Construye el modelo. Con `materialize=False` solo hace la pasada `meta` (validación)."""
    order, preds = topological_order(spec)
    ctx = BuildContext(num_outputs=num_outputs(spec), pretrained_allowed=False, meta=True)
    specs: dict[str, TensorSpec] = {INPUT_NODE: input_tensor_spec(spec)}
    params_by_node: dict[str, dict[str, Any]] = {}
    meta = torch.device("meta")
    values: dict[str, tuple[torch.Tensor, ...]] = {INPUT_NODE: _dummy(specs[INPUT_NODE], meta)}
    num_params = 0
    activations = 0
    meta_blocks: dict[str, nn.Module] = {}
    backbones: list[str] = []

    for node_id in order:
        idx = [n.id for n in spec.nodes].index(node_id)
        path = f"nodes[{idx}]"
        node = spec.nodes[idx]
        try:
            block = get_block(node.block)
        except KeyError as e:
            raise ArchBuildError(f"{path}.block", str(e)) from None
        in_specs = [specs[p] for p in preds[node_id]]
        if len(in_specs) > 1 and not block.multi_input:
            raise ArchBuildError(path, f"{block.key} no acepta múltiples entradas")
        for s in in_specs:
            if s.kind not in block.consumes:
                raise ArchBuildError(
                    path,
                    f"{block.key} no acepta entradas '{s.kind.value}' "
                    f"(acepta {[k.value for k in block.consumes]})",
                )
        params = resolve_params(block, node.params, overrides, path)
        params_by_node[node_id] = params
        if block.is_backbone:
            backbones.append(node_id)
        try:
            with meta:
                module = block.build(params, in_specs, ctx)
                args = [t for p in preds[node_id] for t in values[p]]
                out = module(*args)
        except ArchBuildError:
            raise
        except (ValueError, RuntimeError, TypeError) as e:
            raise ArchBuildError(path, f"{block.key}: {e}") from None
        specs[node_id] = _out_spec(block, in_specs, out)
        values[node_id] = (out,)
        meta_blocks[node_id] = module
        num_params += sum(p.numel() for p in module.parameters())
        activations += out[0].numel()

    output = specs[order[-1]]
    expected = ctx.num_outputs
    from perceptron.tasks import get_adapter

    problem = get_adapter(spec.task.type).check_output(output.kind.value, output.shape, spec)
    if problem:
        raise ArchBuildError(f"nodes[{len(spec.nodes) - 1}]", problem)

    model: nn.Module
    if materialize:
        real_ctx = BuildContext(num_outputs=expected, pretrained_allowed=pretrained_allowed)
        blocks = {
            nid: get_block(spec.node(nid).block).build(
                params_by_node[nid], [specs[p] for p in preds[nid]], real_ctx
            )
            for nid in order
        }
        model = ArchModel(order, preds, blocks, tabular=spec.input.kind == "tabular")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model = ArchModel(order, preds, meta_blocks, tabular=spec.input.kind == "tabular")
        trainable = num_params
    return BuildResult(
        model=model,
        specs=specs,
        output=output,
        num_params=num_params,
        trainable_params=trainable,
        activations_per_sample=activations,
        backbones=backbones,
    )
