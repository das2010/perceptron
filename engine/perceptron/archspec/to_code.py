"""ArchSpec → código PyTorch legible (RF-ARC-07).

El código generado es autocontenido (torch + timm): incluye el fuente de los
bloques usados, copiado de `perceptron.catalog.modules`, y una clase `Model`
cuyo `forward` sigue el grafo. Es equivalente al modelo que construye
`archspec.builder` (mismos parámetros y nombres de pesos, sin el prefijo `blocks.`).
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from perceptron.archspec.builder import build_model
from perceptron.archspec.schema import INPUT_NODE, ArchSpec, Scalar
from perceptron.catalog import modules as m
from perceptron.catalog.registry import TIMM_WEIGHTS, TensorSpec


def identifier(node_id: str) -> str:
    ident = re.sub(r"\W", "_", node_id)
    return f"n_{ident}" if ident[0].isdigit() else ident


def _ctor(
    block: str, p: dict[str, Any], ins: list[TensorSpec], n_out: int, pretrained: bool
) -> tuple[str, str | None]:
    """(expresión del constructor, clase de `modules` que requiere o None)."""
    i = ins[0]
    match block:
        case "input.tabular":
            return (
                f"TabularInput({i.num_numeric}, {list(i.cardinalities)}, "
                f"{p.get('embed_dim')!r}, {p['dropout']!r})",
                "TabularInput",
            )
        case "ft_transformer.encoder":
            args = (
                f"{i.num_numeric}, {list(i.cardinalities)}, {p['d_token']}, "
                f"{p['n_blocks']}, {p['n_heads']}, {p['dropout']!r}"
            )
            return f"FTTransformer({args})", "FTTransformer"
        case "mlp.block":
            args = (
                f"{i.dim}, {p['hidden']}, {p['layers']}, {p['dropout']!r}, "
                f"{p['activation']!r}, {p['batchnorm']!r}"
            )
            return f"MLPBlock({args})", "MLPBlock"
        case "resnet_mlp.block":
            return (
                f"ResidualMLPBlock({i.dim}, {p['d']}, {p['blocks']}, "
                f"{p['hidden_factor']!r}, {p['dropout']!r})",
                "ResidualMLPBlock",
            )
        case "vision.timm_backbone":
            use = bool(p["pretrained"]) and pretrained
            name = f"{p['model']}.{TIMM_WEIGHTS[p['model']].pretrained_tag}" if use else p["model"]
            return (
                f"TimmBackbone({name!r}, pretrained={use}, in_chans={i.channels})",
                "TimmBackbone",
            )
        case "vision.small_cnn":
            return (
                f"SmallCNN({i.channels}, {p['width']}, {p['depth']}, {p['dropout']!r})",
                "SmallCNN",
            )
        case "conv.channel_adapter":
            return f"ChannelAdapter({i.channels}, {p['out_channels']})", "ChannelAdapter"
        case "pool.global_avg":
            return "GlobalAvgPool()", "GlobalAvgPool"
        case "norm.batchnorm":
            return (
                f"nn.BatchNorm1d({i.dim})" if len(i.shape) == 1 else f"nn.BatchNorm2d({i.channels})"
            ), None
        case "reg.dropout":
            return f"nn.Dropout({p['p']!r})", None
        case "head.linear":
            return f"nn.Linear({i.dim}, {p.get('out_features') or n_out})", None
        case "text.embedding":
            return (
                f"TextEmbedding({i.vocab_size}, {p['dim']}, {p['dropout']!r}, {i.pad_id})",
                "TextEmbedding",
            )
        case "text.cnn":
            ks = list(p["kernel_sizes"])
            return f"TextCNN({i.shape[-1]}, {p['filters']}, {ks}, {p['dropout']!r})", "TextCNN"
        case "text.bilstm":
            args = f"{i.shape[-1]}, {p['hidden']}, {p['layers']}, {p['dropout']!r}"
            return f"BiLSTMEncoder({args})", "BiLSTMEncoder"
        case "pool.sequence":
            return f"SequencePool({p['mode']!r})", "SequencePool"
        case "text.hf_encoder":
            use = bool(p["pretrained"]) and pretrained
            args = f"{p['model']!r}, pretrained={use}, pad_id={i.pad_id}, pooling={p['pooling']!r}"
            return f"HFTextEncoder({args})", "HFTextEncoder"
        case "seq.rnn":
            args = f"{i.shape[-1]}, {p['hidden']}, {p['layers']}, {p['dropout']!r}, {p['cell']!r}"
            return f"RNNEncoder({args})", "RNNEncoder"
        case "seq.tcn":
            args = f"{i.shape[-1]}, {p['channels']}, {p['levels']}, {p['kernel']}, {p['dropout']!r}"
            return f"TCN({args})", "TCN"
        case "seq.nbeats":
            args = (
                f"{i.shape[0]}, {i.shape[1]}, {n_out}, {p['hidden']}, {p['blocks']}, "
                f"{p['layers']}, {p['dropout']!r}"
            )
            return f"NBeats({args})", "NBeats"
        case "seq.patchtst":
            patch = min(p["patch_len"], i.shape[0])
            args = (
                f"{i.shape[0]}, {i.shape[1]}, {patch}, {p['stride']}, {p['d_model']}, "
                f"{p['heads']}, {p['layers']}, {p['dropout']!r}"
            )
            return f"PatchTST({args})", "PatchTST"
        case "seq.autoencoder":
            args = f"{i.shape[0]}, {i.shape[1]}, {p['hidden']}, {p['latent']}, {p['kind']!r}"
            return f"SeriesAutoencoder({args})", "SeriesAutoencoder"
        case "merge.concat":
            return "Concat()", "Concat"
        case "merge.add":
            return "Add()", "Add"
    raise ValueError(f"to_code no soporta el bloque {block}")


_DEPENDENCIES = {
    "TabularInput": ["embedding_dim"],
    "MLPBlock": ["_activation"],
}


def archspec_to_code(
    spec: ArchSpec, overrides: dict[str, Scalar] | None = None, *, pretrained: bool = True
) -> str:
    from perceptron.archspec.builder import num_outputs, resolve_params, topological_order
    from perceptron.catalog.registry import get_block

    built = build_model(spec, overrides, pretrained_allowed=False, materialize=False)
    order, preds = topological_order(spec)
    n_out = num_outputs(spec)
    needed: list[str] = []
    ctor_lines: list[str] = []
    for nid in order:
        node = spec.node(nid)
        idx = [n.id for n in spec.nodes].index(nid)
        params = resolve_params(get_block(node.block), node.params, overrides, f"nodes[{idx}]")
        expr, cls = _ctor(
            node.block, params, [built.specs[p] for p in preds[nid]], n_out, pretrained
        )
        ctor_lines.append(f"        self.{identifier(nid)} = {expr}")
        if cls:
            for dep in _DEPENDENCIES.get(cls, []):
                if dep not in needed:
                    needed.append(dep)
            if cls not in needed:
                needed.append(cls)

    tabular = spec.input.kind == "tabular"
    sig = "x_num: torch.Tensor, x_cat: torch.Tensor" if tabular else "x: torch.Tensor"
    names = {INPUT_NODE: "x_num, x_cat" if tabular else "x"}
    fwd: list[str] = []
    for nid in order:
        var = identifier(nid)
        args = ", ".join(names[p] for p in preds[nid])
        fwd.append(f"        {var} = self.{var}({args})")
        names[nid] = var

    sources = "\n\n".join(inspect.getsource(getattr(m, name)).rstrip() for name in needed)
    hp = built.num_params
    return f'''"""Modelo generado por Perceptron desde la ArchSpec "{spec.name}".

ArchSpec hash: {spec.content_hash()}
Parámetros: {hp:,}
Entrada: {spec.input.kind} {spec.input.shape or ""}
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn


{sources}


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
{chr(10).join(ctor_lines)}

    def forward(self, {sig}) -> torch.Tensor:
{chr(10).join(fwd)}
        return {names[order[-1]]}


if __name__ == "__main__":
    model = Model()
    print(model)
    print(sum(p.numel() for p in model.parameters()), "parámetros")
'''
