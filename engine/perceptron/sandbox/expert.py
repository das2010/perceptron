"""Modo experto (RF-ARC-06): ArchSpec de código y código inicial por modalidad (ADR-0025)."""

from __future__ import annotations

from perceptron.archspec.schema import INPUT_NODE, ArchSpec, Node, Provenance
from perceptron.catalog.registry import CODE_BLOCK
from perceptron.domain.enums import Origin, TaskType
from perceptron.sandbox.code import source_sha256

_TASKS = (TaskType.CLASSIFICATION, TaskType.REGRESSION)


def is_code_spec(spec: ArchSpec) -> bool:
    return any(n.block == CODE_BLOCK for n in spec.nodes)


def build_code_spec(base: ArchSpec, source: str, name: str | None = None) -> ArchSpec:
    """ArchSpec de código: entrada, tarea, pérdida, optimizador y entrenamiento de `base`."""
    task = base.task.type
    if task not in _TASKS:
        raise ValueError("el modo experto admite clasificación y regresión")
    node = Node(
        id="model",
        block=CODE_BLOCK,
        params={"code_sha256": source_sha256(source), "task": task.value},
    )
    training = base.training.model_copy(update={"freeze_backbone_epochs": 0})
    return base.model_copy(
        deep=True,
        update={
            "name": name or f"{base.name}-codigo",
            "nodes": [node],
            "edges": [(INPUT_NODE, node.id)],
            "training": training,
            "provenance": Provenance(
                origin=Origin.MANUAL,
                template="code",
                rationale="Código experto (no declarativo, RF-ARC-06), validado en el sandbox.",
            ),
        },
    )


_HEADER = '''"""Modelo escrito a mano (modo experto, RF-ARC-06).

Interfaz fija: `build_model(config) -> nn.Module`. `config["input"]` describe la entrada,
`config["num_outputs"]` el tamaño de la salida y `config["task"]` la tarea. Se ejecuta en un
sandbox sin red ni acceso a archivos fuera del run; solo se pueden importar torch, math,
typing, dataclasses, collections, functools, itertools y numbers.
"""

import torch
from torch import nn

'''

_TABULAR = """input_spec = {"kind": "tabular"}


class Model(nn.Module):
    def __init__(self, num_numeric: int, cardinalities: list[int], num_outputs: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(c + 1, min(16, (c + 1) // 2 + 1)) for c in cardinalities
        )
        width = num_numeric + sum(e.embedding_dim for e in self.embeddings)
        self.net = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_outputs),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.net(torch.cat(parts, dim=1))


def build_model(config: dict) -> nn.Module:
    inp = config["input"]
    return Model(inp["num_numeric"], inp["cardinalities"], config["num_outputs"])
"""

_IMAGE = """input_spec = {"kind": "image"}


def conv(c_in: int, c_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class Model(nn.Module):
    def __init__(self, channels: int, num_outputs: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv(channels, 32), conv(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(config: dict) -> nn.Module:
    return Model(config["input"]["shape"][0], config["num_outputs"])
"""

_TOKENS = """input_spec = {"kind": "tokens"}


class Model(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, num_outputs: int) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=pad_id)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != self.pad_id).unsqueeze(-1).float()
        pooled = (self.embedding(x) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


def build_model(config: dict) -> nn.Module:
    inp = config["input"]
    return Model(inp["vocab_size"], inp["pad_id"], config["num_outputs"])
"""

_SEQUENCE = """input_spec = {"kind": "sequence"}


class Model(nn.Module):
    def __init__(self, features: int, num_outputs: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(features, 64, batch_first=True)
        self.head = nn.Linear(64, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(x)
        return self.head(h[-1])


def build_model(config: dict) -> nn.Module:
    return Model(config["input"]["shape"][-1], config["num_outputs"])
"""


def starter_code(spec: ArchSpec) -> str:
    """Código inicial que respeta la interfaz para la entrada de `spec`."""
    body = {
        "tabular": _TABULAR,
        "image": _IMAGE,
        "spectrogram": _IMAGE,  # espectrogramas: imágenes de 1 canal
        "tokens": _TOKENS,
        "sequence": _SEQUENCE,
    }[spec.input.kind]
    return _HEADER + body
