"""Validación estática del código experto (RF-ARC-06, ADR-0025): nunca lo ejecuta."""

from __future__ import annotations

import pytest

from perceptron.archspec.schema import ArchSpec
from perceptron.catalog.templates import image_template, tabular_template
from perceptron.domain.enums import TaskType
from perceptron.sandbox.expert import build_code_spec, is_code_spec, starter_code
from perceptron.sandbox.static import MAX_SOURCE_BYTES, check_source

OK = """
import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.fc = nn.Linear(n, 2)
        self._cache = None

    def forward(self, x):
        return F.relu(self.fc(x))


def build_model(config):
    return Model(config["input"]["num_numeric"])
"""


def _tab() -> ArchSpec:
    return tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[4]
    )


def test_valid_code_passes() -> None:
    report = check_source(OK)
    assert report.valid, report.issues


@pytest.mark.parametrize(
    ("snippet", "fragment"),
    [
        ("import os", "import no permitido: os"),
        ("import subprocess", "import no permitido: subprocess"),
        ("from socket import socket", "import no permitido: socket"),
        ("from . import x", "relativos"),
        ("from torch import *", "import *"),
        ("from torch import hub", "torch.hub"),
        ("x = open('a.txt')", "nombre no permitido: open"),
        ("x = eval('1')", "nombre no permitido: eval"),
        ("x = getattr(torch, 'load')", "nombre no permitido: getattr"),
        ("x = __import__('os')", "nombre no permitido: __import__"),
        ("x = ().__class__.__bases__", "atributo no permitido: .__class__"),
        ("x = torch.load('w.pt')", "atributo no permitido: .load"),
        ("x = torch.hub.load('a', 'b')", "atributo no permitido: .hub"),
        ("x = torch.utils.cpp_extension", "atributo no permitido: .utils"),
        ("x = torch.ops.load_library('x.so')", "atributo no permitido: .ops"),
        ("x = torch.distributed", "atributo no permitido: .distributed"),
        ("global y", "global"),
    ],
)
def test_forbidden_constructs(snippet: str, fragment: str) -> None:
    report = check_source(f"import torch\n{snippet}\n\ndef build_model(config):\n    pass\n")
    assert not report.valid
    assert any(fragment in i.message for i in report.issues), report.issues
    assert all(i.line >= 1 for i in report.issues)


def test_missing_entrypoint_syntax_and_size() -> None:
    assert any("build_model" in i.message for i in check_source("x = 1\n").issues)
    bad = check_source("def build_model(config)\n    pass\n")
    assert not bad.valid and bad.issues[0].message.startswith("sintaxis")
    huge = check_source("#" * (MAX_SOURCE_BYTES + 1))
    assert not huge.valid


def test_starters_pass_static_check() -> None:
    image = image_template(
        "small_cnn", task=TaskType.CLASSIFICATION, num_classes=2, image_size=32, pretrained=False
    )
    for spec in (_tab(), image):
        report = check_source(starter_code(spec))
        assert report.valid, report.issues


def test_code_spec_keeps_task_and_hashes_the_source() -> None:
    base = _tab()
    spec = build_code_spec(base, OK, "a-mano")
    assert is_code_spec(spec) and not is_code_spec(base)
    assert spec.task == base.task and spec.loss == base.loss and spec.input == base.input
    assert [n.block for n in spec.nodes] == ["code.module"]
    other = build_code_spec(base, OK + "\n# cambio\n", "a-mano")
    assert spec.content_hash() != other.content_hash()
    assert spec.provenance.template == "code"
