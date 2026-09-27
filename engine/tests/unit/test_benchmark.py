"""Partes puras del benchmark O2 (sin red): parsers, organización, brecha y referencia."""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from perceptron.archspec.schema import HP
from perceptron.benchmark.datasets import adult_rows, organize_fsdd, read_idx
from perceptron.benchmark.runner import BenchResult, gap, markdown, with_defaults
from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType


def test_adult_rows_cleaning() -> None:
    line = (
        "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, "
        "White, Male, 2174, 0, 40, United-States, <=50K"
    )
    rows = adult_rows("\n".join([line, line.replace("<=50K", ">50K"), "1, ?, 2", ""]))
    assert [r["income"] for r in rows] == ["bajo", "alto"] and rows[0]["workclass"] == "State-gov"


def test_read_idx() -> None:
    data = struct.pack(">HBB", 0, 8, 3) + struct.pack(">III", 2, 2, 2) + bytes(range(8))
    dims, raw = read_idx(data)
    assert dims == [2, 2, 2] and raw == bytes(range(8))
    with pytest.raises(ValueError, match="uint8"):
        read_idx(struct.pack(">HBB", 0, 9, 1) + struct.pack(">I", 1) + b"\x00")


def test_organize_fsdd(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for d in ("0", "1"):
            for i in range(3):
                z.writestr(f"fsdd-1.0/recordings/{d}_ana_{i}.wav", b"RIFF")
        z.writestr("fsdd-1.0/README.md", "x")
    root = organize_fsdd(buf.getvalue(), tmp_path / "fsdd", per_class=2)
    assert sorted(p.name for p in root.iterdir()) == ["digito_0", "digito_1"]
    assert len(list((root / "digito_0").iterdir())) == 2


def test_gap_and_report() -> None:
    assert gap(0.9, 0.87, "accuracy") == pytest.approx(0.0333, abs=1e-3)
    assert gap(0.5, 0.45, "val_loss") < 0  # menos pérdida: el agente es mejor
    r = BenchResult("adult", "Adult", "CC BY 4.0", "roc_auc", 0.9, 0.88, 0.022, True)
    md = markdown([r], "2026-09-27")
    assert "| Adult | CC BY 4.0 | roc_auc | 0.9000 | 0.8800 | +2.2% | ✅ |" in md


def test_with_defaults_fixes_hp_values() -> None:
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[4]
    )
    fixed = with_defaults(spec, {"lr": 0.0123})
    assert fixed.hyperparameters()["lr"] == 0.0123
    lr = fixed.optimizer.lr
    assert isinstance(lr, HP) and lr.default == 0.0123
