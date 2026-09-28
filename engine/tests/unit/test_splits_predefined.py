"""Split predefinido: la partición viene en una columna (reentrenamiento, RF-MON-05)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from perceptron.api.context import EngineContext
from perceptron.core.errors import ValidationError
from perceptron.data.splits import SPLIT_COLUMN, SplitRequest, assign_splits
from perceptron.domain.enums import SplitStrategy
from perceptron.domain.models import Project
from perceptron.services.workflow import Workflow


def test_assign_predefined_split() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4], "parte": ["train", "val", "test", "train"]})
    req = SplitRequest(strategy=SplitStrategy.PREDEFINED, split_column="parte")
    out = assign_splits(df, req, None)
    assert out.columns == ["x", SPLIT_COLUMN]
    assert out[SPLIT_COLUMN].to_list() == ["train", "val", "test", "train"]
    with pytest.raises(ValidationError, match="inválidos"):
        assign_splits(df.with_columns(pl.lit("otro").alias("parte")), req, None)
    with pytest.raises(ValueError, match="split_column"):
        SplitRequest(strategy=SplitStrategy.PREDEFINED)


def test_ingest_keeps_predefined_split(ctx: EngineContext, tmp_path: Path) -> None:
    project = ctx.projects.add(Project(name="Reentrenar"))
    ctx.files.init_project(project)
    parts = ["train"] * 30 + ["val"] * 10 + ["test"] * 10
    data = pl.DataFrame(
        {"a": list(range(50)), "y": [i % 2 for i in range(50)], "__particion": parts}
    )
    path = tmp_path / "con split.csv"
    data.write_csv(path)
    dv = Workflow(ctx).ingest(
        project.id,
        path,
        target="y",
        split=SplitRequest(strategy=SplitStrategy.PREDEFINED, split_column="__particion"),
    )
    assert dv.split is not None and (dv.split.train, dv.split.val, dv.split.test) == (30, 10, 10)
    view = Workflow(ctx).view(dv)
    assert "__particion" not in view.scan().collect().columns
