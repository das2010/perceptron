from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import (
    FittedPipeline,
    ImageSpec,
    PipelineSpec,
    TargetSpec,
    decode_regression,
    fit_pipeline,
    image_transforms,
    transform_tabular,
)
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.pipeline.steps import StepSpec, build_step
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import Modality, TaskType


def _step(kind: str, cols: list[str], **params: object) -> StepSpec:
    return StepSpec(id=f"t_{kind}", kind=kind, columns=cols, params=params)


def test_unknown_step() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        build_step(_step("magia", ["x"]))


def test_impute_uses_train_statistics_only() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, None, 3.0]})
    test = pl.DataFrame({"x": [None, 100.0]})
    step = build_step(_step("impute_numeric", ["x"], strategy="median"))
    state = step.fit(train)
    assert state == {"fill": {"x": 2.0}}
    assert step.transform(test, state)["x"].to_list() == [2.0, 100.0]


def test_ordinal_unknown_category_maps_to_zero() -> None:
    step = build_step(_step("ordinal", ["c"]))
    state = step.fit(pl.DataFrame({"c": ["b", "a", "b"]}))
    out = step.transform(pl.DataFrame({"c": ["a", "b", "zzz", None]}), state)
    assert out["c"].to_list() == [1, 2, 0, 0]


def test_scalers() -> None:
    df = pl.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 40.0]})
    for method in ("standard", "robust", "minmax", "quantile"):
        step = build_step(_step("scale", ["x"], method=method))
        out = step.transform(df, step.fit(df))["x"].to_numpy()
        assert np.all(np.diff(out) > 0), method  # monótono
    minmax = build_step(_step("scale", ["x"], method="minmax"))
    assert minmax.transform(df, minmax.fit(df))["x"].to_list() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_one_hot_and_dates_and_bools() -> None:
    df = pl.DataFrame(
        {
            "c": ["a", "b", "a"],
            "d": pl.Series(["2026-01-05", "2026-02-10", "2026-03-15"]).str.to_date(),
            "b": ["si", "no", "si"],
        }
    )
    spec = PipelineSpec(
        modality=Modality.TABULAR,
        target=None,
        steps=[_step("one_hot", ["c"]), _step("date_features", ["d"]), _step("to_numeric", ["b"])],
    )
    fitted = fit_pipeline(spec, df)
    assert set(fitted.numeric_features) == {
        "b",
        "c=a",
        "c=b",
        "d.year",
        "d.month",
        "d.weekday",
        "d.ordinal_day",
    }
    arr = transform_tabular(fitted, df)
    assert arr.x_num.shape == (3, 7)
    assert arr.x_cat.shape == (3, 0)


def test_unencoded_columns_are_rejected() -> None:
    spec = PipelineSpec(modality=Modality.TABULAR, target=None, steps=[])
    with pytest.raises(ValueError, match="sin codificar"):
        fit_pipeline(spec, pl.DataFrame({"c": ["a", "b"]}))


def test_regression_target_standardization_roundtrip() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    spec = PipelineSpec(
        modality=Modality.TABULAR,
        target=TargetSpec(name="y", task=TaskType.REGRESSION, standardize=True),
        steps=[_step("scale", ["x"])],
    )
    fitted = fit_pipeline(spec, df)
    arr = transform_tabular(fitted, df)
    assert arr.y is not None
    assert abs(float(arr.y.mean())) < 1e-6
    assert np.allclose(decode_regression(fitted, arr.y), [10, 20, 30, 40], atol=1e-4)


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_pl").ensure()


def test_propose_and_fit_churn(paths: ProjectPaths, fixtures_dir: Path) -> None:
    v = ingest(
        paths, IngestRequest(project_id="prj_pl", source=fixtures_dir / "uc01_churn" / "churn.csv")
    )
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    spec = propose_pipeline(card)
    kinds = [s.kind for s in spec.steps]
    assert kinds[0] == "drop" and "customer_id" in spec.steps[0].columns
    assert "impute_numeric" in kinds and "ordinal" in kinds and "scale" in kinds
    assert len(spec.rationale) == len(spec.steps)
    assert spec.target is not None and spec.target.task is TaskType.CLASSIFICATION

    train = view.read("train")
    fitted = fit_pipeline(spec, train)
    assert set(fitted.categorical_features) == {"region", "plan"}
    assert fitted.cardinalities["plan"] == 4  # 3 planes + desconocido
    assert fitted.classes == ["0", "1"]

    arr = transform_tabular(fitted, view.read("val"))
    assert arr.x_num.dtype == np.float32 and not np.isnan(arr.x_num).any()
    assert arr.x_cat.dtype == np.int64
    assert arr.y is not None and set(np.unique(arr.y)) <= {0, 1}

    # Serialización completa (se empaqueta con el modelo)
    again = FittedPipeline.model_validate_json(fitted.model_dump_json())
    assert np.array_equal(
        transform_tabular(again, train).x_num, transform_tabular(fitted, train).x_num
    )


def test_propose_image_and_transforms(paths: ProjectPaths, fixtures_dir: Path) -> None:
    v = ingest(paths, IngestRequest(project_id="prj_pl", source=fixtures_dir / "uc04_defects"))
    card = profile_dataset(DatasetView(paths.dataset(v.content_hash)))
    spec = propose_pipeline(card)
    assert spec.image is not None
    assert spec.image.size == 32
    assert spec.image.normalize == "dataset"
    assert propose_pipeline(card, pretrained=True).image == ImageSpec(
        size=224, normalize="imagenet", augment=spec.image.augment
    )

    from PIL import Image

    fitted = fit_pipeline(spec, pl.DataFrame({"label": ["ok", "defect"]}))
    assert fitted.classes == ["defect", "ok"]
    img = Image.new("RGB", (40, 30), (120, 130, 140))
    for train in (True, False):
        t = image_transforms(fitted, train=train)(img)
        assert tuple(t.shape) == (3, 32, 32)
