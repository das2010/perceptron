from __future__ import annotations

from pathlib import Path
from typing import Any

import pydantic
import pytest
import torch

from perceptron.archspec.builder import build_model
from perceptron.archspec.schema import HP, ArchSpec, Node
from perceptron.archspec.to_code import archspec_to_code
from perceptron.archspec.validate import Stage, validate_archspec
from perceptron.catalog.registry import BLOCKS, blocks_for
from perceptron.catalog.rules import recommend
from perceptron.catalog.templates import image_template, tabular_template
from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import Modality, TaskType


def _tab(template: str = "mlp", **kw: Any) -> ArchSpec:
    return tabular_template(
        template,
        task=kw.get("task", TaskType.CLASSIFICATION),
        num_classes=kw.get("num_classes", 3),
        num_numeric=5,
        cardinalities=[4, 7],
    )


def _img(backbone: str = "small_cnn", **kw: Any) -> ArchSpec:
    return image_template(
        backbone,
        task=TaskType.CLASSIFICATION,
        num_classes=2,
        image_size=kw.get("size", 32),
        channels=kw.get("channels", 3),
        pretrained=kw.get("pretrained", False),
    )


def _tab_batch(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(n, 5), torch.stack([torch.randint(0, 4, (n,)), torch.randint(0, 7, (n,))], 1)


# ------------------------------------------------------------------ schema


def test_schema_roundtrip_hash_and_hyperparameters() -> None:
    spec = _tab()
    again = ArchSpec.model_validate_json(spec.model_dump_json())
    assert again.content_hash() == spec.content_hash()
    hps = spec.hyperparameters()
    assert {
        "hidden",
        "layers",
        "dropout",
        "lr",
        "weight_decay",
        "epochs",
        "label_smoothing",
    } <= set(hps)
    assert hps["lr"] == 1e-3
    assert isinstance(spec.node("mlp").params["hidden"], HP)
    assert ArchSpec.model_json_schema()["title"] == "ArchSpec"


def test_schema_rejects_duplicate_and_reserved_ids() -> None:
    data = _tab().model_dump(mode="json")
    data["nodes"].append(dict(data["nodes"][0]))
    with pytest.raises(pydantic.ValidationError, match="duplicados"):
        ArchSpec.model_validate(data)
    data = _tab().model_dump(mode="json")
    data["nodes"][0]["id"] = "input"
    with pytest.raises(pydantic.ValidationError, match="reservado"):
        ArchSpec.model_validate(data)


# ------------------------------------------------------------------ builder


@pytest.mark.parametrize("template", ["mlp", "resnet_mlp", "ft_transformer"])
@pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.REGRESSION])
def test_tabular_templates_forward(template: str, task: TaskType) -> None:
    spec = _tab(template, task=task)
    built = build_model(spec)
    out = built.model(*_tab_batch())
    assert out.shape == (4, 3 if task is TaskType.CLASSIFICATION else 1)
    assert built.num_params == sum(p.numel() for p in built.model.parameters())


@pytest.mark.parametrize(
    ("backbone", "channels"),
    [
        ("small_cnn", 3),
        ("small_cnn", 1),
        ("efficientnet_b0", 3),
        ("mobilenetv3_small_100", 1),
        ("resnet18", 3),
    ],
)
def test_image_templates_forward(backbone: str, channels: int) -> None:
    built = build_model(_img(backbone, channels=channels), pretrained_allowed=False)
    out = built.model(torch.randn(2, channels, 32, 32))
    assert out.shape == (2, 2)


def test_hp_overrides_change_the_model() -> None:
    small = build_model(_tab(), {"hidden": 16, "layers": 1})
    big = build_model(_tab(), {"hidden": 256, "layers": 3})
    assert small.num_params < big.num_params


def test_branches_skip_and_concat() -> None:
    spec = _tab().model_copy(
        update={
            "nodes": [
                Node(id="features", block="input.tabular", params={"dropout": 0.0}),
                Node(id="a", block="resnet_mlp.block", params={"d": 32, "blocks": 1}),
                Node(id="b", block="resnet_mlp.block", params={"d": 32, "blocks": 1}),
                Node(id="skip", block="merge.add"),
                Node(id="cat", block="merge.concat"),
                Node(id="head", block="head.linear"),
            ],
            "edges": [
                ("input", "features"),
                ("features", "a"),
                ("features", "b"),
                ("a", "skip"),
                ("b", "skip"),
                ("skip", "cat"),
                ("a", "cat"),
                ("cat", "head"),
            ],
        }
    )
    built = build_model(spec)
    assert built.specs["cat"].shape == (64,)
    assert built.model(*_tab_batch()).shape == (4, 3)


def test_build_is_deterministic_with_seed() -> None:
    torch.manual_seed(0)
    a = build_model(_tab()).model.state_dict()
    torch.manual_seed(0)
    b = build_model(_tab()).model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------ validación (§9.3)


def _mutate(spec: ArchSpec, **update: Any) -> dict[str, Any]:
    data = spec.model_dump(mode="json")
    data.update(update)
    return data


def test_valid_spec_report() -> None:
    report = validate_archspec(_img())
    assert report.valid, report.feedback()
    assert report.num_params and report.num_params > 0
    assert report.output_shape == [2]
    assert report.estimated_memory_mb and report.estimated_memory_mb > 0


def test_stage1_schema_errors_have_paths() -> None:
    data = _mutate(_tab(), loss={"type": "cuadrados"})
    report = validate_archspec(data)
    assert not report.valid
    assert report.issues[0].stage is Stage.SCHEMA
    assert report.issues[0].path.startswith("loss")


def test_stage2_unknown_block_and_wrong_modality() -> None:
    data = _tab().model_dump(mode="json")
    data["nodes"][1]["block"] = "mlp.gigante"
    report = validate_archspec(data)
    assert report.errors[0].stage is Stage.BLOCKS
    assert report.errors[0].path == "nodes[1].block"
    assert "mlp.block" in (report.errors[0].suggestion or "")

    data = _tab().model_dump(mode="json")
    data["nodes"][1]["block"] = "vision.small_cnn"
    assert "modalidad" in validate_archspec(data).errors[0].message


def test_stage2_param_out_of_range() -> None:
    data = _tab().model_dump(mode="json")
    data["nodes"][1]["params"]["layers"] = 50
    report = validate_archspec(data)
    assert report.errors[0].stage is Stage.BLOCKS
    assert report.errors[0].path == "nodes[1].params.layers"


def test_stage3_graph_cycle_and_multiple_outputs() -> None:
    data = _tab().model_dump(mode="json")
    data["edges"].append(["head", "mlp"])
    assert validate_archspec(data).errors[0].stage is Stage.GRAPH

    data = _tab().model_dump(mode="json")
    data["edges"] = [["input", "features"], ["features", "mlp"], ["features", "head"]]
    report = validate_archspec(data)
    assert report.errors[0].stage is Stage.GRAPH
    assert "una salida" in report.errors[0].message


def test_stage4_shape_mismatch_and_wrong_output() -> None:
    data = _img().model_dump(mode="json")
    data["nodes"] = [n for n in data["nodes"] if n["id"] != "pool"]
    data["edges"] = [["input", "encoder"], ["encoder", "drop"], ["drop", "head"]]
    report = validate_archspec(data)
    assert report.errors[0].stage is Stage.SHAPES

    data = _tab().model_dump(mode="json")
    data["nodes"][-1]["params"] = {"out_features": 7}
    report = validate_archspec(data)
    assert report.errors[0].stage is Stage.SHAPES
    assert "3 valores" in report.errors[0].message


def test_stage5_memory() -> None:
    report = validate_archspec(_img("resnet18", size=224), batch_size=512, device_memory_gb=0.05)
    assert report.errors[0].stage is Stage.RESOURCES


def test_stage6_offline_weights_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    report = validate_archspec(_img("efficientnet_b0", pretrained=True), cache_dir=tmp_path)
    assert report.valid  # es advertencia, no error
    assert report.issues[0].stage is Stage.WEIGHTS
    assert "pretrained: false" in (report.issues[0].suggestion or "")


# ------------------------------------------------------------------ to_code (RF-ARC-07)


@pytest.mark.parametrize("which", ["tab", "tab_ft", "img", "img_gray_timm"])
def test_to_code_is_equivalent(which: str) -> None:
    spec = {
        "tab": _tab,
        "tab_ft": lambda: _tab("ft_transformer"),
        "img": _img,
        "img_gray_timm": lambda: _img("mobilenetv3_small_100", channels=1, pretrained=True),
    }[which]()
    code = archspec_to_code(spec, pretrained=False)
    namespace: dict[str, Any] = {"__name__": "generated"}
    exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - código generado por nosotros
    generated = namespace["Model"]()
    built = build_model(spec, pretrained_allowed=False).model
    state = {k.removeprefix("blocks."): v for k, v in built.state_dict().items()}
    generated.load_state_dict(state)
    built.eval()
    generated.eval()
    inputs = (
        _tab_batch()
        if spec.modality is Modality.TABULAR
        else (torch.randn(2, spec.input.shape[0], 32, 32),)
    )  # type: ignore[index]
    with torch.no_grad():
        assert torch.allclose(built(*inputs), generated(*inputs), atol=1e-5)


# ------------------------------------------------------------------ catálogo y reglas


def test_catalog_listing() -> None:
    keys = {b.key for b in blocks_for(Modality.TABULAR)}
    assert "input.tabular" in keys and "vision.small_cnn" not in keys
    assert all(b.public()["key"] == k for k, b in BLOCKS.items())


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_a").ensure()


def _fitted(paths: ProjectPaths, src: Path):  # type: ignore[no-untyped-def]
    v = ingest(paths, IngestRequest(project_id="prj_a", source=src))
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    spec = propose_pipeline(card)
    return card, fit_pipeline(spec, view.read("train"))


def test_rules_churn_and_images(
    paths: ProjectPaths, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERCEPTRON_OFFLINE", "1")
    card, fitted = _fitted(paths, fixtures_dir / "uc01_churn" / "churn.csv")
    rec = recommend(card, fitted)
    assert rec.template == "mlp"  # 400 filas
    assert validate_archspec(rec.spec).valid
    assert rec.spec.input.cardinalities == [
        fitted.cardinalities[c] for c in fitted.categorical_features
    ]

    card, fitted = _fitted(paths, fixtures_dir / "uc04_defects")
    rec = recommend(card, fitted)
    assert rec.template == "small_cnn"  # CPU, 32 px, sin conexión
    assert rec.spec.input.shape == [3, 32, 32]
    assert validate_archspec(rec.spec).valid
