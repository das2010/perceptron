"""Visión avanzada (Capa 1b, sub-hito 4): anotaciones, datasets, modelos y métricas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from perceptron.archspec.builder import build_model
from perceptron.archspec.to_code import archspec_to_code
from perceptron.archspec.validate import validate_archspec
from perceptron.catalog.rules import recommend
from perceptron.catalog.templates import vision_task_template
from perceptron.core.paths import ProjectPaths
from perceptron.data.pipeline.pipeline import fit_pipeline
from perceptron.data.pipeline.propose import propose_pipeline
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.data.vision_tasks import load_coco, load_voc, load_yolo
from perceptron.domain.enums import TaskType
from perceptron.tasks.vision import (
    DiceCELoss,
    centernet_targets,
    decode_centernet,
    edit_distance,
    greedy_ctc,
)
from perceptron.training.data import make_dataset


def test_annotation_formats(tmp_path: Path) -> None:
    coco = {
        "images": [{"id": 1, "file_name": "a.png"}],
        "annotations": [{"image_id": 1, "category_id": 3, "bbox": [2, 3, 4, 5]}],
        "categories": [{"id": 3, "name": "grieta"}],
    }
    (tmp_path / "ann.json").write_text(json.dumps(coco), encoding="utf-8")
    boxes, classes = load_coco(tmp_path / "ann.json")
    assert boxes == {"a.png": [([2.0, 3.0, 6.0, 8.0], "grieta")]} and classes == ["grieta"]

    img = tmp_path / "b.png"
    Image.new("RGB", (100, 50)).save(img)
    (tmp_path / "b.xml").write_text(
        "<annotation><object><name>tornillo</name><bndbox><xmin>1</xmin><ymin>2</ymin>"
        "<xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    assert load_voc(img) == [([1.0, 2.0, 30.0, 40.0], "tornillo")]
    (tmp_path / "b.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    [(box, name)] = load_yolo(img, ["tuerca"], (100, 50))
    assert name == "tuerca" and box == pytest.approx([40.0, 15.0, 60.0, 35.0])


def test_centernet_targets_decode_roundtrip() -> None:
    boxes = torch.tensor([[8.0, 8.0, 24.0, 20.0]])
    heat, reg, mask, _ = centernet_targets(
        [{"boxes": boxes, "labels": torch.tensor([0])}], 1, 8, 8, 4.0
    )
    assert heat.max() == 1 and mask.sum() == 1
    logits = torch.logit(heat.clamp(1e-4, 1 - 1e-4))
    out = torch.cat([logits, reg], dim=1)
    [det] = decode_centernet(out, 1, 32)
    assert det["boxes"][0].tolist() == pytest.approx(boxes[0].tolist(), abs=0.5)
    assert det["scores"][0] > 0.99


def test_dice_ce_and_ctc_helpers() -> None:
    mask = torch.zeros(2, 8, 8, dtype=torch.long)
    mask[:, 2:5, 2:5] = 1
    perfect = torch.nn.functional.one_hot(mask, 2).permute(0, 3, 1, 2).float() * 20
    assert DiceCELoss()(perfect, mask) < DiceCELoss()(torch.zeros(2, 2, 8, 8), mask)
    logits = torch.full((1, 6, 4), -10.0)
    for t, k in enumerate([1, 1, 0, 2, 0, 3]):
        logits[0, t, k] = 10
    assert greedy_ctc(logits, ["a", "b", "c"]) == ["abc"]
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1


@pytest.mark.parametrize(
    ("task", "shape", "classes"),
    [
        (TaskType.OBJECT_DETECTION, [3, 32, 32], 1),
        (TaskType.SEGMENTATION, [3, 32, 32], 2),
        (TaskType.OCR, [1, 32, 64], 12),
    ],
)
def test_vision_templates_forward_and_code(task: TaskType, shape: list[int], classes: int) -> None:
    spec = vision_task_template(task, num_classes=classes, image_shape=shape)
    assert validate_archspec(spec).valid, validate_archspec(spec).feedback()
    built = build_model(spec)
    x = torch.randn(2, *shape)
    out = built.model(x)
    expected = {
        TaskType.OBJECT_DETECTION: (2, classes + 4, 8, 8),
        TaskType.SEGMENTATION: (2, classes, 32, 32),
        TaskType.OCR: (2, 16, classes),
    }[task]
    assert tuple(out.shape) == expected
    ns: dict[str, Any] = {"__name__": "gen"}
    exec(compile(archspec_to_code(spec), "<gen>", "exec"), ns)  # noqa: S102 - código propio
    gen = ns["Model"]()
    gen.load_state_dict({k.removeprefix("blocks."): v for k, v in built.model.state_dict().items()})
    built.model.eval()
    gen.eval()
    with torch.no_grad():
        assert torch.allclose(built.model(x), gen(x), atol=1e-5)


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_vt").ensure()


def _prepare(paths: ProjectPaths, src: Path, **kw: Any) -> tuple[Any, DatasetView, Any, Any]:
    v = ingest(paths, IngestRequest(project_id="prj_vt", source=src, **kw))
    view = DatasetView(paths.dataset(v.content_hash))
    card = profile_dataset(view)
    fitted = fit_pipeline(propose_pipeline(card), view.read("train"), view.files_dir)
    return v, view, card, fitted


def test_detection_uc04(paths: ProjectPaths, fixtures_dir: Path) -> None:
    _, view, card, fitted = _prepare(
        paths, fixtures_dir / "uc04_defects", task=TaskType.OBJECT_DETECTION
    )
    assert view.task is TaskType.OBJECT_DETECTION and view.classes == ["defecto"]
    vt = card.vision_task
    assert vt is not None and 0 < (vt.images_without_objects or 0) < card.profiled_samples
    assert fitted.classes == ["defecto"]
    ds = make_dataset(view, fitted, "train", train=True)
    x, target = ds[0]
    assert x.shape == (3, 32, 32) and target["boxes"].shape[1] == 4
    rec = recommend(card, fitted)
    assert rec.template == "centernet_small" and validate_archspec(rec.spec).valid


def test_segmentation_uc05(paths: ProjectPaths, fixtures_dir: Path) -> None:
    v, view, card, fitted = _prepare(paths, fixtures_dir / "uc05_masks")
    assert view.task is TaskType.SEGMENTATION and v.num_samples == 60
    assert card.vision_task is not None and 0 < (card.vision_task.foreground_fraction or 0) < 0.5
    assert fitted.classes == ["fondo", "objeto"]
    x, mask = make_dataset(view, fitted, "train", train=True)[0]
    assert (
        x.shape == (3, 32, 32) and mask.shape == (32, 32) and set(mask.unique().tolist()) <= {0, 1}
    )
    assert recommend(card, fitted).template == "unet_small"


def test_ocr_uc06(paths: ProjectPaths, fixtures_dir: Path) -> None:
    v, view, card, fitted = _prepare(paths, fixtures_dir / "uc06_ocr")
    assert view.task is TaskType.OCR and v.num_samples == 200
    assert fitted.classes == ["-", *"0123456789"]
    x, ids, length = make_dataset(view, fitted, "train", train=True)[0]
    assert x.shape[0:2] == (1, 32) and int(length) == 13
    assert (ids[:13] > 0).all() and (ids[13:] == 0).all()
    rec = recommend(card, fitted)
    assert rec.template == "crnn" and rec.spec.task.num_classes == 12
    assert np.isclose(x.shape[2] % 4, 0)


def test_detection_evaluate_single_class() -> None:
    """Regresión: con una sola clase torchmetrics devuelve tensores escalares."""
    from perceptron.tasks.base import Predictions
    from perceptron.tasks.vision import DetectionAdapter

    box = torch.tensor([[4.0, 4.0, 20.0, 20.0]])
    preds = Predictions(
        y_true=None,
        y_pred=np.array([1]),
        extra={
            "preds": [{"boxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}],
            "targets": [{"boxes": box, "labels": torch.tensor([0])}],
        },
    )
    spec = vision_task_template(TaskType.OBJECT_DETECTION, num_classes=1, image_shape=[3, 32, 32])

    from types import SimpleNamespace

    pipeline = SimpleNamespace(classes=["defecto"])
    rep = DetectionAdapter().evaluate(preds, spec, pipeline)  # type: ignore[arg-type]
    assert rep.metrics["map_50"] == pytest.approx(1.0)
    assert rep.detail["ap_per_class"] == {"defecto": pytest.approx(1.0)}
