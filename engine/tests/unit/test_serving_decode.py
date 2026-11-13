"""Runtime de inferencia (RF-EXP-02): decodificado en numpy igual al de entrenamiento."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from perceptron.data.pipeline.pipeline import (
    FittedPipeline,
    ImageSpec,
    PipelineSpec,
    TargetSpec,
    vision_task_array,
    vision_task_input,
)
from perceptron.data.series import SeriesConfig, history_columns
from perceptron.domain.enums import Modality, TaskType
from perceptron.serving import runtime
from perceptron.serving.runtime import Prediction, decode_detections, greedy_ctc_texts
from perceptron.tasks import vision


def test_constants_match_training() -> None:
    assert runtime.DECODE_MIN_SCORE == vision.SCORE_THRESHOLD
    assert runtime.MAX_DETECTIONS == vision.MAX_DETECTIONS


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decode_detections_matches_centernet(seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    k, size = 2, 64
    out = torch.randn(3, k + 4, 16, 16, generator=gen) * 2
    out[:, k:] = out[:, k:].abs()  # tamaños positivos, como los que aprende el modelo
    ref = vision.decode_centernet(out, k, size)
    got = decode_detections(out.numpy(), k, size)
    for r, dets in zip(ref, got, strict=True):
        assert len(dets) == len(r["scores"]) > 0
        # Mismas detecciones (ordenadas por puntaje; torch calcula en float32).
        order = np.argsort(-r["scores"].numpy(), kind="stable")
        g = sorted(dets, key=lambda d: -d[1])
        assert [d[0] for d in g] == r["labels"].numpy()[order].tolist()
        assert np.allclose([d[1] for d in g], r["scores"].numpy()[order], atol=1e-5)
        assert np.allclose([d[2] for d in g], r["boxes"].numpy()[order], atol=1e-3)


def test_greedy_ctc_matches_training() -> None:
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(4, 20, 6, generator=gen)
    alphabet = list("abcde")
    assert greedy_ctc_texts(logits.numpy(), alphabet) == vision.greedy_ctc(logits, alphabet)


def test_prediction_details_only_when_present() -> None:
    assert "details" not in Prediction(prediction="x").to_dict()
    d = Prediction(prediction=2, details={"objects": []}).to_dict()
    assert d == {"prediction": 2, "details": {"objects": []}}


def _fitted(task: TaskType, image: ImageSpec | None = None) -> FittedPipeline:
    spec = PipelineSpec(
        modality=Modality.IMAGE, target=TargetSpec(name="y", task=task), image=image
    )
    return FittedPipeline(spec=spec, image_mean=[0.4, 0.5, 0.6], image_std=[0.2, 0.3, 0.4])


def test_vision_task_input_shapes() -> None:
    assert vision_task_input(_fitted(TaskType.CLASSIFICATION)) is None
    assert vision_task_input(_fitted(TaskType.OCR, ImageSpec(size=32, width=96, channels=1))) == (
        (32, 96),
        "L",
    )
    det = _fitted(TaskType.OBJECT_DETECTION, ImageSpec(size=64))
    size, mode = vision_task_input(det) or ((0, 0), "")
    assert size == (64, 64) and mode == "RGB"
    img = Image.new("RGB", (100, 50), (255, 0, 128))
    x = vision_task_array(det, img, size, mode)
    assert x.shape == (3, 64, 64) and x.dtype == np.float32
    assert np.allclose(x[0], (1.0 - 0.4) / 0.2) and np.allclose(x[1], (0.0 - 0.5) / 0.3)


def test_history_columns() -> None:
    fc = SeriesConfig(time_column="t", series_id="sku", target="y", covariates=["precio", "y"])
    assert history_columns(fc) == ["t", "sku", "y", "precio"]
    an = SeriesConfig(
        time_column="t", target="anomalia", task=TaskType.ANOMALY_DETECTION, covariates=["temp"]
    )
    assert history_columns(an) == ["t", "temp"]  # la etiqueta no se conoce al predecir
