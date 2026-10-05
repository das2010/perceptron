"""La `val_loss` se mide sin suavizado de etiquetas: comparable entre trials del HPO.

Con `label_smoothing` en la búsqueda, la entropía cruzada suavizada de validación castigaba a
los trials que suavizaban aunque clasificaran igual, y el análisis del estudio mostraba
`label_smoothing` como el hiperparámetro más importante.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from perceptron.archspec.schema import HP
from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType
from perceptron.training.module import PerceptronModule

LOGITS = torch.tensor([[4.0, -4.0], [-3.0, 3.0], [2.0, -1.0]])
Y = torch.tensor([0, 1, 0])


class _Fixed(nn.Module):
    def forward(self, x_num: torch.Tensor, *_: Any) -> torch.Tensor:
        return LOGITS[: len(x_num)]


def _losses(smoothing: float) -> dict[str, float]:
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=1, cardinalities=[]
    )
    spec.loss.label_smoothing = HP(hp="label_smoothing", default=smoothing)
    spec.loss.class_weights = "none"
    module = PerceptronModule(spec, pretrained_allowed=False)
    module.model = _Fixed()  # type: ignore[assignment]
    module.log = lambda *_a, **_k: None  # type: ignore[method-assign]
    batch = (torch.zeros(3, 1), torch.zeros(3, 0, dtype=torch.long), Y)
    return {
        "train": float(module._step(batch, "train")),
        "val": float(module._step(batch, "val")),
    }


def test_validation_loss_ignores_label_smoothing() -> None:
    plain = float(F.cross_entropy(LOGITS, Y))
    smoothed = _losses(0.2)
    assert smoothed["val"] == pytest.approx(plain)  # comparable con un trial sin suavizado
    assert smoothed["train"] > plain  # el entrenamiento sí suaviza
    assert _losses(0.2)["val"] == pytest.approx(_losses(0.0)["val"])


def test_checkpoint_state_is_unchanged() -> None:
    """La pérdida de validación no es un submódulo: los checkpoints viejos siguen cargando."""
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=1, cardinalities=[]
    )
    spec.loss.label_smoothing = 0.1
    module = PerceptronModule(spec, pretrained_allowed=False)
    assert not any(k.startswith("val_loss") for k in module.state_dict())
