"""Métricas de validación de regresión en las unidades del dato (RF-TRN-06, RF-EVL-01).

Con el target estandarizado, el MAE de validación salía en desvíos (0.8) mientras que el de
test salía en unidades reales (316): el informe los marcaba como «inconsistentes».
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType
from perceptron.training.module import PerceptronModule


class _Zero(nn.Module):
    """Predice 0 en la escala estandarizada, es decir, la media del target."""

    def forward(self, x_num: torch.Tensor, *_: Any) -> torch.Tensor:
        return torch.zeros(len(x_num), 1)


def _val_mae(target_scale: tuple[float, float] | None) -> float:
    spec = tabular_template(
        "mlp", task=TaskType.REGRESSION, num_classes=None, num_numeric=1, cardinalities=[]
    )
    module = PerceptronModule(spec, pretrained_allowed=False, target_scale=target_scale)
    module.model = _Zero()  # type: ignore[assignment]
    module.log = lambda *_a, **_k: None  # type: ignore[method-assign]
    x = torch.zeros(2, 1)
    y = torch.tensor([-1.0, 1.0])  # estandarizado: media ± 1 desvío
    module._step((x, torch.zeros(2, 0, dtype=torch.long), y), "val")
    return float(module.val_metrics.compute()["val_mae"])


def test_validation_metrics_are_reported_in_target_units() -> None:
    # Target con media 600 y desvío 300: errores de ±1 desvío son 300 unidades.
    assert _val_mae((600.0, 300.0)) == pytest.approx(300.0)
    assert _val_mae(None) == pytest.approx(1.0)  # sin estandarizar: tal cual
