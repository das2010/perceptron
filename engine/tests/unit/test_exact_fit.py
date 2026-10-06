"""Ajuste exacto de la regresión lineal por mínimos cuadrados (caso «Tabla X»)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from perceptron.catalog.templates import tabular_template
from perceptron.domain.enums import TaskType
from perceptron.training.exact import least_squares_head
from perceptron.training.module import PerceptronModule

RULE = 3 ** (1 / 5)


def _loader(x: torch.Tensor, y: torch.Tensor) -> DataLoader[tuple[torch.Tensor, ...]]:
    empty = torch.zeros(len(x), 0, dtype=torch.long)
    return DataLoader(TensorDataset(x, empty, y), batch_size=64, shuffle=False)


def test_linear_head_gets_the_exact_solution() -> None:
    spec = tabular_template(
        "linear", task=TaskType.REGRESSION, num_classes=None, num_numeric=1, cardinalities=[]
    )
    module = PerceptronModule(spec, pretrained_allowed=False)
    # Entradas y objetivo estandarizados, como los entrega el pipeline.
    raw = torch.arange(3, 5001, 7, dtype=torch.float32)
    x = ((raw - raw.mean()) / raw.std()).reshape(-1, 1)
    y_raw = raw * RULE
    y = (y_raw - y_raw.mean()) / y_raw.std()
    # Sin entrenar (pesos al azar, BatchNorm con estadísticas iniciales): igual queda exacto.
    assert least_squares_head(module.model, _loader(x, y))
    module.eval()
    with torch.no_grad():
        pred = module.model(x, torch.zeros(len(x), 0, dtype=torch.long)).squeeze(-1)
    assert torch.max(torch.abs(pred - y)).item() < 1e-4


def test_only_a_single_linear_layer_is_solved() -> None:
    spec = tabular_template(
        "mlp", task=TaskType.REGRESSION, num_classes=None, num_numeric=1, cardinalities=[]
    )
    module = PerceptronModule(spec, pretrained_allowed=False)
    x = torch.randn(32, 1)
    assert not least_squares_head(module.model, _loader(x, x.squeeze(-1)))
