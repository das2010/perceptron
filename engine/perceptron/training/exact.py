"""Ajuste exacto de la capa lineal por mínimos cuadrados (regresión lineal pura).

Caso «Tabla X» (salida = entrada × 3^(1/5), sin ruido): la red lineal quedaba con ±8 unidades
de error. El optimizador oscilaba alrededor de la solución (la `val_loss` saltaba ×100 entre
épocas, en parte por la BatchNorm de las numéricas) y el early stopping se quedaba con una época
afortunada. Una regresión lineal tiene solución cerrada: al terminar el entrenamiento se
calculan los pesos óptimos de la única capa lineal sobre lo que esa capa recibe en evaluación
(entradas ya normalizadas), con todo el train. El resultado es el mínimo exacto, no depende del
learning rate y no oscila.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def least_squares_head(model: nn.Module, loader: Any) -> bool:
    """Reemplaza los pesos de la única `nn.Linear` de `model` por la solución de mínimos
    cuadrados sobre `loader` (batches `(*entradas, y)`). Devuelve si se aplicó."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) != 1:
        return False
    head = linears[0]
    device = head.weight.device
    feats: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    def capture(_m: nn.Module, inputs: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        feats.append(inputs[0].detach().to("cpu", torch.float64))

    hook = head.register_forward_hook(capture)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                *inputs, y = batch
                model(*(t.to(device) for t in inputs))
                ys.append(y.detach().to("cpu", torch.float64).reshape(len(y), -1))
    finally:
        hook.remove()
        model.train(was_training)
    if not feats:
        return False
    x = torch.cat(feats)
    y_all = torch.cat(ys)
    if y_all.shape[1] != head.out_features or x.shape[1] != head.in_features:
        return False
    a = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1) if head.bias is not None else x
    solution = torch.linalg.lstsq(a, y_all).solution
    if not torch.isfinite(solution).all():
        return False
    with torch.no_grad():
        head.weight.copy_(solution[: head.in_features].T.to(head.weight.dtype))
        if head.bias is not None:
            head.bias.copy_(solution[head.in_features].to(head.bias.dtype))
    return True
