"""Ajustes previos al entrenamiento (RF-TRN-04): batch size contra OOM y LR finder.

Corren en un `Trainer` descartable, sin callbacks ni checkpoints del run: no ensucian la
historia ni los eventos. Lightning restaura los pesos del modelo al terminar cada búsqueda.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import lightning as L
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

LoaderFactory = Callable[[int, bool], DataLoader[Any]]


class _TunableData(L.LightningDataModule):
    """Loaders reconstruidos con el `batch_size` que prueba el buscador de Lightning."""

    def __init__(self, make: LoaderFactory, batch_size: int) -> None:
        super().__init__()
        self.make = make
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader[Any]:
        return self.make(self.batch_size, True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self.make(self.batch_size, False)


def batch_size_cap(start: int, n_train: int) -> int:
    """Techo por cantidad de datos: con pocos ejemplos, un batch enorme deja pocos pasos de
    optimización por época aunque entre en la GPU."""
    return max(start, min(1024, n_train // 10))


def _tuner_trainer(trainer_kwargs: dict[str, Any], root: Any) -> L.Trainer:
    return L.Trainer(
        **trainer_kwargs,
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[],
        default_root_dir=root,
        num_sanity_val_steps=0,
    )


def tune_batch_size(
    module: L.LightningModule,
    make: LoaderFactory,
    *,
    start: int,
    n_train: int,
    trainer_kwargs: dict[str, Any],
    root: Any,
    max_trials: int = 6,
) -> int:
    """Duplica el batch mientras entra en memoria y, al primer OOM, busca binario entre el
    último que entró y el que falló. Devuelve el mayor batch que entra, con el techo por datos."""
    from lightning.pytorch.tuner import Tuner

    cap = batch_size_cap(start, n_train)
    if cap <= start:
        return start
    data = _TunableData(make, start)
    try:
        found = Tuner(_tuner_trainer(trainer_kwargs, root)).scale_batch_size(
            module, datamodule=data, mode="binsearch", init_val=start, max_trials=max_trials
        )
    except Exception:  # el ajuste nunca impide entrenar: queda el batch por defecto
        logger.warning("no se pudo ajustar el batch size; se usa %d", start, exc_info=True)
        return start
    return max(2, min(int(found or start), cap))


def find_lr(
    module: L.LightningModule,
    make: LoaderFactory,
    *,
    batch_size: int,
    trainer_kwargs: dict[str, Any],
    root: Any,
    num_training: int = 100,
) -> float | None:
    """LR sugerido por el barrido exponencial de Lightning (mayor pendiente de caída de la
    pérdida). None si no se pudo estimar."""
    from lightning.pytorch.tuner import Tuner

    data = _TunableData(make, batch_size)
    try:
        finder = Tuner(_tuner_trainer(trainer_kwargs, root)).lr_find(
            module,
            datamodule=data,
            attr_name="lr_override",
            num_training=num_training,
            update_attr=False,
        )
    except Exception:
        logger.warning("el LR finder falló; se usa el LR de la arquitectura", exc_info=True)
        return None
    suggestion = finder.suggestion() if finder is not None else None
    return float(suggestion) if suggestion else None
