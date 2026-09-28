"""Fine-tuning (RF-TRN-09): descongelado progresivo, LR discriminativo y LoRA."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from perceptron.catalog.templates import image_template, text_template
from perceptron.domain.enums import TaskType
from perceptron.training.callbacks import FreezeBackboneCallback
from perceptron.training.module import PerceptronModule


def _module(**training: object) -> PerceptronModule:
    spec = image_template(
        "resnet18", task=TaskType.CLASSIFICATION, num_classes=3, image_size=64, pretrained=False
    )
    spec = spec.model_copy(update={"training": spec.training.model_copy(update=training)})
    return PerceptronModule(spec, pretrained_allowed=False)


def _trainable(groups: list[list[object]]) -> list[bool]:
    return [all(p.requires_grad for p in g) for g in groups]  # type: ignore[attr-defined]


def test_discriminative_lr_puts_the_backbone_in_its_own_group() -> None:
    module = _module(backbone_lr_mult=0.1)
    groups = module.param_groups(1e-3)
    assert [g["lr"] for g in groups] == [1e-3, pytest.approx(1e-4)]
    total = sum(p.numel() for g in groups for p in g["params"])
    assert total == sum(p.numel() for p in module.parameters())
    assert len(_module().param_groups(1e-3)) == 1  # sin LR discriminativo, un solo grupo


def test_progressive_unfreezing_goes_from_the_output_towards_the_input() -> None:
    module = _module(unfreeze="progressive", freeze_backbone_epochs=1)
    groups = module.backbone_layer_groups()
    assert len(groups) >= 3
    callback = FreezeBackboneCallback(1, progressive=True)
    trainer = SimpleNamespace(current_epoch=0)
    callback.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
    assert not any(_trainable(groups))
    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
    assert _trainable(groups) == [False] * (len(groups) - 1) + [True]
    trainer.current_epoch = 2
    callback.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
    assert _trainable(groups)[-2:] == [True, True] and not _trainable(groups)[0]
    # La cabeza (fuera del backbone) siempre se entrena.
    backbone = {id(p) for g in groups for p in g}
    assert all(p.requires_grad for p in module.parameters() if id(p) not in backbone)


@pytest.mark.network
def test_lora_trains_only_low_rank_adapters() -> None:
    spec = text_template(
        "hf",
        task=TaskType.CLASSIFICATION,
        num_classes=2,
        max_length=8,
        hf_model="hf-internal-testing/tiny-random-bert",
    )
    for node in spec.nodes:
        if node.block == "text.hf_encoder":
            node.params["lora_r"] = 4
    module = PerceptronModule(spec)
    encoder = next(b for b in module.model.blocks.values() if getattr(b, "uses_lora", False))
    trainable = [n for n, p in encoder.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    module.set_backbone_trainable(True)  # el congelado no toca el encoder con LoRA
    assert [n for n, p in encoder.named_parameters() if p.requires_grad] == trainable
