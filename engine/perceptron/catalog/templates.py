"""Plantillas de arquitectura del catálogo (§8), expresadas como ArchSpec."""

from __future__ import annotations

from perceptron.archspec.schema import (
    HP,
    ArchSpec,
    EarlyStopping,
    InputSpec,
    LossSpec,
    Node,
    OptimizerSpec,
    Provenance,
    SchedulerSpec,
    TaskSpec,
    TrainingSpec,
)
from perceptron.domain.enums import Modality, Origin, TaskType

CLASSIFICATION_METRICS = ["accuracy", "f1_macro", "auroc"]
REGRESSION_METRICS = ["mae", "rmse", "r2"]


def _task(task: TaskType, num_classes: int | None) -> TaskSpec:
    if task is TaskType.REGRESSION:
        return TaskSpec(type=task)
    return TaskSpec(type=task, num_classes=num_classes)


def _loss(task: TaskType) -> LossSpec:
    if task is TaskType.REGRESSION:
        return LossSpec(type="mse")
    return LossSpec(
        type="cross_entropy",
        class_weights="auto",
        label_smoothing=HP(hp="label_smoothing", default=0.0),
    )


def _metrics(task: TaskType) -> list[str]:
    return REGRESSION_METRICS if task is TaskType.REGRESSION else CLASSIFICATION_METRICS


def _chain(ids: list[str]) -> list[tuple[str, str]]:
    return list(zip(["input", *ids[:-1]], ids, strict=True))


def _training(epochs: int, patience: int, freeze: int = 0) -> TrainingSpec:
    return TrainingSpec(
        epochs=HP(hp="epochs", default=epochs),
        batch_size="auto",
        early_stopping=EarlyStopping(monitor="val_loss", patience=patience),
        freeze_backbone_epochs=freeze,
    )


def tabular_template(
    template: str,
    *,
    task: TaskType,
    num_classes: int | None,
    num_numeric: int,
    cardinalities: list[int],
    epochs: int = 40,
    rationale: str | None = None,
) -> ArchSpec:
    """`mlp`, `resnet_mlp` o `ft_transformer`."""
    inp = InputSpec(
        kind="tabular",
        num_numeric=num_numeric,
        cardinalities=cardinalities,
        from_pipeline="tabular",
    )
    if template == "mlp":
        nodes = [
            Node(id="features", block="input.tabular", params={"dropout": 0.0}),
            Node(
                id="mlp",
                block="mlp.block",
                params={
                    "hidden": HP(hp="hidden", default=128),
                    "layers": HP(hp="layers", default=2),
                    "dropout": HP(hp="dropout", default=0.1),
                },
            ),
            Node(id="head", block="head.linear"),
        ]
        lr = 1e-3
    elif template == "resnet_mlp":
        nodes = [
            Node(id="features", block="input.tabular", params={"dropout": 0.0}),
            Node(
                id="resmlp",
                block="resnet_mlp.block",
                params={
                    "d": HP(hp="d", default=128),
                    "blocks": HP(hp="blocks", default=2),
                    "dropout": HP(hp="dropout", default=0.1),
                },
            ),
            Node(id="head", block="head.linear"),
        ]
        lr = 1e-3
    elif template == "ft_transformer":
        nodes = [
            Node(
                id="encoder",
                block="ft_transformer.encoder",
                params={
                    "d_token": HP(hp="d_token", default=64),
                    "n_blocks": HP(hp="n_blocks", default=3),
                    "dropout": HP(hp="dropout", default=0.1),
                },
            ),
            Node(id="head", block="head.linear"),
        ]
        lr = 3e-4
    else:
        raise ValueError(f"plantilla tabular desconocida: {template}")
    return ArchSpec(
        name=f"tabular-{template}",
        modality=Modality.TABULAR,
        task=_task(task, num_classes),
        input=inp,
        nodes=nodes,
        edges=_chain([n.id for n in nodes]),
        loss=_loss(task),
        optimizer=OptimizerSpec(
            type="adamw",
            lr=HP(hp="lr", default=lr),
            weight_decay=HP(hp="weight_decay", default=1e-4),
        ),
        scheduler=SchedulerSpec(type="one_cycle"),
        training=_training(epochs, patience=6),
        metrics=_metrics(task),
        provenance=Provenance(origin=Origin.RULES, template=template, rationale=rationale),
    )


def image_template(
    backbone: str,
    *,
    task: TaskType,
    num_classes: int | None,
    image_size: int,
    channels: int = 3,
    pretrained: bool = True,
    freeze_epochs: int = 0,
    epochs: int = 25,
    rationale: str | None = None,
) -> ArchSpec:
    """`small_cnn` o un modelo timm curado (p. ej. `efficientnet_b0`)."""
    nodes: list[Node] = []
    if backbone == "small_cnn":
        nodes.append(
            Node(
                id="encoder",
                block="vision.small_cnn",
                params={
                    "width": HP(hp="width", default=32),
                    "depth": HP(hp="depth", default=3),
                    "dropout": 0.1,
                },
            )
        )
        lr = 1e-3
    else:
        if channels == 1 and pretrained:
            nodes.append(Node(id="stem", block="conv.channel_adapter", params={"out_channels": 3}))
        freeze = f"until_epoch:{freeze_epochs}" if freeze_epochs else "none"
        nodes.append(
            Node(
                id="encoder",
                block="vision.timm_backbone",
                params={"model": backbone, "pretrained": pretrained, "freeze": freeze},
            )
        )
        lr = 3e-4 if pretrained else 1e-3
    nodes += [
        Node(id="pool", block="pool.global_avg"),
        Node(id="drop", block="reg.dropout", params={"p": HP(hp="dropout", default=0.2)}),
        Node(id="head", block="head.linear"),
    ]
    return ArchSpec(
        name=f"image-{backbone}",
        modality=Modality.IMAGE,
        task=_task(task, num_classes),
        input=InputSpec(
            kind="image", shape=[channels, image_size, image_size], from_pipeline="image"
        ),
        nodes=nodes,
        edges=_chain([n.id for n in nodes]),
        loss=_loss(task),
        optimizer=OptimizerSpec(
            type="adamw",
            lr=HP(hp="lr", default=lr),
            weight_decay=HP(hp="weight_decay", default=1e-2),
        ),
        scheduler=SchedulerSpec(type="one_cycle"),
        training=_training(epochs, patience=5, freeze=freeze_epochs),
        metrics=_metrics(task),
        provenance=Provenance(origin=Origin.RULES, template=backbone, rationale=rationale),
    )


def text_template(
    backbone: str,
    *,
    task: TaskType,
    num_classes: int | None,
    max_length: int,
    vocab_size: int | None = None,
    pad_id: int = 0,
    hf_model: str | None = None,
    pretrained: bool = True,
    freeze_epochs: int = 0,
    epochs: int = 20,
    rationale: str | None = None,
) -> ArchSpec:
    """`textcnn`, `bilstm` (vocabulario propio) o `hf` (encoder preentrenado)."""
    inp = InputSpec(
        kind="tokens",
        shape=[max_length],
        vocab_size=vocab_size,
        pad_id=pad_id,
        from_pipeline="text",
    )
    if backbone == "hf":
        freeze = f"until_epoch:{freeze_epochs}" if freeze_epochs else "none"
        model = hf_model or "distilbert-base-multilingual-cased"
        nodes = [
            Node(
                id="encoder",
                block="text.hf_encoder",
                params={"model": model, "pretrained": pretrained, "freeze": freeze},
            ),
            Node(id="drop", block="reg.dropout", params={"p": HP(hp="dropout", default=0.1)}),
        ]
        lr = 3e-5 if pretrained else 1e-4
    elif backbone in ("textcnn", "bilstm"):
        nodes = [
            Node(
                id="embedding",
                block="text.embedding",
                params={"dim": HP(hp="embed_dim", default=128), "dropout": 0.1},
            )
        ]
        if backbone == "textcnn":
            nodes.append(
                Node(
                    id="encoder",
                    block="text.cnn",
                    params={
                        "filters": HP(hp="filters", default=128),
                        "dropout": HP(hp="dropout", default=0.3),
                    },
                )
            )
        else:
            nodes += [
                Node(
                    id="encoder",
                    block="text.bilstm",
                    params={"hidden": HP(hp="hidden", default=128), "dropout": 0.2},
                ),
                Node(id="pool", block="pool.sequence", params={"mode": "max"}),
                Node(id="drop", block="reg.dropout", params={"p": HP(hp="dropout", default=0.3)}),
            ]
        lr = 2e-3
    else:
        raise ValueError(f"plantilla de texto desconocida: {backbone}")
    nodes.append(Node(id="head", block="head.linear"))
    template = f"hf:{hf_model}" if backbone == "hf" else backbone
    return ArchSpec(
        name=f"text-{backbone}",
        modality=Modality.TEXT,
        task=_task(task, num_classes),
        input=inp,
        nodes=nodes,
        edges=_chain([n.id for n in nodes]),
        loss=_loss(task),
        optimizer=OptimizerSpec(
            type="adamw",
            lr=HP(hp="lr", default=lr),
            weight_decay=HP(hp="weight_decay", default=1e-2),
        ),
        scheduler=SchedulerSpec(type="one_cycle"),
        training=_training(epochs, patience=4, freeze=freeze_epochs if backbone == "hf" else 0),
        metrics=_metrics(task),
        provenance=Provenance(origin=Origin.RULES, template=template, rationale=rationale),
    )
