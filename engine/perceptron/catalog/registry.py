"""Registro de bloques del catálogo (SPEC §8).

Cada bloque declara modalidades, tareas, qué tipo de tensor consume y produce,
parámetros con rangos sugeridos (validación y espacio de HPO) y, si usa pesos
preentrenados, su licencia. El LLM (Capa 2) solo puede componer estos bloques.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel
from torch import nn

from perceptron.catalog import modules as m
from perceptron.domain.enums import Modality, TaskType


class TensorKind(StrEnum):
    TABULAR = "tabular"  # par (x_num [B, N], x_cat [B, K])
    FEATURES = "features"  # [B, D]
    FEATURE_MAP = "feature_map"  # [B, C, H, W]
    IMAGE = "image"  # [B, C, H, W] de entrada
    TOKENS = "tokens"  # [B, L] ids enteros
    SEQUENCE = "sequence"  # [B, L, D]


@dataclass(frozen=True)
class TensorSpec:
    kind: TensorKind
    shape: tuple[int, ...]  # sin batch
    num_numeric: int = 0
    cardinalities: tuple[int, ...] = ()
    vocab_size: int = 0
    pad_id: int = 0

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[0]


class ParamSpec(BaseModel):
    type: Literal["int", "float", "bool", "str", "int_list"]
    default: Any = None
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None
    tunable: bool = False
    description: str = ""


class WeightInfo(BaseModel):
    """Pesos preentrenados curados (revisar en la auditoría de licencias, Capa 7)."""

    model: str
    pretrained_tag: str
    license: str
    commercial_ok: bool
    params_m: float
    source: str = "timm"


BuildFn = Callable[[dict[str, Any], list[TensorSpec], "BuildContext"], nn.Module]


@dataclass
class BuildContext:
    num_outputs: int
    pretrained_allowed: bool = True
    meta: bool = False  # pasada de inferencia de shapes: no descargar ni cargar pesos


@dataclass(frozen=True)
class BlockSpec:
    key: str
    description: str
    consumes: tuple[TensorKind, ...]
    produces: TensorKind
    build: BuildFn
    params: dict[str, ParamSpec] = field(default_factory=dict)
    modalities: tuple[Modality, ...] = ()  # vacío = todas
    tasks: tuple[TaskType, ...] = ()  # vacío = todas
    multi_input: bool = False
    is_backbone: bool = False
    internal: bool = False  # no se ofrece en la paleta ni al LLM (p. ej. código experto)

    def public(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "consumes": [k.value for k in self.consumes],
            "produces": self.produces.value,
            "modalities": [x.value for x in self.modalities],
            "tasks": [x.value for x in self.tasks],
            "multi_input": self.multi_input,
            "params": {k: v.model_dump() for k, v in self.params.items()},
        }


# ----------------------------------------------------------------- pesos timm curados

TIMM_WEIGHTS: dict[str, WeightInfo] = {
    w.model: w
    for w in [
        WeightInfo(
            model="resnet18",
            pretrained_tag="a1_in1k",
            license="Apache-2.0",
            commercial_ok=True,
            params_m=11.7,
        ),
        WeightInfo(
            model="resnet50",
            pretrained_tag="a1_in1k",
            license="Apache-2.0",
            commercial_ok=True,
            params_m=25.6,
        ),
        WeightInfo(
            model="efficientnet_b0",
            pretrained_tag="ra_in1k",
            license="Apache-2.0",
            commercial_ok=True,
            params_m=5.3,
        ),
        WeightInfo(
            model="mobilenetv3_small_100",
            pretrained_tag="lamb_in1k",
            license="Apache-2.0",
            commercial_ok=True,
            params_m=2.5,
        ),
        WeightInfo(
            model="mobilenetv3_large_100",
            pretrained_tag="ra_in1k",
            license="Apache-2.0",
            commercial_ok=True,
            params_m=5.5,
        ),
        WeightInfo(
            model="convnext_tiny",
            pretrained_tag="fb_in1k",
            license="MIT",
            commercial_ok=True,
            params_m=28.6,
        ),
        WeightInfo(
            model="vit_small_patch16_224",
            pretrained_tag="augreg_in21k_ft_in1k",
            license="Apache-2.0",
            commercial_ok=True,
            params_m=22.1,
        ),
        WeightInfo(
            model="deit_small_patch16_224",
            pretrained_tag="fb_in1k",
            license="Apache-2.0",
            commercial_ok=True,
            params_m=22.1,
        ),
    ]
}


class TextModelInfo(BaseModel):
    """Encoders de texto de Hugging Face curados (revisar en la auditoría de Capa 7)."""

    model: str
    hidden_size: int
    license: str
    commercial_ok: bool
    params_m: float
    languages: list[str]


HF_TEXT_MODELS: dict[str, TextModelInfo] = {
    t.model: t
    for t in [
        TextModelInfo(
            model="distilbert-base-multilingual-cased",
            hidden_size=768,
            license="Apache-2.0",
            commercial_ok=True,
            params_m=134,
            languages=["multi"],
        ),
        TextModelInfo(
            model="FacebookAI/xlm-roberta-base",
            hidden_size=768,
            license="MIT",
            commercial_ok=True,
            params_m=278,
            languages=["multi"],
        ),
        TextModelInfo(
            model="dccuchile/bert-base-spanish-wwm-cased",
            hidden_size=768,
            license="CC-BY-4.0",
            commercial_ok=True,
            params_m=110,
            languages=["es"],
        ),
        TextModelInfo(
            model="hf-internal-testing/tiny-random-bert",
            hidden_size=32,
            license="Apache-2.0",
            commercial_ok=True,
            params_m=0.1,
            languages=["test"],
        ),
    ]
}
DEFAULT_HF_TEXT_MODEL = "distilbert-base-multilingual-cased"


# ----------------------------------------------------------------- builders


def _single(inputs: list[TensorSpec]) -> TensorSpec:
    if len(inputs) != 1:
        raise ValueError(f"el bloque espera 1 entrada y recibió {len(inputs)}")
    return inputs[0]


def _build_tabular_input(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    return m.TabularInput(t.num_numeric, list(t.cardinalities), p.get("embed_dim"), p["dropout"])


def _build_ft(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    if p["d_token"] % p["n_heads"]:
        raise ValueError("d_token debe ser múltiplo de n_heads")
    return m.FTTransformer(
        t.num_numeric,
        list(t.cardinalities),
        p["d_token"],
        p["n_blocks"],
        p["n_heads"],
        p["dropout"],
    )


def _build_mlp(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    return m.MLPBlock(
        _single(inputs).dim, p["hidden"], p["layers"], p["dropout"], p["activation"], p["batchnorm"]
    )


def _build_resmlp(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    return m.ResidualMLPBlock(
        _single(inputs).dim, p["d"], p["blocks"], p["hidden_factor"], p["dropout"]
    )


def _build_timm(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    name = p["model"]
    if name not in TIMM_WEIGHTS:
        raise ValueError(f"modelo timm no curado: {name}")
    pretrained = bool(p["pretrained"]) and ctx.pretrained_allowed
    full = f"{name}.{TIMM_WEIGHTS[name].pretrained_tag}" if pretrained else name
    return m.TimmBackbone(full, pretrained, _single(inputs).channels)


def _build_code(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    from perceptron.sandbox.code import build

    t = _single(inputs)
    spec = {
        "kind": t.kind.value,
        "shape": list(t.shape),
        "num_numeric": t.num_numeric,
        "cardinalities": list(t.cardinalities),
        "vocab_size": t.vocab_size,
        "pad_id": t.pad_id,
    }
    return build(p, spec, ctx.num_outputs)


def _build_small_cnn(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    return m.SmallCNN(_single(inputs).channels, p["width"], p["depth"], p["dropout"])


def _build_adapter(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    return m.ChannelAdapter(_single(inputs).channels, p["out_channels"])


def _build_pool(_: dict[str, Any], inputs: list[TensorSpec], __: BuildContext) -> nn.Module:
    _single(inputs)
    return m.GlobalAvgPool()


def _build_bn(_: dict[str, Any], inputs: list[TensorSpec], __: BuildContext) -> nn.Module:
    t = _single(inputs)
    return nn.BatchNorm1d(t.dim) if t.kind is TensorKind.FEATURES else nn.BatchNorm2d(t.channels)


def _build_dropout(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    _single(inputs)
    return nn.Dropout(p["p"])


def _build_linear(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    out = p.get("out_features") or ctx.num_outputs
    return nn.Linear(_single(inputs).dim, int(out))


def _build_concat(_: dict[str, Any], inputs: list[TensorSpec], __: BuildContext) -> nn.Module:
    if len(inputs) < 2:
        raise ValueError("merge.concat necesita al menos 2 entradas")
    return m.Concat()


def _build_add(_: dict[str, Any], inputs: list[TensorSpec], __: BuildContext) -> nn.Module:
    if len(inputs) < 2 or len({i.shape for i in inputs}) != 1:
        raise ValueError("merge.add necesita ≥ 2 entradas con la misma forma")
    return m.Add()


def _build_text_embedding(
    p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext
) -> nn.Module:
    t = _single(inputs)
    if not t.vocab_size:
        raise ValueError("la entrada de tokens no declara vocab_size")
    return m.TextEmbedding(t.vocab_size, p["dim"], p["dropout"], t.pad_id)


def _build_textcnn(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    return m.TextCNN(t.shape[-1], p["filters"], list(p["kernel_sizes"]), p["dropout"])


def _build_bilstm(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    return m.BiLSTMEncoder(t.shape[-1], p["hidden"], p["layers"], p["dropout"])


def _build_seq_pool(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    _single(inputs)
    return m.SequencePool(p["mode"])


def _build_hf_text(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    t = _single(inputs)
    info = HF_TEXT_MODELS.get(p["model"])
    if info is None:
        raise ValueError(f"encoder de texto no curado: {p['model']}")
    if ctx.meta:
        return m.FeatureStub(info.hidden_size)
    return m.HFTextEncoder(
        info.model, bool(p["pretrained"]) and ctx.pretrained_allowed, t.pad_id, p["pooling"]
    )


def _build_rnn(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    return m.RNNEncoder(t.shape[-1], p["hidden"], p["layers"], p["dropout"], p["cell"])


def _build_tcn(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    return m.TCN(t.shape[-1], p["channels"], p["levels"], p["kernel"], p["dropout"])


def _build_nbeats(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    t = _single(inputs)
    return m.NBeats(
        t.shape[0], t.shape[1], ctx.num_outputs, p["hidden"], p["blocks"], p["layers"], p["dropout"]
    )


def _build_patchtst(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    if p["d_model"] % p["heads"]:
        raise ValueError("d_model debe ser múltiplo de heads")
    patch = min(p["patch_len"], t.shape[0])
    return m.PatchTST(
        t.shape[0],
        t.shape[1],
        patch,
        p["stride"],
        p["d_model"],
        p["heads"],
        p["layers"],
        p["dropout"],
    )


def _build_series_ae(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    return m.SeriesAutoencoder(t.shape[0], t.shape[1], p["hidden"], p["latent"], p["kind"])


def _build_audio_crnn(p: dict[str, Any], inputs: list[TensorSpec], _: BuildContext) -> nn.Module:
    t = _single(inputs)
    if t.shape[1] % 8:
        raise ValueError("la cantidad de bandas debe ser múltiplo de 8")
    return m.AudioCRNN(t.channels, t.shape[1], p["width"], p["hidden"], p["dropout"])


def _build_centernet(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    return m.CenterNetSmall(_single(inputs).channels, ctx.num_outputs, p["width"], p["depth"])


def _build_unet(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    t = _single(inputs)
    if t.shape[1] % 2 ** p["depth"] or t.shape[2] % 2 ** p["depth"]:
        raise ValueError(f"alto y ancho deben ser múltiplos de {2 ** p['depth']}")
    return m.UNetSmall(t.channels, ctx.num_outputs, p["width"], p["depth"])


def _build_crnn(p: dict[str, Any], inputs: list[TensorSpec], ctx: BuildContext) -> nn.Module:
    t = _single(inputs)
    if t.shape[1] != 32:
        raise ValueError("el CRNN espera líneas de 32 px de alto")
    return m.CRNN(t.channels, ctx.num_outputs, p["width"], p["hidden"])


def _p(t: str, default: Any, **kw: Any) -> ParamSpec:
    return ParamSpec(type=t, default=default, **kw)  # type: ignore[arg-type]


_DROPOUT = _p("float", 0.1, low=0.0, high=0.7, tunable=True, description="Probabilidad de dropout")
_T, _I, _TXT = (Modality.TABULAR,), (Modality.IMAGE,), (Modality.TEXT,)
_TS = (Modality.TIMESERIES,)
_IA = (Modality.IMAGE, Modality.AUDIO)  # espectrogramas como imágenes de 1 canal
_FEAT, _FMAP, _SEQ = TensorKind.FEATURES, TensorKind.FEATURE_MAP, TensorKind.SEQUENCE

BLOCKS: dict[str, BlockSpec] = {
    b.key: b
    for b in [
        BlockSpec(
            "input.tabular",
            "Numéricas normalizadas + embeddings de categóricas",
            (TensorKind.TABULAR,),
            _FEAT,
            _build_tabular_input,
            {
                "embed_dim": _p("int", None, low=2, high=64, description="None = regla automática"),
                "dropout": _DROPOUT,
            },
            modalities=_T,
        ),
        BlockSpec(
            "ft_transformer.encoder",
            "FT-Transformer: cada feature es un token (Gorishniy 2021)",
            (TensorKind.TABULAR,),
            _FEAT,
            _build_ft,
            {
                "d_token": _p(
                    "int", 64, low=16, high=256, choices=[32, 64, 96, 128, 192], tunable=True
                ),
                "n_blocks": _p("int", 3, low=1, high=6, tunable=True),
                "n_heads": _p("int", 8, choices=[4, 8]),
                "dropout": _DROPOUT,
            },
            modalities=_T,
            is_backbone=True,
        ),
        BlockSpec(
            "mlp.block",
            "Capas densas con BatchNorm, activación y dropout",
            (_FEAT,),
            _FEAT,
            _build_mlp,
            {
                "hidden": _p("int", 128, low=16, high=1024, log=True, tunable=True),
                "layers": _p("int", 2, low=1, high=6, tunable=True),
                "dropout": _DROPOUT,
                "activation": _p("str", "relu", choices=["relu", "gelu", "silu"]),
                "batchnorm": _p("bool", True),
            },
        ),
        BlockSpec(
            "resnet_mlp.block",
            "MLP residual (ResNet-MLP)",
            (_FEAT,),
            _FEAT,
            _build_resmlp,
            {
                "d": _p("int", 128, low=32, high=512, log=True, tunable=True),
                "blocks": _p("int", 2, low=1, high=8, tunable=True),
                "hidden_factor": _p("float", 2.0, low=1.0, high=4.0),
                "dropout": _DROPOUT,
            },
        ),
        BlockSpec(
            "vision.timm_backbone",
            "Backbone de visión de timm (lista curada con licencias)",
            (TensorKind.IMAGE, _FMAP),
            _FMAP,
            _build_timm,
            {
                "model": _p("str", "efficientnet_b0", choices=sorted(TIMM_WEIGHTS)),
                "pretrained": _p("bool", True),
                "freeze": _p("str", "none", description="none | until_epoch:N"),
            },
            modalities=_IA,
            is_backbone=True,
        ),
        BlockSpec(
            "vision.small_cnn",
            "CNN compacta desde cero (imágenes chicas / pocos datos)",
            (TensorKind.IMAGE, _FMAP),
            _FMAP,
            _build_small_cnn,
            {
                "width": _p("int", 32, low=8, high=128, log=True, tunable=True),
                "depth": _p("int", 3, low=1, high=5, tunable=True),
                "dropout": _p("float", 0.1, low=0.0, high=0.5, tunable=True),
            },
            modalities=_IA,
            is_backbone=True,
        ),
        BlockSpec(
            "conv.channel_adapter",
            "Conv 1×1 para adaptar canales (p. ej. 1 → 3)",
            (TensorKind.IMAGE, _FMAP),
            _FMAP,
            _build_adapter,
            {"out_channels": _p("int", 3, low=1, high=2048)},
        ),
        BlockSpec(
            "seq.rnn",
            "LSTM/GRU sobre la ventana temporal (estado final)",
            (_SEQ,),
            _FEAT,
            _build_rnn,
            {
                "cell": _p("str", "lstm", choices=["lstm", "gru"]),
                "hidden": _p("int", 64, low=8, high=512, log=True, tunable=True),
                "layers": _p("int", 1, low=1, high=4, tunable=True),
                "dropout": _p("float", 0.1, low=0.0, high=0.5, tunable=True),
            },
            modalities=_TS,
        ),
        BlockSpec(
            "seq.tcn",
            "Temporal Convolutional Network (convoluciones causales dilatadas)",
            (_SEQ,),
            _SEQ,
            _build_tcn,
            {
                "channels": _p("int", 32, low=8, high=256, log=True, tunable=True),
                "levels": _p("int", 3, low=1, high=8, tunable=True),
                "kernel": _p("int", 3, low=2, high=7),
                "dropout": _p("float", 0.1, low=0.0, high=0.5, tunable=True),
            },
            modalities=_TS,
        ),
        BlockSpec(
            "seq.nbeats",
            "N-BEATS genérico: bloques FC con backcast residual (pronóstico directo)",
            (_SEQ,),
            _FEAT,
            _build_nbeats,
            {
                "hidden": _p("int", 128, low=16, high=1024, log=True, tunable=True),
                "blocks": _p("int", 3, low=1, high=8, tunable=True),
                "layers": _p("int", 2, low=1, high=4),
                "dropout": _p("float", 0.0, low=0.0, high=0.5, tunable=True),
            },
            tasks=(TaskType.FORECASTING,),
            modalities=_TS,
            is_backbone=False,
        ),
        BlockSpec(
            "seq.patchtst",
            "PatchTST: parches temporales + encoder Transformer",
            (_SEQ,),
            _FEAT,
            _build_patchtst,
            {
                "patch_len": _p("int", 8, low=2, high=64),
                "stride": _p("int", 4, low=1, high=64),
                "d_model": _p("int", 64, low=16, high=256, choices=[32, 64, 128], tunable=True),
                "heads": _p("int", 4, choices=[2, 4, 8]),
                "layers": _p("int", 2, low=1, high=6, tunable=True),
                "dropout": _p("float", 0.1, low=0.0, high=0.5, tunable=True),
            },
            modalities=_TS,
        ),
        BlockSpec(
            "seq.autoencoder",
            "Autoencoder de ventanas (anomalías por error de reconstrucción)",
            (_SEQ,),
            _SEQ,
            _build_series_ae,
            {
                "kind": _p("str", "conv", choices=["conv", "lstm"]),
                "hidden": _p("int", 64, low=8, high=512, log=True, tunable=True),
                "latent": _p("int", 8, low=2, high=128, log=True, tunable=True),
            },
            tasks=(TaskType.ANOMALY_DETECTION,),
            modalities=_TS,
        ),
        BlockSpec(
            "audio.crnn",
            "CRNN de audio: convoluciones en frecuencia + GRU temporal (patrones en el tiempo)",
            (TensorKind.IMAGE,),
            _FEAT,
            _build_audio_crnn,
            {
                "width": _p("int", 16, low=8, high=64, log=True, tunable=True),
                "hidden": _p("int", 64, low=16, high=256, log=True, tunable=True),
                "dropout": _p("float", 0.2, low=0.0, high=0.5, tunable=True),
            },
            modalities=(Modality.AUDIO,),
        ),
        BlockSpec(
            "detection.centernet_small",
            "Detector CenterNet compacto desde cero (heatmap + tamaño + offset)",
            (TensorKind.IMAGE,),
            _FMAP,
            _build_centernet,
            {
                "width": _p("int", 32, low=8, high=128, log=True, tunable=True),
                "depth": _p("int", 4, low=2, high=8, tunable=True),
            },
            tasks=(TaskType.OBJECT_DETECTION,),
            modalities=_I,
        ),
        BlockSpec(
            "seg.unet_small",
            "U-Net compacta desde cero (logits por píxel)",
            (TensorKind.IMAGE,),
            _FMAP,
            _build_unet,
            {
                "width": _p("int", 16, low=8, high=128, log=True, tunable=True),
                "depth": _p("int", 3, low=1, high=5, tunable=True),
            },
            tasks=(TaskType.SEGMENTATION,),
            modalities=_I,
        ),
        BlockSpec(
            "ocr.crnn",
            "CRNN (CNN + BiLSTM) para OCR de una línea con CTC",
            (TensorKind.IMAGE,),
            _SEQ,
            _build_crnn,
            {
                "width": _p("int", 32, low=8, high=128, log=True, tunable=True),
                "hidden": _p("int", 128, low=32, high=512, log=True, tunable=True),
            },
            tasks=(TaskType.OCR,),
            modalities=_I,
        ),
        BlockSpec("pool.global_avg", "Promedio global espacial", (_FMAP,), _FEAT, _build_pool),
        BlockSpec(
            "text.embedding",
            "Embeddings de tokens (vocabulario propio)",
            (TensorKind.TOKENS,),
            _SEQ,
            _build_text_embedding,
            {"dim": _p("int", 128, low=16, high=512, log=True, tunable=True), "dropout": _DROPOUT},
            modalities=_TXT,
        ),
        BlockSpec(
            "text.cnn",
            "TextCNN: convoluciones 1D de varios anchos + max-pooling",
            (_SEQ,),
            _FEAT,
            _build_textcnn,
            {
                "filters": _p("int", 128, low=16, high=512, log=True, tunable=True),
                "kernel_sizes": _p("int_list", [2, 3, 4], description="Anchos de ventana"),
                "dropout": _p("float", 0.3, low=0.0, high=0.7, tunable=True),
            },
            modalities=_TXT,
        ),
        BlockSpec(
            "text.bilstm",
            "LSTM bidireccional sobre la secuencia",
            (_SEQ,),
            _SEQ,
            _build_bilstm,
            {
                "hidden": _p("int", 128, low=16, high=512, log=True, tunable=True),
                "layers": _p("int", 1, low=1, high=3, tunable=True),
                "dropout": _p("float", 0.2, low=0.0, high=0.6, tunable=True),
            },
        ),
        BlockSpec(
            "pool.sequence",
            "Pooling temporal (media, máximo o último paso)",
            (_SEQ,),
            _FEAT,
            _build_seq_pool,
            {"mode": _p("str", "mean", choices=["mean", "max", "last"])},
        ),
        BlockSpec(
            "text.hf_encoder",
            "Encoder preentrenado de Hugging Face (lista curada con licencias)",
            (TensorKind.TOKENS,),
            _FEAT,
            _build_hf_text,
            {
                "model": _p("str", DEFAULT_HF_TEXT_MODEL, choices=sorted(HF_TEXT_MODELS)),
                "pretrained": _p("bool", True),
                "pooling": _p("str", "cls", choices=["cls", "mean"]),
                "freeze": _p("str", "none", description="none | until_epoch:N"),
            },
            modalities=_TXT,
            is_backbone=True,
        ),
        BlockSpec("norm.batchnorm", "Batch normalization", (_FEAT, _FMAP), _FEAT, _build_bn),
        BlockSpec(
            "reg.dropout",
            "Dropout",
            (_FEAT, _FMAP, _SEQ),
            _FEAT,
            _build_dropout,
            {"p": _p("float", 0.2, low=0.0, high=0.8, tunable=True)},
        ),
        BlockSpec(
            "head.linear",
            "Capa lineal de salida (logits o regresión)",
            (_FEAT,),
            _FEAT,
            _build_linear,
            {
                "out_features": _p(
                    "int", None, low=1, high=100_000, description="None = según la tarea"
                )
            },
        ),
        BlockSpec(
            "merge.concat",
            "Concatena features de varias ramas",
            (_FEAT,),
            _FEAT,
            _build_concat,
            multi_input=True,
        ),
        BlockSpec(
            "merge.add",
            "Suma (skip connection)",
            (_FEAT, _FMAP),
            _FEAT,
            _build_add,
            multi_input=True,
        ),
        BlockSpec(
            "code.module",  # CODE_BLOCK
            "Modelo de código experto (RF-ARC-06): solo se construye dentro del sandbox",
            tuple(TensorKind),
            _FEAT,
            _build_code,
            {
                "code_sha256": _p("str", "", description="SHA-256 del código del usuario"),
                "task": _p("str", "classification", choices=["classification", "regression"]),
            },
            tasks=(TaskType.CLASSIFICATION, TaskType.REGRESSION),
            internal=True,
        ),
    ]
}

# Bloque del código experto (RF-ARC-06): se construye solo dentro del sandbox (ADR-0025).
CODE_BLOCK = "code.module"

# El tipo de salida de estos bloques es el mismo que el de su entrada.
SHAPE_PRESERVING = {"norm.batchnorm", "reg.dropout", "merge.add"}


def get_block(key: str) -> BlockSpec:
    try:
        return BLOCKS[key]
    except KeyError:
        raise KeyError(f"bloque desconocido: {key}") from None


def public_blocks() -> list[BlockSpec]:
    return [b for b in BLOCKS.values() if not b.internal]


def blocks_for(modality: Modality, task: TaskType | None = None) -> list[BlockSpec]:
    return [
        b
        for b in public_blocks()
        if (not b.modalities or modality in b.modalities)
        and (not b.tasks or task is None or task in b.tasks)
    ]
