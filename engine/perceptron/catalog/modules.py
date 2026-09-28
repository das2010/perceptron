"""Implementaciones PyTorch de los bloques del catálogo (§8).

Se mantienen autocontenidas (solo torch/timm): `archspec.to_code` copia su
código fuente al proyecto exportable y a la vista "ver como código".
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn


def embedding_dim(cardinality: int) -> int:
    """Regla usual: min(50, ⌈(n + 1) / 2⌉)."""
    return int(min(50, math.ceil((cardinality + 1) / 2)))


class TabularInput(nn.Module):
    """Numéricas + embeddings de categóricas → vector de features."""

    def __init__(
        self, num_numeric: int, cardinalities: list[int], embed_dim: int | None, dropout: float
    ) -> None:
        super().__init__()
        dims = [embed_dim or embedding_dim(c) for c in cardinalities]
        self.embeddings = nn.ModuleList(
            nn.Embedding(c, d) for c, d in zip(cardinalities, dims, strict=True)
        )
        self.num_norm = nn.BatchNorm1d(num_numeric) if num_numeric else None
        self.dropout = nn.Dropout(dropout)
        self.out_features = num_numeric + sum(dims)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [self.num_norm(x_num)] if self.num_norm is not None else []
        parts += [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        out: torch.Tensor = self.dropout(torch.cat(parts, dim=1))
        return out


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[name]


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int,
        layers: int,
        dropout: float,
        activation: str = "relu",
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        d = in_features
        for _ in range(layers):
            mods.append(nn.Linear(d, hidden))
            if batchnorm:
                mods.append(nn.BatchNorm1d(hidden))
            mods += [_activation(activation), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*mods)
        self.out_features = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class ResidualMLPBlock(nn.Module):
    """ResNet-MLP (Gorishniy et al., 2021): proyección + bloques residuales."""

    def __init__(
        self, in_features: int, d: int, blocks: int, hidden_factor: float, dropout: float
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, d)
        h = int(d * hidden_factor)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm1d(d),
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h, d),
                nn.Dropout(dropout),
            )
            for _ in range(blocks)
        )
        self.out = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU())
        self.out_features = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        out: torch.Tensor = self.out(x)
        return out


class FTTransformer(nn.Module):
    """FT-Transformer (Gorishniy et al., 2021): cada feature es un token; se usa el [CLS]."""

    def __init__(
        self,
        num_numeric: int,
        cardinalities: list[int],
        d_token: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_weight = nn.Parameter(torch.randn(num_numeric, d_token) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(num_numeric, d_token))
        self.cat_embeddings = nn.ModuleList(nn.Embedding(c, d_token) for c in cardinalities)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_token,
            n_heads,
            dim_feedforward=d_token * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.out_features = d_token

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = [x_num.unsqueeze(-1) * self.num_weight + self.num_bias] if x_num.shape[1] else []
        tokens += [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat([self.cls.expand(x_num.shape[0], -1, -1), *tokens], dim=1)
        out: torch.Tensor = self.norm(self.encoder(x)[:, 0])
        return out


class TimmBackbone(nn.Module):
    """Backbone de `timm` sin cabeza: devuelve el mapa de features [B, C, h, w]."""

    def __init__(self, model: str, pretrained: bool, in_chans: int) -> None:
        super().__init__()
        import timm

        body: Any = timm.create_model(
            model, pretrained=pretrained, num_classes=0, global_pool="", in_chans=in_chans
        )
        self.out_channels = int(body.num_features)
        self.body: Any = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.body.forward_features(x)
        if out.dim() == 3:  # ViT: [B, tokens, C] → [B, C, 1, 1] con el token de clase
            out = out[:, 0].unsqueeze(-1).unsqueeze(-1)
        return out


class SmallCNN(nn.Module):
    """CNN compacta desde cero para imágenes chicas o pocos datos."""

    def __init__(self, in_chans: int, width: int, depth: int, dropout: float) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        c = in_chans
        for i in range(depth):
            out = width * 2**i
            mods += [
                nn.Conv2d(c, out, 3, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(),
                nn.Conv2d(out, out, 3, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ]
            c = out
        self.net = nn.Sequential(*mods)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        return out


class GlobalAvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-2, -1))


class Concat(nn.Module):
    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return torch.cat(xs, dim=1)


class Add(nn.Module):
    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out


class TextEmbedding(nn.Module):
    """Ids de tokens [B, L] → vectores [B, L, D] (el índice de padding queda en cero)."""

    def __init__(self, vocab_size: int, dim: int, dropout: float, pad_id: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.dropout(self.embedding(ids))
        return out


class TextCNN(nn.Module):
    """TextCNN (Kim, 2014): convoluciones 1D de varios anchos + max-pooling en el tiempo."""

    def __init__(self, in_dim: int, filters: int, kernel_sizes: list[int], dropout: float) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(in_dim, filters, k, padding=k // 2) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.out_features = filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        pooled = [torch.relu(conv(x)).amax(dim=-1) for conv in self.convs]
        out: torch.Tensor = self.dropout(torch.cat(pooled, dim=1))
        return out


class BiLSTMEncoder(nn.Module):
    """LSTM bidireccional: [B, L, D] → [B, L, 2·hidden]."""

    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        result: torch.Tensor = out
        return result


class SequencePool(nn.Module):
    """[B, L, D] → [B, D] por media, máximo o último paso."""

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "max":
            return x.amax(dim=1)
        if self.mode == "last":
            return x[:, -1]
        return x.mean(dim=1)


class HFTextEncoder(nn.Module):
    """Encoder de Hugging Face (lista curada) con pooling [CLS] o media enmascarada."""

    def __init__(
        self,
        model: str,
        pretrained: bool,
        pad_id: int,
        pooling: str = "cls",
        lora_r: int = 0,
        lora_alpha: int = 16,
    ) -> None:
        super().__init__()
        import transformers

        auto: Any = transformers.AutoModel  # transformers no está completamente tipado
        config: Any = transformers.AutoConfig
        body: Any = (
            auto.from_pretrained(model)
            if pretrained
            else auto.from_config(config.from_pretrained(model))
        )
        self.uses_lora = lora_r > 0
        if self.uses_lora:
            # LoRA (RF-TRN-09): el encoder queda congelado y se entrenan matrices de rango bajo.
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules="all-linear")
            body = get_peft_model(body, cfg)
        self.body: Any = body
        self.pad_id = pad_id
        self.pooling = pooling
        self.out_features = int(body.config.hidden_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != self.pad_id).long()
        hidden: torch.Tensor = self.body(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        return hidden[:, 0]


class FeatureStub(nn.Module):
    """Sustituto sin pesos para inferir shapes en `meta` sin descargar modelos."""

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.out_features, device=x.device)


class RNNEncoder(nn.Module):
    """LSTM/GRU sobre [B, L, C] → estado final [B, hidden]."""

    def __init__(
        self, in_dim: int, hidden: int, layers: int, dropout: float, cell: str = "lstm"
    ) -> None:
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(
            in_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last: torch.Tensor = out[:, -1]
        return last


class TCN(nn.Module):
    """Temporal Convolutional Network: convoluciones causales dilatadas con residuales."""

    def __init__(
        self, in_dim: int, channels: int, levels: int, kernel: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.pads: list[int] = []
        d = in_dim
        for i in range(levels):
            dilation = 2**i
            self.pads.append((kernel - 1) * dilation)
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(d, channels, kernel, dilation=dilation),
                        "skip": nn.Conv1d(d, channels, 1) if d != channels else nn.Identity(),
                        "drop": nn.Dropout(dropout),
                    }
                )
            )
            d = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        for module, pad in zip(self.layers, self.pads, strict=True):
            layer = cast(nn.ModuleDict, module)
            y = layer["conv"](torch.nn.functional.pad(h, (pad, 0)))
            h = torch.relu(layer["drop"](torch.relu(y)) + layer["skip"](h))
        out: torch.Tensor = h.transpose(1, 2)
        return out


class NBeats(nn.Module):
    """N-BEATS genérico (Oreshkin et al., 2020): bloques FC con backcast residual.

    Entrada [B, L, C] (se aplana); salida: pronóstico [B, horizon].
    """

    def __init__(
        self,
        lookback: int,
        channels: int,
        horizon: int,
        hidden: int,
        blocks: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        size = lookback * channels
        self.horizon = horizon
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            mods: list[nn.Module] = []
            d = size
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "fc": nn.Sequential(*mods),
                        "back": nn.Linear(hidden, size),
                        "fore": nn.Linear(hidden, horizon),
                    }
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.flatten(1)
        forecast = torch.zeros(x.shape[0], self.horizon, device=x.device)
        for module in self.blocks:
            block = cast(nn.ModuleDict, module)
            h = block["fc"](residual)
            residual = residual - block["back"](h)
            forecast = forecast + block["fore"](h)
        return forecast


class PatchTST(nn.Module):
    """PatchTST simplificado (Nie et al., 2023): parches temporales + encoder Transformer."""

    def __init__(
        self,
        lookback: int,
        channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (max(lookback, patch_len) - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len * channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 2, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(1, self.patch_len, self.stride)  # [B, P, C, patch]
        tokens = self.embed(patches.flatten(2)) + self.pos
        out: torch.Tensor = self.norm(self.encoder(tokens).mean(dim=1))
        return out


class SeriesAutoencoder(nn.Module):
    """Autoencoder de ventanas [B, L, C] → [B, L, C] (anomalías por error de reconstrucción)."""

    def __init__(
        self, lookback: int, channels: int, hidden: int, latent: int, kind: str = "conv"
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.enc = nn.LSTM(channels, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.dec = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, channels)
        else:
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(lookback * channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent),
                nn.ReLU(),
                nn.Linear(latent, hidden),
                nn.ReLU(),
                nn.Linear(hidden, lookback * channels),
            )
        self.lookback, self.channels = lookback, channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "lstm":
            h, _ = self.enc(x)
            z = self.from_latent(torch.relu(self.to_latent(h[:, -1])))
            d, _ = self.dec(z.unsqueeze(1).expand(-1, self.lookback, -1).contiguous())
            rec: torch.Tensor = self.out(d)
            return rec
        flat: torch.Tensor = self.net(x)
        return flat.view(-1, self.lookback, self.channels)


def _conv_bn(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class CenterNetSmall(nn.Module):
    """Detector CenterNet compacto (Zhou et al., 2019) entrenable desde cero.

    Salida [B, K + 4, H/4, W/4]: K heatmaps de centros (logits), tamaño (w, h) y
    offset (dx, dy) en celdas del mapa de salida.
    """

    def __init__(self, in_chans: int, num_classes: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [_conv_bn(in_chans, width, 2), _conv_bn(width, width * 2, 2)]
        c = width * 2
        for _ in range(max(depth - 2, 0)):
            layers.append(_conv_bn(c, c))
        self.body = nn.Sequential(*layers)
        heat_out = nn.Conv2d(c, num_classes, 1)
        if heat_out.bias is not None:
            nn.init.constant_(heat_out.bias, -2.19)  # prior 0,1 (paper)
        self.heatmap = nn.Sequential(_conv_bn(c, c), heat_out)
        self.regression = nn.Sequential(_conv_bn(c, c), nn.Conv2d(c, 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(x)
        return torch.cat([self.heatmap(f), self.regression(f)], dim=1)


class UNetSmall(nn.Module):
    """U-Net compacta (Ronneberger et al., 2015): [B, C, H, W] → logits [B, K, H, W]."""

    def __init__(self, in_chans: int, num_classes: int, width: int, depth: int) -> None:
        super().__init__()
        self.down = nn.ModuleList()
        c, chans = in_chans, []
        for i in range(depth):
            out = width * 2**i
            self.down.append(nn.Sequential(_conv_bn(c, out), _conv_bn(out, out)))
            chans.append(out)
            c = out
        self.bottleneck = nn.Sequential(_conv_bn(c, c * 2), _conv_bn(c * 2, c * 2))
        c *= 2
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for skip in reversed(chans):
            self.up.append(nn.ConvTranspose2d(c, skip, 2, stride=2))
            self.dec.append(nn.Sequential(_conv_bn(skip * 2, skip), _conv_bn(skip, skip)))
            c = skip
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = torch.nn.functional.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        out: torch.Tensor = self.head(x)
        return out


class CRNN(nn.Module):
    """CRNN (Shi et al., 2015) para OCR de una línea: CNN → BiLSTM → logits por columna.

    Entrada [B, C, 32, W]; salida [B, W/4, vocab] (para CTC, el índice 0 es el blank).
    """

    def __init__(self, in_chans: int, vocab: int, width: int, hidden: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_bn(in_chans, width),
            nn.MaxPool2d(2),
            _conv_bn(width, width * 2),
            nn.MaxPool2d(2),
            _conv_bn(width * 2, width * 4),
            nn.MaxPool2d((2, 1)),
            _conv_bn(width * 4, width * 4),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(width * 4 * 2, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden * 2, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)  # [B, C, H/16, W/4]
        f = f.permute(0, 3, 1, 2).flatten(2)  # [B, W/4, C·H/16]
        seq, _ = self.rnn(f)
        logits: torch.Tensor = self.out(seq)
        return logits


class AudioCRNN(nn.Module):
    """CRNN para audio: convoluciones que reducen solo la frecuencia + GRU sobre el tiempo.

    Entrada: espectrograma [B, 1, bandas, frames]; salida: [B, 4·hidden] (media y máximo
    temporal de la GRU bidireccional). A diferencia del pooling global de una CNN,
    conserva patrones temporales (modulaciones, golpes periódicos).
    """

    def __init__(self, in_chans: int, bins: int, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_chans
        for i in range(3):
            out = width * 2**i
            blocks += [_conv_bn(c, out), nn.MaxPool2d((2, 1))]
            c = out
        self.cnn = nn.Sequential(*blocks)
        self.rnn = nn.GRU(c * (bins // 8), hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out_features = hidden * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)  # [B, C, bins/8, T]
        seq, _ = self.rnn(f.permute(0, 3, 1, 2).flatten(2))  # [B, T, 2·hidden]
        pooled = torch.cat([seq.mean(dim=1), seq.amax(dim=1)], dim=1)
        out: torch.Tensor = self.dropout(pooled)
        return out
