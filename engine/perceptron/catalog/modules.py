"""Implementaciones PyTorch de los bloques del catálogo (§8).

Se mantienen autocontenidas (solo torch/timm): `archspec.to_code` copia su
código fuente al proyecto exportable y a la vista "ver como código".
"""

from __future__ import annotations

import math
from typing import Any

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
