"""Neural network definitions for probing on embeddings."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn


class ConvChannelLayerNorm(nn.Module):
    """LayerNorm on channel dimension for Conv1d tensors [B, C, L]."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # [B, C, L] -> [B, L, C] -> LayerNorm(C) -> [B, C, L]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ProbeMLP(nn.Module):
    """Configurable MLP for binary/multiclass probing.

    The architecture is controlled by ``hidden_dims``:
    - []          -> linear probe (single affine layer)
    - [h1, h2...] -> deeper MLP with optional dropout and batchnorm
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 2,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        if norm_type not in {"none", "batchnorm", "layernorm"}:
            raise ValueError(f"Unknown norm_type '{norm_type}'")

        dims = [in_dim, *hidden_dims, n_classes]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if is_last:
                continue
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            elif norm_type == "layernorm":
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProbeConv1D(nn.Module):
    """1D-convolutional probe over embedding vectors treated as 1D signals."""

    def __init__(
        self,
        n_classes: int = 2,
        conv_channels: Sequence[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.2,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        channels = list(conv_channels or [32, 64])
        if not channels:
            raise ValueError("conv_channels must contain at least one channel size")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if norm_type not in {"batchnorm", "layernorm"}:
            raise ValueError(f"Unknown norm_type '{norm_type}'")

        layers: list[nn.Module] = []
        in_channels = 1
        padding = kernel_size // 2
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm1d(out_channels))
            else:
                layers.append(ConvChannelLayerNorm(out_channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        # x: [batch, embedding_dim] -> [batch, 1, embedding_dim]
        x = x.unsqueeze(1)
        x = self.conv_net(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def _parse_hidden_dims(hidden_dims: str | None) -> list[int]:
    if not hidden_dims:
        return []
    return [int(v.strip()) for v in hidden_dims.split(",") if v.strip()]


def build_probe(
    arch: str,
    in_dim: int,
    n_classes: int = 2,
    hidden_dims: str | None = None,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory for predefined probe architectures."""
    if hidden_dims:
        return ProbeMLP(
            in_dim=in_dim,
            n_classes=n_classes,
            hidden_dims=_parse_hidden_dims(hidden_dims),
            dropout=dropout,
            norm_type="layernorm",
        )

    if arch == "linear":
        return ProbeMLP(in_dim=in_dim, n_classes=n_classes, hidden_dims=[], norm_type="none")
    if arch == "deep":
        return ProbeMLP(
            in_dim=in_dim,
            n_classes=n_classes,
            hidden_dims=[512, 256],
            dropout=dropout,
            norm_type="layernorm",
        )
    if arch == "conv1d":
        return ProbeConv1D(
            n_classes=n_classes,
            conv_channels=[32, 64],
            kernel_size=5,
            dropout=dropout,
            norm_type="layernorm",
        )

    raise ValueError(f"Unknown probe architecture: {arch}")
