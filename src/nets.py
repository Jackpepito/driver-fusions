"""Neural network definitions for probing on embeddings."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn


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
        use_batchnorm: bool = False,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [])

        dims = [in_dim, *hidden_dims, n_classes]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if is_last:
                continue
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
) -> ProbeMLP:
    """Factory for predefined probe architectures."""
    if hidden_dims:
        return ProbeMLP(
            in_dim=in_dim,
            n_classes=n_classes,
            hidden_dims=_parse_hidden_dims(hidden_dims),
            dropout=dropout,
            use_batchnorm=True,
        )

    if arch == "linear":
        return ProbeMLP(in_dim=in_dim, n_classes=n_classes, hidden_dims=[])
    if arch == "deep":
        return ProbeMLP(
            in_dim=in_dim,
            n_classes=n_classes,
            hidden_dims=[512, 256],
            dropout=dropout,
            use_batchnorm=True,
        )

    raise ValueError(f"Unknown probe architecture: {arch}")
