"""Probe checkpoint loading utilities."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from nets import ProbeConv1D, ProbeMLP


def _collect_linear_weights(state: dict) -> list[tuple[int, torch.Tensor]]:
    linear_keys = []
    for key, value in state.items():
        if not str(key).endswith(".weight") or getattr(value, "ndim", None) != 2:
            continue
        match = re.search(r"net\.(\d+)\.weight$", str(key))
        if match is None:
            continue
        linear_keys.append((int(match.group(1)), value))
    linear_keys.sort(key=lambda item: item[0])
    return linear_keys


def infer_hidden_dims_from_state(state: dict) -> tuple[int, list[int], int]:
    linear_keys = _collect_linear_weights(state)
    if not linear_keys:
        raise ValueError("Cannot infer MLP architecture from checkpoint state_dict")

    dims = [int(linear_keys[0][1].shape[1])]
    for _, weight in linear_keys:
        dims.append(int(weight.shape[0]))
    in_dim = dims[0]
    out_dim = dims[-1]
    hidden_dims = dims[1:-1]
    return in_dim, hidden_dims, out_dim


def _is_conv1d_checkpoint(state: dict) -> bool:
    return any(str(k).startswith("conv_net.") for k in state.keys())


def _infer_norm_type_from_state(state: dict, prefix: str) -> str:
    for key in state.keys():
        key = str(key)
        if not key.startswith(prefix):
            continue
        if key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked"):
            return "batchnorm"
    return "layernorm"


def _infer_mlp_norm_type_from_state(state: dict, linear_keys: list[tuple[int, torch.Tensor]]) -> str:
    # BatchNorm carries running statistics in the checkpoint.
    if any(str(k).startswith("net.") and str(k).endswith("running_mean") for k in state.keys()):
        return "batchnorm"

    # LayerNorm carries affine vectors (weight/bias) with index right after each Linear.
    for idx, weight in linear_keys[:-1]:
        out_dim = int(weight.shape[0])
        ln_w = state.get(f"net.{idx + 1}.weight")
        ln_b = state.get(f"net.{idx + 1}.bias")
        if (
            isinstance(ln_w, torch.Tensor)
            and isinstance(ln_b, torch.Tensor)
            and ln_w.ndim == 1
            and ln_b.ndim == 1
            and int(ln_w.shape[0]) == out_dim
            and int(ln_b.shape[0]) == out_dim
        ):
            return "layernorm"

    return "none"


def _infer_mlp_has_dropout(linear_keys: list[tuple[int, torch.Tensor]], norm_type: str) -> bool:
    if len(linear_keys) <= 1:
        return False
    linear_indices = [idx for idx, _ in linear_keys]
    min_stride = min(j - i for i, j in zip(linear_indices, linear_indices[1:]))
    if norm_type in {"layernorm", "batchnorm"}:
        return min_stride >= 4
    return min_stride >= 3


def build_model_from_checkpoint(ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if _is_conv1d_checkpoint(state):
        out_dim = int(state["classifier.weight"].shape[0])
        norm_type = _infer_norm_type_from_state(state, "conv_net.")
        model = ProbeConv1D(n_classes=out_dim, dropout=0.0, norm_type=norm_type).to(device)
        in_dim = None
        hidden_dims = [f"conv1d({norm_type})"]
    else:
        linear_keys = _collect_linear_weights(state)
        if not linear_keys:
            raise ValueError("Cannot infer MLP architecture from checkpoint state_dict")
        in_dim, hidden_dims, out_dim = infer_hidden_dims_from_state(state)
        norm_type = _infer_mlp_norm_type_from_state(state, linear_keys)
        has_dropout = _infer_mlp_has_dropout(linear_keys, norm_type)
        model = ProbeMLP(
            in_dim=in_dim,
            n_classes=out_dim,
            hidden_dims=hidden_dims,
            dropout=0.2 if has_dropout else 0.0,
            norm_type=norm_type,
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, in_dim, hidden_dims, out_dim
