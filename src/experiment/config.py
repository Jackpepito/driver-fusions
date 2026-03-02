"""Configuration and stage parsing utilities for the experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "driver_policies_experiments.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "workspace_root": "/work/H2020DeciderFicarra/gcapitani",
    "policies": ["A", "B", "C", "D"],
    "recon_modes": ["ORF", "Without_ORF"],
    "models": ["esmc", "fuson"],
    "data": {
        "chimerseq_csv": "/homes/gcapitani/Gene-Fusions/data/ChimerSeq4.csv",
        "census_tsv": "/homes/gcapitani/Gene-Fusions/data/Census_allFri Jun 27 11_25_18 2025.tsv",
        "benchmark_csv": "/homes/gcapitani/driver-fusions/data/benchmark_fusions.csv",
        "evaluation_input_csv": "/work/H2020DeciderFicarra/gcapitani/driver-fusion/fusions_set1_17/fusions_set1_17_reconstructed_sequences.csv",
    },
    "reconstruction": {
        "genome_build": "all",
        "seed": 42,
        "n_samples": None,
        "orffinder_path": "/homes/gcapitani/Gene-Fusions/data/ORFfinder",
    },
    "clustering": {
        "min_seq_id": 0.3,
        "coverage": 0.8,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "seed": 42,
    },
    "training": {
        "pool": "mean",
        "epochs": 2000,
        "patience": 400,
        "seed": 42,
        "probe_arch": ["linear"],
        "probe_hidden_dims": "",
        "probe_dropout": 0.2,
        "lr_scheduler": "plateau",
        "lr_reduce_factor": 0.5,
        "lr_min": 1e-7,
        "train_probe_config": "",
        "grid": {
            "lr": [1e-4, 5e-4],
            "batch_size": [32, 64],
            "noise": [0.0, 0.02],
            "focal_gamma": [1.0, 2.0],
        },
    },
    "wandb": {
        "enabled": False,
        "mode": "offline",
        "project": "driver-fusions-policy-comparison",
        "entity": "",
        "tags": ["policy-comparison", "grid-search"],
        "dir": "/work/H2020DeciderFicarra/gcapitani",
        "log_from_train_probe": True,
    },
    "runtime": {
        "skip_existing": True,
        "enable_external_evaluation": False,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        user_cfg = json.load(handle)
    if not isinstance(user_cfg, dict):
        raise ValueError("Config file must contain a JSON object at top level.")
    return deep_merge(DEFAULT_CONFIG, user_cfg)


def stages_from_arg(value: str) -> set[str]:
    canonical = {"label", "reconstruct", "cluster", "embed", "train", "evaluate", "compare"}
    aliases = {
        "labeling": "label",
        "reconstruction": "reconstruct",
        "clustering": "cluster",
        "embeddings": "embed",
        "embedding": "embed",
        "training": "train",
        "evaluation": "evaluate",
    }

    raw_items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw_items or raw_items == ["all"]:
        return canonical

    out: set[str] = set()
    for item in raw_items:
        if item in aliases:
            item = aliases[item]
        if item not in canonical:
            raise ValueError(f"Unknown stage '{item}'. Valid stages: {sorted(canonical)}")
        out.add(item)
    return out
