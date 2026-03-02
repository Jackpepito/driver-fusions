#!/usr/bin/env python3
"""Probe training on pre-computed embeddings for driver/non-driver classification."""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import chi2_contingency, mannwhitneyu

from nets import build_probe
from utils import (
    load_embeddings, plot_umap_binary, plot_umap_cancer,
    compute_metrics, print_test_results, print_per_fusion_results, BENCHMARK_GENE_PAIRS,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

DEFAULT_EMB_DIR = "/work/H2020DeciderFicarra/gcapitani/driver-fusion/embeddings"
DEFAULT_OUTPUT = "/work/H2020DeciderFicarra/gcapitani/driver-fusion/results"
DEFAULT_CONFIG_PATH = Path("configs/train_probe.yaml")

EXPECTED_METADATA_COLS = {
    "seed_reads": "Seed_reads_num",
    "junction_reads": "Junction_reads_num",
    "cancer_type": "Cancertype",
    "chr_info": "Chr_info",
    "has_pub": "has_pub",
    "is_recurrent": "is_recurrent",
    "role_cols": [
        "T_kinase",
        "H_kinase",
        "T_oncogene",
        "H_oncogene",
        "H_tumor_suppressor",
        "T_tumor_suppressor",
    ],
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


def format_test_results(metrics: dict, model_name: str, pool: str, title: str) -> str:
    """Create a text block with the same final test metrics shown in stdout."""
    t = metrics
    labels = [0, 1]
    target_names = ["non-driver", "driver"]
    lines = [
        f"{'='*60}",
        title,
        f"{'='*60}",
        f"TEST RESULTS - {model_name.upper()} (pool={pool})",
        f"  Accuracy:  {t['acc']:.4f}",
        f"  F1:        {t['f1']:.4f}",
        f"  AUROC:     {t['auroc']:.4f}",
        f"  Precision: {t['prec']:.4f}",
        f"  Recall:    {t['rec']:.4f}",
        "",
        "Classification report:",
        classification_report(
            t["labels"],
            t["preds"],
            labels=labels,
            target_names=target_names,
            zero_division=0,
        ),
        "Confusion matrix:",
        str(confusion_matrix(t["labels"], t["preds"], labels=labels)),
    ]
    return "\n".join(lines)


def format_benchmark_per_fusion_results(
    preds: np.ndarray,
    labels: np.ndarray,
    fusion_pairs: list[str],
    title: str = "BENCHMARK PER-FUSION BREAKDOWN (BEST AUROC MODEL)",
) -> str:
    """Create a summary block for benchmark gene1-gene2 fusion performance."""
    preds = np.array(preds)
    labels = np.array(labels)
    fusions = np.array(fusion_pairs, dtype=object)
    benchmark_names = sorted({f"{g1}-{g2}" for g1, g2 in BENCHMARK_GENE_PAIRS})

    lines = [
        f"{'='*60}",
        title,
        f"{'='*60}",
        "fusion_pair,n,pred_driver,pred_non_driver,correct,accuracy",
    ]

    for name in benchmark_names:
        mask = fusions == name
        n = int(mask.sum())
        if n == 0:
            lines.append(f"{name},0,0,0,0,N/A")
            continue

        pred_driver = int((preds[mask] == 1).sum())
        pred_non_driver = n - pred_driver
        correct = int((preds[mask] == labels[mask]).sum())
        accuracy = float(correct / n)
        lines.append(
            f"{name},{n},{pred_driver},{pred_non_driver},{correct},{accuracy:.4f}"
        )

    return "\n".join(lines)


def format_benchmark_per_fusion_results_multi(
    labels: np.ndarray,
    fusion_pairs: list[str],
    preds_by_method: dict[str, np.ndarray],
    title: str = "BENCHMARK PER-FUSION BREAKDOWN (LINEAR PROBE VS BASELINES)",
) -> str:
    labels = np.array(labels)
    fusions = np.array(fusion_pairs, dtype=object)
    benchmark_names = sorted({f"{g1}-{g2}" for g1, g2 in BENCHMARK_GENE_PAIRS})
    lines = [
        f"{'='*60}",
        title,
        f"{'='*60}",
        "fusion_pair,method,n,pred_driver,pred_non_driver,correct,accuracy",
    ]

    for name in benchmark_names:
        mask = fusions == name
        n = int(mask.sum())
        if n == 0:
            for method in preds_by_method:
                lines.append(f"{name},{method},0,0,0,0,N/A")
            continue
        for method, preds in preds_by_method.items():
            method_preds = np.array(preds)
            pred_driver = int((method_preds[mask] == 1).sum())
            pred_non_driver = n - pred_driver
            correct = int((method_preds[mask] == labels[mask]).sum())
            accuracy = float(correct / n)
            lines.append(
                f"{name},{method},{n},{pred_driver},{pred_non_driver},{correct},{accuracy:.4f}"
            )
    return "\n".join(lines)


def format_experiment_config(args, device: str, emb_dir: Path, out_dir: Path, run_id: str,
                             dim: int, n_train: int, n_val: int, n_test: int) -> str:
    loss_name = "cross_entropy" if args.focal_gamma == 0 else f"focal_loss(gamma={args.focal_gamma})"
    lines = [
        f"{'='*60}",
        "EXPERIMENT CONFIG",
        f"{'='*60}",
        f"  run_id:            {run_id}",
        f"  model:             {args.model}",
        f"  pool:              {args.pool}",
        f"  probe_arch:        {args.probe_arch}",
        f"  probe_hidden_dims: {args.probe_hidden_dims if args.probe_hidden_dims else '<default>'}",
        f"  probe_dropout:     {args.probe_dropout}",
        f"  loss:              {loss_name}",
        f"  train_noise_std:   {args.train_noise_std}",
        f"  epochs:            {args.epochs}",
        f"  patience:          {args.patience}",
        f"  batch_size:        {args.batch_size}",
        f"  lr:                {args.lr}",
        f"  lr_scheduler:      {args.lr_scheduler}",
        f"  lr_reduce_factor:  {args.lr_reduce_factor}",
        f"  lr_min:            {args.lr_min}",
        f"  seed:              {args.seed}",
        f"  device:            {device}",
        f"  embedding_dim:     {dim}",
        f"  n_train:           {n_train}",
        f"  n_val:             {n_val}",
        f"  n_test:            {n_test}",
        f"  embeddings_dir:    {emb_dir}",
        f"  output_dir:        {out_dir}",
    ]
    return "\n".join(lines)


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except ModuleNotFoundError:
        # Fallback parser for flat "key: value" YAML configs.
        data = {}
        float_re = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
        int_re = re.compile(r"^[+-]?\d+$")
        with open(config_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    raise ValueError(f"Invalid YAML line {lineno} in {config_path}: {raw.rstrip()}")
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if value == "" or value.lower() in {"null", "none", "~"}:
                    parsed = ""
                elif value.lower() in {"true", "false"}:
                    parsed = value.lower() == "true"
                elif (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    parsed = value[1:-1]
                elif int_re.match(value):
                    parsed = int(value)
                elif float_re.match(value):
                    parsed = float(value)
                else:
                    parsed = value
                data[key] = parsed

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping (dict), got: {type(data).__name__}")

    # Accept both "batch-size" and "batch_size" style keys in config files.
    normalized = {str(k).replace("-", "_"): v for k, v in data.items()}
    return normalized


def train_epoch(model, loader, optim, criterion, device, noise_std: float = 0.0):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if noise_std > 0:
            X = X + torch.randn_like(X) * noise_std
        optim.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_split(model, X: torch.Tensor, y: torch.Tensor, criterion, device: str) -> dict:
    """Evaluate one split returning both loss and standard binary metrics."""
    model.eval()
    logits = model(X.to(device))
    y_dev = y.to(device)
    loss = float(criterion(logits, y_dev).item())
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()

    try:
        auroc = float(roc_auc_score(y_np, probs))
    except ValueError:
        auroc = float("nan")

    return {
        "loss": loss,
        "acc": float(accuracy_score(y_np, preds)),
        "f1": float(f1_score(y_np, preds, zero_division=0)),
        "auroc": auroc,
        "prec": float(precision_score(y_np, preds, zero_division=0)),
        "rec": float(recall_score(y_np, preds, zero_division=0)),
        "preds": preds,
        "probs": probs,
        "labels": y_np,
    }


def _safe_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.where(~s.str.lower().isin({"", ".", "nan", "none"}), np.nan)
    return pd.to_numeric(s, errors="coerce")


def _norm_col(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_to_col = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        col = norm_to_col.get(_norm_col(cand))
        if col is not None:
            return col
    return None


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        return pd.Series(np.where(x.isna(), np.nan, (x > 0).astype(float)), index=series.index)

    s = series.astype(str).str.strip().str.lower()
    s = s.where(~s.isin({"", "nan", "none", "."}), np.nan)
    mapping = {
        "1": 1.0, "true": 1.0, "yes": 1.0, "y": 1.0, "t": 1.0,
        "0": 0.0, "false": 0.0, "no": 0.0, "n": 0.0, "f": 0.0,
    }
    out = s.map(mapping)
    num = pd.to_numeric(s, errors="coerce")
    out = out.where(out.notna(), np.where(num.isna(), np.nan, (num > 0).astype(float)))
    return out


def _to_annotation_presence_binary(series: pd.Series) -> pd.Series:
    """Binary presence for annotation-like columns: empty -> 0, non-empty -> 1."""
    s = series.astype(str).str.strip()
    low = s.str.lower()
    missing = low.isin({"", "nan", "none", ".", "0", "false", "no", "n"})
    return pd.Series(np.where(missing, 0.0, 1.0), index=series.index, dtype="float")


def _compute_random_baseline(labels: np.ndarray, seed: int = 42) -> tuple[np.ndarray, float]:
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        return np.array([], dtype=int), 0.5
    # Jeffreys-smoothed prevalence to avoid degenerate p=0 or p=1 on single-class splits.
    p_driver = float((labels.sum() + 0.5) / (n + 1.0))
    rng = np.random.default_rng(seed)
    y_rand = (rng.random(n) < p_driver).astype(int)
    # Ensure random baseline can emit both classes when enough samples exist.
    if n > 1 and np.unique(y_rand).size == 1:
        idx = int(rng.integers(0, n))
        y_rand[idx] = 1 - y_rand[idx]
    return y_rand, p_driver


def _compute_calibration_metrics(labels: np.ndarray, probs: np.ndarray, random_probs: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    rand = np.clip(np.asarray(random_probs, dtype=float), 0.0, 1.0)
    if len(labels) == 0:
        return {
            "brier": float("nan"),
            "calibration_quality": float("nan"),
            "driver_error": float("nan"),
            "non_driver_error": float("nan"),
            "random_brier": float("nan"),
            "random_calibration_quality": float("nan"),
            "brier_improvement_vs_random": float("nan"),
        }

    brier = float(brier_score_loss(labels, probs))
    rand_brier = float(brier_score_loss(labels, rand))
    driver_mask = labels == 1
    non_driver_mask = labels == 0
    driver_error = float(np.mean(1.0 - probs[driver_mask])) if driver_mask.any() else float("nan")
    non_driver_error = float(np.mean(probs[non_driver_mask])) if non_driver_mask.any() else float("nan")
    return {
        "brier": brier,
        "calibration_quality": float(1.0 - brier),
        "driver_error": driver_error,
        "non_driver_error": non_driver_error,
        "random_brier": rand_brier,
        "random_calibration_quality": float(1.0 - rand_brier),
        "brier_improvement_vs_random": float(rand_brier - brier),
    }


def _init_wandb_run(args, run_id: str, out_dir: Path, dim: int, n_train: int, n_val: int, n_test: int):
    if not args.wandb_enabled or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[WARNING] wandb init skipped: {exc}")
        return None

    os.environ["WANDB_DIR"] = args.wandb_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    policy = str(getattr(args, "policy", "")).strip().upper()
    return wandb.init(
        project=args.wandb_project,
        entity=(str(args.wandb_entity).strip() or None),
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        name=run_id,
        tags=tags,
        config={
            "policy": policy,
            "run_id": run_id,
            "model": args.model,
            "pool": args.pool,
            "probe_arch": args.probe_arch,
            "probe_hidden_dims": args.probe_hidden_dims,
            "probe_dropout": args.probe_dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "lr_reduce_factor": args.lr_reduce_factor,
            "lr_min": args.lr_min,
            "focal_gamma": args.focal_gamma,
            "train_noise_std": args.train_noise_std,
            "seed": args.seed,
            "embedding_dim": int(dim),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "embeddings_dir": str(args.embeddings_dir),
            "output_dir": str(out_dir),
        },
        reinit=True,
    )


def _cm_figure(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_non-driver", "pred_driver"])
    ax.set_yticks([0, 1], labels=["true_non-driver", "true_driver"])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _wandb_log_confusion(run, key: str, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    if run is None:
        return
    try:
        import wandb
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig = _cm_figure(cm, title)
        run.log({key: wandb.Image(fig)})
        plt.close(fig)
    except Exception as exc:
        print(f"[WARNING] wandb confusion-matrix log failed for {key}: {exc}")


def _wandb_log_image_path(run, key: str, path: Path) -> None:
    if run is None or path is None or not Path(path).exists():
        return
    try:
        import wandb
        run.log({key: wandb.Image(str(path))})
    except Exception as exc:
        print(f"[WARNING] wandb image log failed for {path}: {exc}")


def _normalize_gene_name(v) -> str:
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", ".", "na", "n/a"}:
        return ""
    return s.upper()


def _extract_gene_pair_from_row(row: pd.Series) -> tuple[str, str]:
    h_gene = ""
    t_gene = ""
    if "H_gene" in row.index:
        h_gene = _normalize_gene_name(row.get("H_gene", ""))
    if "T_gene" in row.index:
        t_gene = _normalize_gene_name(row.get("T_gene", ""))

    if (not h_gene or not t_gene) and "fusion_pair" in row.index:
        pair = str(row.get("fusion_pair", "")).strip()
        if "-" in pair:
            parts = pair.split("-", 1)
            if len(parts) == 2:
                if not h_gene:
                    h_gene = _normalize_gene_name(parts[0])
                if not t_gene:
                    t_gene = _normalize_gene_name(parts[1])
    return h_gene, t_gene


def _build_gene_frequency_baseline(meta_train: dict) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    rows = meta_train.get("metadata_rows") if isinstance(meta_train, dict) else None
    if not rows:
        return {}, pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return {}, pd.DataFrame()

    if "driver" not in df.columns:
        return {}, pd.DataFrame()

    label_str = df["driver"].astype(str).str.strip().str.lower()
    known = label_str.isin({"driver", "non-driver"})
    df = df.loc[known].copy()
    if df.empty:
        return {}, pd.DataFrame()
    df["y_train"] = np.where(label_str.loc[df.index] == "driver", 1, 0)

    gene_counts: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        h_gene, t_gene = _extract_gene_pair_from_row(row)
        y = int(row["y_train"])
        for g in (h_gene, t_gene):
            if not g:
                continue
            if g not in gene_counts:
                gene_counts[g] = {"driver": 0, "non_driver": 0}
            if y == 1:
                gene_counts[g]["driver"] += 1
            else:
                gene_counts[g]["non_driver"] += 1

    if not gene_counts:
        return {}, pd.DataFrame()

    stats_rows = []
    for gene, cnt in gene_counts.items():
        n_driver = int(cnt["driver"])
        n_non = int(cnt["non_driver"])
        stats_rows.append(
            {
                "gene": gene,
                "n_driver": n_driver,
                "n_non_driver": n_non,
                "total": n_driver + n_non,
                "driver_minus_non_driver": n_driver - n_non,
                "prefer_driver": int(n_driver > n_non),
            }
        )
    stats_df = pd.DataFrame(stats_rows).sort_values(
        ["driver_minus_non_driver", "total", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return gene_counts, stats_df


def _predict_gene_frequency_baseline(df_test: pd.DataFrame, gene_counts: dict[str, dict[str, int]]) -> np.ndarray:
    if len(df_test) == 0:
        return np.array([], dtype=int)
    if not gene_counts:
        return np.zeros(len(df_test), dtype=int)

    preds = np.zeros(len(df_test), dtype=int)
    for i, (_, row) in enumerate(df_test.iterrows()):
        h_gene, t_gene = _extract_gene_pair_from_row(row)
        h_pref = False
        t_pref = False
        if h_gene in gene_counts:
            h_pref = gene_counts[h_gene]["driver"] > gene_counts[h_gene]["non_driver"]
        if t_gene in gene_counts:
            t_pref = gene_counts[t_gene]["driver"] > gene_counts[t_gene]["non_driver"]
        preds[i] = 1 if (h_pref or t_pref) else 0
    return preds


def _baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "predicted_driver_rate": float(np.mean(y_pred)),
    }


def _plot_txt_path(plot_path: Path) -> Path:
    return plot_path.with_suffix(".txt")


def _write_text_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _clean_category_value(v) -> str:
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", ".", "na", "n/a"}:
        return "UNKNOWN"
    return s


def _frame_distribution_by_split(meta: dict, labels: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = meta.get("metadata_rows")
    if not rows:
        return pd.DataFrame()
    if len(rows) != len(labels):
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    frame_col = _find_col(df, ["Frame"])
    if frame_col is None:
        return pd.DataFrame()

    tmp = pd.DataFrame(
        {
            "split": split_name,
            "Frame": df[frame_col].map(_clean_category_value),
            "y_true": labels.astype(int),
        }
    )

    in_frame_col = _find_col(df, ["in_frame", "inframe"])
    if in_frame_col is not None:
        in_frame_bin = _to_binary(df[in_frame_col])
        tmp["in_frame_bin"] = in_frame_bin.values
    else:
        tmp["in_frame_bin"] = np.nan

    out = (
        tmp.groupby(["split", "Frame"], dropna=False)
        .agg(
            n=("y_true", "size"),
            n_driver=("y_true", "sum"),
            n_in_frame=("in_frame_bin", lambda s: int((s == 1).sum())),
            n_not_in_frame=("in_frame_bin", lambda s: int((s == 0).sum())),
            n_in_frame_missing=("in_frame_bin", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    out["n_non_driver"] = out["n"] - out["n_driver"]
    out["driver_rate"] = out["n_driver"] / out["n"]
    known_in_frame = out["n_in_frame"] + out["n_not_in_frame"]
    out["in_frame_rate_known"] = np.where(known_in_frame > 0, out["n_in_frame"] / known_in_frame, np.nan)
    return out.sort_values(["split", "n"], ascending=[True, False]).reset_index(drop=True)


def _print_frame_split_distribution(frame_split_df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("FRAME DISTRIBUTION ACROSS SPLITS (TRUE LABELS)")
    print(f"{'='*60}")
    if frame_split_df is None or frame_split_df.empty:
        print("Frame column unavailable in split metadata.")
        return

    for split in ["train", "val", "test"]:
        part = frame_split_df[frame_split_df["split"] == split]
        if part.empty:
            continue
        total = int(part["n"].sum())
        total_drv = int(part["n_driver"].sum())
        total_non = int(part["n_non_driver"].sum())
        print(f"\n[{split}] total={total} | driver={total_drv} | non-driver={total_non}")
        for _, r in part.iterrows():
            print(
                f"  - Frame={r['Frame']}: n={int(r['n'])}, "
                f"driver={int(r['n_driver'])}, non-driver={int(r['n_non_driver'])}, "
                f"driver_rate={float(r['driver_rate']):.4f}, "
                f"in_frame={int(r['n_in_frame'])}, not_in_frame={int(r['n_not_in_frame'])}, "
                f"in_frame_missing={int(r['n_in_frame_missing'])}, "
                f"in_frame_rate_known={float(r['in_frame_rate_known']):.4f}"
            )


def _print_test_frame_breakdown(frame_stats: pd.DataFrame):
    print(f"\n{'='*60}")
    print("TEST FRAME BREAKDOWN (TRUE VS PRED)")
    print(f"{'='*60}")
    if frame_stats is None or frame_stats.empty:
        print("Frame column unavailable in test metadata.")
        return
    for _, r in frame_stats.iterrows():
        print(
            f"  - Frame={r['Frame']}: n={int(r['n'])}, "
            f"acc={float(r['accuracy']):.4f}, "
            f"driver_correct={int(r['driver_correct'])}/{int(r['n_driver'])}, "
            f"non_driver_correct={int(r['non_driver_correct'])}/{int(r['n_non_driver'])}"
        )


def _normalize_fusion_name(v) -> str:
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", ".", "na", "n/a"}:
        return ""
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "-")
    return s.upper()


def _fusion_label_series(df: pd.DataFrame) -> pd.Series:
    for col in ["Fusion_pair", "fusion_pair", "id"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            if (s != "").any():
                return s
    if "H_gene" in df.columns and "T_gene" in df.columns:
        return df["H_gene"].astype(str).str.strip() + "-" + df["T_gene"].astype(str).str.strip()
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index, dtype="object")


def _extract_fusion_name_set(meta: dict) -> set[str]:
    if "metadata_rows" in meta and meta["metadata_rows"]:
        df = pd.DataFrame(meta["metadata_rows"])
        names = _fusion_label_series(df).map(_normalize_fusion_name)
    else:
        names = pd.Series(meta.get("fusion_pairs", []), dtype="object").map(_normalize_fusion_name)
    names = names[(names != "")]
    return set(names.tolist())


def _print_seen_vs_novel_breakdown(seen_stats: pd.DataFrame):
    print(f"\n{'='*60}")
    print("TEST BREAKDOWN: FUSIONS SEEN IN TRAIN VS NOVEL")
    print(f"{'='*60}")
    if seen_stats is None or seen_stats.empty:
        print("Seen/novel fusion breakdown unavailable.")
        return
    for _, r in seen_stats.iterrows():
        print(
            f"  - {r['group']}: n={int(r['n'])}, unique_fusions={int(r['n_unique_fusions'])}, "
            f"acc={float(r['accuracy']):.4f}, "
            f"driver_correct={int(r['driver_correct'])}/{int(r['n_driver'])}, "
            f"non_driver_correct={int(r['non_driver_correct'])}/{int(r['n_non_driver'])}"
        )


def _top_probability_tables(df: pd.DataFrame, prob_col: str, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prob_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d["fusion_name"] = _fusion_label_series(d)
    d[prob_col] = pd.to_numeric(d[prob_col], errors="coerce")
    d = d.dropna(subset=[prob_col])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    keep = ["fusion_name", prob_col]
    for c in ["predicted_label", "true_label", "y_pred", "y_true"]:
        if c in d.columns:
            keep.append(c)
    top_driver = d.sort_values(prob_col, ascending=False).head(n)[keep].reset_index(drop=True)
    top_non_driver = d.sort_values(prob_col, ascending=True).head(n)[keep].reset_index(drop=True)
    return top_driver, top_non_driver


def _print_top_probability_tables(title: str, top_driver: pd.DataFrame, top_non_driver: pd.DataFrame, prob_col: str):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    if len(top_driver) == 0:
        print("No rows available for probability ranking.")
        return
    print(f"Top {len(top_driver)} highest P(driver):")
    for i, r in top_driver.iterrows():
        pred = f", pred={r['predicted_label']}" if "predicted_label" in top_driver.columns else ""
        true = f", true={r['true_label']}" if "true_label" in top_driver.columns else ""
        print(f"  {i+1:2d}. {r['fusion_name']} | p={float(r[prob_col]):.4f}{pred}{true}")
    print(f"Top {len(top_non_driver)} lowest P(driver):")
    for i, r in top_non_driver.iterrows():
        pred = f", pred={r['predicted_label']}" if "predicted_label" in top_non_driver.columns else ""
        true = f", true={r['true_label']}" if "true_label" in top_non_driver.columns else ""
        print(f"  {i+1:2d}. {r['fusion_name']} | p={float(r[prob_col]):.4f}{pred}{true}")


def _reads_summary_lines(df: pd.DataFrame, read_col: str, read_label: str) -> list[str]:
    x = _safe_numeric(df[read_col])
    valid = x.notna() & (x >= 0)
    d = df.loc[valid, ["y_true", "y_pred"]].copy()
    d["reads"] = x[valid].values
    lines = [
        f"{read_label} summary",
        f"column: {read_col}",
        f"n_valid: {len(d)}",
        "",
    ]
    for src_col, src_name in [("y_true", "true label"), ("y_pred", "predicted label")]:
        lines.append(f"[{src_name}]")
        for v, name in [(0, "non-driver"), (1, "driver")]:
            g = d.loc[d[src_col] == v, "reads"]
            if len(g) == 0:
                lines.append(f"  {name}: n=0")
                continue
            lines.append(
                f"  {name}: n={len(g)}, mean={float(g.mean()):.4f}, std={float(g.std(ddof=1)) if len(g)>1 else 0.0:.4f}, median={float(g.median()):.4f}"
            )
        lines.append("")
    return lines


def _rate_by_category_summary_lines(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    top_n: int = 20,
) -> list[str]:
    tmp = df.copy()
    c = tmp[category_col].astype(str).str.strip()
    tmp[category_col] = c.where(~c.str.lower().isin({"", "nan", "none", "."}), np.nan)
    required = [category_col, "y_true", "y_pred", "y_pred_random"]
    has_gene = "y_pred_gene_baseline" in tmp.columns
    if has_gene:
        required.append("y_pred_gene_baseline")
    tmp = tmp.dropna(subset=required)
    lines = [title, f"category_col: {category_col}", f"n_valid: {len(tmp)}", ""]
    if len(tmp) < 20 or tmp[category_col].nunique() < 2:
        lines.append("Not enough data to compute category rates.")
        return lines
    agg_map = {
        "n": ("y_true", "size"),
        "true_driver_rate": ("y_true", "mean"),
        "pred_driver_rate": ("y_pred", "mean"),
        "random_driver_rate": ("y_pred_random", "mean"),
    }
    if has_gene:
        agg_map["gene_baseline_driver_rate"] = ("y_pred_gene_baseline", "mean")
    stats = tmp.groupby(category_col, dropna=False).agg(
        **agg_map
    ).reset_index().sort_values("n", ascending=False).head(top_n)
    if has_gene:
        lines.append("category,n,true_driver_rate,pred_driver_rate,random_driver_rate,gene_baseline_driver_rate")
    else:
        lines.append("category,n,true_driver_rate,pred_driver_rate,random_driver_rate")
    for _, r in stats.iterrows():
        if has_gene:
            lines.append(
                f"{r[category_col]},{int(r['n'])},{float(r['true_driver_rate']):.6f},{float(r['pred_driver_rate']):.6f},{float(r['random_driver_rate']):.6f},{float(r['gene_baseline_driver_rate']):.6f}"
            )
        else:
            lines.append(
                f"{r[category_col]},{int(r['n'])},{float(r['true_driver_rate']):.6f},{float(r['pred_driver_rate']):.6f},{float(r['random_driver_rate']):.6f}"
            )
    return lines


def _plot_probability_distribution(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "Test-set probability distribution",
) -> Path | None:
    if "y_prob_driver" not in df.columns or "y_true" not in df.columns:
        return None
    probs_non = df.loc[df["y_true"] == 0, "y_prob_driver"].dropna().values
    probs_drv = df.loc[df["y_true"] == 1, "y_prob_driver"].dropna().values
    if len(probs_non) == 0 or len(probs_drv) == 0:
        return None

    bins = np.linspace(0.0, 1.0, 31)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs_non, bins=bins, alpha=0.55, label="true non-driver", color="#4c72b0", density=True)
    ax.hist(probs_drv, bins=bins, alpha=0.55, label="true driver", color="#dd5555", density=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted P(driver)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_reads_true_pred(
    df: pd.DataFrame,
    read_col: str,
    read_label: str,
    out_path: Path,
) -> Path | None:
    x = _safe_numeric(df[read_col])
    valid = x.notna() & (x >= 0)
    if valid.sum() < 30:
        return None

    d = df.loc[valid, ["y_true", "y_pred"]].copy()
    d["reads"] = x[valid].values
    if d["y_true"].nunique() < 2 or d["y_pred"].nunique() < 2:
        return None

    max_read = float(np.nanmax(d["reads"]))
    if max_read <= 0:
        max_read = 1.0
    bins = np.linspace(0, max_read, 35)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, label_col, title in [
        (axes[0], "y_true", f"{read_label} by true label"),
        (axes[1], "y_pred", f"{read_label} by predicted label"),
    ]:
        g0 = d.loc[d[label_col] == 0, "reads"].values
        g1 = d.loc[d[label_col] == 1, "reads"].values
        ax.hist(g0, bins=bins, alpha=0.55, density=True, color="#4c72b0", label="non-driver")
        ax.hist(g1, bins=bins, alpha=0.55, density=True, color="#dd5555", label="driver")
        ax.set_title(title)
        ax.set_xlabel(read_label)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        ax.legend()
    axes[0].set_ylabel("Density")
    fig.suptitle(f"Distribution of {read_label}: true vs predicted", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_cancer_breakdown(cancer_stats: pd.DataFrame, out_path: Path, top_n: int = 20) -> Path | None:
    if cancer_stats is None or cancer_stats.empty:
        return None
    cols = {"Cancertype", "n", "predicted_driver_rate", "true_driver_rate"}
    if not cols.issubset(cancer_stats.columns):
        return None

    top = cancer_stats.sort_values("n", ascending=False).head(top_n).copy()
    top = top.iloc[::-1]
    y = np.arange(len(top))
    h = 0.38

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(top))))
    ax.barh(y - h / 2, top["predicted_driver_rate"].values, height=h, color="#dd5555", label="predicted")
    ax.barh(y + h / 2, top["true_driver_rate"].values, height=h, color="#4c72b0", label="true")
    ax.set_yticks(y, labels=top["Cancertype"].astype(str).tolist())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Driver rate")
    ax.set_title(f"Cancer-type breakdown (top {len(top)} by count)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_rate_by_category(
    df: pd.DataFrame,
    category_col: str,
    out_path: Path,
    title: str,
    top_n: int = 20,
) -> Path | None:
    tmp = df.copy()
    c = tmp[category_col].astype(str).str.strip()
    tmp[category_col] = c.where(~c.str.lower().isin({"", "nan", "none", "."}), np.nan)
    required = [category_col, "y_true", "y_pred", "y_pred_random"]
    has_gene = "y_pred_gene_baseline" in tmp.columns
    if has_gene:
        required.append("y_pred_gene_baseline")
    tmp = tmp.dropna(subset=required)
    if len(tmp) < 20:
        return None
    nun = tmp[category_col].nunique()
    if nun < 2:
        return None

    agg_map = {
        "n": ("y_true", "size"),
        "true_driver_rate": ("y_true", "mean"),
        "pred_driver_rate": ("y_pred", "mean"),
        "random_driver_rate": ("y_pred_random", "mean"),
    }
    if has_gene:
        agg_map["gene_baseline_driver_rate"] = ("y_pred_gene_baseline", "mean")
    stats = tmp.groupby(category_col, dropna=False).agg(
        **agg_map
    ).reset_index().sort_values("n", ascending=False).head(top_n)

    plot_specs = [
        ("true_driver_rate", "true", "#4c72b0"),
        ("pred_driver_rate", "predicted", "#dd5555"),
        ("random_driver_rate", "random baseline", "#999999"),
    ]
    if has_gene:
        plot_specs.append(("gene_baseline_driver_rate", "gene baseline", "#55a868"))

    n_bars = len(plot_specs)
    h = 0.18 if n_bars >= 4 else 0.24
    offsets = np.linspace(-(n_bars - 1) * h / 2, (n_bars - 1) * h / 2, n_bars)
    fig_h = max(6, 0.34 * len(stats))

    stats = stats.sort_values("n", ascending=True)
    y = np.arange(len(stats))

    fig, ax = plt.subplots(figsize=(11, fig_h))
    for off, (col, label, color) in zip(offsets, plot_specs):
        ax.barh(y + off, stats[col].values, height=h, color=color, label=label)
    ax.set_yticks(y, labels=stats[category_col].astype(str).tolist())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Driver rate")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_recurrent_by_label_source(
    df: pd.DataFrame,
    recurrent_col: str,
    out_path: Path,
) -> Path | None:
    rec = _to_binary(df[recurrent_col])
    valid = rec.notna()
    if valid.sum() < 30:
        return None

    source_cols = [("True labels", "y_true"), ("Predicted labels", "y_pred"), ("Random baseline", "y_pred_random")]
    if "y_pred_gene_baseline" in df.columns:
        source_cols.append(("Gene baseline", "y_pred_gene_baseline"))

    tmp_keep = [col for _, col in source_cols]
    tmp = df.loc[valid, tmp_keep].copy()
    tmp["recurrent"] = rec[valid].values

    values_drv = []
    values_non = []
    for _, col in source_cols:
        d = tmp.loc[tmp[col] == 1, "recurrent"]
        n = tmp.loc[tmp[col] == 0, "recurrent"]
        values_drv.append(float(d.mean()) if len(d) else np.nan)
        values_non.append(float(n.mean()) if len(n) else np.nan)

    x = np.arange(len(source_cols))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(source_cols)), 5))
    ax.bar(x - w / 2, values_non, width=w, color="#4c72b0", label="non-driver")
    ax.bar(x + w / 2, values_drv, width=w, color="#dd5555", label="driver")
    ax.set_xticks(x, [s[0] for s in source_cols])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recurrent fusion rate")
    ax.set_title("Recurrent fusions by label source")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_model_vs_baselines_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_rand: np.ndarray,
    y_gene: np.ndarray,
    out_path: Path,
) -> tuple[Path, dict, dict, dict]:
    m_model = _baseline_metrics(y_true, y_pred)
    m_rand = _baseline_metrics(y_true, y_rand)
    m_gene = _baseline_metrics(y_true, y_gene)

    keys = ["accuracy", "f1", "precision", "recall", "predicted_driver_rate"]
    labels = ["Accuracy", "F1", "Precision", "Recall", "Driver rate"]
    x = np.arange(len(keys))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, [m_model[k] for k in keys], width=w, color="#dd5555", label="linear probe")
    ax.bar(x, [m_rand[k] for k in keys], width=w, color="#999999", label="random baseline")
    ax.bar(x + w, [m_gene[k] for k in keys], width=w, color="#55a868", label="gene-frequency baseline")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Linear probe vs baselines")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path, m_model, m_rand, m_gene


def _plot_benchmark_per_fusion_accuracy(
    labels: np.ndarray,
    fusion_pairs: list[str],
    preds_by_method: dict[str, np.ndarray],
    out_path: Path,
) -> Path | None:
    labels = np.array(labels)
    fusions = np.array(fusion_pairs, dtype=object)
    benchmark_names = sorted({f"{g1}-{g2}" for g1, g2 in BENCHMARK_GENE_PAIRS})
    rows = []
    for pair in benchmark_names:
        mask = fusions == pair
        n = int(mask.sum())
        if n == 0:
            continue
        for method, preds in preds_by_method.items():
            p = np.array(preds)
            acc = float((p[mask] == labels[mask]).mean())
            rows.append({"fusion_pair": pair, "method": method, "accuracy": acc, "n": n})

    if not rows:
        return None

    df_plot = pd.DataFrame(rows)
    pairs_present = df_plot["fusion_pair"].drop_duplicates().tolist()
    methods = list(preds_by_method.keys())
    x = np.arange(len(pairs_present))
    w = min(0.26, 0.75 / max(1, len(methods)))

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(pairs_present)), 5))
    for i, method in enumerate(methods):
        vals = []
        for pair in pairs_present:
            s = df_plot[(df_plot["fusion_pair"] == pair) & (df_plot["method"] == method)]["accuracy"]
            vals.append(float(s.iloc[0]) if len(s) else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, width=w, label=method)

    ax.set_xticks(x, pairs_present)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Benchmark per-fusion accuracy: linear probe vs baselines")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_gene_role_distribution_true_labels(
    df: pd.DataFrame,
    out_path: Path,
) -> tuple[Path | None, pd.DataFrame]:
    role_specs = [
        ("Kinase", "H_kinase", "T_kinase"),
        ("Oncogene", "H_oncogene", "T_oncogene"),
        ("Tumor suppressor", "H_tumor_suppressor", "T_tumor_suppressor"),
        ("Receptor", "H_receptor", "T_receptor"),
        ("Transcription factor", "H_transcription_factor", "T_transcription_factor"),
    ]

    rows = []
    for role_name, h_col, t_col in role_specs:
        h_col = h_col if h_col in df.columns else None
        t_col = t_col if t_col in df.columns else None
        if h_col is None and t_col is None:
            continue

        vals = []
        if h_col is not None:
            vals.append(_to_annotation_presence_binary(df[h_col]))
        if t_col is not None:
            vals.append(_to_annotation_presence_binary(df[t_col]))
        role_any = pd.concat(vals, axis=1).max(axis=1, skipna=True)
        valid = role_any.notna()
        if valid.sum() < 20:
            continue

        y = df.loc[valid, "y_true"]
        x = role_any.loc[valid]
        drv = x[y == 1]
        non = x[y == 0]
        if len(drv) == 0 or len(non) == 0:
            continue
        rows.append(
            {
                "role": role_name,
                "n_driver": int(len(drv)),
                "n_non_driver": int(len(non)),
                "driver_rate": float(drv.mean()),
                "non_driver_rate": float(non.mean()),
            }
        )

    stats = pd.DataFrame(rows)
    if stats.empty:
        return None, stats

    fig, ax = plt.subplots(figsize=(9, max(5, 0.8 * len(stats))))
    y_pos = np.arange(len(stats))
    h = 0.36
    ax.barh(y_pos - h / 2, stats["non_driver_rate"].values, height=h, color="#4c72b0", label="true non-driver")
    ax.barh(y_pos + h / 2, stats["driver_rate"].values, height=h, color="#dd5555", label="true driver")
    ax.set_yticks(y_pos, labels=stats["role"].tolist())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share with role present (H or T gene)")
    ax.set_title("Gene-role distribution: true driver vs non-driver")
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path, stats


def _plot_top_associations(
    assoc_df: pd.DataFrame,
    out_path: Path,
    title: str,
    score_col: str = "p_value",
    top_n: int = 15,
) -> Path | None:
    if assoc_df is None or assoc_df.empty:
        return None
    if "feature" not in assoc_df.columns or score_col not in assoc_df.columns:
        return None

    plot_df = assoc_df.copy()
    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[score_col])
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(score_col).head(top_n).copy()
    plot_df["score"] = -np.log10(np.clip(plot_df[score_col].values.astype(float), 1e-300, 1.0))
    plot_df = plot_df.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"].astype(str), plot_df["score"], color="#55a868")
    ax.set_xlabel("-log10(p-value)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def analyze_test_predictions(
    meta_test: dict,
    metrics: dict,
    out_dir: Path,
    train_fusion_names: set[str] | None = None,
    meta_train: dict | None = None,
) -> dict:
    """Analyze test-set predictions by metadata and save analysis artifacts."""
    preds = np.array(metrics["preds"])
    probs = np.array(metrics["probs"])
    labels = np.array(metrics["labels"])

    if "metadata_rows" in meta_test and len(meta_test["metadata_rows"]) == len(preds):
        df = pd.DataFrame(meta_test["metadata_rows"])
    else:
        # Backward-compatible fallback for old embeddings without full metadata.
        df = pd.DataFrame({
            "fusion_pair": meta_test.get("fusion_pairs", [""] * len(preds)),
            "Cancertype": meta_test.get("cancer_types", [""] * len(preds)),
        })

    df["y_true"] = labels
    df["y_pred"] = preds
    df["y_prob_driver"] = probs
    df["predicted_label"] = np.where(df["y_pred"] == 1, "driver", "non-driver")
    df["true_label"] = np.where(df["y_true"] == 1, "driver", "non-driver")
    df["is_correct"] = (df["y_pred"] == df["y_true"]).astype(int)
    y_rand, p_driver = _compute_random_baseline(labels, seed=42)
    df["y_pred_random"] = y_rand
    gene_counts, gene_stats_df = _build_gene_frequency_baseline(meta_train or {})
    y_gene = _predict_gene_frequency_baseline(df, gene_counts)
    df["y_pred_gene_baseline"] = y_gene

    pred_csv = out_dir / "test_predictions_with_metadata.csv"
    df.to_csv(pred_csv, index=False)

    gene_baseline_csv = None
    if not gene_stats_df.empty:
        gene_baseline_csv = out_dir / "gene_frequency_baseline_train_stats.csv"
        gene_stats_df.to_csv(gene_baseline_csv, index=False)

    # ── Cancer type breakdown ────────────────────────────────────────────
    cancer_csv = None
    if "Cancertype" in df.columns:
        ct = df.copy()
        c = ct["Cancertype"].astype(str).str.strip()
        ct["Cancertype"] = c.where(~c.str.lower().isin({"", "nan", "none", "."}), "UNKNOWN")
        cancer_stats = ct.groupby("Cancertype", dropna=False).agg(
            n=("y_pred", "size"),
            predicted_driver_rate=("y_pred", "mean"),
            true_driver_rate=("y_true", "mean"),
            accuracy=("is_correct", "mean"),
            mean_driver_probability=("y_prob_driver", "mean"),
        ).reset_index().sort_values("n", ascending=False)
        cancer_csv = out_dir / "test_cancer_type_breakdown.csv"
        cancer_stats.to_csv(cancer_csv, index=False)

    # ── Frame breakdown (accuracy and per-class correctness) ────────────
    frame_csv = None
    frame_stats = pd.DataFrame()
    frame_col = _find_col(df, ["Frame"])
    if frame_col is not None:
        tf = df.copy()
        tf["Frame"] = tf[frame_col].map(_clean_category_value)
        frame_stats = (
            tf.groupby("Frame", dropna=False)
            .agg(
                n=("y_true", "size"),
                n_driver=("y_true", "sum"),
                correct_total=("is_correct", "sum"),
                driver_correct=("y_true", lambda s: int(((s == 1) & (tf.loc[s.index, "y_pred"] == 1)).sum())),
                non_driver_correct=("y_true", lambda s: int(((s == 0) & (tf.loc[s.index, "y_pred"] == 0)).sum())),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        frame_stats["n_non_driver"] = frame_stats["n"] - frame_stats["n_driver"]
        frame_stats["accuracy"] = frame_stats["correct_total"] / frame_stats["n"]
        frame_stats["driver_accuracy"] = np.where(
            frame_stats["n_driver"] > 0,
            frame_stats["driver_correct"] / frame_stats["n_driver"],
            np.nan,
        )
        frame_stats["non_driver_accuracy"] = np.where(
            frame_stats["n_non_driver"] > 0,
            frame_stats["non_driver_correct"] / frame_stats["n_non_driver"],
            np.nan,
        )
        frame_csv = out_dir / "test_frame_breakdown.csv"
        frame_stats.to_csv(frame_csv, index=False)

    # ── Seen-in-train vs novel fusions (test set) ───────────────────────
    seen_vs_train_csv = None
    seen_vs_train_stats = pd.DataFrame()
    if train_fusion_names:
        tf = df.copy()
        tf["fusion_name"] = _fusion_label_series(tf).map(_normalize_fusion_name)
        tf = tf[tf["fusion_name"] != ""].copy()
        if not tf.empty:
            tf["seen_in_train"] = tf["fusion_name"].isin(train_fusion_names)
            seen_vs_train_stats = (
                tf.groupby("seen_in_train", dropna=False)
                .agg(
                    n=("y_true", "size"),
                    n_unique_fusions=("fusion_name", "nunique"),
                    n_driver=("y_true", "sum"),
                    correct_total=("is_correct", "sum"),
                    driver_correct=("y_true", lambda s: int(((s == 1) & (tf.loc[s.index, "y_pred"] == 1)).sum())),
                    non_driver_correct=("y_true", lambda s: int(((s == 0) & (tf.loc[s.index, "y_pred"] == 0)).sum())),
                )
                .reset_index()
            )
            seen_vs_train_stats["group"] = np.where(
                seen_vs_train_stats["seen_in_train"], "seen_in_train", "novel_not_in_train"
            )
            seen_vs_train_stats["n_non_driver"] = seen_vs_train_stats["n"] - seen_vs_train_stats["n_driver"]
            seen_vs_train_stats["accuracy"] = seen_vs_train_stats["correct_total"] / seen_vs_train_stats["n"]
            seen_vs_train_stats["driver_accuracy"] = np.where(
                seen_vs_train_stats["n_driver"] > 0,
                seen_vs_train_stats["driver_correct"] / seen_vs_train_stats["n_driver"],
                np.nan,
            )
            seen_vs_train_stats["non_driver_accuracy"] = np.where(
                seen_vs_train_stats["n_non_driver"] > 0,
                seen_vs_train_stats["non_driver_correct"] / seen_vs_train_stats["n_non_driver"],
                np.nan,
            )
            seen_vs_train_stats = seen_vs_train_stats.sort_values("seen_in_train", ascending=False).reset_index(drop=True)
            seen_vs_train_csv = out_dir / "test_seen_vs_novel_fusion_breakdown.csv"
            seen_vs_train_stats.to_csv(seen_vs_train_csv, index=False)

    # ── Per-sample test predictions sorted by P(driver) ─────────────────
    pred_sorted_txt = None
    pred_rank = df.copy()
    pred_rank["fusion_name"] = _fusion_label_series(pred_rank).map(_normalize_fusion_name)
    if train_fusion_names:
        pred_rank["seen_in_train"] = pred_rank["fusion_name"].isin(train_fusion_names)
        pred_rank["seen_in_train"] = np.where(pred_rank["seen_in_train"], "yes", "no")
    else:
        pred_rank["seen_in_train"] = "unknown"
    pred_rank["y_prob_driver"] = pd.to_numeric(pred_rank["y_prob_driver"], errors="coerce")
    pred_rank = pred_rank.dropna(subset=["y_prob_driver"]).sort_values("y_prob_driver", ascending=False).reset_index(drop=True)
    pred_sorted_txt = out_dir / "test_predictions_ranked_by_p_driver.txt"
    lines = [
        "rank,fusion_name,p_driver,predicted_label,true_label,seen_in_train",
    ]
    for i, r in pred_rank.iterrows():
        lines.append(
            f"{i+1},{r['fusion_name']},{float(r['y_prob_driver']):.6f},{r['predicted_label']},{r['true_label']},{r['seen_in_train']}"
        )
    _write_text_lines(pred_sorted_txt, lines)

    # ── Generic numeric feature associations (predicted driver vs non-driver) ──
    numeric_rows = []
    excluded = {"y_true", "y_pred", "y_prob_driver", "is_correct", "y_pred_random", "y_pred_gene_baseline"}
    for col in df.columns:
        if col in excluded:
            continue
        x = _safe_numeric(df[col])
        valid = x.notna()
        if valid.sum() < 30:
            continue
        sub = df.loc[valid, ["y_pred"]].copy()
        sub["x"] = x[valid].values
        g_drv = sub.loc[sub["y_pred"] == 1, "x"].values
        g_non = sub.loc[sub["y_pred"] == 0, "x"].values
        if len(g_drv) < 8 or len(g_non) < 8:
            continue
        p_val = mannwhitneyu(g_drv, g_non, alternative="two-sided").pvalue
        numeric_rows.append(
            {
                "feature": col,
                "n_pred_driver": len(g_drv),
                "n_pred_non_driver": len(g_non),
                "median_pred_driver": float(np.median(g_drv)),
                "median_pred_non_driver": float(np.median(g_non)),
                "delta_median": float(np.median(g_drv) - np.median(g_non)),
                "p_value": float(p_val),
            }
        )
    numeric_df = pd.DataFrame(numeric_rows).sort_values("p_value") if numeric_rows else pd.DataFrame()
    numeric_csv = out_dir / "test_numeric_feature_associations.csv"
    numeric_df.to_csv(numeric_csv, index=False)

    # ── Generic categorical feature associations ─────────────────────────
    cat_rows = []
    for col in df.columns:
        if col in excluded:
            continue
        s = df[col].astype(str).str.strip()
        s = s.where(~s.str.lower().isin({"", "nan", "none", "."}), np.nan).dropna()
        if len(s) < 30:
            continue
        if s.nunique() < 2 or s.nunique() > 30:
            continue
        tmp = df.loc[s.index, ["y_pred"]].copy()
        tmp[col] = s.values
        tab = pd.crosstab(tmp["y_pred"], tmp[col])
        if tab.shape[0] != 2 or tab.shape[1] < 2:
            continue
        chi2, p_val, _, _ = chi2_contingency(tab.values)
        cat_rows.append(
            {
                "feature": col,
                "n_total": int(tab.values.sum()),
                "n_levels": int(tab.shape[1]),
                "chi2": float(chi2),
                "p_value": float(p_val),
            }
        )
    cat_df = pd.DataFrame(cat_rows).sort_values("p_value") if cat_rows else pd.DataFrame()
    cat_csv = out_dir / "test_categorical_feature_associations.csv"
    cat_df.to_csv(cat_csv, index=False)

    # ── Plots ────────────────────────────────────────────────────────────
    plot_txt_paths = []
    plot_pred_prob = _plot_probability_distribution(df, out_dir / "plot_test_prob_driver_distribution.png")
    if plot_pred_prob is not None:
        probs_non = df.loc[df["y_true"] == 0, "y_prob_driver"].dropna()
        probs_drv = df.loc[df["y_true"] == 1, "y_prob_driver"].dropna()
        txt = _plot_txt_path(plot_pred_prob)
        _write_text_lines(
            txt,
            [
                "Probability distribution summary",
                f"true non-driver: n={len(probs_non)}, mean={float(probs_non.mean()):.6f}, std={float(probs_non.std(ddof=1)) if len(probs_non)>1 else 0.0:.6f}",
                f"true driver: n={len(probs_drv)}, mean={float(probs_drv.mean()):.6f}, std={float(probs_drv.std(ddof=1)) if len(probs_drv)>1 else 0.0:.6f}",
            ],
        )
        plot_txt_paths.append(txt)
    plot_cancer = _plot_cancer_breakdown(cancer_stats if cancer_csv is not None else pd.DataFrame(),
                                         out_dir / "plot_test_cancer_type_breakdown.png")
    if plot_cancer is not None and cancer_csv is not None:
        txt = _plot_txt_path(plot_cancer)
        lines = ["Cancer-type breakdown summary", "cancer_type,n,predicted_driver_rate,true_driver_rate,accuracy,mean_driver_probability"]
        for _, r in cancer_stats.head(20).iterrows():
            lines.append(
                f"{r['Cancertype']},{int(r['n'])},{float(r['predicted_driver_rate']):.6f},{float(r['true_driver_rate']):.6f},{float(r['accuracy']):.6f},{float(r['mean_driver_probability']):.6f}"
            )
        _write_text_lines(txt, lines)
        plot_txt_paths.append(txt)
    plot_num = _plot_top_associations(
        numeric_df,
        out_dir / "plot_test_numeric_associations.png",
        title="Top numeric associations (predicted driver vs non-driver)",
    )
    if plot_num is not None:
        txt = _plot_txt_path(plot_num)
        lines = ["Top numeric associations summary", "feature,p_value,delta_median,n_pred_driver,n_pred_non_driver"]
        for _, r in numeric_df.head(20).iterrows():
            lines.append(
                f"{r['feature']},{float(r['p_value']):.6e},{float(r['delta_median']):.6f},{int(r['n_pred_driver'])},{int(r['n_pred_non_driver'])}"
            )
        _write_text_lines(txt, lines)
        plot_txt_paths.append(txt)
    plot_cat = _plot_top_associations(
        cat_df,
        out_dir / "plot_test_categorical_associations.png",
        title="Top categorical associations (predicted driver vs non-driver)",
    )
    if plot_cat is not None:
        txt = _plot_txt_path(plot_cat)
        lines = ["Top categorical associations summary", "feature,p_value,chi2,n_total,n_levels"]
        for _, r in cat_df.head(20).iterrows():
            lines.append(
                f"{r['feature']},{float(r['p_value']):.6e},{float(r['chi2']):.6f},{int(r['n_total'])},{int(r['n_levels'])}"
            )
        _write_text_lines(txt, lines)
        plot_txt_paths.append(txt)
    plot_baseline, baseline_model, baseline_random, baseline_gene = _plot_model_vs_baselines_metrics(
        labels, preds, y_rand, y_gene, out_dir / "plot_model_vs_baselines.png"
    )
    if plot_baseline is not None:
        txt = _plot_txt_path(plot_baseline)
        _write_text_lines(
            txt,
            [
                "Linear probe vs baselines summary",
                f"random_reference_driver_rate={p_driver:.6f}",
                f"model: accuracy={baseline_model['accuracy']:.6f}, f1={baseline_model['f1']:.6f}, precision={baseline_model['precision']:.6f}, recall={baseline_model['recall']:.6f}, predicted_driver_rate={baseline_model['predicted_driver_rate']:.6f}",
                f"random: accuracy={baseline_random['accuracy']:.6f}, f1={baseline_random['f1']:.6f}, precision={baseline_random['precision']:.6f}, recall={baseline_random['recall']:.6f}, predicted_driver_rate={baseline_random['predicted_driver_rate']:.6f}",
                f"gene_frequency: accuracy={baseline_gene['accuracy']:.6f}, f1={baseline_gene['f1']:.6f}, precision={baseline_gene['precision']:.6f}, recall={baseline_gene['recall']:.6f}, predicted_driver_rate={baseline_gene['predicted_driver_rate']:.6f}",
            ],
        )
        plot_txt_paths.append(txt)

    seed_col = EXPECTED_METADATA_COLS["seed_reads"] if EXPECTED_METADATA_COLS["seed_reads"] in df.columns else None
    junction_col = EXPECTED_METADATA_COLS["junction_reads"] if EXPECTED_METADATA_COLS["junction_reads"] in df.columns else None
    recurrent_col = EXPECTED_METADATA_COLS["is_recurrent"] if EXPECTED_METADATA_COLS["is_recurrent"] in df.columns else None
    cancer_col = EXPECTED_METADATA_COLS["cancer_type"] if EXPECTED_METADATA_COLS["cancer_type"] in df.columns else None
    chr_info_col = EXPECTED_METADATA_COLS["chr_info"] if EXPECTED_METADATA_COLS["chr_info"] in df.columns else None
    has_pub_col = EXPECTED_METADATA_COLS["has_pub"] if EXPECTED_METADATA_COLS["has_pub"] in df.columns else None

    plot_seed_reads = None
    if seed_col is not None:
        plot_seed_reads = _plot_reads_true_pred(
            df, seed_col, "Seed reads", out_dir / "plot_seed_reads_true_vs_pred.png"
        )
        if plot_seed_reads is not None:
            txt = _plot_txt_path(plot_seed_reads)
            _write_text_lines(txt, _reads_summary_lines(df, seed_col, "Seed reads"))
            plot_txt_paths.append(txt)

    plot_junction_reads = None
    if junction_col is not None:
        plot_junction_reads = _plot_reads_true_pred(
            df, junction_col, "Junction reads", out_dir / "plot_junction_reads_true_vs_pred.png"
        )
        if plot_junction_reads is not None:
            txt = _plot_txt_path(plot_junction_reads)
            _write_text_lines(txt, _reads_summary_lines(df, junction_col, "Junction reads"))
            plot_txt_paths.append(txt)

    plot_recurrent = None
    if recurrent_col is not None:
        plot_recurrent = _plot_recurrent_by_label_source(
            df, recurrent_col, out_dir / "plot_recurrent_rate_true_pred_random.png"
        )
        if plot_recurrent is not None:
            rec = _to_binary(df[recurrent_col])
            valid = rec.notna()
            tmp_rec = df.loc[valid, ["y_true", "y_pred", "y_pred_random"]].copy()
            if "y_pred_gene_baseline" in df.columns:
                tmp_rec["y_pred_gene_baseline"] = df.loc[valid, "y_pred_gene_baseline"].values
            tmp_rec["recurrent"] = rec[valid].values
            txt = _plot_txt_path(plot_recurrent)
            lines = ["Recurrent fusion rate summary", "source,non_driver_rate,driver_rate"]
            source_cols = [("true", "y_true"), ("predicted", "y_pred"), ("random", "y_pred_random")]
            if "y_pred_gene_baseline" in tmp_rec.columns:
                source_cols.append(("gene_baseline", "y_pred_gene_baseline"))
            for name, col in source_cols:
                non = tmp_rec.loc[tmp_rec[col] == 0, "recurrent"]
                drv = tmp_rec.loc[tmp_rec[col] == 1, "recurrent"]
                lines.append(
                    f"{name},{float(non.mean()) if len(non) else float('nan'):.6f},{float(drv.mean()) if len(drv) else float('nan'):.6f}"
                )
            _write_text_lines(txt, lines)
            plot_txt_paths.append(txt)

    plot_cancer_rates = None
    if cancer_col is not None:
        plot_cancer_rates = _plot_rate_by_category(
            df,
            cancer_col,
            out_dir / "plot_cancer_type_true_pred_random.png",
            title="Driver-rate by cancer type (true vs predicted vs random vs gene baseline)",
            top_n=20,
        )
        if plot_cancer_rates is not None:
            txt = _plot_txt_path(plot_cancer_rates)
            _write_text_lines(
                txt,
                _rate_by_category_summary_lines(
                    df,
                    cancer_col,
                    "Driver-rate by cancer type (true vs predicted vs random vs gene baseline)",
                    top_n=20,
                ),
            )
            plot_txt_paths.append(txt)

    role_cols = [c for c in EXPECTED_METADATA_COLS["role_cols"] if c in df.columns]

    role_plot_paths = []
    for role_col in role_cols:
        role_binary = _to_annotation_presence_binary(df[role_col])
        tmp = df.copy()
        role_status = pd.Series(pd.NA, index=df.index, dtype="object")
        role_status.loc[role_binary == 1] = f"{role_col}=1"
        role_status.loc[role_binary == 0] = f"{role_col}=0"
        tmp["role_status"] = role_status
        out_path = out_dir / f"plot_role_{role_col.lower()}_true_pred_random.png"
        p = _plot_rate_by_category(
            tmp,
            "role_status",
            out_path,
            title=f"Driver-rate by {role_col} (true vs predicted vs random vs gene baseline)",
            top_n=2,
        )
        if p is not None:
            role_plot_paths.append(p)
            txt = _plot_txt_path(p)
            _write_text_lines(
                txt,
                _rate_by_category_summary_lines(
                    tmp,
                    "role_status",
                    f"Driver-rate by {role_col} (true vs predicted vs random vs gene baseline)",
                    top_n=2,
                ),
            )
            plot_txt_paths.append(txt)

    plot_chr_info = None
    if chr_info_col is not None:
        plot_chr_info = _plot_rate_by_category(
            df,
            chr_info_col,
            out_dir / "plot_chr_info_true_pred_random.png",
            title="Driver-rate by Chr_info (true vs predicted vs random vs gene baseline)",
            top_n=15,
        )
        if plot_chr_info is not None:
            txt = _plot_txt_path(plot_chr_info)
            _write_text_lines(
                txt,
                _rate_by_category_summary_lines(
                    df,
                    chr_info_col,
                    "Driver-rate by Chr_info (true vs predicted vs random vs gene baseline)",
                    top_n=15,
                ),
            )
            plot_txt_paths.append(txt)

    plot_has_pub = None
    if has_pub_col is not None:
        hp = _to_binary(df[has_pub_col])
        tmp = df.copy()
        has_pub_status = pd.Series(pd.NA, index=df.index, dtype="object")
        has_pub_status.loc[hp == 1] = "has_pub=1"
        has_pub_status.loc[hp == 0] = "has_pub=0"
        tmp["has_pub_status"] = has_pub_status
        plot_has_pub = _plot_rate_by_category(
            tmp,
            "has_pub_status",
            out_dir / "plot_has_pub_true_pred_random.png",
            title="Driver-rate by has_pub (true vs predicted vs random vs gene baseline)",
            top_n=2,
        )
        if plot_has_pub is not None:
            txt = _plot_txt_path(plot_has_pub)
            _write_text_lines(
                txt,
                _rate_by_category_summary_lines(
                    tmp,
                    "has_pub_status",
                    "Driver-rate by has_pub (true vs predicted vs random vs gene baseline)",
                    top_n=2,
                ),
            )
            plot_txt_paths.append(txt)

    plot_gene_roles = None
    plot_gene_roles_stats = pd.DataFrame()
    plot_gene_roles, plot_gene_roles_stats = _plot_gene_role_distribution_true_labels(
        df,
        out_dir / "plot_gene_roles_driver_vs_non_driver.png",
    )
    if plot_gene_roles is not None and not plot_gene_roles_stats.empty:
        txt = _plot_txt_path(plot_gene_roles)
        lines = [
            "Gene-role distribution summary (true labels)",
            "role,n_non_driver,n_driver,non_driver_rate,driver_rate",
        ]
        for _, r in plot_gene_roles_stats.iterrows():
            lines.append(
                f"{r['role']},{int(r['n_non_driver'])},{int(r['n_driver'])},{float(r['non_driver_rate']):.6f},{float(r['driver_rate']):.6f}"
            )
        _write_text_lines(txt, lines)
        plot_txt_paths.append(txt)

    top_driver_prob, top_non_driver_prob = _top_probability_tables(df, prob_col="y_prob_driver", n=10)

    top_num = numeric_df.head(8) if len(numeric_df) else pd.DataFrame()
    top_cat = cat_df.head(8) if len(cat_df) else pd.DataFrame()
    cm_random = confusion_matrix(labels, y_rand)
    cm_gene = confusion_matrix(labels, y_gene)

    return {
        "pred_csv": pred_csv,
        "cancer_csv": cancer_csv,
        "frame_csv": frame_csv,
        "frame_stats": frame_stats,
        "seen_vs_train_csv": seen_vs_train_csv,
        "seen_vs_train_stats": seen_vs_train_stats,
        "pred_sorted_txt": pred_sorted_txt,
        "numeric_csv": numeric_csv,
        "cat_csv": cat_csv,
        "plot_pred_prob": plot_pred_prob,
        "plot_cancer": plot_cancer,
        "plot_numeric": plot_num,
        "plot_categorical": plot_cat,
        "plot_baseline": plot_baseline,
        "plot_seed_reads": plot_seed_reads,
        "plot_junction_reads": plot_junction_reads,
        "plot_recurrent": plot_recurrent,
        "plot_cancer_rates": plot_cancer_rates,
        "plot_chr_info": plot_chr_info,
        "plot_has_pub": plot_has_pub,
        "plot_gene_roles": plot_gene_roles,
        "plot_role_paths": role_plot_paths,
        "plot_text_summaries": plot_txt_paths,
        "top_driver_prob": top_driver_prob,
        "top_non_driver_prob": top_non_driver_prob,
        "gene_baseline_train_csv": gene_baseline_csv,
        "baseline_model": baseline_model,
        "baseline_random": baseline_random,
        "baseline_gene": baseline_gene,
        "baseline_random_confusion_matrix": cm_random,
        "baseline_gene_confusion_matrix": cm_gene,
        "baseline_random_driver_rate_reference": p_driver,
        "y_pred_random": y_rand,
        "y_pred_gene_baseline": y_gene,
        "top_numeric": top_num,
        "top_categorical": top_cat,
    }


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()
    yaml_config = load_yaml_config(Path(pre_args.config))

    parser = argparse.ArgumentParser(description="Train probe model for driver classification")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="YAML config file path. CLI flags override YAML values.")
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMB_DIR)
    parser.add_argument("--model", default="esmc", choices=["esmc", "fuson"])
    parser.add_argument("--pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument(
        "--lr-scheduler",
        default="plateau",
        choices=["none", "plateau"],
        help="Learning-rate scheduler strategy. 'plateau' halves LR on validation plateau.",
    )
    parser.add_argument(
        "--lr-reduce-factor",
        type=float,
        default=0.5,
        help="Factor used by plateau scheduler when patience is reached.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-7,
        help="Minimum LR allowed when using a scheduler.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-id",
        default="",
        help="Run identifier for output folder. If empty, uses timestamp.",
    )
    parser.add_argument(
        "--policy",
        default="",
        help="Policy label (e.g. A/B/C/D) to store in W&B config for filtering.",
    )
    parser.add_argument("--probe-arch", default="linear", choices=["linear", "deep", "conv1d"])
    parser.add_argument(
        "--probe-hidden-dims",
        default="",
        help="Comma-separated hidden dims for custom MLP (overrides --probe-arch), e.g. 1024,256,64",
    )
    parser.add_argument("--probe-dropout", type=float, default=0.2)
    parser.add_argument(
        "--train-noise-std",
        type=float,
        default=0.0,
        help="Std dev of Gaussian noise added to training embeddings (0 disables)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma. Set to 0.0 to recover standard cross-entropy.",
    )
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb-mode",
        default="disabled",
        choices=["disabled", "offline", "online"],
        help="W&B mode (disabled/offline/online).",
    )
    parser.add_argument("--wandb-project", default="driver-fusions-policy-comparison")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument(
        "--wandb-tags",
        default="",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb-dir",
        default="/work/H2020DeciderFicarra/gcapitani",
        help="Directory for local W&B files.",
    )

    valid_dests = {a.dest for a in parser._actions}
    unknown_cfg_keys = sorted(k for k in yaml_config if k not in valid_dests)
    if unknown_cfg_keys:
        raise ValueError(
            f"Unknown keys in config {pre_args.config}: {unknown_cfg_keys}. "
            "Use argument names (e.g., batch_size, probe_hidden_dims)."
        )
    parser.set_defaults(**yaml_config)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_dir = Path(args.embeddings_dir) / args.model
    print(f"Loading embeddings from {emb_dir}")

    X_train, y_train, meta_train = load_embeddings(emb_dir, "train", args.pool)
    X_val, y_val, meta_val = load_embeddings(emb_dir, "val", args.pool)
    X_test, y_test, meta_test = load_embeddings(emb_dir, "test", args.pool)

    dim = X_train.shape[1]
    n0, n1 = (y_train == 0).sum().item(), (y_train == 1).sum().item()
    print(f"Train: {len(X_train)} ({n1} drv, {n0} non-drv) | "
          f"Val: {len(X_val)} | Test: {len(X_test)} | dim: {dim}")

    # UMAP plots and artifacts output
    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not str(args.policy).strip():
        inferred = run_id.split("_", 1)[0].strip().upper()
        if inferred in {"A", "B", "C", "D"}:
            args.policy = inferred
    out_dir = Path(args.output_dir) / args.model / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {out_dir}")
    wandb_run = _init_wandb_run(
        args,
        run_id=run_id,
        out_dir=out_dir,
        dim=int(dim),
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
    )

    # Frame distribution by split (true labels from loaded embeddings)
    frame_dist_parts = [
        _frame_distribution_by_split(meta_train, y_train.cpu().numpy(), "train"),
        _frame_distribution_by_split(meta_val, y_val.cpu().numpy(), "val"),
        _frame_distribution_by_split(meta_test, y_test.cpu().numpy(), "test"),
    ]
    frame_split_df = pd.concat([p for p in frame_dist_parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in frame_dist_parts
    ) else pd.DataFrame()
    _print_frame_split_distribution(frame_split_df)
    if not frame_split_df.empty:
        frame_split_path = out_dir / "frame_split_distribution.csv"
        frame_split_df.to_csv(frame_split_path, index=False)
        print(f"Frame split distribution saved to {frame_split_path}")

    plot_umap_binary(X_train, y_train, out_dir / "umap_driver.png",
                     title=f"{args.model.upper()} — driver vs non-driver")
    if "cancer_types" in meta_train:
        plot_umap_cancer(X_train, meta_train["cancer_types"],
                         out_dir / "umap_cancer.png",
                         title=f"{args.model.upper()} — cancer type")

    # Balanced sampler: equal probability per class
    class_counts = torch.bincount(y_train)
    sample_weights = 1.0 / class_counts[y_train].float()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

    model = build_probe(
        arch=args.probe_arch,
        in_dim=dim,
        hidden_dims=args.probe_hidden_dims,
        dropout=args.probe_dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="max",
            factor=args.lr_reduce_factor,
            patience=args.patience,
            min_lr=args.lr_min,
        )
    criterion = FocalLoss(gamma=args.focal_gamma)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=args.batch_size, sampler=sampler)

    best_auroc = 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            loader,
            optim,
            criterion,
            device,
            noise_std=args.train_noise_std,
        )
        val = evaluate_split(model, X_val, y_val, criterion, device)
        test_epoch = evaluate_split(model, X_test, y_test, criterion, device)

        improved = val["auroc"] > best_auroc
        if improved:
            best_auroc = val["auroc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        prev_lr = optim.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val["auroc"])
        curr_lr = optim.param_groups[0]["lr"]
        flag = " *" if improved else ""
        print(
            f"  ep {ep:3d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val['loss']:.4f} | val_auroc {val['auroc']:.4f} | val_f1 {val['f1']:.4f} | "
            f"test_loss {test_epoch['loss']:.4f} | test_auroc {test_epoch['auroc']:.4f} | test_f1 {test_epoch['f1']:.4f} | "
            f"lr {curr_lr:.6g}{flag}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": ep,
                    "train/loss": float(train_loss),
                    "val/loss": float(val["loss"]),
                    "val/accuracy": float(val["acc"]),
                    "val/f1": float(val["f1"]),
                    "val/auroc": float(val["auroc"]),
                    "val/precision": float(val["prec"]),
                    "val/recall": float(val["rec"]),
                    "test/loss": float(test_epoch["loss"]),
                    "test/accuracy": float(test_epoch["acc"]),
                    "test/f1": float(test_epoch["f1"]),
                    "test/auroc": float(test_epoch["auroc"]),
                    "test/precision": float(test_epoch["prec"]),
                    "test/recall": float(test_epoch["rec"]),
                    "optim/lr": float(curr_lr),
                    "train/best_val_auroc_so_far": float(best_auroc),
                    "train/improved_best": int(improved),
                },
                step=ep,
            )
        if curr_lr < prev_lr:
            print(
                f"  LR reduced at epoch {ep}: {prev_lr:.6g} -> {curr_lr:.6g} "
                f"(factor={args.lr_reduce_factor}, patience={args.patience})"
            )

    # Capture last epoch state
    last_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Test — best AUROC model
    model.load_state_dict(best_state)
    print("\n>>> Best AUROC model:")
    t_best = compute_metrics(model, X_test, y_test, device)
    print_test_results(t_best, args.model, args.pool)
    if "fusion_pairs" in meta_test:
        print_per_fusion_results(t_best["preds"], t_best["labels"], meta_test["fusion_pairs"])

    # Test — last epoch model
    model.load_state_dict(last_state)
    print("\n>>> Last epoch model:")
    t_last = compute_metrics(model, X_test, y_test, device)
    print_test_results(t_last, args.model, args.pool)
    if "fusion_pairs" in meta_test:
        print_per_fusion_results(t_last["preds"], t_last["labels"], meta_test["fusion_pairs"])

    test_rand_for_calib, _ = _compute_random_baseline(np.array(t_best["labels"]), seed=42)
    test_calibration = _compute_calibration_metrics(
        np.array(t_best["labels"]),
        np.array(t_best["probs"]),
        test_rand_for_calib.astype(float),
    )
    print("\n>>> Test calibration (best model):")
    print(
        "  "
        f"Brier={test_calibration['brier']:.6f} | "
        f"Quality(1-Brier)={test_calibration['calibration_quality']:.6f} | "
        f"DriverErr(1-p|y=1)={test_calibration['driver_error']:.6f} | "
        f"NonDriverErr(p|y=0)={test_calibration['non_driver_error']:.6f} | "
        f"RandomBrier={test_calibration['random_brier']:.6f} | "
        f"DeltaBrierVsRandom={test_calibration['brier_improvement_vs_random']:.6f}"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "test_best/calibration_brier": float(test_calibration["brier"]),
                "test_best/calibration_quality": float(test_calibration["calibration_quality"]),
                "test_best/calibration_driver_error": float(test_calibration["driver_error"]),
                "test_best/calibration_non_driver_error": float(test_calibration["non_driver_error"]),
                "test_best/calibration_random_brier": float(test_calibration["random_brier"]),
                "test_best/calibration_random_quality": float(test_calibration["random_calibration_quality"]),
                "test_best/calibration_brier_improvement_vs_random": float(
                    test_calibration["brier_improvement_vs_random"]
                ),
            }
        )

    # Benchmark evaluation — load pre-computed benchmark embeddings
    gene_counts_train, _ = _build_gene_frequency_baseline(meta_train)
    bm_pt = emb_dir / "benchmark.pt"
    bm_artifacts = {
        "metrics_plot": None,
        "metrics_txt": None,
        "per_fusion_plot": None,
        "per_fusion_txt": None,
        "prob_plot": None,
        "prob_txt": None,
        "metrics_model": None,
        "metrics_random": None,
        "metrics_gene": None,
        "cm_random": None,
        "cm_gene": None,
        "random_ref_driver_rate": None,
        "calibration": None,
    }
    if bm_pt.exists():
        X_bm, y_bm, meta_bm = load_embeddings(emb_dir, "benchmark", args.pool)
        bm_pairs = meta_bm.get("fusion_pairs", [])

        # best model
        model.load_state_dict(best_state)
        print("\n>>> Benchmark — best AUROC model:")
        bm_best = compute_metrics(model, X_bm, y_bm, device)
        print_test_results(bm_best, f"{args.model} [benchmark]", args.pool)
        if bm_pairs:
            print_per_fusion_results(bm_best["preds"], bm_best["labels"], bm_pairs)

        # last model
        model.load_state_dict(last_state)
        print("\n>>> Benchmark — last epoch model:")
        bm_last = compute_metrics(model, X_bm, y_bm, device)
        print_test_results(bm_last, f"{args.model} [benchmark]", args.pool)
        if bm_pairs:
            print_per_fusion_results(bm_last["preds"], bm_last["labels"], bm_pairs)

        # baseline comparison on benchmark
        bm_labels = np.array(bm_best["labels"])
        bm_preds_model = np.array(bm_best["preds"])
        bm_rand, bm_p_driver = _compute_random_baseline(bm_labels, seed=42)
        if bm_pairs and len(bm_pairs) == len(bm_labels):
            bm_df = pd.DataFrame({"fusion_pair": bm_pairs})
        else:
            bm_df = pd.DataFrame({"fusion_pair": [""] * len(bm_labels)})
        bm_gene = _predict_gene_frequency_baseline(bm_df, gene_counts_train)

        bm_metrics_plot, bm_m_model, bm_m_rand, bm_m_gene = _plot_model_vs_baselines_metrics(
            bm_labels,
            bm_preds_model,
            bm_rand,
            bm_gene,
            out_dir / "plot_benchmark_model_vs_baselines.png",
        )
        bm_metrics_txt = _plot_txt_path(bm_metrics_plot)
        _write_text_lines(
            bm_metrics_txt,
            [
                "Benchmark linear probe vs baselines summary",
                f"random_reference_driver_rate={bm_p_driver:.6f}",
                f"linear_probe: accuracy={bm_m_model['accuracy']:.6f}, f1={bm_m_model['f1']:.6f}, precision={bm_m_model['precision']:.6f}, recall={bm_m_model['recall']:.6f}, predicted_driver_rate={bm_m_model['predicted_driver_rate']:.6f}",
                f"random: accuracy={bm_m_rand['accuracy']:.6f}, f1={bm_m_rand['f1']:.6f}, precision={bm_m_rand['precision']:.6f}, recall={bm_m_rand['recall']:.6f}, predicted_driver_rate={bm_m_rand['predicted_driver_rate']:.6f}",
                f"gene_frequency: accuracy={bm_m_gene['accuracy']:.6f}, f1={bm_m_gene['f1']:.6f}, precision={bm_m_gene['precision']:.6f}, recall={bm_m_gene['recall']:.6f}, predicted_driver_rate={bm_m_gene['predicted_driver_rate']:.6f}",
            ],
        )

        bm_artifacts.update(
            {
                "metrics_plot": bm_metrics_plot,
                "metrics_txt": bm_metrics_txt,
                "metrics_model": bm_m_model,
                "metrics_random": bm_m_rand,
                "metrics_gene": bm_m_gene,
                "cm_random": confusion_matrix(bm_labels, bm_rand),
                "cm_gene": confusion_matrix(bm_labels, bm_gene),
                "random_ref_driver_rate": bm_p_driver,
            }
        )
        bm_calibration = _compute_calibration_metrics(
            bm_labels,
            np.array(bm_best["probs"]),
            bm_rand.astype(float),
        )
        bm_prob_df = pd.DataFrame(
            {
                "y_true": bm_labels.astype(int),
                "y_prob_driver": np.array(bm_best["probs"], dtype=float),
            }
        )
        bm_prob_plot = _plot_probability_distribution(
            bm_prob_df,
            out_dir / "plot_benchmark_prob_driver_distribution.png",
            title="Benchmark probability distribution",
        )
        bm_prob_txt = None
        if bm_prob_plot is not None:
            probs_non = bm_prob_df.loc[bm_prob_df["y_true"] == 0, "y_prob_driver"].dropna()
            probs_drv = bm_prob_df.loc[bm_prob_df["y_true"] == 1, "y_prob_driver"].dropna()
            bm_prob_txt = _plot_txt_path(bm_prob_plot)
            _write_text_lines(
                bm_prob_txt,
                [
                    "Benchmark probability distribution summary",
                    f"true non-driver: n={len(probs_non)}, mean={float(probs_non.mean()):.6f}, std={float(probs_non.std(ddof=1)) if len(probs_non)>1 else 0.0:.6f}",
                    f"true driver: n={len(probs_drv)}, mean={float(probs_drv.mean()):.6f}, std={float(probs_drv.std(ddof=1)) if len(probs_drv)>1 else 0.0:.6f}",
                ],
            )
        bm_artifacts.update(
            {
                "prob_plot": bm_prob_plot,
                "prob_txt": bm_prob_txt,
            }
        )
        bm_artifacts["calibration"] = bm_calibration
        print(
            "\n>>> Benchmark calibration (best model): "
            f"Brier={bm_calibration['brier']:.6f}, "
            f"Quality(1-Brier)={bm_calibration['calibration_quality']:.6f}, "
            f"RandomBrier={bm_calibration['random_brier']:.6f}, "
            f"DeltaBrierVsRandom={bm_calibration['brier_improvement_vs_random']:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "benchmark_best/calibration_brier": float(bm_calibration["brier"]),
                    "benchmark_best/calibration_quality": float(bm_calibration["calibration_quality"]),
                    "benchmark_best/calibration_driver_error": float(bm_calibration["driver_error"]),
                    "benchmark_best/calibration_non_driver_error": float(bm_calibration["non_driver_error"]),
                    "benchmark_best/calibration_random_brier": float(bm_calibration["random_brier"]),
                    "benchmark_best/calibration_random_quality": float(
                        bm_calibration["random_calibration_quality"]
                    ),
                    "benchmark_best/calibration_brier_improvement_vs_random": float(
                        bm_calibration["brier_improvement_vs_random"]
                    ),
                }
            )

        if bm_pairs:
            preds_by_method = {
                "linear_probe": bm_preds_model,
                "random": bm_rand,
                "gene_frequency": bm_gene,
            }
            bm_per_fusion_plot = _plot_benchmark_per_fusion_accuracy(
                bm_labels,
                bm_pairs,
                preds_by_method,
                out_dir / "plot_benchmark_per_fusion_accuracy_baselines.png",
            )
            bm_per_fusion_txt = out_dir / "benchmark_per_fusion_baselines.txt"
            bm_per_fusion_txt.write_text(
                format_benchmark_per_fusion_results_multi(
                    labels=bm_labels,
                    fusion_pairs=bm_pairs,
                    preds_by_method=preds_by_method,
                    title="BENCHMARK PER-FUSION BREAKDOWN (LINEAR PROBE VS BASELINES)",
                ) + "\n",
                encoding="utf-8",
            )
            bm_artifacts.update(
                {
                    "per_fusion_plot": bm_per_fusion_plot,
                    "per_fusion_txt": bm_per_fusion_txt,
                }
            )
    else:
        print(f"\n[INFO] No benchmark embeddings found at {bm_pt}. "
              "Run compute_embeddings.py first to generate them.")

    # Save both
    torch.save(best_state, out_dir / "probe_best.pt")
    torch.save(last_state, out_dir / "probe_last.pt")
    print(f"\nBest model saved to {out_dir / 'probe_best.pt'}")
    print(f"Last model saved to {out_dir / 'probe_last.pt'}")

    # Save test prediction analysis (based on full test metadata, when available)
    train_fusion_names = _extract_fusion_name_set(meta_train)
    analysis_artifacts = analyze_test_predictions(
        meta_test,
        t_best,
        out_dir,
        train_fusion_names=train_fusion_names,
        meta_train=meta_train,
    )
    print(f"Test predictions + metadata saved to {analysis_artifacts['pred_csv']}")
    if analysis_artifacts["cancer_csv"] is not None:
        print(f"Cancer-type breakdown saved to {analysis_artifacts['cancer_csv']}")
    if analysis_artifacts["frame_csv"] is not None:
        print(f"Frame breakdown (test) saved to {analysis_artifacts['frame_csv']}")
        _print_test_frame_breakdown(analysis_artifacts["frame_stats"])
    if analysis_artifacts["seen_vs_train_csv"] is not None:
        print(f"Seen-vs-novel fusion breakdown (test) saved to {analysis_artifacts['seen_vs_train_csv']}")
        _print_seen_vs_novel_breakdown(analysis_artifacts["seen_vs_train_stats"])
    if analysis_artifacts["pred_sorted_txt"] is not None:
        print(f"Ranked test predictions (with seen_in_train) saved to {analysis_artifacts['pred_sorted_txt']}")
    print(f"Numeric feature associations saved to {analysis_artifacts['numeric_csv']}")
    print(f"Categorical feature associations saved to {analysis_artifacts['cat_csv']}")
    if analysis_artifacts["plot_pred_prob"] is not None:
        print(f"Probability distribution plot saved to {analysis_artifacts['plot_pred_prob']}")
    if analysis_artifacts["plot_cancer"] is not None:
        print(f"Cancer-type plot saved to {analysis_artifacts['plot_cancer']}")
    if analysis_artifacts["plot_numeric"] is not None:
        print(f"Numeric-association plot saved to {analysis_artifacts['plot_numeric']}")
    if analysis_artifacts["plot_categorical"] is not None:
        print(f"Categorical-association plot saved to {analysis_artifacts['plot_categorical']}")
    if analysis_artifacts["plot_baseline"] is not None:
        print(f"Linear-probe-vs-baselines plot saved to {analysis_artifacts['plot_baseline']}")
    if analysis_artifacts["gene_baseline_train_csv"] is not None:
        print(f"Gene-frequency baseline train stats saved to {analysis_artifacts['gene_baseline_train_csv']}")
    if analysis_artifacts["plot_seed_reads"] is not None:
        print(f"Seed-reads plot saved to {analysis_artifacts['plot_seed_reads']}")
    if analysis_artifacts["plot_junction_reads"] is not None:
        print(f"Junction-reads plot saved to {analysis_artifacts['plot_junction_reads']}")
    if analysis_artifacts["plot_recurrent"] is not None:
        print(f"Recurrent-rate plot saved to {analysis_artifacts['plot_recurrent']}")
    if analysis_artifacts["plot_cancer_rates"] is not None:
        print(f"Cancer-type rates (true/pred/random/gene-baseline) plot saved to {analysis_artifacts['plot_cancer_rates']}")
    if analysis_artifacts["plot_chr_info"] is not None:
        print(f"Chr_info plot saved to {analysis_artifacts['plot_chr_info']}")
    if analysis_artifacts["plot_has_pub"] is not None:
        print(f"has_pub plot saved to {analysis_artifacts['plot_has_pub']}")
    if analysis_artifacts["plot_gene_roles"] is not None:
        print(f"Gene-role distribution plot saved to {analysis_artifacts['plot_gene_roles']}")
    if analysis_artifacts["plot_role_paths"]:
        print("Role plots saved:")
        for p in analysis_artifacts["plot_role_paths"]:
            print(f"  - {p}")
    if analysis_artifacts["plot_text_summaries"]:
        print("Plot text summaries saved:")
        for p in analysis_artifacts["plot_text_summaries"]:
            print(f"  - {p}")
    if bm_artifacts["metrics_plot"] is not None:
        print(f"Benchmark linear-probe-vs-baselines plot saved to {bm_artifacts['metrics_plot']}")
    if bm_artifacts["metrics_txt"] is not None:
        print(f"Benchmark baseline metrics summary saved to {bm_artifacts['metrics_txt']}")
    if bm_artifacts["per_fusion_plot"] is not None:
        print(f"Benchmark per-fusion baseline comparison plot saved to {bm_artifacts['per_fusion_plot']}")
    if bm_artifacts["per_fusion_txt"] is not None:
        print(f"Benchmark per-fusion baseline comparison table saved to {bm_artifacts['per_fusion_txt']}")
    if bm_artifacts["prob_plot"] is not None:
        print(f"Benchmark probability distribution plot saved to {bm_artifacts['prob_plot']}")
    if bm_artifacts["prob_txt"] is not None:
        print(f"Benchmark probability distribution summary saved to {bm_artifacts['prob_txt']}")

    if wandb_run is not None:
        wandb_run.summary["paths/out_dir"] = str(out_dir)
        wandb_run.summary["paths/summary_txt"] = str(out_dir / "summary.txt")
        wandb_run.log(
            {
                "test_best/accuracy": float(t_best["acc"]),
                "test_best/f1": float(t_best["f1"]),
                "test_best/auroc": float(t_best["auroc"]),
                "test_best/precision": float(t_best["prec"]),
                "test_best/recall": float(t_best["rec"]),
                "test_last/accuracy": float(t_last["acc"]),
                "test_last/f1": float(t_last["f1"]),
                "test_last/auroc": float(t_last["auroc"]),
                "test_last/precision": float(t_last["prec"]),
                "test_last/recall": float(t_last["rec"]),
            }
        )
        if bm_artifacts["metrics_model"] is not None:
            bmm = bm_artifacts["metrics_model"]
            wandb_run.log(
                {
                    "benchmark_best/accuracy": float(bmm["accuracy"]),
                    "benchmark_best/f1": float(bmm["f1"]),
                    "benchmark_best/precision": float(bmm["precision"]),
                    "benchmark_best/recall": float(bmm["recall"]),
                }
            )

        _wandb_log_confusion(
            wandb_run,
            key="confusion/test_best",
            y_true=np.array(t_best["labels"]),
            y_pred=np.array(t_best["preds"]),
            title="Test best confusion matrix",
        )
        _wandb_log_confusion(
            wandb_run,
            key="confusion/test_last",
            y_true=np.array(t_last["labels"]),
            y_pred=np.array(t_last["preds"]),
            title="Test last confusion matrix",
        )
        _wandb_log_confusion(
            wandb_run,
            key="confusion/test_random_baseline",
            y_true=np.array(t_best["labels"]),
            y_pred=np.array(analysis_artifacts["y_pred_random"]),
            title="Test random baseline confusion matrix",
        )
        _wandb_log_confusion(
            wandb_run,
            key="confusion/test_gene_baseline",
            y_true=np.array(t_best["labels"]),
            y_pred=np.array(analysis_artifacts["y_pred_gene_baseline"]),
            title="Test gene baseline confusion matrix",
        )
        if bm_pt.exists():
            _wandb_log_confusion(
                wandb_run,
                key="confusion/benchmark_best",
                y_true=np.array(bm_best["labels"]),
                y_pred=np.array(bm_best["preds"]),
                title="Benchmark best confusion matrix",
            )
            if bm_artifacts["metrics_model"] is not None:
                _wandb_log_confusion(
                    wandb_run,
                    key="confusion/benchmark_random_baseline",
                    y_true=np.array(bm_best["labels"]),
                    y_pred=np.array(bm_rand),
                    title="Benchmark random baseline confusion matrix",
                )
                _wandb_log_confusion(
                    wandb_run,
                    key="confusion/benchmark_gene_baseline",
                    y_true=np.array(bm_best["labels"]),
                    y_pred=np.array(bm_gene),
                    title="Benchmark gene baseline confusion matrix",
                )

        for png_path in sorted(out_dir.glob("*.png")):
            _wandb_log_image_path(wandb_run, f"plots/{png_path.stem}", png_path)
    _print_top_probability_tables(
        "TEST SET: TOP FUSIONS BY P(driver)",
        analysis_artifacts["top_driver_prob"],
        analysis_artifacts["top_non_driver_prob"],
        prob_col="y_prob_driver",
    )

    # Save final performance summary in results/<model>/summary.txt
    summary_sections = [
        format_experiment_config(
            args=args,
            device=device,
            emb_dir=emb_dir,
            out_dir=out_dir,
            run_id=run_id,
            dim=dim,
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
        ),
        "",
        format_test_results(t_best, args.model, args.pool, "BEST AUROC MODEL"),
        "",
        "Calibration (best test model):",
        f"  - brier: {test_calibration['brier']:.6f}",
        f"  - calibration_quality (1-brier): {test_calibration['calibration_quality']:.6f}",
        f"  - driver_error (mean(1-p_driver|y=1)): {test_calibration['driver_error']:.6f}",
        f"  - non_driver_error (mean(p_driver|y=0)): {test_calibration['non_driver_error']:.6f}",
        f"  - random_brier: {test_calibration['random_brier']:.6f}",
        f"  - random_calibration_quality (1-brier): {test_calibration['random_calibration_quality']:.6f}",
        f"  - brier_improvement_vs_random: {test_calibration['brier_improvement_vs_random']:.6f}",
    ]
    if bm_pt.exists():
        summary_sections.extend([
            "",
            format_test_results(
                bm_best, f"{args.model} [benchmark]", args.pool, "BENCHMARK - BEST AUROC MODEL"
            ),
        ])
        bm_pairs = meta_bm.get("fusion_pairs", [])
        if bm_pairs:
            summary_sections.extend([
                "",
                format_benchmark_per_fusion_results(
                    bm_best["preds"], bm_best["labels"], bm_pairs
                ),
            ])
        if bm_artifacts["metrics_model"] is not None:
            bmm = bm_artifacts["metrics_model"]
            bmr = bm_artifacts["metrics_random"]
            bmg = bm_artifacts["metrics_gene"]
            bm_cmr = bm_artifacts["cm_random"]
            bm_cmg = bm_artifacts["cm_gene"]
            summary_sections.extend([
                "",
                "Benchmark baseline comparison (linear probe vs random vs gene-frequency):",
                f"  - random reference driver rate (from benchmark y_true prevalence): {bm_artifacts['random_ref_driver_rate']:.4f}",
                f"  - linear probe: acc={bmm['accuracy']:.4f}, f1={bmm['f1']:.4f}, prec={bmm['precision']:.4f}, rec={bmm['recall']:.4f}, pred_driver_rate={bmm['predicted_driver_rate']:.4f}",
                f"  - random: acc={bmr['accuracy']:.4f}, f1={bmr['f1']:.4f}, prec={bmr['precision']:.4f}, rec={bmr['recall']:.4f}, pred_driver_rate={bmr['predicted_driver_rate']:.4f}",
                f"  - gene-frequency: acc={bmg['accuracy']:.4f}, f1={bmg['f1']:.4f}, prec={bmg['precision']:.4f}, rec={bmg['recall']:.4f}, pred_driver_rate={bmg['predicted_driver_rate']:.4f}",
                "  - random confusion matrix:",
                str(bm_cmr),
                "  - gene-frequency confusion matrix:",
                str(bm_cmg),
                f"  - benchmark metrics plot: {bm_artifacts['metrics_plot']}",
                f"  - benchmark metrics txt: {bm_artifacts['metrics_txt']}",
                f"  - benchmark per-fusion plot: {bm_artifacts['per_fusion_plot'] if bm_artifacts['per_fusion_plot'] is not None else 'N/A'}",
                f"  - benchmark per-fusion table: {bm_artifacts['per_fusion_txt'] if bm_artifacts['per_fusion_txt'] is not None else 'N/A'}",
                f"  - benchmark probability distribution plot: {bm_artifacts['prob_plot'] if bm_artifacts['prob_plot'] is not None else 'N/A'}",
                f"  - benchmark probability distribution summary: {bm_artifacts['prob_txt'] if bm_artifacts['prob_txt'] is not None else 'N/A'}",
            ])
        if bm_artifacts["calibration"] is not None:
            bcal = bm_artifacts["calibration"]
            summary_sections.extend([
                "",
                "Benchmark calibration (best model):",
                f"  - brier: {bcal['brier']:.6f}",
                f"  - calibration_quality (1-brier): {bcal['calibration_quality']:.6f}",
                f"  - driver_error (mean(1-p_driver|y=1)): {bcal['driver_error']:.6f}",
                f"  - non_driver_error (mean(p_driver|y=0)): {bcal['non_driver_error']:.6f}",
                f"  - random_brier: {bcal['random_brier']:.6f}",
                f"  - random_calibration_quality (1-brier): {bcal['random_calibration_quality']:.6f}",
                f"  - brier_improvement_vs_random: {bcal['brier_improvement_vs_random']:.6f}",
            ])

    summary_sections.extend([
        "",
        f"{'='*60}",
        "TEST SET PREDICTION ANALYSIS (BEST MODEL)",
        f"{'='*60}",
        f"Predictions with metadata: {analysis_artifacts['pred_csv']}",
        f"Ranked test predictions (P(driver), with seen_in_train): {analysis_artifacts['pred_sorted_txt'] if analysis_artifacts['pred_sorted_txt'] else 'N/A'}",
        f"Cancer-type breakdown: {analysis_artifacts['cancer_csv'] if analysis_artifacts['cancer_csv'] else 'N/A (Cancertype missing)'}",
        f"Frame breakdown (test): {analysis_artifacts['frame_csv'] if analysis_artifacts['frame_csv'] else 'N/A (Frame missing)'}",
        f"Seen-vs-novel fusion breakdown (test): {analysis_artifacts['seen_vs_train_csv'] if analysis_artifacts['seen_vs_train_csv'] else 'N/A (fusion names unavailable)'}",
        f"Numeric associations: {analysis_artifacts['numeric_csv']}",
        f"Categorical associations: {analysis_artifacts['cat_csv']}",
        f"Probability distribution plot: {analysis_artifacts['plot_pred_prob'] if analysis_artifacts['plot_pred_prob'] else 'N/A'}",
        f"Cancer-type plot: {analysis_artifacts['plot_cancer'] if analysis_artifacts['plot_cancer'] else 'N/A'}",
        f"Numeric associations plot: {analysis_artifacts['plot_numeric'] if analysis_artifacts['plot_numeric'] else 'N/A'}",
        f"Categorical associations plot: {analysis_artifacts['plot_categorical'] if analysis_artifacts['plot_categorical'] else 'N/A'}",
        f"Linear-probe-vs-baselines plot: {analysis_artifacts['plot_baseline'] if analysis_artifacts['plot_baseline'] else 'N/A'}",
        f"Gene-frequency baseline train stats: {analysis_artifacts['gene_baseline_train_csv'] if analysis_artifacts['gene_baseline_train_csv'] else 'N/A (missing train metadata)'}",
        f"Seed reads (true vs pred) plot: {analysis_artifacts['plot_seed_reads'] if analysis_artifacts['plot_seed_reads'] else 'N/A'}",
        f"Junction reads (true vs pred) plot: {analysis_artifacts['plot_junction_reads'] if analysis_artifacts['plot_junction_reads'] else 'N/A'}",
        f"Recurrent rate plot: {analysis_artifacts['plot_recurrent'] if analysis_artifacts['plot_recurrent'] else 'N/A'}",
        f"Cancer type rates (true/pred/random/gene-baseline) plot: {analysis_artifacts['plot_cancer_rates'] if analysis_artifacts['plot_cancer_rates'] else 'N/A'}",
        f"Chr_info plot: {analysis_artifacts['plot_chr_info'] if analysis_artifacts['plot_chr_info'] else 'N/A'}",
        f"has_pub plot: {analysis_artifacts['plot_has_pub'] if analysis_artifacts['plot_has_pub'] else 'N/A'}",
        f"Gene-role distribution plot (true labels): {analysis_artifacts['plot_gene_roles'] if analysis_artifacts['plot_gene_roles'] else 'N/A'}",
    ])

    role_plot_paths = analysis_artifacts["plot_role_paths"]
    summary_sections.extend(["", "Role plots (true/pred/random/gene-baseline):"])
    if role_plot_paths:
        for p in role_plot_paths:
            summary_sections.append(f"  - {p}")
    else:
        summary_sections.append("  - N/A")

    bm = analysis_artifacts["baseline_model"]
    br = analysis_artifacts["baseline_random"]
    bg = analysis_artifacts["baseline_gene"]
    cmr = analysis_artifacts["baseline_random_confusion_matrix"]
    cmg = analysis_artifacts["baseline_gene_confusion_matrix"]
    summary_sections.extend([
        "",
        "Baseline comparison (linear probe vs random vs gene-frequency):",
        f"  - random reference driver rate (from y_true prevalence): {analysis_artifacts['baseline_random_driver_rate_reference']:.4f}",
        f"  - linear probe: acc={bm['accuracy']:.4f}, f1={bm['f1']:.4f}, prec={bm['precision']:.4f}, rec={bm['recall']:.4f}, pred_driver_rate={bm['predicted_driver_rate']:.4f}",
        f"  - random: acc={br['accuracy']:.4f}, f1={br['f1']:.4f}, prec={br['precision']:.4f}, rec={br['recall']:.4f}, pred_driver_rate={br['predicted_driver_rate']:.4f}",
        f"  - gene-frequency: acc={bg['accuracy']:.4f}, f1={bg['f1']:.4f}, prec={bg['precision']:.4f}, rec={bg['recall']:.4f}, pred_driver_rate={bg['predicted_driver_rate']:.4f}",
        "  - random confusion matrix:",
        str(cmr),
        "  - gene-frequency confusion matrix:",
        str(cmg),
    ])

    summary_sections.extend(["", "Plot text summaries:"])
    if analysis_artifacts["plot_text_summaries"]:
        for p in analysis_artifacts["plot_text_summaries"]:
            summary_sections.append(f"  - {p}")
    else:
        summary_sections.append("  - N/A")

    summary_sections.extend(["", "Top fusions by test-set P(driver):"])
    td = analysis_artifacts["top_driver_prob"]
    tn = analysis_artifacts["top_non_driver_prob"]
    if len(td) > 0:
        summary_sections.append("Highest P(driver):")
        for _, r in td.iterrows():
            summary_sections.append(
                f"  - {r['fusion_name']}: p={float(r['y_prob_driver']):.4f}, pred={r.get('predicted_label', 'NA')}, true={r.get('true_label', 'NA')}"
            )
        summary_sections.append("Lowest P(driver):")
        for _, r in tn.iterrows():
            summary_sections.append(
                f"  - {r['fusion_name']}: p={float(r['y_prob_driver']):.4f}, pred={r.get('predicted_label', 'NA')}, true={r.get('true_label', 'NA')}"
            )
    else:
        summary_sections.append("  - N/A")

    top_num = analysis_artifacts["top_numeric"]
    if len(top_num) > 0:
        summary_sections.extend(["", "Top numeric associations (lowest p-value):"])
        for _, r in top_num.iterrows():
            summary_sections.append(
                f"  - {r['feature']}: p={r['p_value']:.2e}, Δmedian={r['delta_median']:.3g}"
            )

    top_cat = analysis_artifacts["top_categorical"]
    if len(top_cat) > 0:
        summary_sections.extend(["", "Top categorical associations (lowest p-value):"])
        for _, r in top_cat.iterrows():
            summary_sections.append(
                f"  - {r['feature']}: p={r['p_value']:.2e}, chi2={r['chi2']:.3g}"
            )

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_sections) + "\n", encoding="utf-8")
    print(f"Summary saved to {summary_path}")
    if wandb_run is not None:
        try:
            wandb_run.save(str(summary_path))
        except Exception as exc:
            print(f"[WARNING] wandb summary upload failed: {exc}")
        wandb_run.finish()


if __name__ == "__main__":
    main()
