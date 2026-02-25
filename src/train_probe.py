#!/usr/bin/env python3
"""Probe training on pre-computed embeddings for driver/non-driver classification."""

import argparse
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import chi2_contingency, mannwhitneyu

from nets import build_probe
from utils import (
    load_embeddings, plot_umap_binary, plot_umap_cancer,
    compute_metrics, print_test_results, print_per_fusion_results,
)

DEFAULT_EMB_DIR = "/homes/gcapitani/driver-fusions/embeddings"
DEFAULT_OUTPUT = "/homes/gcapitani/driver-fusions/results"

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
        classification_report(t["labels"], t["preds"], target_names=["non-driver", "driver"]),
        "Confusion matrix:",
        str(confusion_matrix(t["labels"], t["preds"])),
    ]
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
    p_driver = float(np.mean(labels))
    rng = np.random.default_rng(seed)
    y_rand = (rng.random(len(labels)) < p_driver).astype(int)
    return y_rand, p_driver


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


def _fusion_label_series(df: pd.DataFrame) -> pd.Series:
    for col in ["Fusion_pair", "fusion_pair", "id"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            if (s != "").any():
                return s
    if "H_gene" in df.columns and "T_gene" in df.columns:
        return df["H_gene"].astype(str).str.strip() + "-" + df["T_gene"].astype(str).str.strip()
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index, dtype="object")


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
    tmp = tmp.dropna(subset=[category_col, "y_true", "y_pred", "y_pred_random"])
    lines = [title, f"category_col: {category_col}", f"n_valid: {len(tmp)}", ""]
    if len(tmp) < 20 or tmp[category_col].nunique() < 2:
        lines.append("Not enough data to compute category rates.")
        return lines
    stats = tmp.groupby(category_col, dropna=False).agg(
        n=("y_true", "size"),
        true_driver_rate=("y_true", "mean"),
        pred_driver_rate=("y_pred", "mean"),
        random_driver_rate=("y_pred_random", "mean"),
    ).reset_index().sort_values("n", ascending=False).head(top_n)
    lines.append("category,n,true_driver_rate,pred_driver_rate,random_driver_rate")
    for _, r in stats.iterrows():
        lines.append(
            f"{r[category_col]},{int(r['n'])},{float(r['true_driver_rate']):.6f},{float(r['pred_driver_rate']):.6f},{float(r['random_driver_rate']):.6f}"
        )
    return lines


def _plot_probability_distribution(df: pd.DataFrame, out_path: Path) -> Path | None:
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
    ax.set_title("Test-set probability distribution")
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
    tmp = tmp.dropna(subset=[category_col, "y_true", "y_pred", "y_pred_random"])
    if len(tmp) < 20:
        return None
    nun = tmp[category_col].nunique()
    if nun < 2:
        return None

    stats = tmp.groupby(category_col, dropna=False).agg(
        n=("y_true", "size"),
        true_driver_rate=("y_true", "mean"),
        pred_driver_rate=("y_pred", "mean"),
        random_driver_rate=("y_pred_random", "mean"),
    ).reset_index().sort_values("n", ascending=False).head(top_n)

    stats = stats.sort_values("n", ascending=True)
    y = np.arange(len(stats))
    h = 0.24

    fig, ax = plt.subplots(figsize=(11, max(6, 0.34 * len(stats))))
    ax.barh(y - h, stats["true_driver_rate"].values, height=h, color="#4c72b0", label="true")
    ax.barh(y, stats["pred_driver_rate"].values, height=h, color="#dd5555", label="predicted")
    ax.barh(y + h, stats["random_driver_rate"].values, height=h, color="#999999", label="random baseline")
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

    tmp = df.loc[valid, ["y_true", "y_pred", "y_pred_random"]].copy()
    tmp["recurrent"] = rec[valid].values

    sources = [("True labels", "y_true"), ("Predicted labels", "y_pred"), ("Random baseline", "y_pred_random")]
    values_drv = []
    values_non = []
    for _, col in sources:
        d = tmp.loc[tmp[col] == 1, "recurrent"]
        n = tmp.loc[tmp[col] == 0, "recurrent"]
        values_drv.append(float(d.mean()) if len(d) else np.nan)
        values_non.append(float(n.mean()) if len(n) else np.nan)

    x = np.arange(len(sources))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, values_non, width=w, color="#4c72b0", label="non-driver")
    ax.bar(x + w / 2, values_drv, width=w, color="#dd5555", label="driver")
    ax.set_xticks(x, [s[0] for s in sources])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recurrent fusion rate")
    ax.set_title("Recurrent fusions by label source")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_model_vs_random_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_rand: np.ndarray,
    out_path: Path,
) -> tuple[Path, dict, dict]:
    m_model = _baseline_metrics(y_true, y_pred)
    m_rand = _baseline_metrics(y_true, y_rand)

    keys = ["accuracy", "f1", "precision", "recall", "predicted_driver_rate"]
    labels = ["Accuracy", "F1", "Precision", "Recall", "Driver rate"]
    x = np.arange(len(keys))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [m_model[k] for k in keys], width=w, color="#dd5555", label="model")
    ax.bar(x + w / 2, [m_rand[k] for k in keys], width=w, color="#999999", label="random baseline")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model vs random-choice baseline")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path, m_model, m_rand


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

    pred_csv = out_dir / "test_predictions_with_metadata.csv"
    df.to_csv(pred_csv, index=False)

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

    # ── Generic numeric feature associations (predicted driver vs non-driver) ──
    numeric_rows = []
    excluded = {"y_true", "y_pred", "y_prob_driver", "is_correct"}
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
    plot_baseline, baseline_model, baseline_random = _plot_model_vs_random_metrics(
        labels, preds, y_rand, out_dir / "plot_model_vs_random_baseline.png"
    )
    if plot_baseline is not None:
        txt = _plot_txt_path(plot_baseline)
        _write_text_lines(
            txt,
            [
                "Model vs random-choice baseline summary",
                f"random_reference_driver_rate={p_driver:.6f}",
                f"model: accuracy={baseline_model['accuracy']:.6f}, f1={baseline_model['f1']:.6f}, precision={baseline_model['precision']:.6f}, recall={baseline_model['recall']:.6f}, predicted_driver_rate={baseline_model['predicted_driver_rate']:.6f}",
                f"random: accuracy={baseline_random['accuracy']:.6f}, f1={baseline_random['f1']:.6f}, precision={baseline_random['precision']:.6f}, recall={baseline_random['recall']:.6f}, predicted_driver_rate={baseline_random['predicted_driver_rate']:.6f}",
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
            tmp_rec["recurrent"] = rec[valid].values
            txt = _plot_txt_path(plot_recurrent)
            lines = ["Recurrent fusion rate summary", "source,non_driver_rate,driver_rate"]
            for name, col in [("true", "y_true"), ("predicted", "y_pred"), ("random", "y_pred_random")]:
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
            title="Driver-rate by cancer type (true vs predicted vs random)",
            top_n=20,
        )
        if plot_cancer_rates is not None:
            txt = _plot_txt_path(plot_cancer_rates)
            _write_text_lines(
                txt,
                _rate_by_category_summary_lines(
                    df,
                    cancer_col,
                    "Driver-rate by cancer type (true vs predicted vs random)",
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
            title=f"Driver-rate by {role_col} (true vs predicted vs random)",
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
                    f"Driver-rate by {role_col} (true vs predicted vs random)",
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
            title="Driver-rate by Chr_info (true vs predicted vs random)",
            top_n=15,
        )
        if plot_chr_info is not None:
            txt = _plot_txt_path(plot_chr_info)
            _write_text_lines(
                txt,
                _rate_by_category_summary_lines(
                    df,
                    chr_info_col,
                    "Driver-rate by Chr_info (true vs predicted vs random)",
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
            title="Driver-rate by has_pub (true vs predicted vs random)",
            top_n=2,
        )
        if plot_has_pub is not None:
            txt = _plot_txt_path(plot_has_pub)
            _write_text_lines(
                txt,
                _rate_by_category_summary_lines(
                    tmp,
                    "has_pub_status",
                    "Driver-rate by has_pub (true vs predicted vs random)",
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

    return {
        "pred_csv": pred_csv,
        "cancer_csv": cancer_csv,
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
        "baseline_model": baseline_model,
        "baseline_random": baseline_random,
        "baseline_random_driver_rate_reference": p_driver,
        "top_numeric": top_num,
        "top_categorical": top_cat,
    }


def main():
    parser = argparse.ArgumentParser(description="Train probe model for driver classification")
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMB_DIR)
    parser.add_argument("--model", default="esmc", choices=["esmc", "fuson"])
    parser.add_argument("--pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-id",
        default="",
        help="Run identifier for output folder. If empty, uses timestamp.",
    )
    parser.add_argument("--probe-arch", default="linear", choices=["linear", "deep"])
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_dir = Path(args.embeddings_dir) / args.model
    print(f"Loading embeddings from {emb_dir}")

    X_train, y_train, meta_train = load_embeddings(emb_dir, "train", args.pool)
    X_val, y_val, _ = load_embeddings(emb_dir, "val", args.pool)
    X_test, y_test, meta_test = load_embeddings(emb_dir, "test", args.pool)

    dim = X_train.shape[1]
    n0, n1 = (y_train == 0).sum().item(), (y_train == 1).sum().item()
    print(f"Train: {len(X_train)} ({n1} drv, {n0} non-drv) | "
          f"Val: {len(X_val)} | Test: {len(X_test)} | dim: {dim}")

    # UMAP plots and artifacts output
    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / args.model / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {out_dir}")
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
    criterion = FocalLoss(gamma=args.focal_gamma)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=args.batch_size, sampler=sampler)

    best_auroc = 0
    best_state = None
    wait = 0

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            loader,
            optim,
            criterion,
            device,
            noise_std=args.train_noise_std,
        )
        val = compute_metrics(model, X_val, y_val, device)

        improved = val["auroc"] > best_auroc
        if improved:
            best_auroc = val["auroc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0

        if ep % 10 == 0 or ep == 1 or improved:
            flag = " *" if improved else ""
            print(f"  ep {ep:3d} | loss {loss:.4f} | "
                  f"val_auroc {val['auroc']:.4f} | val_f1 {val['f1']:.4f}{flag}")

        wait += 0 if improved else 1
        if wait >= args.patience:
            print(f"  Early stopping at epoch {ep}")
            break

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

    # Benchmark evaluation — load pre-computed benchmark embeddings
    bm_pt = emb_dir / "benchmark.pt"
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
    else:
        print(f"\n[INFO] No benchmark embeddings found at {bm_pt}. "
              "Run compute_embeddings.py first to generate them.")

    # Save both
    torch.save(best_state, out_dir / "probe_best.pt")
    torch.save(last_state, out_dir / "probe_last.pt")
    print(f"\nBest model saved to {out_dir / 'probe_best.pt'}")
    print(f"Last model saved to {out_dir / 'probe_last.pt'}")

    # Save test prediction analysis (based on full test metadata, when available)
    analysis_artifacts = analyze_test_predictions(meta_test, t_best, out_dir)
    print(f"Test predictions + metadata saved to {analysis_artifacts['pred_csv']}")
    if analysis_artifacts["cancer_csv"] is not None:
        print(f"Cancer-type breakdown saved to {analysis_artifacts['cancer_csv']}")
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
        print(f"Model-vs-random baseline plot saved to {analysis_artifacts['plot_baseline']}")
    if analysis_artifacts["plot_seed_reads"] is not None:
        print(f"Seed-reads plot saved to {analysis_artifacts['plot_seed_reads']}")
    if analysis_artifacts["plot_junction_reads"] is not None:
        print(f"Junction-reads plot saved to {analysis_artifacts['plot_junction_reads']}")
    if analysis_artifacts["plot_recurrent"] is not None:
        print(f"Recurrent-rate plot saved to {analysis_artifacts['plot_recurrent']}")
    if analysis_artifacts["plot_cancer_rates"] is not None:
        print(f"Cancer-type rates (true/pred/random) plot saved to {analysis_artifacts['plot_cancer_rates']}")
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
    ]
    if bm_pt.exists():
        summary_sections.extend([
            "",
            format_test_results(
                bm_best, f"{args.model} [benchmark]", args.pool, "BENCHMARK - BEST AUROC MODEL"
            ),
        ])

    summary_sections.extend([
        "",
        f"{'='*60}",
        "TEST SET PREDICTION ANALYSIS (BEST MODEL)",
        f"{'='*60}",
        f"Predictions with metadata: {analysis_artifacts['pred_csv']}",
        f"Cancer-type breakdown: {analysis_artifacts['cancer_csv'] if analysis_artifacts['cancer_csv'] else 'N/A (Cancertype missing)'}",
        f"Numeric associations: {analysis_artifacts['numeric_csv']}",
        f"Categorical associations: {analysis_artifacts['cat_csv']}",
        f"Probability distribution plot: {analysis_artifacts['plot_pred_prob'] if analysis_artifacts['plot_pred_prob'] else 'N/A'}",
        f"Cancer-type plot: {analysis_artifacts['plot_cancer'] if analysis_artifacts['plot_cancer'] else 'N/A'}",
        f"Numeric associations plot: {analysis_artifacts['plot_numeric'] if analysis_artifacts['plot_numeric'] else 'N/A'}",
        f"Categorical associations plot: {analysis_artifacts['plot_categorical'] if analysis_artifacts['plot_categorical'] else 'N/A'}",
        f"Model-vs-random baseline plot: {analysis_artifacts['plot_baseline'] if analysis_artifacts['plot_baseline'] else 'N/A'}",
        f"Seed reads (true vs pred) plot: {analysis_artifacts['plot_seed_reads'] if analysis_artifacts['plot_seed_reads'] else 'N/A'}",
        f"Junction reads (true vs pred) plot: {analysis_artifacts['plot_junction_reads'] if analysis_artifacts['plot_junction_reads'] else 'N/A'}",
        f"Recurrent rate plot: {analysis_artifacts['plot_recurrent'] if analysis_artifacts['plot_recurrent'] else 'N/A'}",
        f"Cancer type rates (true/pred/random) plot: {analysis_artifacts['plot_cancer_rates'] if analysis_artifacts['plot_cancer_rates'] else 'N/A'}",
        f"Chr_info plot: {analysis_artifacts['plot_chr_info'] if analysis_artifacts['plot_chr_info'] else 'N/A'}",
        f"has_pub plot: {analysis_artifacts['plot_has_pub'] if analysis_artifacts['plot_has_pub'] else 'N/A'}",
        f"Gene-role distribution plot (true labels): {analysis_artifacts['plot_gene_roles'] if analysis_artifacts['plot_gene_roles'] else 'N/A'}",
    ])

    role_plot_paths = analysis_artifacts["plot_role_paths"]
    summary_sections.extend(["", "Role plots (true/pred/random):"])
    if role_plot_paths:
        for p in role_plot_paths:
            summary_sections.append(f"  - {p}")
    else:
        summary_sections.append("  - N/A")

    bm = analysis_artifacts["baseline_model"]
    br = analysis_artifacts["baseline_random"]
    summary_sections.extend([
        "",
        "Random-choice baseline comparison:",
        f"  - random reference driver rate (from y_true prevalence): {analysis_artifacts['baseline_random_driver_rate_reference']:.4f}",
        f"  - model:  acc={bm['accuracy']:.4f}, f1={bm['f1']:.4f}, prec={bm['precision']:.4f}, rec={bm['recall']:.4f}, pred_driver_rate={bm['predicted_driver_rate']:.4f}",
        f"  - random: acc={br['accuracy']:.4f}, f1={br['f1']:.4f}, prec={br['precision']:.4f}, rec={br['recall']:.4f}, pred_driver_rate={br['predicted_driver_rate']:.4f}",
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


if __name__ == "__main__":
    main()
