"""Shared helpers: data loading, UMAP plots, metric reporting."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report, confusion_matrix,
)


# ── Data loading ──────────────────────────────────────────────────────────

def load_embeddings(emb_dir: Path, split: str, pool: str = "mean"):
    """Load pre-computed embeddings and labels from a .pt file.

    Returns (X, y, metadata) where metadata is a dict with optional keys
    like 'cancer_types'.
    """
    data = torch.load(emb_dir / f"{split}.pt", weights_only=False)
    embeddings = data["embeddings"]
    if not embeddings:
        raise ValueError(f"No embeddings found in {emb_dir / f'{split}.pt'}")

    first = embeddings[0]
    if first.ndim == 1:
        if pool == "cls":
            raise ValueError(
                f"Split '{split}' contains mean-pooled embeddings (1D vectors). "
                "CLS pooling is unavailable on this file."
            )
        X = torch.stack(embeddings)
    elif first.ndim == 2:
        if pool == "mean":
            X = torch.stack([e.mean(dim=0) for e in embeddings])
        elif pool == "cls":
            X = torch.stack([e[0] for e in embeddings])
        else:
            raise ValueError(f"Unknown pool mode: {pool}")
    else:
        raise ValueError(f"Unsupported embedding rank: {first.ndim}")

    y = torch.tensor(data["labels"], dtype=torch.long)
    meta = {k: data[k] for k in data if k not in ("embeddings", "labels")}
    return X, y, meta


# ── UMAP plotting ────────────────────────────────────────────────────────

def _umap_coords(X: torch.Tensor, seed: int = 42):
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.3)
    return reducer.fit_transform(X.numpy())


def plot_umap_binary(X: torch.Tensor, y: torch.Tensor, out_path,
                     title: str = "UMAP — driver vs non-driver"):
    """UMAP coloured by driver (1) / non-driver (0)."""
    print("Computing UMAP (binary)...")
    coords = _umap_coords(X)
    labels_np = y.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, name, color in [(0, "non-driver", "#4c72b0"), (1, "driver", "#dd5555")]:
        mask = labels_np == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5,
                   label=name, color=color, rasterized=True)
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_umap_cancer(X: torch.Tensor, cancer_types: list, out_path,
                     title: str = "UMAP — cancer type", max_legend: int = 15):
    """UMAP coloured by cancer type."""
    print("Computing UMAP (cancer type)...")
    coords = _umap_coords(X)
    types = np.array(cancer_types)

    unique, counts = np.unique(types, return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]

    cmap = plt.cm.get_cmap("tab20", len(unique))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, ct in enumerate(unique):
        mask = types == ct
        label = ct if i < max_legend else None
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5,
                   color=cmap(i), label=label, rasterized=True)
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Metrics ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(model, X: torch.Tensor, y: torch.Tensor, device: str):
    """Run inference and return a dict of standard binary metrics."""
    model.eval()
    logits = model(X.to(device))
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    y_np = y.numpy()
    return {
        "acc": accuracy_score(y_np, preds),
        "f1": f1_score(y_np, preds),
        "auroc": roc_auc_score(y_np, probs),
        "prec": precision_score(y_np, preds),
        "rec": recall_score(y_np, preds),
        "preds": preds,
        "probs": probs,
        "labels": y_np,
    }


BENCHMARK_GENE_PAIRS = [
    ("EWSR1", "FLI1"),
    ("PAX3", "FOXO1"),
    ("BCR", "ABL1"),
    ("EML4", "ALK"),
]


def print_per_fusion_results(preds, labels, fusion_pairs):
    """Print accuracy for top recurring driver fusions and benchmark pairs."""
    preds = np.array(preds)
    labels = np.array(labels)
    fusions = np.array(fusion_pairs)

    # Benchmark pairs
    benchmark_names = {f"{g5}-{g3}" for g5, g3 in BENCHMARK_GENE_PAIRS}

    print(f"\n{'='*60}")
    print("BENCHMARK FUSION RESULTS")
    print(f"{'='*60}")
    for name in sorted(benchmark_names):
        mask = fusions == name
        n = mask.sum()
        if n == 0:
            print(f"  {name:<20}: not in test set")
            continue
        correct = (preds[mask] == labels[mask]).sum()
        print(f"  {name:<20}: {correct}/{n} correct ({correct/n*100:.1f}%)")

    # Top-10 most frequent driver fusions in test
    driver_mask = labels == 1
    driver_fusions = fusions[driver_mask]
    driver_preds = preds[driver_mask]
    driver_labels = labels[driver_mask]

    if len(driver_fusions) == 0:
        return

    unique, counts = np.unique(driver_fusions, return_counts=True)
    order = np.argsort(-counts)

    print(f"\n{'='*60}")
    print("TOP-10 MOST FREQUENT DRIVER FUSIONS (test set)")
    print(f"{'='*60}")
    for idx in order[:10]:
        name = unique[idx]
        mask_f = driver_fusions == name
        n = mask_f.sum()
        correct = (driver_preds[mask_f] == driver_labels[mask_f]).sum()
        bm = " [BM]" if name in benchmark_names else ""
        print(f"  {name:<20}: {correct}/{n} correct ({correct/n*100:.1f}%){bm}")


def print_test_results(metrics: dict, model_name: str, pool: str):
    t = metrics
    print(f"\n{'='*60}")
    print(f"TEST RESULTS — {model_name.upper()} (pool={pool})")
    print(f"{'='*60}")
    print(f"  Accuracy:  {t['acc']:.4f}")
    print(f"  F1:        {t['f1']:.4f}")
    print(f"  AUROC:     {t['auroc']:.4f}")
    print(f"  Precision: {t['prec']:.4f}")
    print(f"  Recall:    {t['rec']:.4f}")
    print(f"\n{classification_report(t['labels'], t['preds'], target_names=['non-driver', 'driver'])}")
    print(f"Confusion matrix:\n{confusion_matrix(t['labels'], t['preds'])}")


def evaluate_on_benchmark(model, X_test: torch.Tensor, y_test: torch.Tensor,
                          fusion_pairs_test: list, device: str,
                          benchmark_csv: Path, model_name: str):
    """Filter test set to benchmark fusions and evaluate separately."""
    if not benchmark_csv.exists():
        print(f"\n[INFO] Benchmark CSV not found: {benchmark_csv}")
        return

    # Load benchmark CSV to get the list of benchmark fusion pairs
    df_bm = pd.read_csv(benchmark_csv)
    if "fusion_h_gene" not in df_bm.columns or "fusion_t_gene" not in df_bm.columns:
        print("[WARNING] Benchmark CSV missing fusion gene columns")
        return

    # Build set of benchmark fusion pairs
    benchmark_pairs = set()
    for _, row in df_bm.iterrows():
        h_gene = str(row["fusion_h_gene"]).strip()
        t_gene = str(row["fusion_t_gene"]).strip()
        benchmark_pairs.add(f"{h_gene}-{t_gene}")

    # Filter test set to benchmark fusions
    fusion_pairs_arr = np.array(fusion_pairs_test)
    mask = np.array([fp in benchmark_pairs for fp in fusion_pairs_arr])
    n_bm = mask.sum()

    if n_bm == 0:
        print(f"\n[INFO] No benchmark fusions found in test set")
        return

    X_bm = X_test[mask]
    y_bm = y_test[mask]
    pairs_bm = fusion_pairs_arr[mask]

    # Evaluate
    print(f"\n{'='*60}")
    print(f"BENCHMARK EVALUATION — {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Benchmark fusions: {n_bm} / {len(y_test)} test samples")

    metrics = compute_metrics(model, X_bm, y_bm, device)
    print(f"  Accuracy:  {metrics['acc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  Precision: {metrics['prec']:.4f}")
    print(f"  Recall:    {metrics['rec']:.4f}")

    # Per-fusion breakdown
    preds_bm = metrics["preds"]
    labels_bm = metrics["labels"]

    print(f"\n{'='*60}")
    print("PER-FUSION ACCURACY (Benchmark)")
    print(f"{'='*60}")
    for pair in sorted(benchmark_pairs):
        mask_pair = pairs_bm == pair
        n = mask_pair.sum()
        if n == 0:
            continue
        correct = (preds_bm[mask_pair] == labels_bm[mask_pair]).sum()
        print(f"  {pair:<20}: {correct}/{n} correct ({correct/n*100:.1f}%)")
