#!/usr/bin/env python3
"""Linear probing on pre-computed embeddings for driver/non-driver classification."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from utils import (
    load_embeddings, plot_umap_binary, plot_umap_cancer,
    compute_metrics, print_test_results, print_per_fusion_results,
)

DEFAULT_EMB_DIR = "/homes/gcapitani/driver-fusions/embeddings"
DEFAULT_OUTPUT = "/homes/gcapitani/driver-fusions/results"


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


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train linear probe for driver classification")
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMB_DIR)
    parser.add_argument("--model", default="esmc", choices=["esmc", "fuson"])
    parser.add_argument("--pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
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

    # UMAP plots
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
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

    model = LinearProbe(dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss(gamma=2.0)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=args.batch_size, sampler=sampler)

    best_auroc = 0
    best_state = None
    wait = 0

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optim, criterion, device)
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


if __name__ == "__main__":
    main()
