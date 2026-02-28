#!/usr/bin/env python3
"""Compute embeddings using ESM-C or FusOn-pLM and save mean-pooled vectors."""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

MAX_SEQ_LEN = 4000

DEFAULT_INPUT = "/homes/gcapitani/driver-fusions/clustering/clustered_splits.csv"
DEFAULT_OUTPUT = "/work/H2020DeciderFicarra/gcapitani/driver-fusion/embeddings"
DEFAULT_BENCHMARK = "/homes/gcapitani/driver-fusions/data/benchmark_fusions.csv"

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

def load_esmc(device):
    from esm.models.esmc import ESMC
    model = ESMC.from_pretrained("esmc_600m", device=torch.device(device))
    model.eval()
    return model


def load_fuson(device):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/FusOn-pLM")
    model = AutoModel.from_pretrained("ChatterjeeLab/FusOn-pLM").to(device)
    model.eval()
    return model, tokenizer


def embed_esmc(model, seq):
    from esm.sdk.api import ESMProtein, LogitsConfig
    protein = ESMProtein(sequence=seq[:MAX_SEQ_LEN])
    pt = model.encode(protein)
    out = model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
    return out.embeddings.squeeze().detach().cpu()


def embed_fuson(model_tok, seq, device):
    model, tokenizer = model_tok
    inputs = tokenizer(seq[:MAX_SEQ_LEN], return_tensors="pt",
                       truncation=True, max_length=MAX_SEQ_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.squeeze(0)[1:-1, :].cpu()


def mean_pool_embedding(emb: torch.Tensor) -> torch.Tensor:
    """Return one vector per sequence (mean over token dimension when needed)."""
    if emb.ndim == 2:
        return emb.mean(dim=0)
    if emb.ndim == 1:
        return emb
    raise ValueError(f"Unexpected embedding shape {tuple(emb.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Compute and save mean-pooled embeddings")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Clustered CSV with reconstructed_seq, driver, split")
    parser.add_argument("--model", default="esmc", choices=["esmc", "fuson"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                        help="Benchmark CSV with 'sequence', 'fusion_h_gene', 'fusion_t_gene', 'driver'")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_csv(args.input)
    df = df[df["reconstructed_seq"].notna() & (df["reconstructed_seq"].str.len() > 0)]
    print(f"Loaded {len(df)} sequences with valid reconstructed_seq")

    if args.model == "esmc":
        model = load_esmc(device)
        embed_fn = lambda seq: embed_esmc(model, seq)
    else:
        model_tok = load_fuson(device)
        embed_fn = lambda seq: embed_fuson(model_tok, seq, device)

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {"driver": 1, "non-driver": 0}

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        if len(split_df) == 0:
            print(f"  {split}: empty, skipping")
            continue

        n = len(split_df)
        n_drv = (split_df["driver"] == "driver").sum()
        print(f"\n  {split}: {n} sequences ({n_drv} driver, {n - n_drv} non-driver)")

        embeddings = []
        labels = []
        cancer_types = []
        fusion_pairs = []
        metadata_rows = []
        for i, row in split_df.iterrows():
            emb = embed_fn(row["reconstructed_seq"])
            embeddings.append(mean_pool_embedding(emb))
            labels.append(label_map.get(row["driver"], -1))
            cancer_types.append(row.get("Cancertype", ""))
            fusion_pairs.append(f"{row['H_gene']}-{row['T_gene']}")
            # Keep full original row metadata for downstream analysis.
            metadata_rows.append(row.to_dict())
            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"    [{i + 1}/{n}]")

        save_path = out_dir / f"{split}.pt"
        torch.save(
            {
                "embeddings": embeddings,
                "labels": labels,
                "cancer_types": cancer_types,
                "fusion_pairs": fusion_pairs,
                "metadata_rows": metadata_rows,
            },
            save_path,
        )
        print(f"    -> {save_path}  (dim={embeddings[0].shape[-1]})")

    # ── Benchmark embeddings ─────────────────────────────────────────────
    bm_path = Path(args.benchmark)
    if bm_path.exists():
        bm_df = pd.read_csv(bm_path)
        bm_df = bm_df[bm_df["sequence"].notna() & (bm_df["sequence"].str.len() > 0)]
        n_bm = len(bm_df)
        print(f"\n  benchmark: {n_bm} sequences")

        bm_embeddings = []
        bm_labels = []
        bm_fusion_pairs = []
        for i, row in bm_df.iterrows():
            emb = embed_fn(row["sequence"])
            bm_embeddings.append(mean_pool_embedding(emb))
            bm_labels.append(label_map.get(str(row["driver"]).strip(), 1))
            bm_fusion_pairs.append(
                f"{row['fusion_h_gene']}-{row['fusion_t_gene']}"
            )
            if (i + 1) % 10 == 0 or (i + 1) == n_bm:
                print(f"    [{i + 1}/{n_bm}]")

        bm_save = out_dir / "benchmark.pt"
        torch.save({"embeddings": bm_embeddings, "labels": bm_labels,
                     "fusion_pairs": bm_fusion_pairs}, bm_save)
        print(f"    -> {bm_save}  (dim={bm_embeddings[0].shape[-1]})")
    else:
        print(f"\n  [INFO] Benchmark CSV not found: {bm_path}, skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
