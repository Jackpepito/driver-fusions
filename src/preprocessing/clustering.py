import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


MMSEQS_DIR = "/homes/gcapitani/Gene-Fusions/mmseqs/bin"

BENCHMARK_GENE_PAIRS = [
    ("EWSR1", "FLI1"),
    ("PAX3", "FOXO1"),
    ("BCR", "ABL1"),
    ("EML4", "ALK"),
]

def ensure_mmseqs_in_path():
    if shutil.which("mmseqs") is None:
        os.environ["PATH"] = f"{MMSEQS_DIR}:{os.environ['PATH']}"


def make_fasta(ids: list, sequences: list, fasta_path: str):
    with open(fasta_path, "w") as f:
        for sid, seq in zip(ids, sequences):
            f.write(f">{sid}\n{seq}\n")


def run_mmseqs(input_fasta: str, output_dir: str, min_seq_id=0.3, c=0.8,
               cov_mode=0, cluster_mode=0):
    ensure_mmseqs_in_path()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    prefix = os.path.join(output_dir, "mmseqs")

    cmd = [
        "mmseqs", "easy-cluster", input_fasta, prefix, output_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(c),
        "--cov-mode", str(cov_mode),
        "--cluster-mode", str(cluster_mode),
        "--dbtype", "1",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return f"{prefix}_cluster.tsv"


def parse_clusters(cluster_tsv: str) -> dict:
    """Return {member_id: cluster_id} mapping (cluster_id = integer)."""
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"],
                     dtype=str)
    rep_to_cid = {rep: cid for cid, rep in enumerate(df["rep"].unique())}
    return {row["member"]: rep_to_cid[row["rep"]] for _, row in df.iterrows()}


def find_benchmark_clusters(df: pd.DataFrame) -> set:
    """Return set of cluster_ids that contain at least one benchmark gene pair."""
    benchmark_cids = set()
    for g5, g3 in BENCHMARK_GENE_PAIRS:
        mask = (df["H_gene"] == g5) & (df["T_gene"] == g3)
        cids = df.loc[mask, "cluster_id"].dropna().unique()
        if len(cids) > 0:
            benchmark_cids.update(cids)
            print(f"  {g5}-{g3}: {mask.sum()} fusions in {len(cids)} cluster(s)")
        else:
            print(f"  {g5}-{g3}: not found in dataset")
    return benchmark_cids


def split_clusters(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15,
                   seed=42) -> pd.DataFrame:
    """
    Assign each row to train/val/test keeping whole clusters together.
    Benchmark clusters go entirely to test. Remaining clusters are shuffled
    and distributed to approximate the target ratios.
    """
    rng = np.random.RandomState(seed)

    print("\nIdentifying benchmark clusters...")
    benchmark_cids = find_benchmark_clusters(df)

    all_cids = set(df["cluster_id"].unique())
    remaining_cids = list(all_cids - benchmark_cids)
    rng.shuffle(remaining_cids)

    cluster_sizes = df.groupby("cluster_id").size().to_dict()
    n_total = len(df)
    n_benchmark = sum(cluster_sizes.get(c, 0) for c in benchmark_cids)

    target_test = max(0, int(n_total * (1 - train_ratio - val_ratio)) - n_benchmark)
    target_val = int(n_total * val_ratio)

    split_map = {cid: "test" for cid in benchmark_cids}

    # Fill test with more clusters until target reached
    test_count = 0
    val_count = 0
    idx = 0
    for cid in remaining_cids:
        size = cluster_sizes[cid]
        if test_count < target_test:
            split_map[cid] = "test"
            test_count += size
        elif val_count < target_val:
            split_map[cid] = "val"
            val_count += size
        else:
            split_map[cid] = "train"
        idx += 1

    df["split"] = df["cluster_id"].map(split_map)
    return df


def print_split_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        n_clusters = subset["cluster_id"].nunique()
        n_drivers = (subset["driver"] == "driver").sum() if "driver" in subset.columns else 0
        print(f"  {split:5s}: {len(subset):5d} samples, {n_clusters:4d} clusters, "
              f"{n_drivers:4d} drivers")
    print(f"  total: {len(df):5d} samples, {df['cluster_id'].nunique():4d} clusters")


def cluster_size_summary(df: pd.DataFrame):
    sizes = df.groupby("cluster_id").size()
    print(f"\nCluster statistics:")
    print(f"  Total clusters:    {len(sizes)}")
    print(f"  Singletons:        {(sizes == 1).sum()}")
    print(f"  Largest cluster:   {sizes.max()}")
    print(f"  Mean cluster size: {sizes.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="MMseqs2 clustering and cluster-aware train/val/test splitting"
    )
    parser.add_argument("--input", type=str,
                        default="/homes/gcapitani/driver-fusions/chimerseq_analysis_results/chimerseq_analysis_results.csv",
                        help="Enriched CSV from reconstruction pipeline")
    parser.add_argument("--output-dir", type=str, default="clustering",
                        help="Output directory (default: clustering)")
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Keep only rows with a reconstructed sequence
    has_seq = df["reconstructed_seq"].notna() & (df["reconstructed_seq"].str.len() > 0)
    df_seq = df[has_seq].copy().reset_index(drop=True)
    print(f"Rows with reconstructed sequence: {len(df_seq)} / {len(df)}")

    # Write FASTA (use row index as ID)
    fasta_path = str(output_dir / "input.fasta")
    make_fasta(
        ids=[str(i) for i in df_seq.index],
        sequences=df_seq["reconstructed_seq"].tolist(),
        fasta_path=fasta_path,
    )
    print(f"FASTA written: {fasta_path}")

    # Run MMseqs2
    mmseqs_dir = str(output_dir / "mmseqs_raw")
    cluster_tsv = run_mmseqs(
        fasta_path, mmseqs_dir,
        min_seq_id=args.min_seq_id, c=args.coverage,
    )

    # Parse cluster assignments
    id_to_cluster = parse_clusters(cluster_tsv)
    df_seq["cluster_id"] = df_seq.index.astype(str).map(id_to_cluster)
    cluster_size_summary(df_seq)

    # Cluster-aware splitting
    df_seq = split_clusters(
        df_seq,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print_split_summary(df_seq)

    # Save output
    out_csv = str(output_dir / "clustered_splits.csv")
    df_seq.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
