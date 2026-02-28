#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_fusions_set.py - Reconstruction pipeline for Fusions_set1-17 dataset.

This script mirrors the current seq_recon/run.py workflow but adapts input
parsing and filtering to the Fusions_set1-17.tsv schema.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Keep local imports consistent with run.py behavior
sys.path.insert(0, str(Path(__file__).parent.parent))

from seq_reconstruction import IsoformAwareFusionReconstructor, generate_final_analysis


DEFAULT_PATHS = {
    "hg19": {
        "gtf": "/homes/gcapitani/Gene-Fusions/data/gencode.v19.annotation.gtf",
        "genome": "/homes/gcapitani/Gene-Fusions/data/hg19.fa",
    },
    "hg38": {
        "gtf": "/homes/gcapitani/Gene-Fusions/data/gencode.v47.annotation.gtf",
        "genome": "/homes/gcapitani/Gene-Fusions/data/hg38.fa",
    },
}


def parse_breakpoint(value):
    """Parse breakpoint values (supports numeric strings and comma-separated values)."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if "," in s:
        s = s.split(",")[0].strip()
    try:
        return int(float(s))
    except Exception:
        return None


def normalize_chr(value):
    """Normalize chromosome labels removing optional 'chr' prefix."""
    s = str(value).strip()
    if not s:
        return ""
    if s.lower().startswith("chr"):
        return s[3:]
    return s


def clean_peptide_sequence(value):
    """Normalize peptide_sequence field into a single AA string."""
    if pd.isna(value):
        return ""
    seq_raw = str(value).strip()
    if not seq_raw or seq_raw == ".":
        return ""

    # Some rows can store two candidates separated by '|': keep longest part.
    if "|" in seq_raw:
        parts = [p.strip() for p in seq_raw.split("|") if p.strip()]
        if parts:
            seq_raw = max(parts, key=len)

    return seq_raw.replace("*", "").strip()


def load_fusions_set_data(
    input_file: str,
    n_samples: int = None,
    only_protein_coding: bool = True,
    only_with_sequence: bool = True,
    random_state: int = 42,
):
    """Load and preprocess Fusions_set1-17 input table."""
    print(f"Loading data: {input_file}")
    df = pd.read_csv(input_file, sep="\t", low_memory=False)
    print(f"Loaded {len(df)} total rows")

    required_cols = [
        "Gene1",
        "Gene2",
        "Chromosome1",
        "Chromosome2",
        "Breakpoint1",
        "Breakpoint2",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Canonical columns used by the reconstructor.
    df["gene5"] = df["Gene1"].astype(str).str.strip()
    df["gene3"] = df["Gene2"].astype(str).str.strip()
    df["chr5"] = df["Chromosome1"].apply(normalize_chr)
    df["chr3"] = df["Chromosome2"].apply(normalize_chr)
    df["bp5"] = df["Breakpoint1"].apply(parse_breakpoint)
    df["bp3"] = df["Breakpoint2"].apply(parse_breakpoint)
    df["original_seq"] = df["peptide_sequence"].apply(clean_peptide_sequence) if "peptide_sequence" in df.columns else ""

    # Basic quality filters.
    df = df.dropna(subset=["bp5", "bp3"]).copy()
    df = df[(df["gene5"] != "") & (df["gene3"] != "") & (df["chr5"] != "") & (df["chr3"] != "")].copy()
    print(f"After basic filters: {len(df)} rows")

    if only_protein_coding and "Biotype1" in df.columns and "Biotype2" in df.columns:
        df = df[(df["Biotype1"] == "protein_coding") & (df["Biotype2"] == "protein_coding")].copy()
        print(f"After protein_coding filter: {len(df)} rows")

    if only_with_sequence and "peptide_sequence" in df.columns:
        df = df[df["original_seq"] != ""].copy()
        print(f"After peptide_sequence filter: {len(df)} rows")

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        print(f"Sampled {n_samples} rows")
    else:
        df = df.reset_index(drop=True)
        print(f"Processing all {len(df)} rows")

    return df


def build_gene_resolver(reconstructor):
    """Return a case-insensitive gene symbol resolver using loaded GTF symbols."""
    cache = getattr(reconstructor, "transcript_cache", {})
    upper_to_canonical = {}
    for g in cache.keys():
        upper_to_canonical.setdefault(str(g).upper(), g)

    def resolve(gene_symbol: str) -> str:
        if gene_symbol in cache:
            return gene_symbol
        return upper_to_canonical.get(str(gene_symbol).upper(), gene_symbol)

    return resolve


def print_coding_diagnostics(df: pd.DataFrame, reconstructor, resolve_gene):
    """Print coding-transcript coverage diagnostics for genes in this dataset."""
    genes = pd.concat([df["gene5"], df["gene3"]], axis=0).dropna().astype(str).str.strip()
    unique_genes = sorted({g for g in genes if g})
    n_total = len(unique_genes)
    n_present = 0
    n_with_coding = 0

    for g in unique_genes:
        g_res = resolve_gene(g)
        txs = reconstructor.get_gene_transcripts(g_res)
        if txs:
            n_present += 1
        if any((t.cds_start is not None and t.cds_end is not None) for t in txs):
            n_with_coding += 1

    print("\nGene coverage diagnostics (hg38 cache):")
    print(f"  Unique genes in dataset:       {n_total}")
    print(f"  Genes found in cache:          {n_present} ({(n_present/max(n_total,1))*100:.1f}%)")
    print(f"  Genes with coding transcripts: {n_with_coding} ({(n_with_coding/max(n_total,1))*100:.1f}%)")


def run_fusions_set_reconstruction(
    input_file: str,
    n_samples: int = None,
    genome_build: str = "hg38",
    output_prefix: str = "fusions_set1_17",
    use_orffinder: bool = True,
    orffinder_path: str = "/homes/gcapitani/Gene-Fusions/data/ORFfinder",
    only_protein_coding: bool = True,
    only_with_sequence: bool = True,
    random_state: int = 42,
    gtf_path: str = None,
    genome_path: str = None,
    refresh_cache: bool = False,
):
    """Run sequence reconstruction for Fusions_set1-17."""
    if genome_build != "hg38":
        raise ValueError("Fusions_set1-17 pipeline supports only hg38.")

    if not gtf_path:
        gtf_path = DEFAULT_PATHS[genome_build]["gtf"]
    if not genome_path:
        genome_path = DEFAULT_PATHS[genome_build]["genome"]

    print("=" * 80)
    print("FUSION PROTEIN RECONSTRUCTION PIPELINE — Fusions_set1-17")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    print("Configuration:")
    print(f"  Input file:          {input_file}")
    print(f"  Genome build:        {genome_build}")
    print(f"  GTF file:            {gtf_path}")
    print(f"  Genome file:         {genome_path}")
    print(f"  Output prefix:       {output_prefix}")
    print(f"  Sample size:         {n_samples if n_samples else 'all'}")
    print(f"  ORFfinder:           {'enabled' if use_orffinder else 'disabled'}")
    if use_orffinder:
        print(f"  ORFfinder path:      {orffinder_path}")
    print(f"  Only protein-coding: {only_protein_coding}")
    print(f"  Only with sequence:  {only_with_sequence}")
    print(f"  Random seed:         {random_state}")
    print()

    start_time = time.time()
    test_df = load_fusions_set_data(
        input_file=input_file,
        n_samples=n_samples,
        only_protein_coding=only_protein_coding,
        only_with_sequence=only_with_sequence,
        random_state=random_state,
    )

    cache_dir = Path("cache") / "gtf_hg38_v47_fusions_set"
    if refresh_cache and cache_dir.exists():
        print(f"Refreshing cache: removing {cache_dir}")
        shutil.rmtree(cache_dir)

    print("\nInitializing reconstructor...")
    reconstructor = IsoformAwareFusionReconstructor(
        mode="local",
        gtf_path=gtf_path,
        genome_path=genome_path,
        cache_dir=str(cache_dir),
        use_orffinder=use_orffinder,
        orffinder_path=orffinder_path,
    )
    print("Reconstructor ready.\n")
    resolve_gene = build_gene_resolver(reconstructor)
    print_coding_diagnostics(test_df, reconstructor, resolve_gene)

    # Reconstruction columns (aligned with run.py)
    recon_cols = [
        "recon_status",
        "reconstructed_seq",
        "reconstructed_length",
        "num_variants",
        "recon_in_frame",
        "recon_quality",
        "bp5_approx",
        "bp3_approx",
        "best_tx_pair",
        "orf_used",
        "orf_frame",
        "recon_error",
        "original_length",
        "length_diff",
    ]
    for col in recon_cols:
        test_df[col] = pd.NA

    n = len(test_df)
    for idx in range(n):
        row = test_df.iloc[idx]
        gene5_resolved = resolve_gene(row["gene5"])
        gene3_resolved = resolve_gene(row["gene3"])
        fusion_name = f"{gene5_resolved}-{gene3_resolved}"
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{n}] {fusion_name}")
        print(f"  chr{row['chr5']}:{int(row['bp5'])} -> chr{row['chr3']}:{int(row['bp3'])}")
        if gene5_resolved != row["gene5"] or gene3_resolved != row["gene3"]:
            print(f"  Resolved symbols: {row['gene5']}->{gene5_resolved}, {row['gene3']}->{gene3_resolved}")
        print(f"{'='*80}")

        try:
            fusion_results = reconstructor.reconstruct_isoform_fusions(
                gene5=gene5_resolved,
                chr5=str(row["chr5"]),
                bp5=int(row["bp5"]),
                gene3=gene3_resolved,
                chr3=str(row["chr3"]),
                bp3=int(row["bp3"]),
                min_cds_len=30,
                allow_approximation=True,
                allow_out_of_frame=True,
            )

            if not fusion_results:
                print("  No fusion variants reconstructed")
                test_df.at[idx, "recon_status"] = "no_variants"
                test_df.at[idx, "num_variants"] = 0
                continue

            # Keep selection strategy aligned with run.py
            in_frame_results = [r for r in fusion_results if r.in_frame]
            if in_frame_results:
                best = max(in_frame_results, key=lambda r: len(r.protein_seq))
            else:
                best = max(fusion_results, key=lambda r: len(r.protein_seq))

            reconstructed_seq = best.protein_seq.replace("*", "")
            original_seq = row["original_seq"] if pd.notna(row["original_seq"]) else ""
            reconstructed_len = len(reconstructed_seq)
            original_len = len(original_seq) if original_seq else 0
            length_diff = reconstructed_len - original_len if original_len > 0 else pd.NA

            print("  Reconstruction result:")
            print(f"    Variants found:    {len(fusion_results)}")
            print(f"    Best transcripts:  {best.tx5_id} + {best.tx3_id}")
            print(f"    Quality:           {best.quality}")
            print(f"    Reconstructed len: {reconstructed_len} aa")
            print(f"    In-frame:          {best.in_frame}")

            test_df.at[idx, "recon_status"] = "success"
            test_df.at[idx, "reconstructed_seq"] = reconstructed_seq
            test_df.at[idx, "reconstructed_length"] = reconstructed_len
            test_df.at[idx, "num_variants"] = len(fusion_results)
            test_df.at[idx, "recon_in_frame"] = best.in_frame
            test_df.at[idx, "recon_quality"] = best.quality
            test_df.at[idx, "bp5_approx"] = best.bp5_approximated
            test_df.at[idx, "bp3_approx"] = best.bp3_approximated
            test_df.at[idx, "best_tx_pair"] = f"{best.tx5_id}+{best.tx3_id}"
            test_df.at[idx, "orf_used"] = best.orf_used
            test_df.at[idx, "orf_frame"] = best.orf_frame
            test_df.at[idx, "original_length"] = original_len
            test_df.at[idx, "length_diff"] = length_diff

        except Exception as e:
            print(f"  ERROR: {e}")
            test_df.at[idx, "recon_status"] = "error"
            test_df.at[idx, "recon_error"] = str(e)[:200]

    # Normalize dtypes for reporting
    for col in ["reconstructed_length", "num_variants", "original_length", "length_diff"]:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    for col in ["recon_in_frame", "bp5_approx", "bp3_approx", "orf_used"]:
        test_df[col] = test_df[col].astype("boolean")

    total_time = time.time() - start_time
    successful = test_df[test_df["recon_status"] == "success"]
    no_variants = test_df[test_df["recon_status"] == "no_variants"]
    errors = test_df[test_df["recon_status"] == "error"]

    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Total fusions tested: {len(test_df)}")
    print(f"Successful:  {len(successful):5d} ({len(successful)/len(test_df)*100:.1f}%)")
    print(f"No variants: {len(no_variants):5d} ({len(no_variants)/len(test_df)*100:.1f}%)")
    print(f"Errors:      {len(errors):5d} ({len(errors)/len(test_df)*100:.1f}%)")

    if len(successful) > 0:
        print(f"\n{'─'*80}")
        print("RECONSTRUCTED PROTEIN LENGTH:")
        print(f"{'─'*80}")
        print(f"  Mean:   {successful['reconstructed_length'].mean():.1f} aa")
        print(f"  Median: {successful['reconstructed_length'].median():.1f} aa")
        print(f"  Min:    {successful['reconstructed_length'].min():.0f} aa")
        print(f"  Max:    {successful['reconstructed_length'].max():.0f} aa")

    # Save merged output with original + reconstruction fields.
    results_tsv = f"{output_prefix}_results.tsv"
    test_df.to_csv(results_tsv, sep="\t", index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_tsv}")

    # Save model-ready CSV:
    # keep all original columns + reconstruction columns, only successful rows.
    model_df = test_df[test_df["recon_status"] == "success"].copy()
    model_csv = f"{output_prefix}_reconstructed_sequences.csv"
    model_df.to_csv(model_csv, index=False)
    print(f"Model-input CSV saved to: {model_csv} (rows={len(model_df)})")
    print(f"{'='*80}\n")

    # Generate analysis from successful reconstructions
    if len(successful) > 0:
        print("=" * 80)
        print("GENERATING COMPREHENSIVE ANALYSIS...")
        print("=" * 80)
        try:
            analysis_df = successful.rename(
                columns={
                    "recon_in_frame": "in_frame",
                    "recon_quality": "quality",
                    "num_variants": "n_isoforms",
                }
            )
            generate_final_analysis(
                analysis_df,
                output_prefix=output_prefix,
                dataset_name="Fusions_set1-17",
            )
            print("Analysis and plots generated successfully!")
        except Exception as e:
            print(f"ERROR generating analysis: {e}")
    else:
        print("No successful reconstructions — skipping analysis generation")

    # Save compact log
    log_file = f"{output_prefix}_log.txt"
    with open(log_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FUSION PROTEIN RECONSTRUCTION PIPELINE — Fusions_set1-17 — LOG\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Input file:          {input_file}\n")
        f.write(f"  Genome build:        {genome_build}\n")
        f.write(f"  GTF file:            {gtf_path}\n")
        f.write(f"  Genome file:         {genome_path}\n")
        f.write(f"  Output prefix:       {output_prefix}\n")
        f.write(f"  Sample size:         {n_samples if n_samples else 'all'}\n")
        f.write(f"  ORFfinder:           {'enabled' if use_orffinder else 'disabled'}\n")
        if use_orffinder:
            f.write(f"  ORFfinder path:      {orffinder_path}\n")
        f.write(f"  Only protein-coding: {only_protein_coding}\n")
        f.write(f"  Only with sequence:  {only_with_sequence}\n")
        f.write(f"  Random seed:         {random_state}\n\n")
        f.write(f"Runtime: {total_time/60:.1f} minutes\n")
        f.write(f"Total fusions: {len(test_df)}\n")
        f.write(f"Successful: {len(successful)} ({len(successful)/len(test_df)*100:.1f}%)\n")
        f.write(f"No variants: {len(no_variants)} ({len(no_variants)/len(test_df)*100:.1f}%)\n")
        f.write(f"Errors: {len(errors)} ({len(errors)/len(test_df)*100:.1f}%)\n")

    print(f"\nLog saved to: {log_file}")
    print(f"\n{'='*80}")
    print(f"Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    return test_df


def main():
    parser = argparse.ArgumentParser(
        description="Fusion Protein Reconstruction Pipeline — Fusions_set1-17"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/work/H2020DeciderFicarra/gcapitani/driver-fusion/data/Fusions_set1-17.csv",
        help="Path to Fusions_set1-17 input file (TSV format)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of random fusions to process (default: all matching rows)",
    )
    parser.add_argument(
        "--genome-build",
        type=str,
        default="hg38",
        choices=["hg38"],
        help="Genome build used for references (fixed: hg38)",
    )
    parser.add_argument(
        "--gtf",
        type=str,
        default=None,
        help="Optional custom GTF path (overrides --genome-build default path)",
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=None,
        help="Optional custom genome FASTA path (overrides --genome-build default path)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/work/H2020DeciderFicarra/gcapitani/driver-fusion/fusions_set1_17",
        help="Output prefix for results/log/analysis",
    )
    parser.add_argument(
        "--orffinder-path",
        type=str,
        default="/homes/gcapitani/Gene-Fusions/data/ORFfinder",
        help="Path to ORFfinder executable",
    )
    parser.add_argument(
        "--no-orffinder",
        action="store_true",
        help="Disable ORFfinder and use direct translation",
    )
    parser.add_argument(
        "--include-non-coding",
        action="store_true",
        help="Include non-protein-coding fusions (default: excluded)",
    )
    parser.add_argument(
        "--include-no-sequence",
        action="store_true",
        help="Include rows without peptide_sequence (default: excluded)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild hg38 GTF cache dedicated to this script",
    )
    args = parser.parse_args()

    run_fusions_set_reconstruction(
        input_file=args.input,
        n_samples=args.n,
        genome_build=args.genome_build,
        output_prefix=args.output,
        use_orffinder=not args.no_orffinder,
        orffinder_path=args.orffinder_path,
        only_protein_coding=not args.include_non_coding,
        only_with_sequence=not args.include_no_sequence,
        random_state=args.seed,
        gtf_path=args.gtf,
        genome_path=args.genome,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
