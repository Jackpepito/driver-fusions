#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run.py - Main entry point for fusion reconstruction pipeline (ChimerSeq dataset)

This script provides a command-line interface to:
1. Load fusion data from ChimerSeq labeled CSV
2. Reconstruct fusion proteins using isoform-aware pipeline
3. Analyze reconstruction quality and frame consistency
4. Stratify results by driver status and cancer type
5. Generate comprehensive analysis plots and statistics

When genome_build='all' (default), a separate reconstructor is loaded for each
genome build present in the data (hg19 and hg38), and each fusion is processed
with the matching reference files automatically.

Usage:
    python run.py --n 1000
    python run.py --input data/chimerseq_labeled.csv --output chimerseq_analysis
    python run.py --genome-build hg19 --output hg19_only
    python run.py --n 500 --drivers-only --output driver_analysis
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Add parent directory to path to find seq_reconstruction package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the seq_reconstruction package
from seq_reconstruction import (
    IsoformAwareFusionReconstructor,
    generate_final_analysis
)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_PATHS = {
    'hg19': {
        'gtf': '/homes/gcapitani/Gene-Fusions/data/gencode.v19.annotation.gtf',
        'genome': '/homes/gcapitani/Gene-Fusions/data/hg19.fa',
    },
    'hg38': {
        'gtf': '/homes/gcapitani/Gene-Fusions/data/gencode.v47.annotation.gtf',
        'genome': '/homes/gcapitani/Gene-Fusions/data/hg38.fa',
    },
}

# Normalize the Frame column from ChimerSeq into broad categories
FRAME_CATEGORIES = {
    'In-Frame': 'in_frame',
    'In-frame': 'in_frame',
    'Out-of-frame': 'out_of_frame',
    'Out-of-Frame': 'out_of_frame',
    "5'UTR-CDS": 'utr',
    "CDS-5'UTR": 'utr',
    "5'UTR-5'UTR": 'utr',
    "CDS-3'UTR": 'utr',
    "3'UTR-CDS": 'utr',
    "5'UTR-3'UTR": 'utr',
    "3'UTR-3'UTR": 'utr',
    "3'UTR-5'UTR": 'utr',
    "5UTR-CDS": 'utr',
    "5UTR-5UTR": 'utr',
    "CDS-5UTR": 'utr',
}


def run_chimerseq_reconstruction(
    input_file: str,
    n_samples: int = None,
    genome_build: str = "all",
    output_prefix: str = "chimerseq_analysis",
    use_orffinder: bool = True,
    orffinder_path: str = "/homes/gcapitani/Gene-Fusions/data/ORFfinder",
    min_protein_len_aa: int = 30,
    drivers_only: bool = False,
    cancer_type: str = None,
    random_state: int = 42
):
    """
    Run fusion reconstruction on ChimerSeq labeled dataset.

    When genome_build='all', a separate reconstructor is loaded for each genome
    build present in the data, and each fusion is processed with the matching
    reference. When a specific build is given, only those fusions are processed.

    Args:
        input_file: Path to chimerseq_labeled.csv
        n_samples: Number of fusions to process (None = all matching fusions)
        genome_build: 'all' to use the correct build per sample, or 'hg19'/'hg38'
                      to restrict to a single build.
        output_prefix: Prefix for output files
        use_orffinder: Whether to use ORFfinder for ORF detection
        orffinder_path: Path to ORFfinder executable
        min_protein_len_aa: Minimum reconstructed protein length in amino acids
        drivers_only: If True, only process driver fusions
        cancer_type: If set, filter to this cancer type (e.g. 'BRCA')
        random_state: Random seed for sampling

    Returns:
        DataFrame with reconstruction results
    """

    # Banner
    print("=" * 80)
    print("FUSION PROTEIN RECONSTRUCTION PIPELINE — ChimerSeq")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    print("Configuration:")
    print(f"  Input file:      {input_file}")
    print(f"  Genome build:    {genome_build}")
    print(f"  Output prefix:   {output_prefix}")
    print(f"  Sample size:     {n_samples if n_samples else 'all'}")
    print(f"  ORFfinder:       {'enabled' if use_orffinder else 'disabled'}")
    if use_orffinder:
        print(f"  ORFfinder path:  {orffinder_path}")
    print(f"  Min protein len: {min_protein_len_aa} aa")
    print(f"  Drivers only:    {drivers_only}")
    print(f"  Cancer type:     {cancer_type if cancer_type else 'all'}")
    print(f"  Random seed:     {random_state}")
    print()

    start_time = time.time()

    # ── Load and filter data ──────────────────────────────────────────────
    print("Loading data...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    print(f"Loaded {len(df)} total rows\n")

    # Show genome build distribution
    build_counts = df['Genome_Build_Version'].value_counts()
    print("Genome build distribution:")
    for build, count in build_counts.items():
        selected = (genome_build == 'all' or build == genome_build)
        marker = " <-- selected" if selected else " (skipped)"
        print(f"  {build}: {count} rows{marker}")
    print()

    # Determine which builds to process
    if genome_build == 'all':
        builds_to_process = sorted(df['Genome_Build_Version'].unique())
    else:
        builds_to_process = [genome_build]
        df = df[df['Genome_Build_Version'] == genome_build].copy()

    # Validate that we have reference paths for all requested builds
    for b in builds_to_process:
        if b not in DEFAULT_PATHS:
            print(f"ERROR: No reference paths configured for genome build '{b}'.")
            print(f"  Available builds in DEFAULT_PATHS: {list(DEFAULT_PATHS.keys())}")
            print(f"  Add paths for '{b}' in DEFAULT_PATHS or restrict with --genome-build")
            sys.exit(1)

    print(f"Builds to process: {builds_to_process}")
    for b in builds_to_process:
        print(f"  {b}: GTF    = {DEFAULT_PATHS[b]['gtf']}")
        print(f"  {' '*len(b)}  Genome = {DEFAULT_PATHS[b]['genome']}")
    print()

    if len(df) == 0:
        print("ERROR: No fusions found after filtering")
        sys.exit(1)

    # Filter by driver status
    if drivers_only:
        df = df[df['driver'] == 'driver'].copy()
        print(f"After filtering for drivers only: {len(df)} rows")

    # Filter by cancer type
    if cancer_type:
        df = df[df['Cancertype'] == cancer_type].copy()
        print(f"After filtering for cancer type '{cancer_type}': {len(df)} rows")

    if len(df) == 0:
        print("ERROR: No fusions left after filtering")
        sys.exit(1)

    # Clean chromosome columns (strip 'chr' prefix for the reconstructor)
    df['H_chr_clean'] = df['H_chr'].astype(str).str.replace('chr', '', regex=False)
    df['T_chr_clean'] = df['T_chr'].astype(str).str.replace('chr', '', regex=False)
    df['H_position'] = pd.to_numeric(df['H_position'], errors='coerce')
    df['T_position'] = pd.to_numeric(df['T_position'], errors='coerce')
    df = df.dropna(subset=['H_gene', 'T_gene', 'H_chr_clean', 'T_chr_clean',
                           'H_position', 'T_position'])

    # Map Frame annotation to broad category
    df['frame_category'] = df['Frame'].map(FRAME_CATEGORIES).fillna('unknown')

    print(f"Clean dataset: {len(df)} rows")
    print(f"  Drivers:     {(df['driver'] == 'driver').sum()}")
    print(f"  Non-drivers: {(df['driver'] == 'non-driver').sum()}")
    print(f"  Frame categories: {df['frame_category'].value_counts().to_dict()}")
    if genome_build == 'all':
        print(f"  Per-build counts: {df['Genome_Build_Version'].value_counts().to_dict()}")
    print()

    # Sample (if requested)
    if n_samples and n_samples < len(df):
        test_df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        print(f"Sampled {n_samples} fusions for reconstruction\n")
    else:
        test_df = df.reset_index(drop=True)
        print(f"Processing all {len(test_df)} fusions\n")

    n = len(test_df)

    # ── Initialize reconstructors (one per genome build) ─────────────────
    reconstructors = {}
    cache_dirs = {}
    for b in builds_to_process:
        cache_dirs[b] = str((Path("cache") / f"gtf_{b}_chimerseq").resolve())
        print(f"Initializing reconstructor for {b}...")
        print(f"  Cache dir: {cache_dirs[b]}")
        try:
            reconstructors[b] = IsoformAwareFusionReconstructor(
                mode='local',
                gtf_path=DEFAULT_PATHS[b]['gtf'],
                genome_path=DEFAULT_PATHS[b]['genome'],
                cache_dir=cache_dirs[b],
                use_orffinder=use_orffinder,
                orffinder_path=orffinder_path
            )
            print(f"  Reconstructor for {b} ready!")
        except Exception as e:
            print(f"ERROR initializing reconstructor for {b}: {e}")
            sys.exit(1)
    print()

    # ── Process fusions ───────────────────────────────────────────────────
    # Add reconstruction columns directly to test_df (preserving all original cols)
    recon_cols = ['recon_status', 'reconstructed_seq', 'reconstructed_length',
                  'num_variants', 'recon_in_frame', 'frame_consistent',
                  'recon_quality', 'bp5_approx', 'bp3_approx',
                  'best_tx_pair', 'orf_used', 'orf_frame', 'recon_error']
    for col in recon_cols:
        test_df[col] = pd.NA

    for idx in range(n):
        row = test_df.iloc[idx]
        fusion_name = f"{row['H_gene']}-{row['T_gene']}"
        sample_build = row['Genome_Build_Version']
        reconstructor = reconstructors[sample_build]

        print(f"\n{'='*80}")
        print(f"[{idx+1}/{n}] {fusion_name}  ({sample_build})")
        print(f"  chr{row['H_chr_clean']}:{int(row['H_position'])} ({row['H_strand']}) -> "
              f"chr{row['T_chr_clean']}:{int(row['T_position'])} ({row['T_strand']})")
        print(f"  Annotated frame: {row['Frame']}  |  Driver: {row['driver']}  |  Cancer: {row['Cancertype']}")
        print(f"{'='*80}")

        try:
            fusion_results = reconstructor.reconstruct_isoform_fusions(
                gene5=row['H_gene'],
                chr5=str(row['H_chr_clean']),
                bp5=int(row['H_position']),
                gene3=row['T_gene'],
                chr3=str(row['T_chr_clean']),
                bp3=int(row['T_position']),
                min_cds_len=30,
                allow_approximation=True,
                allow_out_of_frame=True,
            )

            if not fusion_results:
                print("\n  No fusion variants reconstructed")
                test_df.at[idx, 'recon_status'] = 'no_variants'
                test_df.at[idx, 'num_variants'] = 0
                continue

            if min_protein_len_aa > 0:
                fusion_results = [
                    r for r in fusion_results
                    if len(r.protein_seq.replace('*', '')) >= min_protein_len_aa
                ]
                if not fusion_results:
                    print(f"\n  All variants discarded: reconstructed protein shorter than {min_protein_len_aa} aa")
                    test_df.at[idx, 'recon_status'] = 'no_variants'
                    test_df.at[idx, 'num_variants'] = 0
                    continue

            # Select best result: prioritize in-frame, then longest protein
            in_frame_results = [r for r in fusion_results if r.in_frame]
            if in_frame_results:
                best = max(in_frame_results, key=lambda r: len(r.protein_seq))
            else:
                best = max(fusion_results, key=lambda r: len(r.protein_seq))

            reconstructed_seq = best.protein_seq.replace('*', '')
            reconstructed_len = len(reconstructed_seq)
            annotated_inframe = row['frame_category'] == 'in_frame'
            reconstructed_inframe = best.in_frame
            frame_consistent = (annotated_inframe == reconstructed_inframe)

            print(f"\n{'─'*80}")
            print("RECONSTRUCTION RESULT:")
            print(f"{'─'*80}")
            print(f"  Variants found:     {len(fusion_results)}")
            print(f"  Best transcripts:   {best.tx5_id} + {best.tx3_id}")
            print(f"  Quality:            {best.quality}")
            print(f"  Reconstructed len:  {reconstructed_len} aa")
            print(f"  In-frame (recon):   {reconstructed_inframe}")
            print(f"  Frame consistent:   {frame_consistent}")
            print(f"{'─'*80}")

            test_df.at[idx, 'recon_status'] = 'success'
            test_df.at[idx, 'reconstructed_seq'] = reconstructed_seq
            test_df.at[idx, 'reconstructed_length'] = reconstructed_len
            test_df.at[idx, 'num_variants'] = len(fusion_results)
            test_df.at[idx, 'recon_in_frame'] = reconstructed_inframe
            test_df.at[idx, 'frame_consistent'] = frame_consistent
            test_df.at[idx, 'recon_quality'] = best.quality
            test_df.at[idx, 'bp5_approx'] = best.bp5_approximated
            test_df.at[idx, 'bp3_approx'] = best.bp3_approximated
            test_df.at[idx, 'best_tx_pair'] = f"{best.tx5_id}+{best.tx3_id}"
            test_df.at[idx, 'orf_used'] = best.orf_used
            test_df.at[idx, 'orf_frame'] = best.orf_frame

        except Exception as e:
            print(f"\n  ERROR: {str(e)}")
            test_df.at[idx, 'recon_status'] = 'error'
            test_df.at[idx, 'recon_error'] = str(e)[:200]

    # ── Summary report ────────────────────────────────────────────────────
    total_time = time.time() - start_time
    results_df = test_df

    # Cast reconstruction columns to proper dtypes (pd.NA makes them object)
    for col in ['reconstructed_length', 'num_variants']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    for col in ['recon_in_frame', 'frame_consistent', 'bp5_approx', 'bp3_approx', 'orf_used']:
        results_df[col] = results_df[col].astype('boolean')

    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Total fusions tested: {len(results_df)}")

    successful = results_df[results_df['recon_status'] == 'success']
    no_variants = results_df[results_df['recon_status'] == 'no_variants']
    errors = results_df[results_df['recon_status'] == 'error']

    print(f"Successful:  {len(successful):5d} ({len(successful)/len(results_df)*100:.1f}%)")
    print(f"No variants: {len(no_variants):5d} ({len(no_variants)/len(results_df)*100:.1f}%)")
    print(f"Errors:      {len(errors):5d} ({len(errors)/len(results_df)*100:.1f}%)")

    # Per-build breakdown
    if len(builds_to_process) > 1:
        print(f"\n{'─'*80}")
        print("PER-BUILD BREAKDOWN:")
        print(f"{'─'*80}")
        for b in builds_to_process:
            b_total = results_df[results_df['Genome_Build_Version'] == b]
            b_succ = successful[successful['Genome_Build_Version'] == b] if len(successful) > 0 else pd.DataFrame()
            b_novar = no_variants[no_variants['Genome_Build_Version'] == b] if len(no_variants) > 0 else pd.DataFrame()
            b_err = errors[errors['Genome_Build_Version'] == b] if len(errors) > 0 else pd.DataFrame()
            print(f"  {b}: {len(b_total)} total -> "
                  f"{len(b_succ)} success, {len(b_novar)} no_variants, {len(b_err)} errors")

    if len(successful) > 0:
        # --- Protein length ---
        print(f"\n{'─'*80}")
        print("RECONSTRUCTED PROTEIN LENGTH:")
        print(f"{'─'*80}")
        print(f"  Mean:   {successful['reconstructed_length'].mean():.1f} aa")
        print(f"  Median: {successful['reconstructed_length'].median():.1f} aa")
        print(f"  Min:    {successful['reconstructed_length'].min():.0f} aa")
        print(f"  Max:    {successful['reconstructed_length'].max():.0f} aa")

        # --- Reconstruction quality ---
        print(f"\n{'─'*80}")
        print("RECONSTRUCTION QUALITY:")
        print(f"{'─'*80}")
        quality_counts = successful['recon_quality'].value_counts()
        for qual, count in quality_counts.items():
            print(f"  {qual:<30}: {count:4d} ({count/len(successful)*100:5.1f}%)")

        # --- Frame consistency ---
        print(f"\n{'─'*80}")
        print("FRAME CONSISTENCY (annotated vs reconstructed):")
        print(f"{'─'*80}")
        # Only compare for fusions with in_frame/out_of_frame annotation
        has_frame = successful[successful['frame_category'].isin(['in_frame', 'out_of_frame'])]
        if len(has_frame) > 0:
            consistent = has_frame['frame_consistent'].sum()
            print(f"  Fusions with frame annotation: {len(has_frame)}")
            print(f"  Frame consistent:    {consistent:4d} ({consistent/len(has_frame)*100:5.1f}%)")
            print(f"  Frame inconsistent:  {len(has_frame)-consistent:4d} ({(len(has_frame)-consistent)/len(has_frame)*100:5.1f}%)")
        else:
            print("  No fusions with in-frame/out-of-frame annotation")

        # --- Breakpoint approximations ---
        print(f"\n{'─'*80}")
        print("BREAKPOINT APPROXIMATIONS:")
        print(f"{'─'*80}")
        bp5_approx = successful['bp5_approx'].sum()
        bp3_approx = successful['bp3_approx'].sum()
        print(f"  BP5 approximated:  {bp5_approx} ({bp5_approx/len(successful)*100:.1f}%)")
        print(f"  BP3 approximated:  {bp3_approx} ({bp3_approx/len(successful)*100:.1f}%)")
        both_approx = successful[successful['bp5_approx'] & successful['bp3_approx']]
        print(f"  Both approximated: {len(both_approx)} ({len(both_approx)/len(successful)*100:.1f}%)")

        # --- ORFfinder ---
        print(f"\n{'─'*80}")
        print("ORFFINDER USAGE:")
        print(f"{'─'*80}")
        orf_used_count = successful['orf_used'].sum()
        print(f"  ORFfinder used: {orf_used_count} ({orf_used_count/len(successful)*100:.1f}%)")
        if orf_used_count > 0:
            orf_data = successful[successful['orf_used'] == True]
            frame_counts = orf_data['orf_frame'].value_counts()
            for fr in [0, 1, 2]:
                cnt = frame_counts.get(fr, 0)
                print(f"    Frame {fr}: {cnt} ({cnt/orf_used_count*100:.1f}%)")

        # --- Driver vs Non-driver stratification ---
        print(f"\n{'─'*80}")
        print("DRIVER vs NON-DRIVER STRATIFICATION:")
        print(f"{'─'*80}")
        for label in ['driver', 'non-driver']:
            subset = successful[successful['driver'] == label]
            if len(subset) == 0:
                continue
            n_inframe = subset['recon_in_frame'].sum()
            mean_len = subset['reconstructed_length'].mean()
            mean_vars = subset['num_variants'].mean()
            print(f"\n  [{label.upper()}] (n={len(subset)})")
            print(f"    Mean protein length: {mean_len:.1f} aa")
            print(f"    In-frame:            {n_inframe} ({n_inframe/len(subset)*100:.1f}%)")
            print(f"    Mean isoform combos: {mean_vars:.1f}")

            # Frame consistency for this class
            has_fr = subset[subset['frame_category'].isin(['in_frame', 'out_of_frame'])]
            if len(has_fr) > 0:
                cons = has_fr['frame_consistent'].sum()
                print(f"    Frame consistent:    {cons}/{len(has_fr)} ({cons/len(has_fr)*100:.1f}%)")

            # Quality breakdown
            qc = subset['recon_quality'].value_counts()
            for q, c in qc.items():
                print(f"    {q}: {c} ({c/len(subset)*100:.1f}%)")

        # --- Top cancer types ---
        print(f"\n{'─'*80}")
        print("TOP 10 CANCER TYPES (by success count):")
        print(f"{'─'*80}")
        cancer_counts = successful['Cancertype'].value_counts().head(10)
        for cancer, count in cancer_counts.items():
            in_fr = successful[successful['Cancertype'] == cancer]['recon_in_frame'].sum()
            print(f"  {cancer:<10}: {count:4d} fusions, {in_fr:4d} in-frame ({in_fr/count*100:.1f}%)")

        # --- Top longest / shortest reconstructions ---
        print(f"\n{'─'*80}")
        print("TOP 5 LONGEST RECONSTRUCTIONS:")
        print(f"{'─'*80}")
        top5 = successful.nlargest(5, 'reconstructed_length')
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            fname = f"{r['H_gene']}-{r['T_gene']}"
            print(f"  {i}. {fname:<25} | {int(r['reconstructed_length']):5d} aa | {r['recon_quality']} | {r['driver']} | {r['Genome_Build_Version']}")

        print(f"\n{'─'*80}")
        print("TOP 5 SHORTEST RECONSTRUCTIONS:")
        print(f"{'─'*80}")
        bottom5 = successful.nsmallest(5, 'reconstructed_length')
        for i, (_, r) in enumerate(bottom5.iterrows(), 1):
            fname = f"{r['H_gene']}-{r['T_gene']}"
            print(f"  {i}. {fname:<25} | {int(r['reconstructed_length']):5d} aa | {r['recon_quality']} | {r['driver']} | {r['Genome_Build_Version']}")

    # ── Save results ──────────────────────────────────────────────────────
    # Drop temporary columns before saving
    drop_cols = ['H_chr_clean', 'T_chr_clean']
    if results_df['recon_error'].isna().all():
        drop_cols.append('recon_error')
    results_df = results_df.drop(columns=[c for c in drop_cols if c in results_df.columns])

    results_csv = f"{output_prefix}_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_csv}")
    print(f"{'='*80}\n")

    # ── Generate comprehensive analysis ───────────────────────────────────
    if len(successful) > 0:
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE ANALYSIS...")
        print("=" * 80 + "\n")
        try:
            # Rename columns to match generate_final_analysis expectations
            analysis_df = successful.rename(columns={
                'recon_in_frame': 'in_frame',
                'recon_quality': 'quality',
                'num_variants': 'n_isoforms',
            })
            generate_final_analysis(
                analysis_df, output_prefix=output_prefix,
                dataset_name="ChimerSeq"
            )
            print("\nAnalysis and plots generated successfully!")
        except Exception as e:
            print(f"\nERROR generating analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo successful reconstructions — skipping analysis generation")

    # ── Save log ──────────────────────────────────────────────────────────
    log_file = f"{output_prefix}_log.txt"
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FUSION PROTEIN RECONSTRUCTION PIPELINE — ChimerSeq — LOG\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Input file:      {input_file}\n")
        f.write(f"  Genome build:    {genome_build}\n")
        f.write(f"  Builds loaded:   {builds_to_process}\n")
        for b in builds_to_process:
            f.write(f"    {b}: GTF={DEFAULT_PATHS[b]['gtf']}\n")
            f.write(f"    {' '*len(b)}  Genome={DEFAULT_PATHS[b]['genome']}\n")
            f.write(f"    {' '*len(b)}  Cache={cache_dirs[b]}\n")
        f.write(f"  Output prefix:   {output_prefix}\n")
        f.write(f"  Sample size:     {n_samples if n_samples else 'all'}\n")
        f.write(f"  ORFfinder:       {'enabled' if use_orffinder else 'disabled'}\n")
        if use_orffinder:
            f.write(f"  ORFfinder path:  {orffinder_path}\n")
        f.write(f"  Drivers only:    {drivers_only}\n")
        f.write(f"  Cancer type:     {cancer_type if cancer_type else 'all'}\n")
        f.write(f"  Random seed:     {random_state}\n\n")

        f.write(f"Runtime: {total_time/60:.1f} minutes\n")
        f.write(f"Total fusions: {len(results_df)}\n")
        f.write(f"Successful: {len(successful)} ({len(successful)/len(results_df)*100:.1f}%)\n")
        f.write(f"No variants: {len(no_variants)} ({len(no_variants)/len(results_df)*100:.1f}%)\n")
        f.write(f"Errors: {len(errors)} ({len(errors)/len(results_df)*100:.1f}%)\n")

        if len(builds_to_process) > 1:
            f.write("\nPer-build breakdown:\n")
            for b in builds_to_process:
                b_total = results_df[results_df['Genome_Build_Version'] == b]
                b_succ = successful[successful['Genome_Build_Version'] == b] if len(successful) > 0 else pd.DataFrame()
                f.write(f"  {b}: {len(b_total)} total, {len(b_succ)} success\n")

        if len(successful) > 0:
            f.write(f"\nReconstructed length: "
                    f"{successful['reconstructed_length'].mean():.1f} +/- "
                    f"{successful['reconstructed_length'].std():.1f} aa\n")
            f.write(f"In-frame: {successful['recon_in_frame'].sum()}/{len(successful)} "
                    f"({successful['recon_in_frame'].sum()/len(successful)*100:.1f}%)\n")
            has_frame = successful[successful['frame_category'].isin(['in_frame', 'out_of_frame'])]
            if len(has_frame) > 0:
                cons = has_frame['frame_consistent'].sum()
                f.write(f"Frame consistency: {cons}/{len(has_frame)} "
                        f"({cons/len(has_frame)*100:.1f}%)\n")

    print(f"\nLog saved to: {log_file}")

    print(f"\n{'='*80}")
    print(f"Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    return results_df


def main():
    """Main entry point with argument parsing."""

    parser = argparse.ArgumentParser(
        description='Fusion Protein Reconstruction Pipeline — ChimerSeq Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all fusions (both hg19 and hg38, using correct build per sample)
  python run.py

  # Process only hg19 fusions
  python run.py --genome-build hg19

  # Process 500 random fusions
  python run.py --n 500

  # Only driver fusions
  python run.py --drivers-only --output driver_analysis

  # Specific cancer type
  python run.py --cancer-type BRCA --output brca_analysis

  # Disable ORFfinder
  python run.py --n 1000 --no-orffinder

  # Full custom run (single build)
  python run.py \\
    --input data/chimerseq_labeled.csv \\
    --genome-build hg19 \\
    --n 2000 \\
    --drivers-only \\
    --output driver_hg19 \\
    --seed 123

  # All builds, all fusions
  python run.py --output full_analysis
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='/homes/gcapitani/driver-fusions/data/chimerseq_labeled.csv',
        help='Path to ChimerSeq labeled CSV file (default: data/chimerseq_labeled.csv)'
    )

    parser.add_argument(
        '--n',
        type=int,
        default=None,
        help='Number of fusions to process (default: all matching fusions)'
    )

    parser.add_argument(
        '--genome-build',
        type=str,
        default='all',
        choices=['all', 'hg19', 'hg38'],
        help='Genome build: "all" uses the correct build per sample, or restrict to hg19/hg38 (default: all)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='chimerseq_analysis',
        help='Output prefix for results, plots, and logs (default: chimerseq_analysis)'
    )

    parser.add_argument(
        '--no-orffinder',
        action='store_true',
        help='Disable ORFfinder and use direct translation'
    )

    parser.add_argument(
        '--orffinder-path',
        type=str,
        default='/homes/gcapitani/Gene-Fusions/data/ORFfinder',
        help='Path to ORFfinder executable'
    )
    parser.add_argument(
        '--min-protein-len-aa',
        type=int,
        default=30,
        help='Minimum reconstructed protein length in amino acids (default: 30, set 0 to disable)'
    )

    parser.add_argument(
        '--drivers-only',
        action='store_true',
        help='Process only driver fusions'
    )

    parser.add_argument(
        '--cancer-type',
        type=str,
        default=None,
        help='Filter to a specific cancer type (e.g. BRCA, LUAD, OV)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    args = parser.parse_args()

    run_chimerseq_reconstruction(
        input_file=args.input,
        n_samples=args.n,
        genome_build=args.genome_build,
        output_prefix=args.output,
        use_orffinder=not args.no_orffinder,
        orffinder_path=args.orffinder_path,
        min_protein_len_aa=args.min_protein_len_aa,
        drivers_only=args.drivers_only,
        cancer_type=args.cancer_type,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
