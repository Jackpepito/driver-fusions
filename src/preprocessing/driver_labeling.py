import pandas as pd
import numpy as np
from pathlib import Path
import time

CHIMERSEQ = Path("/homes/gcapitani/Gene-Fusions/data/ChimerSeq4.csv")
CENSUS = "/homes/gcapitani/Gene-Fusions/data/Census_allFri Jun 27 11_25_18 2025.tsv"

# -------------------- Helpers -------------------- #
def parse_first_breakpoint(value):
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if ',' in s:
        s = s.split(',')[0].strip()
    # Remove potential decimals
    try:
        return str(int(float(s)))
    except ValueError:
        return ""

# -------------------- Load & preprocess ChimerSeq -------------------- #
def load_and_prepare_chimerseq():
    # Use low_memory=False to avoid dtype inference issues (mixed types warnings)
    ch = pd.read_csv(CHIMERSEQ, low_memory=False)
    # Normalize positions as string (consistent with fusionpdb first breakpoints)
    ch['H_position_norm'] = ch['H_position'].apply(parse_first_breakpoint)
    ch['T_position_norm'] = ch['T_position'].apply(parse_first_breakpoint)
    ch['H_gene_up'] = ch['H_gene'].astype(str).str.upper()
    ch['T_gene_up'] = ch['T_gene'].astype(str).str.upper()

    # Boolean evidences
    def nonempty(col):
        # Some columns may be missing; handle gracefully by returning False
        if col not in ch.columns:
            return pd.Series(False, index=ch.index)
        return (~ch[col].isna()) & (ch[col].astype(str).str.strip() != '')

    ch['has_pub'] = nonempty('ChimerPub')
    # In ChimerSeq4, the reliability field equivalent to "ChimerSeq+" is "Highly_Reliable_Seq"
    ch['has_seq_plus'] = nonempty('Highly_Reliable_Seq')
    # Do NOT use a non-existent plain "ChimerSeq" column. Presence in ChimerSeq itself implies base evidence.

    # Aggregate by unique quadruple (positions + genes), keeping all columns.
    # Default to first non-null row per group, but preserve boolean evidences as OR.
    group_keys = ['H_position_norm', 'T_position_norm', 'H_gene_up', 'T_gene_up']
    agg_map = {col: 'first' for col in ch.columns if col not in group_keys}
    agg_map['has_pub'] = 'max'
    agg_map['has_seq_plus'] = 'max'
    grouped = ch.groupby(group_keys, as_index=False).agg(agg_map)
    return grouped, ch

# -------------------- Compute recurrence from ChimerSeq -------------------- #
def compute_recurrence(chimerseq_full: pd.DataFrame) -> set:
    """Compute recurrent fusions (≥2 unique patients) from ChimerSeq."""
    # Use the existing Fusion_pair column (already normalized in ChimerSeq4.csv)
    # Count unique patients per fusion
    recurrence = chimerseq_full.groupby('Fusion_pair')['BarcodeID'].nunique()
    recurrent_fusions = set(recurrence[recurrence >= 3].index)
    print(f"  Found {len(recurrent_fusions)} recurrent fusions (≥2 patients) out of {len(recurrence)} total fusion pairs")
    return recurrent_fusions

# -------------------- Load Cancer Gene Census -------------------- #
def load_census_genes() -> set:
    """Load Cancer Gene Census and return set of gene symbols."""
    census = pd.read_csv(CENSUS, sep='\t')
    census_genes = set(census['Gene Symbol'].str.upper())
    print(f"  Loaded {len(census_genes)} genes from Cancer Gene Census")
    return census_genes

# -------------------- Main -------------------- #
def main():
    print("[FAST] Loading data...")
    chimerseq_grouped, chimerseq_full = load_and_prepare_chimerseq()
    
    print("[FAST] Loading Cancer Gene Census...")
    census_genes = load_census_genes()
    
    print("[FAST] Computing fusion recurrence from ChimerSeq...")
    recurrent_fusions = compute_recurrence(chimerseq_full)

    # create a new version of chimerseq_grouped, adding the "is_recurrent" column based on the computed recurrent fusions
    chimerseq_grouped['is_recurrent'] = chimerseq_grouped['Fusion_pair'].isin(recurrent_fusions)

    # add a new column in chimerseq_group based on: if has_seq_plus, is_recurrent, and if either gene is in census, then "driver", else "non-driver" 
    def classify_driver(row):
        if row['has_seq_plus'] and row['is_recurrent'] and row['Frame'] == 'In-Frame' and ((row['H_gene_up'] in census_genes) or (row['T_gene_up'] in census_genes)):
            return "driver"
        if not row['has_seq_plus'] and row['Frame'] == 'In-Frame':#and ((row['H_gene_up'] in census_genes) or (row['T_gene_up'] in census_genes)):# :
            return "non-driver"
        # set to None if it doesn't meet any criteria, to distinguish from non-driver that meets some criteria but not all
        return None

    chimerseq_grouped['driver'] = chimerseq_grouped.apply(classify_driver, axis=1)

    #save a new csv but exlude samples with None in driver
    chimerseq_grouped_filtered = chimerseq_grouped[chimerseq_grouped['driver'].notna()]
    #reset the index of the filtered dataframe
    chimerseq_grouped_filtered = chimerseq_grouped_filtered.reset_index(drop=True)
    chimerseq_grouped_filtered.to_csv("data/chimerseq_labeled.csv", index=False)

    # Output directory for plots and summaries
    output_dir = Path("labeling")
    output_dir.mkdir(parents=True, exist_ok=True)

    #print the number of drivers and non-drivers and some analysis of the data
    num_drivers = (chimerseq_grouped_filtered['driver'] == 'driver').sum()
    num_non_drivers = (chimerseq_grouped_filtered['driver'] == 'non-driver').sum()
    print(f"  Classified {num_drivers} drivers and {num_non_drivers} non-drivers out of {len(chimerseq_grouped_filtered)} total fusion pairs")

    # Plot Junction_reads_num as percentage within each class (drivers/non-drivers).
    # Limit x-axis to 1000 for better visualization.
    import matplotlib.pyplot as plt
    data_drivers = chimerseq_grouped_filtered[chimerseq_grouped_filtered['driver'] == 'driver']
    data_non_drivers = chimerseq_grouped_filtered[chimerseq_grouped_filtered['driver'] == 'non-driver']
    num_drivers_class = len(data_drivers)
    num_non_drivers_class = len(data_non_drivers)

    # Use shared, very narrow bins for both classes.
    bin_width = 10
    bin_edges = np.arange(0, 1000 + bin_width, bin_width)

    plt.figure(figsize=(10, 6))
    if num_drivers_class > 0:
        plt.hist(
            data_drivers['Junction_reads_num'],
            bins=bin_edges,
            alpha=0.5,
            label='Drivers',
            color='blue',
            weights=np.ones(num_drivers_class) * 100 / num_drivers_class,
        )
    if num_non_drivers_class > 0:
        plt.hist(
            data_non_drivers['Junction_reads_num'],
            bins=bin_edges,
            alpha=0.5,
            label='Non-Drivers',
            color='orange',
            weights=np.ones(num_non_drivers_class) * 100 / num_non_drivers_class,
        )
    plt.xlim(0, 1000)
    plt.xlabel('Junction Reads Number')
    plt.ylabel('Percentage Within Class (%)')
    plt.title('Junction Reads Distribution (% Within Each Class)')
    plt.legend()
    plt.savefig(output_dir / "junction_reads_percentage_within_class.png")
    plt.close()

    print(
        f"Junction reads for drivers: "
        f"mean={data_drivers['Junction_reads_num'].mean():.2f}, "
        f"std={data_drivers['Junction_reads_num'].std():.2f}, "
        f"median={data_drivers['Junction_reads_num'].median():.2f}"
    )
    print(
        f"Junction reads for non-drivers: "
        f"mean={data_non_drivers['Junction_reads_num'].mean():.2f}, "
        f"std={data_non_drivers['Junction_reads_num'].std():.2f}, "
        f"median={data_non_drivers['Junction_reads_num'].median():.2f}"
    )

    #analisi sulla frequenza/tipolgoia di tumore ma anche altre colonne presenti che possono essere utili
    tumor_freq_drivers = data_drivers['Cancertype'].value_counts()
    tumor_freq_non_drivers = data_non_drivers['Cancertype'].value_counts()
    print("Tumor frequency for drivers:")
    print(tumor_freq_drivers)
    print("Tumor frequency for non-drivers:")
    print(tumor_freq_non_drivers)

    #plot tumor frequency for drivers and non-drivers as bar plots, and save the plots as png files
    plt.figure(figsize=(12, 6))
    tumor_freq_drivers.plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Tumor Frequency for Drivers')
    plt.xlabel('Cancer Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "tumor_freq_drivers.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    tumor_freq_non_drivers.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Tumor Frequency for Non-Drivers')
    plt.xlabel('Cancer Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "tumor_freq_non_drivers.png")
    plt.close()

    # Additional plots
    plt.figure(figsize=(7, 5))
    class_counts = chimerseq_grouped_filtered['driver'].value_counts().reindex(['driver', 'non-driver']).fillna(0)
    class_counts.index = ['Drivers', 'Non-Drivers']
    class_counts.plot(kind='bar', color=['blue', 'orange'], alpha=0.8)
    plt.title('Class Counts')
    plt.xlabel('Class')
    plt.ylabel('Number of Fusions')
    plt.tight_layout()
    plt.savefig(output_dir / "class_counts.png")
    plt.close()

    violin_generated = False
    if num_drivers_class > 0 and num_non_drivers_class > 0:
        plt.figure(figsize=(8, 5))
        violin_data = [
            data_drivers['Junction_reads_num'].dropna(),
            data_non_drivers['Junction_reads_num'].dropna(),
        ]
        parts = plt.violinplot(violin_data, showmeans=True, showmedians=True)
        # Color each violin by class for consistency with other plots.
        violin_colors = ['blue', 'orange']
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(violin_colors[i])
            body.set_edgecolor('black')
            body.set_alpha(0.5)
        plt.xticks([1, 2], ['Drivers', 'Non-Drivers'])
        plt.ylim(0, 2500)
        plt.title('Junction Reads Comparison by Class (Violin Plot)')
        plt.ylabel('Junction Reads Number')
        plt.tight_layout()
        plt.savefig(output_dir / "junction_reads_violin.png")
        plt.close()
        violin_generated = True

    # Summary file
    total_pairs = len(chimerseq_grouped_filtered)
    driver_pct = (num_drivers / total_pairs * 100) if total_pairs else 0.0
    non_driver_pct = (num_non_drivers / total_pairs * 100) if total_pairs else 0.0

    summary_lines = [
        "Driver Labeling Summary",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total labeled fusion pairs: {total_pairs}",
        f"Drivers: {num_drivers} ({driver_pct:.2f}%)",
        f"Non-drivers: {num_non_drivers} ({non_driver_pct:.2f}%)",
        "",
        "Junction_reads_num stats:",
        f"  Drivers -> mean: {data_drivers['Junction_reads_num'].mean():.2f}, std: {data_drivers['Junction_reads_num'].std():.2f}, median: {data_drivers['Junction_reads_num'].median():.2f}",
        f"  Non-drivers -> mean: {data_non_drivers['Junction_reads_num'].mean():.2f}, std: {data_non_drivers['Junction_reads_num'].std():.2f}, median: {data_non_drivers['Junction_reads_num'].median():.2f}",
        "",
        "Top 10 tumor types (drivers):",
    ]
    summary_lines.extend([f"  {k}: {v}" for k, v in tumor_freq_drivers.head(10).items()])
    summary_lines.append("")
    summary_lines.append("Top 10 tumor types (non-drivers):")
    summary_lines.extend([f"  {k}: {v}" for k, v in tumor_freq_non_drivers.head(10).items()])
    summary_lines.append("")
    summary_lines.append("Generated files:")
    summary_lines.append("  - junction_reads_percentage_within_class.png")
    summary_lines.append("  - tumor_freq_drivers.png")
    summary_lines.append("  - tumor_freq_non_drivers.png")
    summary_lines.append("  - class_counts.png")
    if violin_generated:
        summary_lines.append("  - junction_reads_violin.png")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved plots and summary in: {output_dir.resolve()}")

if __name__ == '__main__':
    main()
