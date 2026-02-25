import pandas as pd
import numpy as np
from pathlib import Path
import time

CHIMERSEQ = Path("/homes/gcapitani/Gene-Fusions/data/ChimerSeq4.csv")
FUSIONPDB_CLEANED = Path("/homes/gcapitani/Gene-Fusions/data/FusionPDB_cleaned.csv")
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

# -------------------- Load & preprocess fusionpdb -------------------- #
def load_and_prepare_fusionpdb():
    fusionpdb = pd.read_csv(FUSIONPDB_CLEANED)
    # Extract gene parts and first breakpoints once
    genes_split = fusionpdb['fusiongenes'].str.split('::', n=1, expand=True)
    fusionpdb['fusion_h_gene'] = genes_split[0].str.upper()
    fusionpdb['fusion_t_gene'] = genes_split[1].str.upper()
    fusionpdb['h_bp_first'] = fusionpdb['hgene_bp'].apply(parse_first_breakpoint)
    fusionpdb['t_bp_first'] = fusionpdb['tgene_bp'].apply(parse_first_breakpoint)
    # Keep only needed columns + aa_seq for join on sequence
    keep_cols = ['aa_seq','fusion_h_gene','fusion_t_gene','h_bp_first','t_bp_first']
    fusionpdb_small = fusionpdb[keep_cols].copy()
    fusionpdb_small = fusionpdb_small.rename(columns={'aa_seq':'sequence'})
    return fusionpdb_small

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
        if not row['has_seq_plus'] and ((row['H_gene_up'] in census_genes) or (row['T_gene_up'] in census_genes)):# and row['Frame'] == 'In-Frame':
            return "non-driver"
        # set to None if it doesn't meet any criteria, to distinguish from non-driver that meets some criteria but not all
        return None

    chimerseq_grouped['driver'] = chimerseq_grouped.apply(classify_driver, axis=1)

    #save a new csv but exlude samples with None in driver
    chimerseq_grouped_filtered = chimerseq_grouped[chimerseq_grouped['driver'].notna()]
    #reset the index of the filtered dataframe
    chimerseq_grouped_filtered = chimerseq_grouped_filtered.reset_index(drop=True)
    chimerseq_grouped_filtered.to_csv("data/chimerseq_labeled.csv", index=False)

    #print the number of drivers and non-drivers and some analysis of the data
    num_drivers = (chimerseq_grouped_filtered['driver'] == 'driver').sum()
    num_non_drivers = (chimerseq_grouped_filtered['driver'] == 'non-driver').sum()
    print(f"  Classified {num_drivers} drivers and {num_non_drivers} non-drivers out of {len(chimerseq_grouped_filtered)} total fusion pairs")

    #plot the density of Junction_reads_num for drivers and non-drivers, and save the plot as a png file. Limit the x-axis to 2500 for better visualization. Use different colors for drivers and non-drivers, and add a legend and title to the plot.
    import matplotlib.pyplot as plt
    data_drivers = chimerseq_grouped_filtered[chimerseq_grouped_filtered['driver'] == 'driver']
    data_non_drivers = chimerseq_grouped_filtered[chimerseq_grouped_filtered['driver'] == 'non-driver']
    plt.figure(figsize=(10, 6))
    plt.hist(data_drivers['Junction_reads_num'], bins=50, alpha=0.5, label='Drivers', color='blue', density=True)
    plt.hist(data_non_drivers['Junction_reads_num'], bins=50, alpha=0.5, label='Non-Drivers', color='orange', density=True)
    plt.xlim(0, 2500)
    plt.xlabel('Junction Reads Number')
    plt.ylabel('Density')
    plt.title('Density of Junction Reads for Drivers vs Non-Drivers')
    plt.legend()
    plt.savefig("junction_reads_density.png")
    plt.close()

    print(f"Junction reads for drivers: mean={data_drivers['Junction_reads_num'].mean():.2f}, median={data_drivers['Junction_reads_num'].median():.2f}")
    print(f"Junction reads for non-drivers: mean={data_non_drivers['Junction_reads_num'].mean():.2f}, median={data_non_drivers['Junction_reads_num'].median():.2f}")

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
    plt.savefig("tumor_freq_drivers.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    tumor_freq_non_drivers.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Tumor Frequency for Non-Drivers')
    plt.xlabel('Cancer Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("tumor_freq_non_drivers.png")
    plt.close()

if __name__ == '__main__':
    main()
