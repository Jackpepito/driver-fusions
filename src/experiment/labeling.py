"""Label construction for A/B/C/D policies."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd


POLICY_DRIVER_MIN_RECURRENCE = {
    "A": 2,
    "B": 3,
    "C": 2,
    "D": 3,
}


def parse_first_breakpoint(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if "," in s:
        s = s.split(",", 1)[0].strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return ""


def nonempty_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    return (~values.isna()) & (values.astype(str).str.strip() != "")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def is_in_frame(value: Any) -> bool:
    if pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    return normalized in {"in-frame", "in frame", "inframe"}


def load_census_genes(census_path: str) -> set[str]:
    census_df = pd.read_csv(census_path, sep="\t", low_memory=False)
    if "Gene Symbol" in census_df.columns:
        source_col = "Gene Symbol"
    else:
        source_col = str(census_df.columns[0])
    genes = set(census_df[source_col].dropna().astype(str).str.upper().str.strip())
    return {g for g in genes if g}


def prepare_chimerseq_base(chimerseq_path: str) -> pd.DataFrame:
    raw = pd.read_csv(chimerseq_path, low_memory=False)

    if "H_gene" not in raw.columns or "T_gene" not in raw.columns:
        raise ValueError("Input ChimerSeq file must include columns H_gene and T_gene")

    raw["H_gene_up"] = raw["H_gene"].astype(str).str.upper().str.strip()
    raw["T_gene_up"] = raw["T_gene"].astype(str).str.upper().str.strip()
    raw["H_position_norm"] = raw.get("H_position", pd.Series(index=raw.index)).apply(parse_first_breakpoint)
    raw["T_position_norm"] = raw.get("T_position", pd.Series(index=raw.index)).apply(parse_first_breakpoint)

    if "Fusion_pair" not in raw.columns:
        raw["Fusion_pair"] = raw["H_gene_up"] + "-" + raw["T_gene_up"]

    raw["has_pub"] = nonempty_series(raw, "ChimerPub")
    # Always derive seq-plus evidence from ChimerSeq's Highly_Reliable_Seq.
    raw["has_seq_plus"] = nonempty_series(raw, "Highly_Reliable_Seq")

    recurrence_source = "BarcodeID" 
    recurrence = raw.groupby("Fusion_pair")[recurrence_source].nunique(dropna=True)
    raw["recurrent_patients"] = raw["Fusion_pair"].map(recurrence).fillna(1).astype(int)

    group_keys = ["H_position_norm", "T_position_norm", "H_gene_up", "T_gene_up"]
    agg_map = {col: "first" for col in raw.columns if col not in group_keys}
    agg_map["has_pub"] = "max"
    agg_map["has_seq_plus"] = "max"
    agg_map["recurrent_patients"] = "max"

    grouped = raw.groupby(group_keys, as_index=False).agg(agg_map)

    grouped["H_gene"] = grouped.get("H_gene", grouped["H_gene_up"]).fillna(grouped["H_gene_up"])
    grouped["T_gene"] = grouped.get("T_gene", grouped["T_gene_up"]).fillna(grouped["T_gene_up"])
    grouped["H_position"] = grouped["H_position_norm"]
    grouped["T_position"] = grouped["T_position_norm"]
    grouped["Fusion_pair"] = grouped.get("Fusion_pair", grouped["H_gene_up"] + "-" + grouped["T_gene_up"])

    return grouped


def classify_row_by_policy(row: pd.Series, policy: str, census_genes: set[str]) -> str | None:
    policy = policy.upper()
    if policy not in POLICY_DRIVER_MIN_RECURRENCE:
        raise ValueError(f"Unknown policy: {policy}")

    has_seq_plus = as_bool(row.get("has_seq_plus"))
    recurrent_raw = pd.to_numeric(row.get("recurrent_patients"), errors="coerce")
    recurrent_patients = int(recurrent_raw) if not pd.isna(recurrent_raw) else 0
    recurrent_min = POLICY_DRIVER_MIN_RECURRENCE[policy]
    one_patient_only = recurrent_patients == 1

    head_gene = str(row.get("H_gene_up", "")).upper().strip()
    tail_gene = str(row.get("T_gene_up", "")).upper().strip()
    has_cancer_gene = (head_gene in census_genes) or (tail_gene in census_genes)

    in_frame = is_in_frame(row.get("Frame"))

    if in_frame and has_seq_plus and recurrent_patients >= recurrent_min and has_cancer_gene:
        return "driver"

    common_non_driver = in_frame and (not has_seq_plus) and one_patient_only
    if policy in {"A", "B"}:
        if common_non_driver and has_cancer_gene:
            return "non-driver"
    else:  # C, D
        if common_non_driver:
            return "non-driver"

    return None


def label_dataset_for_policy(base_df: pd.DataFrame, policy: str, census_genes: set[str]) -> pd.DataFrame:
    policy = policy.upper()
    recurrent_threshold = POLICY_DRIVER_MIN_RECURRENCE[policy]

    labeled = base_df.copy()
    labeled["driver"] = labeled.apply(classify_row_by_policy, axis=1, policy=policy, census_genes=census_genes)
    labeled = labeled[labeled["driver"].notna()].copy().reset_index(drop=True)

    labeled["is_recurrent"] = pd.to_numeric(labeled["recurrent_patients"], errors="coerce").fillna(0).astype(int) >= recurrent_threshold
    labeled["driver_policy"] = policy

    return labeled


def create_policy_summary(policy: str, labeled_df: pd.DataFrame) -> str:
    n_total = len(labeled_df)
    n_driver = int((labeled_df["driver"] == "driver").sum())
    n_non_driver = int((labeled_df["driver"] == "non-driver").sum())

    lines = [
        f"Policy {policy} summary",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total labeled rows: {n_total}",
        f"Drivers: {n_driver}",
        f"Non-drivers: {n_non_driver}",
    ]

    if n_total > 0:
        lines.extend(
            [
                f"Driver rate: {100.0 * n_driver / n_total:.2f}%",
                f"Non-driver rate: {100.0 * n_non_driver / n_total:.2f}%",
            ]
        )

    if "recurrent_patients" in labeled_df.columns:
        stats = pd.to_numeric(labeled_df["recurrent_patients"], errors="coerce")
        lines.extend(
            [
                "",
                "Recurrence stats:",
                f"  mean: {stats.mean():.3f}",
                f"  median: {stats.median():.3f}",
                f"  min: {stats.min():.3f}",
                f"  max: {stats.max():.3f}",
            ]
        )

    lines.extend(_build_cancer_distribution_section(labeled_df))
    lines.extend(_build_junction_reads_section(labeled_df))
    lines.extend(_build_gene_frequency_section(labeled_df))

    return "\n".join(lines) + "\n"


def _find_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        col = normalized.get(name.strip().lower())
        if col is not None:
            return col
    return None


def _format_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * count / total):.2f}%"


def _build_cancer_distribution_section(labeled_df: pd.DataFrame, top_n: int = 15) -> list[str]:
    lines = ["", "Cancer type distribution:"]
    cancer_col = _find_existing_column(labeled_df, ["Cancertype", "CancerType", "cancer_type"])
    if cancer_col is None:
        lines.append("  Column not available (expected: Cancertype).")
        return lines

    for class_name in ["driver", "non-driver"]:
        subset = labeled_df[labeled_df["driver"] == class_name]
        n_class = len(subset)
        lines.append(f"  {class_name} (n={n_class}):")
        if n_class == 0:
            lines.append("    No rows.")
            continue

        counts = (
            subset[cancer_col]
            .fillna("NA")
            .astype(str)
            .str.strip()
            .replace("", "NA")
            .value_counts()
            .head(top_n)
        )
        for cancer, count in counts.items():
            lines.append(f"    {cancer}: {int(count)} ({_format_pct(int(count), n_class)})")
    return lines


def _build_junction_reads_section(labeled_df: pd.DataFrame) -> list[str]:
    lines = ["", "Junction reads by class:"]
    jr_col = _find_existing_column(
        labeled_df,
        ["Junction_reads_num", "JunctionReadsNum", "junction_reads_num", "junction_reads"],
    )
    if jr_col is None:
        lines.append("  Column not available (expected: Junction_reads_num).")
        return lines

    for class_name in ["driver", "non-driver"]:
        subset = labeled_df[labeled_df["driver"] == class_name]
        values = pd.to_numeric(subset[jr_col], errors="coerce").dropna()
        lines.append(f"  {class_name}:")
        if values.empty:
            lines.append("    No valid numeric values.")
            continue
        lines.extend(
            [
                f"    rows_with_value: {len(values)}",
                f"    total_reads: {float(values.sum()):.2f}",
                f"    mean_reads: {float(values.mean()):.2f}",
                f"    median_reads: {float(values.median()):.2f}",
                f"    std_reads: {float(values.std(ddof=1)):.2f}",
                f"    min_reads: {float(values.min()):.2f}",
                f"    max_reads: {float(values.max()):.2f}",
            ]
        )
    return lines


def _build_gene_frequency_section(labeled_df: pd.DataFrame, top_n: int = 20) -> list[str]:
    lines = ["", "Most frequent genes by class (share of class fusions):"]
    head_col = _find_existing_column(labeled_df, ["H_gene_up", "H_gene", "head_gene"])
    tail_col = _find_existing_column(labeled_df, ["T_gene_up", "T_gene", "tail_gene"])
    if head_col is None or tail_col is None:
        lines.append("  Gene columns not available (expected H_gene_up/H_gene and T_gene_up/T_gene).")
        return lines

    for class_name in ["driver", "non-driver"]:
        subset = labeled_df[labeled_df["driver"] == class_name]
        n_class = len(subset)
        lines.append(f"  {class_name} (n={n_class}):")
        if n_class == 0:
            lines.append("    No rows.")
            continue

        head = subset[head_col].fillna("").astype(str).str.upper().str.strip()
        tail = subset[tail_col].fillna("").astype(str).str.upper().str.strip()
        per_row_genes = pd.DataFrame({"head": head, "tail": tail}).apply(
            lambda row: sorted({g for g in row.tolist() if g}), axis=1
        )

        counter: Counter[str] = Counter()
        for genes in per_row_genes.tolist():
            counter.update(genes)

        if not counter:
            lines.append("    No valid gene symbols.")
            continue

        for gene, count in counter.most_common(top_n):
            lines.append(f"    {gene}: {count} ({_format_pct(count, n_class)})")

    return lines
