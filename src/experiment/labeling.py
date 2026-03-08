"""Label construction for A/B/C/D policies."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
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


def save_labeling_distribution_plots(
    labeled_df: pd.DataFrame,
    output_root: Path,
    policy: str,
    top_n_genes: int = 20,
    bins: int = 40,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    generated.extend(
        _save_gene_distribution_artifacts(
            labeled_df=labeled_df,
            output_dir=output_root / "plot_gene_distribution_by_class",
            policy=policy,
            top_n=top_n_genes,
            plt=plt,
        )
    )

    generated.extend(
        _save_numeric_distribution_artifacts(
            labeled_df=labeled_df,
            output_dir=output_root / "plot_junction_reads_distribution",
            value_col_candidates=[
                "Junction_reads_num",
                "JunctionReadsNum",
                "junction_reads_num",
                "junction_reads",
            ],
            value_label="Junction reads",
            value_slug="junction_reads",
            policy=policy,
            bins=bins,
            plt=plt,
            np=np,
        )
    )

    generated.extend(
        _save_numeric_distribution_artifacts(
            labeled_df=labeled_df,
            output_dir=output_root / "plot_seed_reads_distribution",
            value_col_candidates=["Seed_reads_num", "SeedReadsNum", "seed_reads_num", "seed_reads"],
            value_label="Seed reads",
            value_slug="seed_reads",
            policy=policy,
            bins=bins,
            plt=plt,
            np=np,
        )
    )

    generated.extend(
        _save_cancer_distribution_artifacts(
            labeled_df=labeled_df,
            output_dir=output_root / "plot_cancer_distribution_by_class",
            policy=policy,
            plt=plt,
        )
    )

    return generated


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


def _save_gene_distribution_artifacts(
    labeled_df: pd.DataFrame,
    output_dir: Path,
    policy: str,
    top_n: int,
    plt: Any,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    head_col = _find_existing_column(labeled_df, ["H_gene_up", "H_gene", "head_gene"])
    tail_col = _find_existing_column(labeled_df, ["T_gene_up", "T_gene", "tail_gene"])
    if head_col is None or tail_col is None:
        note_path = output_dir / "gene_distribution_by_class_stats.txt"
        note_path.write_text(
            "Gene columns not available (expected H_gene_up/H_gene and T_gene_up/T_gene).\n",
            encoding="utf-8",
        )
        return [note_path]

    records: list[dict[str, Any]] = []
    class_counters: dict[str, Counter[str]] = {}
    class_sizes: dict[str, int] = {}
    for class_name in ["driver", "non-driver"]:
        subset = labeled_df[labeled_df["driver"] == class_name]
        n_class = len(subset)
        class_sizes[class_name] = n_class
        head = subset[head_col].fillna("").astype(str).str.upper().str.strip()
        tail = subset[tail_col].fillna("").astype(str).str.upper().str.strip()
        per_row_genes = pd.DataFrame({"head": head, "tail": tail}).apply(
            lambda row: sorted({g for g in row.tolist() if g}),
            axis=1,
        )
        counter: Counter[str] = Counter()
        for genes in per_row_genes.tolist():
            counter.update(genes)
        class_counters[class_name] = counter
        for gene, count in counter.items():
            records.append(
                {
                    "policy": policy,
                    "class": class_name,
                    "gene": gene,
                    "count": int(count),
                    "class_rows": n_class,
                    "class_share_pct": (100.0 * float(count) / float(n_class)) if n_class > 0 else 0.0,
                }
            )

    full_df = pd.DataFrame(records).sort_values(["class", "count", "gene"], ascending=[True, False, True])
    csv_path = output_dir / "gene_distribution_by_class.csv"
    full_df.to_csv(csv_path, index=False)

    stats_lines = [f"Policy: {policy}", "Gene distribution stats by class:"]
    for class_name in ["driver", "non-driver"]:
        class_df = full_df[full_df["class"] == class_name]
        values = class_df["count"].astype(float) if not class_df.empty else pd.Series(dtype=float)
        stats_lines.extend(
            [
                "",
                f"{class_name}:",
                f"  class_rows={class_sizes[class_name]}",
                f"  unique_genes={len(class_counters[class_name])}",
                f"  mean_count_per_gene={float(values.mean()):.6f}" if not values.empty else "  mean_count_per_gene=nan",
                f"  variance_count_per_gene={float(values.var(ddof=0)):.6f}" if not values.empty else "  variance_count_per_gene=nan",
            ]
        )
    stats_path = output_dir / "gene_distribution_by_class_stats.txt"
    stats_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    plotted_df = (
        full_df.sort_values("count", ascending=False)
        .groupby("class", as_index=False)
        .head(top_n)
        .copy()
    )
    if plotted_df.empty:
        return [csv_path, stats_path]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False)
    for idx, class_name in enumerate(["driver", "non-driver"]):
        ax = axes[idx]
        class_plot = plotted_df[plotted_df["class"] == class_name].copy()
        class_plot = class_plot.sort_values("count", ascending=True)
        if class_plot.empty:
            ax.set_title(f"{class_name} (n=0)")
            ax.set_xlabel("Occurrences")
            ax.set_ylabel("Gene")
            continue
        ax.barh(class_plot["gene"], class_plot["count"], color="#4C72B0" if class_name == "driver" else "#DD8452")
        ax.set_title(f"{class_name} | top {min(top_n, len(class_plot))} genes")
        ax.set_xlabel("Occurrences")
        ax.set_ylabel("Gene")
    fig.suptitle(f"Policy {policy} - Gene distribution by class", fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / "gene_distribution_by_class.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    return [plot_path, csv_path, stats_path]


def _save_numeric_distribution_artifacts(
    labeled_df: pd.DataFrame,
    output_dir: Path,
    value_col_candidates: list[str],
    value_label: str,
    value_slug: str,
    policy: str,
    bins: int,
    plt: Any,
    np: Any,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    value_col = _find_existing_column(labeled_df, value_col_candidates)
    if value_col is None:
        note_path = output_dir / f"{value_slug}_distribution_stats.txt"
        note_path.write_text(
            f"Column not available for {value_label} (candidates: {', '.join(value_col_candidates)}).\n",
            encoding="utf-8",
        )
        return [note_path]

    rows: list[dict[str, Any]] = []
    for class_name in ["driver", "non-driver"]:
        subset = labeled_df[labeled_df["driver"] == class_name]
        values = pd.to_numeric(subset[value_col], errors="coerce")
        for raw_value in values.tolist():
            rows.append({"policy": policy, "class": class_name, "value": raw_value})

    raw_df = pd.DataFrame(rows)
    csv_path = output_dir / f"{value_slug}_distribution.csv"
    raw_df.to_csv(csv_path, index=False)

    valid_df = raw_df.dropna(subset=["value"]).copy()
    valid_df["value"] = pd.to_numeric(valid_df["value"], errors="coerce")
    valid_df = valid_df.dropna(subset=["value"])

    stats_lines = [f"Policy: {policy}", f"{value_label} distribution stats by class:"]
    for class_name in ["driver", "non-driver"]:
        class_values = valid_df.loc[valid_df["class"] == class_name, "value"]
        if class_values.empty:
            stats_lines.extend(["", f"{class_name}:", "  n=0", "  mean=nan", "  variance=nan"])
            continue
        stats_lines.extend(
            [
                "",
                f"{class_name}:",
                f"  n={len(class_values)}",
                f"  mean={float(class_values.mean()):.6f}",
                f"  variance={float(class_values.var(ddof=0)):.6f}",
            ]
        )
    stats_path = output_dir / f"{value_slug}_distribution_stats.txt"
    stats_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    if valid_df.empty:
        return [csv_path, stats_path]

    fig, ax = plt.subplots(figsize=(10, 6))
    classes = ["driver", "non-driver"]
    colors = {"driver": "#4C72B0", "non-driver": "#DD8452"}
    max_val = float(valid_df["value"].max())
    bin_edges = np.linspace(0.0, max_val if max_val > 0 else 1.0, bins + 1)

    for class_name in classes:
        class_values = valid_df.loc[valid_df["class"] == class_name, "value"]
        if class_values.empty:
            continue
        weights = np.ones(len(class_values), dtype=float) * (100.0 / float(len(class_values)))
        ax.hist(
            class_values,
            bins=bin_edges,
            alpha=0.45,
            weights=weights,
            label=f"{class_name} (n={len(class_values)})",
            color=colors[class_name],
        )
    ax.set_title(f"Policy {policy} - {value_label} distribution (% within class)")
    ax.set_xlabel(value_label)
    ax.set_ylabel("Percentage within class (%)")
    ax.legend()
    fig.tight_layout()
    hist_path = output_dir / f"{value_slug}_distribution_hist.png"
    fig.savefig(hist_path, dpi=220)
    plt.close(fig)

    violin_path: Path | None = None
    violin_data = [valid_df.loc[valid_df["class"] == c, "value"].dropna().to_numpy() for c in classes]
    if all(len(arr) > 0 for arr in violin_data):
        fig, ax = plt.subplots(figsize=(8, 6))
        parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[classes[i]])
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(classes)
        ax.set_ylabel(value_label)
        ax.set_title(f"Policy {policy} - {value_label} by class (violin)")
        fig.tight_layout()
        violin_path = output_dir / f"{value_slug}_distribution_violin.png"
        fig.savefig(violin_path, dpi=220)
        plt.close(fig)

    generated = [hist_path]
    if violin_path is not None:
        generated.append(violin_path)
    generated.extend([csv_path, stats_path])
    return generated


def _save_cancer_distribution_artifacts(
    labeled_df: pd.DataFrame,
    output_dir: Path,
    policy: str,
    plt: Any,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cancer_col = _find_existing_column(labeled_df, ["Cancertype", "CancerType", "cancer_type"])
    if cancer_col is None:
        note_path = output_dir / "cancer_distribution_by_class_stats.txt"
        note_path.write_text(
            "Cancer type column not available (expected one of: Cancertype, CancerType, cancer_type).\n",
            encoding="utf-8",
        )
        return [note_path]

    rows: list[dict[str, Any]] = []
    class_sizes: dict[str, int] = {}
    for class_name in ["driver", "non-driver"]:
        subset = labeled_df[labeled_df["driver"] == class_name]
        n_class = len(subset)
        class_sizes[class_name] = n_class
        counts = (
            subset[cancer_col]
            .fillna("NA")
            .astype(str)
            .str.strip()
            .replace("", "NA")
            .value_counts()
        )
        for cancer, count in counts.items():
            rows.append(
                {
                    "policy": policy,
                    "class": class_name,
                    "cancer_type": cancer,
                    "count": int(count),
                    "class_rows": n_class,
                    "class_share_pct": (100.0 * float(count) / float(n_class)) if n_class > 0 else 0.0,
                }
            )

    counts_df = pd.DataFrame(rows).sort_values(["class", "count", "cancer_type"], ascending=[True, False, True])
    csv_path = output_dir / "cancer_distribution_by_class.csv"
    counts_df.to_csv(csv_path, index=False)

    stats_lines = [f"Policy: {policy}", "Cancer distribution stats by class:"]
    for class_name in ["driver", "non-driver"]:
        class_df = counts_df[counts_df["class"] == class_name]
        values = class_df["count"].astype(float) if not class_df.empty else pd.Series(dtype=float)
        stats_lines.extend(
            [
                "",
                f"{class_name}:",
                f"  class_rows={class_sizes[class_name]}",
                f"  unique_cancer_types={len(class_df)}",
                f"  mean_count_per_cancer={float(values.mean()):.6f}" if not values.empty else "  mean_count_per_cancer=nan",
                f"  variance_count_per_cancer={float(values.var(ddof=0)):.6f}" if not values.empty else "  variance_count_per_cancer=nan",
            ]
        )
    stats_path = output_dir / "cancer_distribution_by_class_stats.txt"
    stats_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    if counts_df.empty:
        return [csv_path, stats_path]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = {"driver": "#4C72B0", "non-driver": "#DD8452"}
    for idx, class_name in enumerate(["driver", "non-driver"]):
        ax = axes[idx]
        class_df = counts_df[counts_df["class"] == class_name].copy()
        class_df = class_df.sort_values("count", ascending=True)
        if class_df.empty:
            ax.set_title(f"{class_name} (n=0)")
            ax.set_xlabel("Count")
            ax.set_ylabel("Cancer type")
            continue
        ax.barh(class_df["cancer_type"], class_df["count"], color=colors[class_name], alpha=0.85)
        ax.set_title(f"{class_name} cancer distribution")
        ax.set_xlabel("Count")
        ax.set_ylabel("Cancer type")
    fig.suptitle(f"Policy {policy} - Cancer type distribution by class", fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / "cancer_distribution_by_class.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    return [plot_path, csv_path, stats_path]
