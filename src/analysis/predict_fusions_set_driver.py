#!/usr/bin/env python3
"""Predict driver/non-driver on reconstructed fusions and run statistical analysis."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))

from nets import ProbeConv1D, ProbeMLP

MAX_SEQ_LEN = 4000


def load_esmc(device: str):
    from esm.models.esmc import ESMC
    model = ESMC.from_pretrained("esmc_600m", device=torch.device(device))
    model.eval()
    return model


def load_fuson(device: str):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/FusOn-pLM")
    model = AutoModel.from_pretrained("ChatterjeeLab/FusOn-pLM").to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def embed_esmc(model, seq: str):
    from esm.sdk.api import ESMProtein, LogitsConfig
    protein = ESMProtein(sequence=seq[:MAX_SEQ_LEN])
    pt = model.encode(protein)
    out = model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
    return out.embeddings.squeeze().detach().cpu()


@torch.no_grad()
def embed_fuson(model_tok, seq: str, device: str):
    model, tokenizer = model_tok
    inputs = tokenizer(
        seq[:MAX_SEQ_LEN],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    return out.last_hidden_state.squeeze(0)[1:-1, :].detach().cpu()


def infer_hidden_dims_from_state(state: dict) -> tuple[int, list[int], int]:
    linear_keys = []
    for k, v in state.items():
        if k.endswith(".weight") and getattr(v, "ndim", None) == 2:
            m = re.search(r"net\.(\d+)\.weight$", k)
            if m:
                linear_keys.append((int(m.group(1)), k, v))
    linear_keys.sort(key=lambda x: x[0])
    if not linear_keys:
        raise ValueError("Cannot infer MLP architecture from checkpoint state_dict")

    dims = [int(linear_keys[0][2].shape[1])]
    for _, _, w in linear_keys:
        dims.append(int(w.shape[0]))
    in_dim = dims[0]
    out_dim = dims[-1]
    hidden_dims = dims[1:-1]
    return in_dim, hidden_dims, out_dim


def _is_conv1d_checkpoint(state: dict) -> bool:
    return any(str(k).startswith("conv_net.") for k in state.keys())


def _infer_norm_type_from_state(state: dict, prefix: str) -> str:
    for k in state.keys():
        key = str(k)
        if not key.startswith(prefix):
            continue
        if key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked"):
            return "batchnorm"
    return "layernorm"


def build_model_from_checkpoint(ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if _is_conv1d_checkpoint(state):
        out_dim = int(state["classifier.weight"].shape[0])
        norm_type = _infer_norm_type_from_state(state, "conv_net.")
        model = ProbeConv1D(n_classes=out_dim, dropout=0.0, norm_type=norm_type).to(device)
        in_dim = None
        hidden_dims = [f"conv1d({norm_type})"]
    else:
        in_dim, hidden_dims, out_dim = infer_hidden_dims_from_state(state)
        norm_type = "none" if len(hidden_dims) == 0 else _infer_norm_type_from_state(state, "net.")
        model = ProbeMLP(
            in_dim=in_dim,
            n_classes=out_dim,
            hidden_dims=hidden_dims,
            dropout=0.0,
            norm_type=norm_type,
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, in_dim, hidden_dims, out_dim


def infer_embedding_model(ckpt_path: Path, in_dim: int | None, requested: str) -> str:
    if requested in {"esmc", "fuson"}:
        return requested

    ckpt_lower = str(ckpt_path).lower()
    if "fuson" in ckpt_lower:
        return "fuson"
    if "esmc" in ckpt_lower:
        return "esmc"

    if in_dim == 1280:
        return "fuson"
    if in_dim == 1152:
        return "esmc"

    raise ValueError(
        "Unable to infer embedding model automatically. "
        "Use --embedding-model {esmc,fuson}."
    )


def sequence_signature(sequences: list[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(sequences)).encode("utf-8"))
    for s in sequences:
        h.update(b"\n")
        h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.array(pvals, dtype=float)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        q[i] = prev
    q_final = np.empty(m, dtype=float)
    q_final[order] = np.clip(q, 0, 1)
    return q_final.tolist()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x = np.asarray(x)
    y = np.asarray(y)
    gt = 0
    lt = 0
    for xv in x:
        gt += np.sum(xv > y)
        lt += np.sum(xv < y)
    return (gt - lt) / (len(x) * len(y))


def cramers_v_from_chi2(chi2: float, n: int, r: int, c: int) -> float:
    if n <= 0:
        return np.nan
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return np.nan
    return math.sqrt((chi2 / n) / denom)


def try_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, ".": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def fusion_label_series(df: pd.DataFrame) -> pd.Series:
    for col in ["Fusion", "Fusion_pair", "fusion_pair", "id"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            if (s != "").any() and (~s.str.lower().isin({"nan", "none", "."})).any():
                return s
    if "Gene1" in df.columns and "Gene2" in df.columns:
        g1 = df["Gene1"].astype(str).str.strip()
        g2 = df["Gene2"].astype(str).str.strip()
        return g1 + "-" + g2
    if "gene5" in df.columns and "gene3" in df.columns:
        g5 = df["gene5"].astype(str).str.strip()
        g3 = df["gene3"].astype(str).str.strip()
        return g5 + "-" + g3
    if "H_gene" in df.columns and "T_gene" in df.columns:
        return df["H_gene"].astype(str).str.strip() + "-" + df["T_gene"].astype(str).str.strip()
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index, dtype="object")


def print_top_probability_fusions(df: pd.DataFrame, prob_col: str = "predicted_driver_prob", n: int = 10):
    if prob_col not in df.columns:
        print("\n[INFO] Cannot print top probability fusions: missing probability column.")
        return
    d = df.copy()
    d["fusion_name"] = fusion_label_series(d)
    d[prob_col] = pd.to_numeric(d[prob_col], errors="coerce")
    d = d.dropna(subset=[prob_col])
    if d.empty:
        print("\n[INFO] Cannot print top probability fusions: no valid probabilities.")
        return
    top_hi = d.sort_values(prob_col, ascending=False).head(n)
    top_lo = d.sort_values(prob_col, ascending=True).head(n)
    print(f"\n{'='*70}")
    print("FUSION-SET: TOP FUSIONS BY P(driver)")
    print(f"{'='*70}")
    print(f"Top {len(top_hi)} highest P(driver):")
    for i, (_, r) in enumerate(top_hi.iterrows(), start=1):
        pred = f", pred={r['predicted_label']}" if "predicted_label" in d.columns else ""
        print(f"  {i:2d}. {r['fusion_name']} | p={float(r[prob_col]):.4f}{pred}")
    print(f"Top {len(top_lo)} lowest P(driver):")
    for i, (_, r) in enumerate(top_lo.iterrows(), start=1):
        pred = f", pred={r['predicted_label']}" if "predicted_label" in d.columns else ""
        print(f"  {i:2d}. {r['fusion_name']} | p={float(r[prob_col]):.4f}{pred}")


def save_top_probability_fusions_txt(
    df: pd.DataFrame,
    out_path: Path,
    prob_col: str = "predicted_driver_prob",
    n: int = 10,
) -> Path | None:
    if prob_col not in df.columns:
        return None
    d = df.copy()
    d["fusion_name"] = fusion_label_series(d)
    d[prob_col] = pd.to_numeric(d[prob_col], errors="coerce")
    d = d.dropna(subset=[prob_col])
    if d.empty:
        return None
    top_hi = d.sort_values(prob_col, ascending=False).head(n)
    top_lo = d.sort_values(prob_col, ascending=True).head(n)

    lines = [
        "Top 10 fusioni con P(driver) più alta",
    ]
    for i, (_, r) in enumerate(top_hi.iterrows(), start=1):
        pred = f", pred={r['predicted_label']}" if "predicted_label" in d.columns else ""
        lines.append(f"{i:2d}. {r['fusion_name']} | p={float(r[prob_col]):.6f}{pred}")

    lines.extend([
        "",
        "Top 10 fusioni con P(driver) più bassa",
    ])
    for i, (_, r) in enumerate(top_lo.iterrows(), start=1):
        pred = f", pred={r['predicted_label']}" if "predicted_label" in d.columns else ""
        lines.append(f"{i:2d}. {r['fusion_name']} | p={float(r[prob_col]):.6f}{pred}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def analyze_numeric(df: pd.DataFrame, pred_col: str, min_valid: int = 25):
    rows = []
    excluded = {
        pred_col,
        "predicted_driver_prob",
        "reconstructed_seq",
        "original_seq",
        "peptide_sequence",
        "FUSION_TRANSL",
        "recon_error",
    }
    for col in df.columns:
        if col in excluded:
            continue
        num = try_numeric(df[col])
        valid = num.notna().sum()
        if valid < min_valid:
            continue
        sub = df.loc[num.notna(), [pred_col]].copy()
        sub["x"] = num[num.notna()].values
        g0 = sub.loc[sub[pred_col] == 0, "x"].values
        g1 = sub.loc[sub[pred_col] == 1, "x"].values
        if len(g0) < 8 or len(g1) < 8:
            continue
        stat = mannwhitneyu(g1, g0, alternative="two-sided")
        delta = cliffs_delta(g1, g0)
        rows.append(
            {
                "feature": col,
                "n_total": len(sub),
                "n_non_driver": len(g0),
                "n_driver": len(g1),
                "median_non_driver": float(np.median(g0)),
                "median_driver": float(np.median(g1)),
                "delta_median": float(np.median(g1) - np.median(g0)),
                "cliffs_delta": float(delta),
                "p_value": float(stat.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["q_value"] = benjamini_hochberg(out["p_value"].tolist())
        out = out.sort_values(["q_value", "p_value", "feature"]).reset_index(drop=True)
    return out


def analyze_categorical(df: pd.DataFrame, pred_col: str, max_levels: int = 30):
    rows = []
    excluded = {
        pred_col,
        "predicted_driver_prob",
        "reconstructed_seq",
        "original_seq",
        "peptide_sequence",
        "FUSION_TRANSL",
        "recon_error",
    }
    for col in df.columns:
        if col in excluded:
            continue
        s = df[col].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, ".": np.nan}).dropna()
        if len(s) < 30:
            continue
        nunique = s.nunique()
        if nunique < 2 or nunique > max_levels:
            continue
        temp = df.loc[s.index, [pred_col]].copy()
        temp[col] = s.values
        tab = pd.crosstab(temp[pred_col], temp[col])
        if tab.shape[0] != 2 or tab.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(tab.values)
        cv = cramers_v_from_chi2(chi2, int(tab.values.sum()), *tab.shape)
        rows.append(
            {
                "feature": col,
                "n_total": int(tab.values.sum()),
                "n_levels": int(tab.shape[1]),
                "chi2": float(chi2),
                "cramers_v": float(cv),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["q_value"] = benjamini_hochberg(out["p_value"].tolist())
        out = out.sort_values(["q_value", "p_value", "feature"]).reset_index(drop=True)
    return out


def chromosome_enrichment(df: pd.DataFrame, pred_col: str):
    if "Chromosome1" not in df.columns or "Chromosome2" not in df.columns:
        return pd.DataFrame()
    sub = df[["Chromosome1", "Chromosome2", pred_col]].copy()
    sub["chrom_pair"] = sub["Chromosome1"].astype(str) + "-" + sub["Chromosome2"].astype(str)
    pair_counts = sub["chrom_pair"].value_counts()
    tested_pairs = pair_counts[pair_counts >= 10].index.tolist()
    rows = []
    for pair in tested_pairs:
        is_pair = sub["chrom_pair"] == pair
        a = int(((sub[pred_col] == 1) & is_pair).sum())
        b = int(((sub[pred_col] == 0) & is_pair).sum())
        c = int(((sub[pred_col] == 1) & (~is_pair)).sum())
        d = int(((sub[pred_col] == 0) & (~is_pair)).sum())
        if min(a + b, c + d) == 0:
            continue
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        rows.append(
            {
                "chrom_pair": pair,
                "n_pair": a + b,
                "driver_in_pair": a,
                "non_driver_in_pair": b,
                "odds_ratio": float(odds),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["q_value"] = benjamini_hochberg(out["p_value"].tolist())
        out = out.sort_values(["q_value", "p_value", "n_pair"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def to_binary_flag(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "yes", "y", "driver", "present"}
    false_vals = {"0", "false", "no", "n", "non-driver", "absent"}
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(true_vals)] = 1.0
    out[s.isin(false_vals)] = 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    out[numeric.notna()] = (numeric[numeric.notna()] > 0).astype(float)
    return out


def targeted_feature_analysis(df: pd.DataFrame, pred_col: str, out_dir: Path):
    numeric_features = [
        "SupportingReads", "JunctionReads", "EncompassingReads", "reconstructed_length",
        "num_variants", "original_length", "length_diff", "#Patients", "#Samples",
        "Age at Diagnosis", "BMI at Dg", "Time to progression_Days from Dg",
        "PFI after Primary therapy if Prog_Days", "PFI at outcome update when no prog_Days",
    ]
    categorical_features = [
        "type", "Chromosome1", "Chromosome2", "reading_frame", "PROT_FUSION_TYPE",
        "recon_quality", "Histology", "Stage_FIGO2014", "Primary therapy outcome",
        "Progression Yes_No", "Survival", "BRCA mutation status", "HRD Myriad status",
        "5p_role", "3p_role", "TSG1", "TSG2", "bp5_approx", "bp3_approx", "orf_used",
    ]
    binary_flag_features = ["ProteinCoding", "Kinases", "Druggable", "Cancer", "Actionable"]

    nrows = []
    crows = []

    for col in numeric_features:
        if col not in df.columns:
            continue
        x = try_numeric(df[col])
        tmp = df.loc[x.notna(), [pred_col]].copy()
        tmp["x"] = x[x.notna()].values
        g1 = tmp.loc[tmp[pred_col] == 1, "x"].values
        g0 = tmp.loc[tmp[pred_col] == 0, "x"].values
        if len(g1) < 8 or len(g0) < 8:
            continue
        st = mannwhitneyu(g1, g0, alternative="two-sided")
        nrows.append({
            "feature": col,
            "n_driver": len(g1),
            "n_non_driver": len(g0),
            "median_driver": float(np.median(g1)),
            "median_non_driver": float(np.median(g0)),
            "cliffs_delta": float(cliffs_delta(g1, g0)),
            "p_value": float(st.pvalue),
        })

    for col in categorical_features:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, ".": np.nan}).dropna()
        if len(s) < 30 or s.nunique() < 2:
            continue
        if s.nunique() > 25:
            keep = s.value_counts().head(25).index
            s = s[s.isin(keep)]
        t = df.loc[s.index, [pred_col]].copy()
        t[col] = s.values
        tab = pd.crosstab(t[pred_col], t[col])
        if tab.shape[0] != 2 or tab.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(tab.values)
        crows.append({
            "feature": col,
            "n_total": int(tab.values.sum()),
            "n_levels": int(tab.shape[1]),
            "cramers_v": float(cramers_v_from_chi2(chi2, int(tab.values.sum()), *tab.shape)),
            "p_value": float(p),
        })

    for col in binary_flag_features:
        if col not in df.columns:
            continue
        b = to_binary_flag(df[col])
        t = df.loc[b.notna(), [pred_col]].copy()
        t[col] = b[b.notna()].astype(int).values
        tab = pd.crosstab(t[pred_col], t[col])
        if tab.shape != (2, 2):
            continue
        odds, p = fisher_exact(tab.values, alternative="two-sided")
        crows.append({
            "feature": col,
            "n_total": int(tab.values.sum()),
            "n_levels": 2,
            "cramers_v": float(cramers_v_from_chi2(chi2_contingency(tab.values)[0], int(tab.values.sum()), *tab.shape)),
            "odds_ratio": float(odds),
            "p_value": float(p),
        })

    ndf = pd.DataFrame(nrows)
    cdf = pd.DataFrame(crows)
    if len(ndf) > 0:
        ndf["q_value"] = benjamini_hochberg(ndf["p_value"].tolist())
        ndf = ndf.sort_values(["q_value", "p_value", "feature"]).reset_index(drop=True)
    if len(cdf) > 0:
        cdf["q_value"] = benjamini_hochberg(cdf["p_value"].tolist())
        cdf = cdf.sort_values(["q_value", "p_value", "feature"]).reset_index(drop=True)

    ndf.to_csv(out_dir / "targeted_numeric_associations.csv", index=False)
    cdf.to_csv(out_dir / "targeted_categorical_associations.csv", index=False)
    return ndf, cdf


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] if len(s) > 120 else s


def set_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
        }
    )


def write_plots_guide(out_path: Path):
    text = [
        "PLOTS GUIDE",
        "=" * 80,
        "",
        "General reading rules:",
        "1) All analyses compare model-predicted classes: driver vs non-driver.",
        "2) For numeric plots, central tendency is best read from median/box center.",
        "3) For categorical plots, bar height indicates prevalence or count by class.",
        "4) Statistical significance is reported in tables (p-value and FDR q-value).",
        "5) For associations, prioritize features with low q-value and meaningful effect size.",
        "",
        "Directories:",
        "- figures/all_columns/numeric/: one plot per numeric column.",
        "- figures/all_columns/categorical/: one plot per categorical column.",
        "- figures/: summary panels and global overview figures.",
        "",
        "Caveat:",
        "- These are associations with model predictions, not causal evidence.",
    ]
    out_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def plot_targeted_panels(df: pd.DataFrame, out_dir: Path):
    set_plot_style()

    # Panel A: sequencing evidence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flatten(), ["SupportingReads", "JunctionReads", "EncompassingReads", "num_variants"]):
        if col not in df.columns:
            ax.axis("off")
            continue
        x = try_numeric(df[col])
        p = df.loc[x.notna(), ["predicted_label"]].copy()
        p[col] = x[x.notna()].values
        if len(p) < 20:
            ax.axis("off")
            continue
        sns.boxplot(data=p, x="predicted_label", y=col, ax=ax, showfliers=False, width=0.5)
        sns.stripplot(data=p.sample(min(len(p), 600), random_state=42), x="predicted_label", y=col, ax=ax, size=2, alpha=0.25, color="black")
        ax.set_title(f"{col} by predicted class")
    plt.tight_layout()
    plt.savefig(out_dir / "8_sequencing_evidence_panel.png", dpi=220)
    plt.close()

    # Panel B: molecular/context flags
    mol_cols = [c for c in ["Cancer", "Druggable", "Kinases", "Actionable", "ProteinCoding"] if c in df.columns]
    if mol_cols:
        rows = []
        for c in mol_cols:
            b = to_binary_flag(df[c])
            tmp = df.loc[b.notna(), ["predicted_label"]].copy()
            tmp[c] = b[b.notna()].values
            for lbl in ["driver", "non-driver"]:
                sub = tmp[tmp["predicted_label"] == lbl]
                if len(sub) == 0:
                    continue
                rows.append({"feature": c, "predicted_label": lbl, "pct_positive": float((sub[c] == 1).mean() * 100)})
        mdf = pd.DataFrame(rows)
        if len(mdf) > 0:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=mdf, x="feature", y="pct_positive", hue="predicted_label", palette=["#d62728", "#2ca02c"])
            plt.ylabel("% positive")
            plt.xlabel("")
            plt.title("Molecular flag prevalence by predicted class")
            plt.tight_layout()
            plt.savefig(out_dir / "9_molecular_flags_panel.png", dpi=220)
            plt.close()

    # Panel C: regression-like relationships vs probability
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    reg_cols = ["SupportingReads", "JunctionReads", "#Patients"]
    for ax, col in zip(axes, reg_cols):
        if col not in df.columns:
            ax.axis("off")
            continue
        x = try_numeric(df[col])
        p = df.loc[x.notna(), ["predicted_driver_prob"]].copy()
        p[col] = x[x.notna()].values
        if len(p) < 20:
            ax.axis("off")
            continue
        sns.regplot(data=p, x=col, y="predicted_driver_prob", ax=ax, scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "red"})
        ax.set_title(f"Driver probability vs {col}")
    plt.tight_layout()
    plt.savefig(out_dir / "10_probability_regression_panel.png", dpi=220)
    plt.close()

    # Panel D: domain retention
    if "retained_protein_domains" in df.columns:
        d = df.copy()
        d["has_domains"] = d["retained_protein_domains"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, ".": np.nan}).notna()
        d["n_domains"] = d["retained_protein_domains"].astype(str).apply(
            lambda x: 0 if x.strip() in ("", "nan", ".") else len([t for t in re.split(r"[;|]", x) if t.strip()])
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ct = pd.crosstab(d["has_domains"], d["predicted_label"], normalize="columns") * 100
        ct.index = ["No domains", "Has domains"]
        ct.plot(kind="bar", ax=axes[0], color=["#888", "#4c72b0"])
        axes[0].set_ylabel("%")
        axes[0].set_title("Domain retention prevalence")
        dd = d[["predicted_label", "n_domains"]]
        sns.boxplot(data=dd, x="predicted_label", y="n_domains", ax=axes[1], showfliers=False, width=0.5)
        sns.stripplot(data=dd.sample(min(len(dd), 600), random_state=42), x="predicted_label", y="n_domains", ax=axes[1], size=2, alpha=0.25, color="black")
        axes[1].set_title("Retained domain count")
        plt.tight_layout()
        plt.savefig(out_dir / "11_domain_panel.png", dpi=220)
        plt.close()


def plot_core_figures(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    # 1) Probability distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["predicted_driver_prob"],
        bins=30,
        kde=True,
        kde_kws={"cut": 0, "clip": (0, 1)},
        color="#1f77b4",
    )
    plt.xlabel("Predicted driver probability")
    plt.ylabel("Count")
    plt.title("Distribution of predicted driver probability")
    plt.tight_layout()
    plt.savefig(out_dir / "1_predicted_probability_distribution.png", dpi=220)
    plt.close()

    # 2) Reads vs predicted class
    for idx, col in enumerate(["SupportingReads", "JunctionReads", "EncompassingReads"], start=2):
        if col not in df.columns:
            continue
        x = try_numeric(df[col])
        plot_df = df.copy()
        plot_df[col] = x
        plot_df = plot_df[plot_df[col].notna()].copy()
        if len(plot_df) < 15:
            continue
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=plot_df,
            x="predicted_label",
            y=col,
            showfliers=False,
            palette={"driver": "#d62728", "non-driver": "#2ca02c"},
        )
        sns.stripplot(
            data=plot_df.sample(min(len(plot_df), 600), random_state=42),
            x="predicted_label",
            y=col,
            size=2,
            alpha=0.25,
            color="black",
        )
        plt.xlabel("Predicted class")
        plt.ylabel(col)
        plt.title(f"{col} by predicted class")
        plt.tight_layout()
        plt.savefig(out_dir / f"{idx}_{col}_by_predicted_class.png", dpi=220)
        plt.close()

    # 3) Top chromosome-pair composition
    if "Chromosome1" in df.columns and "Chromosome2" in df.columns:
        tmp = df.copy()
        tmp["chrom_pair"] = tmp["Chromosome1"].astype(str) + "-" + tmp["Chromosome2"].astype(str)
        top_pairs = tmp["chrom_pair"].value_counts().head(15).index
        tmp = tmp[tmp["chrom_pair"].isin(top_pairs)]
        if len(tmp) > 0:
            ct = pd.crosstab(tmp["chrom_pair"], tmp["predicted_label"]).sort_values("driver", ascending=False)
            ct.plot(kind="bar", stacked=True, figsize=(11, 6), color=["#2ca02c", "#d62728"])
            plt.xlabel("Chromosome pair")
            plt.ylabel("Count")
            plt.title("Top chromosome pairs by predicted class")
            plt.tight_layout()
            plt.savefig(out_dir / "5_chromosome_pair_stacked_counts.png", dpi=220)
            plt.close()


def plot_association_summaries(numeric_res: pd.DataFrame, categ_res: pd.DataFrame, out_dir: Path):
    set_plot_style()
    if len(numeric_res) > 0:
        p = numeric_res.copy()
        p["neglog10_q"] = -np.log10(p["q_value"].clip(lower=1e-300))
        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=p, x="cliffs_delta", y="neglog10_q", s=60)
        plt.axhline(-math.log10(0.05), color="red", ls="--", lw=1)
        plt.xlabel("Cliff's delta (driver vs non-driver)")
        plt.ylabel("-log10(FDR q-value)")
        plt.title("Numeric feature associations with predicted class")
        plt.tight_layout()
        plt.savefig(out_dir / "6_numeric_association_volcano.png", dpi=220)
        plt.close()

    if len(categ_res) > 0:
        top = categ_res.head(20).copy()
        top["neglog10_q"] = -np.log10(top["q_value"].clip(lower=1e-300))
        plt.figure(figsize=(10, 7))
        sns.barplot(data=top, y="feature", x="neglog10_q", color="#4c72b0")
        plt.axvline(-math.log10(0.05), color="red", ls="--", lw=1)
        plt.xlabel("-log10(FDR q-value)")
        plt.ylabel("Categorical feature")
        plt.title("Top categorical associations with predicted class")
        plt.tight_layout()
        plt.savefig(out_dir / "7_categorical_association_ranking.png", dpi=220)
        plt.close()


def plot_all_columns(df: pd.DataFrame, out_dir: Path, pred_col: str = "predicted_driver"):
    set_plot_style()
    root = out_dir / "all_columns"
    num_dir = root / "numeric"
    cat_dir = root / "categorical"
    num_dir.mkdir(parents=True, exist_ok=True)
    cat_dir.mkdir(parents=True, exist_ok=True)

    excluded = {
        "reconstructed_seq",
        "original_seq",
        "peptide_sequence",
        "FUSION_TRANSL",
        "recon_error",
    }

    index_rows = []

    for col in df.columns:
        if col in excluded or col == pred_col:
            index_rows.append({"feature": col, "kind": "excluded", "status": "skipped", "reason": "non-plottable"})
            continue

        num = try_numeric(df[col])
        n_num = int(num.notna().sum())
        if n_num >= 20:
            tmp = df.loc[num.notna(), [pred_col, "predicted_label"]].copy()
            tmp[col] = num[num.notna()].values
            g1 = tmp.loc[tmp[pred_col] == 1, col].values
            g0 = tmp.loc[tmp[pred_col] == 0, col].values
            if len(g1) >= 8 and len(g0) >= 8:
                p_val = mannwhitneyu(g1, g0, alternative="two-sided").pvalue
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.boxplot(data=tmp, x="predicted_label", y=col, ax=axes[0], showfliers=False, palette={"driver": "#d62728", "non-driver": "#2ca02c"})
                sns.stripplot(
                    data=tmp.sample(min(len(tmp), 1000), random_state=42),
                    x="predicted_label",
                    y=col,
                    ax=axes[0],
                    size=2,
                    alpha=0.2,
                    color="black",
                )
                axes[0].set_title(f"{col} by class (p={p_val:.2e})")
                kde_kwargs = {"data": tmp, "x": col, "hue": "predicted_label", "ax": axes[1], "fill": True, "common_norm": False, "alpha": 0.3}
                if float(tmp[col].min()) >= 0:
                    kde_kwargs.update({"cut": 0, "clip": (0, None)})
                sns.kdeplot(**kde_kwargs)
                axes[1].set_title(f"{col} distribution")
                plt.tight_layout()
                out = num_dir / f"{sanitize_name(col)}.png"
                plt.savefig(out, dpi=220)
                plt.close()
                index_rows.append({"feature": col, "kind": "numeric", "status": "plotted", "reason": "", "path": str(out)})
                continue

        s = df[col].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, ".": np.nan}).dropna()
        if len(s) < 20:
            index_rows.append({"feature": col, "kind": "categorical", "status": "skipped", "reason": "too_few_values"})
            continue
        tmp = df.loc[s.index, [pred_col, "predicted_label"]].copy()
        if s.nunique() > 25:
            top = s.value_counts().head(24).index
            s = s.where(s.isin(top), other="OTHER")
        tmp[col] = s.values
        tab = pd.crosstab(tmp["predicted_label"], tmp[col])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            index_rows.append({"feature": col, "kind": "categorical", "status": "skipped", "reason": "not_enough_levels"})
            continue
        p_val = chi2_contingency(tab.values)[1]
        pct = pd.crosstab(tmp[col], tmp["predicted_label"], normalize="columns") * 100
        pct = pct.sort_values(by=pct.columns.tolist(), ascending=False).head(20)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        pct.plot(kind="bar", ax=axes[0], color=["#d62728", "#2ca02c"])
        axes[0].set_title(f"{col} prevalence by class (p={p_val:.2e})")
        axes[0].set_ylabel("Percent within class")
        axes[0].tick_params(axis="x", rotation=45)
        tab.T.sort_values(by=tab.index.tolist(), ascending=False).head(20).plot(kind="bar", ax=axes[1], color=["#d62728", "#2ca02c"])
        axes[1].set_title(f"{col} absolute counts")
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        out = cat_dir / f"{sanitize_name(col)}.png"
        plt.savefig(out, dpi=220)
        plt.close()
        index_rows.append({"feature": col, "kind": "categorical", "status": "plotted", "reason": "", "path": str(out)})

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(root / "plot_index.csv", index=False)
    return index_df


def write_report_txt(
    out_path: Path,
    df: pd.DataFrame,
    numeric_res: pd.DataFrame,
    categ_res: pd.DataFrame,
    chrom_res: pd.DataFrame,
    targeted_num: pd.DataFrame,
    targeted_cat: pd.DataFrame,
    plot_index: pd.DataFrame,
    model_path: Path,
    input_csv: Path,
):
    n_total = len(df)
    n_drv = int((df["predicted_driver"] == 1).sum())
    n_non = n_total - n_drv
    lines = [
        "DRIVER/NON-DRIVER PREDICTION ANALYSIS REPORT",
        "=" * 100,
        "",
        f"Input table: {input_csv}",
        f"Model checkpoint: {model_path}",
        f"Analyzed rows: {n_total}",
        f"Predicted driver: {n_drv} ({(n_drv/max(n_total,1))*100:.1f}%)",
        f"Predicted non-driver: {n_non} ({(n_non/max(n_total,1))*100:.1f}%)",
        "",
        "METHODS",
        "-" * 100,
        "1) Numeric features: Mann-Whitney U, effect size Cliff's delta, FDR (Benjamini-Hochberg).",
        "2) Categorical features: chi-square, effect size Cramer's V, FDR.",
        "3) Chromosome pair enrichment: Fisher exact, FDR.",
        "4) Significance criterion: q-value < 0.05.",
        "",
        "OUTPUT QUALITY CONTROL",
        "-" * 100,
        f"Total columns scanned for plotting: {len(plot_index)}",
        f"Columns plotted: {(plot_index['status'] == 'plotted').sum()}",
        f"Columns skipped: {(plot_index['status'] == 'skipped').sum()}",
        "",
    ]

    if len(numeric_res) > 0:
        sig_num = numeric_res[numeric_res["q_value"] < 0.05]
        lines += [
            "GLOBAL NUMERIC ASSOCIATIONS",
            "-" * 100,
            f"Tested numeric features: {len(numeric_res)}",
            f"Significant after FDR (q<0.05): {len(sig_num)}",
            "Top numeric signals:",
        ]
        for _, r in numeric_res.head(10).iterrows():
            lines.append(
                f"  - {r['feature']}: q={r['q_value']:.2e}, p={r['p_value']:.2e}, "
                f"delta={r['cliffs_delta']:.3f}, med(driver)={r['median_driver']:.3g}, "
                f"med(non-driver)={r['median_non_driver']:.3g}"
            )
        lines.append("")

    if len(categ_res) > 0:
        sig_cat = categ_res[categ_res["q_value"] < 0.05]
        lines += [
            "GLOBAL CATEGORICAL ASSOCIATIONS",
            "-" * 100,
            f"Tested categorical features: {len(categ_res)}",
            f"Significant after FDR (q<0.05): {len(sig_cat)}",
            "Top categorical signals:",
        ]
        for _, r in categ_res.head(10).iterrows():
            lines.append(
                f"  - {r['feature']}: q={r['q_value']:.2e}, p={r['p_value']:.2e}, "
                f"Cramer's V={r['cramers_v']:.3f}, levels={int(r['n_levels'])}"
            )
        lines.append("")

    if len(chrom_res) > 0:
        sig_ch = chrom_res[chrom_res["q_value"] < 0.05]
        lines += [
            "CHROMOSOME PAIR ENRICHMENT",
            "-" * 100,
            f"Tested chromosome pairs (n>=10): {len(chrom_res)}",
            f"Significant after FDR (q<0.05): {len(sig_ch)}",
            "Top enrichments:",
        ]
        for _, r in chrom_res.head(10).iterrows():
            lines.append(
                f"  - {r['chrom_pair']}: q={r['q_value']:.2e}, OR={r['odds_ratio']:.2f}, "
                f"driver={int(r['driver_in_pair'])}, non-driver={int(r['non_driver_in_pair'])}"
            )
        lines.append("")

    lines += [
        "TARGETED CONTEXT ANALYSES",
        "-" * 100,
        f"Targeted numeric tests: {len(targeted_num)} | significant: {int((targeted_num.get('q_value', pd.Series(dtype=float)) < 0.05).sum()) if len(targeted_num) else 0}",
        f"Targeted categorical tests: {len(targeted_cat)} | significant: {int((targeted_cat.get('q_value', pd.Series(dtype=float)) < 0.05).sum()) if len(targeted_cat) else 0}",
        "",
        "INTERPRETATION NOTES",
        "-" * 100,
        "1) Associations are with model predictions, not ground-truth experimental labels.",
        "2) FDR correction controls multiple testing but does not imply causality.",
        "3) Use effect size together with q-value for biological prioritization.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Predict driver/non-driver on reconstructed fusion sequences and run association analysis."
    )
    parser.add_argument(
        "--input-csv",
        default="/homes/gcapitani/driver-fusions/fusions_set1_17_results/fusions_set1_17_reconstructed_sequences.csv",
    )
    parser.add_argument(
        "--model-ckpt",
        default="/homes/gcapitani/driver-fusions/results/esmc/20260225_025900/probe_best.pt",
    )
    parser.add_argument(
        "--model",
        dest="model_ckpt",
        help="Alias for --model-ckpt (kept for backward compatibility).",
    )
    parser.add_argument(
        "--output-dir",
        default="/homes/gcapitani/driver-fusions/src/analysis/fusions_set_driver_analysis",
    )
    parser.add_argument(
        "--embedding-model",
        default="auto",
        choices=["auto", "esmc", "fuson"],
        help="Embedding encoder used for inference. 'auto' infers from checkpoint.",
    )
    parser.add_argument("--pool", choices=["mean", "cls"], default="mean")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for probe inference on precomputed embeddings",
    )
    parser.add_argument(
        "--emb-cache-dir",
        default="/work/H2020DeciderFicarra/gcapitani/driver-fusion/embeddings/fusions_set_cache",
        help="Directory where ESM-C embeddings are cached and reused",
    )
    parser.add_argument(
        "--emb-cache-name",
        default="",
        help="Optional custom cache filename (.pt). If empty, derived from input filename + pool.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    model_ckpt = Path(args.model_ckpt)
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading input table: {input_csv}")
    df = pd.read_csv(input_csv)
    if "reconstructed_seq" not in df.columns:
        raise ValueError("Input CSV must contain 'reconstructed_seq'")
    df = df[df["reconstructed_seq"].notna() & (df["reconstructed_seq"].astype(str).str.len() > 0)].copy()
    df = df.reset_index(drop=True)
    print(f"Rows with valid reconstructed sequence: {len(df)}")

    print(f"Loading probe checkpoint: {model_ckpt}")
    model, in_dim, hidden_dims, out_dim = build_model_from_checkpoint(model_ckpt, device)
    print(f"Probe architecture inferred: in_dim={in_dim}, hidden_dims={hidden_dims}, out_dim={out_dim}")
    embedding_model = infer_embedding_model(model_ckpt, in_dim, args.embedding_model)
    print(f"Embedding model selected: {embedding_model}")

    sequences = df["reconstructed_seq"].astype(str).tolist()
    seq_sig = sequence_signature(sequences)
    cache_dir = Path(args.emb_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = args.emb_cache_name.strip() or f"{input_csv.stem}_{embedding_model}_{args.pool}.pt"
    if not cache_name.endswith(".pt"):
        cache_name += ".pt"
    cache_path = cache_dir / cache_name

    emb_rows = None
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            same_sig = cached.get("sequence_signature", "") == seq_sig
            same_pool = cached.get("pool", "") == args.pool
            expected_encoder = "esmc_600m" if embedding_model == "esmc" else "fuson_plm"
            same_encoder = cached.get("encoder", "") == expected_encoder
            if same_sig and same_pool and same_encoder:
                emb_rows = cached.get("embeddings", None)
                print(f"Loaded cached embeddings: {cache_path}")
            else:
                print("Embedding cache found but metadata mismatch, recomputing embeddings.")
        except Exception as e:
            print(f"Failed to read embedding cache ({e}), recomputing embeddings.")

    if emb_rows is None:
        if embedding_model == "esmc":
            print("Loading ESM-C encoder...")
            encoder = load_esmc(device)
            embed_fn = lambda seq: embed_esmc(encoder, seq)
            encoder_tag = "esmc_600m"
        else:
            print("Loading FusOn-pLM encoder...")
            encoder = load_fuson(device)
            embed_fn = lambda seq: embed_fuson(encoder, seq, device)
            encoder_tag = "fuson_plm"
        print("Computing embeddings...")
        emb_rows = []
        for i, seq in enumerate(sequences, start=1):
            emb = embed_fn(seq)
            x = emb.mean(dim=0) if args.pool == "mean" else emb[0]
            if in_dim is not None and int(x.shape[0]) != int(in_dim):
                raise ValueError(f"Embedding dim mismatch at row {i}: got {x.shape[0]}, expected {in_dim}")
            emb_rows.append(x.cpu())
            if i % 50 == 0 or i == len(df):
                print(f"  Embedded {i}/{len(df)}")
        torch.save(
            {
                "encoder": encoder_tag,
                "pool": args.pool,
                "input_csv": str(input_csv),
                "sequence_signature": seq_sig,
                "embeddings": emb_rows,
            },
            cache_path,
        )
        print(f"Saved embedding cache: {cache_path}")

    print("Running batched probe inference...")
    X = torch.stack(emb_rows, dim=0)
    probs = []
    preds = []
    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for start in range(0, len(X), bs):
            end = min(start + bs, len(X))
            xb = X[start:end].to(device, non_blocking=True)
            logits = model(xb)
            pb = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
            yb = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            probs.extend([float(v) for v in pb])
            preds.extend([int(v) for v in yb])
            print(f"  Predicted {end}/{len(X)}")

    df["predicted_driver"] = preds
    df["predicted_driver_prob"] = probs
    df["predicted_label"] = df["predicted_driver"].map({1: "driver", 0: "non-driver"})

    pred_csv = out_dir / "fusions_set_predictions.csv"
    df.to_csv(pred_csv, index=False)
    print(f"Saved predictions: {pred_csv}")
    print_top_probability_fusions(df, prob_col="predicted_driver_prob", n=10)
    top_txt = save_top_probability_fusions_txt(
        df,
        out_dir / "top_probability_fusions.txt",
        prob_col="predicted_driver_prob",
        n=10,
    )
    if top_txt is not None:
        print(f"Saved: {top_txt}")

    print("Running statistical analyses...")
    numeric_res = analyze_numeric(df, pred_col="predicted_driver")
    categ_res = analyze_categorical(df, pred_col="predicted_driver")
    chrom_res = chromosome_enrichment(df, pred_col="predicted_driver")
    targeted_num, targeted_cat = targeted_feature_analysis(df, pred_col="predicted_driver", out_dir=out_dir)

    numeric_csv = out_dir / "numeric_associations.csv"
    categ_csv = out_dir / "categorical_associations.csv"
    chrom_csv = out_dir / "chromosome_pair_enrichment.csv"
    numeric_res.to_csv(numeric_csv, index=False)
    categ_res.to_csv(categ_csv, index=False)
    chrom_res.to_csv(chrom_csv, index=False)
    print(f"Saved: {numeric_csv}")
    print(f"Saved: {categ_csv}")
    print(f"Saved: {chrom_csv}")

    print("Generating figures...")
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    write_plots_guide(figs_dir / "PLOTS_README.txt")
    plot_core_figures(df, figs_dir)
    plot_association_summaries(numeric_res, categ_res, figs_dir)
    plot_targeted_panels(df, figs_dir)
    plot_index = plot_all_columns(df, figs_dir, pred_col="predicted_driver")
    print(f"Saved figures in: {figs_dir}")

    report_path = out_dir / "analysis_report.txt"
    write_report_txt(
        out_path=report_path,
        df=df,
        numeric_res=numeric_res,
        categ_res=categ_res,
        chrom_res=chrom_res,
        targeted_num=targeted_num,
        targeted_cat=targeted_cat,
        plot_index=plot_index,
        model_path=model_ckpt,
        input_csv=input_csv,
    )
    print(f"Saved report: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
