"""Plotting helpers for comparing policy-level experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _best_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df.copy()

    ordered = results_df.sort_values(
        by=["policy", "recon_mode", "model", "auroc", "f1", "accuracy"],
        ascending=[True, True, True, False, False, False],
    )
    return ordered.groupby(["policy", "recon_mode", "model"], as_index=False).first()


def save_policy_comparison_plots(results_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Save compact comparison plots for best runs across policies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if results_df.empty:
        return paths

    best_df = _best_rows(results_df)
    best_csv = out_dir / "best_runs_by_policy_mode_model.csv"
    best_df.to_csv(best_csv, index=False)
    paths.append(best_csv)

    # Plot 1: AUROC per policy across reconstruction+embedding settings.
    plot_df = best_df.copy()
    plot_df["setting"] = plot_df["recon_mode"].astype(str) + " | " + plot_df["model"].astype(str)
    pivot_auroc = plot_df.pivot(index="policy", columns="setting", values="auroc").sort_index()

    if not pivot_auroc.empty:
        ax = pivot_auroc.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Best AUROC by Policy and Setting")
        ax.set_ylabel("AUROC")
        ax.set_xlabel("Policy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Setting", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure()
        fig.tight_layout()
        p = out_dir / "best_auroc_by_policy.png"
        fig.savefig(p, dpi=220)
        plt.close(fig)
        paths.append(p)

    # Plot 2: AUROC distribution across all grid runs by policy.
    policies = sorted(results_df["policy"].dropna().unique().tolist())
    if policies:
        data = [
            pd.to_numeric(results_df.loc[results_df["policy"] == policy, "auroc"], errors="coerce").dropna().values
            for policy in policies
        ]
        if any(len(x) > 0 for x in data):
            fig, ax = plt.subplots(figsize=(10, 5))
            bp = ax.boxplot(
                data,
                labels=policies,
                patch_artist=True,
                showmeans=True,
            )
            palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(palette[i % len(palette)])
                box.set_alpha(0.6)
            ax.set_title("Grid Search AUROC Distribution by Policy")
            ax.set_xlabel("Policy")
            ax.set_ylabel("AUROC")
            ax.set_ylim(0.0, 1.05)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            p = out_dir / "auroc_distribution_by_policy.png"
            fig.savefig(p, dpi=220)
            plt.close(fig)
            paths.append(p)

    # Plot 3: Number of runs per setting.
    counts = (
        results_df.groupby(["policy", "recon_mode", "model"], as_index=False)
        .size()
        .rename(columns={"size": "n_runs"})
    )
    if not counts.empty:
        counts["setting"] = counts["recon_mode"].astype(str) + " | " + counts["model"].astype(str)
        pivot_counts = counts.pivot(index="policy", columns="setting", values="n_runs").fillna(0)
        ax = pivot_counts.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Completed Grid Runs by Policy and Setting")
        ax.set_xlabel("Policy")
        ax.set_ylabel("Runs")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Setting", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure()
        fig.tight_layout()
        p = out_dir / "run_counts_by_policy.png"
        fig.savefig(p, dpi=220)
        plt.close(fig)
        paths.append(p)

    return paths
