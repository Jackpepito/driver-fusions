#!/usr/bin/env python3
"""Orchestrate A/B/C/D experiments from labeling to comparison reports."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.config import DEFAULT_CONFIG_PATH, load_config, stages_from_arg
from experiment.labeling import (
    create_policy_summary,
    label_dataset_for_policy,
    load_census_genes,
    prepare_chimerseq_base,
    save_labeling_distribution_plots,
)
from experiment.logging_utils import setup_logger, write_json, write_text
from experiment.pipeline import (
    build_mode_dirs,
    collect_existing_training_records,
    expected_reconstruction_paths,
    run_clustering,
    run_embeddings,
    run_reconstruction,
    to_float,
    to_metric_rank,
    train_grid,
)
from experiment.plotting import save_policy_comparison_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full A/B/C/D experiment pipeline with config file")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to JSON config file")
    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="Optional policy filter (single value or comma-separated list, e.g. 'A' or 'A,B')",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages: label,reconstruct,cluster,embed,train,evaluate,compare or 'all'",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps with existing outputs (overrides config runtime.skip_existing)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    selected_stages = stages_from_arg(args.stages)
    label_only = selected_stages == {"label"}
    skip_existing = bool(config.get("runtime", {}).get("skip_existing", True)) or args.skip_existing

    workspace_root = Path(str(config["workspace_root"]))
    workspace_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = workspace_root / "pipeline_logs" / f"policy_pipeline_{timestamp}.log"
    logger = setup_logger(log_path)

    logger.info("Pipeline started")
    logger.info("Config: %s", config_path)
    logger.info("Stages: %s", ", ".join(sorted(selected_stages)))
    logger.info("Workspace root: %s", workspace_root)
    logger.info("Dry run: %s", args.dry_run)
    logger.info("Skip existing: %s", skip_existing)

    data_cfg = config["data"]
    chimerseq_path = Path(str(data_cfg["chimerseq_csv"]))
    census_path = Path(str(data_cfg["census_tsv"]))

    base_df: pd.DataFrame | None = None
    census_genes: set[str] | None = None

    all_grid_records: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []

    policies = [str(policy).upper() for policy in config["policies"]]
    recon_modes = [str(mode) for mode in config["recon_modes"]]
    models = [str(model).lower() for model in config["models"]]

    if args.policy.strip():
        requested_policies = [item.strip().upper() for item in args.policy.split(",") if item.strip()]
        if not requested_policies:
            raise ValueError("Received empty --policy filter after parsing.")

        unknown = [policy for policy in requested_policies if policy not in policies]
        if unknown:
            raise ValueError(
                f"Unknown policy in --policy: {unknown}. Available policies in config: {policies}"
            )

        policies = requested_policies
        logger.info("Policy filter enabled: %s", ", ".join(policies))

    # Explicit creation of /work/.../{A,B,C,D} as requested.
    for policy in policies:
        (workspace_root / policy).mkdir(parents=True, exist_ok=True)

    for policy in policies:
        logger.info("Processing policy %s", policy)

        default_policy_dirs = build_mode_dirs(workspace_root, policy, recon_modes[0])
        labeled_csv = default_policy_dirs["labeling"] / f"chimerseq_labeled_policy_{policy}.csv"
        summary_txt = default_policy_dirs["labeling"] / f"summary_policy_{policy}.txt"

        must_build_labels = ("label" in selected_stages) or (
            "reconstruct" in selected_stages and not labeled_csv.exists()
        )
        if must_build_labels:
            if args.dry_run:
                logger.info("[dry-run] Would build labels for policy %s into %s", policy, labeled_csv)
            else:
                if base_df is None:
                    logger.info("Loading and preparing ChimerSeq base table from %s", chimerseq_path)
                    base_df = prepare_chimerseq_base(str(chimerseq_path))
                    logger.info("Prepared base rows: %d", len(base_df))
                if census_genes is None:
                    logger.info("Loading census genes from %s", census_path)
                    census_genes = load_census_genes(str(census_path))
                    logger.info("Loaded census genes: %d", len(census_genes))

                labeled_df = label_dataset_for_policy(base_df, policy, census_genes)
                labeled_csv.parent.mkdir(parents=True, exist_ok=True)
                labeled_df.to_csv(labeled_csv, index=False)
                write_text(summary_txt, create_policy_summary(policy, labeled_df))
                plot_artifacts = save_labeling_distribution_plots(
                    labeled_df=labeled_df,
                    output_root=labeled_csv.parent / "distribution_plots",
                    policy=policy,
                )
                logger.info(
                    "Policy %s labeled rows: %d (drivers=%d, non-drivers=%d)",
                    policy,
                    len(labeled_df),
                    int((labeled_df["driver"] == "driver").sum()),
                    int((labeled_df["driver"] == "non-driver").sum()),
                )
                if plot_artifacts:
                    logger.info(
                        "Policy %s labeling distribution artifacts saved: %s",
                        policy,
                        [str(path) for path in plot_artifacts],
                    )
        else:
            logger.info("Policy labels already exist, reusing %s", labeled_csv)

        if label_only:
            continue

        for recon_mode in recon_modes:
            dirs = build_mode_dirs(workspace_root, policy, recon_mode)
            _, expected_recon_csv = expected_reconstruction_paths(dirs, policy, recon_mode)
            cluster_csv = dirs["clustering"] / "clustered_splits.csv"

            if "reconstruct" in selected_stages:
                recon_csv = run_reconstruction(
                    policy=policy,
                    recon_mode=recon_mode,
                    labeled_csv=labeled_csv,
                    dirs=dirs,
                    config=config,
                    skip_existing=skip_existing,
                    dry_run=args.dry_run,
                    logger=logger,
                )
            else:
                recon_csv = expected_recon_csv
                if not recon_csv.exists() and not args.dry_run and ({"cluster", "embed", "train", "evaluate"} & selected_stages):
                    raise FileNotFoundError(
                        f"Missing reconstruction output for {policy}/{recon_mode}: {recon_csv}. "
                        "Run with --stages reconstruct (or all) first."
                    )

            if "cluster" in selected_stages:
                cluster_csv = run_clustering(
                    policy=policy,
                    recon_mode=recon_mode,
                    recon_csv=recon_csv,
                    dirs=dirs,
                    config=config,
                    skip_existing=skip_existing,
                    dry_run=args.dry_run,
                    logger=logger,
                )
            elif not cluster_csv.exists() and not args.dry_run and ({"embed", "train", "evaluate"} & selected_stages):
                raise FileNotFoundError(
                    f"Missing clustered split for {policy}/{recon_mode}: {cluster_csv}. "
                    "Run with --stages cluster (or all) first."
                )

            for model in models:
                logger.info("Running setting policy=%s mode=%s model=%s", policy, recon_mode, model)

                if "embed" in selected_stages:
                    run_embeddings(
                        policy=policy,
                        recon_mode=recon_mode,
                        model=model,
                        clustered_csv=cluster_csv,
                        dirs=dirs,
                        config=config,
                        skip_existing=skip_existing,
                        dry_run=args.dry_run,
                        logger=logger,
                    )

                records_for_setting: list[dict[str, Any]] = []
                best_for_setting: dict[str, Any] | None = None
                if "train" in selected_stages:
                    records_for_setting, best_for_setting = train_grid(
                        policy=policy,
                        recon_mode=recon_mode,
                        model=model,
                        dirs=dirs,
                        config=config,
                        skip_existing=skip_existing,
                        dry_run=args.dry_run,
                        logger=logger,
                    )
                elif "compare" in selected_stages or "evaluate" in selected_stages:
                    records_for_setting = collect_existing_training_records(
                        policy=policy,
                        recon_mode=recon_mode,
                        model=model,
                        dirs=dirs,
                    )
                    if records_for_setting:
                        best_for_setting = max(
                            records_for_setting,
                            key=lambda r: (
                                to_metric_rank(r["auroc"]),
                                to_metric_rank(r["f1"]),
                                to_metric_rank(r["accuracy"]),
                            ),
                        )

                all_grid_records.extend(records_for_setting)

                if best_for_setting is not None:
                    logger.info(
                        "Best run %s/%s/%s -> run_id=%s, AUROC=%.4f, F1=%.4f",
                        policy,
                        recon_mode,
                        model,
                        best_for_setting["run_id"],
                        to_float(best_for_setting["auroc"]),
                        to_float(best_for_setting["f1"]),
                    )

                    if "evaluate" in selected_stages:
                        logger.info(
                            "[%s/%s/%s] Evaluate stage uses in-training metrics (test + OOD) logged by train_probe/W&B.",
                            policy,
                            recon_mode,
                            model,
                        )

                    best_records.append(best_for_setting)
                else:
                    logger.warning("No successful training runs found for %s/%s/%s", policy, recon_mode, model)

    reports_root = workspace_root / "comparison"
    if args.policy.strip():
        reports_root = reports_root / f"policies_{'_'.join(policies)}"
    reports_root.mkdir(parents=True, exist_ok=True)

    if all_grid_records:
        grid_df = pd.DataFrame(all_grid_records)
        grid_csv = reports_root / "grid_search_results.csv"
        grid_df.to_csv(grid_csv, index=False)
        write_json(reports_root / "grid_search_results.json", all_grid_records)
        logger.info("Saved grid results: %s", grid_csv)

        if "compare" in selected_stages:
            plot_paths = save_policy_comparison_plots(grid_df, reports_root)
            logger.info("Saved comparison artifacts: %s", [str(p) for p in plot_paths])

    if best_records:
        best_df = pd.DataFrame(best_records)
        best_csv = reports_root / "best_runs.csv"
        best_df.to_csv(best_csv, index=False)
        write_json(reports_root / "best_runs.json", best_records)
        logger.info("Saved best-run summary: %s", best_csv)

    manifest = {
        "timestamp": timestamp,
        "config_path": str(config_path),
        "workspace_root": str(workspace_root),
        "stages": sorted(selected_stages),
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(skip_existing),
        "n_grid_records": len(all_grid_records),
        "n_best_records": len(best_records),
        "log_path": str(log_path),
    }
    write_json(reports_root / "run_manifest.json", manifest)

    logger.info("Pipeline completed")
    logger.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
