"""Operational pipeline steps for reconstruction, embedding, training, and evaluation."""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

from experiment.logging_utils import run_command


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def to_metric_rank(value: Any) -> float:
    x = to_float(value)
    if math.isnan(x):
        return float("-inf")
    return x


def metric_summary_from_text(summary_text: str) -> dict[str, float]:
    pattern = re.compile(
        r"TEST RESULTS - .*?\n\s*Accuracy:\s*([0-9eE+\-.]+|nan)\n\s*F1:\s*([0-9eE+\-.]+|nan)\n\s*AUROC:\s*([0-9eE+\-.]+|nan)\n\s*Precision:\s*([0-9eE+\-.]+|nan)\n\s*Recall:\s*([0-9eE+\-.]+|nan)",
        re.S,
    )
    match = pattern.search(summary_text)
    if match is None:
        raise ValueError("Unable to parse test metrics from train_probe summary.")

    accuracy, f1, auroc, precision, recall = match.groups()
    return {
        "accuracy": to_float(accuracy),
        "f1": to_float(f1),
        "auroc": to_float(auroc),
        "precision": to_float(precision),
        "recall": to_float(recall),
    }


def parse_summary_metrics(summary_path: Path) -> dict[str, float]:
    text = summary_path.read_text(encoding="utf-8")
    return metric_summary_from_text(text)


def sanitize_token(value: Any) -> str:
    token = str(value)
    token = token.replace(".", "p")
    token = token.replace("-", "m")
    token = token.replace("+", "")
    return token


def _normalize_probe_arches(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or ["linear"]
    text = str(value).strip()
    if not text:
        return ["linear"]
    if "," in text:
        out = [item.strip() for item in text.split(",") if item.strip()]
        return out or ["linear"]
    return [text]


def build_mode_dirs(workspace_root: Path, policy: str, recon_mode: str) -> dict[str, Path]:
    policy_root = workspace_root / policy
    mode_root = policy_root / recon_mode
    dirs = {
        "policy_root": policy_root,
        "mode_root": mode_root,
        "labeling": policy_root / "labeling",
        "reconstruction": mode_root / "reconstruction",
        "clustering": mode_root / "clustering",
        "embeddings": mode_root / "embeddings",
        "training": mode_root / "training",
        "evaluation": mode_root / "evaluation",
        "reports": mode_root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def expected_reconstruction_paths(dirs: dict[str, Path], policy: str, recon_mode: str) -> tuple[Path, Path]:
    mode_slug = recon_mode.lower().replace("-", "_")
    output_prefix = dirs["reconstruction"] / f"{policy}_{mode_slug}"
    output_csv = Path(f"{output_prefix}_results.csv")
    return output_prefix, output_csv


def _gene_count_table_for_subset(df_subset):
    import pandas as pd

    preferred_pairs = [("H_gene", "T_gene"), ("gene5", "gene3")]
    gene_cols: list[str] = []
    for c1, c2 in preferred_pairs:
        if c1 in df_subset.columns and c2 in df_subset.columns:
            gene_cols = [c1, c2]
            break
    if not gene_cols:
        gene_cols = [c for c in ("H_gene", "T_gene", "gene5", "gene3") if c in df_subset.columns]
    if not gene_cols:
        return pd.DataFrame(columns=["gene", "count"])

    parts = []
    for col in gene_cols:
        s = df_subset[col].dropna().astype(str).str.strip()
        s = s[s != ""]
        parts.append(s)
    if not parts:
        return pd.DataFrame(columns=["gene", "count"])

    genes = pd.concat(parts, ignore_index=True)
    counts = genes.value_counts().rename_axis("gene").reset_index(name="count")
    return counts


def generate_split_gene_distribution_plots(cluster_csv: Path, out_dir: Path, logger, top_n: int = 20) -> dict[str, Path]:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(cluster_csv)
    required = {"split", "driver"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Skipping split gene distribution plots, missing columns in %s: %s", cluster_csv, missing)
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    class_specs = [("driver", "driver"), ("non_driver", "non-driver")]
    split_order = [s for s in ("train", "val", "test") if s in set(df["split"].dropna().astype(str))]
    if not split_order:
        split_order = sorted(set(df["split"].dropna().astype(str)))

    for split in split_order:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (class_slug, class_label) in enumerate(class_specs):
            ax = axes[idx]
            subset = df[(df["split"] == split) & (df["driver"] == class_label)]
            counts = _gene_count_table_for_subset(subset)
            counts_path = out_dir / f"{split}_{class_slug}_gene_counts.csv"
            counts.to_csv(counts_path, index=False)

            if counts.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                ax.set_title(f"{split} | {class_label} (n=0)")
                continue

            top = counts.head(top_n).iloc[::-1]
            ax.barh(top["gene"], top["count"])
            ax.set_xlabel("Occurrences")
            ax.set_ylabel("Gene")
            ax.set_title(f"{split} | {class_label} (rows={len(subset)})")

        fig.suptitle(f"Gene Distribution by Class — Split: {split}", fontsize=14)
        fig.tight_layout()
        plot_path = out_dir / f"{split}_gene_distribution_by_class.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        plot_paths[split] = plot_path

    return plot_paths


def maybe_log_split_gene_plots_to_wandb(
    policy: str,
    recon_mode: str,
    plot_paths: dict[str, Path],
    config: dict[str, Any],
    logger,
) -> None:
    if not plot_paths:
        return

    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return

    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        logger.warning("wandb requested but unavailable for split plots: %s", exc)
        return

    mode = str(wandb_cfg.get("mode", "offline")).strip() or "offline"
    project = str(wandb_cfg.get("project", "driver-fusions-policy-comparison")).strip()
    entity = str(wandb_cfg.get("entity", "")).strip() or None
    tags = [str(t) for t in wandb_cfg.get("tags", [])]
    wandb_dir = str(wandb_cfg.get("dir", "/work/H2020DeciderFicarra/gcapitani")).strip()
    run_name = f"{policy}_{recon_mode}_split_gene_distribution"
    group_name = f"{policy}_{recon_mode}_clustering"

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            dir=wandb_dir,
            name=run_name,
            group=group_name,
            tags=tags + ["clustering", "split-gene-distribution"],
            config={"policy": policy, "recon_mode": recon_mode, "stage": "cluster"},
            reinit=True,
        )
        payload = {
            f"clustering/gene_distribution/{split}": wandb.Image(str(path))
            for split, path in sorted(plot_paths.items())
        }
        wandb.log(payload)
        run.finish()
    except Exception as exc:  # pragma: no cover
        logger.warning("wandb logging failed for split gene distributions (%s/%s): %s", policy, recon_mode, exc)


def run_reconstruction(
    policy: str,
    recon_mode: str,
    labeled_csv: Path,
    dirs: dict[str, Path],
    config: dict[str, Any],
    skip_existing: bool,
    dry_run: bool,
    logger,
) -> Path:
    recon_cfg = config["reconstruction"]
    output_prefix, recon_csv = expected_reconstruction_paths(dirs, policy, recon_mode)

    if skip_existing and recon_csv.exists():
        logger.info("[%s/%s] Reconstruction already exists, skipping: %s", policy, recon_mode, recon_csv)
        return recon_csv

    script = PROJECT_ROOT / "src" / "seq_recon" / "run.py"
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(labeled_csv),
        "--output",
        str(output_prefix),
        "--seed",
        str(recon_cfg.get("seed", 42)),
        "--genome-build",
        str(recon_cfg.get("genome_build", "all")),
        "--min-protein-len-aa",
        str(int(recon_cfg.get("min_protein_len_aa", 30))),
    ]

    n_samples = recon_cfg.get("n_samples")
    if n_samples is not None:
        cmd.extend(["--n", str(int(n_samples))])

    use_orffinder = recon_mode.upper() == "ORF"
    if use_orffinder:
        orffinder_path = str(recon_cfg.get("orffinder_path", "")).strip()
        if orffinder_path:
            cmd.extend(["--orffinder-path", orffinder_path])
    else:
        cmd.append("--no-orffinder")

    run_command(cmd, logger=logger, cwd=PROJECT_ROOT, dry_run=dry_run)

    if not dry_run and not recon_csv.exists():
        raise FileNotFoundError(f"Reconstruction output not found: {recon_csv}")

    return recon_csv


def run_clustering(
    policy: str,
    recon_mode: str,
    recon_csv: Path,
    dirs: dict[str, Path],
    config: dict[str, Any],
    skip_existing: bool,
    dry_run: bool,
    logger,
) -> Path:
    cluster_csv = dirs["clustering"] / "clustered_splits.csv"
    if skip_existing and cluster_csv.exists():
        logger.info("[%s/%s] Clustered split already exists, skipping: %s", policy, recon_mode, cluster_csv)
        if not dry_run:
            try:
                plot_dir = dirs["clustering"] / "split_gene_distribution"
                plot_paths = generate_split_gene_distribution_plots(cluster_csv, plot_dir, logger=logger)
                maybe_log_split_gene_plots_to_wandb(
                    policy=policy,
                    recon_mode=recon_mode,
                    plot_paths=plot_paths,
                    config=config,
                    logger=logger,
                )
                if plot_paths:
                    logger.info(
                        "[%s/%s] Saved split gene distribution plots in %s",
                        policy,
                        recon_mode,
                        plot_dir,
                    )
            except Exception as exc:
                logger.warning("[%s/%s] Failed to generate/log split gene distribution plots: %s", policy, recon_mode, exc)
        return cluster_csv

    cluster_cfg = config["clustering"]
    script = PROJECT_ROOT / "src" / "preprocessing" / "clustering.py"
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(recon_csv),
        "--output-dir",
        str(dirs["clustering"]),
        "--min-seq-id",
        str(cluster_cfg.get("min_seq_id", 0.3)),
        "--coverage",
        str(cluster_cfg.get("coverage", 0.8)),
        "--train-ratio",
        str(cluster_cfg.get("train_ratio", 0.7)),
        "--val-ratio",
        str(cluster_cfg.get("val_ratio", 0.15)),
        "--seed",
        str(cluster_cfg.get("seed", 42)),
    ]

    run_command(cmd, logger=logger, cwd=PROJECT_ROOT, dry_run=dry_run)

    if not dry_run and not cluster_csv.exists():
        raise FileNotFoundError(f"Clustered split output not found: {cluster_csv}")
    if not dry_run:
        try:
            plot_dir = dirs["clustering"] / "split_gene_distribution"
            plot_paths = generate_split_gene_distribution_plots(cluster_csv, plot_dir, logger=logger)
            maybe_log_split_gene_plots_to_wandb(
                policy=policy,
                recon_mode=recon_mode,
                plot_paths=plot_paths,
                config=config,
                logger=logger,
            )
            if plot_paths:
                logger.info(
                    "[%s/%s] Saved split gene distribution plots in %s",
                    policy,
                    recon_mode,
                    plot_dir,
                )
        except Exception as exc:
            logger.warning("[%s/%s] Failed to generate/log split gene distribution plots: %s", policy, recon_mode, exc)

    return cluster_csv


def run_embeddings(
    policy: str,
    recon_mode: str,
    model: str,
    clustered_csv: Path,
    dirs: dict[str, Path],
    config: dict[str, Any],
    skip_existing: bool,
    dry_run: bool,
    logger,
) -> Path:
    model_emb_dir = dirs["embeddings"] / model
    required_files = [model_emb_dir / f"{split}.pt" for split in ("train", "val", "test")]

    if skip_existing and all(path.exists() for path in required_files):
        logger.info("[%s/%s/%s] Embeddings already exist, skipping.", policy, recon_mode, model)
        return model_emb_dir

    script = PROJECT_ROOT / "src" / "compute_embeddings.py"
    data_cfg = config["data"]

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(clustered_csv),
        "--model",
        model,
        "--output-dir",
        str(dirs["embeddings"]),
    ]

    benchmark_csv = str(data_cfg.get("benchmark_csv", "")).strip()
    if benchmark_csv:
        cmd.extend(["--benchmark", benchmark_csv])

    run_command(cmd, logger=logger, cwd=PROJECT_ROOT, dry_run=dry_run)

    if not dry_run and not all(path.exists() for path in required_files):
        missing = [str(path) for path in required_files if not path.exists()]
        raise FileNotFoundError(f"Missing embedding files after run: {missing}")

    return model_emb_dir


def maybe_log_to_wandb(record: dict[str, Any], config: dict[str, Any], logger) -> None:
    wandb_cfg = config.get("wandb", {})
    if bool(wandb_cfg.get("log_from_train_probe", True)):
        return
    if not bool(wandb_cfg.get("enabled", False)):
        return

    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        logger.warning("wandb requested but unavailable: %s", exc)
        return

    mode = str(wandb_cfg.get("mode", "offline")).strip() or "offline"
    project = str(wandb_cfg.get("project", "driver-fusions-policy-comparison")).strip()
    entity = str(wandb_cfg.get("entity", "")).strip() or None
    tags = [str(t) for t in wandb_cfg.get("tags", [])]
    wandb_dir = str(wandb_cfg.get("dir", "/work/H2020DeciderFicarra/gcapitani")).strip()

    run_name = f"{record['policy']}_{record['recon_mode']}_{record['model']}_{record['run_id']}"
    group_name = f"{record['policy']}_{record['recon_mode']}_{record['model']}"

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            dir=wandb_dir,
            name=run_name,
            group=group_name,
            tags=tags,
            config={
                "policy": record["policy"],
                "recon_mode": record["recon_mode"],
                "model": record["model"],
                "run_id": record["run_id"],
                "lr": record["lr"],
                "batch_size": record["batch_size"],
                "noise": record["noise"],
                "focal_gamma": record["focal_gamma"],
            },
            reinit=True,
        )
        wandb.log(
            {
                "accuracy": record["accuracy"],
                "f1": record["f1"],
                "auroc": record["auroc"],
                "precision": record["precision"],
                "recall": record["recall"],
            }
        )
        run.finish()
    except Exception as exc:  # pragma: no cover
        logger.warning("wandb logging failed for run %s: %s", run_name, exc)


def run_aggregate_baselines(
    policy: str,
    recon_mode: str,
    model: str,
    dirs: dict[str, Path],
    config: dict[str, Any],
    records: list[dict[str, Any]],
    skip_existing: bool,
    dry_run: bool,
    logger,
) -> list[dict[str, Any]]:
    if dry_run:
        logger.info("[%s/%s/%s] Dry run: skipping aggregate baselines.", policy, recon_mode, model)
        return []

    ckpts: list[str] = []
    for r in records:
        ckpt = str(r.get("probe_best_ckpt", "")).strip()
        if ckpt and Path(ckpt).exists():
            ckpts.append(ckpt)
    if not ckpts:
        logger.info("[%s/%s/%s] No checkpoints available for aggregate baselines.", policy, recon_mode, model)
        return []

    aggregate_script = PROJECT_ROOT / "src" / "analysis" / "aggregate_policy_models.py"
    setting_id = f"{policy}_{recon_mode}_{model}"
    output_root = dirs["training"] / model
    result_json = output_root / "aggregates" / setting_id / "aggregate_results.json"

    if skip_existing and result_json.exists():
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                out: list[dict[str, Any]] = []
                for row in payload:
                    if not isinstance(row, dict):
                        continue
                    out.append({"policy": policy, "recon_mode": recon_mode, "model": model, **row})
                if out:
                    logger.info("[%s/%s/%s] Reusing aggregate baselines: %s", policy, recon_mode, model, result_json)
                    return out
        except Exception:
            pass

    wandb_cfg = config.get("wandb", {})
    cmd = [
        sys.executable,
        str(aggregate_script),
        "--embeddings-dir",
        str(dirs["embeddings"]),
        "--model",
        model,
        "--pool",
        str(config["training"].get("pool", "mean")),
        "--output-dir",
        str(output_root),
        "--setting-id",
        setting_id,
        "--policy",
        policy,
        "--wandb-mode",
        str(wandb_cfg.get("mode", "disabled")),
        "--wandb-project",
        str(wandb_cfg.get("project", "driver-fusions-policy-comparison")),
        "--wandb-entity",
        str(wandb_cfg.get("entity", "")),
        "--wandb-tags",
        ",".join(str(t) for t in wandb_cfg.get("tags", [])),
        "--wandb-dir",
        str(wandb_cfg.get("dir", "/work/H2020DeciderFicarra/gcapitani")),
    ]
    if bool(wandb_cfg.get("enabled", False)):
        cmd.append("--wandb-enabled")
    if skip_existing:
        cmd.append("--skip-existing")
    for ckpt in ckpts:
        cmd.extend(["--checkpoint", ckpt])

    run_command(cmd, logger=logger, cwd=PROJECT_ROOT, dry_run=False)

    if not result_json.exists():
        logger.warning("[%s/%s/%s] Aggregate results missing after run: %s", policy, recon_mode, model, result_json)
        return []

    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[%s/%s/%s] Failed parsing aggregate results: %s", policy, recon_mode, model, exc)
        return []
    if not isinstance(payload, list):
        return []

    out_records: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        out_records.append({"policy": policy, "recon_mode": recon_mode, "model": model, **row})
    return out_records


def train_grid(
    policy: str,
    recon_mode: str,
    model: str,
    dirs: dict[str, Path],
    config: dict[str, Any],
    skip_existing: bool,
    dry_run: bool,
    logger,
    on_trained_record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    train_cfg = config["training"]
    grid = train_cfg["grid"]

    lrs = list(grid.get("lr", [1e-4]))
    batch_sizes = list(grid.get("batch_size", [64]))
    noises = list(grid.get("noise", [0.0]))
    gammas = list(grid.get("focal_gamma", [2.0]))
    probe_arches = _normalize_probe_arches(train_cfg.get("probe_arch", "linear"))
    wandb_cfg = config.get("wandb", {})

    train_script = PROJECT_ROOT / "src" / "train_probe.py"
    records: list[dict[str, Any]] = []
    trained_records: list[dict[str, Any]] = []

    all_combinations = list(itertools.product(lrs, batch_sizes, noises, gammas, probe_arches))
    for run_idx, (lr, batch_size, noise, gamma, probe_arch) in enumerate(all_combinations, start=1):
        run_id = (
            f"{policy}_{recon_mode}_{model}"
            f"_lr{sanitize_token(lr)}"
            f"_bs{sanitize_token(batch_size)}"
            f"_ns{sanitize_token(noise)}"
            f"_fg{sanitize_token(gamma)}"
            f"_pa{sanitize_token(probe_arch)}"
            f"_{run_idx:03d}"
        )

        run_dir = dirs["training"] / model / run_id
        summary_path = run_dir / "summary.txt"
        ckpt_path = run_dir / "probe_best.pt"

        if skip_existing and summary_path.exists():
            logger.info("[%s/%s/%s] Reusing existing run: %s", policy, recon_mode, model, run_id)
            metrics = parse_summary_metrics(summary_path)
        elif dry_run:
            logger.info("[%s/%s/%s] Dry run training: %s", policy, recon_mode, model, run_id)
            metrics = {
                "accuracy": float("nan"),
                "f1": float("nan"),
                "auroc": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
            }
        else:
            cmd = [
                sys.executable,
                str(train_script),
                "--embeddings-dir",
                str(dirs["embeddings"]),
                "--output-dir",
                str(dirs["training"]),
                "--model",
                model,
                "--run-id",
                run_id,
                "--policy",
                policy,
                "--pool",
                str(train_cfg.get("pool", "mean")),
                "--epochs",
                str(train_cfg.get("epochs", 2000)),
                "--patience",
                str(train_cfg.get("patience", 400)),
                "--seed",
                str(train_cfg.get("seed", 42)),
                "--probe-arch",
                str(probe_arch),
                "--probe-hidden-dims",
                str(train_cfg.get("probe_hidden_dims", "")),
                "--probe-dropout",
                str(train_cfg.get("probe_dropout", 0.2)),
                "--lr-scheduler",
                str(train_cfg.get("lr_scheduler", "plateau")),
                "--lr-reduce-factor",
                str(train_cfg.get("lr_reduce_factor", 0.5)),
                "--lr-min",
                str(train_cfg.get("lr_min", 1e-7)),
                "--lr",
                str(lr),
                "--batch-size",
                str(int(batch_size)),
                "--train-noise-std",
                str(noise),
                "--focal-gamma",
                str(gamma),
            ]

            cfg_path = str(train_cfg.get("train_probe_config", "")).strip()
            if cfg_path:
                cmd.extend(["--config", cfg_path])

            if bool(wandb_cfg.get("enabled", False)):
                cmd.append("--wandb-enabled")
            cmd.extend(
                [
                    "--wandb-mode",
                    str(wandb_cfg.get("mode", "disabled")),
                    "--wandb-project",
                    str(wandb_cfg.get("project", "driver-fusions-policy-comparison")),
                    "--wandb-entity",
                    str(wandb_cfg.get("entity", "")),
                    "--wandb-tags",
                    ",".join(str(t) for t in wandb_cfg.get("tags", [])),
                    "--wandb-dir",
                    str(wandb_cfg.get("dir", "/work/H2020DeciderFicarra/gcapitani")),
                ]
            )

            run_command(cmd, logger=logger, cwd=PROJECT_ROOT, dry_run=False)

            if not summary_path.exists():
                raise FileNotFoundError(f"Summary not found after training: {summary_path}")
            metrics = parse_summary_metrics(summary_path)

        record = {
            "policy": policy,
            "recon_mode": recon_mode,
            "model": model,
            "run_id": run_id,
            "lr": float(lr),
            "batch_size": int(batch_size),
            "noise": float(noise),
            "focal_gamma": float(gamma),
            "probe_arch": str(probe_arch),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auroc": metrics["auroc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "summary_path": str(summary_path),
            "probe_best_ckpt": str(ckpt_path),
        }
        records.append(record)
        trained_records.append(record)
        maybe_log_to_wandb(record, config, logger)
        if on_trained_record is not None:
            on_trained_record(record)

    aggregate_records = run_aggregate_baselines(
        policy=policy,
        recon_mode=recon_mode,
        model=model,
        dirs=dirs,
        config=config,
        records=trained_records,
        skip_existing=skip_existing,
        dry_run=dry_run,
        logger=logger,
    )
    if aggregate_records:
        records.extend(aggregate_records)

    best_record = None
    if trained_records:
        best_record = max(
            trained_records,
            key=lambda r: (
                to_metric_rank(r["auroc"]),
                to_metric_rank(r["f1"]),
                to_metric_rank(r["accuracy"]),
            ),
        )

    return records, best_record


def collect_existing_training_records(policy: str, recon_mode: str, model: str, dirs: dict[str, Path]) -> list[dict[str, Any]]:
    model_train_root = dirs["training"] / model
    if not model_train_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for summary_path in sorted(model_train_root.glob("*/summary.txt")):
        run_dir = summary_path.parent
        try:
            metrics = parse_summary_metrics(summary_path)
        except Exception:
            continue

        records.append(
            {
                "policy": policy,
                "recon_mode": recon_mode,
                "model": model,
                "run_id": run_dir.name,
                "lr": float("nan"),
                "batch_size": float("nan"),
                "noise": float("nan"),
                "focal_gamma": float("nan"),
                "probe_arch": "unknown",
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "summary_path": str(summary_path),
                "probe_best_ckpt": str(run_dir / "probe_best.pt"),
            }
        )
    return records


def find_latest_eval_dir(eval_root: Path) -> Path | None:
    candidates = [path for path in eval_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_external_evaluation(
    policy: str,
    recon_mode: str,
    model: str,
    best_record: dict[str, Any],
    dirs: dict[str, Path],
    config: dict[str, Any],
    skip_existing: bool,
    dry_run: bool,
    logger,
) -> dict[str, str] | None:
    if not bool(config.get("runtime", {}).get("enable_external_evaluation", False)):
        logger.info(
            "[%s/%s/%s] External evaluation disabled by config (runtime.enable_external_evaluation=false).",
            policy,
            recon_mode,
            model,
        )
        return None

    eval_input = Path(str(config["data"].get("evaluation_input_csv", "")))
    if not eval_input.exists() and not dry_run:
        logger.warning(
            "[%s/%s/%s] Evaluation skipped, input CSV not found: %s",
            policy,
            recon_mode,
            model,
            eval_input,
        )
        return None

    model_eval_root = dirs["evaluation"] / model
    model_eval_root.mkdir(parents=True, exist_ok=True)

    if skip_existing and model_eval_root.exists():
        latest_existing = find_latest_eval_dir(model_eval_root)
        if latest_existing is not None and (latest_existing / "analysis_report.txt").exists():
            logger.info("[%s/%s/%s] Reusing existing evaluation: %s", policy, recon_mode, model, latest_existing)
            return {
                "evaluation_dir": str(latest_existing),
                "analysis_report": str(latest_existing / "analysis_report.txt"),
                "predictions_csv": str(latest_existing / "fusions_set_predictions.csv"),
            }

    eval_script = PROJECT_ROOT / "src" / "analysis" / "predict_fusions_set_driver.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--input-csv",
        str(eval_input),
        "--model-ckpt",
        str(best_record["probe_best_ckpt"]),
        "--embedding-model",
        model,
        "--pool",
        str(config["training"].get("pool", "mean")),
        "--output-dir",
        str(model_eval_root),
        "--emb-cache-dir",
        str(model_eval_root / "embedding_cache"),
    ]

    run_command(cmd, logger=logger, cwd=PROJECT_ROOT, dry_run=dry_run)

    if dry_run:
        return None

    latest_dir = find_latest_eval_dir(model_eval_root)
    if latest_dir is None:
        logger.warning("[%s/%s/%s] Evaluation finished but no output directory found.", policy, recon_mode, model)
        return None

    return {
        "evaluation_dir": str(latest_dir),
        "analysis_report": str(latest_dir / "analysis_report.txt"),
        "predictions_csv": str(latest_dir / "fusions_set_predictions.csv"),
    }
