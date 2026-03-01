"""Operational pipeline steps for reconstruction, embedding, training, and evaluation."""

from __future__ import annotations

import itertools
import math
import re
import sys
from pathlib import Path
from typing import Any

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

    run_name = f"{record['policy']}_{record['recon_mode']}_{record['model']}_{record['run_id']}"
    group_name = f"{record['policy']}_{record['recon_mode']}_{record['model']}"

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
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


def train_grid(
    policy: str,
    recon_mode: str,
    model: str,
    dirs: dict[str, Path],
    config: dict[str, Any],
    skip_existing: bool,
    dry_run: bool,
    logger,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    train_cfg = config["training"]
    grid = train_cfg["grid"]

    lrs = list(grid.get("lr", [1e-4]))
    batch_sizes = list(grid.get("batch_size", [64]))
    noises = list(grid.get("noise", [0.0]))
    gammas = list(grid.get("focal_gamma", [2.0]))

    train_script = PROJECT_ROOT / "src" / "train_probe.py"
    records: list[dict[str, Any]] = []

    all_combinations = list(itertools.product(lrs, batch_sizes, noises, gammas))
    for run_idx, (lr, batch_size, noise, gamma) in enumerate(all_combinations, start=1):
        run_id = (
            f"{policy}_{recon_mode}_{model}"
            f"_lr{sanitize_token(lr)}"
            f"_bs{sanitize_token(batch_size)}"
            f"_ns{sanitize_token(noise)}"
            f"_fg{sanitize_token(gamma)}"
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
                "--pool",
                str(train_cfg.get("pool", "mean")),
                "--epochs",
                str(train_cfg.get("epochs", 2000)),
                "--patience",
                str(train_cfg.get("patience", 400)),
                "--seed",
                str(train_cfg.get("seed", 42)),
                "--probe-arch",
                str(train_cfg.get("probe_arch", "linear")),
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
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auroc": metrics["auroc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "summary_path": str(summary_path),
            "probe_best_ckpt": str(ckpt_path),
        }
        records.append(record)
        maybe_log_to_wandb(record, config, logger)

    best_record = None
    if records:
        best_record = max(
            records,
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
