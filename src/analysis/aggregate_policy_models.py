#!/usr/bin/env python3
"""Build ensemble probe baseline for one training setting."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure `src` is importable when this file is executed directly
# (e.g. `python src/analysis/aggregate_policy_models.py`).
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment.probe_io import build_model_from_checkpoint
from train_probe import (
    _compute_calibration_metrics,
    _compute_random_baseline,
    _extract_fusion_name_set,
    analyze_test_predictions,
    format_test_results,
)
from utils import load_embeddings


def _safe_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return float("nan")


def _metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    preds = (probs >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auroc": _safe_auroc(y_true, probs),
        "prec": float(precision_score(y_true, preds, zero_division=0)),
        "rec": float(recall_score(y_true, preds, zero_division=0)),
        "preds": preds,
        "probs": probs,
        "labels": y_true,
    }


@torch.no_grad()
def _predict_probs(ckpt_path: Path, X: torch.Tensor, device: str) -> np.ndarray:
    model, _, _, out_dim = build_model_from_checkpoint(ckpt_path, device)
    if int(out_dim) != 2:
        raise ValueError(f"Only binary probes are supported. Got out_dim={out_dim} for {ckpt_path}")
    logits = model(X.to(device))
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(float)


def _plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_non-driver", "pred_driver"])
    ax.set_yticks([0, 1], labels=["true_non-driver", "true_driver"])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _wandb_log_all(
    args,
    setting_id: str,
    records: list[dict[str, Any]],
    method_outputs: dict[str, dict[str, Path]],
) -> None:
    if not args.wandb_enabled or args.wandb_mode == "disabled":
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[WARNING] wandb logging disabled (import failed): {exc}")
        return

    os.environ["WANDB_DIR"] = args.wandb_dir
    tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=(str(args.wandb_entity).strip() or None),
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        name=f"{setting_id}_aggregate",
        tags=tags + ["aggregate", "ensemble"],
        config={
            "setting_id": setting_id,
            "policy": str(args.policy).strip().upper(),
            "model": args.model,
            "pool": args.pool,
        },
        reinit=True,
    )
    try:
        for r in records:
            method = str(r["method"])
            run.log(
                {
                    f"{method}/accuracy": _to_float(r["accuracy"]),
                    f"{method}/f1": _to_float(r["f1"]),
                    f"{method}/auroc": _to_float(r["auroc"]),
                    f"{method}/precision": _to_float(r["precision"]),
                    f"{method}/recall": _to_float(r["recall"]),
                    f"{method}/calibration_brier": _to_float(r["calibration_brier"]),
                    f"{method}/calibration_quality": _to_float(r["calibration_quality"]),
                    f"{method}/calibration_driver_error": _to_float(r["calibration_driver_error"]),
                    f"{method}/calibration_non_driver_error": _to_float(r["calibration_non_driver_error"]),
                    f"{method}/calibration_random_brier": _to_float(r["calibration_random_brier"]),
                    f"{method}/calibration_brier_improvement_vs_random": _to_float(
                        r["calibration_brier_improvement_vs_random"]
                    ),
                    f"{method}/benchmark_accuracy": _to_float(r.get("benchmark_accuracy")),
                    f"{method}/benchmark_f1": _to_float(r.get("benchmark_f1")),
                    f"{method}/benchmark_auroc": _to_float(r.get("benchmark_auroc")),
                    f"{method}/benchmark_precision": _to_float(r.get("benchmark_precision")),
                    f"{method}/benchmark_recall": _to_float(r.get("benchmark_recall")),
                    f"{method}/benchmark_calibration_brier": _to_float(r.get("benchmark_calibration_brier")),
                    f"{method}/benchmark_calibration_quality": _to_float(
                        r.get("benchmark_calibration_quality")
                    ),
                    f"{method}/benchmark_calibration_brier_improvement_vs_random": _to_float(
                        r.get("benchmark_calibration_brier_improvement_vs_random")
                    ),
                }
            )

        for method, out in method_outputs.items():
            for png in sorted(out["method_dir"].glob("*.png")):
                run.log({f"plots/{method}/{png.stem}": wandb.Image(str(png))})
    finally:
        run.finish()


def _evaluate_method(
    method_name: str,
    probs_test: np.ndarray,
    y_test: np.ndarray,
    meta_train: dict,
    meta_test: dict,
    out_root: Path,
    pool: str,
    probs_benchmark: np.ndarray | None = None,
    y_benchmark: np.ndarray | None = None,
) -> dict[str, Any]:
    method_dir = out_root / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    metrics = _metrics_from_probs(y_test, probs_test)
    y_rand, _ = _compute_random_baseline(y_test, seed=42)
    calibration = _compute_calibration_metrics(y_test, probs_test, y_rand.astype(float))

    t_like = {
        "preds": metrics["preds"],
        "probs": metrics["probs"],
        "labels": metrics["labels"],
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "auroc": metrics["auroc"],
        "prec": metrics["prec"],
        "rec": metrics["rec"],
    }
    train_fusions = _extract_fusion_name_set(meta_train)
    analysis = analyze_test_predictions(
        meta_test,
        t_like,
        method_dir,
        train_fusion_names=train_fusions,
        meta_train=meta_train,
    )

    cm_model = confusion_matrix(y_test, metrics["preds"], labels=[0, 1])
    _plot_confusion(cm_model, f"{method_name} - test confusion", method_dir / "cm_test_model.png")
    _plot_confusion(
        analysis["baseline_random_confusion_matrix"],
        f"{method_name} - test random baseline confusion",
        method_dir / "cm_test_random.png",
    )
    _plot_confusion(
        analysis["baseline_gene_confusion_matrix"],
        f"{method_name} - test gene baseline confusion",
        method_dir / "cm_test_gene.png",
    )

    summary_txt = method_dir / "summary.txt"
    lines = [
        f"SETTING: {out_root.name}",
        f"METHOD: {method_name}",
        "",
        format_test_results(t_like, model_name=method_name, pool=pool, title="TEST RESULTS"),
        "",
        "Calibration:",
        f"  brier={calibration['brier']:.6f}",
        f"  quality_1_minus_brier={calibration['calibration_quality']:.6f}",
        f"  driver_error_mean_1_minus_p={calibration['driver_error']:.6f}",
        f"  non_driver_error_mean_p={calibration['non_driver_error']:.6f}",
        f"  random_brier={calibration['random_brier']:.6f}",
        f"  brier_improvement_vs_random={calibration['brier_improvement_vs_random']:.6f}",
        "",
        f"Prediction analysis folder: {method_dir}",
    ]
    benchmark_metrics = None
    benchmark_calibration = None
    if probs_benchmark is not None and y_benchmark is not None:
        bm_metrics = _metrics_from_probs(y_benchmark, probs_benchmark)
        y_bm_rand, _ = _compute_random_baseline(y_benchmark, seed=42)
        bm_cal = _compute_calibration_metrics(y_benchmark, probs_benchmark, y_bm_rand.astype(float))
        _plot_confusion(
            confusion_matrix(y_benchmark, bm_metrics["preds"], labels=[0, 1]),
            f"{method_name} - benchmark confusion",
            method_dir / "cm_benchmark_model.png",
        )
        lines.extend(
            [
                "",
                format_test_results(
                    bm_metrics, model_name=f"{method_name} [benchmark]", pool=pool, title="BENCHMARK RESULTS"
                ),
                "",
                "Benchmark calibration:",
                f"  brier={bm_cal['brier']:.6f}",
                f"  quality_1_minus_brier={bm_cal['calibration_quality']:.6f}",
                f"  driver_error_mean_1_minus_p={bm_cal['driver_error']:.6f}",
                f"  non_driver_error_mean_p={bm_cal['non_driver_error']:.6f}",
                f"  random_brier={bm_cal['random_brier']:.6f}",
                f"  brier_improvement_vs_random={bm_cal['brier_improvement_vs_random']:.6f}",
            ]
        )
        benchmark_metrics = bm_metrics
        benchmark_calibration = bm_cal
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "method": method_name,
        "metrics": metrics,
        "calibration": calibration,
        "benchmark_metrics": benchmark_metrics,
        "benchmark_calibration": benchmark_calibration,
        "summary_txt": summary_txt,
        "method_dir": method_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate trained probes into ensemble baseline.")
    parser.add_argument("--embeddings-dir", required=True, help="Root embedding directory containing <model>/*.pt")
    parser.add_argument("--model", required=True, choices=["esmc", "fuson"])
    parser.add_argument("--pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--checkpoint", dest="checkpoints", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--setting-id", default="")
    parser.add_argument("--policy", default="", help="Policy label (e.g. A/B/C/D) for W&B config.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-mode", default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb-project", default="driver-fusions-policy-comparison")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-dir", default="/work/H2020DeciderFicarra/gcapitani")
    args = parser.parse_args()

    ckpts = [Path(p).resolve() for p in args.checkpoints if str(p).strip()]
    ckpts = [p for p in ckpts if p.exists()]
    if not ckpts:
        raise FileNotFoundError("No valid checkpoint paths provided for aggregation.")

    setting_id = args.setting_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not str(args.policy).strip():
        inferred = setting_id.split("_", 1)[0].strip().upper()
        if inferred in {"A", "B", "C", "D"}:
            args.policy = inferred
    out_root = Path(args.output_dir).resolve() / "aggregates" / setting_id
    out_root.mkdir(parents=True, exist_ok=True)
    result_json = out_root / "aggregate_results.json"
    if args.skip_existing and result_json.exists():
        print(f"Reusing existing aggregate results: {result_json}")
        return

    emb_dir = Path(args.embeddings_dir) / args.model
    X_test, y_test, meta_test = load_embeddings(emb_dir, "test", args.pool)
    _, _, meta_train = load_embeddings(emb_dir, "train", args.pool)
    y_test_np = y_test.cpu().numpy().astype(int)
    bm_available = (emb_dir / "benchmark.pt").exists()
    X_bm = y_bm_np = None
    if bm_available:
        X_bm_t, y_bm_t, _ = load_embeddings(emb_dir, "benchmark", args.pool)
        X_bm = X_bm_t
        y_bm_np = y_bm_t.cpu().numpy().astype(int)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    probs_by_ckpt: list[np.ndarray] = []
    probs_bm_by_ckpt: list[np.ndarray] = []
    for ckpt in ckpts:
        try:
            probs_test = _predict_probs(ckpt, X_test, device)
            probs_bm = _predict_probs(ckpt, X_bm, device) if X_bm is not None else None
            probs_by_ckpt.append(probs_test)
            if X_bm is not None:
                probs_bm_by_ckpt.append(probs_bm)
        except Exception as exc:
            print(f"[WARNING] Skipping checkpoint due to load/predict failure: {ckpt} ({exc})")

    if not probs_by_ckpt:
        raise RuntimeError("All checkpoints failed during aggregation; no valid model available.")

    ensemble_probs = np.mean(np.stack(probs_by_ckpt, axis=0), axis=0)
    ensemble_bm_probs = np.mean(np.stack(probs_bm_by_ckpt, axis=0), axis=0) if probs_bm_by_ckpt else None
    ensemble_out = _evaluate_method(
        "ensemble",
        ensemble_probs,
        y_test_np,
        meta_train,
        meta_test,
        out_root,
        pool=args.pool,
        probs_benchmark=ensemble_bm_probs,
        y_benchmark=y_bm_np,
    )

    summary_rows: list[dict[str, Any]] = []
    method_outputs = {"ensemble": ensemble_out}
    for out in (ensemble_out,):
        m = out["metrics"]
        c = out["calibration"]
        bm = out["benchmark_metrics"] or {}
        bmc = out["benchmark_calibration"] or {}
        method = str(out["method"])
        row = {
            "method": method,
            "run_id": f"{setting_id}_{method}",
            "accuracy": m["acc"],
            "f1": m["f1"],
            "auroc": m["auroc"],
            "precision": m["prec"],
            "recall": m["rec"],
            "calibration_brier": c["brier"],
            "calibration_quality": c["calibration_quality"],
            "calibration_driver_error": c["driver_error"],
            "calibration_non_driver_error": c["non_driver_error"],
            "calibration_random_brier": c["random_brier"],
            "calibration_brier_improvement_vs_random": c["brier_improvement_vs_random"],
            "benchmark_accuracy": bm.get("acc", float("nan")),
            "benchmark_f1": bm.get("f1", float("nan")),
            "benchmark_auroc": bm.get("auroc", float("nan")),
            "benchmark_precision": bm.get("prec", float("nan")),
            "benchmark_recall": bm.get("rec", float("nan")),
            "benchmark_calibration_brier": bmc.get("brier", float("nan")),
            "benchmark_calibration_quality": bmc.get("calibration_quality", float("nan")),
            "benchmark_calibration_brier_improvement_vs_random": bmc.get(
                "brier_improvement_vs_random", float("nan")
            ),
            "summary_path": str(out["summary_txt"]),
            "analysis_dir": str(out["method_dir"]),
            "probe_best_ckpt": "",
            "n_ensemble_members": int(len(probs_by_ckpt)),
        }
        summary_rows.append(row)

    (out_root / "aggregate_manifest.json").write_text(
        json.dumps(
            {
                "setting_id": setting_id,
                "n_checkpoints_input": len(ckpts),
                "n_checkpoints_used": len(probs_by_ckpt),
                "output_root": str(out_root),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _wandb_log_all(args, setting_id=setting_id, records=summary_rows, method_outputs=method_outputs)
    print(f"Saved aggregate results to {result_json}")


if __name__ == "__main__":
    main()
