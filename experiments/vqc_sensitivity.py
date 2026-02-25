"""Stage 6a.5 — VQC hyperparameter sensitivity analysis.

This module measures how VQC classification accuracy varies with learning
rate and number of variational layers across two encodings (angle and iqp)
and two datasets (moons and circles).

Hyperparameter grid
-------------------
- Learning rates: {0.001, 0.01, 0.05}
- Variational layers: {1, 2, 3}
- Encodings: angle (non-entangling), iqp (entangling), both with n_features=2
- Datasets: moons, circles
- 3 runs × 5-fold CV per grid cell (36 cells total, 540 fold evaluations)

Seed strategy
-------------
- **base_seed** = 42
- **CV fold seed**: ``base_seed + dataset_index`` (encoding-independent,
  identical folds for all encodings — required for valid paired tests)
- **VQC init seed**: ``base_seed + 9000 + grid_cell_index * 10000 + run_index``
  (9000 offset to avoid collision with Stage 6a seeds; 10000 cell spacing
  prevents intra-grid collision since max per-fold offset is 8002)
- **Per-fold init seed**: ``vqc_run_seed + fold_index * 2000``
  (same formula as ``ExperimentConfig.fold_init_seed``)

Usage
-----
As a CLI tool::

    python -m experiments.vqc_sensitivity
    python -m experiments.vqc_sensitivity --quick
    python -m experiments.vqc_sensitivity --output-dir path/to/results --log-level DEBUG

Programmatically::

    from experiments.vqc_sensitivity import run_sensitivity_analysis

    report = run_sensitivity_analysis(
        output_dir="experiments/results/raw/stage6a5_sensitivity",
    )

Dependencies
------------
This module reuses :class:`experiments.vqc.VQCClassifier` for training,
:func:`experiments.datasets.load_dataset` and
:func:`experiments.datasets.get_cv_folds` for data,
:class:`experiments.checkpoint.CheckpointManager` for crash-safe resume,
and :func:`experiments.statistical.compute_ci_metrics` for statistics.

PennyLane is imported lazily (inside functions) to keep module-level
imports lightweight.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from itertools import product
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0"

_DEFAULT_BASE_SEED = 42
_SEED_OFFSET = 9000  # Avoid collision with Stage 6a seeds

_LEARNING_RATES: list[float] = [0.001, 0.01, 0.05]
_N_VAR_LAYERS: list[int] = [1, 2, 3]
_ENCODING_CONFIGS: list[dict[str, Any]] = [
    {"name": "angle", "params": {"n_features": 2, "rotation": "Y"}},
    {"name": "iqp", "params": {"n_features": 2, "reps": 2}},
]
_DATASETS: list[str] = ["moons", "circles"]

_DEFAULT_N_RUNS = 3
_DEFAULT_N_FOLDS = 5
_DEFAULT_EPOCHS = 100

_QUICK_N_RUNS = 1
_QUICK_N_FOLDS = 3

# Dataset index mapping for deterministic CV fold seeds.
_DATASET_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(_DATASETS)}


# ---------------------------------------------------------------------------
# Grid cell definition
# ---------------------------------------------------------------------------


def _build_grid(
    learning_rates: list[float] | None = None,
    n_var_layers: list[int] | None = None,
    encoding_configs: list[dict[str, Any]] | None = None,
    datasets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the full hyperparameter grid as a list of cell specifications.

    Parameters
    ----------
    learning_rates : list[float] or None
        Learning rates to sweep.  Defaults to ``[0.001, 0.01, 0.05]``.
    n_var_layers : list[int] or None
        Layer counts to sweep.  Defaults to ``[1, 2, 3]``.
    encoding_configs : list[dict] or None
        Encoding specifications.  Defaults to angle + iqp, n_features=2.
    datasets : list[str] or None
        Dataset names to evaluate.  Defaults to ``["moons", "circles"]``.

    Returns
    -------
    list[dict[str, Any]]
        Each dict has keys: ``encoding_name``, ``encoding_params``,
        ``dataset``, ``lr``, ``n_var_layers``, ``cell_index``.
    """
    lrs = learning_rates if learning_rates is not None else _LEARNING_RATES
    layers = n_var_layers if n_var_layers is not None else _N_VAR_LAYERS
    encs = encoding_configs if encoding_configs is not None else _ENCODING_CONFIGS
    dss = datasets if datasets is not None else _DATASETS

    grid: list[dict[str, Any]] = []
    cell_index = 0

    for enc_cfg in encs:
        for ds in dss:
            for lr in lrs:
                for n_layers in layers:
                    grid.append(
                        {
                            "encoding_name": enc_cfg["name"],
                            "encoding_params": enc_cfg["params"],
                            "dataset": ds,
                            "lr": lr,
                            "n_var_layers": n_layers,
                            "cell_index": cell_index,
                        }
                    )
                    cell_index += 1

    return grid


def _cell_id(cell: dict[str, Any]) -> str:
    """Build a deterministic checkpoint ID for a grid cell.

    Format: ``sensitivity/<encoding>/<dataset>/lr<lr>_layers<n_var_layers>``

    Parameters
    ----------
    cell : dict[str, Any]
        Grid cell specification from :func:`_build_grid`.

    Returns
    -------
    str
        Unique, deterministic checkpoint identifier.
    """
    return (
        f"sensitivity/{cell['encoding_name']}/{cell['dataset']}"
        f"/lr{cell['lr']}_layers{cell['n_var_layers']}"
    )


# ---------------------------------------------------------------------------
# Seed computation
# ---------------------------------------------------------------------------


def _cv_fold_seed(base_seed: int, dataset_name: str) -> int:
    """Compute the CV fold seed for a dataset.

    This must be encoding-independent so all encodings see the same
    train/test splits, enabling valid paired statistical tests.

    Parameters
    ----------
    base_seed : int
        Base seed (default 42).
    dataset_name : str
        Dataset name.

    Returns
    -------
    int
        Deterministic fold seed.
    """
    dataset_idx = _DATASET_INDEX.get(dataset_name, 0)
    return base_seed + dataset_idx


def _vqc_run_seed(base_seed: int, cell_index: int, run_index: int) -> int:
    """Compute the VQC weight-initialisation seed for a specific run.

    Uses a 9000 offset from the base seed to avoid collision with
    Stage 6a seeds (which use ``stage_seed + encoding_index * 100``).

    The cell spacing of 10000 ensures no seed collision between cells:
    the maximum per-fold offset within a cell is
    ``(n_folds - 1) * 2000 + (n_runs - 1)`` = 8002 for 5 folds / 3 runs,
    which is safely below 10000.

    Parameters
    ----------
    base_seed : int
        Base seed.
    cell_index : int
        Flat index of the grid cell (0-35 for the full grid).
    run_index : int
        Run number within the grid cell (0-based).

    Returns
    -------
    int
        Deterministic VQC init seed.
    """
    return base_seed + _SEED_OFFSET + cell_index * 10000 + run_index


def _fold_init_seed(run_seed: int, fold_index: int) -> int:
    """Compute the per-fold VQC initialisation seed.

    Same formula as ``ExperimentConfig.fold_init_seed``.

    Parameters
    ----------
    run_seed : int
        The run-level seed from :func:`_vqc_run_seed`.
    fold_index : int
        Fold number (0-based).

    Returns
    -------
    int
        Per-fold init seed.
    """
    return run_seed + fold_index * 2000


# ---------------------------------------------------------------------------
# Single grid-cell evaluation
# ---------------------------------------------------------------------------


def _evaluate_cell(
    cell: dict[str, Any],
    *,
    n_runs: int,
    n_folds: int,
    epochs: int,
    base_seed: int,
) -> dict[str, Any]:
    """Train and evaluate VQC for a single grid cell.

    Parameters
    ----------
    cell : dict[str, Any]
        Grid cell specification (encoding, dataset, lr, n_var_layers).
    n_runs : int
        Number of independent runs per cell.
    n_folds : int
        Number of CV folds per run.
    epochs : int
        Training epochs per VQC instance.
    base_seed : int
        Base seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Cell result with per-run/fold details and aggregate statistics.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    from encoding_atlas import get_encoding
    from experiments.datasets import get_cv_folds, load_dataset
    from experiments.statistical import compute_ci_metrics
    from experiments.vqc import VQCClassifier

    encoding = get_encoding(cell["encoding_name"], **cell["encoding_params"])
    dataset_name = cell["dataset"]
    lr = cell["lr"]
    n_var_layers = cell["n_var_layers"]
    cell_index = cell["cell_index"]

    cv_seed = _cv_fold_seed(base_seed, dataset_name)

    # Load dataset with encoding-independent seed.
    X, y = load_dataset(dataset_name, seed=cv_seed)

    t_cell_start = time.monotonic()

    all_test_acc: list[float] = []
    all_train_acc: list[float] = []
    all_final_loss: list[float] = []
    all_epochs_to_converge: list[int] = []
    all_precisions: list[float] = []
    all_recalls: list[float] = []
    all_f1s: list[float] = []
    runs: list[dict[str, Any]] = []

    for run_idx in range(n_runs):
        folds = get_cv_folds(X, y, n_folds=n_folds, seed=cv_seed)
        run_seed = _vqc_run_seed(base_seed, cell_index, run_idx)
        fold_results: list[dict[str, Any]] = []

        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
            vqc_seed = _fold_init_seed(run_seed, fold_idx)
            t_fold = time.monotonic()

            try:
                vqc = VQCClassifier(
                    encoding=encoding,
                    n_var_layers=n_var_layers,
                    lr=lr,
                    epochs=epochs,
                    seed=vqc_seed,
                )
                vqc.fit(X_train, y_train)
                y_pred = vqc.predict(X_test)
                y_pred_train = vqc.predict(X_train)

                fold_wall = round(time.monotonic() - t_fold, 3)

                acc = float(np.mean(y_pred == y_test))
                train_acc = float(np.mean(y_pred_train == y_train))
                prec = float(precision_score(y_test, y_pred, zero_division=0))
                rec = float(recall_score(y_test, y_pred, zero_division=0))
                f1 = float(f1_score(y_test, y_pred, zero_division=0))
                final_loss = vqc.get_final_loss()
                n_epochs_trained = len(vqc.loss_history_)

                fold_results.append(
                    {
                        "fold": fold_idx,
                        "test_accuracy": acc,
                        "train_accuracy": train_acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "final_loss": final_loss,
                        "n_epochs_trained": n_epochs_trained,
                        "wall_time_seconds": fold_wall,
                        "status": vqc.status_,
                    }
                )

                all_test_acc.append(acc)
                all_train_acc.append(train_acc)
                all_precisions.append(prec)
                all_recalls.append(rec)
                all_f1s.append(f1)
                if final_loss is not None:
                    all_final_loss.append(final_loss)
                all_epochs_to_converge.append(n_epochs_trained)

            except Exception as exc:
                fold_wall = round(time.monotonic() - t_fold, 3)
                fold_results.append(
                    {
                        "fold": fold_idx,
                        "status": "failed",
                        "error": str(exc),
                        "wall_time_seconds": fold_wall,
                    }
                )
                logger.error(
                    "Fold %d failed for %s/%s/lr%s_layers%d run%d: %s",
                    fold_idx,
                    cell["encoding_name"],
                    dataset_name,
                    lr,
                    n_var_layers,
                    run_idx,
                    exc,
                )

        run_accs = [
            f["test_accuracy"]
            for f in fold_results
            if f.get("status") != "failed" and "test_accuracy" in f
        ]
        runs.append(
            {
                "run": run_idx,
                "vqc_run_seed": run_seed,
                "folds": fold_results,
                "mean_accuracy": float(np.mean(run_accs)) if run_accs else None,
            }
        )

    cell_wall = round(time.monotonic() - t_cell_start, 3)

    # Aggregate statistics.
    result: dict[str, Any] = {
        "encoding": cell["encoding_name"],
        "dataset": dataset_name,
        "lr": lr,
        "n_var_layers": n_var_layers,
        "n_observations": len(all_test_acc),
        "runs": runs,
        "wall_time_seconds": cell_wall,
    }

    if all_test_acc:
        acc_array = np.array(all_test_acc)
        ci_metrics = compute_ci_metrics(acc_array, seed=base_seed)
        result["mean_accuracy"] = ci_metrics["mean"]
        result["std_accuracy"] = ci_metrics["std"]
        result["ci_lower"] = ci_metrics["ci_lower"]
        result["ci_upper"] = ci_metrics["ci_upper"]
        result["mean_train_accuracy"] = float(np.mean(all_train_acc))
        result["mean_precision"] = float(np.mean(all_precisions))
        result["mean_recall"] = float(np.mean(all_recalls))
        result["mean_f1"] = float(np.mean(all_f1s))
        result["mean_final_loss"] = (
            float(np.mean(all_final_loss)) if all_final_loss else None
        )
        result["mean_epochs_to_converge"] = (
            float(np.mean(all_epochs_to_converge)) if all_epochs_to_converge else None
        )
        result["status"] = "success"
    else:
        result["mean_accuracy"] = None
        result["std_accuracy"] = None
        result["ci_lower"] = None
        result["ci_upper"] = None
        result["mean_final_loss"] = None
        result["mean_epochs_to_converge"] = None
        result["status"] = "failed"
        result["reason"] = "No successful folds"

    return result


# ---------------------------------------------------------------------------
# Analysis summary
# ---------------------------------------------------------------------------


def _compute_analysis(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive sensitivity analysis summary from raw cell results.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of per-cell result dicts (from :func:`_evaluate_cell`).

    Returns
    -------
    dict[str, Any]
        Analysis dict with keys: ``best_config_per_encoding_dataset``,
        ``lr_sensitivity``, ``layer_sensitivity``, ``default_config_rank``.
    """
    # Group results by (encoding, dataset).
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if r.get("mean_accuracy") is None:
            continue
        key = f"{r['encoding']}/{r['dataset']}"
        groups.setdefault(key, []).append(r)

    best_config: dict[str, Any] = {}
    lr_sensitivity: dict[str, dict[str, float]] = {}
    layer_sensitivity: dict[str, dict[str, float]] = {}
    default_config_rank: dict[str, Any] = {}

    for key, group in groups.items():
        # Best configuration.
        best = max(group, key=lambda r: r["mean_accuracy"])
        best_config[key] = {
            "lr": best["lr"],
            "n_var_layers": best["n_var_layers"],
            "mean_accuracy": best["mean_accuracy"],
            "ci_lower": best.get("ci_lower"),
            "ci_upper": best.get("ci_upper"),
        }

        # LR sensitivity: mean accuracy per learning rate (averaged over layers).
        lr_accs: dict[str, list[float]] = {}
        for r in group:
            lr_key = str(r["lr"])
            lr_accs.setdefault(lr_key, []).append(r["mean_accuracy"])
        lr_sensitivity[key] = {
            lr_val: round(float(np.mean(accs)), 6)
            for lr_val, accs in sorted(lr_accs.items())
        }

        # Layer sensitivity: mean accuracy per layer count (averaged over LRs).
        layer_accs: dict[str, list[float]] = {}
        for r in group:
            layer_key = str(r["n_var_layers"])
            layer_accs.setdefault(layer_key, []).append(r["mean_accuracy"])
        layer_sensitivity[key] = {
            lv: round(float(np.mean(accs)), 6)
            for lv, accs in sorted(layer_accs.items())
        }

        # Default config rank (lr=0.01, n_var_layers=2).
        sorted_group = sorted(
            group,
            key=lambda r: r["mean_accuracy"],
            reverse=True,
        )
        for rank, r in enumerate(sorted_group, start=1):
            if r["lr"] == 0.01 and r["n_var_layers"] == 2:
                default_config_rank[key] = {
                    "rank": rank,
                    "total_configs": len(sorted_group),
                    "mean_accuracy": r["mean_accuracy"],
                }
                break
        else:
            # Default config not in grid (shouldn't happen with standard grid).
            default_config_rank[key] = {
                "rank": None,
                "total_configs": len(sorted_group),
            }

    return {
        "best_config_per_encoding_dataset": best_config,
        "lr_sensitivity": lr_sensitivity,
        "layer_sensitivity": layer_sensitivity,
        "default_config_rank": default_config_rank,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_sensitivity_analysis(
    *,
    output_dir: str = "experiments/results/raw/stage6a5_sensitivity",
    base_seed: int = _DEFAULT_BASE_SEED,
    n_runs: int = _DEFAULT_N_RUNS,
    n_folds: int = _DEFAULT_N_FOLDS,
    epochs: int = _DEFAULT_EPOCHS,
    quick: bool = False,
    learning_rates: list[float] | None = None,
    n_var_layers: list[int] | None = None,
    encoding_configs: list[dict[str, Any]] | None = None,
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full VQC hyperparameter sensitivity analysis.

    Iterates over the hyperparameter grid, training VQC models with
    cross-validation, and produces a JSON report with per-cell results
    and sensitivity analysis.

    Parameters
    ----------
    output_dir : str
        Directory for output JSON and checkpoints.
    base_seed : int
        Base seed (default 42).
    n_runs : int
        Number of independent runs per grid cell (default 3).
    n_folds : int
        Number of CV folds per run (default 5).
    epochs : int
        Training epochs (default 100).
    quick : bool
        If ``True``, reduce to 1 run × 3 folds for fast iteration.
    learning_rates : list[float] or None
        Override default learning rates.
    n_var_layers : list[int] or None
        Override default layer counts.
    encoding_configs : list[dict] or None
        Override default encoding configurations.
    datasets : list[str] or None
        Override default datasets.

    Returns
    -------
    dict[str, Any]
        Full sensitivity report (JSON-serializable).
    """
    from experiments.checkpoint import CheckpointManager

    if quick:
        n_runs = _QUICK_N_RUNS
        n_folds = _QUICK_N_FOLDS
        logger.info("Quick mode: %d run(s) × %d folds", n_runs, n_folds)

    grid = _build_grid(
        learning_rates=learning_rates,
        n_var_layers=n_var_layers,
        encoding_configs=encoding_configs,
        datasets=datasets,
    )
    total_cells = len(grid)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint = CheckpointManager(checkpoint_dir)

    logger.info(
        "Starting sensitivity analysis: %d grid cells, %d runs × %d folds",
        total_cells,
        n_runs,
        n_folds,
    )

    t_start = time.monotonic()
    results: list[dict[str, Any]] = []
    completed = 0
    skipped = 0
    failed = 0

    for idx, cell in enumerate(grid):
        cid = _cell_id(cell)
        label = (
            f"{cell['encoding_name']}/{cell['dataset']}"
            f"/lr{cell['lr']}_layers{cell['n_var_layers']}"
        )

        # Check for cached result.
        if checkpoint.is_completed(cid):
            cached = checkpoint.get_result(cid)
            if cached is not None:
                results.append(cached)
                skipped += 1
                logger.debug("Skipping completed cell: %s", label)
                print(
                    f"[SKIP] ({idx + 1:>2}/{total_cells}) {label}",
                    flush=True,
                )
                continue

        print(
            f"[RUN ] ({idx + 1:>2}/{total_cells}) {label}",
            flush=True,
        )

        try:
            cell_result = _evaluate_cell(
                cell,
                n_runs=n_runs,
                n_folds=n_folds,
                epochs=epochs,
                base_seed=base_seed,
            )
            checkpoint.mark_completed(cid, cell_result)
            results.append(cell_result)

            if cell_result.get("status") == "success":
                completed += 1
                acc = cell_result.get("mean_accuracy", 0)
                print(
                    f"[DONE] ({idx + 1:>2}/{total_cells}) {label} " f"— acc={acc:.4f}",
                    flush=True,
                )
            else:
                failed += 1
                print(
                    f"[FAIL] ({idx + 1:>2}/{total_cells}) {label} "
                    f"— {cell_result.get('reason', 'unknown')}",
                    flush=True,
                )

        except Exception as exc:
            failed += 1
            error_result: dict[str, Any] = {
                "encoding": cell["encoding_name"],
                "dataset": cell["dataset"],
                "lr": cell["lr"],
                "n_var_layers": cell["n_var_layers"],
                "status": "failed",
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            }
            results.append(error_result)
            logger.error("Cell %s failed: %s", label, exc)
            print(
                f"[FAIL] ({idx + 1:>2}/{total_cells}) {label} — {exc}",
                flush=True,
            )

    wall_time = round(time.monotonic() - t_start, 3)

    # Build analysis summary.
    analysis = _compute_analysis(results)

    # Strip per-run detail from the top-level results to keep the summary
    # concise; full run data is preserved in checkpoints.
    summary_results: list[dict[str, Any]] = []
    for r in results:
        summary = {k: v for k, v in r.items() if k != "runs"}
        summary_results.append(summary)

    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "grid": {
            "learning_rates": (
                learning_rates if learning_rates is not None else _LEARNING_RATES
            ),
            "n_var_layers": (
                n_var_layers if n_var_layers is not None else _N_VAR_LAYERS
            ),
            "encodings": [
                cfg["name"]
                for cfg in (
                    encoding_configs
                    if encoding_configs is not None
                    else _ENCODING_CONFIGS
                )
            ],
            "datasets": datasets if datasets is not None else _DATASETS,
            "n_runs": n_runs,
            "n_folds": n_folds,
            "epochs": epochs,
        },
        "results": summary_results,
        "analysis": analysis,
        "summary": {
            "total_cells": total_cells,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "wall_time_seconds": wall_time,
        },
    }

    # Save report.
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "sensitivity_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)

    logger.info(
        "Sensitivity analysis complete: %d completed, %d skipped, %d failed "
        "(%.1fs wall time)",
        completed,
        skipped,
        failed,
        wall_time,
    )
    print(f"\nReport saved to: {report_path}", flush=True)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for VQC hyperparameter sensitivity analysis.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  If ``None``, reads from ``sys.argv``.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on partial failure, 2 on fatal error.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="vqc_sensitivity",
        description=(
            "Stage 6a.5: Measure VQC classification accuracy sensitivity "
            "to learning rate and number of variational layers."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/raw/stage6a5_sensitivity",
        help="Output directory for results and checkpoints.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 run × 3 folds per cell (instead of 3 × 5).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        report = run_sensitivity_analysis(
            output_dir=args.output_dir,
            quick=args.quick,
        )

        summary = report.get("summary", {})
        analysis = report.get("analysis", {})
        best_configs = analysis.get("best_config_per_encoding_dataset", {})

        print("\nSensitivity analysis complete.")
        print(
            f"  Cells: {summary.get('completed', 0)} completed, "
            f"{summary.get('skipped', 0)} skipped, "
            f"{summary.get('failed', 0)} failed"
        )
        print(f"  Wall time: {summary.get('wall_time_seconds', 0):.1f}s")

        if best_configs:
            print("\n  Best configurations:")
            for key, cfg in sorted(best_configs.items()):
                print(
                    f"    {key}: lr={cfg['lr']}, "
                    f"layers={cfg['n_var_layers']}, "
                    f"acc={cfg['mean_accuracy']:.4f}"
                )

        default_ranks = analysis.get("default_config_rank", {})
        if default_ranks:
            print("\n  Default config (lr=0.01, layers=2) rank:")
            for key, info in sorted(default_ranks.items()):
                rank = info.get("rank")
                total = info.get("total_configs")
                if rank is not None:
                    print(f"    {key}: #{rank}/{total}")

        if summary.get("failed", 0) > 0:
            return 1
        return 0

    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        return 2


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for numpy types."""
    type_name = type(obj).__name__
    if "int" in type_name and hasattr(obj, "item"):
        return int(obj.item())
    if "float" in type_name and hasattr(obj, "item"):
        return float(obj.item())
    if "ndarray" in type_name and hasattr(obj, "tolist"):
        return obj.tolist()
    if "bool" in type_name and hasattr(obj, "item"):
        return bool(obj.item())
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    sys.exit(main())
