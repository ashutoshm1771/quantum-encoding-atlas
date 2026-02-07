"""Unified experiment runner with automatic dispatch and checkpointing.

This module provides the :class:`ExperimentRunner` which:

1. Reads a stage configuration (via :mod:`experiments.config`).
2. Builds a task list by expanding encoding parameter sweeps.
3. Skips tasks that are already checkpointed (unless ``resume=False``).
4. Dispatches each task to the appropriate ``encoding_atlas.analysis``
   function based on the stage name.
5. Persists results via :class:`experiments.checkpoint.CheckpointManager`.
6. Reports progress to stdout and the Python logger.

Supported Stages
----------------
- ``resources``: Calls :func:`count_resources` and :func:`estimate_execution_time`.
- ``simulability``: Calls :func:`check_simulability`.
- ``expressibility``: Calls :func:`compute_expressibility`.
- ``entanglement``: Calls :func:`compute_entanglement_capability`.
- ``trainability``: Calls :func:`estimate_trainability`.
- ``noise``: Computes ideal vs noisy expressibility, entanglement, and fidelity.
- ``vqc``: Trains VQC classifiers and runs classical baselines (SVM-RBF,
  Random Forest, MLP) for comparison. All methods share the same
  encoding-independent CV fold seed (``config.cv_fold_seed``) to ensure
  identical train/test splits for valid paired statistical comparison.
- ``kernel``: Computes quantum kernel matrices and evaluates kernel-SVM
  classification.

Usage
-----
::

    from experiments.runner import ExperimentRunner

    runner = ExperimentRunner("experiments/configs/stage1_resources.json")
    summary = runner.run()
    print(summary)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from typing import Any

from experiments.checkpoint import CheckpointManager
from experiments.config import ExperimentConfig, config_to_dict, load_config
from experiments.results import ResultStore, TaskResult

logger = logging.getLogger(__name__)

# Maximum number of qubits before we skip simulation-heavy stages.
# Statevector simulation uses 2^n complex numbers × 16 bytes.
_MAX_SIMULATION_QUBITS = 10


class ExperimentRunner:
    """Run a full experiment stage with checkpointing and error handling.

    Parameters
    ----------
    config_path : str
        Path to a JSON configuration file.
    resume : bool, optional
        If ``True`` (default), skip tasks that already appear in the
        checkpoint log. Set to ``False`` to force re-execution.
    quick : bool, optional
        If ``True``, apply quick-mode overrides to reduce sample
        counts and sweep sizes (default ``False``).
    smoke : bool, optional
        If ``True``, apply quick-mode overrides **plus** cap dataset
        sizes to 20 samples for fast end-to-end validation
        (default ``False``). Mutually exclusive with ``quick``.

    Attributes
    ----------
    config : ExperimentConfig
        The loaded and expanded configuration.
    checkpoint : CheckpointManager
        Checkpoint manager for this run.
    store : ResultStore
        In-memory result store, populated as tasks complete.
    """

    def __init__(
        self,
        config_path: str,
        *,
        resume: bool = True,
        quick: bool = False,
        smoke: bool = False,
    ) -> None:
        self.config: ExperimentConfig = load_config(
            config_path, quick=quick, smoke=smoke,
        )

        # Checkpoints live inside the output directory.
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")
        self.checkpoint: CheckpointManager = CheckpointManager(ckpt_dir)
        self.resume: bool = resume
        self.store: ResultStore = ResultStore()

        # Save the effective config for reproducibility.
        os.makedirs(self.config.output_dir, exist_ok=True)
        effective_path = os.path.join(self.config.output_dir, "effective_config.json")
        import json
        with open(effective_path, "w", encoding="utf-8") as fh:
            json.dump(config_to_dict(self.config), fh, indent=2)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute all tasks in the stage.

        Iterates over every encoding spec in the config, dispatches
        to the appropriate analysis function, and checkpoints results.
        Tasks that fail are recorded with ``status="failed"`` and
        execution continues.

        Returns
        -------
        dict[str, Any]
            Summary with keys ``total``, ``completed``, ``skipped``,
            ``failed``, ``wall_time``.
        """
        stage = self.config.stage
        specs = self.config.encoding_specs
        total = len(specs)

        _print_header(stage, total, self.resume)

        completed = 0
        skipped = 0
        failed = 0
        t_start = time.monotonic()

        for idx, spec in enumerate(specs):
            exp_id = self.checkpoint.get_experiment_id(
                stage=stage,
                encoding_name=spec.name,
                encoding_params=spec.params,
                metric=stage,
            )

            # Resume: skip already-completed tasks.
            if self.resume and self.checkpoint.is_completed(exp_id):
                cached = self.checkpoint.get_result(exp_id)
                if cached is not None:
                    task = TaskResult(
                        experiment_id=exp_id,
                        stage=stage,
                        encoding_name=spec.name,
                        encoding_params=spec.params,
                        metric=stage,
                        result=cached,
                        status=cached.get("status", "success"),
                        wall_time=cached.get("wall_time", 0.0),
                    )
                    self.store.add(task)
                skipped += 1
                _print_progress("SKIP", exp_id, idx + 1, total)
                continue

            # Execute the task.
            _print_progress("RUN ", exp_id, idx + 1, total)
            t_task = time.monotonic()

            try:
                result = self._execute_task(spec, idx)
                result["status"] = "success"
            except _SkipTask as exc:
                result = {
                    "status": "skipped",
                    "reason": str(exc),
                    "encoding_params": spec.params,
                }
                skipped += 1
                logger.warning("Task skipped: %s — %s", exp_id, exc)
                _print_progress("SKIP", exp_id, idx + 1, total, note=str(exc))
            except Exception as exc:
                result = {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "encoding_params": spec.params,
                }
                failed += 1
                logger.error("Task failed: %s — %s", exp_id, exc)
                _print_progress("FAIL", exp_id, idx + 1, total, note=str(exc))

            wall = time.monotonic() - t_task
            result["wall_time"] = round(wall, 3)
            result["encoding_params"] = spec.params

            # Persist and store.
            self.checkpoint.mark_completed(exp_id, result)
            task = TaskResult(
                experiment_id=exp_id,
                stage=stage,
                encoding_name=spec.name,
                encoding_params=spec.params,
                metric=stage,
                result=result,
                status=result["status"],
                wall_time=wall,
            )
            self.store.add(task)

            if result["status"] == "success":
                completed += 1

        # Run classical baselines for the VQC stage.
        baseline_counts = self._run_classical_baselines(
            completed, skipped, failed, total,
        )
        completed += baseline_counts["completed"]
        skipped += baseline_counts["skipped"]
        failed += baseline_counts["failed"]
        total += baseline_counts["total"]

        total_wall = time.monotonic() - t_start

        summary = {
            "stage": stage,
            "total": total,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "wall_time": round(total_wall, 2),
        }

        _print_footer(summary)

        # Save summary JSON.
        summary_path = os.path.join(self.config.output_dir, "summary.json")
        self.store.save_summary_json(summary_path)

        return summary

    # ------------------------------------------------------------------
    # Classical baselines
    # ------------------------------------------------------------------

    def _run_classical_baselines(
        self,
        completed: int,
        skipped: int,
        failed: int,
        current_total: int,
    ) -> dict[str, int]:
        """Run classical baselines for the VQC stage.

        Classical baselines (SVM-RBF, Random Forest, MLP) are run once
        per dataset using ``config.cv_fold_seed(dataset_idx)`` — the same
        encoding-independent fold seed used by ``_handle_vqc`` — enabling
        valid paired statistical comparison (Wilcoxon signed-rank).  Each
        baseline is checkpointed independently so it is not re-run on
        resume.

        This method is a no-op for non-VQC stages and when no baselines
        are configured.

        Parameters
        ----------
        completed, skipped, failed : int
            Running counters from the main task loop (used for progress
            display offsets).
        current_total : int
            Total task count so far (used for progress display).

        Returns
        -------
        dict[str, int]
            Counts with keys ``total``, ``completed``, ``skipped``,
            ``failed`` for baseline tasks only.
        """
        counts = {"total": 0, "completed": 0, "skipped": 0, "failed": 0}

        if self.config.stage != "vqc":
            return counts

        baseline_names: list[str] = self.config.analysis_params.get(
            "classical_baselines", [],
        )
        if not baseline_names:
            return counts

        counts["total"] = len(baseline_names)
        display_total = current_total + len(baseline_names)
        display_offset = current_total

        logger.info(
            "Running %d classical baseline(s): %s",
            len(baseline_names),
            baseline_names,
        )

        # Use stage_seed as the base seed (same as task_seed(0) minus 0).
        seed = self.config.stage_seed
        stage = self.config.stage

        for bl_idx, bl_name in enumerate(baseline_names):
            # Baselines use empty params — the classifier name is the
            # encoding_name in checkpoint terminology.
            exp_id = self.checkpoint.get_experiment_id(
                stage=stage,
                encoding_name=f"classical_{bl_name}",
                encoding_params={},
                metric=stage,
            )

            display_num = display_offset + bl_idx + 1

            # Resume: skip already-completed baselines.
            if self.resume and self.checkpoint.is_completed(exp_id):
                cached = self.checkpoint.get_result(exp_id)
                if cached is not None:
                    task = TaskResult(
                        experiment_id=exp_id,
                        stage=stage,
                        encoding_name=f"classical_{bl_name}",
                        encoding_params={},
                        metric=stage,
                        result=cached,
                        status=cached.get("status", "success"),
                        wall_time=cached.get("wall_time", 0.0),
                    )
                    self.store.add(task)
                counts["skipped"] += 1
                _print_progress("SKIP", exp_id, display_num, display_total)
                continue

            _print_progress("RUN ", exp_id, display_num, display_total)
            t_task = time.monotonic()

            try:
                result = _handle_classical_baseline(bl_name, self.config, seed)
                result["status"] = "success"
            except Exception as exc:
                result = {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                counts["failed"] += 1
                logger.error(
                    "Classical baseline failed: %s — %s", exp_id, exc,
                )
                _print_progress(
                    "FAIL", exp_id, display_num, display_total,
                    note=str(exc),
                )

            wall = time.monotonic() - t_task
            result["wall_time"] = round(wall, 3)

            self.checkpoint.mark_completed(exp_id, result)
            task = TaskResult(
                experiment_id=exp_id,
                stage=stage,
                encoding_name=f"classical_{bl_name}",
                encoding_params={},
                metric=stage,
                result=result,
                status=result["status"],
                wall_time=wall,
            )
            self.store.add(task)

            if result["status"] == "success":
                counts["completed"] += 1

        return counts

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    def _execute_task(self, spec: Any, task_index: int) -> dict[str, Any]:
        """Instantiate encoding and dispatch to the stage-specific handler.

        Parameters
        ----------
        spec : EncodingSpec
            The encoding name and parameters for this task.
        task_index : int
            Zero-based index used for deterministic seed computation.

        Returns
        -------
        dict[str, Any]
            Raw result dictionary from the analysis function.

        Raises
        ------
        _SkipTask
            If the task should be skipped (e.g. too many qubits).
        Exception
            Any exception from encoding instantiation or analysis.
        """
        # Stage 7 uses a dummy encoding — skip instantiation.
        if spec.name == "__tradeoff__":
            seed = self.config.task_seed(task_index)
            dispatch = _STAGE_HANDLERS.get(self.config.stage)
            if dispatch is None:
                raise ValueError(
                    f"No handler registered for stage '{self.config.stage}'. "
                    f"Available: {sorted(_STAGE_HANDLERS.keys())}"
                )
            return dispatch(None, self.config, seed)

        from encoding_atlas import get_encoding

        # Instantiate the encoding.
        try:
            encoding = get_encoding(spec.name, **spec.params)
        except Exception as exc:
            raise _SkipTask(
                f"Could not instantiate {spec.name}({spec.params}): {exc}"
            ) from exc

        # Guard: skip simulation-heavy stages for large qubit counts.
        stage = self.config.stage
        if stage in ("expressibility", "entanglement", "trainability"):
            if encoding.n_qubits > _MAX_SIMULATION_QUBITS:
                raise _SkipTask(
                    f"Skipping {spec.name} with {encoding.n_qubits} qubits "
                    f"(max {_MAX_SIMULATION_QUBITS} for {stage} stage)."
                )
        # Noise stage has stricter limit due to density matrix memory
        if stage == "noise":
            max_noisy_qubits = 8
            if encoding.n_qubits > max_noisy_qubits:
                raise _SkipTask(
                    f"Skipping {spec.name} with {encoding.n_qubits} qubits "
                    f"(max {max_noisy_qubits} for noise stage due to density matrix memory)."
                )

        seed = self.config.task_seed(task_index)

        dispatch = _STAGE_HANDLERS.get(stage)
        if dispatch is None:
            raise ValueError(
                f"No handler registered for stage '{stage}'. "
                f"Available: {sorted(_STAGE_HANDLERS.keys())}"
            )

        return dispatch(encoding, self.config, seed)


# ======================================================================
# Stage handlers
# ======================================================================
# Each handler receives (encoding, config, seed) and returns a dict.
# They import analysis functions locally to keep startup fast and to
# avoid import errors if optional backends are missing.
# ======================================================================

def _handle_resources(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: resource profiling (no simulation)."""
    from encoding_atlas.analysis import count_resources, estimate_execution_time
    from encoding_atlas.core.exceptions import ValidationError

    # Some encodings (e.g., BasisEncoding) are data-dependent.
    # For those, count_resources requires input data or we use properties.
    try:
        summary = count_resources(encoding)
    except (ValueError, ValidationError) as exc:
        if "data-dependent" in str(exc).lower():
            # Fall back to properties-based resource counting.
            props = encoding.properties
            summary = {
                "n_qubits": props.n_qubits,
                "depth": props.depth,
                "gate_count": props.gate_count,
                "single_qubit_gates": props.single_qubit_gates,
                "two_qubit_gates": props.two_qubit_gates,
                "parameter_count": props.parameter_count,
                "is_data_dependent": True,
                "note": "Worst-case estimates from properties (data-dependent encoding)",
            }
        else:
            raise

    timing = estimate_execution_time(encoding)

    # count_resources returns a TypedDict — convert to plain dict.
    result: dict[str, Any] = dict(summary)
    result["estimated_time_us"] = timing.get("estimated_time_us", timing)
    result["encoding_name"] = type(encoding).__name__
    return result


def _handle_simulability(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: simulability classification."""
    from encoding_atlas.analysis import check_simulability, is_clifford_circuit, is_matchgate_circuit

    sim_result = check_simulability(encoding, detailed=True)
    result: dict[str, Any] = dict(sim_result)
    result["is_clifford"] = is_clifford_circuit(encoding)
    result["is_matchgate"] = is_matchgate_circuit(encoding)
    result["encoding_name"] = type(encoding).__name__
    return result


def _handle_expressibility(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: expressibility via fidelity sampling."""
    from encoding_atlas.analysis import compute_expressibility

    ap = config.analysis_params
    expr_result = compute_expressibility(
        encoding,
        n_samples=ap.get("n_samples", 5000),
        n_bins=ap.get("n_bins", 75),
        input_range=tuple(ap.get("input_range", (0.0, 6.283185307179586))),
        seed=seed,
        backend=config.backend,
        return_distributions=True,
        verbose=False,
    )

    # expr_result is a TypedDict when return_distributions=True.
    result: dict[str, Any] = dict(expr_result)
    result["encoding_name"] = type(encoding).__name__

    # Remove large array fields to keep checkpoint files small.
    # The full distributions can be recomputed if needed.
    for key in ("fidelity_distribution", "haar_distribution", "bin_edges"):
        result.pop(key, None)

    return result


def _handle_entanglement(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: entanglement capability via Meyer-Wallach."""
    from encoding_atlas.analysis import compute_entanglement_capability

    # Non-entangling encodings have zero entanglement by definition.
    if not encoding.properties.is_entangling:
        return {
            "entanglement_capability": 0.0,
            "std_error": 0.0,
            "measure": "meyer_wallach",
            "per_qubit_entanglement": [0.0] * encoding.n_qubits,
            "note": "Non-entangling encoding; entanglement is zero by definition.",
            "encoding_name": type(encoding).__name__,
        }

    ap = config.analysis_params
    ent_result = compute_entanglement_capability(
        encoding,
        n_samples=ap.get("n_samples", 1000),
        input_range=tuple(ap.get("input_range", (0.0, 6.283185307179586))),
        seed=seed,
        backend=config.backend,
        measure=ap.get("measure", "meyer_wallach"),
        return_details=True,
        verbose=False,
    )

    result: dict[str, Any] = dict(ent_result)
    result["encoding_name"] = type(encoding).__name__

    # Convert numpy arrays to lists for JSON serialization.
    for key in ("per_qubit_entanglement", "entanglement_samples"):
        if key in result and hasattr(result[key], "tolist"):
            result[key] = result[key].tolist()

    return result


def _handle_trainability(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: trainability via gradient variance analysis."""
    from encoding_atlas.analysis import estimate_trainability

    ap = config.analysis_params
    train_result = estimate_trainability(
        encoding,
        n_samples=ap.get("n_samples", 500),
        input_range=tuple(ap.get("input_range", (0.0, 6.283185307179586))),
        observable=ap.get("observable", "computational"),
        seed=seed,
        backend=config.backend,
        return_details=True,
        verbose=False,
    )

    result: dict[str, Any] = dict(train_result)
    result["encoding_name"] = type(encoding).__name__

    # Convert numpy arrays to lists for JSON serialization.
    if "per_parameter_variance" in result and hasattr(result["per_parameter_variance"], "tolist"):
        result["per_parameter_variance"] = result["per_parameter_variance"].tolist()

    return result


def _handle_noise(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: noise resilience analysis (Stage 5b).

    This handler computes:
    1. Ideal (noiseless) expressibility and entanglement baselines
    2. Noisy expressibility for each noise level
    3. Noisy entanglement for each noise level (entangling encodings only)
    4. Fidelity decay for each noise level
    5. Relative change metrics from ideal to noisy

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to analyze.
    config : ExperimentConfig
        Configuration containing noise_levels and sample counts.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing ideal and noisy metrics.
        For each noise level ``lvl``, the following change fields
        are included:

        - ``noisy_{lvl}_expressibility_change``: Relative change in
          expressibility, defined as
          ``(ideal - noisy) / ideal``.  Positive values indicate
          noise-induced degradation; **negative** values indicate
          that noise increased expressibility (noise can push the
          fidelity distribution closer to Haar-random).
        - ``noisy_{lvl}_entanglement_change``: Same convention for
          entanglement capability.
    """
    from experiments.noisy_analysis import (
        compute_noisy_expressibility,
        compute_noisy_entanglement,
        compute_fidelity_decay,
    )
    from encoding_atlas.analysis import (
        compute_expressibility,
        compute_entanglement_capability,
    )

    # Maximum qubits for noisy simulation (density matrices are 4^n)
    MAX_NOISY_QUBITS = 8

    ap = config.analysis_params
    noise_levels = ap.get("noise_levels", ["medium"])
    n_samples_expr = ap.get("n_samples_expressibility", 1000)
    n_samples_ent = ap.get("n_samples_entanglement", 500)
    n_samples_fid = ap.get("n_samples_fidelity", 100)
    input_range = tuple(ap.get("input_range", (0.0, 6.283185307179586)))

    result: dict[str, Any] = {
        "encoding_name": type(encoding).__name__,
        "n_qubits": encoding.n_qubits,
        "n_features": encoding.n_features,
        "is_entangling": encoding.properties.is_entangling,
    }

    # Check qubit count for noisy simulation
    if encoding.n_qubits > MAX_NOISY_QUBITS:
        result["status"] = "skipped"
        result["reason"] = (
            f"Encoding has {encoding.n_qubits} qubits, exceeds maximum "
            f"{MAX_NOISY_QUBITS} for noisy simulation."
        )
        logger.warning(
            "Skipping noisy analysis for %s: %d qubits exceeds max %d",
            type(encoding).__name__,
            encoding.n_qubits,
            MAX_NOISY_QUBITS,
        )
        return result

    # Compute ideal (noiseless) baselines using the library functions
    try:
        ideal_expr_result = compute_expressibility(
            encoding,
            n_samples=n_samples_expr,
            input_range=input_range,
            seed=seed,
            backend=config.backend,
            return_distributions=True,
        )
        result["ideal_expressibility"] = ideal_expr_result["expressibility"]
        result["ideal_kl_divergence"] = ideal_expr_result["kl_divergence"]
        result["ideal_mean_fidelity"] = ideal_expr_result["mean_fidelity"]
    except Exception as e:
        logger.warning("Failed to compute ideal expressibility: %s", e)
        result["ideal_expressibility"] = None
        result["ideal_expressibility_error"] = str(e)

    # Ideal entanglement (only for entangling encodings)
    if encoding.properties.is_entangling and encoding.n_qubits >= 2:
        try:
            ideal_ent_result = compute_entanglement_capability(
                encoding,
                n_samples=n_samples_ent,
                input_range=input_range,
                seed=seed,
                backend=config.backend,
                return_details=True,
            )
            result["ideal_entanglement"] = ideal_ent_result["entanglement_capability"]
            result["ideal_entanglement_std_error"] = ideal_ent_result["std_error"]
        except Exception as e:
            logger.warning("Failed to compute ideal entanglement: %s", e)
            result["ideal_entanglement"] = None
            result["ideal_entanglement_error"] = str(e)
    else:
        result["ideal_entanglement"] = 0.0
        result["ideal_entanglement_note"] = "Non-entangling encoding or single qubit"

    # Compute noisy metrics for each noise level
    for noise_level in noise_levels:
        prefix = f"noisy_{noise_level}"

        # Noisy expressibility
        try:
            noisy_expr = compute_noisy_expressibility(
                encoding,
                noise_level=noise_level,
                n_samples=n_samples_expr,
                seed=seed + 100,  # Offset seed for noisy computation
            )
            result[f"{prefix}_expressibility"] = noisy_expr["noisy_expressibility"]
            result[f"{prefix}_kl_divergence"] = noisy_expr["noisy_kl_divergence"]
            result[f"{prefix}_mean_fidelity"] = noisy_expr["noisy_mean_fidelity"]
            result[f"{prefix}_expr_status"] = noisy_expr["status"]

            # Relative change: (ideal - noisy) / ideal.
            # Positive means noise degraded the metric; negative
            # means noise *increased* expressibility (randomness
            # pushing the distribution closer to Haar).
            if (
                result.get("ideal_expressibility") is not None
                and result["ideal_expressibility"] > 0
            ):
                change = (
                    result["ideal_expressibility"] - noisy_expr["noisy_expressibility"]
                ) / result["ideal_expressibility"]
                result[f"{prefix}_expressibility_change"] = change
            else:
                result[f"{prefix}_expressibility_change"] = None

        except Exception as e:
            logger.warning(
                "Failed to compute noisy expressibility at %s: %s", noise_level, e
            )
            result[f"{prefix}_expressibility"] = None
            result[f"{prefix}_expr_error"] = str(e)
            result[f"{prefix}_expr_status"] = "failed"

        # Noisy entanglement (only for entangling encodings with >= 2 qubits)
        if encoding.properties.is_entangling and encoding.n_qubits >= 2:
            try:
                noisy_ent = compute_noisy_entanglement(
                    encoding,
                    noise_level=noise_level,
                    n_samples=n_samples_ent,
                    seed=seed + 200,
                )
                result[f"{prefix}_entanglement"] = noisy_ent["noisy_entanglement"]
                result[f"{prefix}_linear_entropy"] = noisy_ent["noisy_linear_entropy_avg"]
                result[f"{prefix}_ent_status"] = noisy_ent["status"]

                # Relative change: (ideal - noisy) / ideal.
                # Same sign convention as expressibility_change.
                if (
                    result.get("ideal_entanglement") is not None
                    and result["ideal_entanglement"] > 0
                ):
                    change = (
                        result["ideal_entanglement"] - noisy_ent["noisy_entanglement"]
                    ) / result["ideal_entanglement"]
                    result[f"{prefix}_entanglement_change"] = change
                else:
                    result[f"{prefix}_entanglement_change"] = None

            except Exception as e:
                logger.warning(
                    "Failed to compute noisy entanglement at %s: %s", noise_level, e
                )
                result[f"{prefix}_entanglement"] = None
                result[f"{prefix}_ent_error"] = str(e)
                result[f"{prefix}_ent_status"] = "failed"
        else:
            result[f"{prefix}_entanglement"] = 0.0
            result[f"{prefix}_ent_status"] = "skipped_non_entangling"

        # Fidelity decay
        try:
            fidelity_result = compute_fidelity_decay(
                encoding,
                noise_level=noise_level,
                n_samples=n_samples_fid,
                seed=seed + 300,
            )
            result[f"{prefix}_fidelity_mean"] = fidelity_result["mean_fidelity"]
            result[f"{prefix}_fidelity_std"] = fidelity_result["std_fidelity"]
            result[f"{prefix}_fidelity_decay"] = fidelity_result["fidelity_decay"]
            result[f"{prefix}_fidelity_status"] = fidelity_result["status"]

        except Exception as e:
            logger.warning(
                "Failed to compute fidelity decay at %s: %s", noise_level, e
            )
            result[f"{prefix}_fidelity_mean"] = None
            result[f"{prefix}_fidelity_error"] = str(e)
            result[f"{prefix}_fidelity_status"] = "failed"

    # Overall status
    has_failures = any(
        result.get(f"noisy_{nl}_expr_status") == "failed"
        or result.get(f"noisy_{nl}_expr_status") == "noise_failure"
        for nl in noise_levels
    )
    result["status"] = "partial" if has_failures else "success"

    return result


def _handle_vqc(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: VQC classification benchmark (Stage 6a).

    This handler runs the VQC classification benchmark for a single encoding.
    For each dataset with matching feature dimensions, it:
    1. Runs multiple runs with different seeds
    2. Uses stratified k-fold cross-validation within each run
    3. Trains and evaluates the VQC classifier
    4. Computes aggregate statistics with bootstrap confidence intervals

    Seed strategy (see ``PHASE_4_EXPERIMENT_PLAN.md §Seed Strategy``):

    - **CV fold generation** uses ``config.cv_fold_seed(dataset_idx)`` which
      depends only on (stage, dataset) — *not* on the encoding.  This ensures
      all encodings and classical baselines are evaluated on identical
      train/test splits, enabling valid paired statistical tests.
    - **VQC weight initialisation** uses
      ``config.vqc_run_seed(encoding_idx, dataset_idx, run_idx)`` which
      varies per encoding, dataset, and run.
    - **Per-fold init seed** uses ``config.fold_init_seed(run_seed, fold_idx)``
      to give each fold a unique but deterministic VQC initialisation.

    .. note::
       TrainableEncoding (E13) has learnable parameters in the encoding
       layer itself.  The VQC pipeline only optimises the variational-layer
       parameters (``params_``), not the encoding parameters.  This is by
       design: all encodings are evaluated under a standardised protocol
       where the encoding acts as a fixed feature map.  Joint optimisation
       of encoding and variational parameters is left to future work.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to benchmark.
    config : ExperimentConfig
        Configuration containing datasets, n_runs, n_folds, and vqc_params.
    seed : int
        ``task_seed(task_index)`` — equals ``stage_seed + encoding_index``.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing per-dataset results with:
        - mean/std test accuracy across all folds and runs
        - 95% bootstrap confidence intervals
        - per-run and per-fold detailed results
    """
    import numpy as np
    from experiments.vqc import VQCClassifier
    from experiments.datasets import load_dataset, get_cv_folds, DATASET_N_FEATURES
    from experiments.statistical import bootstrap_ci
    from sklearn.metrics import precision_score, recall_score, f1_score

    ap = config.analysis_params
    datasets = ap.get("datasets", ["moons"])
    n_runs = ap.get("n_runs", 10)
    n_folds = ap.get("n_folds", 5)
    vqc_params = ap.get("vqc_params", {})
    max_samples = ap.get("max_samples")

    # Derive encoding_index from the task seed:
    #   seed == stage_seed + encoding_index  (set by _execute_task)
    stage_seed = config.stage_seed
    encoding_index = seed - stage_seed

    results: dict[str, Any] = {
        "schema_version": "1.0",
        "encoding_name": type(encoding).__name__,
        "n_qubits": encoding.n_qubits,
        "n_features": encoding.n_features,
        "datasets": {},
    }

    for dataset_idx, dataset_name in enumerate(datasets):
        # Check feature dimension compatibility
        dataset_n_features = DATASET_N_FEATURES.get(dataset_name)
        if dataset_n_features is None:
            results["datasets"][dataset_name] = {
                "status": "skipped",
                "reason": f"Unknown dataset: {dataset_name}",
            }
            continue

        if dataset_n_features != encoding.n_features:
            # Skip incompatible dataset/encoding combinations
            results["datasets"][dataset_name] = {
                "status": "skipped",
                "reason": (
                    f"Feature mismatch: dataset has {dataset_n_features}, "
                    f"encoding needs {encoding.n_features}"
                ),
            }
            continue

        # CV fold seed: depends on (base_seed, dataset) only — NOT on
        # encoding or stage.  This is the key invariant for valid paired
        # statistical tests within a stage AND for cross-paradigm
        # comparison (H6) between VQC and kernel stages.
        cv_fold_seed = config.cv_fold_seed(dataset_idx)

        # Load dataset using the same encoding/stage-independent seed so
        # every encoding and every stage sees identical data (critical for
        # synthetic datasets where the seed controls the noise realisation).
        try:
            X, y = load_dataset(
                dataset_name, seed=cv_fold_seed, max_samples=max_samples,
            )
        except Exception as e:
            results["datasets"][dataset_name] = {
                "status": "failed",
                "reason": f"Failed to load dataset: {str(e)}",
            }
            continue

        dataset_results: dict[str, Any] = {
            "runs": [],
            "mean_test_accuracy": None,
            "std_test_accuracy": None,
            "mean_train_accuracy": None,
            "std_train_accuracy": None,
            "mean_precision": None,
            "mean_recall": None,
            "mean_f1": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }

        all_test_accuracies: list[float] = []
        all_train_accuracies: list[float] = []
        all_precisions: list[float] = []
        all_recalls: list[float] = []
        all_f1s: list[float] = []

        for run_idx in range(n_runs):
            # CV folds: encoding-independent (same splits for all encodings).
            folds = get_cv_folds(X, y, n_folds=n_folds, seed=cv_fold_seed)

            # VQC init seed: encoding-dependent (different random restarts).
            # Formula: stage_seed + encoding_index*100 + dataset_idx*10 + run_idx
            vqc_run_seed = config.vqc_run_seed(
                encoding_index, dataset_idx, run_idx,
            )

            fold_accuracies = []
            fold_results = []

            for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
                # Per-fold VQC seed: unique per (encoding, dataset, run, fold).
                vqc_fold_seed = config.fold_init_seed(vqc_run_seed, fold_idx)

                # Train VQC
                vqc = VQCClassifier(
                    encoding=encoding,
                    n_var_layers=vqc_params.get("n_var_layers", 2),
                    lr=vqc_params.get("lr", 0.01),
                    epochs=vqc_params.get("epochs", 100),
                    seed=vqc_fold_seed,
                )

                t_fold = time.monotonic()

                try:
                    vqc.fit(X_train, y_train)
                    y_pred = vqc.predict(X_test)
                    y_pred_train = vqc.predict(X_train)

                    fold_wall = round(time.monotonic() - t_fold, 3)

                    acc = float(np.mean(y_pred == y_test))
                    train_acc = float(np.mean(y_pred_train == y_train))
                    prec = float(precision_score(y_test, y_pred, zero_division=0))
                    rec = float(recall_score(y_test, y_pred, zero_division=0))
                    f1 = float(f1_score(y_test, y_pred, zero_division=0))

                    fold_results.append({
                        "fold": fold_idx,
                        "test_accuracy": acc,
                        "train_accuracy": train_acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "final_loss": vqc.get_final_loss(),
                        "loss_curve": [float(l) for l in vqc.loss_history_],
                        "n_epochs_trained": len(vqc.loss_history_),
                        "wall_time_seconds": fold_wall,
                        "status": vqc.status_,
                    })
                    fold_accuracies.append(acc)
                    all_train_accuracies.append(train_acc)
                    all_precisions.append(prec)
                    all_recalls.append(rec)
                    all_f1s.append(f1)

                except Exception as e:
                    fold_wall = round(time.monotonic() - t_fold, 3)
                    fold_results.append({
                        "fold": fold_idx,
                        "status": "failed",
                        "error": str(e),
                        "wall_time_seconds": fold_wall,
                    })

            if fold_accuracies:
                run_mean = float(np.mean(fold_accuracies))
                all_test_accuracies.extend(fold_accuracies)
            else:
                run_mean = None

            dataset_results["runs"].append({
                "run": run_idx,
                "cv_fold_seed": cv_fold_seed,
                "vqc_run_seed": vqc_run_seed,
                "folds": fold_results,
                "mean_accuracy": run_mean,
            })

        # Aggregate across all runs
        if all_test_accuracies:
            acc_array = np.array(all_test_accuracies)
            dataset_results["mean_test_accuracy"] = float(np.mean(acc_array))
            dataset_results["std_test_accuracy"] = float(np.std(acc_array, ddof=1))
            dataset_results["mean_train_accuracy"] = float(np.mean(all_train_accuracies))
            dataset_results["std_train_accuracy"] = float(np.std(all_train_accuracies, ddof=1))
            dataset_results["mean_precision"] = float(np.mean(all_precisions))
            dataset_results["mean_recall"] = float(np.mean(all_recalls))
            dataset_results["mean_f1"] = float(np.mean(all_f1s))
            ci_low, ci_high = bootstrap_ci(acc_array, confidence=0.95, seed=seed)
            dataset_results["ci_95_lower"] = float(ci_low)
            dataset_results["ci_95_upper"] = float(ci_high)
            dataset_results["status"] = "success"
        else:
            dataset_results["status"] = "failed"
            dataset_results["reason"] = "No successful folds"

        results["datasets"][dataset_name] = dataset_results

    return results


def _handle_kernel(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: Quantum kernel classification benchmark (Stage 6b).

    This handler computes quantum kernel matrices and uses them with
    classical SVM for classification. For each compatible dataset:
    1. Runs multiple runs with different seeds
    2. Uses stratified k-fold cross-validation
    3. Computes kernel matrices for train and test sets
    4. Trains SVM with precomputed kernel
    5. Computes kernel-target alignment
    6. Aggregates statistics with bootstrap confidence intervals

    Seed strategy (see ``PHASE_4_EXPERIMENT_PLAN.md §Seed Strategy``):

    - **CV fold generation** uses ``config.cv_fold_seed(dataset_idx)`` —
      encoding-independent, ensuring identical splits for paired tests.
    - **SVM random state** uses
      ``config.kernel_run_seed(encoding_idx, dataset_idx, run_idx)`` with
      a +500 offset to separate kernel seeds from VQC seeds.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to benchmark.
    config : ExperimentConfig
        Configuration containing datasets, n_runs, n_folds.
    seed : int
        ``task_seed(task_index)`` — equals ``stage_seed + encoding_index``.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing per-dataset results with:
        - mean/std test accuracy across all folds and runs
        - 95% bootstrap confidence intervals
        - kernel-target alignment scores
        - per-run and per-fold detailed results
    """
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.metrics import precision_score, recall_score, f1_score
    from experiments.kernel import (
        compute_kernel_matrix,
        compute_kernel_matrix_cross,
        kernel_target_alignment,
        centered_kernel_target_alignment,
        ensure_psd,
    )
    from experiments.datasets import load_dataset, get_cv_folds, DATASET_N_FEATURES
    from experiments.statistical import bootstrap_ci

    ap = config.analysis_params
    datasets = ap.get("datasets", ["moons"])
    n_runs = ap.get("n_runs", 10)
    n_folds = ap.get("n_folds", 5)
    compute_align = ap.get("compute_alignment", True)
    max_samples = ap.get("max_samples")

    # Derive encoding_index from the task seed.
    stage_seed = config.stage_seed
    encoding_index = seed - stage_seed

    results: dict[str, Any] = {
        "schema_version": "1.0",
        "encoding_name": type(encoding).__name__,
        "n_qubits": encoding.n_qubits,
        "n_features": encoding.n_features,
        "datasets": {},
    }

    for dataset_idx, dataset_name in enumerate(datasets):
        # Check feature dimension compatibility
        dataset_n_features = DATASET_N_FEATURES.get(dataset_name)
        if dataset_n_features is None:
            results["datasets"][dataset_name] = {
                "status": "skipped",
                "reason": f"Unknown dataset: {dataset_name}",
            }
            continue

        if dataset_n_features != encoding.n_features:
            results["datasets"][dataset_name] = {
                "status": "skipped",
                "reason": (
                    f"Feature mismatch: dataset has {dataset_n_features}, "
                    f"encoding needs {encoding.n_features}"
                ),
            }
            continue

        # CV fold seed: depends on (base_seed, dataset) only — NOT on
        # encoding or stage.  This is the key invariant for valid paired
        # statistical tests within a stage AND for cross-paradigm
        # comparison (H6) between VQC and kernel stages.
        cv_fold_seed = config.cv_fold_seed(dataset_idx)

        # Load dataset using the same encoding/stage-independent seed so
        # every encoding and every stage sees identical data (critical for
        # synthetic datasets where the seed controls the noise realisation).
        try:
            X, y = load_dataset(
                dataset_name, seed=cv_fold_seed, max_samples=max_samples,
            )
        except Exception as e:
            results["datasets"][dataset_name] = {
                "status": "failed",
                "reason": f"Failed to load dataset: {str(e)}",
            }
            continue

        dataset_results: dict[str, Any] = {
            "runs": [],
            "mean_test_accuracy": None,
            "std_test_accuracy": None,
            "mean_precision": None,
            "mean_recall": None,
            "mean_f1": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
            "kernel_target_alignment": None,
            "centered_kernel_target_alignment": None,
        }

        all_test_accuracies: list[float] = []
        all_precisions: list[float] = []
        all_recalls: list[float] = []
        all_f1s: list[float] = []

        # Compute full kernel matrix once for alignment (if requested).
        # Both uncentered (Cristianini 2002) and centered (Cortes 2012)
        # variants are computed.  The centered KTA removes the diagonal
        # bias inherent to fidelity kernels where K[i,i] = 1.
        if compute_align:
            try:
                K_full = compute_kernel_matrix(encoding, X, backend=config.backend)
                K_full_psd, was_modified = ensure_psd(K_full)
                alignment = kernel_target_alignment(K_full_psd, y)
                c_alignment = centered_kernel_target_alignment(K_full_psd, y)
                dataset_results["kernel_target_alignment"] = float(alignment)
                dataset_results["centered_kernel_target_alignment"] = float(c_alignment)
                dataset_results["kernel_was_regularized"] = was_modified
            except Exception as e:
                logger.warning(
                    "Failed to compute kernel alignment for %s: %s", dataset_name, e
                )
                dataset_results["kernel_target_alignment"] = None
                dataset_results["centered_kernel_target_alignment"] = None
                dataset_results["alignment_error"] = str(e)

        for run_idx in range(n_runs):
            # CV folds: encoding-independent (same splits for all encodings).
            folds = get_cv_folds(X, y, n_folds=n_folds, seed=cv_fold_seed)

            # Kernel SVM seed: encoding-dependent.
            # Formula: stage_seed + 500 + encoding_index*100 + dataset_idx*10 + run_idx
            ker_run_seed = config.kernel_run_seed(
                encoding_index, dataset_idx, run_idx,
            )

            fold_accuracies = []
            fold_results = []

            for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
                t_fold = time.monotonic()

                try:
                    # Compute training kernel matrix and retain the
                    # pre-computed statevectors so they can be reused
                    # for the cross-kernel (avoids double simulation).
                    K_train, train_states = compute_kernel_matrix(
                        encoding, X_train,
                        backend=config.backend,
                        return_states=True,
                    )
                    K_train_psd, train_regularized = ensure_psd(K_train)

                    # Compute test-train kernel matrix, reusing the
                    # training states obtained above.
                    K_test = compute_kernel_matrix_cross(
                        encoding,
                        X_train,
                        X_test,
                        train_states=train_states,
                        backend=config.backend,
                    )

                    # Train SVM with precomputed kernel — per-fold seed.
                    svm_seed = config.fold_init_seed(ker_run_seed, fold_idx)
                    svm = SVC(kernel="precomputed", random_state=svm_seed)
                    svm.fit(K_train_psd, y_train)

                    # Predict
                    y_pred = svm.predict(K_test)

                    fold_wall = round(time.monotonic() - t_fold, 3)

                    acc = float(np.mean(y_pred == y_test))
                    prec = float(precision_score(y_test, y_pred, zero_division=0))
                    rec = float(recall_score(y_test, y_pred, zero_division=0))
                    f1 = float(f1_score(y_test, y_pred, zero_division=0))

                    fold_results.append({
                        "fold": fold_idx,
                        "test_accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "kernel_regularized": train_regularized,
                        "wall_time_seconds": fold_wall,
                        "status": "success",
                    })
                    fold_accuracies.append(acc)
                    all_precisions.append(prec)
                    all_recalls.append(rec)
                    all_f1s.append(f1)

                except Exception as e:
                    fold_wall = round(time.monotonic() - t_fold, 3)
                    fold_results.append({
                        "fold": fold_idx,
                        "status": "failed",
                        "error": str(e),
                        "wall_time_seconds": fold_wall,
                    })

            if fold_accuracies:
                run_mean = float(np.mean(fold_accuracies))
                all_test_accuracies.extend(fold_accuracies)
            else:
                run_mean = None

            dataset_results["runs"].append({
                "run": run_idx,
                "cv_fold_seed": cv_fold_seed,
                "kernel_run_seed": ker_run_seed,
                "folds": fold_results,
                "mean_accuracy": run_mean,
            })

        # Aggregate across all runs
        if all_test_accuracies:
            acc_array = np.array(all_test_accuracies)
            dataset_results["mean_test_accuracy"] = float(np.mean(acc_array))
            dataset_results["std_test_accuracy"] = float(np.std(acc_array, ddof=1))
            dataset_results["mean_precision"] = float(np.mean(all_precisions))
            dataset_results["mean_recall"] = float(np.mean(all_recalls))
            dataset_results["mean_f1"] = float(np.mean(all_f1s))
            ci_low, ci_high = bootstrap_ci(acc_array, confidence=0.95, seed=seed)
            dataset_results["ci_95_lower"] = float(ci_low)
            dataset_results["ci_95_upper"] = float(ci_high)
            dataset_results["status"] = "success"
        else:
            dataset_results["status"] = "failed"
            dataset_results["reason"] = "No successful folds"

        results["datasets"][dataset_name] = dataset_results

    return results


def _handle_classical_baseline(
    baseline_name: str,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Run a classical baseline classifier across all compatible datasets.

    This handler mirrors the VQC handler structure so that classical and
    quantum results use identical datasets, folds, and seeds for fair
    paired statistical comparison.

    Seed strategy (see ``PHASE_4_EXPERIMENT_PLAN.md §Seed Strategy``):

    - **CV fold generation** uses ``config.cv_fold_seed(dataset_idx)`` —
      the *same* encoding-independent seed used by ``_handle_vqc`` and
      ``_handle_kernel``, ensuring identical train/test splits.
    - **Classifier random state** uses a baseline-specific seed offset
      (``stage_seed + 2000 + ...``) so baseline runs are reproducible
      yet distinct from quantum encoding seeds.

    Parameters
    ----------
    baseline_name : str
        Name of the classical classifier (e.g. ``"svm_rbf"``).
        Must be recognised by :func:`~experiments.classical_baselines.get_classical_baseline`.
    config : ExperimentConfig
        Configuration containing datasets, n_runs, and n_folds.
    seed : int
        ``config.stage_seed`` (set by :meth:`_run_classical_baselines`).

    Returns
    -------
    dict[str, Any]
        Results dictionary keyed by dataset name, each containing
        mean/std accuracy, bootstrap confidence intervals, and
        per-run/per-fold detailed results.
    """
    import numpy as np
    from experiments.classical_baselines import run_baseline_single_fold
    from experiments.datasets import load_dataset, get_cv_folds, DATASET_N_FEATURES
    from experiments.statistical import bootstrap_ci

    ap = config.analysis_params
    datasets = ap.get("datasets", ["moons"])
    n_runs = ap.get("n_runs", 10)
    n_folds = ap.get("n_folds", 5)
    max_samples = ap.get("max_samples")

    # seed is config.stage_seed (encoding-independent by construction).
    stage_seed = seed

    results: dict[str, Any] = {
        "schema_version": "1.0",
        "method": "classical_baseline",
        "baseline_name": baseline_name,
        "datasets": {},
    }

    for dataset_idx, dataset_name in enumerate(datasets):
        dataset_n_features = DATASET_N_FEATURES.get(dataset_name)
        if dataset_n_features is None:
            results["datasets"][dataset_name] = {
                "status": "skipped",
                "reason": f"Unknown dataset: {dataset_name}",
            }
            continue

        # CV fold seed: depends on (base_seed, dataset) only — NOT on
        # encoding or stage.  Identical to _handle_vqc and _handle_kernel,
        # ensuring baselines and quantum methods share the same data and
        # CV splits for valid paired statistical comparison.
        cv_fold_seed = config.cv_fold_seed(dataset_idx)

        # Load dataset using the same encoding/stage-independent seed.
        try:
            X, y = load_dataset(
                dataset_name, seed=cv_fold_seed, max_samples=max_samples,
            )
        except Exception as e:
            results["datasets"][dataset_name] = {
                "status": "failed",
                "reason": f"Failed to load dataset: {str(e)}",
            }
            continue

        dataset_results: dict[str, Any] = {
            "runs": [],
            "mean_test_accuracy": None,
            "std_test_accuracy": None,
            "mean_train_accuracy": None,
            "std_train_accuracy": None,
            "mean_precision": None,
            "mean_recall": None,
            "mean_f1": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }

        all_test_accuracies: list[float] = []
        all_train_accuracies: list[float] = []
        all_precisions: list[float] = []
        all_recalls: list[float] = []
        all_f1s: list[float] = []

        for run_idx in range(n_runs):
            # CV folds: encoding-independent (identical to _handle_vqc).
            folds = get_cv_folds(X, y, n_folds=n_folds, seed=cv_fold_seed)

            fold_accuracies: list[float] = []
            fold_results: list[dict[str, Any]] = []

            for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
                # Classifier seed: offset by 2000 to separate from VQC/kernel.
                baseline_fold_seed = config.fold_init_seed(
                    stage_seed + 2000 + dataset_idx * 10 + run_idx,
                    fold_idx,
                )

                t_fold = time.monotonic()
                fold_result = run_baseline_single_fold(
                    baseline_name,
                    X_train, X_test, y_train, y_test,
                    seed=baseline_fold_seed,
                )
                fold_wall = round(time.monotonic() - t_fold, 3)

                fold_result["fold"] = fold_idx
                fold_result["wall_time_seconds"] = fold_wall
                fold_results.append(fold_result)

                if fold_result["status"] == "success":
                    fold_accuracies.append(fold_result["test_accuracy"])
                    all_train_accuracies.append(fold_result["train_accuracy"])
                    all_precisions.append(fold_result["precision"])
                    all_recalls.append(fold_result["recall"])
                    all_f1s.append(fold_result["f1"])

            if fold_accuracies:
                run_mean = float(np.mean(fold_accuracies))
                all_test_accuracies.extend(fold_accuracies)
            else:
                run_mean = None

            dataset_results["runs"].append({
                "run": run_idx,
                "cv_fold_seed": cv_fold_seed,
                "folds": fold_results,
                "mean_accuracy": run_mean,
            })

        # Aggregate across all runs.
        if all_test_accuracies:
            acc_array = np.array(all_test_accuracies)
            dataset_results["mean_test_accuracy"] = float(np.mean(acc_array))
            dataset_results["std_test_accuracy"] = float(np.std(acc_array, ddof=1))
            dataset_results["mean_train_accuracy"] = float(np.mean(all_train_accuracies))
            dataset_results["std_train_accuracy"] = float(np.std(all_train_accuracies, ddof=1))
            dataset_results["mean_precision"] = float(np.mean(all_precisions))
            dataset_results["mean_recall"] = float(np.mean(all_recalls))
            dataset_results["mean_f1"] = float(np.mean(all_f1s))
            ci_low, ci_high = bootstrap_ci(
                acc_array, confidence=0.95, seed=seed,
            )
            dataset_results["ci_95_lower"] = float(ci_low)
            dataset_results["ci_95_upper"] = float(ci_high)
            dataset_results["status"] = "success"
        else:
            dataset_results["status"] = "failed"
            dataset_results["reason"] = "No successful folds"

        results["datasets"][dataset_name] = dataset_results

    return results


def _handle_tradeoff(
    encoding: Any,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Stage handler: Tradeoff analysis (Stage 7).

    This handler ignores the encoding argument (which is a dummy)
    and instead loads all prior stage results for cross-stage analysis.
    """
    from experiments.tradeoff import run_tradeoff_analysis

    ap = config.analysis_params
    stage_dirs = ap.get("stage_dirs", {})
    figure_dir = ap.get("figure_dir", "experiments/results/figures")
    generate_plots = ap.get("generate_plots", True)
    sensitivity_n_samples = ap.get("sensitivity_n_samples", 1000)

    return run_tradeoff_analysis(
        stage_dirs=stage_dirs,
        output_dir=config.output_dir,
        figure_dir=figure_dir,
        seed=seed,
        generate_plots=generate_plots,
        sensitivity_n_samples=sensitivity_n_samples,
    )


# Handler registry — maps stage name to handler function.
_STAGE_HANDLERS: dict[str, Any] = {
    "resources": _handle_resources,
    "simulability": _handle_simulability,
    "expressibility": _handle_expressibility,
    "entanglement": _handle_entanglement,
    "trainability": _handle_trainability,
    "noise": _handle_noise,
    "vqc": _handle_vqc,
    "kernel": _handle_kernel,
    "tradeoff": _handle_tradeoff,
}


# ======================================================================
# Internal helpers
# ======================================================================

class _SkipTask(Exception):
    """Raised when a task should be skipped (not a real error)."""


def _print_header(stage: str, total: int, resume: bool) -> None:
    """Print a stage execution header to stdout."""
    mode = "RESUME" if resume else "FRESH"
    print(f"\n{'=' * 70}")
    print(f"  Stage: {stage}  |  Tasks: {total}  |  Mode: {mode}")
    print(f"{'=' * 70}")
    sys.stdout.flush()


def _print_progress(
    tag: str,
    exp_id: str,
    current: int,
    total: int,
    *,
    note: str = "",
) -> None:
    """Print a single task progress line."""
    suffix = f"  ({note})" if note else ""
    print(f"  [{tag}] ({current:>4d}/{total}) {exp_id}{suffix}")
    sys.stdout.flush()


def _print_footer(summary: dict[str, Any]) -> None:
    """Print a stage completion footer."""
    print(f"\n{'=' * 70}")
    print(
        f"  Done: {summary['completed']} completed, "
        f"{summary['skipped']} skipped, "
        f"{summary['failed']} failed  |  "
        f"Wall time: {summary['wall_time']:.1f}s"
    )
    print(f"{'=' * 70}\n")
    sys.stdout.flush()
