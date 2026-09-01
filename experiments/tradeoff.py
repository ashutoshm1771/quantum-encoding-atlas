"""Stage 7: Scaling & Tradeoff Analysis.

This module implements all 20 sub-analyses for Stage 7, which joins results
from Stages 1-6b to compute cross-stage correlations, hypothesis verdicts,
Pareto fronts, and final encoding rankings. It also generates publication-
ready tables in Markdown and LaTeX format.

The master entry point is :func:`run_tradeoff_analysis`, called by the
``_handle_tradeoff`` stage handler in ``experiments/runner.py``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.benchmark.scoring import (
    MetricSpec,
    readings_from_values,
    score_encodings,
    weight_sensitivity,
)

from experiments.statistical import (
    spearman_correlation,
    kendall_tau,
    wilcoxon_test,
    mann_whitney_test,
    cliffs_delta,
    interpret_cliffs_delta,
    holm_bonferroni_correction,
    bootstrap_ci,
    pairwise_comparisons,
    compute_ci_metrics,
)

logger = logging.getLogger(__name__)

# ── Encoding Metadata ──────────────────────────────────────────────

REGISTRY_TO_CLASS: dict[str, str] = {
    "angle": "AngleEncoding",
    "amplitude": "AmplitudeEncoding",
    "basis": "BasisEncoding",
    "iqp": "IQPEncoding",
    "zz_feature_map": "ZZFeatureMap",
    "pauli_feature_map": "PauliFeatureMap",
    "data_reuploading": "DataReuploadingEncoding",
    "hardware_efficient": "HardwareEfficientEncoding",
    "higher_order_angle": "HigherOrderAngleEncoding",
    "qaoa_encoding": "QAOAEncoding",
    "hamiltonian_encoding": "HamiltonianEncoding",
    "symmetry_inspired": "SymmetryInspiredFeatureMap",
    "trainable_encoding": "TrainableEncoding",
    "so2_equivariant": "SO2EquivariantFeatureMap",
    "cyclic_equivariant": "CyclicEquivariantFeatureMap",
    "swap_equivariant": "SwapEquivariantFeatureMap",
}
CLASS_TO_REGISTRY: dict[str, str] = {v: k for k, v in REGISTRY_TO_CLASS.items()}

ENCODING_FAMILIES: dict[str, list[str]] = {
    "Non-Entangling": ["angle", "basis", "higher_order_angle"],
    "Amplitude": ["amplitude"],
    "IQP-Based": ["iqp", "zz_feature_map"],
    "Pauli-Based": ["pauli_feature_map"],
    "Data Re-uploading": ["data_reuploading"],
    "Hardware-Efficient": ["hardware_efficient"],
    "QAOA/Hamiltonian": ["qaoa_encoding", "hamiltonian_encoding"],
    "Symmetry-Based": ["symmetry_inspired"],
    "Trainable": ["trainable_encoding"],
    "Equivariant": ["so2_equivariant", "cyclic_equivariant", "swap_equivariant"],
}

ENTANGLING_ENCODINGS: set[str] = {
    "amplitude", "iqp", "zz_feature_map", "pauli_feature_map",
    "data_reuploading", "hardware_efficient", "qaoa_encoding",
    "hamiltonian_encoding", "symmetry_inspired", "trainable_encoding",
    "so2_equivariant", "cyclic_equivariant", "swap_equivariant",
}

EQUIVARIANT_ENCODINGS: set[str] = {
    "so2_equivariant", "cyclic_equivariant", "swap_equivariant",
}

SYMMETRY_MATCHING: dict[str, list[str]] = {
    "circles": ["so2_equivariant"],
    "xor": ["swap_equivariant", "cyclic_equivariant"],
}

DEFAULT_N_FEATURES = 4
DEFAULT_REPS = 2
MIN_SAMPLES_FOR_TEST = 5

# Matched pairs for H3 (data re-uploading advantage)
_MATCHED_PAIRS = [
    ("data_reuploading", {"n_layers": 1}, "iqp", {"reps": 1}),
    ("data_reuploading", {"n_layers": 2}, "iqp", {"reps": 2}),
    ("data_reuploading", {"n_layers": 3}, "zz_feature_map", {"reps": 2}),
    ("data_reuploading", {"n_layers": 1}, "hardware_efficient", {"reps": 1}),
]


# ── Name Canonicalization ──────────────────────────────────────────

def canonicalize_encoding_name(name: str) -> str:
    """Convert encoding name to registry (snake_case) form.

    Parameters
    ----------
    name : str
        Either a class name (e.g. ``"IQPEncoding"``) or a registry
        name (e.g. ``"iqp"``).

    Returns
    -------
    str
        The registry name.
    """
    if name in REGISTRY_TO_CLASS:
        return name
    if name in CLASS_TO_REGISTRY:
        return CLASS_TO_REGISTRY[name]
    name_lower = name.lower()
    for reg_name in REGISTRY_TO_CLASS:
        if reg_name.replace("_", "") == name_lower.replace("_", ""):
            return reg_name
    logger.warning("Unknown encoding name: %s", name)
    return name


def get_encoding_family(registry_name: str) -> str:
    """Return the family name for a given encoding registry name.

    Returns ``"Unknown"`` if the encoding is not found.
    """
    for family, members in ENCODING_FAMILIES.items():
        if registry_name in members:
            return family
    return "Unknown"


# ── Data Loading ───────────────────────────────────────────────────

def load_stage_results(stage_dir: str) -> list[dict[str, Any]]:
    """Load successful results from a stage summary.json.

    Parameters
    ----------
    stage_dir : str
        Path to the stage output directory.

    Returns
    -------
    list[dict[str, Any]]
        List of result entries with ``status == "success"``.
        Returns empty list if file not found.
    """
    path = os.path.join(stage_dir, "summary.json")
    if not os.path.isfile(path):
        logger.warning("Stage results not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data.get("results", []) if r.get("status") == "success"]


def _find_default_config_result(
    results: list[dict[str, Any]],
    encoding_reg_name: str,
    n_features: int = 4,
) -> dict[str, Any] | None:
    """Find the result entry matching the canonical config for an encoding.

    Selection priority:
    1. Exact match on encoding_name and n_features
    2. For SO2Equivariant: accept n_features=2
    3. If multiple configs match, prefer reps=2 or n_layers=2
    """
    candidates = []
    for r in results:
        if canonicalize_encoding_name(r["encoding_name"]) != encoding_reg_name:
            continue
        params = r.get("encoding_params", {})
        nf = params.get("n_features", n_features)
        if nf == n_features:
            candidates.append(r)
        elif encoding_reg_name == "so2_equivariant" and nf == 2:
            candidates.append(r)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Prefer reps=2 or n_layers=2 (the default config from stage6a)
    for c in candidates:
        params = c.get("encoding_params", {})
        if params.get("reps") == DEFAULT_REPS or params.get("n_layers") == DEFAULT_REPS:
            return c
    return candidates[0]


def _extract_fold_accuracies(dataset_result: dict[str, Any]) -> list[float]:
    """Extract a flat list of per-fold test accuracies from VQC/kernel dataset result."""
    accuracies: list[float] = []
    for run in dataset_result.get("runs", []):
        for fold in run.get("folds", []):
            acc = fold.get("test_accuracy")
            if acc is not None and fold.get("status") != "failed":
                accuracies.append(float(acc))
    return accuracies


def build_encoding_profiles(
    stage_dirs: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build a per-encoding profile by joining results from all stages.

    Parameters
    ----------
    stage_dirs : dict[str, str]
        Mapping from stage key to directory path.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from canonical encoding name (registry) to profile dict.
    """
    profiles: dict[str, dict[str, Any]] = {}

    # Initialize profiles for all known encodings
    for reg_name in REGISTRY_TO_CLASS:
        n_feat = 2 if reg_name == "so2_equivariant" else DEFAULT_N_FEATURES
        profiles[reg_name] = {
            "class_name": REGISTRY_TO_CLASS[reg_name],
            "family": get_encoding_family(reg_name),
            "is_entangling": reg_name in ENTANGLING_ENCODINGS,
            "is_equivariant": reg_name in EQUIVARIANT_ENCODINGS,
            "n_features_used": n_feat,
            "depth": None,
            "gate_count": None,
            "two_qubit_gates": None,
            "parameter_count": None,
            "simulability_class": None,
            "is_simulable": None,
            "expressibility": None,
            "entanglement_capability": None,
            "trainability_estimate": None,
            "gradient_variance": None,
            "barren_plateau_risk": None,
            "noise_resilience": None,
            "noisy_medium_fidelity_decay": None,
            "vqc_accuracy": None,
            "vqc_ci": None,
            "vqc_folds": None,
            "vqc_train_accuracy": None,
            "kernel_accuracy": None,
            "kernel_ci": None,
            "kernel_folds": None,
            "kernel_kta": None,
            "expressibility_scaling": None,
            "entanglement_scaling": None,
            "trainability_scaling": None,
            "depth_scaling": None,
        }

    # Stage 1: Resources
    if "resources" in stage_dirs:
        results = load_stage_results(stage_dirs["resources"])
        # Build scaling data for all configs
        scaling_depth: dict[str, dict[int, int]] = {}
        for r in results:
            reg = canonicalize_encoding_name(r["encoding_name"])
            if reg not in profiles:
                continue
            params = r.get("encoding_params", {})
            nf = params.get("n_features", DEFAULT_N_FEATURES)
            res = r.get("result", {})
            scaling_depth.setdefault(reg, {})[nf] = res.get("depth", 0)

        for reg, scale_data in scaling_depth.items():
            if len(scale_data) > 1:
                profiles[reg]["depth_scaling"] = scale_data

        for reg in profiles:
            default_result = _find_default_config_result(results, reg)
            if default_result:
                res = default_result.get("result", {})
                profiles[reg]["depth"] = res.get("depth")
                profiles[reg]["gate_count"] = res.get("gate_count")
                profiles[reg]["two_qubit_gates"] = res.get("two_qubit_gates")
                profiles[reg]["parameter_count"] = res.get("parameter_count")

    # Stage 2: Simulability
    if "simulability" in stage_dirs:
        results = load_stage_results(stage_dirs["simulability"])
        for r in results:
            reg = canonicalize_encoding_name(r["encoding_name"])
            if reg not in profiles:
                continue
            res = r.get("result", {})
            profiles[reg]["simulability_class"] = res.get("simulability_class")
            profiles[reg]["is_simulable"] = res.get("is_simulable")

    # Stage 3: Expressibility
    if "expressibility" in stage_dirs:
        results = load_stage_results(stage_dirs["expressibility"])
        scaling_expr: dict[str, dict[int, float]] = {}
        for r in results:
            reg = canonicalize_encoding_name(r["encoding_name"])
            if reg not in profiles:
                continue
            params = r.get("encoding_params", {})
            nf = params.get("n_features", DEFAULT_N_FEATURES)
            res = r.get("result", {})
            expr_val = res.get("expressibility")
            if expr_val is not None:
                scaling_expr.setdefault(reg, {})[nf] = float(expr_val)

        for reg, scale_data in scaling_expr.items():
            if len(scale_data) > 1:
                profiles[reg]["expressibility_scaling"] = scale_data

        for reg in profiles:
            default_result = _find_default_config_result(results, reg)
            if default_result:
                res = default_result.get("result", {})
                profiles[reg]["expressibility"] = res.get("expressibility")

    # Stage 4: Entanglement
    if "entanglement" in stage_dirs:
        results = load_stage_results(stage_dirs["entanglement"])
        scaling_ent: dict[str, dict[int, float]] = {}
        for r in results:
            reg = canonicalize_encoding_name(r["encoding_name"])
            if reg not in profiles:
                continue
            params = r.get("encoding_params", {})
            nf = params.get("n_features", DEFAULT_N_FEATURES)
            res = r.get("result", {})
            ent_val = res.get("entanglement_capability")
            if ent_val is not None:
                scaling_ent.setdefault(reg, {})[nf] = float(ent_val)

        for reg, scale_data in scaling_ent.items():
            if len(scale_data) > 1:
                profiles[reg]["entanglement_scaling"] = scale_data

        for reg in profiles:
            default_result = _find_default_config_result(results, reg)
            if default_result:
                res = default_result.get("result", {})
                profiles[reg]["entanglement_capability"] = res.get("entanglement_capability")

    # Stage 5: Trainability
    if "trainability" in stage_dirs:
        results = load_stage_results(stage_dirs["trainability"])
        scaling_train: dict[str, dict[int, float]] = {}
        for r in results:
            reg = canonicalize_encoding_name(r["encoding_name"])
            if reg not in profiles:
                continue
            params = r.get("encoding_params", {})
            nf = params.get("n_features", DEFAULT_N_FEATURES)
            res = r.get("result", {})
            gv = res.get("gradient_variance")
            if gv is not None:
                scaling_train.setdefault(reg, {})[nf] = float(gv)

        for reg, scale_data in scaling_train.items():
            if len(scale_data) > 1:
                profiles[reg]["trainability_scaling"] = scale_data

        for reg in profiles:
            default_result = _find_default_config_result(results, reg)
            if default_result:
                res = default_result.get("result", {})
                profiles[reg]["trainability_estimate"] = res.get("trainability_estimate")
                profiles[reg]["gradient_variance"] = res.get("gradient_variance")
                profiles[reg]["barren_plateau_risk"] = res.get("barren_plateau_risk")

    # Stage 5b: Noise
    if "noise" in stage_dirs:
        results = load_stage_results(stage_dirs["noise"])
        for r in results:
            reg = canonicalize_encoding_name(r["encoding_name"])
            if reg not in profiles:
                continue
            res = r.get("result", {})
            fid_decay = res.get("noisy_medium_fidelity_decay")
            expr_change = res.get("noisy_medium_expressibility_change")
            profiles[reg]["noisy_medium_fidelity_decay"] = fid_decay
            if expr_change is not None:
                profiles[reg]["noise_resilience"] = 1.0 - float(expr_change)
            else:
                profiles[reg]["noise_resilience"] = None

    # Stage 6a: VQC
    # Merge dataset results from ALL n_features configs per encoding so that
    # both 2-feature datasets (moons, circles, …) and 4-feature datasets
    # (iris, wine, …) appear in the profile.
    if "vqc" in stage_dirs:
        results = load_stage_results(stage_dirs["vqc"])
        for reg in profiles:
            vqc_acc: dict[str, float] = {}
            vqc_ci: dict[str, tuple[float, float]] = {}
            vqc_folds: dict[str, list[float]] = {}
            vqc_train: dict[str, float] = {}
            for r in results:
                if canonicalize_encoding_name(r["encoding_name"]) != reg:
                    continue
                res = r.get("result", {})
                datasets_data = res.get("datasets", {})
                for ds_name, ds_result in datasets_data.items():
                    if ds_result.get("status") != "success":
                        continue
                    if ds_name in vqc_acc:
                        continue  # keep first successful config for this dataset
                    mean_acc = ds_result.get("mean_test_accuracy")
                    if mean_acc is not None:
                        vqc_acc[ds_name] = float(mean_acc)
                    ci_lo = ds_result.get("ci_95_lower")
                    ci_hi = ds_result.get("ci_95_upper")
                    if ci_lo is not None and ci_hi is not None:
                        vqc_ci[ds_name] = (float(ci_lo), float(ci_hi))
                    folds = _extract_fold_accuracies(ds_result)
                    if folds:
                        vqc_folds[ds_name] = folds
                    train_acc = ds_result.get("mean_train_accuracy")
                    if train_acc is not None:
                        vqc_train[ds_name] = float(train_acc)
            if vqc_acc:
                profiles[reg]["vqc_accuracy"] = vqc_acc
            if vqc_ci:
                profiles[reg]["vqc_ci"] = vqc_ci
            if vqc_folds:
                profiles[reg]["vqc_folds"] = vqc_folds
            if vqc_train:
                profiles[reg]["vqc_train_accuracy"] = vqc_train

    # Stage 6b: Kernel
    # Merge dataset results from ALL n_features configs (same as VQC above).
    if "kernel" in stage_dirs:
        results = load_stage_results(stage_dirs["kernel"])
        for reg in profiles:
            ker_acc: dict[str, float] = {}
            ker_ci: dict[str, tuple[float, float]] = {}
            ker_folds: dict[str, list[float]] = {}
            ker_kta: dict[str, float] = {}
            for r in results:
                if canonicalize_encoding_name(r["encoding_name"]) != reg:
                    continue
                res = r.get("result", {})
                datasets_data = res.get("datasets", {})
                for ds_name, ds_result in datasets_data.items():
                    if ds_result.get("status") != "success":
                        continue
                    if ds_name in ker_acc:
                        continue  # keep first successful config for this dataset
                    mean_acc = ds_result.get("mean_test_accuracy")
                    if mean_acc is not None:
                        ker_acc[ds_name] = float(mean_acc)
                    ci_lo = ds_result.get("ci_95_lower")
                    ci_hi = ds_result.get("ci_95_upper")
                    if ci_lo is not None and ci_hi is not None:
                        ker_ci[ds_name] = (float(ci_lo), float(ci_hi))
                    folds = _extract_fold_accuracies(ds_result)
                    if folds:
                        ker_folds[ds_name] = folds
                    ckta = ds_result.get("centered_kernel_target_alignment")
                    if ckta is not None:
                        ker_kta[ds_name] = float(ckta)
            if ker_acc:
                profiles[reg]["kernel_accuracy"] = ker_acc
            if ker_ci:
                profiles[reg]["kernel_ci"] = ker_ci
            if ker_folds:
                profiles[reg]["kernel_folds"] = ker_folds
            if ker_kta:
                profiles[reg]["kernel_kta"] = ker_kta

    n_with_vqc = sum(1 for p in profiles.values() if p.get("vqc_accuracy"))
    n_with_depth = sum(1 for p in profiles.values() if p.get("depth") is not None)
    logger.info(
        "Built profiles for %d encodings (%d with VQC accuracy, %d with depth)",
        len(profiles), n_with_vqc, n_with_depth,
    )
    return profiles


def extract_classical_baselines(
    stage_dir: str,
) -> dict[str, dict[str, Any]]:
    """Extract classical baseline results from Stage 6a summary.

    Parameters
    ----------
    stage_dir : str
        Path to ``stage6a_vqc`` results directory.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from baseline name to accuracy/folds/ci dicts.
    """
    results = load_stage_results(stage_dir)
    baselines: dict[str, dict[str, Any]] = {}
    for r in results:
        enc_name = r.get("encoding_name", "")
        if not enc_name.startswith("classical_"):
            continue
        res = r.get("result", {})
        datasets_data = res.get("datasets", {})
        if not datasets_data:
            continue
        bl_acc: dict[str, float] = {}
        bl_folds: dict[str, list[float]] = {}
        bl_ci: dict[str, tuple[float, float]] = {}
        for ds_name, ds_result in datasets_data.items():
            if ds_result.get("status") != "success":
                continue
            mean_acc = ds_result.get("mean_test_accuracy")
            if mean_acc is not None:
                bl_acc[ds_name] = float(mean_acc)
            ci_lo = ds_result.get("ci_95_lower")
            ci_hi = ds_result.get("ci_95_upper")
            if ci_lo is not None and ci_hi is not None:
                bl_ci[ds_name] = (float(ci_lo), float(ci_hi))
            folds = _extract_fold_accuracies(ds_result)
            if folds:
                bl_folds[ds_name] = folds
        baselines[enc_name] = {"accuracy": bl_acc, "folds": bl_folds, "ci": bl_ci}
    return baselines


# ── Private Helpers ────────────────────────────────────────────────

def _partial_spearman(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
) -> tuple[float, float]:
    """Compute partial Spearman correlation of X, Y controlling for Z."""
    coeffs_x = np.polyfit(z, x, 1)
    residuals_x = x - np.polyval(coeffs_x, z)
    coeffs_y = np.polyfit(z, y, 1)
    residuals_y = y - np.polyval(coeffs_y, z)
    return spearman_correlation(residuals_x, residuals_y)


def _compute_pareto_set(objectives: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Compute Pareto-optimal set (all objectives maximized).

    Parameters
    ----------
    objectives : NDArray, shape (n, m)
        Objective values; all to be maximized.

    Returns
    -------
    NDArray[np.bool_], shape (n,)
        True for Pareto-optimal rows.
    """
    n = len(objectives)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_pareto[i] = False
                break
    return is_pareto


def _normalize_min_max(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Min-max normalize values to [0, 1]."""
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax - vmin < 1e-12:
        return np.full_like(values, 0.5)
    return (values - vmin) / (vmax - vmin)


def _classify_scaling(values: dict[int, float]) -> str:
    """Classify scaling behavior as increasing/decreasing/flat/non_monotonic."""
    if len(values) < 2:
        return "insufficient_data"
    sorted_nf = sorted(values.keys())
    sorted_vals = [values[nf] for nf in sorted_nf]
    diffs = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
    if all(d > 0.001 for d in diffs):
        return "increasing"
    if all(d < -0.001 for d in diffs):
        return "decreasing"
    if np.std(sorted_vals) < 0.01:
        return "flat"
    return "non_monotonic"


def _write_json(data: dict[str, Any], path: str) -> None:
    """Write a JSON file with schema_version and numpy-safe serialization."""
    from experiments.results import _json_default
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data.setdefault("schema_version", "1.0")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _get_available_datasets(profiles: dict[str, dict[str, Any]]) -> list[str]:
    """Get the list of datasets that have VQC accuracy data."""
    all_ds: set[str] = set()
    for p in profiles.values():
        vqc_acc = p.get("vqc_accuracy")
        if vqc_acc:
            all_ds.update(vqc_acc.keys())
    return sorted(all_ds)


# ── Sub-Analysis Functions (7.1–7.10) ─────────────────────────────

def analyze_accuracy_vs_depth(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """7.1: Spearman correlation between circuit depth and VQC accuracy."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_dataset: dict[str, Any] = {}
    all_depths: list[float] = []
    all_accs: list[float] = []
    all_names: list[str] = []

    for ds in datasets:
        depths, accs, names = [], [], []
        for name, p in profiles.items():
            depth = p.get("depth")
            vqc_acc = (p.get("vqc_accuracy") or {}).get(ds)
            if depth is not None and vqc_acc is not None:
                depths.append(float(depth))
                accs.append(float(vqc_acc))
                names.append(name)
        if len(depths) < MIN_SAMPLES_FOR_TEST:
            per_dataset[ds] = {"status": "insufficient_data", "n_available": len(depths)}
            continue
        rho, p_val = spearman_correlation(np.array(depths), np.array(accs))
        per_dataset[ds] = {
            "rho": rho, "p_value": p_val, "n_encodings": len(depths),
            "encodings": names, "depths": depths, "accuracies": accs,
        }

    # Overall: mean accuracy across datasets per encoding
    for name, p in profiles.items():
        depth = p.get("depth")
        vqc_acc = p.get("vqc_accuracy")
        if depth is not None and vqc_acc:
            mean_acc = float(np.mean(list(vqc_acc.values())))
            all_depths.append(float(depth))
            all_accs.append(mean_acc)
            all_names.append(name)

    overall: dict[str, Any] = {}
    if len(all_depths) >= MIN_SAMPLES_FOR_TEST:
        rho, p_val = spearman_correlation(np.array(all_depths), np.array(all_accs))
        overall = {"rho": rho, "p_value": p_val, "n_encodings": len(all_depths)}
    else:
        overall = {"status": "insufficient_data", "n_available": len(all_depths)}

    # Interpretation
    rho_val = overall.get("rho", 0)
    p_val = overall.get("p_value", 1)
    if abs(rho_val) < 0.3:
        interp = f"Weak correlation (rho={rho_val:.2f}, p={p_val:.3f}): depth has limited impact on accuracy."
    elif rho_val > 0:
        interp = f"Positive correlation (rho={rho_val:.2f}, p={p_val:.3f}): deeper circuits show modest accuracy improvement."
    else:
        interp = f"Negative correlation (rho={rho_val:.2f}, p={p_val:.3f}): deeper circuits may suffer from trainability issues."

    return {"per_dataset": per_dataset, "overall": overall, "interpretation": interp}


def analyze_accuracy_vs_trainability(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """7.2: Spearman + partial Spearman between trainability and accuracy."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_dataset: dict[str, Any] = {}
    all_trains: list[float] = []
    all_accs: list[float] = []
    all_depths: list[float] = []

    for ds in datasets:
        trains, accs, depths, names = [], [], [], []
        for name, p in profiles.items():
            train = p.get("trainability_estimate")
            depth = p.get("depth")
            vqc_acc = (p.get("vqc_accuracy") or {}).get(ds)
            if train is not None and vqc_acc is not None and depth is not None:
                trains.append(float(train))
                accs.append(float(vqc_acc))
                depths.append(float(depth))
                names.append(name)
        if len(trains) < MIN_SAMPLES_FOR_TEST:
            per_dataset[ds] = {"status": "insufficient_data", "n_available": len(trains)}
            continue
        rho, p_val = spearman_correlation(np.array(trains), np.array(accs))
        partial_rho, partial_p = _partial_spearman(
            np.array(trains), np.array(accs), np.array(depths),
        )
        per_dataset[ds] = {
            "rho": rho, "p_value": p_val,
            "partial_rho": partial_rho, "partial_p_value": partial_p,
            "n_encodings": len(trains),
        }

    # Overall
    for name, p in profiles.items():
        train = p.get("trainability_estimate")
        depth = p.get("depth")
        vqc_acc = p.get("vqc_accuracy")
        if train is not None and vqc_acc and depth is not None:
            all_trains.append(float(train))
            all_accs.append(float(np.mean(list(vqc_acc.values()))))
            all_depths.append(float(depth))

    overall: dict[str, Any] = {}
    if len(all_trains) >= MIN_SAMPLES_FOR_TEST:
        rho, p_val = spearman_correlation(np.array(all_trains), np.array(all_accs))
        partial_rho, partial_p = _partial_spearman(
            np.array(all_trains), np.array(all_accs), np.array(all_depths),
        )
        overall = {
            "rho": rho, "p_value": p_val,
            "partial_rho": partial_rho, "partial_p_value": partial_p,
        }
    else:
        overall = {"status": "insufficient_data"}

    return {"per_dataset": per_dataset, "overall": overall}


def analyze_expressibility_vs_accuracy(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """7.3 / H1: Spearman between expressibility and accuracy (entangling only)."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_dataset: dict[str, Any] = {}
    all_exprs: list[float] = []
    all_accs: list[float] = []
    all_names: list[str] = []

    for ds in datasets:
        exprs, accs, names = [], [], []
        for name, p in profiles.items():
            if not p.get("is_entangling"):
                continue
            expr = p.get("expressibility")
            vqc_acc = (p.get("vqc_accuracy") or {}).get(ds)
            if expr is not None and vqc_acc is not None:
                exprs.append(float(expr))
                accs.append(float(vqc_acc))
                names.append(name)
        if len(exprs) < MIN_SAMPLES_FOR_TEST:
            per_dataset[ds] = {"status": "insufficient_data", "n_available": len(exprs)}
            continue
        rho, p_val = spearman_correlation(np.array(exprs), np.array(accs))
        per_dataset[ds] = {
            "rho": rho, "p_value": p_val, "n_encodings": len(exprs),
            "encodings": names, "expressibilities": exprs, "accuracies": accs,
        }

    # Overall
    for name, p in profiles.items():
        if not p.get("is_entangling"):
            continue
        expr = p.get("expressibility")
        vqc_acc = p.get("vqc_accuracy")
        if expr is not None and vqc_acc:
            all_exprs.append(float(expr))
            all_accs.append(float(np.mean(list(vqc_acc.values()))))
            all_names.append(name)

    overall: dict[str, Any] = {}
    h1_verdict: dict[str, Any] = {}

    if len(all_exprs) < MIN_SAMPLES_FOR_TEST:
        overall = {"status": "insufficient_data", "n_available": len(all_exprs)}
        h1_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": f"Insufficient data ({len(all_exprs)} entangling encodings, need {MIN_SAMPLES_FOR_TEST})",
            "test_statistic": {},
        }
    else:
        rho, p_val = spearman_correlation(np.array(all_exprs), np.array(all_accs))
        overall = {"rho": rho, "p_value": p_val, "n_encodings": len(all_exprs)}

        # H1 verdict logic
        median_expr = float(np.median(all_exprs))
        median_acc = float(np.median(all_accs))
        high_expr_low_acc = [
            all_names[i] for i in range(len(all_names))
            if all_exprs[i] > median_expr and all_accs[i] < median_acc
        ]

        if rho > 0 and p_val < 0.10:
            if high_expr_low_acc:
                confidence = "high" if p_val < 0.01 else "moderate"
                h1_verdict = {
                    "verdict": "supported", "confidence": confidence,
                    "evidence": (
                        f"Positive correlation (rho={rho:.2f}, p={p_val:.3f}) but "
                        f"{', '.join(high_expr_low_acc)} have high expressibility with below-median "
                        f"accuracy, confirming necessary but not sufficient."
                    ),
                    "test_statistic": {"spearman_rho": rho, "p_value": p_val},
                }
            else:
                all_high_expr_have_high_acc = all(
                    all_accs[i] >= median_acc
                    for i in range(len(all_names))
                    if all_exprs[i] > median_expr
                )
                if all_high_expr_have_high_acc:
                    h1_verdict = {
                        "verdict": "refuted", "confidence": "moderate",
                        "evidence": "All expressive encodings achieve high accuracy — expressibility appears sufficient.",
                        "test_statistic": {"spearman_rho": rho, "p_value": p_val},
                    }
                else:
                    h1_verdict = {
                        "verdict": "supported", "confidence": "moderate" if p_val < 0.05 else "low",
                        "evidence": f"Positive correlation (rho={rho:.2f}) suggests necessity.",
                        "test_statistic": {"spearman_rho": rho, "p_value": p_val},
                    }
        elif rho <= 0:
            h1_verdict = {
                "verdict": "refuted", "confidence": "moderate",
                "evidence": f"No positive correlation between expressibility and accuracy (rho={rho:.2f}).",
                "test_statistic": {"spearman_rho": rho, "p_value": p_val},
            }
        else:
            h1_verdict = {
                "verdict": "inconclusive", "confidence": "low",
                "evidence": f"Weak positive trend (rho={rho:.2f}) but not significant (p={p_val:.3f}).",
                "test_statistic": {"spearman_rho": rho, "p_value": p_val},
            }

    return {"per_dataset": per_dataset, "overall": overall, "h1_verdict": h1_verdict}


def analyze_equivariant_vs_general(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """7.4 / H2: Mann-Whitney comparing equivariant vs general on symmetric datasets."""
    symmetric_results: dict[str, Any] = {}
    non_symmetric_results: dict[str, Any] = {}
    all_p_values: list[float] = []
    all_p_indices: list[tuple[str, str]] = []  # (type, dataset)

    all_datasets = _get_available_datasets(profiles)
    symmetric_datasets = [ds for ds in all_datasets if ds in SYMMETRY_MATCHING]
    non_symmetric_datasets = [ds for ds in all_datasets if ds not in SYMMETRY_MATCHING]

    for ds in symmetric_datasets:
        matching_equivariants = SYMMETRY_MATCHING.get(ds, [])
        equi_folds: list[float] = []
        general_folds: list[float] = []
        for name, p in profiles.items():
            folds = (p.get("vqc_folds") or {}).get(ds, [])
            if not folds:
                continue
            if name in matching_equivariants:
                equi_folds.extend(folds)
            else:
                general_folds.extend(folds)

        if len(equi_folds) < 3 or len(general_folds) < 3:
            symmetric_results[ds] = {
                "status": "insufficient_data",
                "n_equivariant": len(equi_folds),
                "n_general": len(general_folds),
            }
            continue
        u_stat, p_val, rank_bis = mann_whitney_test(
            np.array(equi_folds), np.array(general_folds),
        )
        symmetric_results[ds] = {
            "equivariant_mean_acc": float(np.mean(equi_folds)),
            "general_mean_acc": float(np.mean(general_folds)),
            "u_stat": u_stat, "p_value": p_val, "rank_biserial": rank_bis,
            "significant": False,  # will be updated after correction
        }
        all_p_values.append(p_val)
        all_p_indices.append(("symmetric", ds))

    for ds in non_symmetric_datasets:
        equi_folds: list[float] = []
        general_folds: list[float] = []
        for name, p in profiles.items():
            folds = (p.get("vqc_folds") or {}).get(ds, [])
            if not folds:
                continue
            if p.get("is_equivariant"):
                equi_folds.extend(folds)
            else:
                general_folds.extend(folds)

        if len(equi_folds) < 3 or len(general_folds) < 3:
            non_symmetric_results[ds] = {
                "status": "insufficient_data",
                "n_equivariant": len(equi_folds),
                "n_general": len(general_folds),
            }
            continue
        u_stat, p_val, rank_bis = mann_whitney_test(
            np.array(equi_folds), np.array(general_folds),
        )
        non_symmetric_results[ds] = {
            "equivariant_mean_acc": float(np.mean(equi_folds)),
            "general_mean_acc": float(np.mean(general_folds)),
            "u_stat": u_stat, "p_value": p_val, "rank_biserial": rank_bis,
            "significant": False,
        }
        all_p_values.append(p_val)
        all_p_indices.append(("non_symmetric", ds))

    # Apply Holm-Bonferroni correction
    if all_p_values:
        rejected = holm_bonferroni_correction(all_p_values)
        for idx, (tp, ds) in enumerate(all_p_indices):
            target = symmetric_results if tp == "symmetric" else non_symmetric_results
            if ds in target and "significant" in target[ds]:
                target[ds]["significant"] = rejected[idx]

    # H2 verdict
    sym_wins = sum(
        1 for ds, r in symmetric_results.items()
        if r.get("significant") and r.get("equivariant_mean_acc", 0) > r.get("general_mean_acc", 0)
    )
    non_sym_wins = sum(
        1 for ds, r in non_symmetric_results.items()
        if r.get("significant") and r.get("equivariant_mean_acc", 0) > r.get("general_mean_acc", 0)
    )
    n_sym_tested = sum(1 for r in symmetric_results.values() if r.get("p_value") is not None)

    if sym_wins == 0 and n_sym_tested > 0:
        h2_verdict = {
            "verdict": "refuted", "confidence": "moderate",
            "evidence": f"General encodings match or exceed equivariant on all {n_sym_tested} symmetric datasets.",
            "test_statistic": {"sym_wins": sym_wins, "non_sym_wins": non_sym_wins},
        }
    elif sym_wins > 0 and non_sym_wins == 0:
        h2_verdict = {
            "verdict": "supported",
            "confidence": "moderate" if sym_wins >= 2 else "low",
            "evidence": (
                f"Equivariant encodings outperform on {sym_wins} symmetric datasets "
                f"but not on non-symmetric datasets, confirming symmetry-specific advantage."
            ),
            "test_statistic": {"sym_wins": sym_wins, "non_sym_wins": non_sym_wins},
        }
    elif sym_wins > 0 and non_sym_wins > 0:
        h2_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": (
                f"Equivariant encodings outperform on both symmetric ({sym_wins}) and "
                f"non-symmetric ({non_sym_wins}) datasets — advantage is not symmetry-specific."
            ),
            "test_statistic": {"sym_wins": sym_wins, "non_sym_wins": non_sym_wins},
        }
    else:
        h2_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": "Insufficient data for symmetric dataset comparisons.",
            "test_statistic": {},
        }

    return {
        "symmetric_datasets": symmetric_results,
        "non_symmetric_datasets": non_symmetric_results,
        "h2_verdict": h2_verdict,
        "limitation_note": "SO2EquivariantFeatureMap supports only n_features=2; comparison limited to toy datasets.",
    }


def analyze_barren_plateau_onset(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """7.5 / H4: Regression of log(gradient_variance) vs depth per encoding family."""
    all_depths: list[float] = []
    all_log_vars: list[float] = []
    all_families: list[str] = []
    all_names: list[str] = []

    # Collect data from scaling (multiple n_features per encoding)
    for name, p in profiles.items():
        train_scaling = p.get("trainability_scaling")
        depth_scaling = p.get("depth_scaling")
        family = p.get("family", "Unknown")
        if train_scaling and depth_scaling:
            for nf in train_scaling:
                gv = train_scaling[nf]
                depth = depth_scaling.get(nf)
                if gv is not None and gv > 0 and depth is not None:
                    all_depths.append(float(depth))
                    all_log_vars.append(float(np.log(gv)))
                    all_families.append(family)
                    all_names.append(name)
        else:
            # Use single-config data
            gv = p.get("gradient_variance")
            depth = p.get("depth")
            if gv is not None and gv > 0 and depth is not None:
                all_depths.append(float(depth))
                all_log_vars.append(float(np.log(gv)))
                all_families.append(family)
                all_names.append(name)

    if len(all_depths) < MIN_SAMPLES_FOR_TEST:
        return {"status": "insufficient_data", "n_available": len(all_depths)}

    # Per-family regression
    per_family: dict[str, Any] = {}
    unique_families = sorted(set(all_families))
    for fam in unique_families:
        mask = [i for i, f in enumerate(all_families) if f == fam]
        if len(mask) < 3:
            per_family[fam] = {"status": "insufficient_data", "n_points": len(mask)}
            continue
        fam_depths = np.array([all_depths[i] for i in mask])
        fam_log_vars = np.array([all_log_vars[i] for i in mask])
        if np.std(fam_depths) < 1e-12:
            per_family[fam] = {"status": "constant_depth", "n_points": len(mask)}
            continue
        coeffs = np.polyfit(fam_depths, fam_log_vars, 1)
        predicted = np.polyval(coeffs, fam_depths)
        ss_res = np.sum((fam_log_vars - predicted) ** 2)
        ss_tot = np.sum((fam_log_vars - np.mean(fam_log_vars)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        per_family[fam] = {
            "slope": float(coeffs[0]), "intercept": float(coeffs[1]),
            "r_squared": float(r_sq), "n_points": len(mask),
            "encodings": [all_names[i] for i in mask],
        }

    # Partial correlations
    depths_arr = np.array(all_depths)
    log_vars_arr = np.array(all_log_vars)

    # Encode family as integer
    fam_to_int = {f: i for i, f in enumerate(unique_families)}
    family_codes = np.array([fam_to_int[f] for f in all_families], dtype=float)

    depth_partial_corr, depth_partial_p = 0.0, 1.0
    family_partial_corr, family_partial_p = 0.0, 1.0

    if len(depths_arr) >= MIN_SAMPLES_FOR_TEST and np.std(family_codes) > 1e-12:
        depth_partial_corr, depth_partial_p = _partial_spearman(
            depths_arr, log_vars_arr, family_codes,
        )
        family_partial_corr, family_partial_p = _partial_spearman(
            family_codes, log_vars_arr, depths_arr,
        )

    # H4 verdict
    if abs(depth_partial_corr) > abs(family_partial_corr) and depth_partial_p < 0.05:
        h4_verdict = {
            "verdict": "supported",
            "confidence": "high" if depth_partial_p < 0.01 else "moderate",
            "evidence": (
                f"Depth is the dominant predictor of gradient variance "
                f"(partial rho={depth_partial_corr:.3f}, p={depth_partial_p:.4f}) "
                f"after controlling for encoding family "
                f"(family partial rho={family_partial_corr:.3f}, p={family_partial_p:.4f})."
            ),
            "test_statistic": {
                "depth_partial_rho": depth_partial_corr,
                "depth_partial_p": depth_partial_p,
                "family_partial_rho": family_partial_corr,
                "family_partial_p": family_partial_p,
            },
        }
    elif abs(family_partial_corr) > abs(depth_partial_corr) and family_partial_p < 0.05:
        h4_verdict = {
            "verdict": "refuted", "confidence": "moderate",
            "evidence": (
                f"Encoding family is the dominant predictor "
                f"(partial rho={family_partial_corr:.3f}, p={family_partial_p:.4f})."
            ),
            "test_statistic": {
                "depth_partial_rho": depth_partial_corr,
                "family_partial_rho": family_partial_corr,
            },
        }
    else:
        h4_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": (
                f"Both depth (partial rho={depth_partial_corr:.3f}) and family "
                f"(partial rho={family_partial_corr:.3f}) contribute comparably."
            ),
            "test_statistic": {
                "depth_partial_rho": depth_partial_corr,
                "family_partial_rho": family_partial_corr,
            },
        }

    return {
        "per_family": per_family,
        "depth_partial_corr": depth_partial_corr,
        "depth_partial_p": depth_partial_p,
        "family_partial_corr": family_partial_corr,
        "family_partial_p": family_partial_p,
        "h4_verdict": h4_verdict,
    }


def analyze_vqc_vs_kernel_ranking(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """7.6 / H6: Kendall tau between VQC and kernel accuracy rankings."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_dataset: dict[str, Any] = {}
    taus: list[float] = []

    for ds in datasets:
        vqc_accs: dict[str, float] = {}
        ker_accs: dict[str, float] = {}
        for name, p in profiles.items():
            v = (p.get("vqc_accuracy") or {}).get(ds)
            k = (p.get("kernel_accuracy") or {}).get(ds)
            if v is not None and k is not None:
                vqc_accs[name] = v
                ker_accs[name] = k

        common = sorted(set(vqc_accs) & set(ker_accs))
        if len(common) < MIN_SAMPLES_FOR_TEST:
            per_dataset[ds] = {"status": "insufficient_data", "n_common": len(common)}
            continue

        vqc_vals = [vqc_accs[n] for n in common]
        ker_vals = [ker_accs[n] for n in common]
        # Rank by accuracy descending (rank 1 = highest accuracy)
        vqc_ranks = np.argsort(np.argsort([-v for v in vqc_vals])) + 1
        ker_ranks = np.argsort(np.argsort([-k for k in ker_vals])) + 1

        tau, p_val = kendall_tau(
            np.array(vqc_ranks, dtype=float),
            np.array(ker_ranks, dtype=float),
        )
        taus.append(tau)

        vqc_ranking = [common[i] for i in np.argsort([-v for v in vqc_vals])]
        ker_ranking = [common[i] for i in np.argsort([-k for k in ker_vals])]

        per_dataset[ds] = {
            "tau": tau, "p_value": p_val, "n_encodings": len(common),
            "vqc_ranking": vqc_ranking, "kernel_ranking": ker_ranking,
            "top3_vqc": vqc_ranking[:3], "top3_kernel": ker_ranking[:3],
        }

    # Overall
    mean_tau = float(np.mean(taus)) if taus else 0.0
    n_significant = sum(
        1 for ds, r in per_dataset.items()
        if r.get("p_value") is not None and r["p_value"] < 0.05
    )
    n_datasets_tested = len(taus)

    # H6 verdict
    if n_datasets_tested < 2:
        h6_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": f"Only {n_datasets_tested} datasets with both VQC and kernel results.",
            "test_statistic": {"mean_tau": mean_tau},
        }
    elif abs(mean_tau) < 0.5 and n_significant > 0:
        h6_verdict = {
            "verdict": "supported",
            "confidence": "high" if n_significant >= 2 else "moderate",
            "evidence": (
                f"Rankings differ substantially (mean tau={mean_tau:.3f}, "
                f"significantly different on {n_significant}/{n_datasets_tested} datasets)."
            ),
            "test_statistic": {"mean_tau": mean_tau, "n_significant": n_significant},
        }
    elif abs(mean_tau) > 0.8:
        h6_verdict = {
            "verdict": "refuted", "confidence": "moderate",
            "evidence": f"Rankings are highly similar (mean tau={mean_tau:.3f}).",
            "test_statistic": {"mean_tau": mean_tau},
        }
    else:
        h6_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": f"Moderate correlation (mean tau={mean_tau:.3f}), unclear whether rankings are meaningfully different.",
            "test_statistic": {"mean_tau": mean_tau},
        }

    return {
        "per_dataset": per_dataset,
        "overall": {"mean_tau": mean_tau, "n_datasets": n_datasets_tested},
        "h6_verdict": h6_verdict,
    }


def analyze_pareto_front(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """7.7 / H7: Multi-objective Pareto front computation."""
    names: list[str] = []
    obj_rows: list[list[float]] = []

    for name, p in profiles.items():
        vqc_acc = p.get("vqc_accuracy")
        depth = p.get("depth")
        train = p.get("trainability_estimate")
        if vqc_acc is None or depth is None or train is None:
            continue
        if depth <= 0:
            continue
        mean_acc = float(np.mean(list(vqc_acc.values())))
        inv_depth = 1.0 / float(depth)
        noise_res = p.get("noise_resilience")
        if noise_res is None:
            noise_res = 0.5  # impute with neutral value
        names.append(name)
        obj_rows.append([mean_acc, inv_depth, float(train), float(noise_res)])

    if len(names) < 3:
        return {
            "status": "insufficient_data", "n_available": len(names),
            "h7_verdict": {
                "verdict": "inconclusive", "confidence": "low",
                "evidence": f"Too few encodings ({len(names)}) for meaningful Pareto analysis.",
                "test_statistic": {},
            },
        }

    objectives = np.array(obj_rows)
    # Normalize each objective
    norm_objectives = np.column_stack([
        _normalize_min_max(objectives[:, i]) for i in range(objectives.shape[1])
    ])

    pareto_mask = _compute_pareto_set(norm_objectives)
    pareto_names = [names[i] for i in range(len(names)) if pareto_mask[i]]
    n_pareto = int(np.sum(pareto_mask))

    # Build per-encoding results
    encodings_result: dict[str, Any] = {}
    for i, name in enumerate(names):
        dominated_by = []
        if not pareto_mask[i]:
            for j in range(len(names)):
                if j != i and pareto_mask[j]:
                    if np.all(norm_objectives[j] >= norm_objectives[i]) and np.any(norm_objectives[j] > norm_objectives[i]):
                        dominated_by.append(names[j])
        encodings_result[name] = {
            "objectives": obj_rows[i],
            "objectives_normalized": norm_objectives[i].tolist(),
            "is_pareto": bool(pareto_mask[i]),
            "dominated_by": dominated_by,
        }

    # H7 verdict
    if n_pareto >= 3:
        h7_verdict = {
            "verdict": "supported",
            "confidence": "high" if n_pareto >= 5 else "moderate",
            "evidence": (
                f"Pareto front contains {n_pareto} encodings ({', '.join(pareto_names)}), "
                f"confirming no single encoding dominates all objectives."
            ),
            "test_statistic": {"n_pareto": n_pareto},
        }
    elif n_pareto == 1:
        h7_verdict = {
            "verdict": "refuted", "confidence": "moderate",
            "evidence": f"{pareto_names[0]} dominates all other encodings across all objectives.",
            "test_statistic": {"n_pareto": n_pareto},
        }
    else:
        h7_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": f"Only {n_pareto} Pareto-optimal encodings; borderline result.",
            "test_statistic": {"n_pareto": n_pareto},
        }

    return {
        "n_objectives": 4,
        "objective_names": ["accuracy", "inv_depth", "trainability", "noise_resilience"],
        "n_encodings_analyzed": len(names),
        "n_pareto_optimal": n_pareto,
        "pareto_optimal": pareto_names,
        "encodings": encodings_result,
        "h7_verdict": h7_verdict,
    }



SCORING_SPECS: list[MetricSpec] = [
    MetricSpec("accuracy", 0.30),
    MetricSpec("expressibility", 0.15),
    MetricSpec("trainability", 0.20),
    MetricSpec("inv_depth", 0.15),
    MetricSpec("entanglement", 0.05),
    MetricSpec("noise_resilience", 0.15),
]


def build_metric_readings(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collect the scoring metrics, recording *why* each absent one is absent.

    A non-entangling encoding has an entanglement capability of exactly 0, and
    the pipeline skips it rather than measuring a foregone conclusion. Treating
    that ``None`` as unknown and filling it with the column median credits
    angle, higher-order-angle and basis with the median capability of the
    encodings that *do* entangle (0.605), which is what previously produced the
    suite's only robustness claim. See encoding_atlas.benchmark.scoring.
    """
    values: dict[str, dict[str, float | None]] = {}
    structural: dict[str, list[str]] = {}
    for name, p in profiles.items():
        vqc_acc = p.get("vqc_accuracy")
        if not vqc_acc:
            continue
        depth = p.get("depth")
        values[name] = {
            "accuracy": float(np.mean(list(vqc_acc.values()))),
            "expressibility": p.get("expressibility"),
            "trainability": p.get("trainability_estimate"),
            "inv_depth": (1.0 / float(depth)) if depth and depth > 0 else None,
            "entanglement": p.get("entanglement_capability"),
            "noise_resilience": p.get("noise_resilience"),
        }
        if p.get("is_entangling") is False:
            structural.setdefault(name, []).append("entanglement")
    return readings_from_values(
        values,
        structural_zeros=structural,
        reason="no entangling gates, so the capability is exactly 0",
    )


def analyze_ranking_sensitivity(
    profiles: dict[str, dict[str, Any]],
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """7.8: Monte Carlo weight sensitivity analysis.

    Absent metrics are resolved by encoding_atlas.benchmark.scoring: a
    structural zero counts as 0 and normalises with everything else, while a
    genuinely unmeasured metric is dropped from that encoding's weighting and
    reported in ``effective_weight``. Nothing is imputed.

    Weight vectors come from numpy's legacy RandomState, whose stream NumPy
    holds stable across releases, so these numbers reproduce on any supported
    NumPy version.
    """
    readings = build_metric_readings(profiles)
    if len(readings) < 3:
        return {"status": "insufficient_data", "n_available": len(readings)}

    result = weight_sensitivity(
        readings, SCORING_SPECS, n_samples=n_samples, seed=seed, k=3, threshold=0.90
    )

    per_encoding: dict[str, Any] = {
        name: {
            "mean_rank": result.mean_rank[name],
            "std_rank": result.std_rank[name],
            "min_rank": result.best_rank[name],
            "max_rank": result.worst_rank[name],
            "pct_top3": result.top_k_fraction[name] * 100.0,
        }
        for name in result.mean_rank
    }

    return {
        "per_encoding": per_encoding,
        "robustly_top3": list(result.robust),
        "default_ranking": list(result.equal_weight_ranking),
        "n_samples": result.n_samples,
        "metric_keys": [spec.name for spec in SCORING_SPECS],
        # Added so a reduced objective is visible rather than absorbed.
        "seed": result.seed,
        "rng": "numpy.random.RandomState (stream-stable across NumPy releases)",
        "effective_weight": dict(result.effective_weight),
        "unusable_metrics": {k: list(v) for k, v in result.unusable.items()},
        "structural_zeros": {k: list(v) for k, v in result.structural_zeros.items()},
    }


def compute_final_ranking(
    profiles: dict[str, dict[str, Any]],
    pareto_result: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """7.9: Final ranking table with weighted score."""
    if weights is None:
        weights = {
            "accuracy": 0.30, "expressibility": 0.15, "trainability": 0.20,
            "inv_depth": 0.15, "entanglement": 0.05, "noise_resilience": 0.15,
        }

    names: list[str] = [n for n, p in profiles.items() if p.get("vqc_accuracy")]

    if not names:
        return {"rankings": [], "weights_used": weights, "normalization_ranges": {}}

    # Score through the shared policy so 7.8 and 7.9 resolve absent metrics the
    # same way. Previously this omitted a missing metric and renormalised while
    # 7.8 imputed the column median, so the two analyses ranked the same data
    # against two different objectives.
    scoring = score_encodings(
        build_metric_readings(profiles),
        [MetricSpec(k, w) for k, w in weights.items()],
    )
    norm_ranges = {k: list(v) for k, v in scoring.normalization.items()}
    names = [n for n in scoring.scores if n in set(names)]

    rankings: list[dict[str, Any]] = []
    pareto_encodings = pareto_result.get("encodings", {})

    for name in names:
        p = profiles[name]
        score = scoring.scores.get(name, 0.0)

        vqc_acc = p.get("vqc_accuracy", {})
        vqc_ci_data = p.get("vqc_ci", {})
        kernel_acc = p.get("kernel_accuracy", {})
        kernel_ci_data = p.get("kernel_ci", {})

        mean_vqc = float(np.mean(list(vqc_acc.values()))) if vqc_acc else None
        mean_kernel = float(np.mean(list(kernel_acc.values()))) if kernel_acc else None

        # CI: average across datasets
        vqc_ci_avg = None
        if vqc_ci_data:
            lows = [c[0] for c in vqc_ci_data.values()]
            highs = [c[1] for c in vqc_ci_data.values()]
            vqc_ci_avg = [float(np.mean(lows)), float(np.mean(highs))]
        kernel_ci_avg = None
        if kernel_ci_data:
            lows = [c[0] for c in kernel_ci_data.values()]
            highs = [c[1] for c in kernel_ci_data.values()]
            kernel_ci_avg = [float(np.mean(lows)), float(np.mean(highs))]

        is_pareto = pareto_encodings.get(name, {}).get("is_pareto", False)
        footnotes: list[str] = []
        if name == "so2_equivariant":
            footnotes.append("n_features=2 only")
        if name == "trainable_encoding":
            footnotes.append("Encoding parameters not jointly optimized")

        rankings.append({
            "encoding": name,
            "class_name": p.get("class_name", ""),
            "family": p.get("family", ""),
            "is_simulable": p.get("is_simulable"),
            "depth": p.get("depth"),
            "expressibility": p.get("expressibility"),
            "entanglement_capability": p.get("entanglement_capability"),
            "trainability_estimate": p.get("trainability_estimate"),
            "noise_resilience": p.get("noise_resilience"),
            "vqc_accuracy": mean_vqc,
            "vqc_ci": vqc_ci_avg,
            "kernel_accuracy": mean_kernel,
            "kernel_ci": kernel_ci_avg,
            "is_pareto": is_pareto,
            "score": round(score, 4),
            "n_features_used": p.get("n_features_used", DEFAULT_N_FEATURES),
            "footnotes": footnotes,
        })

    rankings.sort(key=lambda r: r["score"], reverse=True)
    for rank_idx, r in enumerate(rankings):
        r["rank"] = rank_idx + 1

    return {
        "rankings": rankings,
        "weights_used": weights,
        "normalization_ranges": norm_ranges,
        # Fraction of the intended weight each encoding was actually judged on.
        # Below 1 means it was ranked against a different objective than its
        # rivals, which must be visible rather than absorbed into the score.
        "effective_weight": dict(scoring.effective_weight),
        "unusable_metrics": {k: list(v) for k, v in scoring.unusable.items()},
        "structural_zeros": {k: list(v) for k, v in scoring.structural_zeros.items()},
    }


def analyze_quantum_vs_classical(
    profiles: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """7.10: Compare best quantum encoding accuracy against classical baselines."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_dataset: dict[str, Any] = {}
    n_quantum_wins = 0

    for ds in datasets:
        # Find best quantum encoding for this dataset
        best_q_name = None
        best_q_acc = -1.0
        best_q_folds: list[float] = []
        for name, p in profiles.items():
            acc = (p.get("vqc_accuracy") or {}).get(ds)
            if acc is not None and acc > best_q_acc:
                best_q_acc = acc
                best_q_name = name
                best_q_folds = (p.get("vqc_folds") or {}).get(ds, [])

        if best_q_name is None:
            per_dataset[ds] = {"status": "no_quantum_data"}
            continue

        bl_results: dict[str, Any] = {}
        all_bl_p_values: list[float] = []
        all_bl_keys: list[str] = []

        for bl_name, bl_data in baselines.items():
            bl_acc = bl_data.get("accuracy", {}).get(ds)
            bl_folds_data = bl_data.get("folds", {}).get(ds, [])
            bl_result: dict[str, Any] = {"acc": bl_acc}

            if bl_acc is not None and best_q_folds and bl_folds_data:
                # Truncate to same length for paired test
                min_len = min(len(best_q_folds), len(bl_folds_data))
                if min_len >= MIN_SAMPLES_FOR_TEST:
                    q_arr = np.array(best_q_folds[:min_len])
                    b_arr = np.array(bl_folds_data[:min_len])
                    stat, p_val = wilcoxon_test(q_arr, b_arr)
                    delta = cliffs_delta(q_arr, b_arr)
                    bl_result["wilcoxon_p"] = p_val
                    bl_result["cliffs_delta"] = delta
                    bl_result["effect_size"] = interpret_cliffs_delta(delta)
                    all_bl_p_values.append(p_val)
                    all_bl_keys.append(bl_name)

            bl_results[bl_name] = bl_result

        # Apply Holm-Bonferroni across baselines
        if all_bl_p_values:
            rejected = holm_bonferroni_correction(all_bl_p_values)
            for idx, bl_key in enumerate(all_bl_keys):
                bl_results[bl_key]["significant"] = rejected[idx]

        # Determine if quantum wins
        quantum_wins = best_q_acc > max(
            (bl_data.get("accuracy", {}).get(ds, 0) or 0)
            for bl_data in baselines.values()
        ) if baselines else False
        has_significant = any(
            bl_results.get(k, {}).get("significant", False) for k in all_bl_keys
        )
        quantum_wins = quantum_wins and has_significant
        if quantum_wins:
            n_quantum_wins += 1

        per_dataset[ds] = {
            "best_quantum": best_q_name,
            "best_quantum_acc": best_q_acc,
            "baselines": bl_results,
            "quantum_wins": quantum_wins,
        }

    return {
        "per_dataset": per_dataset,
        "summary": {
            "n_datasets_quantum_wins": n_quantum_wins,
            "total_datasets": len(datasets),
        },
    }


# ── Additional Analysis Functions (4.1–4.10) ──────────────────────

def analyze_family_aggregate(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """4.1: Encoding family aggregate comparison with Kruskal-Wallis test."""
    from scipy.stats import kruskal

    metrics = ["accuracy", "expressibility", "trainability", "inv_depth", "entanglement", "noise_resilience"]
    per_metric: dict[str, Any] = {}

    for metric in metrics:
        family_groups: dict[str, list[float]] = {}
        for name, p in profiles.items():
            family = p.get("family", "Unknown")
            if metric == "accuracy":
                vqc_acc = p.get("vqc_accuracy")
                val = float(np.mean(list(vqc_acc.values()))) if vqc_acc else None
            elif metric == "inv_depth":
                depth = p.get("depth")
                val = 1.0 / float(depth) if depth and depth > 0 else None
            elif metric == "entanglement":
                val = p.get("entanglement_capability")
            elif metric == "noise_resilience":
                val = p.get("noise_resilience")
            else:
                val = p.get(metric) if metric != "trainability" else p.get("trainability_estimate")
            if val is not None:
                family_groups.setdefault(family, []).append(float(val))

        family_stats: dict[str, Any] = {}
        for fam, vals in family_groups.items():
            family_stats[fam] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

        # Kruskal-Wallis test
        valid_groups = [vals for vals in family_groups.values() if len(vals) >= 2]
        kw_result: dict[str, Any] = {}
        if len(valid_groups) >= 3:
            try:
                h_stat, kw_p = kruskal(*valid_groups)
                kw_result = {"kruskal_wallis_h": float(h_stat), "kruskal_wallis_p": float(kw_p)}
            except Exception as e:
                kw_result = {"kruskal_wallis_error": str(e)}

        per_metric[metric] = {
            "family_stats": family_stats,
            **kw_result,
        }

    return {"per_metric": per_metric, "metrics_analyzed": metrics}


def analyze_data_reuploading_advantage(
    profiles: dict[str, dict[str, Any]],
    stage3_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """4.2 / H3: Data re-uploading expressibility advantage at matched depth."""
    matched_pairs: list[dict[str, Any]] = []

    for dr_reg, dr_params, match_reg, match_params in _MATCHED_PAIRS:
        for nf in [2, 4]:
            # Find DataReuploading expressibility
            dr_expr = None
            for r in stage3_results:
                if canonicalize_encoding_name(r["encoding_name"]) != dr_reg:
                    continue
                rp = r.get("encoding_params", {})
                if rp.get("n_features") != nf:
                    continue
                match = all(rp.get(k) == v for k, v in dr_params.items())
                if match:
                    dr_expr = r.get("result", {}).get("expressibility")
                    break

            # Find matched encoding expressibility
            matched_expr = None
            for r in stage3_results:
                if canonicalize_encoding_name(r["encoding_name"]) != match_reg:
                    continue
                rp = r.get("encoding_params", {})
                if rp.get("n_features") != nf:
                    continue
                match = all(rp.get(k) == v for k, v in match_params.items())
                if match:
                    matched_expr = r.get("result", {}).get("expressibility")
                    break

            if dr_expr is not None and matched_expr is not None:
                matched_pairs.append({
                    "dr_config": f"{dr_reg}(n_features={nf}, {dr_params})",
                    "matched_config": f"{match_reg}(n_features={nf}, {match_params})",
                    "dr_expr": float(dr_expr),
                    "matched_expr": float(matched_expr),
                    "diff": float(dr_expr) - float(matched_expr),
                })

    n_pairs = len(matched_pairs)
    if n_pairs < 4:
        h3_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": f"Only {n_pairs} matched pairs available (need at least 4).",
            "test_statistic": {},
        }
        return {"matched_pairs": matched_pairs, "h3_verdict": h3_verdict}

    diffs = [mp["diff"] for mp in matched_pairs]
    positive_pairs = sum(1 for d in diffs if d > 0)

    # Wilcoxon test on differences
    wilcoxon_p = 1.0
    if n_pairs >= MIN_SAMPLES_FOR_TEST:
        dr_vals = np.array([mp["dr_expr"] for mp in matched_pairs])
        match_vals = np.array([mp["matched_expr"] for mp in matched_pairs])
        _, wilcoxon_p = wilcoxon_test(dr_vals, match_vals)

    if positive_pairs >= 3 and wilcoxon_p < 0.05:
        h3_verdict = {
            "verdict": "supported", "confidence": "moderate",
            "evidence": (
                f"DataReuploading achieves higher expressibility in {positive_pairs}/{n_pairs} "
                f"matched pairs (Wilcoxon p={wilcoxon_p:.4f})."
            ),
            "test_statistic": {"wilcoxon_p": wilcoxon_p, "positive_pairs": positive_pairs, "total_pairs": n_pairs},
        }
    elif positive_pairs <= 1:
        h3_verdict = {
            "verdict": "refuted", "confidence": "moderate",
            "evidence": (
                f"Single-pass encodings match or exceed DataReuploading in "
                f"{n_pairs - positive_pairs}/{n_pairs} pairs."
            ),
            "test_statistic": {"wilcoxon_p": wilcoxon_p, "positive_pairs": positive_pairs},
        }
    else:
        h3_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": (
                f"Mixed results: DataReuploading wins {positive_pairs}/{n_pairs} pairs "
                f"but not statistically significant (p={wilcoxon_p:.4f})."
            ),
            "test_statistic": {"wilcoxon_p": wilcoxon_p, "positive_pairs": positive_pairs},
        }

    return {"matched_pairs": matched_pairs, "wilcoxon_p": wilcoxon_p, "h3_verdict": h3_verdict}


def analyze_noise_sensitivity_by_family(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """4.3 / H5: Mann-Whitney comparing noise fidelity decay between entangling/non-entangling."""
    entangling_decays: list[float] = []
    non_entangling_decays: list[float] = []

    for name, p in profiles.items():
        decay = p.get("noisy_medium_fidelity_decay")
        if decay is None:
            continue
        if p.get("is_entangling"):
            entangling_decays.append(float(decay))
        else:
            non_entangling_decays.append(float(decay))

    if len(entangling_decays) < 3 or len(non_entangling_decays) < 3:
        h5_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": (
                f"Insufficient data: {len(entangling_decays)} entangling, "
                f"{len(non_entangling_decays)} non-entangling encodings."
            ),
            "test_statistic": {},
        }
        return {
            "entangling_mean": float(np.mean(entangling_decays)) if entangling_decays else None,
            "non_entangling_mean": float(np.mean(non_entangling_decays)) if non_entangling_decays else None,
            "h5_verdict": h5_verdict,
        }

    ent_mean = float(np.mean(entangling_decays))
    non_ent_mean = float(np.mean(non_entangling_decays))
    u_stat, p_val, rank_bis = mann_whitney_test(
        np.array(entangling_decays), np.array(non_entangling_decays),
    )

    if ent_mean > non_ent_mean and p_val < 0.05:
        h5_verdict = {
            "verdict": "supported", "confidence": "moderate",
            "evidence": (
                f"Entangling encodings have significantly higher fidelity decay "
                f"(mean={ent_mean:.4f}) vs non-entangling (mean={non_ent_mean:.4f}), "
                f"Mann-Whitney p={p_val:.4f}, r={rank_bis:.3f}."
            ),
            "test_statistic": {"u_stat": u_stat, "p_value": p_val, "rank_biserial": rank_bis},
        }
    elif non_ent_mean >= ent_mean:
        h5_verdict = {
            "verdict": "refuted", "confidence": "moderate",
            "evidence": "Non-entangling encodings degrade at equal or higher rate.",
            "test_statistic": {"u_stat": u_stat, "p_value": p_val, "rank_biserial": rank_bis},
        }
    else:
        h5_verdict = {
            "verdict": "inconclusive", "confidence": "low",
            "evidence": f"Trend in expected direction but not significant (p={p_val:.4f}).",
            "test_statistic": {"u_stat": u_stat, "p_value": p_val, "rank_biserial": rank_bis},
        }

    return {
        "entangling_mean": ent_mean,
        "non_entangling_mean": non_ent_mean,
        "u_stat": u_stat, "p_value": p_val, "rank_biserial": rank_bis,
        "h5_verdict": h5_verdict,
    }


def analyze_kta_vs_accuracy(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """4.4: Spearman correlation between centered KTA and kernel-SVM accuracy."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_dataset: dict[str, Any] = {}
    all_ktas: list[float] = []
    all_accs: list[float] = []

    for ds in datasets:
        ktas, accs = [], []
        for name, p in profiles.items():
            kta = (p.get("kernel_kta") or {}).get(ds)
            acc = (p.get("kernel_accuracy") or {}).get(ds)
            if kta is not None and acc is not None:
                ktas.append(float(kta))
                accs.append(float(acc))
        if len(ktas) < MIN_SAMPLES_FOR_TEST:
            per_dataset[ds] = {"status": "insufficient_data", "n_available": len(ktas)}
            continue
        rho, p_val = spearman_correlation(np.array(ktas), np.array(accs))
        per_dataset[ds] = {"rho": rho, "p_value": p_val, "n_encodings": len(ktas)}
        all_ktas.extend(ktas)
        all_accs.extend(accs)

    overall: dict[str, Any] = {}
    kta_is_useful = False
    if len(all_ktas) >= MIN_SAMPLES_FOR_TEST:
        rho, p_val = spearman_correlation(np.array(all_ktas), np.array(all_accs))
        overall = {"rho": rho, "p_value": p_val}
        kta_is_useful = rho > 0.6
    else:
        overall = {"status": "insufficient_data"}

    return {"per_dataset": per_dataset, "overall": overall, "kta_is_useful_proxy": kta_is_useful}


def analyze_overfitting(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """4.5: Train-test gap analysis and correlation with depth."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_encoding: dict[str, Any] = {}
    depths: list[float] = []
    gaps: list[float] = []

    for name, p in profiles.items():
        vqc_acc = p.get("vqc_accuracy") or {}
        vqc_train = p.get("vqc_train_accuracy") or {}
        ds_gaps: list[float] = []
        for ds in datasets:
            train_acc = vqc_train.get(ds)
            test_acc = vqc_acc.get(ds)
            if train_acc is not None and test_acc is not None:
                ds_gaps.append(float(train_acc) - float(test_acc))
        if ds_gaps:
            mean_gap = float(np.mean(ds_gaps))
            per_encoding[name] = {"mean_gap": mean_gap, "flagged": mean_gap > 0.15}
            depth = p.get("depth")
            if depth is not None:
                depths.append(float(depth))
                gaps.append(mean_gap)

    gap_vs_depth: dict[str, Any] = {}
    if len(depths) >= MIN_SAMPLES_FOR_TEST:
        rho, p_val = spearman_correlation(np.array(depths), np.array(gaps))
        gap_vs_depth = {"rho": rho, "p_value": p_val}

    flagged = [name for name, data in per_encoding.items() if data.get("flagged")]

    return {
        "per_encoding": per_encoding,
        "gap_vs_depth_rho": gap_vs_depth.get("rho"),
        "gap_vs_depth_p": gap_vs_depth.get("p_value"),
        "flagged_encodings": flagged,
    }


def analyze_expressibility_entanglement_joint(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """4.6: Joint expressibility-entanglement scatter with quadrant analysis."""
    exprs: list[float] = []
    ents: list[float] = []
    accs: list[float] = []
    names: list[str] = []

    for name, p in profiles.items():
        if not p.get("is_entangling"):
            continue
        expr = p.get("expressibility")
        ent = p.get("entanglement_capability")
        vqc_acc = p.get("vqc_accuracy")
        if expr is not None and ent is not None:
            exprs.append(float(expr))
            ents.append(float(ent))
            mean_acc = float(np.mean(list(vqc_acc.values()))) if vqc_acc else None
            accs.append(mean_acc if mean_acc is not None else 0.0)
            names.append(name)

    if len(exprs) < MIN_SAMPLES_FOR_TEST:
        return {"status": "insufficient_data", "n_available": len(exprs)}

    rho, p_val = spearman_correlation(np.array(exprs), np.array(ents))
    median_expr = float(np.median(exprs))
    median_ent = float(np.median(ents))

    per_encoding: dict[str, Any] = {}
    quadrant_counts = {"HH": 0, "HL": 0, "LH": 0, "LL": 0}
    for i, name in enumerate(names):
        hi_expr = exprs[i] > median_expr
        hi_ent = ents[i] > median_ent
        q = ("H" if hi_expr else "L") + ("H" if hi_ent else "L")
        quadrant_counts[q] += 1
        per_encoding[name] = {
            "expr": exprs[i], "ent": ents[i],
            "quadrant": q, "mean_accuracy": accs[i],
        }

    return {
        "rho": rho, "p_value": p_val,
        "per_encoding": per_encoding,
        "quadrant_counts": quadrant_counts,
    }


def analyze_resource_efficiency(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """4.7: Accuracy / depth and accuracy / gate_count efficiency frontier."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    per_encoding: dict[str, Any] = {}
    gate_acc_pairs: list[tuple[float, float, str]] = []

    for name, p in profiles.items():
        depth = p.get("depth")
        gate_count = p.get("gate_count")
        vqc_acc = p.get("vqc_accuracy")
        if not vqc_acc or depth is None or gate_count is None:
            continue
        if depth <= 0 or gate_count <= 0:
            continue
        mean_acc = float(np.mean(list(vqc_acc.values())))
        acc_per_depth = mean_acc / float(depth)
        acc_per_gate = mean_acc / float(gate_count)
        per_encoding[name] = {
            "acc_per_depth": acc_per_depth,
            "acc_per_gate": acc_per_gate,
            "on_frontier": False,
        }
        gate_acc_pairs.append((float(gate_count), mean_acc, name))

    # Convex efficiency frontier: sort by gate_count, running max accuracy
    gate_acc_pairs.sort(key=lambda x: x[0])
    frontier_names: list[str] = []
    max_acc = -1.0
    for gc, acc, name in gate_acc_pairs:
        if acc >= max_acc:
            max_acc = acc
            frontier_names.append(name)
            if name in per_encoding:
                per_encoding[name]["on_frontier"] = True

    efficiency_ranking = sorted(
        per_encoding.keys(),
        key=lambda n: per_encoding[n]["acc_per_gate"],
        reverse=True,
    )

    return {
        "per_encoding": per_encoding,
        "frontier_encodings": frontier_names,
        "efficiency_ranking": efficiency_ranking,
    }


def analyze_scaling_behavior(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """4.8: Expressibility and entanglement scaling with n_features."""
    per_encoding: dict[str, Any] = {}
    concerning: list[str] = []

    for name, p in profiles.items():
        expr_scaling = p.get("expressibility_scaling")
        ent_scaling = p.get("entanglement_scaling")
        if not expr_scaling and not ent_scaling:
            continue

        entry: dict[str, Any] = {}
        if expr_scaling:
            entry["expr_scaling_class"] = _classify_scaling(expr_scaling)
            entry["expr_values"] = expr_scaling
            if entry["expr_scaling_class"] == "decreasing":
                concerning.append(name)
        if ent_scaling:
            entry["ent_scaling_class"] = _classify_scaling(ent_scaling)
            entry["ent_values"] = ent_scaling

        per_encoding[name] = entry

    return {
        "per_encoding": per_encoding,
        "concerning_encodings": concerning,
    }


def analyze_pairwise_comparisons_analysis(
    profiles: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """4.9: Full pairwise Wilcoxon tests with Holm-Bonferroni correction."""
    if datasets is None:
        datasets = _get_available_datasets(profiles)

    vqc_results: dict[str, Any] = {"per_dataset": {}}
    kernel_results: dict[str, Any] = {"per_dataset": {}}

    for ds in datasets:
        # VQC pairwise
        vqc_folds_dict: dict[str, NDArray[np.floating]] = {}
        for name, p in profiles.items():
            folds = (p.get("vqc_folds") or {}).get(ds, [])
            if len(folds) >= MIN_SAMPLES_FOR_TEST:
                vqc_folds_dict[name] = np.array(folds)

        if len(vqc_folds_dict) >= 2:
            pw = pairwise_comparisons(vqc_folds_dict)
            vqc_results["per_dataset"][ds] = pw

        # Kernel pairwise
        ker_folds_dict: dict[str, NDArray[np.floating]] = {}
        for name, p in profiles.items():
            folds = (p.get("kernel_folds") or {}).get(ds, [])
            if len(folds) >= MIN_SAMPLES_FOR_TEST:
                ker_folds_dict[name] = np.array(folds)

        if len(ker_folds_dict) >= 2:
            pw = pairwise_comparisons(ker_folds_dict)
            kernel_results["per_dataset"][ds] = pw

    return {"vqc": vqc_results, "kernel": kernel_results}


def analyze_convergence(
    profiles: dict[str, dict[str, Any]],
    stage6a_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """4.10: VQC convergence quality check — divergence rates and final loss."""
    per_encoding: dict[str, Any] = {}

    for r in stage6a_results:
        reg = canonicalize_encoding_name(r["encoding_name"])
        if reg.startswith("classical_"):
            continue
        res = r.get("result", {})
        datasets_data = res.get("datasets", {})
        if not datasets_data:
            continue

        total_folds = 0
        diverged_count = 0
        final_losses: list[float] = []

        for ds_name, ds_result in datasets_data.items():
            for run in ds_result.get("runs", []):
                for fold in run.get("folds", []):
                    total_folds += 1
                    status = fold.get("status", "")
                    if status == "diverged":
                        diverged_count += 1
                    fl = fold.get("final_loss")
                    if fl is not None:
                        final_losses.append(float(fl))

        if total_folds == 0:
            continue

        div_rate = diverged_count / total_folds
        mean_loss = float(np.mean(final_losses)) if final_losses else None

        # Only keep the entry with most data for each encoding
        if reg not in per_encoding or total_folds > per_encoding[reg].get("total_runs", 0):
            per_encoding[reg] = {
                "diverged_count": diverged_count,
                "total_runs": total_folds,
                "divergence_rate": div_rate,
                "mean_final_loss": mean_loss,
                "flagged": div_rate > 0.20,
            }

    # Correlation with trainability
    trains: list[float] = []
    div_rates: list[float] = []
    for name, data in per_encoding.items():
        train = profiles.get(name, {}).get("trainability_estimate")
        if train is not None:
            trains.append(float(train))
            div_rates.append(data["divergence_rate"])

    div_vs_train: dict[str, Any] = {}
    if len(trains) >= MIN_SAMPLES_FOR_TEST:
        rho, p_val = spearman_correlation(np.array(trains), np.array(div_rates))
        div_vs_train = {"rho": rho, "p_value": p_val}

    flagged = [name for name, data in per_encoding.items() if data.get("flagged")]

    return {
        "per_encoding": per_encoding,
        "divergence_vs_trainability_rho": div_vs_train.get("rho"),
        "divergence_vs_trainability_p": div_vs_train.get("p_value"),
        "flagged_encodings": flagged,
    }


# ── Hypothesis Verdict Computation ─────────────────────────────────

def compute_all_hypothesis_verdicts(
    results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute verdicts for all 7 hypotheses from sub-analysis results."""
    verdicts: dict[str, dict[str, Any]] = {}

    # H1 from 7.3
    r_7_3 = results.get("7.3", {})
    verdicts["H1"] = r_7_3.get("h1_verdict", {
        "verdict": "inconclusive", "confidence": "low",
        "evidence": "Sub-analysis 7.3 not available.", "test_statistic": {},
    })

    # H2 from 7.4
    r_7_4 = results.get("7.4", {})
    verdicts["H2"] = r_7_4.get("h2_verdict", {
        "verdict": "inconclusive", "confidence": "low",
        "evidence": "Sub-analysis 7.4 not available.", "test_statistic": {},
    })

    # H3 from 4.2
    r_4_2 = results.get("4.2", {})
    verdicts["H3"] = r_4_2.get("h3_verdict", {
        "verdict": "inconclusive", "confidence": "low",
        "evidence": "Sub-analysis 4.2 not available.", "test_statistic": {},
    })

    # H4 from 7.5
    r_7_5 = results.get("7.5", {})
    verdicts["H4"] = r_7_5.get("h4_verdict", {
        "verdict": "inconclusive", "confidence": "low",
        "evidence": "Sub-analysis 7.5 not available.", "test_statistic": {},
    })

    # H5 from 4.3
    r_4_3 = results.get("4.3", {})
    verdicts["H5"] = r_4_3.get("h5_verdict", {
        "verdict": "inconclusive", "confidence": "low",
        "evidence": "Sub-analysis 4.3 not available.", "test_statistic": {},
    })

    # H6 from 7.6
    r_7_6 = results.get("7.6", {})
    verdicts["H6"] = r_7_6.get("h6_verdict", {
        "verdict": "inconclusive", "confidence": "low",
        "evidence": "Sub-analysis 7.6 not available.", "test_statistic": {},
    })

    # H7 from 7.7
    r_7_7 = results.get("7.7", {})
    verdicts["H7"] = r_7_7.get("h7_verdict", {
        "verdict": "inconclusive", "confidence": "low",
        "evidence": "Sub-analysis 7.7 not available.", "test_statistic": {},
    })

    return verdicts


# ── Table Generation ───────────────────────────────────────────────

def _fmt(val: Any, decimals: int = 4) -> str:
    """Format a value for table display."""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _generate_ranking_table_md(ranking_data: dict[str, Any]) -> str:
    """Generate Markdown ranking table."""
    rankings = ranking_data.get("rankings", [])
    if not rankings:
        return "No rankings available."

    headers = [
        "Rank", "Encoding", "Family", "Sim.", "Depth", "Expr.", "Entang.",
        "Train.", "Noise", "VQC Acc", "Kernel Acc", "Pareto", "Score",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for r in rankings:
        vqc_str = _fmt(r.get("vqc_accuracy"), 3)
        if r.get("vqc_ci"):
            vqc_str += f" ({_fmt(r['vqc_ci'][0], 2)}-{_fmt(r['vqc_ci'][1], 2)})"
        ker_str = _fmt(r.get("kernel_accuracy"), 3)
        if r.get("kernel_ci"):
            ker_str += f" ({_fmt(r['kernel_ci'][0], 2)}-{_fmt(r['kernel_ci'][1], 2)})"

        name = r.get("encoding", "")
        if r.get("footnotes"):
            name += " " + " ".join(f"[{fn}]" for fn in r["footnotes"])

        row = [
            str(r.get("rank", "")),
            name,
            r.get("family", ""),
            _fmt(r.get("is_simulable")),
            _fmt(r.get("depth")),
            _fmt(r.get("expressibility"), 3),
            _fmt(r.get("entanglement_capability"), 3),
            _fmt(r.get("trainability_estimate"), 3),
            _fmt(r.get("noise_resilience"), 3),
            vqc_str,
            ker_str,
            "Yes" if r.get("is_pareto") else "No",
            _fmt(r.get("score"), 4),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _generate_ranking_table_tex(ranking_data: dict[str, Any]) -> str:
    """Generate LaTeX ranking table."""
    rankings = ranking_data.get("rankings", [])
    if not rankings:
        return "% No rankings available."

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Quantum Encoding Rankings}",
        r"\label{tab:rankings}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rlllrrrrrrrcc}",
        r"\toprule",
        r"Rank & Encoding & Family & Sim. & Depth & Expr. & Entang. & Train. & Noise & VQC Acc & Kernel Acc & Pareto & Score \\",
        r"\midrule",
    ]

    _checkmark = "$\\checkmark$"
    for r in rankings:
        name = r.get("encoding", "").replace("_", r"\_")
        fam = r.get("family", "").replace("&", r"\&")
        lines.append(
            f"  {r.get('rank', '')} & {name} & {fam} & "
            f"{_fmt(r.get('is_simulable'))} & "
            f"{_fmt(r.get('depth'))} & "
            f"{_fmt(r.get('expressibility'), 3)} & "
            f"{_fmt(r.get('entanglement_capability'), 3)} & "
            f"{_fmt(r.get('trainability_estimate'), 3)} & "
            f"{_fmt(r.get('noise_resilience'), 3)} & "
            f"{_fmt(r.get('vqc_accuracy'), 3)} & "
            f"{_fmt(r.get('kernel_accuracy'), 3)} & "
            f"{_checkmark if r.get('is_pareto') else ''} & "
            f"{_fmt(r.get('score'), 4)} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _generate_hypothesis_table_md(verdicts: dict[str, dict[str, Any]]) -> str:
    """Generate Markdown hypothesis verdict table."""
    headers = ["Hypothesis", "Verdict", "Confidence", "Evidence"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for h_key in sorted(verdicts.keys()):
        v = verdicts[h_key]
        lines.append(
            f"| {h_key} | {v.get('verdict', 'N/A')} | "
            f"{v.get('confidence', 'N/A')} | "
            f"{v.get('evidence', 'N/A')} |"
        )
    return "\n".join(lines)


def _generate_hypothesis_table_tex(verdicts: dict[str, dict[str, Any]]) -> str:
    """Generate LaTeX hypothesis verdict table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hypothesis Verdicts}",
        r"\label{tab:hypotheses}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Hypothesis & Verdict & Confidence & Evidence \\",
        r"\midrule",
    ]

    for h_key in sorted(verdicts.keys()):
        v = verdicts[h_key]
        evidence = v.get("evidence", "N/A")[:80]
        evidence = evidence.replace("&", r"\&").replace("_", r"\_")
        lines.append(
            f"  {h_key} & {v.get('verdict', 'N/A')} & "
            f"{v.get('confidence', 'N/A')} & "
            f"{evidence}... \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ── Master Orchestrator ────────────────────────────────────────────

def run_tradeoff_analysis(
    stage_dirs: dict[str, str],
    output_dir: str,
    figure_dir: str,
    seed: int = 42,
    generate_plots: bool = True,
    sensitivity_n_samples: int = 1000,
) -> dict[str, Any]:
    """Run the complete Stage 7 tradeoff analysis.

    Parameters
    ----------
    stage_dirs : dict[str, str]
        Mapping from stage key to results directory path.
    output_dir : str
        Directory for JSON output files.
    figure_dir : str
        Directory for figure output.
    seed : int
        Random seed for sensitivity analysis.
    generate_plots : bool
        If False, skip plot generation.
    sensitivity_n_samples : int
        Number of Monte Carlo samples for ranking sensitivity (7.8).

    Returns
    -------
    dict[str, Any]
        Complete results dict.
    """
    logger.info("Starting Stage 7 tradeoff analysis...")

    # 1. Build encoding profiles
    profiles = build_encoding_profiles(stage_dirs)
    logger.info("Built profiles for %d encodings", len(profiles))

    # 2. Extract classical baselines
    baselines = {}
    if "vqc" in stage_dirs:
        baselines = extract_classical_baselines(stage_dirs["vqc"])
        logger.info("Extracted %d classical baselines", len(baselines))

    # 3. Load raw stage3 results for H3
    stage3_results: list[dict[str, Any]] = []
    if "expressibility" in stage_dirs:
        stage3_results = load_stage_results(stage_dirs["expressibility"])

    # 4. Load raw stage6a results for convergence analysis
    stage6a_results: list[dict[str, Any]] = []
    if "vqc" in stage_dirs:
        stage6a_results = load_stage_results(stage_dirs["vqc"])

    # 5. Run all 20 sub-analyses
    results: dict[str, dict[str, Any]] = {}

    analyses = [
        ("7.1", lambda: analyze_accuracy_vs_depth(profiles)),
        ("7.2", lambda: analyze_accuracy_vs_trainability(profiles)),
        ("7.3", lambda: analyze_expressibility_vs_accuracy(profiles)),
        ("7.4", lambda: analyze_equivariant_vs_general(profiles)),
        ("7.5", lambda: analyze_barren_plateau_onset(profiles)),
        ("7.6", lambda: analyze_vqc_vs_kernel_ranking(profiles)),
        ("7.7", lambda: analyze_pareto_front(profiles)),
        ("7.8", lambda: analyze_ranking_sensitivity(profiles, n_samples=sensitivity_n_samples, seed=seed)),
        ("7.10", lambda: analyze_quantum_vs_classical(profiles, baselines)),
        ("4.1", lambda: analyze_family_aggregate(profiles)),
        ("4.2", lambda: analyze_data_reuploading_advantage(profiles, stage3_results)),
        ("4.3", lambda: analyze_noise_sensitivity_by_family(profiles)),
        ("4.4", lambda: analyze_kta_vs_accuracy(profiles)),
        ("4.5", lambda: analyze_overfitting(profiles)),
        ("4.6", lambda: analyze_expressibility_entanglement_joint(profiles)),
        ("4.7", lambda: analyze_resource_efficiency(profiles)),
        ("4.8", lambda: analyze_scaling_behavior(profiles)),
        ("4.9", lambda: analyze_pairwise_comparisons_analysis(profiles)),
        ("4.10", lambda: analyze_convergence(profiles, stage6a_results)),
    ]

    for key, func in analyses:
        try:
            results[key] = func()
            # Log key result
            r = results[key]
            if key == "7.1":
                rho = r.get("overall", {}).get("rho", float("nan"))
                logger.info("7.1 Accuracy vs Depth: rho=%.3f", rho)
            elif key == "7.3":
                rho = r.get("overall", {}).get("rho", float("nan"))
                logger.info("7.3 Expressibility vs Accuracy (H1): rho=%.3f", rho)
            elif key == "7.7":
                n_p = r.get("n_pareto_optimal", 0)
                logger.info("7.7 Pareto Front: %d optimal encodings", n_p)
            else:
                logger.info("Sub-analysis %s completed", key)
        except Exception as e:
            logger.warning("Sub-analysis %s failed: %s", key, e)
            results[key] = {"status": "error", "error": str(e)}

    # 6. Compute final ranking (7.9)
    pareto_result = results.get("7.7", {})
    try:
        results["7.9"] = compute_final_ranking(profiles, pareto_result)
        logger.info("7.9 Final ranking computed for %d encodings", len(results["7.9"].get("rankings", [])))
    except Exception as e:
        logger.warning("Sub-analysis 7.9 failed: %s", e)
        results["7.9"] = {"status": "error", "error": str(e)}

    # 7. Compute hypothesis verdicts
    verdicts = compute_all_hypothesis_verdicts(results)
    logger.info("Hypothesis verdicts: %s",
                {k: v.get("verdict", "?") for k, v in verdicts.items()})

    # 8. Write JSON output files
    os.makedirs(output_dir, exist_ok=True)

    _write_json(
        {"schema_version": "1.0", "hypothesis_verdicts": verdicts},
        os.path.join(output_dir, "hypothesis_verdicts.json"),
    )
    _write_json(
        {"schema_version": "1.0", **pareto_result},
        os.path.join(output_dir, "pareto_front.json"),
    )
    _write_json(
        {"schema_version": "1.0", **results.get("7.9", {})},
        os.path.join(output_dir, "rankings.json"),
    )
    _write_json(
        {"schema_version": "1.0", **results.get("4.9", {})},
        os.path.join(output_dir, "pairwise_comparisons.json"),
    )

    # 9. Write tables
    try:
        ranking_md = _generate_ranking_table_md(results.get("7.9", {}))
        with open(os.path.join(output_dir, "ranking_table.md"), "w", encoding="utf-8") as f:
            f.write(ranking_md)
        ranking_tex = _generate_ranking_table_tex(results.get("7.9", {}))
        with open(os.path.join(output_dir, "ranking_table.tex"), "w", encoding="utf-8") as f:
            f.write(ranking_tex)
        hypothesis_md = _generate_hypothesis_table_md(verdicts)
        with open(os.path.join(output_dir, "hypothesis_table.md"), "w", encoding="utf-8") as f:
            f.write(hypothesis_md)
        hypothesis_tex = _generate_hypothesis_table_tex(verdicts)
        with open(os.path.join(output_dir, "hypothesis_table.tex"), "w", encoding="utf-8") as f:
            f.write(hypothesis_tex)
        logger.info("Tables written to %s", output_dir)
    except Exception as e:
        logger.warning("Table generation failed: %s", e)

    # 10. Generate plots
    if generate_plots:
        try:
            from experiments.plotting import generate_all_figures
            generated = generate_all_figures(results, profiles, figure_dir)
            logger.info("Generated %d figures in %s", len(generated), figure_dir)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

        # 10b. Generate journal-quality figures
        try:
            from experiments.plotting_journal import generate_journal_figures
            journal_generated = generate_journal_figures(results, profiles, figure_dir)
            logger.info("Generated %d journal figures in %s", len(journal_generated), figure_dir)
        except Exception as e:
            logger.warning("Journal figure generation failed: %s", e)

    # 11. Return complete result
    return {
        "schema_version": "1.0",
        "status": "success",
        "n_encodings_analyzed": len(profiles),
        "sub_analyses": results,
        "hypothesis_verdicts": verdicts,
    }
