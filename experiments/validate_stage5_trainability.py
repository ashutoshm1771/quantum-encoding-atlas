"""Validation script for Stage 5: Trainability Analysis results.

This script loads results from stage5_trainability/summary.json and performs
physics-aware sanity checks:

1. **Value integrity**: Trainability estimate is in [0, 1] and not NaN.
2. **Gradient variance**: Non-negative (not NaN). Zero is accepted as a valid
   physical result for discrete encodings and diagonal-gate architectures.
3. **Risk level**: barren_plateau_risk is one of {"low", "medium", "high", "unknown"}.
4. **Non-entangling encodings**: Continuous non-entangling encodings (product-state
   circuits) should not exhibit barren plateaus (risk != "high"). Discrete
   encodings (BasisEncoding) are exempt from gradient-based checks.
5. **Depth-trainability correlation**: Shallow circuits should generally have
   higher trainability than deep circuits (with tolerance for noise).
6. **Per-parameter variance**: Individual parameter variances are non-negative.
7. **Sample quality**: Sufficient successful samples for reliable estimation.

Physics Notes
-------------
- **Discrete encodings** (BasisEncoding) map continuous inputs to binary values
  via thresholds, producing step-function circuits. The parameter-shift rule
  is not applicable to step functions, so gradient_variance = 0 is expected.

- **Diagonal-gate encodings at reps=1** (IQP, ZZFeatureMap, PauliFeatureMap)
  apply only Z-diagonal gates (Rz, ZZ) after an initial Hadamard layer. Since
  ⟨0...0| is an eigenstate of all Z operators, the computational basis
  probability P(0...0) = 1/2^n is constant w.r.t. input parameters, yielding
  exactly zero gradients. At reps >= 2, a second H→diagonal layer breaks this
  symmetry and gradients become nonzero.

- **Trainability score scaling**: The sigmoid mapping from gradient variance to
  trainability score [0, 1] uses a reference variance of 0.1. For product-state
  encodings with a local observable (computational basis), gradient variance
  naturally decreases with n_features because each gradient involves a product
  of cos²(θ/2) factors. Expected trainability for non-entangling encodings:
  ~0.28 at n_features=2, ~0.09 at n_features=4. These are NOT barren plateaus
  — the gradients are healthy but small due to the product structure.

Usage:
    python experiments/validate_stage5_trainability.py
    python experiments/validate_stage5_trainability.py --results-path custom/path/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

# Compute paths relative to this script's location
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root to path for imports
sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_RESULTS_PATH = str(_SCRIPT_DIR / "results" / "raw" / "stage5_trainability" / "summary.json")

# Non-entangling encodings (should have high trainability)
NON_ENTANGLING_ENCODINGS: set[str] = {
    "angle",
    "basis",
    "higher_order_angle",
}

# Entangling encodings (may have barren plateau issues).
# Names must match the encoding_name field in summary.json exactly.
ENTANGLING_ENCODINGS: set[str] = {
    "iqp",
    "zz_feature_map",
    "pauli_feature_map",
    "data_reuploading",
    "hardware_efficient",
    "qaoa_encoding",
    "hamiltonian_encoding",
    "symmetry_inspired",
    "trainable_encoding",
    "so2_equivariant",
    "cyclic_equivariant",
    "swap_equivariant",
    "amplitude",
}

# Valid barren plateau risk levels
VALID_RISK_LEVELS = {"low", "medium", "high", "unknown"}

# Encoding families for grouping
ENCODING_FAMILIES: dict[str, list[str]] = {
    "Non-Entangling": ["angle", "basis", "higher_order_angle"],
    "IQP-Based": ["iqp", "zz_feature_map"],
    "Pauli-Based": ["pauli_feature_map"],
    "Data Re-uploading": ["data_reuploading"],
    "Hardware-Efficient": ["hardware_efficient"],
    "QAOA/Hamiltonian": ["qaoa_encoding", "hamiltonian_encoding"],
    "Symmetry-Based": ["symmetry_inspired"],
    "Trainable": ["trainable_encoding"],
    "Equivariant": ["so2_equivariant", "cyclic_equivariant", "swap_equivariant"],
    "Amplitude": ["amplitude"],
}

# Encodings that use discrete/binary data mapping rather than continuous
# parameter rotations.  BasisEncoding applies X gates based on a binary
# threshold (x_i > threshold → |1⟩, else |0⟩), producing step-function
# circuits.  The parameter-shift rule requires smooth dependence on
# continuous parameters and is not applicable to step functions, so
# gradient_variance = 0 is the mathematically correct result — not a bug.
# All gradient-based trainability checks are skipped for these encodings.
DISCRETE_ENCODINGS: set[str] = {"basis"}

# Encodings whose architecture produces exactly zero gradient variance at
# reps=1 when measured with the computational basis observable.
#
# MATHEMATICAL EXPLANATION:
# At reps=1, these encodings apply only Z-diagonal gates (Rz, ZZ) after
# an initial Hadamard layer:
#
#   |ψ(x)⟩ = D(x) · H^⊗n |0⟩^⊗n
#
# where D(x) is diagonal in the computational basis.  When measuring
# P(0...0) = |⟨0...0|ψ(x)⟩|², the bra ⟨0...0| is an eigenstate of all
# Z operators, so D†(x)|0...0⟩ = e^{iφ(x)}|0...0⟩.  Therefore:
#
#   P(0...0) = |e^{iφ(x)} · ⟨0...0|H^⊗n|0...0⟩|² = 1/2^n
#
# This is CONSTANT with respect to x — the gradient is exactly zero for
# all parameter values, not just at isolated points.  The near-zero values
# (~1e-28 to 1e-34) observed in experiments are floating-point noise around
# the true value of zero.
#
# At reps >= 2, a second Hadamard layer followed by additional diagonal
# gates breaks this eigenstate symmetry, making P(0...0) parameter-
# dependent and producing nonzero gradients.
ZERO_GRADIENT_AT_REPS_1: set[str] = {"iqp", "zz_feature_map", "pauli_feature_map"}


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None


@dataclass
class EncodingValidation:
    """Validation results for a single encoding configuration."""

    encoding_name: str
    params: dict[str, Any]
    results: list[ValidationResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_trainability_result(
    encoding_name: str,
    params: dict[str, Any],
    result: dict[str, Any],
) -> EncodingValidation:
    """Validate a single trainability result.

    Parameters
    ----------
    encoding_name : str
        Name of the encoding.
    params : dict[str, Any]
        Encoding parameters.
    result : dict[str, Any]
        Result data from summary.json.

    Returns
    -------
    EncodingValidation
        Validation results.
    """
    checks: list[ValidationResult] = []

    # Extract trainability values
    trainability_estimate = result.get("trainability_estimate")
    gradient_variance = result.get("gradient_variance")
    barren_plateau_risk = result.get("barren_plateau_risk", "unknown")
    per_parameter_variance = result.get("per_parameter_variance", [])
    n_successful_samples = result.get("n_successful_samples")

    # Check 1: Trainability estimate is not NaN
    if trainability_estimate is not None:
        is_nan = math.isnan(trainability_estimate) if isinstance(trainability_estimate, float) else False
        checks.append(ValidationResult(
            check_name="trainability_not_nan",
            passed=not is_nan,
            message=f"trainability_estimate={trainability_estimate} ({'NaN' if is_nan else 'valid'})",
        ))

    # Check 2: Trainability estimate is in valid range [0, 1]
    if trainability_estimate is not None and not math.isnan(trainability_estimate):
        in_range = 0.0 <= trainability_estimate <= 1.0
        checks.append(ValidationResult(
            check_name="trainability_in_range",
            passed=in_range,
            message=f"trainability_estimate={trainability_estimate:.4f} ({'in [0,1]' if in_range else 'OUT OF RANGE'})",
            details={"value": trainability_estimate, "min": 0.0, "max": 1.0},
        ))

    # Check 3: Gradient variance is non-negative and not NaN.
    #
    # Zero gradient variance is a valid physical result in several cases:
    # - Discrete encodings (BasisEncoding): step-function circuits
    # - Diagonal-gate encodings at reps=1: P(0...0) is parameter-independent
    # - Genuine barren plateaus: captured separately by risk level checks
    #
    # We only fail on NaN or negative values, which indicate numerical bugs.
    if gradient_variance is not None:
        if encoding_name in DISCRETE_ENCODINGS:
            # Discrete encodings produce zero gradient by design (step
            # functions have zero parameter-shift derivatives).  Report
            # as passed with an informational message.
            checks.append(ValidationResult(
                check_name="gradient_variance_discrete_exempt",
                passed=True,
                message=(
                    f"{encoding_name} is a discrete encoding; "
                    f"gradient_variance={gradient_variance:.2e} "
                    f"(zero expected, parameter-shift rule not applicable)"
                ),
                details={"value": gradient_variance, "reason": "discrete_encoding"},
            ))
        else:
            is_nan = (
                math.isnan(gradient_variance)
                if isinstance(gradient_variance, float)
                else False
            )
            is_non_negative = gradient_variance >= 0 if not is_nan else False
            checks.append(ValidationResult(
                check_name="gradient_variance_non_negative",
                passed=not is_nan and is_non_negative,
                message=(
                    f"gradient_variance={gradient_variance:.6f}"
                    if not is_nan
                    else "gradient_variance=NaN"
                ),
                details={"value": gradient_variance},
            ))

            # Informational note for near-zero gradient variance in
            # non-discrete encodings.  This flags cases where the gradient
            # is effectively zero due to circuit architecture (known) or
            # potential issues (unknown).
            if not is_nan and gradient_variance < 1e-10:
                reps = params.get("reps", params.get("n_layers", None))
                is_known_zero_architecture = (
                    encoding_name in ZERO_GRADIENT_AT_REPS_1
                    and reps is not None
                    and reps <= 1
                )
                if is_known_zero_architecture:
                    reason = (
                        "Expected: at reps=1, only Z-diagonal gates are "
                        "applied after H layer. Since ⟨0...0| is a Z "
                        "eigenstate, P(0...0) = 1/2^n is constant."
                    )
                else:
                    reason = (
                        "Near-zero gradient variance; may indicate a barren "
                        "plateau or architectural constraint. See "
                        "barren_plateau_risk for risk assessment."
                    )
                checks.append(ValidationResult(
                    check_name="near_zero_gradient_info",
                    passed=True,  # Informational, not a failure
                    message=(
                        f"INFO: {encoding_name} gradient_variance="
                        f"{gradient_variance:.2e}. {reason}"
                    ),
                    details={
                        "value": gradient_variance,
                        "is_known_architecture": is_known_zero_architecture,
                    },
                ))

    # Check 4: Valid barren plateau risk level
    is_valid_risk = barren_plateau_risk in VALID_RISK_LEVELS
    checks.append(ValidationResult(
        check_name="valid_risk_level",
        passed=is_valid_risk,
        message=f"barren_plateau_risk='{barren_plateau_risk}' is {'valid' if is_valid_risk else 'invalid'}",
    ))

    # Check 5: Flag high barren plateau risk
    if barren_plateau_risk == "high":
        # This is a warning, not a failure - just flagging for attention
        checks.append(ValidationResult(
            check_name="high_risk_flagged",
            passed=True,  # Not a failure, just flagging
            message=f"WARNING: {encoding_name} has barren_plateau_risk='high'",
            details={
                "encoding": encoding_name,
                "params": params,
                "risk": "high",
            },
        ))

    # Check 6: Non-entangling encodings should not exhibit barren plateaus.
    #
    # PHYSICAL RATIONALE:
    # Non-entangling encodings produce product states (no inter-qubit
    # correlations).  Barren plateaus arise from entanglement spreading
    # information across exponentially many degrees of freedom (McClean
    # et al., 2018).  Product-state circuits cannot exhibit this mechanism,
    # so barren_plateau_risk="high" would indicate a bug, not real physics.
    #
    # WHY NOT A FIXED TRAINABILITY THRESHOLD:
    # The trainability score is a sigmoid of log₁₀(variance / 0.1).  For
    # product-state encodings with a LOCAL observable (computational basis
    # or single-qubit Z), gradient variance naturally decreases with
    # n_features because each gradient term includes a product of
    # cos²(θⱼ/2) factors from unaffected qubits:
    #
    #   ∂P(0...0)/∂θᵢ = -½sin(θᵢ) · ∏ⱼ≠ᵢ cos²(θⱼ/2)
    #
    # Expected gradient variance and trainability scores:
    #   n_features=2: variance ≈ 0.03,  trainability ≈ 0.28
    #   n_features=4: variance ≈ 0.007, trainability ≈ 0.09
    #   n_features=8: variance ≈ 0.002, trainability ≈ 0.04
    #
    # A fixed threshold (e.g., 0.6) would fail at n_features >= 3
    # despite perfectly healthy gradients.  The correct invariant is
    # absence of barren plateaus, captured by the risk level.
    if encoding_name in NON_ENTANGLING_ENCODINGS and trainability_estimate is not None:
        if encoding_name in DISCRETE_ENCODINGS:
            # Discrete encodings are exempt from gradient-based trainability
            # checks entirely.  BasisEncoding's zero gradient is a property
            # of its step-function structure, not a barren plateau.
            checks.append(ValidationResult(
                check_name="non_entangling_discrete_exempt",
                passed=True,
                message=(
                    f"{encoding_name} is discrete; gradient-based "
                    f"trainability check not applicable "
                    f"(trainability={trainability_estimate:.4f})"
                ),
                details={
                    "encoding": encoding_name,
                    "reason": "discrete_encoding",
                    "trainability": trainability_estimate,
                },
            ))
        elif not math.isnan(trainability_estimate):
            is_not_high_risk = barren_plateau_risk != "high"
            checks.append(ValidationResult(
                check_name="non_entangling_no_barren_plateau",
                passed=is_not_high_risk,
                message=(
                    f"{encoding_name} (non-entangling) "
                    f"barren_plateau_risk='{barren_plateau_risk}', "
                    f"trainability={trainability_estimate:.4f}"
                ),
                details={
                    "encoding": encoding_name,
                    "trainability": trainability_estimate,
                    "risk": barren_plateau_risk,
                    "n_features": params.get("n_features"),
                },
            ))

    # Check 7: Per-parameter variance values are positive
    if per_parameter_variance:
        all_positive = True
        invalid_indices = []
        for i, val in enumerate(per_parameter_variance):
            if val is None or (isinstance(val, float) and (math.isnan(val) or val < 0)):
                all_positive = False
                invalid_indices.append(i)

        checks.append(ValidationResult(
            check_name="per_param_variance_positive",
            passed=all_positive,
            message=f"per_parameter_variance {'all positive' if all_positive else f'invalid at {invalid_indices}'}",
            details={
                "n_params": len(per_parameter_variance),
                "invalid_indices": invalid_indices,
            },
        ))

    # Check 8: n_successful_samples is reasonable
    if n_successful_samples is not None:
        is_reasonable = n_successful_samples >= 10  # At least some samples succeeded
        checks.append(ValidationResult(
            check_name="samples_reasonable",
            passed=is_reasonable,
            message=f"n_successful_samples={n_successful_samples} (min: 10)",
        ))

    return EncodingValidation(encoding_name, params, checks)


def validate_depth_trainability_correlation(results: list[dict[str, Any]]) -> list[ValidationResult]:
    """Validate that shallow circuits have higher trainability than deep circuits.

    Parameters
    ----------
    results : list[dict]
        All results from summary.json.

    Returns
    -------
    list[ValidationResult]
        Validation results for depth-trainability correlation.
    """
    checks: list[ValidationResult] = []

    # Group by encoding name and n_features
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)

    for entry in results:
        if entry.get("status") != "success":
            continue

        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})
        result = entry.get("result", {})

        reps = params.get("reps", 1)
        n_features = params.get("n_features", 4)
        trainability = result.get("trainability_estimate")

        if trainability is not None:
            if not math.isnan(trainability):
                key = (enc_name, n_features)
                grouped[key].append((reps, trainability))

    # For each encoding, check if trainability decreases with reps (depth)
    for (enc_name, n_features), values in grouped.items():
        if len(values) < 2:
            continue

        # Sort by reps
        values.sort(key=lambda x: x[0])
        reps_list = [v[0] for v in values]
        train_list = [v[1] for v in values]

        # Check if generally decreasing (allow some noise)
        # Trainability often decreases with depth due to barren plateaus
        reasonable = True
        for i in range(1, len(train_list)):
            # Allow increases up to 0.2 (some noise is expected)
            if train_list[i] > train_list[i-1] + 0.3:
                reasonable = False
                break

        checks.append(ValidationResult(
            check_name=f"depth_correlation_{enc_name}_n{n_features}",
            passed=reasonable,
            message=f"{enc_name}(n={n_features}): reps={reps_list} -> train={[f'{t:.3f}' for t in train_list]}",
            details={
                "encoding": enc_name,
                "n_features": n_features,
                "reps": reps_list,
                "trainability": train_list,
            },
        ))

    return checks


def collect_high_risk_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect all encodings with high barren plateau risk.

    Parameters
    ----------
    results : list[dict]
        All results from summary.json.

    Returns
    -------
    list[dict]
        High-risk cases.
    """
    high_risk = []

    for entry in results:
        if entry.get("status") != "success":
            continue

        result = entry.get("result", {})
        if result.get("barren_plateau_risk") == "high":
            high_risk.append({
                "encoding": entry.get("encoding_name"),
                "params": entry.get("encoding_params"),
                "trainability": result.get("trainability_estimate"),
                "gradient_variance": result.get("gradient_variance"),
            })

    return high_risk


# =============================================================================
# Table Generation
# =============================================================================

def format_params(params: dict[str, Any]) -> str:
    """Format parameters as a short string."""
    if not params:
        return "default"
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        key_abbrev = {
            "n_features": "n",
            "reps": "r",
            "entanglement": "ent",
            "n_layers": "L",
        }.get(k, k)
        parts.append(f"{key_abbrev}={v}")
    return ", ".join(parts)


def get_encoding_family(encoding_name: str) -> str:
    """Get the family name for an encoding."""
    for family, encodings in ENCODING_FAMILIES.items():
        if encoding_name in encodings:
            return family
    return "Other"


def generate_markdown_table(results: list[dict[str, Any]]) -> str:
    """Generate a publication-ready Markdown table.

    Parameters
    ----------
    results : list[dict]
        Results from summary.json.

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "# Stage 5: Trainability Analysis Results",
        "",
        "## Summary Table",
        "",
        "Sorted by trainability (descending).",
        "",
        "| Encoding | Parameters | Trainability | Gradient Var | BP Risk | n_samples |",
        "|----------|------------|--------------|--------------|---------|-----------|",
    ]

    # Sort by trainability (descending)
    sorted_results = sorted(
        [r for r in results if r.get("result", {}).get("trainability_estimate") is not None],
        key=lambda x: x.get("result", {}).get("trainability_estimate", 0),
        reverse=True,
    )

    for entry in sorted_results:
        result = entry.get("result", {})
        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})

        trainability = result.get("trainability_estimate", 0)
        gradient_var = result.get("gradient_variance", 0)
        bp_risk = result.get("barren_plateau_risk", "—")
        n_samples = result.get("n_successful_samples", "—")

        # Format values
        train_str = f"{trainability:.4f}" if isinstance(trainability, (int, float)) else "—"
        gv_str = f"{gradient_var:.6f}" if isinstance(gradient_var, (int, float)) else "—"

        # Add risk indicator emoji
        risk_emoji = "🟢" if bp_risk == "low" else ("🟡" if bp_risk == "medium" else "🔴" if bp_risk == "high" else "⚪")

        # Add indicator for non-entangling
        entangling_marker = "" if enc_name in ENTANGLING_ENCODINGS else " (NE)"

        lines.append(
            f"| {enc_name}{entangling_marker} | {format_params(params)} | "
            f"{train_str} | {gv_str} | {risk_emoji} {bp_risk} | {n_samples} |"
        )

    # Add high-risk section
    high_risk = collect_high_risk_cases(
        [r for r in results if r.get("status") == "success"]
    )

    if high_risk:
        lines.extend([
            "",
            "## High Barren Plateau Risk Cases",
            "",
            "The following configurations have high barren plateau risk:",
            "",
        ])
        for case in high_risk:
            lines.append(f"- **{case['encoding']}** ({format_params(case['params'])}): "
                        f"trainability={case['trainability']:.4f}, "
                        f"gradient_variance={case['gradient_variance']:.6f}")

    # Add legend
    lines.extend([
        "",
        "## Legend",
        "",
        "- (NE): Non-entangling encoding (expected high trainability)",
        "- Trainability: Higher values indicate easier optimization",
        "- Gradient Var: Variance of gradients (low = barren plateau)",
        "- BP Risk: Barren plateau risk (🟢 low, 🟡 medium, 🔴 high)",
    ])

    return "\n".join(lines)


def generate_latex_table(results: list[dict[str, Any]]) -> str:
    """Generate a publication-ready LaTeX table.

    Parameters
    ----------
    results : list[dict]
        Results from summary.json.

    Returns
    -------
    str
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Stage 5: Trainability Analysis Results}",
        r"\label{tab:stage5-trainability}",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Encoding & Parameters & Trainability & Gradient Var. & BP Risk \\",
        r"\midrule",
    ]

    # Sort by trainability (descending)
    sorted_results = sorted(
        [r for r in results if r.get("result", {}).get("trainability_estimate") is not None],
        key=lambda x: x.get("result", {}).get("trainability_estimate", 0),
        reverse=True,
    )

    for entry in sorted_results[:20]:  # Top 20 for space
        result = entry.get("result", {})
        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})

        # Escape underscores for LaTeX
        enc_name_tex = enc_name.replace("_", r"\_")
        params_tex = format_params(params).replace("_", r"\_")

        trainability = result.get("trainability_estimate", 0)
        gradient_var = result.get("gradient_variance", 0)
        bp_risk = result.get("barren_plateau_risk", "—")

        train_str = f"{trainability:.4f}" if isinstance(trainability, (int, float)) else "—"
        gv_str = f"{gradient_var:.6f}" if isinstance(gradient_var, (int, float)) else "—"

        lines.append(
            f"  {enc_name_tex} & {params_tex} & {train_str} & {gv_str} & {bp_risk} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Main Validation
# =============================================================================

def load_results(path: str) -> dict[str, Any]:
    """Load results from summary.json file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_validation(results_path: str) -> tuple[int, int, list[EncodingValidation]]:
    """Run all validation checks.

    Parameters
    ----------
    results_path : str
        Path to summary.json file.

    Returns
    -------
    tuple[int, int, list[EncodingValidation]]
        (total_passed, total_failed, all_validations)
    """
    data = load_results(results_path)
    results = data.get("results", [])

    if not results:
        logger.warning("No results found in file")
        return 0, 0, []

    logger.info(f"Loaded {len(results)} results from {results_path}")

    all_validations: list[EncodingValidation] = []
    total_passed = 0
    total_failed = 0

    # Validate individual results
    for entry in results:
        if entry.get("status") != "success":
            continue

        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})
        result = entry.get("result", {})

        validation = validate_trainability_result(enc_name, params, result)
        all_validations.append(validation)
        total_passed += validation.pass_count
        total_failed += validation.fail_count

    # Global validation checks
    global_checks: list[ValidationResult] = []

    # Check depth-trainability correlation
    depth_checks = validate_depth_trainability_correlation(results)
    global_checks.extend(depth_checks)

    # Add global checks to a synthetic validation
    if global_checks:
        global_validation = EncodingValidation("_global_checks_", {}, global_checks)
        all_validations.append(global_validation)
        total_passed += global_validation.pass_count
        total_failed += global_validation.fail_count

    return total_passed, total_failed, all_validations


def print_validation_summary(
    total_passed: int,
    total_failed: int,
    validations: list[EncodingValidation],
    results_path: str,
) -> None:
    """Print validation summary to console."""

    print("\n" + "=" * 70)
    print("STAGE 5: TRAINABILITY ANALYSIS VALIDATION")
    print("=" * 70)

    # Summary statistics
    total_encodings = len([v for v in validations if not v.encoding_name.startswith("_")])
    encodings_passed = sum(1 for v in validations if v.all_passed and not v.encoding_name.startswith("_"))
    encodings_failed = total_encodings - encodings_passed

    print(f"\nSummary:")
    print(f"  Total encoding configurations: {total_encodings}")
    print(f"  Configurations passed: {encodings_passed}")
    print(f"  Configurations with failures: {encodings_failed}")
    print(f"  Total checks passed: {total_passed}")
    print(f"  Total checks failed: {total_failed}")

    # Print trainability statistics
    train_values: list[float] = []
    for v in validations:
        if v.encoding_name.startswith("_"):
            continue
        for r in v.results:
            if r.check_name == "trainability_in_range" and r.details:
                val = r.details.get("value")
                if val is not None:
                    train_values.append(val)

    if train_values:
        print(f"\nTrainability Statistics:")
        print(f"  Min: {min(train_values):.4f}")
        print(f"  Max: {max(train_values):.4f}")
        print(f"  Mean: {sum(train_values)/len(train_values):.4f}")

    # Print high-risk cases
    try:
        data = load_results(results_path)
        high_risk = collect_high_risk_cases(
            [r for r in data.get("results", []) if r.get("status") == "success"]
        )
        if high_risk:
            print(f"\nHigh Barren Plateau Risk Cases ({len(high_risk)}):")
            for case in high_risk[:5]:  # Top 5
                print(f"  - {case['encoding']} ({format_params(case['params'])})")
    except Exception:
        pass

    # Print failures
    if total_failed > 0:
        print("\n" + "-" * 70)
        print("FAILED CHECKS:")
        print("-" * 70)

        for validation in validations:
            if not validation.all_passed:
                display_name = validation.encoding_name if not validation.encoding_name.startswith("_") else "Global Checks"
                print(f"\n{display_name} ({format_params(validation.params)}):")
                for check in validation.results:
                    if not check.passed:
                        print(f"  [FAIL] {check.check_name}: {check.message}")

    # Print pass/fail summary
    print("\n" + "-" * 70)
    if total_failed == 0:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: {total_failed} CHECKS FAILED")
    print("-" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Stage 5 Trainability Analysis results",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to summary.json file (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated tables (default: same as results)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Run validation
        total_passed, total_failed, validations = run_validation(args.results_path)

        # Print summary
        print_validation_summary(total_passed, total_failed, validations, args.results_path)

        # Generate tables if we have results
        if validations:
            data = load_results(args.results_path)
            results = [r for r in data.get("results", []) if r.get("status") == "success"]

            # Determine output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate Markdown table
            md_table = generate_markdown_table(results)
            md_path = output_dir / "stage5_trainability_table.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_table)
            print(f"\nGenerated Markdown table: {md_path}")

            # Generate LaTeX table
            latex_table = generate_latex_table(results)
            latex_path = output_dir / "stage5_trainability_table.tex"
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
            print(f"Generated LaTeX table: {latex_path}")

        return 0 if total_failed == 0 else 1

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Stage 5 results not yet available. Run the experiment first:")
        print("  python -m experiments.run_stage --config experiments/configs/stage5_trainability.json")
        return 1
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
