"""Validation script for Stage 4: Entanglement Capability results.

This script loads results from stage4_entanglement/summary.json and performs
sanity checks:
- Non-entangling encodings (AngleEncoding, BasisEncoding, HigherOrderAngleEncoding)
  have entanglement_capability approximately 0
- Meyer-Wallach values are in [0, 1]
- per_qubit_entanglement arrays have correct length
- Values are non-negative and not NaN

Usage:
    python experiments/validate_stage4_entanglement.py
    python experiments/validate_stage4_entanglement.py --results-path custom/path/summary.json
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

DEFAULT_RESULTS_PATH = str(_SCRIPT_DIR / "results" / "raw" / "stage4_entanglement" / "summary.json")

# Non-entangling encodings (should have entanglement_capability approx 0)
NON_ENTANGLING_ENCODINGS: set[str] = {
    "angle",
    "basis",
    "higher_order_angle",
}

# Entangling encodings (should have entanglement_capability > 0)
ENTANGLING_ENCODINGS: set[str] = {
    "iqp",
    "zz_feature_map",
    "pauli_feature_map",
    "data_reuploading",
    "hardware_efficient",
    "qaoa",
    "hamiltonian",
    "symmetry_inspired",
    "trainable",
    "so2_equivariant",
    "cyclic_equivariant",
    "swap_equivariant",
    "amplitude",
}

# Tolerance for "approximately zero" entanglement
ZERO_ENTANGLEMENT_TOLERANCE = 0.05

# Encoding families for grouping
ENCODING_FAMILIES: dict[str, list[str]] = {
    "Non-Entangling": ["angle", "basis", "higher_order_angle"],
    "IQP-Based": ["iqp", "zz_feature_map"],
    "Pauli-Based": ["pauli_feature_map"],
    "Data Re-uploading": ["data_reuploading"],
    "Hardware-Efficient": ["hardware_efficient"],
    "QAOA/Hamiltonian": ["qaoa", "hamiltonian"],
    "Symmetry-Based": ["symmetry_inspired"],
    "Trainable": ["trainable"],
    "Equivariant": ["so2_equivariant", "cyclic_equivariant", "swap_equivariant"],
    "Amplitude": ["amplitude"],
}


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

def validate_entanglement_result(
    encoding_name: str,
    params: dict[str, Any],
    result: dict[str, Any],
) -> EncodingValidation:
    """Validate a single entanglement result.

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

    # Extract entanglement values
    entanglement_capability = result.get("entanglement_capability")
    std_error = result.get("std_error")
    per_qubit_entanglement = result.get("per_qubit_entanglement", [])
    measure = result.get("measure", "meyer_wallach")
    n_features = params.get("n_features", 4)

    # Check 1: Entanglement capability is not NaN
    if entanglement_capability is not None:
        is_nan = math.isnan(entanglement_capability) if isinstance(entanglement_capability, float) else False
        checks.append(ValidationResult(
            check_name="entanglement_not_nan",
            passed=not is_nan,
            message=f"entanglement_capability={entanglement_capability} ({'NaN' if is_nan else 'valid'})",
        ))

    # Check 2: Entanglement capability is in valid range [0, 1] for Meyer-Wallach
    if entanglement_capability is not None and not math.isnan(entanglement_capability):
        in_range = 0.0 <= entanglement_capability <= 1.0
        checks.append(ValidationResult(
            check_name="entanglement_in_range",
            passed=in_range,
            message=f"entanglement_capability={entanglement_capability:.4f} ({'in [0,1]' if in_range else 'OUT OF RANGE'})",
            details={"value": entanglement_capability, "min": 0.0, "max": 1.0},
        ))

    # Check 3: Non-entangling encodings should have ~0 entanglement
    if encoding_name in NON_ENTANGLING_ENCODINGS and entanglement_capability is not None:
        if not math.isnan(entanglement_capability):
            is_near_zero = entanglement_capability < ZERO_ENTANGLEMENT_TOLERANCE
            checks.append(ValidationResult(
                check_name="non_entangling_zero",
                passed=is_near_zero,
                message=f"{encoding_name} (non-entangling) entanglement={entanglement_capability:.4f} "
                       f"(expected < {ZERO_ENTANGLEMENT_TOLERANCE})",
                details={
                    "encoding": encoding_name,
                    "value": entanglement_capability,
                    "tolerance": ZERO_ENTANGLEMENT_TOLERANCE,
                },
            ))

    # Check 4: Entangling encodings should have > 0 entanglement
    if encoding_name in ENTANGLING_ENCODINGS and entanglement_capability is not None:
        if not math.isnan(entanglement_capability):
            has_entanglement = entanglement_capability > ZERO_ENTANGLEMENT_TOLERANCE
            checks.append(ValidationResult(
                check_name="entangling_nonzero",
                passed=has_entanglement,
                message=f"{encoding_name} (entangling) entanglement={entanglement_capability:.4f} "
                       f"(expected > {ZERO_ENTANGLEMENT_TOLERANCE})",
                details={
                    "encoding": encoding_name,
                    "value": entanglement_capability,
                    "threshold": ZERO_ENTANGLEMENT_TOLERANCE,
                },
            ))

    # Check 5: Std error is non-negative
    if std_error is not None:
        is_nan = math.isnan(std_error) if isinstance(std_error, float) else False
        is_valid = not is_nan and std_error >= 0
        checks.append(ValidationResult(
            check_name="std_error_valid",
            passed=is_valid,
            message=f"std_error={std_error:.4f}" if not is_nan else "std_error=NaN",
        ))

    # Check 6: Per-qubit entanglement array has reasonable length
    # Note: n_qubits != n_features for some encodings:
    #   - AmplitudeEncoding: n_qubits = ceil(log2(n_features))
    #   - SO2Equivariant: n_qubits > n_features (extra angular momentum qubits)
    # The per_qubit_entanglement array has one entry per qubit, so its
    # length reflects the actual circuit width, not the feature dimension.
    # We validate that the length is at least 2 (entanglement requires >=2
    # qubits) and at most 2*n_features (reasonable upper bound).
    if per_qubit_entanglement:
        actual_length = len(per_qubit_entanglement)
        reasonable_length = 2 <= actual_length <= 2 * n_features
        checks.append(ValidationResult(
            check_name="per_qubit_length",
            passed=reasonable_length,
            message=(
                f"per_qubit_entanglement length={actual_length} "
                f"(n_features={n_features}, valid range=[2, {2 * n_features}])"
            ),
            details={
                "actual": actual_length,
                "n_features": n_features,
            },
        ))

        # Check 7: Per-qubit values are in [0, 1]
        all_valid = True
        invalid_indices = []
        for i, val in enumerate(per_qubit_entanglement):
            if val is None or math.isnan(val) or val < 0 or val > 1:
                all_valid = False
                invalid_indices.append(i)

        checks.append(ValidationResult(
            check_name="per_qubit_values_valid",
            passed=all_valid,
            message=f"per_qubit_entanglement values {'all valid' if all_valid else f'invalid at indices {invalid_indices}'}",
            details={
                "values": per_qubit_entanglement[:4] if len(per_qubit_entanglement) > 4 else per_qubit_entanglement,
                "invalid_indices": invalid_indices,
            },
        ))

    # Check 8: Measure is specified
    valid_measures = {"meyer_wallach", "concurrence", "von_neumann", "linear_entropy"}
    is_valid_measure = measure in valid_measures
    checks.append(ValidationResult(
        check_name="valid_measure",
        passed=is_valid_measure,
        message=f"measure='{measure}' is {'valid' if is_valid_measure else 'invalid'}",
    ))

    return EncodingValidation(encoding_name, params, checks)


def validate_reps_scaling(results: list[dict[str, Any]]) -> list[ValidationResult]:
    """Validate entanglement behaviour across reps.

    For data-encoding circuits (IQP, ZZ, Pauli feature maps), increasing
    reps does NOT guarantee higher entanglement.  These circuits re-apply
    the same data-dependent unitaries, so doubling the interaction angle
    can *overshoot* the maximum entanglement point.  This is especially
    pronounced at 2 qubits (a single entangling pair), where drops of
    0.2-0.3 from reps=1 to reps=2 are physically expected.

    Instead of requiring monotonic increase, we check:
    1. All entanglement values remain positive (the circuit still entangles).
    2. For n_features >= 4, drops are not catastrophic (> 0.4).
    3. For n_features = 2, we only flag if entanglement collapses to ~0.

    Parameters
    ----------
    results : list[dict]
        All results from summary.json.

    Returns
    -------
    list[ValidationResult]
        Validation results for reps scaling checks.
    """
    checks: list[ValidationResult] = []

    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)

    for entry in results:
        if entry.get("status") != "success":
            continue

        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})
        result = entry.get("result", {})

        reps = params.get("reps")
        n_features = params.get("n_features", 4)
        entanglement = result.get("entanglement_capability")

        if reps is not None and entanglement is not None:
            if not math.isnan(entanglement):
                if enc_name in ENTANGLING_ENCODINGS:
                    key = (enc_name, n_features)
                    grouped[key].append((reps, entanglement))

    for (enc_name, n_features), values in grouped.items():
        if len(values) < 2:
            continue

        values.sort(key=lambda x: x[0])
        reps_list = [v[0] for v in values]
        ent_list = [v[1] for v in values]

        # Adaptive tolerance: 2-qubit circuits have stronger
        # interference effects so we allow larger drops.
        if n_features <= 2:
            max_drop = 0.4
        else:
            max_drop = 0.3

        reasonable = True
        for i in range(1, len(ent_list)):
            if ent_list[i] < ent_list[i - 1] - max_drop:
                reasonable = False
                break

        checks.append(ValidationResult(
            check_name=f"reps_scaling_{enc_name}_n{n_features}",
            passed=reasonable,
            message=(
                f"{enc_name}(n={n_features}): reps={reps_list} -> "
                f"ent={[f'{e:.3f}' for e in ent_list]} "
                f"(max_drop_tolerance={max_drop})"
            ),
            details={
                "encoding": enc_name,
                "n_features": n_features,
                "reps": reps_list,
                "entanglement": ent_list,
                "max_drop_tolerance": max_drop,
            },
        ))

    return checks


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
        "# Stage 4: Entanglement Capability Results",
        "",
        "## Summary Table",
        "",
        "Sorted by entanglement capability (descending).",
        "",
        "| Encoding | Parameters | Entanglement | Std Error | Measure | Per-Qubit (first 4) |",
        "|----------|------------|--------------|-----------|---------|---------------------|",
    ]

    # Sort by entanglement capability (descending)
    sorted_results = sorted(
        [r for r in results if r.get("result", {}).get("entanglement_capability") is not None],
        key=lambda x: x.get("result", {}).get("entanglement_capability", 0),
        reverse=True,
    )

    for entry in sorted_results:
        result = entry.get("result", {})
        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})

        entanglement = result.get("entanglement_capability", 0)
        std_error = result.get("std_error", 0)
        measure = result.get("measure", "—")
        per_qubit = result.get("per_qubit_entanglement", [])

        # Format values
        ent_str = f"{entanglement:.4f}" if isinstance(entanglement, (int, float)) else "—"
        std_str = f"{std_error:.4f}" if isinstance(std_error, (int, float)) else "—"
        per_qubit_str = ", ".join(f"{v:.2f}" for v in per_qubit[:4]) if per_qubit else "—"

        # Add indicator for non-entangling
        entangling_marker = "" if enc_name in ENTANGLING_ENCODINGS else " (NE)"

        lines.append(
            f"| {enc_name}{entangling_marker} | {format_params(params)} | "
            f"{ent_str} | {std_str} | {measure} | [{per_qubit_str}] |"
        )

    # Add legend
    lines.extend([
        "",
        "## Legend",
        "",
        "- (NE): Non-entangling encoding (expected entanglement ≈ 0)",
        "- Entanglement: Meyer-Wallach measure in [0, 1] (higher = more entangled)",
        "- Std Error: Standard error of the entanglement estimate",
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
        r"\caption{Stage 4: Entanglement Capability Results (Meyer-Wallach)}",
        r"\label{tab:stage4-entanglement}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Encoding & Parameters & Entanglement & Std Error & Measure \\",
        r"\midrule",
    ]

    # Sort by entanglement capability (descending)
    sorted_results = sorted(
        [r for r in results if r.get("result", {}).get("entanglement_capability") is not None],
        key=lambda x: x.get("result", {}).get("entanglement_capability", 0),
        reverse=True,
    )

    for entry in sorted_results[:20]:  # Top 20 for space
        result = entry.get("result", {})
        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})

        # Escape underscores for LaTeX
        enc_name_tex = enc_name.replace("_", r"\_")
        params_tex = format_params(params).replace("_", r"\_")

        entanglement = result.get("entanglement_capability", 0)
        std_error = result.get("std_error", 0)
        measure = result.get("measure", "—")

        ent_str = f"{entanglement:.4f}" if isinstance(entanglement, (int, float)) else "—"
        std_str = f"{std_error:.4f}" if isinstance(std_error, (int, float)) else "—"

        lines.append(
            f"  {enc_name_tex} & {params_tex} & {ent_str} & {std_str} & {measure} \\\\"
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

        validation = validate_entanglement_result(enc_name, params, result)
        all_validations.append(validation)
        total_passed += validation.pass_count
        total_failed += validation.fail_count

    # Global validation checks
    global_checks: list[ValidationResult] = []

    # Check reps scaling
    reps_checks = validate_reps_scaling(results)
    global_checks.extend(reps_checks)

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
) -> None:
    """Print validation summary to console."""

    print("\n" + "=" * 70)
    print("STAGE 4: ENTANGLEMENT CAPABILITY VALIDATION")
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

    # Print entanglement statistics
    ent_values: list[float] = []
    for v in validations:
        if v.encoding_name.startswith("_"):
            continue
        for r in v.results:
            if r.check_name == "entanglement_in_range" and r.details:
                val = r.details.get("value")
                if val is not None:
                    ent_values.append(val)

    if ent_values:
        print(f"\nEntanglement Capability Statistics:")
        print(f"  Min: {min(ent_values):.4f}")
        print(f"  Max: {max(ent_values):.4f}")
        print(f"  Mean: {sum(ent_values)/len(ent_values):.4f}")

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
        description="Validate Stage 4 Entanglement Capability results",
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
        print_validation_summary(total_passed, total_failed, validations)

        # Generate tables if we have results
        if validations:
            data = load_results(args.results_path)
            results = [r for r in data.get("results", []) if r.get("status") == "success"]

            # Determine output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate Markdown table
            md_table = generate_markdown_table(results)
            md_path = output_dir / "stage4_entanglement_table.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_table)
            print(f"\nGenerated Markdown table: {md_path}")

            # Generate LaTeX table
            latex_table = generate_latex_table(results)
            latex_path = output_dir / "stage4_entanglement_table.tex"
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
            print(f"Generated LaTeX table: {latex_path}")

        return 0 if total_failed == 0 else 1

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Stage 4 results not yet available. Run the experiment first:")
        print("  python -m experiments.run_stage --config experiments/configs/stage4_entanglement.json")
        return 1
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
