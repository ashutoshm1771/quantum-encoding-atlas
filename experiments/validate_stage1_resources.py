"""Validation script for Stage 1: Resource Profiling results.

This script loads results from stage1_resources/summary.json and performs
sanity checks to verify that gate counts match theoretical formulas from
each encoding's _compute_properties() method.

Usage:
    python experiments/validate_stage1_resources.py
    python experiments/validate_stage1_resources.py --results-path custom/path/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
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

from encoding_atlas.core.registry import get_encoding

DEFAULT_RESULTS_PATH = str(_SCRIPT_DIR / "results" / "raw" / "stage1_resources" / "summary.json")

# Encoding families for grouping in tables
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
# Theoretical Formulas
# =============================================================================

def get_theoretical_properties(encoding_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Compute theoretical properties using the encoding's _compute_properties() method.

    Parameters
    ----------
    encoding_name : str
        Name of the encoding (registry name).
    params : dict[str, Any]
        Encoding parameters.

    Returns
    -------
    dict[str, Any]
        Theoretical properties including gate counts, depth, etc.
    """
    try:
        # Create encoding instance
        encoding = get_encoding(encoding_name, **params)
        props = encoding.properties

        return {
            "n_qubits": props.n_qubits,
            "depth": props.depth,
            "gate_count": props.gate_count,
            "single_qubit_gates": props.single_qubit_gates,
            "two_qubit_gates": props.two_qubit_gates,
            "parameter_count": props.parameter_count,
            "is_entangling": props.is_entangling,
        }
    except Exception as e:
        logger.warning(f"Could not compute theoretical properties for {encoding_name}: {e}")
        return {}


def get_iqp_theoretical_formula(n_features: int, reps: int, entanglement: str) -> dict[str, int]:
    """Compute IQP encoding gate counts using known formulas.

    For IQP with full entanglement:
    - n_pairs = n*(n-1)/2
    - Hadamard gates: n * reps
    - RZ single-qubit: n * reps
    - RZ in ZZ decomposition: n_pairs * reps
    - CNOT gates: 2 * n_pairs * reps
    - Depth: 3 * reps (conceptual layers: H, RZ, ZZ)
    """
    n = n_features

    if entanglement == "full":
        n_pairs = n * (n - 1) // 2
    elif entanglement == "linear":
        n_pairs = n - 1
    elif entanglement == "circular":
        n_pairs = n if n > 2 else n - 1
    else:
        n_pairs = n * (n - 1) // 2  # Default to full

    h_gates = reps * n
    rz_single = reps * n
    rz_zz = reps * n_pairs
    cnot_gates = reps * 2 * n_pairs

    total_single = h_gates + rz_single + rz_zz
    total_two = cnot_gates

    return {
        "n_qubits": n,
        "depth": reps * 3,
        "gate_count": total_single + total_two,
        "single_qubit_gates": total_single,
        "two_qubit_gates": total_two,
    }


# =============================================================================
# Validation Functions
# =============================================================================

def validate_encoding_result(
    encoding_name: str,
    params: dict[str, Any],
    result: dict[str, Any],
) -> EncodingValidation:
    """Validate a single encoding result against theoretical expectations.

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

    # Get theoretical properties
    theoretical = get_theoretical_properties(encoding_name, params)

    if not theoretical:
        checks.append(ValidationResult(
            check_name="theoretical_computation",
            passed=False,
            message=f"Could not compute theoretical properties for {encoding_name}",
        ))
        return EncodingValidation(encoding_name, params, checks)

    # Check 1: n_qubits matches
    actual_qubits = result.get("n_qubits")
    expected_qubits = theoretical.get("n_qubits")
    if actual_qubits is not None and expected_qubits is not None:
        passed = actual_qubits == expected_qubits
        checks.append(ValidationResult(
            check_name="n_qubits",
            passed=passed,
            message=f"n_qubits: expected={expected_qubits}, actual={actual_qubits}",
            details={"expected": expected_qubits, "actual": actual_qubits},
        ))

    # Check 2: gate_count matches (with tolerance for rounding)
    actual_gates = result.get("gate_count")
    expected_gates = theoretical.get("gate_count")
    if actual_gates is not None and expected_gates is not None:
        # Allow small tolerance for floating point issues
        tolerance = max(1, int(expected_gates * 0.05))
        passed = abs(actual_gates - expected_gates) <= tolerance
        checks.append(ValidationResult(
            check_name="gate_count",
            passed=passed,
            message=f"gate_count: expected={expected_gates}, actual={actual_gates} (tol={tolerance})",
            details={"expected": expected_gates, "actual": actual_gates, "tolerance": tolerance},
        ))

    # Check 3: two_qubit_gates matches
    actual_two_qubit = result.get("two_qubit_gates")
    expected_two_qubit = theoretical.get("two_qubit_gates")
    if actual_two_qubit is not None and expected_two_qubit is not None:
        passed = actual_two_qubit == expected_two_qubit
        checks.append(ValidationResult(
            check_name="two_qubit_gates",
            passed=passed,
            message=f"two_qubit_gates: expected={expected_two_qubit}, actual={actual_two_qubit}",
            details={"expected": expected_two_qubit, "actual": actual_two_qubit},
        ))

    # Check 4: single_qubit_gates matches
    actual_single = result.get("single_qubit_gates")
    expected_single = theoretical.get("single_qubit_gates")
    if actual_single is not None and expected_single is not None:
        tolerance = max(1, int(expected_single * 0.05))
        passed = abs(actual_single - expected_single) <= tolerance
        checks.append(ValidationResult(
            check_name="single_qubit_gates",
            passed=passed,
            message=f"single_qubit_gates: expected={expected_single}, actual={actual_single}",
            details={"expected": expected_single, "actual": actual_single},
        ))

    # Check 5: depth is reasonable (at least 1)
    actual_depth = result.get("depth")
    if actual_depth is not None:
        passed = actual_depth >= 1
        checks.append(ValidationResult(
            check_name="depth_valid",
            passed=passed,
            message=f"depth={actual_depth} (must be >= 1)",
            details={"depth": actual_depth},
        ))

    # Check 6: two_qubit_ratio is consistent
    actual_ratio = result.get("two_qubit_ratio")
    if actual_ratio is not None and actual_gates is not None and actual_gates > 0:
        expected_ratio = actual_two_qubit / actual_gates if actual_two_qubit else 0
        passed = abs(actual_ratio - expected_ratio) < 0.01
        checks.append(ValidationResult(
            check_name="two_qubit_ratio_consistent",
            passed=passed,
            message=f"two_qubit_ratio: expected={expected_ratio:.4f}, actual={actual_ratio:.4f}",
            details={"expected": expected_ratio, "actual": actual_ratio},
        ))

    # Check 7: Non-negative values
    for key in ["gate_count", "two_qubit_gates", "single_qubit_gates", "depth", "n_qubits"]:
        val = result.get(key)
        if val is not None:
            passed = val >= 0
            if not passed:
                checks.append(ValidationResult(
                    check_name=f"{key}_non_negative",
                    passed=False,
                    message=f"{key}={val} is negative",
                ))

    return EncodingValidation(encoding_name, params, checks)


def validate_iqp_specific(params: dict[str, Any], result: dict[str, Any]) -> list[ValidationResult]:
    """Perform IQP-specific validation using known formulas.

    Example: IQP with n_features=4, reps=2, entanglement=full
    - n_pairs = 4*3/2 = 6
    - two_qubit_gates = 2 * 6 * 2 = 24
    - depth = 3 * 2 = 6
    """
    checks: list[ValidationResult] = []

    n_features = params.get("n_features", 4)
    reps = params.get("reps", 2)
    entanglement = params.get("entanglement", "full")

    theoretical = get_iqp_theoretical_formula(n_features, reps, entanglement)

    # Check two_qubit_gates matches formula
    actual_two_qubit = result.get("two_qubit_gates")
    expected_two_qubit = theoretical["two_qubit_gates"]

    if actual_two_qubit is not None:
        passed = actual_two_qubit == expected_two_qubit
        checks.append(ValidationResult(
            check_name="iqp_two_qubit_formula",
            passed=passed,
            message=f"IQP(n={n_features}, reps={reps}, ent={entanglement}): "
                   f"two_qubit_gates expected={expected_two_qubit}, actual={actual_two_qubit}",
            details={
                "formula": f"2 * n_pairs * reps = 2 * {n_features*(n_features-1)//2} * {reps}",
                "expected": expected_two_qubit,
                "actual": actual_two_qubit,
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
        # Abbreviate common parameter names
        key_abbrev = {
            "n_features": "n",
            "reps": "r",
            "entanglement": "ent",
            "n_layers": "L",
            "steps": "s",
            "order": "o",
        }.get(k, k)
        parts.append(f"{key_abbrev}={v}")
    return ", ".join(parts)


def get_encoding_family(encoding_name: str) -> str:
    """Get the family name for an encoding."""
    for family, encodings in ENCODING_FAMILIES.items():
        if encoding_name in encodings:
            return family
    return "Other"


def generate_markdown_table(results: list[dict[str, Any]], validations: list[EncodingValidation]) -> str:
    """Generate a publication-ready Markdown table.

    Parameters
    ----------
    results : list[dict]
        Results from summary.json.
    validations : list[EncodingValidation]
        Validation results.

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "# Stage 1: Resource Profiling Results",
        "",
        "## Summary Table",
        "",
        "| Encoding | Parameters | n_qubits | Depth | Gate Count | 2Q Gates | 1Q Gates | 2Q Ratio |",
        "|----------|------------|----------|-------|------------|----------|----------|----------|",
    ]

    # Group by encoding family
    by_family: dict[str, list[dict[str, Any]]] = {}
    for entry in results:
        result = entry.get("result", {})
        enc_name = entry.get("encoding_name", "unknown")
        family = get_encoding_family(enc_name)
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(entry)

    # Sort by gate count within each family
    for family in sorted(by_family.keys()):
        entries = by_family[family]
        entries.sort(key=lambda x: x.get("result", {}).get("gate_count", 0))

        for entry in entries:
            result = entry.get("result", {})
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})

            n_qubits = result.get("n_qubits", "—")
            depth = result.get("depth", "—")
            gate_count = result.get("gate_count", "—")
            two_qubit = result.get("two_qubit_gates", "—")
            single_qubit = result.get("single_qubit_gates", "—")
            two_qubit_ratio = result.get("two_qubit_ratio", 0)

            ratio_str = f"{two_qubit_ratio:.4f}" if isinstance(two_qubit_ratio, (int, float)) else "—"

            lines.append(
                f"| {enc_name} | {format_params(params)} | {n_qubits} | {depth} | "
                f"{gate_count} | {two_qubit} | {single_qubit} | {ratio_str} |"
            )

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
        r"\caption{Stage 1: Circuit Resource Profiling Results}",
        r"\label{tab:stage1-resources}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Encoding & Parameters & $n_q$ & Depth & Gates & 2Q Gates & 1Q Gates \\",
        r"\midrule",
    ]

    # Group by encoding family
    by_family: dict[str, list[dict[str, Any]]] = {}
    for entry in results:
        enc_name = entry.get("encoding_name", "unknown")
        family = get_encoding_family(enc_name)
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(entry)

    for family in sorted(by_family.keys()):
        entries = by_family[family]
        entries.sort(key=lambda x: x.get("result", {}).get("gate_count", 0))

        # Add family separator
        lines.append(r"\multicolumn{7}{l}{\textit{" + family + r"}} \\")

        for entry in entries[:5]:  # Limit to top 5 per family for space
            result = entry.get("result", {})
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})

            # Escape underscores for LaTeX
            enc_name_tex = enc_name.replace("_", r"\_")
            params_tex = format_params(params).replace("_", r"\_")

            n_qubits = result.get("n_qubits", "—")
            depth = result.get("depth", "—")
            gate_count = result.get("gate_count", "—")
            two_qubit = result.get("two_qubit_gates", "—")
            single_qubit = result.get("single_qubit_gates", "—")

            lines.append(
                f"  {enc_name_tex} & {params_tex} & {n_qubits} & {depth} & "
                f"{gate_count} & {two_qubit} & {single_qubit} \\\\"
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
    """Load results from summary.json file.

    Parameters
    ----------
    path : str
        Path to summary.json file.

    Returns
    -------
    dict
        Loaded results.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
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
    # Load results
    data = load_results(results_path)
    results = data.get("results", [])

    if not results:
        logger.warning("No results found in file")
        return 0, 0, []

    logger.info(f"Loaded {len(results)} results from {results_path}")

    all_validations: list[EncodingValidation] = []
    total_passed = 0
    total_failed = 0

    for entry in results:
        if entry.get("status") != "success":
            continue

        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})
        result = entry.get("result", {})

        # Run standard validation
        validation = validate_encoding_result(enc_name, params, result)

        # Add IQP-specific checks
        if enc_name == "iqp":
            iqp_checks = validate_iqp_specific(params, result)
            validation.results.extend(iqp_checks)

        all_validations.append(validation)
        total_passed += validation.pass_count
        total_failed += validation.fail_count

    return total_passed, total_failed, all_validations


def print_validation_summary(
    total_passed: int,
    total_failed: int,
    validations: list[EncodingValidation],
) -> None:
    """Print validation summary to console."""

    print("\n" + "=" * 70)
    print("STAGE 1: RESOURCE PROFILING VALIDATION")
    print("=" * 70)

    # Summary statistics
    total_encodings = len(validations)
    encodings_passed = sum(1 for v in validations if v.all_passed)
    encodings_failed = total_encodings - encodings_passed

    print(f"\nSummary:")
    print(f"  Total encoding configurations: {total_encodings}")
    print(f"  Configurations passed: {encodings_passed}")
    print(f"  Configurations with failures: {encodings_failed}")
    print(f"  Total checks passed: {total_passed}")
    print(f"  Total checks failed: {total_failed}")

    # Print failures
    if total_failed > 0:
        print("\n" + "-" * 70)
        print("FAILED CHECKS:")
        print("-" * 70)

        for validation in validations:
            if not validation.all_passed:
                print(f"\n{validation.encoding_name} ({format_params(validation.params)}):")
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
        description="Validate Stage 1 Resource Profiling results",
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
            # Load raw results for table generation
            data = load_results(args.results_path)
            results = [r for r in data.get("results", []) if r.get("status") == "success"]

            # Determine output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate Markdown table
            md_table = generate_markdown_table(results, validations)
            md_path = output_dir / "stage1_resources_table.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_table)
            print(f"\nGenerated Markdown table: {md_path}")

            # Generate LaTeX table
            latex_table = generate_latex_table(results)
            latex_path = output_dir / "stage1_resources_table.tex"
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
            print(f"Generated LaTeX table: {latex_path}")

        return 0 if total_failed == 0 else 1

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Stage 1 results not yet available. Run the experiment first:")
        print("  python -m experiments.run_stage --config experiments/configs/stage1_resources.json")
        return 1
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
