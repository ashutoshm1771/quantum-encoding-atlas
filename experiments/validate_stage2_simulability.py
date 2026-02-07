"""Validation script for Stage 2: Simulability Classification results.

This script loads results from stage2_simulability/summary.json and verifies
that known simulability classifications are correct:
- AngleEncoding, BasisEncoding, HigherOrderAngleEncoding should be simulable
- IQPEncoding, ZZFeatureMap should be not_simulable
- Checks that is_clifford and is_matchgate flags are consistent with simulability

Usage:
    python experiments/validate_stage2_simulability.py
    python experiments/validate_stage2_simulability.py --results-path custom/path/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
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

DEFAULT_RESULTS_PATH = str(_SCRIPT_DIR / "results" / "raw" / "stage2_simulability" / "summary.json")

# Known simulability classifications
# These are ground-truth values based on theoretical analysis
KNOWN_SIMULABLE: set[str] = {
    "angle",
    "basis",
    "higher_order_angle",
}

# Known ground-truth: these encodings are NOT efficiently simulable in the
# general (asymptotic) case.  They should be classified as either
# ``not_simulable`` (full/unknown topology) or ``conditionally_simulable``
# (linear/circular topology where MPS may work).  They must NEVER be
# classified as ``simulable``.
KNOWN_NOT_EFFICIENTLY_SIMULABLE: set[str] = {
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

# Mapping from simulability class to the ``is_simulable`` boolean flag.
# Only ``simulable`` maps to True.  ``conditionally_simulable`` means the
# circuit *may* be simulable under certain conditions (e.g. bounded
# entanglement entropy) but is NOT guaranteed to be efficient, hence False.
SIMULABILITY_CLASSES = {
    "simulable": True,
    "conditionally_simulable": False,
    "not_simulable": False,
}

# Encoding families for grouping
ENCODING_FAMILIES: dict[str, list[str]] = {
    "Non-Entangling (Simulable)": ["angle", "basis", "higher_order_angle"],
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

def validate_simulability_result(
    encoding_name: str,
    params: dict[str, Any],
    result: dict[str, Any],
) -> EncodingValidation:
    """Validate a single simulability result against known classifications.

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

    # Extract simulability info
    simulability_class = result.get("simulability_class", "unknown")
    is_simulable = result.get("is_simulable")
    is_clifford = result.get("is_clifford", False)
    is_matchgate = result.get("is_matchgate", False)
    reason = result.get("reason", "")

    # Check 1: Verify against known simulable encodings
    if encoding_name in KNOWN_SIMULABLE:
        expected_simulable = True
        passed = simulability_class in ("simulable", "conditionally_simulable")
        checks.append(ValidationResult(
            check_name="known_simulable",
            passed=passed,
            message=f"{encoding_name} expected simulable, got {simulability_class}",
            details={
                "encoding": encoding_name,
                "expected": "simulable",
                "actual": simulability_class,
            },
        ))

    # Check 2: Verify against known NOT efficiently simulable encodings.
    # These must be either "not_simulable" (full/unknown topology) or
    # "conditionally_simulable" (linear/circular topology).  They must
    # never be classified as unconditionally "simulable".
    if encoding_name in KNOWN_NOT_EFFICIENTLY_SIMULABLE:
        passed = simulability_class in ("not_simulable", "conditionally_simulable")
        checks.append(ValidationResult(
            check_name="known_not_efficiently_simulable",
            passed=passed,
            message=(
                f"{encoding_name} expected not efficiently simulable "
                f"(not_simulable or conditionally_simulable), got {simulability_class}"
            ),
            details={
                "encoding": encoding_name,
                "expected": "not_simulable or conditionally_simulable",
                "actual": simulability_class,
            },
        ))

    # Check 3: is_simulable flag consistency
    if is_simulable is not None:
        expected_flag = SIMULABILITY_CLASSES.get(simulability_class, False)
        passed = is_simulable == expected_flag
        checks.append(ValidationResult(
            check_name="is_simulable_consistency",
            passed=passed,
            message=f"is_simulable={is_simulable} vs class={simulability_class} "
                   f"(expected is_simulable={expected_flag})",
            details={
                "is_simulable": is_simulable,
                "simulability_class": simulability_class,
                "expected_flag": expected_flag,
            },
        ))

    # Check 4: Clifford circuits must be simulable
    if is_clifford:
        passed = simulability_class in ("simulable", "conditionally_simulable")
        checks.append(ValidationResult(
            check_name="clifford_simulable",
            passed=passed,
            message=f"is_clifford=True but simulability_class={simulability_class}",
            details={
                "is_clifford": is_clifford,
                "simulability_class": simulability_class,
            },
        ))

    # Check 5: Matchgate circuits must be simulable
    if is_matchgate:
        passed = simulability_class in ("simulable", "conditionally_simulable")
        checks.append(ValidationResult(
            check_name="matchgate_simulable",
            passed=passed,
            message=f"is_matchgate=True but simulability_class={simulability_class}",
            details={
                "is_matchgate": is_matchgate,
                "simulability_class": simulability_class,
            },
        ))

    # Check 6: Non-entangling encodings should be simulable
    # Based on encoding properties
    if encoding_name in KNOWN_SIMULABLE:
        # These encodings are product-state only (non-entangling)
        passed = simulability_class == "simulable"
        checks.append(ValidationResult(
            check_name="product_state_simulable",
            passed=passed,
            message=f"Non-entangling {encoding_name} should be fully simulable",
            details={
                "encoding": encoding_name,
                "simulability_class": simulability_class,
            },
        ))

    # Check 7: Reason field should be non-empty
    if reason:
        passed = len(reason) > 0
    else:
        passed = False
    checks.append(ValidationResult(
        check_name="reason_provided",
        passed=passed,
        message=f"Reason provided: {reason[:50]}..." if reason else "No reason provided",
    ))

    # Check 8: Valid simulability class
    valid_classes = {"simulable", "conditionally_simulable", "not_simulable"}
    passed = simulability_class in valid_classes
    checks.append(ValidationResult(
        check_name="valid_simulability_class",
        passed=passed,
        message=f"simulability_class='{simulability_class}' is {'valid' if passed else 'invalid'}",
    ))

    return EncodingValidation(encoding_name, params, checks)


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
        "# Stage 2: Simulability Classification Results",
        "",
        "## Summary Table",
        "",
        "| Encoding | Parameters | Simulability | Is Clifford | Is Matchgate | Reason |",
        "|----------|------------|--------------|-------------|--------------|--------|",
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
        entries.sort(key=lambda x: x.get("encoding_name", ""))

        for entry in entries:
            result = entry.get("result", {})
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})

            sim_class = result.get("simulability_class", "—")
            is_clifford = "Yes" if result.get("is_clifford") else "No"
            is_matchgate = "Yes" if result.get("is_matchgate") else "No"
            reason = result.get("reason", "—")

            # Truncate reason for table
            if len(reason) > 50:
                reason = reason[:47] + "..."

            # Add emoji for simulability
            sim_emoji = "✅" if sim_class == "simulable" else ("⚠️" if sim_class == "conditionally_simulable" else "❌")

            lines.append(
                f"| {enc_name} | {format_params(params)} | {sim_emoji} {sim_class} | "
                f"{is_clifford} | {is_matchgate} | {reason} |"
            )

    # Add legend
    lines.extend([
        "",
        "## Legend",
        "",
        "- ✅ **simulable**: Can always be efficiently simulated classically",
        "- ⚠️ **conditionally_simulable**: May be simulable depending on parameters",
        "- ❌ **not_simulable**: Believed to be hard to simulate classically",
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
        r"\caption{Stage 2: Simulability Classification of Quantum Encodings}",
        r"\label{tab:stage2-simulability}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Encoding & Parameters & Simulability & Clifford & Matchgate \\",
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
        entries.sort(key=lambda x: x.get("encoding_name", ""))

        # Add family separator
        lines.append(r"\multicolumn{5}{l}{\textit{" + family + r"}} \\")

        for entry in entries:
            result = entry.get("result", {})
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})

            # Escape underscores for LaTeX
            enc_name_tex = enc_name.replace("_", r"\_")
            params_tex = format_params(params).replace("_", r"\_")

            sim_class = result.get("simulability_class", "—")
            is_clifford = r"\checkmark" if result.get("is_clifford") else "—"
            is_matchgate = r"\checkmark" if result.get("is_matchgate") else "—"

            lines.append(
                f"  {enc_name_tex} & {params_tex} & {sim_class} & {is_clifford} & {is_matchgate} \\\\"
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

    for entry in results:
        if entry.get("status") != "success":
            continue

        enc_name = entry.get("encoding_name", "unknown")
        params = entry.get("encoding_params", {})
        result = entry.get("result", {})

        validation = validate_simulability_result(enc_name, params, result)
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
    print("STAGE 2: SIMULABILITY CLASSIFICATION VALIDATION")
    print("=" * 70)

    # Summary statistics
    total_encodings = len(validations)
    encodings_passed = sum(1 for v in validations if v.all_passed)
    encodings_failed = total_encodings - encodings_passed

    # Count by simulability class
    by_class: dict[str, int] = {}
    for v in validations:
        for result in v.results:
            if result.check_name == "valid_simulability_class" and result.details:
                sim_class = result.details.get("simulability_class", "unknown")
                by_class[sim_class] = by_class.get(sim_class, 0) + 1

    print(f"\nSummary:")
    print(f"  Total encoding configurations: {total_encodings}")
    print(f"  Configurations passed: {encodings_passed}")
    print(f"  Configurations with failures: {encodings_failed}")
    print(f"  Total checks passed: {total_passed}")
    print(f"  Total checks failed: {total_failed}")

    # Print simulability classification breakdown
    print("\nSimulability Classification Breakdown:")
    for entry in validations:
        result = entry.results[0] if entry.results else None
        if result and result.details:
            print(f"  {entry.encoding_name}: {result.details.get('actual', 'unknown')}")

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
        description="Validate Stage 2 Simulability Classification results",
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
            md_path = output_dir / "stage2_simulability_table.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_table)
            print(f"\nGenerated Markdown table: {md_path}")

            # Generate LaTeX table
            latex_table = generate_latex_table(results)
            latex_path = output_dir / "stage2_simulability_table.tex"
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
            print(f"Generated LaTeX table: {latex_path}")

        return 0 if total_failed == 0 else 1

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Stage 2 results not yet available. Run the experiment first:")
        print("  python -m experiments.run_stage --config experiments/configs/stage2_simulability.json")
        return 1
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
