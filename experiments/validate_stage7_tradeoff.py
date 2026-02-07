"""Validation script for Stage 7: Scaling & Tradeoff Analysis results.

This script loads results from stage7_tradeoff/ and performs sanity checks
to verify that all JSON outputs, tables, and figures are valid and complete.

Usage:
    python experiments/validate_stage7_tradeoff.py
    python experiments/validate_stage7_tradeoff.py --results-dir custom/path/stage7_tradeoff
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_RESULTS_DIR = str(
    _SCRIPT_DIR / "results" / "raw" / "stage7_tradeoff"
)
DEFAULT_FIGURE_DIR = str(_SCRIPT_DIR / "results" / "figures")

# Expected output files
EXPECTED_JSON_FILES = [
    "hypothesis_verdicts.json",
    "pareto_front.json",
    "rankings.json",
    "pairwise_comparisons.json",
]

EXPECTED_TABLE_FILES = [
    "ranking_table.md",
    "ranking_table.tex",
    "hypothesis_table.md",
    "hypothesis_table.tex",
]

EXPECTED_FIGURE_NAMES = [
    "accuracy_vs_depth",
    "accuracy_vs_trainability",
    "expressibility_vs_accuracy",
    "equivariant_comparison",
    "barren_plateau_regression",
    "vqc_vs_kernel_ranking",
    "pareto_front_2d",
    "pareto_front_parallel",
    "ranking_sensitivity",
    "quantum_vs_classical",
    "family_radar",
    "noise_sensitivity",
    "kta_vs_accuracy",
    "overfitting_gap",
    "expressibility_entanglement",
    "efficiency_frontier",
    "scaling_behavior",
    "pairwise_heatmap",
    "convergence_analysis",
]

VALID_VERDICTS = {"supported", "refuted", "inconclusive"}

HYPOTHESIS_IDS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]

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
class ValidationSuite:
    """Collection of validation results for Stage 7."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def add(self, check_name: str, passed: bool, message: str,
            details: dict[str, Any] | None = None) -> None:
        self.results.append(ValidationResult(check_name, passed, message, details))


# =============================================================================
# Validation Functions
# =============================================================================

def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def validate_json_files_exist(results_dir: Path, suite: ValidationSuite) -> dict[str, Any]:
    """Check that all expected JSON files exist and parse correctly.

    Returns a dict mapping filename -> parsed data for further checks.
    """
    loaded: dict[str, Any] = {}

    for filename in EXPECTED_JSON_FILES:
        path = results_dir / filename
        exists = path.is_file()
        suite.add(
            f"json_exists_{filename}",
            exists,
            f"JSON file {'exists' if exists else 'MISSING'}: {filename}",
        )
        if exists:
            data = _load_json(path)
            parses = data is not None
            suite.add(
                f"json_parses_{filename}",
                parses,
                f"JSON file {'parses' if parses else 'FAILS TO PARSE'}: {filename}",
            )
            if parses:
                loaded[filename] = data

    return loaded


def validate_schema_versions(loaded: dict[str, Any], suite: ValidationSuite) -> None:
    """Check that all JSON files have schema_version == '1.0'."""
    for filename, data in loaded.items():
        version = data.get("schema_version")
        passed = version == "1.0"
        suite.add(
            f"schema_version_{filename}",
            passed,
            f"schema_version in {filename}: "
            f"{'1.0' if passed else repr(version)} "
            f"({'OK' if passed else 'EXPECTED 1.0'})",
        )


def validate_hypothesis_verdicts(loaded: dict[str, Any], suite: ValidationSuite) -> None:
    """Check that each hypothesis verdict is one of supported/refuted/inconclusive."""
    data = loaded.get("hypothesis_verdicts.json")
    if data is None:
        suite.add(
            "hypothesis_verdicts_available",
            False,
            "hypothesis_verdicts.json not loaded — skipping verdict checks",
        )
        return

    verdicts = data.get("hypothesis_verdicts", data.get("verdicts", data))

    for hid in HYPOTHESIS_IDS:
        entry = verdicts.get(hid)
        if entry is None:
            suite.add(
                f"hypothesis_{hid}_present",
                False,
                f"Hypothesis {hid} not found in verdicts",
            )
            continue

        # Verdict may be nested: {"verdict": "supported", ...} or directly a string
        if isinstance(entry, dict):
            verdict_val = entry.get("verdict", entry.get("status"))
        else:
            verdict_val = entry

        valid = verdict_val in VALID_VERDICTS
        suite.add(
            f"hypothesis_{hid}_valid",
            valid,
            f"{hid} verdict = {repr(verdict_val)} "
            f"({'OK' if valid else 'INVALID — expected one of ' + str(VALID_VERDICTS)})",
        )


def validate_pareto_front(loaded: dict[str, Any], suite: ValidationSuite) -> None:
    """Check that Pareto front is non-empty."""
    data = loaded.get("pareto_front.json")
    if data is None:
        suite.add(
            "pareto_front_available",
            False,
            "pareto_front.json not loaded — skipping Pareto checks",
        )
        return

    front = data.get("pareto_optimal", data.get("pareto_front", data.get("front", [])))
    if isinstance(front, dict):
        n_entries = len(front)
    elif isinstance(front, list):
        n_entries = len(front)
    else:
        n_entries = 0

    passed = n_entries > 0
    suite.add(
        "pareto_front_non_empty",
        passed,
        f"Pareto front has {n_entries} entries "
        f"({'OK' if passed else 'EMPTY — expected at least 1'})",
    )


def validate_rankings(loaded: dict[str, Any], suite: ValidationSuite) -> None:
    """Check that rankings have no NaN scores."""
    data = loaded.get("rankings.json")
    if data is None:
        suite.add(
            "rankings_available",
            False,
            "rankings.json not loaded — skipping ranking checks",
        )
        return

    rankings = data.get("rankings", data.get("ranking", []))
    if isinstance(rankings, dict):
        # Convert dict to list for iteration
        items = list(rankings.values()) if rankings else []
    elif isinstance(rankings, list):
        items = rankings
    else:
        items = []

    has_entries = len(items) > 0
    suite.add(
        "rankings_non_empty",
        has_entries,
        f"Rankings has {len(items)} entries "
        f"({'OK' if has_entries else 'EMPTY'})",
    )

    if not has_entries:
        return

    nan_found = []
    for item in items:
        if isinstance(item, dict):
            score = item.get("score", item.get("weighted_score"))
            name = item.get("encoding", item.get("name", "unknown"))
            if score is not None and (
                (isinstance(score, float) and math.isnan(score))
                or score == "NaN"
                or score is None
            ):
                nan_found.append(name)

    no_nans = len(nan_found) == 0
    suite.add(
        "rankings_no_nan",
        no_nans,
        f"Rankings NaN check: "
        f"{'no NaN scores found' if no_nans else f'NaN in: {nan_found}'}",
    )


def validate_tables(results_dir: Path, suite: ValidationSuite) -> None:
    """Check that table files exist and are non-empty."""
    for filename in EXPECTED_TABLE_FILES:
        path = results_dir / filename
        exists = path.is_file()
        suite.add(
            f"table_exists_{filename}",
            exists,
            f"Table file {'exists' if exists else 'MISSING'}: {filename}",
        )
        if exists:
            size = path.stat().st_size
            non_empty = size > 10  # More than just a newline
            suite.add(
                f"table_non_empty_{filename}",
                non_empty,
                f"Table file {filename}: {size} bytes "
                f"({'OK' if non_empty else 'EMPTY or trivial'})",
            )


def validate_figures(figure_dir: Path, suite: ValidationSuite) -> None:
    """Check that figure PNG files exist."""
    if not figure_dir.is_dir():
        suite.add(
            "figure_dir_exists",
            False,
            f"Figure directory MISSING: {figure_dir}",
        )
        return

    suite.add(
        "figure_dir_exists",
        True,
        f"Figure directory exists: {figure_dir}",
    )

    found_count = 0
    missing_names = []

    for name in EXPECTED_FIGURE_NAMES:
        png_path = figure_dir / f"{name}.png"
        exists = png_path.is_file()
        if exists:
            found_count += 1
        else:
            missing_names.append(name)

    all_found = found_count == len(EXPECTED_FIGURE_NAMES)
    suite.add(
        "figures_png_complete",
        all_found,
        f"PNG figures: {found_count}/{len(EXPECTED_FIGURE_NAMES)} found"
        + (f" — missing: {missing_names}" if missing_names else ""),
    )

    # Check PDFs as well (bonus, not required)
    pdf_count = sum(
        1
        for name in EXPECTED_FIGURE_NAMES
        if (figure_dir / f"{name}.pdf").is_file()
    )
    suite.add(
        "figures_pdf_count",
        pdf_count > 0,
        f"PDF figures: {pdf_count}/{len(EXPECTED_FIGURE_NAMES)} found",
    )


def validate_summary_json(results_dir: Path, suite: ValidationSuite) -> None:
    """Check the main summary.json if it exists."""
    path = results_dir / "summary.json"
    if not path.is_file():
        # summary.json is written by the runner, not by tradeoff.py directly
        # So it may or may not exist depending on how the stage was run
        suite.add(
            "summary_json_exists",
            True,  # Not a failure if missing
            "summary.json not present (optional — runner writes this)",
        )
        return

    data = _load_json(path)
    if data is None:
        suite.add(
            "summary_json_parses",
            False,
            "summary.json exists but fails to parse",
        )
        return

    suite.add(
        "summary_json_parses",
        True,
        "summary.json parses correctly",
    )


# =============================================================================
# Main Validation
# =============================================================================

def run_validation(
    results_dir: str,
    figure_dir: str,
    check_figures: bool = True,
) -> ValidationSuite:
    """Run all Stage 7 validation checks.

    Parameters
    ----------
    results_dir : str
        Path to stage7_tradeoff results directory.
    figure_dir : str
        Path to figures directory.
    check_figures : bool
        Whether to check for figure files (set False if --quick was used).

    Returns
    -------
    ValidationSuite
        All validation results.
    """
    suite = ValidationSuite()
    results_path = Path(results_dir)
    figures_path = Path(figure_dir)

    # Check results directory exists
    if not results_path.is_dir():
        suite.add(
            "results_dir_exists",
            False,
            f"Results directory MISSING: {results_path}",
        )
        return suite

    suite.add(
        "results_dir_exists",
        True,
        f"Results directory exists: {results_path}",
    )

    # 1. JSON files exist and parse
    loaded = validate_json_files_exist(results_path, suite)

    # 2. Schema versions
    validate_schema_versions(loaded, suite)

    # 3. Hypothesis verdicts
    validate_hypothesis_verdicts(loaded, suite)

    # 4. Pareto front non-empty
    validate_pareto_front(loaded, suite)

    # 5. Rankings no NaN
    validate_rankings(loaded, suite)

    # 6. Tables non-empty
    validate_tables(results_path, suite)

    # 7. Figures exist
    if check_figures:
        validate_figures(figures_path, suite)

    # 8. Summary JSON (optional)
    validate_summary_json(results_path, suite)

    return suite


def print_validation_summary(suite: ValidationSuite) -> None:
    """Print validation summary to console."""
    print("\n" + "=" * 70)
    print("STAGE 7: SCALING & TRADEOFF ANALYSIS VALIDATION")
    print("=" * 70)

    print(f"\nSummary:")
    print(f"  Total checks: {suite.pass_count + suite.fail_count}")
    print(f"  Passed: {suite.pass_count}")
    print(f"  Failed: {suite.fail_count}")

    # Print all results grouped by pass/fail
    if suite.fail_count > 0:
        print("\n" + "-" * 70)
        print("FAILED CHECKS:")
        print("-" * 70)
        for r in suite.results:
            if not r.passed:
                print(f"  [FAIL] {r.check_name}: {r.message}")

    print("\n" + "-" * 70)
    print("PASSED CHECKS:")
    print("-" * 70)
    for r in suite.results:
        if r.passed:
            print(f"  [PASS] {r.check_name}: {r.message}")

    # Final result
    print("\n" + "=" * 70)
    if suite.all_passed:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: {suite.fail_count} CHECKS FAILED")
    print("=" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Stage 7 Tradeoff Analysis results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to stage7_tradeoff results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=DEFAULT_FIGURE_DIR,
        help=f"Path to figures directory (default: {DEFAULT_FIGURE_DIR})",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure validation (e.g., if run with --quick)",
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
        suite = run_validation(
            results_dir=args.results_dir,
            figure_dir=args.figure_dir,
            check_figures=not args.no_figures,
        )

        print_validation_summary(suite)

        return 0 if suite.all_passed else 1

    except Exception as e:
        logger.exception("Validation failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
