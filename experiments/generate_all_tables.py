"""Generate publication-ready tables for all experiment stages.

This script generates both Markdown and LaTeX tables for Stages 1-5,
combining results into a unified summary suitable for research papers.

Usage:
    python experiments/generate_all_tables.py
    python experiments/generate_all_tables.py --output-dir custom/output/
    python experiments/generate_all_tables.py --stages 1 3 5
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

BASE_RESULTS_DIR = _SCRIPT_DIR / "results" / "raw"

STAGE_CONFIGS = {
    1: {
        "name": "Resource Profiling",
        "dir": "stage1_resources",
        "metrics": ["n_qubits", "depth", "gate_count", "two_qubit_gates", "single_qubit_gates"],
        "sort_by": "gate_count",
        "sort_descending": False,
    },
    2: {
        "name": "Simulability Classification",
        "dir": "stage2_simulability",
        "metrics": ["simulability_class", "is_clifford", "is_matchgate"],
        "sort_by": "simulability_class",
        "sort_descending": False,
    },
    3: {
        "name": "Expressibility Analysis",
        "dir": "stage3_expressibility",
        "metrics": ["expressibility", "kl_divergence", "mean_fidelity"],
        "sort_by": "expressibility",
        "sort_descending": True,
    },
    4: {
        "name": "Entanglement Capability",
        "dir": "stage4_entanglement",
        "metrics": ["entanglement_capability", "std_error", "measure"],
        "sort_by": "entanglement_capability",
        "sort_descending": True,
    },
    5: {
        "name": "Trainability Analysis",
        "dir": "stage5_trainability",
        "metrics": ["trainability_estimate", "gradient_variance", "barren_plateau_risk"],
        "sort_by": "trainability_estimate",
        "sort_descending": True,
    },
}

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
class StageResults:
    """Results for a single stage."""

    stage_num: int
    name: str
    results: list[dict[str, Any]]
    available: bool


# =============================================================================
# Helper Functions
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


def load_stage_results(stage_num: int) -> StageResults:
    """Load results for a specific stage.

    Parameters
    ----------
    stage_num : int
        Stage number (1-5).

    Returns
    -------
    StageResults
        Results for the stage.
    """
    config = STAGE_CONFIGS.get(stage_num)
    if not config:
        return StageResults(stage_num, f"Stage {stage_num}", [], False)

    results_path = BASE_RESULTS_DIR / config["dir"] / "summary.json"

    if not results_path.exists():
        logger.warning(f"Results not found for Stage {stage_num}: {results_path}")
        return StageResults(stage_num, config["name"], [], False)

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = [r for r in data.get("results", []) if r.get("status") == "success"]
        logger.info(f"Loaded {len(results)} results for Stage {stage_num}")
        return StageResults(stage_num, config["name"], results, True)
    except Exception as e:
        logger.error(f"Error loading Stage {stage_num} results: {e}")
        return StageResults(stage_num, config["name"], [], False)


def format_value(val: Any, precision: int = 4) -> str:
    """Format a value for display.

    Parameters
    ----------
    val : Any
        Value to format.
    precision : int
        Decimal precision for floats.

    Returns
    -------
    str
        Formatted value string.
    """
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        if val != val:  # NaN check
            return "NaN"
        return f"{val:.{precision}f}"
    return str(val)


# =============================================================================
# Markdown Table Generation
# =============================================================================

def generate_markdown_summary(stages: list[StageResults], output_dir: Path) -> None:
    """Generate a comprehensive Markdown summary of all stages.

    Parameters
    ----------
    stages : list[StageResults]
        Results for all stages.
    output_dir : Path
        Output directory for generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Quantum Encoding Atlas: Experiment Results Summary",
        "",
        "This document contains publication-ready tables for all completed experiment stages.",
        "",
        "## Table of Contents",
        "",
    ]

    # Add TOC
    for stage in stages:
        if stage.available:
            lines.append(f"- [Stage {stage.stage_num}: {stage.name}](#stage-{stage.stage_num}-{stage.name.lower().replace(' ', '-')})")
        else:
            lines.append(f"- Stage {stage.stage_num}: {stage.name} *(not yet completed)*")

    lines.append("")

    # Generate tables for each available stage
    for stage in stages:
        if not stage.available:
            continue

        lines.extend([
            f"## Stage {stage.stage_num}: {stage.name}",
            "",
        ])

        config = STAGE_CONFIGS[stage.stage_num]
        metrics = config["metrics"]

        # Build header
        header = ["Encoding", "Parameters"] + metrics
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Sort results
        sort_key = config["sort_by"]
        sort_desc = config["sort_descending"]

        sorted_results = sorted(
            stage.results,
            key=lambda x: (
                x.get("result", {}).get(sort_key, 0)
                if isinstance(x.get("result", {}).get(sort_key), (int, float))
                else 0
            ),
            reverse=sort_desc,
        )

        # Build rows
        for entry in sorted_results:
            result = entry.get("result", {})
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})

            cells = [enc_name, format_params(params)]
            for metric in metrics:
                val = result.get(metric)
                cells.append(format_value(val))

            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    # Save summary
    summary_path = output_dir / "all_stages_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated Markdown summary: {summary_path}")


def generate_individual_markdown_tables(stages: list[StageResults], output_dir: Path) -> None:
    """Generate individual Markdown tables for each stage.

    Parameters
    ----------
    stages : list[StageResults]
        Results for all stages.
    output_dir : Path
        Output directory.
    """
    for stage in stages:
        if not stage.available:
            continue

        config = STAGE_CONFIGS[stage.stage_num]
        metrics = config["metrics"]

        lines = [
            f"# Stage {stage.stage_num}: {stage.name}",
            "",
            f"Total configurations: {len(stage.results)}",
            "",
        ]

        # Build header
        header = ["Encoding", "Family", "Parameters"] + metrics
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Sort results
        sort_key = config["sort_by"]
        sort_desc = config["sort_descending"]

        sorted_results = sorted(
            stage.results,
            key=lambda x: (
                x.get("result", {}).get(sort_key, 0)
                if isinstance(x.get("result", {}).get(sort_key), (int, float))
                else 0
            ),
            reverse=sort_desc,
        )

        for entry in sorted_results:
            result = entry.get("result", {})
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})
            family = get_encoding_family(enc_name)

            cells = [enc_name, family, format_params(params)]
            for metric in metrics:
                val = result.get(metric)
                cells.append(format_value(val))

            lines.append("| " + " | ".join(cells) + " |")

        # Save individual table
        md_path = output_dir / f"stage{stage.stage_num}_{config['dir'].replace('stage' + str(stage.stage_num) + '_', '')}_table.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated Markdown table: {md_path}")


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_summary(stages: list[StageResults], output_dir: Path) -> None:
    """Generate LaTeX tables for all stages.

    Parameters
    ----------
    stages : list[StageResults]
        Results for all stages.
    output_dir : Path
        Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        if not stage.available:
            continue

        config = STAGE_CONFIGS[stage.stage_num]
        metrics = config["metrics"]

        # Build LaTeX table
        n_cols = 2 + len(metrics)
        col_spec = "ll" + "r" * len(metrics)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Stage " + str(stage.stage_num) + ": " + stage.name + "}",
            r"\label{tab:stage" + str(stage.stage_num) + "}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
        ]

        # Header
        header_cells = ["Encoding", "Parameters"] + [m.replace("_", " ").title() for m in metrics]
        lines.append(" & ".join(header_cells) + r" \\")
        lines.append(r"\midrule")

        # Sort results
        sort_key = config["sort_by"]
        sort_desc = config["sort_descending"]

        sorted_results = sorted(
            stage.results,
            key=lambda x: (
                x.get("result", {}).get(sort_key, 0)
                if isinstance(x.get("result", {}).get(sort_key), (int, float))
                else 0
            ),
            reverse=sort_desc,
        )

        # Limit rows for LaTeX
        for entry in sorted_results[:30]:
            result = entry.get("result", {})
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})

            # Escape underscores
            enc_name_tex = enc_name.replace("_", r"\_")
            params_tex = format_params(params).replace("_", r"\_")

            cells = [enc_name_tex, params_tex]
            for metric in metrics:
                val = result.get(metric)
                cells.append(format_value(val))

            lines.append("  " + " & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        # Save LaTeX table
        latex_path = output_dir / f"stage{stage.stage_num}_table.tex"
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated LaTeX table: {latex_path}")


def generate_combined_latex(stages: list[StageResults], output_dir: Path) -> None:
    """Generate a combined LaTeX document with all tables.

    Parameters
    ----------
    stages : list[StageResults]
        Results for all stages.
    output_dir : Path
        Output directory.
    """
    lines = [
        r"% Quantum Encoding Atlas: Experiment Results Tables",
        r"% Generated automatically - do not edit manually",
        r"",
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1in}",
        r"",
        r"\begin{document}",
        r"",
        r"\title{Quantum Encoding Atlas: Experiment Results}",
        r"\maketitle",
        r"",
    ]

    for stage in stages:
        if not stage.available:
            lines.append(f"% Stage {stage.stage_num} results not available")
            continue

        lines.append(f"\\input{{stage{stage.stage_num}_table.tex}}")
        lines.append(r"\clearpage")
        lines.append("")

    lines.extend([
        r"\end{document}",
    ])

    # Save combined LaTeX
    combined_path = output_dir / "all_tables.tex"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated combined LaTeX document: {combined_path}")


# =============================================================================
# Comparison Table Generation
# =============================================================================

def generate_encoding_comparison_table(stages: list[StageResults], output_dir: Path) -> None:
    """Generate a comparison table showing all metrics for each encoding.

    Parameters
    ----------
    stages : list[StageResults]
        Results for all stages.
    output_dir : Path
        Output directory.
    """
    # Collect metrics per encoding
    encoding_metrics: dict[str, dict[str, Any]] = {}

    for stage in stages:
        if not stage.available:
            continue

        for entry in stage.results:
            enc_name = entry.get("encoding_name", "unknown")
            params = entry.get("encoding_params", {})
            result = entry.get("result", {})

            key = f"{enc_name}_{format_params(params)}"

            if key not in encoding_metrics:
                encoding_metrics[key] = {
                    "encoding": enc_name,
                    "params": params,
                    "family": get_encoding_family(enc_name),
                }

            # Add stage-specific metrics
            stage_prefix = f"s{stage.stage_num}_"
            for metric_name, metric_value in result.items():
                encoding_metrics[key][stage_prefix + metric_name] = metric_value

    # Generate comparison table (Markdown)
    lines = [
        "# Encoding Comparison: Cross-Stage Metrics",
        "",
        "This table compares key metrics across all stages for each encoding configuration.",
        "",
        "| Encoding | Family | Params | Gate Count | Simulability | Expressibility | Entanglement | Trainability |",
        "|----------|--------|--------|------------|--------------|----------------|--------------|--------------|",
    ]

    # Sort by encoding name
    sorted_encodings = sorted(encoding_metrics.values(), key=lambda x: x.get("encoding", ""))

    for enc_data in sorted_encodings:
        enc_name = enc_data.get("encoding", "unknown")
        family = enc_data.get("family", "Other")
        params = format_params(enc_data.get("params", {}))

        gate_count = format_value(enc_data.get("s1_gate_count"))
        simulability = enc_data.get("s2_simulability_class", "—")
        expressibility = format_value(enc_data.get("s3_expressibility"))
        entanglement = format_value(enc_data.get("s4_entanglement_capability"))
        trainability = format_value(enc_data.get("s5_trainability_estimate"))

        lines.append(
            f"| {enc_name} | {family} | {params} | {gate_count} | {simulability} | "
            f"{expressibility} | {entanglement} | {trainability} |"
        )

    # Save comparison table
    comparison_path = output_dir / "encoding_comparison.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated encoding comparison table: {comparison_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready tables for experiment stages",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_SCRIPT_DIR / "results" / "tables"),
        help="Output directory for generated tables (default: experiments/results/tables)",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Stage numbers to generate tables for (default: 1 2 3 4 5)",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("QUANTUM ENCODING ATLAS: TABLE GENERATION")
    print("=" * 70)

    # Load results for all requested stages
    stages: list[StageResults] = []
    for stage_num in args.stages:
        stage = load_stage_results(stage_num)
        stages.append(stage)

    # Print summary
    available = [s for s in stages if s.available]
    unavailable = [s for s in stages if not s.available]

    print(f"\nStages available: {len(available)}/{len(stages)}")
    for s in available:
        print(f"  - Stage {s.stage_num}: {s.name} ({len(s.results)} results)")
    if unavailable:
        print(f"\nStages not yet completed:")
        for s in unavailable:
            print(f"  - Stage {s.stage_num}: {s.name}")

    if not available:
        print("\nNo results available. Run experiments first.")
        return 1

    # Generate tables
    print(f"\nGenerating tables to: {output_dir}")

    # Markdown tables
    generate_markdown_summary(stages, output_dir)
    generate_individual_markdown_tables(stages, output_dir)

    # LaTeX tables
    generate_latex_summary(stages, output_dir)
    generate_combined_latex(stages, output_dir)

    # Comparison table
    generate_encoding_comparison_table(stages, output_dir)

    print("\n" + "-" * 70)
    print("TABLE GENERATION COMPLETE")
    print("-" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
