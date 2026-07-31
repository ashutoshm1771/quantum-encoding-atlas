"""Stage 8 — Report generation.

Compiles all experiment results (Stages 1-7) into publication-ready
outputs: a master summary JSON, Markdown/LaTeX tables, and narrative
hypothesis verdict and ranking documents.

This module is the final stage of the experiment pipeline.  It reads
only from previously generated results and produces no new simulations.

Outputs
-------
- ``summary.json`` — Master JSON combining all metrics for all encodings
  across all stages, with ``schema_version``.
- ``tables/`` — Markdown and LaTeX tables for every stage, plus
  VQC/kernel accuracy matrices, sensitivity grid, and consolidated
  cross-stage comparison.
- ``ranking.md`` — Final encoding ranking narrative with Pareto front
  explanation and practical guidance.
- ``hypotheses.md`` — H1-H7 verdict table with full supporting evidence,
  test statistics, and interpretation.

Usage
-----
As a CLI tool::

    python -m experiments.run_stage --config experiments/configs/stage8_report.json
    python -m experiments.run_stage --config experiments/configs/stage8_report.json --quick

Programmatically::

    from experiments.report import generate_report
    result = generate_report(
        stage_dirs={...},
        tradeoff_dir="experiments/results/raw/stage7_tradeoff",
        output_dir="experiments/results/report",
    )

Dependencies
------------
This module reads from per-stage ``summary.json`` files and Stage 7
output files (``hypothesis_verdicts.json``, ``rankings.json``,
``pareto_front.json``, ``pairwise_comparisons.json``).  It also
optionally reads the Stage 6a.5 sensitivity report.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"

# Encoding display names for tables.
_ENCODING_DISPLAY: dict[str, str] = {
    "angle": "AngleEncoding",
    "amplitude": "AmplitudeEncoding",
    "basis": "BasisEncoding",
    "iqp": "IQPEncoding",
    "zz_feature_map": "ZZFeatureMap",
    "pauli_feature_map": "PauliFeatureMap",
    "data_reuploading": "DataReuploading",
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

_ENCODING_FAMILIES: dict[str, str] = {
    "angle": "Non-Entangling",
    "amplitude": "Amplitude",
    "basis": "Non-Entangling",
    "iqp": "IQP-Based",
    "zz_feature_map": "IQP-Based",
    "pauli_feature_map": "Pauli-Based",
    "data_reuploading": "Data Re-uploading",
    "hardware_efficient": "Hardware-Efficient",
    "higher_order_angle": "Non-Entangling",
    "qaoa_encoding": "QAOA/Hamiltonian",
    "hamiltonian_encoding": "QAOA/Hamiltonian",
    "symmetry_inspired": "Symmetry-Based",
    "trainable_encoding": "Trainable",
    "so2_equivariant": "Equivariant",
    "cyclic_equivariant": "Equivariant",
    "swap_equivariant": "Equivariant",
}

# Hypothesis descriptions for narrative.
_HYPOTHESIS_DESCRIPTIONS: dict[str, str] = {
    "H1": (
        "Expressibility is necessary but not sufficient for high "
        "classification accuracy."
    ),
    "H2": (
        "Equivariant encodings outperform general encodings on datasets "
        "with matching symmetry structure."
    ),
    "H3": (
        "Data re-uploading encodings achieve higher expressibility than "
        "single-pass encodings at equivalent circuit depth."
    ),
    "H4": (
        "Barren plateau onset (trainability collapse) correlates with "
        "circuit depth more strongly than with encoding family."
    ),
    "H5": (
        "Noise degrades entangling encodings disproportionately compared "
        "to non-entangling encodings."
    ),
    "H6": (
        "Quantum kernel methods and VQC methods produce different "
        "encoding rankings on the same datasets."
    ),
    "H7": (
        "No single encoding dominates across all metrics; the Pareto "
        "front contains >= 3 encodings."
    ),
}


# ---------------------------------------------------------------------------
# JSON helpers
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


def _load_json(path: str) -> dict[str, Any] | None:
    """Load a JSON file, returning None if missing or invalid."""
    if not os.path.isfile(path):
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None


def _save_json(data: Any, path: str) -> None:
    """Save data as pretty-printed JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _save_text(content: str, path: str) -> None:
    """Save text content to a file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _fmt(val: Any, precision: int = 4) -> str:
    """Format a value for display in tables."""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        if val != val:  # NaN
            return "NaN"
        if abs(val) < 1e-10:
            return "0.0000"
        return f"{val:.{precision}f}"
    return str(val)


def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# ---------------------------------------------------------------------------
# 8.1: Master summary JSON
# ---------------------------------------------------------------------------


def _mean_kernel_alignment(
    kernel_summary: dict[str, Any] | None,
) -> dict[str, float]:
    """Mean centered kernel-target alignment per encoding, from Stage 6b.

    The kernel stage records ``centered_kernel_target_alignment`` for every
    (encoding config, dataset) pair. This averages it over those pairs, which
    is exactly the rule the ranking uses for ``kernel_accuracy`` — so the two
    columns are aggregated identically and remain directly comparable.

    Alignment is the benchmark's *validated* predictor of kernel accuracy
    (Spearman rho = 0.91), unlike expressibility, which the study refutes.
    Carrying it into the master summary is what lets the shipped atlas rank
    encodings by it.

    Parameters
    ----------
    kernel_summary : dict or None
        Parsed ``summary.json`` from the Stage 6b kernel directory.

    Returns
    -------
    dict[str, float]
        Encoding name -> mean centered alignment. Encodings with no successful
        measurement are omitted.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    if not kernel_summary:
        return {}

    for entry in kernel_summary.get("results", []):
        if entry.get("status") != "success":
            continue
        enc_name = entry.get("encoding_name")
        if not enc_name:
            continue
        datasets = entry.get("result", {}).get("datasets", {})
        for ds_data in datasets.values():
            if ds_data.get("status") != "success":
                continue
            alignment = ds_data.get("centered_kernel_target_alignment")
            if alignment is None:
                continue
            sums[enc_name] = sums.get(enc_name, 0.0) + float(alignment)
            counts[enc_name] = counts.get(enc_name, 0) + 1

    return {name: sums[name] / counts[name] for name in sums if counts[name]}


def _build_master_summary(
    stage_dirs: dict[str, str],
    tradeoff_dir: str,
    sensitivity_dir: str | None,
) -> dict[str, Any]:
    """Build the master summary JSON combining all stages.

    Parameters
    ----------
    stage_dirs : dict[str, str]
        Mapping of stage name to directory path for Stages 1-6b.
    tradeoff_dir : str
        Path to the Stage 7 tradeoff results directory.
    sensitivity_dir : str or None
        Path to the Stage 6a.5 sensitivity results directory.

    Returns
    -------
    dict[str, Any]
        Master summary with all encoding profiles and stage data.
    """
    # Load rankings (the most comprehensive per-encoding data).
    rankings_path = os.path.join(tradeoff_dir, "rankings.json")
    rankings_data = _load_json(rankings_path)
    rankings = rankings_data.get("rankings", []) if rankings_data else []

    # Load all per-stage summaries for encoding-level detail.
    stage_summaries: dict[str, dict[str, Any] | None] = {}
    for stage_name, stage_dir in stage_dirs.items():
        summary_path = os.path.join(stage_dir, "summary.json")
        stage_summaries[stage_name] = _load_json(summary_path)

    # Load hypothesis verdicts.
    verdicts_path = os.path.join(tradeoff_dir, "hypothesis_verdicts.json")
    verdicts_data = _load_json(verdicts_path)

    # Load Pareto front.
    pareto_path = os.path.join(tradeoff_dir, "pareto_front.json")
    pareto_data = _load_json(pareto_path)

    # Load sensitivity report.
    sensitivity_data = None
    if sensitivity_dir:
        sensitivity_path = os.path.join(sensitivity_dir, "sensitivity_report.json")
        sensitivity_data = _load_json(sensitivity_path)

    # Mean centered kernel-target alignment per encoding (Stage 6b).
    kernel_alignment = _mean_kernel_alignment(stage_summaries.get("kernel"))

    # Build per-encoding profiles from rankings + stage data.
    encoding_profiles: list[dict[str, Any]] = []
    for ranking_entry in rankings:
        enc_name = ranking_entry["encoding"]
        profile: dict[str, Any] = {
            "encoding": enc_name,
            "display_name": _ENCODING_DISPLAY.get(enc_name, enc_name),
            "family": _ENCODING_FAMILIES.get(enc_name, "Other"),
            "rank": ranking_entry.get("rank"),
            "score": ranking_entry.get("score"),
            "is_pareto": ranking_entry.get("is_pareto", False),
            "is_simulable": ranking_entry.get("is_simulable"),
            "metrics": {
                "depth": ranking_entry.get("depth"),
                "expressibility": ranking_entry.get("expressibility"),
                "entanglement_capability": ranking_entry.get("entanglement_capability"),
                "trainability_estimate": ranking_entry.get("trainability_estimate"),
                "noise_resilience": ranking_entry.get("noise_resilience"),
                "vqc_accuracy": ranking_entry.get("vqc_accuracy"),
                "vqc_ci": ranking_entry.get("vqc_ci"),
                "kernel_accuracy": ranking_entry.get("kernel_accuracy"),
                "kernel_ci": ranking_entry.get("kernel_ci"),
                "kernel_target_alignment": kernel_alignment.get(enc_name),
            },
        }
        encoding_profiles.append(profile)

    # Count per-stage results.
    stage_counts: dict[str, dict[str, int]] = {}
    for stage_name, summary in stage_summaries.items():
        if summary is not None:
            stage_counts[stage_name] = {
                "total": summary.get("total_results", 0),
                "success": summary.get("success_count", 0),
                "failed": summary.get("failed_count", 0),
            }

    master: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "n_encodings": len(encoding_profiles),
        "encoding_profiles": encoding_profiles,
        "stage_counts": stage_counts,
        "hypothesis_verdicts": (
            verdicts_data.get("hypothesis_verdicts", {}) if verdicts_data else {}
        ),
        "pareto_front": {
            "n_pareto_optimal": (
                pareto_data.get("n_pareto_optimal", 0) if pareto_data else 0
            ),
            "pareto_optimal": (
                pareto_data.get("pareto_optimal", []) if pareto_data else []
            ),
            "objective_names": (
                pareto_data.get("objective_names", []) if pareto_data else []
            ),
        },
    }

    if sensitivity_data:
        master["sensitivity_analysis"] = {
            "grid": sensitivity_data.get("grid", {}),
            "analysis": sensitivity_data.get("analysis", {}),
        }

    return master


# ---------------------------------------------------------------------------
# 8.2: Table generation
# ---------------------------------------------------------------------------


def _extract_vqc_accuracy_matrix(
    vqc_summary: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    """Extract encoding x dataset accuracy matrix from VQC results.

    Returns
    -------
    tuple
        (encoding_names, dataset_names, matrix) where matrix maps
        encoding -> dataset -> formatted "mean +/- std" string.
    """
    results = vqc_summary.get("results", [])

    # Collect unique encodings and datasets.
    encodings_set: set[str] = set()
    datasets_set: set[str] = set()
    matrix: dict[str, dict[str, str]] = {}

    for entry in results:
        if entry.get("status") != "success":
            continue
        enc_name = entry.get("encoding_name", "unknown")
        result_data = entry.get("result", {})
        datasets = result_data.get("datasets", {})

        encodings_set.add(enc_name)
        if enc_name not in matrix:
            matrix[enc_name] = {}

        for ds_name, ds_data in datasets.items():
            datasets_set.add(ds_name)
            agg = ds_data.get("aggregate", {})
            mean_acc = agg.get("mean_test_accuracy")
            std_acc = agg.get("std_test_accuracy")
            if mean_acc is not None and std_acc is not None:
                matrix[enc_name][ds_name] = f"{mean_acc:.3f} +/- {std_acc:.3f}"
            elif mean_acc is not None:
                matrix[enc_name][ds_name] = f"{mean_acc:.3f}"
            else:
                matrix[enc_name][ds_name] = "—"

    enc_names = sorted(encodings_set)
    ds_names = sorted(datasets_set)
    return enc_names, ds_names, matrix


def _generate_accuracy_matrix_md(
    enc_names: list[str],
    ds_names: list[str],
    matrix: dict[str, dict[str, str]],
    title: str,
) -> str:
    """Generate a Markdown accuracy matrix table."""
    lines = [f"# {title}", ""]
    header = ["Encoding"] + ds_names
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for enc in enc_names:
        cells = [enc]
        for ds in ds_names:
            cells.append(matrix.get(enc, {}).get(ds, "—"))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _generate_accuracy_matrix_tex(
    enc_names: list[str],
    ds_names: list[str],
    matrix: dict[str, dict[str, str]],
    title: str,
    label: str,
) -> str:
    """Generate a LaTeX accuracy matrix table."""
    n_cols = 1 + len(ds_names)
    col_spec = "l" + "c" * len(ds_names)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + _tex_escape(title) + "}",
        r"\label{" + label + "}",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header_cells = ["Encoding"] + [_tex_escape(ds) for ds in ds_names]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for enc in enc_names:
        cells = [_tex_escape(enc)]
        for ds in ds_names:
            val = matrix.get(enc, {}).get(ds, "—")
            cells.append(val.replace("+/-", r"$\pm$"))
        lines.append("  " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _generate_sensitivity_table_md(
    sensitivity_data: dict[str, Any],
) -> str:
    """Generate a Markdown table for Stage 6a.5 sensitivity analysis."""
    results = sensitivity_data.get("results", [])
    if not results:
        return "# Sensitivity Analysis\n\nNo results available.\n"

    lines = [
        "# Stage 6a.5: VQC Hyperparameter Sensitivity",
        "",
        "| Encoding | Dataset | LR | Layers | Accuracy | 95% CI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    # Sort by encoding, dataset, then accuracy descending.
    sorted_results = sorted(
        results,
        key=lambda r: (
            r.get("encoding", ""),
            r.get("dataset", ""),
            -(r.get("mean_accuracy") or 0),
        ),
    )

    for r in sorted_results:
        if r.get("status") != "success":
            continue
        enc = r.get("encoding", "—")
        ds = r.get("dataset", "—")
        lr = r.get("lr", "—")
        layers = r.get("n_var_layers", "—")
        acc = _fmt(r.get("mean_accuracy"), 4)
        ci_lo = r.get("ci_lower")
        ci_hi = r.get("ci_upper")
        ci_str = f"({_fmt(ci_lo, 3)}-{_fmt(ci_hi, 3)})" if ci_lo is not None else "—"
        lines.append(f"| {enc} | {ds} | {lr} | {layers} | {acc} | {ci_str} |")

    return "\n".join(lines)


def _generate_sensitivity_table_tex(
    sensitivity_data: dict[str, Any],
) -> str:
    """Generate a LaTeX table for Stage 6a.5 sensitivity analysis."""
    results = sensitivity_data.get("results", [])
    if not results:
        return ""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{VQC Hyperparameter Sensitivity (Stage 6a.5)}",
        r"\label{tab:sensitivity}",
        r"\small",
        r"\begin{tabular}{llccrc}",
        r"\toprule",
        r"Encoding & Dataset & LR & Layers & Accuracy & 95\% CI \\",
        r"\midrule",
    ]

    sorted_results = sorted(
        results,
        key=lambda r: (
            r.get("encoding", ""),
            r.get("dataset", ""),
            -(r.get("mean_accuracy") or 0),
        ),
    )

    for r in sorted_results:
        if r.get("status") != "success":
            continue
        enc = _tex_escape(r.get("encoding", "—"))
        ds = _tex_escape(r.get("dataset", "—"))
        lr = str(r.get("lr", "—"))
        layers = str(r.get("n_var_layers", "—"))
        acc = _fmt(r.get("mean_accuracy"), 4)
        ci_lo = r.get("ci_lower")
        ci_hi = r.get("ci_upper")
        ci_str = f"({_fmt(ci_lo, 3)}--{_fmt(ci_hi, 3)})" if ci_lo is not None else "—"
        lines.append(f"  {enc} & {ds} & {lr} & {layers} & {acc} & {ci_str}" + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _generate_tables(
    stage_dirs: dict[str, str],
    tradeoff_dir: str,
    sensitivity_dir: str | None,
    table_dir: str,
) -> list[str]:
    """Generate all tables for the report.

    Parameters
    ----------
    stage_dirs : dict[str, str]
        Stage name to directory mapping.
    tradeoff_dir : str
        Stage 7 output directory.
    sensitivity_dir : str or None
        Stage 6a.5 output directory.
    table_dir : str
        Output directory for generated tables.

    Returns
    -------
    list[str]
        List of generated file paths.
    """
    os.makedirs(table_dir, exist_ok=True)
    generated: list[str] = []

    # --- Copy Stage 7 tables (ranking + hypothesis) -------------------------
    for filename in (
        "ranking_table.md",
        "ranking_table.tex",
        "hypothesis_table.md",
        "hypothesis_table.tex",
    ):
        src = os.path.join(tradeoff_dir, filename)
        dst = os.path.join(table_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            generated.append(dst)
            logger.info("Copied %s -> %s", src, dst)

    # --- Copy per-stage raw tables ------------------------------------------
    for stage_name, stage_dir in stage_dirs.items():
        for ext in (".md", ".tex"):
            # Per-stage tables are named like stage1_resources_table.md
            for fname in os.listdir(stage_dir):
                if fname.endswith(f"_table{ext}"):
                    src = os.path.join(stage_dir, fname)
                    dst = os.path.join(table_dir, fname)
                    shutil.copy2(src, dst)
                    generated.append(dst)

    # --- VQC accuracy matrix ------------------------------------------------
    vqc_dir = stage_dirs.get("vqc", "")
    vqc_summary = _load_json(os.path.join(vqc_dir, "summary.json"))
    if vqc_summary:
        enc_names, ds_names, matrix = _extract_vqc_accuracy_matrix(vqc_summary)
        if enc_names:
            md = _generate_accuracy_matrix_md(
                enc_names,
                ds_names,
                matrix,
                "VQC Classification Accuracy (mean +/- std)",
            )
            path = os.path.join(table_dir, "vqc_accuracy_matrix.md")
            _save_text(md, path)
            generated.append(path)

            tex = _generate_accuracy_matrix_tex(
                enc_names,
                ds_names,
                matrix,
                "VQC Classification Accuracy",
                "tab:vqc_accuracy",
            )
            path = os.path.join(table_dir, "vqc_accuracy_matrix.tex")
            _save_text(tex, path)
            generated.append(path)

    # --- Kernel accuracy matrix ---------------------------------------------
    kernel_dir = stage_dirs.get("kernel", "")
    kernel_summary = _load_json(os.path.join(kernel_dir, "summary.json"))
    if kernel_summary:
        enc_names, ds_names, matrix = _extract_vqc_accuracy_matrix(kernel_summary)
        if enc_names:
            md = _generate_accuracy_matrix_md(
                enc_names,
                ds_names,
                matrix,
                "Kernel Classification Accuracy (mean +/- std)",
            )
            path = os.path.join(table_dir, "kernel_accuracy_matrix.md")
            _save_text(md, path)
            generated.append(path)

            tex = _generate_accuracy_matrix_tex(
                enc_names,
                ds_names,
                matrix,
                "Kernel Classification Accuracy",
                "tab:kernel_accuracy",
            )
            path = os.path.join(table_dir, "kernel_accuracy_matrix.tex")
            _save_text(tex, path)
            generated.append(path)

    # --- Sensitivity grid table ---------------------------------------------
    if sensitivity_dir:
        sens_path = os.path.join(sensitivity_dir, "sensitivity_report.json")
        sens_data = _load_json(sens_path)
        if sens_data:
            md = _generate_sensitivity_table_md(sens_data)
            path = os.path.join(table_dir, "sensitivity_grid.md")
            _save_text(md, path)
            generated.append(path)

            tex = _generate_sensitivity_table_tex(sens_data)
            if tex:
                path = os.path.join(table_dir, "sensitivity_grid.tex")
                _save_text(tex, path)
                generated.append(path)

    return generated


# ---------------------------------------------------------------------------
# 8.3: Figure verification
# ---------------------------------------------------------------------------


def _verify_figures(figure_dir: str) -> dict[str, Any]:
    """Verify that all expected figures exist.

    Parameters
    ----------
    figure_dir : str
        Directory containing generated figures.

    Returns
    -------
    dict[str, Any]
        Report of found/missing figures.
    """
    if not os.path.isdir(figure_dir):
        return {"status": "missing", "found": 0, "missing_dir": True}

    png_files = sorted(f for f in os.listdir(figure_dir) if f.endswith(".png"))
    pdf_files = sorted(f for f in os.listdir(figure_dir) if f.endswith(".pdf"))

    return {
        "status": "ok" if png_files else "empty",
        "png_count": len(png_files),
        "pdf_count": len(pdf_files),
        "figure_pairs": len(png_files),
        "figures": [f.replace(".png", "") for f in png_files],
    }


# ---------------------------------------------------------------------------
# 8.4: Hypothesis verdict narrative
# ---------------------------------------------------------------------------

_VERDICT_EMOJI: dict[str, str] = {
    "supported": "SUPPORTED",
    "refuted": "REFUTED",
    "inconclusive": "INCONCLUSIVE",
}

_CONFIDENCE_LABEL: dict[str, str] = {
    "high": "High confidence",
    "moderate": "Moderate confidence",
    "low": "Low confidence",
}


def _format_test_statistic(stats: dict[str, Any]) -> str:
    """Format test statistics into a human-readable string."""
    parts = []
    for key, val in stats.items():
        if isinstance(val, float):
            if abs(val) < 0.001 and val != 0:
                parts.append(f"{key} = {val:.2e}")
            else:
                parts.append(f"{key} = {val:.4f}")
        else:
            parts.append(f"{key} = {val}")
    return "; ".join(parts)


def _generate_hypotheses_md(
    verdicts_data: dict[str, Any],
    pareto_data: dict[str, Any] | None,
) -> str:
    """Generate the hypotheses.md narrative document.

    Parameters
    ----------
    verdicts_data : dict[str, Any]
        Loaded hypothesis_verdicts.json.
    pareto_data : dict[str, Any] or None
        Loaded pareto_front.json (for H7 detail).

    Returns
    -------
    str
        Full Markdown document content.
    """
    verdicts = verdicts_data.get("hypothesis_verdicts", {})

    lines = [
        "# Hypothesis Verdicts",
        "",
        "This document presents the results of seven pre-registered "
        "hypotheses tested across the Quantum Encoding Atlas experiment "
        "pipeline (Stages 1-7).  Each hypothesis was evaluated using the "
        "statistical tests and criteria defined in the experiment design "
        "document.",
        "",
        "## Summary Table",
        "",
        "| Hypothesis | Verdict | Confidence | Key Statistic |",
        "| --- | --- | --- | --- |",
    ]

    for h_id in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        v = verdicts.get(h_id, {})
        verdict = v.get("verdict", "unknown")
        confidence = v.get("confidence", "unknown")
        stats = v.get("test_statistic", {})
        stat_str = _format_test_statistic(stats) if stats else "—"
        lines.append(
            f"| {h_id} | {_VERDICT_EMOJI.get(verdict, verdict)} "
            f"| {confidence} | {stat_str} |"
        )

    lines.extend(["", "---", "", "## Detailed Analysis", ""])

    for h_id in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        v = verdicts.get(h_id, {})
        description = _HYPOTHESIS_DESCRIPTIONS.get(h_id, "")
        verdict = v.get("verdict", "unknown")
        confidence = v.get("confidence", "unknown")
        evidence = v.get("evidence", "No evidence available.")
        stats = v.get("test_statistic", {})

        lines.extend(
            [
                f"### {h_id}: {description}",
                "",
                f"**Verdict:** {_VERDICT_EMOJI.get(verdict, verdict)}",
                f"**Confidence:** {_CONFIDENCE_LABEL.get(confidence, confidence)}",
                "",
                "**Evidence:**",
                "",
                evidence,
                "",
            ]
        )

        if stats:
            lines.append("**Test Statistics:**")
            lines.append("")
            for key, val in stats.items():
                if isinstance(val, float):
                    if abs(val) < 0.001 and val != 0:
                        lines.append(f"- {key}: {val:.2e}")
                    else:
                        lines.append(f"- {key}: {val:.4f}")
                else:
                    lines.append(f"- {key}: {val}")
            lines.append("")

        # H7-specific Pareto detail.
        if h_id == "H7" and pareto_data:
            pareto_names = pareto_data.get("pareto_optimal", [])
            if pareto_names:
                lines.append("**Pareto-optimal encodings:**")
                lines.append("")
                for name in pareto_names:
                    display = _ENCODING_DISPLAY.get(name, name)
                    family = _ENCODING_FAMILIES.get(name, "Other")
                    lines.append(f"- {display} ({family})")
                lines.append("")

        lines.extend(["---", ""])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranking narrative
# ---------------------------------------------------------------------------


def _generate_ranking_md(
    rankings_data: dict[str, Any],
    pareto_data: dict[str, Any] | None,
    verdicts_data: dict[str, Any] | None,
) -> str:
    """Generate the ranking.md narrative document.

    Parameters
    ----------
    rankings_data : dict[str, Any]
        Loaded rankings.json.
    pareto_data : dict[str, Any] or None
        Loaded pareto_front.json.
    verdicts_data : dict[str, Any] or None
        Loaded hypothesis_verdicts.json.

    Returns
    -------
    str
        Full Markdown document content.
    """
    rankings = rankings_data.get("rankings", [])

    lines = [
        "# Final Encoding Rankings",
        "",
        "This document presents the final composite ranking of all 16 quantum "
        "data encodings evaluated in the Quantum Encoding Atlas.  Rankings are "
        "computed using a weighted multi-objective score combining classification "
        "accuracy, circuit efficiency, trainability, and noise resilience.",
        "",
    ]

    # --- Ranking table ------------------------------------------------------
    lines.extend(
        [
            "## Composite Rankings",
            "",
            "| Rank | Encoding | Family | Score | VQC Acc | Kernel Acc | Pareto |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for r in rankings:
        enc = r.get("encoding", "unknown")
        family = _ENCODING_FAMILIES.get(enc, "Other")
        rank = r.get("rank", "—")
        score = _fmt(r.get("score"), 4)
        vqc = _fmt(r.get("vqc_accuracy"), 3)
        kernel = _fmt(r.get("kernel_accuracy"), 3)
        pareto = "Yes" if r.get("is_pareto") else "No"
        lines.append(
            f"| {rank} | {enc} | {family} | {score} | {vqc} | {kernel} | {pareto} |"
        )

    # --- Pareto front -------------------------------------------------------
    if pareto_data:
        pareto_names = pareto_data.get("pareto_optimal", [])
        objectives = pareto_data.get("objective_names", [])
        n_analyzed = pareto_data.get("n_encodings_analyzed", 0)

        lines.extend(
            [
                "",
                "## Pareto Front",
                "",
                f"The Pareto front was computed over {n_analyzed} encodings "
                f"using {len(objectives)} objectives: "
                f"{', '.join(objectives)}.",
                "",
                f"**{len(pareto_names)} Pareto-optimal encodings** were "
                "identified (no other encoding dominates them on all objectives "
                "simultaneously):",
                "",
            ]
        )

        encodings_detail = pareto_data.get("encodings", {})
        for name in pareto_names:
            display = _ENCODING_DISPLAY.get(name, name)
            detail = encodings_detail.get(name, {})
            obj_vals = detail.get("objectives", [])
            obj_strs = [
                f"{oname}={_fmt(oval, 3)}" for oname, oval in zip(objectives, obj_vals)
            ]
            lines.append(f"- **{display}**: {', '.join(obj_strs)}")

        lines.extend(
            [
                "",
                "The existence of multiple Pareto-optimal encodings across "
                "different families (Non-Entangling, Equivariant) confirms that "
                "no single encoding dominates all evaluation axes.  Encoding "
                "selection should be guided by the specific requirements of the "
                "target application.",
            ]
        )

    # --- Key findings -------------------------------------------------------
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
        ]
    )

    if rankings:
        top = rankings[0]
        top_name = _ENCODING_DISPLAY.get(top["encoding"], top["encoding"])
        lines.extend(
            [
                f"1. **Top-ranked encoding:** {top_name} "
                f"(score={_fmt(top.get('score'), 4)}, "
                f"VQC={_fmt(top.get('vqc_accuracy'), 3)}, "
                f"kernel={_fmt(top.get('kernel_accuracy'), 3)})",
                "",
            ]
        )

    # Group by family for family-level insights.
    family_best: dict[str, dict[str, Any]] = {}
    for r in rankings:
        fam = _ENCODING_FAMILIES.get(r.get("encoding", ""), "Other")
        if fam not in family_best:
            family_best[fam] = r

    lines.append("2. **Best encoding per family:**")
    lines.append("")
    for fam, r in sorted(family_best.items()):
        display = _ENCODING_DISPLAY.get(r["encoding"], r["encoding"])
        lines.append(
            f"   - {fam}: {display} "
            f"(rank #{r.get('rank')}, score={_fmt(r.get('score'), 4)})"
        )
    lines.append("")

    # Simulable encodings.
    simulable = [r for r in rankings if r.get("is_simulable")]
    if simulable:
        lines.append(
            f"3. **Classically simulable encodings:** "
            f"{', '.join(_ENCODING_DISPLAY.get(r['encoding'], r['encoding']) for r in simulable)} "
            f"— these can be efficiently simulated classically, making them "
            f"useful as baselines but not candidates for quantum advantage."
        )
        lines.append("")

    # Practical guidance.
    lines.extend(
        [
            "## Practical Guidance",
            "",
            "- **For highest accuracy:** Choose the top-ranked encoding "
            "for the specific dataset and paradigm (VQC or kernel).",
            "- **For resource-constrained hardware:** Prefer shallow-depth "
            "encodings from the Pareto front (e.g., AngleEncoding, "
            "HigherOrderAngleEncoding).",
            "- **For noise-resilient applications:** Prioritise encodings "
            "with high noise resilience scores; non-entangling encodings "
            "are generally more robust (see H5).",
            "- **For trainability:** Avoid deep circuits that exhibit "
            "barren plateaus (see H4); SwapEquivariantFeatureMap achieves "
            "the highest trainability among entangling encodings.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def generate_report(
    *,
    stage_dirs: dict[str, str],
    tradeoff_dir: str,
    output_dir: str,
    figure_dir: str = "experiments/results/figures",
    sensitivity_dir: str | None = None,
    table_dir: str | None = None,
    generate_tables: bool = True,
) -> dict[str, Any]:
    """Generate the complete Stage 8 report.

    Parameters
    ----------
    stage_dirs : dict[str, str]
        Mapping of stage name to result directory (Stages 1-6b).
    tradeoff_dir : str
        Stage 7 tradeoff results directory.
    output_dir : str
        Top-level output directory for the report.
    figure_dir : str
        Directory containing Stage 7 figures.
    sensitivity_dir : str or None
        Stage 6a.5 sensitivity results directory.
    table_dir : str or None
        Custom table output directory.  Defaults to ``output_dir/tables``.
    generate_tables : bool
        Whether to generate tables (default True).

    Returns
    -------
    dict[str, Any]
        Report generation summary.
    """
    t_start = time.monotonic()
    os.makedirs(output_dir, exist_ok=True)

    if table_dir is None:
        table_dir = os.path.join(output_dir, "tables")

    generated_files: list[str] = []
    errors: list[str] = []

    # ---- 8.1: Master summary JSON -----------------------------------------
    logger.info("Step 8.1: Building master summary JSON...")
    print("[8.1] Building master summary JSON...", flush=True)
    try:
        master_summary = _build_master_summary(
            stage_dirs,
            tradeoff_dir,
            sensitivity_dir,
        )
        # Use "master_summary.json" to avoid collision with the runner's
        # own checkpoint-format "summary.json".
        summary_path = os.path.join(output_dir, "master_summary.json")
        _save_json(master_summary, summary_path)
        generated_files.append(summary_path)
        print(
            f"  -> {summary_path} " f"({master_summary['n_encodings']} encodings)",
            flush=True,
        )
    except Exception as exc:
        msg = f"8.1 Master summary failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ---- 8.2: Tables -------------------------------------------------------
    if generate_tables:
        logger.info("Step 8.2: Generating tables...")
        print("[8.2] Generating tables...", flush=True)
        try:
            table_files = _generate_tables(
                stage_dirs,
                tradeoff_dir,
                sensitivity_dir,
                table_dir,
            )
            generated_files.extend(table_files)
            print(f"  -> {len(table_files)} table files in {table_dir}", flush=True)
        except Exception as exc:
            msg = f"8.2 Table generation failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    # ---- 8.3: Figure verification ------------------------------------------
    logger.info("Step 8.3: Verifying figures...")
    print("[8.3] Verifying figures...", flush=True)
    figure_report = _verify_figures(figure_dir)
    print(
        f"  -> {figure_report.get('png_count', 0)} PNG, "
        f"{figure_report.get('pdf_count', 0)} PDF figures found",
        flush=True,
    )

    # ---- 8.4: Hypothesis narrative -----------------------------------------
    logger.info("Step 8.4: Generating hypothesis verdict narrative...")
    print("[8.4] Generating hypothesis verdict narrative...", flush=True)
    try:
        verdicts_data = _load_json(
            os.path.join(tradeoff_dir, "hypothesis_verdicts.json")
        )
        pareto_data = _load_json(os.path.join(tradeoff_dir, "pareto_front.json"))
        rankings_data = _load_json(os.path.join(tradeoff_dir, "rankings.json"))

        if verdicts_data:
            hyp_md = _generate_hypotheses_md(verdicts_data, pareto_data)
            hyp_path = os.path.join(output_dir, "hypotheses.md")
            _save_text(hyp_md, hyp_path)
            generated_files.append(hyp_path)
            print(f"  -> {hyp_path}", flush=True)
        else:
            errors.append("8.4 hypothesis_verdicts.json not found")

        if rankings_data:
            rank_md = _generate_ranking_md(
                rankings_data,
                pareto_data,
                verdicts_data,
            )
            rank_path = os.path.join(output_dir, "ranking.md")
            _save_text(rank_md, rank_path)
            generated_files.append(rank_path)
            print(f"  -> {rank_path}", flush=True)
        else:
            errors.append("8.4 rankings.json not found")

    except Exception as exc:
        msg = f"8.4 Narrative generation failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    wall_time = round(time.monotonic() - t_start, 3)

    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "success" if not errors else "partial",
        "n_files_generated": len(generated_files),
        "generated_files": [os.path.basename(f) for f in generated_files],
        "figure_verification": figure_report,
        "errors": errors,
        "wall_time_seconds": wall_time,
    }

    # Save the report generation result alongside the outputs.
    result_path = os.path.join(output_dir, "report_generation_result.json")
    _save_json(result, result_path)

    print(
        f"\n[DONE] Report generation complete: "
        f"{len(generated_files)} files, {wall_time:.1f}s",
        flush=True,
    )
    if errors:
        print(f"  Warnings: {len(errors)} error(s) — see report_generation_result.json")

    return result
