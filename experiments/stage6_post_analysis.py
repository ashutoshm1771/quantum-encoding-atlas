"""Stage 6 post-analysis: statistical tests, comparisons, and report generation.

This module performs rigorous statistical post-analysis on the raw results
from Stage 6a (VQC classification) and Stage 6b (quantum kernel classification).
It implements plan items 6a.6-6a.8 and 6b.5-6b.7.

Analyses Performed
------------------
1. **Per-encoding aggregation**: Collects per-fold accuracies across all runs
   and datasets, computes mean/std/CI for each encoding-dataset pair.

2. **Pairwise statistical tests** (6a.6, 6b.6): Wilcoxon signed-rank tests
   between all encoding pairs on each dataset, with Holm-Bonferroni correction
   for family-wise error control.

3. **Effect sizes** (6a.7): Cliff's delta for every significant pair, with
   Vargha-Delaney interpretation (negligible/small/medium/large).

4. **Bootstrap confidence intervals** (6a.7): 95% bootstrap CI for each
   encoding's mean accuracy on each dataset.

5. **VQC vs kernel comparison** (6b.7): Paired accuracy differences per
   encoding per dataset, with Wilcoxon tests and Cliff's delta.

6. **Quantum vs classical comparison**: Per-dataset comparison between every
   quantum encoding and every classical baseline (SVM-RBF, Random Forest, MLP).

7. **Summary report**: Structured JSON output consumed by Stage 7
   (:mod:`experiments.tradeoff`) and human-readable Markdown tables.

Usage
-----
As a CLI tool::

    python -m experiments.stage6_post_analysis \\
        --vqc-dir experiments/results/raw/stage6a_vqc \\
        --kernel-dir experiments/results/raw/stage6b_kernel \\
        --output-dir experiments/results/post_analysis/stage6

Programmatically::

    from experiments.stage6_post_analysis import run_stage6_post_analysis

    report = run_stage6_post_analysis(
        vqc_dir="experiments/results/raw/stage6a_vqc",
        kernel_dir="experiments/results/raw/stage6b_kernel",
        output_dir="experiments/results/post_analysis/stage6",
    )

Dependencies
------------
This module reuses the statistical primitives from :mod:`experiments.statistical`
and the data-loading utilities from :mod:`experiments.tradeoff`. It does not
import heavy simulation libraries (PennyLane, Qiskit, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.statistical import (
    bootstrap_ci,
    cliffs_delta,
    compute_ci_metrics,
    holm_bonferroni_correction,
    interpret_cliffs_delta,
    kendall_tau,
    pairwise_comparisons,
    wilcoxon_test,
)
from experiments.tradeoff import (
    ENTANGLING_ENCODINGS,
    EQUIVARIANT_ENCODINGS,
    REGISTRY_TO_CLASS,
    canonicalize_encoding_name,
    extract_classical_baselines,
    get_encoding_family,
    load_stage_results,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum number of paired fold-level observations required to run a
#: Wilcoxon signed-rank test.  With fewer observations the test has no
#: statistical power and may raise errors.
_MIN_PAIRED_SAMPLES: int = 5

#: Significance level for hypothesis tests (before Holm-Bonferroni).
_ALPHA: float = 0.05

#: Schema version for the JSON output to enable forward-compatible parsing.
_SCHEMA_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodingDatasetProfile:
    """Aggregated metrics for one encoding on one dataset.

    This is the fundamental unit of analysis: a single encoding evaluated
    on a single dataset via a specific paradigm (VQC or kernel).

    Attributes
    ----------
    encoding_name : str
        Registry name of the encoding (snake_case).
    dataset : str
        Name of the dataset.
    paradigm : str
        Either ``"vqc"`` or ``"kernel"``.
    fold_accuracies : tuple[float, ...]
        Per-fold test accuracies across all runs.  For 10 runs × 5 folds
        this is a length-50 tuple.
    mean_accuracy : float
        Mean of *fold_accuracies*.
    std_accuracy : float
        Sample standard deviation (ddof=1) of *fold_accuracies*.
    ci_lower : float
        Lower bound of the 95% bootstrap confidence interval.
    ci_upper : float
        Upper bound of the 95% bootstrap confidence interval.
    n_observations : int
        Length of *fold_accuracies*.
    """

    encoding_name: str
    dataset: str
    paradigm: str
    fold_accuracies: tuple[float, ...]
    mean_accuracy: float
    std_accuracy: float
    ci_lower: float
    ci_upper: float
    n_observations: int


@dataclass(frozen=True)
class PairwiseResult:
    """Result of a pairwise comparison between two methods.

    Attributes
    ----------
    method_a : str
        First method (encoding or baseline name).
    method_b : str
        Second method.
    dataset : str
        Dataset on which the comparison was performed.
    p_value : float
        Raw Wilcoxon signed-rank p-value.
    significant : bool
        Whether the pair is significant after Holm-Bonferroni correction.
    cliffs_delta : float
        Cliff's delta effect size.  Positive means *method_a* > *method_b*.
    effect_size : str
        Interpretation: ``"negligible"``, ``"small"``, ``"medium"``, or ``"large"``.
    """

    method_a: str
    method_b: str
    dataset: str
    p_value: float
    significant: bool
    cliffs_delta: float
    effect_size: str


@dataclass
class Stage6Report:
    """Complete Stage 6 post-analysis report.

    Serializable to JSON via :meth:`to_dict`.  This is the primary output
    of :func:`run_stage6_post_analysis`.

    Attributes
    ----------
    schema_version : str
        Output schema version for forward compatibility.
    vqc_profiles : dict
        ``{encoding_name: {dataset: EncodingDatasetProfile}}`` for VQC.
    kernel_profiles : dict
        ``{encoding_name: {dataset: EncodingDatasetProfile}}`` for kernel.
    vqc_pairwise : dict
        ``{dataset: list[PairwiseResult]}`` from VQC pairwise tests.
    kernel_pairwise : dict
        ``{dataset: list[PairwiseResult]}`` from kernel pairwise tests.
    vqc_vs_kernel : dict
        Per-encoding VQC-vs-kernel accuracy comparison.
    quantum_vs_classical : dict
        Per-dataset quantum-vs-classical baseline comparison.
    ranking_summary : dict
        Mean accuracy rankings for VQC and kernel paradigms.
    """

    schema_version: str = _SCHEMA_VERSION
    vqc_profiles: dict[str, dict[str, EncodingDatasetProfile]] = field(
        default_factory=dict,
    )
    kernel_profiles: dict[str, dict[str, EncodingDatasetProfile]] = field(
        default_factory=dict,
    )
    vqc_pairwise: dict[str, list[PairwiseResult]] = field(default_factory=dict)
    kernel_pairwise: dict[str, list[PairwiseResult]] = field(default_factory=dict)
    vqc_vs_kernel: dict[str, Any] = field(default_factory=dict)
    quantum_vs_classical: dict[str, Any] = field(default_factory=dict)
    ranking_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "vqc_profiles": _profiles_to_dict(self.vqc_profiles),
            "kernel_profiles": _profiles_to_dict(self.kernel_profiles),
            "vqc_pairwise": _pairwise_to_dict(self.vqc_pairwise),
            "kernel_pairwise": _pairwise_to_dict(self.kernel_pairwise),
            "vqc_vs_kernel": self.vqc_vs_kernel,
            "quantum_vs_classical": self.quantum_vs_classical,
            "ranking_summary": self.ranking_summary,
        }


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def aggregate_profiles(
    results: list[dict[str, Any]],
    paradigm: str,
    seed: int = 42,
) -> dict[str, dict[str, EncodingDatasetProfile]]:
    """Aggregate raw stage results into per-encoding, per-dataset profiles.

    Reads the summary JSON results and extracts per-fold test accuracies
    for every encoding-dataset combination.  Classical baselines (whose
    encoding names start with ``"classical_"``) are excluded.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Successful result entries from :func:`load_stage_results`.
    paradigm : str
        ``"vqc"`` or ``"kernel"``.
    seed : int
        Seed passed to :func:`~experiments.statistical.bootstrap_ci`.

    Returns
    -------
    dict[str, dict[str, EncodingDatasetProfile]]
        Nested mapping ``{encoding_name: {dataset: profile}}``.
    """
    profiles: dict[str, dict[str, EncodingDatasetProfile]] = {}

    for entry in results:
        raw_name = entry.get("encoding_name", "")
        if raw_name.startswith("classical_"):
            continue

        reg_name = canonicalize_encoding_name(raw_name)
        result_data = entry.get("result", {})
        datasets_data = result_data.get("datasets", {})

        for ds_name, ds_result in datasets_data.items():
            if ds_result.get("status") != "success":
                continue

            # Skip if we already have this encoding-dataset pair from a
            # config with more features (keeps first successful config).
            if reg_name in profiles and ds_name in profiles[reg_name]:
                continue

            fold_accs = _extract_fold_accuracies(ds_result)
            if not fold_accs:
                continue

            arr = np.array(fold_accs)
            ci_metrics = compute_ci_metrics(arr, confidence=0.95, seed=seed)

            profile = EncodingDatasetProfile(
                encoding_name=reg_name,
                dataset=ds_name,
                paradigm=paradigm,
                fold_accuracies=tuple(fold_accs),
                mean_accuracy=ci_metrics["mean"],
                std_accuracy=ci_metrics["std"],
                ci_lower=ci_metrics["ci_lower"],
                ci_upper=ci_metrics["ci_upper"],
                n_observations=len(fold_accs),
            )

            profiles.setdefault(reg_name, {})[ds_name] = profile

    n_total = sum(len(ds) for ds in profiles.values())
    logger.info(
        "Aggregated %d %s profiles across %d encodings",
        n_total, paradigm, len(profiles),
    )
    return profiles


def run_pairwise_tests(
    profiles: dict[str, dict[str, EncodingDatasetProfile]],
    alpha: float = _ALPHA,
) -> dict[str, list[PairwiseResult]]:
    """Run pairwise statistical tests for all encoding pairs per dataset.

    For each dataset, performs Wilcoxon signed-rank tests on all
    ``C(n, 2)`` encoding pairs, applies Holm-Bonferroni correction, and
    computes Cliff's delta effect sizes.

    Only encoding pairs that share the same dataset and have at least
    :data:`_MIN_PAIRED_SAMPLES` observations each are compared.  The
    comparison uses the minimum-length overlap of fold accuracies (all
    encodings use the same CV folds, so this is typically all folds).

    Parameters
    ----------
    profiles : dict[str, dict[str, EncodingDatasetProfile]]
        Output of :func:`aggregate_profiles`.
    alpha : float
        Family-wise significance level (default 0.05).

    Returns
    -------
    dict[str, list[PairwiseResult]]
        ``{dataset_name: [PairwiseResult, ...]}``.
    """
    # Collect all datasets that appear in the profiles.
    datasets: set[str] = set()
    for enc_profiles in profiles.values():
        datasets.update(enc_profiles.keys())

    results: dict[str, list[PairwiseResult]] = {}

    for ds_name in sorted(datasets):
        # Gather encoding fold-accuracy arrays for this dataset.
        scores: dict[str, NDArray[np.floating]] = {}
        for enc_name, enc_profiles in profiles.items():
            if ds_name not in enc_profiles:
                continue
            profile = enc_profiles[ds_name]
            if profile.n_observations >= _MIN_PAIRED_SAMPLES:
                scores[enc_name] = np.array(profile.fold_accuracies)

        if len(scores) < 2:
            logger.debug(
                "Dataset '%s': fewer than 2 encodings with sufficient data; "
                "skipping pairwise tests.",
                ds_name,
            )
            continue

        pw = pairwise_comparisons(scores, alpha=alpha)

        ds_results: list[PairwiseResult] = []
        for pair in pw["pairs"]:
            ds_results.append(PairwiseResult(
                method_a=pair["method_a"],
                method_b=pair["method_b"],
                dataset=ds_name,
                p_value=pair["p_value"],
                significant=pair["significant"],
                cliffs_delta=pair["cliffs_delta"],
                effect_size=pair["effect_size"],
            ))

        results[ds_name] = ds_results
        n_sig = sum(1 for r in ds_results if r.significant)
        logger.info(
            "Dataset '%s': %d/%d pairs significant after Holm-Bonferroni",
            ds_name, n_sig, len(ds_results),
        )

    return results


def compare_vqc_vs_kernel(
    vqc_profiles: dict[str, dict[str, EncodingDatasetProfile]],
    kernel_profiles: dict[str, dict[str, EncodingDatasetProfile]],
    alpha: float = _ALPHA,
) -> dict[str, Any]:
    """Compare VQC and kernel accuracy for each encoding per dataset.

    For each encoding-dataset pair where both VQC and kernel data exist,
    performs a Wilcoxon signed-rank test on the paired fold accuracies and
    computes Cliff's delta.  Also computes per-dataset ranking agreement
    via Kendall's tau.

    Parameters
    ----------
    vqc_profiles, kernel_profiles : dict
        Output of :func:`aggregate_profiles` for each paradigm.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict[str, Any]
        Report with keys:

        - ``"per_encoding"``: per-encoding comparison results.
        - ``"per_dataset"``: per-dataset ranking agreement (Kendall's tau).
        - ``"summary"``: aggregate counts.
    """
    per_encoding: dict[str, dict[str, Any]] = {}
    datasets_with_data: set[str] = set()

    all_encodings = sorted(
        set(vqc_profiles.keys()) & set(kernel_profiles.keys()),
    )

    for enc_name in all_encodings:
        enc_results: dict[str, Any] = {}

        for ds_name in sorted(vqc_profiles.get(enc_name, {})):
            if ds_name not in kernel_profiles.get(enc_name, {}):
                continue

            vqc_prof = vqc_profiles[enc_name][ds_name]
            ker_prof = kernel_profiles[enc_name][ds_name]

            # Use minimum-length alignment for paired test.
            n = min(vqc_prof.n_observations, ker_prof.n_observations)
            if n < _MIN_PAIRED_SAMPLES:
                continue

            vqc_folds = np.array(vqc_prof.fold_accuracies[:n])
            ker_folds = np.array(ker_prof.fold_accuracies[:n])

            _, p_val = wilcoxon_test(vqc_folds, ker_folds)
            delta = cliffs_delta(vqc_folds, ker_folds)

            enc_results[ds_name] = {
                "vqc_mean": vqc_prof.mean_accuracy,
                "kernel_mean": ker_prof.mean_accuracy,
                "difference": vqc_prof.mean_accuracy - ker_prof.mean_accuracy,
                "p_value": p_val,
                "significant": p_val < alpha,
                "cliffs_delta": delta,
                "effect_size": interpret_cliffs_delta(delta),
                "better_paradigm": (
                    "vqc" if delta > 0 else "kernel" if delta < 0 else "tie"
                ),
            }
            datasets_with_data.add(ds_name)

        if enc_results:
            per_encoding[enc_name] = enc_results

    # Per-dataset ranking agreement (Kendall's tau).
    per_dataset: dict[str, Any] = {}

    for ds_name in sorted(datasets_with_data):
        vqc_means: dict[str, float] = {}
        ker_means: dict[str, float] = {}

        for enc_name in all_encodings:
            vqc_ds = vqc_profiles.get(enc_name, {}).get(ds_name)
            ker_ds = kernel_profiles.get(enc_name, {}).get(ds_name)
            if vqc_ds is not None and ker_ds is not None:
                vqc_means[enc_name] = vqc_ds.mean_accuracy
                ker_means[enc_name] = ker_ds.mean_accuracy

        if len(vqc_means) < 3:
            per_dataset[ds_name] = {
                "status": "insufficient_data",
                "n_encodings": len(vqc_means),
            }
            continue

        # Rank by accuracy (descending), then compute Kendall's tau.
        enc_order = sorted(vqc_means.keys())
        vqc_ranks = _rank_values(
            [vqc_means[e] for e in enc_order], descending=True,
        )
        ker_ranks = _rank_values(
            [ker_means[e] for e in enc_order], descending=True,
        )

        tau, tau_p = kendall_tau(
            np.array(vqc_ranks, dtype=float),
            np.array(ker_ranks, dtype=float),
        )

        per_dataset[ds_name] = {
            "tau": tau,
            "p_value": tau_p,
            "significant": tau_p < alpha,
            "n_encodings": len(enc_order),
            "vqc_ranking": [
                enc_order[i]
                for i in np.argsort(vqc_ranks)
            ],
            "kernel_ranking": [
                enc_order[i]
                for i in np.argsort(ker_ranks)
            ],
        }

    # Summary counts.
    n_vqc_better = 0
    n_kernel_better = 0
    n_tie = 0
    for enc_data in per_encoding.values():
        for ds_data in enc_data.values():
            bp = ds_data.get("better_paradigm", "tie")
            if bp == "vqc":
                n_vqc_better += 1
            elif bp == "kernel":
                n_kernel_better += 1
            else:
                n_tie += 1

    return {
        "per_encoding": per_encoding,
        "per_dataset": per_dataset,
        "summary": {
            "n_comparisons": n_vqc_better + n_kernel_better + n_tie,
            "n_vqc_better": n_vqc_better,
            "n_kernel_better": n_kernel_better,
            "n_tie": n_tie,
        },
    }


def compare_quantum_vs_classical(
    vqc_profiles: dict[str, dict[str, EncodingDatasetProfile]],
    baselines: dict[str, dict[str, Any]],
    alpha: float = _ALPHA,
) -> dict[str, Any]:
    """Compare quantum VQC encodings against classical baselines.

    For each dataset, finds the best quantum encoding and compares it
    against each classical baseline.  Also performs pairwise Wilcoxon
    tests between every quantum encoding and every baseline.

    Parameters
    ----------
    vqc_profiles : dict
        VQC profiles from :func:`aggregate_profiles`.
    baselines : dict
        Classical baseline data from :func:`extract_classical_baselines`.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict[str, Any]
        Report with per-dataset and overall comparison results.
    """
    # Collect all datasets present in VQC profiles.
    all_datasets: set[str] = set()
    for enc_ds in vqc_profiles.values():
        all_datasets.update(enc_ds.keys())

    per_dataset: dict[str, Any] = {}

    # Global counters.
    total_comparisons = 0
    n_quantum_wins = 0
    n_classical_wins = 0
    n_ties = 0

    for ds_name in sorted(all_datasets):
        # Best quantum encoding for this dataset.
        best_enc_name: str | None = None
        best_enc_acc: float = -1.0

        for enc_name, enc_ds in vqc_profiles.items():
            prof = enc_ds.get(ds_name)
            if prof is not None and prof.mean_accuracy > best_enc_acc:
                best_enc_acc = prof.mean_accuracy
                best_enc_name = enc_name

        # Comparison with each baseline.
        baseline_results: dict[str, Any] = {}

        for bl_name, bl_data in baselines.items():
            bl_acc_dict = bl_data.get("accuracy", {})
            bl_folds_dict = bl_data.get("folds", {})

            bl_mean = bl_acc_dict.get(ds_name)
            bl_folds = bl_folds_dict.get(ds_name, [])

            if bl_mean is None:
                continue

            comparison: dict[str, Any] = {
                "baseline_mean": bl_mean,
                "best_quantum_mean": best_enc_acc,
                "best_quantum_encoding": best_enc_name,
                "difference": best_enc_acc - bl_mean,
            }

            # Paired test (if best quantum has folds and baseline has folds).
            if best_enc_name is not None and bl_folds:
                best_prof = vqc_profiles[best_enc_name].get(ds_name)
                if best_prof is not None:
                    n = min(best_prof.n_observations, len(bl_folds))
                    if n >= _MIN_PAIRED_SAMPLES:
                        q_arr = np.array(best_prof.fold_accuracies[:n])
                        bl_arr = np.array(bl_folds[:n])
                        _, p_val = wilcoxon_test(q_arr, bl_arr)
                        delta = cliffs_delta(q_arr, bl_arr)
                        comparison["p_value"] = p_val
                        comparison["significant"] = p_val < alpha
                        comparison["cliffs_delta"] = delta
                        comparison["effect_size"] = interpret_cliffs_delta(delta)

            winner = (
                "quantum" if comparison["difference"] > 0
                else "classical" if comparison["difference"] < 0
                else "tie"
            )
            comparison["winner"] = winner

            total_comparisons += 1
            if winner == "quantum":
                n_quantum_wins += 1
            elif winner == "classical":
                n_classical_wins += 1
            else:
                n_ties += 1

            baseline_results[bl_name] = comparison

        per_dataset[ds_name] = {
            "best_quantum_encoding": best_enc_name,
            "best_quantum_accuracy": best_enc_acc,
            "baselines": baseline_results,
        }

    return {
        "per_dataset": per_dataset,
        "summary": {
            "n_comparisons": total_comparisons,
            "n_quantum_wins": n_quantum_wins,
            "n_classical_wins": n_classical_wins,
            "n_ties": n_ties,
            "quantum_win_rate": (
                n_quantum_wins / total_comparisons
                if total_comparisons > 0 else 0.0
            ),
        },
    }


def compute_ranking_summary(
    vqc_profiles: dict[str, dict[str, EncodingDatasetProfile]],
    kernel_profiles: dict[str, dict[str, EncodingDatasetProfile]],
) -> dict[str, Any]:
    """Compute overall encoding rankings for VQC and kernel paradigms.

    Ranks encodings by their mean accuracy across all compatible datasets.

    Parameters
    ----------
    vqc_profiles, kernel_profiles : dict
        Profiles from :func:`aggregate_profiles`.

    Returns
    -------
    dict[str, Any]
        Ranking tables with per-encoding mean accuracy and rank.
    """
    summary: dict[str, Any] = {}

    for paradigm, profiles in [("vqc", vqc_profiles), ("kernel", kernel_profiles)]:
        enc_mean_accs: dict[str, float] = {}

        for enc_name, enc_ds in profiles.items():
            accs = [p.mean_accuracy for p in enc_ds.values()]
            if accs:
                enc_mean_accs[enc_name] = float(np.mean(accs))

        if not enc_mean_accs:
            summary[paradigm] = {"rankings": [], "status": "no_data"}
            continue

        # Sort descending by mean accuracy.
        sorted_encs = sorted(
            enc_mean_accs.items(), key=lambda x: x[1], reverse=True,
        )

        rankings: list[dict[str, Any]] = []
        for rank, (enc_name, mean_acc) in enumerate(sorted_encs, start=1):
            rankings.append({
                "rank": rank,
                "encoding": enc_name,
                "family": get_encoding_family(enc_name),
                "is_entangling": enc_name in ENTANGLING_ENCODINGS,
                "is_equivariant": enc_name in EQUIVARIANT_ENCODINGS,
                "mean_accuracy_across_datasets": round(mean_acc, 6),
                "n_datasets": len(profiles.get(enc_name, {})),
            })

        summary[paradigm] = {
            "rankings": rankings,
            "n_encodings": len(rankings),
        }

    return summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown_summary(report: Stage6Report) -> str:
    """Generate a human-readable Markdown summary of the analysis.

    Parameters
    ----------
    report : Stage6Report
        Complete post-analysis report.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    lines: list[str] = []
    lines.append("# Stage 6 Post-Analysis Report")
    lines.append("")

    # --- VQC Rankings ---
    lines.append("## VQC Encoding Rankings")
    lines.append("")
    vqc_ranking = report.ranking_summary.get("vqc", {}).get("rankings", [])
    if vqc_ranking:
        lines.append(
            "| Rank | Encoding | Family | Entangling | "
            "Mean Accuracy | Datasets |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in vqc_ranking:
            lines.append(
                f"| {row['rank']} "
                f"| {row['encoding']} "
                f"| {row['family']} "
                f"| {'Yes' if row['is_entangling'] else 'No'} "
                f"| {row['mean_accuracy_across_datasets']:.4f} "
                f"| {row['n_datasets']} |"
            )
    else:
        lines.append("*No VQC data available.*")
    lines.append("")

    # --- Kernel Rankings ---
    lines.append("## Kernel Encoding Rankings")
    lines.append("")
    ker_ranking = report.ranking_summary.get("kernel", {}).get("rankings", [])
    if ker_ranking:
        lines.append(
            "| Rank | Encoding | Family | Entangling | "
            "Mean Accuracy | Datasets |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in ker_ranking:
            lines.append(
                f"| {row['rank']} "
                f"| {row['encoding']} "
                f"| {row['family']} "
                f"| {'Yes' if row['is_entangling'] else 'No'} "
                f"| {row['mean_accuracy_across_datasets']:.4f} "
                f"| {row['n_datasets']} |"
            )
    else:
        lines.append("*No kernel data available.*")
    lines.append("")

    # --- VQC vs Kernel ---
    lines.append("## VQC vs Kernel Comparison")
    lines.append("")
    vk_summary = report.vqc_vs_kernel.get("summary", {})
    if vk_summary:
        lines.append(
            f"- Total comparisons: {vk_summary.get('n_comparisons', 0)}"
        )
        lines.append(f"- VQC better: {vk_summary.get('n_vqc_better', 0)}")
        lines.append(f"- Kernel better: {vk_summary.get('n_kernel_better', 0)}")
        lines.append(f"- Ties: {vk_summary.get('n_tie', 0)}")
    lines.append("")

    # --- Quantum vs Classical ---
    lines.append("## Quantum vs Classical Baselines")
    lines.append("")
    qc_summary = report.quantum_vs_classical.get("summary", {})
    if qc_summary:
        lines.append(
            f"- Total comparisons: {qc_summary.get('n_comparisons', 0)}"
        )
        lines.append(
            f"- Quantum wins: {qc_summary.get('n_quantum_wins', 0)}"
        )
        lines.append(
            f"- Classical wins: {qc_summary.get('n_classical_wins', 0)}"
        )
        win_rate = qc_summary.get("quantum_win_rate", 0)
        lines.append(f"- Quantum win rate: {win_rate:.1%}")
    lines.append("")

    # --- Pairwise Summary ---
    lines.append("## Pairwise Statistical Tests (VQC)")
    lines.append("")
    for ds_name, pairs in sorted(report.vqc_pairwise.items()):
        n_sig = sum(1 for p in pairs if p.significant)
        n_total = len(pairs)
        lines.append(f"- **{ds_name}**: {n_sig}/{n_total} significant pairs")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def run_stage6_post_analysis(
    vqc_dir: str,
    kernel_dir: str,
    output_dir: str,
    *,
    alpha: float = _ALPHA,
    seed: int = 42,
) -> Stage6Report:
    """Execute the complete Stage 6 post-analysis pipeline.

    This is the primary entry point.  It loads raw results from Stage 6a
    and 6b, runs all statistical analyses, and saves the report as JSON
    and Markdown.

    Parameters
    ----------
    vqc_dir : str
        Path to Stage 6a (VQC) results directory containing ``summary.json``.
    kernel_dir : str
        Path to Stage 6b (kernel) results directory containing ``summary.json``.
    output_dir : str
        Directory for writing analysis outputs.
    alpha : float
        Significance level for hypothesis tests (default 0.05).
    seed : int
        Random seed for bootstrap CI computation (default 42).

    Returns
    -------
    Stage6Report
        Complete analysis report.

    Raises
    ------
    FileNotFoundError
        If neither VQC nor kernel results can be loaded.
    """
    logger.info("Starting Stage 6 post-analysis")
    logger.info("  VQC dir:    %s", vqc_dir)
    logger.info("  Kernel dir: %s", kernel_dir)
    logger.info("  Output dir: %s", output_dir)

    # Load raw results.
    vqc_results = load_stage_results(vqc_dir)
    kernel_results = load_stage_results(kernel_dir)

    if not vqc_results and not kernel_results:
        raise FileNotFoundError(
            f"No results found in either {vqc_dir} or {kernel_dir}. "
            "Ensure Stage 6a and/or 6b have completed successfully."
        )

    report = Stage6Report()

    # 1. Aggregate profiles.
    logger.info("Aggregating VQC profiles...")
    report.vqc_profiles = aggregate_profiles(
        vqc_results, paradigm="vqc", seed=seed,
    )

    logger.info("Aggregating kernel profiles...")
    report.kernel_profiles = aggregate_profiles(
        kernel_results, paradigm="kernel", seed=seed,
    )

    # 2. Pairwise statistical tests (6a.6, 6b.6).
    logger.info("Running VQC pairwise tests...")
    report.vqc_pairwise = run_pairwise_tests(
        report.vqc_profiles, alpha=alpha,
    )

    logger.info("Running kernel pairwise tests...")
    report.kernel_pairwise = run_pairwise_tests(
        report.kernel_profiles, alpha=alpha,
    )

    # 3. VQC vs kernel comparison (6b.7).
    logger.info("Comparing VQC vs kernel...")
    report.vqc_vs_kernel = compare_vqc_vs_kernel(
        report.vqc_profiles, report.kernel_profiles, alpha=alpha,
    )

    # 4. Quantum vs classical comparison.
    logger.info("Comparing quantum vs classical baselines...")
    baselines = extract_classical_baselines(vqc_dir)
    report.quantum_vs_classical = compare_quantum_vs_classical(
        report.vqc_profiles, baselines, alpha=alpha,
    )

    # 5. Ranking summary.
    logger.info("Computing ranking summary...")
    report.ranking_summary = compute_ranking_summary(
        report.vqc_profiles, report.kernel_profiles,
    )

    # Save outputs.
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "stage6_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=_json_default)
    logger.info("JSON report saved: %s", json_path)

    md_path = os.path.join(output_dir, "stage6_report.md")
    md_content = generate_markdown_summary(report)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)
    logger.info("Markdown report saved: %s", md_path)

    logger.info("Stage 6 post-analysis complete.")
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Stage 6 post-analysis.

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
    import sys

    parser = argparse.ArgumentParser(
        prog="stage6_post_analysis",
        description="Run Stage 6 post-analysis: statistical tests and comparisons.",
    )
    parser.add_argument(
        "--vqc-dir",
        default="experiments/results/raw/stage6a_vqc",
        help="Path to Stage 6a VQC results directory.",
    )
    parser.add_argument(
        "--kernel-dir",
        default="experiments/results/raw/stage6b_kernel",
        help="Path to Stage 6b kernel results directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/post_analysis/stage6",
        help="Output directory for analysis report.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=_ALPHA,
        help="Significance level (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap CI (default: 42).",
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
        report = run_stage6_post_analysis(
            vqc_dir=args.vqc_dir,
            kernel_dir=args.kernel_dir,
            output_dir=args.output_dir,
            alpha=args.alpha,
            seed=args.seed,
        )

        # Print a short summary to stdout.
        vqc_n = report.ranking_summary.get("vqc", {}).get("n_encodings", 0)
        ker_n = report.ranking_summary.get("kernel", {}).get("n_encodings", 0)
        print(f"\nStage 6 post-analysis complete.")
        print(f"  VQC encodings ranked:    {vqc_n}")
        print(f"  Kernel encodings ranked: {ker_n}")

        vqc_pw_sig = sum(
            sum(1 for p in pairs if p.significant)
            for pairs in report.vqc_pairwise.values()
        )
        vqc_pw_total = sum(
            len(pairs) for pairs in report.vqc_pairwise.values()
        )
        print(f"  VQC pairwise significant: {vqc_pw_sig}/{vqc_pw_total}")

        vk = report.vqc_vs_kernel.get("summary", {})
        print(
            f"  VQC vs kernel: "
            f"{vk.get('n_vqc_better', 0)} VQC wins, "
            f"{vk.get('n_kernel_better', 0)} kernel wins"
        )

        qc = report.quantum_vs_classical.get("summary", {})
        print(
            f"  Quantum vs classical: "
            f"{qc.get('quantum_win_rate', 0):.1%} quantum win rate"
        )

        print(f"\nOutputs saved to: {args.output_dir}/")
        return 0

    except FileNotFoundError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2

    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        return 2


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_fold_accuracies(dataset_result: dict[str, Any]) -> list[float]:
    """Extract a flat list of per-fold test accuracies from a dataset result.

    Iterates over all runs and folds, collecting ``test_accuracy`` values
    from non-failed folds.

    Parameters
    ----------
    dataset_result : dict[str, Any]
        A single dataset entry from a VQC or kernel result dict.

    Returns
    -------
    list[float]
        Per-fold test accuracies.  Empty if no valid folds found.
    """
    accuracies: list[float] = []
    for run in dataset_result.get("runs", []):
        for fold in run.get("folds", []):
            acc = fold.get("test_accuracy")
            if acc is not None and fold.get("status") != "failed":
                accuracies.append(float(acc))
    return accuracies


def _rank_values(values: list[float], *, descending: bool = True) -> list[int]:
    """Assign integer ranks to a list of values.

    Parameters
    ----------
    values : list[float]
        Values to rank.
    descending : bool
        If ``True``, the largest value gets rank 1.

    Returns
    -------
    list[int]
        Ranks (1-indexed).  Ties are broken by position order.
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=descending)
    ranks = [0] * len(values)
    for rank, (original_idx, _) in enumerate(indexed, start=1):
        ranks[original_idx] = rank
    return ranks


def _profiles_to_dict(
    profiles: dict[str, dict[str, EncodingDatasetProfile]],
) -> dict[str, Any]:
    """Convert profile dataclasses to JSON-serializable dicts."""
    out: dict[str, Any] = {}
    for enc_name, enc_ds in profiles.items():
        enc_out: dict[str, Any] = {}
        for ds_name, profile in enc_ds.items():
            enc_out[ds_name] = {
                "encoding_name": profile.encoding_name,
                "dataset": profile.dataset,
                "paradigm": profile.paradigm,
                "mean_accuracy": profile.mean_accuracy,
                "std_accuracy": profile.std_accuracy,
                "ci_lower": profile.ci_lower,
                "ci_upper": profile.ci_upper,
                "n_observations": profile.n_observations,
                # Omit fold_accuracies from JSON to keep file size reasonable.
                # They can be reconstructed from the raw summary.json.
            }
        out[enc_name] = enc_out
    return out


def _pairwise_to_dict(
    pairwise: dict[str, list[PairwiseResult]],
) -> dict[str, Any]:
    """Convert pairwise result dataclasses to JSON-serializable dicts."""
    out: dict[str, Any] = {}
    for ds_name, pairs in pairwise.items():
        out[ds_name] = {
            "pairs": [
                {
                    "method_a": p.method_a,
                    "method_b": p.method_b,
                    "p_value": p.p_value,
                    "significant": p.significant,
                    "cliffs_delta": p.cliffs_delta,
                    "effect_size": p.effect_size,
                }
                for p in pairs
            ],
            "n_significant": sum(1 for p in pairs if p.significant),
            "n_comparisons": len(pairs),
        }
    return out


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for numpy types."""
    type_name = type(obj).__name__
    if "int" in type_name and hasattr(obj, "item"):
        return int(obj.item())
    if "float" in type_name and hasattr(obj, "item"):
        return float(obj.item())
    if "ndarray" in type_name and hasattr(obj, "tolist"):
        return obj.tolist()
    if "bool_" in type_name and hasattr(obj, "item"):
        return bool(obj.item())
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    import sys

    sys.exit(main())
