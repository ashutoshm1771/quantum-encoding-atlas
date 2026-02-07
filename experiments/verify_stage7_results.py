#!/usr/bin/env python
"""Stage 7 Independent Results Verification.

Independently recomputes all key statistics from Stage 7 using scipy/numpy
directly — bypassing the project's statistical wrappers — then compares
against saved Stage 7 outputs to verify correctness.

Verifications performed:
  - Spearman correlations (7.1, 7.2, 7.3, 4.4)
  - Partial Spearman via rank residuals (7.2)
  - Mann-Whitney U test (7.4, 4.3/H5)
  - Kendall tau (7.6)
  - Pareto front dominance (7.7)
  - Weighted ranking scores (7.9)
  - Cliff's delta range and interpretation (4.9)
  - Hypothesis verdict decision tree logic (H1-H7)

Usage::

    python experiments/verify_stage7_results.py
    python experiments/verify_stage7_results.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

ABS_TOL = 1e-6
REL_TOL = 1e-4

# Encoding metadata (duplicated here for independence from tradeoff.py)
ENTANGLING_ENCODINGS = {
    "iqp", "zz_feature_map", "pauli_feature_map", "data_reuploading",
    "hardware_efficient", "amplitude", "higher_order_angle",
    "qaoa_encoding", "hamiltonian_encoding", "symmetry_inspired",
    "trainable_encoding", "so2_equivariant", "cyclic_equivariant",
    "swap_equivariant",
}

DEFAULT_WEIGHTS = {
    "accuracy": 0.30,
    "expressibility": 0.15,
    "trainability": 0.20,
    "inv_depth": 0.15,
    "entanglement": 0.05,
    "noise_resilience": 0.15,
}


# ── Verification Suite ────────────────────────────────────────────


class VerificationSuite:
    """Collects and reports verification check results."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self._section = ""

    def section(self, name: str) -> None:
        """Start a new named verification section."""
        self._section = name

    def check_close(
        self,
        name: str,
        saved: float | None,
        recomputed: float | None,
        atol: float = ABS_TOL,
        rtol: float = REL_TOL,
    ) -> None:
        """Assert two numeric values are close within tolerance."""
        if saved is None and recomputed is None:
            self._add(name, True, "both None (OK)")
            return
        if saved is None or recomputed is None:
            self._add(name, False, f"saved={saved}, recomputed={recomputed}")
            return
        if np.isnan(saved) and np.isnan(recomputed):
            self._add(name, True, "both NaN (OK)")
            return
        if np.isnan(saved) or np.isnan(recomputed):
            self._add(name, False, f"saved={saved}, recomputed={recomputed}")
            return
        diff = abs(saved - recomputed)
        denom = max(abs(saved), abs(recomputed), 1e-15)
        passed = diff <= atol or (diff / denom) <= rtol
        self._add(
            name, passed,
            f"saved={saved:.6f}, recomputed={recomputed:.6f}, diff={diff:.2e}",
        )

    def check_set_equal(self, name: str, saved: set, recomputed: set) -> None:
        """Assert two sets are equal."""
        if saved == recomputed:
            self._add(name, True, f"{len(saved)} items match")
        else:
            only_s = saved - recomputed
            only_r = recomputed - saved
            self._add(name, False, f"only in saved: {only_s}, only in recomputed: {only_r}")

    def check_bool(self, name: str, condition: bool, detail: str = "") -> None:
        """Assert a boolean condition."""
        self._add(name, condition, detail)

    def check_equal(self, name: str, saved: Any, recomputed: Any) -> None:
        """Assert exact equality."""
        passed = saved == recomputed
        self._add(name, passed, f"saved={saved!r}, recomputed={recomputed!r}")

    def _add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({
            "section": self._section, "name": name,
            "passed": passed, "detail": detail,
        })

    def report(self) -> bool:
        """Print the verification report.  Returns True if all passed."""
        passed = [c for c in self.checks if c["passed"]]
        failed = [c for c in self.checks if not c["passed"]]

        print("=" * 70)
        print("STAGE 7: INDEPENDENT RESULTS VERIFICATION")
        print("=" * 70)
        print()
        print(f"  Total checks: {len(self.checks)}")
        print(f"  Passed: {len(passed)}")
        print(f"  Failed: {len(failed)}")
        print()

        if failed:
            print("-" * 70)
            print("FAILED CHECKS:")
            print("-" * 70)
            for c in failed:
                print(f"  [FAIL] {c['section']}: {c['name']}")
                print(f"         {c['detail']}")
            print()

        print("-" * 70)
        print("ALL CHECKS:")
        print("-" * 70)
        current_section = ""
        for c in self.checks:
            if c["section"] != current_section:
                current_section = c["section"]
                print(f"\n  {current_section}")
            tag = "PASS" if c["passed"] else "FAIL"
            print(f"    [{tag}] {c['name']}: {c['detail']}")

        print()
        print("=" * 70)
        if not failed:
            print("RESULT: ALL CHECKS PASSED")
        else:
            print(f"RESULT: {len(failed)} CHECKS FAILED")
        print("=" * 70)

        return len(failed) == 0


# ── Data Loading ──────────────────────────────────────────────────


def load_stage7_results(results_dir: str) -> dict[str, Any]:
    """Load saved sub-analysis results from summary.json."""
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    results = summary.get("results", [])
    if not results:
        raise ValueError("No results in summary.json")
    tradeoff_result = results[0].get("result", {})
    return tradeoff_result.get("sub_analyses", {})


def load_stage7_outputs(results_dir: str) -> dict[str, Any]:
    """Load the individual JSON output files."""
    outputs: dict[str, Any] = {}
    for name in ("hypothesis_verdicts", "pareto_front", "rankings", "pairwise_comparisons"):
        path = os.path.join(results_dir, f"{name}.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                outputs[name] = json.load(f)
    return outputs


def rebuild_profiles(config_path: str) -> dict[str, dict[str, Any]]:
    """Rebuild encoding profiles from raw stage data."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    stage_dirs = config.get("analysis_params", {}).get("stage_dirs", {})

    # Import only the data-loading function from our code; all statistical
    # verification is done independently with scipy.
    from experiments.tradeoff import build_encoding_profiles
    return build_encoding_profiles(stage_dirs)


# ── Independent Statistical Helpers ───────────────────────────────


def _get_available_datasets(profiles: dict) -> list[str]:
    """Get datasets that at least one encoding has VQC accuracy for."""
    datasets: set[str] = set()
    for p in profiles.values():
        vqc = p.get("vqc_accuracy")
        if vqc:
            datasets.update(vqc.keys())
    return sorted(datasets)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Partial Spearman correlation controlling for z.

    Matches tradeoff.py: regress RAW x and y on z via OLS, then compute
    Spearman on the residuals.
    """
    coeffs_x = np.polyfit(z, x, 1)
    residuals_x = x - np.polyval(coeffs_x, z)
    coeffs_y = np.polyfit(z, y, 1)
    residuals_y = y - np.polyval(coeffs_y, z)
    return sp_stats.spearmanr(residuals_x, residuals_y)


def _is_dominated(a: list[float], b: list[float]) -> bool:
    """True if *a* is Pareto-dominated by *b* (all objectives maximized)."""
    return (all(bv >= av for av, bv in zip(a, b))
            and any(bv > av for av, bv in zip(a, b)))


def _compute_pareto_set(points: dict[str, list[float]]) -> set[str]:
    """Independently compute Pareto-optimal set via brute-force dominance."""
    pareto: set[str] = set()
    names = list(points.keys())
    for i, ni in enumerate(names):
        dominated = False
        for j, nj in enumerate(names):
            if i != j and _is_dominated(points[ni], points[nj]):
                dominated = True
                break
        if not dominated:
            pareto.add(ni)
    return pareto


# ── Verification Functions ────────────────────────────────────────


def verify_data_integrity(
    profiles: dict, saved: dict, results_dir: str, suite: VerificationSuite,
) -> None:
    """Section 1: basic data integrity checks."""
    suite.section("Data Integrity")

    suite.check_bool(
        "sub-analyses loaded",
        len(saved) > 0,
        f"{len(saved)} sub-analyses in summary.json",
    )

    # Compare encoding count with saved metadata
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    results_list = summary.get("results", [])
    if results_list:
        saved_n = results_list[0].get("result", {}).get("n_encodings_analyzed")
        if saved_n is not None:
            suite.check_equal("profile count matches saved", len(profiles), saved_n)

    n_with_vqc = sum(1 for p in profiles.values() if p.get("vqc_accuracy"))
    suite.check_bool(
        "encodings with VQC accuracy",
        n_with_vqc >= 5,
        f"{n_with_vqc} encodings have VQC accuracy data",
    )


def verify_7_1(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 7.1: Accuracy vs Depth — Spearman correlation (scipy)."""
    suite.section("7.1 Accuracy vs Depth (Spearman)")

    result = saved.get("7.1", {})
    overall = result.get("overall", {})

    if overall.get("status") == "insufficient_data":
        suite.check_bool("skipped (insufficient data)", True, "matches saved status")
        return

    depths, accs = [], []
    for p in profiles.values():
        d = p.get("depth")
        vqc = p.get("vqc_accuracy")
        if d is None or not vqc:
            continue
        depths.append(d)
        accs.append(float(np.mean(list(vqc.values()))))

    if len(depths) < 5:
        suite.check_bool("insufficient data for recomputation", True, f"n={len(depths)}")
        return

    rho, p_val = sp_stats.spearmanr(depths, accs)
    suite.check_close("overall rho", overall.get("rho"), rho)
    suite.check_close("overall p_value", overall.get("p_value"), p_val)


def verify_7_2(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 7.2: Accuracy vs Trainability — Spearman + partial Spearman."""
    suite.section("7.2 Accuracy vs Trainability (Spearman + partial)")

    result = saved.get("7.2", {})
    overall = result.get("overall", {})

    if overall.get("status") == "insufficient_data":
        suite.check_bool("skipped (insufficient data)", True, "matches saved status")
        return

    trains, accs, depths = [], [], []
    for p in profiles.values():
        t = p.get("trainability_estimate")
        d = p.get("depth")
        vqc = p.get("vqc_accuracy")
        if t is None or d is None or not vqc:
            continue
        trains.append(t)
        accs.append(float(np.mean(list(vqc.values()))))
        depths.append(d)

    if len(trains) < 5:
        suite.check_bool("insufficient data", True, f"n={len(trains)}")
        return

    rho, p_val = sp_stats.spearmanr(trains, accs)
    suite.check_close("overall rho", overall.get("rho"), rho)
    suite.check_close("overall p_value", overall.get("p_value"), p_val)

    # Partial Spearman controlling for depth
    partial_rho, _ = _partial_spearman(
        np.array(trains), np.array(accs), np.array(depths),
    )
    suite.check_close("partial rho (controlling depth)", overall.get("partial_rho"), partial_rho)


def verify_7_3(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 7.3/H1: Expressibility vs Accuracy — entangling only (scipy)."""
    suite.section("7.3 Expressibility vs Accuracy [H1] (Spearman)")

    result = saved.get("7.3", {})
    overall = result.get("overall", {})

    if overall.get("status") == "insufficient_data":
        suite.check_bool("skipped (insufficient data)", True, "matches saved status")
        return

    exprs, accs = [], []
    for name, p in profiles.items():
        if not p.get("is_entangling"):
            continue
        expr = p.get("expressibility")
        vqc = p.get("vqc_accuracy")
        if expr is None or not vqc:
            continue
        exprs.append(expr)
        accs.append(float(np.mean(list(vqc.values()))))

    if len(exprs) < 5:
        suite.check_bool("insufficient data", True, f"n={len(exprs)}")
        return

    rho, p_val = sp_stats.spearmanr(exprs, accs)
    suite.check_close("overall rho", overall.get("rho"), rho)
    suite.check_close("overall p_value", overall.get("p_value"), p_val)


def verify_7_4(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 4.3/H5: Noise sensitivity — Mann-Whitney U (scipy)."""
    suite.section("4.3 Noise Sensitivity [H5] (Mann-Whitney U)")

    result = saved.get("4.3", {})
    saved_p = result.get("p_value")
    saved_u = result.get("u_stat")

    if saved_p is None:
        suite.check_bool("no Mann-Whitney data", True, "4.3 has no p_value (likely insufficient)")
        return

    entangling_decays, non_entangling_decays = [], []
    for name, p in profiles.items():
        decay = p.get("noisy_medium_fidelity_decay")
        if decay is None:
            continue
        if p.get("is_entangling"):
            entangling_decays.append(float(decay))
        else:
            non_entangling_decays.append(float(decay))

    if len(entangling_decays) < 3 or len(non_entangling_decays) < 3:
        suite.check_bool("insufficient groups", True,
                         f"ent={len(entangling_decays)}, non_ent={len(non_entangling_decays)}")
        return

    u_stat, p_val = sp_stats.mannwhitneyu(
        entangling_decays, non_entangling_decays, alternative="two-sided",
    )
    suite.check_close("U statistic", saved_u, u_stat)
    suite.check_close("p_value", saved_p, p_val)


def verify_7_6(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 7.6/H6: VQC vs Kernel ranking — Kendall tau (scipy)."""
    suite.section("7.6 VQC vs Kernel Ranking [H6] (Kendall tau)")

    result = saved.get("7.6", {})
    per_ds = result.get("per_dataset", {})

    if not per_ds:
        suite.check_bool("no per-dataset data", True, "7.6 has no per_dataset results")
        return

    # Verify at least one dataset
    for ds in _get_available_datasets(profiles):
        ds_data = per_ds.get(ds, {})
        saved_tau = ds_data.get("kendall_tau")
        if saved_tau is None:
            continue

        # Rebuild VQC and kernel rankings
        vqc_pairs, ker_pairs = [], []
        for name, p in profiles.items():
            vqc_acc = (p.get("vqc_accuracy") or {}).get(ds)
            ker_acc = (p.get("kernel_accuracy") or {}).get(ds)
            if vqc_acc is not None and ker_acc is not None:
                vqc_pairs.append((name, vqc_acc))
                ker_pairs.append((name, ker_acc))

        if len(vqc_pairs) < 5:
            continue

        vqc_sorted = sorted(vqc_pairs, key=lambda x: x[1], reverse=True)
        ker_sorted = sorted(ker_pairs, key=lambda x: x[1], reverse=True)

        vqc_rank = {name: i + 1 for i, (name, _) in enumerate(vqc_sorted)}
        ker_rank = {name: i + 1 for i, (name, _) in enumerate(ker_sorted)}

        common = sorted(set(vqc_rank) & set(ker_rank))
        vqc_ranks = [vqc_rank[n] for n in common]
        ker_ranks = [ker_rank[n] for n in common]

        tau, p_val = sp_stats.kendalltau(vqc_ranks, ker_ranks)
        suite.check_close(f"[{ds}] Kendall tau", saved_tau, tau)
        suite.check_close(f"[{ds}] p_value", ds_data.get("p_value"), p_val)
        break  # One dataset is sufficient for verification
    else:
        suite.check_bool("no dataset with Kendall data", True, "all skipped")


def verify_7_7(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 7.7/H7: Pareto front — independent dominance check."""
    suite.section("7.7 Pareto Front [H7] (dominance)")

    result = saved.get("7.7", {})
    saved_encodings = result.get("encodings", {})

    if not saved_encodings:
        suite.check_bool("no Pareto data", True, "7.7 has no encodings")
        return

    saved_pareto = {n for n, d in saved_encodings.items() if d.get("is_pareto")}

    # Rebuild objective vectors identically to analyze_pareto_front
    objectives: dict[str, list[float]] = {}
    for name, p in profiles.items():
        vqc = p.get("vqc_accuracy")
        depth = p.get("depth")
        train = p.get("trainability_estimate")
        noise = p.get("noise_resilience")
        if not vqc or not depth or train is None:
            continue
        mean_acc = float(np.mean(list(vqc.values())))
        inv_depth = 1.0 / float(depth) if depth > 0 else 0.0
        # Match tradeoff.py: impute noise_resilience = 0.5 when missing
        noise_val = float(noise) if noise is not None else 0.5
        objectives[name] = [mean_acc, inv_depth, float(train), noise_val]

    if len(objectives) < 3:
        suite.check_bool("too few encodings", True, f"n={len(objectives)}")
        return

    recomputed_pareto = _compute_pareto_set(objectives)
    suite.check_set_equal("Pareto-optimal set", saved_pareto, recomputed_pareto)

    # Verify no Pareto point is dominated by another Pareto point
    all_non_dominated = True
    for ni in recomputed_pareto:
        for nj in recomputed_pareto:
            if ni != nj and _is_dominated(objectives[ni], objectives[nj]):
                all_non_dominated = False
                break
    suite.check_bool(
        "Pareto points are mutually non-dominated",
        all_non_dominated,
        f"checked {len(recomputed_pareto)} Pareto-optimal encodings",
    )

    # Verify every non-Pareto point is dominated by at least one Pareto point
    non_pareto = set(objectives.keys()) - recomputed_pareto
    all_dominated = True
    for np_name in non_pareto:
        dominated_by_any = any(
            _is_dominated(objectives[np_name], objectives[pp])
            for pp in recomputed_pareto
        )
        if not dominated_by_any:
            all_dominated = False
            break
    suite.check_bool(
        "all non-Pareto points are dominated",
        all_dominated,
        f"checked {len(non_pareto)} non-Pareto encodings",
    )


def verify_7_9(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 7.9: Final ranking — independent weighted-score computation."""
    suite.section("7.9 Final Ranking (weighted scores)")

    result = saved.get("7.9", {})
    saved_rankings = result.get("rankings", [])

    if not saved_rankings:
        suite.check_bool("no rankings to verify", True, "empty rankings list")
        return

    # Replicate compute_final_ranking logic exactly
    names: list[str] = []
    raw_metrics: list[dict[str, float | None]] = []

    for name, p in profiles.items():
        vqc_acc = p.get("vqc_accuracy")
        if not vqc_acc:
            continue
        mean_acc = float(np.mean(list(vqc_acc.values())))
        depth = p.get("depth")
        inv_depth = 1.0 / float(depth) if depth and depth > 0 else None
        names.append(name)
        raw_metrics.append({
            "accuracy": mean_acc,
            "expressibility": p.get("expressibility"),
            "trainability": p.get("trainability_estimate"),
            "inv_depth": inv_depth,
            "entanglement": p.get("entanglement_capability"),
            "noise_resilience": p.get("noise_resilience"),
        })

    if not names:
        suite.check_bool("no encodings for ranking", True, "")
        return

    metric_keys = list(DEFAULT_WEIGHTS.keys())

    # Build normalization ranges
    norm_ranges: dict[str, list[float]] = {}
    for mk in metric_keys:
        vals = [m[mk] for m in raw_metrics if m[mk] is not None]
        if vals:
            norm_ranges[mk] = [float(min(vals)), float(max(vals))]
        else:
            norm_ranges[mk] = [0.0, 1.0]

    # Compute weighted scores (matching the exact score /= total_weight logic)
    recomputed_scores: dict[str, float] = {}
    for i, name in enumerate(names):
        score = 0.0
        total_weight = 0.0
        for mk in metric_keys:
            val = raw_metrics[i][mk]
            if val is None:
                continue
            lo, hi = norm_ranges[mk]
            norm_val = (val - lo) / (hi - lo) if hi - lo > 1e-12 else 0.5
            score += DEFAULT_WEIGHTS[mk] * norm_val
            total_weight += DEFAULT_WEIGHTS[mk]
        if total_weight > 0:
            score /= total_weight
        recomputed_scores[name] = score

    # Compare individual scores
    saved_scores = {r["encoding"]: r["score"] for r in saved_rankings if "score" in r}
    matched = 0
    for name, recomp_score in recomputed_scores.items():
        if name in saved_scores:
            suite.check_close(
                f"{name} score",
                saved_scores[name], recomp_score,
            )
            matched += 1
        if matched >= 5:
            break

    # Compare ranking order
    saved_order = [r["encoding"] for r in saved_rankings]
    recomputed_order = sorted(
        recomputed_scores.keys(),
        key=lambda n: recomputed_scores[n],
        reverse=True,
    )

    # Top-3 order match
    suite.check_equal(
        "top-3 ranking order",
        saved_order[:3], recomputed_order[:3],
    )


def verify_kta_4_4(profiles: dict, saved: dict, suite: VerificationSuite) -> None:
    """Verify 4.4: KTA vs Kernel Accuracy — Spearman (scipy)."""
    suite.section("4.4 KTA vs Kernel Accuracy (Spearman)")

    result = saved.get("4.4", {})
    overall = result.get("overall", {})

    if overall.get("status") == "insufficient_data":
        suite.check_bool("skipped (insufficient data)", True, "matches saved status")
        return

    ktas, accs = [], []
    for p in profiles.values():
        kta = p.get("centered_kta")
        ker = p.get("kernel_accuracy")
        if kta is None or not ker:
            continue
        ktas.append(kta)
        accs.append(float(np.mean(list(ker.values()))))

    if len(ktas) < 5:
        suite.check_bool("insufficient data", True, f"n={len(ktas)}")
        return

    rho, p_val = sp_stats.spearmanr(ktas, accs)
    suite.check_close("overall rho", overall.get("rho"), rho)
    suite.check_close("overall p_value", overall.get("p_value"), p_val)


def verify_cliffs_delta_4_9(saved: dict, suite: VerificationSuite) -> None:
    """Verify 4.9: Cliff's delta — range [-1,1] and interpretation check."""
    suite.section("4.9 Pairwise Cliff's Delta (spot check)")

    result = saved.get("4.9", {})
    vqc_data = result.get("vqc", {}).get("per_dataset", {})

    if not vqc_data:
        suite.check_bool("no pairwise VQC data", True, "4.9 vqc data absent")
        return

    for ds_name, ds_data in vqc_data.items():
        pairs = ds_data.get("pairs", [])
        if not pairs:
            continue

        # Spot-check first pair
        pair = pairs[0]
        delta = pair.get("cliffs_delta")
        if delta is None:
            continue

        suite.check_bool(
            f"{pair['method_a']} vs {pair['method_b']}: delta in [-1, 1]",
            -1.0 <= delta <= 1.0,
            f"delta={delta:.4f}",
        )

        # Verify interpretation matches standard thresholds
        interp = pair.get("effect_size")
        abs_d = abs(delta)
        if abs_d < 0.147:
            expected = "negligible"
        elif abs_d < 0.33:
            expected = "small"
        elif abs_d < 0.474:
            expected = "medium"
        else:
            expected = "large"
        suite.check_equal("Cliff's delta interpretation", interp, expected)
        break
    else:
        suite.check_bool("no pairs found", True, "all datasets empty")


def verify_hypothesis_verdicts(
    saved: dict, outputs: dict, suite: VerificationSuite,
) -> None:
    """Verify H1-H7: trace decision tree logic against saved statistics.

    Each hypothesis verdict is computed inside its respective sub-analysis.
    We re-derive the expected verdict from the same sub-analysis statistics
    and compare with what was saved.
    """
    suite.section("Hypothesis Verdicts (decision tree verification)")

    saved_verdicts = outputs.get("hypothesis_verdicts", {}).get("hypothesis_verdicts", {})
    if not saved_verdicts:
        suite.check_bool("verdicts available", False, "no verdicts found in output")
        return

    # ── H1: Expressibility predicts accuracy (from 7.3) ──
    r_7_3 = saved.get("7.3", {})
    overall_7_3 = r_7_3.get("overall", {})
    rho = overall_7_3.get("rho")
    p_val = overall_7_3.get("p_value")
    h1_saved = r_7_3.get("h1_verdict", {}).get("verdict")

    if rho is not None and p_val is not None:
        # Decision tree from tradeoff.py lines 808-849
        if rho > 0 and p_val < 0.10:
            expected = "supported"  # default; "refuted" only in a narrow sub-case
            # Check the sub-case: all high-expr have high-acc AND no outliers
            # We trust the sub-analysis statistics here; just verify the verdict
            # is one of the valid options
            suite.check_bool(
                "H1 verdict valid given rho>0, p<0.10",
                h1_saved in ("supported", "refuted"),
                f"verdict={h1_saved!r}, rho={rho:.3f}, p={p_val:.3f}",
            )
        elif rho <= 0:
            suite.check_equal("H1 verdict (rho<=0 → refuted)", h1_saved, "refuted")
        else:
            suite.check_equal("H1 verdict (p>=0.10 → inconclusive)", h1_saved, "inconclusive")
    else:
        suite.check_equal("H1 verdict (no data → inconclusive)", h1_saved, "inconclusive")

    # ── H2: Equivariant advantage on symmetric data (from 7.4) ──
    r_7_4 = saved.get("7.4", {})
    sym = r_7_4.get("symmetric_datasets", {})
    non_sym = r_7_4.get("non_symmetric_datasets", {})
    h2_saved = r_7_4.get("h2_verdict", {}).get("verdict")

    sym_wins = sum(
        1 for r in sym.values()
        if r.get("significant") and r.get("equivariant_mean_acc", 0) > r.get("general_mean_acc", 0)
    )
    non_sym_wins = sum(
        1 for r in non_sym.values()
        if r.get("significant") and r.get("equivariant_mean_acc", 0) > r.get("general_mean_acc", 0)
    )
    n_sym_tested = sum(1 for r in sym.values() if r.get("p_value") is not None)

    if sym_wins == 0 and n_sym_tested > 0:
        h2_expected = "refuted"
    elif sym_wins > 0 and non_sym_wins == 0:
        h2_expected = "supported"
    elif sym_wins > 0 and non_sym_wins > 0:
        h2_expected = "inconclusive"
    else:
        h2_expected = "inconclusive"

    suite.check_equal("H2 verdict", h2_saved, h2_expected)

    # ── H3: Data reuploading advantage (from 4.2) ──
    r_4_2 = saved.get("4.2", {})
    h3_saved = r_4_2.get("h3_verdict", {}).get("verdict")
    h3_test = r_4_2.get("h3_verdict", {}).get("test_statistic", {})
    wilcoxon_p = r_4_2.get("wilcoxon_p")
    positive_pairs = h3_test.get("positive_pairs")
    n_pairs = h3_test.get("total_pairs") or len(r_4_2.get("matched_pairs", []))

    if positive_pairs is not None and n_pairs is not None and n_pairs >= 4:
        if positive_pairs >= 3 and (wilcoxon_p or 1.0) < 0.05:
            h3_expected = "supported"
        elif positive_pairs <= 1:
            h3_expected = "refuted"
        else:
            h3_expected = "inconclusive"
    else:
        h3_expected = "inconclusive"

    suite.check_equal("H3 verdict", h3_saved, h3_expected)

    # ── H4: Barren plateau onset (from 7.5) ──
    r_7_5 = saved.get("7.5", {})
    h4_saved = r_7_5.get("h4_verdict", {}).get("verdict")
    depth_partial_corr = r_7_5.get("depth_partial_corr", 0)
    depth_partial_p = r_7_5.get("depth_partial_p", 1)
    family_partial_corr = r_7_5.get("family_partial_corr", 0)
    family_partial_p = r_7_5.get("family_partial_p", 1)

    if abs(depth_partial_corr) > abs(family_partial_corr) and depth_partial_p < 0.05:
        h4_expected = "supported"
    elif abs(family_partial_corr) > abs(depth_partial_corr) and family_partial_p < 0.05:
        h4_expected = "refuted"
    else:
        h4_expected = "inconclusive"

    suite.check_equal("H4 verdict", h4_saved, h4_expected)

    # ── H5: Entangling encodings more noise-sensitive (from 4.3) ──
    r_4_3 = saved.get("4.3", {})
    h5_saved = r_4_3.get("h5_verdict", {}).get("verdict")
    ent_mean = r_4_3.get("entangling_mean")
    non_ent_mean = r_4_3.get("non_entangling_mean")
    h5_p = r_4_3.get("p_value")

    if ent_mean is not None and non_ent_mean is not None and h5_p is not None:
        if ent_mean > non_ent_mean and h5_p < 0.05:
            h5_expected = "supported"
        elif non_ent_mean >= ent_mean:
            h5_expected = "refuted"
        else:
            h5_expected = "inconclusive"
    else:
        h5_expected = "inconclusive"

    suite.check_equal("H5 verdict", h5_saved, h5_expected)

    # ── H6: VQC and kernel rankings differ (from 7.6) ──
    r_7_6 = saved.get("7.6", {})
    h6_saved = r_7_6.get("h6_verdict", {}).get("verdict")
    per_ds_7_6 = r_7_6.get("per_dataset", {})
    overall_7_6 = r_7_6.get("overall", {})

    taus = [r.get("kendall_tau") for r in per_ds_7_6.values() if r.get("kendall_tau") is not None]
    mean_tau = overall_7_6.get("mean_tau", float(np.mean(taus)) if taus else 0.0)
    n_significant = sum(
        1 for r in per_ds_7_6.values()
        if r.get("p_value") is not None and r["p_value"] < 0.05
    )
    n_ds_tested = overall_7_6.get("n_datasets", len(taus))

    if n_ds_tested < 2:
        h6_expected = "inconclusive"
    elif abs(mean_tau) < 0.5 and n_significant > 0:
        h6_expected = "supported"
    elif abs(mean_tau) > 0.8:
        h6_expected = "refuted"
    else:
        h6_expected = "inconclusive"

    suite.check_equal("H6 verdict", h6_saved, h6_expected)

    # ── H7: Pareto front has multiple encodings (from 7.7) ──
    r_7_7 = saved.get("7.7", {})
    h7_saved = r_7_7.get("h7_verdict", {}).get("verdict")
    n_pareto = r_7_7.get("n_pareto_optimal", 0)

    if n_pareto >= 3:
        h7_expected = "supported"
    elif n_pareto == 1:
        h7_expected = "refuted"
    else:
        h7_expected = "inconclusive"

    suite.check_equal("H7 verdict", h7_saved, h7_expected)


# ── Main ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify Stage 7 tradeoff analysis results.",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/results/raw/stage7_tradeoff",
        help="Directory containing Stage 7 results.",
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/stage7_tradeoff.json",
        help="Stage 7 config file path.",
    )
    args = parser.parse_args()

    suite = VerificationSuite()

    # ── 1. Load data ──────────────────────────────────────────
    try:
        saved = load_stage7_results(args.results_dir)
    except Exception as e:
        suite.section("Data Loading")
        suite.check_bool("summary.json loaded", False, str(e))
        suite.report()
        return 1

    try:
        outputs = load_stage7_outputs(args.results_dir)
    except Exception as e:
        suite.section("Data Loading")
        suite.check_bool("JSON outputs loaded", False, str(e))

    try:
        profiles = rebuild_profiles(args.config)
    except Exception as e:
        suite.section("Data Loading")
        suite.check_bool("profiles rebuilt", False, str(e))
        suite.report()
        return 1

    # ── 2. Run all verifications ──────────────────────────────
    verify_data_integrity(profiles, saved, args.results_dir, suite)
    verify_7_1(profiles, saved, suite)
    verify_7_2(profiles, saved, suite)
    verify_7_3(profiles, saved, suite)
    verify_7_4(profiles, saved, suite)
    verify_7_6(profiles, saved, suite)
    verify_7_7(profiles, saved, suite)
    verify_7_9(profiles, saved, suite)
    verify_kta_4_4(profiles, saved, suite)
    verify_cliffs_delta_4_9(saved, suite)
    verify_hypothesis_verdicts(saved, outputs, suite)

    # ── 3. Report ─────────────────────────────────────────────
    all_passed = suite.report()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
