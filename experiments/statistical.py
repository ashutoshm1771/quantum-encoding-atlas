"""Statistical analysis utilities for Stage 6 experiments.

This module provides statistical tools for rigorous analysis of experiment
results, including:

- Hypothesis testing (Wilcoxon signed-rank test)
- Multiple comparison correction (Holm-Bonferroni)
- Effect size estimation (Cliff's delta)
- Confidence intervals (bootstrap)

These tools implement the Statistical Analysis Protocol specified in
the Phase 4 experiment plan.

Statistical Analysis Protocol
-----------------------------
1. Cross-validation: 5-fold stratified CV (not single train/test split)
2. Paired tests: Wilcoxon signed-rank for paired fold comparisons
3. Correction: Holm-Bonferroni for multiple comparisons
4. Effect sizes: Cliff's delta with standard thresholds
5. Uncertainty: 95% bootstrap confidence intervals
"""

from __future__ import annotations

import logging
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Hypothesis Testing
# =============================================================================


def wilcoxon_test(
    scores_a: NDArray[np.floating],
    scores_b: NDArray[np.floating],
) -> Tuple[float, float]:
    """Perform paired Wilcoxon signed-rank test.

    Tests whether there is a significant difference between two sets of
    paired scores (e.g., accuracies across folds).

    Parameters
    ----------
    scores_a : array-like
        Scores from method A (e.g., encoding 1 accuracies per fold).
    scores_b : array-like
        Scores from method B (e.g., encoding 2 accuracies per fold).
        Must have the same length as scores_a.

    Returns
    -------
    statistic : float
        The Wilcoxon test statistic.
    p_value : float
        Two-sided p-value for the test.

    Notes
    -----
    - Requires paired samples (same folds for both methods)
    - Non-parametric: doesn't assume normal distribution
    - Use Holm-Bonferroni correction when doing multiple comparisons

    Examples
    --------
    >>> scores_enc1 = np.array([0.85, 0.87, 0.84, 0.86, 0.88])
    >>> scores_enc2 = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
    >>> stat, p = wilcoxon_test(scores_enc1, scores_enc2)
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Arrays must have same length: {len(scores_a)} vs {len(scores_b)}"
        )

    # Check for zero differences (Wilcoxon can't handle identical scores)
    differences = scores_a - scores_b
    if np.allclose(differences, 0):
        # No difference detected
        return 0.0, 1.0

    try:
        result = stats.wilcoxon(scores_a, scores_b)
        return float(result.statistic), float(result.pvalue)
    except ValueError as e:
        # Can happen with very small samples or identical values
        logger.warning("Wilcoxon test failed: %s", str(e))
        return 0.0, 1.0


def mann_whitney_test(
    scores_a: NDArray[np.floating],
    scores_b: NDArray[np.floating],
) -> Tuple[float, float, float]:
    """Perform Mann-Whitney U test for independent samples.

    Used for comparing encoding families (e.g., equivariant vs general)
    where samples are not paired.

    Parameters
    ----------
    scores_a : array-like
        Scores from group A.
    scores_b : array-like
        Scores from group B.

    Returns
    -------
    statistic : float
        The Mann-Whitney U statistic.
    p_value : float
        Two-sided p-value for the test.
    rank_biserial : float
        Effect size (rank-biserial correlation). Range [-1, 1].

    Notes
    -----
    The rank-biserial correlation is computed as:
        r = 2*U / (n1*n2) - 1

    Effect size interpretation:
    - |r| < 0.1: negligible
    - |r| < 0.3: small
    - |r| < 0.5: medium
    - |r| >= 0.5: large
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    result = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    # Compute rank-biserial correlation as effect size
    n1, n2 = len(scores_a), len(scores_b)
    rank_biserial = 2 * statistic / (n1 * n2) - 1

    return statistic, p_value, float(rank_biserial)


# =============================================================================
# Multiple Comparison Correction
# =============================================================================


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    The Holm-Bonferroni method is a step-down procedure that controls
    the family-wise error rate while being less conservative than the
    standard Bonferroni correction.

    Parameters
    ----------
    p_values : list of float
        Raw p-values from multiple hypothesis tests.
    alpha : float, default=0.05
        Significance level (family-wise error rate).

    Returns
    -------
    rejected : list of bool
        For each p-value, True if null hypothesis is rejected.

    Notes
    -----
    Procedure:
    1. Sort p-values in ascending order: p₁ ≤ p₂ ≤ ... ≤ pₙ
    2. For rank k, reject if pₖ < α/(n - k + 1)
    3. Stop at first non-rejection; all subsequent are non-significant

    With 16 encodings, there are C(16,2) = 120 pairwise comparisons.
    Without correction, ~6 would appear significant by chance at α=0.05.

    Examples
    --------
    >>> p_values = [0.001, 0.03, 0.04, 0.05]
    >>> rejected = holm_bonferroni_correction(p_values)
    >>> rejected  # [True, True, False, False] (with thresholds 0.0125, 0.0167, ...)
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    rejected = [False] * n

    for k, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        # Holm-Bonferroni threshold for rank k (0-indexed)
        threshold = alpha / (n - k)

        if p < threshold:
            rejected[idx] = True
        else:
            # Stop at first non-rejection
            break

    n_rejected = sum(rejected)
    logger.debug(
        "Holm-Bonferroni: %d/%d null hypotheses rejected at α=%.3f",
        n_rejected, n, alpha
    )

    return rejected


# =============================================================================
# Effect Size Estimation
# =============================================================================


def cliffs_delta(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> float:
    """Compute Cliff's delta effect size.

    Cliff's delta is a non-parametric effect size measure that quantifies
    how often values in one group are larger than values in another.

    Parameters
    ----------
    x : array-like
        Scores from group X.
    y : array-like
        Scores from group Y.

    Returns
    -------
    delta : float
        Cliff's delta, in range [-1, 1].
        - δ = +1: all x > all y
        - δ = -1: all x < all y
        - δ = 0: distributions are identical

    Notes
    -----
    Effect size interpretation (Vargha & Delaney thresholds):
    - |δ| < 0.147: negligible
    - |δ| < 0.33: small
    - |δ| < 0.474: medium
    - |δ| >= 0.474: large

    Cliff's delta is the difference in dominance:
        δ = (n_more - n_less) / (n_x * n_y)

    where n_more = count(x_i > y_j) and n_less = count(x_i < y_j).

    Examples
    --------
    >>> x = np.array([0.90, 0.92, 0.88, 0.91, 0.89])
    >>> y = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
    >>> delta = cliffs_delta(x, y)  # Should be close to 1.0 (x dominates)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n_x, n_y = len(x), len(y)

    if n_x == 0 or n_y == 0:
        return 0.0

    # Count dominance relations
    more = 0
    less = 0

    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1

    delta = (more - less) / (n_x * n_y)
    return float(delta)


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta effect size.

    Parameters
    ----------
    delta : float
        Cliff's delta value.

    Returns
    -------
    str
        Effect size category: 'negligible', 'small', 'medium', or 'large'.
    """
    abs_delta = abs(delta)

    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


# =============================================================================
# Confidence Intervals
# =============================================================================


def bootstrap_ci(
    scores: NDArray[np.floating],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses the percentile method to estimate the confidence interval
    for the population mean from a sample.

    Parameters
    ----------
    scores : array-like
        Sample of scores.
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI).
    n_bootstrap : int, default=10000
        Number of bootstrap resamples.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.

    Notes
    -----
    The percentile method:
    1. Resample with replacement n_bootstrap times
    2. Compute mean of each resample
    3. Take α/2 and 1-α/2 percentiles as CI bounds

    For 95% CI: lower = 2.5th percentile, upper = 97.5th percentile

    Examples
    --------
    >>> scores = np.array([0.85, 0.87, 0.84, 0.86, 0.88])
    >>> lower, upper = bootstrap_ci(scores, confidence=0.95)
    """
    scores = np.asarray(scores)
    n = len(scores)

    if n == 0:
        return np.nan, np.nan

    if n == 1:
        return float(scores[0]), float(scores[0])

    rng = np.random.default_rng(seed)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Percentile confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100

    lower = float(np.percentile(bootstrap_means, lower_percentile))
    upper = float(np.percentile(bootstrap_means, upper_percentile))

    return lower, upper


def compute_ci_metrics(
    scores: NDArray[np.floating],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute mean, std, and confidence interval for a set of scores.

    Convenience function that computes all relevant statistics.

    Parameters
    ----------
    scores : array-like
        Sample of scores.
    confidence : float, default=0.95
        Confidence level.
    n_bootstrap : int, default=10000
        Number of bootstrap resamples for CI.
    seed : int, default=42
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mean': float
        - 'std': float
        - 'ci_lower': float
        - 'ci_upper': float
    """
    scores = np.asarray(scores)

    if len(scores) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    ci_lower, ci_upper = bootstrap_ci(
        scores,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# =============================================================================
# Correlation Analysis
# =============================================================================


def spearman_correlation(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> Tuple[float, float]:
    """Compute Spearman rank correlation coefficient.

    Used for measuring monotonic relationships between variables
    (e.g., accuracy vs circuit depth).

    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.

    Returns
    -------
    rho : float
        Spearman correlation coefficient, in range [-1, 1].
    p_value : float
        Two-sided p-value for testing non-correlation.

    Notes
    -----
    Interpretation:
    - |ρ| < 0.3: weak
    - |ρ| < 0.6: moderate
    - |ρ| >= 0.6: strong
    """
    x = np.asarray(x)
    y = np.asarray(y)

    result = stats.spearmanr(x, y)
    return float(result.correlation), float(result.pvalue)


def kendall_tau(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> Tuple[float, float]:
    """Compute Kendall's tau rank correlation.

    Used for measuring agreement between two rankings
    (e.g., VQC vs kernel encoding rankings).

    Parameters
    ----------
    x : array-like
        First ranking.
    y : array-like
        Second ranking.

    Returns
    -------
    tau : float
        Kendall's tau coefficient, in range [-1, 1].
    p_value : float
        Two-sided p-value for testing independence.

    Notes
    -----
    τ = 1: Perfect agreement
    τ = -1: Perfect disagreement
    τ = 0: No association
    """
    x = np.asarray(x)
    y = np.asarray(y)

    result = stats.kendalltau(x, y)
    return float(result.correlation), float(result.pvalue)


# =============================================================================
# Pairwise Comparison Analysis
# =============================================================================


def pairwise_comparisons(
    results: dict[str, NDArray[np.floating]],
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Perform pairwise statistical comparisons between methods.

    Runs Wilcoxon tests for all pairs, applies Holm-Bonferroni correction,
    and computes Cliff's delta effect sizes.

    Parameters
    ----------
    results : dict
        Dictionary mapping method name to array of scores (e.g., per-fold).
    alpha : float, default=0.05
        Significance level for hypothesis testing.

    Returns
    -------
    dict
        Dictionary with comparison results:
        - 'pairs': list of dicts, each with:
            - 'method_a': str
            - 'method_b': str
            - 'p_value': float (raw)
            - 'significant': bool (after Holm-Bonferroni)
            - 'cliffs_delta': float
            - 'effect_size': str ('negligible', 'small', 'medium', 'large')
        - 'n_significant': int
        - 'n_comparisons': int
    """
    method_names = list(results.keys())
    n_methods = len(method_names)

    # Generate all pairs
    pairs = []
    p_values = []

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            name_a = method_names[i]
            name_b = method_names[j]
            scores_a = results[name_a]
            scores_b = results[name_b]

            _, p_value = wilcoxon_test(scores_a, scores_b)
            delta = cliffs_delta(scores_a, scores_b)

            pairs.append({
                "method_a": name_a,
                "method_b": name_b,
                "p_value": p_value,
                "cliffs_delta": delta,
                "effect_size": interpret_cliffs_delta(delta),
            })
            p_values.append(p_value)

    # Apply Holm-Bonferroni correction
    significant = holm_bonferroni_correction(p_values, alpha=alpha)

    for pair, is_sig in zip(pairs, significant):
        pair["significant"] = is_sig

    return {
        "pairs": pairs,
        "n_significant": sum(significant),
        "n_comparisons": len(pairs),
    }
