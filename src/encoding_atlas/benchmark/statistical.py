"""Statistical tests for benchmark comparison.

Provides paired Wilcoxon signed-rank testing, Holm-Bonferroni correction across
the family of pairwise comparisons, and Cliff's delta effect sizes — the same
methodology used to produce the empirical atlas.
"""

from __future__ import annotations

from typing import Any

# Cliff's delta magnitude thresholds (Romano et al., 2006).
_CLIFF_THRESHOLDS = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"))


def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
) -> tuple[float, float]:
    """Perform a paired Wilcoxon signed-rank test on two score lists.

    Returns ``(statistic, p_value)``. If all paired differences are zero
    (identical scores), returns ``(0.0, 1.0)`` rather than raising, so a sweep
    of comparisons does not abort on ties.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise ImportError("scipy required for statistical tests") from exc

    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have equal length")
    if all(a == b for a, b in zip(scores_a, scores_b)):
        return 0.0, 1.0

    stat, p_value = wilcoxon(scores_a, scores_b)
    return float(stat), float(p_value)


def cliffs_delta(scores_a: list[float], scores_b: list[float]) -> tuple[float, str]:
    """Compute Cliff's delta effect size and its magnitude label.

    Cliff's delta is ``(#(a > b) - #(a < b)) / (n_a * n_b)`` over all pairs,
    a non-parametric, sign-based effect size in ``[-1, 1]``. The magnitude
    label follows Romano et al. (2006): negligible / small / medium / large.
    """
    if not scores_a or not scores_b:
        return 0.0, "negligible"

    greater = sum(a > b for a in scores_a for b in scores_b)
    less = sum(a < b for a in scores_a for b in scores_b)
    delta = (greater - less) / (len(scores_a) * len(scores_b))

    magnitude = "large"
    for threshold, label in _CLIFF_THRESHOLDS:
        if abs(delta) < threshold:
            magnitude = label
            break
    return float(delta), magnitude


def holm_bonferroni(pvalues: dict[Any, float]) -> dict[Any, float]:
    """Apply Holm-Bonferroni step-down correction to a family of p-values.

    Parameters
    ----------
    pvalues : dict
        Mapping of comparison key -> raw p-value.

    Returns
    -------
    dict
        Mapping of the same keys -> corrected p-value (monotone, clipped to 1).
    """
    if not pvalues:
        return {}

    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected: dict[Any, float] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        corrected[key] = running_max
    return corrected


def compare_encodings(
    results: dict[str, list[float]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compare all encoding pairs with the Wilcoxon test (raw p-values).

    Returns a nested mapping ``name_a -> name_b -> (statistic, p_value)`` for
    each unordered pair. Retained for backward compatibility; see
    :func:`compare_encodings_corrected` for the full analysis.
    """
    names = list(results.keys())
    comparisons: dict[str, dict[str, tuple[float, float]]] = {}
    for i, name_a in enumerate(names):
        comparisons[name_a] = {}
        for name_b in names[i + 1 :]:
            comparisons[name_a][name_b] = wilcoxon_test(
                results[name_a], results[name_b]
            )
    return comparisons


def compare_encodings_corrected(
    results: dict[str, list[float]],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Full pairwise comparison: Wilcoxon + Holm-Bonferroni + Cliff's delta.

    Parameters
    ----------
    results : dict
        Mapping of encoding name -> list of paired per-fold/run scores. All
        lists must have equal length (paired across the same CV splits).
    alpha : float, default=0.05
        Family-wise significance level used to flag ``significant`` pairs.

    Returns
    -------
    dict
        ``{"n_comparisons": int, "alpha": float, "pairs": [...]}`` where each
        pair record holds the two encodings, raw and corrected p-values,
        Wilcoxon statistic, Cliff's delta with magnitude, the mean-score
        difference, and a ``significant`` boolean (corrected p < alpha).
    """
    names = list(results.keys())
    raw: dict[tuple[str, str], dict[str, Any]] = {}

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            a, b = results[name_a], results[name_b]
            stat, p_value = wilcoxon_test(a, b)
            delta, magnitude = cliffs_delta(a, b)
            mean_diff = (sum(a) / len(a) - sum(b) / len(b)) if a and b else 0.0
            raw[(name_a, name_b)] = {
                "encoding_a": name_a,
                "encoding_b": name_b,
                "statistic": stat,
                "p_value": p_value,
                "cliffs_delta": delta,
                "effect_magnitude": magnitude,
                "mean_difference": mean_diff,
            }

    corrected = holm_bonferroni({key: rec["p_value"] for key, rec in raw.items()})

    pairs = []
    for key, rec in raw.items():
        rec["p_value_corrected"] = corrected[key]
        rec["significant"] = corrected[key] < alpha
        pairs.append(rec)
    pairs.sort(key=lambda r: r["p_value_corrected"])

    return {"n_comparisons": len(pairs), "alpha": alpha, "pairs": pairs}
