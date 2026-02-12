"""Statistical tests for benchmark comparison."""


def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test."""
    try:
        from scipy.stats import wilcoxon

        stat, p_value = wilcoxon(scores_a, scores_b)
        return float(stat), float(p_value)
    except ImportError:
        raise ImportError("scipy required for statistical tests")


def compare_encodings(
    results: dict[str, list[float]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compare all encoding pairs using Wilcoxon test."""
    encoding_names = list(results.keys())
    comparisons = {}

    for i, name_a in enumerate(encoding_names):
        comparisons[name_a] = {}
        for name_b in encoding_names[i + 1 :]:
            stat, p_value = wilcoxon_test(results[name_a], results[name_b])
            comparisons[name_a][name_b] = (stat, p_value)

    return comparisons
