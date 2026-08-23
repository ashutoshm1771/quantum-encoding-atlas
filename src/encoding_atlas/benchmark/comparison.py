"""Comparing many encodings across many datasets (Demsar, 2006).

The rest of :mod:`encoding_atlas.benchmark.statistical` answers "is encoding A
better than encoding B on this data?". This module answers the question the
atlas actually asks: "across a suite of datasets, which of these sixteen
encodings can be told apart at all?"

Why all-pairs Wilcoxon cannot answer it
---------------------------------------
The obvious approach — run :func:`~encoding_atlas.benchmark.statistical.
wilcoxon_test` on every pair and correct with Holm-Bonferroni — is not merely
weak at this scale, it is *arithmetically incapable* of returning a positive
result. A paired Wilcoxon signed-rank test over ``N`` datasets has an exact
two-sided minimum attainable p-value of ``2 / 2**N``; comparing ``k`` methods
generates ``m = k (k - 1) / 2`` comparisons, and Holm's strictest threshold is
``alpha / m``. For the atlas's ``k = 15``, ``N = 8``::

    min attainable p = 2 / 2**8   = 0.0078
    Holm threshold   = 0.05 / 105 = 0.00048

Since ``0.0078 > 0.00048``, no pair can reach significance no matter how large
the true effect is. This is exactly the situation Demsar (2006) was written to
address, and the reason the omnibus-plus-post-hoc route below is the correct
one rather than merely the conventional one.

The procedure
-------------
1. **Rank** the methods within each dataset: 1 = best. Ranking per dataset,
   rather than comparing raw accuracies, is what makes datasets of wildly
   different difficulty commensurable.
2. **Omnibus.** The Friedman test asks whether the average ranks differ more
   than chance would allow. Its chi-square form is known to be conservative, so
   the Iman-Davenport F is reported alongside and is what
   :attr:`FriedmanResult.rejected` uses.
3. **Post hoc, only if the omnibus rejects.** Nemenyi for all-pairs, or
   Bonferroni-Dunn when every method is compared against one control. Two
   methods differ if their average ranks differ by at least the critical
   difference

       CD = q_alpha * sqrt(k (k + 1) / (6 N)) .

Running a post-hoc test without a rejecting omnibus inflates the error rate and
is the single most common misuse of this procedure; :func:`compare_over_datasets`
therefore returns ``posthoc=None`` in that case rather than a table the caller
might quote.

Missing cells
-------------
Friedman requires a *complete block design*: every method scored on every
dataset. Real suites violate this — ``SO2EquivariantFeatureMap`` accepts only
two features, so it has no result on the atlas's four 4-feature datasets.

Imputing a missing cell (with zero, or the mean) and ranking anyway is not a
neutral choice: it assigns worst rank for being *inapplicable* rather than for
performing badly. On the atlas data that single mistake moved SO(2)'s average
rank from 5.00 to 10.50 under VQC and from 4.38 to 10.19 under kernel-SVM —
from the top third to the bottom half — while leaving every other encoding
untouched.

So nothing is imputed here. Incomplete methods are excluded from the omnibus
and post-hoc, and named in :attr:`ComparisonResult.excluded` together with the
datasets they lacked, so the exclusion is reported rather than silent. Their
descriptive ranks remain available through :func:`average_ranks` with
``missing="available"`` — which must not be fed back into the post-hoc, because
a rank earned against 15 competitors is not on the same scale as one earned
against 11.

Critical values
---------------
:func:`studentized_range_quantile` computes ``q_alpha`` from the studentised
range distribution rather than reading a lookup table, so any ``k`` and any
``alpha`` work. It reproduces Demsar's Table 5 to within that table's own
rounding (max deviation 0.0007 over k = 2..20); the published table is used as
a test oracle rather than as the implementation.

References
----------
Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data
Sets. *Journal of Machine Learning Research*, 7, 1-30.

Iman, R. L., & Davenport, J. M. (1980). Approximations of the critical region
of the Friedman statistic. *Communications in Statistics*, 9(6), 571-595.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ComparisonResult",
    "FriedmanResult",
    "PostHocResult",
    "average_ranks",
    "bonferroni_dunn_quantile",
    "compare_over_datasets",
    "complete_cases",
    "critical_difference",
    "describe_exclusions",
    "friedman_test",
    "iman_davenport",
    "rank_matrix",
    "studentized_range_quantile",
]

#: How :func:`average_ranks` treats a method that lacks some datasets.
MissingPolicy = Literal["complete", "available", "error"]

#: Which post-hoc test :func:`compare_over_datasets` applies.
PostHocTest = Literal["nemenyi", "bonferroni-dunn"]

_SCORES = Mapping[str, Mapping[str, float]]


# ──────────────────────────────────────────────────────────────────────
# Layout: turning {method: {dataset: score}} into a complete rank matrix
# ──────────────────────────────────────────────────────────────────────


def complete_cases(
    scores: _SCORES,
) -> tuple[list[str], list[str], dict[str, tuple[str, ...]]]:
    """Split a score table into its complete block design and the leftovers.

    Parameters
    ----------
    scores : mapping
        ``{method_name: {dataset_name: score}}``. Scores must be finite; a
        method may omit datasets entirely, but a present entry must be a real
        number.

    Returns
    -------
    methods : list of str
        Methods scored on *every* dataset, sorted. These form the complete
        block design the Friedman test requires.
    datasets : list of str
        Union of all dataset names, sorted.
    excluded : dict
        ``{method_name: (datasets it lacked, ...)}`` for every method not in
        ``methods``. Empty when the design is already complete.

    Raises
    ------
    ValueError
        If ``scores`` is empty, or any present score is not finite.

    Examples
    --------
    >>> methods, datasets, excluded = complete_cases(
    ...     {"a": {"d1": 0.9, "d2": 0.8}, "b": {"d1": 0.7}}
    ... )
    >>> methods, datasets
    (['a'], ['d1', 'd2'])
    >>> excluded
    {'b': ('d2',)}
    """
    if not scores:
        raise ValueError("scores must not be empty")

    datasets = sorted({name for row in scores.values() for name in row})
    if not datasets:
        raise ValueError("no datasets found: every method's score table is empty")

    for method, row in scores.items():
        for dataset, value in row.items():
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"score for method {method!r} on dataset {dataset!r} is "
                    f"{value!r}; scores must be finite"
                )

    methods: list[str] = []
    excluded: dict[str, tuple[str, ...]] = {}
    for method in sorted(scores):
        missing = tuple(d for d in datasets if d not in scores[method])
        if missing:
            excluded[method] = missing
        else:
            methods.append(method)
    return methods, datasets, excluded


def rank_matrix(
    scores: _SCORES,
    methods: Sequence[str],
    datasets: Sequence[str],
    *,
    higher_is_better: bool = True,
) -> NDArray[np.float64]:
    """Rank ``methods`` within each dataset; 1 is best.

    Ties receive their average rank, so each row sums to ``k (k + 1) / 2``
    regardless of ties — the invariant the Friedman statistic relies on.

    Parameters
    ----------
    scores : mapping
        ``{method_name: {dataset_name: score}}``.
    methods, datasets : sequence of str
        The complete block design to rank over. Every ``methods[j]`` must have
        a score for every ``datasets[i]``.
    higher_is_better : bool, default=True
        Whether a larger score is a better one. Set ``False`` for error rates
        or losses.

    Returns
    -------
    ndarray of shape (n_datasets, n_methods)
        Row ``i`` holds the ranks of every method on ``datasets[i]``.

    Raises
    ------
    ValueError
        If a required score is missing.
    """
    from scipy.stats import rankdata

    matrix = np.empty((len(datasets), len(methods)), dtype=np.float64)
    for i, dataset in enumerate(datasets):
        row = []
        for method in methods:
            try:
                row.append(float(scores[method][dataset]))
            except KeyError as exc:
                raise ValueError(
                    f"method {method!r} has no score for dataset {dataset!r}; "
                    f"rank_matrix requires a complete block design (see "
                    f"complete_cases)"
                ) from exc
        signed = [-value for value in row] if higher_is_better else row
        matrix[i, :] = rankdata(signed)
    return matrix


def average_ranks(
    scores: _SCORES,
    *,
    higher_is_better: bool = True,
    missing: MissingPolicy = "complete",
) -> dict[str, float]:
    """Mean rank of each method across datasets; lower is better.

    Parameters
    ----------
    scores : mapping
        ``{method_name: {dataset_name: score}}``.
    higher_is_better : bool, default=True
        Whether a larger score is a better one.
    missing : {"complete", "available", "error"}, default="complete"
        What to do about a method that lacks some datasets.

        ``"complete"``
            Drop it. The returned ranks form a valid complete block design and
            are the ones the post-hoc tests consume.
        ``"available"``
            Rank it on the datasets it does have, against whichever other
            methods also have them. **Descriptive only.** A rank earned against
            fewer competitors is not comparable with one earned against more,
            so these values must not be fed to :func:`critical_difference` or
            compared with a critical difference computed for the full ``k``.
        ``"error"``
            Raise if the design is incomplete.

        Nothing is ever imputed: see the module docstring for what imputation
        did to the atlas's SO(2) ranks.

    Returns
    -------
    dict
        ``{method_name: average_rank}``, ordered best (lowest) first.

    Raises
    ------
    ValueError
        If ``missing`` is unknown, or is ``"error"`` and cells are absent, or
        ``"complete"`` leaves no method standing.

    Examples
    --------
    >>> ranks = average_ranks({"a": {"d1": 0.9, "d2": 0.8},
    ...                        "b": {"d1": 0.5, "d2": 0.6}})
    >>> ranks["a"], ranks["b"]
    (1.0, 2.0)
    """
    if missing not in ("complete", "available", "error"):
        raise ValueError(
            f"missing must be 'complete', 'available' or 'error', got {missing!r}"
        )

    methods, datasets, excluded = complete_cases(scores)

    if excluded and missing == "error":
        raise ValueError(
            f"incomplete block design: {describe_exclusions(excluded)}. "
            f"Pass missing='complete' to drop these methods, or "
            f"missing='available' for descriptive ranks."
        )

    if missing == "available":
        return _available_average_ranks(
            scores, datasets, higher_is_better=higher_is_better
        )

    if not methods:
        raise ValueError(
            f"no method has a score on every dataset, so there is no complete "
            f"block design to rank: {describe_exclusions(excluded)}"
        )

    matrix = rank_matrix(scores, methods, datasets, higher_is_better=higher_is_better)
    means = {method: float(matrix[:, j].mean()) for j, method in enumerate(methods)}
    return dict(sorted(means.items(), key=lambda kv: (kv[1], kv[0])))


def _available_average_ranks(
    scores: _SCORES,
    datasets: Sequence[str],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    """Per-dataset ranks over whichever methods have that dataset."""
    from scipy.stats import rankdata

    collected: dict[str, list[float]] = {method: [] for method in scores}
    for dataset in datasets:
        present = sorted(m for m in scores if dataset in scores[m])
        if not present:  # pragma: no cover - datasets come from the union of keys
            continue
        values = [float(scores[m][dataset]) for m in present]
        signed = [-v for v in values] if higher_is_better else values
        for method, rank in zip(present, rankdata(signed)):
            collected[method].append(float(rank))

    means = {m: float(np.mean(v)) for m, v in collected.items() if v}
    return dict(sorted(means.items(), key=lambda kv: (kv[1], kv[0])))


def describe_exclusions(excluded: Mapping[str, Sequence[str]]) -> str:
    """Render ``{method: missing datasets}`` as a readable clause."""
    if not excluded:
        return "none"
    parts = [
        f"{method} lacks {list(missing)}"
        for method, missing in sorted(excluded.items())
    ]
    return "; ".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Omnibus
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FriedmanResult:
    """Outcome of the Friedman omnibus test over a complete block design.

    Attributes
    ----------
    n_methods, n_datasets : int
        ``k`` and ``N``. Both statistics below depend on them.
    chi_square : float
        Friedman's statistic, asymptotically chi-square with ``k - 1``
        degrees of freedom. Known to be conservative, especially for small
        ``N``; reported for comparability with the literature.
    chi_square_p_value : float
        Its p-value.
    chi_square_dof : int
        ``k - 1``.
    f_statistic : float
        The Iman-Davenport correction,
        ``(N - 1) chi2 / (N (k - 1) - chi2)``, distributed F with
        ``(k - 1, (k - 1)(N - 1))`` degrees of freedom. Demsar recommends this
        over ``chi_square``, so it is what :attr:`rejected` uses. Infinite when
        the ranks agree perfectly across datasets, in which case the p-value is
        exactly 0.
    f_p_value : float
        Its p-value.
    f_dof : tuple of int
        ``(k - 1, (k - 1)(N - 1))``.
    alpha : float
        Significance level the ``rejected`` flags were evaluated at.
    """

    n_methods: int
    n_datasets: int
    chi_square: float
    chi_square_p_value: float
    chi_square_dof: int
    f_statistic: float
    f_p_value: float
    f_dof: tuple[int, int]
    alpha: float

    @property
    def rejected(self) -> bool:
        """Whether the Iman-Davenport F rejects "all methods are equivalent"."""
        return self.f_p_value < self.alpha

    @property
    def chi_square_rejected(self) -> bool:
        """Whether the (more conservative) chi-square form rejects."""
        return self.chi_square_p_value < self.alpha

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, including the derived flags."""
        return {
            "n_methods": self.n_methods,
            "n_datasets": self.n_datasets,
            "chi_square": self.chi_square,
            "chi_square_p_value": self.chi_square_p_value,
            "chi_square_dof": self.chi_square_dof,
            "chi_square_rejected": self.chi_square_rejected,
            "f_statistic": self.f_statistic,
            "f_p_value": self.f_p_value,
            "f_dof": list(self.f_dof),
            "alpha": self.alpha,
            "rejected": self.rejected,
        }


def _every_dataset_is_fully_tied(matrix: NDArray[np.float64]) -> bool:
    """Whether every dataset ranked all methods equal — the degenerate case."""
    if matrix.size == 0:  # pragma: no cover - callers guarantee a non-empty design
        return True
    return bool(np.all(matrix.max(axis=1) == matrix.min(axis=1)))


def iman_davenport(
    chi_square: float,
    n_methods: int,
    n_datasets: int,
) -> tuple[float, float]:
    """Convert a Friedman chi-square into the Iman-Davenport F and its p-value.

    Friedman's chi-square is undesirably conservative, so Iman & Davenport
    (1980) proposed

        F = (N - 1) chi2 / (N (k - 1) - chi2) ,

    distributed F with ``(k - 1, (k - 1)(N - 1))`` degrees of freedom. Demsar
    recommends it over the chi-square form, and it is what
    :attr:`FriedmanResult.rejected` uses.

    Parameters
    ----------
    chi_square : float
        Friedman's statistic. Must be non-negative and at most its maximum
        ``N (k - 1)``.
    n_methods : int
        ``k >= 2``.
    n_datasets : int
        ``N >= 2``.

    Returns
    -------
    f_statistic : float
        Infinite when ``chi_square`` attains its maximum ``N (k - 1)``, i.e.
        the methods rank identically on every dataset.
    p_value : float
        Exactly 0 in that degenerate case, which is a decisive rejection
        rather than an error.

    Raises
    ------
    ValueError
        If ``chi_square`` is negative or exceeds ``N (k - 1)``, or the counts
        are too small.

    Examples
    --------
    Demsar (2006) Section 3.2.2 reports ``chi2 = 9.28`` for 4 algorithms over
    14 datasets, and an F of 3.69:

    >>> f_statistic, _ = iman_davenport(9.28, 4, 14)
    >>> round(f_statistic, 2)
    3.69
    """
    from scipy.stats import f as f_dist

    if n_methods < 2:
        raise ValueError(f"n_methods must be at least 2, got {n_methods}")
    if n_datasets < 2:
        raise ValueError(f"n_datasets must be at least 2, got {n_datasets}")
    maximum = float(n_datasets * (n_methods - 1))
    if chi_square < 0.0:
        raise ValueError(f"chi_square must be non-negative, got {chi_square!r}")
    if chi_square > maximum:
        raise ValueError(
            f"chi_square {chi_square!r} exceeds its maximum N(k-1) = {maximum} "
            f"for k={n_methods}, N={n_datasets}"
        )

    denominator = maximum - chi_square
    if denominator <= 0.0:
        return math.inf, 0.0
    f_statistic = ((n_datasets - 1) * chi_square) / denominator
    p_value = float(
        f_dist.sf(f_statistic, n_methods - 1, (n_methods - 1) * (n_datasets - 1))
    )
    return f_statistic, p_value


def friedman_test(
    scores: _SCORES,
    *,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> FriedmanResult:
    """Test whether any method's average rank differs from the others'.

    Only the complete block design is used; methods lacking a dataset are
    dropped, because the statistic is undefined for unbalanced blocks. Use
    :func:`complete_cases` first if you need to know which those were.

    Parameters
    ----------
    scores : mapping
        ``{method_name: {dataset_name: score}}``. At least 3 complete methods
        and 2 datasets are required.
    alpha : float, default=0.05
        Significance level for the ``rejected`` properties.
    higher_is_better : bool, default=True
        Whether a larger score is a better one.

    Returns
    -------
    FriedmanResult

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``(0, 1)``, or fewer than 3 complete methods or
        2 datasets remain.

    Notes
    -----
    With ``k`` methods and ``N`` datasets the test has little power unless
    ``N`` is comfortably larger than ``k``; a non-rejection is evidence of an
    underpowered suite at least as much as of genuinely equivalent methods.
    """
    from scipy.stats import friedmanchisquare

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    methods, datasets, excluded = complete_cases(scores)
    n_methods, n_datasets = len(methods), len(datasets)

    if n_methods < 3:
        raise ValueError(
            f"the Friedman test needs at least 3 methods scored on every "
            f"dataset, got {n_methods} ({describe_exclusions(excluded)})"
        )
    if n_datasets < 2:
        raise ValueError(
            f"the Friedman test needs at least 2 datasets, got {n_datasets}"
        )

    matrix = rank_matrix(scores, methods, datasets, higher_is_better=higher_is_better)
    if _every_dataset_is_fully_tied(matrix):
        # scipy's tie correction divides by zero here and returns NaN. The
        # situation is not an error: every method scored identically on every
        # dataset, which is exactly zero evidence that they differ.
        chi_square, chi_p = 0.0, 1.0
    else:
        statistic, p_value = friedmanchisquare(
            *(matrix[:, j] for j in range(n_methods))
        )
        chi_square, chi_p = float(statistic), float(p_value)
        if not (  # pragma: no cover - the only known NaN source is handled above
            math.isfinite(chi_square) and math.isfinite(chi_p)
        ):
            raise ValueError(
                f"the Friedman statistic is not finite (chi2={chi_square}, "
                f"p={chi_p}) for {n_methods} methods over {n_datasets} "
                f"datasets; this usually means the ranks are degenerate"
            )

    f_statistic, f_p = iman_davenport(chi_square, n_methods, n_datasets)

    return FriedmanResult(
        n_methods=n_methods,
        n_datasets=n_datasets,
        chi_square=chi_square,
        chi_square_p_value=chi_p,
        chi_square_dof=n_methods - 1,
        f_statistic=f_statistic,
        f_p_value=f_p,
        f_dof=(n_methods - 1, (n_methods - 1) * (n_datasets - 1)),
        alpha=alpha,
    )


# ──────────────────────────────────────────────────────────────────────
# Critical values and post-hoc tests
# ──────────────────────────────────────────────────────────────────────


def studentized_range_quantile(n_methods: int, alpha: float = 0.05) -> float:
    """Nemenyi's ``q_alpha``: the studentised range quantile over ``sqrt(2)``.

    Computed from :class:`scipy.stats.studentized_range` with infinite degrees
    of freedom, rather than read from a table, so any ``k`` and ``alpha`` work.
    It agrees with Demsar (2006) Table 5 to within that table's rounding — at
    most 0.0007 across ``k = 2..20`` — which the test suite pins.

    Parameters
    ----------
    n_methods : int
        Number of methods being compared, ``k >= 2``.
    alpha : float, default=0.05
        Two-sided family-wise significance level.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``n_methods < 2``, ``alpha`` is outside ``(0, 1)``, or scipy returns
        a non-finite quantile.
    """
    from scipy.stats import studentized_range

    if n_methods < 2:
        raise ValueError(f"n_methods must be at least 2, got {n_methods}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    quantile = float(studentized_range.ppf(1.0 - alpha, n_methods, np.inf))
    value = quantile / math.sqrt(2.0)
    if not math.isfinite(value):  # pragma: no cover - scipy is finite for k >= 2
        raise ValueError(
            f"scipy returned a non-finite studentised-range quantile for "
            f"k={n_methods}, alpha={alpha}"
        )
    return value


def bonferroni_dunn_quantile(n_methods: int, alpha: float = 0.05) -> float:
    """Bonferroni-Dunn's critical value for comparing ``k - 1`` methods to one.

    The ``k - 1`` comparisons against a single control are Bonferroni-corrected,
    giving the two-sided normal quantile at ``alpha / (k - 1)``.

    Because only ``k - 1`` comparisons are made rather than Nemenyi's
    ``k (k - 1) / 2``, this critical value is smaller — the test is more
    powerful, at the cost of answering only "which methods differ from the
    control?" rather than "which pairs differ?". The control must be chosen
    before looking at the ranks.

    Parameters
    ----------
    n_methods : int
        Total number of methods including the control, ``k >= 2``.
    alpha : float, default=0.05
        Two-sided family-wise significance level.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``n_methods < 2`` or ``alpha`` is outside ``(0, 1)``.

    Notes
    -----
    Reproduces Demsar (2006) Table 6 except at ``k = 9``, where the published
    2.724 does not follow from ``alpha / (k - 1) = 0.00625``; the value is
    2.7344, and the surrounding entries (2.690 at ``k = 8``, 2.773 at
    ``k = 10``) confirm the table entry is a typo rather than a different
    convention.
    """
    from scipy.stats import norm

    if n_methods < 2:
        raise ValueError(f"n_methods must be at least 2, got {n_methods}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    adjusted = alpha / (n_methods - 1)
    return float(norm.ppf(1.0 - adjusted / 2.0))


def critical_difference(
    n_methods: int,
    n_datasets: int,
    *,
    alpha: float = 0.05,
    test: PostHocTest = "nemenyi",
) -> float:
    """Smallest average-rank gap that counts as significant.

    ``CD = q_alpha * sqrt(k (k + 1) / (6 N))``, with ``q_alpha`` from
    :func:`studentized_range_quantile` (Nemenyi) or
    :func:`bonferroni_dunn_quantile` (Bonferroni-Dunn).

    Parameters
    ----------
    n_methods : int
        ``k >= 2``.
    n_datasets : int
        ``N >= 1``.
    alpha : float, default=0.05
        Two-sided family-wise significance level.
    test : {"nemenyi", "bonferroni-dunn"}, default="nemenyi"
        Which family of comparisons the critical value must cover.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``n_datasets < 1`` or ``test`` is unknown.

    Examples
    --------
    The atlas's own configuration — 15 encodings over 8 datasets — leaves a
    critical difference of more than half the available rank range:

    >>> round(critical_difference(15, 8), 3)
    7.583
    """
    if n_datasets < 1:
        raise ValueError(f"n_datasets must be at least 1, got {n_datasets}")
    if test == "nemenyi":
        q_alpha = studentized_range_quantile(n_methods, alpha)
    elif test == "bonferroni-dunn":
        q_alpha = bonferroni_dunn_quantile(n_methods, alpha)
    else:
        raise ValueError(f"test must be 'nemenyi' or 'bonferroni-dunn', got {test!r}")
    return q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))


@dataclass(frozen=True)
class PostHocResult:
    """Which methods a post-hoc test could actually separate.

    Attributes
    ----------
    test : {"nemenyi", "bonferroni-dunn"}
        Which test produced this.
    alpha : float
        Family-wise significance level.
    q_alpha : float
        Critical value from the studentised range (Nemenyi) or normal
        (Bonferroni-Dunn) distribution.
    critical_difference : float
        The rank gap two methods must exceed to be called different.
    average_ranks : dict
        ``{method: average_rank}``, best (lowest) first.
    control : str or None
        The reference method for Bonferroni-Dunn; ``None`` for Nemenyi.
    significant_pairs : tuple
        ``(better, worse, rank_difference)`` for every separated pair, widest
        gap first. For Bonferroni-Dunn only pairs involving ``control`` appear.
    rank_range : float
        Gap between the best and worst average rank. Compare it against
        ``critical_difference``: when the two are close, the suite cannot
        separate much regardless of the result, and saying so is part of
        reporting the outcome honestly.
    """

    test: PostHocTest
    alpha: float
    q_alpha: float
    critical_difference: float
    average_ranks: dict[str, float]
    control: str | None
    significant_pairs: tuple[tuple[str, str, float], ...]
    rank_range: float

    @property
    def n_significant(self) -> int:
        """How many pairs were separated."""
        return len(self.significant_pairs)

    def separated_from(self, method: str) -> tuple[str, ...]:
        """Methods this test could tell apart from ``method``, either way."""
        out = []
        for better, worse, _ in self.significant_pairs:
            if better == method:
                out.append(worse)
            elif worse == method:
                out.append(better)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "test": self.test,
            "alpha": self.alpha,
            "q_alpha": self.q_alpha,
            "critical_difference": self.critical_difference,
            "average_ranks": dict(self.average_ranks),
            "control": self.control,
            "significant_pairs": [list(p) for p in self.significant_pairs],
            "rank_range": self.rank_range,
            "n_significant": self.n_significant,
        }


def _posthoc(
    ranks: Mapping[str, float],
    n_datasets: int,
    *,
    alpha: float,
    test: PostHocTest,
    control: str | None,
) -> PostHocResult:
    """Apply a post-hoc test to already-computed average ranks."""
    n_methods = len(ranks)
    if test == "nemenyi":
        q_alpha = studentized_range_quantile(n_methods, alpha)
    else:
        q_alpha = bonferroni_dunn_quantile(n_methods, alpha)
    cd = q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))

    if test == "bonferroni-dunn":
        if control is None:  # pragma: no cover - validated by compare_over_datasets
            raise ValueError("bonferroni-dunn requires a control method")
        if control not in ranks:  # pragma: no cover - validated by the caller
            raise ValueError(
                f"control {control!r} is not among the compared methods "
                f"{sorted(ranks)}"
            )
        candidate_pairs = [(control, other) for other in ranks if other != control]
    else:
        names = list(ranks)
        candidate_pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    significant: list[tuple[str, str, float]] = []
    for a, b in candidate_pairs:
        gap = ranks[a] - ranks[b]
        if abs(gap) >= cd:
            better, worse = (b, a) if gap > 0 else (a, b)
            significant.append((better, worse, abs(gap)))
    significant.sort(key=lambda item: (-item[2], item[0], item[1]))

    values = list(ranks.values())
    return PostHocResult(
        test=test,
        alpha=alpha,
        q_alpha=q_alpha,
        critical_difference=cd,
        average_ranks=dict(ranks),
        control=control,
        significant_pairs=tuple(significant),
        rank_range=float(max(values) - min(values)) if values else 0.0,
    )


# ──────────────────────────────────────────────────────────────────────
# The full procedure
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComparisonResult:
    """Complete Demsar comparison of several methods over several datasets.

    Attributes
    ----------
    methods : tuple of str
        Methods that formed the complete block design, best-ranked first.
    datasets : tuple of str
        Datasets they were compared over, sorted.
    average_ranks : dict
        ``{method: average_rank}``, best (lowest) first. Complete cases only.
    friedman : FriedmanResult
        The omnibus test.
    posthoc : PostHocResult or None
        ``None`` when the omnibus did not reject — in which case no post-hoc
        was run, because running one anyway inflates the error rate.
    excluded : dict
        ``{method: (datasets it lacked, ...)}`` for methods left out of the
        analysis. Never silently dropped; report these alongside any ranking.
    higher_is_better : bool
        Whether larger scores were treated as better.
    """

    methods: tuple[str, ...]
    datasets: tuple[str, ...]
    average_ranks: dict[str, float]
    friedman: FriedmanResult
    posthoc: PostHocResult | None
    excluded: dict[str, tuple[str, ...]]
    higher_is_better: bool

    @property
    def best(self) -> str:
        """The method with the lowest average rank."""
        return next(iter(self.average_ranks))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the whole comparison."""
        return {
            "methods": list(self.methods),
            "datasets": list(self.datasets),
            "average_ranks": dict(self.average_ranks),
            "friedman": self.friedman.to_dict(),
            "posthoc": self.posthoc.to_dict() if self.posthoc else None,
            "excluded": {k: list(v) for k, v in self.excluded.items()},
            "higher_is_better": self.higher_is_better,
            "best": self.best,
        }

    def summary(self) -> str:
        """A short human-readable report, safe to paste into a paper draft."""
        friedman = self.friedman
        lines = [
            f"{friedman.n_methods} methods over {friedman.n_datasets} datasets",
            f"  Friedman chi2({friedman.chi_square_dof}) = "
            f"{friedman.chi_square:.3f}, p = {friedman.chi_square_p_value:.3g}",
            f"  Iman-Davenport F{friedman.f_dof} = {friedman.f_statistic:.3f}, "
            f"p = {friedman.f_p_value:.3g}",
        ]
        if self.posthoc is None:
            lines.append(
                f"  omnibus does NOT reject at alpha = {friedman.alpha}: no "
                f"post-hoc test was run, and no pair may be called different."
            )
            return "\n".join(lines)

        posthoc = self.posthoc
        lines.append(
            f"  omnibus rejects at alpha = {friedman.alpha}; {posthoc.test} "
            f"CD = {posthoc.critical_difference:.3f} over a rank range of "
            f"{posthoc.rank_range:.2f}"
        )
        lines.append(f"  best: {self.best} (rank {self.average_ranks[self.best]:.2f})")
        beaten = posthoc.separated_from(self.best)
        lines.append(
            f"  separated from best: {len(beaten)}/{friedman.n_methods - 1}"
            + (f" -> {list(beaten)}" if beaten else "")
        )
        if self.excluded:
            lines.append(
                f"  excluded (incomplete): " f"{describe_exclusions(self.excluded)}"
            )
        return "\n".join(lines)


def compare_over_datasets(
    scores: _SCORES,
    *,
    alpha: float = 0.05,
    higher_is_better: bool = True,
    test: PostHocTest = "nemenyi",
    control: str | None = None,
) -> ComparisonResult:
    """Run the full Demsar procedure: rank, omnibus, then post-hoc if licensed.

    Parameters
    ----------
    scores : mapping
        ``{method_name: {dataset_name: score}}``. Methods that lack a dataset
        are excluded from the analysis and reported in
        :attr:`ComparisonResult.excluded`; nothing is imputed.
    alpha : float, default=0.05
        Family-wise significance level, used by both the omnibus and post-hoc.
    higher_is_better : bool, default=True
        Whether a larger score is a better one.
    test : {"nemenyi", "bonferroni-dunn"}, default="nemenyi"
        All-pairs, or every method against one control.
    control : str, optional
        Required for ``"bonferroni-dunn"``, and must be chosen before looking
        at the ranks for the correction to be valid. Nemenyi does not use it,
        but an unknown name is still rejected rather than ignored, so a typo
        cannot pass silently.

    Returns
    -------
    ComparisonResult
        With ``posthoc=None`` if the omnibus failed to reject.

    Raises
    ------
    ValueError
        If the inputs cannot form a complete block design of at least 3
        methods and 2 datasets, or ``control`` is missing or unknown.

    Examples
    --------
    >>> scores = {
    ...     "good":   {"d1": 0.95, "d2": 0.93, "d3": 0.97, "d4": 0.94},
    ...     "middle": {"d1": 0.85, "d2": 0.83, "d3": 0.87, "d4": 0.84},
    ...     "poor":   {"d1": 0.60, "d2": 0.62, "d3": 0.58, "d4": 0.61},
    ... }
    >>> result = compare_over_datasets(scores)
    >>> result.best
    'good'
    >>> result.friedman.rejected
    True
    """
    if test not in ("nemenyi", "bonferroni-dunn"):
        raise ValueError(f"test must be 'nemenyi' or 'bonferroni-dunn', got {test!r}")
    if test == "bonferroni-dunn" and control is None:
        raise ValueError(
            "bonferroni-dunn compares every method against one control; pass "
            "control=<method name>"
        )

    methods, datasets, excluded = complete_cases(scores)
    friedman = friedman_test(scores, alpha=alpha, higher_is_better=higher_is_better)
    ranks = average_ranks(scores, higher_is_better=higher_is_better, missing="complete")

    if control is not None and control not in ranks:
        raise ValueError(
            f"control {control!r} is not part of the complete block design "
            f"{sorted(ranks)}"
            + (f" ({describe_exclusions(excluded)})" if excluded else "")
        )

    posthoc = None
    if friedman.rejected:
        posthoc = _posthoc(
            ranks,
            friedman.n_datasets,
            alpha=alpha,
            test=test,
            control=control if test == "bonferroni-dunn" else None,
        )

    return ComparisonResult(
        methods=tuple(ranks),
        datasets=tuple(datasets),
        average_ranks=ranks,
        friedman=friedman,
        posthoc=posthoc,
        excluded=excluded,
        higher_is_better=higher_is_better,
    )
