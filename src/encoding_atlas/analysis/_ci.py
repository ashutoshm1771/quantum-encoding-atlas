"""Bootstrap confidence-interval helpers shared by the analysis pipeline.

The three core analysis functions (``compute_expressibility``,
``compute_entanglement_capability``, ``estimate_trainability``) all
return a single scalar summary of a sampled distribution. For
research use those scalars need an uncertainty envelope — otherwise
``expressibility = 0.042`` is unfalsifiable in a paper.

This module supplies a small, deterministic, well-tested percentile
bootstrap implementation that every analysis function uses to attach
a confidence interval to its result. Determinism is the key
property: for a fixed ``rng`` the bootstrap CI is byte-identical
across runs, so users can quote CIs in publications and reviewers
can reproduce them exactly.

Design notes
------------
* **Percentile bootstrap** (Efron 1979) rather than BCa: simpler,
  identical asymptotically for our smooth statistics
  (mean / variance / KL), and easier to verify in tests. BCa would
  pull in additional jackknife logic for marginal accuracy gains.
* **Pre-sampled indices**: ``rng.integers`` is called once for the
  full ``(n_bootstrap, n_samples)`` index matrix. This is faster
  than per-iteration draws and — crucially — produces an identical
  index sequence regardless of how the loop is structured, so
  changes to the inner statistic don't perturb the random state.
* **Validation upfront**: bad ``confidence_level`` (outside (0, 1))
  or non-positive ``n_bootstrap`` raise ``ValueError`` before any
  bootstrap work happens.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def validate_ci_args(
    confidence_level: float,
    n_bootstrap: int,
) -> None:
    """Validate the shared CI arguments. Raises a clean ``ValueError``.

    Parameters
    ----------
    confidence_level : float
        Must lie strictly in the open interval (0, 1). Common
        choices are 0.90, 0.95 (default for almost all analysis
        functions), and 0.99.
    n_bootstrap : int
        Number of bootstrap resamples; must be a positive integer.
        Larger values give smoother CIs at proportional CPU cost.

    Raises
    ------
    ValueError
        With a message that names exactly which argument was bad.
    """
    if not isinstance(confidence_level, (int, float)) or not (
        0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError(
            f"confidence_level must be a number in (0, 1), " f"got {confidence_level!r}"
        )
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be a positive integer, got {n_bootstrap!r}")


def percentile_bootstrap_ci(
    samples: NDArray[np.floating[Any]],
    statistic_fn: Callable[[NDArray[np.floating[Any]]], float],
    rng: np.random.Generator,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a scalar statistic.

    Resamples ``samples`` with replacement ``n_bootstrap`` times,
    applies ``statistic_fn`` to each resample, and returns the
    ``(lower, upper)`` percentile bounds at the requested
    ``confidence_level``.

    Parameters
    ----------
    samples : NDArray[np.floating]
        1-D sample array (length n). Per-sample statistics of the
        analysis (fidelities, entanglement values, etc.).
    statistic_fn : callable
        Receives a resampled view of ``samples`` and returns a
        ``float``. Common choices: ``np.mean``, ``np.var``, a
        composite (e.g. variance-to-trainability mapping).
    rng : numpy.random.Generator
        Random source for the resampling. Caller is responsible for
        seeding for determinism.
    n_bootstrap : int, default=200
        Number of bootstrap resamples. 200 keeps the cost negligible
        next to the simulation cost and gives ~1% jitter on the
        percentile endpoints.
    confidence_level : float, default=0.95
        Two-sided confidence level in (0, 1).

    Returns
    -------
    (lower, upper) : tuple of float
        Bounds of the bootstrap CI. If ``samples`` has length < 2 or
        every resample yields the same value (degenerate distribution),
        both bounds equal ``statistic_fn(samples)``.

    Raises
    ------
    ValueError
        If ``confidence_level`` is out of range or ``n_bootstrap``
        is not a positive integer.

    Notes
    -----
    The percentile bootstrap is consistent for smooth statistics and
    matches the bias-corrected ``BCa`` method asymptotically. We
    deliberately use it (rather than ``BCa``) here because it is
    simpler, has fewer numerical edge cases, and is easier to verify
    in tests — and because the analysis statistics it operates on
    (mean of a bounded value, variance) satisfy the smoothness
    assumption.
    """
    validate_ci_args(confidence_level, n_bootstrap)

    samples = np.asarray(samples)
    n = samples.shape[0]

    # Degenerate input — no resampling is meaningful. Return the point
    # estimate as both bounds so the public surface remains uniform.
    if n < 2:
        point = float(statistic_fn(samples)) if n == 1 else float("nan")
        return point, point

    # One vectorized index draw covers all resamples; downstream changes
    # to ``statistic_fn`` cannot perturb the random sequence.
    indices = rng.integers(0, n, size=(n_bootstrap, n))

    bootstrap_stats: NDArray[np.float64] = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        bootstrap_stats[b] = float(statistic_fn(samples[indices[b]]))

    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(bootstrap_stats, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(bootstrap_stats, 100.0 * (1.0 - alpha / 2.0)))

    # Numerical guard: if the statistic is constant across resamples the
    # two percentiles will collapse; clip any tiny negative gap to 0 so
    # downstream code can rely on ``lower <= upper``.
    if upper < lower:
        upper = lower
    return lower, upper
