"""Feature-scaling sensitivity analysis.

Encodings turn numbers into rotation angles, so the numeric range you scale
your features into is not a preprocessing detail — it decides how much of each
qubit's Bloch circle the data sweeps, and therefore the geometry of the kernel
the model actually sees.

Rotation gates have period ``2*pi``. Scaling into ``[0, 2*pi]`` drives every
feature over a full period, which is the regime
:mod:`encoding_atlas.analysis.concentration` identifies as maximally
Haar-like: pairwise fidelities collapse towards the ``1/2**n`` floor and the
kernel approaches the identity. Narrower ranges keep the encoded states in a
cone, preserving the spread a kernel method needs. The effect is large, and it
is largest for exactly the circuits that scramble fastest.

What this module measures
-------------------------
:func:`scan_feature_ranges` sweeps an encoding across candidate ranges and
reports, for each, the kernel-target alignment and the kernel concentration on
the caller's data. Alignment is training-free and predicts downstream accuracy
(Spearman rho = 0.91 in the benchmark), so it doubles as a cheap criterion for
*choosing* a range — which is what :func:`recommend_feature_range` returns.

Why it matters beyond tuning
----------------------------
The benchmark's own pipeline scales every dataset to ``[0, 2*pi]``
(``experiments/datasets.py``), and its headline negative result — that
expressibility is *anti*-correlated with accuracy — is measured under that
choice. The bundled scaling-sensitivity scan (see
:mod:`encoding_atlas.atlas.scaling`) records how that correlation moves as the
range changes. Treat the range as a variable to report, not a constant to
assume.

Reading ``concentration_ratio`` here
------------------------------------
The ratio is an off-diagonal *variance* relative to the Haar variance, and it
is **not** monotone in the range width. A very narrow range leaves every kernel
entry close to 1 — degenerate in its own way, but with tiny variance, so a
*low* ratio. Only a ratio near 1 **together with** an off-diagonal mean near
``1/2**n`` means the Haar floor. Read
:attr:`FeatureRangeResult.offdiagonal_mean` alongside it; that quantity does
move monotonically towards the floor as the range widens, and
:attr:`FeatureRangeResult.alignment` falls with it.

Notes
-----
Scaling here is per-feature min-max, matching the benchmark and
:class:`sklearn.preprocessing.MinMaxScaler`. This is *not* the same as
:func:`encoding_atlas.utils.preprocessing.scale_features`, which rescales using
one global minimum and maximum across all features.

References
----------
Thanasilp, Wang, Cerezo & Holmes (2024), *Nat. Commun.* 15:5200 — exponential
concentration in quantum kernel methods.
Cortes, Mohri & Rostamizadeh (2012), *JMLR* 13:795 — centered alignment.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FEATURE_RANGES",
    "FeatureRangeResult",
    "FeatureRangeScan",
    "recommend_feature_range",
    "scale_to_range",
    "scan_feature_ranges",
]

# Candidate ranges, as fractions of the 2*pi rotation period. A quarter, a
# half, a full half-turn, and the whole period: enough to expose the trend
# without making the scan expensive. ``[0, 2*pi]`` is last because it is the
# benchmark's own default, and the one the scan exists to question.
DEFAULT_FEATURE_RANGES: tuple[tuple[float, float], ...] = (
    (0.0, math.pi / 4),
    (0.0, math.pi / 2),
    (0.0, math.pi),
    (0.0, 2.0 * math.pi),
)

# Sub-sample above this many rows; alignment is O(n^2) in kernel entries.
_DEFAULT_MAX_SAMPLES: int = 200

# Span below this is treated as a constant feature (mapped to the range floor).
_CONSTANT_FEATURE_EPS: float = 1e-12


def scale_to_range(
    X: NDArray[np.floating[Any]],
    low: float,
    high: float,
    *,
    reference: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.float64]:
    """Min-max scale each feature of ``X`` into ``[low, high]``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data to transform.
    low, high : float
        Target interval. Must satisfy ``low < high`` and both be finite.
    reference : ndarray of shape (m, n_features), optional
        Fit the per-feature minimum and span on this array instead of on
        ``X``. Pass the training fold here to transform a test fold without
        leaking its extremes into the scaler — transformed values may then
        fall outside ``[low, high]``, which is correct and matches
        :class:`sklearn.preprocessing.MinMaxScaler`. Defaults to ``X``.

    Returns
    -------
    ndarray of shape (n_samples, n_features)
        The scaled data. Constant features map to ``low``.

    Raises
    ------
    ValueError
        If ``X`` is not 2D, the range is invalid or non-finite, or
        ``reference`` has a different feature count.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0, 10.0], [1.0, 20.0]])
    >>> np.round(scale_to_range(X, 0.0, np.pi), 4)
    array([[0.    , 0.    ],
           [3.1416, 3.1416]])
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"range must be finite, got ({low}, {high})")
    if low >= high:
        raise ValueError(f"range must satisfy low < high, got ({low}, {high})")

    X_array = np.asarray(X, dtype=np.float64)
    if X_array.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_array.shape}")

    source = X_array if reference is None else np.asarray(reference, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"reference must be a 2D array, got shape {source.shape}")
    if source.shape[1] != X_array.shape[1]:
        raise ValueError(
            f"reference has {source.shape[1]} features but X has " f"{X_array.shape[1]}"
        )
    if source.shape[0] == 0:
        raise ValueError("reference must contain at least one row")

    mins = source.min(axis=0)
    span = source.max(axis=0) - mins
    span = np.where(span <= _CONSTANT_FEATURE_EPS, 1.0, span)
    scaled: NDArray[np.float64] = low + (X_array - mins) / span * (high - low)
    # A constant feature has no information; pin it to the range floor rather
    # than letting the placeholder span produce an arbitrary offset.
    constant = (source.max(axis=0) - mins) <= _CONSTANT_FEATURE_EPS
    if np.any(constant):
        scaled[:, constant] = low
    return scaled


@dataclass(frozen=True)
class FeatureRangeResult:
    """An encoding measured at one feature-scaling range.

    Attributes
    ----------
    low, high : float
        The range the features were scaled into.
    width : float
        ``high - low``, as a convenience for plotting against period coverage.
    alignment : float
        Kernel-target alignment at this range, in ``[-1, 1]``. Higher predicts
        better downstream accuracy.
    concentration_ratio : float
        Off-diagonal kernel variance over the Haar variance. Near 1 *and* with
        ``offdiagonal_mean`` near ``1/2**n_qubits`` means the kernel has
        collapsed onto the Haar floor. Not monotone in the range width on its
        own — see the module docstring.
    offdiagonal_mean, offdiagonal_variance : float
        Raw moments of the kernel's off-diagonal entries. The mean is the
        cleaner read on how close the kernel is to the floor: it decreases
        towards ``1/2**n_qubits`` as the range widens.
    """

    low: float
    high: float
    width: float
    alignment: float
    concentration_ratio: float
    offdiagonal_mean: float
    offdiagonal_variance: float


@dataclass(frozen=True)
class FeatureRangeScan:
    """How an encoding's kernel geometry responds to feature scaling.

    Attributes
    ----------
    encoding_name : str
        Class name of the scanned encoding.
    n_qubits, n_features : int
        Circuit width and input dimensionality.
    n_samples_used, n_samples_supplied : int
        Rows used after sub-sampling, and rows supplied.
    centered : bool
        Whether the centred alignment variant was used.
    results : tuple[FeatureRangeResult, ...]
        One measurement per range, in the order scanned.
    """

    encoding_name: str
    n_qubits: int
    n_features: int
    n_samples_used: int
    n_samples_supplied: int
    centered: bool
    results: tuple[FeatureRangeResult, ...]

    @property
    def best(self) -> FeatureRangeResult:
        """The range with the highest alignment.

        Ties break towards the narrower range, which is the more conservative
        choice: it keeps the kernel further from the concentration floor.

        Raises
        ------
        RuntimeError
            If the scan produced no measurements.
        """
        if not self.results:
            raise RuntimeError("scan produced no measurements")
        return max(self.results, key=lambda r: (r.alignment, -r.width))

    @property
    def best_range(self) -> tuple[float, float]:
        """``(low, high)`` of the highest-aligned range."""
        return (self.best.low, self.best.high)

    @property
    def alignment_spread(self) -> float:
        """Highest minus lowest alignment across the scanned ranges.

        A large spread means the scaling choice matters as much as, or more
        than, the choice of encoding — so it should be reported, not assumed.
        """
        if not self.results:
            return 0.0
        alignments = [r.alignment for r in self.results]
        return float(max(alignments) - min(alignments))

    def at(self, low: float, high: float) -> FeatureRangeResult | None:
        """Return the measurement for a given range, or ``None``."""
        for result in self.results:
            if math.isclose(result.low, low) and math.isclose(result.high, high):
                return result
        return None


def _subsample(n_samples: int, max_samples: int, seed: int | None) -> NDArray[np.intp]:
    """Return a deterministic row subset, or all rows when small enough."""
    if n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.intp)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)[:max_samples].astype(np.intp)
    indices.sort()
    return indices


def _validate_ranges(
    ranges: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Validate and normalise the requested ranges."""
    resolved = tuple((float(low), float(high)) for low, high in ranges)
    if not resolved:
        raise ValueError("ranges must not be empty")
    for low, high in resolved:
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"range must be finite, got ({low}, {high})")
        if low >= high:
            raise ValueError(f"range must satisfy low < high, got ({low}, {high})")
    return resolved


def scan_feature_ranges(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
    *,
    ranges: Sequence[tuple[float, float]] = DEFAULT_FEATURE_RANGES,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    centered: bool = True,
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> FeatureRangeScan:
    """Measure an encoding's kernel geometry across feature-scaling ranges.

    For each candidate range the data is min-max scaled into it, the fidelity
    kernel is built, and both the kernel-target alignment and the kernel
    concentration are recorded. No training is involved.

    Use it to choose a range (see :func:`recommend_feature_range`) and to see
    how much the choice matters — :attr:`FeatureRangeScan.alignment_spread`
    is often comparable to the spread across *encodings*.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding to scan. Built once and reused across ranges.
    X : ndarray of shape (n_samples, n_features)
        Unscaled feature matrix. Whatever scaling it already carries is
        overwritten, so pass raw features.
    y : ndarray of shape (n_samples,)
        Two-class labels. Any two distinct values work.
    ranges : sequence of (float, float), default=:data:`DEFAULT_FEATURE_RANGES`
        Candidate ranges. Each must satisfy ``low < high``.
    max_samples : int, default=200
        Sub-sample above this many rows. Must be at least 2.
    centered : bool, default=True
        Use the centred alignment (Cortes et al., 2012).
    seed : int or None, default=None
        Seed for sub-sampling.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector simulation backend.

    Returns
    -------
    FeatureRangeScan
        One measurement per range, plus the best range and the spread.

    Raises
    ------
    ValueError
        If ``X``/``y`` are malformed or mismatched, ``y`` is not two-class,
        ``max_samples`` is invalid, or any range is invalid.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.benchmark import get_dataset
    >>> X, y = get_dataset("moons", n_samples=40, seed=0)
    >>> scan = scan_feature_ranges(IQPEncoding(n_features=2, reps=2), X, y, seed=0)
    >>> len(scan.results)
    4

    See Also
    --------
    recommend_feature_range : Just the best range.
    encoding_atlas.analysis.compute_kernel_concentration : The concentration axis.
    """
    from encoding_atlas.analysis.concentration import summarize_kernel_concentration
    from encoding_atlas.analysis.generalization import (
        centered_kernel_target_alignment,
        compute_fidelity_kernel,
        kernel_target_alignment,
        validate_binary_labels,
    )

    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise ValueError(f"max_samples must be an integer >= 2, got {max_samples!r}")
    if max_samples < 2:
        raise ValueError(f"max_samples must be an integer >= 2, got {max_samples!r}")

    resolved_ranges = _validate_ranges(ranges)

    X_array = np.asarray(X, dtype=np.float64)
    if X_array.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_array.shape}")
    if X_array.shape[0] < 2:
        raise ValueError(f"X must contain at least 2 samples, got {X_array.shape[0]}")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X contains NaN or infinite values")

    y_float = validate_binary_labels(y)
    if y_float.shape[0] != X_array.shape[0]:
        raise ValueError(
            f"X and y must have the same length, got {X_array.shape[0]} and "
            f"{y_float.shape[0]}"
        )

    n_supplied = X_array.shape[0]
    indices = _subsample(n_supplied, max_samples, seed)
    X_used, y_used = X_array[indices], y_float[indices]

    score = centered_kernel_target_alignment if centered else kernel_target_alignment
    n_qubits = int(encoding.n_qubits)

    results: list[FeatureRangeResult] = []
    for low, high in resolved_ranges:
        K = compute_fidelity_kernel(
            encoding, scale_to_range(X_used, low, high), backend=backend
        )
        summary = summarize_kernel_concentration(K, n_qubits)
        results.append(
            FeatureRangeResult(
                low=low,
                high=high,
                width=high - low,
                alignment=float(score(K, y_used)),
                concentration_ratio=summary.concentration_ratio,
                offdiagonal_mean=summary.offdiagonal_mean,
                offdiagonal_variance=summary.offdiagonal_variance,
            )
        )

    logger.debug(
        "Feature-range scan for %s: best=%s spread=%.3f",
        type(encoding).__name__,
        max(results, key=lambda r: r.alignment) if results else None,
        max(r.alignment for r in results) - min(r.alignment for r in results),
    )

    return FeatureRangeScan(
        encoding_name=type(encoding).__name__,
        n_qubits=n_qubits,
        n_features=int(X_array.shape[1]),
        n_samples_used=int(len(X_used)),
        n_samples_supplied=int(n_supplied),
        centered=bool(centered),
        results=tuple(results),
    )


def recommend_feature_range(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
    *,
    ranges: Sequence[tuple[float, float]] = DEFAULT_FEATURE_RANGES,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    centered: bool = True,
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> tuple[float, float]:
    """Return the feature range that maximises alignment for ``encoding``.

    A thin wrapper over :func:`scan_feature_ranges` for the common case of
    "just tell me what to scale into". Ties break towards the narrower range.

    Parameters
    ----------
    encoding, X, y, ranges, max_samples, centered, seed, backend
        As for :func:`scan_feature_ranges`.

    Returns
    -------
    (float, float)
        The recommended ``(low, high)``. Feed it to :func:`scale_to_range`.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.benchmark import get_dataset
    >>> X, y = get_dataset("moons", n_samples=40, seed=0)
    >>> low, high = recommend_feature_range(
    ...     IQPEncoding(n_features=2, reps=2), X, y, seed=0
    ... )
    >>> low == 0.0
    True
    """
    return scan_feature_ranges(
        encoding,
        X,
        y,
        ranges=ranges,
        max_samples=max_samples,
        centered=centered,
        seed=seed,
        backend=backend,
    ).best_range
