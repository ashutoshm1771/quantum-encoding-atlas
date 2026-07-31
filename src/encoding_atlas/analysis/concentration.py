"""Fidelity-kernel concentration analysis.

Every other axis in this package measures an encoding at a *fixed* circuit
width. This module measures how an encoding's fidelity kernel behaves as that
width grows — the property that decides whether a result obtained at 2-8 qubits
says anything about the same encoding at 20.

The quantity
------------
For encoded states ``|phi(x)> = U(x)|0>`` the fidelity (overlap) kernel is

    K(x, x') = |<phi(x)|phi(x')>|^2 .

A kernel method learns from the *spread* of the off-diagonal entries: if every
distinct pair looks equally similar, the matrix is the identity up to sampling
noise and there is no geometry to fit. Exponential concentration is the
statement that this spread vanishes exponentially in the qubit count,

    Var_{x, x'}[K(x, x')] in O(b^-n) for some b > 1,

at which point separating two entries requires a shot budget growing like
``1 / Var`` — the kernel becomes unusable on hardware well before it becomes
uninformative in exact arithmetic.

The reference point is the Haar-random ensemble. For two independent Haar
states in dimension ``d = 2^n`` the overlap follows ``Beta(1, d - 1)``, so

    E_Haar[K] = 1/d ,   Var_Haar[K] = (d - 1) / (d^2 (d + 1)) ~ d^-2 .

That gives the dimensionless order parameter reported here:

    concentration_ratio = Var[K] / Var_Haar[K] .

A value near 1 means the kernel's spread has collapsed to what a maximally
scrambling ensemble would produce — fully concentrated. Larger values mean the
encoding retains structure the circuit width has not yet destroyed.

Why the variance and not the mean
---------------------------------
The *mean* cannot tell these regimes apart. An encoding that puts each qubit in
an independent, uniformly random single-qubit state has mean overlap exactly
``2^-n = 1/d`` — indistinguishable from Haar — while its variance is
``(3/8)^n - (1/4)^n``, exponentially *larger* than the Haar ``~4^-n``. Angle
encoding is exactly this case: its mean ratio is 1.00 at every width, while its
variance ratio grows by a factor of 1.5 per qubit. The mean is reported as
:attr:`ConcentrationResult.mean_ratio` for completeness, but it is degenerate
and must not be used as the order parameter.

Relation to the rest of the atlas
---------------------------------
This is the mechanism behind the benchmark's headline negative result. High
expressibility *means* Haar-likeness, Haar-likeness *implies* a kernel whose
variance sits at the ``~d^-2`` floor, and a kernel at the floor cannot
generalize — so the encodings that score best on expressibility are the ones
whose kernels are least usable. Concentration turns that correlation into an
explanation, and unlike expressibility it says *at which qubit count* the
problem bites.

Concentration is a joint property of the encoding **and** the input
distribution. Both entry points default to inputs drawn uniformly from
``[0, 2*pi)``, matching the rest of the analysis package; narrowing the range
(as feature scaling does) generally reduces concentration.

Notes
-----
Shot figures assume the standard compute-uncompute estimator, whose all-zeros
count is exactly ``Binomial(shots, K)`` — see
:func:`encoding_atlas.analysis.generalization.sample_shot_kernel`. They are a
noiseless lower bound on hardware cost: gate noise, readout error, and
compilation overhead are not modelled.

References
----------
Thanasilp, Wang, Cerezo & Holmes (2024), *Nat. Commun.* 15:5200 — exponential
concentration in quantum kernel methods.
Huang et al. (2021), *Nat. Commun.* 12:2631 — kernel geometry and learnability.
Sim, Johnson & Aspuru-Guzik (2019), *Adv. Quantum Technol.* 2:1900070 —
expressibility as Haar-likeness.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.analysis._sampling import (
    SamplingMethod,
    generate_sample_batch,
    validate_sampling,
)
from encoding_atlas.analysis.generalization import compute_fidelity_kernel

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)

__all__ = [
    "ConcentrationResult",
    "ScalingResult",
    "compute_kernel_concentration",
    "estimate_concentration_scaling",
    "summarize_kernel_concentration",
    "haar_kernel_moments",
    "CONCENTRATION_THRESHOLD",
]

# Default number of random inputs used to build the kernel. Statistics are over
# the n(n-1)/2 off-diagonal pairs, so 40 samples already gives 780 of them.
_DEFAULT_N_SAMPLES: int = 40

# Default feature sampling range, matching the rest of the analysis package.
_DEFAULT_INPUT_RANGE: tuple[float, float] = (0.0, 2.0 * math.pi)

# Default width sweep, matching the range the bundled empirical atlas covers.
_DEFAULT_FEATURE_COUNTS: tuple[int, ...] = (2, 4, 6, 8)

# ``concentration_ratio`` below this is treated as "at the Haar floor": the
# kernel's off-diagonal variance is within a factor of two of what a
# Haar-random ensemble gives, leaving no usable geometry. The factor of two is
# a convention, not a theorem — it is exposed so callers can pick their own
# operating point.
CONCENTRATION_THRESHOLD: float = 2.0

# Statevector batches cost n_samples * 2^n_qubits complex128. Warn once the
# working set passes ~10 MB; the hard cap is the simulator's own 20-qubit limit.
_WARN_QUBITS: int = 14

# Minimum number of measured widths needed to fit a decay rate.
_MIN_FIT_POINTS: int = 2

# Floor for log-space fitting: values below this are treated as unresolvable
# rather than driving the fit to -inf.
_LOG_FLOOR: float = 1e-15


def haar_kernel_moments(n_qubits: int) -> tuple[float, float]:
    """Return the Haar-random ``(mean, variance)`` of the fidelity kernel.

    For two independent Haar-random pure states in dimension ``d = 2**n_qubits``
    the overlap ``|<psi|phi>|^2`` follows ``Beta(1, d - 1)``, giving

        mean = 1 / d ,    variance = (d - 1) / (d^2 (d + 1)) .

    The variance is the floor an encoding's kernel decays *towards*; it is the
    reference for :attr:`ConcentrationResult.concentration_ratio`.

    Parameters
    ----------
    n_qubits : int
        Circuit width. Must be a positive integer.

    Returns
    -------
    (float, float)
        The Haar mean and variance of a single off-diagonal kernel entry.

    Raises
    ------
    ValueError
        If ``n_qubits`` is not a positive integer.

    Examples
    --------
    >>> mean, variance = haar_kernel_moments(4)
    >>> round(mean, 6)
    0.0625
    >>> round(variance, 8)
    0.00344669
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)):
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")
    d = float(2 ** int(n_qubits))
    return 1.0 / d, (d - 1.0) / (d * d * (d + 1.0))


def _shots_to_resolve(mean: float, variance: float) -> float:
    """Shots per kernel entry needed to resolve the off-diagonal spread.

    A compute-uncompute estimate of one entry from ``M`` shots has standard
    error ``sqrt(K (1 - K) / M)``. Requiring that to fall below half the
    observed off-diagonal standard deviation gives

        M >= 4 K_bar (1 - K_bar) / Var .

    Returns ``inf`` when the entries carry no spread (``Var == 0``): no shot
    budget separates identical values. Returns ``1.0`` when the estimator is
    itself noiseless (every entry exactly 0 or exactly 1).
    """
    if variance <= 0.0:
        return math.inf
    entry_variance = mean * (1.0 - mean)
    if entry_variance <= 0.0:
        return 1.0
    return 4.0 * entry_variance / variance


@dataclass(frozen=True)
class ConcentrationResult:
    """Fidelity-kernel concentration measured at one circuit width.

    Attributes
    ----------
    n_qubits : int
        Circuit width the kernel was measured at.
    n_samples : int
        Number of inputs used; statistics are over the
        ``n_samples (n_samples - 1) / 2`` off-diagonal pairs.
    offdiagonal_mean : float
        Mean off-diagonal kernel entry, in ``[0, 1]``.
    offdiagonal_variance, offdiagonal_std : float
        Spread of the off-diagonal entries — the signal a kernel method
        actually uses. Its vanishing *is* concentration.
    offdiagonal_min, offdiagonal_max : float
        Extremes of the off-diagonal entries.
    haar_mean, haar_variance : float
        Haar-random reference moments at this width, from
        :func:`haar_kernel_moments`.
    concentration_ratio : float
        ``offdiagonal_variance / haar_variance`` — **the order parameter**.
        Near 1 means the kernel's spread has reached the Haar floor and
        carries no usable geometry; larger means structure remains. Directly
        comparable across qubit counts.
    mean_ratio : float
        ``offdiagonal_mean / haar_mean``. Reported for completeness only:
        it is 1 for *any* ensemble of independent random single-qubit states
        as well as for Haar, so it cannot distinguish the two regimes. Do not
        use it as a concentration measure.
    identity_distance : float
        Root-mean-square off-diagonal entry, i.e.
        ``||K - I||_F / sqrt(n (n - 1))``. Zero means ``K`` is exactly ``I``.
    is_concentrated : bool
        Whether ``concentration_ratio`` is below :attr:`threshold`.
    threshold : float
        The ``concentration_ratio`` cut used for :attr:`is_concentrated`.
    shots_per_entry : float
        Shots per kernel entry for the compute-uncompute estimator's standard
        error to fall below half the observed spread —
        ``4 K_bar (1 - K_bar) / Var``. ``inf`` if the entries have no spread.
        A noiseless lower bound on hardware cost.
    """

    n_qubits: int
    n_samples: int
    offdiagonal_mean: float
    offdiagonal_variance: float
    offdiagonal_std: float
    offdiagonal_min: float
    offdiagonal_max: float
    haar_mean: float
    haar_variance: float
    concentration_ratio: float
    mean_ratio: float
    identity_distance: float
    is_concentrated: bool
    threshold: float
    shots_per_entry: float

    def shots_for_dataset(self, n_samples: int) -> float:
        """Total shots to estimate a full ``n_samples x n_samples`` kernel.

        The kernel is symmetric with a deterministic unit diagonal, so only
        ``n (n - 1) / 2`` entries need measuring.

        Parameters
        ----------
        n_samples : int
            Dataset size. Must be a positive integer.

        Returns
        -------
        float
            ``shots_per_entry * n (n - 1) / 2``; ``inf`` if the per-entry cost
            is unbounded.

        Raises
        ------
        ValueError
            If ``n_samples`` is not a positive integer.
        """
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
        n = int(n_samples)
        return self.shots_per_entry * (n * (n - 1) / 2.0)


@dataclass(frozen=True)
class ScalingResult:
    """How an encoding's kernel concentration scales with circuit width.

    Attributes
    ----------
    encoding_name : str
        Class name of the encodings the factory produced.
    n_qubits : tuple[int, ...]
        Circuit widths successfully measured, ascending.
    feature_counts : tuple[int, ...]
        Feature counts that produced those widths, in the same order.
    results : tuple[ConcentrationResult, ...]
        Per-width measurements, in the same order.
    decay_rate : float
        Per-qubit factor by which the off-diagonal **variance** shrinks, from
        a least-squares fit of ``log(variance)`` against ``n_qubits``. ``1.0``
        means no decay. The Haar floor itself decays at ``4.0``, so a rate
        near 4 means the encoding is scrambling as fast as it possibly can.
        ``nan`` if fewer than two widths were measurable.
    mean_decay_rate : float
        The same quantity for the off-diagonal mean. The Haar floor decays at
        ``2.0``. ``nan`` if unfittable.
    haar_normalized_slope : float
        Slope of ``log(concentration_ratio)`` against ``n_qubits``. Negative
        means the kernel is collapsing *towards* the Haar floor as the circuit
        widens; non-negative means it holds structure or pulls away. This is
        the honest discriminator — a large ``decay_rate`` alone proves nothing,
        because the floor is falling too.
    r_squared : float
        Coefficient of determination of the ``log(variance)`` fit, in
        ``[0, 1]``. Well below ~0.9 means the log-linear model — and any
        extrapolation from it — should not be trusted. ``nan`` for fewer than
        two points.
    skipped : dict[int, str]
        Feature counts that could not be measured, mapped to the reason.
    """

    encoding_name: str
    n_qubits: tuple[int, ...]
    feature_counts: tuple[int, ...]
    results: tuple[ConcentrationResult, ...]
    decay_rate: float
    mean_decay_rate: float
    haar_normalized_slope: float
    r_squared: float
    skipped: dict[int, str]

    @property
    def concentration_ratios(self) -> tuple[float, ...]:
        """The measured ``concentration_ratio`` at each width."""
        return tuple(r.concentration_ratio for r in self.results)

    @property
    def offdiagonal_variances(self) -> tuple[float, ...]:
        """The measured off-diagonal variance at each width."""
        return tuple(r.offdiagonal_variance for r in self.results)

    @property
    def offdiagonal_means(self) -> tuple[float, ...]:
        """The measured mean off-diagonal entry at each width."""
        return tuple(r.offdiagonal_mean for r in self.results)

    def concentration_horizon(
        self, threshold: float = CONCENTRATION_THRESHOLD
    ) -> int | None:
        """Smallest qubit count at which the kernel reaches the Haar floor.

        Resolution order:

        1. If the measured widths show a *sustained* crossing — some width
           from which every wider measured point is also below ``threshold``
           — the smallest such width is returned. No extrapolation involved.
           The crossing must be sustained: a single dip below the threshold at
           the narrowest width, followed by growth away from the floor, is not
           a horizon.
        2. Otherwise, if :attr:`haar_normalized_slope` is negative, the fit is
           extrapolated from the widest measured point to the first integer
           width whose predicted ratio falls below ``threshold``.
        3. Otherwise ``None``: the encoding shows no sign of approaching the
           floor over the measured range.

        Parameters
        ----------
        threshold : float, default=:data:`CONCENTRATION_THRESHOLD`
            ``concentration_ratio`` defining the floor. Must be positive.

        Returns
        -------
        int or None
            The horizon, or ``None`` if the measurement does not support one.

        Raises
        ------
        ValueError
            If ``threshold`` is not positive.

        Warnings
        --------
        Case 2 is an extrapolation from a fitted log-linear model. Check
        :attr:`r_squared` before relying on it.
        """
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        # Walk inwards from the widest measurement: the horizon is the start of
        # the longest unbroken run of below-threshold ratios that reaches the
        # widest width measured.
        ratios = self.concentration_ratios
        sustained: int | None = None
        for index in range(len(ratios) - 1, -1, -1):
            if ratios[index] >= threshold:
                break
            sustained = self.n_qubits[index]
        if sustained is not None:
            return sustained

        slope = self.haar_normalized_slope
        if not math.isfinite(slope) or slope >= 0.0:
            return None

        # log(ratio(n)) is linear in n. Anchor the extrapolation on the widest
        # measured point so it starts from data, not from the fitted intercept.
        n_last = self.n_qubits[-1]
        ratio_last = self.concentration_ratios[-1]
        if ratio_last <= 0.0:
            return n_last
        steps = (math.log(threshold) - math.log(ratio_last)) / slope
        return int(n_last + math.ceil(max(steps, 0.0)))

    def shots_per_entry_at(self, n_qubits: int) -> float:
        """Extrapolate the per-entry shot requirement to a wider circuit.

        Propagates the mean and the variance forward with their own fitted
        decay rates and re-evaluates ``4 K_bar (1 - K_bar) / Var``. Using both
        fits matters: the relative spread of the entries is not preserved as
        the circuit widens, so scaling the standard deviation with the mean
        would be wrong by an exponential factor.

        Parameters
        ----------
        n_qubits : int
            Target circuit width. Must be a positive integer. Widths at or
            below the measured range return the nearest measurement instead of
            an extrapolation.

        Returns
        -------
        float
            Extrapolated shots per kernel entry, or ``nan`` if no fit exists.

        Raises
        ------
        ValueError
            If ``n_qubits`` is not a positive integer.

        Warnings
        --------
        Beyond the measured range this is an extrapolation, not a measurement.
        Check :attr:`r_squared`.
        """
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)):
            raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")
        if not self.results:
            return math.nan

        target = int(n_qubits)
        steps = target - self.n_qubits[-1]
        if steps <= 0:
            closest = min(self.results, key=lambda r: abs(r.n_qubits - target))
            return closest.shots_per_entry

        if not (
            math.isfinite(self.decay_rate)
            and math.isfinite(self.mean_decay_rate)
            and self.decay_rate > 0.0
            and self.mean_decay_rate > 0.0
        ):
            return math.nan

        last = self.results[-1]
        mean = last.offdiagonal_mean / (self.mean_decay_rate**steps)
        variance = last.offdiagonal_variance / (self.decay_rate**steps)
        return _shots_to_resolve(mean, variance)


def _validate_common(n_samples: int, input_range: tuple[float, float]) -> None:
    """Validate the arguments shared by both public entry points."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError(f"n_samples must be an integer >= 2, got {n_samples!r}")
    if n_samples < 2:
        raise ValueError(
            f"n_samples must be an integer >= 2 (a kernel needs at least one "
            f"off-diagonal pair), got {n_samples!r}"
        )
    if not isinstance(input_range, (tuple, list)) or len(input_range) != 2:
        raise ValueError(f"input_range must be a (min, max) tuple, got {input_range!r}")
    low, high = float(input_range[0]), float(input_range[1])
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"input_range must be finite, got {input_range!r}")
    if low >= high:
        raise ValueError(f"input_range must satisfy min < max, got ({low}, {high})")


def compute_kernel_concentration(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]] | None = None,
    *,
    n_samples: int = _DEFAULT_N_SAMPLES,
    input_range: tuple[float, float] = _DEFAULT_INPUT_RANGE,
    sampling: SamplingMethod = "uniform",
    threshold: float = CONCENTRATION_THRESHOLD,
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> ConcentrationResult:
    """Measure how concentrated an encoding's fidelity kernel is.

    Builds the fidelity kernel on ``X`` (or on random inputs when ``X`` is not
    supplied) and reduces it to the concentration order parameter, the spread
    of its off-diagonal entries, and the shot budget that spread implies.

    A ``concentration_ratio`` near 1 means the kernel's off-diagonal variance
    has collapsed to the Haar floor: every pair of distinct inputs is equally
    dissimilar, the matrix is the identity up to sampling noise, and a kernel
    method built on it cannot generalize regardless of shot budget or noise
    level. Values well above 1 mean the encoding still separates inputs.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding to analyse.
    X : ndarray of shape (n_samples, n_features), optional
        Inputs to build the kernel from. When ``None`` (default), inputs are
        drawn from ``input_range``, making this a data-free axis comparable
        across encodings — the same convention as expressibility and
        entanglement capability.
    n_samples : int, default=40
        Number of random inputs when ``X`` is not supplied. Must be >= 2.
        Ignored when ``X`` is given.
    input_range : (float, float), default=(0, 2*pi)
        Range for randomly sampled inputs. Concentration depends on the input
        distribution as well as the circuit; narrowing this range (as feature
        scaling does) generally reduces it. Ignored when ``X`` is given.
    sampling : {"uniform", "sobol"}, default="uniform"
        Random-input strategy. Ignored when ``X`` is given.
    threshold : float, default=2.0
        ``concentration_ratio`` below which ``is_concentrated`` is set.
    seed : int or None, default=None
        Seed for input sampling. Ignored when ``X`` is given.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector simulation backend.

    Returns
    -------
    ConcentrationResult
        The measured concentration statistics.

    Raises
    ------
    ValueError
        If ``n_samples``, ``input_range``, ``threshold`` or ``sampling`` is
        invalid, or if ``X`` is not a 2D array with at least two rows.

    Warns
    -----
    UserWarning
        If the encoding is wide enough that the statevector batch becomes
        memory-hungry (more than 14 qubits).

    Examples
    --------
    An entangling feature map sits at the Haar floor; a non-entangling one
    does not:

    >>> from encoding_atlas import AngleEncoding, IQPEncoding
    >>> angle = compute_kernel_concentration(
    ...     AngleEncoding(n_features=6), n_samples=30, seed=0
    ... )
    >>> iqp = compute_kernel_concentration(
    ...     IQPEncoding(n_features=6, reps=2), n_samples=30, seed=0
    ... )
    >>> angle.concentration_ratio > iqp.concentration_ratio
    True

    See Also
    --------
    estimate_concentration_scaling : The same measurement across widths.
    encoding_atlas.analysis.generalization.compute_fidelity_kernel : The kernel.
    """
    validate_sampling(sampling)
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    n_qubits = int(encoding.n_qubits)
    if n_qubits > _WARN_QUBITS:
        n_states = n_samples if X is None else len(X)
        warnings.warn(
            f"Kernel concentration at {n_qubits} qubits simulates a "
            f"{n_states} x 2^{n_qubits} statevector batch "
            f"(~{n_states * (2**n_qubits) * 16 / 1024**2:.0f} MB) and may be "
            f"slow. Consider fewer samples or a narrower circuit.",
            UserWarning,
            stacklevel=2,
        )

    if X is None:
        _validate_common(n_samples, input_range)
        rng = np.random.default_rng(seed)
        X_used = generate_sample_batch(
            int(n_samples),
            int(encoding.n_features),
            input_range,
            rng,
            sampling,
        )
    else:
        X_used = np.asarray(X, dtype=np.float64)
        if X_used.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_used.shape}")
        if X_used.shape[0] < 2:
            raise ValueError(
                f"X must contain at least 2 samples to form an off-diagonal "
                f"pair, got {X_used.shape[0]}"
            )

    K = compute_fidelity_kernel(encoding, X_used, backend=backend)
    result = summarize_kernel_concentration(K, n_qubits, threshold=threshold)

    logger.debug(
        "Concentration for %s at %d qubits: var=%.6g ratio=%.3f",
        type(encoding).__name__,
        n_qubits,
        result.offdiagonal_variance,
        result.concentration_ratio,
    )
    return result


def summarize_kernel_concentration(
    K: NDArray[np.floating[Any]],
    n_qubits: int,
    *,
    threshold: float = CONCENTRATION_THRESHOLD,
) -> ConcentrationResult:
    """Reduce an already-computed fidelity kernel to concentration statistics.

    Split out from :func:`compute_kernel_concentration` so a caller that
    already holds the kernel — screening several encodings, for instance —
    gets the concentration axis without paying for a second simulation, while
    both paths keep a single definition of the statistics.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Fidelity kernel with entries in ``[0, 1]``. Must have at least two
        rows so an off-diagonal pair exists.
    n_qubits : int
        Circuit width the kernel was produced at; sets the Haar reference.
    threshold : float, default=2.0
        ``concentration_ratio`` below which ``is_concentrated`` is set.

    Returns
    -------
    ConcentrationResult
        The same statistics :func:`compute_kernel_concentration` returns.

    Raises
    ------
    ValueError
        If ``K`` is not a square 2D matrix of size >= 2, ``n_qubits`` is not a
        positive integer, or ``threshold`` is not positive.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K must be a square 2D matrix, got shape {K_arr.shape}")
    if K_arr.shape[0] < 2:
        raise ValueError(
            f"K must be at least 2x2 to have an off-diagonal pair, got "
            f"{K_arr.shape[0]}x{K_arr.shape[0]}"
        )

    n = K_arr.shape[0]
    offdiag = K_arr[np.triu_indices(n, k=1)]
    mean = float(offdiag.mean())
    variance = float(offdiag.var())
    haar_mean, haar_variance = haar_kernel_moments(n_qubits)
    ratio = variance / haar_variance if haar_variance > 0.0 else math.inf

    return ConcentrationResult(
        n_qubits=int(n_qubits),
        n_samples=n,
        offdiagonal_mean=mean,
        offdiagonal_variance=variance,
        offdiagonal_std=float(math.sqrt(variance)),
        offdiagonal_min=float(offdiag.min()),
        offdiagonal_max=float(offdiag.max()),
        haar_mean=haar_mean,
        haar_variance=haar_variance,
        concentration_ratio=ratio,
        mean_ratio=mean / haar_mean if haar_mean > 0.0 else math.inf,
        identity_distance=float(math.sqrt(float(np.mean(offdiag**2)))),
        is_concentrated=bool(ratio < threshold),
        threshold=float(threshold),
        shots_per_entry=_shots_to_resolve(mean, variance),
    )


def _fit_log_linear(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, float]:
    """Least-squares fit of ``log(y) = a + b x``; returns ``(slope, r_squared)``.

    Returns ``(nan, nan)`` when fewer than two points are usable or the
    abscissa has no spread.
    """
    usable = y > _LOG_FLOOR
    x_fit = x[usable]
    if x_fit.size < _MIN_FIT_POINTS or float(np.ptp(x_fit)) == 0.0:
        return math.nan, math.nan

    y_fit = np.log(y[usable])
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    residual = y_fit - (slope * x_fit + intercept)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r_squared = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), float(r_squared)


def estimate_concentration_scaling(
    factory: Callable[[int], BaseEncoding],
    *,
    feature_counts: tuple[int, ...] = _DEFAULT_FEATURE_COUNTS,
    n_samples: int = _DEFAULT_N_SAMPLES,
    input_range: tuple[float, float] = _DEFAULT_INPUT_RANGE,
    sampling: SamplingMethod = "uniform",
    threshold: float = CONCENTRATION_THRESHOLD,
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> ScalingResult:
    """Measure kernel concentration across a range of circuit widths.

    Calls ``factory(n_features)`` for each requested feature count, measures
    the concentration at each resulting width, and fits a log-linear decay
    model. Feature counts the factory rejects (an encoding requiring exactly
    two features, or an even count) are recorded in
    :attr:`ScalingResult.skipped` rather than aborting the sweep.

    The fit is against the encoding's **qubit count**, not its feature count,
    because concentration is a statement about Hilbert-space dimension. The
    distinction matters for amplitude encoding, where ``n_qubits =
    ceil(log2(n_features))``.

    Parameters
    ----------
    factory : callable
        ``n_features -> BaseEncoding``, e.g.
        ``lambda d: IQPEncoding(n_features=d, reps=2)``.
    feature_counts : tuple[int, ...], default=(2, 4, 6, 8)
        Feature counts to sweep. Must be non-empty and all positive.
    n_samples : int, default=40
        Random inputs per width. Must be >= 2.
    input_range : (float, float), default=(0, 2*pi)
        Range for randomly sampled inputs.
    sampling : {"uniform", "sobol"}, default="uniform"
        Random-input strategy.
    threshold : float, default=2.0
        ``concentration_ratio`` defining the Haar floor.
    seed : int or None, default=None
        Base seed. Width ``i`` uses ``seed + i``, so widths are independent
        draws yet the whole sweep is reproducible.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector simulation backend.

    Returns
    -------
    ScalingResult
        Per-width measurements plus the fitted decay rates, Haar-normalized
        slope, and fit quality.

    Raises
    ------
    ValueError
        If ``feature_counts`` is empty or contains non-positive values, or if
        any shared argument is invalid.
    RuntimeError
        If the factory failed for *every* requested feature count; the
        per-count reasons are included in the message.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> scaling = estimate_concentration_scaling(
    ...     lambda d: IQPEncoding(n_features=d, reps=2),
    ...     feature_counts=(2, 4),
    ...     n_samples=12,
    ...     seed=0,
    ... )
    >>> len(scaling.results)
    2

    See Also
    --------
    compute_kernel_concentration : Single-width measurement.
    """
    validate_sampling(sampling)
    _validate_common(n_samples, input_range)
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    counts = tuple(int(c) for c in feature_counts)
    if not counts:
        raise ValueError("feature_counts must not be empty")
    if any(c < 1 for c in counts):
        raise ValueError(f"feature_counts must all be positive, got {feature_counts!r}")

    encoding_name = ""
    measured: list[tuple[int, int, ConcentrationResult]] = []
    skipped: dict[int, str] = {}

    for index, n_features in enumerate(sorted(set(counts))):
        try:
            encoding = factory(n_features)
            result = compute_kernel_concentration(
                encoding,
                n_samples=n_samples,
                input_range=input_range,
                sampling=sampling,
                threshold=threshold,
                seed=None if seed is None else seed + index,
                backend=backend,
            )
        except Exception as exc:  # noqa: BLE001 - one bad width must not abort
            skipped[n_features] = f"{type(exc).__name__}: {exc}"
            logger.debug(
                "Concentration sweep skipped n_features=%d: %s", n_features, exc
            )
            continue
        encoding_name = encoding_name or type(encoding).__name__
        measured.append((int(encoding.n_qubits), n_features, result))

    if not measured:
        detail = "; ".join(f"n_features={k}: {v}" for k, v in sorted(skipped.items()))
        raise RuntimeError(
            f"Concentration sweep produced no measurable widths. Reasons: {detail}"
        )

    # A factory may map distinct feature counts onto the same qubit count
    # (amplitude encoding). Sort by width and keep every measurement; the fit
    # tolerates repeated abscissae.
    measured.sort(key=lambda item: (item[0], item[1]))
    qubit_counts = tuple(item[0] for item in measured)
    used_features = tuple(item[1] for item in measured)
    results = tuple(item[2] for item in measured)

    x = np.asarray(qubit_counts, dtype=np.float64)
    variances = np.asarray([r.offdiagonal_variance for r in results], dtype=np.float64)
    means = np.asarray([r.offdiagonal_mean for r in results], dtype=np.float64)
    ratios = np.asarray([r.concentration_ratio for r in results], dtype=np.float64)

    variance_slope, r_squared = _fit_log_linear(x, variances)
    mean_slope, _ = _fit_log_linear(x, means)
    ratio_slope, _ = _fit_log_linear(x, ratios)

    return ScalingResult(
        encoding_name=encoding_name,
        n_qubits=qubit_counts,
        feature_counts=used_features,
        results=results,
        decay_rate=(
            math.exp(-variance_slope) if math.isfinite(variance_slope) else math.nan
        ),
        mean_decay_rate=(
            math.exp(-mean_slope) if math.isfinite(mean_slope) else math.nan
        ),
        haar_normalized_slope=ratio_slope,
        r_squared=r_squared,
        skipped=skipped,
    )
