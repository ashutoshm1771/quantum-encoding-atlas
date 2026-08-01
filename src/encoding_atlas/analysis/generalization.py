"""Generalization diagnostics for quantum encodings.

The descriptive axes (expressibility, entanglement, trainability) do not, on
their own, predict downstream accuracy — the benchmark finds *no* positive
correlation between expressibility and accuracy. This module adds the
learning-theoretic, kernel-geometry quantities that the QML literature uses to
explain *why* an encoding generalizes:

- **Kernel-target alignment** — how well the encoding's kernel matches the label
  structure (predicts kernel-method accuracy).
- **Geometric difference** — how distinct the quantum kernel is from a classical
  one (a necessary condition for a quantum advantage).
- **Effective dimension** — the effective degrees of freedom the encoding uses
  in feature space (a capacity measure).

All three are cheap classical post-processing of the fidelity (overlap) kernel

    K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2 ,

computed from simulated statevectors — no training required.

Finite shots
------------
By default the kernel is exact (infinite-shot). Passing ``shots=M`` to
:func:`compute_fidelity_kernel`, or to any of the three diagnostics, instead
returns the estimate a real device would produce from ``M`` measurements per
entry. On hardware the standard construction is the *compute-uncompute*
circuit ``U(x_j)^dagger U(x_i)|0>``, whose probability of returning the
all-zeros bitstring is exactly ``K(x_i, x_j)``; the count of all-zeros
outcomes over ``M`` shots is therefore exactly ``Binomial(M, K)``. Sampling
that binomial from the exact kernel is statistically identical to running the
circuits, so shot realism costs one random draw rather than a second
simulation. See :func:`sample_shot_kernel`.

Notes
-----
The effective dimension implemented here is the *feature-space* (kernel ridge)
effective dimension, i.e. the effective degrees of freedom of the kernel's
eigenspectrum. It is a well-defined, directly verifiable quantity and is
distinct from the parameter-space Fisher-information effective dimension of
Abbas et al. (2021); the latter characterises a full variational model rather
than an encoding's feature map.

References
----------
Cristianini et al. (2002), *NeurIPS* 14 — kernel-target alignment.
Cortes, Mohri & Rostamizadeh (2012), *JMLR* 13:795 — centered alignment.
Huang et al. (2021), *Nat. Commun.* 12:2631 — geometric difference.
Zhang (2005), *Neural Comput.* 17:2077; Bach (2013) — effective degrees of
freedom of a kernel.
Havlicek et al. (2019), *Nature* 567:209 — compute-uncompute kernel estimation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.analysis._utils import simulate_encoding_statevectors_batch

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

# Numerical floor below which Frobenius norms are treated as degenerate.
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def _trace_normalize(K: NDArray[np.floating[Any]], n: int) -> NDArray[np.floating[Any]]:
    """Scale ``K`` so that ``trace(K) == n`` (mean eigenvalue 1)."""
    trace = float(np.trace(K))
    if trace <= _EPS:
        return K
    return K * (n / trace)


def _symmetric_psd_sqrt(K: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Return the symmetric PSD square root of a (near-)PSD matrix ``K``."""
    K_sym = (K + K.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(K_sym)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_matrix: NDArray[np.floating[Any]] = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    return sqrt_matrix


# ---------------------------------------------------------------------------
# Fidelity kernel
# ---------------------------------------------------------------------------


def sample_shot_kernel(
    K: NDArray[np.floating[Any]],
    shots: int,
    *,
    seed: int | None = None,
) -> NDArray[np.floating[Any]]:
    """Return the finite-shot estimate of an exact fidelity kernel.

    Models the *compute-uncompute* estimator used on hardware: for each pair
    ``(i, j)`` the circuit ``U(x_j)^dagger U(x_i)|0>`` is measured ``shots``
    times and the fraction of all-zeros outcomes is taken as ``K[i, j]``.
    Because the all-zeros probability *is* ``K[i, j]``, that count is exactly
    ``Binomial(shots, K[i, j])`` — so drawing from the binomial is
    statistically identical to simulating the circuits, at a fraction of the
    cost.

    The estimate is unbiased entrywise and symmetric, and its diagonal is
    exactly 1 (the compute-uncompute circuit for ``x_i = x_i`` returns
    all-zeros with certainty). It is **not** guaranteed to be positive
    semidefinite: independent noise on each entry routinely pushes the
    smallest eigenvalues negative. Project with
    :func:`encoding_atlas.benchmark.kernel.ensure_psd` before handing it to a
    precomputed-kernel estimator.

    Parameters
    ----------
    K : ndarray, shape (n, n)
        Exact kernel with entries in ``[0, 1]``, e.g. from
        :func:`compute_fidelity_kernel`.
    shots : int
        Measurements per entry. Must be a positive integer.
    seed : int or None, default=None
        Seed for the binomial draws.

    Returns
    -------
    ndarray, shape (n, n)
        Symmetric finite-shot estimate with unit diagonal. Entries are
        multiples of ``1 / shots``.

    Raises
    ------
    ValueError
        If ``shots`` is not a positive integer, or ``K`` is not a square 2D
        matrix with entries in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> K = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> K_hat = sample_shot_kernel(K, shots=1000, seed=0)
    >>> bool(K_hat[0, 0] == 1.0 and K_hat[0, 1] == K_hat[1, 0])
    True

    Notes
    -----
    The variance of each off-diagonal estimate is ``K (1 - K) / shots``. When
    the kernel has concentrated — every off-diagonal entry near ``2**-n`` —
    that noise swamps the signal unless ``shots`` grows with the Hilbert-space
    dimension; see
    :func:`encoding_atlas.analysis.concentration.compute_kernel_concentration`
    for the resulting budget.
    """
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if shots < 1:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")

    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K must be a square 2D matrix, got shape {K_arr.shape}")
    if K_arr.size and (float(K_arr.min()) < -_EPS or float(K_arr.max()) > 1.0 + _EPS):
        raise ValueError(
            f"K entries must lie in [0, 1], got "
            f"[{float(K_arr.min()):.6g}, {float(K_arr.max()):.6g}]"
        )

    n = K_arr.shape[0]
    estimate = np.eye(n, dtype=np.float64)
    if n < 2:
        return estimate

    rng = np.random.default_rng(seed)
    upper = np.triu_indices(n, k=1)
    probabilities = np.clip(K_arr[upper], 0.0, 1.0)
    sampled = rng.binomial(int(shots), probabilities) / float(shots)
    estimate[upper] = sampled
    estimate[(upper[1], upper[0])] = sampled
    return estimate


def compute_fidelity_kernel(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    *,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
    shots: int | None = None,
    seed: int | None = None,
) -> NDArray[np.floating[Any]]:
    """Compute the fidelity (overlap) kernel matrix for ``X``.

    ``K[i, j] = |<phi(x_i)|phi(x_j)>|^2`` where ``|phi(x)> = U(x)|0>`` is the
    encoded state. With the default ``shots=None`` the matrix is exact:
    symmetric, PSD, unit diagonal, entries in ``[0, 1]``.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding used for state preparation. ``encoding.n_features`` must equal
        ``X.shape[1]``.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector simulation backend.
    shots : int or None, default=None
        When given, return the finite-shot compute-uncompute estimate from
        ``shots`` measurements per entry instead of the exact kernel (see
        :func:`sample_shot_kernel`). The result stays symmetric with a unit
        diagonal but is **no longer guaranteed PSD**.
    seed : int or None, default=None
        Seed for the shot sampling. Ignored when ``shots`` is ``None``.

    Returns
    -------
    ndarray, shape (n_samples, n_samples)
        The fidelity kernel matrix.

    Raises
    ------
    ValueError
        If ``shots`` is given but is not a positive integer.
    """
    states = simulate_encoding_statevectors_batch(encoding, X, backend=backend)
    overlaps = states.conj() @ states.T
    K = np.abs(overlaps) ** 2
    np.fill_diagonal(K, 1.0)
    exact = np.clip(K.real.astype(np.float64), 0.0, 1.0)
    if shots is None:
        return exact
    return sample_shot_kernel(exact, shots, seed=seed)


# ---------------------------------------------------------------------------
# Kernel-target alignment
# ---------------------------------------------------------------------------


def validate_binary_labels(
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> NDArray[np.float64]:
    """Validate that ``y`` is a 1D, finite, exactly-two-class label vector.

    Kernel-target alignment is a two-class quantity, so anything else — a
    single class, a multi-class vector, or a continuous regression target —
    would be scored meaninglessly rather than rejected. Callers that rank or
    select encodings by alignment should gate on this first.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Candidate label vector. The two classes may be any two distinct
        values; see :func:`_signed_labels` for how they are mapped.

    Returns
    -------
    ndarray of shape (n_samples,), dtype float64
        The labels as float64, unchanged apart from the cast.

    Raises
    ------
    ValueError
        If ``y`` is not 1D, contains non-finite values, or does not have
        exactly two distinct values.

    Examples
    --------
    >>> import numpy as np
    >>> validate_binary_labels(np.array([0, 1, 1, 0])).dtype
    dtype('float64')
    """
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError(f"y must be a 1D label vector, got shape {y_array.shape}")
    y_float = y_array.astype(np.float64)
    if not np.all(np.isfinite(y_float)):
        raise ValueError("y contains NaN or infinite values")
    classes = np.unique(y_float)
    if classes.size != 2:
        raise ValueError(
            f"kernel-target alignment is defined for two-class labels; got "
            f"{classes.size} distinct value(s). For multi-class or regression "
            f"targets, evaluate candidates directly with "
            f"encoding_atlas.benchmark.evaluate_encoding instead."
        )
    return y_float


def _signed_labels(
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> NDArray[np.float64]:
    """Map labels onto the ``{-1, +1}`` convention the alignment is defined on.

    A two-class label vector is mapped *by partition*: the lower class becomes
    ``-1`` and the higher ``+1``, whatever the two values happen to be. This
    makes the alignment invariant to the label convention — ``{0, 1}``,
    ``{1, 2}`` and ``{-1, +1}`` all describe the same split and must score the
    same. For ``{0, 1}`` the result is identical to the ``2y - 1`` transform
    used previously, so published numbers are unaffected.

    Anything that is not exactly two-valued (a single class, a multi-class
    vector, or a continuous regression target) falls back to ``2y - 1``. That
    keeps continuous targets meaningful, and the centred alignment is invariant
    to affine relabelling anyway.
    """
    y_array = np.asarray(y, dtype=np.float64)
    classes = np.unique(y_array[np.isfinite(y_array)])
    if classes.size == 2:
        return np.where(y_array > classes[0], 1.0, -1.0).astype(np.float64)
    return 2.0 * y_array - 1.0


def kernel_target_alignment(
    K: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> float:
    """Uncentred kernel-target alignment in ``[-1, 1]`` (Cristianini, 2002).

    ``A(K, y) = <K, yy^T>_F / (||K||_F ||yy^T||_F)`` with labels mapped to
    ``{-1, +1}``. Returns ``0.0`` for degenerate kernels or single-class labels.

    Two-class labels are mapped by partition, so the score does not depend on
    whether the classes are spelled ``{0, 1}``, ``{1, 2}`` or ``{-1, +1}``.
    Unlike :func:`centered_kernel_target_alignment`, the uncentred score is
    *not* invariant to that choice on its own, which is why the mapping is
    normalised here.
    """
    y_signed = _signed_labels(y)
    y_outer = np.outer(y_signed, y_signed)
    norm_K = float(np.linalg.norm(K, "fro"))
    norm_y = float(np.linalg.norm(y_outer, "fro"))
    if norm_K < _EPS or norm_y < _EPS:
        return 0.0
    return float(np.sum(K * y_outer) / (norm_K * norm_y))


def centered_kernel_target_alignment(
    K: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> float:
    """Centred kernel-target alignment in ``[-1, 1]`` (Cortes et al., 2012).

    Centring removes the bias the uncentred score suffers for fidelity kernels
    (whose diagonal is always 1), giving a more faithful measure of task
    alignment. Returns ``0.0`` for degenerate inputs or ``n < 2``.

    This is the variant the benchmark uses and the one to screen encodings
    with. It is already invariant to affine relabelling of ``y``; labels are
    normalised through :func:`_signed_labels` so both variants agree on what a
    two-class split means.
    """
    n = K.shape[0]
    if n < 2:
        return 0.0
    y_signed = _signed_labels(y)
    y_outer = np.outer(y_signed, y_signed)
    K_c = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    y_c = (
        y_outer
        - y_outer.mean(axis=0, keepdims=True)
        - y_outer.mean(axis=1, keepdims=True)
        + y_outer.mean()
    )
    norm_K = float(np.linalg.norm(K_c, "fro"))
    norm_y = float(np.linalg.norm(y_c, "fro"))
    if norm_K < _EPS or norm_y < _EPS:
        return 0.0
    return float(np.sum(K_c * y_c) / (norm_K * norm_y))


def compute_kernel_target_alignment(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
    *,
    centered: bool = True,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
    shots: int | None = None,
    seed: int | None = None,
) -> float:
    """Kernel-target alignment of an encoding's fidelity kernel on ``(X, y)``.

    Higher alignment predicts stronger quantum-kernel classification accuracy.
    Uses the centred variant by default.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding under test.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    y : ndarray, shape (n_samples,)
        Binary labels.
    centered : bool, default=True
        Use the centred alignment (Cortes et al., 2012).
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector backend.
    shots : int or None, default=None
        Measure the alignment on a finite-shot kernel estimate instead of the
        exact one. Alignment aggregates over all ``n(n-1)/2`` entries, so it
        tolerates shot noise far better than the kernel matrix itself does.
    seed : int or None, default=None
        Seed for the shot sampling. Ignored when ``shots`` is ``None``.
    """
    K = compute_fidelity_kernel(encoding, X, backend=backend, shots=shots, seed=seed)
    if centered:
        return centered_kernel_target_alignment(K, y)
    return kernel_target_alignment(K, y)


# ---------------------------------------------------------------------------
# Geometric difference
# ---------------------------------------------------------------------------


def geometric_difference(
    K1: NDArray[np.floating[Any]],
    K2: NDArray[np.floating[Any]],
    *,
    regularization: float = 1e-6,
) -> float:
    """Asymmetric geometric difference ``g(K1 || K2)`` (Huang et al., 2021).

    ``g(K1 || K2) = sqrt( || sqrt(K2) (K1 + reg I)^-1 sqrt(K2) ||_inf )`` where
    ``||.||_inf`` is the spectral norm and both kernels are trace-normalised to
    ``n``. It quantifies how well a model expressed in ``K1``'s space can
    reproduce ``K2``'s geometry: ``g`` is large when ``K2`` explores directions
    ``K1`` cannot. ``g(K, K) = 1`` (in the ``reg -> 0`` limit).

    Parameters
    ----------
    K1, K2 : ndarray, shape (n, n)
        Kernel matrices (symmetric PSD). ``K1`` is the reference (inverted);
        for a quantum-advantage diagnostic use ``K1 = classical`` and
        ``K2 = quantum``.
    regularization : float, default=1e-6
        Ridge added to ``K1`` before inversion for numerical stability.
    """
    if K1.shape != K2.shape:
        raise ValueError(f"kernel shapes differ: {K1.shape} vs {K2.shape}")
    n = K1.shape[0]
    K1n = _trace_normalize((K1 + K1.T) / 2.0, n)
    K2n = _trace_normalize((K2 + K2.T) / 2.0, n)

    sqrt_K2 = _symmetric_psd_sqrt(K2n)
    inv_K1 = np.linalg.inv(K1n + regularization * np.eye(n))
    M = sqrt_K2 @ inv_K1 @ sqrt_K2
    M = (M + M.T) / 2.0
    max_eig = float(np.linalg.eigvalsh(M)[-1])
    return float(np.sqrt(max(max_eig, 0.0)))


def _classical_kernel(
    X: NDArray[np.floating[Any]],
    kind: str,
    gamma: float | None,
) -> NDArray[np.floating[Any]]:
    """Build a classical reference kernel (``"rbf"`` or ``"linear"``)."""
    if kind == "rbf":
        from sklearn.metrics.pairwise import rbf_kernel

        return np.asarray(rbf_kernel(X, gamma=gamma), dtype=np.float64)
    if kind == "linear":
        from sklearn.metrics.pairwise import linear_kernel

        return np.asarray(linear_kernel(X), dtype=np.float64)
    raise ValueError(f"classical_kernel must be 'rbf' or 'linear', got {kind!r}")


def compute_geometric_difference(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    *,
    classical_kernel: Literal["rbf", "linear"] = "rbf",
    gamma: float | None = None,
    regularization: float = 1e-6,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
    shots: int | None = None,
    seed: int | None = None,
) -> float:
    """Geometric difference ``g(K_classical || K_quantum)`` for an encoding.

    A large value means the encoding's fidelity kernel is geometrically distinct
    from the classical reference kernel — a necessary (not sufficient) condition
    for a quantum advantage over that classical kernel.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding under test.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    classical_kernel : {"rbf", "linear"}, default="rbf"
        Classical reference kernel.
    gamma : float or None, default=None
        RBF bandwidth (``None`` -> ``1 / n_features``, scikit-learn default).
    regularization : float, default=1e-6
        Ridge for the classical-kernel inversion.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector backend.
    shots : int or None, default=None
        Measure against a finite-shot kernel estimate instead of the exact
        one. Shot noise inflates ``g`` — noise is a direction no classical
        kernel reproduces — so a large value under finite shots is not by
        itself evidence of an advantage.
    seed : int or None, default=None
        Seed for the shot sampling. Ignored when ``shots`` is ``None``.
    """
    K_q = compute_fidelity_kernel(encoding, X, backend=backend, shots=shots, seed=seed)
    K_c = _classical_kernel(np.asarray(X, dtype=np.float64), classical_kernel, gamma)
    return geometric_difference(K_c, K_q, regularization=regularization)


# ---------------------------------------------------------------------------
# Effective dimension
# ---------------------------------------------------------------------------


def kernel_effective_dimension(
    K: NDArray[np.floating[Any]],
    *,
    regularization: float = 1.0,
    normalize: bool = True,
) -> float:
    """Effective degrees of freedom of a kernel's eigenspectrum.

    ``d_eff(lambda) = sum_i lambda_i / (lambda_i + lambda)`` over the eigenvalues
    ``lambda_i`` of ``K`` (Zhang, 2005). It is a smooth "soft rank" measuring how
    many feature-space directions the kernel effectively uses, in ``[0, n]``.

    Parameters
    ----------
    K : ndarray, shape (n, n)
        Kernel matrix (symmetric PSD).
    regularization : float, default=1.0
        Ridge ``lambda``. With ``normalize=True`` the mean eigenvalue is 1, so
        ``lambda = 1`` weights each direction by its relative strength.
    normalize : bool, default=True
        Trace-normalise ``K`` to ``n`` (mean eigenvalue 1) before the sum, making
        the value comparable across encodings and dataset sizes.
    """
    if regularization <= 0:
        raise ValueError(f"regularization must be positive, got {regularization}")
    n = K.shape[0]
    K_sym: NDArray[np.floating[Any]] = (K + K.T) / 2.0
    if normalize:
        K_sym = _trace_normalize(K_sym, n)
    eigvals = np.clip(np.linalg.eigvalsh(K_sym), 0.0, None)
    return float(np.sum(eigvals / (eigvals + regularization)))


def compute_effective_dimension(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    *,
    regularization: float = 1.0,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
    shots: int | None = None,
    seed: int | None = None,
) -> float:
    """Feature-space effective dimension of an encoding's fidelity kernel.

    Larger values indicate the encoding spreads data across more effective
    feature-space directions (higher capacity). See
    :func:`kernel_effective_dimension` for the definition.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding under test.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    regularization : float, default=1.0
        Ridge parameter of the effective-dimension sum.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector backend.
    shots : int or None, default=None
        Measure on a finite-shot kernel estimate instead of the exact one.
        Shot noise raises the apparent dimension by spreading the eigenvalue
        spectrum, so finite-shot values overstate capacity.
    seed : int or None, default=None
        Seed for the shot sampling. Ignored when ``shots`` is ``None``.
    """
    K = compute_fidelity_kernel(encoding, X, backend=backend, shots=shots, seed=seed)
    return kernel_effective_dimension(K, regularization=regularization)
