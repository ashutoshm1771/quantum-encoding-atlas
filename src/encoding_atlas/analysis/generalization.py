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


def compute_fidelity_kernel(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    *,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> NDArray[np.floating[Any]]:
    """Compute the fidelity (overlap) kernel matrix for ``X``.

    ``K[i, j] = |<phi(x_i)|phi(x_j)>|^2`` where ``|phi(x)> = U(x)|0>`` is the
    encoded state. The matrix is symmetric, PSD, has unit diagonal, and entries
    in ``[0, 1]``.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding used for state preparation. ``encoding.n_features`` must equal
        ``X.shape[1]``.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector simulation backend.

    Returns
    -------
    ndarray, shape (n_samples, n_samples)
        The fidelity kernel matrix.
    """
    states = simulate_encoding_statevectors_batch(encoding, X, backend=backend)
    overlaps = states.conj() @ states.T
    K = np.abs(overlaps) ** 2
    np.fill_diagonal(K, 1.0)
    return np.clip(K.real.astype(np.float64), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Kernel-target alignment
# ---------------------------------------------------------------------------


def kernel_target_alignment(
    K: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> float:
    """Uncentred kernel-target alignment in ``[-1, 1]`` (Cristianini, 2002).

    ``A(K, y) = <K, yy^T>_F / (||K||_F ||yy^T||_F)`` with labels mapped to
    ``{-1, +1}``. Returns ``0.0`` for degenerate kernels or single-class labels.
    """
    y_signed = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
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
    """
    n = K.shape[0]
    if n < 2:
        return 0.0
    y_signed = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
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
) -> float:
    """Kernel-target alignment of an encoding's fidelity kernel on ``(X, y)``.

    Higher alignment predicts stronger quantum-kernel classification accuracy.
    Uses the centred variant by default.
    """
    K = compute_fidelity_kernel(encoding, X, backend=backend)
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
    """
    K_q = compute_fidelity_kernel(encoding, X, backend=backend)
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
) -> float:
    """Feature-space effective dimension of an encoding's fidelity kernel.

    Larger values indicate the encoding spreads data across more effective
    feature-space directions (higher capacity). See
    :func:`kernel_effective_dimension` for the definition.
    """
    K = compute_fidelity_kernel(encoding, X, backend=backend)
    return kernel_effective_dimension(K, regularization=regularization)
