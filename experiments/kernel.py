"""Quantum kernel computation for Stage 6b experiments.

This module implements quantum kernel matrix computation using the fidelity
kernel approach. The kernel entry K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|² measures the
squared overlap between encoded quantum states.

Key functions:
    - simulate_encoding_state: Get statevector from encoding
    - compute_kernel_entry: Compute single kernel matrix entry
    - compute_kernel_matrix: Compute full kernel matrix
    - compute_kernel_matrix_cross: Compute kernel between train/test sets
    - kernel_target_alignment: Uncentered kernel-label alignment (Cristianini 2002)
    - centered_kernel_target_alignment: Centered variant (Cortes 2012)
    - ensure_psd: Enforce positive semi-definiteness

Mathematical Background
-----------------------
The fidelity kernel (also called quantum kernel or overlap kernel) is:

    K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|² = |⟨0|U†(xᵢ)U(xⱼ)|0⟩|²

where U(x) is the encoding circuit and |φ(x)⟩ = U(x)|0⟩.

Properties:
- K is symmetric: K(xᵢ, xⱼ) = K(xⱼ, xᵢ)
- K is positive semi-definite (by construction)
- K(x, x) = 1 (self-fidelity)
- 0 ≤ K(xᵢ, xⱼ) ≤ 1

References
----------
.. [1] Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
       feature spaces." Nature, 567(7747), 209-212.

.. [2] Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature
       Hilbert spaces." Physical Review Letters, 122(4), 040504.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)

__all__ = [
    "simulate_encoding_state",
    "compute_kernel_entry",
    "compute_kernel_matrix",
    "compute_kernel_matrix_cross",
    "kernel_target_alignment",
    "centered_kernel_target_alignment",
    "ensure_psd",
]


def simulate_encoding_state(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    backend: str = "pennylane",
) -> NDArray[np.complexfloating[Any, Any]]:
    """Simulate the encoding circuit and return the statevector.

    This function wraps the encoding's circuit execution to obtain the
    final quantum state for a given input.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to use.
    x : np.ndarray, shape (n_features,)
        Input features to encode.
    backend : str, default="pennylane"
        Backend to use for simulation. Only "pennylane" is currently
        supported for statevector simulation.

    Returns
    -------
    np.ndarray, shape (2**n_qubits,)
        Complex statevector representing the encoded quantum state.

    Raises
    ------
    ValueError
        If backend is not supported.
    SimulationError
        If circuit simulation fails.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> import numpy as np
    >>> enc = IQPEncoding(n_features=2, reps=1)
    >>> x = np.array([0.5, 1.0])
    >>> state = simulate_encoding_state(enc, x)
    >>> print(f"State dimension: {len(state)}")
    """
    # Use the analysis module's simulation utility for consistency
    from encoding_atlas.analysis._utils import simulate_encoding_statevector

    return simulate_encoding_statevector(encoding, x, backend=backend)


def compute_kernel_entry(
    state_i: NDArray[np.complexfloating[Any, Any]],
    state_j: NDArray[np.complexfloating[Any, Any]],
) -> float:
    """Compute a single kernel matrix entry: |⟨φ(xᵢ)|φ(xⱼ)⟩|².

    This is the fidelity kernel, measuring the squared overlap between
    two quantum states.

    Parameters
    ----------
    state_i : np.ndarray
        First statevector.
    state_j : np.ndarray
        Second statevector.

    Returns
    -------
    float
        Kernel entry value in [0, 1].

    Notes
    -----
    The fidelity F = |⟨ψ|φ⟩|² is:
    - 1 if states are identical (up to global phase)
    - 0 if states are orthogonal
    """
    # Complex inner product: ⟨state_i|state_j⟩
    overlap = np.vdot(state_i, state_j)
    # Squared magnitude: |⟨state_i|state_j⟩|²
    fidelity = float(np.abs(overlap) ** 2)
    # Clamp to [0, 1] for numerical safety
    return float(np.clip(fidelity, 0.0, 1.0))


def compute_kernel_matrix(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    backend: str = "pennylane",
    verbose: bool = False,
    return_states: bool = False,
) -> (
    NDArray[np.floating[Any]]
    | Tuple[NDArray[np.floating[Any]], List[NDArray[np.complexfloating[Any, Any]]]]
):
    """Compute the full quantum kernel matrix.

    For n samples, computes the n×n symmetric kernel matrix where
    K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|².

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to use for state preparation.
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix.
    backend : str, default="pennylane"
        Backend for circuit simulation.
    verbose : bool, default=False
        If True, log progress information.
    return_states : bool, default=False
        If True, return the pre-computed statevectors alongside the
        kernel matrix. This avoids redundant simulation when the
        states are needed later (e.g. for cross-kernel computation).

    Returns
    -------
    K : np.ndarray, shape (n_samples, n_samples)
        Symmetric kernel matrix with K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|².
        Diagonal elements are 1 (self-fidelity).
        Returned alone when ``return_states=False``.
    states : list[np.ndarray]
        Pre-computed statevectors, one per sample.
        Only returned when ``return_states=True``, as the second
        element of a ``(K, states)`` tuple.

    Notes
    -----
    **Complexity:**
    - Time: O(n²) circuit simulations (exploiting symmetry)
    - Memory: O(n) for storing statevectors + O(n²) for kernel matrix

    **Optimization:**
    Statevectors are pre-computed once (n simulations), then pairwise
    fidelities are computed from the stored states (n(n+1)/2 inner products).

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> import numpy as np
    >>> enc = IQPEncoding(n_features=2, reps=1)
    >>> X = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1]])
    >>> K = compute_kernel_matrix(enc, X)
    >>> print(f"Kernel shape: {K.shape}")
    >>> print(f"Diagonal (should be 1): {np.diag(K)}")

    Return states for reuse in cross-kernel computation:

    >>> K, states = compute_kernel_matrix(enc, X, return_states=True)
    """
    n = len(X)
    K = np.zeros((n, n), dtype=np.float64)

    # Pre-compute all statevectors
    if verbose:
        logger.info("Computing %d statevectors...", n)

    states: List[NDArray[np.complexfloating[Any, Any]]] = []
    for i, xi in enumerate(X):
        state = simulate_encoding_state(encoding, xi, backend)
        states.append(state)
        if verbose and (i + 1) % 50 == 0:
            logger.info("Computed %d/%d statevectors", i + 1, n)

    # Compute kernel entries (upper triangle + diagonal)
    if verbose:
        logger.info("Computing kernel entries...")

    for i in range(n):
        K[i, i] = 1.0  # Self-fidelity is always 1
        for j in range(i + 1, n):
            K[i, j] = compute_kernel_entry(states[i], states[j])
            K[j, i] = K[i, j]  # Symmetric

    if verbose:
        logger.info("Kernel matrix computation complete.")

    if return_states:
        return K, states
    return K


def compute_kernel_matrix_cross(
    encoding: BaseEncoding,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    train_states: Optional[List[NDArray[np.complexfloating[Any, Any]]]] = None,
    backend: str = "pennylane",
) -> NDArray[np.floating[Any]]:
    """Compute kernel matrix between test and training samples.

    This is used during prediction: for each test sample, we need its
    kernel values against all training samples.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to use.
    X_train : np.ndarray, shape (n_train, n_features)
        Training data.
    X_test : np.ndarray, shape (n_test, n_features)
        Test data.
    train_states : list, optional
        Pre-computed training statevectors. If provided, avoids
        recomputing them (useful when K_train was already computed).
    backend : str, default="pennylane"
        Backend for simulation.

    Returns
    -------
    K : np.ndarray, shape (n_test, n_train)
        Kernel matrix where K[i,j] = |⟨φ(x_test_i)|φ(x_train_j)⟩|².

    Notes
    -----
    When using SVM with precomputed kernel for prediction, the test kernel
    matrix should have shape (n_test, n_train), where each row corresponds
    to a test sample and contains its kernel values with all training samples.
    """
    n_test = len(X_test)
    n_train = len(X_train)

    # Get training states if not provided
    if train_states is None:
        train_states = [
            simulate_encoding_state(encoding, x, backend) for x in X_train
        ]

    # Compute test-train kernel
    K = np.zeros((n_test, n_train), dtype=np.float64)

    for i, xi in enumerate(X_test):
        state_i = simulate_encoding_state(encoding, xi, backend)
        for j, state_j in enumerate(train_states):
            K[i, j] = compute_kernel_entry(state_i, state_j)

    return K


def kernel_target_alignment(
    K: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> float:
    """Compute kernel-target alignment.

    The kernel-target alignment measures how well the kernel matrix matches
    the ideal kernel induced by the labels. Higher alignment suggests the
    encoding is well-suited for the classification task.

    The alignment is defined as:

        A(K, y) = ⟨K, yy^T⟩_F / (||K||_F · ||yy^T||_F)

    where ⟨·,·⟩_F is the Frobenius inner product.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Kernel matrix.
    y : np.ndarray, shape (n,)
        Labels (will be converted to {-1, +1} for alignment computation).

    Returns
    -------
    float
        Alignment score in [-1, 1]. Higher is better.
        - A = 1: Kernel perfectly matches label structure
        - A = 0: Kernel is uncorrelated with labels
        - A = -1: Kernel is anti-correlated with labels

    Notes
    -----
    The ideal kernel for classification is the outer product yy^T, which
    has value +1 for same-class pairs and -1 for different-class pairs.
    Higher alignment indicates the quantum kernel naturally separates
    the classes.

    References
    ----------
    .. [1] Cristianini, N., et al. (2002). "On kernel-target alignment."
           Advances in Neural Information Processing Systems, 14.
    """
    # Convert labels to {-1, +1}
    y_array = np.asarray(y, dtype=np.float64)
    y_signed = 2.0 * y_array - 1.0

    # Ideal kernel: y @ y.T (outer product)
    y_outer = np.outer(y_signed, y_signed)

    # Frobenius inner product: ⟨K, yy^T⟩ = sum(K * yy^T)
    inner = np.sum(K * y_outer)

    # Frobenius norms
    norm_K = np.linalg.norm(K, "fro")
    norm_y = np.linalg.norm(y_outer, "fro")

    # Handle degenerate cases
    if norm_K < 1e-10 or norm_y < 1e-10:
        logger.warning(
            "Degenerate kernel or labels: norm_K=%.2e, norm_y=%.2e",
            norm_K,
            norm_y,
        )
        return 0.0

    alignment = inner / (norm_K * norm_y)
    return float(alignment)


def centered_kernel_target_alignment(
    K: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> float:
    """Compute *centered* kernel-target alignment (Cortes et al., 2012).

    The centered KTA removes the bias present in the uncentered version
    by projecting both the kernel matrix and the ideal kernel into the
    centred feature space before measuring alignment.  This is important
    for fidelity kernels where ``K[i, i] = 1`` always: the uncentered
    KTA inflates alignment for any kernel with high diagonal values, even
    an identity kernel with zero discriminative power (uncentered KTA of
    the identity kernel equals ``1 / sqrt(n)``).

    The centered alignment is defined as:

    .. math::
        A_c(K, y) = \\frac{\\langle H K H,\\, H y y^T H \\rangle_F}
                          {\\|H K H\\|_F \\;\\|H y y^T H\\|_F}

    where :math:`H = I - \\frac{1}{n} \\mathbf{1}\\mathbf{1}^T` is the
    centering matrix.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Kernel matrix (should be symmetric).
    y : np.ndarray, shape (n,)
        Labels (will be converted to {-1, +1} for alignment computation).

    Returns
    -------
    float
        Centered alignment score in [-1, 1].  Higher is better.

    References
    ----------
    .. [1] Cortes, C., Mohri, M., & Rostamizadeh, A. (2012).
           "Algorithms for learning kernels based on centered alignment."
           Journal of Machine Learning Research, 13, 795-828.

    See Also
    --------
    kernel_target_alignment : Uncentered variant (Cristianini et al., 2002).
    """
    n = K.shape[0]
    if n < 2:
        return 0.0

    # Convert labels to {-1, +1}
    y_array = np.asarray(y, dtype=np.float64)
    y_signed = 2.0 * y_array - 1.0

    # Ideal kernel: y @ y.T (outer product)
    y_outer = np.outer(y_signed, y_signed)

    # Centre both matrices via HMH where H = I - (1/n)*11^T.
    # Expanding HMH yields: M - row_mean - col_mean + grand_mean.
    K_c = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    y_c = y_outer - y_outer.mean(axis=0, keepdims=True) - y_outer.mean(axis=1, keepdims=True) + y_outer.mean()

    # Frobenius inner product and norms of the centred matrices
    inner = np.sum(K_c * y_c)
    norm_K = np.linalg.norm(K_c, "fro")
    norm_y = np.linalg.norm(y_c, "fro")

    # Handle degenerate cases (e.g. constant kernel or single-class labels)
    if norm_K < 1e-10 or norm_y < 1e-10:
        logger.warning(
            "Degenerate centred kernel or labels: norm_K=%.2e, norm_y=%.2e",
            norm_K,
            norm_y,
        )
        return 0.0

    alignment = inner / (norm_K * norm_y)
    return float(alignment)


def ensure_psd(
    K: NDArray[np.floating[Any]],
    epsilon: float = 1e-8,
) -> Tuple[NDArray[np.floating[Any]], bool]:
    """Ensure kernel matrix is positive semi-definite.

    Numerical noise in kernel computation can cause small negative
    eigenvalues, which makes the matrix invalid for SVM. This function
    clips negative eigenvalues to enforce PSD property.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Kernel matrix (should be symmetric).
    epsilon : float, default=1e-8
        Small positive value to replace negative eigenvalues.

    Returns
    -------
    K_psd : np.ndarray, shape (n, n)
        Positive semi-definite kernel matrix.
    was_modified : bool
        True if the matrix had negative eigenvalues that were clipped.

    Notes
    -----
    The procedure is:
    1. Symmetrize: K = (K + K^T) / 2
    2. Eigendecompose: K = V @ diag(λ) @ V^T
    3. Clip: λ_i = max(λ_i, epsilon) for all i
    4. Reconstruct: K_psd = V @ diag(λ_clipped) @ V^T

    This is the "nearest PSD matrix" in terms of eigenvalue modification.
    """
    # Ensure symmetry
    K_sym = (K + K.T) / 2.0

    # Eigendecomposition (eigh for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(K_sym)

    # Check for negative eigenvalues
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue >= -epsilon:
        # Already PSD (within tolerance)
        return K_sym, False

    # Log the correction
    n_negative = int(np.sum(eigenvalues < 0))
    logger.debug(
        "Kernel has %d negative eigenvalues (min=%.2e). Clipping to epsilon=%.2e.",
        n_negative,
        min_eigenvalue,
        epsilon,
    )

    # Clip negative eigenvalues
    eigenvalues_clipped = np.maximum(eigenvalues, epsilon)

    # Reconstruct PSD matrix
    K_psd = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T

    return K_psd, True


def compute_kernel_statistics(
    K: NDArray[np.floating[Any]],
) -> dict[str, float]:
    """Compute summary statistics of a kernel matrix.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Kernel matrix.

    Returns
    -------
    dict
        Dictionary containing:
        - mean: Mean of off-diagonal elements
        - std: Standard deviation of off-diagonal elements
        - min: Minimum off-diagonal value
        - max: Maximum off-diagonal value
        - rank: Numerical rank (count of eigenvalues > 1e-10)
        - trace: Trace of the matrix (should be n for normalized kernel)
    """
    n = K.shape[0]

    # Extract off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    off_diag = K[mask]

    # Compute eigenvalues for rank
    eigenvalues = np.linalg.eigvalsh(K)
    rank = int(np.sum(np.abs(eigenvalues) > 1e-10))

    return {
        "mean": float(np.mean(off_diag)),
        "std": float(np.std(off_diag)),
        "min": float(np.min(off_diag)),
        "max": float(np.max(off_diag)),
        "rank": rank,
        "trace": float(np.trace(K)),
    }
