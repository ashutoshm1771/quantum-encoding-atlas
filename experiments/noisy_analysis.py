"""Noisy analysis functions for Stage 5b.

This module provides noisy versions of expressibility and entanglement
analysis functions that use density matrix simulation with depolarizing
noise channels.

Unlike the library's analysis functions (which use pure-state simulation),
these functions use PennyLane's ``default.mixed`` device to simulate
quantum channels and compute metrics for mixed states.

Key Differences from Pure-State Analysis
----------------------------------------
1. **Expressibility**: Uses fidelity between density matrices instead of
   pure state overlap. For mixed states:
   F(ρ, σ) = (Tr√(√ρ σ √ρ))²

2. **Entanglement**: Meyer-Wallach is defined for pure states only. For
   mixed states, we use the linear entropy of reduced density matrices
   as a proxy measure.

3. **Fidelity Decay**: Compares ideal (pure) states to noisy (mixed) states
   using the state fidelity measure.

Memory Considerations
---------------------
Density matrices are 2^n × 2^n complex128 values (total 4^n entries),
compared to 2^n entries for statevectors. Memory usage (complex128,
16 bytes per entry):

- 4 qubits:  16 × 16 density matrix       = 4 KB    (statevector: 256 B)
- 6 qubits:  64 × 64 density matrix       = 64 KB   (statevector: 1 KB)
- 8 qubits:  256 × 256 density matrix     = 1 MB    (statevector: 4 KB)
- 10 qubits: 1024 × 1024 density matrix   = 16 MB   (statevector: 16 KB)
- 12 qubits: 4096 × 4096 density matrix   = 256 MB  (statevector: 64 KB)

Keep n_qubits ≤ 8 strictly for noisy simulation.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
import scipy.linalg

from experiments.noise_models import (
    NOISE_LEVELS,
    execute_noisy_circuit,
    execute_ideal_circuit,
    get_noise_params,
)

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

__all__ = [
    "compute_noisy_expressibility",
    "compute_noisy_entanglement",
    "compute_fidelity_decay",
]

_logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Numerical stability constant
_EPSILON: float = 1e-15

# Maximum qubits for density matrix simulation
_MAX_NOISY_SIMULATION_QUBITS: int = 8

# Default sample counts (reduced from pure-state analysis due to slower simulation)
_DEFAULT_N_SAMPLES_EXPRESSIBILITY: int = 1000
_DEFAULT_N_SAMPLES_ENTANGLEMENT: int = 500
_DEFAULT_N_SAMPLES_FIDELITY: int = 100

# Minimum sample counts
_MIN_SAMPLES_WARNING: int = 50
_MIN_SAMPLES_ERROR: int = 10


# =============================================================================
# Fidelity Computation for Mixed States
# =============================================================================


def compute_mixed_state_fidelity(
    rho: NDArray[np.complexfloating[Any, Any]],
    sigma: NDArray[np.complexfloating[Any, Any]],
) -> float:
    """Compute the fidelity between two density matrices.

    For mixed states, fidelity is defined as:
        F(ρ, σ) = (Tr√(√ρ σ √ρ))²

    For a pure state |ψ⟩ (ρ = |ψ⟩⟨ψ|) and a density matrix σ:
        F(|ψ⟩⟨ψ|, σ) = ⟨ψ|σ|ψ⟩

    Parameters
    ----------
    rho : NDArray[np.complexfloating]
        First density matrix.
    sigma : NDArray[np.complexfloating]
        Second density matrix.

    Returns
    -------
    float
        Fidelity value in [0, 1].

    Notes
    -----
    This uses scipy's matrix square root function which can be slow
    for large matrices. For pure states, the computation simplifies
    significantly.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.complex128)

    # Check for pure state optimization
    # A pure state has Tr(ρ²) = 1
    rho_purity = np.real(np.trace(rho @ rho))

    if np.isclose(rho_purity, 1.0, atol=1e-10):
        # rho is pure: F = ⟨ψ|σ|ψ⟩
        # Find the dominant eigenvector of rho
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        # Get the eigenvector with eigenvalue ≈ 1
        idx = np.argmax(eigenvalues)
        psi = eigenvectors[:, idx]
        fidelity = np.real(np.conj(psi) @ sigma @ psi)
        return float(np.clip(fidelity, 0.0, 1.0))

    # General case: F = (Tr√(√ρ σ √ρ))²
    try:
        sqrt_rho = scipy.linalg.sqrtm(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho

        # Handle potential numerical issues
        if np.any(np.isnan(inner)) or np.any(np.isinf(inner)):
            _logger.warning("Numerical issues in fidelity computation, using fallback")
            return 0.5  # Return neutral value

        sqrt_inner = scipy.linalg.sqrtm(inner)
        trace_val = np.real(np.trace(sqrt_inner))
        fidelity = trace_val ** 2

    except (np.linalg.LinAlgError, ValueError) as e:
        _logger.warning("Matrix square root failed: %s, using fallback", e)
        # Fallback: use trace overlap as approximation
        fidelity = np.real(np.trace(rho @ sigma))

    return float(np.clip(fidelity, 0.0, 1.0))


def compute_pure_mixed_fidelity(
    statevector: NDArray[np.complexfloating[Any, Any]],
    rho: NDArray[np.complexfloating[Any, Any]],
) -> float:
    """Compute fidelity between a pure state and a density matrix.

    For pure state |ψ⟩ and density matrix ρ:
        F(|ψ⟩, ρ) = ⟨ψ|ρ|ψ⟩

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        Pure state vector of shape (d,).
    rho : NDArray[np.complexfloating]
        Density matrix of shape (d, d).

    Returns
    -------
    float
        Fidelity value in [0, 1].
    """
    psi = np.asarray(statevector, dtype=np.complex128).ravel()
    rho = np.asarray(rho, dtype=np.complex128)

    # F = ⟨ψ|ρ|ψ⟩
    fidelity = np.real(np.conj(psi) @ rho @ psi)
    return float(np.clip(fidelity, 0.0, 1.0))


# =============================================================================
# Linear Entropy for Mixed States
# =============================================================================


def compute_linear_entropy(
    rho: NDArray[np.complexfloating[Any, Any]],
) -> float:
    """Compute the linear entropy of a density matrix.

    Linear entropy is defined as:
        S_L(ρ) = 1 - Tr(ρ²)

    This measures the "mixedness" of a quantum state:
    - S_L = 0 for pure states
    - S_L = 1 - 1/d for maximally mixed states of dimension d

    Parameters
    ----------
    rho : NDArray[np.complexfloating]
        Density matrix.

    Returns
    -------
    float
        Linear entropy in [0, 1 - 1/d].
    """
    rho = np.asarray(rho, dtype=np.complex128)
    purity = np.real(np.trace(rho @ rho))
    return float(max(0.0, 1.0 - purity))


def partial_trace_single_qubit(
    rho: NDArray[np.complexfloating[Any, Any]],
    n_qubits: int,
    keep_qubit: int,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Compute partial trace of a density matrix, keeping one qubit.

    Parameters
    ----------
    rho : NDArray[np.complexfloating]
        Density matrix of shape (2^n_qubits, 2^n_qubits).
    n_qubits : int
        Total number of qubits.
    keep_qubit : int
        Index of qubit to keep (0-indexed, MSB first).

    Returns
    -------
    NDArray[np.complexfloating]
        Reduced density matrix of shape (2, 2).
    """
    dim = 2 ** n_qubits
    rho = np.asarray(rho, dtype=np.complex128)

    if rho.shape != (dim, dim):
        raise ValueError(
            f"Density matrix shape {rho.shape} doesn't match "
            f"{n_qubits} qubits (expected {(dim, dim)})"
        )

    # Reshape to tensor form
    tensor_shape = [2] * (2 * n_qubits)
    rho_tensor = rho.reshape(tensor_shape)

    # Build einsum contraction to trace out all but keep_qubit
    # Input indices: 0..n-1 for row, n..2n-1 for column
    # For traced qubits, set column index = row index
    # For kept qubit, keep row and column separate

    indices_in = list(range(2 * n_qubits))
    for q in range(n_qubits):
        if q != keep_qubit:
            indices_in[n_qubits + q] = q  # Contract row with column

    indices_out = [keep_qubit, keep_qubit + n_qubits]

    # Convert to einsum letters
    def idx_to_letter(idx: int) -> str:
        return chr(ord("a") + idx)

    input_str = "".join(idx_to_letter(i) for i in indices_in)
    output_str = "".join(idx_to_letter(i) for i in indices_out)

    rho_reduced = np.einsum(f"{input_str}->{output_str}", rho_tensor)
    rho_reduced = rho_reduced.reshape(2, 2).astype(np.complex128)

    # Normalize trace to 1
    trace_val = np.trace(rho_reduced)
    if np.abs(trace_val) > _EPSILON:
        rho_reduced = rho_reduced / trace_val

    return rho_reduced


# =============================================================================
# Noisy Expressibility
# =============================================================================


def compute_noisy_expressibility(
    encoding: "BaseEncoding",
    noise_level: str,
    n_samples: int = _DEFAULT_N_SAMPLES_EXPRESSIBILITY,
    n_bins: int = 75,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute expressibility using density matrix simulation with noise.

    This function computes expressibility by sampling fidelities between
    noisy encoded states (density matrices) and comparing to the Haar
    distribution, similar to the pure-state version but with noise.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    noise_level : str
        One of "low", "medium", or "high".
    n_samples : int, default=1000
        Number of random input pairs to sample.
    n_bins : int, default=75
        Number of histogram bins for KL divergence.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - noisy_expressibility: float in [0, 1]
        - noisy_kl_divergence: raw KL divergence
        - noisy_mean_fidelity: mean fidelity between sampled pairs
        - noisy_std_fidelity: std of fidelities
        - noise_level: the noise level used
        - noise_params: the noise parameters used
        - n_samples: number of samples used
        - status: "success" or "noise_failure"

    Notes
    -----
    For mixed states, fidelity is computed using the Uhlmann fidelity:
    F(ρ, σ) = (Tr√(√ρ σ √ρ))²

    This is slower than pure-state fidelity due to matrix square roots.
    """
    # Validate inputs
    n_qubits = encoding.n_qubits
    n_features = encoding.n_features

    if n_qubits > _MAX_NOISY_SIMULATION_QUBITS:
        raise ValueError(
            f"Noisy simulation limited to {_MAX_NOISY_SIMULATION_QUBITS} qubits, "
            f"but encoding has {n_qubits} qubits."
        )

    if n_samples < _MIN_SAMPLES_ERROR:
        raise ValueError(
            f"n_samples must be at least {_MIN_SAMPLES_ERROR}, got {n_samples}"
        )

    if n_samples < _MIN_SAMPLES_WARNING:
        warnings.warn(
            f"n_samples={n_samples} is low for reliable results. "
            f"Recommended minimum: {_MIN_SAMPLES_WARNING}.",
            UserWarning,
            stacklevel=2,
        )

    noise_params = get_noise_params(noise_level)
    rng = np.random.default_rng(seed)

    _logger.info(
        "Computing noisy expressibility for %s (n_qubits=%d, noise=%s, n_samples=%d)",
        encoding.__class__.__name__,
        n_qubits,
        noise_level,
        n_samples,
    )

    # Sample fidelities between random pairs
    fidelities = np.zeros(n_samples, dtype=np.float64)
    failed_samples = 0

    for i in range(n_samples):
        # Generate two random inputs
        x1 = rng.uniform(0.0, 2 * np.pi, size=n_features)
        x2 = rng.uniform(0.0, 2 * np.pi, size=n_features)

        try:
            # Execute noisy circuits
            rho1 = execute_noisy_circuit(encoding, x1, noise_params)
            rho2 = execute_noisy_circuit(encoding, x2, noise_params)

            # Check for non-physical density matrices
            if _is_density_matrix_invalid(rho1) or _is_density_matrix_invalid(rho2):
                _logger.debug("Sample %d: non-physical density matrix, skipping", i)
                fidelities[i] = np.nan
                failed_samples += 1
                continue

            # Compute fidelity between density matrices
            fidelity = compute_mixed_state_fidelity(rho1, rho2)
            fidelities[i] = fidelity

        except Exception as e:
            _logger.debug("Sample %d failed: %s", i, e)
            fidelities[i] = np.nan
            failed_samples += 1

    # Filter out failed samples
    valid_fidelities = fidelities[~np.isnan(fidelities)]

    if len(valid_fidelities) < _MIN_SAMPLES_ERROR:
        _logger.warning(
            "Only %d valid samples (out of %d), returning noise_failure",
            len(valid_fidelities),
            n_samples,
        )
        return {
            "noisy_expressibility": 0.0,
            "noisy_kl_divergence": float("inf"),
            "noisy_mean_fidelity": 0.0,
            "noisy_std_fidelity": 0.0,
            "noise_level": noise_level,
            "noise_params": noise_params,
            "n_samples": n_samples,
            "valid_samples": len(valid_fidelities),
            "status": "noise_failure",
        }

    # Compute statistics
    mean_fidelity = float(np.mean(valid_fidelities))
    std_fidelity = float(np.std(valid_fidelities, ddof=1))

    # Build histogram and compute KL divergence
    hist, bin_edges = np.histogram(
        valid_fidelities, bins=n_bins, range=(0.0, 1.0), density=True
    )
    bin_width = bin_edges[1] - bin_edges[0]
    P_encoding = hist * bin_width
    P_encoding = P_encoding / (P_encoding.sum() + _EPSILON)

    # Haar distribution for n_qubits
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    P_haar = _compute_haar_distribution(n_qubits, bin_centers)

    # KL divergence
    P_encoding_safe = P_encoding + _EPSILON
    P_haar_safe = P_haar + _EPSILON
    P_encoding_safe = P_encoding_safe / P_encoding_safe.sum()
    P_haar_safe = P_haar_safe / P_haar_safe.sum()

    kl_divergence = float(np.sum(P_encoding_safe * np.log(P_encoding_safe / P_haar_safe)))
    kl_divergence = max(0.0, kl_divergence)

    # Normalize to expressibility score
    MAX_KL = 10.0
    expressibility = 1.0 - min(1.0, kl_divergence / MAX_KL)

    return {
        "noisy_expressibility": float(expressibility),
        "noisy_kl_divergence": float(kl_divergence),
        "noisy_mean_fidelity": mean_fidelity,
        "noisy_std_fidelity": std_fidelity,
        "noise_level": noise_level,
        "noise_params": noise_params,
        "n_samples": n_samples,
        "valid_samples": len(valid_fidelities),
        "failed_samples": failed_samples,
        "status": "success",
    }


# =============================================================================
# Noisy Entanglement
# =============================================================================


def compute_noisy_entanglement(
    encoding: "BaseEncoding",
    noise_level: str,
    n_samples: int = _DEFAULT_N_SAMPLES_ENTANGLEMENT,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute entanglement capability with noise using linear entropy.

    For mixed states, the Meyer-Wallach measure is not directly applicable.
    Instead, we use the average linear entropy of single-qubit reduced
    density matrices as a proxy measure for entanglement.

    The linear entropy S_L(ρ) = 1 - Tr(ρ²) measures mixedness. For the
    reduced state of qubit i after tracing out all other qubits, higher
    linear entropy indicates that qubit i is more entangled with the rest.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    noise_level : str
        One of "low", "medium", or "high".
    n_samples : int, default=500
        Number of random inputs to sample.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - noisy_entanglement: float in [0, 1]
        - noisy_linear_entropy_avg: average linear entropy
        - per_qubit_entropy: linear entropy per qubit
        - noise_level: the noise level used
        - noise_params: the noise parameters used
        - status: "success" or "noise_failure"

    Notes
    -----
    Unlike the pure-state Meyer-Wallach measure, this mixed-state measure
    can be affected by both entanglement AND noise-induced mixing. High
    noise can increase the linear entropy even without entanglement.
    """
    n_qubits = encoding.n_qubits
    n_features = encoding.n_features

    if n_qubits > _MAX_NOISY_SIMULATION_QUBITS:
        raise ValueError(
            f"Noisy simulation limited to {_MAX_NOISY_SIMULATION_QUBITS} qubits, "
            f"but encoding has {n_qubits} qubits."
        )

    if n_qubits < 2:
        # Single qubit cannot have entanglement
        return {
            "noisy_entanglement": 0.0,
            "noisy_linear_entropy_avg": 0.0,
            "per_qubit_entropy": [0.0],
            "noise_level": noise_level,
            "noise_params": get_noise_params(noise_level),
            "n_samples": n_samples,
            "status": "success",
            "note": "Single qubit cannot have entanglement",
        }

    noise_params = get_noise_params(noise_level)
    rng = np.random.default_rng(seed)

    _logger.info(
        "Computing noisy entanglement for %s (n_qubits=%d, noise=%s, n_samples=%d)",
        encoding.__class__.__name__,
        n_qubits,
        noise_level,
        n_samples,
    )

    # Accumulate entanglement measures
    entanglement_samples = np.zeros(n_samples, dtype=np.float64)
    per_qubit_sum = np.zeros(n_qubits, dtype=np.float64)
    failed_samples = 0

    for i in range(n_samples):
        x = rng.uniform(0.0, 2 * np.pi, size=n_features)

        try:
            # Execute noisy circuit
            rho = execute_noisy_circuit(encoding, x, noise_params)

            if _is_density_matrix_invalid(rho):
                _logger.debug("Sample %d: non-physical density matrix, skipping", i)
                entanglement_samples[i] = np.nan
                failed_samples += 1
                continue

            # Compute linear entropy for each qubit's reduced state
            per_qubit_entropy = np.zeros(n_qubits, dtype=np.float64)
            for q in range(n_qubits):
                rho_q = partial_trace_single_qubit(rho, n_qubits, q)
                per_qubit_entropy[q] = compute_linear_entropy(rho_q)

            # Average linear entropy (similar to Meyer-Wallach normalization)
            # Max linear entropy for a qubit is 0.5 (maximally mixed)
            # Normalize by (2/n_qubits) × sum, capped at 1.0
            avg_entropy = np.mean(per_qubit_entropy)
            # Scale to [0, 1]: max per-qubit entropy is 0.5, so multiply by 2
            entanglement = min(1.0, 2.0 * avg_entropy)

            entanglement_samples[i] = entanglement
            per_qubit_sum += per_qubit_entropy

        except Exception as e:
            _logger.debug("Sample %d failed: %s", i, e)
            entanglement_samples[i] = np.nan
            failed_samples += 1

    # Filter valid samples
    valid_samples = entanglement_samples[~np.isnan(entanglement_samples)]

    if len(valid_samples) < _MIN_SAMPLES_ERROR:
        return {
            "noisy_entanglement": 0.0,
            "noisy_linear_entropy_avg": 0.0,
            "per_qubit_entropy": [0.0] * n_qubits,
            "noise_level": noise_level,
            "noise_params": noise_params,
            "n_samples": n_samples,
            "valid_samples": len(valid_samples),
            "status": "noise_failure",
        }

    # Compute final statistics
    noisy_entanglement = float(np.mean(valid_samples))
    per_qubit_avg = per_qubit_sum / len(valid_samples)

    return {
        "noisy_entanglement": noisy_entanglement,
        "noisy_linear_entropy_avg": float(np.mean(per_qubit_avg)),
        "per_qubit_entropy": per_qubit_avg.tolist(),
        "std_error": float(np.std(valid_samples, ddof=1) / np.sqrt(len(valid_samples))),
        "noise_level": noise_level,
        "noise_params": noise_params,
        "n_samples": n_samples,
        "valid_samples": len(valid_samples),
        "failed_samples": failed_samples,
        "status": "success",
    }


# =============================================================================
# Fidelity Decay
# =============================================================================


def compute_fidelity_decay(
    encoding: "BaseEncoding",
    noise_level: str,
    n_samples: int = _DEFAULT_N_SAMPLES_FIDELITY,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute fidelity decay between ideal and noisy states.

    This measures how much the quantum state degrades due to noise by
    computing the fidelity between ideal (noiseless) states and noisy
    (with depolarizing channels) states for the same inputs.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    noise_level : str
        One of "low", "medium", or "high".
    n_samples : int, default=100
        Number of random inputs to sample.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - mean_fidelity: average fidelity between ideal and noisy
        - std_fidelity: standard deviation of fidelities
        - min_fidelity: minimum observed fidelity
        - max_fidelity: maximum observed fidelity
        - fidelity_decay: 1 - mean_fidelity (how much fidelity is lost)
        - noise_level: the noise level used
        - noise_params: the noise parameters used
        - status: "success" or "noise_failure"
    """
    n_qubits = encoding.n_qubits
    n_features = encoding.n_features

    if n_qubits > _MAX_NOISY_SIMULATION_QUBITS:
        raise ValueError(
            f"Noisy simulation limited to {_MAX_NOISY_SIMULATION_QUBITS} qubits, "
            f"but encoding has {n_qubits} qubits."
        )

    noise_params = get_noise_params(noise_level)
    rng = np.random.default_rng(seed)

    _logger.info(
        "Computing fidelity decay for %s (n_qubits=%d, noise=%s, n_samples=%d)",
        encoding.__class__.__name__,
        n_qubits,
        noise_level,
        n_samples,
    )

    fidelities = np.zeros(n_samples, dtype=np.float64)
    failed_samples = 0

    for i in range(n_samples):
        x = rng.uniform(0.0, 2 * np.pi, size=n_features)

        try:
            # Execute ideal (noiseless) circuit
            psi_ideal = execute_ideal_circuit(encoding, x)

            # Execute noisy circuit
            rho_noisy = execute_noisy_circuit(encoding, x, noise_params)

            if _is_density_matrix_invalid(rho_noisy):
                _logger.debug("Sample %d: non-physical density matrix, skipping", i)
                fidelities[i] = np.nan
                failed_samples += 1
                continue

            # Compute fidelity between pure ideal state and mixed noisy state
            fidelity = compute_pure_mixed_fidelity(psi_ideal, rho_noisy)
            fidelities[i] = fidelity

        except Exception as e:
            _logger.debug("Sample %d failed: %s", i, e)
            fidelities[i] = np.nan
            failed_samples += 1

    # Filter valid samples
    valid_fidelities = fidelities[~np.isnan(fidelities)]

    if len(valid_fidelities) < _MIN_SAMPLES_ERROR:
        return {
            "mean_fidelity": 0.0,
            "std_fidelity": 0.0,
            "min_fidelity": 0.0,
            "max_fidelity": 0.0,
            "fidelity_decay": 1.0,
            "noise_level": noise_level,
            "noise_params": noise_params,
            "n_samples": n_samples,
            "valid_samples": len(valid_fidelities),
            "status": "noise_failure",
        }

    mean_fidelity = float(np.mean(valid_fidelities))
    std_fidelity = float(np.std(valid_fidelities, ddof=1))
    min_fidelity = float(np.min(valid_fidelities))
    max_fidelity = float(np.max(valid_fidelities))

    return {
        "mean_fidelity": mean_fidelity,
        "std_fidelity": std_fidelity,
        "min_fidelity": min_fidelity,
        "max_fidelity": max_fidelity,
        "fidelity_decay": 1.0 - mean_fidelity,
        "std_error": std_fidelity / np.sqrt(len(valid_fidelities)),
        "noise_level": noise_level,
        "noise_params": noise_params,
        "n_samples": n_samples,
        "valid_samples": len(valid_fidelities),
        "failed_samples": failed_samples,
        "status": "success",
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_haar_distribution(
    n_qubits: int,
    fidelity_values: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the Haar-random fidelity distribution.

    For d-dimensional Hilbert space (d = 2^n_qubits), the PDF of fidelities
    between Haar-random states is:
        P_Haar(F) = (d - 1)(1 - F)^(d - 2)

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    fidelity_values : NDArray[np.floating]
        Array of fidelity values at which to evaluate.

    Returns
    -------
    NDArray[np.floating]
        Probability values, normalized to sum to 1.
    """
    d = 2 ** n_qubits
    F = np.asarray(fidelity_values, dtype=np.float64)

    if d == 2:
        # Single qubit: uniform distribution
        P_haar = np.ones_like(F)
    else:
        one_minus_F = np.clip(1.0 - F, _EPSILON, 1.0)
        if d > 100:
            # Use log-space for numerical stability
            log_P = np.log(d - 1) + (d - 2) * np.log(one_minus_F)
            log_P_max = np.max(log_P)
            P_haar = np.exp(log_P - log_P_max)
        else:
            P_haar = (d - 1) * np.power(one_minus_F, d - 2)

    P_haar = np.nan_to_num(P_haar, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    P_sum = P_haar.sum()
    if P_sum > _EPSILON:
        P_haar = P_haar / P_sum
    else:
        P_haar = np.ones_like(P_haar) / len(P_haar)

    return P_haar


def _is_density_matrix_invalid(
    rho: NDArray[np.complexfloating[Any, Any]],
) -> bool:
    """Check if a density matrix is non-physical.

    A valid density matrix must be:
    1. Hermitian: ρ = ρ†
    2. Positive semi-definite: all eigenvalues ≥ 0
    3. Unit trace: Tr(ρ) = 1

    Noise can cause numerical issues leading to non-physical states.

    Parameters
    ----------
    rho : NDArray[np.complexfloating]
        Density matrix to check.

    Returns
    -------
    bool
        True if the density matrix is invalid, False otherwise.
    """
    # Check for NaN or Inf
    if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
        return True

    # Check trace
    trace = np.real(np.trace(rho))
    if not np.isclose(trace, 1.0, atol=0.1):  # Allow some tolerance
        return True

    # Check for negative eigenvalues (sign of non-physical state)
    try:
        eigenvalues = np.linalg.eigvalsh(rho)
        min_eigenvalue = np.min(eigenvalues)
        if min_eigenvalue < -0.1:  # Allow small negative values from numerical error
            return True
    except np.linalg.LinAlgError:
        return True

    return False
