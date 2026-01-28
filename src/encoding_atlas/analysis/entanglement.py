"""Entanglement capability analysis for quantum encodings.

This module implements entanglement capability computation using the
Meyer-Wallach measure as defined in Meyer & Wallach (2002).

Entanglement capability quantifies the average amount of entanglement
produced by a quantum encoding across random inputs. This is a key metric
for understanding the computational expressiveness of quantum circuits.

Mathematical Background
-----------------------
The Meyer-Wallach entanglement measure Q(|psi>) quantifies the average
bipartite entanglement across all qubit bipartitions:

    Q(|psi>) = (2/n) * sum_i (1 - Tr(rho_i^2))

where:
    - n is the number of qubits
    - rho_i is the reduced density matrix of qubit i after tracing out
      all other qubits
    - Tr(rho_i^2) is the purity of that reduced state

Properties:
    - Q = 0: Completely separable (product state, no entanglement)
    - Q = 1: Maximally entangled (all qubits maximally mixed when traced out)

Entanglement capability is the expected value of this measure over
random input states:

    Ent_cap = E_x[Q(|psi(x)>)]

The Scott measure generalizes Meyer-Wallach to k-body reduced states,
providing additional insight into the entanglement structure.

References
----------
.. [1] Meyer, D. A., & Wallach, N. R. (2002).
       "Global entanglement in multiparticle systems."
       Journal of Mathematical Physics, 43(9), 4273-4278.
       DOI: 10.1063/1.1497700

.. [2] Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019).
       "Expressibility and entangling capability of parameterized quantum
       circuits for hybrid quantum-classical algorithms."
       Advanced Quantum Technologies, 2(12), 1900070.
       DOI: 10.1002/qute.201900070

.. [3] Brennen, G. K. (2003).
       "An observable measure of entanglement for pure states of
       multi-qubit systems."
       Quantum Information and Computation, 3(6), 619-626.

.. [4] Scott, A. J. (2004).
       "Multipartite entanglement, quantum-error-correcting codes, and
       entangling power of quantum evolutions."
       Physical Review A, 69(5), 052330.
       DOI: 10.1103/PhysRevA.69.052330

Examples
--------
Basic usage with an entangling encoding:

>>> from encoding_atlas import IQPEncoding
>>> from encoding_atlas.analysis import compute_entanglement_capability
>>>
>>> enc = IQPEncoding(n_features=4, reps=2)
>>> result = compute_entanglement_capability(enc, n_samples=1000, seed=42)
>>> print(f"Entanglement capability: {result:.4f}")

Comparing different entanglement patterns:

>>> enc_linear = IQPEncoding(n_features=4, reps=2, entanglement="linear")
>>> enc_full = IQPEncoding(n_features=4, reps=2, entanglement="full")
>>> ent_linear = compute_entanglement_capability(enc_linear, seed=42)
>>> ent_full = compute_entanglement_capability(enc_full, seed=42)
>>> print(f"Linear: {ent_linear:.4f}, Full: {ent_full:.4f}")

Getting detailed results:

>>> result = compute_entanglement_capability(
...     enc, n_samples=500, seed=42, return_details=True
... )
>>> print(f"Mean: {result['entanglement_capability']:.4f}")
>>> print(f"Std Error: {result['std_error']:.4f}")

Computing Meyer-Wallach measure for a specific state:

>>> import numpy as np
>>> from encoding_atlas.analysis.entanglement import compute_meyer_wallach
>>>
>>> # GHZ state: (|000> + |111>) / sqrt(2)
>>> ghz = np.zeros(8, dtype=complex)
>>> ghz[0] = ghz[7] = 1.0 / np.sqrt(2)
>>> mw = compute_meyer_wallach(ghz, n_qubits=3)
>>> print(f"GHZ state MW: {mw:.4f}")  # Should be 1.0

Notes
-----
For non-entangling encodings (e.g., AngleEncoding without entangling layers),
the entanglement capability will be exactly 0.0, as all generated states
are product states.

The computation requires simulating the quantum circuit for each sample,
which can be computationally expensive for large qubit counts. For systems
with more than 15 qubits, consider using fewer samples or the Scott measure
with smaller subsystem sizes.

See Also
--------
encoding_atlas.analysis.expressibility : Related metric for Hilbert space coverage.
encoding_atlas.analysis._utils.partial_trace_single_qubit : Core partial trace operation.
encoding_atlas.analysis._utils.compute_purity : Purity computation for reduced states.
"""

from __future__ import annotations

import logging
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Union, overload

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.exceptions import (
    AnalysisError,
    InsufficientSamplesError,
    NumericalInstabilityError,
    SimulationError,
    ValidationError,
)
from encoding_atlas.analysis._utils import (
    compute_purity,
    create_rng,
    partial_trace_single_qubit,
    partial_trace_subsystem,
    simulate_encoding_statevector,
    validate_encoding_for_analysis,
    validate_statevector,
)


# =============================================================================
# Type Checking Imports and Type Definitions
# =============================================================================

if TYPE_CHECKING:
    pass

# Type aliases for clarity
StatevectorType = NDArray[np.complexfloating[Any, Any]]
"""Complex array representing a quantum statevector, shape (2^n_qubits,)."""

FloatArray = NDArray[np.floating[Any]]
"""Array of floating-point values."""


class EntanglementResult(TypedDict):
    """Result dictionary from entanglement capability computation.

    Attributes
    ----------
    entanglement_capability : float
        Average entanglement measure in [0, 1]. Higher values indicate
        the encoding creates more entangled states on average.
    entanglement_samples : NDArray[np.floating]
        Array of entanglement measure values for each sampled input.
        Shape: (n_samples,). The measure type depends on the ``measure``
        parameter used (Meyer-Wallach or Scott).
    std_error : float
        Standard error of the mean, computed as std / sqrt(n_samples).
        Useful for assessing the reliability of the estimate.
    n_samples : int
        Number of random samples used in the computation.
    per_qubit_entanglement : NDArray[np.floating]
        Average linear entropy contribution from each qubit.
        Shape: (n_qubits,). Useful for identifying which qubits
        are most entangled. Note: Only meaningful for Meyer-Wallach
        measure; returns zeros for Scott measure.
    measure : str
        The entanglement measure used: ``"meyer_wallach"`` or ``"scott"``.
    scott_k : int | None
        The k value used for Scott measure, or None if Meyer-Wallach was used.
    """

    entanglement_capability: float
    entanglement_samples: FloatArray
    std_error: float
    n_samples: int
    per_qubit_entanglement: FloatArray
    measure: str
    scott_k: int | None


# =============================================================================
# Module-Level Configuration
# =============================================================================

__all__ = [
    # Main public function
    "compute_entanglement_capability",
    # Supporting functions for direct state analysis
    "compute_meyer_wallach",
    "compute_meyer_wallach_with_breakdown",
    "compute_scott_measure",
    # Type exports
    "EntanglementResult",
]

_logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Default number of random samples for entanglement capability estimation.
#
# RATIONALE: Entanglement capability uses FEWER samples than expressibility
# (which uses 5000) but MORE than trainability (which uses 500) because:
#
# 1. POINT ESTIMATE: Unlike expressibility (histogram-based), entanglement
#    capability is a simple average of Meyer-Wallach values. Point estimates
#    converge faster than histogram estimates.
#
# 2. VARIANCE REDUCTION: The Meyer-Wallach measure Q ∈ [0, 1] is bounded,
#    so its variance is limited. Central Limit Theorem guarantees standard
#    error ≈ σ/√n. With n=1000 samples, error ≈ σ/32.
#
# 3. COMPUTATIONAL COST: Each sample requires 1 circuit simulation plus
#    n_qubits partial trace computations. This is more expensive than
#    expressibility per sample but no gradients are needed.
#
# 4. LITERATURE STANDARD: Entanglement capability studies typically use
#    100-1000 samples (Sim et al., 2019; Meyer & Wallach, 2002).
#
# 1000 samples provides a good balance between accuracy (SE ≈ 3% for typical
# variance) and computational cost.
_DEFAULT_N_SAMPLES: int = 1000

# Minimum samples for meaningful statistics (warning threshold).
# Below 100 samples, the standard error exceeds 10% of the typical range,
# making estimates unreliable for comparison purposes.
_MIN_SAMPLES_WARNING: int = 100

# Absolute minimum samples (error threshold).
# Below 10 samples, statistical estimates are meaningless and may not
# capture the entanglement structure of the encoding at all.
_MIN_SAMPLES_ERROR: int = 10

# Default input range for random sampling (covers full rotation range)
_DEFAULT_INPUT_RANGE: tuple[float, float] = (0.0, 2.0 * np.pi)

# Numerical stability epsilon for floating-point comparisons
_EPSILON: float = 1e-15

# Maximum number of qubits for which detailed logging is practical
_MAX_VERBOSE_QUBITS: int = 10


# =============================================================================
# Main Public Function
# =============================================================================


@overload
def compute_entanglement_capability(
    encoding: BaseEncoding,
    n_samples: int = ...,
    input_range: tuple[float, float] = ...,
    seed: int | None = ...,
    backend: Literal["pennylane", "qiskit"] = ...,
    measure: Literal["meyer_wallach", "scott"] = ...,
    scott_k: int | None = ...,
    return_details: Literal[False] = ...,
    verbose: bool = ...,
) -> float: ...


@overload
def compute_entanglement_capability(
    encoding: BaseEncoding,
    n_samples: int = ...,
    input_range: tuple[float, float] = ...,
    seed: int | None = ...,
    backend: Literal["pennylane", "qiskit"] = ...,
    measure: Literal["meyer_wallach", "scott"] = ...,
    scott_k: int | None = ...,
    return_details: Literal[True] = ...,
    verbose: bool = ...,
) -> EntanglementResult: ...


def compute_entanglement_capability(
    encoding: BaseEncoding,
    n_samples: int = _DEFAULT_N_SAMPLES,
    input_range: tuple[float, float] = _DEFAULT_INPUT_RANGE,
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit"] = "pennylane",
    measure: Literal["meyer_wallach", "scott"] = "meyer_wallach",
    scott_k: int | None = None,
    return_details: bool = False,
    verbose: bool = False,
) -> Union[float, EntanglementResult]:
    """Compute the entanglement capability of a quantum encoding.

    Entanglement capability quantifies the average amount of entanglement
    produced by the encoding across random inputs. Higher values indicate
    the encoding creates more entangled states on average.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to analyze. Must implement ``get_circuit()``.
    n_samples : int, default=1000
        Number of random inputs to sample for averaging. Higher values
        give more accurate estimates but increase computation time.
        Must be at least 10.
    input_range : tuple[float, float], default=(0, 2*pi)
        Range for sampling random input values. Features are sampled
        uniformly from this range.
    seed : int, optional
        Random seed for reproducibility. If None, uses system entropy.
    backend : {"pennylane", "qiskit"}, default="pennylane"
        Backend for circuit simulation.
    measure : {"meyer_wallach", "scott"}, default="meyer_wallach"
        Entanglement measure to use:

        - ``"meyer_wallach"``: Average single-qubit purity deficit.
          Faster and most commonly used. Equivalent to Scott measure with k=1.
        - ``"scott"``: Scott measure with configurable k-body subsystems.
          More detailed but slower. Use ``scott_k`` to specify subsystem size.

    scott_k : int, optional
        Subsystem size for Scott measure. Only used when ``measure="scott"``.
        Must satisfy ``1 <= scott_k <= n_qubits // 2``.

        - If None (default): Automatically selects the maximum valid k,
          i.e., ``min(2, n_qubits // 2)``. For 2-3 qubit systems, this
          falls back to k=1 (equivalent to Meyer-Wallach).
        - If specified: Uses the given k value, raising an error if invalid.

        Ignored when ``measure="meyer_wallach"``.

    return_details : bool, default=False
        If True, return full result dictionary with statistics.
        If False, return only the entanglement capability score.
    verbose : bool, default=False
        If True, log progress during computation.

    Returns
    -------
    float or EntanglementResult
        If ``return_details=False``: Entanglement capability in [0, 1].

        If ``return_details=True``: Dictionary containing:
            - ``entanglement_capability``: float in [0, 1]
            - ``entanglement_samples``: per-sample entanglement values
            - ``std_error``: standard error of mean
            - ``n_samples``: number of samples used
            - ``per_qubit_entanglement``: average entanglement per qubit
            - ``measure``: the measure used ("meyer_wallach" or "scott")
            - ``scott_k``: the k value used (None for Meyer-Wallach)

    Raises
    ------
    InsufficientSamplesError
        If ``n_samples < 10``.
    SimulationError
        If circuit simulation fails.
    ValueError
        If encoding has < 2 qubits (entanglement requires multiple qubits).
    ValidationError
        If encoding is invalid or input parameters are malformed.

    Warns
    -----
    UserWarning
        If ``n_samples < 100``, results may be unreliable.

    Examples
    --------
    Basic usage:

    >>> from encoding_atlas import IQPEncoding
    >>> enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
    >>> ent = compute_entanglement_capability(enc, seed=42)
    >>> print(f"Entanglement: {ent:.4f}")

    Compare different entanglement patterns:

    >>> enc_linear = IQPEncoding(n_features=4, reps=2, entanglement="linear")
    >>> enc_full = IQPEncoding(n_features=4, reps=2, entanglement="full")
    >>> print(f"Linear: {compute_entanglement_capability(enc_linear, seed=42):.4f}")
    >>> print(f"Full: {compute_entanglement_capability(enc_full, seed=42):.4f}")

    Get detailed statistics:

    >>> result = compute_entanglement_capability(
    ...     enc, n_samples=500, seed=42, return_details=True
    ... )
    >>> print(f"Mean: {result['entanglement_capability']:.4f}")
    >>> print(f"Std Error: {result['std_error']:.4f}")
    >>> print(f"Per-qubit: {result['per_qubit_entanglement']}")

    Notes
    -----
    For non-entangling encodings (e.g., AngleEncoding), the entanglement
    capability will be exactly 0.0, as all generated states are product states.

    The computation time scales linearly with ``n_samples`` and exponentially
    with the number of qubits (due to statevector simulation).

    See Also
    --------
    compute_meyer_wallach : Compute MW measure for a single state.
    compute_scott_measure : Compute Scott measure for a single state.
    compute_expressibility : Related metric for Hilbert space coverage.
    """
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    validate_encoding_for_analysis(encoding)

    n_qubits = encoding.n_qubits
    n_features = encoding.n_features

    if n_qubits < 2:
        raise ValueError(
            f"Entanglement requires at least 2 qubits. "
            f"Encoding '{encoding.__class__.__name__}' has {n_qubits} qubit(s). "
            f"Consider using an encoding with more features or a different "
            f"encoding type that uses multiple qubits."
        )

    if n_samples < _MIN_SAMPLES_ERROR:
        raise InsufficientSamplesError(
            f"n_samples must be at least {_MIN_SAMPLES_ERROR}, got {n_samples}. "
            f"Increase n_samples for statistically meaningful results.",
            requested_samples=n_samples,
            minimum_samples=_MIN_SAMPLES_ERROR,
            metric="entanglement_capability",
        )

    if n_samples < _MIN_SAMPLES_WARNING:
        warnings.warn(
            f"n_samples={n_samples} is low. For reliable entanglement capability "
            f"estimates, use at least {_MIN_SAMPLES_WARNING} samples.",
            UserWarning,
            stacklevel=2,
        )

    # Validate input_range - check type before length
    if not isinstance(input_range, (tuple, list)):
        raise ValidationError(
            f"input_range must be a tuple or list of (min, max), "
            f"got {type(input_range).__name__}: {input_range!r}"
        )
    if len(input_range) != 2:
        raise ValidationError(
            f"input_range must be a tuple of (min, max) with exactly 2 elements, "
            f"got {len(input_range)} elements: {input_range}"
        )
    if input_range[0] >= input_range[1]:
        raise ValidationError(
            f"input_range[0] must be less than input_range[1]: "
            f"{input_range[0]} >= {input_range[1]}"
        )

    # Validate measure parameter
    if measure not in ("meyer_wallach", "scott"):
        raise ValueError(
            f"measure must be 'meyer_wallach' or 'scott', got {measure!r}"
        )

    # Validate backend parameter
    if backend not in ("pennylane", "qiskit"):
        raise ValueError(
            f"backend must be 'pennylane' or 'qiskit', got {backend!r}"
        )

    # Validate and resolve scott_k parameter
    effective_scott_k: int | None = None
    if measure == "scott":
        max_k = max(1, n_qubits // 2)
        if scott_k is None:
            # Auto-select: prefer k=2 if valid, otherwise use max valid k
            effective_scott_k = min(2, max_k)
            _logger.debug(
                "Auto-selected scott_k=%d for %d qubits (max_k=%d)",
                effective_scott_k,
                n_qubits,
                max_k,
            )
        else:
            # Validate user-provided k
            if not isinstance(scott_k, (int, np.integer)) or scott_k < 1:
                raise ValueError(
                    f"scott_k must be a positive integer, got {scott_k!r}"
                )
            if scott_k > max_k:
                raise ValueError(
                    f"scott_k={scott_k} exceeds maximum valid value of {max_k} "
                    f"for {n_qubits} qubits. For meaningful entanglement measures, "
                    f"scott_k must satisfy 1 <= scott_k <= n_qubits // 2."
                )
            effective_scott_k = int(scott_k)
    elif scott_k is not None:
        # User specified scott_k but measure is not "scott" - warn them
        warnings.warn(
            f"scott_k={scott_k} is ignored when measure='meyer_wallach'. "
            f"To use Scott measure, set measure='scott'.",
            UserWarning,
            stacklevel=2,
        )

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    rng = create_rng(seed)

    if verbose:
        measure_desc = measure
        if measure == "scott" and effective_scott_k is not None:
            measure_desc = f"scott (k={effective_scott_k})"
        _logger.info(
            "Computing entanglement capability for %s "
            "(n_qubits=%d, n_features=%d, n_samples=%d, measure=%s)",
            encoding.__class__.__name__,
            n_qubits,
            n_features,
            n_samples,
            measure_desc,
        )

    # -------------------------------------------------------------------------
    # Sample Entanglement Values
    # -------------------------------------------------------------------------
    entanglement_samples = np.zeros(n_samples, dtype=np.float64)
    per_qubit_sum = np.zeros(n_qubits, dtype=np.float64)

    # Progress logging interval (every 10%)
    log_interval = max(1, n_samples // 10)

    for i in range(n_samples):
        # Generate random input features
        x = rng.uniform(input_range[0], input_range[1], size=n_features)
        x = x.astype(np.float64)

        try:
            # Simulate circuit to get statevector
            statevector = simulate_encoding_statevector(
                encoding, x, backend=backend
            )

            # Compute entanglement measure
            if measure == "meyer_wallach":
                ent_value, per_qubit = compute_meyer_wallach_with_breakdown(
                    statevector, n_qubits
                )
            else:  # scott
                # effective_scott_k is guaranteed to be valid here
                assert effective_scott_k is not None  # Type narrowing for mypy
                ent_value = compute_scott_measure(
                    statevector, n_qubits, k=effective_scott_k
                )
                # For Scott measure, per-qubit breakdown is not directly available
                per_qubit = np.zeros(n_qubits, dtype=np.float64)

            entanglement_samples[i] = ent_value
            per_qubit_sum += per_qubit

        except SimulationError:
            # Re-raise simulation errors with context
            raise
        except Exception as e:
            raise SimulationError(
                f"Entanglement computation failed at sample {i}: {e}",
                backend=backend,
                details={
                    "sample_index": i,
                    "input": x.tolist(),
                    "error_type": type(e).__name__,
                    "measure": measure,
                    "scott_k": effective_scott_k,
                },
            ) from e

        # Progress logging
        if verbose and (i + 1) % log_interval == 0:
            current_mean = np.mean(entanglement_samples[: i + 1])
            _logger.debug(
                "Processed %d/%d samples (current mean: %.4f)",
                i + 1,
                n_samples,
                current_mean,
            )

    # -------------------------------------------------------------------------
    # Compute Statistics
    # -------------------------------------------------------------------------
    entanglement_capability = float(np.mean(entanglement_samples))

    # Standard error of the mean: std / sqrt(n)
    std_dev = float(np.std(entanglement_samples, ddof=1))  # Sample std with Bessel correction
    std_error = std_dev / np.sqrt(n_samples)

    # Per-qubit average entanglement
    per_qubit_entanglement = per_qubit_sum / n_samples

    # Ensure entanglement_capability is in valid range [0, 1]
    # (handle potential numerical errors)
    entanglement_capability = float(np.clip(entanglement_capability, 0.0, 1.0))

    if verbose:
        _logger.info(
            "Entanglement capability: %.4f +/- %.4f (n=%d)",
            entanglement_capability,
            std_error,
            n_samples,
        )
        if n_qubits <= _MAX_VERBOSE_QUBITS:
            _logger.debug(
                "Per-qubit entanglement: %s",
                np.array2string(per_qubit_entanglement, precision=4),
            )

    # -------------------------------------------------------------------------
    # Return Results
    # -------------------------------------------------------------------------
    if return_details:
        return EntanglementResult(
            entanglement_capability=entanglement_capability,
            entanglement_samples=entanglement_samples,
            std_error=std_error,
            n_samples=n_samples,
            per_qubit_entanglement=per_qubit_entanglement,
            measure=measure,
            scott_k=effective_scott_k,
        )

    return entanglement_capability


# =============================================================================
# Supporting Public Functions
# =============================================================================


def compute_meyer_wallach(
    statevector: StatevectorType,
    n_qubits: int,
) -> float:
    """Compute the Meyer-Wallach entanglement measure for a pure state.

    The Meyer-Wallach measure Q(|psi>) quantifies global entanglement as
    the average linear entropy across all single-qubit reduced states:

        Q(|psi>) = (2/n) * sum_i (1 - Tr(rho_i^2))

    where rho_i is the reduced density matrix of qubit i.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        Pure state vector, shape ``(2^n_qubits,)``.
    n_qubits : int
        Number of qubits in the system.

    Returns
    -------
    float
        Meyer-Wallach measure in [0, 1].

        - 0 = completely separable (product state)
        - 1 = maximally entangled

    Raises
    ------
    ValueError
        If statevector length doesn't match ``2^n_qubits``.
    ValidationError
        If statevector is invalid (NaN, zero norm, etc.).

    Examples
    --------
    GHZ state: (|000> + |111>) / sqrt(2) has MW = 1:

    >>> import numpy as np
    >>> ghz = np.zeros(8, dtype=complex)
    >>> ghz[0] = ghz[7] = 1.0 / np.sqrt(2)
    >>> mw = compute_meyer_wallach(ghz, n_qubits=3)
    >>> print(f"GHZ Meyer-Wallach: {mw:.4f}")
    GHZ Meyer-Wallach: 1.0000

    Product state |000> has MW = 0:

    >>> product = np.zeros(8, dtype=complex)
    >>> product[0] = 1.0
    >>> mw = compute_meyer_wallach(product, n_qubits=3)
    >>> print(f"Product Meyer-Wallach: {mw:.4f}")
    Product Meyer-Wallach: 0.0000

    Bell state (|00> + |11>) / sqrt(2) has MW = 1:

    >>> bell = np.zeros(4, dtype=complex)
    >>> bell[0] = bell[3] = 1.0 / np.sqrt(2)
    >>> mw = compute_meyer_wallach(bell, n_qubits=2)
    >>> print(f"Bell Meyer-Wallach: {mw:.4f}")
    Bell Meyer-Wallach: 1.0000

    Notes
    -----
    The Meyer-Wallach measure is invariant under local unitary operations.
    It reaches its maximum value of 1 for maximally entangled states like
    the GHZ state (|00...0> + |11...1>) / sqrt(2).

    See Also
    --------
    compute_meyer_wallach_with_breakdown : Also returns per-qubit contributions.
    compute_scott_measure : Generalization to k-body reduced states.
    """
    mw_value, _ = compute_meyer_wallach_with_breakdown(statevector, n_qubits)
    return mw_value


def compute_meyer_wallach_with_breakdown(
    statevector: StatevectorType,
    n_qubits: int,
) -> tuple[float, FloatArray]:
    """Compute Meyer-Wallach measure with per-qubit breakdown.

    This function returns both the total Meyer-Wallach measure and the
    contribution from each qubit, which is useful for understanding
    which qubits are most entangled.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        Pure state vector, shape ``(2^n_qubits,)``.
    n_qubits : int
        Number of qubits in the system.

    Returns
    -------
    mw_value : float
        Total Meyer-Wallach measure in [0, 1].
    per_qubit_entropy : NDArray[np.floating]
        Linear entropy (1 - purity) for each qubit's reduced state.
        Shape: ``(n_qubits,)``.

    Raises
    ------
    ValueError
        If statevector length doesn't match ``2^n_qubits``, or if
        ``n_qubits < 1``.
    ValidationError
        If statevector is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> # W state: (|001> + |010> + |100>) / sqrt(3)
    >>> w_state = np.zeros(8, dtype=complex)
    >>> w_state[1] = w_state[2] = w_state[4] = 1.0 / np.sqrt(3)
    >>> mw, per_qubit = compute_meyer_wallach_with_breakdown(w_state, n_qubits=3)
    >>> print(f"Total MW: {mw:.4f}")
    >>> print(f"Per-qubit entropies: {per_qubit}")

    Notes
    -----
    The per-qubit breakdown is useful for:

    1. Identifying which qubits contribute most to entanglement
    2. Detecting asymmetric entanglement structures
    3. Debugging encoding circuits

    For the GHZ state, all qubits have equal entropy (= 0.5).
    For the W state, all qubits also have equal entropy (= 2/3).
    """
    # Validate inputs
    if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")

    expected_dim = 2**n_qubits
    state = validate_statevector(statevector, expected_qubits=n_qubits)

    if len(state) != expected_dim:
        raise ValueError(
            f"Statevector length {len(state)} doesn't match "
            f"2^{n_qubits} = {expected_dim}"
        )

    # Handle single-qubit case (no entanglement possible)
    if n_qubits == 1:
        return 0.0, np.zeros(1, dtype=np.float64)

    # Compute per-qubit linear entropy
    per_qubit_entropy = np.zeros(n_qubits, dtype=np.float64)

    for i in range(n_qubits):
        # Compute reduced density matrix for qubit i
        rho_i = partial_trace_single_qubit(state, n_qubits, keep_qubit=i)

        # Compute purity: Tr(rho^2)
        purity = compute_purity(rho_i)

        # Linear entropy: 1 - Tr(rho^2)
        # For pure state (no entanglement): purity = 1, entropy = 0
        # For maximally mixed (max entanglement): purity = 0.5, entropy = 0.5
        per_qubit_entropy[i] = 1.0 - purity

    # Meyer-Wallach measure: (2/n) * sum of linear entropies
    # The factor of 2 normalizes to [0, 1] range
    # (maximum linear entropy per qubit is 0.5 for maximally mixed state)
    mw_value = (2.0 / n_qubits) * np.sum(per_qubit_entropy)

    # Clamp to [0, 1] for numerical stability
    mw_value = float(np.clip(mw_value, 0.0, 1.0))

    return mw_value, per_qubit_entropy


def compute_scott_measure(
    statevector: StatevectorType,
    n_qubits: int,
    k: int = 2,
) -> float:
    """Compute the Scott entanglement measure.

    The Scott measure generalizes Meyer-Wallach to k-body reduced states,
    averaging the linear entropy over all k-qubit subsystems.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        Pure state vector.
    n_qubits : int
        Number of qubits.
    k : int, default=2
        Size of subsystems to consider. Must satisfy 1 <= k <= n_qubits // 2.

        - k=1: Equivalent to Meyer-Wallach measure
        - k=2: Pairwise entanglement (default)
        - Higher k: More detailed entanglement structure

    Returns
    -------
    float
        Scott measure in [0, 1].

    Raises
    ------
    ValueError
        If k is out of valid range or inputs are invalid.
    ValidationError
        If statevector is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> # GHZ state
    >>> ghz = np.zeros(8, dtype=complex)
    >>> ghz[0] = ghz[7] = 1.0 / np.sqrt(2)
    >>> scott_1 = compute_scott_measure(ghz, n_qubits=3, k=1)
    >>> scott_2 = compute_scott_measure(ghz, n_qubits=3, k=2)
    >>> print(f"Scott k=1: {scott_1:.4f}")  # Same as Meyer-Wallach
    >>> print(f"Scott k=2: {scott_2:.4f}")

    Notes
    -----
    The Scott measure with k > 1 is more computationally expensive than
    Meyer-Wallach, as it requires computing reduced density matrices for
    all C(n, k) subsystems.

    References
    ----------
    .. [1] Scott, A. J. (2004).
           "Multipartite entanglement, quantum-error-correcting codes, and
           entangling power of quantum evolutions."
           Physical Review A, 69(5), 052330.
    """
    # Validate inputs
    if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")

    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    # For k=1, this reduces to Meyer-Wallach
    if k == 1:
        return compute_meyer_wallach(statevector, n_qubits)

    # k cannot exceed n_qubits // 2 for meaningful measure
    # (otherwise we're looking at most of the system)
    max_k = max(1, n_qubits // 2)
    if k > max_k:
        raise ValueError(
            f"k={k} exceeds maximum allowed value of {max_k} for {n_qubits} qubits. "
            f"For meaningful entanglement measures, k should be at most n_qubits // 2."
        )

    if k > n_qubits:
        raise ValueError(
            f"k={k} cannot exceed n_qubits={n_qubits}"
        )

    # Validate statevector
    state = validate_statevector(statevector, expected_qubits=n_qubits)

    # Special case: single qubit cannot have entanglement
    if n_qubits == 1:
        return 0.0

    # Compute average linear entropy over all k-qubit subsystems
    total_linear_entropy = 0.0
    n_subsystems = 0

    for qubits_to_keep in combinations(range(n_qubits), k):
        # Compute reduced density matrix for this subsystem
        rho_k = partial_trace_subsystem(state, n_qubits, list(qubits_to_keep))

        # Compute purity and linear entropy
        purity = compute_purity(rho_k)
        linear_entropy = 1.0 - purity

        total_linear_entropy += linear_entropy
        n_subsystems += 1

    # Average and normalize
    if n_subsystems == 0:
        return 0.0

    # Normalization factor to bring to [0, 1] range
    # Maximum linear entropy for k qubits is 1 - 1/2^k
    max_linear_entropy = 1.0 - 1.0 / (2**k)
    avg_linear_entropy = total_linear_entropy / n_subsystems

    if max_linear_entropy > _EPSILON:
        normalized_measure = avg_linear_entropy / max_linear_entropy
    else:
        normalized_measure = 0.0

    # Clamp to [0, 1]
    return float(np.clip(normalized_measure, 0.0, 1.0))


# =============================================================================
# Private Helper Functions
# =============================================================================


def _validate_entanglement_inputs(
    n_samples: int,
    input_range: tuple[float, float] | list[float],
    measure: str,
    backend: str,
) -> None:
    """Validate common input parameters for entanglement functions.

    This is a helper function that consolidates input validation logic.

    Parameters
    ----------
    n_samples : int
        Number of samples to validate.
    input_range : tuple[float, float] or list[float]
        Input range to validate.
    measure : str
        Measure name to validate.
    backend : str
        Backend name to validate.

    Raises
    ------
    InsufficientSamplesError
        If n_samples is too low.
    ValidationError
        If any parameter is invalid.
    """
    if n_samples < _MIN_SAMPLES_ERROR:
        raise InsufficientSamplesError(
            f"n_samples must be at least {_MIN_SAMPLES_ERROR}, got {n_samples}",
            requested_samples=n_samples,
            minimum_samples=_MIN_SAMPLES_ERROR,
            metric="entanglement_capability",
        )

    # Validate input_range - check type before length
    if not isinstance(input_range, (tuple, list)):
        raise ValidationError(
            f"input_range must be a tuple or list of (min, max), "
            f"got {type(input_range).__name__}: {input_range!r}"
        )

    if len(input_range) != 2:
        raise ValidationError(
            f"input_range must be a tuple of (min, max) with exactly 2 elements, "
            f"got {len(input_range)} elements: {input_range}"
        )

    if input_range[0] >= input_range[1]:
        raise ValidationError(
            f"input_range[0] must be less than input_range[1]: "
            f"{input_range[0]} >= {input_range[1]}"
        )

    if measure not in ("meyer_wallach", "scott"):
        raise ValueError(
            f"measure must be 'meyer_wallach' or 'scott', got {measure!r}"
        )

    if backend not in ("pennylane", "qiskit"):
        raise ValueError(
            f"backend must be 'pennylane' or 'qiskit', got {backend!r}"
        )
