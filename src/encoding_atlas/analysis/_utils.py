"""Shared utilities for the analysis module.

This module provides common functionality used across all analysis functions
in the encoding_atlas.analysis package. It includes:

- **Circuit Simulation**: Statevector simulation for PennyLane, Qiskit, and Cirq backends
- **Partial Trace Computation**: Reduced density matrix calculations
- **Quantum Fidelity**: State overlap computations
- **Purity Calculation**: Density matrix purity measures
- **Gradient Computation**: Parameter-shift rule implementation
- **Input Validation**: Encoding validation for analysis operations
- **Random Sampling**: Reproducible random state generation

These utilities are designed with numerical stability in mind and include
comprehensive error handling for edge cases.

Mathematical Background
-----------------------
The core mathematical operations implemented here are:

**Fidelity** (for pure states):
    F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²

**Purity** (for density matrices):
    γ(ρ) = Tr(ρ²)

For a pure state, γ = 1. For a maximally mixed state of dimension d, γ = 1/d.

**Partial Trace**:
    For a bipartite system AB, the reduced density matrix of subsystem A is:
    ρ_A = Tr_B(ρ_AB)

**Parameter-Shift Rule**:
    For a parameterized gate U(θ) = exp(-iθG/2) where G² = I:
    ∂⟨O⟩/∂θ = (⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}) / 2

References
----------
.. [1] Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and
       Quantum Information." Cambridge University Press.

.. [2] Schuld, M., et al. (2019). "Evaluating analytic gradients on quantum
       hardware." Physical Review A, 99(3), 032331.

.. [3] Mitarai, K., et al. (2018). "Quantum circuit learning." Physical
       Review A, 98(3), 032309.

Examples
--------
Basic statevector simulation:

>>> from encoding_atlas import AngleEncoding
>>> from encoding_atlas.analysis._utils import simulate_encoding_statevector
>>> import numpy as np
>>>
>>> enc = AngleEncoding(n_features=2)
>>> x = np.array([0.5, 1.0])
>>> statevector = simulate_encoding_statevector(enc, x)
>>> print(f"State dimension: {len(statevector)}")
State dimension: 4

Computing fidelity between states:

>>> from encoding_atlas.analysis._utils import compute_fidelity
>>> state1 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
>>> state2 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
>>> fidelity = compute_fidelity(state1, state2)
>>> print(f"Fidelity: {fidelity}")
Fidelity: 1.0

Notes
-----
All functions in this module are designed to be:

1. **Numerically Stable**: Using epsilon values and clamping to prevent
   floating-point issues
2. **Backend Agnostic**: Supporting multiple quantum simulation backends
3. **Reproducible**: Accepting random seeds for deterministic behavior
4. **Well-Documented**: Including mathematical formulas and references

Qubit Ordering Convention
-------------------------
This module uses the **MSB (most significant bit) first** convention for
qubit ordering. All simulation functions in this module return statevectors
using this convention, regardless of which backend is used:

- **Qubit 0** corresponds to the **leftmost** (most significant) bit
- For an n-qubit state |q₀q₁...qₙ₋₁⟩, the statevector index i encodes the
  computational basis state where q₀ is the MSB

**Native Backend Conventions:**

- **PennyLane**: Uses MSB natively (qubit 0 = leftmost bit) — no conversion needed.
- **Qiskit**: Uses LSB natively (qubit 0 = rightmost bit) — converted to MSB via
  ``_reverse_qubit_order()`` in ``_simulate_qiskit()``.
- **Cirq**: Uses MSB natively (``LineQubit(0)`` = leftmost bit) — no conversion needed.

The conversion is handled automatically by the simulation functions, so users
always receive statevectors in the consistent MSB format.

**Example for 3 qubits:**

========  ===========  =======
Index     Bitstring    State
========  ===========  =======
0         000          |000⟩
1         001          |001⟩
2         010          |010⟩
3         011          |011⟩
4         100          |100⟩
5         101          |101⟩
6         110          |110⟩
7         111          |111⟩
========  ===========  =======

To determine the state of qubit 0 from index i:
    qubit_0_state = (i >> (n_qubits - 1)) & 1

This convention is important for:
- Partial trace computations (knowing which indices to sum)
- Local observable expectations (e.g., ⟨Z₀⟩ on first qubit)
- Cross-backend consistency (same analysis results regardless of backend)

See Also
--------
encoding_atlas.analysis.expressibility : Expressibility computation
encoding_atlas.analysis.entanglement : Entanglement capability measures
encoding_atlas.analysis.trainability : Trainability estimation
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.exceptions import (
    AnalysisError,
    NumericalInstabilityError,
    SimulationError,
    ValidationError,
)

if TYPE_CHECKING:
    pass

# =============================================================================
# Module-Level Logger
# =============================================================================

_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Simulation
    "simulate_encoding_statevector",
    "simulate_encoding_statevectors_batch",
    # Validation
    "validate_encoding_for_analysis",
    "validate_statevector",
    # Quantum operations
    "partial_trace_single_qubit",
    "partial_trace_subsystem",
    "compute_fidelity",
    "compute_purity",
    "compute_linear_entropy",
    "compute_von_neumann_entropy",
    # Gradient computation
    "compute_parameter_gradient",
    "compute_all_parameter_gradients",
    # Random sampling
    "generate_random_parameters",
    "create_rng",
]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Numerical stability constant to prevent division by zero
# This value is chosen to be small enough to not affect results
# but large enough to prevent floating-point underflow issues
_EPSILON: float = 1e-15

# Minimum norm threshold for statevector validation
# States with norm below this are considered invalid
_MIN_NORM_THRESHOLD: float = 1e-12

# Maximum qubit count for full statevector simulation.
#
# MEMORY ANALYSIS:
# - Statevector memory: 2^n × 16 bytes (complex128)
# - Density matrix memory: 2^(2n) × 16 bytes (for partial trace operations)
#
# At 20 qubits:
# - Statevector: 2^20 × 16 = 16 MB (manageable)
# - Density matrix: 2^40 × 16 = 16 TB (impossible)
#
# The limit of 20 qubits is set for statevector-only operations. Analysis
# functions that require density matrices (entanglement, etc.) have their
# own practical limits around 13-15 qubits due to density matrix memory.
_MAX_SIMULATION_QUBITS: int = 20

# Warning threshold for qubit count in simulation.
#
# At 15 qubits, analysis operations that create density matrices require:
# - Density matrix: 2^30 × 16 = 16 GB
#
# This threshold warns users that they are entering memory-intensive territory.
# Pure statevector operations remain fast (only ~0.5 MB for statevector).
_SIMULATION_QUBIT_WARNING_THRESHOLD: int = 15

# Default parameter range for random sampling
_DEFAULT_PARAM_MIN: float = 0.0
_DEFAULT_PARAM_MAX: float = 2.0 * np.pi

# =============================================================================
# Type Aliases
# =============================================================================

StatevectorType = NDArray[np.complexfloating[Any, Any]]
"""Complex array representing a quantum statevector, shape (2^n_qubits,)."""

DensityMatrixType = NDArray[np.complexfloating[Any, Any]]
"""Complex array representing a density matrix, shape (2^n, 2^n)."""

FloatArray = NDArray[np.floating[Any]]
"""Array of floating-point values."""


# =============================================================================
# Internal Helper Functions (Qubit Ordering)
# =============================================================================


def _reverse_bits(i: int, n_bits: int) -> int:
    """Reverse the bit order of an integer.

    Parameters
    ----------
    i : int
        The integer whose bits to reverse.
    n_bits : int
        The number of bits to consider.

    Returns
    -------
    int
        The bit-reversed integer.

    Examples
    --------
    >>> _reverse_bits(1, 3)  # 001 -> 100 = 4
    4
    >>> _reverse_bits(2, 3)  # 010 -> 010 = 2
    2
    >>> _reverse_bits(5, 3)  # 101 -> 101 = 5
    5
    """
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (i & 1)
        i >>= 1
    return result


def _reverse_qubit_order(
    statevector: StatevectorType,
    n_qubits: int,
) -> StatevectorType:
    """Reverse the qubit ordering of a statevector.

    Converts between LSB (least significant bit first) and MSB (most
    significant bit first) qubit ordering conventions by permuting the
    statevector amplitudes.

    Parameters
    ----------
    statevector : StatevectorType
        Input statevector of shape (2^n_qubits,).
    n_qubits : int
        Number of qubits in the system.

    Returns
    -------
    StatevectorType
        Statevector with reversed qubit ordering.

    Notes
    -----
    This function is used to convert Qiskit's LSB convention to this
    library's MSB convention (consistent with PennyLane).

    **Example for 2 qubits:**

    In LSB (Qiskit): qubit 0 is rightmost bit
        - Index 0 = |00⟩ (q1=0, q0=0)
        - Index 1 = |01⟩ (q1=0, q0=1)  <- qubit 0 is 1
        - Index 2 = |10⟩ (q1=1, q0=0)
        - Index 3 = |11⟩ (q1=1, q0=1)

    In MSB (PennyLane/this library): qubit 0 is leftmost bit
        - Index 0 = |00⟩ (q0=0, q1=0)
        - Index 1 = |01⟩ (q0=0, q1=1)
        - Index 2 = |10⟩ (q0=1, q1=0)  <- qubit 0 is 1
        - Index 3 = |11⟩ (q0=1, q1=1)

    Converting LSB → MSB requires:
        - Index 0 → Index 0 (00 → 00)
        - Index 1 → Index 2 (01 → 10)
        - Index 2 → Index 1 (10 → 01)
        - Index 3 → Index 3 (11 → 11)
    """
    dim = 2**n_qubits
    reordered = np.zeros(dim, dtype=np.complex128)

    for i in range(dim):
        j = _reverse_bits(i, n_qubits)
        reordered[j] = statevector[i]

    return reordered


# =============================================================================
# Simulation Functions
# =============================================================================


def simulate_encoding_statevector(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> StatevectorType:
    """Simulate an encoding circuit and return the resulting statevector.

    This function executes a quantum circuit defined by the encoding with
    the given input data and returns the final quantum state as a complex
    vector in the computational basis.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to simulate.
    x : NDArray[np.floating]
        Input data vector of shape ``(n_features,)``. The values are used
        as parameters for the encoding circuit.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        The quantum simulation backend to use.

        - ``"pennylane"``: Uses PennyLane's default.qubit simulator
        - ``"qiskit"``: Uses Qiskit's Statevector class
        - ``"cirq"``: Uses Cirq's Simulator for statevector simulation

    Returns
    -------
    StatevectorType
        Complex statevector of shape ``(2^n_qubits,)`` representing the
        final quantum state in the computational basis. The state is
        normalized (sum of |amplitude|² = 1).

    Raises
    ------
    SimulationError
        If the simulation fails due to backend errors, missing dependencies,
        or invalid circuit structure.
    ValidationError
        If the encoding or input data is invalid.
    ValueError
        If an unknown backend is specified.

    Warnings
    --------
    UserWarning
        If the number of qubits exceeds ``_SIMULATION_QUBIT_WARNING_THRESHOLD``
        (15 by default), a warning is issued about memory usage.

    Examples
    --------
    Simulate an angle encoding circuit:

    >>> from encoding_atlas import AngleEncoding
    >>> import numpy as np
    >>>
    >>> enc = AngleEncoding(n_features=3)
    >>> x = np.array([0.1, 0.2, 0.3])
    >>> state = simulate_encoding_statevector(enc, x)
    >>> print(f"State shape: {state.shape}")
    State shape: (8,)
    >>> print(f"State norm: {np.linalg.norm(state):.6f}")
    State norm: 1.000000

    Using Qiskit backend:

    >>> state_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")
    >>> # Results should be equivalent (up to global phase)
    >>> fidelity = np.abs(np.vdot(state, state_qiskit))**2
    >>> print(f"Cross-backend fidelity: {fidelity:.6f}")
    Cross-backend fidelity: 1.000000

    Notes
    -----
    The statevector is ordered in the computational basis with the standard
    convention where qubit 0 is the most significant bit:

    - Index 0 corresponds to |00...0⟩
    - Index 1 corresponds to |00...1⟩
    - Index 2^n - 1 corresponds to |11...1⟩

    **Memory Requirements**

    Memory grows exponentially with qubit count:

    - Statevector: 2^n × 16 bytes (e.g., 15 qubits → 0.5 MB, 20 qubits → 16 MB)
    - Density matrix (for analysis): 2^(2n) × 16 bytes (e.g., 15 qubits → 16 GB)

    Analysis operations like entanglement capability create full density matrices,
    so their practical limit is around 13-15 qubits. For larger systems, consider
    tensor network methods or sampling-based approaches.

    See Also
    --------
    simulate_encoding_statevectors_batch : Simulate multiple inputs at once.
    compute_fidelity : Compute fidelity between two statevectors.
    """
    # Validate inputs
    validate_encoding_for_analysis(encoding)

    n_qubits = encoding.n_qubits

    # Check for excessive qubit count
    if n_qubits > _MAX_SIMULATION_QUBITS:
        # Calculate memory requirements
        statevector_bytes = 2**n_qubits * 16
        density_matrix_bytes = 2 ** (2 * n_qubits) * 16
        statevector_mb = statevector_bytes / (1024**2)
        density_matrix_tb = density_matrix_bytes / (1024**4)

        raise SimulationError(
            f"Cannot simulate {n_qubits} qubits: exceeds maximum of "
            f"{_MAX_SIMULATION_QUBITS}. "
            f"Statevector would require {statevector_mb:.0f} MB, and analysis "
            f"operations (entanglement, etc.) would require {density_matrix_tb:.0f} TB "
            f"for density matrix computations.",
            backend=backend,
            details={"n_qubits": n_qubits, "max_qubits": _MAX_SIMULATION_QUBITS},
        )

    if n_qubits > _SIMULATION_QUBIT_WARNING_THRESHOLD:
        # Calculate memory requirements for statevector and density matrix
        statevector_bytes = 2**n_qubits * 16
        density_matrix_bytes = 2 ** (2 * n_qubits) * 16

        # Format statevector memory (typically MB range)
        statevector_mb = statevector_bytes / (1024**2)

        # Format density matrix memory (typically GB range at this threshold)
        density_matrix_gb = density_matrix_bytes / (1024**3)

        warnings.warn(
            f"Simulating {n_qubits} qubits: statevector requires {statevector_mb:.1f} MB, "
            f"but analysis operations (entanglement, partial trace) create density "
            f"matrices requiring {density_matrix_gb:.0f} GB. "
            f"Consider using fewer qubits for analysis tasks.",
            UserWarning,
            stacklevel=2,
        )

    # Validate input shape
    x_array = np.asarray(x, dtype=np.float64)
    if x_array.ndim != 1:
        raise ValidationError(f"Input x must be 1D array, got shape {x_array.shape}")
    if x_array.shape[0] != encoding.n_features:
        raise ValidationError(
            f"Input x has {x_array.shape[0]} features, but encoding expects "
            f"{encoding.n_features} features"
        )

    # Check for invalid values
    if np.any(np.isnan(x_array)) or np.any(np.isinf(x_array)):
        raise ValidationError("Input x contains NaN or infinite values")

    _logger.debug(
        "Simulating encoding %s with %d qubits using %s backend",
        encoding.__class__.__name__,
        n_qubits,
        backend,
    )

    # Dispatch to backend-specific implementation
    if backend == "pennylane":
        return _simulate_pennylane(encoding, x_array)
    elif backend == "qiskit":
        return _simulate_qiskit(encoding, x_array)
    elif backend == "cirq":
        return _simulate_cirq(encoding, x_array)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Supported backends are "
            f"'pennylane', 'qiskit', and 'cirq'."
        )


def simulate_encoding_statevectors_batch(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> NDArray[np.complexfloating[Any, Any]]:
    """Simulate encoding circuits for multiple input vectors.

    This is a convenience function that applies :func:`simulate_encoding_statevector`
    to each row of a 2D input array and returns a pre-allocated 2D array of
    statevectors.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to simulate.
    X : NDArray[np.floating]
        Input data array of shape ``(n_samples, n_features)``.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        The quantum simulation backend to use.

    Returns
    -------
    NDArray[np.complexfloating], shape ``(n_samples, 2**n_qubits)``
        2D array of statevectors, one row per input sample.

    Raises
    ------
    SimulationError
        If simulation fails for any input.
    ValidationError
        If input shape is incorrect.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> import numpy as np
    >>>
    >>> enc = AngleEncoding(n_features=2)
    >>> X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> states = simulate_encoding_statevectors_batch(enc, X)
    >>> print(f"Number of states: {len(states)}")
    Number of states: 3
    """
    X_array = np.asarray(X, dtype=np.float64)
    if X_array.ndim != 2:
        raise ValidationError(f"Input X must be 2D array, got shape {X_array.shape}")

    n_samples = X_array.shape[0]
    dim = 2**encoding.n_qubits

    _logger.debug(
        "Batch simulating %d samples for encoding %s",
        n_samples,
        encoding.__class__.__name__,
    )

    states = np.zeros((n_samples, dim), dtype=np.complex128)
    for i, x in enumerate(X_array):
        try:
            state = simulate_encoding_statevector(encoding, x, backend)
            states[i] = np.asarray(state, dtype=np.complex128).ravel()
        except SimulationError as e:
            raise SimulationError(
                f"Simulation failed for sample {i}: {e}",
                backend=backend,
                details={"sample_index": i, "original_error": str(e)},
            ) from e

    return states


def _simulate_pennylane(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
) -> StatevectorType:
    """Simulate encoding using PennyLane's default.qubit device.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to simulate.
    x : NDArray[np.floating]
        Input data vector.

    Returns
    -------
    StatevectorType
        Statevector of the encoded state.

    Raises
    ------
    SimulationError
        If PennyLane is not installed, the encoding returns a non-callable
        circuit, or simulation fails for any other reason.

    Notes
    -----
    PennyLane encodings are expected to return a callable (function) that,
    when invoked inside a QNode, applies quantum gates to the circuit.
    If the encoding's ``get_circuit()`` method returns something that is
    not callable (e.g., None, a QuantumCircuit object, or other non-function
    types), this function raises a ``SimulationError`` with a descriptive
    message to help diagnose the issue.
    """
    try:
        import pennylane as qml
    except ImportError as e:
        raise SimulationError(
            "PennyLane is required for simulation but is not installed. "
            "Install it with: pip install pennylane",
            backend="pennylane",
        ) from e

    n_qubits = encoding.n_qubits
    encoding_name = encoding.__class__.__name__

    try:
        # Create a device for statevector simulation
        dev = qml.device("default.qubit", wires=n_qubits)

        # Get the circuit operations from the encoding
        # PennyLane encodings return a callable that applies gates
        circuit_fn = encoding.get_circuit(x, backend="pennylane")

        # Validate that the circuit is callable
        # This check is performed outside the QNode to provide a clear error
        # message before simulation begins
        if not callable(circuit_fn):
            raise SimulationError(
                f"Encoding '{encoding_name}' returned a non-callable circuit "
                f"of type '{type(circuit_fn).__name__}' for the PennyLane backend. "
                f"Expected a callable (function) that applies gates to the circuit. "
                f"This may indicate that the encoding does not properly support "
                f"the PennyLane backend, or there is a bug in its get_circuit() method.",
                backend="pennylane",
                details={
                    "encoding": encoding_name,
                    "circuit_type": type(circuit_fn).__name__,
                    "circuit_value": repr(circuit_fn)[:200],  # Truncate for safety
                },
            )

        @qml.qnode(dev)
        def statevector_circuit():
            """Execute circuit and return statevector."""
            # Apply the encoding circuit (validated as callable above)
            circuit_fn()
            return qml.state()

        # Execute and convert to numpy
        result = statevector_circuit()
        statevector = np.array(result, dtype=np.complex128)

        # Verify normalization
        norm = np.linalg.norm(statevector)
        if not np.isclose(norm, 1.0, atol=_MIN_NORM_THRESHOLD):
            _logger.warning("Statevector norm is %.6f, renormalizing", norm)
            statevector = statevector / norm

        return statevector

    except SimulationError:
        # Re-raise SimulationError without wrapping
        raise
    except Exception as e:
        raise SimulationError(
            f"PennyLane simulation failed for encoding '{encoding_name}': {e}",
            backend="pennylane",
            details={
                "encoding": encoding_name,
                "error_type": type(e).__name__,
            },
        ) from e


def _simulate_qiskit(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
) -> StatevectorType:
    """Simulate encoding using Qiskit's Statevector class.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to simulate.
    x : NDArray[np.floating]
        Input data vector.

    Returns
    -------
    StatevectorType
        Statevector of the encoded state, using the same qubit ordering
        convention as PennyLane (MSB = qubit 0).

    Raises
    ------
    SimulationError
        If Qiskit is not installed or simulation fails.

    Notes
    -----
    **Qubit Ordering Conversion**

    Qiskit natively uses **LSB (least significant bit)** ordering, where
    qubit 0 corresponds to the rightmost bit. This library uses **MSB (most
    significant bit)** ordering (consistent with PennyLane), where qubit 0
    corresponds to the leftmost bit.

    This function automatically converts Qiskit's output to MSB ordering by
    permuting the statevector amplitudes. The conversion reverses the bit
    order of each index:

    - Qiskit index 1 (binary: 01) → MSB index 2 (binary: 10)
    - Qiskit index 2 (binary: 10) → MSB index 1 (binary: 01)

    **Cross-Backend Consistency**

    After this conversion, all simulation backends in this module
    (_simulate_pennylane, _simulate_qiskit, _simulate_cirq) return statevectors
    using the same MSB convention. This ensures that analysis functions
    (expressibility, entanglement, trainability) produce consistent results
    regardless of which backend is used.

    For detailed qubit ordering documentation, see the module docstring.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError as e:
        raise SimulationError(
            "Qiskit is required for simulation but is not installed. "
            "Install it with: pip install qiskit",
            backend="qiskit",
        ) from e

    try:
        # Get the Qiskit circuit from the encoding
        circuit = encoding.get_circuit(x, backend="qiskit")

        if not isinstance(circuit, QuantumCircuit):
            raise SimulationError(
                f"Expected Qiskit QuantumCircuit, got {type(circuit).__name__}",
                backend="qiskit",
            )

        # Compute statevector
        statevector_obj = Statevector(circuit)
        statevector_raw = np.array(statevector_obj.data, dtype=np.complex128)

        # Convert Qiskit's LSB qubit ordering to this library's MSB convention.
        # Qiskit uses qubit 0 = LSB (rightmost bit), while PennyLane and this
        # library use qubit 0 = MSB (leftmost bit).
        #
        # To convert: permute statevector indices by reversing the bit order.
        # For index i in Qiskit, the corresponding index in MSB convention is
        # the bit-reversal of i for n_qubits bits.
        n_qubits = circuit.num_qubits
        statevector = _reverse_qubit_order(statevector_raw, n_qubits)

        # Verify normalization
        norm = np.linalg.norm(statevector)
        if not np.isclose(norm, 1.0, atol=_MIN_NORM_THRESHOLD):
            _logger.warning("Statevector norm is %.6f, renormalizing", norm)
            statevector = statevector / norm

        return statevector

    except SimulationError:
        raise
    except Exception as e:
        raise SimulationError(
            f"Qiskit simulation failed: {e}",
            backend="qiskit",
            details={"error_type": type(e).__name__},
        ) from e


def _simulate_cirq(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
) -> StatevectorType:
    """Simulate encoding using Cirq's Simulator.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to simulate.
    x : NDArray[np.floating]
        Input data vector.

    Returns
    -------
    StatevectorType
        Statevector of the encoded state.

    Raises
    ------
    SimulationError
        If Cirq is not installed or simulation fails.

    Notes
    -----
    Cirq uses a different qubit ordering convention than some other frameworks.
    This function handles the conversion to ensure consistent statevector
    ordering across all backends.
    """
    try:
        import cirq
    except ImportError as e:
        raise SimulationError(
            "Cirq is required for simulation but is not installed. "
            "Install it with: pip install cirq",
            backend="cirq",
        ) from e

    try:
        # Get the Cirq circuit from the encoding
        circuit = encoding.get_circuit(x, backend="cirq")

        if not isinstance(circuit, cirq.Circuit):
            raise SimulationError(
                f"Expected Cirq Circuit, got {type(circuit).__name__}",
                backend="cirq",
            )

        # Create simulator and compute final state.
        # Use complex128 to match the double-precision simulation used by
        # PennyLane's default.qubit and Qiskit's Statevector.  Cirq's
        # Simulator defaults to complex64 which silently truncates to
        # single-precision, introducing ~1e-8 discrepancies that pollute
        # downstream analysis (expressibility, entanglement, etc.).
        simulator = cirq.Simulator(dtype=np.complex128)

        # Get the result with the final state.
        # Specify qubit_order so that Cirq always returns a 2^n_qubits
        # vector even when some qubits have no operations (e.g.
        # BasisEncoding with sparse inputs).  Without this, Cirq only
        # simulates the active qubits and returns a shorter vector.
        all_qubits = cirq.LineQubit.range(encoding.n_qubits)
        result = simulator.simulate(circuit, qubit_order=all_qubits)

        # Extract the statevector from the result
        # Cirq's final_state_vector is already a numpy array
        statevector = np.array(result.final_state_vector, dtype=np.complex128)

        # Verify normalization
        norm = np.linalg.norm(statevector)
        if not np.isclose(norm, 1.0, atol=_MIN_NORM_THRESHOLD):
            _logger.warning("Statevector norm is %.6f, renormalizing", norm)
            statevector = statevector / norm

        return statevector

    except SimulationError:
        raise
    except Exception as e:
        raise SimulationError(
            f"Cirq simulation failed: {e}",
            backend="cirq",
            details={"error_type": type(e).__name__},
        ) from e


# =============================================================================
# Validation Functions
# =============================================================================


def validate_encoding_for_analysis(encoding: BaseEncoding) -> None:
    """Validate that an encoding is suitable for analysis operations.

    This function performs comprehensive validation to ensure the encoding
    can be safely used in analysis functions. It checks the encoding type,
    configuration, and derived properties.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to validate.

    Raises
    ------
    AnalysisError
        If the encoding is not a valid BaseEncoding instance.
    ValidationError
        If the encoding has invalid configuration (e.g., zero features
        or qubits).

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> validate_encoding_for_analysis(enc)  # No exception raised

    >>> validate_encoding_for_analysis("not an encoding")
    Traceback (most recent call last):
        ...
    AnalysisError: Expected BaseEncoding instance, got str

    Notes
    -----
    This function is called internally by analysis functions. You generally
    don't need to call it directly unless implementing custom analysis.
    """
    # Check type
    if not isinstance(encoding, BaseEncoding):
        raise AnalysisError(
            f"Expected BaseEncoding instance, got {type(encoding).__name__}",
            details={"actual_type": type(encoding).__name__},
        )

    # Check n_features
    if encoding.n_features < 1:
        raise ValidationError(
            f"Encoding must have at least 1 feature, got {encoding.n_features}"
        )

    # Check n_qubits
    try:
        n_qubits = encoding.n_qubits
    except Exception as e:
        raise AnalysisError(
            f"Failed to get n_qubits from encoding: {e}",
            details={"error": str(e)},
        ) from e

    if n_qubits < 1:
        raise ValidationError(f"Encoding must have at least 1 qubit, got {n_qubits}")

    _logger.debug(
        "Validated encoding %s: n_features=%d, n_qubits=%d",
        encoding.__class__.__name__,
        encoding.n_features,
        n_qubits,
    )


def validate_statevector(
    statevector: NDArray[Any],
    expected_qubits: int | None = None,
    check_normalization: bool = True,
    tolerance: float = 1e-10,
) -> StatevectorType:
    """Validate and normalize a statevector.

    Parameters
    ----------
    statevector : NDArray
        The statevector to validate.
    expected_qubits : int, optional
        If provided, verify the statevector has dimension 2^expected_qubits.
    check_normalization : bool, default=True
        If True, verify the statevector is normalized and renormalize if
        needed.
    tolerance : float, default=1e-10
        Tolerance for normalization check.

    Returns
    -------
    StatevectorType
        Validated (and possibly renormalized) statevector as complex128.

    Raises
    ------
    ValidationError
        If the statevector has invalid shape or values.
    NumericalInstabilityError
        If the statevector has near-zero norm.

    Examples
    --------
    >>> import numpy as np
    >>> state = np.array([1, 0, 0, 0], dtype=complex)
    >>> validated = validate_statevector(state, expected_qubits=2)
    >>> validated.dtype
    dtype('complex128')
    """
    state = np.asarray(statevector, dtype=np.complex128)

    # Check shape
    if state.ndim != 1:
        raise ValidationError(f"Statevector must be 1D, got shape {state.shape}")

    dim = state.shape[0]

    # Check power of 2
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValidationError(f"Statevector dimension must be a power of 2, got {dim}")

    # Check expected qubits
    if expected_qubits is not None:
        expected_dim = 2**expected_qubits
        if dim != expected_dim:
            raise ValidationError(
                f"Statevector has dimension {dim}, but expected {expected_dim} "
                f"for {expected_qubits} qubits"
            )

    # Check for invalid values
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        raise ValidationError("Statevector contains NaN or infinite values")

    # Check and fix normalization
    if check_normalization:
        norm = np.linalg.norm(state)
        if norm < _MIN_NORM_THRESHOLD:
            raise NumericalInstabilityError(
                f"Statevector has near-zero norm: {norm}",
                value=norm,
                operation="validate_statevector",
            )
        if not np.isclose(norm, 1.0, atol=tolerance):
            # If the norm deviates significantly from 1.0, the state is
            # genuinely non-physical and should be rejected rather than
            # silently renormalized.  Only tiny floating-point drift
            # (within a generous tolerance) is corrected automatically.
            _renorm_tolerance = max(tolerance * 1e4, 1e-6)
            if abs(norm - 1.0) > _renorm_tolerance:
                raise ValidationError(
                    f"Statevector is not normalized: norm = {norm:.10f} "
                    f"(deviation {abs(norm - 1.0):.2e} exceeds tolerance "
                    f"{_renorm_tolerance:.2e})"
                )
            _logger.debug("Renormalizing statevector: norm was %.10f", norm)
            state = state / norm

    return state


# =============================================================================
# Partial Trace Functions
# =============================================================================


def partial_trace_single_qubit(
    statevector: StatevectorType,
    n_qubits: int,
    keep_qubit: int,
) -> DensityMatrixType:
    """Compute the partial trace, keeping only a single qubit.

    This function traces out all qubits except the specified one,
    returning the reduced density matrix for that qubit.

    Parameters
    ----------
    statevector : StatevectorType
        Full system statevector of shape ``(2^n_qubits,)``.
    n_qubits : int
        Total number of qubits in the system.
    keep_qubit : int
        Index of the qubit to keep (0-indexed). Qubit 0 is the most
        significant bit in the computational basis ordering.

    Returns
    -------
    DensityMatrixType
        Reduced density matrix for the kept qubit, shape ``(2, 2)``.
        This is a valid density matrix: Hermitian, positive semidefinite,
        with trace 1.

    Raises
    ------
    ValueError
        If ``keep_qubit`` is out of range ``[0, n_qubits-1]``.
    ValidationError
        If the statevector is invalid.

    Examples
    --------
    For a separable state |00⟩, each qubit's reduced state is |0⟩:

    >>> import numpy as np
    >>> state_00 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    >>> rho_0 = partial_trace_single_qubit(state_00, n_qubits=2, keep_qubit=0)
    >>> print(rho_0)
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    For a Bell state (|00⟩ + |11⟩)/√2, each qubit is maximally mixed:

    >>> bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    >>> rho_0 = partial_trace_single_qubit(bell, n_qubits=2, keep_qubit=0)
    >>> print(np.round(rho_0, 3))
    [[0.5+0.j 0. +0.j]
     [0. +0.j 0.5+0.j]]

    Notes
    -----
    The convention used is that qubit 0 is the most significant bit.
    For a 2-qubit system:

    - Index 0: |00⟩ (qubit 0 = 0, qubit 1 = 0)
    - Index 1: |01⟩ (qubit 0 = 0, qubit 1 = 1)
    - Index 2: |10⟩ (qubit 0 = 1, qubit 1 = 0)
    - Index 3: |11⟩ (qubit 0 = 1, qubit 1 = 1)

    See Also
    --------
    partial_trace_subsystem : Trace out arbitrary qubits.
    compute_purity : Compute purity of a density matrix.
    """
    # Validate inputs
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")

    if not isinstance(keep_qubit, int):
        raise ValueError(f"keep_qubit must be an integer, got {type(keep_qubit)}")

    if keep_qubit < 0 or keep_qubit >= n_qubits:
        raise ValueError(
            f"keep_qubit must be in range [0, {n_qubits - 1}], got {keep_qubit}"
        )

    # Validate statevector
    state = validate_statevector(statevector, expected_qubits=n_qubits)

    # Special case: single qubit system
    if n_qubits == 1:
        return np.outer(state, np.conj(state))

    # Reshape statevector to tensor form
    # Shape: (2, 2, 2, ...) with n_qubits dimensions
    tensor_shape = [2] * n_qubits

    # Compute full density matrix as outer product
    # ρ = |ψ⟩⟨ψ| has shape (2^n, 2^n) = (2, 2, ..., 2, 2, ...) tensor
    rho_full = np.outer(state, np.conj(state))
    rho_tensor = rho_full.reshape(tensor_shape + tensor_shape)

    # Trace out all qubits except keep_qubit
    # We need to contract pairs of indices (i, i+n_qubits) for i != keep_qubit
    #
    # Strategy: Use einsum for clean implementation
    # Create einsum subscripts that trace out unwanted qubits

    # Build einsum string
    # Input: indices 0..n-1 for ket, n..2n-1 for bra
    # Output: keep_qubit for ket, keep_qubit+n for bra
    # Traced pairs: (i, i+n) for i != keep_qubit

    output_indices = [keep_qubit, keep_qubit + n_qubits]

    # For traced qubits, we need both ket and bra indices to be the same
    # So we replace bra index (i+n_qubits) with ket index (i) for traced qubits
    modified_input = list(range(2 * n_qubits))
    for i in range(n_qubits):
        if i != keep_qubit:
            modified_input[i + n_qubits] = i  # Contract bra with ket

    # Convert to einsum format using letters
    def idx_to_letter(idx: int) -> str:
        """Convert index to einsum letter."""
        return chr(ord("a") + idx)

    input_str = "".join(idx_to_letter(i) for i in modified_input)
    output_str = "".join(idx_to_letter(i) for i in output_indices)

    einsum_str = f"{input_str}->{output_str}"

    # Perform the partial trace
    rho_reduced = np.einsum(einsum_str, rho_tensor)

    # Ensure output is 2x2 complex matrix
    rho_reduced = rho_reduced.reshape(2, 2).astype(np.complex128)

    # Verify trace = 1 (within numerical precision)
    trace_val = np.trace(rho_reduced)
    if not np.isclose(np.real(trace_val), 1.0, atol=1e-10):
        _logger.warning(
            "Reduced density matrix trace is %.10f, renormalizing",
            np.real(trace_val),
        )
        rho_reduced = rho_reduced / trace_val

    return rho_reduced


def partial_trace_subsystem(
    statevector: StatevectorType,
    n_qubits: int,
    keep_qubits: Sequence[int],
) -> DensityMatrixType:
    """Compute partial trace, keeping a specified subsystem.

    This function traces out all qubits not in the keep list,
    returning the reduced density matrix for the kept subsystem.

    Parameters
    ----------
    statevector : StatevectorType
        Full system statevector of shape ``(2^n_qubits,)``.
    n_qubits : int
        Total number of qubits in the system.
    keep_qubits : Sequence[int]
        Indices of qubits to keep (0-indexed). Must be a non-empty
        sequence of unique integers in range ``[0, n_qubits-1]``.

    Returns
    -------
    DensityMatrixType
        Reduced density matrix for the kept subsystem, shape
        ``(2^len(keep_qubits), 2^len(keep_qubits))``.

    Raises
    ------
    ValueError
        If ``keep_qubits`` is empty, contains duplicates, or has
        out-of-range indices.
    ValidationError
        If the statevector is invalid.

    Examples
    --------
    **Example 1: Keep the first two qubits of a 3-qubit product state**

    >>> import numpy as np
    >>> # |000⟩ state
    >>> state = np.zeros(8, dtype=complex)
    >>> state[0] = 1.0
    >>> rho_01 = partial_trace_subsystem(state, n_qubits=3, keep_qubits=[0, 1])
    >>> print(rho_01.shape)
    (4, 4)
    >>> # Result is |00⟩⟨00| (pure state)
    >>> print(np.round(rho_01, 3))
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]

    **Example 2: GHZ state partial trace (keeping first qubit)**

    For the GHZ state (|000⟩ + |111⟩)/√2, tracing out qubits 1 and 2
    leaves qubit 0 in a maximally mixed state:

    >>> ghz = np.zeros(8, dtype=complex)
    >>> ghz[0] = 1.0 / np.sqrt(2)  # |000⟩
    >>> ghz[7] = 1.0 / np.sqrt(2)  # |111⟩
    >>> rho_0 = partial_trace_subsystem(ghz, n_qubits=3, keep_qubits=[0])
    >>> # Single qubit is maximally mixed (I/2)
    >>> print(np.round(rho_0, 3))
    [[0.5+0.j 0. +0.j]
     [0. +0.j 0.5+0.j]]

    **Example 3: Non-contiguous qubit selection**

    Keep qubits 0 and 2, tracing out qubit 1:

    >>> # |000⟩ state
    >>> state = np.zeros(8, dtype=complex)
    >>> state[0] = 1.0
    >>> rho_02 = partial_trace_subsystem(state, n_qubits=3, keep_qubits=[0, 2])
    >>> print(rho_02.shape)
    (4, 4)
    >>> # Result is |00⟩⟨00| for qubits 0 and 2
    >>> print(np.round(np.diag(rho_02).real, 3))
    [1. 0. 0. 0.]

    **Example 4: W state partial trace**

    The W state (|001⟩ + |010⟩ + |100⟩)/√3 has different entanglement
    structure than GHZ. Tracing out one qubit leaves a mixed state:

    >>> w_state = np.zeros(8, dtype=complex)
    >>> w_state[1] = 1.0 / np.sqrt(3)  # |001⟩
    >>> w_state[2] = 1.0 / np.sqrt(3)  # |010⟩
    >>> w_state[4] = 1.0 / np.sqrt(3)  # |100⟩
    >>> rho_01 = partial_trace_subsystem(w_state, n_qubits=3, keep_qubits=[0, 1])
    >>> # Trace = 1, but not maximally mixed
    >>> print(f"Trace: {np.trace(rho_01).real:.3f}")
    Trace: 1.000

    Notes
    -----
    The ordering of qubits in the output follows the order specified
    in ``keep_qubits``. For example, if ``keep_qubits=[1, 0]``, the
    first index of the output density matrix corresponds to qubit 1.

    **Qubit Indexing Convention**

    Qubit 0 is the most significant bit in the computational basis:

    - For 3 qubits: |q0 q1 q2⟩
    - Index 0 = |000⟩, Index 1 = |001⟩, Index 4 = |100⟩, Index 7 = |111⟩

    **Use Cases**

    - Computing entanglement entropy of subsystems
    - Analyzing local properties of entangled states
    - Verifying separability of quantum states
    """
    # Validate inputs
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")

    keep_list = list(keep_qubits)

    if len(keep_list) == 0:
        raise ValueError("keep_qubits must not be empty")

    if len(keep_list) != len(set(keep_list)):
        raise ValueError("keep_qubits contains duplicate indices")

    for q in keep_list:
        if not isinstance(q, (int, np.integer)):
            raise ValueError(f"keep_qubits must contain integers, got {type(q)}")
        if q < 0 or q >= n_qubits:
            raise ValueError(f"Qubit index {q} out of range [0, {n_qubits - 1}]")

    # Validate statevector
    state = validate_statevector(statevector, expected_qubits=n_qubits)

    # Special case: keeping all qubits
    if len(keep_list) == n_qubits:
        return np.outer(state, np.conj(state))

    # Compute using tensor reshaping and einsum
    trace_qubits = [q for q in range(n_qubits) if q not in keep_list]
    n_keep = len(keep_list)
    dim_keep = 2**n_keep

    # Reshape to tensor
    tensor_shape = [2] * n_qubits

    # Full density matrix tensor
    rho_full = np.outer(state, np.conj(state))
    rho_tensor = rho_full.reshape(tensor_shape + tensor_shape)

    # Build einsum contraction
    # Trace over qubits not in keep_list by making their ket and bra indices equal
    modified_input = list(range(2 * n_qubits))
    for q in trace_qubits:
        modified_input[q + n_qubits] = q

    # Output indices: keep_qubits in specified order, then their bra counterparts
    output_indices = keep_list + [q + n_qubits for q in keep_list]

    def idx_to_letter(idx: int) -> str:
        return chr(ord("a") + idx)

    input_str = "".join(idx_to_letter(i) for i in modified_input)
    output_str = "".join(idx_to_letter(i) for i in output_indices)

    einsum_str = f"{input_str}->{output_str}"

    rho_reduced = np.einsum(einsum_str, rho_tensor)
    rho_reduced = rho_reduced.reshape(dim_keep, dim_keep).astype(np.complex128)

    # Verify trace
    trace_val = np.trace(rho_reduced)
    if not np.isclose(np.real(trace_val), 1.0, atol=1e-10):
        _logger.warning(
            "Reduced density matrix trace is %.10f, renormalizing",
            np.real(trace_val),
        )
        rho_reduced = rho_reduced / trace_val

    return rho_reduced


# =============================================================================
# Fidelity and Purity Functions
# =============================================================================


def compute_fidelity(
    state1: StatevectorType,
    state2: StatevectorType,
) -> float:
    """Compute the fidelity between two pure quantum states.

    The fidelity between two pure states is defined as:

        F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²

    This measures how "close" two quantum states are, with F = 1 for
    identical states (up to global phase) and F = 0 for orthogonal states.

    Parameters
    ----------
    state1 : StatevectorType
        First statevector.
    state2 : StatevectorType
        Second statevector. Must have the same dimension as ``state1``.

    Returns
    -------
    float
        Fidelity value in the range ``[0, 1]``.

    Raises
    ------
    ValueError
        If states have different dimensions.
    ValidationError
        If states are invalid.

    Examples
    --------
    Identical states have fidelity 1:

    >>> import numpy as np
    >>> state = np.array([1, 0, 0, 0], dtype=complex)
    >>> fidelity = compute_fidelity(state, state)
    >>> print(f"Fidelity: {fidelity}")
    Fidelity: 1.0

    Orthogonal states have fidelity 0:

    >>> state1 = np.array([1, 0], dtype=complex)  # |0⟩
    >>> state2 = np.array([0, 1], dtype=complex)  # |1⟩
    >>> fidelity = compute_fidelity(state1, state2)
    >>> print(f"Fidelity: {fidelity}")
    Fidelity: 0.0

    States differing by a global phase have fidelity 1:

    >>> state1 = np.array([1, 0], dtype=complex)
    >>> state2 = np.array([1j, 0], dtype=complex)  # |0⟩ with phase
    >>> fidelity = compute_fidelity(state1, state2)
    >>> print(f"Fidelity: {fidelity:.6f}")
    Fidelity: 1.000000

    Notes
    -----
    The implementation uses :func:`numpy.vdot` which conjugates the first
    argument, computing ⟨state1|state2⟩ = Σᵢ state1ᵢ* · state2ᵢ.

    For mixed states (density matrices), use the more general formula:
    F(ρ, σ) = (Tr√(√ρ σ √ρ))²

    See Also
    --------
    compute_purity : Compute purity of a density matrix.
    """
    # Convert to numpy arrays
    s1 = np.asarray(state1, dtype=np.complex128).ravel()
    s2 = np.asarray(state2, dtype=np.complex128).ravel()

    # Check dimensions match
    if len(s1) != len(s2):
        raise ValueError(
            f"States must have same dimension: got {len(s1)} and {len(s2)}"
        )

    # Check for invalid values
    if np.any(np.isnan(s1)) or np.any(np.isinf(s1)):
        raise ValidationError("state1 contains NaN or infinite values")
    if np.any(np.isnan(s2)) or np.any(np.isinf(s2)):
        raise ValidationError("state2 contains NaN or infinite values")

    # Compute inner product ⟨ψ₁|ψ₂⟩
    # np.vdot conjugates the first argument
    overlap = np.vdot(s1, s2)

    # Fidelity = |⟨ψ₁|ψ₂⟩|²
    fidelity = np.abs(overlap) ** 2

    # Clamp to [0, 1] for numerical stability
    # Small numerical errors can push the value slightly outside this range
    fidelity = float(np.clip(fidelity, 0.0, 1.0))

    return fidelity


def _compute_fidelities_batch(
    states1: NDArray[np.complexfloating[Any, Any]],
    states2: NDArray[np.complexfloating[Any, Any]],
) -> NDArray[np.floating[Any]]:
    """Compute fidelities between pairs of statevectors in batch.

    Vectorized version of :func:`compute_fidelity` for arrays of states.
    Computes F_i = |⟨ψ₁ⁱ|ψ₂ⁱ⟩|² for each pair (i).

    Parameters
    ----------
    states1 : NDArray[np.complexfloating], shape ``(n, d)``
        First set of statevectors.
    states2 : NDArray[np.complexfloating], shape ``(n, d)``
        Second set of statevectors.

    Returns
    -------
    NDArray[np.floating], shape ``(n,)``
        Fidelity values, each in [0, 1].

    Raises
    ------
    ValueError
        If shapes of ``states1`` and ``states2`` do not match.
    ValidationError
        If any state contains NaN or infinite values.
    """
    if states1.shape != states2.shape:
        raise ValueError(
            f"States must have same shape: got {states1.shape} and {states2.shape}"
        )
    if states1.ndim != 2:
        raise ValueError(f"States must be 2D arrays, got ndim={states1.ndim}")

    if np.any(np.isnan(states1)) or np.any(np.isinf(states1)):
        raise ValidationError("states1 contains NaN or infinite values")
    if np.any(np.isnan(states2)) or np.any(np.isinf(states2)):
        raise ValidationError("states2 contains NaN or infinite values")

    overlaps = np.sum(np.conj(states1) * states2, axis=1)
    fidelities = np.abs(overlaps) ** 2
    fidelities = np.clip(fidelities, 0.0, 1.0).astype(np.float64)

    return fidelities


def compute_purity(
    density_matrix: DensityMatrixType,
) -> float:
    """Compute the purity of a density matrix.

    The purity of a quantum state ρ is defined as:

        γ(ρ) = Tr(ρ²)

    Properties:
        - For a pure state: γ = 1
        - For a maximally mixed state of dimension d: γ = 1/d
        - In general: 1/d ≤ γ ≤ 1

    Parameters
    ----------
    density_matrix : DensityMatrixType
        Density matrix (must be square). Can be any valid quantum state,
        including mixed states.

    Returns
    -------
    float
        Purity value in the range ``[1/d, 1]`` where d is the dimension.

    Raises
    ------
    ValidationError
        If the density matrix is not square or contains invalid values.

    Examples
    --------
    Pure state |0⟩ has purity 1:

    >>> import numpy as np
    >>> rho_pure = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> purity = compute_purity(rho_pure)
    >>> print(f"Purity: {purity}")
    Purity: 1.0

    Maximally mixed state has purity 1/d:

    >>> rho_mixed = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    >>> purity = compute_purity(rho_mixed)
    >>> print(f"Purity: {purity}")
    Purity: 0.5

    Notes
    -----
    The purity is related to the linear entropy by:
        S_L(ρ) = 1 - γ(ρ)

    And to the participation ratio by:
        PR = 1/γ(ρ)

    which measures the "effective dimension" of the state.

    See Also
    --------
    compute_linear_entropy : Compute linear entropy from purity.
    compute_fidelity : Compute fidelity between pure states.
    """
    rho = np.asarray(density_matrix, dtype=np.complex128)

    # Validate shape
    if rho.ndim != 2:
        raise ValidationError(f"Density matrix must be 2D, got shape {rho.shape}")
    if rho.shape[0] != rho.shape[1]:
        raise ValidationError(f"Density matrix must be square, got shape {rho.shape}")

    # Check for invalid values
    if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
        raise ValidationError("Density matrix contains NaN or infinite values")

    # Compute Tr(ρ²)
    # Using matrix multiplication: Tr(ρ²) = Tr(ρ @ ρ)
    rho_squared = rho @ rho
    purity = np.real(np.trace(rho_squared))

    # Clamp to valid range [1/d, 1]
    d = rho.shape[0]
    min_purity = 1.0 / d
    purity = float(np.clip(purity, min_purity - _EPSILON, 1.0 + _EPSILON))

    return purity


def compute_linear_entropy(
    density_matrix: DensityMatrixType,
) -> float:
    """Compute the linear entropy of a density matrix.

    The linear entropy is a measure of mixedness defined as:

        S_L(ρ) = 1 - Tr(ρ²) = 1 - γ(ρ)

    where γ(ρ) is the purity.

    Properties:
        - For a pure state: S_L = 0
        - For a maximally mixed state of dimension d: S_L = 1 - 1/d
        - In general: 0 ≤ S_L ≤ 1 - 1/d

    Parameters
    ----------
    density_matrix : DensityMatrixType
        Density matrix (must be square).

    Returns
    -------
    float
        Linear entropy value in the range ``[0, 1 - 1/d]``.

    Examples
    --------
    Pure state has zero linear entropy:

    >>> import numpy as np
    >>> rho_pure = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> entropy = compute_linear_entropy(rho_pure)
    >>> print(f"Linear entropy: {entropy}")
    Linear entropy: 0.0

    Maximally mixed state has maximum linear entropy:

    >>> rho_mixed = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    >>> entropy = compute_linear_entropy(rho_mixed)
    >>> print(f"Linear entropy: {entropy}")
    Linear entropy: 0.5

    See Also
    --------
    compute_purity : Compute purity (1 - linear entropy).
    compute_von_neumann_entropy : Compute von Neumann entropy.
    """
    purity = compute_purity(density_matrix)
    linear_entropy = 1.0 - purity
    return float(np.clip(linear_entropy, 0.0, 1.0))


def compute_von_neumann_entropy(
    density_matrix: DensityMatrixType,
) -> float:
    """Compute the von Neumann entropy of a density matrix.

    The von Neumann entropy is the quantum analog of Shannon entropy:

        S(ρ) = -Tr(ρ log₂ ρ) = -Σᵢ λᵢ log₂(λᵢ)

    where λᵢ are the eigenvalues of ρ.

    Properties:
        - For a pure state: S = 0
        - For a maximally mixed state of dimension d: S = log₂(d)
        - In general: 0 ≤ S ≤ log₂(d)

    Parameters
    ----------
    density_matrix : DensityMatrixType
        Density matrix (must be square).

    Returns
    -------
    float
        Von Neumann entropy in bits (log base 2).

    Examples
    --------
    Pure state has zero entropy:

    >>> import numpy as np
    >>> rho_pure = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> entropy = compute_von_neumann_entropy(rho_pure)
    >>> print(f"von Neumann entropy: {entropy:.6f}")
    von Neumann entropy: 0.000000

    Maximally mixed state has maximum entropy:

    >>> rho_mixed = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    >>> entropy = compute_von_neumann_entropy(rho_mixed)
    >>> print(f"von Neumann entropy: {entropy:.6f}")
    von Neumann entropy: 1.000000

    Notes
    -----
    The computation uses the eigenvalue decomposition of the density matrix.
    Eigenvalues that are zero or negative (due to numerical errors) are
    filtered out to avoid log(0) issues.

    See Also
    --------
    compute_linear_entropy : Compute linear entropy (simpler approximation).
    compute_purity : Compute state purity.
    """
    rho = np.asarray(density_matrix, dtype=np.complex128)

    # Validate
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValidationError(
            f"Density matrix must be square 2D array, got shape {rho.shape}"
        )

    # Get eigenvalues (should be real for Hermitian matrix)
    eigenvalues = np.linalg.eigvalsh(rho)

    # Filter to positive eigenvalues only (handle numerical errors)
    positive_eigenvalues = eigenvalues[eigenvalues > _EPSILON]

    if len(positive_eigenvalues) == 0:
        # All eigenvalues are effectively zero - maximally mixed or invalid
        _logger.warning("All eigenvalues are zero or negative")
        return 0.0

    # S = -Σᵢ λᵢ log₂(λᵢ)
    entropy = -np.sum(positive_eigenvalues * np.log2(positive_eigenvalues))

    # Clamp to valid range
    d = rho.shape[0]
    max_entropy = np.log2(d)
    entropy = float(np.clip(entropy, 0.0, max_entropy + _EPSILON))

    return entropy


# =============================================================================
# Gradient Computation Functions
# =============================================================================


def compute_parameter_gradient(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    param_index: int,
    observable: Literal[
        "computational", "global_z", "local_z", "pauli_z"
    ] = "computational",
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> float:
    """Compute gradient of expectation value with respect to one parameter.

    Uses the parameter-shift rule for quantum gradients:

        ∂⟨O⟩/∂θ = (⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}) / 2

    This is exact for gates of the form U(θ) = exp(-iθG/2) where G² = I,
    which includes all standard rotation gates (RX, RY, RZ).

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance.
    x : NDArray[np.floating]
        Input data vector.
    param_index : int
        Index of the parameter to differentiate with respect to.
    observable : {"computational", "global_z", "local_z", "pauli_z"}, default="computational"
        Observable to measure:

        - ``"computational"``: Probability of |0...0⟩ state, i.e., ⟨0...0|ρ|0...0⟩
        - ``"global_z"``: Expectation value of Z⊗Z⊗...⊗Z (tensor product of Z on
          all qubits). Eigenvalue is (-1)^(number of 1s in bitstring).
        - ``"local_z"``: Expectation value of Z on first qubit only, ⟨Z₀⟩.
          Computed as P(0 on qubit 0) - P(1 on qubit 0) = 2·P(0) - 1.
        - ``"pauli_z"``: **Deprecated alias for "global_z"**. For clarity,
          use ``"global_z"`` for the global Z-string observable or ``"local_z"``
          for the single-qubit Z observable.

    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Simulation backend.

    Returns
    -------
    float
        Gradient of the expectation value.

    Raises
    ------
    ValidationError
        If param_index is out of range.
    ValueError
        If observable is not a recognized option.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> import numpy as np
    >>>
    >>> enc = AngleEncoding(n_features=2)
    >>> x = np.array([0.5, 1.0])
    >>> grad = compute_parameter_gradient(enc, x, param_index=0)
    >>> print(f"Gradient: {grad:.6f}")

    Notes
    -----
    The parameter-shift rule requires two circuit evaluations per gradient
    component. For full gradient vectors, use :func:`compute_all_parameter_gradients`.

    **Observable Semantics**

    The distinction between ``"global_z"`` and ``"local_z"`` is important:

    - ``"global_z"`` measures the parity of all qubits simultaneously. The
      expectation value is ⟨Z⊗Z⊗...⊗Z⟩ where each Z acts on a different qubit.
      This captures global correlations in the encoded state.
    - ``"local_z"`` measures only the first qubit (MSB in computational basis
      ordering), ignoring the state of other qubits. This is useful for
      local sensitivity analysis.

    See Also
    --------
    compute_all_parameter_gradients : Compute gradients for all parameters.
    """
    x_array = np.asarray(x, dtype=np.float64).copy()

    if param_index < 0 or param_index >= len(x_array):
        raise ValidationError(
            f"param_index {param_index} out of range [0, {len(x_array) - 1}]"
        )

    # Parameter shift value (π/2 for standard rotation gates)
    shift = np.pi / 2.0

    # Forward shift
    x_plus = x_array.copy()
    x_plus[param_index] += shift

    # Backward shift
    x_minus = x_array.copy()
    x_minus[param_index] -= shift

    # Simulate both circuits
    state_plus = simulate_encoding_statevector(encoding, x_plus, backend)
    state_minus = simulate_encoding_statevector(encoding, x_minus, backend)

    # Compute expectation values based on observable type
    if observable == "computational":
        # ⟨0...0|ρ|0...0⟩ = |⟨0...0|ψ⟩|² = |ψ[0]|²
        # Probability of measuring all qubits in state |0⟩
        exp_plus = np.abs(state_plus[0]) ** 2
        exp_minus = np.abs(state_minus[0]) ** 2

    elif observable in ("global_z", "pauli_z"):
        # Handle deprecated "pauli_z" alias
        if observable == "pauli_z":
            warnings.warn(
                'Observable "pauli_z" is deprecated. Use "global_z" for the '
                'global Z⊗Z⊗...⊗Z observable, or "local_z" for Z on the first '
                "qubit only. This will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        # ⟨Z⊗Z⊗...⊗Z⟩ - Global Pauli Z-string expectation value
        # For computational basis state |i⟩, the eigenvalue is (-1)^(popcount(i))
        # where popcount(i) is the number of 1-bits (Hamming weight).
        n = len(state_plus)
        z_eigenvalues = np.array(
            [1.0 - 2.0 * (bin(i).count("1") % 2) for i in range(n)],
            dtype=np.float64,
        )
        exp_plus = np.real(np.sum(z_eigenvalues * np.abs(state_plus) ** 2))
        exp_minus = np.real(np.sum(z_eigenvalues * np.abs(state_minus) ** 2))

    elif observable == "local_z":
        # ⟨Z₀⟩ - Expectation value of Z on the first qubit (MSB)
        # = P(0 on qubit 0) - P(1 on qubit 0) = 2·P(0) - 1
        #
        # The first qubit (MSB) is 0 when bit (n_qubits-1) is 0.
        # For a statevector of dimension 2^n, indices 0 to 2^(n-1)-1 have
        # the first qubit in state |0⟩.
        n_qubits = int(np.log2(len(state_plus)))
        exp_plus = _compute_local_z_expectation(state_plus, n_qubits)
        exp_minus = _compute_local_z_expectation(state_minus, n_qubits)

    else:
        raise ValueError(
            f"Unknown observable: {observable!r}. "
            f'Valid options are: "computational", "global_z", "local_z", "pauli_z".'
        )

    # Parameter-shift formula
    gradient = (exp_plus - exp_minus) / 2.0

    return float(gradient)


def _compute_local_z_expectation(
    statevector: NDArray[np.complexfloating[Any, Any]],
    n_qubits: int,
) -> float:
    """Compute expectation value of Z on the first qubit (MSB).

    The first qubit corresponds to the most significant bit (MSB) in the
    computational basis ordering used by this library.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        Statevector of shape (2^n_qubits,).
    n_qubits : int
        Number of qubits in the system.

    Returns
    -------
    float
        Expectation value ⟨Z₀⟩ = P(0 on qubit 0) - P(1 on qubit 0).

    Notes
    -----
    **Qubit Ordering Convention**

    This library uses MSB (most significant bit) ordering:
    - Qubit 0 corresponds to the leftmost (highest) bit
    - For index i, qubit 0 is in state |0⟩ if bit (n_qubits-1) of i is 0

    For example, with 3 qubits:
    - |000⟩ = index 0: qubit 0 = 0
    - |001⟩ = index 1: qubit 0 = 0
    - |010⟩ = index 2: qubit 0 = 0
    - |011⟩ = index 3: qubit 0 = 0
    - |100⟩ = index 4: qubit 0 = 1
    - |101⟩ = index 5: qubit 0 = 1
    - |110⟩ = index 6: qubit 0 = 1
    - |111⟩ = index 7: qubit 0 = 1

    So indices 0-3 have qubit 0 in state |0⟩, and indices 4-7 have qubit 0
    in state |1⟩.
    """
    dim = 2**n_qubits

    # Sum probabilities where first qubit (MSB) is 0
    # First qubit is 0 when bit (n_qubits-1) is 0, i.e., for indices < dim/2
    prob_zero = 0.0
    for i in range(dim):
        # Check if the MSB (qubit 0) is 0
        if (i >> (n_qubits - 1)) & 1 == 0:
            prob_zero += np.abs(statevector[i]) ** 2

    # ⟨Z₀⟩ = P(0) - P(1) = P(0) - (1 - P(0)) = 2·P(0) - 1
    expectation = 2.0 * prob_zero - 1.0

    return float(expectation)


def compute_all_parameter_gradients(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    observable: Literal[
        "computational", "global_z", "local_z", "pauli_z"
    ] = "computational",
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> FloatArray:
    """Compute gradients for all parameters using parameter-shift rule.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance.
    x : NDArray[np.floating]
        Input data vector.
    observable : {"computational", "global_z", "local_z", "pauli_z"}, default="computational"
        Observable to measure. See :func:`compute_parameter_gradient` for details.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Simulation backend.

    Returns
    -------
    FloatArray
        Array of gradients, one per parameter.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> import numpy as np
    >>>
    >>> enc = AngleEncoding(n_features=3)
    >>> x = np.array([0.5, 1.0, 1.5])
    >>> gradients = compute_all_parameter_gradients(enc, x)
    >>> print(f"Gradients shape: {gradients.shape}")
    Gradients shape: (3,)
    """
    x_array = np.asarray(x, dtype=np.float64)
    n_params = len(x_array)

    gradients = np.zeros(n_params, dtype=np.float64)

    for i in range(n_params):
        gradients[i] = compute_parameter_gradient(
            encoding, x_array, i, observable, backend
        )

    return gradients


# =============================================================================
# Random Sampling Functions
# =============================================================================


def create_rng(seed: int | None = None) -> np.random.Generator:
    """Create a numpy random number generator.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. If None, uses system entropy.

    Returns
    -------
    Generator
        NumPy random number generator instance.

    Examples
    --------
    >>> rng = create_rng(42)
    >>> values = rng.random(5)
    >>> # Same seed produces same values
    >>> rng2 = create_rng(42)
    >>> values2 = rng2.random(5)
    >>> np.allclose(values, values2)
    True
    """
    return np.random.default_rng(seed)


def generate_random_parameters(
    encoding_or_n_features: BaseEncoding | int,
    n_samples: int = 1,
    param_min: float = _DEFAULT_PARAM_MIN,
    param_max: float = _DEFAULT_PARAM_MAX,
    seed: int | None = None,
) -> FloatArray:
    """Generate random parameter vectors for encoding analysis.

    Parameters
    ----------
    encoding_or_n_features : BaseEncoding or int
        Either an encoding instance (``n_features`` is extracted
        automatically) or an integer specifying the number of features
        directly.
    n_samples : int, default=1
        Number of parameter vectors to generate.
    param_min : float, default=0.0
        Minimum parameter value.
    param_max : float, default=2π
        Maximum parameter value.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    FloatArray
        Random parameters of shape ``(n_samples, n_features)`` if
        ``n_samples > 1``, or ``(n_features,)`` if ``n_samples == 1``.

    Examples
    --------
    Generate parameters from an encoding:

    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> params = generate_random_parameters(enc, n_samples=10, seed=42)
    >>> print(params.shape)
    (10, 4)

    Generate parameters by specifying n_features directly:

    >>> params = generate_random_parameters(4, seed=42)
    >>> print(params.shape)
    (4,)

    With custom range:

    >>> params = generate_random_parameters(
    ...     4, n_samples=10, param_min=-np.pi, param_max=np.pi
    ... )
    """
    # Accept either an encoding instance or an integer
    if isinstance(encoding_or_n_features, BaseEncoding):
        n_features = encoding_or_n_features.n_features
    elif isinstance(encoding_or_n_features, (int, np.integer)):
        n_features = int(encoding_or_n_features)
    else:
        raise TypeError(
            f"Expected BaseEncoding or int, got {type(encoding_or_n_features).__name__}"
        )

    if n_features < 1:
        raise ValueError(f"n_features must be at least 1, got {n_features}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if param_min >= param_max:
        raise ValueError(
            f"param_min must be less than param_max: {param_min} >= {param_max}"
        )

    rng = create_rng(seed)

    # Generate uniform random parameters
    if n_samples == 1:
        params = rng.uniform(param_min, param_max, size=n_features)
    else:
        params = rng.uniform(param_min, param_max, size=(n_samples, n_features))

    return params.astype(np.float64)
