"""Amplitude-based quantum data encoding module.

This module implements AmplitudeEncoding, a quantum data encoding technique
that maps classical features directly to the amplitudes of a quantum state.
This provides exponential compression: n classical features can be encoded
using only log₂(n) qubits.

Amplitude encoding is fundamental in quantum machine learning, offering:

1. **Exponential Compression**: n features stored in log₂(n) qubits
2. **Full Hilbert Space Utilization**: Accesses the full 2^n dimensional space
3. **Natural for Quantum Algorithms**: Direct input for algorithms like HHL, QSVM
4. **Preserves Geometric Structure**: Normalized data maps to unit sphere

The encoding creates quantum states of the form:

    |ψ(x)⟩ = Σᵢ xᵢ|i⟩  where Σ|xᵢ|² = 1

where xᵢ is the i-th normalized feature value and |i⟩ is the computational
basis state corresponding to the binary representation of index i.

Mathematical Background
-----------------------
Amplitude encoding embeds an n-dimensional classical vector x = (x₀, ..., xₙ₋₁)
into the amplitudes of a quantum state:

    |ψ(x)⟩ = (1/‖x‖) Σᵢ xᵢ|i⟩

For n features, we need ⌈log₂(n)⌉ qubits to represent all basis states.
If n is not a power of 2, the state is zero-padded to the next power of 2.

**State Preparation Complexity**:
General amplitude encoding requires O(2^n) gates for n qubits, making it
exponentially expensive. This is a fundamental limitation, as preparing
an arbitrary quantum state is known to require exponential resources in
the worst case.

Note on Data Preprocessing
--------------------------
Input data should typically be preprocessed before amplitude encoding:

- **Normalization**: The encoding automatically normalizes input vectors
  to create valid quantum states (‖ψ‖ = 1). This can be disabled if data
  is already normalized.

- **Feature Scaling**: Consider scaling features to similar ranges before
  encoding to prevent any single feature from dominating the state.

- **Power-of-Two Padding**: Non-power-of-two feature counts are automatically
  padded with zeros. Consider padding with meaningful values if zeros would
  bias downstream quantum algorithms.

Use Cases
---------
Amplitude encoding is particularly suited for:

- **Quantum linear solvers (HHL algorithm)**: The HHL algorithm requires input
  vectors encoded as quantum state amplitudes to solve linear systems Ax = b
  with exponential speedup for sparse, well-conditioned matrices.

- **Quantum kernel methods (QSVM)**: Quantum support vector machines use
  amplitude-encoded feature vectors to compute quantum kernels that may
  capture complex patterns inaccessible to classical kernels.

- **Quantum neural networks**: Serves as an input encoding layer for variational
  quantum circuits, enabling the processing of classical data in quantum
  neural network architectures.

- **Quantum principal component analysis (QPCA)**: Amplitude encoding enables
  quantum PCA algorithms that can extract principal components with exponential
  speedup for low-rank density matrices.

- **Quantum random access memory (QRAM)**: Forms the basis for QRAM
  implementations where classical data must be queried in superposition,
  enabling quantum speedups in database search and machine learning.

- **Quantum sampling and Monte Carlo**: Amplitude encoding can represent
  probability distributions for quantum-enhanced sampling algorithms.

Limitations
-----------
Users should be aware of these critical constraints:

- **O(2^n) gate complexity**: General state preparation requires exponentially
  many gates in the number of qubits. For n qubits, the circuit contains
  O(2^n) gates, which grows rapidly even for modest qubit counts.

- **Exponential circuit depth**: The deep circuits required for amplitude
  encoding accumulate significant errors on NISQ (Noisy Intermediate-Scale
  Quantum) hardware, limiting practical applicability to small feature counts
  or fault-tolerant quantum computers.

- **Normalization destroys magnitude information**: Since quantum states must
  satisfy |ψ|² = 1, the original L2 norm of the input vector is lost during
  encoding. Only relative magnitudes (ratios between features) are preserved.

- **Memory-intensive classical simulation**: Simulating amplitude encoding
  circuits requires storing the full 2^n-dimensional state vector. The Cirq
  backend additionally constructs a 2^n × 2^n unitary matrix, consuming
  O(4^n) memory (see ``resource_summary()`` for estimates).

- **Limited parallelization**: Unlike angle encoding where each qubit can be
  prepared independently, amplitude encoding creates entangled states that
  cannot be trivially parallelized across qubits.

Resource Analysis
-----------------
AmplitudeEncoding provides methods to analyze circuit resources and plan
hardware execution:

- ``gate_count_breakdown()``: Get estimated gate counts by type (rotations, CNOTs).
  Returns theoretical estimates based on Möttönen et al. decomposition.

- ``resource_summary()``: Comprehensive breakdown of circuit resources including
  qubit count, state dimension, compression ratio, gate count estimates,
  and memory requirements for each backend.

- ``properties``: Access the ``EncodingProperties`` dataclass with theoretical
  circuit metrics (depth, gate counts, simulability classification).

These methods help understand resource requirements before execution:

    >>> enc = AmplitudeEncoding(n_features=16)
    >>> summary = enc.resource_summary()
    >>> summary['n_qubits']
    4
    >>> summary['compression_ratio']
    4.0
    >>> summary['cirq_unitary_memory_human']
    '4.0 KB'

    >>> # For larger feature counts, memory becomes significant
    >>> enc_large = AmplitudeEncoding(n_features=1024)
    >>> summary = enc_large.resource_summary()
    >>> summary['n_qubits']
    10
    >>> summary['cirq_unitary_memory_human']
    '16.0 MB'

References
----------
.. [1] Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
       Quantum Computers." Springer. Chapter 4: Quantum Feature Maps.
.. [2] Mottonen, M., et al. (2004). "Transformation of quantum states using
       uniformly controlled rotations." Quantum Information & Computation.
.. [3] Shende, V., Bullock, S., & Markov, I. (2006). "Synthesis of quantum
       logic circuits." IEEE Transactions on CAD.
.. [4] Araujo, I. F., et al. (2021). "A divide-and-conquer algorithm for
       quantum state preparation." Scientific Reports, 11, 6329.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

# =============================================================================
# Module-Level Logger
# =============================================================================

# Configure module-level logger for production debugging.
#
# Usage:
#   To enable debug logging for amplitude encoding:
#
#   >>> import logging
#   >>> logging.getLogger('encoding_atlas.encodings.amplitude').setLevel(logging.DEBUG)
#   >>> handler = logging.StreamHandler()
#   >>> handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
#   >>> logging.getLogger('encoding_atlas.encodings.amplitude').addHandler(handler)
#
# Log Levels:
#   - DEBUG: Detailed information for diagnosing issues (normalization values,
#            padding details, backend selection, state vector properties)
#   - INFO: General operational information (circuit generation started/completed)
#   - WARNING: Potential issues that don't prevent operation (handled by warnings module)
#
# The logger is named after the module to allow granular control over log output.
# By default, no handlers are attached, so no output is produced unless the
# application configures logging.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["AmplitudeEncoding"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Threshold for Cirq memory warning (in qubits).
#
# The Cirq backend constructs a full unitary matrix of size 2^n × 2^n for n qubits,
# where each element is a complex128 (16 bytes). Memory usage scales as:
#
#   Memory (bytes) = 2^(2n) × 16 = 4^n × 16
#
# Examples:
#   - 10 qubits: 16 MB
#   - 12 qubits: 256 MB (threshold)
#   - 13 qubits: 1 GB
#   - 14 qubits: 4 GB
#   - 15 qubits: 16 GB
#   - 16 qubits: 64 GB
#
# At 12 qubits (256 MB), the memory usage becomes significant enough to warrant
# a warning. Users on memory-constrained systems should consider alternative
# backends (PennyLane, Qiskit) which use more memory-efficient decompositions.
_CIRQ_MEMORY_WARNING_THRESHOLD_QUBITS: int = 12


def _format_memory_size(size_bytes: float) -> str:
    """Format a byte size as a human-readable string.

    Converts raw byte counts to appropriate units (KB, MB, GB, TB) with
    one decimal place precision.

    Parameters
    ----------
    size_bytes : float
        Size in bytes to format.

    Returns
    -------
    str
        Human-readable size string (e.g., "256.0 MB", "4.0 GB").

    Examples
    --------
    >>> _format_memory_size(1024)
    '1.0 KB'
    >>> _format_memory_size(268435456)
    '256.0 MB'
    >>> _format_memory_size(17179869184)
    '16.0 GB'
    """
    units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]
    for unit_name, unit_size in units:
        if size_bytes >= unit_size:
            return f"{size_bytes / unit_size:.1f} {unit_name}"
    return f"{size_bytes:.0f} bytes"


# Threshold for detecting truly zero input vectors.
#
# This threshold is used in get_circuit() to detect input vectors with
# zero or near-zero L2 norm, which cannot form valid quantum states.
# The value 1e-15 is chosen to be:
#   - Well above floating-point epsilon (~2.2e-16)
#   - Close enough to zero to only catch truly problematic inputs
#   - Permissive enough to allow very small but legitimate data
#
# Example: A vector [1e-14, 0, 0, 0] has norm 1e-14 > 1e-15, so it passes
# validation and gets normalized to [1, 0, 0, 0].
_ZERO_NORM_THRESHOLD: float = 1e-15


# =============================================================================
# Type Definitions
# =============================================================================


class GateCountBreakdown(TypedDict):
    """Type definition for the gate_count_breakdown() return value.

    Provides a detailed breakdown of gate counts for amplitude encoding.
    Note that amplitude encoding uses general state preparation, so gate
    counts are **theoretical estimates** based on the Möttönen et al.
    decomposition algorithm. Actual gate counts may vary by backend.

    .. note::
        Unlike encodings with fixed circuit structure (e.g., AngleEncoding,
        IQPEncoding), amplitude encoding creates data-dependent circuits
        where the exact gate count depends on the input amplitudes. These
        estimates represent the worst-case for general state preparation.

    Attributes
    ----------
    rotation_gates : int
        Estimated number of single-qubit rotation gates (RY/RZ).
        For general state preparation: approximately 2^n gates.

    cnot : int
        Estimated number of CNOT gates for creating entanglement.
        For general state preparation: approximately 2^n - 2 gates.

    total_single_qubit : int
        Total number of single-qubit gates (equals rotation_gates).

    total_two_qubit : int
        Total number of two-qubit gates (equals cnot count).

    total : int
        Total estimated gate count.

    state_dimension : int
        Hilbert space dimension (2^n_qubits). Included for context
        since gate counts scale with this value.

    is_estimate : bool
        Always True for amplitude encoding, indicating these are
        theoretical estimates rather than exact counts.

    Examples
    --------
    >>> enc = AmplitudeEncoding(n_features=8)
    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['total_single_qubit']
    8
    >>> breakdown['cnot']
    6
    >>> breakdown['is_estimate']
    True
    """

    rotation_gates: int
    """Estimated number of rotation gates (RY/RZ)."""

    cnot: int
    """Estimated number of CNOT gates."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates."""

    total: int
    """Total estimated gate count."""

    state_dimension: int
    """Hilbert space dimension (2^n_qubits)."""

    is_estimate: bool
    """True, indicating these are theoretical estimates."""


class AmplitudeEncoding(BaseEncoding):
    """Amplitude encoding maps classical features to quantum state amplitudes.

    AmplitudeEncoding provides exponential data compression by encoding n
    classical features into the amplitudes of a log₂(n)-qubit quantum state.
    This is one of the most powerful quantum data encodings in terms of
    compression, but comes with significant circuit depth requirements.

    The encoding prepares a quantum state where each computational basis
    state |i⟩ has amplitude proportional to the i-th feature value:

        |0⟩ → (1/‖x‖) Σᵢ xᵢ|i⟩

    The circuit structure requires state preparation gates:

        |0⟩⊗ⁿ ─── [State Preparation Unitary U] ─── |ψ(x)⟩

    where U|0⟩⊗ⁿ = |ψ(x)⟩ and the state preparation uses O(2^n) gates
    for n qubits in the general case.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
        The number of qubits is ⌈log₂(n_features)⌉, with zero-padding if
        n_features is not a power of 2.
    normalize : bool, default=True
        Whether to normalize input data to create a valid quantum state.
        If True (default), input vectors are divided by their L2 norm.
        If False, the user must ensure inputs are already normalized.

        .. warning::
            Setting normalize=False with unnormalized data may cause
            undefined behavior or errors in quantum backends.

    Attributes
    ----------
    normalize : bool
        Whether automatic normalization is enabled.
    n_features : int
        Number of classical features (inherited from BaseEncoding).
    n_qubits : int
        Number of qubits: max(1, ⌈log₂(n_features)⌉).
    depth : int
        Circuit depth: O(2^n_qubits) for general state preparation.

    Examples
    --------
    Create a basic amplitude encoding for 4 features (requires 2 qubits):

    >>> from encoding_atlas import AmplitudeEncoding
    >>> import numpy as np
    >>> enc = AmplitudeEncoding(n_features=4)
    >>> enc.n_qubits
    2
    >>> enc.n_features
    4

    Encode a feature vector (automatically normalized):

    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Single feature requires 1 qubit (padded to 2 amplitudes):

    >>> enc_single = AmplitudeEncoding(n_features=1)
    >>> enc_single.n_qubits
    1

    Generate circuits for different backends:

    >>> qiskit_circuit = enc.get_circuit(x, backend='qiskit')
    >>> qiskit_circuit.num_qubits
    2

    >>> cirq_circuit = enc.get_circuit(x, backend='cirq')
    >>> len(cirq_circuit.all_qubits())
    2

    Non-power-of-two features are zero-padded:

    >>> enc_5 = AmplitudeEncoding(n_features=5)
    >>> enc_5.n_qubits  # ceil(log2(5)) = 3, so 8 amplitudes
    3

    Batch processing multiple samples:

    >>> X = np.random.randn(10, 4)
    >>> circuits = enc.get_circuits(X, backend='pennylane')
    >>> len(circuits)
    10

    Access encoding properties:

    >>> props = enc.properties
    >>> props.is_entangling
    True
    >>> props.simulability
    'not_simulable'

    References
    ----------
    .. [1] Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
           Quantum Computers." Springer.
    .. [2] Mottonen, M., et al. (2004). "Transformation of quantum states
           using uniformly controlled rotations."

    See Also
    --------
    AngleEncoding : Linear scaling (n features = n qubits), no entanglement.
    BasisEncoding : Encodes discrete data in computational basis states.
    IQPEncoding : Adds entanglement with quadratic feature interactions.
    DataReuploading : Re-encodes data multiple times with trainable layers.

    Notes
    -----
    **Compression vs. Circuit Depth Tradeoff**:
    While amplitude encoding provides exponential compression (n features
    in log₂(n) qubits), it requires O(2^n) gates for state preparation.
    This means the encoding circuit depth scales linearly with the number
    of features, not logarithmically.

    **Qubit Calculation Edge Case**:
    For n_features=1, the formula log₂(1) = 0 would yield 0 qubits. The
    implementation enforces a minimum of 1 qubit, padding the single
    feature to a 2-element state vector [x, 0] → |0⟩ after normalization.

    **Normalization Behavior**:
    - If normalize=True: Input vector x is divided by ‖x‖ before encoding
    - If normalize=False: Input is used as-is (must be pre-normalized)
    - Zero-norm vectors will cause errors (division by zero)

    **Backend-Specific Implementation**:
    - PennyLane: Uses qml.AmplitudeEmbedding for optimal performance
    - Qiskit: Uses QuantumCircuit.initialize() with automatic decomposition
    - Cirq: Uses a custom unitary gate constructed via Gram-Schmidt

    **Qubit Ordering Conventions**:
    Quantum computing frameworks differ in how they map statevector
    indices to qubit labels.  This library adopts the **MSB (most
    significant bit)** convention used by PennyLane and Cirq, where
    **qubit 0 is the leftmost (most significant) bit**.

    - **PennyLane** — MSB natively.  ``qml.state()`` returns a
      statevector where index *i* encodes the computational basis state
      with qubit 0 as the most significant bit.  No conversion needed.

    - **Cirq** — MSB natively.  ``Simulator.simulate()`` returns a
      statevector with ``LineQubit(0)`` as the most significant bit.
      No conversion needed.

    - **Qiskit** — LSB natively.  ``Statevector(circuit).data`` returns
      amplitudes where qubit 0 is the *least* significant bit.
      ``_to_qiskit()`` compensates for this by bit-reversing the
      amplitude indices before calling ``initialize()``, and the
      analysis module's ``_reverse_qubit_order()`` reverses the
      statevector when reading results back.  The net effect is
      transparent MSB semantics for the caller.

    **Example**: For 4 features (2 qubits), amplitude vector [a₀, a₁, a₂, a₃]
    maps to quantum state: a₀|00⟩ + a₁|01⟩ + a₂|10⟩ + a₃|11⟩

    In the MSB convention used by all backends after conversion:

    - Index 0 (binary 00) → qubit 0 = 0, qubit 1 = 0
    - Index 1 (binary 01) → qubit 0 = 0, qubit 1 = 1
    - Index 2 (binary 10) → qubit 0 = 1, qubit 1 = 0
    - Index 3 (binary 11) → qubit 0 = 1, qubit 1 = 1

    This consistent ordering ensures that circuits generated by different
    backends produce equivalent quantum states, enabling reliable cross-
    backend testing and backend-agnostic algorithm development.

    **Hardware Considerations**:
    Due to the deep circuit depth, amplitude encoding may accumulate
    significant errors on NISQ hardware. Consider using shallower
    encodings (AngleEncoding, IQPEncoding) for near-term applications.
    """

    # ==========================================================================
    # Class Constants
    # ==========================================================================

    # Memory-efficient slot-based attribute storage
    __slots__ = ("normalize", "_n_qubits")

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def __init__(self, n_features: int, normalize: bool = True) -> None:
        """Initialize the amplitude encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode. Must be a positive integer.
        normalize : bool, default=True
            Whether to automatically normalize input vectors.

        Raises
        ------
        ValueError
            If n_features is not a positive integer (raised by parent class).
        TypeError
            If normalize is not a boolean.

        Examples
        --------
        >>> enc = AmplitudeEncoding(n_features=8)
        >>> enc.n_qubits
        3

        >>> enc = AmplitudeEncoding(n_features=4, normalize=False)
        >>> enc.normalize
        False
        """
        # Validate normalize parameter type
        # This catches common mistakes like passing strings ("True"), integers (1),
        # or None, which could lead to subtle bugs in conditional logic.
        if not isinstance(normalize, bool):
            raise TypeError(
                f"normalize must be a boolean (True or False), "
                f"got {type(normalize).__name__!r} with value {normalize!r}. "
                f"Use normalize=True to enable automatic normalization or "
                f"normalize=False to use pre-normalized input data."
            )

        # Call parent constructor with configuration
        super().__init__(n_features, normalize=normalize)

        # Store encoding-specific parameters
        self.normalize: bool = normalize

        # Calculate number of qubits needed: ceil(log2(n_features))
        # Special case: n_features=1 yields log2(1)=0, but we need at least
        # 1 qubit to create a valid quantum circuit. The single feature will
        # be padded to a 2-element vector [x, 0] representing state x|0⟩.
        self._n_qubits: int = max(1, int(np.ceil(np.log2(n_features))))

        # Log initialization for debugging
        _logger.debug(
            "AmplitudeEncoding initialized: n_features=%d, n_qubits=%d, "
            "normalize=%s, state_dim=%d",
            n_features,
            self._n_qubits,
            normalize,
            2**self._n_qubits,
        )

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        Calculated as max(1, ⌈log₂(n_features)⌉) to ensure at least 1 qubit
        even for single-feature inputs.

        Returns
        -------
        int
            Number of qubits. For n features, uses ⌈log₂(n)⌉ qubits.

        Examples
        --------
        >>> AmplitudeEncoding(n_features=4).n_qubits
        2
        >>> AmplitudeEncoding(n_features=5).n_qubits  # Padded to 8
        3
        >>> AmplitudeEncoding(n_features=1).n_qubits  # Minimum 1
        1
        """
        return self._n_qubits

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        For general amplitude encoding, the depth scales as O(2^n_qubits)
        because state preparation requires accessing all 2^n basis states.

        Returns
        -------
        int
            Circuit depth, equal to 2^n_qubits (the state dimension).

        Notes
        -----
        This depth is an upper bound for general state preparation. Some
        backends may optimize for specific cases (e.g., sparse amplitudes).
        """
        return 2**self._n_qubits

    # ==========================================================================
    # Circuit Generation
    # ==========================================================================

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for a single data sample.

        Creates a quantum circuit that encodes the input features as
        amplitudes of a quantum state. The input is validated, padded
        to the next power of 2 if necessary, and normalized (if enabled).

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (1, n_features).
            Values are encoded as quantum state amplitudes.
        backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
            Target quantum computing framework:

            - "pennylane": Returns a callable function that applies gates
            - "qiskit": Returns a Qiskit QuantumCircuit object
            - "cirq": Returns a Cirq Circuit object

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.

        Raises
        ------
        ValueError
            If input shape doesn't match n_features.
        ValueError
            If input contains NaN or infinite values.
        ValueError
            If input vector has zero (or near-zero) norm. A valid quantum
            state requires at least one non-zero amplitude to satisfy the
            normalization constraint |ψ|² = 1.
        ValueError
            If backend is not one of the supported options.
        ImportError
            If the requested backend is not installed.

        Examples
        --------
        >>> enc = AmplitudeEncoding(n_features=4)
        >>> x = np.array([0.5, 0.5, 0.5, 0.5])
        >>> circuit = enc.get_circuit(x, backend='pennylane')
        >>> callable(circuit)
        True

        >>> qc = enc.get_circuit(x, backend='qiskit')
        >>> qc.num_qubits
        2

        Notes
        -----
        The circuit prepares the state |ψ⟩ = Σᵢ xᵢ|i⟩ where x is the
        (optionally normalized) input vector, padded to length 2^n_qubits.
        """
        _logger.debug(
            "get_circuit called: backend=%r, input_shape=%s",
            backend,
            getattr(x, "shape", f"len={len(x)}" if hasattr(x, "__len__") else "scalar"),
        )

        # Validate and preprocess input
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]

        # Pad to power of 2 (state dimension must be 2^n_qubits)
        target_size = 2**self._n_qubits
        original_len = len(x_validated)
        if len(x_validated) < target_size:
            x_validated = np.pad(x_validated, (0, target_size - len(x_validated)))
            _logger.debug(
                "Input padded: original_len=%d, padded_len=%d, padding_zeros=%d",
                original_len,
                target_size,
                target_size - original_len,
            )

        # Compute the L2 norm for normalization and validation
        norm = np.linalg.norm(x_validated)

        # Validate: Zero-norm vectors cannot form valid quantum states.
        # A quantum state |ψ⟩ must satisfy the normalization constraint ⟨ψ|ψ⟩ = 1,
        # which is impossible for a zero vector. This check applies regardless of
        # the normalize setting, since even pre-normalized data cannot be all zeros.
        if norm < _ZERO_NORM_THRESHOLD:
            raise ValueError(
                "Input vector has zero (or near-zero) norm and cannot be encoded "
                "as a valid quantum state. Quantum states must satisfy the "
                "normalization constraint |ψ|² = 1, which requires at least one "
                "non-zero amplitude. Please provide input data with at least one "
                "non-zero feature value."
            )

        # Normalize to create valid quantum state (|ψ|² = 1)
        if self.normalize:
            x_validated = x_validated / norm
            _logger.debug(
                "Input normalized: original_norm=%.10f, normalized_norm=1.0", norm
            )
        else:
            # When normalize=False, user claims data is pre-normalized.
            # We validate that the norm is reasonably close to 1.0 to catch
            # obvious errors (like forgetting to normalize), while allowing
            # small numerical deviations that occur in practice.
            #
            # Tolerance: 1% deviation from unit norm (rtol=0.01).
            # This catches errors like passing raw data [1, 2, 3, 4] (norm=5.48)
            # while allowing legitimate numerical imprecision like norm=1.00005.
            if not np.isclose(norm, 1.0, rtol=0.01, atol=1e-10):
                raise ValueError(
                    f"normalize=False but input vector has L2 norm {norm:.6f}, "
                    f"which differs from 1.0 by more than 1%. When normalization "
                    f"is disabled, input data must be pre-normalized to satisfy "
                    f"the quantum state constraint |ψ|² = 1.\n\n"
                    f"Options:\n"
                    f"  1. Set normalize=True to enable automatic normalization\n"
                    f"  2. Pre-normalize your data: x = x / np.linalg.norm(x)"
                )

        # Dispatch to backend-specific implementation
        _logger.debug(
            "Dispatching to backend: %r, state_vector_len=%d, n_qubits=%d",
            backend,
            len(x_validated),
            self._n_qubits,
        )

        if backend == "pennylane":
            circuit = self._to_pennylane(x_validated)
            _logger.info("PennyLane circuit generated: n_qubits=%d", self._n_qubits)
            return circuit
        elif backend == "qiskit":
            circuit = self._to_qiskit(x_validated)
            _logger.info("Qiskit circuit generated: n_qubits=%d", self._n_qubits)
            return circuit
        elif backend == "cirq":
            circuit = self._to_cirq(x_validated)
            _logger.info("Cirq circuit generated: n_qubits=%d", self._n_qubits)
            return circuit
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
            )

    def get_circuits(
        self,
        X: ArrayLike,
        backend: BackendType = "pennylane",
        *,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> list[CircuitType]:
        """Generate quantum circuits for multiple data samples.

        Parameters
        ----------
        X : array-like
            Input features of shape (n_samples, n_features) or (n_features,).
            If 1D, treated as a single sample.
        backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
            Target quantum computing framework.
        parallel : bool, default=False
            If True, use parallel processing via ThreadPoolExecutor for
            circuit generation. This can significantly speed up processing
            for large batches, especially with the Cirq backend which
            constructs full unitary matrices.

            Parallel processing is thread-safe because AmplitudeEncoding's
            circuit generation is stateless (no instance mutation occurs).
        max_workers : int or None, default=None
            Maximum number of worker threads for parallel processing.
            Only used when ``parallel=True``. If None, uses the default
            from ThreadPoolExecutor (typically min(32, cpu_count + 4)).

            For CPU-bound workloads (like Cirq's unitary construction),
            set to ``os.cpu_count()``. For I/O-bound or memory-constrained
            scenarios, lower values may be preferable.

        Returns
        -------
        list[CircuitType]
            List of circuits, one per sample. Order is preserved even
            when using parallel processing.

        Examples
        --------
        Sequential processing (default):

        >>> enc = AmplitudeEncoding(n_features=4)
        >>> X = np.random.randn(10, 4)
        >>> circuits = enc.get_circuits(X, backend='pennylane')
        >>> len(circuits)
        10

        Parallel processing for large batches:

        >>> enc = AmplitudeEncoding(n_features=8)
        >>> X_large = np.random.randn(1000, 8)
        >>> circuits = enc.get_circuits(X_large, backend='cirq', parallel=True)
        >>> len(circuits)
        1000

        Custom worker count:

        >>> import os
        >>> circuits = enc.get_circuits(
        ...     X_large, backend='qiskit', parallel=True, max_workers=os.cpu_count()
        ... )

        Notes
        -----
        **When to use parallel processing:**

        - Large batches (>100 samples): Parallel processing overhead is
          amortized across many samples.
        - Cirq backend: Unitary matrix construction is computationally
          expensive and benefits significantly from parallelization.
        - Qiskit backend: Circuit object creation has moderate overhead.

        **When to use sequential processing:**

        - Small batches (<100 samples): Overhead of thread pool management
          may exceed the benefit.
        - PennyLane backend: Circuit generation creates lightweight closures
          and is already very fast.
        - Memory-constrained environments: Parallel Cirq circuit generation
          may consume significant memory (each worker builds a full unitary).
        - Already in a parallel context: Avoid nested parallelism.

        **Thread Safety:**

        This method is thread-safe. The encoding object is not modified
        during circuit generation, and each circuit is generated
        independently. Input validation creates defensive copies to
        prevent data races.

        **Order Preservation:**

        When ``parallel=True``, the returned list maintains the same order
        as the input samples. This is achieved using ThreadPoolExecutor.map()
        which preserves ordering.

        **Memory Considerations for Cirq:**

        When using ``parallel=True`` with the Cirq backend, each worker
        thread constructs a full 2^n × 2^n unitary matrix. For n_qubits >= 10,
        this can consume substantial memory per worker. Consider limiting
        ``max_workers`` for large qubit counts to avoid memory exhaustion.

        See Also
        --------
        get_circuit : Generate a single circuit (useful for individual samples).
        resource_summary : Analyze resource requirements for this encoding.
        """
        X_validated = self._validate_input(X)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)

        n_samples = X_validated.shape[0]

        # Log batch processing start
        _logger.debug(
            "Batch processing started: n_samples=%d, backend=%r, parallel=%s, "
            "max_workers=%s",
            n_samples,
            backend,
            parallel,
            max_workers,
        )

        if parallel and n_samples > 1:
            # Parallel processing using ThreadPoolExecutor
            # ThreadPoolExecutor.map() preserves order of results
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a helper function that captures the backend parameter
                # Uses internal method to avoid re-validating each sample
                def generate_single(x: NDArray[np.floating[Any]]) -> CircuitType:
                    return self._get_circuit_from_validated(x, backend)

                # Map preserves order: result[i] corresponds to X_validated[i]
                circuits = list(executor.map(generate_single, X_validated))

            _logger.debug(
                "Parallel batch processing completed: generated %d circuits "
                "using ThreadPoolExecutor",
                len(circuits),
            )
        else:
            # Sequential processing (default behavior)
            # Use internal method to avoid re-validating each sample.
            # The batch was already validated above, so we can safely skip
            # per-sample validation for better performance.
            circuits = [
                self._get_circuit_from_validated(x, backend) for x in X_validated
            ]

            _logger.debug(
                "Sequential batch processing completed: generated %d circuits",
                len(circuits),
            )

        _logger.info(
            "Batch circuit generation complete: n_circuits=%d, backend=%r",
            len(circuits),
            backend,
        )
        return circuits

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input (internal use only).

        This method skips input validation, assuming the caller has already
        validated the input. Used by get_circuits() to avoid double validation.

        Parameters
        ----------
        x : NDArray
            Pre-validated input features of shape (n_features,).
        backend : BackendType
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.
        """
        # Pad to power of 2 (state dimension must be 2^n_qubits)
        target_size = 2**self._n_qubits
        if len(x) < target_size:
            x = np.pad(x, (0, target_size - len(x)))

        # Compute the L2 norm for normalization and validation
        norm = np.linalg.norm(x)

        # Validate: Zero-norm vectors cannot form valid quantum states
        if norm < _ZERO_NORM_THRESHOLD:
            raise ValueError(
                "Input vector has zero (or near-zero) norm and cannot be encoded "
                "as a valid quantum state."
            )

        # Normalize to create valid quantum state (|ψ|² = 1)
        if self.normalize:
            x = x / norm
        else:
            # Validate pre-normalized data (1% tolerance for practical use)
            if not np.isclose(norm, 1.0, rtol=0.01, atol=1e-10):
                raise ValueError(
                    f"normalize=False but input vector has L2 norm {norm:.6f}. "
                    f"Input must be pre-normalized when normalization is disabled."
                )

        # Dispatch to backend-specific implementation
        if backend == "pennylane":
            return self._to_pennylane(x)
        elif backend == "qiskit":
            return self._to_qiskit(x)
        elif backend == "cirq":
            return self._to_cirq(x)
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
            )

    # ==========================================================================
    # Backend-Specific Implementations
    # ==========================================================================

    def _to_pennylane(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate PennyLane circuit function.

        Uses PennyLane's built-in AmplitudeEmbedding for optimal performance
        and compatibility with automatic differentiation.

        Parameters
        ----------
        x : NDArray
            Preprocessed input features (validated, padded, normalized).

        Returns
        -------
        callable
            Function that applies the encoding gates when called within
            a PennyLane QNode context.

        Raises
        ------
        ImportError
            If PennyLane is not installed.

        Notes
        -----
        **Qubit Ordering**: PennyLane uses **MSB (most significant bit)**
        ordering, where wire 0 is the *leftmost* (most significant) bit.
        Amplitude index *i* maps to the computational basis state whose
        binary representation has wire 0 as the MSB.

        Example for 2 qubits:

        - amplitudes[0] → |00⟩ (wire 0 = 0, wire 1 = 0)
        - amplitudes[1] → |01⟩ (wire 0 = 0, wire 1 = 1)
        - amplitudes[2] → |10⟩ (wire 0 = 1, wire 1 = 0)
        - amplitudes[3] → |11⟩ (wire 0 = 1, wire 1 = 1)
        """
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(
                "PennyLane is required for the 'pennylane' backend. "
                "Install it with: pip install pennylane"
            ) from e

        # Capture values for closure (avoid mutable state issues)
        n_qubits = self.n_qubits
        amplitudes = x.copy()

        def circuit() -> None:
            """Apply amplitude embedding gates."""
            # normalize=False because we already normalized in get_circuit
            qml.AmplitudeEmbedding(amplitudes, wires=range(n_qubits), normalize=False)

        return circuit

    def _to_qiskit(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Qiskit QuantumCircuit.

        Uses Qiskit's initialize() method which automatically decomposes
        the state preparation into elementary gates.

        Parameters
        ----------
        x : NDArray
            Preprocessed input features (validated, padded, normalized).

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit implementing the encoding.

        Raises
        ------
        ImportError
            If Qiskit is not installed.

        Notes
        -----
        **Qubit Ordering Conversion (MSB → LSB)**:

        This library and PennyLane use **MSB (most significant bit)**
        ordering, where qubit 0 is the leftmost bit.  Qiskit uses **LSB
        (least significant bit)** ordering, where qubit 0 is the rightmost
        bit.  These conventions cause the *same* amplitude index to refer
        to *different* physical basis states:

        =========  ==================  =================
        Index      MSB meaning          LSB meaning
        =========  ==================  =================
        0 (00)     q0=0, q1=0          q0=0, q1=0
        1 (01)     q0=0, q1=1          q0=1, q1=0
        2 (10)     q0=1, q1=0          q0=0, q1=1
        3 (11)     q0=1, q1=1          q0=1, q1=1
        =========  ==================  =================

        To produce the same physical state as PennyLane, the amplitude
        vector must be permuted from MSB index order to LSB index order
        before passing to ``initialize()``.  The permutation is a
        bit-reversal of the indices, implemented efficiently via a tensor
        transpose.

        After this conversion, the analysis module's
        ``_reverse_qubit_order()`` (applied when reading back the Qiskit
        statevector) correctly recovers the MSB-ordered result, matching
        PennyLane and Cirq.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError(
                "Qiskit is required for the 'qiskit' backend. "
                "Install it with: pip install qiskit"
            ) from e

        qc = QuantumCircuit(self.n_qubits, name="AmplitudeEncoding")

        # Input 'x' arrives pre-processed from get_circuit():
        # - Validated (no NaN/Inf, correct shape)
        # - Padded to power of 2
        # - Normalized to unit L2 norm (if self.normalize=True)
        # - Or user-provided normalized data (if self.normalize=False)
        #
        # Convert amplitude ordering from this library's MSB convention
        # to Qiskit's LSB convention.  The conversion reshapes the 1-D
        # amplitude vector into an n-qubit tensor (shape [2]*n_qubits),
        # reverses the axis order (which bit-reverses the multi-index),
        # and flattens back to 1-D.  This is equivalent to permuting
        # amplitudes[i] → amplitudes[bit_reverse(i)] but avoids an
        # explicit per-index loop.
        n_qubits = self.n_qubits
        amplitudes_lsb = (
            x.reshape([2] * n_qubits)
            .transpose(list(reversed(range(n_qubits))))
            .flatten()
            .copy()  # Ensure contiguous memory for Qiskit
        )

        qc.initialize(amplitudes_lsb, range(n_qubits))

        return qc

    def _to_cirq(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Cirq Circuit.

        Implements state preparation using a custom unitary gate that maps
        |0...0⟩ to the target state. The unitary is constructed by placing
        the target state as the first column and completing to an orthonormal
        basis using Gram-Schmidt orthogonalization.

        Parameters
        ----------
        x : NDArray
            Preprocessed input features (validated, padded, normalized).

        Returns
        -------
        cirq.Circuit
            Cirq circuit implementing the encoding.

        Raises
        ------
        ImportError
            If Cirq is not installed.

        Warns
        -----
        UserWarning
            When n_qubits >= 12, a warning is issued about potentially high
            memory usage. The Cirq backend constructs a full 2^n × 2^n unitary
            matrix which can consume significant memory for large n.

        Notes
        -----
        **Qubit Ordering**: Cirq uses **MSB (most significant bit)**
        ordering, where ``LineQubit(0)`` is the *leftmost* (most
        significant) bit.  ``Simulator.simulate().final_state_vector``
        returns amplitudes with this convention, matching PennyLane.

        Example for 2 qubits:

        - x[0] → |00⟩ (LineQubit(0) = 0, LineQubit(1) = 0)
        - x[1] → |01⟩ (LineQubit(0) = 0, LineQubit(1) = 1)
        - x[2] → |10⟩ (LineQubit(0) = 1, LineQubit(1) = 0)
        - x[3] → |11⟩ (LineQubit(0) = 1, LineQubit(1) = 1)

        No index permutation is needed because Cirq and PennyLane share
        the same MSB convention.

        **Memory Usage**: This implementation creates a custom gate with the
        full unitary matrix. For large numbers of qubits, this may consume
        significant memory (O(4^n) complex numbers for n qubits).

        The Gram-Schmidt process is numerically stable for typical use cases.
        For near-linearly-dependent states, numerical precision issues may
        arise.

        Memory usage examples for the unitary matrix:

        - 10 qubits: 16 MB
        - 12 qubits: 256 MB
        - 14 qubits: 4 GB
        - 16 qubits: 64 GB

        For memory-constrained environments, consider using the 'pennylane'
        or 'qiskit' backends which use more efficient decomposition strategies.
        """
        try:
            import cirq
        except ImportError as e:
            raise ImportError(
                "Cirq is required for the 'cirq' backend. "
                "Install it with: pip install cirq-core"
            ) from e

        # Emit memory warning for large state vectors.
        #
        # The Cirq backend constructs a full unitary matrix of size 2^n × 2^n,
        # which requires 4^n × 16 bytes of memory (complex128 per element).
        # For n_qubits >= 12, this exceeds 256 MB and may cause memory pressure
        # or allocation failures on constrained systems.
        #
        # We issue a warning rather than raising an error because:
        # 1. Some users may have sufficient memory and want to proceed
        # 2. The threshold is conservative; 256 MB is manageable on most systems
        # 3. It allows users to make an informed decision about backend choice
        #
        # The warning is emitted every time (no filtering) to ensure visibility,
        # especially in interactive/notebook environments where warnings may be
        # suppressed by default settings.
        if self.n_qubits >= _CIRQ_MEMORY_WARNING_THRESHOLD_QUBITS:
            # Calculate memory usage: 2^n × 2^n × 16 bytes = 4^n × 16 bytes
            state_dim = 2**self.n_qubits
            matrix_elements = state_dim * state_dim
            memory_bytes = matrix_elements * 16  # complex128 = 16 bytes
            memory_str = _format_memory_size(memory_bytes)

            warnings.warn(
                f"Cirq backend memory warning: Constructing a {state_dim}×{state_dim} "
                f"unitary matrix for {self.n_qubits}-qubit amplitude encoding requires "
                f"approximately {memory_str} of memory. This may cause performance issues "
                f"or memory allocation failures on constrained systems.\n\n"
                f"Recommendations:\n"
                f"  - For memory-efficient encoding, use backend='pennylane' or backend='qiskit'\n"
                f"  - Reduce n_features to decrease the number of qubits required\n"
                f"  - Ensure sufficient system memory is available before proceeding",
                UserWarning,
                stacklevel=3,  # Points to user's get_circuit() call
            )

        # Create qubits
        qubits = cirq.LineQubit.range(self.n_qubits)

        # Input 'x' arrives pre-processed from get_circuit():
        # - Validated (no NaN/Inf, correct shape)
        # - Padded to power of 2 (len(x) == 2^n_qubits)
        # - Normalized to unit L2 norm
        #
        # Convert to complex for unitary construction.
        # No re-padding or re-normalization needed.
        target_state = x.astype(complex)

        # Create custom state preparation gate
        amp_gate = _AmplitudePreparationGate(target_state)

        # Build circuit with the custom gate
        circuit = cirq.Circuit()
        circuit.append(amp_gate.on(*qubits))

        return circuit

    # ==========================================================================
    # Resource Analysis
    # ==========================================================================

    def resource_summary(self) -> dict[str, Any]:
        """Generate a comprehensive resource summary for this encoding.

        Provides a detailed breakdown of circuit resources including qubit
        requirements, state dimension, compression ratio, memory estimates,
        and encoding characteristics. This is particularly useful for:

        - Hardware resource planning and capacity estimation
        - Understanding the exponential compression trade-offs
        - Comparing amplitude encoding with other encoding methods
        - Memory budget planning for Cirq backend

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``n_features``: Number of input classical features
            - ``n_qubits``: Number of qubits required (⌈log₂(n_features)⌉)
            - ``state_dimension``: Hilbert space dimension (2^n_qubits)
            - ``compression_ratio``: Features per qubit (n_features / n_qubits)
            - ``padding_zeros``: Number of zero-padding elements added
            - ``depth``: Circuit depth for state preparation
            - ``normalize``: Whether automatic normalization is enabled
            - ``theoretical_gate_count``: Estimated total gates (2n - 2)
            - ``theoretical_single_qubit_gates``: Estimated single-qubit gates
            - ``theoretical_two_qubit_gates``: Estimated two-qubit gates
            - ``is_entangling``: Always True for amplitude encoding
            - ``simulability``: Always "not_simulable" (exponential classical cost)
            - ``cirq_unitary_memory_bytes``: Memory for Cirq's unitary matrix
            - ``cirq_unitary_memory_human``: Human-readable memory string
            - ``backend_notes``: Dict with backend-specific information

        Examples
        --------
        Get complete resource analysis:

        >>> enc = AmplitudeEncoding(n_features=8)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        3
        >>> summary['compression_ratio']
        2.6666666666666665
        >>> summary['state_dimension']
        8

        Understand memory requirements for Cirq:

        >>> enc = AmplitudeEncoding(n_features=1024)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        10
        >>> summary['cirq_unitary_memory_human']
        '16.0 MB'

        Compare compression for different feature counts:

        >>> for n in [4, 8, 16, 32, 64]:
        ...     enc = AmplitudeEncoding(n_features=n)
        ...     summary = enc.resource_summary()
        ...     print(f"n={n}: {summary['n_qubits']} qubits, "
        ...           f"compression={summary['compression_ratio']:.1f}x")
        n=4: 2 qubits, compression=2.0x
        n=8: 3 qubits, compression=2.7x
        n=16: 4 qubits, compression=4.0x
        n=32: 5 qubits, compression=6.4x
        n=64: 6 qubits, compression=10.7x

        Use for capacity planning:

        >>> enc = AmplitudeEncoding(n_features=16)
        >>> summary = enc.resource_summary()
        >>> print(f"Encoding {summary['n_features']} features requires "
        ...       f"{summary['n_qubits']} qubits with depth {summary['depth']}")
        Encoding 16 features requires 4 qubits with depth 16

        Notes
        -----
        **Theoretical vs. Actual Gate Counts:**

        The gate counts reported here are theoretical estimates based on
        general state preparation algorithms. Actual gate counts depend on:

        - **PennyLane**: Uses optimized ``AmplitudeEmbedding`` decomposition
        - **Qiskit**: Uses ``initialize()`` with automatic optimization
        - **Cirq**: Uses a single custom unitary gate (full matrix approach)

        For precise gate counts after compilation, use the respective
        backend's transpiler or decomposition tools.

        **Memory Estimation:**

        The ``cirq_unitary_memory_bytes`` field estimates memory for Cirq's
        unitary matrix construction (2^n × 2^n complex128 matrix). This is
        relevant when using the Cirq backend, especially with parallel
        processing where multiple workers each construct their own matrix.

        **Compression Ratio:**

        The compression ratio shows how many classical features are encoded
        per qubit. Higher ratios indicate better compression but come with
        exponentially increasing circuit depth. This is the fundamental
        trade-off of amplitude encoding.

        See Also
        --------
        properties : Access the EncodingProperties dataclass.
        get_circuit : Generate a circuit for specific input data.
        """
        n = self.n_features
        n_qubits = self._n_qubits
        state_dim = 2**n_qubits

        # Theoretical gate counts (estimates for general state preparation)
        # Based on Möttönen et al. decomposition: O(2^n) gates
        theoretical_total = 2 * state_dim - 2
        theoretical_single = state_dim
        theoretical_two = state_dim - 2

        # Memory estimation for Cirq backend
        # Unitary matrix: 2^n × 2^n complex128 (16 bytes per element)
        cirq_memory_bytes = state_dim * state_dim * 16

        summary: dict[str, Any] = {
            # Basic structure
            "n_features": n,
            "n_qubits": n_qubits,
            "state_dimension": state_dim,
            "compression_ratio": n / n_qubits if n_qubits > 0 else 0.0,
            "padding_zeros": state_dim - n,
            # Circuit metrics
            "depth": self.depth,
            "normalize": self.normalize,
            # Theoretical gate counts
            "theoretical_gate_count": theoretical_total,
            "theoretical_single_qubit_gates": theoretical_single,
            "theoretical_two_qubit_gates": theoretical_two,
            # Encoding characteristics
            "is_entangling": True,
            "simulability": "not_simulable",
            # Memory estimates
            "cirq_unitary_memory_bytes": cirq_memory_bytes,
            "cirq_unitary_memory_human": _format_memory_size(cirq_memory_bytes),
            # Backend-specific notes
            "backend_notes": {
                "pennylane": "Uses qml.AmplitudeEmbedding with optimized decomposition",
                "qiskit": "Uses QuantumCircuit.initialize() with automatic synthesis",
                "cirq": (
                    f"Constructs {state_dim}×{state_dim} unitary matrix "
                    f"(~{_format_memory_size(cirq_memory_bytes)} memory)"
                ),
            },
        }

        _logger.debug(
            "Resource summary: n_features=%d, n_qubits=%d, state_dim=%d, "
            "compression=%.2fx, depth=%d, cirq_memory=%s",
            n,
            n_qubits,
            state_dim,
            summary["compression_ratio"],
            self.depth,
            summary["cirq_unitary_memory_human"],
        )

        return summary

    def transform_input(self, x: ArrayLike) -> NDArray[np.floating[Any]]:
        """Transform input data according to the encoding's preprocessing rules.

        This method implements the ``DataTransformable`` protocol, exposing
        the encoding's internal data transformation logic. For AmplitudeEncoding,
        this applies normalization to create a valid quantum state (unit L2 norm),
        and optionally pads the data to the next power of 2.

        Parameters
        ----------
        x : ArrayLike
            Input features of shape ``(n_features,)`` or ``(1, n_features)``.
            Can be any numeric type that can be converted to float.

        Returns
        -------
        NDArray[np.floating]
            Transformed data with the following properties:

            - **L2 norm of 1.0** (if ``normalize=True``, the default)
            - **Length equal to state dimension** (2^n_qubits, zero-padded if needed)

            If ``normalize=False``, returns the padded but unnormalized input.

        Raises
        ------
        ValueError
            If input shape doesn't match ``n_features``.
        ValueError
            If input contains NaN or infinite values.
        ValueError
            If input vector has zero (or near-zero) norm and cannot be
            normalized to a valid quantum state.

        Examples
        --------
        Transform and normalize continuous data:

        >>> enc = AmplitudeEncoding(n_features=4)
        >>> x = np.array([3.0, 4.0, 0.0, 0.0])
        >>> result = enc.transform_input(x)
        >>> result
        array([0.6, 0.8, 0. , 0. ])
        >>> np.linalg.norm(result)  # Unit norm
        1.0

        Non-power-of-2 features are zero-padded:

        >>> enc = AmplitudeEncoding(n_features=3)
        >>> x = np.array([1.0, 0.0, 0.0])
        >>> result = enc.transform_input(x)
        >>> len(result)  # Padded to 4 (2^2)
        4
        >>> result
        array([1., 0., 0., 0.])

        Using the protocol interface:

        >>> from encoding_atlas.core.protocols import DataTransformable
        >>> enc = AmplitudeEncoding(n_features=4)
        >>> if isinstance(enc, DataTransformable):
        ...     normalized = enc.transform_input([1.0, 2.0, 2.0, 0.0])
        ...     print(f"Normalized: {normalized}")
        ...     print(f"Norm: {np.linalg.norm(normalized):.6f}")
        Normalized: [0.33333333 0.66666667 0.66666667 0.        ]
        Norm: 1.000000

        Write generic code that works with any DataTransformable encoding:

        >>> def analyze_transformation(enc, x):
        ...     '''Analyze how an encoding transforms input data.'''
        ...     from encoding_atlas.core.protocols import DataTransformable
        ...     if isinstance(enc, DataTransformable):
        ...         original_norm = np.linalg.norm(x)
        ...         transformed = enc.transform_input(x)
        ...         new_norm = np.linalg.norm(transformed)
        ...         print(f"{enc.__class__.__name__}:")
        ...         print(f"  Original norm: {original_norm:.4f}")
        ...         print(f"  Transformed norm: {new_norm:.4f}")

        Notes
        -----
        **Relationship to get_circuit()**:

        This method performs the same preprocessing as ``get_circuit()``:

        1. Validates input shape and values
        2. Pads to power-of-2 length (state dimension)
        3. Normalizes to unit L2 norm (if ``normalize=True``)

        The difference is that ``transform_input()`` returns the transformed
        data for inspection, while ``get_circuit()`` uses it to build a circuit.

        **Normalization Behavior**:

        - If ``normalize=True`` (default): Input is divided by its L2 norm.
          This ensures the transformed data represents a valid quantum state.
        - If ``normalize=False``: Input is only padded, not normalized.
          User must ensure input is already normalized.

        **Zero-Vector Handling**:

        A zero vector cannot be normalized (division by zero) and cannot
        represent a valid quantum state. This method raises ``ValueError``
        for inputs with near-zero norm (< 1e-15).

        **Idempotency**:

        For already-normalized data of the correct length, this method
        is approximately idempotent (numerical precision may cause tiny
        differences due to floating-point arithmetic).

        See Also
        --------
        resource_summary : Comprehensive resource analysis for this encoding.
        DataTransformable : The protocol this method implements.
        get_circuit : Generate quantum circuit (uses same transformation).
        """
        # Validate input shape and values
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            if x_validated.shape[0] != 1:
                raise ValueError(
                    f"transform_input expects a single sample, got shape "
                    f"{x_validated.shape}. For batch processing, iterate over "
                    f"samples: [enc.transform_input(xi) for xi in X]"
                )
            x_validated = x_validated[0]

        # Pad to power of 2 (state dimension must be 2^n_qubits)
        target_size = 2**self._n_qubits
        if len(x_validated) < target_size:
            x_validated = np.pad(x_validated, (0, target_size - len(x_validated)))

        # Compute the L2 norm for validation and normalization
        norm = np.linalg.norm(x_validated)

        # Validate: Zero-norm vectors cannot form valid quantum states
        if norm < _ZERO_NORM_THRESHOLD:
            raise ValueError(
                "Input vector has zero (or near-zero) norm and cannot be "
                "transformed for amplitude encoding. Quantum states must satisfy "
                "the normalization constraint |ψ|² = 1, which requires at least "
                "one non-zero amplitude. Please provide input data with at least "
                "one non-zero feature value."
            )

        # Normalize to create valid quantum state (|ψ|² = 1) if enabled
        if self.normalize:
            x_validated = x_validated / norm

        _logger.debug(
            "transform_input: original_norm=%.10f, normalized=%s, output_len=%d",
            norm,
            self.normalize,
            len(x_validated),
        )

        return x_validated

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of estimated gate counts by type.

        Computes theoretical gate count estimates for amplitude encoding based
        on the Möttönen et al. state preparation algorithm. These are **estimates**
        because amplitude encoding creates data-dependent circuits whose exact
        structure varies with input amplitudes.

        This method is useful for:

        - Resource estimation before circuit execution
        - Hardware capacity planning
        - Comparing encoding methods
        - Understanding scaling behavior

        Returns
        -------
        GateCountBreakdown
            TypedDict with estimated gate counts:

            - ``'rotation_gates'``: Estimated single-qubit rotations (~2^n)
            - ``'cnot'``: Estimated CNOT gates (~2^n - 2)
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Total two-qubit gates
            - ``'total'``: Total estimated gate count
            - ``'state_dimension'``: Hilbert space dimension for context
            - ``'is_estimate'``: Always True (theoretical estimates)

        Examples
        --------
        >>> enc = AmplitudeEncoding(n_features=8)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['rotation_gates']
        8
        >>> breakdown['cnot']
        6
        >>> breakdown['total']
        14
        >>> breakdown['is_estimate']
        True

        Gate counts scale exponentially with qubits:

        >>> enc_small = AmplitudeEncoding(n_features=4)
        >>> enc_large = AmplitudeEncoding(n_features=16)
        >>> enc_small.gate_count_breakdown()['total']
        6
        >>> enc_large.gate_count_breakdown()['total']
        30

        Notes
        -----
        **Why estimates?**

        Unlike encodings with fixed circuit structure (AngleEncoding, IQPEncoding),
        amplitude encoding prepares arbitrary quantum states. The Möttönen et al.
        algorithm [1] provides a general decomposition with O(2^n) gates, but:

        1. Specific input values may allow simpler circuits
        2. Different backends use different synthesis algorithms
        3. Optimization passes may reduce gate counts

        **Scaling behavior:**

        For n qubits (encoding up to 2^n features):

        - Single-qubit gates: ~2^n (one rotation per amplitude)
        - Two-qubit gates: ~2^n - 2 (for entanglement creation)
        - Total: ~2^(n+1) - 2

        **Backend-specific notes:**

        - PennyLane: Uses qml.AmplitudeEmbedding with optimized decomposition
        - Qiskit: Uses QuantumCircuit.initialize() with automatic synthesis
        - Cirq: Constructs full unitary matrix (different approach)

        See Also
        --------
        resource_summary : Comprehensive resource analysis including memory estimates.
        properties : Access the EncodingProperties dataclass.

        References
        ----------
        .. [1] Möttönen, M., et al. (2004). "Transformation of quantum states using
               uniformly controlled rotations." Quantum Information & Computation.
        """
        state_dim = 2**self._n_qubits

        # Theoretical estimates based on Möttönen et al. decomposition
        # General state preparation requires O(2^n) gates
        rotation_gates = state_dim
        cnot_gates = max(0, state_dim - 2)  # Ensure non-negative
        total = rotation_gates + cnot_gates

        _logger.debug(
            "Gate count breakdown (estimate): rotations=%d, cnot=%d, total=%d, "
            "state_dim=%d",
            rotation_gates,
            cnot_gates,
            total,
            state_dim,
        )

        return GateCountBreakdown(
            rotation_gates=rotation_gates,
            cnot=cnot_gates,
            total_single_qubit=rotation_gates,
            total_two_qubit=cnot_gates,
            total=total,
            state_dimension=state_dim,
            is_estimate=True,
        )

    # ==========================================================================
    # Properties Computation
    # ==========================================================================

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Calculates circuit metrics including qubit count, depth, gate counts,
        and encoding characteristics for theoretical analysis.

        Returns
        -------
        EncodingProperties
            Computed properties including:
            - n_qubits: Number of qubits (⌈log₂(n_features)⌉)
            - depth: Circuit depth (2^n_qubits for general state prep)
            - gate_count: Approximate total gates
            - is_entangling: True (general states require entanglement)
            - simulability: "not_simulable" (exponential classical cost)

        Notes
        -----
        Gate counts are estimates based on theoretical state preparation
        complexity. Actual counts may vary by backend and optimization level.
        """
        n = 2**self._n_qubits  # State dimension

        return EncodingProperties(
            n_qubits=self._n_qubits,
            depth=n,  # O(n) depth for state preparation
            gate_count=2 * n - 2,  # Approximate for general state preparation
            single_qubit_gates=n,  # Approximate single-qubit rotations
            two_qubit_gates=n - 2,  # Approximate CNOT count
            parameter_count=n,  # Each amplitude is a parameter
            is_entangling=True,  # General states require entanglement
            simulability="not_simulable",  # Exponential classical simulation cost
            trainability_estimate=0.5,  # Deep circuits have trainability issues
            notes=(
                f"Exponential compression: {self.n_features} features in "
                f"{self._n_qubits} qubits. "
                f"Circuit depth O(2^n) limits NISQ applicability."
            ),
        )

    # ==========================================================================
    # String Representation
    # ==========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation.

        Returns
        -------
        str
            String representation showing all configuration parameters.
        """
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, "
            f"normalize={self.normalize!r})"
        )


# =============================================================================
# Helper Classes
# =============================================================================


class _AmplitudePreparationGate:
    """Custom Cirq gate for amplitude encoding state preparation.

    This gate prepares an arbitrary quantum state by constructing a unitary
    matrix that maps |0...0⟩ to the target state. The unitary is built using
    QR decomposition (Householder reflections via LAPACK):

    1. Construct matrix A = [target_state | I_{n-1}]
    2. Compute QR decomposition: A = QR
    3. Apply phase correction to ensure U[:, 0] = target_state exactly

    This approach is numerically stable (backward stable) and highly optimized,
    using BLAS Level 3 operations. It replaces the previous Gram-Schmidt
    implementation which was numerically unstable for large matrices.

    This is an internal helper class and should not be used directly.

    Parameters
    ----------
    state_vector : np.ndarray
        Target state vector (must be normalized, length must be power of 2).

    Raises
    ------
    ValueError
        If state vector length is not a power of 2.

    Notes
    -----
    Memory usage: O(4^n) for n qubits (full unitary matrix storage).

    Time complexity: O(n³) where n = 2^(n_qubits), implemented via LAPACK's
    zgeqrf routine which is highly optimized (cache-efficient, vectorized,
    and multi-threaded when linked with OpenBLAS or Intel MKL).

    Numerical stability: The QR decomposition via Householder reflections is
    backward stable, guaranteeing orthogonality to machine precision (~1e-15)
    regardless of matrix size. This is critical for quantum computing where
    non-unitary gates would violate probability conservation.

    References
    ----------
    .. [1] Golub, G. H., & Van Loan, C. F. (2013). "Matrix Computations"
           (4th ed.). Johns Hopkins University Press.
    """

    # Memory-efficient slot-based attribute storage
    __slots__ = ("_state_vector", "_num_qubits", "_unitary_matrix")

    def __init__(self, state_vector: np.ndarray) -> None:
        """Initialize the amplitude preparation gate.

        Parameters
        ----------
        state_vector : np.ndarray
            Normalized target state vector. Must have length equal to a power of 2.

        Raises
        ------
        ValueError
            If state vector length is not a power of 2.
        """
        n = len(state_vector)

        # Validate that length is a power of 2.
        # For quantum state preparation, the state vector dimension must be 2^k
        # for some integer k (the number of qubits).
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError(
                f"State vector length must be a power of 2, got {n}. "
                f"This is required for valid quantum state preparation."
            )

        # Defensive copy to prevent external modifications from corrupting
        # the gate's internal state. This ensures thread safety and prevents
        # subtle bugs if the caller modifies the array after passing it.
        self._state_vector: np.ndarray = state_vector.copy()
        self._num_qubits: int = int(np.log2(n))

        # Pre-compute and cache the unitary matrix (lazy initialization)
        self._unitary_matrix: np.ndarray | None = None

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits this gate acts on.

        Returns
        -------
        int
            Number of qubits.
        """
        return self._num_qubits

    def _unitary_(self) -> np.ndarray:
        """Return the unitary matrix that prepares the target state.

        Constructs a unitary U such that U|0...0⟩ = |ψ_target⟩. The first
        column of U is the target state vector, and remaining columns form
        an orthonormal completion.

        Algorithm
        ---------
        This implementation uses QR decomposition via Householder reflections
        (numpy.linalg.qr, backed by LAPACK's zgeqrf for complex matrices).

        **Why QR instead of Gram-Schmidt?**

        1. **Numerical Stability**: Classical Gram-Schmidt (CGS) is numerically
           unstable—it loses orthogonality due to catastrophic cancellation when
           subtracting nearly-equal vectors. QR via Householder reflections is
           backward stable, maintaining orthogonality to machine precision.

        2. **Performance**: While both are O(n³), QR uses optimized BLAS Level 3
           operations (cache-efficient, vectorized, multi-threaded with OpenBLAS
           or MKL). For n=4096 (12 qubits), QR is ~100-1000x faster than pure
           Python Gram-Schmidt loops.

        3. **Robustness**: LAPACK's QR handles all edge cases (near-zero vectors,
           linear dependence) internally with decades of battle-tested code.

        **Mathematical Correctness**:

        Given target state |ψ⟩, we construct A = [|ψ⟩ | I_{n-1}] where I_{n-1}
        is the identity matrix with the first column removed. QR decomposition
        gives A = QR where Q is unitary and R is upper triangular.

        Since A[:, 0] = |ψ⟩ and R is upper triangular:
            A[:, 0] = Q @ R[:, 0] = Q[:, 0] * R[0, 0]

        For normalized |ψ⟩, |R[0, 0]| = 1, so R[0, 0] = e^{iθ} for some phase θ.
        Thus Q[:, 0] = |ψ⟩ * e^{-iθ}. We apply phase correction to ensure
        U[:, 0] = |ψ⟩ exactly.

        **Phase Correction Preserves Unitarity**:

        The correction U[:, 0] *= e^{iθ} is equivalent to U = Q @ D where
        D = diag(e^{iθ}, 1, 1, ..., 1). Since D is a diagonal unitary matrix
        (all diagonal entries have unit modulus), U = Q @ D is unitary:
            U†U = D†Q†QD = D†ID = D†D = I  ✓

        Returns
        -------
        np.ndarray
            Unitary matrix of shape (2^n, 2^n) with dtype complex128.
            The first column is exactly the target state vector.
            Satisfies U†U = UU† = I to machine precision (~1e-15).

        Notes
        -----
        The result is cached after first computation. Subsequent calls return
        the cached matrix without recomputation.

        References
        ----------
        .. [1] Golub, G. H., & Van Loan, C. F. (2013). "Matrix Computations"
               (4th ed.). Johns Hopkins University Press. Section 5.2.
        .. [2] Higham, N. J. (2002). "Accuracy and Stability of Numerical
               Algorithms" (2nd ed.). SIAM. Chapter 19.
        """
        # Return cached result if available
        if self._unitary_matrix is not None:
            return self._unitary_matrix

        n = len(self._state_vector)

        # Construct matrix A with target state as first column.
        # The remaining columns are standard basis vectors e_1, e_2, ..., e_{n-1}.
        # This ensures A has full rank (the target state plus n-1 linearly
        # independent vectors span the full n-dimensional space).
        A = np.eye(n, dtype=np.complex128)
        A[:, 0] = self._state_vector

        # QR decomposition using LAPACK's Householder-based algorithm.
        # Q is unitary (Q†Q = I), R is upper triangular.
        # Time complexity: O(n³) with highly optimized BLAS operations.
        # Space complexity: O(n²) for the Q matrix.
        Q, R = np.linalg.qr(A)

        # Phase correction to ensure U[:, 0] = target_state exactly.
        #
        # After QR decomposition:
        #   A[:, 0] = Q[:, 0] * R[0, 0]
        #   target_state = Q[:, 0] * R[0, 0]
        #
        # Since target_state is normalized, |R[0, 0]| = 1, meaning R[0, 0] = e^{iθ}.
        # Therefore Q[:, 0] = target_state * e^{-iθ}.
        #
        # To recover target_state as the first column:
        #   U[:, 0] = Q[:, 0] * e^{iθ} = Q[:, 0] * conj(e^{-iθ})
        #
        # We compute e^{iθ} = <target_state | Q[:, 0]>* = conj(<target|Q[:,0]>)
        # and multiply the first column by this phase.
        #
        # Equivalently, this is U = Q @ diag(e^{iθ}, 1, ..., 1), which preserves
        # unitarity since diagonal matrices with unit-modulus entries are unitary.
        alpha = np.vdot(self._state_vector, Q[:, 0])  # <target|Q[:,0]> = e^{iθ}
        Q[:, 0] *= np.conj(alpha)  # Multiply by e^{-iθ} to get target_state

        # Cache the result for subsequent calls
        self._unitary_matrix = Q
        return Q

    def on(self, *qubits: Any) -> Any:
        """Apply this gate to the specified qubits.

        Creates a Cirq operation by wrapping the computed unitary in a
        Cirq-compatible gate object.

        Parameters
        ----------
        *qubits : cirq.LineQubit
            Qubits to apply the gate to.

        Returns
        -------
        cirq.Operation
            Gate operation on the specified qubits.

        Raises
        ------
        ImportError
            If Cirq is not installed.
        """
        # Get the cached Cirq gate class (created once, reused thereafter)
        gate_class = _get_or_create_cirq_gate_class()

        # Instantiate the gate with our computed unitary
        gate = gate_class(self._unitary_(), self._num_qubits)
        return gate.on(*qubits)


# =============================================================================
# Cirq Gate Implementation (Module-Level for Performance)
# =============================================================================

# Module-level cache for the dynamically-created Cirq gate class.
# This is created lazily to avoid import errors when Cirq is not installed.
# The class is created exactly once and reused for all subsequent gate
# instantiations, avoiding the overhead of class creation on each call.
_CirqAmplitudeGateClass: type | None = None


def _get_or_create_cirq_gate_class() -> type:
    """Get the Cirq gate class, creating it lazily on first use.

    This function creates a class that inherits from ``cirq.Gate`` exactly
    once, caching it for subsequent calls. This avoids the overhead of
    creating a new class definition on every gate instantiation.

    The lazy creation pattern is necessary because:
    1. Cirq may not be installed, so we can't define the class at module load
    2. Creating classes inside methods on every call is inefficient

    Returns
    -------
    type
        A class that inherits from ``cirq.Gate`` and can be instantiated
        with ``(unitary, n_qubits)`` arguments.

    Raises
    ------
    ImportError
        If Cirq is not installed.

    Notes
    -----
    **Thread Safety:**
    The caching uses a simple global variable without locking. In a race
    condition where two threads call this function simultaneously before
    the class is created, both may create the class. This is benign because:

    1. Both threads create identical classes
    2. One assignment will win, and subsequent calls use that class
    3. The extra class creation is wasted work, not incorrect behavior

    The simplicity of this approach is preferred over adding lock overhead
    for what is typically a one-time initialization.
    """
    global _CirqAmplitudeGateClass

    # Fast path: class already created
    if _CirqAmplitudeGateClass is not None:
        return _CirqAmplitudeGateClass

    # Slow path: create the class (happens once per process)
    try:
        import cirq
    except ImportError as e:
        raise ImportError(
            "Cirq is required for the 'cirq' backend. "
            "Install it with: pip install cirq-core"
        ) from e

    class _CirqAmplitudeGate(cirq.Gate):
        """Cirq gate for amplitude encoding state preparation.

        This gate wraps a pre-computed unitary matrix for preparing
        an arbitrary quantum state via amplitude encoding. It inherits
        from ``cirq.Gate`` to integrate properly with Cirq's circuit
        operations, simulation, and visualization.

        Parameters
        ----------
        unitary : np.ndarray
            The unitary matrix that prepares the target state.
            Must be of shape (2^n, 2^n) for n qubits.
        n_qubits : int
            Number of qubits this gate acts on.

        Notes
        -----
        This class is created dynamically at runtime to avoid import
        errors when Cirq is not installed. It is cached at module level
        after first creation for performance.
        """

        __slots__ = ("_unitary", "_n_qubits")

        def __init__(self, unitary: np.ndarray, n_qubits: int) -> None:
            """Initialize the amplitude encoding gate."""
            self._unitary = unitary
            self._n_qubits = n_qubits

        def _num_qubits_(self) -> int:
            """Return number of qubits this gate acts on."""
            return self._n_qubits

        def _unitary_(self) -> np.ndarray:
            """Return the unitary matrix for this gate."""
            return self._unitary

        def _circuit_diagram_info_(self, args: Any) -> cirq.CircuitDiagramInfo:
            """Return circuit diagram symbols for visualization.

            Returns 'AmpEnc' for the first qubit and '#' for continuation
            qubits, following Cirq's multi-qubit gate convention.
            """
            wire_symbols = tuple(
                "AmpEnc" if i == 0 else "#" for i in range(self._n_qubits)
            )
            return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    # Cache the class for future use
    _CirqAmplitudeGateClass = _CirqAmplitudeGate
    return _CirqAmplitudeGateClass
