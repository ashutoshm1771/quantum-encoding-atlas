"""Basis encoding for discrete/binary quantum data embedding.

This module implements BasisEncoding, a fundamental quantum data encoding
technique that maps binary (discrete) classical data directly to computational
basis states. Each classical bit is encoded as the state of a corresponding
qubit: 0 → |0⟩ and 1 → |1⟩.

Basis encoding is one of the simplest quantum data encodings, offering:

1. **Simplicity**: Direct one-to-one mapping from bits to qubits
2. **Minimal Depth**: Circuit depth of 1 (only X gates on select qubits)
3. **Classical Simulability**: Trivially simulable (product of basis states)
4. **Hardware Efficiency**: No two-qubit gates required
5. **Deterministic**: No probabilistic amplitude—pure computational basis states

The encoding creates quantum states of the form:

    |ψ(x)⟩ = |x₀x₁...xₙ₋₁⟩

where xᵢ ∈ {0, 1} is the i-th binary feature. This corresponds to applying
X gates to qubits where the corresponding classical bit is 1:

    |ψ(x)⟩ = X^{x₀} ⊗ X^{x₁} ⊗ ... ⊗ X^{xₙ₋₁} |0⟩⊗ⁿ

Mathematical Background
-----------------------
Basis encoding provides a direct classical-to-quantum mapping where each
classical bit string corresponds to exactly one computational basis state.
For n features:

    x = (x₀, x₁, ..., xₙ₋₁) ∈ {0, 1}ⁿ  →  |x₀x₁...xₙ₋₁⟩

The circuit implementation applies an X (NOT) gate to each qubit whose
corresponding classical bit is 1:

    |0⟩ ──X^{xᵢ}── → |xᵢ⟩

The X gate is defined as:

    X = |0⟩⟨1| + |1⟩⟨0| = [[0, 1], [1, 0]]

So X|0⟩ = |1⟩ and X|1⟩ = |0⟩. When xᵢ = 0, no gate is applied (identity).
When xᵢ = 1, the X gate flips |0⟩ to |1⟩.

Note on Data Preprocessing
--------------------------
Basis encoding is designed for binary data. For continuous inputs, this
implementation applies a configurable threshold (default 0.5):

- Values > threshold are mapped to 1 (X gate applied)
- Values ≤ threshold are mapped to 0 (no gate applied)

The threshold can be customized at construction time:

    >>> enc = BasisEncoding(n_features=4, threshold=0.0)  # signed data
    >>> enc = BasisEncoding(n_features=4, threshold=0.7)  # custom boundary

For more complex binarization logic, preprocess your data before encoding:

- **Median split**: ``x_binary = (x > np.median(x)).astype(int)``
- **Quantile-based**: Use pandas ``qcut`` or sklearn's ``Binarizer``

Debugging
---------
This module supports Python's standard logging for debugging binarization
and circuit generation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.basis').setLevel(logging.DEBUG)

Debug logs include:
- Initialization parameters (n_features, threshold)
- Binarization results (input range, binary output, count of 0s and 1s)
- Batch processing progress

Resource Analysis
-----------------
BasisEncoding provides methods to analyze circuit resources:

- ``gate_count_breakdown()``: Get worst-case gate counts by type (for API consistency)
- ``binarize(x)``: Inspect how input data will be converted to binary
- ``actual_gate_count(x)``: Count exact X gates for specific input (data-dependent)
- ``resource_summary(x)``: Complete breakdown of circuit resources for specific data

These complement ``properties.gate_count`` (which reports worst-case bounds):

    >>> enc = BasisEncoding(n_features=8)
    >>> enc.properties.gate_count  # Theoretical maximum (worst case)
    8
    >>> enc.actual_gate_count([1, 0, 0, 0, 0, 0, 0, 0])  # Actual for sparse data
    1

Use Cases
---------
Basis encoding is particularly suited for:

- **Discrete/categorical data**: One-hot encoded features, binary flags
- **Combinatorial optimization**: QAOA, VQE for discrete problems
- **Database search**: Grover's algorithm with marked states
- **Quantum random access memory (QRAM)**: Address encoding
- **Benchmarking**: Baseline for comparing other encodings

Limitations
-----------
- No amplitude-based information (all basis state coefficients are 0 or 1)
- Linear qubit scaling (n features require n qubits)
- Limited expressivity for continuous data patterns
- No entanglement created (product states only)

References
----------
.. [1] Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and
       Quantum Information." Cambridge University Press. Chapter 1.
.. [2] Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
       Quantum Computers." Springer. Chapter 4: Quantum Feature Maps.
.. [3] Weigold, M., et al. (2020). "Data encoding patterns for quantum
       computing." IEEE International Conference on Quantum Computing
       and Engineering (QCE).
.. [4] LaRose, R., & Coyle, B. (2020). "Robust data encodings for quantum
       classifiers." Physical Review A, 102(3), 032420.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

# =============================================================================
# Module-Level Logger
# =============================================================================
# Configure a logger for this module to enable optional debug output.
# By default, Python's logging is disabled (level WARNING), so these
# debug statements have no performance impact in production.
#
# To enable debug logging for this module:
#   import logging
#   logging.getLogger('encoding_atlas.encodings.basis').setLevel(logging.DEBUG)
#   logging.basicConfig(level=logging.DEBUG)
#
# Or configure via your application's logging setup.
logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["BasisEncoding"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default threshold for binarizing continuous inputs.
# Values strictly greater than this threshold are mapped to 1 (X gate applied).
# Values less than or equal to this threshold are mapped to 0 (no gate).
#
# The choice of 0.5 is conventional for data normalized to [0, 1].
# For other data ranges, users should preprocess their data or use a
# custom threshold via manual binarization before encoding.
_DEFAULT_BINARIZATION_THRESHOLD: float = 0.5


# =============================================================================
# Type Definitions
# =============================================================================


class GateCountBreakdown(TypedDict):
    """Type definition for the gate_count_breakdown() return value.

    Provides a detailed breakdown of gate counts for basis encoding.
    Since basis encoding uses only X gates (no two-qubit gates), the
    breakdown is simpler than other encodings.

    .. note::
        Basis encoding has **data-dependent** gate counts: only qubits
        corresponding to 1-valued features receive X gates. The values
        returned by ``gate_count_breakdown()`` represent the **worst case**
        (all features = 1). Use ``actual_gate_count(x)`` for data-specific
        counts.

    Attributes
    ----------
    x_gates : int
        Maximum number of X (NOT) gates (equals n_features).
        Actual count depends on input data; use ``actual_gate_count(x)``
        for data-specific counts.

    total_single_qubit : int
        Total number of single-qubit gates (equals x_gates).

    total_two_qubit : int
        Total number of two-qubit gates (always 0 for basis encoding).

    total : int
        Total gate count (equals total_single_qubit since no two-qubit gates).

    is_worst_case : bool
        Always True, indicating this is the worst-case estimate.

    Examples
    --------
    >>> enc = BasisEncoding(n_features=8)
    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['x_gates']
    8
    >>> breakdown['total_two_qubit']
    0
    >>> breakdown['is_worst_case']
    True

    For actual gate count with specific data, use actual_gate_count():

    >>> enc.actual_gate_count([1, 0, 0, 0, 0, 0, 0, 0])  # Only 1 X gate
    1
    """

    x_gates: int
    """Maximum number of X gates (worst case: n_features)."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates (always 0)."""

    total: int
    """Total gate count (worst case)."""

    is_worst_case: bool
    """True, indicating this is the worst-case estimate."""


class BasisEncoding(BaseEncoding):
    """Basis encoding for binary/discrete data into computational basis states.

    BasisEncoding provides the simplest quantum data encoding by mapping
    binary classical data directly to computational basis states. Each qubit
    represents one classical bit, with X gates applied to encode 1s.

    This encoding creates pure computational basis states (no superposition),
    making it:

    - **Trivially simulable**: Classical computers can efficiently track basis states
    - **Hardware efficient**: Only single-qubit X gates, no entanglement
    - **Deterministic**: Measurement always yields the encoded bit string
    - **Minimal depth**: Circuit depth of 1 regardless of feature count

    The circuit structure is:

        |0⟩ ─── X^{x₀} ─── |x₀⟩
        |0⟩ ─── X^{x₁} ─── |x₁⟩
        |0⟩ ─── X^{x₂} ─── |x₂⟩
        ...

    where X^{xᵢ} means "apply X gate if xᵢ = 1, otherwise identity".

    Parameters
    ----------
    n_features : int
        Number of binary features to encode. Must be a positive integer.
        Each feature requires one qubit, so this also determines the
        number of qubits in the circuit.
    threshold : float, default=0.5
        Binarization threshold for continuous inputs. Values strictly greater
        than this threshold are mapped to 1 (X gate applied), while values
        less than or equal to the threshold are mapped to 0 (no gate).

        Common threshold choices:
        - 0.5: Standard for data normalized to [0, 1]
        - 0.0: Treat negative values as 0, positive as 1
        - Custom: Match your data's natural decision boundary

    Attributes
    ----------
    n_features : int
        Number of binary features (inherited from BaseEncoding).
    n_qubits : int
        Number of qubits, equal to n_features.
    depth : int
        Circuit depth, always 1 for basis encoding.
    threshold : float
        Binarization threshold. Values > threshold become 1.

    Examples
    --------
    Create a basic basis encoding for 4 binary features:

    >>> from encoding_atlas import BasisEncoding
    >>> import numpy as np
    >>> enc = BasisEncoding(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.depth
    1
    >>> enc.threshold
    0.5

    Encode binary data directly:

    >>> x_binary = np.array([1, 0, 1, 1])
    >>> circuit = enc.get_circuit(x_binary, backend='pennylane')
    >>> callable(circuit)
    True

    Continuous data is automatically binarized at threshold 0.5:

    >>> x_continuous = np.array([0.8, 0.2, 0.6, 0.9])  # -> [1, 0, 1, 1]
    >>> circuit = enc.get_circuit(x_continuous, backend='pennylane')

    Custom threshold for data with different scales:

    >>> enc_custom = BasisEncoding(n_features=4, threshold=0.0)
    >>> enc_custom.threshold
    0.0
    >>> # Now: positive values -> 1, non-positive values -> 0
    >>> x_signed = np.array([-0.5, 0.5, -0.1, 0.1])  # -> [0, 1, 0, 1]
    >>> circuit = enc_custom.get_circuit(x_signed, backend='pennylane')

    Generate circuits for different backends:

    >>> x = np.array([1, 0, 1, 0])
    >>> qiskit_circuit = enc.get_circuit(x, backend='qiskit')
    >>> qiskit_circuit.num_qubits
    4

    >>> cirq_circuit = enc.get_circuit(x, backend='cirq')
    >>> len(cirq_circuit.all_qubits())
    4

    Batch processing multiple samples:

    >>> X = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1]])
    >>> circuits = enc.get_circuits(X, backend='pennylane')
    >>> len(circuits)
    3

    Access encoding properties:

    >>> props = enc.properties
    >>> props.is_entangling
    False
    >>> props.simulability
    'simulable'
    >>> props.depth
    1

    Analyze actual resource usage for specific data:

    >>> enc = BasisEncoding(n_features=4)
    >>> enc.properties.gate_count  # Theoretical worst-case
    4
    >>> enc.actual_gate_count([1, 0, 0, 0])  # Actual for sparse data
    1
    >>> enc.binarize([0.8, 0.2, 0.6, 0.4])  # Inspect binarization
    array([1, 0, 1, 0])

    References
    ----------
    .. [1] Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and
           Quantum Information." Cambridge University Press.
    .. [2] Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
           Quantum Computers." Springer.

    See Also
    --------
    AngleEncoding : Encodes continuous features as rotation angles.
    AmplitudeEncoding : Encodes features in state amplitudes (logarithmic qubits).
    IQPEncoding : Adds entanglement via ZZ interactions for richer expressivity.

    Notes
    -----
    **Binarization Behavior**:
    This implementation automatically binarizes continuous inputs using the
    configured threshold (default 0.5):

    - x > threshold → 1 (X gate applied)
    - x ≤ threshold → 0 (no gate applied)

    For pre-binarized data (0s and 1s), this threshold has no effect.
    The threshold can be customized at construction time to match your
    data's natural decision boundary.

    **Comparison with Other Encodings**:

    +------------------+----------+-------+------------+---------------+
    | Encoding         | Qubits   | Depth | Entangling | Data Type     |
    +==================+==========+=======+============+===============+
    | BasisEncoding    | n        | 1     | No         | Binary        |
    +------------------+----------+-------+------------+---------------+
    | AngleEncoding    | n        | 1     | No         | Continuous    |
    +------------------+----------+-------+------------+---------------+
    | AmplitudeEncoding| log₂(n)  | O(2ⁿ) | Yes        | Continuous    |
    +------------------+----------+-------+------------+---------------+
    | IQPEncoding      | n        | O(n)  | Yes        | Continuous    |
    +------------------+----------+-------+------------+---------------+

    **When to Use Basis Encoding**:

    - Your data is naturally binary or categorical
    - You need a simple baseline encoding for comparison
    - Circuit depth must be minimal (NISQ hardware)
    - You're implementing algorithms that work with marked basis states
      (Grover's search, QAOA for combinatorial problems)

    **When NOT to Use Basis Encoding**:

    - Your data is continuous and pattern-rich (use AngleEncoding or IQP)
    - You need amplitude-based information processing
    - Your algorithm requires superposition or entanglement from the encoding

    **Cirq Backend Note**:

    Cirq's ``circuit.all_qubits()`` only returns qubits that have operations.
    For sparse inputs (many zeros), this may return fewer qubits than expected.
    Always use ``encoding.n_qubits`` to determine the actual qubit count
    required for your circuit, not ``circuit.all_qubits()``.

    **Understanding Gate Counts (Important)**:

    BasisEncoding provides two ways to analyze gate counts:

    1. **Theoretical (worst-case)**: ``properties.gate_count`` returns
       ``n_features``, assuming all features binarize to 1. This is
       computed once at construction time and cached. Use this for:

       - Capacity planning and resource allocation
       - Worst-case guarantees
       - Comparing encoding algorithms theoretically
       - Documentation and specifications

    2. **Actual (data-dependent)**: ``actual_gate_count(x)`` returns the
       exact count of X gates for specific input data. This equals the
       number of 1s after binarization. Use this for:

       - Accurate hardware resource estimation
       - Benchmarking with real datasets
       - Cost analysis for sparse data
       - Debugging unexpected behavior

    Example showing the difference:

    >>> enc = BasisEncoding(n_features=100)
    >>> enc.properties.gate_count  # Theoretical: always 100
    100
    >>> sparse_data = np.zeros(100)
    >>> sparse_data[0] = 1  # Only one feature is "on"
    >>> enc.actual_gate_count(sparse_data)  # Actual: only 1 gate
    1

    For sparse binary data, the actual gate count can be significantly
    lower than the theoretical maximum, which matters for:

    - Hardware execution time (fewer gates = faster circuits)
    - Error rates (fewer gates = less decoherence)
    - Classical simulation cost (sparser circuits = easier to simulate)
    """

    # =========================================================================
    # Class Constants
    # =========================================================================

    # Memory-efficient slot-based attribute storage.
    # BasisEncoding has no additional instance attributes beyond BaseEncoding,
    # but we declare empty __slots__ to maintain the optimization pattern
    # and prevent accidental __dict__ creation.
    __slots__ = ("threshold",)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, n_features: int, threshold: float = 0.5) -> None:
        """Initialize the basis encoding.

        Parameters
        ----------
        n_features : int
            Number of binary features to encode. Each feature requires one
            qubit, so this determines both the input dimension and the
            number of qubits in the circuit.
        threshold : float, default=0.5
            Binarization threshold for continuous inputs. Values strictly
            greater than this threshold are mapped to 1 (X gate applied),
            while values less than or equal are mapped to 0 (no gate).

            Must be a finite real number. Common choices:
            - 0.5 for data normalized to [0, 1]
            - 0.0 for signed data (negative -> 0, positive -> 1)

        Raises
        ------
        ValueError
            If n_features is not a positive integer (raised by parent class).
        TypeError
            If threshold is not a numeric type (int or float).
        ValueError
            If threshold is NaN or infinite.

        Examples
        --------
        >>> enc = BasisEncoding(n_features=4)
        >>> enc.n_qubits
        4
        >>> enc.depth
        1
        >>> enc.threshold
        0.5

        >>> enc = BasisEncoding(n_features=8)
        >>> enc.properties.gate_count  # Worst case: all X gates
        8

        >>> enc = BasisEncoding(n_features=4, threshold=0.0)
        >>> enc.threshold
        0.0
        """
        # =======================================================================
        # THRESHOLD VALIDATION
        # =======================================================================
        # Validate threshold before calling parent constructor to fail fast
        # on invalid parameters.

        # Type check: must be a numeric type (int or float, but not bool)
        # Note: bool is a subclass of int in Python, so we explicitly exclude it
        if isinstance(threshold, bool):
            raise TypeError(
                f"threshold must be a numeric type (int or float), "
                f"got {type(threshold).__name__}"
            )
        if not isinstance(threshold, (int, float)):
            raise TypeError(
                f"threshold must be a numeric type (int or float), "
                f"got {type(threshold).__name__}"
            )

        # Convert to float for consistent internal representation
        threshold_float = float(threshold)

        # Value check: must be finite (not NaN or Inf)
        if not np.isfinite(threshold_float):
            raise ValueError(
                f"threshold must be a finite number, got {threshold_float}"
            )

        # =======================================================================
        # INITIALIZATION
        # =======================================================================
        # Call parent constructor with threshold in config for equality/hashing
        # Note: We pass threshold to config so that two BasisEncodings with
        # different thresholds are correctly identified as different objects
        super().__init__(n_features, threshold=threshold_float)

        # Store the binarization threshold as an instance attribute
        # for easy access via the threshold property
        self.threshold: float = threshold_float

        # Log initialization for debugging
        logger.debug(
            "BasisEncoding initialized: n_features=%d, threshold=%.6f, n_qubits=%d",
            self.n_features,
            self.threshold,
            self.n_qubits,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        For basis encoding, each feature is encoded on a dedicated qubit,
        so n_qubits equals n_features.

        Returns
        -------
        int
            Number of qubits, equal to n_features.

        Examples
        --------
        >>> BasisEncoding(n_features=4).n_qubits
        4
        >>> BasisEncoding(n_features=10).n_qubits
        10
        """
        return self.n_features

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        Basis encoding has constant depth of 1, regardless of the number
        of features. All X gates can be applied in parallel since they
        act on different qubits.

        Returns
        -------
        int
            Circuit depth, always 1.

        Notes
        -----
        The depth of 1 makes basis encoding ideal for NISQ devices where
        circuit depth directly impacts error accumulation.
        """
        return 1

    # =========================================================================
    # Circuit Generation
    # =========================================================================

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input.

        Encoding-specific seam called by the inherited ``get_circuit``/
        ``get_circuits`` template methods. Binarizes the input via the
        configured threshold and dispatches to the backend.

        Accepts a 1D array (the normal case) or a 2D single-sample
        ``(1, n_features)`` array for robustness when called directly.
        """
        # Defensive 2D-to-1D handling for direct callers; the template
        # methods already pass 1D input.
        if x.ndim == 2:
            x = x[0]

        # Binarize continuous inputs using threshold:
        # values > threshold become 1; values <= threshold become 0.
        x_binary = (x > self.threshold).astype(int)

        # Log binarization details — invaluable for diagnosing unexpected
        # encoding behavior in production data pipelines.
        if logger.isEnabledFor(logging.DEBUG):
            n_ones = int(np.sum(x_binary))
            n_zeros = len(x_binary) - n_ones
            logger.debug(
                "Binarization complete: threshold=%.6f, input_range=[%.6f, %.6f], "
                "binary_result=%s, ones=%d, zeros=%d",
                self.threshold,
                float(np.min(x)),
                float(np.max(x)),
                x_binary.tolist(),
                n_ones,
                n_zeros,
            )

        if backend == "pennylane":
            return self._to_pennylane(x_binary)
        elif backend == "qiskit":
            return self._to_qiskit(x_binary)
        elif backend == "cirq":
            return self._to_cirq(x_binary)
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
            )

    # =========================================================================
    # Resource Analysis Methods
    # =========================================================================
    #
    # These methods provide data-dependent analysis of circuit resources.
    # Unlike the `properties` attribute (which reports theoretical worst-case
    # bounds computed at construction time), these methods analyze specific
    # input data to return exact resource counts.
    #
    # Use cases:
    # - Accurate hardware resource estimation for specific datasets
    # - Comparing actual vs. theoretical circuit complexity
    # - Debugging binarization behavior
    # - Performance benchmarking with realistic data

    def binarize(self, x: ArrayLike) -> NDArray[np.integer[Any]]:
        """Binarize input data using this encoding's threshold.

        This method exposes the internal binarization logic, allowing users
        to inspect how their data will be converted to binary before encoding.
        This is useful for debugging, understanding encoding behavior, and
        preprocessing validation.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (n_samples, n_features).
            Can be any numeric type; will be compared against the threshold.

        Returns
        -------
        NDArray[np.integer]
            Binarized data with same shape as input.
            Each element is 0 if original value ≤ threshold, else 1.

        Raises
        ------
        ValueError
            If input shape doesn't match n_features.
        ValueError
            If input contains NaN or infinite values.

        Examples
        --------
        Inspect binarization for a single sample:

        >>> enc = BasisEncoding(n_features=4)
        >>> x = np.array([0.8, 0.2, 0.6, 0.4])
        >>> enc.binarize(x)
        array([1, 0, 1, 0])

        Custom threshold affects binarization:

        >>> enc_custom = BasisEncoding(n_features=4, threshold=0.0)
        >>> x_signed = np.array([-0.5, 0.5, -0.1, 0.1])
        >>> enc_custom.binarize(x_signed)
        array([0, 1, 0, 1])

        Batch binarization for multiple samples:

        >>> enc = BasisEncoding(n_features=3)
        >>> X = np.array([[0.1, 0.9, 0.5], [0.6, 0.4, 0.7]])
        >>> enc.binarize(X)
        array([[0, 1, 0],
               [1, 0, 1]])

        Verify before encoding:

        >>> enc = BasisEncoding(n_features=4)
        >>> x = np.array([0.51, 0.49, 0.50, 0.501])
        >>> binary = enc.binarize(x)
        >>> binary  # Exactly at threshold (0.5) -> 0
        array([1, 0, 0, 1])
        >>> enc.actual_gate_count(x)  # 2 ones = 2 X gates
        2

        Notes
        -----
        The binarization rule is:

        - x > threshold → 1
        - x ≤ threshold → 0

        Note that values exactly equal to the threshold are mapped to 0.
        This is consistent with the mathematical definition of "strictly
        greater than" used in the encoding.

        This method is useful for:

        1. **Debugging**: Understanding why certain inputs produce
           unexpected encodings
        2. **Validation**: Verifying that your data preprocessing
           produces the expected binary patterns
        3. **Analysis**: Computing statistics about your binarized data
           (e.g., sparsity, balance between 0s and 1s)

        See Also
        --------
        actual_gate_count : Count X gates for specific input data.
        get_circuit : Generate the quantum circuit for input data.
        """
        x_validated = self._validate_input(x)

        # Apply threshold comparison
        # Values > threshold become 1, values <= threshold become 0
        x_binary = (x_validated > self.threshold).astype(np.int32)

        logger.debug(
            "Binarization: threshold=%.6f, shape=%s, ones=%d, zeros=%d",
            self.threshold,
            x_binary.shape,
            int(np.sum(x_binary)),
            int(np.prod(x_binary.shape)) - int(np.sum(x_binary)),
        )

        return x_binary

    def transform_input(self, x: ArrayLike) -> NDArray[np.integer[Any]]:
        """Transform input data according to the encoding's preprocessing rules.

        This method implements the ``DataTransformable`` protocol, exposing
        the encoding's internal data transformation logic. For BasisEncoding,
        this applies the binarization process: values above the threshold
        become 1, values at or below become 0.

        This is an alias for :meth:`binarize` that provides a standardized
        interface consistent with other encodings implementing the
        ``DataTransformable`` protocol (e.g., ``AmplitudeEncoding``).

        Parameters
        ----------
        x : ArrayLike
            Input features of shape ``(n_features,)`` or ``(n_samples, n_features)``.
            Can be any numeric type; will be compared against the threshold.

        Returns
        -------
        NDArray[np.integer]
            Binarized data with the same shape as input.
            Each element is 0 if original value ≤ threshold, else 1.

        Raises
        ------
        ValueError
            If input shape doesn't match ``n_features``.
        ValueError
            If input contains NaN or infinite values.

        Examples
        --------
        Transform continuous data to binary:

        >>> enc = BasisEncoding(n_features=4)
        >>> x = np.array([0.8, 0.2, 0.6, 0.4])
        >>> enc.transform_input(x)
        array([1, 0, 1, 0])

        Using the protocol interface:

        >>> from encoding_atlas.core.protocols import DataTransformable
        >>> enc = BasisEncoding(n_features=4)
        >>> if isinstance(enc, DataTransformable):
        ...     binary = enc.transform_input([0.9, 0.1, 0.8, 0.3])
        ...     print(binary)
        [1 0 1 0]

        Write generic code that works with any DataTransformable encoding:

        >>> def show_transformation(enc, x):
        ...     '''Show how an encoding transforms data.'''
        ...     from encoding_atlas.core.protocols import DataTransformable
        ...     if isinstance(enc, DataTransformable):
        ...         transformed = enc.transform_input(x)
        ...         print(f"{enc.__class__.__name__}: {x} -> {transformed}")
        ...     else:
        ...         print(f"{enc.__class__.__name__}: No transformation")

        Notes
        -----
        This method is idempotent for binary data: if the input is already
        binary (0s and 1s), the output equals the input (assuming the default
        threshold of 0.5).

        The transformation preserves shape: a 1D input returns a 1D output,
        and a 2D input returns a 2D output.

        See Also
        --------
        binarize : The underlying binarization method (identical behavior).
        actual_gate_count : Count X gates for specific input data.
        DataTransformable : The protocol this method implements.
        """
        # Delegate to binarize() for the actual implementation.
        # This ensures consistent behavior and avoids code duplication.
        return self.binarize(x)

    def actual_gate_count(self, x: ArrayLike) -> int:
        """Count the actual number of X gates for specific input data.

        Unlike ``properties.gate_count`` which reports the theoretical
        worst-case (all features are 1), this method analyzes the actual
        input data to return the exact number of X gates that will be
        applied in the generated circuit.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (1, n_features).
            Will be binarized using the configured threshold before counting.

        Returns
        -------
        int
            Number of X gates that will be applied. This equals the count
            of 1s in the binarized input (i.e., values > threshold).

            - Minimum: 0 (all features ≤ threshold)
            - Maximum: n_features (all features > threshold)

        Raises
        ------
        ValueError
            If input shape doesn't match n_features.
        ValueError
            If input contains NaN or infinite values.

        Examples
        --------
        Compare theoretical vs. actual gate counts:

        >>> enc = BasisEncoding(n_features=4)
        >>> enc.properties.gate_count  # Theoretical worst case
        4
        >>> enc.actual_gate_count([0, 0, 0, 0])  # All zeros
        0
        >>> enc.actual_gate_count([1, 0, 0, 0])  # One bit set
        1
        >>> enc.actual_gate_count([1, 1, 1, 1])  # All ones (worst case)
        4

        Continuous data is binarized first:

        >>> enc = BasisEncoding(n_features=4, threshold=0.5)
        >>> enc.actual_gate_count([0.8, 0.2, 0.6, 0.4])  # -> [1, 0, 1, 0]
        2

        Useful for resource estimation:

        >>> enc = BasisEncoding(n_features=100)
        >>> sparse_data = np.zeros(100)
        >>> sparse_data[:5] = 1  # Only 5 features are "on"
        >>> enc.actual_gate_count(sparse_data)
        5
        >>> enc.properties.gate_count  # Theoretical would overestimate
        100

        Calculate average gate count for a dataset:

        >>> enc = BasisEncoding(n_features=4)
        >>> X = np.random.rand(1000, 4)  # 1000 samples
        >>> avg_gates = np.mean([enc.actual_gate_count(x) for x in X])
        >>> print(f"Average gates per sample: {avg_gates:.2f}")  # ~2.0 for uniform data
        Average gates per sample: ...

        Notes
        -----
        **When to use this method vs. ``properties.gate_count``:**

        +---------------------------+----------------------------------+
        | Use ``properties.gate_count`` for: | Use ``actual_gate_count()`` for: |
        +===========================+==================================+
        | Capacity planning         | Specific dataset analysis        |
        +---------------------------+----------------------------------+
        | Worst-case guarantees     | Accurate resource estimation     |
        +---------------------------+----------------------------------+
        | Comparing encodings       | Benchmarking with real data      |
        +---------------------------+----------------------------------+
        | Documentation             | Hardware cost calculation        |
        +---------------------------+----------------------------------+

        **Performance:**

        This method is O(n) in the number of features, with minimal overhead.
        It's safe to call repeatedly for large datasets.

        **Relationship to other metrics:**

        - ``actual_gate_count(x) == sum(binarize(x))`` for 1D input
        - ``actual_gate_count(x) <= properties.gate_count`` always
        - ``actual_gate_count(x) == properties.gate_count`` iff all
          binarized values are 1

        See Also
        --------
        binarize : Inspect the binarized form of input data.
        properties : Access theoretical worst-case circuit properties.
        """
        # Validate and flatten input
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            if x_validated.shape[0] != 1:
                raise ValueError(
                    f"actual_gate_count expects a single sample, got shape "
                    f"{x_validated.shape}. For batch processing, iterate over "
                    f"samples or use: [enc.actual_gate_count(xi) for xi in X]"
                )
            x_validated = x_validated[0]

        # Binarize using threshold
        x_binary = (x_validated > self.threshold).astype(int)

        # Count ones = number of X gates
        gate_count = int(np.sum(x_binary))

        logger.debug(
            "Gate count analysis: input_range=[%.6f, %.6f], threshold=%.6f, "
            "actual_gates=%d, max_gates=%d",
            float(np.min(x_validated)),
            float(np.max(x_validated)),
            self.threshold,
            gate_count,
            self.n_features,
        )

        return gate_count

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type (worst case).

        Returns the maximum possible gate counts for basis encoding, which
        occurs when all features are above the binarization threshold (all 1s).
        For data-specific gate counts, use ``actual_gate_count(x)`` instead.

        This method is useful for:

        - Resource planning (worst-case capacity)
        - Comparing encoding methods
        - Consistency with other encoding APIs

        Returns
        -------
        GateCountBreakdown
            TypedDict with worst-case gate counts:

            - ``'x_gates'``: Maximum X gates (equals n_features)
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Always 0 (no entanglement)
            - ``'total'``: Total gate count (equals n_features worst case)
            - ``'is_worst_case'``: True (use ``actual_gate_count`` for specific data)

        Examples
        --------
        >>> enc = BasisEncoding(n_features=8)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['x_gates']
        8
        >>> breakdown['total']
        8
        >>> breakdown['total_two_qubit']
        0

        Compare with actual gate count for specific data:

        >>> enc = BasisEncoding(n_features=4)
        >>> enc.gate_count_breakdown()['total']  # Worst case
        4
        >>> enc.actual_gate_count([1, 0, 0, 0])  # Actual for sparse data
        1

        Notes
        -----
        **Why worst-case?**

        Basis encoding has data-dependent circuit structure: only qubits
        corresponding to 1-valued features receive X gates. The worst case
        (all features = 1) is returned for:

        1. Consistency with other encodings' ``gate_count_breakdown()`` API
        2. Conservative resource planning
        3. Hardware capacity estimation

        **Getting actual counts:**

        For data-specific gate counts:

        - Use ``actual_gate_count(x)`` for a single sample
        - Use ``resource_summary(x)`` for comprehensive analysis

        **Basis encoding characteristics:**

        - No two-qubit gates (``total_two_qubit`` always 0)
        - Circuit depth always 1 (all X gates in parallel)
        - Trivially simulable (product states only)

        See Also
        --------
        actual_gate_count : Get exact gate count for specific input data.
        resource_summary : Get comprehensive resource breakdown for specific data.
        properties : Access theoretical circuit properties.
        """
        logger.debug(
            "Gate count breakdown (worst case): x_gates=%d, total=%d",
            self.n_features,
            self.n_features,
        )

        return GateCountBreakdown(
            x_gates=self.n_features,
            total_single_qubit=self.n_features,
            total_two_qubit=0,
            total=self.n_features,
            is_worst_case=True,
        )

    def resource_summary(self, x: ArrayLike) -> dict[str, Any]:
        """Generate a comprehensive resource summary for specific input data.

        Provides a detailed breakdown of circuit resources for the given
        input, including both theoretical bounds and actual data-dependent
        counts. Useful for reporting, optimization decisions, and debugging.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (1, n_features).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``n_qubits``: Number of qubits (always n_features)
            - ``depth``: Circuit depth (always 1)
            - ``actual_gate_count``: Exact X gates for this input
            - ``max_gate_count``: Theoretical maximum (n_features)
            - ``gate_efficiency``: Ratio actual/max (0.0 to 1.0)
            - ``binarized_input``: The binary representation of input
            - ``threshold``: Binarization threshold used
            - ``ones_count``: Number of 1s in binarized input
            - ``zeros_count``: Number of 0s in binarized input
            - ``sparsity``: Fraction of zeros (zeros_count / n_features)

        Examples
        --------
        Get detailed resource analysis:

        >>> enc = BasisEncoding(n_features=4)
        >>> summary = enc.resource_summary([0.8, 0.2, 0.6, 0.4])
        >>> summary['actual_gate_count']
        2
        >>> summary['max_gate_count']
        4
        >>> summary['gate_efficiency']
        0.5
        >>> summary['sparsity']
        0.5

        Use for reporting:

        >>> enc = BasisEncoding(n_features=8)
        >>> summary = enc.resource_summary([1, 0, 0, 0, 0, 0, 0, 1])
        >>> print(f"Using {summary['actual_gate_count']}/{summary['max_gate_count']} gates")
        Using 2/8 gates
        >>> print(f"Circuit is {summary['sparsity']:.0%} sparse")
        Circuit is 75% sparse

        Notes
        -----
        This method combines the functionality of ``binarize()`` and
        ``actual_gate_count()`` into a single comprehensive report.
        For performance-critical code where you only need the gate count,
        prefer ``actual_gate_count()`` directly.
        """
        # Validate input
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            if x_validated.shape[0] != 1:
                raise ValueError(
                    f"resource_summary expects a single sample, got shape "
                    f"{x_validated.shape}. For batch processing, iterate over "
                    f"samples."
                )
            x_validated = x_validated[0]

        # Compute binarization
        x_binary = (x_validated > self.threshold).astype(int)
        ones_count = int(np.sum(x_binary))
        zeros_count = self.n_features - ones_count

        # Build summary dictionary
        summary: dict[str, Any] = {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            # Gate counts
            "actual_gate_count": ones_count,
            "max_gate_count": self.n_features,
            "gate_efficiency": (
                ones_count / self.n_features if self.n_features > 0 else 0.0
            ),
            # Binarization details
            "binarized_input": x_binary.tolist(),
            "threshold": self.threshold,
            "ones_count": ones_count,
            "zeros_count": zeros_count,
            "sparsity": zeros_count / self.n_features if self.n_features > 0 else 0.0,
        }

        logger.debug(
            "Resource summary: actual_gates=%d/%d (%.1f%% efficiency), "
            "sparsity=%.1f%%",
            ones_count,
            self.n_features,
            summary["gate_efficiency"] * 100,
            summary["sparsity"] * 100,
        )

        return summary

    # =========================================================================
    # Backend-Specific Implementations
    # =========================================================================

    def _to_pennylane(self, x: NDArray[np.integer[Any]]) -> Any:
        """Generate PennyLane circuit function.

        Manually applies X gates to qubits corresponding to 1-valued
        features, ensuring consistent qubit ordering with Qiskit and Cirq.

        Parameters
        ----------
        x : NDArray[np.integer]
            Binarized input features of shape (n_features,).

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
        This implementation uses manual X gate application instead of
        PennyLane's built-in BasisEmbedding to ensure consistent qubit
        ordering conventions across all backends (Qiskit, Cirq, PennyLane).
        All backends follow the convention: feature x[i] controls qubit i.
        """
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(
                "PennyLane is required for the 'pennylane' backend. "
                "Install it with: pip install pennylane"
            ) from e

        # Capture values for closure to avoid mutable state issues
        # This ensures the circuit function is self-contained and thread-safe
        n_qubits = self.n_qubits
        features = x.copy()

        def circuit() -> None:
            """Apply basis embedding gates."""
            # Apply X gate to each qubit where the corresponding bit is 1.
            # PennyLane uses big-endian (wire 0 = MSB), so feature[i] maps
            # directly to wire i — no reversal needed.  The analysis module's
            # _simulate_qiskit handles the Qiskit LSB→MSB conversion separately.
            for i in range(n_qubits):
                if features[i] == 1:
                    qml.PauliX(wires=i)

        return circuit

    def _to_qiskit(self, x: NDArray[np.integer[Any]]) -> Any:
        """Generate Qiskit QuantumCircuit.

        Manually applies X gates to qubits corresponding to 1-valued
        features for maximum control and transparency.

        Parameters
        ----------
        x : NDArray[np.integer]
            Binarized input features of shape (n_features,).

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit implementing the encoding.

        Raises
        ------
        ImportError
            If Qiskit is not installed.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError(
                "Qiskit is required for the 'qiskit' backend. "
                "Install it with: pip install qiskit"
            ) from e

        qc = QuantumCircuit(self.n_qubits, name="BasisEncoding")

        # Apply X gate to each qubit where the corresponding bit is 1
        for i, val in enumerate(x):
            if val == 1:
                qc.x(i)

        return qc

    def _to_cirq(self, x: NDArray[np.integer[Any]]) -> Any:
        """Generate Cirq Circuit.

        Manually applies X gates to qubits corresponding to 1-valued
        features for maximum control and transparency.

        Parameters
        ----------
        x : NDArray[np.integer]
            Binarized input features of shape (n_features,).

        Returns
        -------
        cirq.Circuit
            Cirq circuit implementing the encoding.

        Raises
        ------
        ImportError
            If Cirq is not installed.
        """
        try:
            import cirq
        except ImportError as e:
            raise ImportError(
                "Cirq is required for the 'cirq' backend. "
                "Install it with: pip install cirq-core"
            ) from e

        qubits = cirq.LineQubit.range(self.n_qubits)

        # Collect all X gates into a single moment (parallel execution).
        # Cirq uses big-endian ordering (qubit 0 = MSB), same as PennyLane,
        # so feature[i] maps directly to qubit[i] — no reversal needed.
        x_gates = [cirq.X(qubits[i]) for i, val in enumerate(x) if val == 1]

        circuit = cirq.Circuit(cirq.Moment(x_gates)) if x_gates else cirq.Circuit()

        return circuit

    # =========================================================================
    # Properties Computation
    # =========================================================================

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical (worst-case) properties of this encoding.

        Calculates circuit metrics including qubit count, depth, gate counts,
        and encoding characteristics. These are **theoretical bounds** computed
        at construction time, independent of any specific input data.

        Returns
        -------
        EncodingProperties
            Computed properties including:

            - n_qubits: Number of qubits (= n_features)
            - depth: Circuit depth (always 1)
            - gate_count: **Worst-case** maximum (= n_features, all 1s)
            - single_qubit_gates: Maximum X gates (= n_features)
            - two_qubit_gates: Always 0 (no entanglement)
            - is_entangling: Always False (product states only)
            - simulability: Always "simulable" (trivial classical simulation)

        Notes
        -----
        **Important: Theoretical vs. Actual Gate Counts**

        The ``gate_count`` reported here is the **worst-case maximum**,
        assuming all input features binarize to 1. This is appropriate for:

        - Capacity planning and worst-case resource allocation
        - Comparing encodings on equal theoretical footing
        - Documentation and specifications

        For **actual** gate counts with specific input data, use:

        - ``actual_gate_count(x)``: Returns exact X gate count for input x
        - ``binarize(x)``: Inspect the binarized form of your data
        - ``resource_summary(x)``: Complete resource breakdown

        Example of the difference:

            >>> enc = BasisEncoding(n_features=8)
            >>> enc.properties.gate_count  # Theoretical maximum
            8
            >>> enc.actual_gate_count([1, 0, 0, 0, 0, 0, 0, 0])  # Actual
            1

        For sparse data, actual gate count can be significantly lower than
        the theoretical maximum, which impacts hardware execution time,
        error rates, and simulation costs.
        """
        n = self.n_features

        return EncodingProperties(
            n_qubits=n,
            depth=1,  # All X gates can be applied in parallel
            gate_count=n,  # WORST CASE: all features are 1
            single_qubit_gates=n,  # Maximum X gates (worst case)
            two_qubit_gates=0,  # No entangling gates ever
            parameter_count=0,  # No continuous parameters
            is_entangling=False,  # Product states only
            simulability="simulable",  # Trivially classically simulable
            trainability_estimate=1.0,  # No trainability issues (no parameters)
            notes=(
                f"GATE COUNTS ARE WORST-CASE (max {n} X gates if all features=1). "
                f"Actual gates depend on input data; use actual_gate_count(x) for "
                f"data-specific counts. Encodes binary data into computational "
                f"basis states |x⟩. Trivially classically simulable (product states). "
                f"Ideal for discrete optimization, Grover search, and QAOA."
            ),
        )

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return detailed string representation.

        Returns
        -------
        str
            String representation showing configuration parameters.
            Includes threshold only when non-default (not 0.5).

        Examples
        --------
        >>> enc = BasisEncoding(n_features=4)
        >>> repr(enc)
        'BasisEncoding(n_features=4)'

        >>> enc = BasisEncoding(n_features=4, threshold=0.0)
        >>> repr(enc)
        'BasisEncoding(n_features=4, threshold=0.0)'
        """
        # Only include threshold in repr if it differs from the default
        # This keeps the repr clean for typical usage while being explicit
        # when custom thresholds are used
        if self.threshold == _DEFAULT_BINARIZATION_THRESHOLD:
            return f"{self.__class__.__name__}(n_features={self.n_features})"
        else:
            return (
                f"{self.__class__.__name__}"
                f"(n_features={self.n_features}, threshold={self.threshold})"
            )
