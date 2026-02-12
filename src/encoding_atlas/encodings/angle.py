"""Angle-based quantum data encoding module.

This module implements AngleEncoding, a fundamental quantum data encoding
technique that maps classical features to rotation angles of single-qubit
gates. Each classical feature is encoded as a rotation angle on a dedicated
qubit, creating product states (no entanglement).

Angle encoding is one of the simplest and most widely used quantum data
encodings, offering several advantages:

1. **Simplicity**: Direct mapping from features to rotation angles
2. **Efficiency**: O(n) circuit depth for n features
3. **Trainability**: No barren plateau issues due to lack of entanglement
4. **Classical Simulability**: Product states can be efficiently simulated

The encoding creates quantum states of the form:

    |ψ(x)⟩ = ⊗ᵢ Rₐ(xᵢ)|0⟩

where Rₐ ∈ {RX, RY, RZ} is the rotation gate around axis a, and xᵢ is
the i-th feature value used as the rotation angle.

Mathematical Background
-----------------------
Single-qubit rotation gates are defined as:

    RX(θ) = exp(-i θ/2 X) = cos(θ/2)I - i·sin(θ/2)X
    RY(θ) = exp(-i θ/2 Y) = cos(θ/2)I - i·sin(θ/2)Y
    RZ(θ) = exp(-i θ/2 Z) = exp(-iθ/2)|0⟩⟨0| + exp(iθ/2)|1⟩⟨1|

For n features, the resulting quantum state is a tensor product:

    |ψ(x)⟩ = Rₐ(x₀)|0⟩ ⊗ Rₐ(x₁)|0⟩ ⊗ ... ⊗ Rₐ(xₙ₋₁)|0⟩

Note on Data Preprocessing
--------------------------
For optimal expressivity, input features should typically be scaled to
the range [0, 2π] or [-π, π]. Common preprocessing approaches include:

- Min-max scaling: x' = 2π · (x - x_min) / (x_max - x_min)
- Arctan scaling: x' = arctan(x) or x' = 2·arctan(x)
- Standard scaling followed by arctan: x' = arctan((x - μ) / σ)

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.angle').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters (n_features, rotation, reps)
- Circuit generation progress for each backend
- Batch processing statistics
- Input value range warnings (when values are outside typical [-2π, 2π] range)

Resource Analysis
-----------------
AngleEncoding has minimal resource requirements, making it ideal for
NISQ (Noisy Intermediate-Scale Quantum) devices:

- **Qubits**: n (one per feature)
- **Circuit depth**: reps (all rotations execute in parallel per layer)
- **Single-qubit gates**: n × reps (one rotation per qubit per layer)
- **Two-qubit gates**: 0 (no entanglement)

The ``gate_count_breakdown()`` and ``resource_summary()`` methods provide
programmatic access to these metrics:

    >>> enc = AngleEncoding(n_features=4, reps=2)
    >>> enc.gate_count_breakdown()
    {'rx': 0, 'ry': 8, 'rz': 0, 'total_single_qubit': 8, 'total_two_qubit': 0, 'total': 8}
    >>> enc.resource_summary()['hardware_requirements']
    {'connectivity': 'none', 'native_gates': ['RY']}

Use Cases
---------
AngleEncoding is particularly suited for:

- **Simple classification tasks**: Where the simplicity and trainability
  of product states is advantageous over complex entangling encodings
- **Baseline comparisons**: As a simple, entanglement-free baseline for
  benchmarking more complex encodings like IQP or ZZFeatureMap
- **NISQ hardware**: Where minimizing circuit depth is critical to
  reduce decoherence and gate errors
- **Classical simulation**: Where efficient classical simulation is needed
  for development, debugging, and hyperparameter tuning
- **Variational quantum classifiers**: As a trainable input layer with
  guaranteed gradient flow (no barren plateaus due to lack of entanglement)
- **Hybrid algorithms**: Combined with classical neural network layers
  in quantum-classical hybrid architectures

Limitations
-----------
- **Limited expressivity**: Product states cannot represent entangled
  quantum correlations, limiting the decision boundary complexity
- **Linear qubit scaling**: Requires n qubits for n features (unlike
  amplitude encoding which uses log₂(n) qubits)
- **No quantum advantage**: Classically simulable states provide no
  computational advantage over classical methods
- **Periodic rotations**: Features should typically be scaled to [0, 2π]
  or [-π, π] to fully utilize the rotation range without redundancy

References
----------
.. [1] Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
       Quantum Computers." Springer.
.. [2] LaRose, R., & Coyle, B. (2020). "Robust data encodings for quantum
       classifiers." Physical Review A, 102(3), 032420.
.. [3] Weigold, M., et al. (2020). "Data encoding patterns for quantum
       computing." IEEE International Conference on Quantum Computing
       and Engineering (QCE).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict

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
#   To enable debug logging for angle encoding:
#
#   >>> import logging
#   >>> logging.getLogger('encoding_atlas.encodings.angle').setLevel(logging.DEBUG)
#   >>> handler = logging.StreamHandler()
#   >>> handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
#   >>> logging.getLogger('encoding_atlas.encodings.angle').addHandler(handler)
#
# Log Levels:
#   - DEBUG: Detailed information for diagnosing issues (circuit generation,
#            batch processing progress, rotation gate selection)
#   - INFO: General operational information (circuit generation completed)
#   - WARNING: Potential issues that don't prevent operation
#
# The logger is named after the module to allow granular control over log output.
# By default, no handlers are attached, so no output is produced unless the
# application configures logging.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["AngleEncoding"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Threshold for emitting debug warnings about input value ranges.
# If input values exceed ±4π (approximately ±12.57), a debug log is emitted
# suggesting that the user may want to scale their features to [0, 2π] or
# [-π, π] for optimal encoding. The threshold of 4π accounts for full Bloch
# sphere coverage (4π for RZ phase accumulation) plus some margin.
_INPUT_RANGE_DEBUG_THRESHOLD: float = 4.0 * np.pi


# =============================================================================
# Type Definitions
# =============================================================================


class GateCountBreakdown(TypedDict):
    """Type definition for the gate_count_breakdown() return value.

    Provides a detailed breakdown of gate counts by type for circuit
    resource estimation and hardware compatibility analysis.
    """

    rx: int
    """Number of RX gates (non-zero only if rotation='X')."""

    ry: int
    """Number of RY gates (non-zero only if rotation='Y')."""

    rz: int
    """Number of RZ gates (non-zero only if rotation='Z')."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates (always 0 for AngleEncoding)."""

    total: int
    """Total gate count."""


class AngleEncoding(BaseEncoding):
    """Angle encoding using single-qubit rotation gates.

    AngleEncoding maps classical features directly to rotation angles of
    single-qubit gates. Each feature is encoded on a dedicated qubit using
    a rotation gate around a specified axis (X, Y, or Z).

    This encoding creates product states (no entanglement), making it:
    - Classically simulable (efficient classical simulation possible)
    - Free from barren plateaus (good trainability)
    - Hardware-efficient (only single-qubit gates)

    The circuit structure for each repetition is:

        |0⟩ ─ Rₐ(x₀) ─
        |0⟩ ─ Rₐ(x₁) ─
        |0⟩ ─ Rₐ(x₂) ─
        ...

    where Rₐ is the rotation gate for axis a ∈ {X, Y, Z}.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
        Each feature requires one qubit, so this also determines the number
        of qubits in the circuit.
    rotation : {"X", "Y", "Z"}, default="Y"
        The axis of rotation to use for encoding:

        - "X": Uses RX gates, encoding features in the YZ plane
        - "Y": Uses RY gates (default), encoding features in the XZ plane
        - "Z": Uses RZ gates, encoding features as phases

        The Y rotation is commonly used as it creates real-valued amplitudes
        when starting from |0⟩, and can represent any point on the Bloch
        sphere's XZ great circle.
    reps : int, default=1
        Number of times to repeat the encoding layer. Higher values increase
        the effective rotation angle by a factor of `reps`. Must be at least 1.

        With reps > 1, the effective rotation for feature xᵢ becomes:
        Rₐ(xᵢ)^reps = Rₐ(reps · xᵢ)

    Attributes
    ----------
    rotation : str
        The rotation axis used ("X", "Y", or "Z").
    reps : int
        Number of encoding layer repetitions.
    n_features : int
        Number of classical features (inherited from BaseEncoding).
    n_qubits : int
        Number of qubits, equal to n_features.

    Examples
    --------
    Create a basic angle encoding with default Y rotation:

    >>> from encoding_atlas import AngleEncoding
    >>> import numpy as np
    >>> enc = AngleEncoding(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.rotation
    'Y'

    Generate a PennyLane circuit:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Use X rotation for different encoding behavior:

    >>> enc_x = AngleEncoding(n_features=4, rotation='X')
    >>> qiskit_circuit = enc_x.get_circuit(x, backend='qiskit')
    >>> qiskit_circuit.num_qubits
    4

    Use multiple repetitions to increase rotation range:

    >>> enc_reps = AngleEncoding(n_features=2, reps=3)
    >>> enc_reps.depth
    3
    >>> enc_reps.properties.gate_count
    6

    Access encoding properties:

    >>> enc = AngleEncoding(n_features=4)
    >>> props = enc.properties
    >>> props.is_entangling
    False
    >>> props.simulability
    'simulable'

    References
    ----------
    .. [1] Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
           Quantum Computers." Springer.
    .. [2] LaRose, R., & Coyle, B. (2020). "Robust data encodings for
           quantum classifiers." Physical Review A, 102(3), 032420.

    See Also
    --------
    AmplitudeEncoding : Encodes features in state amplitudes (logarithmic qubits).
    IQPEncoding : Adds entanglement via ZZ interactions.
    DataReuploading : Re-uploads data multiple times with trainable layers.
    PauliFeatureMap : Configurable Pauli rotations with entanglement.

    Notes
    -----
    **Expressivity**: Angle encoding creates product states, which limits its
    expressivity compared to entangling encodings. However, this simplicity
    is advantageous for trainability and classical simulation.

    **Rotation Axis Choice**:

    - RY is most common as it creates real amplitudes from |0⟩
    - RX creates complex amplitudes with imaginary components
    - RZ only adds phases to |0⟩ (no population change from |0⟩)

    **Scaling Considerations**: The rotation gates have period 2π (for RX, RY)
    or 4π (for full Bloch sphere coverage). Input features should be scaled
    appropriately to utilize the full encoding range.

    **Comparison with Other Encodings**:

    +-------------------+----------+--------+------------+---------------+
    | Encoding          | Qubits   | Depth  | Entangling | Simulability  |
    +===================+==========+========+============+===============+
    | AngleEncoding     | n        | reps   | No         | Simulable     |
    +-------------------+----------+--------+------------+---------------+
    | BasisEncoding     | n        | 1      | No         | Simulable     |
    +-------------------+----------+--------+------------+---------------+
    | AmplitudeEncoding | log₂(n)  | O(2^n) | Yes        | Not simulable |
    +-------------------+----------+--------+------------+---------------+
    | IQPEncoding       | n        | O(n²)  | Yes        | Not simulable |
    +-------------------+----------+--------+------------+---------------+
    | ZZFeatureMap      | n        | O(n²)  | Yes        | Not simulable |
    +-------------------+----------+--------+------------+---------------+
    """

    # ==========================================================================
    # Class Constants
    # ==========================================================================

    # Valid rotation axes
    _VALID_ROTATIONS: frozenset[str] = frozenset({"X", "Y", "Z"})

    # Memory-efficient slot-based attribute storage
    __slots__ = ("rotation", "reps")

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def __init__(
        self,
        n_features: int,
        rotation: Literal["X", "Y", "Z"] = "Y",
        reps: int = 1,
    ) -> None:
        """Initialize the angle encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        rotation : {"X", "Y", "Z"}, default="Y"
            Rotation axis for the encoding gates.
        reps : int, default=1
            Number of encoding layer repetitions.

        Raises
        ------
        ValueError
            If rotation is not one of "X", "Y", "Z".
        ValueError
            If reps is less than 1.
        ValueError
            If n_features is less than 1 (raised by parent class).
        """
        # Validate rotation axis
        if rotation not in self._VALID_ROTATIONS:
            raise ValueError(
                f"rotation must be one of {sorted(self._VALID_ROTATIONS)}, "
                f"got {rotation!r}"
            )

        # Validate repetitions
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
            raise ValueError(f"reps must be a positive integer, got {reps!r}")

        # Call parent constructor
        super().__init__(n_features, rotation=rotation, reps=reps)

        # Store encoding-specific parameters
        self.rotation: Literal["X", "Y", "Z"] = rotation
        self.reps: int = reps

        # Log initialization
        _logger.debug(
            "AngleEncoding initialized: n_features=%d, rotation=%r, reps=%d",
            n_features,
            rotation,
            reps,
        )

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        For angle encoding, each feature is encoded on a dedicated qubit,
        so n_qubits equals n_features.

        Returns
        -------
        int
            Number of qubits, equal to n_features.
        """
        return self._n_features

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        The depth equals the number of repetitions, as each repetition
        consists of a single layer of parallel single-qubit rotations.

        Returns
        -------
        int
            Circuit depth, equal to reps.
        """
        return self.reps

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
        rotation angles applied to qubits initialized in the |0⟩ state.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (1, n_features).
            Values are used directly as rotation angles (in radians).
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
            If backend is not recognized.

        Examples
        --------
        >>> enc = AngleEncoding(n_features=4)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> circuit = enc.get_circuit(x, backend='pennylane')
        >>> callable(circuit)
        True

        >>> qiskit_circuit = enc.get_circuit(x, backend='qiskit')
        >>> qiskit_circuit.num_qubits
        4
        """
        _logger.debug(
            "Generating circuit: backend=%r, input_shape=%s",
            backend,
            getattr(x, "shape", f"len={len(x)}"),
        )

        # Validate and preprocess input
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]

        # Debug logging for input value range
        # This helps users identify when feature scaling may be needed
        if _logger.isEnabledFor(logging.DEBUG):
            x_min, x_max = float(x_validated.min()), float(x_validated.max())
            if (
                abs(x_min) > _INPUT_RANGE_DEBUG_THRESHOLD
                or abs(x_max) > _INPUT_RANGE_DEBUG_THRESHOLD
            ):
                _logger.debug(
                    "Input values [%.3g, %.3g] are outside typical range "
                    "[-2π, 2π]. Rotation gates are periodic with period 2π "
                    "(RX, RY) or 4π (full Bloch). Consider scaling features "
                    "to [0, 2π] or [-π, π] for optimal encoding.",
                    x_min,
                    x_max,
                )

        # Dispatch to backend-specific implementation
        if backend == "pennylane":
            circuit = self._to_pennylane(x_validated)
        elif backend == "qiskit":
            circuit = self._to_qiskit(x_validated)
        elif backend == "cirq":
            circuit = self._to_cirq(x_validated)
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
            )

        _logger.debug("Circuit generated successfully for backend=%r", backend)
        return circuit

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
            circuit generation. This can speed up processing for large
            batches (>100 samples) but adds overhead for small batches.

            Parallel processing is thread-safe because AngleEncoding's circuit
            generation is stateless (no instance mutation occurs).
        max_workers : int or None, default=None
            Maximum number of worker threads for parallel processing.
            Only used when ``parallel=True``. If None, uses the default
            from ThreadPoolExecutor (typically min(32, cpu_count + 4)).

            For CPU-bound workloads, set to ``os.cpu_count()``.
            For I/O-bound workloads, higher values may help.

        Returns
        -------
        list[CircuitType]
            List of circuits, one per sample. Order is preserved even
            when using parallel processing.

        Examples
        --------
        Sequential processing (default):

        >>> enc = AngleEncoding(n_features=4)
        >>> X = np.random.randn(10, 4)
        >>> circuits = enc.get_circuits(X, backend='pennylane')
        >>> len(circuits)
        10

        Parallel processing for large batches:

        >>> enc = AngleEncoding(n_features=4)
        >>> X_large = np.random.randn(1000, 4)
        >>> circuits = enc.get_circuits(X_large, backend='qiskit', parallel=True)
        >>> len(circuits)
        1000

        Custom worker count:

        >>> import os
        >>> circuits = enc.get_circuits(
        ...     X_large, backend='cirq', parallel=True, max_workers=os.cpu_count()
        ... )

        Notes
        -----
        **When to use parallel processing:**

        - Large batches (>100 samples): Parallel processing overhead is
          amortized across many samples.
        - Qiskit/Cirq backends: These create full circuit objects which
          has more overhead than PennyLane's lightweight closures.

        **When to use sequential processing:**

        - Small batches (<100 samples): Overhead of thread pool management
          may exceed the benefit.
        - PennyLane backend: Circuit generation is extremely fast.
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
        """
        X_validated = self._validate_input(X)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)

        n_samples = X_validated.shape[0]

        _logger.debug(
            "Batch circuit generation: n_samples=%d, backend=%r, parallel=%s, "
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
                "Parallel batch generation completed: %d circuits using "
                "ThreadPoolExecutor",
                len(circuits),
            )
        else:
            # Sequential processing using internal method to avoid re-validation
            # The batch was already validated above, so we can safely skip
            # per-sample validation for better performance
            circuits = [
                self._get_circuit_from_validated(x, backend) for x in X_validated
            ]

            _logger.debug(
                "Sequential batch generation completed: %d circuits",
                len(circuits),
            )

        return circuits

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input (internal use only).

        This method skips input validation, assuming the caller has already
        validated the input. Used by get_circuits() to avoid double validation
        when processing batches, improving performance for large datasets.

        Parameters
        ----------
        x : NDArray
            Pre-validated input features of shape (n_features,).
            Must be a 1D array with exactly n_features elements.
            Must not contain NaN or infinite values.
        backend : BackendType
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.

        Raises
        ------
        ValueError
            If backend is not one of the supported options.

        Notes
        -----
        **Internal Method**: This method is not part of the public API.
        External callers should use ``get_circuit()`` which includes
        full input validation.

        **Thread Safety**: This method is thread-safe. It does not modify
        any instance state and operates only on the provided input array.

        **Performance**: By skipping validation, this method is faster than
        ``get_circuit()`` for batch processing where the entire batch has
        already been validated once. The performance gain is proportional
        to the batch size.
        """
        # Handle 2D input (single sample as row)
        if x.ndim == 2:
            x = x[0]

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

        Creates a callable that applies the angle encoding gates when
        invoked within a PennyLane QNode context.

        Parameters
        ----------
        x : NDArray
            Validated input features of shape (n_features,).

        Returns
        -------
        callable
            Function that applies the encoding gates when called.

        Raises
        ------
        ImportError
            If PennyLane is not installed.
        """
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(
                "PennyLane is required for the 'pennylane' backend. "
                "Install it with: pip install pennylane"
            ) from e

        # Capture parameters for closure
        n_qubits = self.n_qubits
        reps = self.reps
        rotation = self.rotation

        # Select rotation gate
        rotation_gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[rotation]

        def circuit() -> None:
            """Apply the angle encoding gates."""
            for _ in range(reps):
                for i in range(n_qubits):
                    rotation_gate(x[i], wires=i)

        return circuit

    def _to_qiskit(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Qiskit QuantumCircuit.

        Parameters
        ----------
        x : NDArray
            Validated input features of shape (n_features,).

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

        qc = QuantumCircuit(self.n_qubits, name="AngleEncoding")

        # Select rotation method
        rotation_method = {"X": qc.rx, "Y": qc.ry, "Z": qc.rz}[self.rotation]

        for _ in range(self.reps):
            for i in range(self.n_qubits):
                rotation_method(float(x[i]), i)

        return qc

    def _to_cirq(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Cirq Circuit.

        Parameters
        ----------
        x : NDArray
            Validated input features of shape (n_features,).

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

        # Select rotation gate factory
        rotation_gate = {"X": cirq.rx, "Y": cirq.ry, "Z": cirq.rz}[self.rotation]

        moments = []
        for _ in range(self.reps):
            # Create a moment with all rotations in parallel
            gates = [
                rotation_gate(float(x[i]))(qubits[i]) for i in range(self.n_qubits)
            ]
            moments.append(cirq.Moment(gates))

        return cirq.Circuit(moments)

    # ==========================================================================
    # Properties Computation
    # ==========================================================================

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Calculates circuit metrics including qubit count, depth, gate
        counts, and encoding characteristics.

        Returns
        -------
        EncodingProperties
            Computed properties including:
            - n_qubits: Number of qubits (= n_features)
            - depth: Circuit depth (= reps)
            - gate_count: Total gates (= n_features * reps)
            - is_entangling: Always False (product states)
            - simulability: Always "simulable" (classical simulation possible)
        """
        n = self._n_features
        total_gates = n * self.reps

        return EncodingProperties(
            n_qubits=n,
            depth=self.reps,
            gate_count=total_gates,
            single_qubit_gates=total_gates,
            two_qubit_gates=0,
            parameter_count=total_gates,  # Each gate has one data-dependent parameter
            is_entangling=False,
            simulability="simulable",  # Product states are classically simulable
            trainability_estimate=0.9,  # High trainability (no barren plateaus)
            notes=(
                f"Rotation axis: {self.rotation}, "
                f"Creates product states only (no entanglement). "
                f"Classically simulable with O(n) complexity."
            ),
        )

    # ==========================================================================
    # Resource Analysis
    # ==========================================================================

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type.

        Provides a comprehensive breakdown of gates used in this encoding,
        enabling hardware resource estimation and comparison with other
        encodings. This method is useful for:

        - Hardware compatibility analysis (checking native gate support)
        - Resource estimation for quantum hardware budgeting
        - Programmatic comparison of different encoding configurations

        Returns
        -------
        GateCountBreakdown
            Dictionary with gate counts:

            - ``'rx'``: Number of RX gates (non-zero only if rotation='X')
            - ``'ry'``: Number of RY gates (non-zero only if rotation='Y')
            - ``'rz'``: Number of RZ gates (non-zero only if rotation='Z')
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Always 0 (no entanglement)
            - ``'total'``: Total gate count

        Examples
        --------
        >>> enc = AngleEncoding(n_features=4, rotation='Y', reps=2)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['ry']
        8
        >>> breakdown['total_single_qubit']
        8
        >>> breakdown['total_two_qubit']
        0

        >>> enc_x = AngleEncoding(n_features=4, rotation='X', reps=1)
        >>> enc_x.gate_count_breakdown()['rx']
        4

        See Also
        --------
        resource_summary : Comprehensive resource analysis including
            hardware requirements and encoding characteristics.
        properties : Basic encoding properties via EncodingProperties object.
        """
        n = self.n_features
        total = n * self.reps

        rx = total if self.rotation == "X" else 0
        ry = total if self.rotation == "Y" else 0
        rz = total if self.rotation == "Z" else 0

        return GateCountBreakdown(
            rx=rx,
            ry=ry,
            rz=rz,
            total_single_qubit=total,
            total_two_qubit=0,
            total=total,
        )

    def resource_summary(self) -> dict[str, Any]:
        """Generate a comprehensive resource summary for this encoding.

        Provides a complete view of circuit resources, encoding characteristics,
        and hardware requirements in a single call. This method is useful for:

        - Generating reports comparing different encoding configurations
        - Hardware planning and resource allocation
        - Documenting encoding choices in experiments

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            **Circuit Structure**:
                - ``n_qubits``: Number of qubits
                - ``n_features``: Number of classical features encoded
                - ``depth``: Circuit depth
                - ``reps``: Number of encoding repetitions
                - ``rotation``: Rotation axis ('X', 'Y', or 'Z')

            **Gate Counts**:
                - ``gate_counts``: Full breakdown from ``gate_count_breakdown()``

            **Encoding Characteristics**:
                - ``is_entangling``: Always False for angle encoding
                - ``simulability``: Always 'simulable' for product states
                - ``trainability_estimate``: Trainability score (0.0-1.0)

            **Hardware Requirements**:
                - ``connectivity``: 'none' (no two-qubit gates required)
                - ``native_gates``: List of required gate types

        Examples
        --------
        >>> enc = AngleEncoding(n_features=4, rotation='Y', reps=2)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['depth']
        2
        >>> summary['is_entangling']
        False
        >>> summary['hardware_requirements']['connectivity']
        'none'
        >>> summary['hardware_requirements']['native_gates']
        ['RY']

        See Also
        --------
        gate_count_breakdown : Detailed gate count breakdown.
        properties : Basic encoding properties.
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties

        return {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            "rotation": self.rotation,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": False,
            "simulability": "simulable",
            "trainability_estimate": props.trainability_estimate,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": "none",  # No two-qubit gates needed
                "native_gates": [f"R{self.rotation}"],
            },
        }

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
            f"rotation={self.rotation!r}, "
            f"reps={self.reps})"
        )
