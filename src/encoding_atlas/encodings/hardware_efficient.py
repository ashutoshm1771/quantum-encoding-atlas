"""Hardware-efficient encoding module.

This module implements HardwareEfficientEncoding, a quantum data encoding
technique designed to match the native gate sets and connectivity constraints
of near-term quantum hardware. By using only gates that are native to the
target device and respecting qubit connectivity, this encoding minimizes
gate decomposition overhead and reduces circuit depth.

Hardware-efficient encoding is particularly important for NISQ devices, offering:

1. **Native Gate Compatibility**: Uses standard single-qubit rotations and
   CNOT gates available on most quantum hardware
2. **Connectivity-Aware**: Matches entanglement topology to physical qubit
   layouts (linear, circular)
3. **Reduced Compilation Overhead**: Circuits require minimal or no additional
   gate decomposition
4. **Error Mitigation**: Shorter circuits accumulate less noise from gate errors

The encoding creates quantum states using:

    |ψ(x)⟩ = [U_ent · U_rot(x)]^reps |0⟩^⊗n

where U_rot applies data-dependent rotations and U_ent provides entanglement
using connectivity-respecting CNOT gates.

Mathematical Background
-----------------------
Hardware-efficient ansätze were introduced to address the practical
constraints of quantum hardware. The key considerations are:

1. **Gate Fidelity**: Native gates (RX, RY, RZ, CNOT) have higher fidelity
   than composite gates requiring decomposition
2. **Connectivity**: Physical qubit connectivity limits which two-qubit gates
   can be applied directly. Linear and circular topologies are common
3. **Depth-Error Trade-off**: Deeper circuits are more expressive but
   accumulate more errors

The circuit structure alternates between:
- **Rotation Layer**: R_α(xᵢ) for each qubit, where α ∈ {X, Y, Z}
- **Entanglement Layer**: CNOT gates following the connectivity pattern

Note on Data Preprocessing
--------------------------
For hardware-efficient encoding, input features are typically:

- Scaled to [0, 2π] or [-π, π] for rotation gates
- Features should be standardized if they have very different magnitudes
- The choice of rotation axis (X, Y, Z) may affect which preprocessing
  is optimal for a given task

The simpler structure compared to IQP or ZZ feature maps makes this encoding
more robust to noise, though potentially less expressive.

Use Cases
---------
HardwareEfficientEncoding is particularly suited for:

- **NISQ hardware deployment**: Minimizes gate errors by using native gates
  and respecting physical connectivity constraints
- **Variational quantum algorithms**: Natural choice for VQE, QAOA, and
  quantum neural networks due to hardware compatibility
- **Error-sensitive applications**: When gate fidelity is critical, the
  reduced compilation overhead preserves quantum coherence
- **Rapid prototyping**: Simple structure allows quick iteration on
  circuit designs without complex transpilation
- **Superconducting qubits**: Linear and circular topologies match common
  chip architectures (IBM, Google, Rigetti)
- **Ion trap devices**: Full entanglement topology matches all-to-all
  connectivity available on IonQ, Quantinuum, and similar platforms

Limitations
-----------
- **Limited expressivity**: Shallow hardware-efficient circuits may not
  capture all relevant quantum correlations for complex tasks
- **Topology constraints**: Circular entanglement requires wrap-around
  connectivity; full entanglement requires all-to-all connectivity
  (available on ion traps but not most superconducting devices)
- **Full entanglement scaling**: Full topology has O(n²) two-qubit gates
  per layer, which may be prohibitive for large qubit counts
- **Barren plateaus**: Deep hardware-efficient circuits can exhibit
  vanishing gradients, limiting trainability
- **Feature scaling sensitivity**: Performance depends on appropriate
  preprocessing of input features to rotation angle ranges

Resource Analysis
-----------------
HardwareEfficientEncoding has predictable resource requirements:

- **Qubits**: n (one per feature)
- **Circuit depth**: 2 × reps (rotation + entanglement layers)
- **Single-qubit gates**: n × reps
- **Two-qubit gates**:
  - Linear: (n-1) × reps
  - Circular: n × reps
  - Full: n(n-1)/2 × reps (O(n²) scaling)

The ``gate_count_breakdown()`` and ``resource_summary()`` methods provide
programmatic access to these metrics:

    >>> enc = HardwareEfficientEncoding(n_features=4, reps=2)
    >>> enc.gate_count_breakdown()
    {'rx': 0, 'ry': 8, 'rz': 0, 'cnot': 6, 'total_single_qubit': 8, ...}
    >>> enc.resource_summary()['hardware_requirements']
    {'connectivity': 'linear', 'native_gates': ['RY', 'CNOT']}

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.hardware_efficient').setLevel(
    ...     logging.DEBUG
    ... )

Debug logs include:

- Initialization parameters (n_features, reps, rotation, entanglement)
- Circuit generation progress for each backend
- Batch processing statistics
- Entanglement pair computation
- Input value range warnings (when values are outside typical [-2π, 2π] range)

References
----------
.. [1] Kandala, A., et al. (2017). "Hardware-efficient variational quantum
       eigensolver for small molecules and quantum magnets." Nature,
       549(7671), 242-246.
.. [2] Cerezo, M., et al. (2021). "Variational quantum algorithms."
       Nature Reviews Physics, 3(9), 625-644.
.. [3] Bharti, K., et al. (2022). "Noisy intermediate-scale quantum algorithms."
       Reviews of Modern Physics, 94(1), 015004.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

# =============================================================================
# Module-Level Logger
# =============================================================================

# Configure module-level logger for production debugging.
#
# Usage:
#   To enable debug logging for hardware-efficient encoding:
#
#   >>> import logging
#   >>> logging.getLogger('encoding_atlas.encodings.hardware_efficient').setLevel(
#   ...     logging.DEBUG
#   ... )
#   >>> handler = logging.StreamHandler()
#   >>> handler.setFormatter(
#   ...     logging.Formatter('%(name)s - %(levelname)s - %(message)s')
#   ... )
#   >>> logging.getLogger('encoding_atlas.encodings.hardware_efficient').addHandler(
#   ...     handler
#   ... )
#
# Log Levels:
#   - DEBUG: Detailed information for diagnosing issues (circuit generation,
#            batch processing progress, entanglement pair computation)
#   - INFO: General operational information (circuit generation completed)
#   - WARNING: Potential issues like large qubit counts with full entanglement
#
# The logger is named after the module to allow granular control over log output.
# By default, no handlers are attached, so no output is produced unless the
# application configures logging.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["HardwareEfficientEncoding"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Threshold for emitting debug warnings about input value ranges.
# If input values exceed ±4π (approximately ±12.57), a debug log is emitted
# suggesting that the user may want to scale their features to [0, 2π] or
# [-π, π] for optimal encoding.
_INPUT_RANGE_DEBUG_THRESHOLD: float = 4.0 * np.pi

# Threshold for emitting warnings about large qubit counts with circular
# entanglement. Circular topology requires all-to-all connectivity which
# may not be available on hardware with >20 qubits.
_LARGE_QUBIT_WARNING_THRESHOLD: int = 20


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

    cnot: int
    """Number of CNOT (CX) entangling gates."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates."""

    total: int
    """Total gate count."""


class HardwareEfficientEncoding(BaseEncoding):
    """Hardware-efficient encoding optimized for NISQ devices.

    HardwareEfficientEncoding implements a quantum data encoding that uses
    only native gates and respects physical qubit connectivity constraints.
    This makes it particularly suitable for near-term quantum devices where
    gate errors and limited connectivity are significant concerns.

    The encoding uses simple alternating layers of single-qubit rotations
    (parameterized by input features) and entangling CNOT gates following
    a linear or circular topology.

    The circuit structure for each repetition is:

        |0⟩ ─ Rₐ(x₀) ─╭────╮─ Rₐ(x₀) ─╭────╮─
        |0⟩ ─ Rₐ(x₁) ─│CNOT│─ Rₐ(x₁) ─│CNOT│─
        |0⟩ ─ Rₐ(x₂) ─│    │─ Rₐ(x₂) ─│    │─
        ...           ╰────╯          ╰────╯

    where Rₐ is the chosen rotation gate (RX, RY, or RZ) and CNOT gates
    connect neighboring qubits according to the entanglement topology.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
        Each feature requires one qubit, so this also determines the number
        of qubits in the circuit.
    reps : int, default=2
        Number of times to repeat the encoding layers. Higher values increase
        circuit expressivity but also depth and accumulated errors.
        Must be at least 1.
    rotation : {"X", "Y", "Z"}, default="Y"
        Axis of rotation for single-qubit gates:

        - "X": Uses RX gates, commonly native on superconducting qubits
        - "Y": Uses RY gates (default), creates real-valued amplitudes
        - "Z": Uses RZ gates, often "virtual" gates with zero error

        The optimal choice depends on the target hardware's native gate set.
    entanglement : {"linear", "circular", "full"}, default="linear"
        Topology of CNOT entangling gates:

        - "linear": Nearest-neighbor connectivity. CNOT gates connect pairs
          (0,1), (1,2), ..., (n-2, n-1). Creates n-1 entangling gates per layer.
          Matches linear qubit arrays common in superconducting devices.
        - "circular": Linear connectivity plus periodic boundary. Adds a CNOT
          between qubits n-1 and 0. Creates n entangling gates per layer.
          Matches circular/ring qubit topologies.
        - "full": All-to-all connectivity. CNOT gates connect every pair of
          qubits. Creates n(n-1)/2 entangling gates per layer (O(n²) scaling).
          Matches ion trap devices (IonQ, Quantinuum) with global connectivity.

    Attributes
    ----------
    reps : int
        Number of encoding layer repetitions.
    rotation : str
        The rotation axis ("X", "Y", or "Z").
    entanglement : str
        The entanglement topology ("linear", "circular", or "full").
    n_features : int
        Number of classical features (inherited from BaseEncoding).
    n_qubits : int
        Number of qubits, equal to n_features.

    Examples
    --------
    Create a basic hardware-efficient encoding:

    >>> from encoding_atlas import HardwareEfficientEncoding
    >>> import numpy as np
    >>> enc = HardwareEfficientEncoding(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.rotation
    'Y'

    Generate a PennyLane circuit:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Use RZ rotations (often "virtual" gates with zero error):

    >>> enc_rz = HardwareEfficientEncoding(n_features=4, rotation='Z')
    >>> enc_rz.rotation
    'Z'

    Use circular entanglement for ring topologies:

    >>> enc_circ = HardwareEfficientEncoding(n_features=4, entanglement='circular')
    >>> enc_circ.properties.two_qubit_gates
    8

    Generate circuits for different backends:

    >>> qiskit_circuit = enc.get_circuit(x, backend='qiskit')
    >>> qiskit_circuit.num_qubits
    4
    >>> cirq_circuit = enc.get_circuit(x, backend='cirq')
    >>> len(cirq_circuit.all_qubits())
    4

    Access encoding properties:

    >>> props = enc.properties
    >>> props.is_entangling
    True
    >>> props.trainability_estimate
    0.8

    References
    ----------
    .. [1] Kandala, A., et al. (2017). "Hardware-efficient variational quantum
           eigensolver for small molecules and quantum magnets." Nature.
    .. [2] Cerezo, M., et al. (2021). "Variational quantum algorithms."
           Nature Reviews Physics.

    See Also
    --------
    AngleEncoding : Simpler encoding without entanglement.
    DataReuploading : Re-uploads data for universal approximation.
    IQPEncoding : Higher expressivity with ZZ interactions.
    ZZFeatureMap : Qiskit-style feature map encoding.

    Notes
    -----
    **Hardware Native Gates**: Most superconducting quantum computers support
    RX, RY, RZ rotations and CNOT (or CZ) as native gates. Ion trap devices
    typically support different gate sets. Choose the rotation type that
    matches your target hardware.

    **Virtual RZ Gates**: On many superconducting devices, RZ gates are
    implemented virtually (by updating the phase tracking) with essentially
    zero error. Using rotation='Z' can significantly reduce circuit errors.

    **Depth Considerations**: Each repetition adds one rotation layer and one
    entanglement layer. For n qubits with linear entanglement:
    - Total single-qubit gates: reps × n
    - Total two-qubit gates: reps × (n-1)
    - Circuit depth: approximately 2 × reps

    **Expressivity vs. Trainability**: Hardware-efficient encodings typically
    have good trainability (gradients don't vanish as quickly as random
    circuits) but may have lower expressivity than IQP-style encodings.
    This trade-off often favors hardware-efficient designs on NISQ devices.
    """

    # Valid rotation axes and entanglement topologies
    _VALID_ROTATIONS: frozenset[str] = frozenset({"X", "Y", "Z"})
    _VALID_ENTANGLEMENTS: frozenset[str] = frozenset({"linear", "circular", "full"})

    __slots__ = ("reps", "rotation", "entanglement", "_entanglement_pairs")

    def __init__(
        self,
        n_features: int,
        reps: int = 2,
        rotation: Literal["X", "Y", "Z"] = "Y",
        entanglement: Literal["linear", "circular", "full"] = "linear",
    ) -> None:
        """Initialize the hardware-efficient encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        reps : int, default=2
            Number of encoding layer repetitions.
        rotation : {"X", "Y", "Z"}, default="Y"
            Rotation axis for single-qubit gates.
        entanglement : {"linear", "circular", "full"}, default="linear"
            Topology for entangling CNOT gates.

        Raises
        ------
        ValueError
            If reps is less than 1.
        ValueError
            If rotation is not one of "X", "Y", "Z".
        ValueError
            If entanglement is not one of "linear", "circular", "full".
        ValueError
            If n_features is less than 1 (raised by parent class).
        """
        # Validate reps
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
            raise ValueError(f"reps must be a positive integer, got {reps!r}")

        # Validate rotation
        if rotation not in self._VALID_ROTATIONS:
            raise ValueError(
                f"rotation must be one of {sorted(self._VALID_ROTATIONS)}, "
                f"got {rotation!r}"
            )

        # Validate entanglement
        if entanglement not in self._VALID_ENTANGLEMENTS:
            raise ValueError(
                f"entanglement must be one of {sorted(self._VALID_ENTANGLEMENTS)}, "
                f"got {entanglement!r}"
            )

        super().__init__(
            n_features, reps=reps, rotation=rotation, entanglement=entanglement
        )
        self.reps: int = reps
        self.rotation: Literal["X", "Y", "Z"] = rotation
        self.entanglement: Literal["linear", "circular", "full"] = entanglement

        # Cache entanglement pairs at initialization for performance
        # This avoids recomputation on every circuit generation
        self._entanglement_pairs: list[tuple[int, int]] = (
            self._compute_entanglement_pairs()
        )

        # Warn for large qubit counts with non-linear entanglement topologies
        # Circular topology requires wrap-around connectivity
        # Full topology requires all-to-all connectivity (quadratic gate scaling)
        if n_features > _LARGE_QUBIT_WARNING_THRESHOLD:
            if entanglement == "circular":
                warnings.warn(
                    f"Using circular entanglement with {n_features} qubits. "
                    f"Circular topology requires wrap-around connectivity which "
                    f"may not be available on all quantum hardware. Consider "
                    f"using 'linear' entanglement for better hardware compatibility.",
                    UserWarning,
                    stacklevel=2,
                )
                _logger.warning(
                    "Large qubit count (%d) with circular entanglement may have "
                    "hardware compatibility issues",
                    n_features,
                )
            elif entanglement == "full":
                n_pairs = n_features * (n_features - 1) // 2
                warnings.warn(
                    f"Using full entanglement with {n_features} qubits creates "
                    f"{n_pairs} CNOT pairs per layer (O(n²) scaling). This requires "
                    f"all-to-all qubit connectivity, available on ion trap devices "
                    f"but not most superconducting hardware. Circuit depth and gate "
                    f"count grow quadratically with qubit count.",
                    UserWarning,
                    stacklevel=2,
                )
                _logger.warning(
                    "Large qubit count (%d) with full entanglement: %d CNOT pairs "
                    "per layer (O(n²) scaling)",
                    n_features,
                    n_pairs,
                )

        # Log initialization
        _logger.debug(
            "HardwareEfficientEncoding initialized: n_features=%d, reps=%d, "
            "rotation=%r, entanglement=%r, n_entanglement_pairs=%d",
            n_features,
            reps,
            rotation,
            entanglement,
            len(self._entanglement_pairs),
        )

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        For hardware-efficient encoding, each feature is encoded on a
        dedicated qubit, so n_qubits equals n_features.

        Returns
        -------
        int
            Number of qubits, equal to n_features.
        """
        return self.n_features

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        Each repetition consists of one rotation layer and one entanglement
        layer, contributing depth 2 per repetition.

        Returns
        -------
        int
            Circuit depth, equal to 2 * reps.
        """
        return self.reps * 2

    def _compute_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Compute qubit pairs for CNOT entanglement based on topology.

        This method performs the actual computation of entanglement pairs.
        Results are cached in ``_entanglement_pairs`` at initialization.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for CNOT gates.

        Notes
        -----
        - Linear: n-1 pairs (nearest neighbors)
        - Circular: n pairs (nearest neighbors + wrap-around)
        - Full: n(n-1)/2 pairs (all-to-all connectivity)
        """
        n = self.n_qubits
        if self.entanglement == "linear":
            pairs = [(i, i + 1) for i in range(n - 1)]
        elif self.entanglement == "circular":
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs.append((n - 1, 0))
        else:  # full
            # All-to-all connectivity: every qubit pair connected
            # Total pairs: n(n-1)/2 (each pair appears once with i < j)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        _logger.debug(
            "Computed entanglement pairs for %s topology: %d pairs",
            self.entanglement,
            len(pairs),
        )
        return pairs

    def _get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get cached qubit pairs for CNOT entanglement.

        Returns the pre-computed entanglement pairs cached at initialization.
        This method provides backward compatibility and is used internally.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for CNOT gates.

        See Also
        --------
        get_entanglement_pairs : Public method for accessing entanglement pairs.
        """
        return self._entanglement_pairs

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get the qubit pairs used for CNOT entanglement.

        Returns the entanglement pairs based on the configured topology.
        This is useful for analyzing circuit structure, debugging, or
        implementing custom circuit modifications.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for CNOT gates.
            For linear entanglement with n qubits: [(0,1), (1,2), ..., (n-2,n-1)]
            For circular entanglement: adds (n-1, 0) wrap-around pair.

        Examples
        --------
        >>> enc = HardwareEfficientEncoding(n_features=4, entanglement='linear')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        >>> enc_circ = HardwareEfficientEncoding(n_features=4, entanglement='circular')
        >>> enc_circ.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3), (3, 0)]

        >>> enc_full = HardwareEfficientEncoding(n_features=4, entanglement='full')
        >>> enc_full.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        Notes
        -----
        The returned list is the cached copy computed at initialization.
        Modifying the returned list will not affect the encoding's behavior.
        """
        # Return a copy to prevent external modification
        return list(self._entanglement_pairs)

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input.

        Encoding-specific seam called by the inherited ``get_circuit``/
        ``get_circuits`` template methods. Emits a debug log when input
        values fall outside the typical [-2π, 2π] range and dispatches to
        the backend-specific implementation.

        Accepts a 1D array (the normal case) or a 2D single-sample
        ``(1, n_features)`` array for robustness when called directly.
        """
        # Defensive 2D-to-1D handling for direct callers; the inherited
        # template methods already pass 1D input.
        if x.ndim == 2:
            x = x[0]

        if _logger.isEnabledFor(logging.DEBUG):
            x_min, x_max = float(x.min()), float(x.max())
            if (
                abs(x_min) > _INPUT_RANGE_DEBUG_THRESHOLD
                or abs(x_max) > _INPUT_RANGE_DEBUG_THRESHOLD
            ):
                _logger.debug(
                    "Input values [%.3g, %.3g] are outside typical range "
                    "[-2π, 2π]. Rotation gates are periodic with period 2π. "
                    "Consider scaling features to [0, 2π] or [-π, π] for "
                    "optimal encoding.",
                    x_min,
                    x_max,
                )

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

    def _to_pennylane(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate PennyLane circuit function.

        Creates a callable that applies the hardware-efficient encoding
        when invoked within a PennyLane QNode context.

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

        pairs = self._get_entanglement_pairs()
        rot_gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[self.rotation]
        reps = self.reps

        def circuit() -> None:
            """Apply the hardware-efficient encoding gates."""
            for _ in range(reps):
                # Rotation layer - data encoding
                for i, val in enumerate(x):
                    rot_gate(val, wires=i)
                # Entangling layer - CNOT gates
                for i, j in pairs:
                    qml.CNOT(wires=[i, j])

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

        pairs = self._get_entanglement_pairs()
        qc = QuantumCircuit(self.n_qubits, name="HardwareEfficientEncoding")
        rot_method = {"X": "rx", "Y": "ry", "Z": "rz"}[self.rotation]

        for _ in range(self.reps):
            # Rotation layer
            for i, val in enumerate(x):
                getattr(qc, rot_method)(float(val), i)
            # Entangling layer
            for i, j in pairs:
                qc.cx(i, j)

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
            Cirq circuit implementing the hardware-efficient encoding.

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
        pairs = self._get_entanglement_pairs()
        circuit = cirq.Circuit()

        # Select rotation gate factory
        rotation_gates = {
            "X": cirq.rx,
            "Y": cirq.ry,
            "Z": cirq.rz,
        }
        rot_gate = rotation_gates[self.rotation]

        for _ in range(self.reps):
            # Rotation layer
            for i, val in enumerate(x):
                circuit.append(rot_gate(float(val))(qubits[i]))

            # Entangling layer
            for i, j in pairs:
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))

        return circuit

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Returns
        -------
        EncodingProperties
            Computed properties including:
            - n_qubits: Number of qubits (= n_features)
            - depth: Circuit depth (= 2 * reps)
            - gate_count: Total gates including rotations and CNOT
            - is_entangling: True when n_qubits > 1
            - simulability: Depends on circuit size
        """
        n = self.n_qubits
        n_pairs = len(self._get_entanglement_pairs())

        # Gates per rep: n rotations + n_pairs CNOTs
        rotation_gates = self.reps * n
        cnot_gates = self.reps * n_pairs

        return EncodingProperties(
            n_qubits=n,
            depth=self.reps * 2,
            gate_count=rotation_gates + cnot_gates,
            single_qubit_gates=rotation_gates,
            two_qubit_gates=cnot_gates,
            parameter_count=rotation_gates,  # Each rotation uses one data parameter
            is_entangling=(n > 1 and n_pairs > 0),
            simulability="not_simulable" if n > 1 else "simulable",
            trainability_estimate=0.8,  # Good trainability due to simple structure
            notes=(
                f"Hardware-efficient with {self.rotation} rotations and "
                f"{self.entanglement} entanglement. Optimized for NISQ devices."
            ),
        )

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
            - ``'cnot'``: Number of CNOT entangling gates
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Total two-qubit gates
            - ``'total'``: Total gate count

        Examples
        --------
        >>> enc = HardwareEfficientEncoding(n_features=4, reps=2, rotation='Y')
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['ry']
        8
        >>> breakdown['cnot']
        6
        >>> breakdown['total']
        14

        >>> enc_circ = HardwareEfficientEncoding(
        ...     n_features=4, reps=2, rotation='X', entanglement='circular'
        ... )
        >>> enc_circ.gate_count_breakdown()['cnot']
        8

        See Also
        --------
        resource_summary : Comprehensive resource analysis including
            hardware requirements and encoding characteristics.
        properties : Basic encoding properties via EncodingProperties object.
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        rotation_gates = self.reps * n
        cnot_gates = self.reps * n_pairs
        total = rotation_gates + cnot_gates

        rx = rotation_gates if self.rotation == "X" else 0
        ry = rotation_gates if self.rotation == "Y" else 0
        rz = rotation_gates if self.rotation == "Z" else 0

        return GateCountBreakdown(
            rx=rx,
            ry=ry,
            rz=rz,
            cnot=cnot_gates,
            total_single_qubit=rotation_gates,
            total_two_qubit=cnot_gates,
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
                - ``entanglement``: Entanglement topology ('linear' or 'circular')

            **Gate Counts**:
                - ``gate_counts``: Full breakdown from ``gate_count_breakdown()``

            **Encoding Characteristics**:
                - ``is_entangling``: True for multi-qubit configurations
                - ``simulability``: 'not_simulable' for entangled states
                - ``trainability_estimate``: Trainability score (0.0-1.0)

            **Hardware Requirements**:
                - ``connectivity``: Required qubit connectivity topology
                - ``native_gates``: List of required gate types

            **Entanglement Details**:
                - ``n_entanglement_pairs``: Number of CNOT pairs per layer
                - ``entanglement_pairs``: List of (control, target) qubit pairs

        Examples
        --------
        >>> enc = HardwareEfficientEncoding(
        ...     n_features=4, reps=2, rotation='Y', entanglement='linear'
        ... )
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['depth']
        4
        >>> summary['is_entangling']
        True
        >>> summary['hardware_requirements']['connectivity']
        'linear'
        >>> summary['hardware_requirements']['native_gates']
        ['RY', 'CNOT']

        Full entanglement for ion trap devices:

        >>> enc_full = HardwareEfficientEncoding(n_features=4, entanglement='full')
        >>> enc_full.resource_summary()['n_entanglement_pairs']
        6

        See Also
        --------
        gate_count_breakdown : Detailed gate count breakdown.
        properties : Basic encoding properties.
        get_entanglement_pairs : Access entanglement pair list.
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
            "entanglement": self.entanglement,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": props.is_entangling,
            "simulability": props.simulability,
            "trainability_estimate": props.trainability_estimate,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": self.entanglement,
                "native_gates": [f"R{self.rotation}", "CNOT"],
            },
            # Entanglement details
            "n_entanglement_pairs": len(self._entanglement_pairs),
            "entanglement_pairs": self.get_entanglement_pairs(),
        }

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
            f"reps={self.reps}, "
            f"rotation={self.rotation!r}, "
            f"entanglement={self.entanglement!r})"
        )
