"""IQP (Instantaneous Quantum Polynomial) encoding module.

This module implements IQPEncoding, a quantum data encoding technique based on
Instantaneous Quantum Polynomial (IQP) circuits. IQP circuits consist of
diagonal gates in the computational basis separated by layers of Hadamard
gates, creating highly entangled quantum states that encode classical data
with provable classical hardness guarantees.

IQP encoding is particularly notable for quantum machine learning because:

1. **Classical Hardness**: Simulating IQP circuits is provably hard under
   standard complexity-theoretic assumptions (polynomial hierarchy collapse)
2. **Quantum Advantage**: Provides a pathway to demonstrating quantum
   computational advantage in machine learning contexts
3. **Feature Interactions**: ZZ interactions naturally capture pairwise
   feature correlations in the quantum state
4. **Expressivity**: Creates highly entangled states that can represent
   complex decision boundaries

The encoding creates quantum states of the form:

    |ψ(x)⟩ = U_IQP(x)|+⟩^⊗n

where U_IQP consists of alternating layers of Hadamard gates, single-qubit
Z rotations, and two-qubit ZZ interactions parameterized by input features.

Mathematical Background
-----------------------
The IQP circuit structure for each layer consists of:

1. **Hadamard Layer**: H^⊗n creates uniform superposition
2. **Single-Qubit Phase**: RZ(2xᵢ) applies data-dependent phases
3. **Two-Qubit Interactions**: ZZ(xᵢxⱼ) = exp(-i xᵢxⱼ ZᵢZⱼ)

The ZZ interaction is implemented as:

    ZZ(θ) = CNOT · (I ⊗ RZ(2θ)) · CNOT

This creates entanglement between qubits while encoding feature products.

The output state encodes both individual features (via RZ) and feature
interactions (via ZZ), enabling the quantum classifier to learn nonlinear
decision boundaries.

Note on Data Preprocessing
--------------------------
For IQP encoding, input features are typically:

- Scaled to [0, 2π] or [-π, π] for effective phase encoding
- Standardized if features have very different scales
- The ZZ interactions encode products xᵢ·xⱼ, so feature scaling affects
  the strength of learned interactions

The factor of 2 in RZ(2x) ensures full coverage of the Bloch sphere's
phase when features span [0, π].

Use Cases
---------
IQP encoding is particularly suited for:

- **Quantum kernel methods**: Computing quantum kernels for classification
  and regression tasks where classical simulation is provably hard
- **Quantum advantage demonstrations**: Benchmarking quantum vs classical
  performance on machine learning tasks
- **Feature interaction modeling**: Problems where pairwise feature
  correlations are important (e.g., interaction terms in physics)
- **Variational quantum classifiers**: As a fixed or trainable feature map
  layer in hybrid quantum-classical models
- **Quantum neural networks**: Input encoding layer for QNN architectures

Limitations
-----------
- **Qubit scaling**: Requires n qubits for n features (linear scaling)
- **Gate count**: Full entanglement creates O(n²) two-qubit gates per layer,
  which may exceed NISQ device capabilities for large feature counts
- **Trainability**: Deep circuits with full entanglement may exhibit barren
  plateaus, making gradient-based optimization challenging
- **Hardware constraints**: Full entanglement may require all-to-all qubit
  connectivity, which is not available on all quantum hardware
- **No amplitude encoding**: Unlike AmplitudeEncoding, does not achieve
  exponential compression of classical data

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation and entanglement pattern computation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.iqp').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters (n_features, reps, entanglement)
- Entanglement pair computation details
- Circuit generation progress for each backend
- Gate count breakdowns

Resource Analysis
-----------------
IQPEncoding provides methods to analyze circuit resources:

- ``get_entanglement_pairs()``: Inspect which qubit pairs have ZZ interactions
- ``gate_count_breakdown()``: Get detailed gate counts by type
- ``properties``: Access computed circuit properties (depth, gate counts, etc.)

These methods help understand resource requirements before execution:

    >>> enc = IQPEncoding(n_features=8, entanglement='full')
    >>> len(enc.get_entanglement_pairs())  # Number of ZZ interactions
    28
    >>> enc.gate_count_breakdown()
    {'hadamard': 16, 'rz_single': 16, 'rz_zz': 56, 'cnot': 112, 'total': 200}

References
----------
.. [1] Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
       feature spaces." Nature, 567(7747), 209-212.
.. [2] Bremner, M. J., Montanaro, A., & Shepherd, D. J. (2016). "Average-case
       complexity versus approximate simulation of commuting quantum
       computations." Physical Review Letters, 117(8), 080501.
.. [3] Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature
       Hilbert spaces." Physical Review Letters, 122(4), 040504.
.. [4] McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
       network training landscapes." Nature Communications, 9, 4812.
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
#   To enable debug logging for IQP encoding:
#
#   >>> import logging
#   >>> logging.getLogger('encoding_atlas.encodings.iqp').setLevel(logging.DEBUG)
#   >>> handler = logging.StreamHandler()
#   >>> handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
#   >>> logging.getLogger('encoding_atlas.encodings.iqp').addHandler(handler)
#
# Log Levels:
#   - DEBUG: Detailed information for diagnosing issues (entanglement pairs,
#            gate counts, circuit generation steps)
#   - INFO: General operational information (encoding created, circuit generated)
#   - WARNING: Potential issues (large feature count with full entanglement)
#
# The logger is named after the module to allow granular control over log output.
# By default, no handlers are attached, so no output is produced unless the
# application configures logging.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["IQPEncoding"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Threshold for warning about full entanglement with many features.
#
# Full entanglement creates n(n-1)/2 ZZ interaction pairs, each requiring
# 2 CNOT gates. The gate count scales quadratically:
#
#   Two-qubit gates per layer = n(n-1)
#
# Examples (per layer):
#   - 4 features:  12 CNOTs (manageable)
#   - 8 features:  56 CNOTs (moderate)
#   - 12 features: 132 CNOTs (threshold - warning issued)
#   - 16 features: 240 CNOTs (may exceed NISQ capabilities)
#   - 20 features: 380 CNOTs (likely impractical on current hardware)
#
# At 12 features, the circuit complexity becomes significant enough to warrant
# a warning. Users should consider 'linear' or 'circular' entanglement for
# larger feature counts, or ensure their target hardware supports the required
# gate count and connectivity.
_FULL_ENTANGLEMENT_WARNING_THRESHOLD: int = 12

# Default number of repetitions for IQP encoding layers.
#
# The choice of 2 repetitions provides a good balance between:
#   - Expressivity: Multiple layers increase feature map expressiveness
#   - Trainability: Fewer layers reduce barren plateau risk
#   - Gate count: Each rep multiplies the total gate count
#
# Research suggests 2-3 repetitions often suffice for classification tasks.
# Higher values may be needed for more complex decision boundaries but
# increase the risk of trainability issues.
_DEFAULT_REPS: int = 2

# Default entanglement topology.
#
# 'full' entanglement provides maximum expressivity by creating ZZ interactions
# between all pairs of qubits. While this maximizes the encoding's ability to
# capture feature correlations, it also:
#   - Creates O(n²) two-qubit gates
#   - Requires all-to-all qubit connectivity
#   - May face trainability challenges for deep circuits
#
# For hardware-constrained applications, consider 'linear' or 'circular'.
_DEFAULT_ENTANGLEMENT: Literal["full", "linear", "circular"] = "full"


# =============================================================================
# Type Definitions
# =============================================================================


class GateCountBreakdown(TypedDict):
    """Type definition for the gate_count_breakdown() return value.

    Provides a detailed breakdown of gate counts by type for circuit
    resource estimation and hardware compatibility analysis. This TypedDict
    ensures type safety and IDE autocompletion for the returned dictionary.

    Attributes
    ----------
    hadamard : int
        Number of Hadamard gates. Each qubit receives one H gate per
        repetition, creating the initial superposition state.

    rz_single : int
        Number of single-qubit RZ gates for data encoding. Each qubit
        receives one RZ(2xᵢ) gate per repetition.

    rz_zz : int
        Number of RZ gates used in ZZ interaction decomposition. Each
        ZZ interaction uses one RZ gate.

    cnot : int
        Number of CNOT gates. Each ZZ interaction requires 2 CNOT gates
        for its decomposition: ZZ(θ) = CNOT · RZ(2θ) · CNOT.

    total_single_qubit : int
        Total number of single-qubit gates (hadamard + rz_single + rz_zz).

    total_two_qubit : int
        Total number of two-qubit gates (equals cnot count).

    total : int
        Total gate count (total_single_qubit + total_two_qubit).

    Examples
    --------
    >>> enc = IQPEncoding(n_features=4, reps=1, entanglement='full')
    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['hadamard']
    4
    >>> breakdown['cnot']
    12
    >>> breakdown['total_single_qubit']
    14
    >>> breakdown['total']
    26
    """

    hadamard: int
    """Number of Hadamard gates (n_qubits × reps)."""

    rz_single: int
    """Number of single-qubit RZ gates for data encoding."""

    rz_zz: int
    """Number of RZ gates in ZZ interaction decomposition."""

    cnot: int
    """Number of CNOT gates (2 × n_pairs × reps)."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates."""

    total: int
    """Total gate count."""


class IQPEncoding(BaseEncoding):
    """IQP encoding using Hadamard gates and ZZ interactions.

    IQPEncoding implements Instantaneous Quantum Polynomial circuits for
    quantum machine learning. It creates highly entangled quantum states
    by combining Hadamard gates, single-qubit Z rotations, and two-qubit
    ZZ interactions parameterized by classical input features.

    This encoding is notable for its provable classical hardness: simulating
    IQP circuits is believed to be intractable for classical computers under
    standard complexity-theoretic assumptions, making it a candidate for
    demonstrating quantum advantage in machine learning.

    The circuit structure for each repetition is:

        |0⟩ ─ H ─ RZ(2x₀) ─╭─────╮─╭─────╮─
        |0⟩ ─ H ─ RZ(2x₁) ─│ ZZ  │─│     │─
        |0⟩ ─ H ─ RZ(2x₂) ─╰─────╯─│ ZZ  │─
        ...                        ╰─────╯

    where ZZ(xᵢxⱼ) interactions are applied according to the entanglement
    topology (full, linear, or circular connectivity).

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
        Each feature requires one qubit, so this also determines the number
        of qubits in the circuit.
    reps : int, default=2
        Number of times to repeat the encoding layers. Higher values create
        deeper circuits with stronger entanglement but may face trainability
        issues. Must be at least 1.
    entanglement : {"full", "linear", "circular"}, default="full"
        Topology of ZZ interactions between qubits:

        - "full": All-to-all connectivity. Every pair (i, j) with i < j has a
          ZZ interaction. Creates n(n-1)/2 entangling gates per layer.
          Maximum expressivity but highest gate count.
        - "linear": Nearest-neighbor connectivity. Only pairs (i, i+1) have
          ZZ interactions. Creates n-1 entangling gates per layer.
          Hardware-friendly for linear qubit topologies.
        - "circular": Nearest-neighbor with periodic boundary. Like linear,
          but adds a ZZ interaction between qubits n-1 and 0.
          Creates n entangling gates per layer.

    Attributes
    ----------
    reps : int
        Number of encoding layer repetitions.
    entanglement : str
        The entanglement topology ("full", "linear", or "circular").
    n_features : int
        Number of classical features (inherited from BaseEncoding).
    n_qubits : int
        Number of qubits, equal to n_features.

    Examples
    --------
    Create a basic IQP encoding with default settings:

    >>> from encoding_atlas import IQPEncoding
    >>> import numpy as np
    >>> enc = IQPEncoding(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.entanglement
    'full'

    Generate a PennyLane circuit:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Use linear entanglement for hardware-friendly circuits:

    >>> enc_linear = IQPEncoding(n_features=4, entanglement='linear')
    >>> props = enc_linear.properties
    >>> props.two_qubit_gates < enc.properties.two_qubit_gates
    True

    Generate Qiskit and Cirq circuits:

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
    >>> props.simulability
    'not_simulable'

    References
    ----------
    .. [1] Havlíček, V., et al. (2019). "Supervised learning with
           quantum-enhanced feature spaces." Nature, 567(7747), 209-212.
    .. [2] Bremner, M. J., et al. (2016). "Average-case complexity versus
           approximate simulation of commuting quantum computations."
           Physical Review Letters, 117(8), 080501.
    .. [3] McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
           network training landscapes." Nature Communications, 9, 4812.

    See Also
    --------
    ZZFeatureMap : Similar encoding with different phase conventions.
    AngleEncoding : Simpler encoding without entanglement.
    PauliFeatureMap : Generalized Pauli-based feature map.
    DataReuploading : Re-uploads data with trainable intermediate layers.

    Warnings
    --------
    **Full Entanglement with Many Features**: When using ``entanglement='full'``
    with more than 12 features, a warning is issued because the circuit
    complexity may exceed practical limits for NISQ devices. The number of
    two-qubit gates scales as O(n²), which can lead to:

    - Excessive circuit depth and gate errors
    - Hardware connectivity constraints (all-to-all required)
    - Potential trainability issues (barren plateaus)

    Consider using ``entanglement='linear'`` or ``entanglement='circular'``
    for large feature counts.

    Notes
    -----
    **Classical Hardness**: The output distribution of IQP circuits is
    provably hard to sample from classically under the assumption that
    the polynomial hierarchy does not collapse [2]_. This provides theoretical
    motivation for potential quantum advantage in machine learning.

    **Feature Interactions**: The ZZ(xᵢxⱼ) gates encode products of features,
    allowing the quantum kernel to capture pairwise correlations. This is
    analogous to polynomial feature expansion in classical machine learning,
    but executed in superposition.

    **Entanglement Topology Trade-offs**:

    +------------+------------------+---------------+------------------------+
    | Topology   | Gate Scaling     | Connectivity  | Best For               |
    +============+==================+===============+========================+
    | full       | O(n²)            | All-to-all    | Max expressivity       |
    +------------+------------------+---------------+------------------------+
    | linear     | O(n)             | Nearest-only  | NISQ devices           |
    +------------+------------------+---------------+------------------------+
    | circular   | O(n)             | Ring          | Periodic problems      |
    +------------+------------------+---------------+------------------------+

    **Trainability**: Deeper circuits (higher reps) with full entanglement
    may exhibit barren plateaus, where gradients become exponentially small
    [3]_. Mitigation strategies include:

    - Using fewer repetitions (reps=1 or 2)
    - Using sparser entanglement (linear or circular)
    - Layer-wise training approaches
    - Careful parameter initialization

    **Gate Decomposition**: The ZZ interaction is implemented using the
    standard decomposition: ``ZZ(θ) = CNOT · RZ(2θ) · CNOT``. This requires
    2 CNOT gates per interaction, which dominates the circuit's two-qubit
    gate count.
    """

    # Valid entanglement topologies
    _VALID_ENTANGLEMENTS: frozenset[str] = frozenset({"full", "linear", "circular"})

    __slots__ = ("reps", "entanglement", "_entanglement_pairs")

    def __init__(
        self,
        n_features: int,
        reps: int = _DEFAULT_REPS,
        entanglement: Literal["full", "linear", "circular"] = _DEFAULT_ENTANGLEMENT,
    ) -> None:
        """Initialize the IQP encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        reps : int, default=2
            Number of encoding layer repetitions.
        entanglement : {"full", "linear", "circular"}, default="full"
            Topology of ZZ interactions between qubits.

        Raises
        ------
        ValueError
            If reps is less than 1.
        ValueError
            If entanglement is not one of "full", "linear", "circular".
        ValueError
            If n_features is less than 1 (raised by parent class).

        Warns
        -----
        UserWarning
            If using full entanglement with more than 12 features, as the
            circuit complexity may exceed practical NISQ device limits.
        """
        # Validate repetitions
        # Accept both Python int and numpy integer types, but reject bool
        # (bool is a subclass of int in Python, so check it first)
        if (
            isinstance(reps, bool)
            or not isinstance(reps, (int, np.integer))
            or reps < 1
        ):
            raise ValueError(
                f"reps must be at least 1 (got {reps!r}). "
                "Each repetition adds one layer of H, RZ, and ZZ gates."
            )
        # Convert numpy integer to Python int for consistency
        reps = int(reps)

        # Validate entanglement topology
        if entanglement not in self._VALID_ENTANGLEMENTS:
            raise ValueError(
                f"entanglement must be one of {sorted(self._VALID_ENTANGLEMENTS)}, "
                f"got {entanglement!r}"
            )

        super().__init__(n_features, reps=reps, entanglement=entanglement)
        self.reps: int = reps
        self.entanglement: Literal["full", "linear", "circular"] = entanglement

        # Pre-compute and cache entanglement pairs as an immutable tuple.
        # This eliminates redundant computation during circuit generation:
        # - Each get_circuit() call would otherwise recompute pairs
        # - Batch processing of N samples would compute pairs N times
        # - With caching, pairs are computed once at initialization
        #
        # Stored as tuple for:
        # - Immutability (prevents accidental modification)
        # - Thread safety (safe to share across threads)
        # - Hashability (could be used as dict key if needed)
        self._entanglement_pairs: tuple[tuple[int, int], ...] = (
            self._compute_entanglement_pairs(n_features, entanglement)
        )

        # Log initialization
        _logger.debug(
            "IQPEncoding initialized: n_features=%d, reps=%d, entanglement=%r",
            n_features,
            reps,
            entanglement,
        )

        # Warn about potentially excessive gate count with full entanglement
        if entanglement == "full" and n_features > _FULL_ENTANGLEMENT_WARNING_THRESHOLD:
            n_pairs = n_features * (n_features - 1) // 2
            cnot_count = 2 * n_pairs * reps
            warnings.warn(
                f"Full entanglement with {n_features} features creates "
                f"{n_pairs} ZZ interaction pairs per layer ({cnot_count} total "
                f"CNOT gates for {reps} reps). This may exceed practical limits "
                f"for NISQ devices. Consider using entanglement='linear' or "
                f"'circular' for better hardware compatibility.",
                UserWarning,
                stacklevel=2,
            )
            _logger.warning(
                "Large feature count with full entanglement: %d features, "
                "%d CNOT gates total",
                n_features,
                cnot_count,
            )

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        For IQP encoding, each feature is encoded on a dedicated qubit,
        so n_qubits equals n_features.

        Returns
        -------
        int
            Number of qubits, equal to n_features.
        """
        return self.n_features

    @property
    def depth(self) -> int:
        """Logical depth of the encoding (number of layer types per repetition).

        This property returns the number of distinct layer types in the IQP
        circuit structure, NOT the actual compiled circuit depth. Each
        repetition consists of three conceptual phases:

        1. **Hadamard layer**: H gates on all qubits (actual depth: 1)
        2. **Single-qubit RZ layer**: RZ(2xᵢ) on each qubit (actual depth: 1)
        3. **ZZ interaction layer**: CNOT-RZ-CNOT sequences (actual depth: varies)

        The actual compiled circuit depth depends on the entanglement topology
        and the quantum compiler's ability to parallelize gates:

        +------------+--------------------+--------------------------------+
        | Topology   | ZZ Layer Depth     | Approximate Total per Rep      |
        +============+====================+================================+
        | linear     | O(1) parallelizable| ~5 (H + RZ + 3 for ZZ chain)   |
        +------------+--------------------+--------------------------------+
        | circular   | O(1) parallelizable| ~5-6 (similar to linear)       |
        +------------+--------------------+--------------------------------+
        | full       | O(n) sequential    | 2 + O(n) (limited parallelism) |
        +------------+--------------------+--------------------------------+

        For hardware execution planning, use ``gate_count_breakdown()`` for
        precise gate counts, or consult your quantum compiler's output for
        the actual transpiled circuit depth.

        Returns
        -------
        int
            Logical depth, equal to 3 * reps. This represents the number of
            conceptual layer types, not the gate-level circuit depth.

        See Also
        --------
        gate_count_breakdown : Get exact gate counts by type.
        properties : Access all computed circuit properties.
        get_entanglement_pairs : Understand entanglement structure.

        Notes
        -----
        The distinction between logical and actual depth matters for:

        - **NISQ hardware**: Actual depth determines decoherence impact
        - **Simulation**: Gate-level depth affects classical simulation cost
        - **Compilation**: Quantum compilers may optimize actual depth

        For benchmarking or hardware resource estimation, the actual compiled
        depth from your target backend's transpiler is more accurate than
        this logical depth value.
        """
        return self.reps * 3

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs for ZZ entanglement based on topology.

        Returns the list of qubit pairs that will have ZZ interactions applied
        based on the configured entanglement topology. This is useful for:

        - Understanding circuit structure before generation
        - Verifying hardware connectivity requirements
        - Debugging entanglement patterns
        - Computing resource requirements

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for ZZ interactions.
            Each tuple (i, j) indicates a ZZ(xᵢ·xⱼ) gate between qubits i and j.

        Examples
        --------
        >>> enc = IQPEncoding(n_features=4, entanglement='full')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        >>> enc_linear = IQPEncoding(n_features=4, entanglement='linear')
        >>> enc_linear.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        >>> enc_circular = IQPEncoding(n_features=4, entanglement='circular')
        >>> enc_circular.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3), (3, 0)]

        Notes
        -----
        The number of pairs depends on the entanglement topology:

        - **full**: n(n-1)/2 pairs (all unique combinations)
        - **linear**: n-1 pairs (nearest neighbors only)
        - **circular**: n pairs for n>2 (nearest neighbors + wrap-around),
          or n-1 pairs for n≤2 (no wrap-around needed)

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts by type.
        properties : Access all computed circuit properties.
        """
        return self._get_entanglement_pairs()

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type.

        Computes the exact number of each gate type in the IQP encoding
        circuit. This is useful for:

        - Resource estimation before circuit execution
        - Hardware compatibility assessment
        - Comparing different encoding configurations
        - Debugging and verification

        Returns
        -------
        GateCountBreakdown
            TypedDict with gate counts:

            - ``'hadamard'``: Number of Hadamard gates (n_qubits × reps)
            - ``'rz_single'``: Number of single-qubit RZ gates (n_qubits × reps)
            - ``'rz_zz'``: Number of RZ gates in ZZ interactions (n_pairs × reps)
            - ``'cnot'``: Number of CNOT gates (2 × n_pairs × reps)
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Total two-qubit gates
            - ``'total'``: Total gate count

        Examples
        --------
        >>> enc = IQPEncoding(n_features=4, reps=1, entanglement='full')
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['hadamard']
        4
        >>> breakdown['cnot']
        12
        >>> breakdown['total']
        26

        Compare entanglement topologies:

        >>> enc_full = IQPEncoding(n_features=8, entanglement='full')
        >>> enc_linear = IQPEncoding(n_features=8, entanglement='linear')
        >>> enc_full.gate_count_breakdown()['cnot']
        112
        >>> enc_linear.gate_count_breakdown()['cnot']
        28

        See Also
        --------
        get_entanglement_pairs : Get the qubit pairs for ZZ interactions.
        properties : Access all computed circuit properties.
        """
        n = self.n_features
        # Use cached pairs directly - no list conversion needed for len()
        n_pairs = len(self._entanglement_pairs)

        h_gates = self.reps * n
        rz_single = self.reps * n
        rz_zz = self.reps * n_pairs
        cnot_gates = self.reps * 2 * n_pairs

        total_single_qubit = h_gates + rz_single + rz_zz
        total_two_qubit = cnot_gates
        total = total_single_qubit + total_two_qubit

        _logger.debug(
            "Gate breakdown: H=%d, RZ_single=%d, RZ_zz=%d, CNOT=%d, total=%d",
            h_gates,
            rz_single,
            rz_zz,
            cnot_gates,
            total,
        )

        return GateCountBreakdown(
            hadamard=h_gates,
            rz_single=rz_single,
            rz_zz=rz_zz,
            cnot=cnot_gates,
            total_single_qubit=total_single_qubit,
            total_two_qubit=total_two_qubit,
            total=total,
        )

    def resource_summary(self) -> dict[str, Any]:
        """Generate a comprehensive resource summary for this encoding.

        Provides a detailed breakdown of circuit resources including qubit
        requirements, circuit depth, gate counts, entanglement information,
        and encoding characteristics. Useful for hardware planning, comparing
        encoding configurations, and generating reports.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``n_qubits``: Number of qubits required
            - ``n_features``: Number of input features
            - ``depth``: Circuit depth (conceptual layers)
            - ``reps``: Number of encoding repetitions
            - ``entanglement``: Entanglement topology ("full", "linear", "circular")
            - ``entanglement_pairs``: List of qubit pairs with ZZ interactions
            - ``n_entanglement_pairs``: Number of ZZ interaction pairs per layer
            - ``gate_counts``: Detailed breakdown from gate_count_breakdown()
            - ``is_entangling``: Always True for IQP
            - ``simulability``: "not_simulable" (provably hard classically)
            - ``trainability_estimate``: Estimated trainability (0.0 to 1.0)
            - ``hardware_requirements``: Dict with connectivity and gate requirements

        Examples
        --------
        Get complete resource analysis:

        >>> enc = IQPEncoding(n_features=4, reps=2, entanglement='full')
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['n_entanglement_pairs']
        6
        >>> summary['gate_counts']['cnot']
        24

        Compare different entanglement topologies:

        >>> enc_full = IQPEncoding(n_features=8, entanglement='full')
        >>> enc_linear = IQPEncoding(n_features=8, entanglement='linear')
        >>> enc_full.resource_summary()['n_entanglement_pairs']
        28
        >>> enc_linear.resource_summary()['n_entanglement_pairs']
        7

        Use for hardware planning:

        >>> enc = IQPEncoding(n_features=6, entanglement='full')
        >>> summary = enc.resource_summary()
        >>> print(f"Requires {summary['hardware_requirements']['connectivity']}")
        Requires all-to-all
        >>> print(f"Total two-qubit gates: {summary['gate_counts']['total_two_qubit']}")
        Total two-qubit gates: 60

        Notes
        -----
        This method combines information from multiple sources:

        - ``properties``: Theoretical encoding characteristics
        - ``gate_count_breakdown()``: Detailed gate analysis
        - ``get_entanglement_pairs()``: Entanglement structure

        For quick access to just gate counts, use ``gate_count_breakdown()``.
        For just entanglement pairs, use ``get_entanglement_pairs()``.

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts by type.
        get_entanglement_pairs : Get the qubit pairs for ZZ interactions.
        properties : Access computed circuit properties.
        """
        pairs = self._get_entanglement_pairs()
        gate_counts = self.gate_count_breakdown()
        props = self.properties

        # Determine connectivity requirement based on entanglement topology
        if self.entanglement == "full":
            connectivity = "all-to-all"
        elif self.entanglement == "linear":
            connectivity = "linear"
        else:  # circular
            connectivity = "ring"

        summary: dict[str, Any] = {
            # Basic circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            # Entanglement configuration
            "entanglement": self.entanglement,
            "entanglement_pairs": pairs,
            "n_entanglement_pairs": len(pairs),
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": True,
            "simulability": "not_simulable",
            "trainability_estimate": props.trainability_estimate,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": connectivity,
                "native_gates": ["H", "RZ", "CNOT"],
                "min_two_qubit_gate_fidelity": 0.99,  # Recommended for NISQ
            },
        }

        _logger.debug(
            "Resource summary: n_qubits=%d, depth=%d, total_gates=%d, "
            "entanglement=%s, pairs=%d",
            self.n_qubits,
            self.depth,
            gate_counts["total"],
            self.entanglement,
            len(pairs),
        )

        return summary

    @staticmethod
    def _compute_entanglement_pairs(
        n_features: int,
        entanglement: Literal["full", "linear", "circular"],
    ) -> tuple[tuple[int, int], ...]:
        """Compute qubit pairs for ZZ entanglement based on topology.

        This is a static method used during initialization to compute the
        entanglement pairs once. The result is cached as an immutable tuple.

        Parameters
        ----------
        n_features : int
            Number of features (qubits).
        entanglement : {"full", "linear", "circular"}
            Entanglement topology.

        Returns
        -------
        tuple[tuple[int, int], ...]
            Immutable tuple of (control, target) qubit pairs for ZZ interactions.

        Notes
        -----
        - Full: n(n-1)/2 pairs (all combinations)
        - Linear: n-1 pairs (nearest neighbors)
        - Circular: n pairs for n>2 (nearest neighbors + wrap-around),
          or n-1 pairs for n≤2 (no wrap-around needed)
        """
        if entanglement == "full":
            pairs = [
                (i, j) for i in range(n_features) for j in range(i + 1, n_features)
            ]
        elif entanglement == "linear":
            pairs = [(i, i + 1) for i in range(n_features - 1)]
        else:  # circular
            pairs = [(i, i + 1) for i in range(n_features - 1)]
            if n_features > 2:
                pairs.append((n_features - 1, 0))

        return tuple(pairs)

    def _get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs for ZZ entanglement based on topology.

        Returns a list for API compatibility. Internally, pairs are cached
        as an immutable tuple computed at initialization time.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for ZZ interactions.
            This is a new list created from the cached tuple, so callers
            can safely modify it without affecting the encoding.

        Notes
        -----
        - Full: n(n-1)/2 pairs (all combinations)
        - Linear: n-1 pairs (nearest neighbors)
        - Circular: n pairs for n>2 (nearest neighbors + wrap-around)

        See Also
        --------
        get_entanglement_pairs : Public API that returns the same data.
        _compute_entanglement_pairs : Static method that computes pairs.
        """
        return list(self._entanglement_pairs)

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Dispatch to backend after pre-validation.

        Accepts a 1D array (the normal case) or a 2D single-sample
        ``(1, n_features)`` array for robustness when called directly.
        """
        # Defensive 2D-to-1D handling for direct callers; the inherited
        # template methods already pass 1D input.
        if x.ndim == 2:
            x = x[0]

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

        Creates a callable that applies the IQP encoding gates when
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

        # Use cached pairs directly (tuple) - no list conversion needed for iteration
        pairs = self._entanglement_pairs
        n_qubits = self.n_qubits
        reps = self.reps

        def circuit() -> None:
            """Apply the IQP encoding gates."""
            for _ in range(reps):
                # Hadamard layer - create superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                # Single-qubit Z rotations - encode individual features
                for i, val in enumerate(x):
                    qml.RZ(2 * val, wires=i)
                # ZZ interactions - encode feature products
                for i, j in pairs:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * x[i] * x[j], wires=j)
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

        # Use cached pairs directly (tuple) - no list conversion needed for iteration
        pairs = self._entanglement_pairs
        qc = QuantumCircuit(self.n_qubits, name="IQPEncoding")

        for _ in range(self.reps):
            # Hadamard layer
            for i in range(self.n_qubits):
                qc.h(i)
            # Single-qubit Z rotations
            for i, val in enumerate(x):
                qc.rz(2 * float(val), i)
            # ZZ interactions
            for i, j in pairs:
                qc.cx(i, j)
                qc.rz(2 * float(x[i]) * float(x[j]), j)
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

        # Use cached pairs directly (tuple) - no list conversion needed for iteration
        pairs = self._entanglement_pairs
        qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()

        for _ in range(self.reps):
            # Hadamard layer
            circuit.append([cirq.H(q) for q in qubits])
            # Single-qubit Z rotations
            circuit.append(
                [cirq.rz(2 * float(x[i]))(qubits[i]) for i in range(self.n_qubits)]
            )
            # ZZ interactions
            for i, j in pairs:
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))
                circuit.append(cirq.rz(2 * float(x[i]) * float(x[j]))(qubits[j]))
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))

        return circuit

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Returns
        -------
        EncodingProperties
            Computed properties including:
            - n_qubits: Number of qubits (= n_features)
            - depth: Circuit depth (= 3 * reps)
            - gate_count: Total gates including H, RZ, and CNOT
            - is_entangling: Always True (ZZ interactions create entanglement)
            - simulability: "not_simulable" (provably hard classically)
        """
        n = self.n_features
        # Use cached pairs directly - no list conversion needed for len()
        n_pairs = len(self._entanglement_pairs)

        # Gates per rep: n Hadamards + n RZ + n_pairs * (CNOT + RZ + CNOT)
        h_gates = self.reps * n
        rz_single = self.reps * n
        rz_zz = self.reps * n_pairs
        cnot_gates = self.reps * 2 * n_pairs

        return EncodingProperties(
            n_qubits=n,
            depth=self.reps * 3,
            gate_count=h_gates + rz_single + rz_zz + cnot_gates,
            single_qubit_gates=h_gates + rz_single + rz_zz,
            two_qubit_gates=cnot_gates,
            parameter_count=self.reps * (n + n_pairs),
            is_entangling=True,
            simulability="not_simulable",
            trainability_estimate=max(0.3, 0.9 - 0.1 * self.reps),
            notes=(
                f"IQP encoding with {self.entanglement} entanglement. "
                f"Provably hard to simulate classically under standard assumptions."
            ),
        )

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
            f"entanglement={self.entanglement!r})"
        )
