"""ZZ Feature Map encoding module.

This module implements ZZFeatureMap, a quantum data encoding technique that
follows the Qiskit convention for second-order Pauli-Z feature maps. It creates
entangled quantum states using alternating layers of Hadamard gates, phase
rotations, and ZZ entangling interactions with a specific phase convention.

The ZZ Feature Map is widely used in quantum machine learning, particularly in
Quantum Support Vector Machines (QSVM) and quantum kernel methods, offering:

1. **Qiskit Compatibility**: Follows the standard Qiskit ZZFeatureMap convention
2. **Feature Interactions**: Captures pairwise feature correlations via ZZ gates
3. **Periodic Structure**: Uses (π - x) phase convention for symmetry
4. **Entanglement Control**: Configurable connectivity for different hardware

The encoding creates quantum states using:

    |ψ(x)⟩ = [U_ZZ(x) · H^⊗n]^reps |0⟩^⊗n

where U_ZZ consists of single-qubit phase gates and two-qubit ZZ interactions
parameterized by (π - xᵢ)(π - xⱼ) for feature interactions.

Mathematical Background
-----------------------
The ZZ Feature Map circuit structure for each layer consists of:

1. **Hadamard Layer**: H^⊗n creates uniform superposition
2. **Phase Gates**: P(2xᵢ) applies data-dependent phases to each qubit
3. **ZZ Interactions**: ZZ((π-xᵢ)(π-xⱼ)) for pairwise feature encoding

The phase convention 2(π - xᵢ)(π - xⱼ) differs from IQP's direct product xᵢxⱼ.
This creates a different kernel function when used in QSVM applications:

    K(x, x') = |⟨ψ(x)|ψ(x')⟩|²

The specific phase convention affects the geometry of the feature space and
can be more suitable for certain types of data distributions.

Note on Data Preprocessing
--------------------------
For ZZ Feature Map encoding, input features are typically:

- Scaled to [0, π] for the (π - x) convention to work properly
- Alternatively scaled to [0, 2π] for full phase coverage
- The (π - xᵢ)(π - xⱼ) interaction means features near π have minimal
  interaction strength, while features near 0 or 2π have maximum interaction

The asymmetric nature of the (π - x) convention can be exploited when data
has meaningful relationships at the boundaries of the feature range.

Debugging
---------
This module supports Python's standard logging for debugging circuit generation
and parameter validation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.zz_feature_map').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters (n_features, reps, entanglement)
- Circuit generation details (backend, input shape)
- Entanglement pair computation
- Batch processing progress
- Resource scaling warnings for large feature counts

References
----------
.. [1] Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
       feature spaces." Nature, 567(7747), 209-212.
.. [2] Qiskit Development Team. "Qiskit Machine Learning: ZZFeatureMap."
       https://qiskit.org/documentation/machine-learning/
.. [3] Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature
       Hilbert spaces." Physical Review Letters, 122(4), 040504.
"""

from __future__ import annotations

import logging
import warnings
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
# Configure a logger for this module to enable optional debug output.
# By default, Python's logging is disabled (level WARNING), so these
# debug statements have no performance impact in production.
#
# To enable debug logging for this module:
#   import logging
#   logging.getLogger('encoding_atlas.encodings.zz_feature_map').setLevel(logging.DEBUG)
#   logging.basicConfig(level=logging.DEBUG)
#
# Or configure via your application's logging setup.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ['ZZFeatureMap']

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default number of repetitions for the ZZ Feature Map encoding layers.
#
# The choice of 2 repetitions is conventional in quantum machine learning:
# - 1 rep provides basic feature encoding but limited expressivity
# - 2 reps (default) balances expressivity with circuit depth
# - Higher reps increase expressivity but also circuit depth and noise
#
# This value matches Qiskit's default for ZZFeatureMap.
_DEFAULT_REPS: int = 2

# Default entanglement topology for ZZ interactions.
#
# "full" entanglement creates all-to-all connectivity, providing maximum
# feature correlation capture at the cost of O(n²) entangling gates.
# For hardware with limited connectivity, consider "linear" or "circular".
_DEFAULT_ENTANGLEMENT: Literal["full", "linear", "circular"] = "full"

# Valid entanglement topology options.
#
# - "full": All-to-all connectivity, n(n-1)/2 ZZ pairs
# - "linear": Nearest-neighbor, n-1 ZZ pairs
# - "circular": Nearest-neighbor with periodic boundary, n ZZ pairs (for n>2)
_VALID_ENTANGLEMENTS: frozenset[str] = frozenset({"full", "linear", "circular"})

# Threshold for warning about full entanglement with many features.
#
# Full entanglement creates n(n-1)/2 ZZ interaction pairs, each requiring
# 2 CNOT gates. The gate count scales quadratically:
#
#   Two-qubit gates per layer = n(n-1)
#
# Examples (per layer, reps=2):
#   - 4 features:  12 CNOTs (manageable)
#   - 8 features:  56 CNOTs (moderate)
#   - 10 features: 90 CNOTs (threshold - warning issued)
#   - 12 features: 132 CNOTs (may exceed NISQ capabilities)
#   - 16 features: 240 CNOTs (likely impractical on current hardware)
#
# At 10 features, the circuit complexity becomes significant enough to warrant
# a warning. Users should consider 'linear' or 'circular' entanglement for
# larger feature counts, or ensure their target hardware supports the required
# gate count and connectivity.
_FULL_ENTANGLEMENT_WARNING_THRESHOLD: int = 10

# Threshold for DEBUG logging about input values outside optimal range.
# The (π - x) phase convention works best with inputs in [0, 2π].
# Values beyond this threshold (in absolute value) trigger a debug log.
# Set to 4π to allow some flexibility while catching extreme values.
_INPUT_RANGE_DEBUG_THRESHOLD: float = 4.0 * np.pi


# =============================================================================
# Type Definitions
# =============================================================================

class GateCountBreakdown(TypedDict):
    """Type definition for gate count breakdown dictionary.

    Attributes
    ----------
    hadamard : int
        Number of Hadamard gates.
    phase_single : int
        Number of single-qubit phase gates (data encoding).
    phase_zz : int
        Number of phase gates in ZZ decomposition.
    cnot : int
        Number of CNOT gates.
    total_single_qubit : int
        Total single-qubit gates (hadamard + phase_single + phase_zz).
    total_two_qubit : int
        Total two-qubit gates (cnot).
    total : int
        Total gate count.
    """

    hadamard: int
    phase_single: int
    phase_zz: int
    cnot: int
    total_single_qubit: int
    total_two_qubit: int
    total: int


class ZZFeatureMap(BaseEncoding):
    """ZZ Feature Map encoding following Qiskit conventions.

    ZZFeatureMap implements the second-order Pauli-Z expansion feature map
    commonly used in quantum kernel methods and QSVM. It creates entangled
    quantum states by applying Hadamard gates, phase rotations on single
    qubits, and ZZ entangling interactions between pairs of qubits.

    This encoding follows the Qiskit convention for ZZFeatureMap, using the
    phase formula 2(π - xᵢ)(π - xⱼ) for two-qubit interactions, which creates
    a different kernel geometry compared to the standard IQP encoding.

    The circuit structure for each repetition is:

        |0⟩ ─ H ─ P(2x₀) ─╭──────────────╮─╭──────────────╮─
        |0⟩ ─ H ─ P(2x₁) ─│ ZZ(φ₀₁)     │─│              │─
        |0⟩ ─ H ─ P(2x₂) ─╰──────────────╯─│ ZZ(φ₁₂)     │─
        ...                                ╰──────────────╯

    where φᵢⱼ = 2(π - xᵢ)(π - xⱼ) and ZZ interactions are applied according
    to the specified entanglement topology.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
        Each feature requires one qubit, so this also determines the number
        of qubits in the circuit.
    reps : int, default=2
        Number of times to repeat the encoding layers. Higher values create
        deeper circuits that may capture more complex feature relationships.
        Must be at least 1.
    entanglement : {"full", "linear", "circular"}, default="full"
        Topology of ZZ interactions between qubits:

        - "full": All-to-all connectivity. Every pair (i, j) with i < j has a
          ZZ interaction. Creates n(n-1)/2 entangling gates per layer.
        - "linear": Nearest-neighbor connectivity. Only pairs (i, i+1) have
          ZZ interactions. Creates n-1 entangling gates per layer.
        - "circular": Nearest-neighbor with periodic boundary. Like linear,
          but adds a ZZ interaction between qubits n-1 and 0 when n > 2.

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

    Use Cases
    ---------
    **Quantum Kernel Methods (QSVM)**:
        ZZFeatureMap is the canonical choice for quantum kernel estimation
        in supervised learning. The (π-x) phase convention creates kernels
        that can separate data classes that are not linearly separable in
        the original feature space.

    **Quantum Neural Networks**:
        Use as a feature embedding layer before variational ansatz circuits.
        The entanglement creates correlations that parameterized layers
        can learn to exploit.

    **Benchmarking**:
        Standard circuit for comparing quantum machine learning approaches
        across different platforms and simulators.

    **Research**:
        Studying the effect of different phase conventions on kernel geometry
        and quantum advantage in machine learning tasks.

    Limitations
    -----------
    **Scalability**: Full entanglement creates O(n²) ZZ pairs, leading to
    deep circuits for large feature counts. For n > 10 features, consider
    using ``entanglement='linear'`` or ``entanglement='circular'``.

    **Hardware Connectivity**: Full entanglement requires all-to-all qubit
    connectivity, which is not available on most near-term quantum hardware.
    SWAP gates will be inserted during transpilation, increasing circuit depth.

    **Noise Sensitivity**: Deep circuits with many two-qubit gates are more
    susceptible to decoherence and gate errors on NISQ devices.

    **Classical Simulation**: Unlike product-state encodings (e.g., AngleEncoding),
    ZZFeatureMap creates entangled states that cannot be efficiently simulated
    classically for large qubit counts.

    Resource Analysis
    -----------------
    The circuit resources scale as follows per repetition:

    **Full entanglement** (``entanglement='full'``):
        - ZZ pairs: n(n-1)/2
        - CNOT gates: n(n-1) per rep
        - Example: n=10 → 45 pairs, 90 CNOTs per rep

    **Linear entanglement** (``entanglement='linear'``):
        - ZZ pairs: n-1
        - CNOT gates: 2(n-1) per rep
        - Example: n=10 → 9 pairs, 18 CNOTs per rep

    **Circular entanglement** (``entanglement='circular'``):
        - ZZ pairs: n (for n > 2)
        - CNOT gates: 2n per rep
        - Example: n=10 → 10 pairs, 20 CNOTs per rep

    Use ``gate_count_breakdown()`` for detailed gate counts, or
    ``get_entanglement_pairs()`` to inspect the connectivity.

    Entanglement Topology Trade-offs
    --------------------------------
    +------------+------------------+----------------+------------------+
    | Topology   | Connectivity     | ZZ Pairs       | Hardware         |
    +============+==================+================+==================+
    | full       | All-to-all       | n(n-1)/2       | Requires SWAP    |
    |            | (maximum)        | (quadratic)    | gates on most HW |
    +------------+------------------+----------------+------------------+
    | linear     | Nearest-neighbor | n-1            | Native on most   |
    |            | (chain)          | (linear)       | superconducting  |
    +------------+------------------+----------------+------------------+
    | circular   | Nearest-neighbor | n (for n > 2)  | Native on ring   |
    |            | + wrap-around    | (linear)       | topologies       |
    +------------+------------------+----------------+------------------+

    **Recommendation**: Use ``entanglement='linear'`` for NISQ hardware
    experiments to minimize SWAP overhead. Use ``entanglement='full'``
    for simulation studies or when maximum expressivity is needed.

    Examples
    --------
    Create a basic ZZ Feature Map with default settings:

    >>> from encoding_atlas import ZZFeatureMap
    >>> import numpy as np
    >>> enc = ZZFeatureMap(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.entanglement
    'full'

    Generate a PennyLane circuit:

    >>> x = np.array([0.5, 1.0, 1.5, 2.0])
    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Use linear entanglement for hardware-friendly circuits:

    >>> enc_linear = ZZFeatureMap(n_features=4, entanglement='linear')
    >>> enc_linear.properties.two_qubit_gates
    6

    Generate Qiskit circuit (compatible with Qiskit's ZZFeatureMap):

    >>> qiskit_circuit = enc.get_circuit(x, backend='qiskit')
    >>> qiskit_circuit.num_qubits
    4

    Batch processing with parallel execution:

    >>> X = np.random.randn(100, 4)
    >>> circuits = enc.get_circuits(X, backend='pennylane', parallel=True)
    >>> len(circuits)
    100

    Inspect entanglement connectivity:

    >>> enc.get_entanglement_pairs()
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    Get detailed gate count breakdown:

    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['cnot']
    24
    >>> breakdown['total']
    52

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
    .. [2] Qiskit Development Team. "ZZFeatureMap documentation."

    See Also
    --------
    IQPEncoding : Similar IQP-style encoding with direct xᵢxⱼ interactions.
    PauliFeatureMap : Generalized Pauli-based feature map.
    AngleEncoding : Simpler encoding without entanglement.
    DataReuploading : Re-uploads data with trainable intermediate layers.

    Notes
    -----
    **Phase Convention**: The (π - x) convention in ZZ interactions means:

    - Features at x = π have zero interaction strength
    - Features at x = 0 or x = 2π have maximum interaction
    - This creates a kernel with different geometry than standard IQP

    **Comparison with IQPEncoding**:

    - ZZFeatureMap uses P(2x) phase gates; IQP uses RZ(2x)
    - ZZFeatureMap uses (π-xᵢ)(π-xⱼ) for ZZ; IQP uses xᵢxⱼ
    - Both create classically hard-to-simulate circuits
    - Choice depends on data distribution and kernel requirements

    **Qiskit Compatibility**: This implementation follows Qiskit conventions
    and produces equivalent circuits to qiskit.circuit.library.ZZFeatureMap.

    **Thread Safety**: This class is thread-safe for circuit generation.
    Multiple threads can safely call ``get_circuit()`` or ``get_circuits()``
    concurrently on the same encoding instance. The encoding object is not
    modified during circuit generation, and input validation creates
    defensive copies to prevent data races.

    **Circular Entanglement Edge Case**: For n_features=2, circular
    entanglement produces the same connectivity as linear (only one pair).
    The wrap-around edge (n-1, 0) is only added when n > 2 to avoid
    duplicate pairs.
    """

    # =========================================================================
    # Class Constants
    # =========================================================================

    # Memory-efficient slot-based attribute storage.
    # Only instance attributes specific to this class are listed here.
    # Attributes from BaseEncoding are handled by its own __slots__.
    __slots__ = ('reps', 'entanglement', '_entanglement_pairs')

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        n_features: int,
        reps: int = _DEFAULT_REPS,
        entanglement: Literal["full", "linear", "circular"] = _DEFAULT_ENTANGLEMENT,
    ) -> None:
        """Initialize the ZZ Feature Map encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode. Must be a positive integer.
            Each feature requires one qubit, so this also determines the number
            of qubits in the circuit.
        reps : int, default=2
            Number of encoding layer repetitions. Must be at least 1.
        entanglement : {"full", "linear", "circular"}, default="full"
            Topology of ZZ interactions between qubits.

        Raises
        ------
        ValueError
            If reps is not a positive integer.
        ValueError
            If entanglement is not one of "full", "linear", "circular".
        ValueError
            If n_features is less than 1 (raised by parent class).

        Examples
        --------
        >>> enc = ZZFeatureMap(n_features=4)
        >>> enc.n_qubits
        4
        >>> enc.reps
        2
        >>> enc.entanglement
        'full'

        >>> enc = ZZFeatureMap(n_features=8, reps=3, entanglement='linear')
        >>> len(enc.get_entanglement_pairs())
        7
        """
        # =====================================================================
        # REPS VALIDATION
        # =====================================================================
        # Validate reps before calling parent constructor to fail fast.
        # Note: bool is a subclass of int in Python, so we explicitly exclude it
        # to prevent confusing behavior where True would be accepted as reps=1.
        if isinstance(reps, bool):
            raise ValueError(
                f"reps must be a positive integer, got {type(reps).__name__} "
                f"(boolean values are not accepted)"
            )
        if not isinstance(reps, int):
            raise ValueError(
                f"reps must be a positive integer, got {type(reps).__name__}"
            )
        if reps < 1:
            raise ValueError(
                f"reps must be a positive integer (>= 1), got {reps}"
            )

        # =====================================================================
        # ENTANGLEMENT VALIDATION
        # =====================================================================
        if entanglement not in _VALID_ENTANGLEMENTS:
            raise ValueError(
                f"entanglement must be one of {sorted(_VALID_ENTANGLEMENTS)}, "
                f"got {entanglement!r}"
            )

        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        # Call parent constructor with configuration for equality/hashing.
        # The config is stored in BaseEncoding for __eq__ and __hash__.
        super().__init__(n_features, reps=reps, entanglement=entanglement)

        # Store encoding-specific parameters as instance attributes
        self.reps: int = reps
        self.entanglement: Literal["full", "linear", "circular"] = entanglement

        # Pre-compute and cache entanglement pairs at initialization.
        # This is computed once and reused for all circuit generation calls,
        # avoiding redundant O(n²) computation for full entanglement.
        self._entanglement_pairs: list[tuple[int, int]] = (
            self._compute_entanglement_pairs(n_features, entanglement)
        )

        _logger.debug(
            "Entanglement pairs computed: topology=%r, n_qubits=%d, n_pairs=%d",
            entanglement,
            n_features,
            len(self._entanglement_pairs),
        )

        # =====================================================================
        # RESOURCE SCALING WARNING
        # =====================================================================
        # Warn about potentially excessive gate count with full entanglement.
        # This matches the pattern used by IQPEncoding and other encodings.
        if (
            entanglement == "full"
            and n_features > _FULL_ENTANGLEMENT_WARNING_THRESHOLD
        ):
            n_pairs = len(self._entanglement_pairs)
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

        # Log initialization for debugging
        _logger.debug(
            "ZZFeatureMap initialized: n_features=%d, n_qubits=%d, "
            "reps=%d, entanglement=%r",
            self.n_features,
            self.n_qubits,
            self.reps,
            self.entanglement,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        For ZZ Feature Map, each feature is encoded on a dedicated qubit,
        so n_qubits equals n_features.

        Returns
        -------
        int
            Number of qubits, equal to n_features.

        Examples
        --------
        >>> ZZFeatureMap(n_features=4).n_qubits
        4
        >>> ZZFeatureMap(n_features=10).n_qubits
        10
        """
        return self.n_features

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        The depth is computed based on the entanglement topology:

        **Per repetition:**

        1. Hadamard gates (depth 1, all qubits in parallel)
        2. Single-qubit phase gates (depth 1, all qubits in parallel)
        3. ZZ entangling block (depth depends on topology)

        **ZZ block depth by topology:**

        - **Linear/Circular**: Consecutive pairs share qubits, so ZZ
          operations must be sequential. Depth = 3 × n_pairs (each ZZ
          requires CNOT + P + CNOT = 3 layers).

        - **Full**: Non-overlapping pairs can be parallelized. This is
          an edge-coloring problem on the complete graph. The chromatic
          index is (n-1) for even n, or n for odd n. With optimal
          scheduling, depth = 3 × chromatic_index.

        Returns
        -------
        int
            Circuit depth estimate. For linear/circular topologies, this
            is exact. For full entanglement, this assumes optimal
            parallelization (actual depth may be higher on hardware with
            limited connectivity).

        Notes
        -----
        The depth formula per repetition is:

        - Linear: 2 + 3×(n-1)
        - Circular: 2 + 3×n (for n > 2), or 2 + 3 = 5 (for n = 2)
        - Full: 2 + 3×(n-1) for even n, 2 + 3×n for odd n

        After transpilation to specific hardware, actual depth may differ
        due to SWAP gate insertion and native gate decomposition.

        Examples
        --------
        >>> ZZFeatureMap(n_features=4, reps=2, entanglement='linear').depth
        22
        >>> ZZFeatureMap(n_features=4, reps=2, entanglement='full').depth
        22
        >>> ZZFeatureMap(n_features=4, reps=1, entanglement='circular').depth
        14
        """
        n = self.n_features

        # Single-qubit operations: H + P per qubit, all parallel = depth 2
        single_qubit_depth = 2

        # ZZ layer depth depends on topology
        if self.entanglement == "linear":
            # Each pair shares a qubit with the next, must be sequential
            # Each ZZ = CNOT + P + CNOT = depth 3
            n_pairs = n - 1
            zz_depth = 3 * n_pairs
        elif self.entanglement == "circular":
            # Same as linear but with wrap-around pair for n > 2
            n_pairs = n if n > 2 else 1
            zz_depth = 3 * n_pairs
        else:  # full
            # Can parallelize non-overlapping pairs (edge coloring)
            # Complete graph K_n has chromatic index: n-1 for even n, n for odd n
            chromatic_index = n if n % 2 == 1 else n - 1
            zz_depth = 3 * chromatic_index

        return self.reps * (single_qubit_depth + zz_depth)

    # =========================================================================
    # Entanglement Pairs (Public and Private)
    # =========================================================================

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs for ZZ entanglement based on topology.

        Returns the list of qubit pairs that will have ZZ interactions
        applied between them. This is useful for:

        - Inspecting the connectivity pattern
        - Verifying hardware compatibility
        - Understanding circuit structure
        - Resource estimation

        Returns
        -------
        list[tuple[int, int]]
            List of (qubit_i, qubit_j) pairs for ZZ interactions.
            The order is deterministic for reproducibility.

        Notes
        -----
        Pair counts by topology:

        - Full: n(n-1)/2 pairs (all combinations where i < j)
        - Linear: n-1 pairs (nearest neighbors only)
        - Circular: n-1 pairs for n<=2, n pairs for n>2 (adds wrap-around)

        Examples
        --------
        >>> enc = ZZFeatureMap(n_features=4, entanglement='full')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        >>> enc = ZZFeatureMap(n_features=4, entanglement='linear')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        >>> enc = ZZFeatureMap(n_features=4, entanglement='circular')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3), (3, 0)]

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts including ZZ-related gates.
        """
        return self._get_entanglement_pairs()

    @staticmethod
    def _compute_entanglement_pairs(
        n_features: int,
        entanglement: Literal["full", "linear", "circular"],
    ) -> list[tuple[int, int]]:
        """Compute qubit pairs for ZZ entanglement based on topology.

        This is a static method used during initialization to compute the
        entanglement pairs once. The result is cached as an instance attribute.

        Parameters
        ----------
        n_features : int
            Number of features (qubits).
        entanglement : {"full", "linear", "circular"}
            Entanglement topology.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for ZZ interactions.

        Notes
        -----
        Pair counts by topology:

        - Full: n(n-1)/2 pairs (all combinations where i < j)
        - Linear: n-1 pairs (nearest neighbors only)
        - Circular: n-1 pairs for n<=2, n pairs for n>2 (adds wrap-around)
        """
        if entanglement == "full":
            # All-to-all connectivity: every pair (i, j) with i < j
            pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
        elif entanglement == "linear":
            # Nearest-neighbor connectivity: only (i, i+1) pairs
            pairs = [(i, i + 1) for i in range(n_features - 1)]
        else:  # circular
            # Nearest-neighbor with periodic boundary
            pairs = [(i, i + 1) for i in range(n_features - 1)]
            # Add wrap-around only for n > 2 to avoid duplicate pair
            # For n=2, linear already has (0, 1), circular would add (1, 0)
            # which is redundant (same qubits, different order doesn't matter for ZZ)
            if n_features > 2:
                pairs.append((n_features - 1, 0))

        return pairs

    def _get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs for ZZ entanglement (internal accessor).

        Returns the cached entanglement pairs computed at initialization.
        This method provides a consistent internal interface for accessing
        the pairs throughout the class.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for ZZ interactions.
            This returns the cached list directly (not a copy) for
            performance. Callers should not modify the returned list.

        See Also
        --------
        get_entanglement_pairs : Public API that returns the same data.
        _compute_entanglement_pairs : Static method that computes pairs.
        """
        return self._entanglement_pairs

    # =========================================================================
    # Resource Analysis
    # =========================================================================

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get detailed breakdown of gate counts by type.

        Returns a dictionary with counts for each gate type used in the
        circuit. This is useful for:

        - Resource estimation and circuit analysis
        - Comparing different configurations
        - Hardware resource planning
        - Understanding circuit complexity

        Returns
        -------
        GateCountBreakdown
            Dictionary with the following keys:

            - ``hadamard``: Number of Hadamard gates (n * reps)
            - ``phase_single``: Single-qubit phase gates for data encoding
            - ``phase_zz``: Phase gates in ZZ decomposition
            - ``cnot``: Number of CNOT gates (2 per ZZ interaction)
            - ``total_single_qubit``: Sum of all single-qubit gates
            - ``total_two_qubit``: Sum of all two-qubit gates (= cnot)
            - ``total``: Total gate count

        Examples
        --------
        >>> enc = ZZFeatureMap(n_features=4, reps=2)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['hadamard']
        8
        >>> breakdown['cnot']
        24
        >>> breakdown['total']
        52

        Compare topologies:

        >>> enc_full = ZZFeatureMap(n_features=10, entanglement='full')
        >>> enc_linear = ZZFeatureMap(n_features=10, entanglement='linear')
        >>> enc_full.gate_count_breakdown()['cnot']
        180
        >>> enc_linear.gate_count_breakdown()['cnot']
        36

        See Also
        --------
        get_entanglement_pairs : Inspect the qubit pairs for ZZ interactions.
        properties : Get comprehensive encoding properties.
        """
        n = self.n_features
        n_pairs = len(self._get_entanglement_pairs())

        # Gate counts per component
        hadamard = self.reps * n
        phase_single = self.reps * n
        phase_zz = self.reps * n_pairs
        cnot = self.reps * 2 * n_pairs

        # Aggregate counts
        total_single_qubit = hadamard + phase_single + phase_zz
        total_two_qubit = cnot
        total = total_single_qubit + total_two_qubit

        return GateCountBreakdown(
            hadamard=hadamard,
            phase_single=phase_single,
            phase_zz=phase_zz,
            cnot=cnot,
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
            - ``depth``: Circuit depth (layer count)
            - ``reps``: Number of encoding repetitions
            - ``entanglement``: Entanglement topology ("full", "linear", "circular")
            - ``entanglement_pairs``: List of qubit pairs with ZZ interactions
            - ``n_entanglement_pairs``: Number of ZZ interaction pairs per layer
            - ``gate_counts``: Detailed breakdown from gate_count_breakdown()
            - ``is_entangling``: Always True for ZZ Feature Map
            - ``simulability``: "not_simulable" (hard classical simulation)
            - ``trainability_estimate``: Estimated trainability (0.0 to 1.0)
            - ``phase_convention``: Description of the (π-x) phase convention
            - ``hardware_requirements``: Dict with connectivity and gate requirements
            - ``qiskit_compatible``: Always True (follows Qiskit ZZFeatureMap convention)

        Examples
        --------
        Get complete resource analysis:

        >>> enc = ZZFeatureMap(n_features=4, reps=2, entanglement='full')
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['n_entanglement_pairs']
        6
        >>> summary['gate_counts']['cnot']
        24

        Compare different entanglement topologies:

        >>> enc_full = ZZFeatureMap(n_features=8, entanglement='full')
        >>> enc_linear = ZZFeatureMap(n_features=8, entanglement='linear')
        >>> enc_full.resource_summary()['n_entanglement_pairs']
        28
        >>> enc_linear.resource_summary()['n_entanglement_pairs']
        7

        Use for hardware planning:

        >>> enc = ZZFeatureMap(n_features=6, entanglement='full')
        >>> summary = enc.resource_summary()
        >>> print(f"Requires {summary['hardware_requirements']['connectivity']}")
        Requires all-to-all
        >>> print(f"Total two-qubit gates: {summary['gate_counts']['total_two_qubit']}")
        Total two-qubit gates: 60

        Check Qiskit compatibility:

        >>> enc = ZZFeatureMap(n_features=4)
        >>> summary = enc.resource_summary()
        >>> summary['qiskit_compatible']
        True
        >>> summary['phase_convention']
        '2(π - xᵢ)(π - xⱼ) for ZZ interactions'

        Notes
        -----
        This method combines information from multiple sources:

        - ``properties``: Theoretical encoding characteristics
        - ``gate_count_breakdown()``: Detailed gate analysis
        - ``get_entanglement_pairs()``: Entanglement structure

        The ZZ Feature Map follows Qiskit's convention for the ZZFeatureMap class,
        using the phase formula 2(π - xᵢ)(π - xⱼ) for two-qubit interactions.
        This creates a different kernel geometry compared to IQP encoding's
        direct xᵢxⱼ product.

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
            # Gate counts (using GateCountBreakdown TypedDict)
            "gate_counts": dict(gate_counts),
            # Encoding characteristics
            "is_entangling": True,
            "simulability": "not_simulable",
            "trainability_estimate": props.trainability_estimate,
            # ZZ Feature Map specific information
            "phase_convention": "2(π - xᵢ)(π - xⱼ) for ZZ interactions",
            "qiskit_compatible": True,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": connectivity,
                "native_gates": ["H", "P", "CNOT"],
                "min_two_qubit_gate_fidelity": 0.99,  # Recommended for NISQ
            },
        }

        _logger.debug(
            "Resource summary: n_qubits=%d, depth=%d, total_gates=%d, "
            "entanglement=%s, pairs=%d",
            self.n_qubits,
            self.depth,
            gate_counts['total'],
            self.entanglement,
            len(pairs),
        )

        return summary

    # =========================================================================
    # Circuit Generation
    # =========================================================================

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for a single data sample.

        Creates a quantum circuit that encodes the input features using
        Hadamard gates, phase rotations, and ZZ entangling interactions
        with the (π - x) phase convention.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (1, n_features).
            Values are used as rotation angles (in radians). For optimal
            encoding, scale features to [0, 2π] or [-π, π].
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
            If backend is not one of the supported options.
        ImportError
            If the requested backend is not installed.

        Examples
        --------
        >>> enc = ZZFeatureMap(n_features=4)
        >>> x = np.array([0.5, 1.0, 1.5, 2.0])
        >>> circuit = enc.get_circuit(x, backend='pennylane')
        >>> callable(circuit)
        True

        >>> qc = enc.get_circuit(x, backend='qiskit')
        >>> qc.num_qubits
        4

        Notes
        -----
        The circuit prepares the state:

            |ψ(x)⟩ = [U_ZZ(x) · H^⊗n]^reps |0⟩^⊗n

        where U_ZZ applies single-qubit phase gates P(2xᵢ) and two-qubit
        ZZ interactions with angle 2(π - xᵢ)(π - xⱼ).
        """
        _logger.debug(
            "get_circuit called: backend=%r, input_shape=%s",
            backend,
            getattr(x, 'shape', f'len={len(x)}' if hasattr(x, '__len__') else 'scalar'),
        )

        # Validate and preprocess input
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]

        # Debug log if values are far outside the optimal [0, 2π] range.
        # The (π - x) phase convention has specific geometric meaning in that range.
        # Values outside still work (rotations are periodic) but may produce
        # unexpected kernel geometry.
        if _logger.isEnabledFor(logging.DEBUG):
            x_min, x_max = float(x_validated.min()), float(x_validated.max())
            if abs(x_min) > _INPUT_RANGE_DEBUG_THRESHOLD or abs(x_max) > _INPUT_RANGE_DEBUG_THRESHOLD:
                _logger.debug(
                    "Input values [%.3g, %.3g] are outside optimal range [0, 2π]. "
                    "The (π - x) phase convention works best with scaled inputs. "
                    "Consider normalizing features to [0, 2π] or [-π, π].",
                    x_min,
                    x_max,
                )

        # Dispatch to backend-specific implementation
        if backend == "pennylane":
            circuit = self._to_pennylane(x_validated)
            _logger.debug("PennyLane circuit generated: n_qubits=%d", self.n_qubits)
            return circuit
        elif backend == "qiskit":
            circuit = self._to_qiskit(x_validated)
            _logger.debug("Qiskit circuit generated: n_qubits=%d", self.n_qubits)
            return circuit
        elif backend == "cirq":
            circuit = self._to_cirq(x_validated)
            _logger.debug("Cirq circuit generated: n_qubits=%d", self.n_qubits)
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
            circuit generation. This can speed up processing for large
            batches (>100 samples) but adds overhead for small batches.

            Parallel processing is thread-safe due to ZZFeatureMap's
            stateless circuit generation design.
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

        >>> enc = ZZFeatureMap(n_features=4)
        >>> X = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        >>> circuits = enc.get_circuits(X, backend='pennylane')
        >>> len(circuits)
        2

        Parallel processing for large batches:

        >>> enc = ZZFeatureMap(n_features=4)
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
            # Use optimized internal method to avoid re-validation
            circuits = [
                self._get_circuit_from_validated(x, backend)
                for x in X_validated
            ]

            _logger.debug(
                "Sequential batch processing completed: generated %d circuits",
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
        when processing batches.

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

        Raises
        ------
        ValueError
            If backend is not one of the supported options.
        ImportError
            If the requested backend is not installed.

        Notes
        -----
        This is an internal optimization method. External callers should use
        ``get_circuit()`` which includes proper input validation.
        """
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

    # =========================================================================
    # Backend-Specific Implementations
    # =========================================================================

    def _to_pennylane(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate PennyLane circuit function.

        Creates a callable that applies the ZZ Feature Map encoding gates
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

        Notes
        -----
        The returned function captures all necessary values in its closure,
        making it self-contained and thread-safe. The closure captures:

        - x: Copy of input data (immutable due to validation)
        - n_qubits: Number of qubits
        - reps: Number of repetitions
        - pairs: Entanglement pairs (cached)

        This ensures the circuit function can be safely used even if the
        encoding object or original input array is modified after creation.
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
        pairs = self._get_entanglement_pairs()
        n_qubits = self.n_qubits
        reps = self.reps
        # Create a copy of x to ensure closure captures immutable data
        x_captured = x.copy()

        def circuit() -> None:
            """Apply the ZZ Feature Map encoding gates."""
            for _ in range(reps):
                # Hadamard layer - create superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                # Single-qubit phase rotations
                for i in range(n_qubits):
                    qml.PhaseShift(2 * x_captured[i], wires=i)
                # ZZ interactions with (π - x) convention
                for i, j in pairs:
                    qml.CNOT(wires=[i, j])
                    qml.PhaseShift(
                        2 * (np.pi - x_captured[i]) * (np.pi - x_captured[j]),
                        wires=j,
                    )
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

        Notes
        -----
        This produces circuits equivalent to Qiskit's built-in ZZFeatureMap.
        The circuit uses:

        - H gates for creating superposition
        - P (phase) gates for single-qubit rotations: P(2xᵢ)
        - CX (CNOT) gates for ZZ decomposition
        - P gates for ZZ interaction: P(2(π-xᵢ)(π-xⱼ))

        The resulting circuit can be directly used with Qiskit's simulators
        and hardware backends.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError(
                "Qiskit is required for the 'qiskit' backend. "
                "Install it with: pip install qiskit"
            ) from e

        pairs = self._get_entanglement_pairs()
        qc = QuantumCircuit(self.n_qubits, name="ZZFeatureMap")

        for _ in range(self.reps):
            # Hadamard layer
            for i in range(self.n_qubits):
                qc.h(i)
            # Single-qubit phase rotations
            for i in range(self.n_qubits):
                qc.p(2 * float(x[i]), i)
            # ZZ interactions with (π - x) convention
            for i, j in pairs:
                angle = 2 * (np.pi - float(x[i])) * (np.pi - float(x[j]))
                qc.cx(i, j)
                qc.p(angle, j)
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
            Cirq circuit implementing the ZZ Feature Map encoding.

        Raises
        ------
        ImportError
            If Cirq is not installed.

        Notes
        -----
        Cirq uses RZ gates to implement the phase rotations. RZ and P (phase)
        gates differ by a global phase that doesn't affect measurement
        outcomes, so the circuits are functionally equivalent.

        The circuit structure mirrors the PennyLane and Qiskit implementations:

        - H gates for superposition
        - RZ gates for single-qubit phases
        - CNOT-RZ-CNOT for ZZ interactions

        This ensures cross-backend consistency in quantum state preparation.
        """
        try:
            import cirq
        except ImportError as e:
            raise ImportError(
                "Cirq is required for the 'cirq' backend. "
                "Install it with: pip install cirq-core"
            ) from e

        pairs = self._get_entanglement_pairs()
        qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()

        for _ in range(self.reps):
            # Hadamard layer
            circuit.append([cirq.H(q) for q in qubits])

            # Phase gates for single-qubit rotations
            for i in range(self.n_qubits):
                circuit.append(cirq.rz(2 * float(x[i]))(qubits[i]))

            # ZZ interactions using CNOT-Phase-CNOT decomposition
            for i, j in pairs:
                angle = 2 * (np.pi - float(x[i])) * (np.pi - float(x[j]))
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))
                circuit.append(cirq.rz(angle)(qubits[j]))
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))

        return circuit

    # =========================================================================
    # Properties Computation
    # =========================================================================

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Calculates circuit metrics including qubit count, depth, gate counts,
        and encoding characteristics for theoretical analysis.

        Returns
        -------
        EncodingProperties
            Computed properties including:

            - n_qubits: Number of qubits (= n_features)
            - depth: Circuit depth (= 4 * reps)
            - gate_count: Total gates including H, P, and CNOT
            - single_qubit_gates: H gates + P gates (single and ZZ)
            - two_qubit_gates: CNOT gates (2 per ZZ interaction)
            - is_entangling: Always True (ZZ interactions create entanglement)
            - simulability: "not_simulable" (hard classical simulation)

        Notes
        -----
        Gate count breakdown per repetition:

        - Hadamard gates: n (one per qubit)
        - Single-qubit P gates: n (one per qubit for data encoding)
        - ZZ interactions: n_pairs (depends on entanglement topology)
        - Each ZZ uses: 2 CNOTs + 1 P gate

        Total per rep: n + n + n_pairs + 2*n_pairs = 2n + 3*n_pairs

        For full entanglement: n_pairs = n(n-1)/2
        For linear entanglement: n_pairs = n-1
        For circular entanglement: n_pairs = n (for n > 2)
        """
        breakdown = self.gate_count_breakdown()

        return EncodingProperties(
            n_qubits=self.n_features,
            depth=self.depth,
            gate_count=breakdown['total'],
            single_qubit_gates=breakdown['total_single_qubit'],
            two_qubit_gates=breakdown['total_two_qubit'],
            parameter_count=self.reps * (self.n_features + len(self._get_entanglement_pairs())),
            is_entangling=True,
            simulability="not_simulable",
            trainability_estimate=max(0.3, 0.85 - 0.1 * self.reps),
            notes=(
                f"ZZ Feature Map with {self.entanglement} entanglement, "
                f"{self.reps} rep(s). Qiskit-compatible (pi-x) phase convention. "
                f"Creates {len(self._get_entanglement_pairs())} ZZ interactions per layer."
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
            String representation showing all configuration parameters.

        Examples
        --------
        >>> enc = ZZFeatureMap(n_features=4)
        >>> repr(enc)
        "ZZFeatureMap(n_features=4, reps=2, entanglement='full')"

        >>> enc = ZZFeatureMap(n_features=8, reps=3, entanglement='linear')
        >>> repr(enc)
        "ZZFeatureMap(n_features=8, reps=3, entanglement='linear')"
        """
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, "
            f"reps={self.reps}, "
            f"entanglement={self.entanglement!r})"
        )
