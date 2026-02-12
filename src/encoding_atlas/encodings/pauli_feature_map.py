"""Pauli Feature Map encoding module.

This module implements PauliFeatureMap, a highly configurable quantum data
encoding that uses arbitrary Pauli rotation gates to encode classical features
into quantum states. It generalizes simpler encodings like ZZFeatureMap by
allowing users to specify which Pauli terms (Z, ZZ, Y, YY, XZ, etc.) to include.

The encoding creates quantum states by:
1. Applying Hadamard gates to create superposition
2. Applying configurable Pauli rotations based on input features
3. Repeating for multiple layers (reps)

Mathematical Background
-----------------------
The PauliFeatureMap circuit structure for each layer consists of:

1. **Hadamard Layer**: H^⊗n creates uniform superposition
2. **Single-Qubit Paulis**: Parameterized rotations R_P(2xᵢ) for P ∈ {X, Y, Z}
3. **Two-Qubit Paulis**: Entangling rotations exp(-iφP⊗P) for pairs (i,j)

For single-qubit Pauli terms (e.g., "Z", "Y", "X"), rotations use:

    R_P(φ) = exp(-i φ/2 P)

where φ = 2xᵢ and P is the corresponding Pauli matrix.

For two-qubit Pauli terms (e.g., "ZZ", "XX", "YY", "XY"), the rotation is:

    exp(-i φ/2 (P_a ⊗ P_b))

where φ = 2(π - xᵢ)(π - xⱼ). This is decomposed using:

    exp(-i φ/2 P_a⊗P_b) = BasisChange · CNOT · R_Z(φ) · CNOT · BasisChange†

The basis change depends on the Pauli types:
- Z: No change needed (computational basis)
- X: Hadamard (H) gate
- Y: Sdg-H (rotate to Y-basis)

Note on Data Preprocessing
--------------------------
For PauliFeatureMap encoding, input features are typically:

- Scaled to [0, 2π] for effective phase encoding
- Standardized if features have very different scales
- The two-qubit interactions encode (π - xᵢ)(π - xⱼ), so feature scaling
  affects the strength of learned interactions

Use Cases
---------
PauliFeatureMap is particularly suited for:

- **Quantum kernel methods**: Computing quantum kernels with custom Pauli
  structures that may better match specific problem geometries
- **Research and experimentation**: Exploring which Pauli terms provide
  best expressivity for a given dataset
- **Hardware-aware design**: Selecting Pauli terms that decompose efficiently
  on target hardware (e.g., avoiding Y-rotations on some devices)
- **Variational quantum classifiers**: As a configurable feature map layer
  in hybrid quantum-classical models
- **Generalized quantum neural networks**: Building blocks for QNN architectures
  that extend beyond standard IQP or ZZ structures

Limitations
-----------
- **Qubit scaling**: Requires n qubits for n features (linear scaling)
- **Gate count**: Full entanglement with many Pauli terms creates O(n²)
  two-qubit gates per layer, which may exceed NISQ device capabilities
- **Trainability**: Deep circuits with extensive entanglement may exhibit
  barren plateaus, making gradient-based optimization challenging
- **Hardware constraints**: Full entanglement requires all-to-all qubit
  connectivity; Y-rotations require additional basis changes
- **Complexity**: More configuration options than simpler encodings like
  ZZFeatureMap; requires careful tuning for optimal performance

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation and entanglement pattern computation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.pauli_feature_map').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters (n_features, reps, paulis, entanglement)
- Entanglement pair computation details
- Circuit generation progress for each backend
- Gate count breakdowns
- Input value range warnings

Resource Analysis
-----------------
PauliFeatureMap provides methods to analyze circuit resources:

- ``get_entanglement_pairs()``: Inspect which qubit pairs have interactions
- ``gate_count_breakdown()``: Get detailed gate counts by type
- ``resource_summary()``: Comprehensive resource analysis
- ``properties``: Access computed circuit properties (depth, gate counts, etc.)

These methods help understand resource requirements before execution:

    >>> enc = PauliFeatureMap(n_features=8, paulis=["Z", "ZZ"], entanglement='full')
    >>> len(enc.get_entanglement_pairs())  # Number of ZZ interactions
    28
    >>> enc.gate_count_breakdown()
    {'hadamard': 16, 'rx': 0, 'ry': 0, 'rz': 16, 'rz_two_qubit': 56, ...}

Entanglement Topology Trade-offs
--------------------------------
+------------+------------------+---------------+------------------------+
| Topology   | Gate Scaling     | Connectivity  | Best For               |
+============+==================+===============+========================+
| full       | O(n²)            | All-to-all    | Max expressivity       |
+------------+------------------+---------------+------------------------+
| linear     | O(n)             | Nearest-only  | NISQ devices           |
+------------+------------------+---------------+------------------------+
| circular   | O(n)             | Ring          | Periodic problems      |
+------------+------------------+---------------+------------------------+

References
----------
.. [1] Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
       feature spaces." Nature, 567(7747), 209-212.
.. [2] Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature
       Hilbert Spaces." Physical Review Letters, 122(4), 040504.
.. [3] McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
       network training landscapes." Nature Communications, 9, 4812.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

if TYPE_CHECKING:
    pass

# =============================================================================
# Module-Level Logger
# =============================================================================

# Configure module-level logger for production debugging.
#
# Usage:
#   To enable debug logging for Pauli feature map encoding:
#
#   >>> import logging
#   >>> logging.getLogger('encoding_atlas.encodings.pauli_feature_map').setLevel(logging.DEBUG)
#   >>> handler = logging.StreamHandler()
#   >>> handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
#   >>> logging.getLogger('encoding_atlas.encodings.pauli_feature_map').addHandler(handler)
#
# Log Levels:
#   - DEBUG: Detailed information for diagnosing issues (entanglement pairs,
#            gate counts, circuit generation steps, input value ranges)
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

__all__ = ["PauliFeatureMap"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default number of repetitions for Pauli feature map encoding layers.
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

# Default Pauli terms to use when none are specified.
#
# ["Z", "ZZ"] is equivalent to ZZFeatureMap and provides a good baseline:
#   - Z rotations encode individual feature values
#   - ZZ interactions capture pairwise feature correlations
#
# This matches the structure used in seminal quantum ML papers.
_DEFAULT_PAULIS: list[str] = ["Z", "ZZ"]

# Default entanglement topology.
#
# 'full' entanglement provides maximum expressivity by creating interactions
# between all pairs of qubits. While this maximizes the encoding's ability to
# capture feature correlations, it also:
#   - Creates O(n²) two-qubit gates
#   - Requires all-to-all qubit connectivity
#   - May face trainability challenges for deep circuits
#
# For hardware-constrained applications, consider 'linear' or 'circular'.
_DEFAULT_ENTANGLEMENT: Literal["full", "linear", "circular"] = "full"

# Threshold for warning about full entanglement with many features.
#
# Full entanglement creates n(n-1)/2 interaction pairs per two-qubit Pauli term,
# each requiring 2 CNOT gates. The gate count scales quadratically:
#
#   Two-qubit gates per layer per Pauli term = n(n-1)
#
# Examples (per layer, per two-qubit Pauli):
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

# Threshold for DEBUG logging about input values outside optimal range.
# The Pauli feature map works best with inputs in [0, 2π].
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
        Number of Hadamard gates per repetition.
    rx : int
        Number of RX gates (for X Pauli terms).
    ry : int
        Number of RY gates (for Y Pauli terms).
    rz : int
        Number of RZ gates (for Z Pauli terms).
    rz_two_qubit : int
        Number of RZ gates in two-qubit Pauli decompositions.
    cnot : int
        Number of CNOT gates (2 per two-qubit Pauli interaction).
    basis_change : int
        Number of basis change gates (H, S, Sdg for X and Y Paulis).
    total_single_qubit : int
        Total single-qubit gates.
    total_two_qubit : int
        Total two-qubit gates (= cnot).
    total : int
        Total gate count.
    """

    hadamard: int
    rx: int
    ry: int
    rz: int
    rz_two_qubit: int
    cnot: int
    basis_change: int
    total_single_qubit: int
    total_two_qubit: int
    total: int


class PauliFeatureMap(BaseEncoding):
    """Configurable Pauli feature map encoding for quantum machine learning.

    PauliFeatureMap encodes classical data into quantum states using a
    configurable set of Pauli rotation gates. This provides flexibility
    to tailor the encoding to specific problem structures.

    The circuit structure for each repetition consists of:
    1. A Hadamard layer creating uniform superposition
    2. Pauli rotation layers as specified by the `paulis` parameter

    For single-qubit Paulis (e.g., "Z", "Y", "X"), rotations R_P(2φ(x_i))
    are applied where φ is a feature mapping function.

    For two-qubit Paulis (e.g., "ZZ", "YY", "XX"), entangling rotations
    exp(-i φ(x_i, x_j) P⊗P) are applied using the specified entanglement
    pattern.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
    reps : int, default=2
        Number of repetitions (layers) of the feature map circuit.
        Higher values increase expressibility but also circuit depth.
        Must be at least 1.
    paulis : list[str] or None, default=None
        List of Pauli strings specifying which rotations to apply.
        Valid single-qubit terms: "X", "Y", "Z"
        Valid two-qubit terms: "XX", "YY", "ZZ", "XY", "XZ", "YZ"
        If None, defaults to ["Z", "ZZ"] (equivalent to ZZFeatureMap).
    entanglement : {"full", "linear", "circular"}, default="full"
        Pattern for two-qubit interactions:
        - "full": All pairs (i, j) where i < j
        - "linear": Consecutive pairs (i, i+1)
        - "circular": Linear plus (n-1, 0)

    Attributes
    ----------
    reps : int
        Number of circuit repetitions.
    paulis : list[str]
        List of Pauli terms used in the encoding.
    entanglement : str
        Entanglement pattern for two-qubit gates.

    Examples
    --------
    Create a basic Pauli feature map with default settings:

    >>> from encoding_atlas import PauliFeatureMap
    >>> import numpy as np
    >>> enc = PauliFeatureMap(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.paulis
    ['Z', 'ZZ']

    Create a custom feature map with Y and YY rotations:

    >>> enc = PauliFeatureMap(n_features=4, paulis=["Y", "YY"], reps=3)
    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')

    Use linear entanglement for hardware-friendly circuits:

    >>> enc_full = PauliFeatureMap(n_features=8, entanglement="full")
    >>> enc_linear = PauliFeatureMap(n_features=8, entanglement="linear")
    >>> enc_linear.properties.two_qubit_gates < enc_full.properties.two_qubit_gates
    True

    Batch processing with parallel execution:

    >>> X = np.random.randn(100, 4)
    >>> circuits = enc.get_circuits(X, backend='pennylane', parallel=True)
    >>> len(circuits)
    100

    Inspect entanglement connectivity:

    >>> enc = PauliFeatureMap(n_features=4)
    >>> enc.get_entanglement_pairs()
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    Get detailed gate count breakdown:

    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['cnot']
    24
    >>> breakdown['total']
    52

    References
    ----------
    .. [1] Havlíček, V., et al. (2019). "Supervised learning with quantum-
           enhanced feature spaces." Nature, 567(7747), 209-212.

    See Also
    --------
    ZZFeatureMap : Specialized version with fixed Z and ZZ paulis.
    IQPEncoding : Similar encoding with IQP-style structure.
    AngleEncoding : Simpler encoding without entanglement.

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
    **Feature Interactions**: The two-qubit Pauli gates encode products of
    features using the (π - xᵢ)(π - xⱼ) convention, allowing the quantum
    kernel to capture pairwise correlations.

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

    **Pauli Term Trade-offs**:

    +-------+-------------------+------------------------------------------+
    | Pauli | Gate Overhead     | Notes                                    |
    +=======+===================+==========================================+
    | Z     | Minimal           | Native on most hardware                  |
    +-------+-------------------+------------------------------------------+
    | X     | Low (1 H gate)    | Requires basis change                    |
    +-------+-------------------+------------------------------------------+
    | Y     | Medium (H + S)    | Requires two basis change gates          |
    +-------+-------------------+------------------------------------------+
    | ZZ    | 2 CNOTs           | Standard for IQP/ZZFeatureMap            |
    +-------+-------------------+------------------------------------------+
    | XX    | 2 CNOTs + 4 H     | Basis change adds 4 single-qubit gates   |
    +-------+-------------------+------------------------------------------+
    | YY    | 2 CNOTs + 8 gates | Most overhead due to Y-basis changes     |
    +-------+-------------------+------------------------------------------+

    **Trainability**: Deeper circuits (higher reps) with full entanglement
    and many Pauli terms may exhibit barren plateaus, where gradients become
    exponentially small. Mitigation strategies include:

    - Using fewer repetitions (reps=1 or 2)
    - Using sparser entanglement (linear or circular)
    - Selecting fewer Pauli terms
    - Layer-wise training approaches

    **Thread Safety**: This class is thread-safe for circuit generation.
    Multiple threads can safely call ``get_circuit()`` or ``get_circuits()``
    concurrently on the same encoding instance.
    """

    # Valid Pauli terms
    _VALID_SINGLE_PAULIS: frozenset[str] = frozenset({"X", "Y", "Z"})
    _VALID_TWO_PAULIS: frozenset[str] = frozenset(
        {"XX", "YY", "ZZ", "XY", "XZ", "YZ", "YX", "ZX", "ZY"}
    )
    _VALID_PAULIS: frozenset[str] = _VALID_SINGLE_PAULIS | _VALID_TWO_PAULIS

    # Valid entanglement patterns
    _VALID_ENTANGLEMENT: frozenset[str] = frozenset({"full", "linear", "circular"})

    # Memory-efficient slot-based attribute storage.
    # Only instance attributes specific to this class are listed here.
    # Attributes from BaseEncoding are handled by its own __slots__.
    __slots__ = (
        "reps",
        "paulis",
        "entanglement",
        "_entanglement_pairs",
        "_single_paulis",
        "_two_paulis",
    )

    def __init__(
        self,
        n_features: int,
        reps: int = _DEFAULT_REPS,
        paulis: list[str] | None = None,
        entanglement: Literal["full", "linear", "circular"] = _DEFAULT_ENTANGLEMENT,
    ) -> None:
        """Initialize the Pauli feature map encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        reps : int, default=2
            Number of repetitions of the feature map circuit.
        paulis : list[str] or None, default=None
            List of Pauli strings. Defaults to ["Z", "ZZ"].
        entanglement : {"full", "linear", "circular"}, default="full"
            Entanglement pattern for two-qubit gates.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        TypeError
            If paulis is not a list or None.

        Warns
        -----
        UserWarning
            If using full entanglement with more than 12 features, as the
            circuit complexity may exceed practical NISQ device limits.
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
        if not isinstance(reps, int) or reps < 1:
            raise ValueError(f"reps must be a positive integer, got {reps!r}")

        # =====================================================================
        # ENTANGLEMENT VALIDATION
        # =====================================================================
        if entanglement not in self._VALID_ENTANGLEMENT:
            raise ValueError(
                f"entanglement must be one of {sorted(self._VALID_ENTANGLEMENT)}, "
                f"got {entanglement!r}"
            )

        # =====================================================================
        # PAULIS VALIDATION
        # =====================================================================
        if paulis is None:
            paulis = list(_DEFAULT_PAULIS)  # Copy to avoid mutation
        else:
            # Validate paulis is a list
            if not isinstance(paulis, list):
                raise TypeError(
                    f"paulis must be a list of strings or None, got {type(paulis).__name__}"
                )
            # Validate each pauli term
            paulis = [p.upper() for p in paulis]  # Normalize to uppercase
            for p in paulis:
                if p not in self._VALID_PAULIS:
                    raise ValueError(
                        f"Invalid Pauli term {p!r}. Valid terms are: "
                        f"{sorted(self._VALID_PAULIS)}"
                    )
            # Remove duplicates while preserving order
            seen: set[str] = set()
            unique_paulis: list[str] = []
            for p in paulis:
                if p not in seen:
                    seen.add(p)
                    unique_paulis.append(p)
            paulis = unique_paulis

        if len(paulis) == 0:
            raise ValueError("paulis list cannot be empty")

        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        # Call parent constructor with configuration for equality/hashing.
        super().__init__(
            n_features,
            reps=reps,
            paulis=paulis,
            entanglement=entanglement,
        )

        # Store encoding-specific parameters as instance attributes
        self.reps: int = reps
        self.paulis: list[str] = paulis
        self.entanglement: Literal["full", "linear", "circular"] = entanglement

        # Precompute entanglement pairs for efficiency
        self._entanglement_pairs: list[tuple[int, int]] = (
            self._compute_entanglement_pairs()
        )

        # Categorize paulis for circuit construction
        self._single_paulis: list[str] = [p for p in paulis if len(p) == 1]
        self._two_paulis: list[str] = [p for p in paulis if len(p) == 2]

        # Log initialization
        _logger.debug(
            "PauliFeatureMap initialized: n_features=%d, reps=%d, "
            "paulis=%r, entanglement=%r, n_pairs=%d",
            n_features,
            reps,
            paulis,
            entanglement,
            len(self._entanglement_pairs),
        )

        # =====================================================================
        # RESOURCE SCALING WARNING
        # =====================================================================
        # Warn about potentially excessive gate count with full entanglement
        if (
            entanglement == "full"
            and n_features > _FULL_ENTANGLEMENT_WARNING_THRESHOLD
            and len(self._two_paulis) > 0
        ):
            n_pairs = len(self._entanglement_pairs)
            n_two_paulis = len(self._two_paulis)
            cnot_count = 2 * n_pairs * n_two_paulis * reps
            warnings.warn(
                f"Full entanglement with {n_features} features and "
                f"{n_two_paulis} two-qubit Pauli term(s) creates "
                f"{n_pairs} interaction pairs per layer ({cnot_count} total "
                f"CNOT gates for {reps} reps). This may exceed practical limits "
                f"for NISQ devices. Consider using entanglement='linear' or "
                f"'circular' for better hardware compatibility.",
                UserWarning,
                stacklevel=2,
            )
            _logger.warning(
                "Large feature count with full entanglement: %d features, "
                "%d two-qubit Paulis, %d CNOT gates total",
                n_features,
                n_two_paulis,
                cnot_count,
            )

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        Returns
        -------
        int
            Number of qubits, equal to number of features.
        """
        return self._n_features

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        The depth depends on the Pauli terms and number of repetitions.
        Each repetition contributes:
        - 1 layer for Hadamards
        - 1 layer per single-qubit Pauli type
        - 3 layers per two-qubit Pauli type (decomposed as CNOT-RZ-CNOT)

        Note: This is an estimate assuming sequential execution of each
        Pauli type. Actual depth may vary based on parallelization.

        Returns
        -------
        int
            Total circuit depth.
        """
        # Each repetition has:
        # - 1 layer: Hadamard gates (parallel on all qubits)
        # - 1 layer per single-qubit Pauli (parallel on all qubits)
        # - 3 layers per two-qubit Pauli (CNOT-RZ-CNOT, sequential for each pair)
        depth_per_rep = 1  # Hadamard layer
        depth_per_rep += len(self._single_paulis)  # Single-qubit rotations
        depth_per_rep += 3 * len(self._two_paulis)  # Two-qubit interactions
        return self.reps * depth_per_rep

    def _compute_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Compute qubit pairs for two-qubit interactions.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs.
        """
        n = self._n_features

        if n < 2:
            return []

        if self.entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        elif self.entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        elif self.entanglement == "circular":
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs.append((n - 1, 0))
            return pairs
        else:
            # Should never reach here due to validation
            raise ValueError(f"Unknown entanglement: {self.entanglement}")

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs for two-qubit Pauli interactions.

        Returns the list of qubit pairs that will have two-qubit Pauli
        interactions applied based on the configured entanglement topology.
        This is useful for:

        - Understanding circuit structure before generation
        - Verifying hardware connectivity requirements
        - Debugging entanglement patterns
        - Computing resource requirements

        Returns
        -------
        list[tuple[int, int]]
            List of (qubit_i, qubit_j) pairs for two-qubit interactions.
            Each tuple (i, j) indicates entangling gates between qubits i and j.

        Examples
        --------
        >>> enc = PauliFeatureMap(n_features=4, entanglement='full')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        >>> enc_linear = PauliFeatureMap(n_features=4, entanglement='linear')
        >>> enc_linear.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        >>> enc_circular = PauliFeatureMap(n_features=4, entanglement='circular')
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
        # Return a copy to prevent external modification
        return list(self._entanglement_pairs)

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type.

        Computes the exact number of each gate type in the PauliFeatureMap
        circuit. This is useful for:

        - Resource estimation before circuit execution
        - Hardware compatibility assessment
        - Comparing different encoding configurations
        - Debugging and verification

        Returns
        -------
        GateCountBreakdown
            Dictionary with gate counts:

            - ``'hadamard'``: Number of Hadamard gates (n_qubits × reps)
            - ``'rx'``: Number of RX gates (for X Pauli)
            - ``'ry'``: Number of RY gates (for Y Pauli)
            - ``'rz'``: Number of RZ gates (for Z Pauli, single-qubit)
            - ``'rz_two_qubit'``: RZ gates in two-qubit decomposition
            - ``'cnot'``: Number of CNOT gates (2 per two-qubit interaction)
            - ``'basis_change'``: H, S, Sdg gates for non-Z Paulis
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Total two-qubit gates (= cnot)
            - ``'total'``: Total gate count

        Examples
        --------
        >>> enc = PauliFeatureMap(n_features=4, reps=1, paulis=["Z", "ZZ"])
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['hadamard']
        4
        >>> breakdown['cnot']
        12
        >>> breakdown['total']
        26

        Compare Pauli configurations:

        >>> enc_z = PauliFeatureMap(n_features=4, paulis=["Z"])
        >>> enc_xyz = PauliFeatureMap(n_features=4, paulis=["X", "Y", "Z"])
        >>> enc_z.gate_count_breakdown()['total']
        16
        >>> enc_xyz.gate_count_breakdown()['total']
        32

        See Also
        --------
        get_entanglement_pairs : Get the qubit pairs for interactions.
        properties : Access all computed circuit properties.
        resource_summary : Get comprehensive resource analysis.
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        # Count gates by type
        h_gates = self.reps * n

        # Single-qubit rotations (one per qubit per single-qubit Pauli per rep)
        rx_gates = self.reps * n * self._single_paulis.count("X")
        ry_gates = self.reps * n * self._single_paulis.count("Y")
        rz_gates = self.reps * n * self._single_paulis.count("Z")

        # Two-qubit interactions
        n_two_paulis = len(self._two_paulis)
        rz_two_qubit = self.reps * n_pairs * n_two_paulis  # One RZ per interaction
        cnot_gates = self.reps * 2 * n_pairs * n_two_paulis  # Two CNOTs per interaction

        # Basis change gates for non-Z Paulis in two-qubit terms
        # Each X requires 2 H gates (before and after), each Y requires 4 gates (Sdg, H, H, S)
        basis_change_gates = 0
        for pauli in self._two_paulis:
            p1, p2 = pauli[0], pauli[1]
            # Count basis change gates per pair
            gates_per_pair = 0
            if p1 == "X":
                gates_per_pair += 2  # H before and after
            elif p1 == "Y":
                gates_per_pair += 4  # Sdg, H before; H, S after
            if p2 == "X":
                gates_per_pair += 2
            elif p2 == "Y":
                gates_per_pair += 4
            basis_change_gates += self.reps * n_pairs * gates_per_pair

        single_qubit = (
            h_gates + rx_gates + ry_gates + rz_gates + rz_two_qubit + basis_change_gates
        )
        two_qubit = cnot_gates
        total = single_qubit + two_qubit

        _logger.debug(
            "Gate breakdown: H=%d, RX=%d, RY=%d, RZ=%d, RZ_2q=%d, "
            "CNOT=%d, basis_change=%d, total=%d",
            h_gates,
            rx_gates,
            ry_gates,
            rz_gates,
            rz_two_qubit,
            cnot_gates,
            basis_change_gates,
            total,
        )

        return GateCountBreakdown(
            hadamard=h_gates,
            rx=rx_gates,
            ry=ry_gates,
            rz=rz_gates,
            rz_two_qubit=rz_two_qubit,
            cnot=cnot_gates,
            basis_change=basis_change_gates,
            total_single_qubit=single_qubit,
            total_two_qubit=two_qubit,
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
            - ``depth``: Circuit depth
            - ``reps``: Number of encoding repetitions
            - ``paulis``: List of Pauli terms
            - ``single_paulis``: List of single-qubit Pauli terms
            - ``two_paulis``: List of two-qubit Pauli terms
            - ``entanglement``: Entanglement topology
            - ``entanglement_pairs``: List of qubit pairs with interactions
            - ``n_entanglement_pairs``: Number of interaction pairs per layer
            - ``gate_counts``: Detailed breakdown from gate_count_breakdown()
            - ``is_entangling``: Whether circuit creates entanglement
            - ``simulability``: Simulation difficulty classification
            - ``trainability_estimate``: Estimated trainability (0.0 to 1.0)
            - ``hardware_requirements``: Dict with connectivity and gate requirements

        Examples
        --------
        Get complete resource analysis:

        >>> enc = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['n_entanglement_pairs']
        6
        >>> summary['gate_counts']['cnot']
        24

        Compare different Pauli configurations:

        >>> enc_z = PauliFeatureMap(n_features=4, paulis=["Z"])
        >>> enc_zz = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"])
        >>> enc_z.resource_summary()['is_entangling']
        False
        >>> enc_zz.resource_summary()['is_entangling']
        True

        Use for hardware planning:

        >>> enc = PauliFeatureMap(n_features=6, entanglement='linear')
        >>> summary = enc.resource_summary()
        >>> print(f"Requires {summary['hardware_requirements']['connectivity']}")
        Requires linear
        >>> print(f"Total two-qubit gates: {summary['gate_counts']['total_two_qubit']}")
        Total two-qubit gates: 20

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
        get_entanglement_pairs : Get the qubit pairs for interactions.
        properties : Access computed circuit properties.
        """
        pairs = self.get_entanglement_pairs()
        gate_counts = self.gate_count_breakdown()
        props = self.properties

        # Determine connectivity requirement based on entanglement topology
        if self.entanglement == "full":
            connectivity = "all-to-all"
        elif self.entanglement == "linear":
            connectivity = "linear"
        else:  # circular
            connectivity = "ring"

        # Determine native gates needed based on Paulis
        native_gates = ["H"]
        if "X" in self._single_paulis or any("X" in p for p in self._two_paulis):
            native_gates.append("RX")
        if "Y" in self._single_paulis or any("Y" in p for p in self._two_paulis):
            native_gates.extend(["RY", "S", "Sdg"])
        if "Z" in self._single_paulis or any("Z" in p for p in self._two_paulis):
            native_gates.append("RZ")
        if len(self._two_paulis) > 0:
            native_gates.append("CNOT")

        summary: dict[str, Any] = {
            # Basic circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            # Pauli configuration
            "paulis": self.paulis,
            "single_paulis": self._single_paulis,
            "two_paulis": self._two_paulis,
            # Entanglement configuration
            "entanglement": self.entanglement,
            "entanglement_pairs": pairs,
            "n_entanglement_pairs": len(pairs),
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": props.is_entangling,
            "simulability": props.simulability,
            "trainability_estimate": props.trainability_estimate,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": connectivity,
                "native_gates": sorted(set(native_gates)),
                "min_two_qubit_gate_fidelity": 0.99,  # Recommended for NISQ
            },
        }

        _logger.debug(
            "Resource summary: n_qubits=%d, depth=%d, total_gates=%d, "
            "entanglement=%s, pairs=%d, paulis=%r",
            self.n_qubits,
            self.depth,
            gate_counts["total"],
            self.entanglement,
            len(pairs),
            self.paulis,
        )

        return summary

    def _feature_map_single(self, x: NDArray[np.floating[Any]], idx: int) -> float:
        """Compute feature map value for single-qubit rotation.

        Parameters
        ----------
        x : NDArray
            Input feature vector.
        idx : int
            Feature index.

        Returns
        -------
        float
            Mapped feature value (2 * x[idx]).
        """
        return 2.0 * float(x[idx])

    def _feature_map_two(
        self,
        x: NDArray[np.floating[Any]],
        idx_i: int,
        idx_j: int,
    ) -> float:
        """Compute feature map value for two-qubit rotation.

        Uses the product feature map: φ(x_i, x_j) = 2(π - x_i)(π - x_j)
        This is the standard feature map used in quantum kernel methods.

        Parameters
        ----------
        x : NDArray
            Input feature vector.
        idx_i : int
            First feature index.
        idx_j : int
            Second feature index.

        Returns
        -------
        float
            Mapped feature value.
        """
        return 2.0 * (np.pi - float(x[idx_i])) * (np.pi - float(x[idx_j]))

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for a single data sample.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,).
        backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format:
            - PennyLane: callable function that applies gates
            - Qiskit: QuantumCircuit object
            - Cirq: Circuit object

        Raises
        ------
        ValueError
            If input shape doesn't match n_features or backend is unknown.

        Examples
        --------
        >>> enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"])
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> circuit = enc.get_circuit(x, backend='pennylane')
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

        # Debug log if values are far outside the optimal [0, 2π] range.
        if _logger.isEnabledFor(logging.DEBUG):
            x_min, x_max = float(x_validated.min()), float(x_validated.max())
            if (
                abs(x_min) > _INPUT_RANGE_DEBUG_THRESHOLD
                or abs(x_max) > _INPUT_RANGE_DEBUG_THRESHOLD
            ):
                _logger.debug(
                    "Input values [%.3g, %.3g] are outside optimal range [0, 2π]. "
                    "Consider normalizing features to [0, 2π] or [-π, π].",
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
        backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
            Target quantum computing framework.
        parallel : bool, default=False
            If True, use parallel processing via ThreadPoolExecutor for
            circuit generation. This can speed up processing for large
            batches (>100 samples) but adds overhead for small batches.

            Parallel processing is thread-safe because PauliFeatureMap's
            circuit generation is stateless (no instance mutation occurs).
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

        >>> enc = PauliFeatureMap(n_features=4)
        >>> X = np.random.randn(10, 4)
        >>> circuits = enc.get_circuits(X, backend='pennylane')
        >>> len(circuits)
        10

        Parallel processing for large batches:

        >>> enc = PauliFeatureMap(n_features=4)
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

    def _to_pennylane(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate PennyLane circuit function.

        Parameters
        ----------
        x : NDArray
            Validated input features.

        Returns
        -------
        callable
            Function that applies the encoding gates when called.
        """
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(
                "PennyLane is required for the 'pennylane' backend. "
                "Install it with: pip install pennylane"
            ) from e

        # Capture values for closure
        n_qubits = self.n_qubits
        reps = self.reps
        single_paulis = self._single_paulis
        two_paulis = self._two_paulis
        pairs = self._entanglement_pairs

        # Precompute rotation angles for efficiency
        single_angles: dict[str, list[float]] = {
            p: [self._feature_map_single(x, i) for i in range(n_qubits)]
            for p in single_paulis
        }
        two_angles: dict[str, list[float]] = {
            p: [self._feature_map_two(x, i, j) for i, j in pairs] for p in two_paulis
        }

        # Gate mapping for single-qubit Paulis
        single_gate_map = {
            "X": qml.RX,
            "Y": qml.RY,
            "Z": qml.RZ,
        }

        def circuit() -> None:
            """Apply the Pauli feature map encoding."""
            for _ in range(reps):
                # Hadamard layer
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)

                # Single-qubit Pauli rotations
                for pauli in single_paulis:
                    gate = single_gate_map[pauli]
                    for i in range(n_qubits):
                        gate(single_angles[pauli][i], wires=i)

                # Two-qubit Pauli rotations
                for pauli in two_paulis:
                    for pair_idx, (i, j) in enumerate(pairs):
                        angle = two_angles[pauli][pair_idx]
                        _apply_two_qubit_pauli_pennylane(qml, pauli, angle, i, j)

        return circuit

    def _to_qiskit(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Qiskit QuantumCircuit.

        Parameters
        ----------
        x : NDArray
            Validated input features.

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit implementing the encoding.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError(
                "Qiskit is required for the 'qiskit' backend. "
                "Install it with: pip install qiskit"
            ) from e

        qc = QuantumCircuit(self.n_qubits, name="PauliFeatureMap")

        for _ in range(self.reps):
            # Hadamard layer
            for i in range(self.n_qubits):
                qc.h(i)

            # Single-qubit Pauli rotations
            for pauli in self._single_paulis:
                for i in range(self.n_qubits):
                    angle = self._feature_map_single(x, i)
                    if pauli == "X":
                        qc.rx(angle, i)
                    elif pauli == "Y":
                        qc.ry(angle, i)
                    elif pauli == "Z":
                        qc.rz(angle, i)

            # Two-qubit Pauli rotations
            for pauli in self._two_paulis:
                for i, j in self._entanglement_pairs:
                    angle = self._feature_map_two(x, i, j)
                    _apply_two_qubit_pauli_qiskit(qc, pauli, angle, i, j)

        return qc

    def _to_cirq(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Cirq Circuit.

        Parameters
        ----------
        x : NDArray
            Validated input features.

        Returns
        -------
        cirq.Circuit
            Cirq circuit implementing the encoding.
        """
        try:
            import cirq
        except ImportError as e:
            raise ImportError(
                "Cirq is required for the 'cirq' backend. "
                "Install it with: pip install cirq-core"
            ) from e

        qubits = cirq.LineQubit.range(self.n_qubits)
        moments: list[Any] = []

        for _ in range(self.reps):
            # Hadamard layer
            moments.append(
                cirq.Moment([cirq.H(qubits[i]) for i in range(self.n_qubits)])
            )

            # Single-qubit Pauli rotations
            for pauli in self._single_paulis:
                gates = []
                for i in range(self.n_qubits):
                    angle = self._feature_map_single(x, i)
                    if pauli == "X":
                        gates.append(cirq.rx(angle)(qubits[i]))
                    elif pauli == "Y":
                        gates.append(cirq.ry(angle)(qubits[i]))
                    elif pauli == "Z":
                        gates.append(cirq.rz(angle)(qubits[i]))
                moments.append(cirq.Moment(gates))

            # Two-qubit Pauli rotations
            for pauli in self._two_paulis:
                for i, j in self._entanglement_pairs:
                    angle = self._feature_map_two(x, i, j)
                    two_qubit_moments = _apply_two_qubit_pauli_cirq(
                        cirq, pauli, angle, qubits[i], qubits[j]
                    )
                    moments.extend(two_qubit_moments)

        return cirq.Circuit(moments)

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Returns
        -------
        EncodingProperties
            Computed properties including qubit count, depth,
            gate count, and other characteristics.
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        # Get detailed gate counts
        breakdown = self.gate_count_breakdown()

        # Determine if entangling
        is_entangling = len(self._two_paulis) > 0 and n_pairs > 0

        # Determine simulability
        # Circuits with only Z rotations and no entanglement are simulable
        # Circuits with entanglement and non-Clifford gates are not simulable
        if not is_entangling:
            simulability: Literal[
                "simulable", "conditionally_simulable", "not_simulable"
            ] = "simulable"
        else:
            # Non-Clifford entangling circuits are not efficiently simulable
            simulability = "not_simulable"

        # Trainability estimate: decreases with depth
        trainability = max(0.1, 0.9 - 0.05 * self.depth)

        return EncodingProperties(
            n_qubits=n,
            depth=self.depth,
            gate_count=breakdown["total"],
            single_qubit_gates=breakdown["total_single_qubit"],
            two_qubit_gates=breakdown["total_two_qubit"],
            parameter_count=0,  # No trainable parameters, only data-dependent
            is_entangling=is_entangling,
            simulability=simulability,
            trainability_estimate=trainability,
            notes=(
                f"Pauli terms: {self.paulis}, "
                f"Entanglement: {self.entanglement}, "
                f"Pairs: {n_pairs}, "
                f"Reps: {self.reps}"
            ),
        )

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, "
            f"reps={self.reps}, "
            f"paulis={self.paulis!r}, "
            f"entanglement={self.entanglement!r})"
        )


# =============================================================================
# Helper functions for two-qubit Pauli rotations
# =============================================================================


def _apply_two_qubit_pauli_pennylane(
    qml: Any,
    pauli: str,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply two-qubit Pauli rotation in PennyLane.

    Implements exp(-i * angle/2 * P_i ⊗ P_j) using CNOT decomposition.

    Parameters
    ----------
    qml : module
        PennyLane module.
    pauli : str
        Two-character Pauli string (e.g., "ZZ", "XX").
    angle : float
        Rotation angle.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    p1, p2 = pauli[0], pauli[1]

    # Basis change for first qubit
    if p1 == "X":
        qml.Hadamard(wires=qubit_i)
    elif p1 == "Y":
        qml.adjoint(qml.S)(wires=qubit_i)
        qml.Hadamard(wires=qubit_i)

    # Basis change for second qubit
    if p2 == "X":
        qml.Hadamard(wires=qubit_j)
    elif p2 == "Y":
        qml.adjoint(qml.S)(wires=qubit_j)
        qml.Hadamard(wires=qubit_j)

    # CNOT-RZ-CNOT structure
    qml.CNOT(wires=[qubit_i, qubit_j])
    qml.RZ(angle, wires=qubit_j)
    qml.CNOT(wires=[qubit_i, qubit_j])

    # Undo basis change for second qubit
    if p2 == "X":
        qml.Hadamard(wires=qubit_j)
    elif p2 == "Y":
        qml.Hadamard(wires=qubit_j)
        qml.S(wires=qubit_j)

    # Undo basis change for first qubit
    if p1 == "X":
        qml.Hadamard(wires=qubit_i)
    elif p1 == "Y":
        qml.Hadamard(wires=qubit_i)
        qml.S(wires=qubit_i)


def _apply_two_qubit_pauli_qiskit(
    qc: Any,
    pauli: str,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply two-qubit Pauli rotation in Qiskit.

    Parameters
    ----------
    qc : QuantumCircuit
        Qiskit quantum circuit.
    pauli : str
        Two-character Pauli string.
    angle : float
        Rotation angle.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    p1, p2 = pauli[0], pauli[1]

    # Basis change for first qubit
    if p1 == "X":
        qc.h(qubit_i)
    elif p1 == "Y":
        qc.sdg(qubit_i)
        qc.h(qubit_i)

    # Basis change for second qubit
    if p2 == "X":
        qc.h(qubit_j)
    elif p2 == "Y":
        qc.sdg(qubit_j)
        qc.h(qubit_j)

    # CNOT-RZ-CNOT structure
    qc.cx(qubit_i, qubit_j)
    qc.rz(angle, qubit_j)
    qc.cx(qubit_i, qubit_j)

    # Undo basis change for second qubit
    if p2 == "X":
        qc.h(qubit_j)
    elif p2 == "Y":
        qc.h(qubit_j)
        qc.s(qubit_j)

    # Undo basis change for first qubit
    if p1 == "X":
        qc.h(qubit_i)
    elif p1 == "Y":
        qc.h(qubit_i)
        qc.s(qubit_i)


def _apply_two_qubit_pauli_cirq(
    cirq: Any,
    pauli: str,
    angle: float,
    qubit_i: Any,
    qubit_j: Any,
) -> list[Any]:
    """Apply two-qubit Pauli rotation in Cirq.

    Parameters
    ----------
    cirq : module
        Cirq module.
    pauli : str
        Two-character Pauli string.
    angle : float
        Rotation angle.
    qubit_i : cirq.LineQubit
        First qubit.
    qubit_j : cirq.LineQubit
        Second qubit.

    Returns
    -------
    list[cirq.Moment]
        List of circuit moments implementing the rotation.
    """
    p1, p2 = pauli[0], pauli[1]
    moments: list[Any] = []

    # Basis change gates - organized by gate type for proper sequencing
    s_gates: list[Any] = []
    h_gates_forward: list[Any] = []
    h_gates_backward: list[Any] = []
    s_inv_gates: list[Any] = []

    if p1 == "X":
        h_gates_forward.append(cirq.H(qubit_i))
        h_gates_backward.append(cirq.H(qubit_i))
    elif p1 == "Y":
        s_inv_gates.append(cirq.S(qubit_i) ** -1)
        h_gates_forward.append(cirq.H(qubit_i))
        h_gates_backward.append(cirq.H(qubit_i))
        s_gates.append(cirq.S(qubit_i))

    if p2 == "X":
        h_gates_forward.append(cirq.H(qubit_j))
        h_gates_backward.append(cirq.H(qubit_j))
    elif p2 == "Y":
        s_inv_gates.append(cirq.S(qubit_j) ** -1)
        h_gates_forward.append(cirq.H(qubit_j))
        h_gates_backward.append(cirq.H(qubit_j))
        s_gates.append(cirq.S(qubit_j))

    # Apply basis changes in proper sequence
    if s_inv_gates:
        moments.append(cirq.Moment(s_inv_gates))
    if h_gates_forward:
        moments.append(cirq.Moment(h_gates_forward))

    # CNOT-RZ-CNOT
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))
    moments.append(cirq.Moment([cirq.rz(angle)(qubit_j)]))
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))

    # Undo basis changes in reverse order
    if h_gates_backward:
        moments.append(cirq.Moment(h_gates_backward))
    if s_gates:
        moments.append(cirq.Moment(s_gates))

    return moments
