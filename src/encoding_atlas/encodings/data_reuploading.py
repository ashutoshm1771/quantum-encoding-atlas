"""Data re-uploading encoding module.

This module implements DataReuploading, a quantum feature map that repeatedly
encodes classical data throughout the circuit, interleaved with entangling
layers. This multi-layer encoding strategy creates quantum states with rich
Fourier spectra, enabling high expressivity in quantum machine learning models.

Data re-uploading is a foundational technique in quantum machine learning,
offering several key advantages:

1. **High Expressivity**: Multiple encoding layers increase the accessible
   Fourier frequencies, enabling complex decision boundaries
2. **Efficient Depth-Width Trade-off**: Uses fewer qubits than angle encoding
   while maintaining expressivity through depth
3. **Rich Feature Space**: Creates quantum kernels with higher-order feature
   interactions through repeated encoding and entanglement
4. **Gradient-Friendly Structure**: Interleaved data-entanglement pattern
   provides better gradient flow than purely random circuits

The encoding creates quantum states by repeatedly uploading data:

    |ψ(x)⟩ = [U_ent · U_data(x)]^L |0⟩^⊗n

where U_data(x) encodes the classical features via RY rotations and U_ent
provides entanglement via a CNOT ladder. The key insight is that re-encoding
data at multiple circuit depths increases the Fourier frequency content of
the resulting quantum state.

Relationship to Universal Approximation
---------------------------------------
This module implements the **encoding component** of the data re-uploading
architecture. The full universal approximation capability described in
Pérez-Salinas et al. [1]_ requires **trainable parameters** interleaved
with data uploads:

    |ψ(θ, x)⟩ = U(θₗ) · U_data(x) · U(θₗ₋₁) · U_data(x) · ... · U(θ₁) |0⟩

This module provides U_data(x) repeated L times with fixed entanglement.
To achieve universal approximation, users should:

1. Use this encoding as the feature map in a variational circuit
2. Add trainable rotation layers (e.g., via PennyLane's StronglyEntanglingLayers)
3. Optimize the trainable parameters for their specific task

The encoding alone creates a fixed (but expressive) quantum kernel suitable
for kernel methods, classification, and as input to variational algorithms.

Mathematical Background
-----------------------
The expressivity of data re-uploading arises from Fourier analysis of
parameterized quantum circuits [2]_. For a single data feature x encoded
L times through rotation gates, the expectation value of any observable
can be written as a Fourier series:

    f(x) = Σₖ cₖ exp(ikx)  for k ∈ {-L, ..., L}

The number of accessible Fourier frequencies grows linearly with L (the
number of encoding layers). For multi-qubit circuits with entanglement,
the frequency spectrum becomes richer due to:

- Product terms from multi-qubit observables
- Interference between qubits via entanglement

The Fourier coefficients cₖ are determined by the fixed circuit structure.
To control these coefficients (and thus approximate arbitrary functions),
trainable parameters must be added between data uploads.

Note on Data Preprocessing
--------------------------
For data re-uploading, input features are typically:

- Scaled to [0, 2π] or [-π, π] for rotation gates
- Standardized to zero mean and unit variance before scaling
- Features beyond n_qubits are cyclically mapped (feature i → qubit i mod n_qubits)

The repeated encoding means the circuit sees each feature value L times,
effectively amplifying the importance of the data in the output state.

Use Cases
---------
Data re-uploading encoding is particularly suited for:

- **Quantum kernel methods**: As a high-expressivity feature map for
  quantum kernel estimation and quantum support vector machines
- **Variational quantum classifiers**: As the data encoding layer in
  hybrid quantum-classical classifiers (add trainable layers for full
  universal approximation capability)
- **Quantum neural networks**: Input encoding for QNN architectures where
  expressivity is more important than circuit width
- **Function approximation**: When combined with trainable parameters,
  enables approximation of complex continuous functions
- **Single-qubit demonstrations**: Proof-of-concept experiments showing
  how re-uploading increases expressivity with minimal qubits

Limitations
-----------
- **Circuit depth**: Each layer adds depth proportional to qubit count,
  which may exceed NISQ device coherence times for many layers
- **Trainability**: Very deep circuits (>10 layers) may exhibit barren
  plateaus despite the interleaved structure
- **Feature capacity**: When n_features > n_qubits with cyclic mapping,
  different features encoded on the same qubit may interfere
- **Gate count scaling**: Total gates scale as O(n_layers × n_features),
  which can be significant for deep circuits
- **Fixed coefficients**: Without trainable parameters, the Fourier
  coefficients are fixed by the circuit structure

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation and resource analysis. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.data_reuploading').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters (n_features, n_layers, n_qubits)
- Deep circuit warnings (when n_layers exceeds threshold)
- Input value range advisories (when values are outside [0, 2π])
- Circuit generation progress for each backend
- Gate count and resource analysis details

Resource Analysis
-----------------
DataReuploading provides methods to analyze circuit resources:

- ``gate_count_breakdown()``: Get detailed gate counts by type (RY, CNOT)
- ``resource_summary()``: Comprehensive resource analysis including depth,
  gate counts, expressivity metrics, and hardware recommendations

These methods help understand resource requirements before execution:

    >>> enc = DataReuploading(n_features=4, n_layers=5)
    >>> enc.gate_count_breakdown()
    {'ry_gates': 20, 'cnot_gates': 15, 'total': 35, ...}
    >>> summary = enc.resource_summary()
    >>> summary['trainability_estimate']
    0.65

References
----------
.. [1] Pérez-Salinas, A., et al. (2020). "Data re-uploading for a universal
       quantum classifier." Quantum, 4, 226.
.. [2] Schuld, M., Sweke, R., & Meyer, J. J. (2021). "Effect of data encoding
       on the expressive power of variational quantum machine learning models."
       Physical Review A, 103(3), 032430.
.. [3] Goto, T., et al. (2021). "Universal approximation property of quantum
       machine learning models in quantum-enhanced feature spaces."
       Physical Review Letters, 127(9), 090506.
.. [4] McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
       network training landscapes." Nature Communications, 9, 4812.
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
# Configure a logger for this module to enable optional debug output.
# By default, Python's logging is disabled (level WARNING), so these
# debug statements have no performance impact in production.
#
# To enable debug logging for this module:
#   import logging
#   logging.getLogger('encoding_atlas.encodings.data_reuploading').setLevel(logging.DEBUG)
#   logging.basicConfig(level=logging.DEBUG)
#
# Or configure via your application's logging setup.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ['DataReuploading']

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default number of re-uploading layers.
#
# The choice of 3 layers provides a good balance between:
#   - Expressivity: Multiple layers increase Fourier frequency content
#   - Trainability: Fewer layers reduce barren plateau risk
#   - Gate count: Each layer adds O(n) gates
#
# Research suggests 2-5 layers often suffice for many classification tasks.
# Higher values may be needed for more complex function approximation.
_DEFAULT_N_LAYERS: int = 3

# Threshold for warning about deep circuits.
#
# Deep circuits with many re-uploading layers may face trainability challenges
# due to barren plateaus (exponentially vanishing gradients). While data
# re-uploading's interleaved structure provides better gradient flow than
# purely entangling circuits, very deep circuits can still be problematic.
#
# At 10 layers, the circuit depth becomes significant enough to warrant
# a warning. Users should consider:
#   - Whether the task requires such expressivity
#   - Using gradient-free optimization methods
#   - Layer-wise training strategies
#
# References:
#   McClean et al. (2018) "Barren plateaus in quantum neural network
#   training landscapes." Nature Communications, 9, 4812.
_DEEP_CIRCUIT_WARNING_THRESHOLD: int = 10

# Threshold for DEBUG logging about input values outside optimal range.
#
# Data re-uploading uses RY rotations which are 2π-periodic, so any input
# value produces valid results. However, the encoding works best with inputs
# in [0, 2π] or [-π, π] to ensure meaningful phase differences between states.
#
# Values beyond this threshold (in absolute value) trigger a debug log to
# help users identify potential preprocessing issues. Set to 4π to allow
# some flexibility while catching extreme values that suggest unscaled data.
_INPUT_RANGE_DEBUG_THRESHOLD: float = 4.0 * np.pi


# =============================================================================
# Type Definitions
# =============================================================================

class GateCountBreakdown(TypedDict):
    """Type definition for gate count breakdown dictionary.

    This TypedDict provides type safety and IDE autocompletion for the
    dictionary returned by ``gate_count_breakdown()``.

    Attributes
    ----------
    ry_gates : int
        Number of RY rotation gates (data encoding gates).
    cnot_gates : int
        Number of CNOT entangling gates.
    total_single_qubit : int
        Total single-qubit gates (equals ry_gates).
    total_two_qubit : int
        Total two-qubit gates (equals cnot_gates).
    total : int
        Total gate count (ry_gates + cnot_gates).
    ry_per_layer : int
        RY gates per layer.
    cnot_per_layer : int
        CNOT gates per layer.
    gates_per_layer : int
        Total gates per layer.
    """

    ry_gates: int
    cnot_gates: int
    total_single_qubit: int
    total_two_qubit: int
    total: int
    ry_per_layer: int
    cnot_per_layer: int
    gates_per_layer: int


class DataReuploading(BaseEncoding):
    """Data re-uploading quantum feature map with high expressivity.

    DataReuploading implements a quantum encoding strategy where classical
    data is encoded multiple times throughout the circuit, interleaved with
    entangling layers. This multi-layer approach creates quantum states with
    rich Fourier spectra, enabling expressive quantum kernels and serving as
    a foundation for variational quantum classifiers.

    The key insight is that re-uploading data at multiple circuit depths
    increases the number of accessible Fourier frequencies, similar to how
    depth increases expressivity in classical neural networks.

    The circuit structure repeats L times:

        |0⟩ ─ RY(x₀) ─╭────╮─ RY(x₀) ─╭────╮─ ... ─
        |0⟩ ─ RY(x₁) ─│CNOT│─ RY(x₁) ─│CNOT│─ ... ─
        |0⟩ ─ RY(x₂) ─│    │─ RY(x₂) ─│    │─ ... ─
        ...           ╰────╯          ╰────╯

    Each layer consists of data encoding (RY rotations) followed by
    entanglement (CNOT ladder), repeated n_layers times.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
        If n_features > n_qubits, features are cyclically mapped to qubits.
    n_layers : int, default=3
        Number of re-uploading layers. Higher values increase the number of
        accessible Fourier frequencies but also circuit depth. Must be at
        least 1. More layers provide richer feature representations.
    n_qubits : int, optional
        Number of qubits in the circuit. If not specified, defaults to
        n_features. Can be set lower than n_features to use fewer qubits
        with cyclic feature mapping.

    Attributes
    ----------
    n_layers : int
        Number of re-uploading layers.
    n_features : int
        Number of classical features (inherited from BaseEncoding).
    n_qubits : int
        Number of qubits in the circuit.

    Examples
    --------
    Create a basic data re-uploading encoding:

    >>> from encoding_atlas import DataReuploading
    >>> import numpy as np
    >>> enc = DataReuploading(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.n_layers
    3

    Generate a PennyLane circuit:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Use fewer qubits than features (cyclic mapping):

    >>> enc_compact = DataReuploading(n_features=8, n_qubits=4)
    >>> enc_compact.n_qubits
    4

    Increase layers for higher expressivity:

    >>> enc_deep = DataReuploading(n_features=4, n_layers=10)
    >>> enc_deep.depth
    30

    Generate circuits for different backends:

    >>> qiskit_circuit = enc.get_circuit(x, backend='qiskit')
    >>> qiskit_circuit.num_qubits
    4

    Access encoding properties:

    >>> props = enc.properties
    >>> props.is_entangling
    True

    References
    ----------
    .. [1] Pérez-Salinas, A., et al. (2020). "Data re-uploading for a
           universal quantum classifier." Quantum, 4, 226.
    .. [2] Schuld, M., et al. (2021). "Effect of data encoding on the
           expressive power of variational quantum machine learning models."

    See Also
    --------
    AngleEncoding : Single-layer encoding without re-uploading.
    IQPEncoding : Entangling encoding without data re-uploading.
    HardwareEfficientEncoding : Similar structure optimized for hardware.
    PauliFeatureMap : Alternative multi-layer encoding approach.

    Notes
    -----
    **Feature Map vs. Universal Approximation**: This class implements a
    *fixed feature map* that provides high expressivity through repeated
    data encoding. For full universal function approximation as described
    in [1]_, trainable parameters must be added between data uploads. Use
    this encoding as the feature map layer in a variational circuit, then
    add trainable rotations to achieve universal approximation capability.

    **Fourier Expressivity**: With L encoding layers, the quantum state
    can represent functions with Fourier frequencies up to ±L. More layers
    enable the representation of higher-frequency components. The specific
    Fourier coefficients are determined by the fixed circuit structure.

    **Trainability Considerations**: While data re-uploading helps with
    expressivity, very deep circuits may still face trainability challenges.
    The interleaved structure generally provides better gradient flow than
    purely entangling circuits.

    **Feature Mapping**: When n_features > n_qubits, feature xᵢ is encoded
    on qubit (i mod n_qubits). This allows compact representations but may
    create interference between features mapped to the same qubit.

    **Gate Set Choice**: This implementation uses RY rotations for data
    encoding and a CNOT ladder for entanglement. The original single-qubit
    formulation in [1]_ uses general SU(2) (U3) rotations with three
    parameters per gate, and the multi-qubit extension uses CZ gates with
    varying entanglement topologies. The RY + CNOT variant used here is a
    common simplification adopted by many QML frameworks; it is less
    expressive per gate but sufficient for most feature map applications.
    The Fourier frequency analysis from [2]_ applies regardless of the
    specific rotation axis.

    **Comparison with Classical Deep Learning**: Data re-uploading is
    analogous to residual connections in deep neural networks, where
    information about the input is preserved and refined through layers.
    """

    __slots__ = ('n_layers', '_n_qubits', '_entanglement_pairs')

    def __init__(
        self,
        n_features: int,
        n_layers: int = _DEFAULT_N_LAYERS,
        n_qubits: int | None = None,
    ) -> None:
        """Initialize the data re-uploading encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        n_layers : int, default=3
            Number of re-uploading layers for expressivity.
        n_qubits : int, optional
            Number of qubits. Defaults to n_features if not specified.

        Raises
        ------
        ValueError
            If n_layers is less than 1.
        ValueError
            If n_qubits is less than 1 (when specified).
        ValueError
            If n_features is less than 1 (raised by parent class).

        Warns
        -----
        UserWarning
            If n_layers exceeds the deep circuit threshold (default 10),
            as very deep circuits may face trainability challenges due to
            barren plateaus.

        Examples
        --------
        >>> enc = DataReuploading(n_features=4)
        >>> enc.n_qubits
        4
        >>> enc.n_layers
        3

        >>> enc = DataReuploading(n_features=4, n_layers=5, n_qubits=2)
        >>> enc.n_qubits
        2
        >>> enc.n_layers
        5
        """
        # =====================================================================
        # PARAMETER VALIDATION
        # =====================================================================
        # Validate n_layers before calling parent constructor to fail fast.
        # Note: bool is a subclass of int in Python, so we explicitly exclude it
        # to prevent confusing behavior where True would be accepted as n_layers=1.
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError(f"n_layers must be a positive integer, got {n_layers!r}")

        # Validate n_qubits if specified
        if n_qubits is not None:
            if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
                raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")

        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        super().__init__(n_features, n_layers=n_layers, n_qubits_override=n_qubits)
        self.n_layers: int = n_layers
        self._n_qubits: int = n_qubits if n_qubits is not None else n_features

        # Pre-compute and cache entanglement pairs as an immutable tuple.
        # This eliminates redundant computation during circuit generation:
        # - Each get_circuit() call would otherwise recompute pairs
        # - Batch processing of N samples would compute pairs N times
        # - With caching, pairs are computed once at initialization
        #
        # DataReuploading uses a CNOT ladder topology (linear entanglement).
        # For n qubits, we have n-1 pairs: (0,1), (1,2), ..., (n-2, n-1)
        if self._n_qubits <= 1:
            self._entanglement_pairs: tuple[tuple[int, int], ...] = ()
        else:
            self._entanglement_pairs = tuple(
                (i, i + 1) for i in range(self._n_qubits - 1)
            )

        # Log initialization for debugging
        _logger.debug(
            "DataReuploading initialized: n_features=%d, n_layers=%d, n_qubits=%d, "
            "n_entanglement_pairs=%d",
            self.n_features,
            self.n_layers,
            self._n_qubits,
            len(self._entanglement_pairs),
        )

        # =====================================================================
        # DEEP CIRCUIT WARNING
        # =====================================================================
        # Warn about potential trainability issues with very deep circuits.
        # Deep circuits may exhibit barren plateaus (exponentially vanishing
        # gradients), making optimization difficult. While data re-uploading's
        # interleaved structure provides better gradient flow than purely
        # entangling circuits, very deep configurations can still be problematic.
        if n_layers > _DEEP_CIRCUIT_WARNING_THRESHOLD:
            total_gates = n_layers * n_features + n_layers * max(0, self._n_qubits - 1)
            # Compute exact depth: ceil(n_features/n_qubits) + (n_qubits-1) per layer
            encoding_depth = -(-n_features // self._n_qubits)
            entangling_depth = max(0, self._n_qubits - 1)
            circuit_depth = n_layers * (encoding_depth + entangling_depth)
            estimated_trainability = max(0.4, 0.9 - 0.05 * n_layers)
            warnings.warn(
                f"DataReuploading with {n_layers} layers creates a deep circuit "
                f"({total_gates} total gates, depth={circuit_depth}). "
                f"Very deep circuits may face trainability challenges due to barren "
                f"plateaus (estimated trainability: {estimated_trainability:.2f}). "
                f"Consider: (1) reducing n_layers if task permits, "
                f"(2) using gradient-free optimization, or "
                f"(3) layer-wise training strategies.",
                UserWarning,
                stacklevel=2,
            )
            _logger.warning(
                "Deep circuit configuration: n_layers=%d exceeds threshold=%d, "
                "total_gates=%d, estimated_trainability=%.2f",
                n_layers,
                _DEEP_CIRCUIT_WARNING_THRESHOLD,
                total_gates,
                estimated_trainability,
            )

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit.

        If not explicitly specified during initialization, this equals
        n_features. Can be less than n_features for compact representations
        using cyclic feature mapping.

        Returns
        -------
        int
            Number of qubits.
        """
        return self._n_qubits

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        Each layer consists of data encoding (RY rotations) and entanglement
        (CNOT ladder). The depth is computed exactly based on circuit structure:

        - **Encoding sublayer**: When features are cyclically mapped to qubits,
          each qubit may receive multiple RY gates sequentially. The depth is
          ceil(n_features / n_qubits).
        - **Entangling sublayer**: CNOT ladder with n_qubits-1 sequential gates.

        Returns
        -------
        int
            Exact circuit depth = n_layers × (ceil(n_features/n_qubits) + n_qubits - 1).
            For single-qubit circuits, depth = n_layers × ceil(n_features/1).

        Examples
        --------
        >>> enc = DataReuploading(n_features=4, n_qubits=4, n_layers=3)
        >>> enc.depth  # 3 × (1 + 3) = 12
        12

        >>> enc = DataReuploading(n_features=8, n_qubits=4, n_layers=2)
        >>> enc.depth  # 2 × (2 + 3) = 10
        10

        >>> enc = DataReuploading(n_features=1, n_qubits=1, n_layers=5)
        >>> enc.depth  # 5 × (1 + 0) = 5
        5
        """
        # Encoding depth: ceil(n_features / n_qubits) for cyclic feature mapping
        # Using integer arithmetic: -(-a // b) is equivalent to ceil(a / b)
        encoding_depth = -(-self.n_features // self._n_qubits)
        # Entangling depth: CNOT ladder has n_qubits - 1 sequential gates
        entangling_depth = max(0, self._n_qubits - 1)
        return self.n_layers * (encoding_depth + entangling_depth)

    # =========================================================================
    # Resource Analysis Methods
    # =========================================================================

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type.

        Computes the exact number of each gate type in the data re-uploading
        circuit. This is useful for:

        - Resource estimation before circuit execution
        - Hardware compatibility assessment (gate budget planning)
        - Comparing different layer configurations
        - Understanding circuit complexity scaling

        Returns
        -------
        GateCountBreakdown
            Dictionary with gate counts:

            - ``ry_gates``: Number of RY rotation gates (data encoding)
            - ``cnot_gates``: Number of CNOT entangling gates
            - ``total_single_qubit``: Total single-qubit gates (= ry_gates)
            - ``total_two_qubit``: Total two-qubit gates (= cnot_gates)
            - ``total``: Total gate count
            - ``ry_per_layer``: RY gates per layer
            - ``cnot_per_layer``: CNOT gates per layer
            - ``gates_per_layer``: Total gates per layer

        Examples
        --------
        Basic gate count analysis:

        >>> enc = DataReuploading(n_features=4, n_layers=3)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['ry_gates']
        12
        >>> breakdown['cnot_gates']
        9
        >>> breakdown['total']
        21

        Compare different layer counts:

        >>> enc_shallow = DataReuploading(n_features=4, n_layers=2)
        >>> enc_deep = DataReuploading(n_features=4, n_layers=10)
        >>> enc_shallow.gate_count_breakdown()['total']
        14
        >>> enc_deep.gate_count_breakdown()['total']
        70

        Per-layer breakdown:

        >>> enc = DataReuploading(n_features=8, n_layers=5)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['ry_per_layer']
        8
        >>> breakdown['cnot_per_layer']
        7
        >>> breakdown['gates_per_layer']
        15

        Notes
        -----
        **Gate Count Formulas:**

        - RY gates per layer = n_features (all features encoded via cyclic mapping)
        - CNOT gates per layer = max(0, n_qubits - 1)
        - Total RY gates = n_layers × n_features
        - Total CNOT gates = n_layers × max(0, n_qubits - 1)

        **Single-Qubit Case (n_qubits=1):**

        When n_qubits=1, there are no CNOT gates (no entanglement possible).
        The circuit consists only of repeated RY rotations, creating a
        feature map with L Fourier frequencies for L layers.

        See Also
        --------
        resource_summary : Get comprehensive resource analysis.
        properties : Access computed circuit properties.
        """
        n = self._n_qubits
        n_entangling = max(0, n - 1)

        # Per-layer counts
        # With cyclic mapping, ALL features are encoded (one RY per feature)
        ry_per_layer = self.n_features
        cnot_per_layer = n_entangling
        gates_per_layer = ry_per_layer + cnot_per_layer

        # Total counts
        ry_gates = self.n_layers * ry_per_layer
        cnot_gates = self.n_layers * cnot_per_layer
        total = ry_gates + cnot_gates

        _logger.debug(
            "Gate breakdown: RY=%d, CNOT=%d, total=%d (per layer: %d)",
            ry_gates,
            cnot_gates,
            total,
            gates_per_layer,
        )

        return GateCountBreakdown(
            ry_gates=ry_gates,
            cnot_gates=cnot_gates,
            total_single_qubit=ry_gates,
            total_two_qubit=cnot_gates,
            total=total,
            ry_per_layer=ry_per_layer,
            cnot_per_layer=cnot_per_layer,
            gates_per_layer=gates_per_layer,
        )

    def resource_summary(self) -> dict[str, Any]:
        """Generate a comprehensive resource summary for this encoding.

        Provides a detailed breakdown of circuit resources including qubit
        requirements, circuit depth, gate counts, expressivity metrics,
        trainability estimates, and hardware recommendations. Useful for
        hardware planning, comparing encoding configurations, and generating
        reports.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            **Circuit Structure:**
            - ``n_qubits``: Number of qubits required
            - ``n_features``: Number of input features
            - ``n_layers``: Number of re-uploading layers
            - ``depth``: Circuit depth

            **Gate Counts:**
            - ``gate_counts``: Detailed breakdown from gate_count_breakdown()

            **Encoding Characteristics:**
            - ``is_entangling``: Whether circuit creates entanglement
            - ``simulability``: Classical simulation difficulty
            - ``trainability_estimate``: Estimated trainability (0.0 to 1.0)
            - ``fourier_frequencies``: Maximum Fourier frequencies (expressivity)

            **Hardware Requirements:**
            - ``hardware_requirements``: Dict with connectivity and gate needs

            **Recommendations:**
            - ``recommendations``: List of usage recommendations based on config

        Examples
        --------
        Get complete resource analysis:

        >>> enc = DataReuploading(n_features=4, n_layers=3)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['depth']
        9
        >>> summary['trainability_estimate']
        0.75

        Examine hardware requirements:

        >>> enc = DataReuploading(n_features=8, n_layers=5)
        >>> summary = enc.resource_summary()
        >>> summary['hardware_requirements']['connectivity']
        'linear'
        >>> summary['hardware_requirements']['native_gates']
        ['RY', 'CNOT']

        Check expressivity metrics:

        >>> enc = DataReuploading(n_features=4, n_layers=10)
        >>> summary = enc.resource_summary()
        >>> summary['fourier_frequencies']
        10
        >>> summary['is_entangling']
        True

        Use recommendations for optimization:

        >>> enc = DataReuploading(n_features=4, n_layers=15)  # doctest: +SKIP
        >>> summary = enc.resource_summary()  # doctest: +SKIP
        >>> for rec in summary['recommendations']:  # doctest: +SKIP
        ...     print(rec)

        Notes
        -----
        **Trainability Estimate:**

        The trainability estimate is computed as:

            trainability = max(0.4, 0.9 - 0.05 × n_layers)

        This heuristic reflects the observation that deeper circuits tend to
        have flatter loss landscapes (barren plateaus). The interleaved
        structure of data re-uploading provides better trainability than
        purely random or hardware-efficient ansätze, hence the relatively
        slow decay rate.

        **Fourier Frequencies:**

        The maximum number of Fourier frequencies that can be represented
        equals n_layers. This determines the encoding's ability to approximate
        functions with high-frequency components. For multi-qubit circuits,
        the effective frequency content can be higher due to entanglement.

        **Hardware Connectivity:**

        DataReuploading uses a CNOT ladder topology (qubit i → qubit i+1),
        which requires only linear (nearest-neighbor) connectivity. This
        makes it hardware-friendly for most superconducting and ion trap
        architectures.

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts by type.
        properties : Access computed circuit properties.
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties

        # Compute expressivity metric (Fourier frequencies)
        fourier_frequencies = self.n_layers

        # Build recommendations based on configuration
        recommendations: list[str] = []

        if self.n_layers > _DEEP_CIRCUIT_WARNING_THRESHOLD:
            recommendations.append(
                f"Consider reducing n_layers (currently {self.n_layers}) if task "
                f"permits, as deep circuits may face trainability challenges."
            )

        if self._n_qubits == 1:
            recommendations.append(
                "Single-qubit mode: No entanglement, but repeated encoding still "
                "increases Fourier frequency content for expressive feature maps."
            )

        if self.n_features > self._n_qubits:
            recommendations.append(
                f"Features ({self.n_features}) exceed qubits ({self._n_qubits}): "
                f"Using cyclic feature mapping. Consider if feature interference "
                f"on shared qubits is acceptable for your task."
            )

        if self.n_layers < 3 and self._n_qubits > 1:
            recommendations.append(
                f"Shallow circuit ({self.n_layers} layers): May have limited "
                f"expressivity for complex functions. Consider increasing n_layers "
                f"if accuracy is insufficient."
            )

        if not recommendations:
            recommendations.append(
                "Configuration looks good for typical quantum ML tasks."
            )

        summary: dict[str, Any] = {
            # Circuit structure
            "n_qubits": self._n_qubits,
            "n_features": self.n_features,
            "n_layers": self.n_layers,
            "depth": self.depth,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": self._n_qubits > 1,
            "simulability": "not_simulable" if self._n_qubits > 1 else "simulable",
            "trainability_estimate": props.trainability_estimate,
            "fourier_frequencies": fourier_frequencies,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": "linear",  # CNOT ladder only needs nearest-neighbor
                "native_gates": ["RY", "CNOT"],
                "min_qubit_count": self._n_qubits,
                "estimated_circuit_time_us": gate_counts["total"] * 0.5,  # ~0.5μs per gate
            },
            # Recommendations
            "recommendations": recommendations,
        }

        _logger.debug(
            "Resource summary: n_qubits=%d, depth=%d, total_gates=%d, "
            "trainability=%.2f, fourier_frequencies=%d",
            self._n_qubits,
            self.depth,
            gate_counts["total"],
            props.trainability_estimate,
            fourier_frequencies,
        )

        return summary

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs for CNOT entanglement in the ladder topology.

        Returns the list of qubit pairs that will have CNOT gates applied
        between them. DataReuploading uses a linear (ladder) entanglement
        topology where each qubit is connected to its nearest neighbor.
        This is useful for:

        - Understanding circuit structure before generation
        - Verifying hardware connectivity requirements
        - Debugging entanglement patterns
        - Computing resource requirements

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for CNOT gates.
            Each tuple (i, i+1) indicates a CNOT from qubit i to qubit i+1.
            Returns an empty list for single-qubit configurations.
            This is a new list created from the cached tuple, so callers
            can safely modify it without affecting the encoding.

        Notes
        -----
        **Linear Topology:**

        DataReuploading always uses linear (ladder) entanglement:

        - Pairs: [(0, 1), (1, 2), (2, 3), ..., (n-2, n-1)]
        - Count: n_qubits - 1 pairs
        - Connectivity: Only nearest-neighbor connections required

        This makes DataReuploading hardware-friendly for most quantum
        processors with linear or grid connectivity.

        **Single-Qubit Case:**

        When n_qubits=1, no entanglement is possible, so an empty list
        is returned. Single-qubit data re-uploading still creates an
        expressive feature map with L Fourier frequencies for L layers.

        **Caching:**

        Entanglement pairs are computed once at initialization and cached
        as an immutable tuple. This method returns a new list copy for
        API compatibility, ensuring callers cannot modify the internal state.

        Examples
        --------
        Standard multi-qubit configuration:

        >>> enc = DataReuploading(n_features=4, n_layers=3)
        >>> enc.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        Fewer qubits than features:

        >>> enc = DataReuploading(n_features=8, n_qubits=4)
        >>> enc.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        Single-qubit (no entanglement):

        >>> enc = DataReuploading(n_features=1, n_layers=5)
        >>> enc.get_entanglement_pairs()
        []

        Two qubits:

        >>> enc = DataReuploading(n_features=2)
        >>> enc.get_entanglement_pairs()
        [(0, 1)]

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts including CNOT count.
        resource_summary : Comprehensive resource analysis.
        """
        # Return a new list from the cached tuple for API compatibility
        # Callers can safely modify the returned list
        return list(self._entanglement_pairs)

    # =========================================================================
    # Circuit Generation
    # =========================================================================

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for a single data sample.

        Creates a quantum circuit that repeatedly encodes the input features
        across multiple layers, interleaved with entangling operations.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (1, n_features).
            Values are used as RY rotation angles (in radians).
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
        >>> enc = DataReuploading(n_features=4, n_layers=3)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> circuit = enc.get_circuit(x, backend='pennylane')
        >>> callable(circuit)
        True
        """
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]

        # Debug log if values are far outside the optimal [0, 2π] range.
        # RY rotations are 2π-periodic, so any input works mathematically,
        # but extreme values often indicate unscaled data that may not
        # produce meaningful phase differences between encoded states.
        if _logger.isEnabledFor(logging.DEBUG):
            x_min, x_max = float(x_validated.min()), float(x_validated.max())
            if abs(x_min) > _INPUT_RANGE_DEBUG_THRESHOLD or abs(x_max) > _INPUT_RANGE_DEBUG_THRESHOLD:
                _logger.debug(
                    "Input values [%.3g, %.3g] are outside optimal range [0, 2π]. "
                    "RY rotations are 2π-periodic, so any value works, but large "
                    "values may indicate unscaled data. Consider normalizing "
                    "features to [0, 2π] or [-π, π] for meaningful phase encoding.",
                    x_min,
                    x_max,
                )

        if backend == "pennylane":
            return self._to_pennylane(x_validated)
        elif backend == "qiskit":
            return self._to_qiskit(x_validated)
        elif backend == "cirq":
            return self._to_cirq(x_validated)
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

            Parallel processing is thread-safe due to DataReuploading's
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

        >>> enc = DataReuploading(n_features=4)
        >>> X = np.random.randn(10, 4)
        >>> circuits = enc.get_circuits(X, backend='pennylane')
        >>> len(circuits)
        10

        Parallel processing for large batches:

        >>> enc = DataReuploading(n_features=4)
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
                # Use _get_circuit_from_validated to skip redundant validation
                def generate_single(x: NDArray[np.floating[Any]]) -> CircuitType:
                    return self._get_circuit_from_validated(x, backend)

                # Map preserves order: result[i] corresponds to X_validated[i]
                circuits = list(executor.map(generate_single, X_validated))

            _logger.debug(
                "Parallel batch processing completed: generated %d circuits "
                "with max_workers=%s",
                len(circuits),
                max_workers,
            )
        else:
            # Sequential processing - use optimized path
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
        when processing batches, improving performance for large datasets.

        Parameters
        ----------
        x : NDArray
            Pre-validated input features of shape (n_features,).
            Must have already been validated by _validate_input().
        backend : BackendType
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.

        Notes
        -----
        This is an internal optimization method. External callers should use
        get_circuit() which includes full input validation.

        The method assumes x is a 1D array with the correct number of features
        and no NaN/Inf values.
        """
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

        Creates a callable that applies the data re-uploading encoding
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

        n_qubits = self._n_qubits
        n_layers = self.n_layers
        n_features = len(x)

        def circuit() -> None:
            """Apply the data re-uploading encoding gates."""
            for _ in range(n_layers):
                # Data encoding layer - RY rotations with cyclic feature mapping
                # Features beyond n_qubits are cyclically mapped to qubits
                for i in range(n_features):
                    qml.RY(x[i], wires=i % n_qubits)
                # Entangling layer - CNOT ladder
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

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

        qc = QuantumCircuit(self._n_qubits, name="DataReuploading")
        n_features = len(x)

        for _ in range(self.n_layers):
            # Data encoding layer with cyclic feature mapping
            # Features beyond n_qubits are cyclically mapped to qubits
            for i in range(n_features):
                qc.ry(float(x[i]), i % self._n_qubits)
            # Entangling layer
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

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
            Cirq circuit implementing the data re-uploading encoding.

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

        qubits = cirq.LineQubit.range(self._n_qubits)
        circuit = cirq.Circuit()
        n_features = len(x)

        for _ in range(self.n_layers):
            # Data encoding layer - RY rotations with cyclic feature mapping
            # Features beyond n_qubits are cyclically mapped to qubits
            for i in range(n_features):
                circuit.append(cirq.ry(float(x[i]))(qubits[i % self._n_qubits]))

            # Entangling layer - CNOT ladder
            for i in range(self._n_qubits - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

        return circuit

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Returns
        -------
        EncodingProperties
            Computed properties including:
            - n_qubits: Number of qubits in the circuit
            - depth: Exact circuit depth based on structure
            - gate_count: Total gates including RY and CNOT
            - is_entangling: True when n_qubits > 1
            - simulability: "not_simulable" for entangling circuits
        """
        n = self._n_qubits
        n_entangling = max(0, n - 1)

        # Gates per layer: n_features RY + (n_qubits-1) CNOT
        # With cyclic mapping, all features are encoded
        ry_gates = self.n_layers * self.n_features
        cnot_gates = self.n_layers * n_entangling

        return EncodingProperties(
            n_qubits=n,
            depth=self.depth,  # Use the property for exact calculation
            gate_count=ry_gates + cnot_gates,
            single_qubit_gates=ry_gates,
            two_qubit_gates=cnot_gates,
            parameter_count=ry_gates,  # Each RY gate uses one data parameter
            is_entangling=(n > 1),
            simulability="not_simulable" if n > 1 else "simulable",
            trainability_estimate=max(0.4, 0.9 - 0.05 * self.n_layers),
            notes=(
                f"Data re-uploading feature map with {self.n_layers} layers. "
                f"High expressivity via repeated encoding; add trainable "
                f"parameters for universal approximation."
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
            f"n_layers={self.n_layers}, "
            f"n_qubits={self._n_qubits})"
        )
