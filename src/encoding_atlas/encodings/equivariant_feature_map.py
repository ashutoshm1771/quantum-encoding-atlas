"""Equivariant quantum feature maps with provable symmetry guarantees.

This module implements mathematically rigorous equivariant quantum feature maps
that satisfy the equivariance property:

    U(g)|ψ(x)⟩ = |ψ(g·x)⟩   for all group elements g

where g is a group element, x is input data, U(g) is a unitary representation
of the group, and ψ is the feature map. This guarantees that the quantum state
transforms predictably under group actions on the input data.

Equivariant encodings provide several key advantages:

1. **Guaranteed Symmetry Preservation**: The quantum state transforms in a
   mathematically provable way under input transformations
2. **Reduced Hypothesis Space**: Learning is constrained to symmetric functions,
   improving sample efficiency
3. **Provable Generalization**: Theoretical guarantees on learning performance
4. **Algebraic Verification**: Can mathematically verify equivariance property

This module provides three main encoding classes:

- :class:`SO2EquivariantFeatureMap`: Equivariant to 2D rotations (SO(2) group)
- :class:`CyclicEquivariantFeatureMap`: Equivariant to cyclic shifts (Z_n group)
- :class:`SwapEquivariantFeatureMap`: Equivariant to pair swaps (S_2 group)

Mathematical Background
-----------------------
A quantum feature map ψ: X → H is **equivariant** with respect to a group G
if there exist:

1. A group action ·: G × X → X on the input space
2. A unitary representation U: G → U(H) on the Hilbert space

Such that: |ψ(g·x)⟩ = U(g)|ψ(x)⟩ for all g ∈ G and x ∈ X.

**SO(2) Rotation Equivariance**:

For 2D points (x, y), rotation by angle φ transforms:

    Input: (r cos θ, r sin θ) → (r cos(θ + φ), r sin(θ + φ))
    State: |ψ(r, θ)⟩ → |ψ(r, θ + φ)⟩ = U(φ)|ψ(r, θ)⟩

where U(φ) = exp(-iφL_z) is the rotation operator and L_z is the angular
momentum operator.

**Cyclic (Z_n) Equivariance**:

For n features, cyclic shift by k positions transforms:

    Input: (x₀, x₁, ..., xₙ₋₁) → (xₖ, xₖ₊₁, ..., xₖ₊ₙ₋₁ mod n)
    State: |ψ(x)⟩ → |ψ(σᵏ·x)⟩ = SWAP_k|ψ(x)⟩

where SWAP_k permutes qubits cyclically.

**Swap (S_2) Equivariance**:

For pairs of features, swapping each pair transforms:

    Input: (x₀, x₁, x₂, x₃, ...) → (x₁, x₀, x₃, x₂, ...)
    State: |ψ(x)⟩ → |ψ(swap·x)⟩ = (SWAP₀₁ ⊗ SWAP₂₃ ⊗ ...)|ψ(x)⟩

where SWAP_ij is the swap gate on qubits i and j.

Note on Data Preprocessing
--------------------------
Each equivariant encoding has specific input requirements tied to its symmetry
group. Proper preprocessing ensures the encoding correctly captures the intended
symmetry structure.

**SO2EquivariantFeatureMap (2D Rotation Symmetry)**:

Input must be 2D Cartesian coordinates [x, y]. The encoding internally converts
to polar coordinates (r, θ):

- **No normalization required**: The encoding handles arbitrary magnitudes.
  The radial component r is encoded via amplitude coefficients cₘ(r), while
  the angular component θ is encoded as phases e^{imθ}.

- **Centering recommended**: If your data has a natural center (e.g., centroid
  of a point cloud), translate data so this center is at the origin. Rotational
  symmetry is then meaningful around this point.

- **Radial function choice**: For data where distance from origin matters, use
  ``radial_function="gaussian"`` (default). For purely angular data where only
  direction matters, use ``radial_function="uniform"``.

- **Example preprocessing**::

      # Center data around mean
      x_centered = x - x.mean(axis=0)
      # Optionally scale to unit variance per dimension
      x_scaled = x_centered / x_centered.std(axis=0)

**CyclicEquivariantFeatureMap (Cyclic Shift Symmetry)**:

Input is an n-dimensional feature vector where cyclic permutation symmetry
is meaningful (e.g., periodic signals, ring-structured data):

- **Scaling to [0, 2π]**: Features are encoded via RY(xᵢ) rotations. For full
  expressivity, scale features to span [0, 2π] or [-π, π]::

      x_scaled = 2 * np.pi * (x - x.min()) / (x.max() - x.min())

- **Preserve relative ordering**: Since the encoding respects cyclic shifts,
  the relative ordering of features matters. Do not shuffle features randomly.

- **Periodic signals**: For time series with periodic structure, ensure the
  feature indices correspond to the natural ordering (e.g., time steps,
  spatial positions around a ring).

- **Uniform scaling**: Apply the same scaling to all features to preserve
  the cyclic symmetry. Feature-wise scaling breaks translational invariance.

**SwapEquivariantFeatureMap (Pair Swap Symmetry)**:

Input is a vector with an even number of features, organized into pairs
(x₀, x₁), (x₂, x₃), ... where swapping within pairs is a symmetry:

- **Pair structure**: Ensure features are organized so that indices (2i, 2i+1)
  form meaningful pairs. Each feature x[i] is directly encoded on qubit i via
  RY(x[i]), ensuring that swapping data within a pair is exactly equivalent
  to applying a SWAP gate on the corresponding qubits.

- **Scaling to rotation range**: Features are encoded via RY(xᵢ) rotations.
  For full expressivity, scale features to span [0, 2π] or [-π, π]::

      # Standardize features
      x_scaled = (x - x.mean()) / x.std()
      # Then scale to rotation-friendly range
      x_scaled = x_scaled * np.pi / 3  # Maps ~3σ to ±π

- **Physical pairs**: This encoding is ideal for data with natural pair
  structure: particle-antiparticle features, bidirectional measurements,
  or any data where pair exchanges are physically meaningful.

**General Recommendations**:

1. **Avoid extreme values**: Very large values (>10⁶) or very small values
   (<10⁻¹⁰) may cause numerical instability in some backends.

2. **NaN/Inf handling**: Remove or impute NaN and Inf values before encoding.
   The encodings do not perform explicit NaN checks for performance reasons.

3. **Batch consistency**: When processing batches, apply identical preprocessing
   to all samples to ensure consistent encoding behavior.

Verification
------------
All encoding classes provide built-in equivariance verification:

    >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=1)
    >>> x = np.array([0.5, 0.3])  # 2D point
    >>> phi = np.pi / 4  # Rotation angle
    >>> enc.verify_equivariance(x, phi)
    True

This verification checks that U(φ)|ψ(x)⟩ equals |ψ(rotate(x, φ))⟩ up to
numerical tolerance and global phase.

Use Cases
---------
Equivariant encodings are particularly suited for:

- **Symmetry-aware learning**: Problems with known symmetries (rotation,
  translation, permutation)
- **Physics simulations**: Systems with rotational or translational symmetry
- **Molecular property prediction**: Molecules with rotational/permutation
  symmetry
- **Graph neural networks**: Permutation-equivariant quantum encodings
- **Image classification**: Rotation-equivariant feature extraction
- **Sample-efficient learning**: Leveraging symmetry for better generalization

Limitations
-----------
- **Specialized structure**: More complex than general-purpose encodings
- **Group-specific**: Each encoding is tailored to a specific symmetry group
- **Circuit complexity**: May require specialized gates or decompositions
- **Limited flexibility**: Cannot handle approximate or broken symmetries

Debugging
---------
This module supports Python's standard logging for debugging equivariance
verification and circuit generation:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logger = logging.getLogger('encoding_atlas.encodings.equivariant_feature_map')
    >>> logger.setLevel(logging.DEBUG)

Debug logs include:
- Initialization parameters
- Equivariance verification results
- Group action and unitary computation details
- Circuit generation progress

Resource Analysis
-----------------
All equivariant encoding classes provide methods to analyze circuit resources
and plan hardware execution:

- ``gate_count_breakdown()``: Get detailed gate counts by type (RY, RZZ, CZ,
  state preparation gates, etc.). Returns a TypedDict with breakdown.

- ``resource_summary()``: Comprehensive breakdown including qubit count, depth,
  gate counts, symmetry group information, hardware requirements, and
  equivariance verification cost estimates.

- ``get_entanglement_pairs()``: List of qubit pairs with entangling gates.
  Useful for mapping to hardware connectivity.

- ``properties``: Access the ``EncodingProperties`` dataclass with theoretical
  circuit metrics (depth, gate counts, simulability classification).

**Equivariance Verification Cost**:

Each encoding provides multiple verification methods with different trade-offs:

+------------------------+------------------+------------------+----------------+
| Method                 | Memory           | Time             | Scalability    |
+========================+==================+==================+================+
| ``verify_equivariance``| O(2^n)           | O(2^n)           | ≤15 qubits     |
+------------------------+------------------+------------------+----------------+
| ``verify_statistical`` | O(n_shots)       | O(shots × depth) | Any size       |
+------------------------+------------------+------------------+----------------+
| ``verify_auto``        | Auto-selected    | Auto-selected    | Any size       |
+------------------------+------------------+------------------+----------------+

**Resource Summary Examples**::

    >>> # SO(2) encoding resources
    >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=2)
    >>> summary = enc.resource_summary()
    >>> summary['n_qubits']
    3
    >>> summary['symmetry_group']
    'SO(2)'
    >>> summary['angular_momenta']
    [-2, -1, 0, 1, 2]

    >>> # Cyclic encoding resources
    >>> enc = CyclicEquivariantFeatureMap(n_features=8, reps=2)
    >>> summary = enc.resource_summary()
    >>> summary['gate_counts']['total_two_qubit']
    16
    >>> summary['hardware_requirements']['connectivity']
    'ring'

    >>> # Swap encoding resources
    >>> enc = SwapEquivariantFeatureMap(n_features=6)
    >>> summary = enc.resource_summary()
    >>> summary['n_pairs']
    3
    >>> summary['symmetry_group']
    'S_2^3'

References
----------
.. [1] Larocca, M., et al. (2022). "Group-Invariant Quantum Machine Learning."
       PRX Quantum, 3, 030341.
.. [2] Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational Quantum
       Machine Learning." PRX Quantum, 4, 010328.
.. [3] Schatzki, L., et al. (2022). "Theoretical Guarantees for
       Permutation-Equivariant Quantum Neural Networks." npj Quantum
       Information, 8, 74.
.. [4] Skolik, A., et al. (2023). "Equivariant quantum circuits for learning
       on weighted graphs." npj Quantum Information, 9, 47.
.. [5] Nguyen, Q., et al. (2022). "Theory for Equivariant Quantum Neural
       Networks." arXiv:2210.08566.

See Also
--------
PauliFeatureMap : General-purpose feature map with configurable entanglement
IQPEncoding : IQP circuits with quadratic feature interactions
ZZFeatureMap : Feature map with ZZ entanglement
"""

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

# =============================================================================
# Module-Level Logger
# =============================================================================

_logger = logging.getLogger(__name__)

# =============================================================================
# Module-Level Constants
# =============================================================================

# Threshold for emitting warnings about large angular momentum in SO(2) encoding.
# max_angular_momentum > 3 requires more than 8 qubits (ceil(log2(2*3+1)) = 3 qubits)
# and O(2^n) state preparation gates, which may be impractical on NISQ devices.
_MAX_ANGULAR_MOMENTUM_WARNING_THRESHOLD: int = 3

# Threshold for emitting warnings about large feature count in cyclic encoding.
# Ring topology with >20 qubits may have limited connectivity on some hardware.
_LARGE_CYCLIC_FEATURE_WARNING_THRESHOLD: int = 20

# Threshold for automatic verification method selection.
# For systems with more than this many qubits, statistical verification is recommended.
#
# Memory analysis for exact verification:
#   - State vector: O(2^n) complex numbers = 2^n × 16 bytes
#   - Unitary matrix: O(2^2n) complex numbers = 2^2n × 16 bytes  ← BOTTLENECK
#
# The unitary matrix dominates memory usage:
#   - 10 qubits: ~16 MB unitary matrix (fast)
#   - 12 qubits: ~256 MB unitary matrix (reasonable for most machines)
#   - 13 qubits: ~1 GB unitary matrix (requires good hardware)
#   - 14 qubits: ~4 GB unitary matrix (requires workstation)
#   - 15 qubits: ~16 GB unitary matrix (exceeds most laptop RAM)
#
# We choose 12 as a conservative default that works reliably on typical hardware
# (laptops with 8-16 GB RAM). Users with more powerful machines can call
# verify_equivariance() directly for larger systems.
_EXACT_VERIFICATION_QUBIT_THRESHOLD: int = 12

# Default number of measurement shots for statistical verification.
# Higher values give more confidence but increase runtime.
_DEFAULT_STATISTICAL_VERIFICATION_SHOTS: int = 10000

# Default significance level for statistical hypothesis testing.
# Probability of Type I error (rejecting equivariance when it's true).
_DEFAULT_STATISTICAL_SIGNIFICANCE: float = 0.01

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base class and implementations
    "EquivariantFeatureMap",
    "SO2EquivariantFeatureMap",
    "CyclicEquivariantFeatureMap",
    "SwapEquivariantFeatureMap",
    # Type definitions for gate count breakdowns
    "SO2GateCountBreakdown",
    "CyclicGateCountBreakdown",
    "SwapGateCountBreakdown",
    # Type definitions for verification results
    "EquivarianceVerificationResult",
    "StatisticalVerificationResult",
]

# =============================================================================
# Type Variables
# =============================================================================

# Group element type variable (e.g., float for SO(2), int for Z_n)
G = TypeVar("G")

# =============================================================================
# Type Definitions
# =============================================================================


class SO2GateCountBreakdown(TypedDict):
    """Gate count breakdown for SO(2) equivariant encoding."""

    state_preparation: int
    """Number of state preparation gates for amplitude encoding."""

    phase_gates: int
    """Number of phase gates for angle encoding."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates (for state preparation)."""

    total: int
    """Total gate count."""


class CyclicGateCountBreakdown(TypedDict):
    """Gate count breakdown for cyclic equivariant encoding."""

    ry: int
    """Number of RY gates for feature encoding."""

    rzz: int
    """Number of RZZ gates for ring entanglement."""

    rx: int
    """Number of RX gates for translationally-invariant rotation."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates."""

    total: int
    """Total gate count."""


class SwapGateCountBreakdown(TypedDict):
    """Gate count breakdown for swap equivariant encoding.

    The swap equivariant circuit uses CZ (Controlled-Z) gates for entanglement
    rather than CNOT gates. CZ is symmetric under qubit exchange, which is
    essential for maintaining swap equivariance:
    ``SWAP · CZ(q0, q1) · SWAP = CZ(q1, q0) = CZ(q0, q1)``.
    """

    ry: int
    """Number of RY gates for direct feature encoding (one per qubit per layer)."""

    hadamard: int
    """Number of Hadamard gates for basis mixing."""

    cz: int
    """Number of CZ (Controlled-Z) gates for symmetric pair entanglement."""

    total_single_qubit: int
    """Total number of single-qubit gates (RY + Hadamard)."""

    total_two_qubit: int
    """Total number of two-qubit gates (CZ)."""

    total: int
    """Total gate count."""


class EquivarianceVerificationResult(TypedDict):
    """Result of exact equivariance verification."""

    equivariant: bool
    """Whether the encoding is equivariant within tolerance."""

    overlap: float
    """Absolute value of inner product between U(g)|ψ(x)⟩ and |ψ(g·x)⟩."""

    tolerance: float
    """Numerical tolerance used for verification."""

    group_element: Any
    """The group element tested."""


class StatisticalVerificationResult(TypedDict):
    """Result of statistical equivariance verification.

    Statistical verification uses measurement sampling to compare probability
    distributions, making it scalable to larger systems where exact state
    vector verification would require exponential memory.
    """

    equivariant: bool
    """Whether equivariance holds at the given confidence level."""

    p_value: float
    """P-value from the statistical test (chi-squared or similar)."""

    test_statistic: float
    """Test statistic value (chi-squared statistic)."""

    significance: float
    """Significance level used for the test."""

    n_shots: int
    """Number of measurement shots used."""

    group_element: Any
    """The group element tested."""

    method: str
    """Statistical test method used (e.g., 'chi_squared')."""

    confidence_level: float
    """Confidence level of the result (1 - significance)."""


# =============================================================================
# Abstract Base Class
# =============================================================================


class EquivariantFeatureMap(BaseEncoding, Generic[G]):
    """Abstract base class for equivariant quantum feature maps.

    This class defines the interface for quantum encodings that satisfy
    mathematical equivariance with respect to a symmetry group G:

        U(g)|ψ(x)⟩ = |ψ(g·x)⟩   for all g ∈ G, x ∈ X

    Subclasses must implement:
    - group_action: How the group acts on input data
    - unitary_representation: Unitary matrix for group elements
    - group_generators: Generators of the symmetry group
    - Standard BaseEncoding methods (get_circuit, etc.)

    The base class provides generic equivariance verification that works
    for any symmetry group.

    Type Parameters
    ---------------
    G : type
        The type of group elements (e.g., float for SO(2), int for Z_n)

    Notes
    -----
    **Equivariance vs. Invariance**:

    - Equivariance: Output transforms with input (U(g)|ψ(x)⟩ = |ψ(g·x)⟩)
    - Invariance: Output unchanged under transformation (ψ(g·x) = ψ(x))

    Invariance is a special case of equivariance where U(g) = I.

    **Verification Strategy**:

    Testing equivariance on group generators is sufficient to verify
    equivariance for the entire group, since any group element can be
    expressed as a product of generators.
    """

    __slots__ = ()

    @abstractmethod
    def group_action(
        self, g: G, x: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Apply group element g to input data x.

        This defines how the symmetry group acts on the classical input space.

        Parameters
        ----------
        g : G
            Group element (type depends on specific group).
        x : ndarray
            Input data to transform.

        Returns
        -------
        ndarray
            Transformed data g·x.

        Examples
        --------
        For SO(2) rotations:

        >>> enc = SO2EquivariantFeatureMap()
        >>> x = np.array([1.0, 0.0])
        >>> g = np.pi / 2  # 90-degree rotation
        >>> enc.group_action(g, x)
        array([0., 1.])

        For cyclic shifts:

        >>> enc = CyclicEquivariantFeatureMap(n_features=4)
        >>> x = np.array([1, 2, 3, 4])
        >>> g = 1  # Shift by 1
        >>> enc.group_action(g, x)
        array([2, 3, 4, 1])
        """
        ...

    @abstractmethod
    def unitary_representation(self, g: G) -> NDArray[np.complexfloating[Any, Any]]:
        """Get unitary matrix U(g) representing group element g.

        This defines the group representation on the quantum Hilbert space.

        Parameters
        ----------
        g : G
            Group element.

        Returns
        -------
        ndarray
            Unitary matrix U(g) of shape (2^n_qubits, 2^n_qubits).

        Notes
        -----
        The unitary must satisfy the group homomorphism property:

            U(g₁ ∘ g₂) = U(g₁) @ U(g₂)

        where ∘ is the group operation.

        Examples
        --------
        For SO(2), returns rotation operator exp(-iφL_z):

        >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=1)
        >>> phi = np.pi / 4
        >>> U = enc.unitary_representation(phi)
        >>> U.shape
        (8, 8)
        >>> np.allclose(U @ U.conj().T, np.eye(8))
        True
        """
        ...

    @abstractmethod
    def group_generators(self) -> list[G]:
        """Return generators of the symmetry group.

        Testing equivariance on generators is sufficient to verify
        equivariance for the entire group (by the group homomorphism property).

        Returns
        -------
        list of G
            List of group generators.

        Examples
        --------
        For Z_n (cyclic group), the generator is a single shift:

        >>> enc = CyclicEquivariantFeatureMap(n_features=4)
        >>> enc.group_generators()
        [1]

        For SO(2) (continuous), return representative angles:

        >>> enc = SO2EquivariantFeatureMap()
        >>> enc.group_generators()
        [0.7853981633974483, 1.5707963267948966, 3.141592653589793]
        """
        ...

    def _get_state_vector(
        self,
        x: NDArray[np.floating[Any]],
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Get quantum state vector |ψ(x)⟩ for input data x.

        This is used internally for equivariance verification.

        Parameters
        ----------
        x : ndarray
            Input data.

        Returns
        -------
        ndarray
            Quantum state vector of shape (2^n_qubits,).

        Notes
        -----
        This method generates a PennyLane circuit and executes it to
        obtain the state vector. For verification purposes, PennyLane
        is used as the default backend due to its efficient state
        vector simulation.
        """
        import pennylane as qml

        # Generate circuit
        circuit = self.get_circuit(x, backend="pennylane")

        # Create a device and execute the circuit
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def get_state():
            circuit()
            return qml.state()

        state = get_state()
        return np.array(state, dtype=complex)

    def verify_equivariance(
        self,
        x: ArrayLike,
        g: G,
        atol: float = 1e-10,
    ) -> bool:
        """Verify equivariance property: U(g)|ψ(x)⟩ = |ψ(g·x)⟩.

        This checks that the encoding satisfies the equivariance condition
        for a specific input and group element. The verification accounts
        for global phase ambiguity (two states are equal if they differ
        only by a global phase factor).

        Parameters
        ----------
        x : array_like
            Input data to test.
        g : G
            Group element to test.
        atol : float, default=1e-10
            Absolute tolerance for numerical comparison.

        Returns
        -------
        bool
            True if equivariance holds within tolerance, False otherwise.

        Notes
        -----
        **Phase Ambiguity**: Two quantum states are physically equivalent
        if they differ by a global phase: |ψ⟩ ≈ e^{iα}|ψ⟩. The verification
        uses the overlap |⟨ψ|φ⟩| which is invariant under global phase.

        **Numerical Tolerance**: The default tolerance of 1e-10 is suitable
        for exact state vector simulation. For hardware or noisy simulation,
        use a larger tolerance (e.g., 1e-6).

        Examples
        --------
        Verify SO(2) rotation equivariance:

        >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=1)
        >>> x = np.array([0.5, 0.3])
        >>> phi = np.pi / 4
        >>> enc.verify_equivariance(x, phi)
        True

        Verify cyclic shift equivariance:

        >>> enc = CyclicEquivariantFeatureMap(n_features=4)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> enc.verify_equivariance(x, 1)  # Shift by 1
        True
        """
        _logger.debug(f"Verifying equivariance for group element {g}")

        # Validate and convert input
        x_array = self._validate_input(x)
        if x_array.ndim == 2:
            if x_array.shape[0] != 1:
                raise ValueError(
                    "verify_equivariance requires a single sample, got batch"
                )
            x_array = x_array[0]

        # Get |ψ(x)⟩
        state_x = self._get_state_vector(x_array)

        # Compute U(g)|ψ(x)⟩
        U_g = self.unitary_representation(g)
        left_side = U_g @ state_x

        # Compute |ψ(g·x)⟩
        gx = self.group_action(g, x_array)
        right_side = self._get_state_vector(gx)

        # Compare states up to global phase
        # Two states are equal up to phase iff |⟨ψ|φ⟩| = 1
        overlap = np.abs(np.vdot(left_side, right_side))
        is_equivariant = np.isclose(overlap, 1.0, atol=atol)

        _logger.debug(
            f"Equivariance check: overlap = {overlap:.10f}, "
            f"threshold = {1.0 - atol:.10f}, "
            f"result = {is_equivariant}"
        )

        return bool(is_equivariant)

    def verify_equivariance_detailed(
        self,
        x: ArrayLike,
        g: G,
        atol: float = 1e-10,
    ) -> EquivarianceVerificationResult:
        """Verify equivariance and return detailed results.

        This is similar to :meth:`verify_equivariance` but returns
        detailed information about the verification, including the
        overlap value and tolerance used.

        Parameters
        ----------
        x : array_like
            Input data to test.
        g : G
            Group element to test.
        atol : float, default=1e-10
            Absolute tolerance for numerical comparison.

        Returns
        -------
        EquivarianceVerificationResult
            Dictionary containing:
            - equivariant: bool (whether equivariance holds)
            - overlap: float (|⟨U(g)ψ(x)|ψ(g·x)⟩|)
            - tolerance: float (tolerance used)
            - group_element: G (the group element tested)

        Examples
        --------
        >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=1)
        >>> x = np.array([0.5, 0.3])
        >>> result = enc.verify_equivariance_detailed(x, np.pi / 4)
        >>> result['equivariant']
        True
        >>> result['overlap']
        0.9999999999...
        """
        # Validate and convert input
        x_array = self._validate_input(x)
        if x_array.ndim == 2:
            if x_array.shape[0] != 1:
                raise ValueError(
                    "verify_equivariance_detailed requires a single sample"
                )
            x_array = x_array[0]

        # Get |ψ(x)⟩
        state_x = self._get_state_vector(x_array)

        # Compute U(g)|ψ(x)⟩
        U_g = self.unitary_representation(g)
        left_side = U_g @ state_x

        # Compute |ψ(g·x)⟩
        gx = self.group_action(g, x_array)
        right_side = self._get_state_vector(gx)

        # Compute overlap
        overlap = float(np.abs(np.vdot(left_side, right_side)))
        is_equivariant = np.isclose(overlap, 1.0, atol=atol)

        return {
            "equivariant": bool(is_equivariant),
            "overlap": overlap,
            "tolerance": atol,
            "group_element": g,
        }

    def verify_equivariance_on_generators(
        self,
        x: ArrayLike,
        atol: float = 1e-10,
    ) -> bool:
        """Verify equivariance on all group generators.

        Testing equivariance on generators is sufficient to guarantee
        equivariance for the entire group (by the group homomorphism
        property). This is more efficient than testing all group elements.

        Parameters
        ----------
        x : array_like
            Input data to test.
        atol : float, default=1e-10
            Absolute tolerance for numerical comparison.

        Returns
        -------
        bool
            True if equivariance holds for all generators, False otherwise.

        Examples
        --------
        >>> enc = CyclicEquivariantFeatureMap(n_features=4)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> enc.verify_equivariance_on_generators(x)
        True
        """
        generators = self.group_generators()
        _logger.debug(f"Testing equivariance on {len(generators)} generators")

        for i, g in enumerate(generators):
            _logger.debug(f"Testing generator {i + 1}/{len(generators)}: {g}")
            if not self.verify_equivariance(x, g, atol):
                _logger.warning(f"Equivariance failed for generator {g}")
                return False

        _logger.debug("Equivariance verified for all generators")
        return True

    # -------------------------------------------------------------------------
    # Statistical Verification Methods (Scalable)
    # -------------------------------------------------------------------------

    @abstractmethod
    def _apply_group_unitary_as_gates(self, g: G) -> None:
        """Apply U(g) as quantum gates for statistical verification.

        This method applies the unitary representation of group element g
        directly as quantum gates, enabling measurement-based verification
        that doesn't require full state vector computation.

        Parameters
        ----------
        g : G
            Group element to represent as gates.

        Notes
        -----
        This method is called within a quantum circuit context (e.g., inside
        a PennyLane QNode). Implementations should use the backend's native
        gate operations.

        Subclasses must implement this method to enable statistical
        verification via :meth:`verify_equivariance_statistical`.
        """
        ...

    def _samples_to_distribution(
        self,
        samples: NDArray[np.integer[Any]],
        n_qubits: int,
    ) -> NDArray[np.floating[Any]]:
        """Convert measurement samples to probability distribution.

        Parameters
        ----------
        samples : ndarray
            Array of measurement outcomes, shape (n_shots,) or (n_shots, n_qubits).
            Each row represents a measurement outcome in the computational basis.
        n_qubits : int
            Number of qubits in the system.

        Returns
        -------
        ndarray
            Probability distribution over all 2^n computational basis states.
            Shape: (2^n_qubits,)

        Notes
        -----
        This method handles both single-integer samples (e.g., from Qiskit)
        and per-qubit bit arrays (e.g., from PennyLane).
        """
        dim = 2**n_qubits
        counts = np.zeros(dim, dtype=float)

        # Handle different sample formats
        samples_array = np.asarray(samples)

        if samples_array.ndim == 1:
            # Samples are integers representing basis states
            for sample in samples_array:
                if 0 <= sample < dim:
                    counts[int(sample)] += 1
        elif samples_array.ndim == 2:
            # Samples are bit arrays (n_shots, n_qubits)
            for shot in samples_array:
                # Convert bit array to integer (little-endian convention)
                idx = sum(int(bit) << i for i, bit in enumerate(shot))
                if 0 <= idx < dim:
                    counts[idx] += 1
        else:
            raise ValueError(
                f"Expected samples with 1 or 2 dimensions, got {samples_array.ndim}"
            )

        # Convert to probability distribution
        total = counts.sum()
        if total > 0:
            return counts / total
        else:
            # Return uniform distribution if no valid samples
            return np.ones(dim, dtype=float) / dim

    def verify_equivariance_statistical(
        self,
        x: ArrayLike,
        g: G,
        n_shots: int = _DEFAULT_STATISTICAL_VERIFICATION_SHOTS,
        significance: float = _DEFAULT_STATISTICAL_SIGNIFICANCE,
    ) -> StatisticalVerificationResult:
        """Verify equivariance using statistical measurement comparison.

        This method verifies the equivariance property U(g)|ψ(x)⟩ = |ψ(g·x)⟩
        by comparing measurement probability distributions from both circuits.
        Unlike exact verification, this method:

        - Scales polynomially: O(n_shots × circuit_depth) instead of O(2^n)
        - Works on real quantum hardware (not just simulators)
        - Provides statistical confidence rather than exact verification

        Parameters
        ----------
        x : array_like
            Input data to test.
        g : G
            Group element to test.
        n_shots : int, default=10000
            Number of measurement shots for each circuit.
            Higher values increase confidence but also runtime.
        significance : float, default=0.01
            Significance level for the statistical test.
            This is the probability of Type I error (falsely rejecting
            equivariance when it actually holds).

        Returns
        -------
        StatisticalVerificationResult
            Dictionary containing:
            - equivariant: bool - whether test passed
            - p_value: float - p-value from chi-squared test
            - test_statistic: float - chi-squared statistic
            - significance: float - significance level used
            - n_shots: int - number of shots used
            - group_element: G - the group element tested
            - method: str - "chi_squared"
            - confidence_level: float - (1 - significance)

        Raises
        ------
        ValueError
            If n_shots < 100 (insufficient for statistical testing).
        ValueError
            If significance not in (0, 1).

        Notes
        -----
        **Statistical Test**: Uses the chi-squared test for homogeneity to
        compare the two probability distributions. The null hypothesis is
        that both circuits produce the same distribution (equivariance holds).

        **Interpretation**:
        - p_value > significance → equivariance likely holds
        - p_value ≤ significance → equivariance likely violated

        **Recommended Shots**:
        - Quick check: n_shots=1000 (faster, less confident)
        - Standard: n_shots=10000 (good balance)
        - High confidence: n_shots=100000 (slower, more confident)

        **Memory Efficiency**: This method uses O(n_shots) memory regardless
        of the number of qubits, making it practical for large systems.

        Examples
        --------
        Verify SO(2) rotation equivariance statistically:

        >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=2)
        >>> x = np.array([0.5, 0.3])
        >>> result = enc.verify_equivariance_statistical(x, np.pi / 4)
        >>> result['equivariant']
        True
        >>> result['confidence_level']
        0.99

        Verify cyclic equivariance with higher confidence:

        >>> enc = CyclicEquivariantFeatureMap(n_features=8)
        >>> x = np.random.randn(8)
        >>> result = enc.verify_equivariance_statistical(
        ...     x, 1, n_shots=50000, significance=0.001
        ... )
        >>> result['confidence_level']
        0.999

        See Also
        --------
        verify_equivariance : Exact verification (O(2^n) memory).
        verify_equivariance_detailed : Exact verification with details.
        """
        # Validate parameters
        if n_shots < 100:
            raise ValueError(
                f"n_shots must be at least 100 for meaningful statistics, got {n_shots}"
            )
        if not (0 < significance < 1):
            raise ValueError(f"significance must be in (0, 1), got {significance}")

        _logger.debug(
            f"Statistical verification: g={g}, n_shots={n_shots}, "
            f"significance={significance}"
        )

        # Validate and convert input
        x_array = self._validate_input(x)
        if x_array.ndim == 2:
            if x_array.shape[0] != 1:
                raise ValueError(
                    "verify_equivariance_statistical requires a single sample"
                )
            x_array = x_array[0]

        import pennylane as qml

        # Create device with shots
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=n_shots)

        # Circuit 1: Prepare |ψ(x)⟩, apply U(g), measure
        @qml.qnode(dev)
        def circuit_transformed() -> NDArray[np.integer[Any]]:
            """Generate samples from U(g)|ψ(x)⟩."""
            # Prepare |ψ(x)⟩
            circuit_func = self.get_circuit(x_array, backend="pennylane")
            circuit_func()
            # Apply U(g) as gates
            self._apply_group_unitary_as_gates(g)
            # Return computational basis samples
            return qml.sample()

        # Circuit 2: Prepare |ψ(g·x)⟩, measure
        gx = self.group_action(g, x_array)

        @qml.qnode(dev)
        def circuit_direct() -> NDArray[np.integer[Any]]:
            """Generate samples from |ψ(g·x)⟩."""
            circuit_func = self.get_circuit(gx, backend="pennylane")
            circuit_func()
            return qml.sample()

        # Execute circuits and get samples
        samples_transformed = circuit_transformed()
        samples_direct = circuit_direct()

        # Convert samples to probability distributions
        dist_transformed = self._samples_to_distribution(
            samples_transformed, self.n_qubits
        )
        dist_direct = self._samples_to_distribution(samples_direct, self.n_qubits)

        # Perform chi-squared test for homogeneity
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        dist_transformed_safe = dist_transformed + epsilon
        dist_direct_safe = dist_direct + epsilon

        # Normalize after adding epsilon
        dist_transformed_safe /= dist_transformed_safe.sum()
        dist_direct_safe /= dist_direct_safe.sum()

        # Expected counts under null hypothesis (distributions are equal)
        # Use pooled estimate as expected distribution
        pooled = (dist_transformed_safe + dist_direct_safe) / 2
        expected_counts = pooled * n_shots

        # Observed counts
        observed_transformed = dist_transformed * n_shots
        observed_direct = dist_direct * n_shots

        # Chi-squared statistic: sum of (O - E)^2 / E for both samples
        # Only include bins where expected count > 5 (standard chi-squared assumption)
        valid_bins = expected_counts > 5
        if valid_bins.sum() < 2:
            # Not enough valid bins for chi-squared test
            # Fall back to comparing distributions directly
            _logger.warning(
                "Insufficient valid bins for chi-squared test, "
                "using direct distribution comparison"
            )
            # Use total variation distance as a simple metric
            tv_distance = 0.5 * np.abs(dist_transformed - dist_direct).sum()
            # Consider equivariant if TV distance is small
            is_equivariant = tv_distance < 0.1
            return {
                "equivariant": is_equivariant,
                "p_value": 1.0 - tv_distance if is_equivariant else tv_distance,
                "test_statistic": tv_distance,
                "significance": significance,
                "n_shots": n_shots,
                "group_element": g,
                "method": "total_variation",
                "confidence_level": 1.0 - significance,
            }

        # Compute chi-squared statistic
        chi_squared = (
            (observed_transformed[valid_bins] - expected_counts[valid_bins]) ** 2
            / expected_counts[valid_bins]
        ).sum() + (
            (observed_direct[valid_bins] - expected_counts[valid_bins]) ** 2
            / expected_counts[valid_bins]
        ).sum()

        # Degrees of freedom: (number of valid bins - 1)
        dof = max(1, valid_bins.sum() - 1)

        # Compute p-value using scipy's chi-squared distribution
        try:
            from scipy.stats import chi2

            p_value = float(1.0 - chi2.cdf(chi_squared, dof))
        except ImportError:
            # Fallback if scipy not available: use simple approximation
            _logger.warning(
                "scipy not available, using approximate p-value calculation"
            )
            # Approximate p-value using normal approximation for large dof
            z = (chi_squared - dof) / np.sqrt(2 * dof)
            p_value = float(0.5 * (1 - np.tanh(z / np.sqrt(2))))

        # Determine if equivariant based on significance level
        is_equivariant = p_value > significance

        _logger.debug(
            f"Statistical verification: chi2={chi_squared:.4f}, dof={dof}, "
            f"p_value={p_value:.6f}, equivariant={is_equivariant}"
        )

        return {
            "equivariant": is_equivariant,
            "p_value": p_value,
            "test_statistic": float(chi_squared),
            "significance": significance,
            "n_shots": n_shots,
            "group_element": g,
            "method": "chi_squared",
            "confidence_level": 1.0 - significance,
        }

    def verify_equivariance_auto(
        self,
        x: ArrayLike,
        g: G,
        atol: float = 1e-10,
        n_shots: int = _DEFAULT_STATISTICAL_VERIFICATION_SHOTS,
        significance: float = _DEFAULT_STATISTICAL_SIGNIFICANCE,
    ) -> bool:
        """Verify equivariance using automatically selected method.

        Automatically chooses between exact and statistical verification
        based on the number of qubits:

        - n_qubits ≤ 15: Uses exact verification (more precise)
        - n_qubits > 15: Uses statistical verification (scalable)

        This provides the best of both worlds: precise results for small
        systems and scalable verification for large systems.

        Parameters
        ----------
        x : array_like
            Input data to test.
        g : G
            Group element to test.
        atol : float, default=1e-10
            Absolute tolerance for exact verification.
        n_shots : int, default=10000
            Number of shots for statistical verification.
        significance : float, default=0.01
            Significance level for statistical verification.

        Returns
        -------
        bool
            True if equivariance holds, False otherwise.

        Examples
        --------
        >>> enc = CyclicEquivariantFeatureMap(n_features=4)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> enc.verify_equivariance_auto(x, 1)  # Uses exact (n_qubits=4)
        True

        >>> enc_large = CyclicEquivariantFeatureMap(n_features=20)
        >>> x_large = np.random.randn(20)
        >>> enc_large.verify_equivariance_auto(x_large, 1)  # Uses statistical
        True

        See Also
        --------
        verify_equivariance : Always uses exact verification.
        verify_equivariance_statistical : Always uses statistical verification.
        """
        if self.n_qubits <= _EXACT_VERIFICATION_QUBIT_THRESHOLD:
            _logger.debug(
                f"Auto-selecting exact verification (n_qubits={self.n_qubits} "
                f"<= threshold={_EXACT_VERIFICATION_QUBIT_THRESHOLD})"
            )
            return self.verify_equivariance(x, g, atol)
        else:
            _logger.debug(
                f"Auto-selecting statistical verification (n_qubits={self.n_qubits} "
                f"> threshold={_EXACT_VERIFICATION_QUBIT_THRESHOLD})"
            )
            result = self.verify_equivariance_statistical(x, g, n_shots, significance)
            return result["equivariant"]


# =============================================================================
# SO(2) Rotation-Equivariant Feature Map
# =============================================================================


class SO2EquivariantFeatureMap(EquivariantFeatureMap[float]):
    """SO(2) rotation-equivariant quantum feature map for 2D data.

    This encoding is equivariant to 2D rotations, satisfying:

        U(φ)|ψ(r, θ)⟩ = |ψ(r, θ + φ)⟩

    where (r, θ) are polar coordinates of a 2D point and φ is a rotation
    angle. The encoding uses angular momentum eigenstates |m⟩ with phase
    encoding for the angle θ and amplitude encoding for the radius r:

        |ψ(r, θ)⟩ = Σₘ cₘ(r) · e^{imθ} |m⟩

    where:
    - m ∈ {-max_m, ..., max_m} are angular momentum quantum numbers
    - cₘ(r) are radial amplitude functions (rotation-invariant)
    - e^{imθ} encodes the angle as phases
    - |m⟩ are angular momentum basis states

    The rotation operator is U(φ) = exp(-iφL_z) where L_z is the angular
    momentum operator with eigenvalues m.

    Parameters
    ----------
    n_features : int, default=2
        Number of classical features to encode. For SO(2) equivariance,
        this must be exactly 2 (representing 2D Cartesian coordinates [x, y]).
        This parameter is provided for API consistency with other encodings
        in the registry. Passing any value other than 2 will raise a ValueError.
    max_angular_momentum : int, default=1
        Maximum absolute angular momentum quantum number.
        The encoding uses m ∈ {-max_m, ..., +max_m}, requiring
        n_qubits = ⌈log₂(2·max_m + 1)⌉ qubits.
    radial_function : {"gaussian", "uniform"}, default="gaussian"
        Radial amplitude function cₘ(r):
        - "gaussian": cₘ(r) ∝ exp(-(r - |m|)²/(2σ²))
        - "uniform": cₘ(r) = 1/√(2·max_m + 1) (all equal)
    radial_sigma : float, default=1.0
        Width parameter σ for Gaussian radial function.

    Attributes
    ----------
    max_angular_momentum : int
        Maximum angular momentum quantum number.
    radial_function : str
        Type of radial amplitude function.
    radial_sigma : float
        Width parameter for Gaussian radial function.
    angular_momenta : list of int
        List of angular momentum quantum numbers used.

    Examples
    --------
    Create rotation-equivariant encoding and verify equivariance:

    >>> from encoding_atlas import SO2EquivariantFeatureMap
    >>> import numpy as np
    >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=1)
    >>> enc.n_features
    2
    >>> enc.n_qubits
    2

    Encode a 2D point and verify rotation equivariance:

    >>> x = np.array([0.5, 0.3])
    >>> phi = np.pi / 4  # 45-degree rotation
    >>> enc.verify_equivariance(x, phi)
    True

    Test on multiple rotation angles:

    >>> angles = [0, np.pi/6, np.pi/4, np.pi/2, np.pi]
    >>> all(enc.verify_equivariance(x, phi) for phi in angles)
    True

    Generate PennyLane circuit:

    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Notes
    -----
    **Input Format**: This encoding expects 2D Cartesian coordinates [x, y]
    and internally converts to polar coordinates (r, θ).

    **Angular Momentum Basis**: For max_angular_momentum=1 with 2 qubits,
    the encoding uses the spin-1 triplet states:

    - |m = +1⟩ ↔ |00⟩
    - |m =  0⟩ ↔ (|01⟩ + |10⟩)/√2
    - |m = -1⟩ ↔ |11⟩

    **Rotation Operator**: The unitary U(φ) = exp(-iφL_z) acts as:

        U(φ)|m⟩ = e^{-imφ}|m⟩

    This causes the phase factors e^{imθ} to become e^{im(θ-φ)}, but we
    use the convention where input θ → θ + φ, so U(φ) = exp(+iφL_z).

    **Radial Encoding**: The radial function cₘ(r) controls how the radius
    is encoded in state amplitudes:

    - Gaussian: Encodes radius by distributing amplitude across m values
    - Uniform: Ignores radius (pure angular encoding)

    References
    ----------
    .. [1] Larocca, M., et al. (2022). "Group-Invariant Quantum Machine
           Learning." PRX Quantum, 3, 030341.
    .. [2] Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational
           Quantum Machine Learning." PRX Quantum, 4, 010328.

    See Also
    --------
    CyclicEquivariantFeatureMap : Cyclic shift equivariance
    SwapEquivariantFeatureMap : Pair swap equivariance
    PauliFeatureMap : General-purpose feature map
    """

    _VALID_RADIAL_FUNCTIONS: frozenset[str] = frozenset({"gaussian", "uniform"})

    __slots__ = (
        "max_angular_momentum",
        "radial_function",
        "radial_sigma",
        "angular_momenta",
        "_n_states",
        "_n_qubits",
        "_basis_map",
    )

    def __init__(
        self,
        n_features: int = 2,
        max_angular_momentum: int = 1,
        radial_function: Literal["gaussian", "uniform"] = "gaussian",
        radial_sigma: float = 1.0,
    ) -> None:
        """Initialize SO(2) equivariant feature map.

        Parameters
        ----------
        n_features : int, default=2
            Number of classical features to encode. Must be exactly 2 for
            SO(2) equivariance (representing 2D Cartesian coordinates [x, y]).
            This parameter is provided for API consistency with other encodings.
        max_angular_momentum : int, default=1
            Maximum angular momentum quantum number.
        radial_function : {"gaussian", "uniform"}, default="gaussian"
            Type of radial amplitude function.
        radial_sigma : float, default=1.0
            Width parameter for Gaussian radial function.

        Raises
        ------
        TypeError
            If n_features is not an integer.
        ValueError
            If n_features is not exactly 2 (SO(2) requires 2D input).
        ValueError
            If max_angular_momentum is not a non-negative integer.
        ValueError
            If radial_function is not one of {"gaussian", "uniform"}.
        ValueError
            If radial_sigma is not positive.

        Notes
        -----
        The ``n_features`` parameter enforces the mathematical constraint that
        SO(2) rotation equivariance is only defined for 2D data. Attempting to
        use a different value will raise a clear error message explaining this
        constraint.
        """
        # =================================================================
        # Validate n_features (SO(2) requires exactly 2D input)
        # =================================================================
        # Type check: must be integer (excluding bool which is int subclass)
        if isinstance(n_features, bool) or not isinstance(n_features, int):
            raise TypeError(
                f"n_features must be an integer, got {type(n_features).__name__}"
            )

        # Value check: SO(2) equivariance requires exactly 2D input
        # This is a fundamental mathematical constraint, not an arbitrary limit
        _SO2_REQUIRED_FEATURES = 2
        if n_features != _SO2_REQUIRED_FEATURES:
            raise ValueError(
                f"SO2EquivariantFeatureMap requires n_features={_SO2_REQUIRED_FEATURES} "
                f"(2D Cartesian coordinates [x, y]) for SO(2) rotation equivariance. "
                f"Got n_features={n_features}. "
                f"For higher-dimensional data with rotational symmetry, consider "
                f"projecting to 2D or using a different encoding."
            )

        # =================================================================
        # Validate max_angular_momentum
        # =================================================================
        # Type check: must be integer (excluding bool)
        if isinstance(max_angular_momentum, bool) or not isinstance(
            max_angular_momentum, int
        ):
            raise TypeError(
                f"max_angular_momentum must be an integer, "
                f"got {type(max_angular_momentum).__name__}"
            )

        # Value check: must be non-negative
        if max_angular_momentum < 0:
            raise ValueError(
                f"max_angular_momentum must be a non-negative integer, "
                f"got {max_angular_momentum!r}"
            )

        # =================================================================
        # Validate radial_function
        # =================================================================
        if radial_function not in self._VALID_RADIAL_FUNCTIONS:
            raise ValueError(
                f"radial_function must be one of {sorted(self._VALID_RADIAL_FUNCTIONS)}, "
                f"got {radial_function!r}"
            )

        # =================================================================
        # Validate radial_sigma
        # =================================================================
        # Type check for radial_sigma (must be numeric)
        if not isinstance(radial_sigma, (int, float)) or isinstance(radial_sigma, bool):
            raise TypeError(
                f"radial_sigma must be a positive number, "
                f"got {type(radial_sigma).__name__}"
            )

        # Value check: must be positive and finite
        if radial_sigma <= 0:
            raise ValueError(f"radial_sigma must be positive, got {radial_sigma!r}")

        if not np.isfinite(radial_sigma):
            raise ValueError(f"radial_sigma must be finite, got {radial_sigma!r}")

        # =================================================================
        # Call parent constructor
        # =================================================================
        super().__init__(
            n_features,
            max_angular_momentum=max_angular_momentum,
            radial_function=radial_function,
            radial_sigma=radial_sigma,
        )

        # Store encoding-specific parameters
        self.max_angular_momentum = max_angular_momentum
        self.radial_function = radial_function
        self.radial_sigma = radial_sigma

        # Build angular momentum quantum numbers
        self.angular_momenta = list(
            range(-max_angular_momentum, max_angular_momentum + 1)
        )
        self._n_states = len(self.angular_momenta)

        # Number of qubits needed (at least 1 qubit)
        self._n_qubits = (
            max(1, int(np.ceil(np.log2(self._n_states)))) if self._n_states > 0 else 1
        )

        # Precompute basis state mapping
        self._basis_map = self._compute_angular_momentum_basis()

        # Warn about large angular momentum (exponential state preparation cost)
        if max_angular_momentum > _MAX_ANGULAR_MOMENTUM_WARNING_THRESHOLD:
            warnings.warn(
                f"Large max_angular_momentum ({max_angular_momentum}) requires "
                f"{self._n_qubits} qubits and O(2^{self._n_qubits}) gates for state "
                f"preparation. This may not be practical for hardware deployment. "
                f"Consider using a smaller max_angular_momentum for NISQ devices.",
                UserWarning,
                stacklevel=2,
            )
            _logger.warning(
                "Large max_angular_momentum (%d) with %d qubits may have "
                "scalability issues due to O(2^n) state preparation",
                max_angular_momentum,
                self._n_qubits,
            )

        _logger.debug(
            f"Initialized SO2EquivariantFeatureMap with "
            f"n_features={n_features}, max_angular_momentum={max_angular_momentum}, "
            f"n_states={self._n_states}, n_qubits={self._n_qubits}"
        )

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._n_qubits

    @property
    def depth(self) -> int:
        """Circuit depth."""
        # Amplitude preparation depth is O(2^n) in general
        # Phase encoding is O(1) per qubit
        # Conservative estimate
        return 2 * self._n_states

    def _compute_angular_momentum_basis(
        self,
    ) -> dict[int, NDArray[np.complexfloating[Any, Any]]]:
        """Compute angular momentum basis states in computational basis.

        Returns
        -------
        dict
            Mapping from angular momentum m to computational basis state vector.

        Notes
        -----
        For small systems (max_m=1, 2 qubits), uses explicit spin-1 triplet
        states. For larger systems, uses standard computational basis ordering.
        """
        basis = {}
        dim = 2**self._n_qubits

        if self.max_angular_momentum == 1 and self._n_qubits == 2:
            # Spin-1 triplet states (symmetric subspace)
            basis[+1] = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
            basis[0] = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(
                2
            )  # (|01⟩+|10⟩)/√2
            basis[-1] = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        else:
            # General case: use standard basis ordering
            for i, m in enumerate(self.angular_momenta):
                state = np.zeros(dim, dtype=complex)
                if i < dim:  # Ensure we don't exceed Hilbert space dimension
                    state[i] = 1
                basis[m] = state

        return basis

    def _radial_amplitudes(
        self,
        r: float,
    ) -> dict[int, float]:
        """Compute radial amplitude coefficients cₘ(r).

        Parameters
        ----------
        r : float
            Radial coordinate.

        Returns
        -------
        dict
            Normalized amplitude coefficients for each m.
        """
        amplitudes = {}

        if self.radial_function == "gaussian":
            # Gaussian amplitude: cₘ(r) ∝ exp(-(r - |m|)²/(2σ²))
            for m in self.angular_momenta:
                amplitudes[m] = np.exp(
                    -((r - abs(m)) ** 2) / (2 * self.radial_sigma**2)
                )
        elif self.radial_function == "uniform":
            # Uniform amplitude: all equal (ignores radius)
            for m in self.angular_momenta:
                amplitudes[m] = 1.0
        else:
            raise ValueError(f"Unknown radial function: {self.radial_function}")

        # Normalize (handle edge case where all amplitudes are effectively zero)
        norm = np.sqrt(sum(abs(amplitudes[m]) ** 2 for m in amplitudes))
        if norm < 1e-300 or not np.isfinite(norm):
            # For very large radii, all amplitudes may underflow to 0
            # Fall back to uniform distribution to maintain valid quantum state
            uniform_amp = 1.0 / np.sqrt(len(amplitudes))
            return dict.fromkeys(amplitudes, uniform_amp)
        return {m: amplitudes[m] / norm for m in amplitudes}

    def _encode_state(
        self,
        x: NDArray[np.floating[Any]],
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Encode 2D point into quantum state vector.

        Parameters
        ----------
        x : ndarray
            2D point [x, y].

        Returns
        -------
        ndarray
            Quantum state vector.
        """
        # Convert to polar coordinates
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = np.arctan2(x[1], x[0])

        # Get radial amplitudes
        c = self._radial_amplitudes(r)

        # Construct state: |ψ⟩ = Σₘ cₘ(r) · e^{imθ} |m⟩
        dim = 2**self._n_qubits
        state = np.zeros(dim, dtype=complex)

        for m in self.angular_momenta:
            phase = np.exp(1j * m * theta)
            state += c[m] * phase * self._basis_map[m]

        return state

    def group_action(
        self,
        phi: float,
        x: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Rotate 2D point by angle phi.

        Parameters
        ----------
        phi : float
            Rotation angle in radians.
        x : ndarray
            2D point [x, y] to rotate.

        Returns
        -------
        ndarray
            Rotated point.
        """
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = np.arctan2(x[1], x[0])
        new_theta = theta + phi
        return np.array([r * np.cos(new_theta), r * np.sin(new_theta)])

    def unitary_representation(
        self,
        phi: float,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Get rotation operator U(φ) = exp(+iφL_z).

        Parameters
        ----------
        phi : float
            Rotation angle in radians.

        Returns
        -------
        ndarray
            Unitary rotation matrix.

        Notes
        -----
        The rotation operator acts as U(φ)|m⟩ = e^{-imφ}|m⟩ in the standard
        convention. However, since we want θ → θ + φ in the input space to
        correspond to e^{imθ} → e^{im(θ+φ)} in the state, we use the opposite
        sign convention: U(φ) = exp(+iφL_z), giving e^{+imφ} phase factors.

        The operator acts as identity on the subspace orthogonal to the
        angular momentum basis (e.g., the singlet state for spin-1).
        """
        dim = 2**self._n_qubits
        # Start with identity matrix (acts as identity on orthogonal complement)
        U = np.eye(dim, dtype=complex)

        # Apply phase shifts to angular momentum basis states
        for m in self.angular_momenta:
            basis_state = self._basis_map[m]
            # Use +imφ to match the θ → θ + φ convention
            phase = np.exp(1j * m * phi)
            # Subtract identity contribution and add rotated state
            # U = I - |m⟩⟨m| + e^{imφ}|m⟩⟨m| = I + (e^{imφ} - 1)|m⟩⟨m|
            projector = np.outer(basis_state, basis_state.conj())
            U += (phase - 1) * projector

        return U

    def group_generators(self) -> list[float]:
        """Return representative rotation angles for testing.

        Returns
        -------
        list of float
            Representative rotation angles.

        Notes
        -----
        SO(2) is a continuous group, so we return multiple test angles
        to verify equivariance across the group.
        """
        return [np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]

    def _apply_group_unitary_as_gates(self, phi: float) -> None:
        """Apply rotation U(φ) = exp(+iφL_z) as quantum gates.

        For SO(2) encoding, the rotation operator applies phase shifts
        to angular momentum basis states: U(φ)|m⟩ = e^{+imφ}|m⟩.

        Parameters
        ----------
        phi : float
            Rotation angle in radians.

        Notes
        -----
        This method is called within a PennyLane QNode context during
        statistical verification. It applies phase gates to implement
        the rotation operator in the computational basis.

        For the spin-1 case (max_angular_momentum=1, 2 qubits), we use
        the symmetric subspace representation. For general cases, we
        apply phase shifts to each computational basis state.
        """
        import pennylane as qml

        if self.max_angular_momentum == 1 and self._n_qubits == 2:
            # Special case: spin-1 triplet states
            # |+1⟩ = |00⟩, |0⟩ = (|01⟩+|10⟩)/√2, |-1⟩ = |11⟩
            # Apply phases: e^{+iφ} to |00⟩, e^{0} to |01⟩,|10⟩, e^{-iφ} to |11⟩
            # This is equivalent to RZ(φ) on qubit 0 and RZ(φ) on qubit 1
            # since phase(00) = phase(0)+phase(0), phase(11) = phase(1)+phase(1)
            # We need: |00⟩ → e^{iφ}|00⟩, |11⟩ → e^{-iφ}|11⟩
            # Using RZ(-φ/2) on both qubits gives the right relative phases
            for q in range(self._n_qubits):
                qml.PhaseShift(-phi / 2, wires=q)
            # Apply global Z rotation to shift |00⟩ and |11⟩ appropriately
            qml.ctrl(qml.PhaseShift(phi, wires=1), control=0, control_values=[0])
            qml.ctrl(qml.PhaseShift(-phi, wires=1), control=0, control_values=[1])
        else:
            # General case: apply diagonal phase operator
            # Each computational basis state |i⟩ corresponds to angular momentum m_i
            # Apply phase e^{+im_i φ} to each state
            # This is implemented as controlled phase gates
            for idx, m in enumerate(self.angular_momenta):
                if idx >= 2**self._n_qubits:
                    break
                if m == 0:
                    continue  # No phase for m=0
                phase = m * phi
                # Apply phase to computational basis state |idx⟩
                # Use multi-controlled phase gate
                bits = [(idx >> q) & 1 for q in range(self._n_qubits)]
                # Apply controlled phase gate
                if self._n_qubits == 1:
                    if bits[0] == 1:
                        qml.PhaseShift(phase, wires=0)
                else:
                    # For multi-qubit case, use controlled operations
                    # Create control pattern based on bit string
                    control_wires = list(range(self._n_qubits - 1))
                    control_values = bits[:-1]
                    target_wire = self._n_qubits - 1
                    target_phase = phase if bits[-1] == 1 else 0
                    if target_phase != 0:
                        qml.ctrl(
                            qml.PhaseShift(target_phase, wires=target_wire),
                            control=control_wires,
                            control_values=control_values,
                        )

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for input data.

        Strict variant: a 2D input with more than one row raises
        :class:`ValueError`. Use :meth:`get_circuits` for batches.

        Parameters
        ----------
        x : array_like
            2D input point [x, y] (shape (2,) or (1, 2)).
        backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Quantum circuit in the specified backend format.

        Raises
        ------
        ValueError
            If input has multiple samples or backend is not supported.
        """
        x_array = self._validate_input(x)
        if x_array.ndim == 2:
            if x_array.shape[0] != 1:
                raise ValueError(
                    "get_circuit requires a single sample, use get_circuits for batches"
                )
            x_array = x_array[0]
        return self._get_circuit_from_validated(x_array, backend)

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Dispatch SO(2)-equivariant circuit construction.

        Encoding-specific seam called by ``get_circuit`` and the inherited
        ``get_circuits`` template method. Accepts a 1D array (the normal
        case) or a 2D single-sample ``(1, 2)`` array for robustness when
        called directly.
        """
        if x.ndim == 2:
            x = x[0]

        if backend == "pennylane":
            return self._to_pennylane(x)
        elif backend == "qiskit":
            return self._to_qiskit(x)
        elif backend == "cirq":
            return self._to_cirq(x)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _to_pennylane(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate PennyLane circuit.

        Parameters
        ----------
        x : ndarray
            2D input point.

        Returns
        -------
        callable
            PennyLane quantum function.
        """
        import pennylane as qml

        # Precompute state vector for efficiency
        target_state = self._encode_state(x)

        def circuit() -> None:
            """SO(2) equivariant encoding circuit."""
            # Use PennyLane's state preparation
            qml.StatePrep(target_state, wires=range(self._n_qubits))

        return circuit

    def _to_qiskit(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate Qiskit circuit.

        Parameters
        ----------
        x : ndarray
            2D input point.

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit.
        """
        from qiskit import QuantumCircuit

        # Precompute state vector
        target_state = self._encode_state(x)

        # Create circuit and initialize state
        circuit = QuantumCircuit(self._n_qubits)
        circuit.initialize(target_state, range(self._n_qubits))

        return circuit

    def _to_cirq(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate Cirq circuit using unitary state preparation.

        Constructs a unitary matrix U such that U|0...0⟩ = |ψ(x)⟩ and
        applies it via ``cirq.MatrixGate``. The unitary is built by placing
        the target state as the first column of a matrix and completing it
        to an orthonormal basis via QR decomposition (Householder
        reflections).

        Parameters
        ----------
        x : ndarray
            2D input point.

        Returns
        -------
        cirq.Circuit
            Cirq quantum circuit that prepares the encoded state.

        Notes
        -----
        Cirq does not provide a built-in state preparation primitive
        comparable to PennyLane's ``StatePrep`` or Qiskit's
        ``QuantumCircuit.initialize``.  Instead we construct the full
        2^n × 2^n unitary explicitly.  For the qubit counts used by
        SO(2) equivariant encoding (typically 1--4 qubits for
        ``max_angular_momentum`` up to ~7), the matrix is small and the
        construction is negligible in cost.

        The QR-based orthonormal completion is the same numerically
        stable algorithm used by the amplitude encoding Cirq backend
        (see ``amplitude.py``).
        """
        import cirq

        target_state = self._encode_state(x)
        dim = len(target_state)

        # Build unitary U with U[:, 0] = target_state via QR decomposition.
        # Place target_state as the first column, remaining columns are
        # standard basis vectors to guarantee full rank.
        A = np.eye(dim, dtype=np.complex128)
        A[:, 0] = target_state

        Q, R = np.linalg.qr(A)

        # Phase correction: QR gives Q[:, 0] = target_state * e^{-iθ}
        # where R[0, 0] = e^{iθ}.  Multiply column 0 by conj(alpha) to
        # recover the exact target state as the first column.
        alpha = np.vdot(target_state, Q[:, 0])
        Q[:, 0] *= np.conj(alpha)

        qubits = cirq.LineQubit.range(self._n_qubits)
        circuit = cirq.Circuit()
        circuit.append(cirq.MatrixGate(Q).on(*qubits))

        return circuit

    def gate_count_breakdown(self) -> SO2GateCountBreakdown:
        """Get breakdown of gate counts by type.

        Returns
        -------
        SO2GateCountBreakdown
            Dictionary with detailed gate counts.

        Notes
        -----
        Gate counts for SO(2) encoding depend on state preparation method:
        - State preparation: O(2^n) single-qubit gates
        - Phase encoding: O(n) single-qubit gates
        """
        # State preparation requires preparing an arbitrary state
        # This is O(2^n) gates in general
        n_state_prep = 2**self._n_qubits - 1  # Typical state prep cost

        return {
            "state_preparation": n_state_prep,
            "phase_gates": self._n_qubits,  # Phase rotations
            "total_single_qubit": n_state_prep + self._n_qubits,
            "total_two_qubit": 0,  # State prep may use CNOTs, but exact count varies
            "total": n_state_prep + self._n_qubits,
        }

    def _compute_properties(self) -> EncodingProperties:
        """Compute encoding properties."""
        gate_breakdown = self.gate_count_breakdown()

        return EncodingProperties(
            n_qubits=self.n_qubits,
            depth=self.depth,
            gate_count=gate_breakdown["total"],
            single_qubit_gates=gate_breakdown["total_single_qubit"],
            two_qubit_gates=gate_breakdown["total_two_qubit"],
            parameter_count=0,  # No trainable parameters
            is_entangling=True,  # State preparation generally creates entanglement
            simulability="not_simulable",  # Requires full quantum state
        )

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get entanglement pairs (empty for state preparation).

        Returns
        -------
        list[tuple[int, int]]
            Empty list. SO(2) encoding uses state preparation
            which implicitly creates entanglement without explicit
            two-qubit gates.

        Notes
        -----
        State preparation (qml.StatePrep) may internally use
        entangling gates, but this is backend-dependent and not
        part of the logical circuit structure.

        Examples
        --------
        >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=1)
        >>> enc.get_entanglement_pairs()
        []
        """
        return []

    def resource_summary(self) -> dict[str, Any]:
        """Generate comprehensive resource summary.

        Provides a complete view of circuit resources, encoding characteristics,
        and hardware requirements. This method combines information from
        multiple sources for hardware planning and comparison.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            **Circuit Structure**:
                - ``n_qubits``: Number of qubits
                - ``n_features``: Number of classical features (always 2)
                - ``depth``: Circuit depth
                - ``max_angular_momentum``: Maximum angular momentum quantum number

            **Gate Counts**:
                - ``gate_counts``: Full breakdown from ``gate_count_breakdown()``

            **Encoding Characteristics**:
                - ``is_entangling``: True (state preparation creates entanglement)
                - ``simulability``: "not_simulable"
                - ``trainability_estimate``: Trainability score (0.0-1.0)

            **Symmetry Information**:
                - ``symmetry_group``: "SO(2)"
                - ``angular_momenta``: List of angular momentum values used

            **Equivariance Verification Cost**:
                - ``verification_cost``: Memory and time complexity for verification

            **Hardware Requirements**:
                - ``connectivity``: Required qubit connectivity
                - ``native_gates``: List of required gate types
                - ``state_preparation``: Note about state preparation requirements

        Examples
        --------
        >>> enc = SO2EquivariantFeatureMap(max_angular_momentum=1)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        2
        >>> summary['symmetry_group']
        'SO(2)'
        >>> summary['angular_momenta']
        [-1, 0, 1]

        See Also
        --------
        gate_count_breakdown : Detailed gate count breakdown.
        properties : Basic encoding properties.
        verify_equivariance : Verify equivariance property.
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties

        return {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "max_angular_momentum": self.max_angular_momentum,
            "radial_function": self.radial_function,
            "radial_sigma": self.radial_sigma,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": props.is_entangling,
            "simulability": props.simulability,
            "trainability_estimate": props.trainability_estimate,
            # Symmetry information
            "symmetry_group": "SO(2)",
            "angular_momenta": self.angular_momenta,
            "n_angular_states": self._n_states,
            # Equivariance verification cost
            "verification_cost": {
                "exact": {
                    "memory": f"O(2^{self.n_qubits})",
                    "time": f"O(2^{self.n_qubits})",
                    "recommended_max_qubits": _EXACT_VERIFICATION_QUBIT_THRESHOLD,
                },
                "statistical": {
                    "memory": "O(n_shots)",
                    "time": "O(n_shots × circuit_depth)",
                    "default_shots": _DEFAULT_STATISTICAL_VERIFICATION_SHOTS,
                    "scalable": True,
                },
            },
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": "all-to-all",
                "native_gates": ["StatePrep"],
                "state_preparation": (
                    f"Requires arbitrary state preparation with "
                    f"{self._n_states} angular momentum states"
                ),
            },
            # Entanglement details
            "n_entanglement_pairs": 0,
            "entanglement_pairs": [],
            # Verification methods available
            "verification_methods": [
                "verify_equivariance (exact)",
                "verify_equivariance_statistical (scalable)",
                "verify_equivariance_auto (automatic selection)",
            ],
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
            f"max_angular_momentum={self.max_angular_momentum}, "
            f"radial_function={self.radial_function!r}, "
            f"radial_sigma={self.radial_sigma})"
        )


# =============================================================================
# Cyclic (Z_n) Equivariant Feature Map
# =============================================================================


class CyclicEquivariantFeatureMap(EquivariantFeatureMap[int]):
    """Z_n cyclic-equivariant quantum feature map.

    This encoding is equivariant to cyclic shifts of input features, satisfying:

        SWAP_σ|ψ(x)⟩ = |ψ(σ·x)⟩

    where σ is a cyclic shift operator and SWAP_σ permutes qubits accordingly.

    The encoding uses a translationally-invariant circuit structure:
    - Same single-qubit gates on all qubits
    - Ring entanglement with periodic boundary conditions
    - All circuit elements respect the cyclic symmetry

    This ensures that cyclically shifting the input is equivalent to
    cyclically permuting the qubits in the output state.

    Parameters
    ----------
    n_features : int
        Number of features (also determines number of qubits).
    reps : int, default=2
        Number of encoding layer repetitions.
    coupling_strength : float, default=π/4
        Strength of RZZ entangling gates.

    Attributes
    ----------
    reps : int
        Number of encoding layer repetitions.
    coupling_strength : float
        Entangling gate strength.

    Examples
    --------
    Create cyclic-equivariant encoding and verify equivariance:

    >>> from encoding_atlas import CyclicEquivariantFeatureMap
    >>> import numpy as np
    >>> enc = CyclicEquivariantFeatureMap(n_features=4)
    >>> enc.n_qubits
    4

    Verify cyclic shift equivariance:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> enc.verify_equivariance(x, 1)  # Shift by 1
    True
    >>> enc.verify_equivariance(x, 2)  # Shift by 2
    True

    Test on all possible shifts:

    >>> all(enc.verify_equivariance(x, k) for k in range(4))
    True

    Notes
    -----
    **Translational Invariance**: The circuit has identical structure
    at each qubit position, creating a ring topology with periodic
    boundary conditions.

    **Circuit Structure**: Each layer consists of:
    1. RY(xᵢ) on qubit i (feature encoding)
    2. RZZ(θ) on all nearest-neighbor pairs in a ring
    3. RX(π/6) on all qubits (uniform rotation)

    **Group Structure**: Z_n is the cyclic group with elements
    {0, 1, 2, ..., n-1} and operation (k₁ + k₂) mod n.

    References
    ----------
    .. [1] Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational
           Quantum Machine Learning." PRX Quantum, 4, 010328.

    See Also
    --------
    SO2EquivariantFeatureMap : Rotation equivariance for 2D data
    SwapEquivariantFeatureMap : Pair swap equivariance
    """

    __slots__ = ("reps", "coupling_strength", "_entanglement_pairs")

    def __init__(
        self,
        n_features: int,
        reps: int = 2,
        coupling_strength: float = np.pi / 4,
    ) -> None:
        """Initialize cyclic equivariant feature map.

        Parameters
        ----------
        n_features : int
            Number of features (and qubits). Must be at least 2 for
            meaningful cyclic symmetry (Z_n with n >= 2).
        reps : int, default=2
            Number of encoding layer repetitions.
        coupling_strength : float, default=π/4
            RZZ coupling strength. Must be a finite number.

        Raises
        ------
        TypeError
            If n_features is not an integer (including bool).
        ValueError
            If n_features < 2 (cyclic symmetry requires at least 2 features).
        ValueError
            If reps < 1.
        TypeError
            If coupling_strength is not a number (including bool).
        ValueError
            If coupling_strength is not finite (inf or NaN).
        """
        # =================================================================
        # Validate n_features (cyclic symmetry requires n >= 2)
        # =================================================================
        if isinstance(n_features, bool) or not isinstance(n_features, int):
            raise TypeError(
                f"n_features must be an integer, got {type(n_features).__name__}"
            )
        if n_features < 2:
            raise ValueError(
                f"CyclicEquivariantFeatureMap requires n_features >= 2 for "
                f"meaningful cyclic symmetry (Z_n group with n >= 2). "
                f"Got n_features={n_features}. "
                f"For single-feature encoding, use AngleEncoding instead."
            )

        # =================================================================
        # Validate reps
        # =================================================================
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
            raise ValueError(f"reps must be a positive integer, got {reps!r}")

        # =================================================================
        # Validate coupling_strength
        # =================================================================
        if not isinstance(coupling_strength, (int, float)) or isinstance(
            coupling_strength, bool
        ):
            raise TypeError(
                f"coupling_strength must be a number, "
                f"got {type(coupling_strength).__name__}"
            )
        if not np.isfinite(coupling_strength):
            raise ValueError(
                f"coupling_strength must be finite, got {coupling_strength!r}"
            )

        # Call parent constructor
        super().__init__(
            n_features,
            reps=reps,
            coupling_strength=coupling_strength,
        )

        self.reps = reps
        self.coupling_strength = coupling_strength

        # Cache entanglement pairs for performance
        self._entanglement_pairs: list[tuple[int, int]] = [
            (i, (i + 1) % n_features) for i in range(n_features)
        ]

        # Warn about large feature counts with ring topology
        if n_features > _LARGE_CYCLIC_FEATURE_WARNING_THRESHOLD:
            warnings.warn(
                f"Large n_features ({n_features}) with ring topology may have "
                f"limited connectivity on some quantum hardware. Linear chains "
                f"are more common than circular topologies on superconducting devices.",
                UserWarning,
                stacklevel=2,
            )
            _logger.warning(
                "Large feature count (%d) with ring topology may have "
                "hardware compatibility issues",
                n_features,
            )

        _logger.debug(
            f"Initialized CyclicEquivariantFeatureMap with "
            f"n_features={n_features}, reps={reps}"
        )

    @property
    def n_qubits(self) -> int:
        """Number of qubits (equal to n_features)."""
        return self.n_features

    @property
    def depth(self) -> int:
        """Circuit depth."""
        # Each rep has: RY layer (depth 1) + RZZ ring (depth 1) + RX layer (depth 1)
        return 3 * self.reps

    def group_action(
        self,
        k: int,
        x: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Cyclically shift features by k positions.

        Parameters
        ----------
        k : int
            Number of positions to shift (can be negative).
        x : ndarray
            Input features.

        Returns
        -------
        ndarray
            Cyclically shifted features.
        """
        return np.roll(x, -k)

    def unitary_representation(
        self,
        k: int,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Get qubit permutation unitary for shift by k.

        Parameters
        ----------
        k : int
            Number of positions to shift.

        Returns
        -------
        ndarray
            Permutation unitary matrix.

        Notes
        -----
        The permutation maps computational basis states:
        |i⟩ → |σᵏ(i)⟩ where σᵏ cyclically permutes qubit indices.
        """
        n = self.n_qubits
        dim = 2**n
        U = np.zeros((dim, dim))

        for i in range(dim):
            # Extract bit string
            bits = [(i >> q) & 1 for q in range(n)]
            # Apply cyclic shift
            shifted = bits[-k:] + bits[:-k]
            # Reconstruct integer
            j = sum(b << q for q, b in enumerate(shifted))
            U[j, i] = 1

        return U

    def group_generators(self) -> list[int]:
        """Return generator of cyclic group.

        Returns
        -------
        list of int
            Single generator [1] (shift by one position).
        """
        return [1]

    def _apply_group_unitary_as_gates(self, k: int) -> None:
        """Apply cyclic permutation by k positions as SWAP gates.

        For cyclic equivariant encoding, the unitary representation of
        a k-position shift is a cyclic permutation of qubits:
        |q_0, q_1, ..., q_{n-1}⟩ → |q_k, q_{k+1}, ..., q_{k-1}⟩

        Parameters
        ----------
        k : int
            Number of positions to shift (can be negative).

        Notes
        -----
        This method is called within a PennyLane QNode context during
        statistical verification. It implements the cyclic permutation
        using a sequence of SWAP gates.

        A cyclic shift by k positions can be decomposed into k single-position
        shifts, where each single shift is implemented using n-1 SWAP gates.
        """
        import pennylane as qml

        n = self.n_qubits
        # Normalize k to be in range [0, n)
        k_normalized = k % n

        if k_normalized == 0:
            return  # No operation needed

        # Implement cyclic shift using SWAP gates
        # A cyclic shift by 1 position: q_0 ← q_1, q_1 ← q_2, ..., q_{n-1} ← q_0
        # This is achieved by: SWAP(n-2, n-1), SWAP(n-3, n-2), ..., SWAP(0, 1)
        # Repeat k times for shift by k positions
        for _ in range(k_normalized):
            # Single cyclic shift: move qubit 0 to position n-1
            for i in range(n - 1):
                qml.SWAP(wires=[i, i + 1])

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get RZZ entanglement pairs for ring topology.

        Returns
        -------
        list[tuple[int, int]]
            List of (qubit_i, qubit_j) pairs for RZZ gates.
            Includes periodic boundary: (n-1, 0).

        Examples
        --------
        >>> enc = CyclicEquivariantFeatureMap(n_features=4)
        >>> enc.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3), (3, 0)]
        """
        return list(self._entanglement_pairs)

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for input data.

        Strict variant: a 2D input with more than one row raises
        :class:`ValueError`. Use :meth:`get_circuits` for batches.

        Parameters
        ----------
        x : array_like
            Input features.
        backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Quantum circuit in the specified backend format.
        """
        x_array = self._validate_input(x)
        if x_array.ndim == 2:
            if x_array.shape[0] != 1:
                raise ValueError("get_circuit requires a single sample")
            x_array = x_array[0]
        return self._get_circuit_from_validated(x_array, backend)

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Dispatch cyclic-equivariant circuit construction.

        Encoding-specific seam called by ``get_circuit`` and the inherited
        ``get_circuits`` template method. Accepts a 1D array (the normal
        case) or a 2D single-sample ``(1, n_features)`` array for
        robustness when called directly.
        """
        if x.ndim == 2:
            x = x[0]

        if backend == "pennylane":
            return self._to_pennylane(x)
        elif backend == "qiskit":
            return self._to_qiskit(x)
        elif backend == "cirq":
            return self._to_cirq(x)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _to_pennylane(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate PennyLane circuit.

        Parameters
        ----------
        x : ndarray
            Input features.

        Returns
        -------
        callable
            PennyLane quantum function.
        """
        import pennylane as qml

        n = self.n_qubits

        def circuit() -> None:
            """Cyclic-equivariant encoding circuit."""
            for _rep in range(self.reps):
                # Layer 1: Equivariant local encoding
                for i in range(n):
                    qml.RY(x[i], wires=i)

                # Layer 2: Ring entanglement (translationally invariant)
                for i in range(n):
                    j = (i + 1) % n  # Periodic boundary
                    qml.IsingZZ(self.coupling_strength, wires=[i, j])

                # Layer 3: Translationally invariant rotation
                for i in range(n):
                    qml.RX(np.pi / 6, wires=i)

        return circuit

    def _to_qiskit(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate Qiskit circuit.

        Parameters
        ----------
        x : ndarray
            Input features.

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit.
        """
        import numpy as np
        from qiskit import QuantumCircuit

        n = self.n_qubits
        circuit = QuantumCircuit(n)

        for _rep in range(self.reps):
            # Layer 1: RY encoding
            for i in range(n):
                circuit.ry(x[i], i)

            # Layer 2: RZZ ring entanglement
            # RZZ(θ) implements exp(-i θ/2 Z⊗Z), same convention as
            # PennyLane's IsingZZ(θ) — no conversion factor needed.
            for i in range(n):
                j = (i + 1) % n
                circuit.rzz(self.coupling_strength, i, j)

            # Layer 3: RX rotation
            for i in range(n):
                circuit.rx(np.pi / 6, i)

        return circuit

    def _to_cirq(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate Cirq circuit.

        Parameters
        ----------
        x : ndarray
            Input features.

        Returns
        -------
        Circuit
            Cirq quantum circuit.
        """
        import cirq

        n = self.n_qubits
        qubits = cirq.LineQubit.range(n)
        circuit = cirq.Circuit()

        for _rep in range(self.reps):
            # Layer 1: RY encoding
            circuit.append([cirq.ry(x[i])(qubits[i]) for i in range(n)])

            # Layer 2: RZZ ring entanglement
            # IsingZZ(φ) = exp(-i φ/2 Z⊗Z)
            # Cirq's ZZPowGate(exponent=t) = exp(i π t Z⊗Z) up to global phase,
            # with relative phase e^(iπt) on the -1 eigenspace.
            # Matching: t = φ / π  (see also qaoa_encoding.py).
            for i in range(n):
                j = (i + 1) % n
                circuit.append(
                    cirq.ZZPowGate(exponent=self.coupling_strength / np.pi)(
                        qubits[i], qubits[j]
                    )
                )

            # Layer 3: RX rotation
            circuit.append([cirq.rx(np.pi / 6)(qubits[i]) for i in range(n)])

        return circuit

    def gate_count_breakdown(self) -> CyclicGateCountBreakdown:
        """Get breakdown of gate counts by type.

        Returns
        -------
        CyclicGateCountBreakdown
            Dictionary with detailed gate counts.
        """
        n = self.n_qubits

        ry_count = n * self.reps
        rzz_count = n * self.reps  # Ring topology: n pairs
        rx_count = n * self.reps

        return {
            "ry": ry_count,
            "rzz": rzz_count,
            "rx": rx_count,
            "total_single_qubit": ry_count + rx_count,
            "total_two_qubit": rzz_count,
            "total": ry_count + rzz_count + rx_count,
        }

    def _compute_properties(self) -> EncodingProperties:
        """Compute encoding properties."""
        gate_breakdown = self.gate_count_breakdown()

        return EncodingProperties(
            n_qubits=self.n_qubits,
            depth=self.depth,
            gate_count=gate_breakdown["total"],
            single_qubit_gates=gate_breakdown["total_single_qubit"],
            two_qubit_gates=gate_breakdown["total_two_qubit"],
            parameter_count=0,
            is_entangling=True,
            simulability="not_simulable",
        )

    def resource_summary(self) -> dict[str, Any]:
        """Generate comprehensive resource summary.

        Provides a complete view of circuit resources, encoding characteristics,
        and hardware requirements. This method combines information from
        multiple sources for hardware planning and comparison.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            **Circuit Structure**:
                - ``n_qubits``: Number of qubits
                - ``n_features``: Number of classical features
                - ``depth``: Circuit depth
                - ``reps``: Number of encoding repetitions

            **Gate Counts**:
                - ``gate_counts``: Full breakdown from ``gate_count_breakdown()``

            **Encoding Characteristics**:
                - ``is_entangling``: True
                - ``simulability``: "not_simulable"
                - ``trainability_estimate``: Trainability score (0.0-1.0)

            **Symmetry Information**:
                - ``symmetry_group``: "Z_n" with the value of n
                - ``cyclic_order``: The order of the cyclic group

            **Equivariance Verification Cost**:
                - ``verification_cost``: Memory and time complexity for verification

            **Hardware Requirements**:
                - ``connectivity``: Required qubit connectivity (ring)
                - ``native_gates``: List of required gate types

            **Entanglement Details**:
                - ``n_entanglement_pairs``: Number of RZZ pairs per layer
                - ``entanglement_pairs``: List of (control, target) qubit pairs

        Examples
        --------
        >>> enc = CyclicEquivariantFeatureMap(n_features=4, reps=2)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['symmetry_group']
        'Z_4'
        >>> summary['n_entanglement_pairs']
        4

        See Also
        --------
        gate_count_breakdown : Detailed gate count breakdown.
        properties : Basic encoding properties.
        verify_equivariance : Verify equivariance property.
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties
        pairs = self.get_entanglement_pairs()

        return {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            "coupling_strength": self.coupling_strength,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": props.is_entangling,
            "simulability": props.simulability,
            "trainability_estimate": props.trainability_estimate,
            # Symmetry information
            "symmetry_group": f"Z_{self.n_features}",
            "cyclic_order": self.n_features,
            # Equivariance verification cost
            "verification_cost": {
                "exact": {
                    "memory": f"O(2^{self.n_qubits})",
                    "time": f"O(2^{self.n_qubits})",
                    "recommended_max_qubits": _EXACT_VERIFICATION_QUBIT_THRESHOLD,
                },
                "statistical": {
                    "memory": "O(n_shots)",
                    "time": "O(n_shots × circuit_depth)",
                    "default_shots": _DEFAULT_STATISTICAL_VERIFICATION_SHOTS,
                    "scalable": True,
                },
            },
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": "ring",
                "native_gates": ["RY", "RZZ", "RX"],
            },
            # Entanglement details
            "n_entanglement_pairs": len(pairs),
            "entanglement_pairs": pairs,
            # Verification methods available
            "verification_methods": [
                "verify_equivariance (exact)",
                "verify_equivariance_statistical (scalable)",
                "verify_equivariance_auto (automatic selection)",
            ],
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
            f"coupling_strength={self.coupling_strength})"
        )


# =============================================================================
# Swap (S_2) Equivariant Feature Map
# =============================================================================


class SwapEquivariantFeatureMap(EquivariantFeatureMap[list[bool]]):
    r"""S_2 swap-equivariant quantum feature map for paired features.

    This encoding is equivariant to swapping pairs of features, satisfying:

    .. math::

        (\mathrm{SWAP}_{01} \otimes \mathrm{SWAP}_{23} \otimes \ldots)
        |\psi(x)\rangle = |\psi(\mathrm{swap} \cdot x)\rangle

    where swap exchanges ``(x₀, x₁) → (x₁, x₀)``,
    ``(x₂, x₃) → (x₃, x₂)``, etc.

    The encoding uses **direct feature encoding** (``RY(x[i])`` on qubit *i*)
    combined with **symmetric entanglement** (CZ gates), ensuring that
    swapping features within a pair is exactly equivalent to applying a
    SWAP gate on the corresponding qubits.

    Parameters
    ----------
    n_features : int
        Number of features (must be even).
    reps : int, default=2
        Number of encoding layer repetitions.

    Attributes
    ----------
    n_pairs : int
        Number of feature pairs.
    reps : int
        Number of encoding layer repetitions.

    Examples
    --------
    Create swap-equivariant encoding for 4 features (2 pairs):

    >>> from encoding_atlas import SwapEquivariantFeatureMap
    >>> import numpy as np
    >>> enc = SwapEquivariantFeatureMap(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.n_pairs
    2

    Verify swap equivariance:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> swaps = [True, False]  # Swap first pair only
    >>> enc.verify_equivariance(x, swaps)
    True

    Swap both pairs:

    >>> swaps = [True, True]
    >>> enc.verify_equivariance(x, swaps)
    True

    Notes
    -----
    **Pair Structure**: Features are organized into pairs:
    ``(x₀, x₁), (x₂, x₃), (x₄, x₅), ...``

    Each pair is encoded on two qubits with symmetric structure.

    **Circuit Structure**: Each layer consists of:

    1. **RY gates** for direct feature encoding: ``RY(x[i])`` on qubit *i*.
       Swapping data ``(x[i] ↔ x[j])`` is equivalent to swapping the RY
       angles between qubits, which is exactly what the SWAP gate does.
    2. **Hadamard gates** for basis mixing. Since ``H ⊗ H`` commutes with
       SWAP (tensor product of identical operators), this layer preserves
       equivariance.
    3. **CZ gates** for symmetric pair entanglement. CZ is used instead of
       CNOT because CZ is symmetric under qubit exchange:
       ``SWAP · CZ(q0, q1) · SWAP = CZ(q1, q0) = CZ(q0, q1)``.
       This symmetry is essential for equivariance; CNOT would break it
       because ``CNOT(q0, q1) ≠ CNOT(q1, q0)``.

    **Equivariance Proof Sketch** (for one pair on qubits 0, 1):

    The circuit per layer is ``CZ · (H ⊗ H) · (RY(x₀) ⊗ RY(x₁))``.
    Applying SWAP to the state:

    .. math::

        \mathrm{SWAP} \cdot \mathrm{CZ} \cdot (H \otimes H)
        \cdot (R_Y(x_0) \otimes R_Y(x_1)) |00\rangle

    Since CZ commutes with SWAP, push SWAP through CZ. Since
    ``H ⊗ H`` commutes with SWAP, push through again. SWAP exchanges
    the single-qubit gates: ``RY(x₀) ⊗ RY(x₁) → RY(x₁) ⊗ RY(x₀)``.
    Since ``SWAP|00⟩ = |00⟩``, the result is ``|ψ(x₁, x₀)⟩``,
    which is exactly ``|ψ(swap · x)⟩``.

    References
    ----------
    .. [1] Schatzki, L., et al. (2022). "Theoretical Guarantees for
           Permutation-Equivariant Quantum Neural Networks." npj Quantum
           Information, 8, 74.

    See Also
    --------
    SO2EquivariantFeatureMap : Rotation equivariance
    CyclicEquivariantFeatureMap : Cyclic shift equivariance
    """

    __slots__ = ("n_pairs", "reps", "_entanglement_pairs")

    def __init__(
        self,
        n_features: int,
        reps: int = 2,
    ) -> None:
        """Initialize swap equivariant feature map.

        Parameters
        ----------
        n_features : int
            Number of features (must be even).
        reps : int, default=2
            Number of encoding layer repetitions.

        Raises
        ------
        ValueError
            If n_features is odd.
        ValueError
            If reps < 1.
        """
        # Validate n_features is even
        if n_features % 2 != 0:
            raise ValueError(f"n_features must be even, got {n_features}")

        # Validate reps
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
            raise ValueError(f"reps must be a positive integer, got {reps!r}")

        # Call parent constructor
        super().__init__(n_features, reps=reps)

        self.n_pairs = n_features // 2
        self.reps = reps

        # Cache entanglement pairs for performance
        # CZ gates are applied within each feature pair
        self._entanglement_pairs: list[tuple[int, int]] = [
            (2 * i, 2 * i + 1) for i in range(self.n_pairs)
        ]

        _logger.debug(
            f"Initialized SwapEquivariantFeatureMap with "
            f"n_features={n_features}, n_pairs={self.n_pairs}, reps={reps}"
        )

    @property
    def n_qubits(self) -> int:
        """Number of qubits (equal to n_features)."""
        return self.n_features

    @property
    def depth(self) -> int:
        """Circuit depth."""
        # Each rep: RY layer + H layer + CZ layer
        return 3 * self.reps

    def group_action(
        self,
        swaps: list[bool],
        x: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Swap specified feature pairs.

        Parameters
        ----------
        swaps : list of bool
            Boolean list indicating which pairs to swap.
            swaps[i] = True means swap pair i.
        x : ndarray
            Input features.

        Returns
        -------
        ndarray
            Features with specified pairs swapped.
        """
        result = x.copy()
        for i, should_swap in enumerate(swaps):
            if should_swap:
                result[2 * i], result[2 * i + 1] = result[2 * i + 1], result[2 * i]
        return result

    def unitary_representation(
        self,
        swaps: list[bool],
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Get SWAP gate unitary for specified pairs.

        Parameters
        ----------
        swaps : list of bool
            Boolean list indicating which pairs to swap.

        Returns
        -------
        ndarray
            Tensor product of SWAP/Identity operators.
        """
        # Build tensor product of SWAP and Identity gates
        SWAP = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        I2 = np.eye(4, dtype=complex)

        U = np.array([[1]], dtype=complex)
        for should_swap in swaps:
            gate = SWAP if should_swap else I2
            U = np.kron(U, gate)

        return U

    def group_generators(self) -> list[list[bool]]:
        """Return generators for the swap group.

        Returns
        -------
        list of list of bool
            One generator per pair (swap that pair only).
        """
        generators = []
        for i in range(self.n_pairs):
            swaps = [False] * self.n_pairs
            swaps[i] = True
            generators.append(swaps)
        return generators

    def _apply_group_unitary_as_gates(self, swaps: list[bool]) -> None:
        """Apply pair swaps as SWAP gates.

        For swap equivariant encoding, the unitary representation swaps
        pairs of qubits according to the swap pattern:
        - swaps[i] = True: swap qubits (2i, 2i+1)
        - swaps[i] = False: no swap for pair i

        Parameters
        ----------
        swaps : list of bool
            Boolean list indicating which pairs to swap.
            swaps[i] = True means swap qubits (2*i, 2*i+1).

        Notes
        -----
        This method is called within a PennyLane QNode context during
        statistical verification. It implements the swap operations
        directly using SWAP gates.
        """
        import pennylane as qml

        for i, should_swap in enumerate(swaps):
            if should_swap:
                qubit_a = 2 * i
                qubit_b = 2 * i + 1
                qml.SWAP(wires=[qubit_a, qubit_b])

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get CZ gate qubit pairs within each feature pair.

        Returns
        -------
        list[tuple[int, int]]
            List of qubit pairs for CZ gates.
            Each pair ``(2i, 2i+1)`` represents symmetric entanglement
            within feature pair *i*. CZ is symmetric, so the order
            within each tuple does not affect the gate operation.

        Examples
        --------
        >>> enc = SwapEquivariantFeatureMap(n_features=4)
        >>> enc.get_entanglement_pairs()
        [(0, 1), (2, 3)]
        """
        return list(self._entanglement_pairs)

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for input data.

        Strict variant: a 2D input with more than one row raises
        :class:`ValueError`. Use :meth:`get_circuits` for batches.

        Parameters
        ----------
        x : array_like
            Input features.
        backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Quantum circuit in the specified backend format.
        """
        x_array = self._validate_input(x)
        if x_array.ndim == 2:
            if x_array.shape[0] != 1:
                raise ValueError("get_circuit requires a single sample")
            x_array = x_array[0]
        return self._get_circuit_from_validated(x_array, backend)

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Dispatch swap-equivariant circuit construction.

        Encoding-specific seam called by ``get_circuit`` and the inherited
        ``get_circuits`` template method. Accepts a 1D array (the normal
        case) or a 2D single-sample ``(1, n_features)`` array for
        robustness when called directly.
        """
        if x.ndim == 2:
            x = x[0]

        if backend == "pennylane":
            return self._to_pennylane(x)
        elif backend == "qiskit":
            return self._to_qiskit(x)
        elif backend == "cirq":
            return self._to_cirq(x)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _to_pennylane(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate PennyLane circuit.

        Parameters
        ----------
        x : ndarray
            Input features.

        Returns
        -------
        callable
            PennyLane quantum function.
        """
        import pennylane as qml

        n = self.n_qubits

        def circuit() -> None:
            """Swap-equivariant encoding circuit.

            Each feature x[i] is directly encoded on qubit i via RY(x[i]).
            Hadamard gates provide basis mixing, and CZ gates provide
            symmetric entanglement within pairs.

            Equivariance holds because:
            - Direct encoding: swapping data ↔ swapping qubits
            - H ⊗ H commutes with SWAP (identical single-qubit gates)
            - CZ commutes with SWAP (symmetric two-qubit gate)
            - SWAP|00⟩ = |00⟩ (initial state is invariant)
            """
            for _rep in range(self.reps):
                # Layer 1: Direct feature encoding (RY(x[i]) on qubit i)
                for i in range(n):
                    qml.RY(x[i], wires=i)

                # Layer 2: Hadamard for basis mixing
                for i in range(n):
                    qml.Hadamard(wires=i)

                # Layer 3: Symmetric pair entanglement (CZ within each pair)
                for pair_idx in range(self.n_pairs):
                    i = 2 * pair_idx
                    j = 2 * pair_idx + 1
                    qml.CZ(wires=[i, j])

        return circuit

    def _to_qiskit(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate Qiskit circuit.

        Parameters
        ----------
        x : ndarray
            Input features.

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit.
        """
        from qiskit import QuantumCircuit

        n = self.n_qubits
        circuit = QuantumCircuit(n)

        for _rep in range(self.reps):
            # Layer 1: Direct feature encoding (RY(x[i]) on qubit i)
            for i in range(n):
                circuit.ry(float(x[i]), i)

            # Layer 2: Hadamard gates
            for i in range(n):
                circuit.h(i)

            # Layer 3: Symmetric pair entanglement (CZ within each pair)
            for pair_idx in range(self.n_pairs):
                i = 2 * pair_idx
                j = 2 * pair_idx + 1
                circuit.cz(i, j)

        return circuit

    def _to_cirq(
        self,
        x: NDArray[np.floating[Any]],
    ) -> CircuitType:
        """Generate Cirq circuit.

        Parameters
        ----------
        x : ndarray
            Input features.

        Returns
        -------
        Circuit
            Cirq quantum circuit.
        """
        import cirq

        n = self.n_qubits
        qubits = cirq.LineQubit.range(n)
        circuit = cirq.Circuit()

        for _rep in range(self.reps):
            # Layer 1: Direct feature encoding (RY(x[i]) on qubit i)
            for i in range(n):
                circuit.append(cirq.ry(float(x[i]))(qubits[i]))

            # Layer 2: Hadamard gates
            circuit.append([cirq.H(qubits[i]) for i in range(n)])

            # Layer 3: Symmetric pair entanglement (CZ within each pair)
            for pair_idx in range(self.n_pairs):
                i = 2 * pair_idx
                j = 2 * pair_idx + 1
                circuit.append(cirq.CZ(qubits[i], qubits[j]))

        return circuit

    def gate_count_breakdown(self) -> SwapGateCountBreakdown:
        """Get breakdown of gate counts by type.

        Returns
        -------
        SwapGateCountBreakdown
            Dictionary with detailed gate counts including RY, Hadamard,
            and CZ gates per layer and in total.

        Examples
        --------
        >>> enc = SwapEquivariantFeatureMap(n_features=4, reps=2)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['ry']
        8
        >>> breakdown['cz']
        4
        """
        n = self.n_qubits

        ry_count = n * self.reps
        hadamard_count = n * self.reps
        cz_count = self.n_pairs * self.reps

        return {
            "ry": ry_count,
            "hadamard": hadamard_count,
            "cz": cz_count,
            "total_single_qubit": ry_count + hadamard_count,
            "total_two_qubit": cz_count,
            "total": ry_count + hadamard_count + cz_count,
        }

    def _compute_properties(self) -> EncodingProperties:
        """Compute encoding properties."""
        gate_breakdown = self.gate_count_breakdown()

        return EncodingProperties(
            n_qubits=self.n_qubits,
            depth=self.depth,
            gate_count=gate_breakdown["total"],
            single_qubit_gates=gate_breakdown["total_single_qubit"],
            two_qubit_gates=gate_breakdown["total_two_qubit"],
            parameter_count=0,
            is_entangling=True,
            simulability="not_simulable",
        )

    def resource_summary(self) -> dict[str, Any]:
        """Generate comprehensive resource summary.

        Provides a complete view of circuit resources, encoding characteristics,
        and hardware requirements. This method combines information from
        multiple sources for hardware planning and comparison.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            **Circuit Structure**:
                - ``n_qubits``: Number of qubits
                - ``n_features``: Number of classical features
                - ``depth``: Circuit depth
                - ``reps``: Number of encoding repetitions
                - ``n_pairs``: Number of feature pairs

            **Gate Counts**:
                - ``gate_counts``: Full breakdown from ``gate_count_breakdown()``

            **Encoding Characteristics**:
                - ``is_entangling``: True
                - ``simulability``: "not_simulable"
                - ``trainability_estimate``: Trainability score (0.0-1.0)

            **Symmetry Information**:
                - ``symmetry_group``: "S_2^n" (direct product of swap groups)
                - ``n_pair_swaps``: Number of independent pair swaps

            **Equivariance Verification Cost**:
                - ``verification_cost``: Memory and time complexity for verification

            **Hardware Requirements**:
                - ``connectivity``: Required qubit connectivity (pairs)
                - ``native_gates``: List of required gate types

            **Entanglement Details**:
                - ``n_entanglement_pairs``: Number of CZ pairs per layer
                - ``entanglement_pairs``: List of qubit pairs for CZ gates

        Examples
        --------
        >>> enc = SwapEquivariantFeatureMap(n_features=4, reps=2)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['n_pairs']
        2
        >>> summary['symmetry_group']
        'S_2^2'

        See Also
        --------
        gate_count_breakdown : Detailed gate count breakdown.
        properties : Basic encoding properties.
        verify_equivariance : Verify equivariance property.
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties
        pairs = self.get_entanglement_pairs()

        return {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            "n_pairs": self.n_pairs,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding characteristics
            "is_entangling": props.is_entangling,
            "simulability": props.simulability,
            "trainability_estimate": props.trainability_estimate,
            # Symmetry information
            "symmetry_group": f"S_2^{self.n_pairs}",
            "n_pair_swaps": self.n_pairs,
            # Equivariance verification cost
            "verification_cost": {
                "exact": {
                    "memory": f"O(2^{self.n_qubits})",
                    "time": f"O(2^{self.n_qubits})",
                    "recommended_max_qubits": _EXACT_VERIFICATION_QUBIT_THRESHOLD,
                },
                "statistical": {
                    "memory": "O(n_shots)",
                    "time": "O(n_shots × circuit_depth)",
                    "default_shots": _DEFAULT_STATISTICAL_VERIFICATION_SHOTS,
                    "scalable": True,
                },
            },
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": "pairs",
                "native_gates": ["RY", "H", "CZ"],
            },
            # Entanglement details
            "n_entanglement_pairs": len(pairs),
            "entanglement_pairs": pairs,
            # Verification methods available
            "verification_methods": [
                "verify_equivariance (exact)",
                "verify_equivariance_statistical (scalable)",
                "verify_equivariance_auto (automatic selection)",
            ],
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
            f"n_pairs={self.n_pairs}, "
            f"reps={self.reps})"
        )
