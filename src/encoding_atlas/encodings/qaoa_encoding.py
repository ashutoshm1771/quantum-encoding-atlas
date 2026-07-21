"""QAOA-style Quantum Data Encoding module.

This module implements QAOA-style encoding, which adapts the structure of the
Quantum Approximate Optimization Algorithm (QAOA) for classical data encoding.
The encoding creates quantum states through alternating layers of data-dependent
rotations (analogous to the cost Hamiltonian) and mixer operations (analogous
to the mixer Hamiltonian).

The encoding structure follows the QAOA ansatz:
1. Optional initial Hadamard layer for superposition
2. Repeated blocks of:
   a. Data encoding layer: Single-qubit rotations encoding classical features
   b. Mixer layer: Mixing rotations + entangling gates

Mathematical Definition:
    |ψ(x)⟩ = ∏_{p=1}^{reps} U_M(β_p) U_C(γ, x) |+⟩^⊗n

    where:
    - U_C(γ, x) = ∏_i R_data(γ · x_i) applies data-dependent rotations
    - U_M(β) = U_entangle · ∏_i R_mixer(β) applies mixing operations
    - |+⟩^⊗n is the uniform superposition state

This encoding is particularly suitable for:
- Problems with graph or combinatorial structure
- Variational quantum classifiers
- Quantum kernel methods requiring QAOA-like feature maps

References
----------
.. [1] Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate
       Optimization Algorithm." arXiv:1411.4028.
.. [2] Hadfield, S., et al. (2019). "From the Quantum Approximate Optimization
       Algorithm to a Quantum Alternating Operator Ansatz." Algorithms, 12(2), 34.
.. [3] Lloyd, S., et al. (2020). "Quantum embeddings for machine learning."
       arXiv:2001.03622.
.. [4] Schuld, M., & Petruccione, F. (2021). "Machine Learning with Quantum
       Computers." Springer.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict, cast

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
#   logging.getLogger('encoding_atlas.encodings.qaoa_encoding').setLevel(logging.DEBUG)
#   logging.basicConfig(level=logging.DEBUG)
#
# Or configure via your application's logging setup.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["QAOAEncoding", "GateCountBreakdown"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Threshold for warning about full entanglement with many features.
#
# Full entanglement creates n(n-1)/2 entangling gate pairs, each applied
# per repetition. The gate count scales quadratically:
#
#   Entangling gates per layer = n(n-1)/2
#
# Examples (per layer):
#   - 4 features:  6 entangling gates (manageable)
#   - 8 features:  28 entangling gates (moderate)
#   - 12 features: 66 entangling gates (threshold - warning issued)
#   - 16 features: 120 entangling gates (may exceed NISQ capabilities)
#   - 20 features: 190 entangling gates (likely impractical on current hardware)
#
# At 12 features, the circuit complexity becomes significant enough to warrant
# a warning. Users should consider 'linear' or 'circular' entanglement for
# larger feature counts, or ensure their target hardware supports the required
# gate count and connectivity.
_FULL_ENTANGLEMENT_WARNING_THRESHOLD: int = 12

# Default number of repetitions for QAOA-style encoding.
#
# The choice of 2 repetitions provides a good balance between:
# - Expressibility (sufficient layers for feature interaction)
# - Circuit depth (manageable on NISQ devices)
# - Trainability (avoids deep barren plateaus)
#
# This value is inspired by typical QAOA implementations where p=2 provides
# a reasonable approximation ratio for many optimization problems.
_DEFAULT_REPS: int = 2


# =============================================================================
# Type Definitions
# =============================================================================


class GateCountBreakdown(TypedDict):
    """Type definition for gate count breakdown dictionary.

    This TypedDict provides type-safe access to gate count information
    returned by the ``gate_count_breakdown()`` method, enabling IDE
    autocompletion and static type checking.

    Attributes
    ----------
    hadamard : int
        Number of Hadamard gates (initial layer, if included).
    data_rotation : int
        Number of data encoding rotation gates (R_data).
    mixer_rotation : int
        Number of mixer rotation gates (R_mixer).
    entangling : int
        Number of two-qubit entangling gates (CX, CZ, or RZZ).
    total_single_qubit : int
        Total single-qubit gates (hadamard + data_rotation + mixer_rotation).
    total_two_qubit : int
        Total two-qubit gates (= entangling).
    total : int
        Total gate count.
    """

    hadamard: int
    data_rotation: int
    mixer_rotation: int
    entangling: int
    total_single_qubit: int
    total_two_qubit: int
    total: int


class QAOAEncoding(BaseEncoding):
    """QAOA-style quantum data encoding with alternating cost and mixer layers.

    This encoding adapts the Quantum Approximate Optimization Algorithm (QAOA)
    structure for encoding classical data into quantum states. It alternates
    between data-encoding layers (analogous to the cost Hamiltonian evolution)
    and mixer layers (analogous to the mixer Hamiltonian evolution).

    The circuit structure for each repetition consists of:
    1. Data encoding layer: R_data(γ · x_i) rotations on each qubit
    2. Mixer layer: R_mixer(β) rotations followed by entangling gates

    This creates a rich feature space suitable for quantum machine learning
    tasks, particularly those with graph or combinatorial structure.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be at least 1.
    reps : int, default=2
        Number of QAOA-style repetitions (often denoted as 'p' in QAOA
        literature). Higher values increase expressibility but also circuit
        depth. Must be at least 1.
    data_rotation : {'X', 'Y', 'Z'}, default='Z'
        Rotation axis for data encoding (cost layer).
        - 'Z': RZ gates (standard in QAOA for diagonal Hamiltonians)
        - 'Y': RY gates (useful for amplitude-sensitive encodings)
        - 'X': RX gates (alternative encoding basis)
    mixer_rotation : {'X', 'Y', 'Z'}, default='X'
        Rotation axis for mixer layer.
        - 'X': RX gates (standard QAOA mixer)
        - 'Y': RY gates (alternative mixer)
        - 'Z': RZ gates (phase-only mixer)
    entanglement : {'linear', 'full', 'circular', 'none'}, default='linear'
        Entanglement pattern for mixer layer:
        - 'linear': Consecutive pairs (i, i+1) - hardware-friendly
        - 'full': All pairs (i, j) where i < j - maximum entanglement
        - 'circular': Linear plus (n-1, 0) - periodic boundary
        - 'none': No entanglement (product state encoding)
    entangling_gate : {'cx', 'cz', 'rzz'}, default='cz'
        Type of entangling gate in mixer layer:
        - 'cx': CNOT gates
        - 'cz': Controlled-Z gates (symmetric, hardware-native)
        - 'rzz': RZZ gates with angle β (parameterized entanglement)
    gamma : float, default=1.0
        Scaling factor for data encoding rotations.
        The rotation angle is γ · x_i for feature x_i.
    beta : float, default=1.0
        Scaling factor for mixer rotations.
        The rotation angle is β for each mixer gate.
    include_initial_h : bool, default=True
        Whether to apply Hadamard gates at the start to create
        uniform superposition. Set to False if starting from |0⟩.
    feature_map : {'linear', 'quadratic'}, default='linear'
        How to map features to rotation angles:
        - 'linear': φ(x_i) = γ · x_i (direct mapping, scaled by gamma)
        - 'quadratic': φ(x_i) = γ · x_i² (nonlinear mapping, scaled by gamma)

    Attributes
    ----------
    reps : int
        Number of QAOA-style repetitions.
    data_rotation : str
        Rotation axis for data encoding.
    mixer_rotation : str
        Rotation axis for mixer.
    entanglement : str
        Entanglement pattern.
    entangling_gate : str
        Type of entangling gate.
    gamma : float
        Data encoding scaling factor.
    beta : float
        Mixer scaling factor.
    include_initial_h : bool
        Whether initial Hadamards are applied.
    feature_map : str
        Feature mapping type.

    Examples
    --------
    Basic QAOA-style encoding with default settings:

    >>> from encoding_atlas.encodings.qaoa_encoding import QAOAEncoding
    >>> import numpy as np
    >>> enc = QAOAEncoding(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.reps
    2

    Check depth calculation:

    >>> enc = QAOAEncoding(n_features=4, reps=2, entanglement='linear')
    >>> enc.depth
    9

    Compute data angles:

    >>> enc = QAOAEncoding(n_features=4, gamma=2.0)
    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> angles = enc.compute_data_angles(x)
    >>> np.allclose(angles, [0.2, 0.4, 0.6, 0.8])
    True

    Product state encoding (no entanglement):

    >>> enc = QAOAEncoding(n_features=4, entanglement='none')
    >>> enc.properties.is_entangling
    False

    High-depth encoding with full entanglement:

    >>> enc = QAOAEncoding(
    ...     n_features=4,
    ...     reps=5,
    ...     entanglement='full',
    ...     entangling_gate='cz'
    ... )
    >>> enc.depth
    26

    See Also
    --------
    PauliFeatureMap : Encoding with configurable Pauli rotations.
    ZZFeatureMap : Encoding using ZZ entanglement structure.
    DataReuploading : Alternative data re-uploading approach.
    HardwareEfficientEncoding : Hardware-optimized variational encoding.

    Notes
    -----
    **Connection to QAOA:**

    The standard QAOA circuit has the form:

    .. math::

        |ψ(γ, β)⟩ = ∏_{p=1}^{P} e^{-iβ_p H_M} e^{-iγ_p H_C} |+⟩^{⊗n}

    where H_C is the cost Hamiltonian and H_M is the mixer Hamiltonian.
    In this encoding:

    - H_C is replaced by data-dependent rotations R(γ · x_i)
    - H_M is approximated by single-qubit mixers + entangling gates

    **Expressibility:**

    The expressibility increases with:
    - Number of repetitions (reps)
    - Entanglement connectivity (full > circular > linear > none)
    - Use of parameterized entangling gates (rzz)

    **Hardware Considerations:**

    For near-term quantum devices:
    - Use 'linear' entanglement to match hardware topology
    - Use 'cz' gates which are native on many platforms
    - Keep reps low (2-4) to minimize noise accumulation

    **Backend Support:**

    This encoding supports multiple quantum computing frameworks:

    - **PennyLane**: Returns a callable function that applies gates when invoked
      within a QNode context. Ideal for variational algorithms and automatic
      differentiation workflows.
    - **Qiskit**: Returns a ``QuantumCircuit`` object compatible with IBM Quantum
      hardware and Qiskit's transpiler. Barriers are inserted between repetitions
      for visual clarity.
    - **Cirq**: Returns a ``cirq.Circuit`` with optimized Moment structure for
      minimal depth. Supports Google's quantum processors.

    Use the ``backend`` parameter in ``get_circuit()`` and ``get_circuits()``
    to select the target framework::

        circuit_pl = encoding.get_circuit(x, backend='pennylane')
        circuit_qk = encoding.get_circuit(x, backend='qiskit')
        circuit_cq = encoding.get_circuit(x, backend='cirq')
    """

    # Memory optimization: Use __slots__ to avoid __dict__ overhead.
    # This is consistent with BaseEncoding and reduces memory by ~30%.
    # Important for hyperparameter searches creating many instances.
    __slots__ = (
        "reps",
        "data_rotation",
        "mixer_rotation",
        "entanglement",
        "entangling_gate",
        "gamma",
        "beta",
        "include_initial_h",
        "feature_map",
        "_entanglement_pairs",
    )

    # Valid rotation axes
    _VALID_ROTATIONS: frozenset[str] = frozenset({"X", "Y", "Z"})

    # Valid entanglement patterns
    _VALID_ENTANGLEMENT: frozenset[str] = frozenset(
        {"linear", "full", "circular", "none"}
    )

    # Valid entangling gates
    _VALID_ENTANGLING_GATES: frozenset[str] = frozenset({"cx", "cz", "rzz"})

    # Valid feature maps
    _VALID_FEATURE_MAPS: frozenset[str] = frozenset({"linear", "quadratic"})

    def __init__(
        self,
        n_features: int,
        reps: int = 2,
        data_rotation: Literal["X", "Y", "Z"] = "Z",
        mixer_rotation: Literal["X", "Y", "Z"] = "X",
        entanglement: Literal["linear", "full", "circular", "none"] = "linear",
        entangling_gate: Literal["cx", "cz", "rzz"] = "cz",
        gamma: float = 1.0,
        beta: float = 1.0,
        include_initial_h: bool = True,
        feature_map: Literal["linear", "quadratic"] = "linear",
    ) -> None:
        """Initialize the QAOA-style encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        reps : int, default=2
            Number of QAOA-style repetitions.
        data_rotation : {'X', 'Y', 'Z'}, default='Z'
            Rotation axis for data encoding.
        mixer_rotation : {'X', 'Y', 'Z'}, default='X'
            Rotation axis for mixer.
        entanglement : {'linear', 'full', 'circular', 'none'}, default='linear'
            Entanglement pattern for mixer layer.
        entangling_gate : {'cx', 'cz', 'rzz'}, default='cz'
            Type of entangling gate.
        gamma : float, default=1.0
            Scaling factor for data encoding.
        beta : float, default=1.0
            Scaling factor for mixer rotations.
        include_initial_h : bool, default=True
            Whether to apply initial Hadamard gates.
        feature_map : {'linear', 'quadratic'}, default='linear'
            Feature mapping type.

        Raises
        ------
        ValueError
            If any parameter has an invalid value.
        TypeError
            If any parameter has an incorrect type.
        """
        # =================================================================
        # PARAMETER VALIDATION
        # =================================================================

        # Validate n_features
        if not isinstance(n_features, (int, np.integer)):
            raise TypeError(
                f"n_features must be an integer, got {type(n_features).__name__}"
            )
        n_features = int(n_features)
        if n_features < 1:
            raise ValueError(f"n_features must be at least 1, got {n_features}")

        # Validate reps
        if not isinstance(reps, (int, np.integer)):
            raise TypeError(f"reps must be an integer, got {type(reps).__name__}")
        reps = int(reps)
        if reps < 1:
            raise ValueError(f"reps must be at least 1, got {reps}")

        # Warn about potential barren plateaus with deep circuits
        if reps > 10:
            import warnings

            warnings.warn(
                f"reps={reps} creates a deep circuit that may suffer from barren "
                f"plateaus, making training difficult. Consider reps <= 10 for "
                f"practical trainability. See: McClean et al. (2018) 'Barren "
                f"plateaus in quantum neural network training landscapes'.",
                UserWarning,
                stacklevel=2,
            )

        # Validate data_rotation
        if not isinstance(data_rotation, str):
            raise TypeError(
                f"data_rotation must be a string, got {type(data_rotation).__name__}"
            )
        data_rotation_upper = data_rotation.upper()
        if data_rotation_upper not in self._VALID_ROTATIONS:
            raise ValueError(
                f"data_rotation must be one of {sorted(self._VALID_ROTATIONS)}, "
                f"got '{data_rotation}'"
            )

        # Validate mixer_rotation
        if not isinstance(mixer_rotation, str):
            raise TypeError(
                f"mixer_rotation must be a string, got {type(mixer_rotation).__name__}"
            )
        mixer_rotation_upper = mixer_rotation.upper()
        if mixer_rotation_upper not in self._VALID_ROTATIONS:
            raise ValueError(
                f"mixer_rotation must be one of {sorted(self._VALID_ROTATIONS)}, "
                f"got '{mixer_rotation}'"
            )

        # Validate entanglement
        if not isinstance(entanglement, str):
            raise TypeError(
                f"entanglement must be a string, got {type(entanglement).__name__}"
            )
        entanglement_lower = entanglement.lower()
        if entanglement_lower not in self._VALID_ENTANGLEMENT:
            raise ValueError(
                f"entanglement must be one of {sorted(self._VALID_ENTANGLEMENT)}, "
                f"got '{entanglement}'"
            )

        # Validate entangling_gate
        if not isinstance(entangling_gate, str):
            raise TypeError(
                f"entangling_gate must be a string, "
                f"got {type(entangling_gate).__name__}"
            )
        entangling_gate_lower = entangling_gate.lower()
        if entangling_gate_lower not in self._VALID_ENTANGLING_GATES:
            raise ValueError(
                f"entangling_gate must be one of {sorted(self._VALID_ENTANGLING_GATES)}, "
                f"got '{entangling_gate}'"
            )

        # Validate gamma
        if not isinstance(gamma, (int, float, np.integer, np.floating)):
            raise TypeError(f"gamma must be a number, got {type(gamma).__name__}")
        gamma = float(gamma)
        if not np.isfinite(gamma):
            raise ValueError(f"gamma must be finite, got {gamma}")

        # Validate beta
        if not isinstance(beta, (int, float, np.integer, np.floating)):
            raise TypeError(f"beta must be a number, got {type(beta).__name__}")
        beta = float(beta)
        if not np.isfinite(beta):
            raise ValueError(f"beta must be finite, got {beta}")

        # Validate include_initial_h
        if not isinstance(include_initial_h, bool):
            raise TypeError(
                f"include_initial_h must be a bool, "
                f"got {type(include_initial_h).__name__}"
            )

        # Validate feature_map
        if not isinstance(feature_map, str):
            raise TypeError(
                f"feature_map must be a string, got {type(feature_map).__name__}"
            )
        feature_map_lower = feature_map.lower()
        if feature_map_lower not in self._VALID_FEATURE_MAPS:
            raise ValueError(
                f"feature_map must be one of {sorted(self._VALID_FEATURE_MAPS)}, "
                f"got '{feature_map}'"
            )

        # =================================================================
        # INITIALIZATION
        # =================================================================

        # Call parent constructor
        super().__init__(
            n_features,
            reps=reps,
            data_rotation=data_rotation_upper,
            mixer_rotation=mixer_rotation_upper,
            entanglement=entanglement_lower,
            entangling_gate=entangling_gate_lower,
            gamma=gamma,
            beta=beta,
            include_initial_h=include_initial_h,
            feature_map=feature_map_lower,
        )

        # Store validated parameters
        self.reps: int = reps
        self.data_rotation: str = data_rotation_upper
        self.mixer_rotation: str = mixer_rotation_upper
        self.entanglement: str = entanglement_lower
        self.entangling_gate: str = entangling_gate_lower
        self.gamma: float = gamma
        self.beta: float = beta
        self.include_initial_h: bool = include_initial_h
        self.feature_map: str = feature_map_lower

        # Precompute entanglement pairs
        self._entanglement_pairs: list[tuple[int, int]] = (
            self._compute_entanglement_pairs()
        )

        # Warn about full entanglement with many features
        # Full entanglement creates O(n²) two-qubit gates which can be impractical
        # for NISQ devices when n exceeds the warning threshold.
        if (
            entanglement_lower == "full"
            and n_features >= _FULL_ENTANGLEMENT_WARNING_THRESHOLD
        ):
            import warnings

            n_pairs = len(self._entanglement_pairs)
            total_entangling = n_pairs * reps
            warnings.warn(
                f"Full entanglement with n_features={n_features} creates "
                f"{n_pairs} entangling gate pairs per layer ({total_entangling} total "
                f"with reps={reps}). This may exceed NISQ device capabilities. "
                f"Consider 'linear' or 'circular' entanglement for better "
                f"hardware compatibility and reduced circuit depth.",
                UserWarning,
                stacklevel=2,
            )

        # Log initialization details at DEBUG level
        _logger.debug(
            "QAOAEncoding initialized: n_features=%d, n_qubits=%d, reps=%d, "
            "data_rotation=%s, mixer_rotation=%s, entanglement=%s, "
            "entangling_gate=%s, gamma=%.4f, beta=%.4f, include_initial_h=%s, "
            "feature_map=%s, n_entanglement_pairs=%d",
            n_features,
            n_features,  # n_qubits = n_features
            reps,
            data_rotation_upper,
            mixer_rotation_upper,
            entanglement_lower,
            entangling_gate_lower,
            gamma,
            beta,
            include_initial_h,
            feature_map_lower,
            len(self._entanglement_pairs),
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
        """Theoretical circuit depth estimate (fast, no circuit generation).

        This property returns a theoretical depth estimate based on optimal
        parallel scheduling, without generating the actual circuit. It's useful
        for quick comparisons and circuit design decisions.

        The depth is computed based on theoretical circuit structure:
        - 1 for initial Hadamard layer (if included)
        - Per repetition:
            - 1 for data encoding layer (parallel single-qubit gates)
            - 1 for mixer rotation layer (parallel single-qubit gates)
            - depth of entangling layer (optimal parallel scheduling)

        Backend-Specific Considerations
        -------------------------------
        Actual circuit depth may differ from this estimate due to:
        - **Qiskit**: Transpiler optimizations, gate decomposition
        - **Cirq**: Moment structure (our impl. is optimized for this)
        - **PennyLane**: Device-specific gate decomposition

        For accurate backend-specific depth, use ``get_depth(backend)``.

        Returns
        -------
        int
            Theoretical circuit depth estimate.

        See Also
        --------
        get_depth : Get actual depth for a specific backend.

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=8, reps=2, entanglement='linear')
        >>> enc.depth  # Fast theoretical estimate
        9
        >>> enc.get_depth('cirq')  # Actual Cirq depth
        9
        """
        depth = 0
        n = self._n_features

        # Initial Hadamard layer
        if self.include_initial_h:
            depth += 1

        # Per repetition
        for _ in range(self.reps):
            # Data encoding layer (all parallel)
            depth += 1

            # Mixer rotation layer (all parallel)
            depth += 1

            # Entangling layer depth - computed based on sequential application
            if self.entanglement == "none" or n < 2:
                # No entangling gates
                pass
            elif self.entanglement == "linear":
                # Linear entanglement: CZ(0,1), CZ(1,2), ..., CZ(n-2, n-1)
                # Total of (n-1) two-qubit gates.
                #
                # With OPTIMAL PARALLEL SCHEDULING, gates can be grouped as:
                #   Layer 1 (even pairs): CZ(0,1), CZ(2,3), CZ(4,5), ...
                #   Layer 2 (odd pairs):  CZ(1,2), CZ(3,4), CZ(5,6), ...
                #
                # Gates within each layer operate on disjoint qubits and can
                # execute in parallel. This gives constant depth:
                #   - n = 2: depth = 1 (single pair, min(2, 1) = 1)
                #   - n >= 3: depth = 2 (two layers, min(2, n-1) = 2)
                #
                # NOTE: The previous implementation assumed sequential application
                # (depth = n-1), which overestimated by ~(n-1)/2 for large n.
                # For n=100, the old estimate was 99; optimal is 2 (50x reduction).
                #
                # Safety: n >= 2 is guaranteed by the outer conditional check.
                # The min() ensures we never exceed 2 or go below 1.
                depth += min(2, n - 1)
            elif self.entanglement == "circular":
                # Circular entanglement forms a cycle graph C_n.
                # By edge coloring theory, the chromatic index is:
                #   χ'(C_n) = 2  if n is even (alternating edge colors)
                #   χ'(C_n) = 3  if n is odd (one edge needs a third color)
                #
                # This gives optimal parallel scheduling depth.
                # Example for n=4 (even): {(0,1),(2,3)}, {(1,2),(3,0)} → depth 2
                # Example for n=5 (odd):  {(0,1),(2,3)}, {(1,2),(3,4)}, {(4,0)} → depth 3
                if n % 2 == 0:
                    depth += 2
                else:
                    depth += 3
            elif self.entanglement == "full":
                # Full: n(n-1)/2 pairs applied in order (i,j) for i<j
                # Actual depth depends on DAG scheduling
                # We compute it by simulating qubit availability
                depth += self._compute_full_entanglement_depth()

        return depth

    @property
    def n_data_parameters(self) -> int:
        """Number of data-dependent rotation angles per input sample.

        This is the number of angles that are computed from the input
        data during encoding. Each qubit receives one data rotation
        per repetition.

        Note: This is different from trainable parameters (which is 0
        for this encoding). The gamma and beta values are fixed
        hyperparameters, not learned during training.

        Returns
        -------
        int
            Number of data-dependent angles = n_qubits × reps.

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, reps=3)
        >>> enc.n_data_parameters
        12
        """
        return self.n_qubits * self.reps

    def _compute_full_entanglement_depth(self) -> int:
        """Compute the optimal depth of full entanglement layer.

        Full entanglement applies two-qubit gates between all pairs (i, j)
        where i < j, giving n(n-1)/2 gates total. The optimal scheduling
        is determined by **edge coloring** of the complete graph K_n.

        Graph Theory Background
        -----------------------
        - Each qubit is a vertex in K_n (complete graph on n vertices)
        - Each two-qubit gate is an edge in K_n
        - Two gates can execute in parallel iff they don't share a qubit
          (i.e., their edges don't share a vertex)
        - Finding minimum layers = finding the chromatic index χ'(K_n)

        By Vizing's theorem and the known result for complete graphs:
            χ'(K_n) = n - 1  if n is even
            χ'(K_n) = n      if n is odd

        This is optimal and achievable via round-robin tournament scheduling.

        Example for n=4 (even):
            Layer 1: (0,1), (2,3)  - disjoint pairs
            Layer 2: (0,2), (1,3)  - disjoint pairs
            Layer 3: (0,3), (1,2)  - disjoint pairs
            Total: 3 layers = n-1 ✓

        Example for n=5 (odd):
            Layer 1: (0,1), (2,3)
            Layer 2: (0,2), (1,4)
            Layer 3: (0,3), (2,4)
            Layer 4: (0,4), (1,3)
            Layer 5: (1,2), (3,4)
            Total: 5 layers = n ✓

        NOTE: The previous implementation used greedy sequential scheduling
        which gave depth ~2n-3. For n=12, old=21 vs optimal=11 (1.9x reduction).

        Returns
        -------
        int
            Optimal depth of the full entanglement layer.

        References
        ----------
        .. [1] Vizing, V. G. (1964). "On an estimate of the chromatic class
               of a p-graph." Diskret. Analiz, 3, 25-30.
        .. [2] König, D. (1916). "Über Graphen und ihre Anwendung auf
               Determinantentheorie und Mengenlehre." Math. Ann., 77, 453-465.
        """
        n = self._n_features

        # Safety check: no entanglement possible with fewer than 2 qubits
        if n < 2:
            return 0

        # Optimal depth from edge coloring of complete graph K_n
        # Chromatic index: χ'(K_n) = n-1 (even n) or n (odd n)
        #
        # This is a fundamental result in graph theory and is optimal.
        # The schedule can be constructed using round-robin tournament
        # algorithms, but for depth calculation we only need the formula.
        if n % 2 == 0:
            return n - 1
        else:
            return n

    def _compute_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Compute qubit pairs for entangling operations.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs.
        """
        n = self._n_features

        if n < 2 or self.entanglement == "none":
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
            return []

    def _apply_feature_map(
        self,
        x: NDArray[np.floating[Any]],
        idx: int,
    ) -> float:
        """Apply feature mapping to compute rotation angle.

        Parameters
        ----------
        x : NDArray
            Input feature vector.
        idx : int
            Feature index.

        Returns
        -------
        float
            Mapped rotation angle.
        """
        value = float(x[idx])

        if self.feature_map == "linear":
            return self.gamma * value
        elif self.feature_map == "quadratic":
            return self.gamma * value * value
        else:
            return self.gamma * value

    def _check_input_range(
        self,
        x: NDArray[np.floating[Any]],
        warn_threshold: float = 2 * np.pi,
    ) -> None:
        """Check if input values are within recommended range and warn if not.

        Rotation gates are periodic with period 2π. Input values significantly
        outside the range [-2π, 2π] will wrap around, which may lead to:
        - Loss of precision due to floating-point limitations
        - Unexpected behavior if data was not properly normalized
        - Reduced discriminative power of the encoding

        This method issues a warning (not an error) to alert users while
        allowing the computation to proceed. Users can suppress these warnings
        if they intentionally use large values.

        Parameters
        ----------
        x : NDArray
            Input feature vector.
        warn_threshold : float, default=2π
            Threshold for warning. Values with |x[i] * gamma| > threshold
            will trigger a warning.

        Notes
        -----
        For quantum machine learning, it's recommended to normalize input
        features to a range like [0, 2π] or [-π, π] before encoding.
        Common normalization strategies:
        - Min-max scaling to [0, 2π]
        - Standardization followed by arctan scaling
        - Feature-specific domain knowledge
        """
        import warnings

        # Compute the actual rotation angles that will be used
        # Account for gamma scaling and feature map transformation
        if self.feature_map == "quadratic":
            # For quadratic, angles are gamma * x^2
            effective_angles = np.abs(self.gamma * x * x)
        else:
            # For linear (and default), angles are gamma * x
            effective_angles = np.abs(self.gamma * x)

        # Check if any angle exceeds the threshold
        max_angle = float(np.max(effective_angles))
        n_exceeding = int(np.sum(effective_angles > warn_threshold))

        if n_exceeding > 0:
            # Compute how many full rotations the max angle represents
            full_rotations = max_angle / (2 * np.pi)

            warnings.warn(
                f"Input values produce rotation angles outside recommended range. "
                f"{n_exceeding}/{len(x)} features exceed |angle| > 2π "
                f"(max angle: {max_angle:.2f} rad ≈ {full_rotations:.1f} full rotations). "
                f"This may cause precision loss due to angle wrapping. "
                f"Consider normalizing inputs to [-2π, 2π] range. "
                f"To suppress this warning, normalize your data or use "
                f"warnings.filterwarnings('ignore', category=UserWarning).",
                UserWarning,
                stacklevel=3,  # Points to user's call to get_circuit
            )

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for a single data sample.

        Strict variant: a 2D input with more than one row raises
        :class:`ValueError`. Use :meth:`get_circuits` for batches.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,) or (1, n_features).
        backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.

        Raises
        ------
        ValueError
            If input shape doesn't match n_features, contains multiple
            samples, or backend is unknown.

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, reps=2)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> circuit = enc.get_circuit(x, backend='pennylane')
        """
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            if x_validated.shape[0] != 1:
                raise ValueError(
                    f"get_circuit expects a single sample, got shape "
                    f"{x_validated.shape}. Use get_circuits for batches."
                )
            x_validated = x_validated[0]
        return self._get_circuit_from_validated(x_validated, backend)

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input.

        Encoding-specific seam called by the public ``get_circuit`` and
        the inherited ``get_circuits`` template method. Performs the QAOA
        input-range check and dispatches to the backend implementation.

        Accepts a 1D array (the normal case) or a 2D single-sample
        ``(1, n_features)`` array for robustness when called directly.
        """
        # Defensive 2D-to-1D handling for direct callers; ``get_circuit``
        # and the inherited ``get_circuits`` already pass 1D input.
        if x.ndim == 2:
            x = x[0]

        # Warn when input values fall outside the recommended range.
        # Rotation gates are 2π-periodic so values wrap around silently;
        # this can mask user data-scaling issues.
        self._check_input_range(x)

        if backend == "pennylane":
            return self._to_pennylane(x)
        elif backend == "qiskit":
            return self._to_qiskit(x)
        elif backend == "cirq":
            return self._to_cirq(x)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
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
        include_initial_h = self.include_initial_h
        data_rotation = self.data_rotation
        mixer_rotation = self.mixer_rotation
        entangling_gate = self.entangling_gate
        pairs = self._entanglement_pairs
        beta = self.beta

        # Precompute data rotation angles
        data_angles = [self._apply_feature_map(x, i) for i in range(n_qubits)]

        # Select rotation gates
        data_gate_map = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}
        mixer_gate_map = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}

        data_gate = data_gate_map[data_rotation]
        mixer_gate = mixer_gate_map[mixer_rotation]

        def circuit() -> None:
            """Apply the QAOA-style encoding gates."""
            # Initial Hadamard layer
            if include_initial_h:
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)

            # QAOA-style repetitions
            for _ in range(reps):
                # Data encoding layer (cost layer)
                for i in range(n_qubits):
                    data_gate(data_angles[i], wires=i)

                # Mixer layer: rotations
                for i in range(n_qubits):
                    mixer_gate(beta, wires=i)

                # Mixer layer: entanglement
                for ctrl, tgt in pairs:
                    if entangling_gate == "cx":
                        qml.CNOT(wires=[ctrl, tgt])
                    elif entangling_gate == "cz":
                        qml.CZ(wires=[ctrl, tgt])
                    elif entangling_gate == "rzz":
                        qml.IsingZZ(beta, wires=[ctrl, tgt])

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

        qc = QuantumCircuit(self.n_qubits, name="QAOAEncoding")

        # Initial Hadamard layer
        if self.include_initial_h:
            for i in range(self.n_qubits):
                qc.h(i)

        # Select rotation methods
        data_gate_map = {"X": qc.rx, "Y": qc.ry, "Z": qc.rz}
        mixer_gate_map = {"X": qc.rx, "Y": qc.ry, "Z": qc.rz}

        data_gate = data_gate_map[self.data_rotation]
        mixer_gate = mixer_gate_map[self.mixer_rotation]

        # QAOA-style repetitions
        for rep in range(self.reps):
            # Data encoding layer
            for i in range(self.n_qubits):
                angle = self._apply_feature_map(x, i)
                data_gate(angle, i)

            # Mixer layer: rotations
            for i in range(self.n_qubits):
                mixer_gate(self.beta, i)

            # Mixer layer: entanglement
            for ctrl, tgt in self._entanglement_pairs:
                if self.entangling_gate == "cx":
                    qc.cx(ctrl, tgt)
                elif self.entangling_gate == "cz":
                    qc.cz(ctrl, tgt)
                elif self.entangling_gate == "rzz":
                    qc.rzz(self.beta, ctrl, tgt)

            # Add barrier between repetitions for clarity
            if rep < self.reps - 1:
                qc.barrier()

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

        # Select rotation gate constructors
        def get_data_gate(angle: float) -> cirq.Gate:
            if self.data_rotation == "X":
                return cirq.rx(angle)
            elif self.data_rotation == "Y":
                return cirq.ry(angle)
            else:  # Z
                return cirq.rz(angle)

        def get_mixer_gate(angle: float) -> cirq.Gate:
            if self.mixer_rotation == "X":
                return cirq.rx(angle)
            elif self.mixer_rotation == "Y":
                return cirq.ry(angle)
            else:  # Z
                return cirq.rz(angle)

        # Initial Hadamard layer
        if self.include_initial_h:
            h_ops = [cirq.H(qubits[i]) for i in range(self.n_qubits)]
            moments.append(cirq.Moment(h_ops))

        # QAOA-style repetitions
        for _ in range(self.reps):
            # Data encoding layer
            data_ops = []
            for i in range(self.n_qubits):
                angle = self._apply_feature_map(x, i)
                data_ops.append(get_data_gate(angle).on(qubits[i]))
            moments.append(cirq.Moment(data_ops))

            # Mixer layer: rotations
            mixer_ops = []
            for i in range(self.n_qubits):
                mixer_ops.append(get_mixer_gate(self.beta).on(qubits[i]))
            moments.append(cirq.Moment(mixer_ops))

            # Mixer layer: entanglement
            # Note on gate parallelization:
            # - CZ and RZZ gates commute and can be safely parallelized
            # - CNOT gates do NOT commute in general, so must be applied
            #   sequentially to match PennyLane/Qiskit behavior
            if self._entanglement_pairs:
                if self.entangling_gate == "cx":
                    # CNOT gates must be applied sequentially to preserve
                    # circuit semantics across all backends
                    for ctrl, tgt in self._entanglement_pairs:
                        moments.append(
                            cirq.Moment([cirq.CNOT(qubits[ctrl], qubits[tgt])])
                        )
                else:
                    # CZ and RZZ gates commute - can parallelize for efficiency
                    ops_with_qubits: list[tuple[Any, tuple[int, int]]] = []
                    for ctrl, tgt in self._entanglement_pairs:
                        if self.entangling_gate == "cz":
                            op = cirq.CZ(qubits[ctrl], qubits[tgt])
                        elif self.entangling_gate == "rzz":
                            # RZZ(β) applies: exp(-i β/2 Z⊗Z)
                            # Cirq's ZZPowGate(exponent=t) applies: exp(-i π t/2 Z⊗Z)
                            # To match: t = β/π
                            op = cirq.ZZPowGate(exponent=self.beta / np.pi).on(
                                qubits[ctrl], qubits[tgt]
                            )
                        else:
                            raise ValueError(
                                f"Unknown entangling gate: {self.entangling_gate}"
                            )
                        ops_with_qubits.append((op, (ctrl, tgt)))

                    # Greedy parallelization for commuting gates
                    entangle_moments: list[tuple[list[Any], set[int]]] = []
                    for op, (ctrl, tgt) in ops_with_qubits:
                        placed = False
                        for moment_ops, used_qubits in entangle_moments:
                            if ctrl not in used_qubits and tgt not in used_qubits:
                                moment_ops.append(op)
                                used_qubits.add(ctrl)
                                used_qubits.add(tgt)
                                placed = True
                                break
                        if not placed:
                            entangle_moments.append(([op], {ctrl, tgt}))

                    for moment_ops, _ in entangle_moments:
                        moments.append(cirq.Moment(moment_ops))

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

        # Initial Hadamard gates
        h_gates = n if self.include_initial_h else 0

        # Per repetition gate counts
        data_rotation_gates_per_rep = n
        mixer_rotation_gates_per_rep = n
        entangling_gates_per_rep = n_pairs

        # Total counts
        single_qubit_gates = h_gates + self.reps * (
            data_rotation_gates_per_rep + mixer_rotation_gates_per_rep
        )
        two_qubit_gates = self.reps * entangling_gates_per_rep
        total_gates = single_qubit_gates + two_qubit_gates

        # Determine if entangling
        is_entangling = n_pairs > 0 and self.reps > 0

        # Determine simulability
        if not is_entangling:
            simulability: Literal[
                "simulable", "conditionally_simulable", "not_simulable"
            ] = "simulable"
        else:
            # Entangling circuits with non-Clifford gates are hard to simulate
            simulability = "not_simulable"

        # Trainability estimate (HEURISTIC - use with caution)
        #
        # WARNING: This is a rough heuristic estimate, NOT a rigorous theoretical
        # prediction. Actual trainability depends on many factors not captured here:
        # - Cost function structure and locality
        # - Initialization strategy
        # - Optimization algorithm
        # - Hardware noise characteristics
        #
        # The estimate considers:
        # 1. Circuit depth (deeper circuits tend toward barren plateaus)
        # 2. Entanglement connectivity (higher connectivity = harder to train)
        # 3. Number of qubits (more qubits = exponentially harder)
        #
        # References for barren plateau phenomena:
        # - McClean et al. (2018) "Barren plateaus in quantum neural network
        #   training landscapes" Nature Communications 9, 4812.
        # - Cerezo et al. (2021) "Cost function dependent barren plateaus"
        #   Nature Communications 12, 1791.
        #
        # Formula: trainability = base - depth_penalty - entanglement_penalty - qubit_penalty
        # Clamped to [0.05, 0.95] range to avoid extreme predictions.
        base_trainability = 0.85

        # Depth penalty: logarithmic scaling to avoid overly harsh penalties
        # Shallow circuits (depth <= 5) have zero penalty
        depth_penalty = 0.02 * np.log1p(max(0, self.depth - 5))

        # Entanglement penalty: full > circular > linear > none
        entanglement_penalties = {
            "none": 0.0,
            "linear": 0.05,
            "circular": 0.08,
            "full": 0.15,
        }
        entanglement_penalty = entanglement_penalties.get(self.entanglement, 0.05)

        # Qubit penalty: exponential decay becomes significant above ~10 qubits
        # Based on barren plateau scaling with system size
        qubit_penalty = 0.01 * max(0, n - 6)

        trainability = (
            base_trainability - depth_penalty - entanglement_penalty - qubit_penalty
        )
        trainability = float(np.clip(trainability, 0.05, 0.95))

        # Build notes
        notes_parts = [
            f"QAOA-style p={self.reps}",
            f"data:{self.data_rotation} mixer:{self.mixer_rotation}",
            f"entanglement:{self.entanglement}",
        ]
        if self.entanglement != "none":
            notes_parts.append(f"gate:{self.entangling_gate}")

        return EncodingProperties(
            n_qubits=n,
            depth=self.depth,
            gate_count=total_gates,
            single_qubit_gates=single_qubit_gates,
            two_qubit_gates=two_qubit_gates,
            parameter_count=0,  # No trainable parameters, all data-dependent
            is_entangling=is_entangling,
            simulability=simulability,
            trainability_estimate=trainability,
            notes=", ".join(notes_parts),
        )

    def get_layer_info(self) -> dict[str, Any]:
        """Get detailed information about the circuit layers.

        Returns
        -------
        dict
            Dictionary containing:
            - 'n_qubits': Number of qubits
            - 'reps': Number of repetitions
            - 'layers_per_rep': Description of layers in each repetition
            - 'entanglement_pairs': List of qubit pairs for entanglement
            - 'total_data_gates': Total number of data encoding gates
            - 'total_mixer_gates': Total number of mixer gates
            - 'total_entangling_gates': Total number of entangling gates

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, reps=2)
        >>> info = enc.get_layer_info()
        >>> info['total_entangling_gates']
        6
        """
        n_pairs = len(self._entanglement_pairs)

        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "layers_per_rep": {
                "data_encoding": f"{self.n_qubits} R{self.data_rotation} gates",
                "mixer_rotation": f"{self.n_qubits} R{self.mixer_rotation} gates",
                "entanglement": f"{n_pairs} {self.entangling_gate.upper()} gates",
            },
            "entanglement_pairs": self._entanglement_pairs,
            "total_data_gates": self.n_qubits * self.reps,
            "total_mixer_gates": self.n_qubits * self.reps,
            "total_entangling_gates": n_pairs * self.reps,
            "include_initial_h": self.include_initial_h,
            "gamma": self.gamma,
            "beta": self.beta,
        }

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get the list of qubit pairs used for entanglement.

        Returns the list of qubit pairs that will have entangling gates applied
        based on the configured entanglement topology. This is useful for:

        - Understanding circuit structure before generation
        - Verifying hardware connectivity requirements
        - Debugging entanglement patterns
        - Computing resource requirements

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for entangling operations.
            Each tuple (i, j) indicates an entangling gate between qubits i and j.

        Notes
        -----
        Pair counts by topology:

        - Full: n(n-1)/2 pairs (all combinations where i < j)
        - Linear: n-1 pairs (nearest neighbors only)
        - Circular: n-1 pairs for n<=2, n pairs for n>2 (adds wrap-around)
        - None: 0 pairs (no entanglement)

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, entanglement='full')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        >>> enc_linear = QAOAEncoding(n_features=4, entanglement='linear')
        >>> enc_linear.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        >>> enc_circular = QAOAEncoding(n_features=4, entanglement='circular')
        >>> enc_circular.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3), (3, 0)]

        >>> enc_none = QAOAEncoding(n_features=4, entanglement='none')
        >>> enc_none.get_entanglement_pairs()
        []

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts by type.
        resource_summary : Get comprehensive resource summary.
        """
        # Return a copy to prevent external mutation of internal state
        return list(self._entanglement_pairs)

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type.

        Computes the exact number of each gate type in the QAOA encoding
        circuit. This is useful for:

        - Resource estimation before circuit execution
        - Hardware compatibility assessment
        - Comparing different encoding configurations
        - Debugging and verification

        Returns
        -------
        GateCountBreakdown
            TypedDict with gate counts:

            - ``'hadamard'``: Number of Hadamard gates (n_qubits if include_initial_h)
            - ``'data_rotation'``: Number of data encoding R_data gates
            - ``'mixer_rotation'``: Number of mixer R_mixer gates
            - ``'entangling'``: Number of two-qubit entangling gates
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Total two-qubit gates (= entangling)
            - ``'total'``: Total gate count

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, reps=2, entanglement='linear')
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['hadamard']
        4
        >>> breakdown['data_rotation']
        8
        >>> breakdown['mixer_rotation']
        8
        >>> breakdown['entangling']
        6
        >>> breakdown['total']
        26

        Compare configurations:

        >>> enc_full = QAOAEncoding(n_features=4, reps=2, entanglement='full')
        >>> enc_full.gate_count_breakdown()['entangling']
        12

        See Also
        --------
        get_entanglement_pairs : Get qubit pairs for entanglement.
        resource_summary : Get comprehensive resource summary.
        properties : Get computed encoding properties.
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        # Gate counts
        hadamard = n if self.include_initial_h else 0
        data_rotation = n * self.reps
        mixer_rotation = n * self.reps
        entangling = n_pairs * self.reps

        # Totals
        total_single_qubit = hadamard + data_rotation + mixer_rotation
        total_two_qubit = entangling
        total = total_single_qubit + total_two_qubit

        _logger.debug(
            "Gate count breakdown: H=%d, data_rot=%d, mixer_rot=%d, "
            "entangling=%d, total=%d",
            hadamard,
            data_rotation,
            mixer_rotation,
            entangling,
            total,
        )

        return GateCountBreakdown(
            hadamard=hadamard,
            data_rotation=data_rotation,
            mixer_rotation=mixer_rotation,
            entangling=entangling,
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

            **Circuit Structure**:
                - ``n_qubits``: Number of qubits required
                - ``n_features``: Number of input features
                - ``depth``: Circuit depth (theoretical estimate)
                - ``reps``: Number of QAOA-style repetitions

            **Entanglement Information**:
                - ``entanglement``: Entanglement topology ('full', 'linear', etc.)
                - ``entanglement_pairs``: List of qubit pairs with entangling gates
                - ``n_entanglement_pairs``: Number of qubit pairs per layer
                - ``entangling_gate``: Type of entangling gate ('cx', 'cz', 'rzz')

            **Gate Counts**:
                - ``gate_counts``: Detailed breakdown from ``gate_count_breakdown()``

            **Encoding Parameters**:
                - ``gamma``: Data encoding scaling factor
                - ``beta``: Mixer rotation scaling factor
                - ``data_rotation``: Rotation axis for data encoding
                - ``mixer_rotation``: Rotation axis for mixer
                - ``include_initial_h``: Whether initial Hadamards are applied
                - ``feature_map``: Feature mapping type ('linear', 'quadratic')

            **Encoding Characteristics**:
                - ``is_entangling``: Whether the encoding creates entanglement
                - ``simulability``: Classical simulability status
                - ``trainability_estimate``: Estimated trainability (0.0 to 1.0)

            **Hardware Requirements**:
                - ``hardware_requirements``: Dict with connectivity and gate info

        Examples
        --------
        Get complete resource analysis:

        >>> enc = QAOAEncoding(n_features=4, reps=2, entanglement='full')
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['gate_counts']['total']
        32
        >>> summary['is_entangling']
        True

        Compare different configurations:

        >>> enc_linear = QAOAEncoding(n_features=8, entanglement='linear')
        >>> enc_full = QAOAEncoding(n_features=8, entanglement='full')
        >>> enc_linear.resource_summary()['n_entanglement_pairs']
        7
        >>> enc_full.resource_summary()['n_entanglement_pairs']
        28

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts.
        get_entanglement_pairs : Get qubit pairs for entanglement.
        properties : Get computed encoding properties.
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties
        n_pairs = len(self._entanglement_pairs)

        # Determine hardware connectivity requirement
        if self.entanglement == "full":
            connectivity = "all-to-all"
        elif self.entanglement == "circular":
            connectivity = "ring"
        elif self.entanglement == "linear":
            connectivity = "linear (nearest-neighbor)"
        else:
            connectivity = "none required"

        summary = {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            # Entanglement information
            "entanglement": self.entanglement,
            "entanglement_pairs": self.get_entanglement_pairs(),
            "n_entanglement_pairs": n_pairs,
            "entangling_gate": self.entangling_gate,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding parameters
            "gamma": self.gamma,
            "beta": self.beta,
            "data_rotation": self.data_rotation,
            "mixer_rotation": self.mixer_rotation,
            "include_initial_h": self.include_initial_h,
            "feature_map": self.feature_map,
            # Encoding characteristics
            "is_entangling": props.is_entangling,
            "simulability": props.simulability,
            "trainability_estimate": props.trainability_estimate,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": connectivity,
                "native_gates": [
                    "H" if self.include_initial_h else None,
                    f"R{self.data_rotation}",
                    f"R{self.mixer_rotation}",
                    self.entangling_gate.upper() if n_pairs > 0 else None,
                ],
                "min_qubits": self.n_qubits,
            },
        }

        # Clean up None values from native_gates
        hw_req = cast("dict[str, Any]", summary["hardware_requirements"])
        hw_req["native_gates"] = [g for g in hw_req["native_gates"] if g is not None]

        _logger.debug(
            "Resource summary generated: n_qubits=%d, depth=%d, total_gates=%d, "
            "n_entanglement_pairs=%d, entanglement=%s",
            self.n_qubits,
            self.depth,
            gate_counts["total"],
            n_pairs,
            self.entanglement,
        )

        return summary

    def get_depth(
        self,
        backend: BackendType = "qiskit",
        x: ArrayLike | None = None,
    ) -> int:
        """Get the actual circuit depth for a specific backend.

        Unlike the `depth` property which returns a theoretical estimate,
        this method generates the actual circuit and returns the depth
        as computed by the backend's native method.

        Backend Depth Semantics
        -----------------------
        Different backends compute depth differently:

        - **Qiskit**: ``circuit.depth()`` counts the longest path through
          the circuit DAG. The transpiler may optimize gates, potentially
          reducing depth. Use ``transpile(circuit).depth()`` for hardware.

        - **Cirq**: ``len(circuit)`` counts Moments. Our implementation
          already groups parallel gates into Moments for optimal depth.

        - **PennyLane**: Returns the theoretical depth since PennyLane
          circuits are functions without a native depth method. The actual
          depth on hardware depends on the device's gate decomposition.

        Parameters
        ----------
        backend : {'pennylane', 'qiskit', 'cirq'}, default='qiskit'
            Target quantum computing framework.
        x : array-like, optional
            Input features of shape (n_features,). If not provided, random
            values in [0, 2π] are used. The depth is independent of input
            values but a valid input is needed to construct the circuit.

        Returns
        -------
        int
            Actual circuit depth as computed by the backend.

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, reps=2, entanglement='linear')
        >>> enc.depth  # Theoretical estimate
        9
        >>> enc.get_depth(backend='cirq')  # Actual Cirq depth (Moments)
        9
        >>> enc.get_depth(backend='qiskit') >= enc.depth  # Qiskit may be higher
        True

        Notes
        -----
        For production use, always verify depth on your target backend
        and hardware, as transpilation and gate decomposition can affect
        the final circuit depth.

        See Also
        --------
        depth : Theoretical depth estimate (faster, no circuit generation).
        """
        # Generate sample input if not provided
        if x is None:
            # Use deterministic seed for reproducibility in depth queries
            rng = np.random.default_rng(seed=42)
            x = rng.uniform(0, 2 * np.pi, size=self.n_qubits)

        # Validate input
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]

        # Generate circuit and compute backend-specific depth
        if backend == "qiskit":
            try:
                from qiskit import QuantumCircuit
            except ImportError as e:
                raise ImportError(
                    "Qiskit is required for the 'qiskit' backend. "
                    "Install it with: pip install qiskit"
                ) from e

            circuit = self._to_qiskit(x_validated)
            return int(circuit.depth())

        elif backend == "cirq":
            try:
                import cirq
            except ImportError as e:
                raise ImportError(
                    "Cirq is required for the 'cirq' backend. "
                    "Install it with: pip install cirq-core"
                ) from e

            circuit = self._to_cirq(x_validated)
            # Cirq depth = number of Moments
            return len(circuit)

        elif backend == "pennylane":
            # PennyLane circuits are functions; no native depth method.
            # Return the theoretical depth as a best approximation.
            # Note: Actual depth on hardware may differ due to decomposition.
            return self.depth

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
            )

    def compute_data_angles(
        self,
        x: ArrayLike,
    ) -> NDArray[np.floating[Any]]:
        """Compute the data encoding rotation angles for given input.

        This is useful for debugging and understanding the encoding.

        Parameters
        ----------
        x : array-like
            Input feature vector of shape (n_features,).

        Returns
        -------
        ndarray
            Array of shape (n_qubits,) containing the rotation angle for
            each qubit in the data encoding layer.

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, gamma=2.0)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> angles = enc.compute_data_angles(x)
        >>> angles
        array([0.2, 0.4, 0.6, 0.8])
        """
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]

        return np.array(
            [self._apply_feature_map(x_validated, i) for i in range(self.n_qubits)]
        )

    def copy(self, **kwargs: Any) -> QAOAEncoding:
        """Create a copy of this encoding with optional parameter overrides.

        This method enables creating variations of an encoding without
        modifying the original, following the immutability pattern.

        Parameters
        ----------
        **kwargs : Any
            Parameters to override in the copy. Any parameter accepted
            by __init__ can be overridden.

        Returns
        -------
        QAOAEncoding
            New encoding instance with updated parameters.

        Examples
        --------
        Create a deeper version of an existing encoding:

        >>> enc = QAOAEncoding(n_features=4, reps=2, gamma=1.5)
        >>> enc_deep = enc.copy(reps=5)
        >>> enc_deep.reps
        5
        >>> enc_deep.gamma  # Preserved from original
        1.5

        Create a variation with different entanglement:

        >>> enc_full = enc.copy(entanglement='full', entangling_gate='cx')
        >>> enc_full.entanglement
        'full'
        """
        # Start with current parameters
        params: dict[str, Any] = {
            "n_features": self.n_features,
            "reps": self.reps,
            "data_rotation": self.data_rotation,
            "mixer_rotation": self.mixer_rotation,
            "entanglement": self.entanglement,
            "entangling_gate": self.entangling_gate,
            "gamma": self.gamma,
            "beta": self.beta,
            "include_initial_h": self.include_initial_h,
            "feature_map": self.feature_map,
        }
        # Override with provided kwargs
        params.update(kwargs)
        return QAOAEncoding(**params)

    def to_dict(self) -> dict[str, Any]:
        """Serialize encoding configuration to a dictionary.

        Returns a JSON-compatible dictionary containing all parameters
        needed to reconstruct this encoding.

        Returns
        -------
        dict[str, Any]
            Dictionary with encoding class name and all parameters.

        Examples
        --------
        >>> enc = QAOAEncoding(n_features=4, reps=3, gamma=2.0)
        >>> config = enc.to_dict()
        >>> config['class']
        'QAOAEncoding'
        >>> config['reps']
        3

        Save to JSON file:

        >>> import json
        >>> with open('encoding_config.json', 'w') as f:  # doctest: +SKIP
        ...     json.dump(enc.to_dict(), f, indent=2)

        See Also
        --------
        from_dict : Reconstruct encoding from dictionary.
        """
        return {
            "class": "QAOAEncoding",
            "n_features": self.n_features,
            "reps": self.reps,
            "data_rotation": self.data_rotation,
            "mixer_rotation": self.mixer_rotation,
            "entanglement": self.entanglement,
            "entangling_gate": self.entangling_gate,
            "gamma": self.gamma,
            "beta": self.beta,
            "include_initial_h": self.include_initial_h,
            "feature_map": self.feature_map,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> QAOAEncoding:
        """Reconstruct encoding from a dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary containing encoding parameters, typically
            created by ``to_dict()``.

        Returns
        -------
        QAOAEncoding
            Reconstructed encoding instance.

        Raises
        ------
        ValueError
            If the config specifies a different encoding class.

        Examples
        --------
        >>> config = {
        ...     'class': 'QAOAEncoding',
        ...     'n_features': 4,
        ...     'reps': 3,
        ...     'gamma': 2.0
        ... }
        >>> enc = QAOAEncoding.from_dict(config)
        >>> enc.reps
        3

        Load from JSON file:

        >>> import json
        >>> with open('encoding_config.json', 'r') as f:  # doctest: +SKIP
        ...     config = json.load(f)
        >>> enc = QAOAEncoding.from_dict(config)  # doctest: +SKIP

        See Also
        --------
        to_dict : Serialize encoding to dictionary.
        """
        config = config.copy()  # Don't modify original
        class_name = config.pop("class", "QAOAEncoding")

        if class_name != "QAOAEncoding":
            raise ValueError(
                f"Config specifies class '{class_name}', expected 'QAOAEncoding'. "
                f"Use the appropriate class's from_dict method."
            )

        return cls(**config)

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, "
            f"reps={self.reps}, "
            f"data_rotation='{self.data_rotation}', "
            f"mixer_rotation='{self.mixer_rotation}', "
            f"entanglement='{self.entanglement}', "
            f"entangling_gate='{self.entangling_gate}', "
            f"gamma={self.gamma}, "
            f"beta={self.beta}, "
            f"include_initial_h={self.include_initial_h}, "
            f"feature_map='{self.feature_map}')"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoding."""
        if not isinstance(other, QAOAEncoding):
            return NotImplemented
        return (
            self.n_features == other.n_features
            and self.reps == other.reps
            and self.data_rotation == other.data_rotation
            and self.mixer_rotation == other.mixer_rotation
            and self.entanglement == other.entanglement
            and self.entangling_gate == other.entangling_gate
            and self.gamma == other.gamma
            and self.beta == other.beta
            and self.include_initial_h == other.include_initial_h
            and self.feature_map == other.feature_map
        )

    def __hash__(self) -> int:
        """Return hash of the encoding."""
        return hash(
            (
                self.__class__.__name__,
                self.n_features,
                self.reps,
                self.data_rotation,
                self.mixer_rotation,
                self.entanglement,
                self.entangling_gate,
                self.gamma,
                self.beta,
                self.include_initial_h,
                self.feature_map,
            )
        )
