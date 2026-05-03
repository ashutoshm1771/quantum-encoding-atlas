"""Hamiltonian Quantum Data Encoding module.

This module implements Hamiltonian-based quantum data encoding, where classical
data is encoded into quantum states through time evolution under a data-dependent
Hamiltonian operator. This approach creates rich entanglement structures and
provides a natural connection to quantum simulation and dynamics.

Hamiltonian encoding is particularly notable for quantum machine learning because:

1. **Physics-Motivated Design**: The encoding is grounded in quantum dynamics,
   providing a natural framework for encoding classical data into quantum states
   with explicit control over evolution time and Trotter decomposition.

2. **Flexible Hamiltonian Types**: Supports multiple Hamiltonian structures
   (IQP, XY, Heisenberg, Pauli-Z) allowing users to choose the interaction
   pattern best suited for their specific machine learning task.

3. **Rich Entanglement Structures**: Creates highly entangled quantum states
   through configurable two-qubit interactions that capture pairwise feature
   correlations in the quantum state.

4. **Expressivity Control**: The evolution time parameter provides explicit
   control over the "strength" of the encoding, allowing fine-tuning of the
   quantum feature map's expressivity.

5. **Trotterization Framework**: Uses well-understood Trotterization techniques
   from quantum simulation, providing a clear path to higher-order approximations
   if needed.

The encoding generates quantum states via:

    |ψ(x)⟩ = exp(-iH(x)t) |+⟩^⊗n

where:
    - |+⟩ = H|0⟩ = (|0⟩ + |1⟩)/√2 is the equal superposition state
    - H(x) is a data-dependent Hamiltonian
    - t is the evolution time
    - n is the number of qubits (features)

The initial Hadamard layer creates an equal superposition state, which is then
evolved under the data-dependent Hamiltonian. This is a standard approach in
quantum machine learning that enables richer feature maps compared to starting
from the computational basis |0...0⟩.

Mathematical Framework
----------------------
For a Hamiltonian H(x) = Σⱼ cⱼ(x) Pⱼ where Pⱼ are Pauli strings,
the time evolution is decomposed into products of simpler exponentials:

    exp(-iH(x)t) ≈ [∏ⱼ exp(-icⱼ(x)t/r Pⱼ)]^r

where r is the number of Trotter steps (reps).

The complete circuit structure is:

1. **Initial State Preparation**: H^⊗n |0...0⟩ = |+...+⟩
2. **Trotterized Time Evolution**: [∏ⱼ exp(-icⱼ(x)t/r Pⱼ)]^r

For two-qubit terms, the rotation angle uses the quantum kernel feature map:

    angle = time_step × (π - xᵢ)(π - xⱼ)

This formula has a critical point at x = π where interactions vanish.

Supported Hamiltonian Types
---------------------------
- **'iqp'**: Diagonal Hamiltonians with ZZ interactions (IQP-style)
  H(x) = Σᵢ xᵢZᵢ + Σᵢⱼ (π - xᵢ)(π - xⱼ)ZᵢZⱼ

- **'xy'**: XY model with data-dependent couplings
  H(x) = Σᵢⱼ (π - xᵢ)(π - xⱼ)(XᵢXⱼ + YᵢYⱼ)

- **'heisenberg'**: Heisenberg model with all Pauli interactions
  H(x) = Σᵢⱼ (π - xᵢ)(π - xⱼ)(XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ)

- **'pauli_z'**: Single-qubit Z rotations with pairwise ZZ interactions
  H(x) = Σᵢ xᵢZᵢ + Σᵢⱼ (π - xᵢ)(π - xⱼ)ZᵢZⱼ

Note on Data Preprocessing
--------------------------
For Hamiltonian encoding, input features should be carefully scaled:

- **Recommended ranges**: Scale features to [0, 1] or [-1, 1] for consistent
  behavior. These ranges avoid the critical point and provide good rotation
  coverage.

- **Critical point at π**: The two-qubit rotation formula (π - xᵢ)(π - xⱼ)
  produces zero interaction when features equal π (~3.14159). Avoid scaling
  features to ranges that include π in the interior (e.g., [0, 2π]).

- **Evolution time scaling**: Larger ``evolution_time`` values amplify rotation
  angles. If using pre-normalized features, adjust ``evolution_time`` to
  achieve desired rotation magnitudes.

- **Single-qubit terms**: Use angle = time_step × xᵢ directly, so larger
  feature values produce larger rotations.

Use Cases
---------
Hamiltonian encoding is particularly suited for:

- **Quantum simulation-inspired ML**: Problems where physics-based encoding
  provides natural inductive bias (e.g., molecular property prediction,
  materials science)

- **Quantum kernel methods**: Computing quantum kernels with configurable
  Hamiltonian structures for classification and regression tasks

- **Variational quantum classifiers**: As a physics-motivated feature map
  layer in hybrid quantum-classical models

- **Research and experimentation**: Exploring how different Hamiltonian
  structures affect quantum machine learning performance

- **Time-series encoding**: The explicit time evolution framework provides
  a natural connection to temporal data encoding

- **Hardware-aware design**: Linear and circular entanglement patterns
  map efficiently to hardware with limited qubit connectivity

Limitations
-----------
- **Qubit scaling**: Requires n qubits for n features (linear scaling)
- **Gate count**: Full entanglement creates O(n²) two-qubit gates per Trotter
  step, which may exceed NISQ device capabilities for large feature counts
- **Trainability**: Deep circuits (high reps) with full entanglement may
  exhibit barren plateaus, making gradient-based optimization challenging
- **Hardware constraints**: Full entanglement requires all-to-all qubit
  connectivity, which is not available on all quantum hardware
- **Critical point sensitivity**: Features near π produce minimal two-qubit
  interactions, which may be undesirable for some applications
- **Trotterization error**: First-order Trotter decomposition introduces
  approximation errors; increase ``reps`` for higher accuracy

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation and entanglement pattern computation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.hamiltonian').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters (n_features, hamiltonian_type, reps, entanglement)
- Entanglement pair computation details
- Circuit generation progress for each backend
- Gate count breakdowns
- Critical point warnings for feature values near π

For more detailed logging with custom formatting:

    >>> import logging
    >>> logger = logging.getLogger('encoding_atlas.encodings.hamiltonian')
    >>> logger.setLevel(logging.DEBUG)
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    >>> logger.addHandler(handler)

Resource Analysis
-----------------
HamiltonianEncoding provides methods to analyze circuit resources:

- ``get_entanglement_pairs()``: Inspect which qubit pairs have interactions
- ``gate_count_breakdown()``: Get detailed gate counts by type (upper bound)
- ``count_gates(x)``: Get exact gate counts for specific input data
- ``resource_summary()``: Comprehensive resource analysis
- ``properties``: Access computed circuit properties (depth, gate counts, etc.)

These methods help understand resource requirements before execution:

    >>> enc = HamiltonianEncoding(n_features=8, entanglement='full')
    >>> len(enc.get_entanglement_pairs())  # Number of ZZ interactions
    28
    >>> enc.gate_count_breakdown()
    {'hadamard': 8, 'rz': 36, 's_gates': 0, 'cnot': 56, ...}
    >>> summary = enc.resource_summary()
    >>> summary['hardware_requirements']['connectivity']
    'all-to-all'

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
.. [1] Schuld, M., Sweke, R., & Meyer, J. J. (2021). "Effect of data encoding on
       the expressive power of variational quantum-machine-learning models."
       Physical Review A, 103(3), 032430.
.. [2] Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
       feature spaces." Nature, 567(7747), 209-212.
.. [3] Lloyd, S., Mohseni, M., & Rebentrost, P. (2014). "Quantum algorithms for
       supervised and unsupervised machine learning." arXiv:1307.0411.
.. [4] Childs, A. M., et al. (2018). "Toward the first quantum simulation with
       quantum speedup." PNAS, 115(38), 9456-9461.
.. [5] McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
       network training landscapes." Nature Communications, 9, 4812.
"""

from __future__ import annotations

import logging
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
#   To enable debug logging for Hamiltonian encoding:
#
#   >>> import logging
#   >>> logging.getLogger('encoding_atlas.encodings.hamiltonian').setLevel(logging.DEBUG)
#   >>> handler = logging.StreamHandler()
#   >>> handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
#   >>> logging.getLogger('encoding_atlas.encodings.hamiltonian').addHandler(handler)
#
# Log Levels:
#   - DEBUG: Detailed information for diagnosing issues (entanglement pairs,
#            gate counts, circuit generation steps, input value ranges)
#   - INFO: General operational information (encoding created, circuit generated)
#   - WARNING: Potential issues (large feature count with full entanglement,
#              values near critical point π)
#
# The logger is named after the module to allow granular control over log output.
# By default, no handlers are attached, so no output is produced unless the
# application configures logging.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["HamiltonianEncoding", "GateCountBreakdown"]

# =============================================================================
# Module-level constants for numerical precision
# =============================================================================

# Tolerance for floating-point angle comparisons.
# Angles smaller than this threshold are considered effectively zero and
# the corresponding gates are skipped to avoid unnecessary circuit depth.
#
# Rationale for 1e-10:
# - Much larger than typical floating-point noise (~1e-16 for double precision)
# - Much smaller than any physically meaningful rotation angle
# - A rotation of 1e-10 radians produces state changes of O(1e-20) in probability,
#   which is far below any measurement precision
# - Conservative enough to not skip intentionally small rotations
#
# This value can be adjusted if needed for specific use cases, but 1e-10
# provides a good balance between numerical stability and correctness.
_ANGLE_TOLERANCE: float = 1e-10

# Tolerance for detecting values near the critical point π in the two-qubit
# rotation formula (π - x_i)(π - x_j). When a feature value is within this
# tolerance of π, the two-qubit interaction becomes negligible.
#
# A tolerance of 0.1 (~3% of π) is used to catch values that are "close enough"
# to π that users should be warned, while not being overly sensitive.
_CRITICAL_POINT_TOLERANCE: float = 0.1

# =============================================================================
# Backend version requirements
# =============================================================================
#
# Minimum tested versions for each backend. These are the versions that the
# implementation has been tested against and is known to work with. Older
# versions may work but are not guaranteed.
#
# Version format: (major, minor, patch) or (major, minor) for flexibility
_BACKEND_MIN_VERSIONS: dict[str, tuple[int, ...]] = {
    "pennylane": (0, 32),  # PennyLane 0.32+ for stable gate API
    "qiskit": (1, 0),  # Qiskit 1.0+ for new API (post-terra deprecation)
    "cirq": (1, 0),  # Cirq 1.0+ for stable API
}

# Whether to warn about untested versions (vs. raising an error)
_BACKEND_VERSION_WARN_ONLY: bool = True


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Parameters
    ----------
    version_str : str
        Version string like "1.2.3" or "0.32.0".

    Returns
    -------
    tuple[int, ...]
        Version as tuple of integers, e.g., (1, 2, 3).

    Examples
    --------
    >>> _parse_version("1.2.3")
    (1, 2, 3)
    >>> _parse_version("0.32")
    (0, 32)
    """
    # Handle versions with suffixes like "1.2.3.dev0" or "1.2.3rc1"
    # by only taking the numeric parts
    parts = []
    for part in version_str.split(".")[:3]:  # Only take major.minor.patch
        # Extract numeric prefix (handles "3rc1" -> "3")
        numeric = ""
        for char in part:
            if char.isdigit():
                numeric += char
            else:
                break
        if numeric:
            parts.append(int(numeric))
    return tuple(parts) if parts else (0,)


def _check_backend_version(backend: str, module: Any) -> None:
    """Check if a backend module meets minimum version requirements.

    Issues a warning if the installed version is older than the minimum
    tested version. This helps users identify potential compatibility issues.

    Parameters
    ----------
    backend : str
        Backend name ('pennylane', 'qiskit', 'cirq').
    module : module
        The imported backend module.

    Warns
    -----
    UserWarning
        If the installed version is older than the minimum tested version.
    """
    if backend not in _BACKEND_MIN_VERSIONS:
        return  # Unknown backend, skip version check

    min_version = _BACKEND_MIN_VERSIONS[backend]

    # Get version from module
    version_str = getattr(module, "__version__", None)
    if version_str is None:
        # Some modules use VERSION instead
        version_str = getattr(module, "VERSION", None)

    if version_str is None:
        # Cannot determine version, skip check
        return

    try:
        installed_version = _parse_version(str(version_str))
    except (ValueError, AttributeError):
        # Cannot parse version, skip check
        return

    # Compare versions (tuple comparison works for version ordering)
    # Pad shorter tuple with zeros for fair comparison
    max_len = max(len(installed_version), len(min_version))
    installed_padded = installed_version + (0,) * (max_len - len(installed_version))
    min_padded = min_version + (0,) * (max_len - len(min_version))

    if installed_padded < min_padded:
        import warnings

        min_version_str = ".".join(map(str, min_version))
        warnings.warn(
            f"BACKEND VERSION WARNING: {backend} version {version_str} is older than "
            f"the minimum tested version {min_version_str}.\n"
            f"  The HamiltonianEncoding has been tested with {backend} >= {min_version_str}.\n"
            f"  Older versions may have API differences that cause errors.\n"
            f"  Recommended: pip install --upgrade {backend}",
            UserWarning,
            stacklevel=4,
        )


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
        Number of Hadamard gates. Each qubit receives one H gate in the
        initial state preparation layer.

    rz : int
        Number of RZ rotation gates. These are used for both single-qubit
        Z rotations and in the decomposition of ZZ/XX/YY interactions.

    s_gates : int
        Number of S and Sdg gates. Used in YY interaction decomposition
        for basis changes to/from the Y basis.

    cnot : int
        Number of CNOT gates. Each two-qubit interaction (ZZ, XX, YY)
        requires 2 CNOT gates for its decomposition.

    total_single_qubit : int
        Total number of single-qubit gates (hadamard + rz + s_gates).

    total_two_qubit : int
        Total number of two-qubit gates (equals cnot count).

    total : int
        Total gate count (total_single_qubit + total_two_qubit).

    Examples
    --------
    >>> enc = HamiltonianEncoding(n_features=4, reps=1, entanglement='full')
    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['hadamard']
    4
    >>> breakdown['cnot']
    12
    >>> breakdown['total']
    26

    Notes
    -----
    Gate counts are computed as upper bounds assuming all rotation angles
    are significant (not skipped due to being near zero). For exact counts
    with specific input data, use ``count_gates(x)``.
    """

    hadamard: int
    """Number of Hadamard gates (initial state preparation)."""

    rz: int
    """Number of RZ rotation gates."""

    s_gates: int
    """Number of S and Sdg gates (for YY interactions)."""

    cnot: int
    """Number of CNOT gates (2 per two-qubit interaction)."""

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates."""

    total: int
    """Total gate count."""


def _is_significant_angle(angle: float) -> bool:
    """Check if a rotation angle is significant (not effectively zero).

    This function provides a tolerance-based comparison to avoid floating-point
    equality issues. Very small angles (smaller than _ANGLE_TOLERANCE) are
    considered effectively zero, and the corresponding gates can be skipped.

    Parameters
    ----------
    angle : float
        The rotation angle to check.

    Returns
    -------
    bool
        True if the angle is significant (should apply the gate),
        False if the angle is effectively zero (can skip the gate).

    Examples
    --------
    >>> _is_significant_angle(0.5)
    True
    >>> _is_significant_angle(1e-16)
    False
    >>> _is_significant_angle(0.0)
    False
    >>> _is_significant_angle(-0.5)
    True

    Notes
    -----
    The tolerance is set to 1e-10, which is:
    - 6 orders of magnitude larger than double-precision epsilon (~1e-16)
    - Small enough that skipped rotations have no measurable effect
    - Handles numerical artifacts from operations like `time_step * x[i]`
    """
    # Use abs() to handle both positive and negative angles
    # The comparison is strict (>) rather than (>=) to be conservative
    return abs(angle) > _ANGLE_TOLERANCE


class HamiltonianEncoding(BaseEncoding):
    """Hamiltonian-based quantum data encoding via time evolution.

    This encoding maps classical data to quantum states through evolution
    under a data-dependent Hamiltonian operator. The Hamiltonian structure
    determines the type of correlations and entanglement in the encoded state.

    The circuit implements time evolution using Trotterization, decomposing
    the evolution operator into a product of simpler gates. Different
    Hamiltonian types provide different encoding strategies suitable for
    various machine learning tasks.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be at least 1.
    hamiltonian_type : {'iqp', 'xy', 'heisenberg', 'pauli_z'}, default='iqp'
        Type of Hamiltonian structure to use:
        - 'iqp': Diagonal Hamiltonian with Z and ZZ terms (IQP-style)
        - 'xy': XY model with XX and YY interactions
        - 'heisenberg': Full Heisenberg model with XX, YY, and ZZ terms
        - 'pauli_z': Single-qubit Z rotations with ZZ interactions
    evolution_time : float, default=1.0
        Total evolution time parameter t in exp(-iHt).
        Larger values create more complex states but may lead to
        harder optimization landscapes.
    reps : int, default=2
        Number of Trotter steps (repetitions). Higher values improve
        the approximation of the exponential but increase circuit depth.
        Must be at least 1.
    entanglement : {'full', 'linear', 'circular'}, default='full'
        Entanglement pattern for two-qubit terms:
        - 'full': All pairs (i,j) where i < j
        - 'linear': Nearest-neighbor pairs (i, i+1)
        - 'circular': Linear plus wraparound (n-1, 0)
    insert_barriers : bool, default=True
        Whether to insert barriers between Trotter steps in circuits.
        Useful for visualization and preventing gate optimizations.
        Only effective for the Qiskit backend. PennyLane does not support
        barriers, and the Cirq backend does not currently insert them.

    Attributes
    ----------
    hamiltonian_type : str
        The Hamiltonian structure type.
    evolution_time : float
        Evolution time parameter.
    reps : int
        Number of Trotter steps.
    entanglement : str
        Entanglement pattern.
    insert_barriers : bool
        Whether barriers are inserted.

    Examples
    --------
    Create a basic IQP-style Hamiltonian encoding:

    >>> from encoding_atlas import HamiltonianEncoding
    >>> import numpy as np
    >>> enc = HamiltonianEncoding(n_features=4)
    >>> enc.hamiltonian_type
    'iqp'
    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')

    Use XY model with linear connectivity:

    >>> enc = HamiltonianEncoding(
    ...     n_features=8,
    ...     hamiltonian_type='xy',
    ...     entanglement='linear',
    ...     reps=3
    ... )
    >>> enc.properties.is_entangling
    True

    Custom evolution time for different encoding strength:

    >>> enc = HamiltonianEncoding(
    ...     n_features=4,
    ...     evolution_time=2.0,
    ...     reps=4
    ... )
    >>> enc.evolution_time
    2.0

    References
    ----------
    .. [1] Havlíček, V., et al. (2019). "Supervised learning with quantum-
           enhanced feature spaces." Nature, 567(7747), 209-212.
    .. [2] Schuld, M., Sweke, R., & Meyer, J. J. (2021). "Effect of data
           encoding on the expressive power of variational quantum-machine-
           learning models." Physical Review A, 103(3), 032430.

    See Also
    --------
    PauliFeatureMap : Related encoding using Pauli rotations.
    IQPEncoding : IQP-style encoding without explicit time evolution.
    ZZFeatureMap : Simplified encoding with Z and ZZ terms.

    Notes
    -----
    **Hamiltonian Types:**

    The choice of Hamiltonian determines the encoding properties:

    - **IQP**: Efficient for encoding pairwise feature correlations,
      creates diagonal evolution in computational basis.
    - **XY**: Creates coherent superpositions, useful for capturing
      complex feature relationships.
    - **Heisenberg**: Most expressive, combines all interaction types.
    - **Pauli-Z**: Simplest, similar to ZZFeatureMap but with explicit
      time evolution framework.

    **Trotterization:**

    The time evolution operator is approximated using first-order
    Trotterization. For higher accuracy, increase `reps`, though this
    increases circuit depth proportionally.

    **Classical Simulability:**

    - IQP and Pauli-Z Hamiltonians create states in the Clifford hierarchy,
      which are hard to simulate classically but provide quantum advantage.
    - XY and Heisenberg models create general entangled states.

    **Comparison with PauliFeatureMap:**

    While PauliFeatureMap applies rotations directly, HamiltonianEncoding
    uses the time evolution framework, providing a more physics-motivated
    approach with explicit control over evolution time and Trotter steps.

    **Input Scaling and Feature Ranges:**

    The two-qubit rotation angle formula uses (π - xᵢ)(π - xⱼ), which has
    important implications for input scaling:

    - **Critical point at x = π**: When feature values equal π (~3.14159),
      the two-qubit interaction becomes zero. Avoid scaling features to
      ranges that include π in the interior.

    - **Recommended ranges**:
        - [0, 1]: Safe, produces angles in range [0, ~9.87] before time scaling
        - [-1, 1]: Safe, produces angles in range [0, ~17.2] before time scaling
        - [0, 2π]: CAUTION - includes critical point at π, features near π
          will have minimal two-qubit interactions

    - **Single-qubit terms**: Use angle = time_step × xᵢ directly, so
      larger feature values produce larger rotations.

    - **Scaling guidance**:
        - Normalize features to [0, 1] or [-1, 1] for consistent behavior
        - If using [0, 2π], be aware that features near π (~3.14) will have
          reduced two-qubit interactions
        - The `evolution_time` parameter scales all rotation angles uniformly

    **Complexity Analysis:**

    - **Time Complexity (circuit construction)**:
        - Full entanglement: O(reps × n²) where n is number of qubits
        - Linear/circular entanglement: O(reps × n)

    - **Circuit Depth**:
        - Full entanglement: O(reps × n) layers
        - Linear/circular entanglement: O(reps) layers

    - **Gate Count**:
        - Full entanglement: O(reps × n²) gates
        - Linear/circular entanglement: O(reps × n) gates

    **Scalability Guidelines:**

    For large systems (n > 20 qubits), consider:
    1. Use `entanglement='linear'` for O(n) scaling instead of O(n²)
    2. Use `max_pairs` parameter to limit two-qubit interactions
    3. Reduce `reps` if depth is a concern
    4. Consider using hardware-efficient entanglement patterns

    **Thread Safety:**

    Instances of HamiltonianEncoding are **not thread-safe** for concurrent
    circuit generation with the same input data. Each call to `get_circuit()`
    or `get_circuits()` may modify internal warning flags. For thread-safe
    usage:
    - Create separate encoding instances per thread
    - Or use thread-local storage for encoding objects
    - The class is safe for concurrent reads of properties (n_qubits, depth, etc.)

    **Serialization:**

    This class supports pickling for distributed computing and caching:
    ```python
    import pickle
    enc = HamiltonianEncoding(n_features=4)
    serialized = pickle.dumps(enc)
    enc_restored = pickle.loads(serialized)
    ```
    """

    # Valid Hamiltonian types
    _VALID_HAMILTONIAN_TYPES: frozenset[str] = frozenset(
        {"iqp", "xy", "heisenberg", "pauli_z"}
    )

    # Valid entanglement patterns
    _VALID_ENTANGLEMENT: frozenset[str] = frozenset({"full", "linear", "circular"})

    # Memory-efficient slot-based attribute storage.
    # Only instance attributes specific to this class are listed here.
    # Attributes from BaseEncoding are handled by its own __slots__.
    __slots__ = (
        "hamiltonian_type",
        "evolution_time",
        "reps",
        "entanglement",
        "insert_barriers",
        "max_pairs",
        "include_single_qubit_terms",
        "_entanglement_pairs",
        "_pairs_truncated",
        "_time_step",
        "_cached_depth",
        "_critical_point_warning_issued",
    )

    def __init__(
        self,
        n_features: int,
        hamiltonian_type: Literal["iqp", "xy", "heisenberg", "pauli_z"] = "iqp",
        evolution_time: float = 1.0,
        reps: int = 2,
        entanglement: Literal["full", "linear", "circular"] = "full",
        insert_barriers: bool = True,
        max_pairs: int | None = None,
        include_single_qubit_terms: bool = True,
    ) -> None:
        """Initialize the Hamiltonian encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        hamiltonian_type : {'iqp', 'xy', 'heisenberg', 'pauli_z'}, default='iqp'
            Type of Hamiltonian structure.
        evolution_time : float, default=1.0
            Total evolution time parameter.
        reps : int, default=2
            Number of Trotter steps.
        entanglement : {'full', 'linear', 'circular'}, default='full'
            Entanglement pattern for two-qubit terms.
        insert_barriers : bool, default=True
            Whether to insert barriers between Trotter steps.
            Only effective for the Qiskit backend.
        max_pairs : int or None, default=None
            Maximum number of qubit pairs for two-qubit interactions.
            If None, all pairs from the entanglement pattern are used.
            Use this to limit circuit depth for large systems with full
            entanglement. For example, with n=100 qubits and full entanglement,
            setting max_pairs=200 limits interactions to the first 200 pairs
            instead of all 4950 pairs.
        include_single_qubit_terms : bool, default=True
            Whether to include single-qubit Z rotations in the Hamiltonian.
            For IQP and Pauli-Z types, this is always True regardless of
            this setting. For XY and Heisenberg types:
            - True: Adds Σᵢ xᵢZᵢ single-qubit terms (more expressive)
            - False: Only two-qubit terms (original behavior)
            This allows XY/Heisenberg encodings to capture both individual
            feature magnitudes and pairwise correlations.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        TypeError
            If parameter types are incorrect.
        """
        # =================================================================
        # PARAMETER VALIDATION
        # =================================================================

        # Validate hamiltonian_type
        if not isinstance(hamiltonian_type, str):
            raise TypeError(
                f"hamiltonian_type must be a string, got {type(hamiltonian_type).__name__}"
            )
        hamiltonian_type_lower = hamiltonian_type.lower()
        if hamiltonian_type_lower not in self._VALID_HAMILTONIAN_TYPES:
            raise ValueError(
                f"hamiltonian_type must be one of {sorted(self._VALID_HAMILTONIAN_TYPES)}, "
                f"got '{hamiltonian_type}'"
            )

        # Validate evolution_time
        if not isinstance(evolution_time, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"evolution_time must be a number, got {type(evolution_time).__name__}"
            )
        evolution_time = float(evolution_time)
        if not np.isfinite(evolution_time):
            raise ValueError(f"evolution_time must be finite, got {evolution_time}")

        # Validate reps
        if not isinstance(reps, (int, np.integer)):
            raise TypeError(f"reps must be an integer, got {type(reps).__name__}")
        reps = int(reps)
        if reps < 1:
            raise ValueError(f"reps must be at least 1, got {reps}")

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

        # Validate insert_barriers
        if not isinstance(insert_barriers, bool):
            raise TypeError(
                f"insert_barriers must be a bool, got {type(insert_barriers).__name__}"
            )

        # Validate max_pairs
        if max_pairs is not None:
            if not isinstance(max_pairs, (int, np.integer)):
                raise TypeError(
                    f"max_pairs must be an integer or None, got {type(max_pairs).__name__}"
                )
            max_pairs = int(max_pairs)
            if max_pairs < 0:
                raise ValueError(f"max_pairs must be non-negative, got {max_pairs}")

        # Validate include_single_qubit_terms
        if not isinstance(include_single_qubit_terms, bool):
            raise TypeError(
                f"include_single_qubit_terms must be a bool, "
                f"got {type(include_single_qubit_terms).__name__}"
            )

        # =================================================================
        # INITIALIZATION
        # =================================================================

        # Call parent constructor
        super().__init__(
            n_features,
            hamiltonian_type=hamiltonian_type_lower,
            evolution_time=evolution_time,
            reps=reps,
            entanglement=entanglement_lower,
            insert_barriers=insert_barriers,
            max_pairs=max_pairs,
            include_single_qubit_terms=include_single_qubit_terms,
        )

        # Store validated parameters
        self.hamiltonian_type: str = hamiltonian_type_lower
        self.evolution_time: float = evolution_time
        self.reps: int = reps
        self.entanglement: str = entanglement_lower
        self.insert_barriers: bool = insert_barriers
        self.max_pairs: int | None = max_pairs
        self.include_single_qubit_terms: bool = include_single_qubit_terms

        # Precompute entanglement pairs for efficiency
        all_pairs = self._compute_entanglement_pairs()

        # Apply max_pairs limit if specified
        if max_pairs is not None and len(all_pairs) > max_pairs:
            self._entanglement_pairs: list[tuple[int, int]] = all_pairs[:max_pairs]
            self._pairs_truncated: bool = True
            import warnings

            warnings.warn(
                f"Limiting entanglement to {max_pairs} pairs (full pattern has "
                f"{len(all_pairs)} pairs). This reduces circuit depth but may "
                f"affect encoding expressiveness. Pairs are selected in order, "
                f"prioritizing lower-indexed qubits.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self._entanglement_pairs = all_pairs
            self._pairs_truncated: bool = False

        # Compute time step for Trotterization
        self._time_step: float = evolution_time / reps

        # Initialize depth cache (computed lazily for efficiency)
        self._cached_depth: int | None = None

        # Initialize warning flag for critical point detection
        # This ensures the warning is only issued once per instance
        self._critical_point_warning_issued: bool = False

        # Performance warning for large systems
        if (
            entanglement_lower == "full"
            and n_features > 20
            and not self._pairs_truncated
        ):
            total_possible_pairs = n_features * (n_features - 1) // 2
            import warnings

            warnings.warn(
                f"SCALABILITY WARNING: Full entanglement with {n_features} qubits creates "
                f"{total_possible_pairs} qubit pairs (O(n²) scaling).\n"
                f"  Current configuration: {len(self._entanglement_pairs)} pairs × {reps} reps = "
                f"{len(self._entanglement_pairs) * reps} two-qubit interaction layers.\n"
                f"\n"
                f"  RECOMMENDED MITIGATIONS:\n"
                f"  1. Use entanglement='linear' for O(n) scaling ({n_features - 1} pairs)\n"
                f"  2. Use entanglement='circular' for O(n) scaling ({n_features} pairs)\n"
                f"  3. Set max_pairs={min(100, total_possible_pairs // 4)} to limit interactions\n"
                f"  4. Reduce reps if depth is acceptable with fewer Trotter steps\n"
                f"\n"
                f"  Example: HamiltonianEncoding(n_features={n_features}, entanglement='linear')\n"
                f"           HamiltonianEncoding(n_features={n_features}, max_pairs=100)",
                UserWarning,
                stacklevel=2,
            )

        # Numerical precision warning
        if abs(evolution_time) > 100 and reps < 10:
            import warnings

            warnings.warn(
                f"Large evolution_time ({evolution_time}) with few Trotter steps ({reps}) "
                f"may result in poor Trotterization accuracy. Consider increasing reps.",
                UserWarning,
                stacklevel=2,
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
        """Compute exact circuit depth of the encoding.

        The depth is computed exactly based on:
        - Initial Hadamard layer (depth 1)
        - Hamiltonian type determining gate structure
        - Entanglement pattern affecting parallelization opportunities
        - Number of Trotter repetitions

        Gate Parallelization:
            Two-qubit gates on disjoint qubit pairs can execute in parallel.
            - Linear entanglement: Pairs can be grouped into 2 parallel layers
            - Circular entanglement: Pairs can be grouped into 2-3 parallel layers
            - Full entanglement: Uses edge coloring (requires n-1 or n colors)

        Depth per Gate Type:
            - ZZ interaction: 3 layers (CNOT, RZ, CNOT)
            - XX interaction: 5 layers (H⊗H, CNOT, RZ, CNOT, H⊗H)
            - YY interaction: 7 layers (Sdg⊗Sdg, H⊗H, CNOT, RZ, CNOT, H⊗H, S⊗S)

        Returns
        -------
        int
            Exact circuit depth.

        Notes
        -----
        The depth is cached after first computation since it only depends on
        static configuration parameters, not input data.

        See Also
        --------
        _compute_exact_depth : Internal method that performs the calculation.
        _compute_parallel_layers : Computes parallelization groups for pairs.
        """
        # Use cached value if available
        if hasattr(self, "_cached_depth") and self._cached_depth is not None:
            return self._cached_depth

        # Compute and cache the exact depth
        self._cached_depth: int = self._compute_exact_depth()
        return self._cached_depth

    def _compute_exact_depth(self) -> int:
        """Compute exact circuit depth based on actual circuit structure.

        This method performs a detailed analysis of the circuit structure,
        accounting for gate parallelization opportunities based on the
        entanglement pattern.

        Returns
        -------
        int
            Exact circuit depth.

        Notes
        -----
        **Depth Calculation Breakdown:**

        1. Initial Hadamard layer: 1 (all qubits in parallel)

        2. Per Trotter step for IQP/Pauli-Z:
           - Single-qubit RZ: 1 layer (all parallel)
           - ZZ interactions: 3 × (number of parallel groups)

        3. Per Trotter step for XY:
           - XX interactions: 5 × (number of parallel groups)
           - YY interactions: 7 × (number of parallel groups)

        4. Per Trotter step for Heisenberg:
           - XX: 5 × (number of parallel groups)
           - YY: 7 × (number of parallel groups)
           - ZZ: 3 × (number of parallel groups)

        **Parallel Group Calculation:**

        For linear entanglement with pairs (0,1), (1,2), (2,3), ...:
        - Odd-indexed pairs {(0,1), (2,3), ...} are disjoint
        - Even-indexed pairs {(1,2), (3,4), ...} are disjoint
        - Result: 2 parallel groups

        For circular entanglement (adds (n-1, 0) pair):
        - If n is even: 2 groups (same as linear)
        - If n is odd: 3 groups (wrap-around conflicts)

        For full entanglement (all pairs):
        - Uses edge coloring of complete graph K_n
        - Chromatic index = n-1 (even n) or n (odd n)
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        # Safety check: handle edge cases
        if n < 1:
            return 0

        # Initial Hadamard layer: all qubits in parallel = depth 1
        total_depth = 1

        # If no pairs, only single-qubit operations per rep
        if n_pairs == 0:
            if self.hamiltonian_type in ["iqp", "pauli_z"]:
                # Only single-qubit RZ gates (1 layer per rep, all parallel)
                total_depth += self.reps * 1
            elif (
                self.hamiltonian_type in ["xy", "heisenberg"]
                and self.include_single_qubit_terms
            ):
                # Single-qubit RZ gates when include_single_qubit_terms is enabled
                total_depth += self.reps * 1
            return total_depth

        # Compute number of parallel layers needed for two-qubit gates
        parallel_groups = self._compute_parallel_layers()

        # Safety check: ensure we have at least 1 group if we have pairs
        if parallel_groups < 1 and n_pairs > 0:
            parallel_groups = n_pairs  # Fallback to sequential (worst case)

        # Depth constants for each interaction type
        ZZ_DEPTH = 3  # CNOT, RZ, CNOT
        XX_DEPTH = 5  # H⊗H, CNOT, RZ, CNOT, H⊗H
        YY_DEPTH = 7  # Sdg⊗Sdg, H⊗H, CNOT, RZ, CNOT, H⊗H, S⊗S

        for _ in range(self.reps):
            if self.hamiltonian_type in ["iqp", "pauli_z"]:
                # Single-qubit RZ layer: all qubits in parallel
                total_depth += 1
                # ZZ interactions: parallelized based on entanglement pattern
                total_depth += ZZ_DEPTH * parallel_groups

            elif self.hamiltonian_type == "xy":
                # Optional single-qubit RZ layer
                if self.include_single_qubit_terms:
                    total_depth += 1
                # XX interactions followed by YY interactions
                # Each type is parallelized independently
                total_depth += XX_DEPTH * parallel_groups
                total_depth += YY_DEPTH * parallel_groups

            elif self.hamiltonian_type == "heisenberg":
                # Optional single-qubit RZ layer
                if self.include_single_qubit_terms:
                    total_depth += 1
                # XX, YY, and ZZ interactions (all executed for each pair)
                total_depth += XX_DEPTH * parallel_groups
                total_depth += YY_DEPTH * parallel_groups
                total_depth += ZZ_DEPTH * parallel_groups

        return total_depth

    def _compute_parallel_layers(self) -> int:
        """Compute the number of parallel layers needed for two-qubit gates.

        Two-qubit gates can be parallelized when they act on disjoint qubit
        pairs. This method computes the minimum number of sequential layers
        needed based on the entanglement pattern.

        Returns
        -------
        int
            Number of parallel layers (groups) required.

        Notes
        -----
        **Graph Coloring Approach:**

        The parallelization problem is equivalent to edge coloring:
        - Create a graph where qubits are nodes
        - Entanglement pairs are edges
        - Edges sharing a node cannot be same color (executed together)
        - Chromatic index = minimum colors = minimum parallel layers

        **Results by Entanglement Type:**

        - Linear (path graph P_n): chromatic index = 2
        - Circular (cycle graph C_n): 2 if n even, 3 if n odd
        - Full (complete graph K_n): n-1 if n even, n if n odd

        Raises
        ------
        RuntimeError
            If parallel layer computation fails unexpectedly.
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        # Safety checks
        if n_pairs == 0:
            return 0
        if n < 2:
            return 0

        if self.entanglement == "linear":
            # Linear entanglement: pairs (0,1), (1,2), ..., (n-2, n-1)
            # This forms a path graph, chromatic index = 2
            # Odd pairs (0,1), (2,3), ... are independent
            # Even pairs (1,2), (3,4), ... are independent
            return 2 if n_pairs > 1 else 1

        elif self.entanglement == "circular":
            # Circular entanglement: linear + (n-1, 0) wrap-around
            # This forms a cycle graph C_n
            # Chromatic index = 2 if n even, 3 if n odd
            if n % 2 == 0:
                return 2
            else:
                return 3

        elif self.entanglement == "full":
            # Full entanglement: all pairs (i,j) for i < j
            # This forms a complete graph K_n
            # Edge chromatic index (Vizing's theorem):
            #   - n-1 for even n (Class 1)
            #   - n for odd n (Class 1, but n is odd so n-1 is even, need n)
            # Actually for K_n: chromatic index = n-1 if n even, n if n odd
            if n % 2 == 0:
                return n - 1
            else:
                return n

        else:
            # Unknown entanglement type: use conservative sequential estimate
            # This should not happen due to validation, but provides safety
            import warnings

            warnings.warn(
                f"Unknown entanglement pattern '{self.entanglement}', "
                f"using sequential depth estimate.",
                RuntimeWarning,
                stacklevel=3,
            )
            return n_pairs

    def _compute_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Compute qubit pairs for two-qubit interactions.

        Returns
        -------
        list[tuple[int, int]]
            List of (qubit_i, qubit_j) pairs.
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
            raise ValueError(f"Unknown entanglement: {self.entanglement}")

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs for two-qubit interactions.

        Returns the list of qubit pairs that will have two-qubit interactions
        applied based on the configured entanglement topology. This is useful for:

        - Understanding circuit structure before generation
        - Verifying hardware connectivity requirements
        - Debugging entanglement patterns
        - Computing resource requirements

        Returns
        -------
        list[tuple[int, int]]
            List of (qubit_i, qubit_j) pairs for two-qubit interactions.
            Each tuple (i, j) indicates an interaction gate between qubits i and j.
            Returns a copy to prevent external modification of internal state.

        Examples
        --------
        >>> enc = HamiltonianEncoding(n_features=4, entanglement='full')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        >>> enc_linear = HamiltonianEncoding(n_features=4, entanglement='linear')
        >>> enc_linear.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        >>> enc_circular = HamiltonianEncoding(n_features=4, entanglement='circular')
        >>> enc_circular.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3), (3, 0)]

        Notes
        -----
        The number of pairs depends on the entanglement topology:

        - **full**: n(n-1)/2 pairs (all unique combinations)
        - **linear**: n-1 pairs (nearest neighbors only)
        - **circular**: n pairs for n>2 (nearest neighbors + wrap-around),
          or n-1 pairs for n≤2 (no wrap-around needed)

        If ``max_pairs`` was set during initialization, the returned list may be
        truncated to that number of pairs.

        See Also
        --------
        gate_count_breakdown : Get detailed gate counts by type.
        resource_summary : Get comprehensive resource analysis.
        properties : Access all computed circuit properties.
        """
        # Return a copy to prevent external modification of internal state
        return list(self._entanglement_pairs)

    def _compute_rotation_angle(
        self,
        x: NDArray[np.floating[Any]],
        idx_i: int,
        idx_j: int | None = None,
    ) -> float:
        """Compute rotation angle for Hamiltonian term.

        For single-qubit terms: angle = time_step × xᵢ
        For two-qubit terms: angle = time_step × (π - xᵢ)(π - xⱼ)

        Parameters
        ----------
        x : NDArray
            Input feature vector. For optimal results, features should be
            scaled to [0, 1] or [-1, 1]. See Notes for details.
        idx_i : int
            First feature index.
        idx_j : int or None
            Second feature index (for two-qubit terms).

        Returns
        -------
        float
            Rotation angle in radians.

        Notes
        -----
        **Single-Qubit Terms:**

        The rotation angle is computed as:
            angle = time_step × xᵢ

        This scales linearly with the feature value. Larger features produce
        larger rotations.

        **Two-Qubit Terms:**

        The rotation angle uses the quantum kernel feature map:
            angle = time_step × (π - xᵢ)(π - xⱼ)

        This formula has a **critical point at x = π** where the interaction
        becomes zero. This is intentional for the standard quantum kernel,
        but users should be aware of the implications:

        - Features equal to π (~3.14159) produce zero two-qubit interaction
        - Features in [0, 1] produce angles in [0, ~9.87 × time_step]
        - Features in [-1, 1] produce angles in [0, ~17.2 × time_step]
        - Features in [0, 2π] include the critical point

        **Recommended Input Scaling:**

        For consistent and predictable behavior, scale your features to:
        - [0, 1]: Safe range, good for normalized data
        - [-1, 1]: Safe range, good for standardized data (mean=0, std=1)

        Avoid scaling to [0, 2π] unless you specifically want features near
        π to have reduced interactions.

        Warnings
        --------
        If feature values are detected near π (within ±0.1), a warning is
        issued on first occurrence to alert users to potential issues with
        two-qubit interactions.

        See Also
        --------
        _validate_input : Validates input array shape and type.

        References
        ----------
        .. [1] Havlíček, V., et al. (2019). "Supervised learning with
               quantum-enhanced feature spaces." Nature, 567(7747), 209-212.
        """
        # Extract feature value with type safety
        xi = float(x[idx_i])

        if idx_j is None:
            # Single-qubit term: linear scaling
            return self._time_step * xi
        else:
            # Two-qubit term: quantum kernel feature map (π - xᵢ)(π - xⱼ)
            xj = float(x[idx_j])

            # Check for values near the critical point π
            # Only warn once per instance to avoid log spam
            if not self._critical_point_warning_issued and (
                abs(xi - np.pi) < _CRITICAL_POINT_TOLERANCE
                or abs(xj - np.pi) < _CRITICAL_POINT_TOLERANCE
            ):
                import warnings

                warnings.warn(
                    f"Feature value(s) detected near π (critical point). "
                    f"Values: x[{idx_i}]={xi:.4f}, x[{idx_j}]={xj:.4f}. "
                    f"Two-qubit interactions will be minimal for features "
                    f"near π (~3.14159). Consider scaling features to [0, 1] "
                    f"or [-1, 1] for stronger two-qubit correlations. "
                    f"See class docstring 'Input Scaling' section for details.",
                    UserWarning,
                    stacklevel=4,
                )
                self._critical_point_warning_issued = True

            # Compute the two-qubit angle using quantum kernel formula
            return self._time_step * (np.pi - xi) * (np.pi - xj)

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
        >>> enc = HamiltonianEncoding(n_features=4, hamiltonian_type='iqp')
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

        Encoding-specific seam called by ``get_circuit`` and the inherited
        ``get_circuits`` template method.

        Accepts a 1D array (the normal case) or a 2D single-sample
        ``(1, n_features)`` array for robustness when called directly.

        Notes
        -----
        Thread-safe for the core circuit generation logic. May set the
        ``_critical_point_warning_issued`` flag, which is benign for
        correctness (worst case: a warning is issued multiple times under
        concurrent invocation).
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
                f"Unknown backend '{backend}'. "
                f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
            )

    def iter_circuits(
        self,
        X: ArrayLike,
        backend: BackendType = "pennylane",
    ):
        """Generate quantum circuits lazily using a generator.

        This method is a memory-efficient alternative to `get_circuits()` for
        processing large batches of data. Instead of creating all circuits in
        memory at once, it yields circuits one at a time, allowing processing
        of arbitrarily large datasets with constant memory overhead.

        Parameters
        ----------
        X : array-like
            Input features of shape (n_samples, n_features) or (n_features,).
        backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
            Target quantum computing framework.

        Yields
        ------
        CircuitType
            Circuit for each sample, yielded one at a time.

        Examples
        --------
        Process large dataset without loading all circuits into memory:

        >>> enc = HamiltonianEncoding(n_features=4)
        >>> X = np.random.randn(100000, 4)  # Large dataset
        >>> for i, circuit in enumerate(enc.iter_circuits(X, backend='qiskit')):
        ...     # Process circuit immediately
        ...     result = execute_circuit(circuit)
        ...     if i % 10000 == 0:
        ...         print(f"Processed {i} circuits")

        Use with itertools for batched processing:

        >>> from itertools import islice
        >>> enc = HamiltonianEncoding(n_features=4)
        >>> X = np.random.randn(1000, 4)
        >>> # Get first 10 circuits
        >>> first_10 = list(islice(enc.iter_circuits(X), 10))

        Combine with tqdm for progress tracking:

        >>> from tqdm import tqdm
        >>> enc = HamiltonianEncoding(n_features=4)
        >>> X = np.random.randn(10000, 4)
        >>> for circuit in tqdm(enc.iter_circuits(X), total=len(X)):
        ...     process(circuit)

        Notes
        -----
        **Memory Efficiency:**

        - `get_circuits()`: O(n_samples) memory - stores all circuits
        - `iter_circuits()`: O(1) memory - yields circuits one at a time

        For a batch of 100,000 samples with 10-qubit circuits:
        - `get_circuits()`: May use several GB of memory
        - `iter_circuits()`: Uses only memory for one circuit at a time

        **When to Use:**

        - Datasets larger than available memory
        - Streaming data processing
        - When circuits are processed and discarded immediately
        - Distributed computing pipelines

        **Thread Safety:**

        The generator is not thread-safe. For parallel processing, either:
        - Create separate encoding instances per thread/process
        - Materialize slices of the generator for each worker

        See Also
        --------
        get_circuits : List-based alternative for smaller batches.
        """
        # Validate input once
        X_validated = self._validate_input(X)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)

        # Yield circuits one at a time
        for x in X_validated:
            yield self.get_circuit(x, backend)

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

        # Check backend version compatibility
        _check_backend_version("pennylane", qml)

        # Capture values for closure
        n_qubits = self.n_qubits
        reps = self.reps
        hamiltonian_type = self.hamiltonian_type
        pairs = self._entanglement_pairs
        include_single = self.include_single_qubit_terms

        # Precompute all rotation angles
        single_angles = [self._compute_rotation_angle(x, i) for i in range(n_qubits)]
        two_qubit_angles = [self._compute_rotation_angle(x, i, j) for i, j in pairs]

        def circuit() -> None:
            """Apply the Hamiltonian encoding via time evolution."""
            # Initial Hadamard layer to create superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            for _ in range(reps):
                # Apply Hamiltonian terms based on type
                if hamiltonian_type == "iqp":
                    # Single-qubit Z rotations (always included for IQP)
                    for i in range(n_qubits):
                        if _is_significant_angle(single_angles[i]):
                            qml.RZ(2 * single_angles[i], wires=i)

                    # ZZ interactions
                    for pair_idx, (i, j) in enumerate(pairs):
                        angle = two_qubit_angles[pair_idx]
                        if _is_significant_angle(angle):
                            _apply_zz_interaction_pennylane(qml, angle, i, j)

                elif hamiltonian_type == "pauli_z":
                    # Single-qubit Z rotations (always included for Pauli-Z)
                    for i in range(n_qubits):
                        if _is_significant_angle(single_angles[i]):
                            qml.RZ(2 * single_angles[i], wires=i)

                    for pair_idx, (i, j) in enumerate(pairs):
                        angle = two_qubit_angles[pair_idx]
                        if _is_significant_angle(angle):
                            _apply_zz_interaction_pennylane(qml, angle, i, j)

                elif hamiltonian_type == "xy":
                    # Optional single-qubit Z rotations for XY model
                    # This adds Σᵢ xᵢZᵢ terms to capture individual feature magnitudes
                    if include_single:
                        for i in range(n_qubits):
                            if _is_significant_angle(single_angles[i]):
                                qml.RZ(2 * single_angles[i], wires=i)

                    # XX and YY interactions
                    for pair_idx, (i, j) in enumerate(pairs):
                        angle = two_qubit_angles[pair_idx]
                        if _is_significant_angle(angle):
                            # XX interaction
                            _apply_xx_interaction_pennylane(qml, angle, i, j)
                            # YY interaction
                            _apply_yy_interaction_pennylane(qml, angle, i, j)

                elif hamiltonian_type == "heisenberg":
                    # Optional single-qubit Z rotations for Heisenberg model
                    # This adds external field terms Σᵢ hᵢZᵢ (typical in Heisenberg model)
                    if include_single:
                        for i in range(n_qubits):
                            if _is_significant_angle(single_angles[i]):
                                qml.RZ(2 * single_angles[i], wires=i)

                    # XX, YY, and ZZ interactions
                    for pair_idx, (i, j) in enumerate(pairs):
                        angle = two_qubit_angles[pair_idx]
                        if _is_significant_angle(angle):
                            _apply_xx_interaction_pennylane(qml, angle, i, j)
                            _apply_yy_interaction_pennylane(qml, angle, i, j)
                            _apply_zz_interaction_pennylane(qml, angle, i, j)

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
            import qiskit
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError(
                "Qiskit is required for the 'qiskit' backend. "
                "Install it with: pip install qiskit"
            ) from e

        # Check backend version compatibility
        _check_backend_version("qiskit", qiskit)

        qc = QuantumCircuit(self.n_qubits, name="HamiltonianEncoding")

        # Initial Hadamard layer to create superposition
        for i in range(self.n_qubits):
            qc.h(i)

        for rep in range(self.reps):
            # Apply Hamiltonian terms based on type
            if self.hamiltonian_type == "iqp":
                # Single-qubit Z rotations (always included for IQP)
                for i in range(self.n_qubits):
                    angle = self._compute_rotation_angle(x, i)
                    if _is_significant_angle(angle):
                        qc.rz(2 * angle, i)

                # ZZ interactions
                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        _apply_zz_interaction_qiskit(qc, angle, i, j)

            elif self.hamiltonian_type == "pauli_z":
                # Single-qubit Z rotations (always included for Pauli-Z)
                for i in range(self.n_qubits):
                    angle = self._compute_rotation_angle(x, i)
                    if _is_significant_angle(angle):
                        qc.rz(2 * angle, i)

                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        _apply_zz_interaction_qiskit(qc, angle, i, j)

            elif self.hamiltonian_type == "xy":
                # Optional single-qubit Z rotations for XY model
                if self.include_single_qubit_terms:
                    for i in range(self.n_qubits):
                        angle = self._compute_rotation_angle(x, i)
                        if _is_significant_angle(angle):
                            qc.rz(2 * angle, i)

                # XX and YY interactions
                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        _apply_xx_interaction_qiskit(qc, angle, i, j)
                        _apply_yy_interaction_qiskit(qc, angle, i, j)

            elif self.hamiltonian_type == "heisenberg":
                # Optional single-qubit Z rotations for Heisenberg model
                if self.include_single_qubit_terms:
                    for i in range(self.n_qubits):
                        angle = self._compute_rotation_angle(x, i)
                        if _is_significant_angle(angle):
                            qc.rz(2 * angle, i)

                # XX, YY, and ZZ interactions
                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        _apply_xx_interaction_qiskit(qc, angle, i, j)
                        _apply_yy_interaction_qiskit(qc, angle, i, j)
                        _apply_zz_interaction_qiskit(qc, angle, i, j)

            # Add barrier between Trotter steps
            if self.insert_barriers and rep < self.reps - 1:
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

        # Check backend version compatibility
        _check_backend_version("cirq", cirq)

        qubits = cirq.LineQubit.range(self.n_qubits)
        moments: list[Any] = []

        # Initial Hadamard layer to create superposition
        moments.append(cirq.Moment([cirq.H(qubits[i]) for i in range(self.n_qubits)]))

        for _ in range(self.reps):
            # Apply Hamiltonian terms based on type
            if self.hamiltonian_type == "iqp":
                # Single-qubit Z rotations (always included for IQP)
                z_gates = []
                for i in range(self.n_qubits):
                    angle = self._compute_rotation_angle(x, i)
                    if _is_significant_angle(angle):
                        z_gates.append(cirq.rz(2 * angle)(qubits[i]))
                if z_gates:
                    moments.append(cirq.Moment(z_gates))

                # ZZ interactions
                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        zz_moments = _apply_zz_interaction_cirq(
                            cirq, angle, qubits[i], qubits[j]
                        )
                        moments.extend(zz_moments)

            elif self.hamiltonian_type == "pauli_z":
                # Single-qubit Z rotations (always included for Pauli-Z)
                z_gates = []
                for i in range(self.n_qubits):
                    angle = self._compute_rotation_angle(x, i)
                    if _is_significant_angle(angle):
                        z_gates.append(cirq.rz(2 * angle)(qubits[i]))
                if z_gates:
                    moments.append(cirq.Moment(z_gates))

                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        zz_moments = _apply_zz_interaction_cirq(
                            cirq, angle, qubits[i], qubits[j]
                        )
                        moments.extend(zz_moments)

            elif self.hamiltonian_type == "xy":
                # Optional single-qubit Z rotations for XY model
                if self.include_single_qubit_terms:
                    z_gates = []
                    for i in range(self.n_qubits):
                        angle = self._compute_rotation_angle(x, i)
                        if _is_significant_angle(angle):
                            z_gates.append(cirq.rz(2 * angle)(qubits[i]))
                    if z_gates:
                        moments.append(cirq.Moment(z_gates))

                # XX and YY interactions
                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        xx_moments = _apply_xx_interaction_cirq(
                            cirq, angle, qubits[i], qubits[j]
                        )
                        moments.extend(xx_moments)
                        yy_moments = _apply_yy_interaction_cirq(
                            cirq, angle, qubits[i], qubits[j]
                        )
                        moments.extend(yy_moments)

            elif self.hamiltonian_type == "heisenberg":
                # Optional single-qubit Z rotations for Heisenberg model
                if self.include_single_qubit_terms:
                    z_gates = []
                    for i in range(self.n_qubits):
                        angle = self._compute_rotation_angle(x, i)
                        if _is_significant_angle(angle):
                            z_gates.append(cirq.rz(2 * angle)(qubits[i]))
                    if z_gates:
                        moments.append(cirq.Moment(z_gates))

                # XX, YY, and ZZ interactions
                for i, j in self._entanglement_pairs:
                    angle = self._compute_rotation_angle(x, i, j)
                    if _is_significant_angle(angle):
                        xx_moments = _apply_xx_interaction_cirq(
                            cirq, angle, qubits[i], qubits[j]
                        )
                        moments.extend(xx_moments)
                        yy_moments = _apply_yy_interaction_cirq(
                            cirq, angle, qubits[i], qubits[j]
                        )
                        moments.extend(yy_moments)
                        zz_moments = _apply_zz_interaction_cirq(
                            cirq, angle, qubits[i], qubits[j]
                        )
                        moments.extend(zz_moments)

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

        # Gate count estimation based on Hamiltonian type
        # Note: These are UPPER BOUND estimates. Actual gate count may be lower
        # if some rotation angles are effectively zero (within tolerance).
        # Use count_gates(x) for exact counts with specific input data.
        if self.hamiltonian_type in ["iqp", "pauli_z"]:
            # Per Trotter step: n single-qubit RZ + n_pairs ZZ terms (2 CNOTs + 1 RZ each)
            single_qubit_gates_per_rep = n + n_pairs  # RZ gates
            two_qubit_gates_per_rep = 2 * n_pairs  # CNOTs

        elif self.hamiltonian_type == "xy":
            # Per Trotter step: n_pairs XX terms + n_pairs YY terms
            # Each term: 2 H gates + 2 CNOTs + 1 RZ or similar
            # Plus optional single-qubit RZ gates if include_single_qubit_terms
            single_qubit_gates_per_rep = 4 * n_pairs  # Basis change gates
            if self.include_single_qubit_terms:
                single_qubit_gates_per_rep += n  # Single-qubit RZ gates
            two_qubit_gates_per_rep = 4 * n_pairs  # CNOTs for XX and YY

        elif self.hamiltonian_type == "heisenberg":
            # XX + YY + ZZ terms
            # Plus optional single-qubit RZ gates if include_single_qubit_terms
            single_qubit_gates_per_rep = 4 * n_pairs  # Basis change gates
            if self.include_single_qubit_terms:
                single_qubit_gates_per_rep += n  # Single-qubit RZ gates
            two_qubit_gates_per_rep = 6 * n_pairs  # CNOTs for all three types

        else:
            # Default estimate
            single_qubit_gates_per_rep = n + 2 * n_pairs
            two_qubit_gates_per_rep = 2 * n_pairs

        # Total counts
        # Include initial Hadamard layer (H^⊗n) for |+⟩^⊗n state preparation
        single_qubit_gates = n + self.reps * single_qubit_gates_per_rep
        two_qubit_gates = self.reps * two_qubit_gates_per_rep
        total_gates = single_qubit_gates + two_qubit_gates

        # Determine if entangling
        is_entangling = n_pairs > 0

        # Determine simulability
        if self.hamiltonian_type in ["iqp", "pauli_z"]:
            # IQP circuits are in Clifford hierarchy, hard to simulate
            simulability: Literal[
                "simulable", "conditionally_simulable", "not_simulable"
            ] = "not_simulable"
        else:
            # General entangling circuits
            simulability = "not_simulable"

        # Trainability estimate: Set to None as we don't have a theoretically
        # grounded formula for this encoding. Trainability depends on many factors
        # including the specific optimization landscape, loss function, and
        # parameter initialization strategy. Users should empirically evaluate
        # trainability for their specific use case.
        #
        # General guidance:
        # - Deeper circuits (higher reps) may experience barren plateaus
        # - Full entanglement creates more complex landscapes than linear
        # - IQP-style circuits have been shown to have favorable trainability
        #   properties in certain regimes (see Cerezo et al., 2021)
        trainability_estimate: float | None = None

        return EncodingProperties(
            n_qubits=n,
            depth=self.depth,
            gate_count=total_gates,
            single_qubit_gates=single_qubit_gates,
            two_qubit_gates=two_qubit_gates,
            parameter_count=0,  # No trainable parameters, only data-dependent
            is_entangling=is_entangling,
            simulability=simulability,
            trainability_estimate=trainability_estimate,
            notes=(
                f"Hamiltonian type: {self.hamiltonian_type}, "
                f"Evolution time: {self.evolution_time}, "
                f"Trotter steps: {self.reps}, "
                f"Entanglement: {self.entanglement} ({n_pairs} pairs), "
                f"Single-qubit terms: {self.include_single_qubit_terms}. "
                f"Gate counts are UPPER BOUNDS - actual count depends on input data "
                f"(gates with zero-angle rotations are skipped). "
                f"Use count_gates(x) for exact counts. "
                f"Trainability not estimated - depends on optimization context."
            ),
        )

    def __repr__(self) -> str:
        """Return detailed string representation."""
        parts = [
            f"n_features={self.n_features}",
            f"hamiltonian_type='{self.hamiltonian_type}'",
            f"evolution_time={self.evolution_time}",
            f"reps={self.reps}",
            f"entanglement='{self.entanglement}'",
            f"insert_barriers={self.insert_barriers}",
        ]
        # Only include optional params if non-default
        if self.max_pairs is not None:
            parts.append(f"max_pairs={self.max_pairs}")
        if not self.include_single_qubit_terms:
            parts.append(
                f"include_single_qubit_terms={self.include_single_qubit_terms}"
            )
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another HamiltonianEncoding instance.

        Two HamiltonianEncoding instances are considered equal if all their
        configuration parameters are identical. This comparison is based on
        the parameters that affect circuit generation:

        - n_features
        - hamiltonian_type
        - evolution_time
        - reps
        - entanglement
        - insert_barriers
        - max_pairs
        - include_single_qubit_terms

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            True if both encodings have identical configuration, False otherwise.

        Examples
        --------
        >>> enc1 = HamiltonianEncoding(n_features=4, hamiltonian_type='iqp')
        >>> enc2 = HamiltonianEncoding(n_features=4, hamiltonian_type='iqp')
        >>> enc1 == enc2
        True

        >>> enc3 = HamiltonianEncoding(n_features=4, hamiltonian_type='xy')
        >>> enc1 == enc3
        False

        >>> enc1 == "not an encoding"
        False

        Notes
        -----
        This method enables:

        - Using encodings in sets
        - Using encodings as dictionary keys (with __hash__)
        - Comparing configurations for equivalence

        The comparison is strict: all parameters must match exactly.
        Floating-point comparison for ``evolution_time`` uses exact equality,
        so encodings created with slightly different float values will not
        be considered equal.

        See Also
        --------
        __hash__ : Hash computation for dictionary/set usage.
        """
        if not isinstance(other, HamiltonianEncoding):
            return NotImplemented

        return (
            self.n_features == other.n_features
            and self.hamiltonian_type == other.hamiltonian_type
            and self.evolution_time == other.evolution_time
            and self.reps == other.reps
            and self.entanglement == other.entanglement
            and self.insert_barriers == other.insert_barriers
            and self.max_pairs == other.max_pairs
            and self.include_single_qubit_terms == other.include_single_qubit_terms
        )

    def __hash__(self) -> int:
        """Compute hash value for this encoding.

        The hash is based on all configuration parameters that affect
        circuit generation, allowing HamiltonianEncoding instances to be
        used as dictionary keys and in sets.

        Returns
        -------
        int
            Hash value based on encoding configuration.

        Examples
        --------
        >>> enc1 = HamiltonianEncoding(n_features=4)
        >>> enc2 = HamiltonianEncoding(n_features=4)
        >>> hash(enc1) == hash(enc2)
        True

        >>> enc_set = {enc1, enc2}
        >>> len(enc_set)
        1

        >>> enc_dict = {enc1: "first encoding"}
        >>> enc_dict[enc2]
        'first encoding'

        Notes
        -----
        The hash is consistent with __eq__: encodings that compare equal
        will have the same hash value. This allows correct behavior when
        used in dictionaries and sets.

        The hash includes:
        - n_features
        - hamiltonian_type
        - evolution_time
        - reps
        - entanglement
        - insert_barriers
        - max_pairs
        - include_single_qubit_terms

        Warning: If evolution_time was set to a float that is not exactly
        representable, the hash may differ between identical-looking values.
        For reliable hashing, use exact float values or round to a fixed
        number of decimal places before creating the encoding.

        See Also
        --------
        __eq__ : Equality comparison.
        """
        return hash(
            (
                self.n_features,
                self.hamiltonian_type,
                self.evolution_time,
                self.reps,
                self.entanglement,
                self.insert_barriers,
                self.max_pairs,
                self.include_single_qubit_terms,
            )
        )

    def count_gates(
        self,
        x: ArrayLike,
    ) -> dict[str, int]:
        """Count the actual gates that would be applied for specific input data.

        Unlike the `properties.gate_count` which provides an upper bound estimate,
        this method computes the exact gate count for the given input, accounting
        for gates that are skipped when rotation angles are effectively zero.

        Parameters
        ----------
        x : array-like
            Input features of shape (n_features,).

        Returns
        -------
        dict[str, int]
            Dictionary with gate counts:
            - 'hadamard': Number of Hadamard gates (initial layer)
            - 'rz': Number of RZ rotation gates
            - 'cnot': Number of CNOT gates
            - 'single_qubit': Total single-qubit gates (H + RZ)
            - 'two_qubit': Total two-qubit gates (CNOT)
            - 'total': Total gate count

        Examples
        --------
        >>> enc = HamiltonianEncoding(n_features=4, hamiltonian_type='iqp')
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> counts = enc.count_gates(x)
        >>> counts['total']
        42
        >>> counts['cnot']
        24

        Notes
        -----
        This method is useful for:
        - Accurate resource estimation for specific inputs
        - Comparing actual vs. theoretical gate counts
        - Identifying inputs that produce sparse circuits
        - Hardware cost estimation

        The count is independent of backend - all backends generate
        equivalent gate sequences.
        """
        # Validate input
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            if x_validated.shape[0] != 1:
                raise ValueError(
                    f"count_gates expects a single sample, got shape {x_validated.shape}"
                )
            x_validated = x_validated[0]

        n = self.n_qubits
        pairs = self._entanglement_pairs

        # Precompute angles
        single_angles = [self._compute_rotation_angle(x_validated, i) for i in range(n)]
        two_qubit_angles = [
            self._compute_rotation_angle(x_validated, i, j) for i, j in pairs
        ]

        # Initialize counts
        hadamard_count = n  # Initial Hadamard layer (always applied)
        rz_count = 0
        cnot_count = 0

        for _ in range(self.reps):
            if self.hamiltonian_type in ["iqp", "pauli_z"]:
                # Single-qubit RZ gates
                for i in range(n):
                    if _is_significant_angle(single_angles[i]):
                        rz_count += 1

                # ZZ interactions: CNOT-RZ-CNOT
                for pair_idx in range(len(pairs)):
                    if _is_significant_angle(two_qubit_angles[pair_idx]):
                        cnot_count += 2
                        rz_count += 1

            elif self.hamiltonian_type == "xy":
                # Optional single-qubit RZ gates
                if self.include_single_qubit_terms:
                    for i in range(n):
                        if _is_significant_angle(single_angles[i]):
                            rz_count += 1

                # XX: H-H, CNOT, RZ, CNOT, H-H (2H + 1RZ per qubit, 2 CNOT)
                # YY: Sdg-Sdg, H-H, CNOT, RZ, CNOT, H-H, S-S (same structure)
                for pair_idx in range(len(pairs)):
                    if _is_significant_angle(two_qubit_angles[pair_idx]):
                        # XX interaction
                        hadamard_count += 4  # 2 before, 2 after
                        cnot_count += 2
                        rz_count += 1
                        # YY interaction (uses S/Sdg but we count as single-qubit)
                        hadamard_count += 4  # 2 before, 2 after (plus S gates below)
                        cnot_count += 2
                        rz_count += 1

            elif self.hamiltonian_type == "heisenberg":
                # Optional single-qubit RZ gates
                if self.include_single_qubit_terms:
                    for i in range(n):
                        if _is_significant_angle(single_angles[i]):
                            rz_count += 1

                # XX + YY + ZZ interactions
                for pair_idx in range(len(pairs)):
                    if _is_significant_angle(two_qubit_angles[pair_idx]):
                        # XX interaction
                        hadamard_count += 4
                        cnot_count += 2
                        rz_count += 1
                        # YY interaction
                        hadamard_count += 4
                        cnot_count += 2
                        rz_count += 1
                        # ZZ interaction
                        cnot_count += 2
                        rz_count += 1

        # Note: For YY, we also have S and Sdg gates which are single-qubit
        # These are tracked separately for precision
        s_gate_count = 0
        if self.hamiltonian_type in ["xy", "heisenberg"]:
            for pair_idx in range(len(pairs)):
                if _is_significant_angle(two_qubit_angles[pair_idx]):
                    # 2 Sdg before + 2 S after per YY interaction
                    s_gate_count += 4 * self.reps

        single_qubit_total = hadamard_count + rz_count + s_gate_count
        two_qubit_total = cnot_count

        return {
            "hadamard": hadamard_count,
            "rz": rz_count,
            "s_gates": s_gate_count,  # S and Sdg gates
            "cnot": cnot_count,
            "single_qubit": single_qubit_total,
            "two_qubit": two_qubit_total,
            "total": single_qubit_total + two_qubit_total,
        }

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type (upper bound).

        Unlike ``count_gates(x)`` which computes exact counts for specific input
        data, this method returns the maximum possible gate counts assuming all
        rotation angles are significant. This is useful for:

        - Resource estimation before having specific input data
        - Hardware compatibility assessment
        - Comparing different encoding configurations
        - Debugging and verification

        Returns
        -------
        GateCountBreakdown
            TypedDict with gate counts:

            - ``'hadamard'``: Number of Hadamard gates (initial layer + basis changes)
            - ``'rz'``: Number of RZ rotation gates
            - ``'s_gates'``: Number of S and Sdg gates (for YY interactions)
            - ``'cnot'``: Number of CNOT gates (2 per two-qubit interaction)
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Total two-qubit gates (= cnot)
            - ``'total'``: Total gate count

        Examples
        --------
        >>> enc = HamiltonianEncoding(n_features=4, reps=1, entanglement='full')
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['hadamard']
        4
        >>> breakdown['cnot']
        12
        >>> breakdown['total']
        26

        Compare entanglement topologies:

        >>> enc_full = HamiltonianEncoding(n_features=8, entanglement='full')
        >>> enc_linear = HamiltonianEncoding(n_features=8, entanglement='linear')
        >>> enc_full.gate_count_breakdown()['cnot'] > enc_linear.gate_count_breakdown()['cnot']
        True

        See Also
        --------
        count_gates : Get exact counts for specific input data.
        resource_summary : Get comprehensive resource analysis.
        properties : Access all computed circuit properties.
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        # Initial Hadamard layer (always present)
        hadamard_count = n

        # Initialize counts
        rz_count = 0
        cnot_count = 0
        s_gate_count = 0

        # Gate counts based on Hamiltonian type
        if self.hamiltonian_type in ["iqp", "pauli_z"]:
            # Per Trotter step:
            # - n single-qubit RZ gates
            # - n_pairs ZZ interactions (2 CNOTs + 1 RZ each)
            rz_count = self.reps * (n + n_pairs)  # Single-qubit + ZZ RZ gates
            cnot_count = self.reps * 2 * n_pairs  # 2 CNOTs per ZZ

        elif self.hamiltonian_type == "xy":
            # Per Trotter step:
            # - n single-qubit RZ gates (if include_single_qubit_terms)
            # - n_pairs XX interactions (4 H + 2 CNOT + 1 RZ each)
            # - n_pairs YY interactions (4 H + 4 S/Sdg + 2 CNOT + 1 RZ each)
            if self.include_single_qubit_terms:
                rz_count = self.reps * n
            rz_count += self.reps * n_pairs * 2  # 1 RZ per XX, 1 RZ per YY
            hadamard_count += self.reps * n_pairs * 8  # 4 H per XX, 4 H per YY
            cnot_count = self.reps * n_pairs * 4  # 2 CNOTs per XX, 2 per YY
            s_gate_count = self.reps * n_pairs * 4  # 2 Sdg + 2 S per YY

        elif self.hamiltonian_type == "heisenberg":
            # Per Trotter step:
            # - n single-qubit RZ gates (if include_single_qubit_terms)
            # - n_pairs XX interactions
            # - n_pairs YY interactions
            # - n_pairs ZZ interactions
            if self.include_single_qubit_terms:
                rz_count = self.reps * n
            rz_count += self.reps * n_pairs * 3  # 1 RZ per XX, YY, ZZ
            hadamard_count += self.reps * n_pairs * 8  # 4 H per XX, 4 H per YY
            cnot_count = self.reps * n_pairs * 6  # 2 CNOTs per XX, YY, ZZ
            s_gate_count = self.reps * n_pairs * 4  # 2 Sdg + 2 S per YY

        total_single_qubit = hadamard_count + rz_count + s_gate_count
        total_two_qubit = cnot_count

        _logger.debug(
            "Gate breakdown: H=%d, RZ=%d, S=%d, CNOT=%d, total=%d",
            hadamard_count,
            rz_count,
            s_gate_count,
            cnot_count,
            total_single_qubit + total_two_qubit,
        )

        return GateCountBreakdown(
            hadamard=hadamard_count,
            rz=rz_count,
            s_gates=s_gate_count,
            cnot=cnot_count,
            total_single_qubit=total_single_qubit,
            total_two_qubit=total_two_qubit,
            total=total_single_qubit + total_two_qubit,
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
            - ``reps``: Number of Trotter steps (repetitions)
            - ``hamiltonian_type``: Type of Hamiltonian ('iqp', 'xy', etc.)
            - ``evolution_time``: Total evolution time parameter
            - ``entanglement``: Entanglement topology
            - ``entanglement_pairs``: List of qubit pairs with interactions
            - ``n_entanglement_pairs``: Number of interaction pairs
            - ``gate_counts``: Detailed breakdown from gate_count_breakdown()
            - ``is_entangling``: Whether circuit creates entanglement
            - ``simulability``: Simulation difficulty classification
            - ``hardware_requirements``: Dict with connectivity and gate requirements

        Examples
        --------
        Get complete resource analysis:

        >>> enc = HamiltonianEncoding(n_features=4, reps=2, entanglement='full')
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['n_entanglement_pairs']
        6
        >>> summary['gate_counts']['cnot']
        24

        Compare different Hamiltonian types:

        >>> enc_iqp = HamiltonianEncoding(n_features=4, hamiltonian_type='iqp')
        >>> enc_heisenberg = HamiltonianEncoding(n_features=4, hamiltonian_type='heisenberg')
        >>> enc_heisenberg.resource_summary()['gate_counts']['cnot'] > enc_iqp.resource_summary()['gate_counts']['cnot']
        True

        Use for hardware planning:

        >>> enc = HamiltonianEncoding(n_features=6, entanglement='linear')
        >>> summary = enc.resource_summary()
        >>> print(f"Requires {summary['hardware_requirements']['connectivity']}")
        Requires linear
        >>> print(f"Native gates: {summary['hardware_requirements']['native_gates']}")
        Native gates: ['H', 'RZ', 'CNOT']

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
        count_gates : Get exact gate counts for specific input data.
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

        # Determine native gates needed based on Hamiltonian type
        native_gates = ["H", "RZ", "CNOT"]
        if self.hamiltonian_type in ["xy", "heisenberg"]:
            native_gates.extend(["S", "Sdg"])

        summary: dict[str, Any] = {
            # Basic circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            # Hamiltonian configuration
            "hamiltonian_type": self.hamiltonian_type,
            "evolution_time": self.evolution_time,
            "include_single_qubit_terms": self.include_single_qubit_terms,
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
            "hamiltonian_type=%s, entanglement=%s, pairs=%d",
            self.n_qubits,
            self.depth,
            gate_counts["total"],
            self.hamiltonian_type,
            self.entanglement,
            len(pairs),
        )

        return summary

    # =========================================================================
    # Serialization support for distributed computing and caching
    # =========================================================================

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling.

        Returns
        -------
        dict
            State dictionary containing all necessary attributes to
            reconstruct the encoding instance.

        Notes
        -----
        This method enables serialization for:
        - Distributed computing (multiprocessing, Ray, Dask)
        - Caching encoded circuits to disk
        - Saving/loading encoding configurations

        The cached depth value is excluded and will be recomputed on
        deserialization to ensure correctness.
        """
        # Get all instance attributes
        state = self.__dict__.copy()

        # Remove cached values that can be recomputed
        # This ensures the pickled object is minimal and always valid
        state.pop("_cached_depth", None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickle.

        Parameters
        ----------
        state : dict
            State dictionary from __getstate__.

        Notes
        -----
        After deserialization:
        - The depth cache is reset (will be recomputed on first access)
        - All other attributes are restored exactly
        - The object is immediately usable for circuit generation
        """
        # Restore all saved attributes
        self.__dict__.update(state)

        # Reset cache (will be recomputed on first access)
        self._cached_depth = None

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        """Support for pickle reduction.

        This provides an alternative pickling mechanism that reconstructs
        the object using the constructor, which guarantees all validation
        and initialization logic is executed.

        Returns
        -------
        tuple
            Tuple of (class, constructor_args) for reconstruction.
        """
        return (
            self.__class__,
            (
                self.n_features,
                self.hamiltonian_type,
                self.evolution_time,
                self.reps,
                self.entanglement,
                self.insert_barriers,
                self.max_pairs,
                self.include_single_qubit_terms,
            ),
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> HamiltonianEncoding:
        """Create a deep copy of this encoding.

        Parameters
        ----------
        memo : dict
            Memoization dictionary for handling circular references.

        Returns
        -------
        HamiltonianEncoding
            Deep copy of this encoding with identical configuration.

        Notes
        -----
        The deep copy creates a new instance with the same parameters.
        This is useful for:
        - Creating independent copies for parallel processing
        - Modifying configurations without affecting the original
        - Thread-safe operations
        """
        # Create a new instance with the same parameters
        # This ensures all initialization logic runs correctly
        return HamiltonianEncoding(
            n_features=self.n_features,
            hamiltonian_type=self.hamiltonian_type,
            evolution_time=self.evolution_time,
            reps=self.reps,
            entanglement=self.entanglement,
            insert_barriers=self.insert_barriers,
            max_pairs=self.max_pairs,
            include_single_qubit_terms=self.include_single_qubit_terms,
        )


# =============================================================================
# Helper functions for two-qubit interactions
# =============================================================================


def _apply_zz_interaction_pennylane(
    qml: Any,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply ZZ interaction exp(-i*angle*ZZ) in PennyLane.

    Parameters
    ----------
    qml : module
        PennyLane module.
    angle : float
        Interaction strength.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    qml.CNOT(wires=[qubit_i, qubit_j])
    qml.RZ(2 * angle, wires=qubit_j)
    qml.CNOT(wires=[qubit_i, qubit_j])


def _apply_xx_interaction_pennylane(
    qml: Any,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply XX interaction exp(-i*angle*XX) in PennyLane.

    Parameters
    ----------
    qml : module
        PennyLane module.
    angle : float
        Interaction strength.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    # Basis change to X basis
    qml.Hadamard(wires=qubit_i)
    qml.Hadamard(wires=qubit_j)
    # ZZ interaction in X basis
    qml.CNOT(wires=[qubit_i, qubit_j])
    qml.RZ(2 * angle, wires=qubit_j)
    qml.CNOT(wires=[qubit_i, qubit_j])
    # Basis change back
    qml.Hadamard(wires=qubit_i)
    qml.Hadamard(wires=qubit_j)


def _apply_yy_interaction_pennylane(
    qml: Any,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply YY interaction exp(-i*angle*YY) in PennyLane.

    Parameters
    ----------
    qml : module
        PennyLane module.
    angle : float
        Interaction strength.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    # Basis change to Y basis
    qml.adjoint(qml.S)(wires=qubit_i)
    qml.Hadamard(wires=qubit_i)
    qml.adjoint(qml.S)(wires=qubit_j)
    qml.Hadamard(wires=qubit_j)
    # ZZ interaction in Y basis
    qml.CNOT(wires=[qubit_i, qubit_j])
    qml.RZ(2 * angle, wires=qubit_j)
    qml.CNOT(wires=[qubit_i, qubit_j])
    # Basis change back
    qml.Hadamard(wires=qubit_i)
    qml.S(wires=qubit_i)
    qml.Hadamard(wires=qubit_j)
    qml.S(wires=qubit_j)


def _apply_zz_interaction_qiskit(
    qc: Any,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply ZZ interaction in Qiskit.

    Parameters
    ----------
    qc : QuantumCircuit
        Qiskit quantum circuit.
    angle : float
        Interaction strength.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    qc.cx(qubit_i, qubit_j)
    qc.rz(2 * angle, qubit_j)
    qc.cx(qubit_i, qubit_j)


def _apply_xx_interaction_qiskit(
    qc: Any,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply XX interaction in Qiskit.

    Parameters
    ----------
    qc : QuantumCircuit
        Qiskit quantum circuit.
    angle : float
        Interaction strength.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    qc.h(qubit_i)
    qc.h(qubit_j)
    qc.cx(qubit_i, qubit_j)
    qc.rz(2 * angle, qubit_j)
    qc.cx(qubit_i, qubit_j)
    qc.h(qubit_i)
    qc.h(qubit_j)


def _apply_yy_interaction_qiskit(
    qc: Any,
    angle: float,
    qubit_i: int,
    qubit_j: int,
) -> None:
    """Apply YY interaction in Qiskit.

    Parameters
    ----------
    qc : QuantumCircuit
        Qiskit quantum circuit.
    angle : float
        Interaction strength.
    qubit_i : int
        First qubit index.
    qubit_j : int
        Second qubit index.
    """
    qc.sdg(qubit_i)
    qc.h(qubit_i)
    qc.sdg(qubit_j)
    qc.h(qubit_j)
    qc.cx(qubit_i, qubit_j)
    qc.rz(2 * angle, qubit_j)
    qc.cx(qubit_i, qubit_j)
    qc.h(qubit_i)
    qc.s(qubit_i)
    qc.h(qubit_j)
    qc.s(qubit_j)


def _apply_zz_interaction_cirq(
    cirq: Any,
    angle: float,
    qubit_i: Any,
    qubit_j: Any,
) -> list[Any]:
    """Apply ZZ interaction in Cirq.

    Parameters
    ----------
    cirq : module
        Cirq module.
    angle : float
        Interaction strength.
    qubit_i : cirq.LineQubit
        First qubit.
    qubit_j : cirq.LineQubit
        Second qubit.

    Returns
    -------
    list[cirq.Moment]
        List of circuit moments.
    """
    moments = []
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))
    moments.append(cirq.Moment([cirq.rz(2 * angle)(qubit_j)]))
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))
    return moments


def _apply_xx_interaction_cirq(
    cirq: Any,
    angle: float,
    qubit_i: Any,
    qubit_j: Any,
) -> list[Any]:
    """Apply XX interaction in Cirq.

    Parameters
    ----------
    cirq : module
        Cirq module.
    angle : float
        Interaction strength.
    qubit_i : cirq.LineQubit
        First qubit.
    qubit_j : cirq.LineQubit
        Second qubit.

    Returns
    -------
    list[cirq.Moment]
        List of circuit moments.
    """
    moments = []
    moments.append(cirq.Moment([cirq.H(qubit_i), cirq.H(qubit_j)]))
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))
    moments.append(cirq.Moment([cirq.rz(2 * angle)(qubit_j)]))
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))
    moments.append(cirq.Moment([cirq.H(qubit_i), cirq.H(qubit_j)]))
    return moments


def _apply_yy_interaction_cirq(
    cirq: Any,
    angle: float,
    qubit_i: Any,
    qubit_j: Any,
) -> list[Any]:
    """Apply YY interaction in Cirq.

    Parameters
    ----------
    cirq : module
        Cirq module.
    angle : float
        Interaction strength.
    qubit_i : cirq.LineQubit
        First qubit.
    qubit_j : cirq.LineQubit
        Second qubit.

    Returns
    -------
    list[cirq.Moment]
        List of circuit moments.
    """
    moments = []
    # Basis change to Y
    moments.append(cirq.Moment([cirq.S(qubit_i) ** -1, cirq.S(qubit_j) ** -1]))
    moments.append(cirq.Moment([cirq.H(qubit_i), cirq.H(qubit_j)]))
    # ZZ in Y basis
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))
    moments.append(cirq.Moment([cirq.rz(2 * angle)(qubit_j)]))
    moments.append(cirq.Moment([cirq.CNOT(qubit_i, qubit_j)]))
    # Basis change back
    moments.append(cirq.Moment([cirq.H(qubit_i), cirq.H(qubit_j)]))
    moments.append(cirq.Moment([cirq.S(qubit_i), cirq.S(qubit_j)]))
    return moments
