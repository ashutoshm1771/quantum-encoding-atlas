"""Higher-Order Angle Encoding module.

This module implements Higher-Order Angle Encoding, which creates feature
interactions by applying rotation gates with polynomial combinations of
input features. Unlike standard angle encoding which only uses first-order
terms (x_i), this encoding uses products like x_i * x_j (second-order),
x_i * x_j * x_k (third-order), etc.

The key insight is that this encoding creates non-linear feature interactions
WITHOUT using entangling gates, meaning the resulting quantum state is still
a product state. This makes it classically simulable but can still be useful
for capturing feature correlations.

Mathematical Definition:
    |ψ(x)⟩ = ⊗_q R_axis(θ_q) |0⟩

    where θ_q = Σ_S c_S · ∏_{i∈S} x_i

    and S ranges over subsets of features up to the specified order.

Example:
    For order=2 with features [x_0, x_1, x_2]:
    - First-order terms: x_0, x_1, x_2
    - Second-order terms: x_0*x_1, x_0*x_2, x_1*x_2

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.higher_order_angle').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters (n_features, order, rotation, combination)
- Term generation and assignment details
- Circuit generation progress for each backend
- Batch processing statistics

Resource Analysis
-----------------
HigherOrderAngleEncoding provides methods to analyze circuit resources:

- ``get_term_info()``: Inspect polynomial terms and qubit assignments
- ``gate_count_breakdown()``: Get detailed gate counts by type
- ``resource_summary()``: Complete breakdown of circuit resources
- ``compute_angles(x)``: Debug rotation angles for specific input

These methods help understand resource requirements before execution:

    >>> enc = HigherOrderAngleEncoding(n_features=4, order=2)
    >>> enc.gate_count_breakdown()
    {'rx': 0, 'ry': 4, 'rz': 0, 'total_single_qubit': 4, 'total_two_qubit': 0, 'total': 4}
    >>> enc.resource_summary()['n_terms']
    10

Thread Safety
-------------
HigherOrderAngleEncoding is thread-safe for all read operations including
circuit generation. The encoding object is immutable after construction:

- All parameters are stored as read-only attributes
- Terms and qubit assignments are pre-computed and cached as immutable tuples
- Circuit generation methods do not modify instance state

Multiple threads can safely call ``get_circuit()`` and ``get_circuits()``
concurrently on the same encoding instance. The ``parallel=True`` option
in ``get_circuits()`` uses ThreadPoolExecutor internally.

References
----------
.. [1] Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in
       Feature Hilbert Spaces." Physical Review Letters, 122(4), 040504.
.. [2] Havlíček, V., et al. (2019). "Supervised learning with quantum-
       enhanced feature spaces." Nature, 567(7747), 209-212.
.. [3] Stoudenmire, E., & Schwab, D. J. (2016). "Supervised learning with
       tensor networks." Advances in Neural Information Processing Systems.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterator
from itertools import combinations
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
#   To enable debug logging for higher-order angle encoding:
#
#   >>> import logging
#   >>> logging.getLogger('encoding_atlas.encodings.higher_order_angle').setLevel(logging.DEBUG)
#   >>> handler = logging.StreamHandler()
#   >>> handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
#   >>> logging.getLogger('encoding_atlas.encodings.higher_order_angle').addHandler(handler)
#
# Log Levels:
#   - DEBUG: Detailed information for diagnosing issues (term generation,
#            angle computation, circuit generation steps)
#   - INFO: General operational information (encoding created, circuit generated)
#   - WARNING: Potential issues (high term count, exponential scaling)
#
# The logger is named after the module to allow granular control over log output.
# By default, no handlers are attached, so no output is produced unless the
# application configures logging.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["HigherOrderAngleEncoding", "GateCountBreakdown", "count_terms"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default backend when None is provided.
_DEFAULT_BACKEND: BackendType = "pennylane"

# Supported backend frameworks for circuit generation.
_SUPPORTED_BACKENDS: frozenset[str] = frozenset({"pennylane", "qiskit", "cirq"})


def _normalize_backend(backend: BackendType | None) -> BackendType:
    """Normalize backend parameter to a valid lowercase backend name.

    Handles None by returning the default backend and performs
    case-insensitive matching for user convenience.

    Parameters
    ----------
    backend : BackendType or None
        Backend to normalize. If None, returns the default backend.

    Returns
    -------
    BackendType
        Normalized lowercase backend name.

    Raises
    ------
    ValueError
        If backend is not a supported value.

    Examples
    --------
    >>> _normalize_backend(None)
    'pennylane'
    >>> _normalize_backend("QISKIT")
    'qiskit'
    >>> _normalize_backend("PennyLane")
    'pennylane'
    """
    if backend is None:
        return _DEFAULT_BACKEND

    if not isinstance(backend, str):
        raise ValueError(f"backend must be a string, got {type(backend).__name__}")

    backend_lower = backend.lower()
    if backend_lower not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends: {', '.join(repr(b) for b in sorted(_SUPPORTED_BACKENDS))}"
        )

    return backend_lower  # type: ignore[return-value]


# Warning threshold for exponential term growth.
#
# Higher-order angle encoding generates polynomial terms combinatorially.
# The number of k-th order terms is C(n, k), and the total terms can grow
# exponentially when order approaches n_features:
#
#   - order=2, n=10: 55 terms
#   - order=3, n=10: 175 terms
#   - order=5, n=10: 637 terms
#   - order=n, n=10: 1023 terms (2^n - 1)
#
# At 100 terms, computational overhead becomes noticeable. Users should be
# aware of the scaling behavior and consider limiting the order for large
# feature counts.
_TERM_COUNT_WARNING_THRESHOLD: int = 100


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
    """Total number of two-qubit gates (always 0 for HigherOrderAngleEncoding)."""

    total: int
    """Total gate count."""


class HigherOrderAngleEncoding(BaseEncoding):
    """Higher-order angle encoding with polynomial feature interactions.

    This encoding applies rotation gates where the rotation angles are
    polynomial combinations of input features. It generalizes standard
    angle encoding by including higher-order interaction terms like
    products of features (x_i * x_j).

    The encoding creates feature interactions without entanglement,
    resulting in a product state that is classically simulable but can
    still capture non-linear relationships between features.

    Parameters
    ----------
    n_features : int
        Number of classical input features. Must be at least 1.
    order : int, default=2
        Maximum polynomial order for feature interactions.
        - order=1: Only single features (standard angle encoding)
        - order=2: Single features + pairwise products
        - order=3: Single features + pairs + triples
        Must be between 1 and n_features (inclusive).
    rotation : {'X', 'Y', 'Z'}, default='Y'
        Rotation axis for the encoding gates.
        - 'X': RX gates (rotation around X-axis)
        - 'Y': RY gates (rotation around Y-axis, most common)
        - 'Z': RZ gates (rotation around Z-axis)
    combination : {'product', 'sum'}, default='product'
        How to combine features in higher-order terms.
        - 'product': Multiply features (x_i * x_j)
        - 'sum': Add features (x_i + x_j)
    include_first_order : bool, default=True
        Whether to include first-order (single feature) terms.
        If False, only higher-order interaction terms are used.
    scaling : float, default=1.0
        Scaling factor applied to all rotation angles.
        Useful for controlling the range of rotations.
    reps : int, default=1
        Number of repetitions of the encoding circuit.
        Each repetition applies all terms again.

    Attributes
    ----------
    n_features : int
        Number of input features.
    order : int
        Maximum polynomial order.
    rotation : str
        Rotation axis ('X', 'Y', or 'Z').
    combination : str
        Feature combination method.
    include_first_order : bool
        Whether first-order terms are included.
    scaling : float
        Rotation angle scaling factor.
    reps : int
        Number of circuit repetitions.
    n_qubits : int
        Number of qubits (equals n_features).
    depth : int
        Circuit depth.
    terms : list[tuple[int, ...]]
        List of feature index tuples representing each term.

    Examples
    --------
    Basic usage with default parameters:

    >>> import numpy as np
    >>> enc = HigherOrderAngleEncoding(n_features=4, order=2)
    >>> enc.n_qubits
    4
    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')

    Custom configuration with third-order interactions:

    >>> enc = HigherOrderAngleEncoding(
    ...     n_features=3,
    ...     order=3,
    ...     rotation='Z',
    ...     combination='product',
    ...     scaling=2.0,
    ...     reps=2
    ... )
    >>> len(enc.terms)  # 3 first-order + 3 second-order + 1 third-order
    7

    Only second-order interactions (no first-order terms):

    >>> enc = HigherOrderAngleEncoding(
    ...     n_features=4,
    ...     order=2,
    ...     include_first_order=False
    ... )
    >>> len(enc.terms)  # Only 6 pairwise terms
    6

    See Also
    --------
    AngleEncoding : Standard first-order angle encoding.
    IQPEncoding : Encoding with entanglement using ZZ interactions.
    PauliFeatureMap : Generalized encoding with configurable Pauli terms.

    Notes
    -----
    **Circuit Structure:**

    The circuit applies rotation gates to each qubit, where the rotation
    angles are computed from polynomial terms. For n_features=3, order=2:

    ::

        q0: ─RY(θ_0)─
        q1: ─RY(θ_1)─
        q2: ─RY(θ_2)─

    **Term Generation and Assignment:**

    Terms are generated in lexicographic order by polynomial degree, then
    assigned to qubits using **round-robin by term index** (not by feature).

    For n_features=3, order=2, the terms are generated as:

    ::

        Index 0: (0,)   → x_0        (first-order)
        Index 1: (1,)   → x_1        (first-order)
        Index 2: (2,)   → x_2        (first-order)
        Index 3: (0,1)  → x_0 * x_1  (second-order)
        Index 4: (0,2)  → x_0 * x_2  (second-order)
        Index 5: (1,2)  → x_1 * x_2  (second-order)

    Round-robin assignment (qubit_idx = term_index % n_qubits):

    ::

        Qubit 0: indices 0, 3  → terms (0,), (0,1)
        Qubit 1: indices 1, 4  → terms (1,), (0,2)
        Qubit 2: indices 2, 5  → terms (2,), (1,2)

    Therefore, the rotation angles are:

    ::

        θ_0 = scaling * (x_0 + x_0*x_1)      [sum of terms at indices 0, 3]
        θ_1 = scaling * (x_1 + x_0*x_2)      [sum of terms at indices 1, 4]
        θ_2 = scaling * (x_2 + x_1*x_2)      [sum of terms at indices 2, 5]

    Note that the assignment is purely index-based, not feature-based. A term
    like x_0*x_2 (index 4) goes to qubit 1, even though it contains features
    0 and 2. Use ``get_term_info()`` to inspect the exact assignments for any
    configuration.

    **Complexity:**

    - Number of k-th order terms: C(n, k) = n! / (k! * (n-k)!)
    - Total terms: Σ_{k=1}^{order} C(n, k)
    - For order=2: n + n*(n-1)/2 terms
    - For order=n: 2^n - 1 terms (exponential)

    **Classical Simulability:**

    Since no entangling gates are used, the resulting state is a product
    state. Product states can be efficiently simulated classically, meaning
    this encoding alone cannot provide quantum advantage. However, it can
    still be useful when combined with entangling ansatz layers.

    **Comparison with IQP/ZZ Feature Maps:**

    Unlike IQP or ZZ feature maps which use entangling gates to create
    feature interactions in the quantum state, this encoding creates
    interactions in the rotation angles themselves. The key difference:

    - Higher-Order Angle: Interactions in angles → Product state (simulable)
    - IQP/ZZ: Interactions via entanglement → Entangled state (hard to simulate)

    **Thread Safety:**

    This encoding is thread-safe for all read operations including circuit
    generation. The encoding object is immutable after construction:

    - All parameters are stored as read-only attributes
    - Terms and qubit assignments are pre-computed and cached as immutable tuples
    - Circuit generation methods do not modify instance state

    Multiple threads can safely call ``get_circuit()`` and ``get_circuits()``
    concurrently on the same encoding instance. The ``parallel=True`` option
    in ``get_circuits()`` uses ThreadPoolExecutor internally for efficient
    batch processing.

    Warnings
    --------
    **High Term Count**: When the polynomial order approaches n_features, the
    number of terms grows exponentially (up to 2^n - 1 terms). A warning is
    issued when the term count exceeds 100. For large feature counts, consider
    using a lower order to manage circuit complexity.
    """

    # Valid rotation axes
    _VALID_ROTATIONS: frozenset[str] = frozenset({"X", "Y", "Z"})

    # Memory-efficient slot-based attribute storage.
    # This eliminates the per-instance __dict__ overhead and ensures
    # consistent attribute access patterns.
    __slots__ = (
        "order",
        "rotation",
        "combination",
        "include_first_order",
        "scaling",
        "reps",
        "_terms",
        "_qubit_terms",
    )

    # Valid combination methods
    _VALID_COMBINATIONS: frozenset[str] = frozenset({"product", "sum"})

    def __init__(
        self,
        n_features: int,
        order: int = 2,
        rotation: Literal["X", "Y", "Z"] = "Y",
        combination: Literal["product", "sum"] = "product",
        include_first_order: bool = True,
        scaling: float = 1.0,
        reps: int = 1,
    ) -> None:
        """Initialize the Higher-Order Angle Encoding.

        Parameters
        ----------
        n_features : int
            Number of classical input features.
        order : int, default=2
            Maximum polynomial order (1 to n_features).
        rotation : {'X', 'Y', 'Z'}, default='Y'
            Rotation axis for encoding gates.
        combination : {'product', 'sum'}, default='product'
            How to combine features in higher-order terms.
        include_first_order : bool, default=True
            Whether to include first-order terms.
        scaling : float, default=1.0
            Scaling factor for rotation angles.
        reps : int, default=1
            Number of circuit repetitions.

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

        # Validate n_features first (before using it in other checks)
        if not isinstance(n_features, (int, np.integer)):
            raise TypeError(
                f"n_features must be an integer, got {type(n_features).__name__}"
            )
        n_features = int(n_features)
        if n_features < 1:
            raise ValueError(f"n_features must be at least 1, got {n_features}")

        # Validate order
        if not isinstance(order, (int, np.integer)):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        order = int(order)
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        if order > n_features:
            raise ValueError(f"order ({order}) cannot exceed n_features ({n_features})")

        # Validate rotation
        if not isinstance(rotation, str):
            raise TypeError(f"rotation must be a string, got {type(rotation).__name__}")
        rotation_upper = rotation.upper()
        if rotation_upper not in self._VALID_ROTATIONS:
            raise ValueError(
                f"rotation must be one of {sorted(self._VALID_ROTATIONS)}, "
                f"got '{rotation}'"
            )

        # Validate combination
        if not isinstance(combination, str):
            raise TypeError(
                f"combination must be a string, got {type(combination).__name__}"
            )
        combination_lower = combination.lower()
        if combination_lower not in self._VALID_COMBINATIONS:
            raise ValueError(
                f"combination must be one of {sorted(self._VALID_COMBINATIONS)}, "
                f"got '{combination}'"
            )

        # Validate include_first_order
        if not isinstance(include_first_order, bool):
            raise TypeError(
                f"include_first_order must be a bool, "
                f"got {type(include_first_order).__name__}"
            )

        # Validate scaling
        if not isinstance(scaling, (int, float, np.integer, np.floating)):
            raise TypeError(f"scaling must be a number, got {type(scaling).__name__}")
        scaling = float(scaling)
        if not np.isfinite(scaling):
            raise ValueError(f"scaling must be finite, got {scaling}")

        # Validate reps
        if not isinstance(reps, (int, np.integer)):
            raise TypeError(f"reps must be an integer, got {type(reps).__name__}")
        reps = int(reps)
        if reps < 1:
            raise ValueError(f"reps must be at least 1, got {reps}")

        # Check for degenerate case
        if not include_first_order and order == 1:
            raise ValueError(
                "Cannot have order=1 with include_first_order=False "
                "(would result in no terms)"
            )

        # =================================================================
        # INITIALIZATION
        # =================================================================

        # Call parent constructor
        super().__init__(
            n_features,
            order=order,
            rotation=rotation_upper,
            combination=combination_lower,
            include_first_order=include_first_order,
            scaling=scaling,
            reps=reps,
        )

        # Store validated parameters
        self.order: int = order
        self.rotation: str = rotation_upper
        self.combination: str = combination_lower
        self.include_first_order: bool = include_first_order
        self.scaling: float = scaling
        self.reps: int = reps

        # Precompute terms for efficiency
        self._terms: tuple[tuple[int, ...], ...] = tuple(self._generate_terms())

        # Precompute term assignments to qubits (round-robin)
        self._qubit_terms: tuple[tuple[tuple[int, ...], ...], ...] = (
            self._assign_terms_to_qubits()
        )

        # Log initialization
        _logger.debug(
            "HigherOrderAngleEncoding initialized: n_features=%d, order=%d, "
            "rotation=%r, combination=%r, include_first_order=%s, scaling=%.4f, "
            "reps=%d, n_terms=%d",
            n_features,
            order,
            rotation_upper,
            combination_lower,
            include_first_order,
            scaling,
            reps,
            len(self._terms),
        )

        # Warn about exponential term growth
        if len(self._terms) > _TERM_COUNT_WARNING_THRESHOLD:
            warnings.warn(
                f"High term count ({len(self._terms)} terms) may impact performance. "
                f"With order={order} and n_features={n_features}, the term count "
                f"grows combinatorially. Consider using a lower order for better "
                f"performance, or ensure this complexity is intentional.",
                UserWarning,
                stacklevel=2,
            )
            _logger.warning(
                "High term count: %d terms with order=%d, n_features=%d",
                len(self._terms),
                order,
                n_features,
            )

    @property
    def terms(self) -> list[tuple[int, ...]]:
        """List of feature index tuples representing each polynomial term.

        Returns
        -------
        list[tuple[int, ...]]
            Each tuple contains the indices of features in that term.
            For example, (0, 1) represents x_0 * x_1 (or x_0 + x_1).

        Examples
        --------
        >>> enc = HigherOrderAngleEncoding(n_features=3, order=2)
        >>> enc.terms
        [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
        """
        return list(self._terms)

    @property
    def n_terms(self) -> int:
        """Total number of polynomial terms.

        Returns
        -------
        int
            Number of terms in the encoding.
        """
        return len(self._terms)

    def _generate_terms(self) -> Iterator[tuple[int, ...]]:
        """Generate all polynomial terms up to the specified order.

        Yields
        ------
        tuple[int, ...]
            Feature index tuple for each term.
        """
        start_order = 1 if self.include_first_order else 2

        for k in range(start_order, self.order + 1):
            for combo in combinations(range(self.n_features), k):
                yield combo

    def _assign_terms_to_qubits(
        self,
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Assign polynomial terms to qubits using round-robin distribution.

        The assignment uses term index modulo n_qubits:

            qubit_idx = term_index % n_qubits

        This distributes terms evenly across qubits regardless of which
        features appear in each term. For example, with 6 terms and 3 qubits:

        - Qubit 0 receives terms at indices 0, 3
        - Qubit 1 receives terms at indices 1, 4
        - Qubit 2 receives terms at indices 2, 5

        Returns
        -------
        tuple[tuple[tuple[int, ...], ...], ...]
            Outer tuple indexed by qubit. Each inner tuple contains the
            terms (as feature index tuples) assigned to that qubit.
            All structures are immutable for thread safety.
        """
        # Initialize empty lists for each qubit
        qubit_assignments: list[list[tuple[int, ...]]] = [
            [] for _ in range(self.n_features)
        ]

        # Distribute terms round-robin
        for idx, term in enumerate(self._terms):
            qubit_idx = idx % self.n_features
            qubit_assignments[qubit_idx].append(term)

        # Convert to immutable tuples
        return tuple(tuple(terms) for terms in qubit_assignments)

    def _compute_term_value(
        self,
        x: NDArray[np.floating[Any]],
        term: tuple[int, ...],
    ) -> float:
        """Compute the value of a polynomial term for given input.

        Parameters
        ----------
        x : ndarray
            Input feature vector.
        term : tuple[int, ...]
            Feature indices in the term.

        Returns
        -------
        float
            Computed term value.
        """
        if self.combination == "product":
            value = 1.0
            for idx in term:
                value *= x[idx]
            return value
        else:  # sum
            return sum(x[idx] for idx in term)

    def _compute_qubit_angle(
        self,
        x: NDArray[np.floating[Any]],
        qubit_idx: int,
    ) -> float:
        """Compute the total rotation angle for a specific qubit.

        Parameters
        ----------
        x : ndarray
            Input feature vector.
        qubit_idx : int
            Index of the qubit.

        Returns
        -------
        float
            Total rotation angle (sum of all term values for this qubit).
        """
        total = 0.0
        for term in self._qubit_terms[qubit_idx]:
            total += self._compute_term_value(x, term)
        return self.scaling * total

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding.

        Returns
        -------
        int
            Number of qubits (equals n_features).
        """
        return self.n_features

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        Returns
        -------
        int
            Circuit depth (reps, since all gates are parallel).
        """
        # All rotation gates can be applied in parallel
        return self.reps

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType | None = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for the given input data.

        Strict variant: a 2D input with more than one row raises
        :class:`ValueError`. Use :meth:`get_circuits` for batches. Backend
        names are case-insensitive and ``None`` resolves to the default
        backend.

        Parameters
        ----------
        x : array-like
            Input feature vector of shape (n_features,) or (1, n_features).
        backend : {'pennylane', 'qiskit', 'cirq'} or None, default='pennylane'
            Target quantum computing framework. Case-insensitive.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.

        Examples
        --------
        >>> enc = HigherOrderAngleEncoding(n_features=4, order=2)
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> circuit = enc.get_circuit(x, backend='pennylane')
        >>> callable(circuit)
        True
        """
        backend = _normalize_backend(backend)

        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            if x_validated.shape[0] != 1:
                raise ValueError(
                    f"get_circuit expects a single sample, got shape "
                    f"{x_validated.shape}. Use get_circuits for batches."
                )
            x_validated = x_validated[0]
        return self._get_circuit_from_validated(x_validated, backend)

    def get_circuits(
        self,
        X: ArrayLike,
        backend: BackendType | None = "pennylane",
        *,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> list[CircuitType]:
        """Generate quantum circuits for multiple input samples.

        Thin wrapper around :meth:`BaseEncoding.get_circuits` that
        normalizes the backend (case-insensitive, ``None`` allowed) before
        delegating.

        Parameters
        ----------
        X : array-like
            Input data of shape (n_samples, n_features) or (n_features,).
        backend : {'pennylane', 'qiskit', 'cirq'} or None, default='pennylane'
            Target quantum computing framework. Case-insensitive.
        parallel : bool, default=False
            See :meth:`BaseEncoding.get_circuits`.
        max_workers : int or None, default=None
            See :meth:`BaseEncoding.get_circuits`.

        Returns
        -------
        list[CircuitType]
            List of circuits, one per sample, in input order.
        """
        return super().get_circuits(
            X,
            _normalize_backend(backend),
            parallel=parallel,
            max_workers=max_workers,
        )

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input.

        Encoding-specific seam called by ``get_circuit`` and ``get_circuits``.
        Accepts a 1D array (the normal case) or a 2D single-sample
        ``(1, n_features)`` array for robustness when called directly.
        """
        # Defensive 2D-to-1D handling for direct callers.
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
        """Generate PennyLane circuit.

        Parameters
        ----------
        x : ndarray
            Input feature vector.

        Returns
        -------
        callable
            Function that applies the encoding gates.
        """
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(
                "PennyLane is required for the 'pennylane' backend. "
                "Install it with: pip install pennylane"
            ) from e

        # Precompute angles for all qubits
        angles = [self._compute_qubit_angle(x, q) for q in range(self.n_qubits)]

        # Select rotation gate
        if self.rotation == "X":
            rotation_gate = qml.RX
        elif self.rotation == "Y":
            rotation_gate = qml.RY
        else:  # Z
            rotation_gate = qml.RZ

        def circuit() -> None:
            """Apply the higher-order angle encoding gates.

            All rotation gates are applied unconditionally, including zero-angle
            rotations. This ensures consistent circuit structure across all backends
            and maintains compatibility with frameworks like Cirq that require
            explicit gate operations for qubit registration.
            """
            for _ in range(self.reps):
                for qubit_idx in range(self.n_qubits):
                    rotation_gate(angles[qubit_idx], wires=qubit_idx)

        return circuit

    def _to_qiskit(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Qiskit circuit.

        Parameters
        ----------
        x : ndarray
            Input feature vector.

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit with encoding gates.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError(
                "Qiskit is required for the 'qiskit' backend. "
                "Install it with: pip install qiskit"
            ) from e

        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits, name="HigherOrderAngle")

        # Select rotation method
        if self.rotation == "X":
            apply_rotation = qc.rx
        elif self.rotation == "Y":
            apply_rotation = qc.ry
        else:  # Z
            apply_rotation = qc.rz

        # Apply encoding gates unconditionally for all qubits.
        # Zero-angle rotations are included to maintain consistent circuit
        # structure across backends and ensure predictable gate counts.
        for rep in range(self.reps):
            for qubit_idx in range(self.n_qubits):
                apply_rotation(self._compute_qubit_angle(x, qubit_idx), qubit_idx)

            # Add barrier between repetitions for clarity
            if rep < self.reps - 1:
                qc.barrier()

        return qc

    def _to_cirq(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Cirq circuit.

        Parameters
        ----------
        x : ndarray
            Input feature vector.

        Returns
        -------
        cirq.Circuit
            Cirq circuit with encoding gates.
        """
        try:
            import cirq
        except ImportError as e:
            raise ImportError(
                "Cirq is required for the 'cirq' backend. "
                "Install it with: pip install cirq-core"
            ) from e

        # Create qubits
        qubits = cirq.LineQubit.range(self.n_qubits)

        # Select rotation gate constructor
        if self.rotation == "X":

            def rotation_gate(angle: float) -> cirq.Gate:
                return cirq.rx(angle)

        elif self.rotation == "Y":

            def rotation_gate(angle: float) -> cirq.Gate:
                return cirq.ry(angle)

        else:  # Z

            def rotation_gate(angle: float) -> cirq.Gate:
                return cirq.rz(angle)

        # Build circuit
        circuit = cirq.Circuit()

        for _ in range(self.reps):
            # Collect all gates for this repetition (parallel application)
            # Note: Unlike PennyLane/Qiskit which pre-allocate qubits, Cirq only
            # registers qubits that have operations. We must include all qubits
            # to ensure consistent state vector dimensions across backends.
            ops = [
                rotation_gate(self._compute_qubit_angle(x, qubit_idx)).on(
                    qubits[qubit_idx]
                )
                for qubit_idx in range(self.n_qubits)
            ]
            circuit.append(ops)

        return circuit

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Returns
        -------
        EncodingProperties
            Computed properties including qubit count, depth, gate counts, etc.
        """
        n = self.n_qubits
        n_terms = self.n_terms

        # Gate count: one rotation per qubit per rep (deterministic).
        # All rotations are applied unconditionally for cross-backend consistency.
        single_qubit_gates = n * self.reps

        # Depth: all rotations are parallel, so depth = reps
        depth = self.reps

        # This encoding creates no entanglement (product states)
        is_entangling = False

        # Product states are classically simulable
        simulability: Literal[
            "simulable", "conditionally_simulable", "not_simulable"
        ] = "simulable"

        # Trainability: product states have good trainability
        # No barren plateau issues from this encoding alone
        trainability_estimate = 0.95 - 0.02 * self.reps

        # Build notes
        notes_parts = [
            f"Order-{self.order} polynomial encoding",
            f"with {n_terms} terms",
            f"({self.combination} combination)",
        ]
        if not self.include_first_order:
            notes_parts.append("excluding first-order terms")

        return EncodingProperties(
            n_qubits=n,
            depth=depth,
            gate_count=single_qubit_gates,
            single_qubit_gates=single_qubit_gates,
            two_qubit_gates=0,  # No entangling gates
            parameter_count=0,  # All data-dependent, no trainable params
            is_entangling=is_entangling,
            simulability=simulability,
            trainability_estimate=trainability_estimate,
            notes=", ".join(notes_parts),
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
            - ``'total_single_qubit'``: Total single-qubit gates
            - ``'total_two_qubit'``: Always 0 (no entanglement)
            - ``'total'``: Total gate count

        Examples
        --------
        >>> enc = HigherOrderAngleEncoding(n_features=4, order=2, rotation='Y', reps=2)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['ry']
        8
        >>> breakdown['total_single_qubit']
        8
        >>> breakdown['total_two_qubit']
        0

        >>> enc_x = HigherOrderAngleEncoding(n_features=4, order=2, rotation='X')
        >>> enc_x.gate_count_breakdown()['rx']
        4

        See Also
        --------
        resource_summary : Comprehensive resource analysis including
            hardware requirements and encoding characteristics.
        properties : Basic encoding properties via EncodingProperties object.
        """
        n = self.n_qubits
        total = n * self.reps

        rx = total if self.rotation == "X" else 0
        ry = total if self.rotation == "Y" else 0
        rz = total if self.rotation == "Z" else 0

        _logger.debug(
            "Gate breakdown: rx=%d, ry=%d, rz=%d, total=%d",
            rx,
            ry,
            rz,
            total,
        )

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

            **Polynomial Configuration**:
                - ``order``: Maximum polynomial order
                - ``combination``: Combination method ('product' or 'sum')
                - ``include_first_order``: Whether first-order terms are included
                - ``scaling``: Scaling factor for rotation angles
                - ``n_terms``: Total number of polynomial terms
                - ``terms_by_order``: Dict mapping order to term count

            **Gate Counts**:
                - ``gate_counts``: Full breakdown from ``gate_count_breakdown()``

            **Encoding Characteristics**:
                - ``is_entangling``: Always False for higher-order angle encoding
                - ``simulability``: Always 'simulable' for product states
                - ``trainability_estimate``: Trainability score (0.0-1.0)

            **Hardware Requirements**:
                - ``connectivity``: 'none' (no two-qubit gates required)
                - ``native_gates``: List of required gate types

        Examples
        --------
        >>> enc = HigherOrderAngleEncoding(n_features=4, order=2, rotation='Y', reps=2)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['n_terms']
        10
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
        get_term_info : Detailed polynomial term information.
        properties : Basic encoding properties.
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties

        # Count terms by order
        terms_by_order: dict[int, int] = {}
        for term in self._terms:
            order = len(term)
            terms_by_order[order] = terms_by_order.get(order, 0) + 1

        summary: dict[str, Any] = {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "depth": self.depth,
            "reps": self.reps,
            "rotation": self.rotation,
            # Polynomial configuration
            "order": self.order,
            "combination": self.combination,
            "include_first_order": self.include_first_order,
            "scaling": self.scaling,
            "n_terms": self.n_terms,
            "terms_by_order": terms_by_order,
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

        _logger.debug(
            "Resource summary: n_qubits=%d, depth=%d, n_terms=%d, " "total_gates=%d",
            self.n_qubits,
            self.depth,
            self.n_terms,
            gate_counts["total"],
        )

        return summary

    def get_term_info(self) -> dict[str, Any]:
        """Get detailed information about the polynomial terms and their assignments.

        This method is useful for understanding and debugging the encoding's
        term distribution. Terms are assigned to qubits using round-robin by
        term index: ``qubit_idx = term_index % n_qubits``.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``'terms'``: List of all terms as tuples of feature indices
            - ``'n_terms'``: Total number of terms
            - ``'terms_by_order'``: Dict mapping polynomial order to terms
            - ``'qubit_assignments'``: Dict mapping qubit index to assigned terms

        Examples
        --------
        >>> enc = HigherOrderAngleEncoding(n_features=3, order=2)
        >>> info = enc.get_term_info()
        >>> info['n_terms']
        6
        >>> info['terms_by_order'][1]  # First-order terms
        [(0,), (1,), (2,)]
        >>> info['terms_by_order'][2]  # Second-order terms
        [(0, 1), (0, 2), (1, 2)]

        Inspect qubit assignments (round-robin by term index):

        >>> info['qubit_assignments'][0]  # Qubit 0 gets term indices 0, 3
        [(0,), (0, 1)]
        >>> info['qubit_assignments'][1]  # Qubit 1 gets term indices 1, 4
        [(1,), (0, 2)]
        >>> info['qubit_assignments'][2]  # Qubit 2 gets term indices 2, 5
        [(2,), (1, 2)]

        See Also
        --------
        compute_angles : Compute rotation angles for specific input data.
        terms : Direct access to the list of terms.
        """
        # Group terms by order
        terms_by_order: dict[int, list[tuple[int, ...]]] = {}
        for term in self._terms:
            order = len(term)
            if order not in terms_by_order:
                terms_by_order[order] = []
            terms_by_order[order].append(term)

        # Qubit assignments
        qubit_assignments = {
            q: list(self._qubit_terms[q]) for q in range(self.n_qubits)
        }

        return {
            "terms": self.terms,
            "n_terms": self.n_terms,
            "terms_by_order": terms_by_order,
            "qubit_assignments": qubit_assignments,
        }

    def compute_angles(
        self,
        x: ArrayLike,
    ) -> NDArray[np.floating[Any]]:
        """Compute the rotation angles for each qubit given input data.

        This is useful for debugging and understanding the encoding.

        Parameters
        ----------
        x : array-like
            Input feature vector of shape (n_features,).

        Returns
        -------
        ndarray
            Array of shape (n_qubits,) containing the rotation angle for
            each qubit.

        Examples
        --------
        >>> enc = HigherOrderAngleEncoding(n_features=3, order=2)
        >>> x = np.array([0.5, 0.5, 0.5])
        >>> angles = enc.compute_angles(x)
        >>> angles.shape
        (3,)
        """
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]

        return np.array(
            [self._compute_qubit_angle(x_validated, q) for q in range(self.n_qubits)]
        )

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, "
            f"order={self.order}, "
            f"rotation='{self.rotation}', "
            f"combination='{self.combination}', "
            f"include_first_order={self.include_first_order}, "
            f"scaling={self.scaling}, "
            f"reps={self.reps})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoding."""
        if not isinstance(other, HigherOrderAngleEncoding):
            return NotImplemented
        return (
            self.n_features == other.n_features
            and self.order == other.order
            and self.rotation == other.rotation
            and self.combination == other.combination
            and self.include_first_order == other.include_first_order
            and self.scaling == other.scaling
            and self.reps == other.reps
        )

    def __hash__(self) -> int:
        """Return hash of the encoding."""
        return hash(
            (
                self.__class__.__name__,
                self.n_features,
                self.order,
                self.rotation,
                self.combination,
                self.include_first_order,
                self.scaling,
                self.reps,
            )
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def count_terms(n_features: int, order: int, include_first_order: bool = True) -> int:
    """Count the total number of polynomial terms.

    Parameters
    ----------
    n_features : int
        Number of input features.
    order : int
        Maximum polynomial order.
    include_first_order : bool, default=True
        Whether to include first-order terms.

    Returns
    -------
    int
        Total number of terms.

    Examples
    --------
    >>> count_terms(4, 2, True)   # 4 + 6 = 10
    10
    >>> count_terms(4, 2, False)  # Only 6 pairs
    6
    >>> count_terms(4, 3, True)   # 4 + 6 + 4 = 14
    14
    """
    start = 1 if include_first_order else 2
    total = 0
    for k in range(start, order + 1):
        total += math.comb(n_features, k)
    return total
