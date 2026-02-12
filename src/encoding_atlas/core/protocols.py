"""Capability protocols for quantum encodings.

This module defines **optional capability protocols** that encodings can implement
to provide additional functionality beyond the core `BaseEncoding` contract. These
protocols follow Python's structural subtyping (PEP 544) and the Interface
Segregation Principle from SOLID.

Design Philosophy
-----------------
The Quantum Encoding Atlas uses a **Layered Contract Architecture**:

1. **Layer 1 (Core Contract)**: `BaseEncoding` - mandatory methods all encodings
   must implement (`get_circuit`, `get_circuits`, `n_qubits`, `depth`, etc.)

2. **Layer 2 (Capability Protocols)**: This module - optional standardized
   interfaces that encodings implement if applicable

3. **Layer 3 (Encoding-Specific)**: Unique methods that don't generalize
   (e.g., `BasisEncoding.binarize()`)

**Why Protocols?**

- **Flexibility**: Encodings only implement what makes sense for them
- **Discoverability**: Users can check capabilities with `isinstance()`
- **Type Safety**: IDEs and type checkers understand the contracts
- **No Deep Inheritance**: Avoids fragile class hierarchies

Protocol Reference
------------------
.. list-table:: Available Protocols
   :header-rows: 1
   :widths: 25 50 25

   * - Protocol
     - Description
     - Applicable Encodings
   * - `ResourceAnalyzable`
     - Resource analysis independent of input data
     - Most encodings (11/13)
   * - `DataDependentResourceAnalyzable`
     - Resource analysis that varies with input
     - BasisEncoding
   * - `EntanglementQueryable`
     - Query entanglement structure
     - Entangling encodings (8/13)
   * - `DataTransformable`
     - Expose data preprocessing logic
     - AmplitudeEncoding, BasisEncoding

Usage Examples
--------------
**Checking capabilities at runtime:**

>>> from encoding_atlas import IQPEncoding, BasisEncoding
>>> from encoding_atlas.core.protocols import (
...     ResourceAnalyzable,
...     EntanglementQueryable,
... )
>>>
>>> enc = IQPEncoding(n_features=4)
>>>
>>> # Check if encoding supports resource analysis
>>> if isinstance(enc, ResourceAnalyzable):
...     summary = enc.resource_summary()
...     breakdown = enc.gate_count_breakdown()
...     print(f"Total gates: {breakdown['total']}")
Total gates: 28
>>>
>>> # Check if encoding has entanglement info
>>> if isinstance(enc, EntanglementQueryable):
...     pairs = enc.get_entanglement_pairs()
...     print(f"Entanglement pairs: {len(pairs)}")
Entanglement pairs: 6

**Writing generic functions that work with any capable encoding:**

>>> from typing import Any
>>> from encoding_atlas.core.base import BaseEncoding
>>> from encoding_atlas.core.protocols import ResourceAnalyzable
>>>
>>> def analyze_if_possible(enc: BaseEncoding) -> dict[str, Any] | None:
...     '''Analyze encoding resources if supported.'''
...     if isinstance(enc, ResourceAnalyzable):
...         return enc.resource_summary()
...     return None
>>>
>>> # Works with IQPEncoding (implements ResourceAnalyzable)
>>> result = analyze_if_possible(IQPEncoding(n_features=4))
>>> result is not None
True
>>>
>>> # Returns None for encodings that don't support it
>>> # (hypothetical encoding without resource analysis)

**Type-safe code with static analysis:**

>>> from encoding_atlas.core.protocols import EntanglementQueryable
>>>
>>> def get_entanglement_info(enc: EntanglementQueryable) -> list[tuple[int, int]]:
...     '''Get entanglement pairs from an encoding.
...
...     Type checker enforces that only EntanglementQueryable
...     encodings can be passed to this function.
...     '''
...     return enc.get_entanglement_pairs()

Industry Standards
------------------
This module implements patterns from established software engineering practices:

- **PEP 544**: Protocols: Structural subtyping (static duck typing)
  https://peps.python.org/pep-0544/

- **Interface Segregation Principle (ISP)**: From SOLID principles
  "Clients should not be forced to depend on interfaces they do not use."
  https://en.wikipedia.org/wiki/Interface_segregation_principle

- **Similar patterns in other libraries**:
  - scikit-learn: `ClassifierMixin`, `RegressorMixin`, `TransformerMixin`
  - Rust: Traits
  - Go: Interfaces (implicit satisfaction)
  - TypeScript: Structural typing

See Also
--------
- :class:`encoding_atlas.core.base.BaseEncoding` : Core encoding contract
- :class:`encoding_atlas.core.types.BaseEncodingProtocol` : Type-checking protocol
- :mod:`encoding_atlas.core.properties` : Property data structures

Notes
-----
All protocols in this module are decorated with ``@runtime_checkable``, which
enables ``isinstance()`` checks at runtime. However, runtime checking only
verifies method existence, not signatures. For full type safety, use a static
type checker like mypy.

Protocol methods are defined with ``...`` (Ellipsis) as the body, following
PEP 544 conventions. This indicates that the Protocol doesn't provide an
implementation - conforming classes must implement these methods.

References
----------
.. [1] Python PEP 544 - Protocols: Structural subtyping (static duck typing)
       https://peps.python.org/pep-0544/

.. [2] Martin, R. C. (2000). "Design Principles and Design Patterns."
       Interface Segregation Principle.

.. [3] Gamma, E., et al. (1994). "Design Patterns: Elements of Reusable
       Object-Oriented Software." Strategy Pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Protocols
    "ResourceAnalyzable",
    "DataDependentResourceAnalyzable",
    "EntanglementQueryable",
    "DataTransformable",
    # Type guards
    "is_resource_analyzable",
    "is_data_dependent_resource_analyzable",
    "is_entanglement_queryable",
    "is_data_transformable",
]


# =============================================================================
# Resource Analysis Protocols
# =============================================================================


@runtime_checkable
class ResourceAnalyzable(Protocol):
    """Protocol for encodings that provide data-independent resource analysis.

    This protocol defines a standardized interface for querying circuit
    resources (gate counts, depth, etc.) when the resource usage does NOT
    depend on the specific input data being encoded.

    **When to Implement**

    Implement this protocol when your encoding's resource requirements are
    determined entirely by configuration parameters (n_features, reps, etc.)
    and do not vary with input data values.

    **When NOT to Implement**

    Do NOT implement this protocol if resource usage depends on input data.
    For example, `BasisEncoding` applies X gates only where input values
    exceed a threshold, so actual gate count varies per input. Such encodings
    should implement `DataDependentResourceAnalyzable` instead.

    Applicable Encodings
    --------------------
    - AmplitudeEncoding
    - AngleEncoding
    - IQPEncoding
    - ZZFeatureMap
    - PauliFeatureMap
    - HardwareEfficientEncoding
    - DataReuploading
    - HigherOrderAngleEncoding
    - HamiltonianEncoding
    - QAOAEncoding
    - CovariantFeatureMap

    Examples
    --------
    **Checking if an encoding implements this protocol:**

    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.core.protocols import ResourceAnalyzable
    >>>
    >>> enc = IQPEncoding(n_features=4, reps=2)
    >>> isinstance(enc, ResourceAnalyzable)
    True

    **Using the protocol methods:**

    >>> summary = enc.resource_summary()
    >>> summary['n_qubits']
    4
    >>> summary['is_entangling']
    True
    >>>
    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['total']
    28

    **Writing generic resource analysis code:**

    >>> def compare_resources(encodings: list) -> None:
    ...     for enc in encodings:
    ...         if isinstance(enc, ResourceAnalyzable):
    ...             summary = enc.resource_summary()
    ...             print(f"{enc.__class__.__name__}: {summary['gate_counts']['total']} gates")
    ...         else:
    ...             print(f"{enc.__class__.__name__}: Resource analysis not supported")

    See Also
    --------
    DataDependentResourceAnalyzable : For encodings with data-dependent resources.
    encoding_atlas.core.properties.EncodingProperties : Property data structure.
    """

    def resource_summary(self) -> dict[str, Any]:
        """Get comprehensive resource summary for this encoding.

        This method provides a complete view of circuit resources, encoding
        characteristics, and hardware requirements in a single call.

        Returns
        -------
        dict[str, Any]
            Dictionary containing at minimum:

            **Required Fields** (guaranteed present):

            - ``n_qubits`` : int
                Number of qubits required for the circuit.
            - ``n_features`` : int
                Number of classical features encoded.
            - ``depth`` : int
                Circuit depth (number of time steps/layers).
            - ``gate_counts`` : dict[str, int]
                Breakdown of gates by type, including:
                - ``total``: Total gate count
                - ``single_qubit``: Total single-qubit gates
                - ``two_qubit``: Total two-qubit gates
                - Additional encoding-specific gate types
            - ``is_entangling`` : bool
                Whether the encoding creates quantum entanglement.
            - ``simulability`` : Literal["simulable", "conditionally_simulable", "not_simulable"]
                Classical simulability status of the encoding.

            **Optional Fields** (encoding-specific):

            - ``reps`` : int
                Number of circuit repetitions (if applicable).
            - ``entanglement`` : str
                Entanglement topology (if applicable).
            - ``entanglement_pairs`` : list[tuple[int, int]]
                Qubit pairs with entangling interactions.
            - ``trainability_estimate`` : float
                Estimated trainability score (0.0-1.0).
            - ``hardware_requirements`` : dict
                Hardware compatibility information.

        Examples
        --------
        >>> enc = IQPEncoding(n_features=4, reps=2)
        >>> summary = enc.resource_summary()
        >>> summary['n_qubits']
        4
        >>> summary['gate_counts']['total']
        28
        >>> summary['is_entangling']
        True

        Notes
        -----
        The exact fields returned may vary by encoding, but the required
        fields listed above are guaranteed to be present for any encoding
        implementing this protocol.
        """
        ...

    def gate_count_breakdown(self) -> dict[str, int]:
        """Get detailed breakdown of gate counts by type.

        This method provides granular gate count information for hardware
        resource estimation, transpilation planning, and encoding comparison.

        Returns
        -------
        dict[str, int]
            Dictionary with gate counts. Guaranteed fields:

            - ``total`` : int
                Total number of gates in the circuit.
            - ``total_single_qubit`` : int
                Total number of single-qubit gates.
            - ``total_two_qubit`` : int
                Total number of two-qubit gates.

            Additional encoding-specific fields may include:

            - ``hadamard`` : int - Hadamard gates
            - ``rx``, ``ry``, ``rz`` : int - Rotation gates
            - ``cnot`` : int - CNOT gates
            - ``cz`` : int - Controlled-Z gates
            - ``zz`` : int - ZZ interaction gates

        Examples
        --------
        >>> enc = IQPEncoding(n_features=4, reps=2)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['total']
        28
        >>> breakdown['total_single_qubit']
        16
        >>> breakdown['total_two_qubit']
        12

        Notes
        -----
        The specific gate types included depend on the encoding. Use
        `breakdown.get('gate_name', 0)` to safely access optional fields.
        """
        ...


@runtime_checkable
class DataDependentResourceAnalyzable(Protocol):
    """Protocol for encodings where resource usage depends on input data.

    This protocol defines a standardized interface for querying circuit
    resources when the actual resource usage varies depending on the
    specific input data being encoded.

    **When to Implement**

    Implement this protocol when your encoding's gate count or structure
    changes based on input values. The canonical example is `BasisEncoding`,
    which applies X gates only where input values exceed a threshold.

    **Difference from ResourceAnalyzable**

    - `ResourceAnalyzable`: Resources fixed at construction time
      - `resource_summary()` takes no parameters
    - `DataDependentResourceAnalyzable`: Resources vary with input
      - `resource_summary(x)` requires input data

    Applicable Encodings
    --------------------
    - BasisEncoding

    Examples
    --------
    **Checking if an encoding implements this protocol:**

    >>> from encoding_atlas import BasisEncoding
    >>> from encoding_atlas.core.protocols import DataDependentResourceAnalyzable
    >>>
    >>> enc = BasisEncoding(n_features=4)
    >>> isinstance(enc, DataDependentResourceAnalyzable)
    True

    **Using the protocol methods:**

    >>> import numpy as np
    >>> x = np.array([0.1, 0.9, 0.3, 0.8])  # 2 values > 0.5 threshold
    >>> actual_gates = enc.actual_gate_count(x)
    >>> actual_gates
    2
    >>>
    >>> summary = enc.resource_summary(x)
    >>> summary['actual_x_gates']
    2

    **Comparing theoretical vs actual resources:**

    >>> # Theoretical maximum (worst case)
    >>> theoretical = enc.properties.gate_count
    >>> # Actual for specific input
    >>> actual = enc.actual_gate_count(x)
    >>> print(f"Theoretical max: {theoretical}, Actual: {actual}")
    Theoretical max: 4, Actual: 2

    See Also
    --------
    ResourceAnalyzable : For encodings with data-independent resources.
    """

    def resource_summary(self, x: ArrayLike) -> dict[str, Any]:
        """Get resource summary for specific input data.

        Parameters
        ----------
        x : ArrayLike
            Input data to analyze. Shape should be (n_features,) or
            (1, n_features) for a single sample.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the same structure as
            `ResourceAnalyzable.resource_summary()`, but with actual
            values computed for the given input data.

            Additional data-dependent fields:

            - ``actual_gate_count`` : int
                Actual number of gates for this specific input.
            - ``actual_x_gates`` : int (for BasisEncoding)
                Number of X gates actually applied.
            - ``sparsity`` : float
                Ratio of gates used vs theoretical maximum.

        Examples
        --------
        >>> import numpy as np
        >>> enc = BasisEncoding(n_features=4)
        >>> x = np.array([0.1, 0.9, 0.3, 0.8])
        >>> summary = enc.resource_summary(x)
        >>> summary['actual_x_gates']
        2
        """
        ...

    def actual_gate_count(self, x: ArrayLike) -> int:
        """Get actual gate count for specific input data.

        Parameters
        ----------
        x : ArrayLike
            Input data to analyze.

        Returns
        -------
        int
            Exact number of gates that will be applied for this input.

        Examples
        --------
        >>> import numpy as np
        >>> enc = BasisEncoding(n_features=4, threshold=0.5)
        >>> # Only values > 0.5 get X gates
        >>> x_sparse = np.array([0.1, 0.2, 0.3, 0.4])  # All <= 0.5
        >>> enc.actual_gate_count(x_sparse)
        0
        >>> x_dense = np.array([0.6, 0.7, 0.8, 0.9])  # All > 0.5
        >>> enc.actual_gate_count(x_dense)
        4
        """
        ...


# =============================================================================
# Entanglement Query Protocol
# =============================================================================


@runtime_checkable
class EntanglementQueryable(Protocol):
    """Protocol for encodings with queryable entanglement structure.

    This protocol defines a standardized interface for querying which
    qubit pairs have entangling interactions in the encoding circuit.
    This information is valuable for:

    - Understanding encoding expressivity
    - Hardware mapping and transpilation
    - Debugging circuit structure
    - Theoretical analysis of encoding power

    **When to Implement**

    Implement this protocol when your encoding creates entanglement
    between qubits and you want to expose the entanglement topology
    to users.

    **When NOT to Implement**

    Do NOT implement this protocol for non-entangling encodings
    (e.g., `AngleEncoding`, which applies only single-qubit gates).

    Applicable Encodings
    --------------------
    - IQPEncoding
    - ZZFeatureMap
    - PauliFeatureMap
    - HardwareEfficientEncoding
    - DataReuploading
    - QAOAEncoding
    - CovariantFeatureMap
    - HamiltonianEncoding (topology-dependent)

    Examples
    --------
    **Checking if an encoding implements this protocol:**

    >>> from encoding_atlas import IQPEncoding, AngleEncoding
    >>> from encoding_atlas.core.protocols import EntanglementQueryable
    >>>
    >>> iqp = IQPEncoding(n_features=4)
    >>> isinstance(iqp, EntanglementQueryable)
    True
    >>>
    >>> angle = AngleEncoding(n_features=4)
    >>> isinstance(angle, EntanglementQueryable)
    False

    **Querying entanglement pairs:**

    >>> pairs = iqp.get_entanglement_pairs()
    >>> pairs
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    >>> len(pairs)
    6

    **Using entanglement info for analysis:**

    >>> def connectivity_analysis(enc) -> dict:
    ...     if not isinstance(enc, EntanglementQueryable):
    ...         return {"entangling": False, "pairs": []}
    ...     pairs = enc.get_entanglement_pairs()
    ...     return {
    ...         "entangling": True,
    ...         "pairs": pairs,
    ...         "n_pairs": len(pairs),
    ...         "max_connectivity": max(
    ...             sum(1 for p in pairs if q in p)
    ...             for q in range(enc.n_qubits)
    ...         ) if pairs else 0
    ...     }

    See Also
    --------
    ResourceAnalyzable : Often implemented together for full analysis.
    """

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get qubit pairs that have entangling interactions.

        Returns the list of qubit index pairs where two-qubit entangling
        gates (CNOT, CZ, ZZ, etc.) are applied. Each pair represents a
        direct entangling interaction in the circuit.

        Returns
        -------
        list[tuple[int, int]]
            List of (qubit_i, qubit_j) pairs where entangling gates
            are applied. Pairs are typically ordered with qubit_i < qubit_j.

            - For "linear" entanglement: [(0,1), (1,2), (2,3), ...]
            - For "circular" entanglement: linear + [(n-1, 0)]
            - For "full" entanglement: all pairs [(i,j) for i<j]

            Returns an empty list if the encoding creates no entanglement.

        Examples
        --------
        >>> from encoding_atlas import HardwareEfficientEncoding
        >>>
        >>> # Linear entanglement (nearest-neighbor)
        >>> enc_linear = HardwareEfficientEncoding(n_features=4, entanglement='linear')
        >>> enc_linear.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]
        >>>
        >>> # Circular entanglement (with wrap-around)
        >>> enc_circ = HardwareEfficientEncoding(n_features=4, entanglement='circular')
        >>> enc_circ.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>>
        >>> # Full entanglement (all-to-all)
        >>> enc_full = HardwareEfficientEncoding(n_features=4, entanglement='full')
        >>> enc_full.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        Notes
        -----
        The returned list represents the entanglement topology per layer.
        For encodings with multiple repetitions, the same pairs may be
        applied multiple times. Use `resource_summary()` or
        `gate_count_breakdown()` to get the total number of entangling gates.
        """
        ...


# =============================================================================
# Data Transformation Protocol
# =============================================================================


@runtime_checkable
class DataTransformable(Protocol):
    """Protocol for encodings that transform input data before encoding.

    This protocol defines a standardized interface for exposing the
    data preprocessing logic that an encoding applies before quantum
    circuit generation. This is useful for:

    - Understanding how data is transformed
    - Debugging unexpected encoding behavior
    - Pre-computing transformations for efficiency
    - Validating data compatibility

    **When to Implement**

    Implement this protocol when your encoding applies non-trivial
    preprocessing to input data before encoding into quantum gates.

    **Common Transformations**

    - **BasisEncoding**: Binarization (continuous -> 0/1)
    - **AmplitudeEncoding**: Normalization (vector -> unit norm)
    - **HigherOrderAngleEncoding**: Feature product computation

    Applicable Encodings
    --------------------
    - AmplitudeEncoding (normalization to unit vector)
    - BasisEncoding (binarization via threshold)

    Examples
    --------
    **Checking if an encoding implements this protocol:**

    >>> from encoding_atlas import BasisEncoding, AmplitudeEncoding
    >>> from encoding_atlas.core.protocols import DataTransformable
    >>>
    >>> basis = BasisEncoding(n_features=4)
    >>> isinstance(basis, DataTransformable)
    True
    >>>
    >>> amp = AmplitudeEncoding(n_features=4)
    >>> isinstance(amp, DataTransformable)
    True

    **Using transform_input to understand preprocessing:**

    >>> import numpy as np
    >>>
    >>> # BasisEncoding binarization
    >>> basis = BasisEncoding(n_features=4, threshold=0.5)
    >>> x = np.array([0.1, 0.9, 0.3, 0.8])
    >>> basis.transform_input(x)
    array([0, 1, 0, 1])
    >>>
    >>> # AmplitudeEncoding normalization
    >>> amp = AmplitudeEncoding(n_features=4)
    >>> x = np.array([1.0, 2.0, 2.0, 0.0])
    >>> transformed = amp.transform_input(x)
    >>> np.linalg.norm(transformed)  # Unit norm
    1.0

    **Debugging encoding behavior:**

    >>> # Why did my circuit behave unexpectedly?
    >>> x_input = np.array([0.49, 0.51, 0.50, 0.52])
    >>> x_binary = basis.transform_input(x_input)
    >>> print(f"Input: {x_input}")
    >>> print(f"Binary: {x_binary}")  # Shows threshold behavior
    Input: [0.49 0.51 0.5  0.52]
    Binary: [0 1 0 1]

    See Also
    --------
    BasisEncoding.binarize : Specific binarization method.
    AmplitudeEncoding : Normalization behavior documentation.
    """

    def transform_input(self, x: ArrayLike) -> NDArray[np.floating[Any]]:
        """Transform input data according to encoding requirements.

        Applies the encoding's preprocessing logic to input data. This
        method exposes what happens to data before it's encoded into
        quantum gates, allowing users to inspect and understand the
        transformation.

        Parameters
        ----------
        x : ArrayLike
            Raw input data to transform. Shape should be (n_features,)
            or (n_samples, n_features).

        Returns
        -------
        NDArray
            Transformed data ready for quantum encoding.
            Shape matches input shape.

            The specific transformation depends on the encoding:

            - **BasisEncoding**: Returns int array of 0s and 1s
            - **AmplitudeEncoding**: Returns float array with unit L2 norm

        Raises
        ------
        ValueError
            If input shape doesn't match n_features.
        ValueError
            If input contains invalid values (NaN, inf).
        TypeError
            If input cannot be converted to numeric array.

        Examples
        --------
        **BasisEncoding binarization:**

        >>> basis = BasisEncoding(n_features=4, threshold=0.5)
        >>> x = np.array([0.1, 0.6, 0.4, 0.9])
        >>> basis.transform_input(x)
        array([0, 1, 0, 1])

        **AmplitudeEncoding normalization:**

        >>> amp = AmplitudeEncoding(n_features=4, normalize=True)
        >>> x = np.array([3.0, 4.0, 0.0, 0.0])
        >>> result = amp.transform_input(x)
        >>> result
        array([0.6, 0.8, 0.0, 0.0])
        >>> np.linalg.norm(result)
        1.0

        Notes
        -----
        This method should be idempotent for most encodings - applying
        it twice should give the same result as applying it once
        (for already-transformed data).
        """
        ...


# =============================================================================
# Type Guards
# =============================================================================


def is_resource_analyzable(obj: object) -> bool:
    """Check if an object implements the ResourceAnalyzable protocol.

    This is a convenience function equivalent to
    ``isinstance(obj, ResourceAnalyzable)`` but may be more readable
    in some contexts.

    Parameters
    ----------
    obj : object
        Object to check (typically an encoding instance).

    Returns
    -------
    bool
        True if obj implements ResourceAnalyzable, False otherwise.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.core.protocols import is_resource_analyzable
    >>>
    >>> enc = IQPEncoding(n_features=4)
    >>> is_resource_analyzable(enc)
    True

    See Also
    --------
    ResourceAnalyzable : The protocol being checked.
    """
    return isinstance(obj, ResourceAnalyzable)


def is_data_dependent_resource_analyzable(obj: object) -> bool:
    """Check if an object implements the DataDependentResourceAnalyzable protocol.

    Parameters
    ----------
    obj : object
        Object to check (typically an encoding instance).

    Returns
    -------
    bool
        True if obj implements DataDependentResourceAnalyzable, False otherwise.

    Examples
    --------
    >>> from encoding_atlas import BasisEncoding
    >>> from encoding_atlas.core.protocols import is_data_dependent_resource_analyzable
    >>>
    >>> enc = BasisEncoding(n_features=4)
    >>> is_data_dependent_resource_analyzable(enc)
    True

    See Also
    --------
    DataDependentResourceAnalyzable : The protocol being checked.
    """
    return isinstance(obj, DataDependentResourceAnalyzable)


def is_entanglement_queryable(obj: object) -> bool:
    """Check if an object implements the EntanglementQueryable protocol.

    Parameters
    ----------
    obj : object
        Object to check (typically an encoding instance).

    Returns
    -------
    bool
        True if obj implements EntanglementQueryable, False otherwise.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding, AngleEncoding
    >>> from encoding_atlas.core.protocols import is_entanglement_queryable
    >>>
    >>> is_entanglement_queryable(IQPEncoding(n_features=4))
    True
    >>> is_entanglement_queryable(AngleEncoding(n_features=4))
    False

    See Also
    --------
    EntanglementQueryable : The protocol being checked.
    """
    return isinstance(obj, EntanglementQueryable)


def is_data_transformable(obj: object) -> bool:
    """Check if an object implements the DataTransformable protocol.

    Parameters
    ----------
    obj : object
        Object to check (typically an encoding instance).

    Returns
    -------
    bool
        True if obj implements DataTransformable, False otherwise.

    Examples
    --------
    >>> from encoding_atlas import BasisEncoding, AmplitudeEncoding
    >>> from encoding_atlas.core.protocols import is_data_transformable
    >>>
    >>> is_data_transformable(BasisEncoding(n_features=4))
    True
    >>> is_data_transformable(AmplitudeEncoding(n_features=4))
    True

    See Also
    --------
    DataTransformable : The protocol being checked.
    """
    return isinstance(obj, DataTransformable)
