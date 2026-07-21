"""Type definitions for the encoding_atlas package.

This module provides type definitions used throughout the encoding_atlas package,
including type aliases, TypedDicts, and Protocols for consistent typing across
all encoding implementations.

Type Definitions
----------------
- **BackendType**: Literal type for supported quantum backends
- **CircuitType**: Union type for quantum circuit objects across backends
- **FloatArray/IntArray**: NumPy array type aliases
- **FeatureData**: Input data type for encoding methods
- **GateCountBreakdownBase**: Base TypedDict for gate count analysis

Protocols
---------
- **BaseEncodingProtocol**: Protocol defining the encoding interface
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import cirq
    import pennylane as qml
    from qiskit import QuantumCircuit
    from typing_extensions import TypeAlias

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Type aliases
    "BackendType",
    "CircuitType",
    "FloatArray",
    "IntArray",
    "FeatureData",
    "E",
    # TypedDicts
    "GateCountBreakdownBase",
    # Protocols
    "BaseEncodingProtocol",
]

# =============================================================================
# Type Aliases
# =============================================================================

BackendType = Literal["pennylane", "qiskit", "cirq"]
"""Supported quantum computing backends."""

CircuitType = Union["qml.QNode", "QuantumCircuit", "cirq.Circuit", Any]
"""Union type for quantum circuit objects from different backends."""

FloatArray: TypeAlias = NDArray[np.floating[Any]]
"""NumPy array of floating-point values."""

IntArray: TypeAlias = NDArray[np.integer[Any]]
"""NumPy array of integer values."""

FeatureData = Union[FloatArray, list[float], tuple[float, ...]]
"""Input data types accepted by encoding methods."""

E = TypeVar("E", bound="BaseEncodingProtocol")
"""Type variable for encoding classes."""


# =============================================================================
# Gate Count Type Definitions
# =============================================================================


class GateCountBreakdownBase(TypedDict):
    """Base type definition for gate count breakdown dictionaries.

    This TypedDict defines the **minimum required fields** that all encoding
    implementations must return from their ``gate_count_breakdown()`` method.
    Individual encodings may extend this with encoding-specific fields (e.g.,
    specific gate types like ``rx``, ``hadamard``, ``zz``).

    Having a consistent base ensures that:

    1. All encodings can be compared using common metrics
    2. Resource estimation code can rely on these fields being present
    3. IDE autocompletion works for the common fields

    Attributes
    ----------
    total_single_qubit : int
        Total number of single-qubit gates in the circuit.
        Includes all rotation gates (RX, RY, RZ), Hadamard, Phase, etc.

    total_two_qubit : int
        Total number of two-qubit gates in the circuit.
        Includes CNOT, CZ, ZZ interactions, etc.

    total : int
        Total gate count (total_single_qubit + total_two_qubit).

    Examples
    --------
    All encodings return at least these fields:

    >>> enc = SomeEncoding(n_features=4)
    >>> breakdown = enc.gate_count_breakdown()
    >>> breakdown['total_single_qubit']
    8
    >>> breakdown['total_two_qubit']
    6
    >>> breakdown['total']
    14

    Encodings may include additional fields:

    >>> breakdown.get('rx', 0)  # Encoding-specific field
    0
    >>> breakdown.get('ry', 0)  # Encoding-specific field
    8

    Notes
    -----
    This base type uses ``TypedDict`` which provides:

    - Static type checking for dictionary keys and values
    - IDE autocompletion for known fields
    - Runtime compatibility with regular dictionaries

    Individual encoding modules define their own ``GateCountBreakdown``
    TypedDict that includes these base fields plus encoding-specific ones.
    """

    total_single_qubit: int
    """Total number of single-qubit gates."""

    total_two_qubit: int
    """Total number of two-qubit gates."""

    total: int
    """Total gate count."""


# =============================================================================
# Protocols
# =============================================================================


class BaseEncodingProtocol(Protocol):
    """Protocol defining the encoding interface.

    All encoding classes should implement this protocol to ensure
    consistent API across the library. This enables type-safe code
    that works with any encoding implementation.

    Required Properties
    -------------------
    n_features : int
        Number of classical features the encoding accepts.
    n_qubits : int
        Number of qubits in the generated circuit.
    depth : int
        Circuit depth (number of layers).

    Required Methods
    ----------------
    get_circuit(x, backend)
        Generate a quantum circuit for the given input data.
    gate_count_breakdown()
        Return a dictionary with gate count statistics.
    """

    @property
    def n_features(self) -> int:
        """Number of classical features to encode."""
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit."""
        ...

    @property
    def depth(self) -> int:
        """Circuit depth."""
        ...

    def get_circuit(self, x: FeatureData, backend: BackendType = "pennylane") -> Any:
        """Generate quantum circuit for input data.

        Parameters
        ----------
        x : FeatureData
            Input features to encode.
        backend : BackendType, default="pennylane"
            Target quantum computing framework.

        Returns
        -------
        Any
            Circuit in the specified backend's format.
        """
        ...

    def gate_count_breakdown(self) -> GateCountBreakdownBase:
        """Get a breakdown of gate counts by type.

        Returns
        -------
        GateCountBreakdownBase
            Dictionary containing at minimum:
            - ``total_single_qubit``: Total single-qubit gates
            - ``total_two_qubit``: Total two-qubit gates
            - ``total``: Total gate count

            May include additional encoding-specific fields.
        """
        ...
