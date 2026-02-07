"""Simulability analysis for quantum encodings.

This module determines whether quantum encoding circuits can be efficiently
simulated on classical computers. This is important for:

- **Quantum Advantage Assessment**: Non-simulable circuits may offer quantum speedup
- **Development Workflow**: Simulable circuits are easier to debug and test
- **Resource Planning**: Determines whether classical or quantum resources are needed
- **Algorithm Selection**: Guides choice of classical vs quantum approaches

The main function :func:`check_simulability` analyzes circuit structure and
returns a simulability classification with detailed reasoning.

Mathematical Background
-----------------------
Several classes of quantum circuits are known to be efficiently simulable:

1. **Clifford Circuits** (Gottesman-Knill Theorem): Circuits using only
   H, S, CNOT, CZ, and Pauli gates can be simulated in polynomial time
   using the stabilizer formalism.

2. **Low-Entanglement Circuits**: Circuits where entanglement across any
   bipartition is bounded can be simulated using tensor networks (MPS/MPO).

3. **Matchgate Circuits**: Circuits with nearest-neighbor matchgates on
   a line are classically simulable.

4. **Product State Circuits**: Circuits that never create entanglement
   (separable encodings) are trivially simulable as independent single-qubit
   systems.

Simulability Classes
--------------------
The analysis returns one of three classifications:

- ``simulable``: Circuit can always be efficiently simulated classically.
  Examples: Clifford circuits, product state circuits, circuits with
  O(log n) entanglement.

- ``conditionally_simulable``: Circuit may be simulable depending on
  input data, circuit parameters, or specific structure.
  Examples: Circuits with linear/circular entanglement topology where
  tensor network (MPS) simulation may be efficient if entanglement
  entropy is bounded.

- ``not_simulable``: Circuit is believed to be hard to simulate classically.
  Examples: IQP circuits, random circuits with high entanglement,
  circuits with many T gates.

References
----------
.. [1] Gottesman, D. (1998). "The Heisenberg representation of quantum
       computers." arXiv:quant-ph/9807006.

.. [2] Jozsa, R., & Linden, N. (2003). "On the role of entanglement in
       quantum-computational speed-up." Proc. R. Soc. A, 459(2036), 2011-2032.

.. [3] Vidal, G. (2003). "Efficient classical simulation of slightly
       entangled quantum computations." Physical Review Letters, 91(14), 147902.

.. [4] Bremner, M. J., Montanaro, A., & Shepherd, D. J. (2016). "Average-case
       complexity versus approximate simulation of commuting quantum
       computations." Physical Review Letters, 117(8), 080501.

Examples
--------
>>> from encoding_atlas import AngleEncoding, IQPEncoding
>>> from encoding_atlas.analysis import check_simulability

Non-entangling encoding is simulable:

>>> enc_angle = AngleEncoding(n_features=4)
>>> result = check_simulability(enc_angle)
>>> print(f"Simulable: {result['is_simulable']}")
Simulable: True
>>> print(f"Class: {result['simulability_class']}")
Class: simulable

IQP encoding is not efficiently simulable:

>>> enc_iqp = IQPEncoding(n_features=4, reps=2)
>>> result = check_simulability(enc_iqp)
>>> print(f"Simulable: {result['is_simulable']}")
Simulable: False
>>> print(f"Class: {result['simulability_class']}")
Class: not_simulable

Quick check without details:

>>> result = check_simulability(enc_angle, detailed=False)
>>> if result['is_simulable']:
...     print("Can use efficient classical simulation")
Can use efficient classical simulation

See Also
--------
count_resources : Get detailed gate counts for an encoding.
compute_expressibility : Measure Hilbert space coverage.
compute_entanglement_capability : Measure entanglement generation.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.exceptions import AnalysisError

if TYPE_CHECKING:
    pass

# =============================================================================
# Module-Level Logger
# =============================================================================

_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "check_simulability",
    "get_simulability_reason",
    "is_clifford_circuit",
    "is_matchgate_circuit",
    "estimate_entanglement_bound",
    "SimulabilityResult",
]

# =============================================================================
# Type Definitions
# =============================================================================


class SimulabilityResult(TypedDict):
    """Result of simulability analysis.

    Attributes
    ----------
    is_simulable : bool
        True if the encoding circuit can be efficiently simulated classically.
    simulability_class : str
        One of "simulable", "conditionally_simulable", "not_simulable".
    reason : str
        Human-readable explanation of the classification.
    details : dict[str, Any]
        Additional analysis details including:

        - is_entangling: Whether circuit creates entanglement
        - is_clifford: Whether circuit uses only Clifford gates
        - is_matchgate: Whether circuit uses only matchgate operations
        - entanglement_pattern: Description of entanglement structure
        - two_qubit_gate_count: Number of two-qubit gates
        - n_qubits: Number of qubits
        - declared_simulability: Simulability from encoding properties

    recommendations : list[str]
        Suggestions for simulation approaches.
    """

    is_simulable: bool
    simulability_class: Literal["simulable", "conditionally_simulable", "not_simulable"]
    reason: str
    details: dict[str, Any]
    recommendations: list[str]


# =============================================================================
# Constants
# =============================================================================

# Gate sets for simulability classification
# Clifford gates can be efficiently simulated using stabilizer formalism
_CLIFFORD_GATES: frozenset[str] = frozenset(
    {
        # Single-qubit Clifford gates
        "h",
        "hadamard",
        "s",
        "sdg",
        "s_dagger",
        "sdagger",
        "x",
        "y",
        "z",
        "paulix",
        "pauliy",
        "pauliz",
        # Two-qubit Clifford gates
        "cnot",
        "cx",
        "cz",
        "swap",
        # Identity (trivially Clifford)
        "id",
        "i",
        "identity",
    }
)

# Non-Clifford gates that break efficient stabilizer simulation
_NON_CLIFFORD_GATES: frozenset[str] = frozenset(
    {
        # T gate and inverse
        "t",
        "tdg",
        "t_dagger",
        "tdagger",
        # Parameterized rotation gates (non-Clifford except at special angles)
        "rx",
        "ry",
        "rz",
        "u1",
        "u2",
        "u3",
        "u",
        "p",
        "phase",
        # Controlled rotations
        "crx",
        "cry",
        "crz",
        "cu1",
        "cu2",
        "cu3",
        "cu",
        "cp",
        "cphase",
        # Multi-qubit rotations
        "rxx",
        "ryy",
        "rzz",
        "rzx",
        # Other non-Clifford
        "ccx",
        "toffoli",
        "ccz",
    }
)

# Matchgate set for classical simulability on nearest-neighbor topologies
# Matchgates are a specific class of two-qubit gates that preserve particle
# number parity and can be efficiently simulated classically when applied
# in nearest-neighbor fashion on a line topology.
#
# Reference: Jozsa & Miyake (2008), "Matchgates and classical simulation of
#            quantum circuits" Proc. R. Soc. A 464, 3089-3106.
_MATCHGATE_GATES: frozenset[str] = frozenset(
    {
        # Core matchgates (preserve fermionic parity)
        "iswap",
        "fswap",
        "fermionic_swap",
        "fswapgate",
        # XY interaction gates (parameterized matchgates)
        "xy",
        "xygate",
        "givens",
        "givensrotation",
        # Single-qubit gates compatible with matchgate circuits
        # (these are "free" operations in the matchgate formalism)
        "rx",
        "rz",
        "x",
        "z",
        "h",
        "hadamard",
    }
)

# Known matchgate-based encodings (efficient classical simulation possible)
_MATCHGATE_ENCODING_NAMES: frozenset[str] = frozenset(
    {
        "givensencoding",
        "particlepreservingencoding",
        "fermionicencoding",
        "matchgateencoding",
    }
)

# Threshold for "small" circuits that can always be simulated via brute force
# 2^20 = ~1 million amplitudes, requiring ~16 MB for statevector
_SMALL_CIRCUIT_QUBITS: int = 20

# Warning threshold for memory-intensive simulations
# At 25+ qubits, memory requirements become significant (512 MB+),
# warranting a user warning. This is separate from _SMALL_CIRCUIT_QUBITS
# which defines the upper limit of feasibility.
#
# Memory requirements by qubit count:
#   15 qubits: 2^15 * 16 bytes =   512 KB
#   20 qubits: 2^20 * 16 bytes =    16 MB
#   25 qubits: 2^25 * 16 bytes =   512 MB
#   30 qubits: 2^30 * 16 bytes =    16 GB
_MEMORY_WARNING_QUBITS: int = 25

# Numerical constants for entropy calculations
_ENTROPY_EPSILON: float = 1e-15


# =============================================================================
# Main Public Functions
# =============================================================================


def check_simulability(
    encoding: BaseEncoding,
    detailed: bool = True,
) -> SimulabilityResult:
    """Check whether an encoding circuit is classically simulable.

    Analyzes the encoding's circuit structure to determine if it can
    be efficiently simulated on a classical computer. This analysis
    considers entanglement structure, gate set, and circuit topology.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to analyze.
    detailed : bool, default=True
        If True, include detailed analysis in the result.
        If False, return minimal result with just classification.

    Returns
    -------
    SimulabilityResult
        Dictionary containing:

        - ``is_simulable`` : bool
            True if efficiently classically simulable.
        - ``simulability_class`` : str
            One of "simulable", "conditionally_simulable", "not_simulable".
        - ``reason`` : str
            Human-readable explanation of the classification.
        - ``details`` : dict
            Additional analysis details (if detailed=True):

            - is_entangling: Whether circuit creates entanglement
            - is_clifford: Whether circuit uses only Clifford gates
            - entanglement_pattern: Description of entanglement structure
            - two_qubit_gate_count: Number of two-qubit gates
            - n_qubits: Number of qubits
            - declared_simulability: Simulability from encoding properties

        - ``recommendations`` : list[str]
            Suggestions for simulation approaches.

    Raises
    ------
    AnalysisError
        If encoding is not a valid BaseEncoding instance or analysis fails.

    Examples
    --------
    Check a non-entangling encoding:

    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> result = check_simulability(enc)
    >>> print(result['simulability_class'])
    simulable
    >>> print(result['reason'])
    Encoding produces only product states (no entanglement)

    Check an entangling encoding:

    >>> from encoding_atlas import IQPEncoding
    >>> enc = IQPEncoding(n_features=4, reps=2)
    >>> result = check_simulability(enc)
    >>> print(result['simulability_class'])
    not_simulable
    >>> print(result['recommendations'])
    ['Use statevector simulation for small instances (< 20 qubits)', ...]

    Quick check without details:

    >>> result = check_simulability(enc, detailed=False)
    >>> if result['is_simulable']:
    ...     print("Can use classical simulation")

    Notes
    -----
    **Simulability Classes**:

    - ``simulable``: Circuit can always be efficiently simulated.
      Examples: Clifford circuits, product state circuits.

    - ``conditionally_simulable``: Circuit may be simulable depending on
      input data, circuit parameters, or specific structure.
      Examples: Circuits with linear/circular entanglement topology.

    - ``not_simulable``: Circuit is believed to be hard to simulate
      classically. Examples: IQP circuits, random circuits with high
      entanglement.

    **Limitations**:

    This analysis is based on known theoretical results and heuristics.
    It may be conservative (classifying some simulable circuits as
    not_simulable) but aims not to falsely claim simulability.

    See Also
    --------
    is_clifford_circuit : Check if circuit uses only Clifford gates.
    estimate_entanglement_bound : Estimate entanglement entropy bound.
    count_resources : Get detailed gate counts.
    """
    # Validate encoding
    if not isinstance(encoding, BaseEncoding):
        raise AnalysisError(
            f"Expected BaseEncoding instance, got {type(encoding).__name__}",
            details={"actual_type": type(encoding).__name__},
        )

    _logger.debug(
        "Checking simulability for %s encoding",
        encoding.__class__.__name__,
    )

    # Gather circuit properties
    try:
        props = encoding.properties
        n_qubits = encoding.n_qubits
    except Exception as e:
        raise AnalysisError(
            f"Failed to access encoding properties: {e}",
            details={"error": str(e)},
        ) from e

    is_entangling = props.is_entangling
    two_qubit_gates = props.two_qubit_gates
    declared_simulability = props.simulability

    # Initialize result containers
    recommendations: list[str] = []

    # =========================================================================
    # Compute circuit properties needed for decision logic
    # These are always computed because the simulability classification
    # depends on them regardless of whether detailed output is requested.
    # =========================================================================
    is_clifford = _check_clifford_property(encoding)
    is_matchgate = _check_matchgate_property(encoding)
    entanglement_pattern = _get_entanglement_pattern(encoding)
    non_clifford_analysis = _check_non_clifford_gates(encoding)

    # Build details dict only if detailed output is requested
    if detailed:
        details: dict[str, Any] = {
            "is_entangling": is_entangling,
            "is_clifford": is_clifford,
            "is_matchgate": is_matchgate,
            "entanglement_pattern": entanglement_pattern,
            "two_qubit_gate_count": two_qubit_gates,
            "n_qubits": n_qubits,
            "n_features": encoding.n_features,
            "declared_simulability": declared_simulability,
            "encoding_name": encoding.__class__.__name__,
            # Non-Clifford gate analysis (T-gates break stabilizer simulation)
            "has_non_clifford_gates": non_clifford_analysis["has_non_clifford"],
            "has_t_gates": non_clifford_analysis["has_t_gates"],
            "has_parameterized_rotations": non_clifford_analysis[
                "has_parameterized_rotations"
            ],
        }
        # Only include non_clifford_gates set if non-empty
        if non_clifford_analysis["non_clifford_gates"]:
            details["non_clifford_gates"] = non_clifford_analysis["non_clifford_gates"]
    else:
        details: dict[str, Any] = {}

    # =========================================================================
    # Simulability Decision Logic
    # =========================================================================

    # Case 1: Non-entangling encodings are always efficiently simulable
    # Product states can be simulated as n independent single-qubit systems
    if not is_entangling:
        _logger.debug(
            "Encoding %s is non-entangling (product states only)",
            encoding.__class__.__name__,
        )
        return SimulabilityResult(
            is_simulable=True,
            simulability_class="simulable",
            reason="Encoding produces only product states (no entanglement)",
            details=details,
            recommendations=[
                "Can be simulated as independent single-qubit systems",
                "Classical computation scales linearly with qubit count O(n)",
                "Use standard numerical linear algebra for efficient simulation",
            ],
        )

    # Case 2: Small circuits are always simulable via brute-force statevector
    # Add recommendation with appropriate memory units
    if n_qubits <= _SMALL_CIRCUIT_QUBITS:
        memory_bytes = 2**n_qubits * 16  # Complex128 = 16 bytes per amplitude
        memory_str = _format_memory_size(memory_bytes)
        recommendations.append(
            f"Statevector simulation feasible ({n_qubits} qubits, ~{memory_str} memory)"
        )

    # Case 3: Check if Clifford-only (Gottesman-Knill theorem applies)
    if is_clifford:
        _logger.debug(
            "Encoding %s uses only Clifford gates",
            encoding.__class__.__name__,
        )
        return SimulabilityResult(
            is_simulable=True,
            simulability_class="simulable",
            reason="Circuit uses only Clifford gates (Gottesman-Knill theorem)",
            details=details,
            recommendations=[
                "Use stabilizer formalism for efficient simulation",
                "Scales polynomially O(n²) with qubit count",
                "Consider tools like Stim for high-performance simulation",
            ],
        )

    # Case 4: Check if matchgate circuit with nearest-neighbor topology
    if is_matchgate:
        _logger.debug(
            "Encoding %s uses matchgate operations with linear topology",
            encoding.__class__.__name__,
        )
        return SimulabilityResult(
            is_simulable=True,
            simulability_class="simulable",
            reason=(
                "Matchgate circuit with nearest-neighbor connectivity "
                "(maps to free-fermion system)"
            ),
            details=details,
            recommendations=[
                "Use free-fermion simulation for efficient classical computation",
                "Scales polynomially O(n³) with qubit count",
                "Consider fermionic Gaussian state methods",
            ],
        )

    # Case 5: Entangling circuits with non-Clifford gates
    if is_entangling and two_qubit_gates > 0:
        # Check for special structures that might still be simulable

        # Linear/1D entanglement might be simulable with MPS
        if entanglement_pattern in ("linear", "nearest_neighbor"):
            _logger.debug(
                "Encoding %s has linear entanglement pattern",
                encoding.__class__.__name__,
            )
            recommendations.extend(
                [
                    "Consider MPS (Matrix Product State) simulation",
                    "May be efficient if entanglement entropy is bounded",
                    "Tensor network methods scale with bond dimension",
                ]
            )

            return SimulabilityResult(
                is_simulable=False,
                simulability_class="conditionally_simulable",
                reason=(
                    "Linear entanglement structure may allow tensor network "
                    "simulation if entanglement entropy is bounded"
                ),
                details=details,
                recommendations=recommendations,
            )

        # Circular entanglement is similar to linear
        if entanglement_pattern == "circular":
            _logger.debug(
                "Encoding %s has circular entanglement pattern",
                encoding.__class__.__name__,
            )
            recommendations.extend(
                [
                    "Consider MPS with periodic boundary conditions",
                    "May be efficient for bounded entanglement",
                    "DMRG-style algorithms may be applicable",
                ]
            )

            return SimulabilityResult(
                is_simulable=False,
                simulability_class="conditionally_simulable",
                reason=(
                    "Circular entanglement structure may allow tensor network "
                    "simulation if entanglement entropy is bounded"
                ),
                details=details,
                recommendations=recommendations,
            )

        # Full/all-to-all entanglement - log for debugging
        if entanglement_pattern == "full":
            _logger.debug(
                "Encoding %s has full entanglement pattern",
                encoding.__class__.__name__,
            )

        # =====================================================================
        # Theoretical Complexity Classification
        #
        # For circuits with full, partial, or unknown entanglement topology
        # and non-Clifford gates, no known efficient classical simulation
        # algorithm exists.  The classification is based on the circuit
        # *family's* asymptotic complexity, not a specific instance size.
        #
        # - IQP circuits have provable hardness under polynomial hierarchy
        #   assumptions [Bremner, Montanaro & Shepherd, PRL 117(8), 2016].
        # - General entangling circuits with parameterized rotations are
        #   believed to be hard based on the absence of known efficient
        #   algorithms and connections to #P-hard problems.
        #
        # Small instances (≤ _SMALL_CIRCUIT_QUBITS qubits) can always be
        # brute-force simulated via statevector methods.  This practical
        # feasibility is noted in *recommendations* but does not change the
        # theoretical classification, because simulability is a property of
        # the circuit family, not a single instance.
        # =====================================================================

        # Practical feasibility note for small circuits (recommendation only)
        if n_qubits <= _SMALL_CIRCUIT_QUBITS:
            memory_bytes = 2**n_qubits * 16
            memory_str = _format_memory_size(memory_bytes)
            recommendations.append(
                f"Brute-force statevector simulation is feasible at this "
                f"circuit size ({n_qubits} qubits, ~{memory_str} memory)"
            )

        recommendations.extend(
            [
                f"Use statevector simulation for instances with < {_SMALL_CIRCUIT_QUBITS} qubits",
                "Consider tensor network methods for structured entanglement",
                "May require quantum hardware for large instances",
            ]
        )

        # Determine reason based on circuit characteristics
        encoding_name = encoding.__class__.__name__.lower()
        if "iqp" in encoding_name:
            reason = (
                "IQP circuits have provable classical hardness under "
                "polynomial hierarchy assumptions"
            )
        elif two_qubit_gates > n_qubits * 2:
            reason = (
                f"High entanglement circuit with {two_qubit_gates} two-qubit gates "
                f"and non-Clifford operations"
            )
        else:
            reason = (
                f"Entangling circuit with {two_qubit_gates} two-qubit gates "
                f"and non-Clifford operations"
            )

        return SimulabilityResult(
            is_simulable=False,
            simulability_class="not_simulable",
            reason=reason,
            details=details,
            recommendations=recommendations,
        )

    # Case 6: Fallback - use declared simulability from encoding properties
    _logger.debug(
        "Using declared simulability for %s: %s",
        encoding.__class__.__name__,
        declared_simulability,
    )

    # Validate declared_simulability to ensure it's a valid literal value.
    # This guards against encoding implementations that may return unexpected
    # values from their simulability property. We use a conservative fallback
    # to "not_simulable" to avoid incorrectly claiming simulability.
    #
    # Note: We use explicit string comparisons for proper mypy type narrowing,
    # as mypy cannot narrow types from `in frozenset` checks.
    validated_simulability: Literal[
        "simulable", "conditionally_simulable", "not_simulable"
    ]
    if declared_simulability == "simulable":
        validated_simulability = "simulable"
    elif declared_simulability == "conditionally_simulable":
        validated_simulability = "conditionally_simulable"
    elif declared_simulability == "not_simulable":
        validated_simulability = "not_simulable"
    else:
        _logger.warning(
            "Encoding %s declared invalid simulability value '%s'. "
            "Using conservative fallback 'not_simulable'.",
            encoding.__class__.__name__,
            declared_simulability,
        )
        validated_simulability = "not_simulable"

    is_simulable = validated_simulability == "simulable"

    if not recommendations:
        if is_simulable:
            recommendations = [
                "Use standard simulation methods",
                "Consider backend-specific optimizations",
            ]
        else:
            recommendations = [
                f"Use statevector simulation for < {_SMALL_CIRCUIT_QUBITS} qubits",
                "Consider tensor network methods for structured problems",
            ]

    return SimulabilityResult(
        is_simulable=is_simulable,
        simulability_class=validated_simulability,
        reason=f"Based on encoding's declared simulability: {validated_simulability}",
        details=details,
        recommendations=recommendations,
    )


def get_simulability_reason(encoding: BaseEncoding) -> str:
    """Get a concise explanation of why an encoding is or isn't simulable.

    This is a convenience function that provides a quick summary without
    the full details returned by :func:`check_simulability`.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.

    Returns
    -------
    str
        Human-readable explanation of simulability status.

    Raises
    ------
    AnalysisError
        If encoding is not a valid BaseEncoding instance.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding, IQPEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> print(get_simulability_reason(enc))
    Simulable: Encoding produces only product states (no entanglement)

    >>> enc = IQPEncoding(n_features=4, reps=2)
    >>> reason = get_simulability_reason(enc)
    >>> print(reason.startswith("Not simulable:"))
    True

    See Also
    --------
    check_simulability : Get full simulability analysis with details.
    """
    result = check_simulability(encoding, detailed=False)
    prefix = "Simulable" if result["is_simulable"] else "Not simulable"
    return f"{prefix}: {result['reason']}"


def is_clifford_circuit(encoding: BaseEncoding) -> bool:
    """Check if an encoding uses only Clifford gates.

    Clifford circuits can be efficiently simulated using the stabilizer
    formalism, as proven by the Gottesman-Knill theorem. The Clifford
    gate set includes: H, S, CNOT, CZ, and Pauli gates (X, Y, Z).

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to check.

    Returns
    -------
    bool
        True if the circuit is believed to use only Clifford gates.

    Raises
    ------
    AnalysisError
        If encoding is not a valid BaseEncoding instance.

    Notes
    -----
    This is a conservative check based on encoding properties and
    known gate sets. Some encodings may be classified as non-Clifford
    even if specific parameter choices yield Clifford circuits.

    For example, RZ(π/2) is equivalent to S (a Clifford gate), but
    general RZ(θ) rotations are non-Clifford.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding, BasisEncoding
    >>> # AngleEncoding uses parameterized rotations (non-Clifford in general)
    >>> enc = AngleEncoding(n_features=4)
    >>> is_clifford_circuit(enc)
    False

    See Also
    --------
    check_simulability : Full simulability analysis.
    """
    if not isinstance(encoding, BaseEncoding):
        raise AnalysisError(
            f"Expected BaseEncoding instance, got {type(encoding).__name__}",
            details={"actual_type": type(encoding).__name__},
        )

    return _check_clifford_property(encoding)


def is_matchgate_circuit(encoding: BaseEncoding) -> bool:
    """Check if an encoding uses only matchgate operations.

    Matchgate circuits with nearest-neighbor connectivity on a line topology
    can be efficiently simulated classically in polynomial time. This is based
    on the fact that matchgates preserve fermionic parity and can be mapped
    to free-fermion systems.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to check.

    Returns
    -------
    bool
        True if the circuit is believed to use only matchgate operations
        with appropriate (nearest-neighbor) topology.

    Raises
    ------
    AnalysisError
        If encoding is not a valid BaseEncoding instance.

    Notes
    -----
    **What are Matchgates?**

    Matchgates are a class of two-qubit gates that act on the computational
    basis states in a specific way that preserves particle number parity.
    Common matchgates include:

    - iSWAP: Swaps |01⟩ ↔ |10⟩ with a phase
    - fSWAP (fermionic SWAP): Used in fermionic simulations
    - Givens rotations: Parameterized rotations in the {|01⟩, |10⟩} subspace

    **Simulability Conditions**

    Matchgate circuits are classically simulable when:

    1. All two-qubit gates are matchgates
    2. Qubits are arranged in a line topology
    3. Two-qubit gates act only on nearest neighbors

    **Limitations**

    This check is heuristic and may be conservative. It identifies known
    matchgate-based encodings by name and checks for linear entanglement
    patterns. Some matchgate circuits may not be detected if they don't
    follow recognized naming conventions.

    References
    ----------
    .. [1] Jozsa, R., & Miyake, A. (2008). "Matchgates and classical simulation
           of quantum circuits." Proc. R. Soc. A, 464(2100), 3089-3106.

    .. [2] Terhal, B. M., & DiVincenzo, D. P. (2002). "Classical simulation of
           noninteracting-fermion quantum circuits." Physical Review A, 65(3).

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> is_matchgate_circuit(enc)
    False

    See Also
    --------
    is_clifford_circuit : Check for Clifford-only circuits.
    check_simulability : Full simulability analysis.
    """
    if not isinstance(encoding, BaseEncoding):
        raise AnalysisError(
            f"Expected BaseEncoding instance, got {type(encoding).__name__}",
            details={"actual_type": type(encoding).__name__},
        )

    return _check_matchgate_property(encoding)


def estimate_entanglement_bound(
    encoding: BaseEncoding,
    n_samples: int = 100,
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> float:
    """Estimate upper bound on entanglement entropy.

    Samples random inputs and estimates the maximum entanglement
    entropy across the middle bipartition. Low entanglement suggests
    the circuit may be simulable using tensor network methods.

    The entanglement entropy is computed as the von Neumann entropy
    of the reduced density matrix for one half of the system:

        S = -Tr(ρ_A log₂ ρ_A)

    where ρ_A is obtained by tracing out the other half.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    n_samples : int, default=100
        Number of random inputs to sample. More samples give
        more reliable estimates but take longer.
    seed : int, optional
        Random seed for reproducibility.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Backend to use for circuit simulation.

        - ``"pennylane"``: Uses PennyLane's default.qubit simulator (recommended)
        - ``"qiskit"``: Uses Qiskit's Statevector class
        - ``"cirq"``: Uses Cirq's Simulator for statevector simulation

    Returns
    -------
    float
        Estimated maximum entanglement entropy (in bits).
        For n qubits, the maximum possible value is n/2 bits
        (achieved by maximally entangled states).

    Raises
    ------
    AnalysisError
        If encoding is not valid or simulation fails.
    ValueError
        If an unknown backend is specified.

    Warnings
    --------
    UserWarning
        If the encoding has more than 25 qubits, a warning is issued
        about memory usage for simulation.

    Notes
    -----
    This is a statistical estimate based on random sampling. The true
    maximum may be higher for adversarial inputs not sampled.

    For non-entangling encodings, this will return 0.0.

    For tensor network simulability, entanglement entropy bounded by
    O(log n) suggests efficient MPS simulation is possible.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding, IQPEncoding
    >>> # Non-entangling encoding has zero entanglement
    >>> enc = AngleEncoding(n_features=4)
    >>> entropy = estimate_entanglement_bound(enc, n_samples=50, seed=42)
    >>> entropy < 0.01  # Essentially zero
    True

    >>> # Entangling encoding has non-zero entanglement
    >>> enc = IQPEncoding(n_features=4, reps=2)
    >>> entropy = estimate_entanglement_bound(enc, n_samples=50, seed=42)
    >>> entropy > 0.1
    True

    >>> # Using Cirq backend
    >>> entropy_cirq = estimate_entanglement_bound(enc, n_samples=50, seed=42, backend="cirq")
    >>> entropy_cirq > 0.1
    True

    See Also
    --------
    check_simulability : Full simulability analysis.
    compute_entanglement_capability : Related entanglement measure.
    """
    if not isinstance(encoding, BaseEncoding):
        raise AnalysisError(
            f"Expected BaseEncoding instance, got {type(encoding).__name__}",
            details={"actual_type": type(encoding).__name__},
        )

    # Import simulation utilities
    from encoding_atlas.analysis._utils import (
        create_rng,
        simulate_encoding_statevector,
    )

    # Validate parameters
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")

    # Validate backend
    if backend not in ("pennylane", "qiskit", "cirq"):
        raise ValueError(
            f"backend must be 'pennylane', 'qiskit', or 'cirq', got {backend!r}"
        )

    n_features = encoding.n_features
    n_qubits = encoding.n_qubits

    # Check for non-entangling encodings (optimization)
    if not encoding.properties.is_entangling:
        _logger.debug(
            "Encoding %s is non-entangling, returning entropy = 0",
            encoding.__class__.__name__,
        )
        return 0.0

    # Warn for large qubit counts
    if n_qubits > _MEMORY_WARNING_QUBITS:
        memory_gb = 2**n_qubits * 16 / (1024**3)
        warnings.warn(
            f"Estimating entanglement for {n_qubits} qubits requires "
            f"approximately {memory_gb:.2f} GB memory per sample",
            UserWarning,
            stacklevel=2,
        )

    rng = create_rng(seed)
    max_entropy: float = 0.0
    successful_samples = 0
    first_error: Exception | None = None

    _logger.debug(
        "Estimating entanglement bound for %s with %d samples using %s backend",
        encoding.__class__.__name__,
        n_samples,
        backend,
    )

    # =========================================================================
    # Early validation: Test backend availability with a single simulation
    # This catches import errors and backend configuration issues early
    # rather than after many failed samples.
    # =========================================================================
    test_x = np.zeros(n_features, dtype=np.float64)
    try:
        _ = simulate_encoding_statevector(encoding, test_x, backend=backend)
    except ImportError as e:
        raise AnalysisError(
            f"Backend '{backend}' is not available. Please install the required "
            f"package: {e}",
            details={
                "backend": backend,
                "encoding": encoding.__class__.__name__,
                "error": str(e),
            },
        ) from e
    except Exception as e:
        # Log but continue - the error might be input-dependent
        _logger.debug(
            "Initial test simulation failed (may be input-dependent): %s",
            str(e),
        )

    # =========================================================================
    # Main sampling loop
    # =========================================================================
    for i in range(n_samples):
        # Generate random input parameters
        x = rng.uniform(0, 2 * np.pi, size=n_features).astype(np.float64)

        try:
            # Simulate encoding circuit
            statevector = simulate_encoding_statevector(encoding, x, backend=backend)

            # Compute entanglement entropy for middle bipartition
            mid_qubit = n_qubits // 2
            if mid_qubit == 0:
                mid_qubit = 1  # Ensure at least 1 qubit on each side

            entropy = _compute_bipartite_entropy(statevector, n_qubits, mid_qubit)
            max_entropy = max(max_entropy, entropy)
            successful_samples += 1

            _logger.debug(
                "Sample %d/%d: entropy = %.4f (max so far: %.4f)",
                i + 1,
                n_samples,
                entropy,
                max_entropy,
            )

        except ImportError as e:
            # Backend not available - fail fast
            raise AnalysisError(
                f"Backend '{backend}' is not available: {e}",
                details={"backend": backend, "error": str(e)},
            ) from e

        except (np.linalg.LinAlgError, ValueError) as e:
            # Numerical issues - log and continue
            if first_error is None:
                first_error = e
            _logger.debug(
                "Numerical issue in sample %d: %s",
                i + 1,
                str(e),
            )
            continue

        except Exception as e:
            # Other errors - log and continue, but track
            if first_error is None:
                first_error = e
            _logger.warning(
                "Simulation failed for sample %d/%d: %s",
                i + 1,
                n_samples,
                str(e),
            )
            continue

    # =========================================================================
    # Validate results
    # =========================================================================
    if successful_samples == 0:
        error_detail = str(first_error) if first_error else "Unknown error"
        raise AnalysisError(
            f"All {n_samples} simulations failed for entanglement estimation. "
            f"First error: {error_detail}",
            details={
                "encoding": encoding.__class__.__name__,
                "backend": backend,
                "n_samples": n_samples,
                "first_error": error_detail,
            },
        )

    if successful_samples < n_samples:
        success_rate = successful_samples / n_samples * 100
        _logger.warning(
            "Entanglement estimation: %d/%d samples succeeded (%.1f%%). "
            "Results may be less reliable.",
            successful_samples,
            n_samples,
            success_rate,
        )

    _logger.debug(
        "Entanglement bound estimate for %s: %.4f bits (from %d samples)",
        encoding.__class__.__name__,
        max_entropy,
        successful_samples,
    )

    return float(max_entropy)


# =============================================================================
# Private Helper Functions
# =============================================================================


def _format_memory_size(bytes_count: int | float) -> str:
    """Format a byte count as a human-readable memory size string.

    Uses appropriate units (KB, MB, GB, TB) based on magnitude.

    Parameters
    ----------
    bytes_count : int or float
        Number of bytes to format.

    Returns
    -------
    str
        Human-readable string like "512 KB", "16.0 MB", "2.1 GB".

    Examples
    --------
    >>> _format_memory_size(1024)
    '1.0 KB'
    >>> _format_memory_size(16 * 1024 * 1024)
    '16.0 MB'
    >>> _format_memory_size(2.5 * 1024 * 1024 * 1024)
    '2.5 GB'
    """
    if bytes_count < 1024:
        return f"{bytes_count:.0f} bytes"
    elif bytes_count < 1024**2:
        return f"{bytes_count / 1024:.1f} KB"
    elif bytes_count < 1024**3:
        return f"{bytes_count / (1024**2):.1f} MB"
    elif bytes_count < 1024**4:
        return f"{bytes_count / (1024**3):.1f} GB"
    else:
        return f"{bytes_count / (1024**4):.1f} TB"


def _check_clifford_property(encoding: BaseEncoding) -> bool:
    """Check if encoding appears to be Clifford-only.

    This is a heuristic check based on encoding type, properties, and gate sets.
    Uses the defined _CLIFFORD_GATES and _NON_CLIFFORD_GATES sets for analysis.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to check.

    Returns
    -------
    bool
        True if encoding is believed to use only Clifford gates.

    Notes
    -----
    The Clifford gate set includes: H, S, CNOT, CZ, and Pauli gates (X, Y, Z).
    Circuits using only these gates can be efficiently simulated using the
    stabilizer formalism (Gottesman-Knill theorem).

    Parameterized rotation gates (RX, RY, RZ) are Clifford only at special
    angles (multiples of π/2). Since we cannot determine parameter values
    statically, we conservatively assume non-Clifford for parameterized gates.
    """
    props = encoding.properties
    encoding_name = encoding.__class__.__name__.lower()

    # Known Clifford-only encoding types by name
    clifford_encoding_names = frozenset({
        "basisencoding",  # Only X gates (Clifford)
    })

    # Check if this is a known Clifford-only encoding
    if encoding_name in clifford_encoding_names:
        return True

    # If encoding has a gate_set attribute, analyze it directly
    if hasattr(encoding, "gate_set") or hasattr(encoding, "gates"):
        gate_set_attr = getattr(
            encoding, "gate_set", getattr(encoding, "gates", None)
        )
        if gate_set_attr is not None:
            gates = _normalize_gate_set(gate_set_attr)
            if gates:
                # Check if all gates are Clifford gates
                non_clifford_in_set = gates - _CLIFFORD_GATES
                if not non_clifford_in_set:
                    _logger.debug(
                        "Encoding %s has gate set %s - all Clifford",
                        encoding.__class__.__name__,
                        gates,
                    )
                    return True
                else:
                    _logger.debug(
                        "Encoding %s has non-Clifford gates: %s",
                        encoding.__class__.__name__,
                        non_clifford_in_set,
                    )
                    return False

    # Encodings with parameterized rotations are generally not Clifford
    # (except at special angles like multiples of π/2, which we can't detect)
    if props.parameter_count > 0:
        return False

    # Non-entangling encodings with no parameters could be Clifford
    # Conservative: assume non-Clifford unless we know otherwise
    return props.two_qubit_gates == 0 and not props.is_entangling


def _check_non_clifford_gates(encoding: BaseEncoding) -> dict[str, Any]:
    """Analyze encoding for non-Clifford gate content.

    This function checks for the presence of non-Clifford gates, particularly
    T gates which are important for magic state distillation costs.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - has_non_clifford: bool - Whether non-Clifford gates are present
        - has_t_gates: bool - Whether T gates specifically are present
        - non_clifford_gates: set[str] - Set of non-Clifford gate names found
        - has_parameterized_rotations: bool - Whether parameterized rotations exist

    Notes
    -----
    T gates (and their inverse Tdg) are particularly important because:
    1. They break efficient stabilizer simulation
    2. They require magic state distillation for fault-tolerant computation
    3. The T-count is a common metric for circuit complexity
    """
    result: dict[str, Any] = {
        "has_non_clifford": False,
        "has_t_gates": False,
        "non_clifford_gates": set(),
        "has_parameterized_rotations": False,
    }

    props = encoding.properties

    # Parameterized gates are generally non-Clifford
    if props.parameter_count > 0:
        result["has_non_clifford"] = True
        result["has_parameterized_rotations"] = True

    # Check gate set if available
    if hasattr(encoding, "gate_set") or hasattr(encoding, "gates"):
        gate_set_attr = getattr(
            encoding, "gate_set", getattr(encoding, "gates", None)
        )
        if gate_set_attr is not None:
            gates = _normalize_gate_set(gate_set_attr)
            non_clifford_found = gates & _NON_CLIFFORD_GATES
            if non_clifford_found:
                result["has_non_clifford"] = True
                result["non_clifford_gates"] = non_clifford_found

                # Check specifically for T gates
                t_gate_variants = {"t", "tdg", "t_dagger", "tdagger"}
                if non_clifford_found & t_gate_variants:
                    result["has_t_gates"] = True

    return result


def _normalize_gate_set(gate_set_attr: Any) -> set[str]:
    """Normalize a gate set attribute to a set of lowercase strings.

    Parameters
    ----------
    gate_set_attr : Any
        Gate set attribute from encoding (could be list, tuple, set, or str).

    Returns
    -------
    set[str]
        Normalized set of lowercase gate names.
    """
    if isinstance(gate_set_attr, (list, tuple, set, frozenset)):
        return {str(g).lower() for g in gate_set_attr}
    elif isinstance(gate_set_attr, str):
        return {gate_set_attr.lower()}
    return set()


def _check_matchgate_property(encoding: BaseEncoding) -> bool:
    """Check if encoding appears to use only matchgate operations.

    Matchgate circuits are classically simulable when applied with
    nearest-neighbor connectivity on a line topology.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to check.

    Returns
    -------
    bool
        True if encoding is believed to be a matchgate circuit with
        appropriate topology for classical simulation.

    Notes
    -----
    This check considers:

    1. Known matchgate-based encoding names
    2. Entanglement topology (must be linear/nearest-neighbor)
    3. Encoding properties that suggest matchgate structure

    The check is conservative - it may return False for some valid
    matchgate circuits that don't follow recognized patterns.
    """
    encoding_name = encoding.__class__.__name__.lower()

    # Compute entanglement pattern once (used by all checks below)
    entanglement_pattern = _get_entanglement_pattern(encoding)
    _has_linear_topology = entanglement_pattern in ("linear", "none")

    # Check for known matchgate-based encoding types
    if encoding_name in _MATCHGATE_ENCODING_NAMES:
        # Verify linear topology for simulability
        if _has_linear_topology:
            _logger.debug(
                "Encoding %s identified as matchgate circuit with %s topology",
                encoding.__class__.__name__,
                entanglement_pattern,
            )
            return True
        else:
            _logger.debug(
                "Encoding %s is matchgate-based but has %s topology "
                "(not efficiently simulable)",
                encoding.__class__.__name__,
                entanglement_pattern,
            )
            return False

    # Check if encoding has matchgate-related attributes
    if hasattr(encoding, "gate_set") or hasattr(encoding, "gates"):
        gate_set_attr = getattr(
            encoding, "gate_set", getattr(encoding, "gates", None)
        )
        if gate_set_attr is not None:
            gates = _normalize_gate_set(gate_set_attr)

            # Check if all gates are matchgates
            if gates and gates.issubset(_MATCHGATE_GATES):
                # Also verify topology
                if _has_linear_topology:
                    _logger.debug(
                        "Encoding %s uses matchgate set with linear topology",
                        encoding.__class__.__name__,
                    )
                    return True

    # Check for fermionic/particle-preserving keywords in encoding name
    fermionic_keywords = {"fermionic", "fermion", "givens", "particle", "matchgate"}
    if any(keyword in encoding_name for keyword in fermionic_keywords):
        if _has_linear_topology:
            _logger.debug(
                "Encoding %s appears to be fermionic/matchgate-based",
                encoding.__class__.__name__,
            )
            return True

    # Non-entangling encodings are trivially simulable (but not matchgate-specific)
    # Return False here as they're handled by product state simulability
    return False


def _get_entanglement_pattern(encoding: BaseEncoding) -> str:
    """Determine the entanglement pattern of an encoding.

    Analyzes the encoding to identify its entanglement topology using
    multiple detection strategies with graceful fallbacks.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.

    Returns
    -------
    str
        One of: "none", "linear", "circular", "full", "partial", "unknown".
        Note that "nearest_neighbor" inputs are normalized to "linear".

    Notes
    -----
    Detection strategies (in order of priority):

    1. Check if encoding is non-entangling (returns "none")
    2. Check encoding's `entanglement` attribute (many encodings store this)
    3. Analyze `get_entanglement_pairs()` if implemented
    4. Infer from encoding name patterns
    5. Fallback to "unknown"

    The function is designed to be robust against missing methods and
    unexpected data types.
    """
    # Check if encoding is non-entangling
    if not encoding.properties.is_entangling:
        return "none"

    n_qubits = encoding.n_qubits

    # =========================================================================
    # Strategy 1: Check encoding's entanglement attribute
    # Many encodings store their entanglement pattern as an attribute
    # =========================================================================
    if hasattr(encoding, "entanglement"):
        entanglement = getattr(encoding, "entanglement", None)
        if isinstance(entanglement, str):
            pattern = entanglement.lower()
            # Normalize common patterns
            if pattern in ("linear", "nearest_neighbor", "nearest-neighbor"):
                return "linear"
            elif pattern in ("circular", "ring", "pbc", "periodic"):
                return "circular"
            elif pattern in ("full", "all", "all-to-all", "complete"):
                return "full"
            elif pattern in ("none", "empty"):
                return "none"
            elif pattern == "partial":
                return "partial"

    # =========================================================================
    # Strategy 2: Analyze get_entanglement_pairs() if implemented
    # =========================================================================
    if hasattr(encoding, "get_entanglement_pairs"):
        try:
            pairs = encoding.get_entanglement_pairs()

            if not pairs:
                return "none"

            # Safely convert pairs to list of (int, int) tuples
            normalized_pairs: list[tuple[int, int]] = []
            for p in pairs:
                try:
                    # Handle various pair formats: (0, 1), [0, 1], etc.
                    if hasattr(p, "__iter__") and not isinstance(p, str):
                        pair_list = list(p)
                        if len(pair_list) >= 2:
                            normalized_pairs.append((int(pair_list[0]), int(pair_list[1])))
                except (TypeError, ValueError, IndexError):
                    continue

            if not normalized_pairs:
                _logger.debug(
                    "Could not normalize entanglement pairs for %s",
                    encoding.__class__.__name__,
                )
            else:
                n_pairs = len(normalized_pairs)
                max_pairs = n_qubits * (n_qubits - 1) // 2

                # Check for linear (nearest-neighbor) pattern
                is_linear = all(
                    abs(p[0] - p[1]) == 1 for p in normalized_pairs
                )

                # Linear pattern: n-1 nearest-neighbor edges
                if is_linear and n_pairs == n_qubits - 1:
                    return "linear"

                # Check for circular pattern (linear + wrap-around edge)
                has_wrap_around = any(
                    (min(p[0], p[1]) == 0 and max(p[0], p[1]) == n_qubits - 1)
                    for p in normalized_pairs
                )
                is_circular = all(
                    abs(p[0] - p[1]) == 1
                    or (min(p[0], p[1]) == 0 and max(p[0], p[1]) == n_qubits - 1)
                    for p in normalized_pairs
                )

                # Circular pattern: n edges with wrap-around
                if is_circular and has_wrap_around and n_pairs == n_qubits:
                    return "circular"

                # Full connectivity: all possible pairs
                if n_pairs >= max_pairs:
                    return "full"

                # Partial entanglement: some but not all pairs
                if n_pairs > 0:
                    return "partial"

        except Exception as e:
            _logger.debug(
                "Failed to analyze entanglement pairs for %s: %s",
                encoding.__class__.__name__,
                str(e),
            )

    # =========================================================================
    # Strategy 3: Infer from encoding name or class
    # =========================================================================
    encoding_name = encoding.__class__.__name__.lower()

    # IQP and ZZ encodings typically use full entanglement by default
    if any(pattern in encoding_name for pattern in ("iqp", "zz", "pauli")):
        # Check config for entanglement specification
        config = getattr(encoding, "config", {}) or {}
        if isinstance(config, dict):
            ent_config = config.get("entanglement", "")
            if isinstance(ent_config, str):
                if ent_config.lower() in ("linear", "nearest_neighbor"):
                    return "linear"
                elif ent_config.lower() in ("circular", "ring"):
                    return "circular"
        # Default for these encodings is typically full
        return "full"

    # Hardware efficient encodings often use linear entanglement
    if "hardware" in encoding_name or "efficient" in encoding_name:
        return "linear"

    # =========================================================================
    # Fallback: unknown pattern
    # =========================================================================
    return "unknown"


def _compute_bipartite_entropy(
    statevector: NDArray[np.complexfloating[Any, Any]],
    n_qubits: int,
    cut_position: int,
) -> float:
    """Compute entanglement entropy across a bipartition.

    Uses von Neumann entropy of the reduced density matrix:
        S = -Tr(ρ_A log₂ ρ_A) = -Σᵢ λᵢ log₂(λᵢ)

    where λᵢ are the eigenvalues of the reduced density matrix.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        Full system statevector of shape (2^n_qubits,).
    n_qubits : int
        Total number of qubits in the system.
    cut_position : int
        Position of the bipartition cut. The left subsystem contains
        qubits 0 to cut_position-1, and the right subsystem contains
        qubits cut_position to n_qubits-1.

        **Valid range**: ``1 <= cut_position <= n_qubits - 1``

        For values outside this range (degenerate bipartitions where one
        subsystem is empty), the function returns 0.0 by design.

    Returns
    -------
    float
        Entanglement entropy in bits (log base 2). Returns 0.0 for:

        - Degenerate bipartitions (cut_position <= 0 or cut_position >= n_qubits)
        - Product states (no entanglement between subsystems)
        - Single-qubit systems (n_qubits = 1)

    Notes
    -----
    **Degenerate Bipartition Handling**:

    When ``cut_position <= 0`` or ``cut_position >= n_qubits``, one of the
    subsystems would be empty. This is not a meaningful bipartition:

    - ``cut_position = 0``: Left subsystem has 0 qubits (empty)
    - ``cut_position = n_qubits``: Right subsystem has 0 qubits (empty)

    In these cases, 0.0 is returned because:

    1. Mathematically, there's no tensor product structure to trace over
    2. Physically, entanglement requires at least one qubit on each side
    3. This is consistent with the definition S(ρ_∅) = 0 for the empty system

    **Computation Steps**:

    1. Reshape the statevector into a matrix of shape (dim_left, dim_right)
    2. Compute reduced density matrix: ρ_L = Tr_R(|ψ⟩⟨ψ|) = ψψ†
    3. Find eigenvalues of ρ_L (real for Hermitian matrix)
    4. Compute von Neumann entropy: S = -Σᵢ λᵢ log₂(λᵢ)
    """
    # =========================================================================
    # Handle degenerate bipartitions
    # =========================================================================
    # A valid bipartition requires at least 1 qubit on each side.
    # - cut_position <= 0: Left subsystem would be empty
    # - cut_position >= n_qubits: Right subsystem would be empty
    # - n_qubits < 2: Cannot form a bipartition with single qubit
    # In all these cases, return 0.0 as there's no meaningful entanglement
    # to measure across the cut.
    if cut_position <= 0 or cut_position >= n_qubits:
        return 0.0

    # Dimensions of left and right subsystems
    dim_left = 2**cut_position
    dim_right = 2 ** (n_qubits - cut_position)

    # Reshape statevector into matrix form
    # Each row corresponds to left subsystem, columns to right subsystem
    try:
        psi_matrix = statevector.reshape(dim_left, dim_right)
    except ValueError:
        _logger.warning(
            "Failed to reshape statevector for entropy calculation: "
            "shape %s, expected %d elements",
            statevector.shape,
            dim_left * dim_right,
        )
        return 0.0

    # Compute reduced density matrix for left subsystem
    # ρ_L = Tr_R(|ψ⟩⟨ψ|) = ψ @ ψ†
    rho_left = psi_matrix @ psi_matrix.conj().T

    # Get eigenvalues of reduced density matrix
    # For Hermitian matrix, eigenvalues are real
    try:
        eigenvalues = np.linalg.eigvalsh(rho_left)
    except np.linalg.LinAlgError:
        _logger.warning("Eigenvalue computation failed for entropy calculation")
        return 0.0

    # Filter to positive eigenvalues (numerical noise can give small negatives)
    positive_eigenvalues = eigenvalues[eigenvalues > _ENTROPY_EPSILON]

    if len(positive_eigenvalues) == 0:
        # Pure product state - no entanglement
        return 0.0

    # Compute von Neumann entropy: S = -Σᵢ λᵢ log₂(λᵢ)
    entropy = -np.sum(positive_eigenvalues * np.log2(positive_eigenvalues))

    # Ensure non-negative (numerical precision)
    entropy = max(0.0, float(entropy))

    return entropy
