"""Resource counting and analysis for quantum encodings.

This module provides comprehensive resource analysis for quantum encoding
circuits, including gate counts, circuit depth, and detailed breakdowns
by gate type. This information is crucial for:

- **Hardware Planning**: Evaluating quantum hardware requirements
- **Encoding Comparison**: Objective comparison of circuit complexity
- **Compilation Planning**: Estimating transpilation overhead
- **Optimization Guidance**: Identifying resource-heavy components
- **Performance Prediction**: Understanding execution time on quantum hardware

The main function :func:`count_resources` returns a detailed breakdown of
all circuit resources without requiring circuit simulation.

Key Metrics
-----------
.. list-table:: Resource Metrics
   :header-rows: 1
   :widths: 25 40 35

   * - Metric
     - Description
     - Use Case
   * - ``n_qubits``
     - Total qubits required
     - Hardware capacity planning
   * - ``depth``
     - Circuit depth (layers)
     - Decoherence time constraints
   * - ``gate_count``
     - Total gates
     - Overall complexity
   * - ``single_qubit_gates``
     - 1Q gates
     - Typically fast, high fidelity
   * - ``two_qubit_gates``
     - 2Q gates
     - Typically slower, lower fidelity
   * - ``parameter_count``
     - Parameterized gates
     - Training complexity
   * - ``two_qubit_ratio``
     - 2Q / total ratio
     - Hardware noise impact

Design Principles
-----------------
1. **Protocol-Based**: Uses ``ResourceAnalyzable`` protocol when available
2. **Data-Aware**: Handles data-dependent encodings like ``BasisEncoding``
3. **Fallback Strategy**: Uses ``encoding.properties`` when protocol not available
4. **Type Safety**: Full TypedDict definitions for return types
5. **Numerical Robustness**: Safe handling of division and edge cases

Examples
--------
Basic resource counting:

>>> from encoding_atlas import AngleEncoding, IQPEncoding
>>> from encoding_atlas.analysis import count_resources
>>>
>>> enc = AngleEncoding(n_features=4)
>>> resources = count_resources(enc)
>>> print(f"Total gates: {resources['gate_count']}")
>>> print(f"Circuit depth: {resources['depth']}")

Comparing encodings:

>>> enc_simple = AngleEncoding(n_features=4)
>>> enc_complex = IQPEncoding(n_features=4, reps=2)
>>>
>>> res_simple = count_resources(enc_simple)
>>> res_complex = count_resources(enc_complex)
>>>
>>> print(f"Simple: {res_simple['gate_count']} gates")
>>> print(f"Complex: {res_complex['gate_count']} gates")

Detailed breakdown:

>>> breakdown = count_resources(enc_complex, detailed=True)
>>> print(f"CNOT gates: {breakdown['cnot']}")
>>> print(f"Hadamard gates: {breakdown['h']}")

Data-dependent encoding (BasisEncoding):

>>> from encoding_atlas import BasisEncoding
>>> import numpy as np
>>>
>>> enc = BasisEncoding(n_features=4)
>>> x = np.array([0, 1, 1, 0])  # Binary input
>>> res = count_resources(enc, x=x)
>>> print(f"X gates (depends on input): {res['gate_count']}")

Multi-encoding comparison:

>>> from encoding_atlas.analysis import compare_resources
>>> encodings = [
...     AngleEncoding(n_features=4),
...     IQPEncoding(n_features=4, reps=1),
...     IQPEncoding(n_features=4, reps=2),
... ]
>>> comparison = compare_resources(encodings)
>>> print(comparison['gate_count'])  # [4, 26, 52]

Notes
-----
Resource counting is performed analytically based on encoding parameters,
not by circuit simulation. This makes it fast and deterministic.

For data-dependent encodings like BasisEncoding, the gate count
depends on the specific input values (e.g., number of 1s in binary
input determines number of X gates).

See Also
--------
check_simulability : Determine if classical simulation is efficient.
compute_expressibility : Measure Hilbert space coverage.
compute_entanglement_capability : Measure entanglement generation.

References
----------
.. [1] Cross, A. W., et al. (2017). "Open Quantum Assembly Language."
       arXiv:1707.03429.
.. [2] Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond."
       Quantum, 2, 79.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.exceptions import AnalysisError, ValidationError
from encoding_atlas.core.protocols import (
    is_data_dependent_resource_analyzable,
    is_resource_analyzable,
)

if TYPE_CHECKING:
    from typing import TypedDict

    class ResourceCountSummary(TypedDict):
        """Complete resource summary for an encoding.

        This TypedDict defines the structure of the dictionary returned by
        :func:`count_resources` when ``detailed=False``.

        Attributes
        ----------
        n_qubits : int
            Number of qubits required for the encoding circuit.
        depth : int
            Circuit depth (number of time steps/layers).
        gate_count : int
            Total number of gates in the circuit.
        single_qubit_gates : int
            Number of single-qubit gates (RX, RY, RZ, H, X, Y, Z, S, T).
        two_qubit_gates : int
            Number of two-qubit gates (CNOT, CZ, SWAP).
        parameter_count : int
            Number of parameterized gates (data-dependent rotations).
        cnot_count : int
            Number of CNOT/CX gates specifically.
        cz_count : int
            Number of CZ gates.
        t_gate_count : int
            Number of T gates (relevant for fault-tolerant computing).
        hadamard_count : int
            Number of Hadamard gates.
        rotation_gates : int
            Total number of rotation gates (RX + RY + RZ).
        two_qubit_ratio : float
            Fraction of gates that are two-qubit (0.0 to 1.0).
        gates_per_qubit : float
            Average number of gates per qubit.
        encoding_name : str
            Name of the encoding class.
        is_data_dependent : bool
            Whether gate counts depend on input data.
        """

        n_qubits: int
        depth: int
        gate_count: int
        single_qubit_gates: int
        two_qubit_gates: int
        parameter_count: int
        cnot_count: int
        cz_count: int
        t_gate_count: int
        hadamard_count: int
        rotation_gates: int
        two_qubit_ratio: float
        gates_per_qubit: float
        encoding_name: str
        is_data_dependent: bool

    class DetailedGateBreakdown(TypedDict):
        """Detailed breakdown by specific gate types.

        This TypedDict defines the structure of the dictionary returned by
        :func:`count_resources` when ``detailed=True``.

        Attributes
        ----------
        rx : int
            Number of RX (rotation around X) gates.
        ry : int
            Number of RY (rotation around Y) gates.
        rz : int
            Number of RZ (rotation around Z) gates.
        h : int
            Number of Hadamard gates.
        x : int
            Number of Pauli-X gates.
        y : int
            Number of Pauli-Y gates.
        z : int
            Number of Pauli-Z gates.
        s : int
            Number of S (phase) gates.
        t : int
            Number of T gates.
        cnot : int
            Number of CNOT gates.
        cx : int
            Number of CX gates (same as CNOT).
        cz : int
            Number of CZ gates.
        swap : int
            Number of SWAP gates.
        total_single_qubit : int
            Total number of single-qubit gates.
        total_two_qubit : int
            Total number of two-qubit gates.
        total : int
            Total gate count.
        encoding_name : str
            Name of the encoding class.
        """

        rx: int
        ry: int
        rz: int
        h: int
        x: int
        y: int
        z: int
        s: int
        t: int
        cnot: int
        cx: int
        cz: int
        swap: int
        total_single_qubit: int
        total_two_qubit: int
        total: int
        encoding_name: str


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "count_resources",
    "get_resource_summary",
    "get_gate_breakdown",
    "compare_resources",
    "estimate_execution_time",
]

# =============================================================================
# Module-Level Logger
# =============================================================================

_logger = logging.getLogger(__name__)

# =============================================================================
# Module-Level Constants
# =============================================================================

# Numerical stability constant to prevent division by zero
_EPSILON: float = 1e-15

# Default gate times in microseconds (typical superconducting qubit values)
# Used for execution time estimation
_DEFAULT_SINGLE_QUBIT_GATE_TIME_US: float = 0.02  # 20 ns
_DEFAULT_TWO_QUBIT_GATE_TIME_US: float = 0.2  # 200 ns
_DEFAULT_MEASUREMENT_TIME_US: float = 1.0  # 1 μs


# =============================================================================
# Main Public Functions
# =============================================================================


@overload
def count_resources(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]] | None = None,
    *,
    detailed: Literal[False] = False,
) -> ResourceCountSummary: ...


@overload
def count_resources(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]] | None = None,
    *,
    detailed: Literal[True],
) -> DetailedGateBreakdown: ...


def count_resources(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]] | None = None,
    *,
    detailed: bool = False,
) -> ResourceCountSummary | DetailedGateBreakdown:
    """Count the computational resources required by an encoding.

    Analyzes the encoding circuit structure to determine gate counts,
    circuit depth, and other resource metrics. This analysis is performed
    analytically (without simulation) for most encodings.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to analyze. Must be a valid encoding that
        inherits from :class:`BaseEncoding`.
    x : NDArray[np.floating], optional
        Input data vector. Required only for data-dependent encodings
        (e.g., BasisEncoding) where gate count depends on input values.
        For most encodings, this can be omitted.
    detailed : bool, default=False
        If True, return detailed gate-by-gate breakdown.
        If False, return summary with totals and derived metrics.

    Returns
    -------
    ResourceCountSummary or DetailedGateBreakdown
        If ``detailed=False`` (default): Dictionary with summary metrics including:

        - ``n_qubits``: Number of qubits
        - ``depth``: Circuit depth
        - ``gate_count``: Total gates
        - ``single_qubit_gates``: Single-qubit gate count
        - ``two_qubit_gates``: Two-qubit gate count
        - ``parameter_count``: Number of parameterized gates
        - ``two_qubit_ratio``: Fraction of gates that are two-qubit
        - ``gates_per_qubit``: Average gates per qubit
        - ``encoding_name``: Name of the encoding class
        - ``is_data_dependent``: Whether counts depend on input data

        If ``detailed=True``: Dictionary with per-gate-type counts including:

        - ``rx``, ``ry``, ``rz``: Rotation gate counts
        - ``h``: Hadamard count
        - ``x``, ``y``, ``z``: Pauli gate counts
        - ``cnot``, ``cz``, ``swap``: Two-qubit gate counts
        - ``total_single_qubit``, ``total_two_qubit``, ``total``
        - ``encoding_name``: Name of the encoding class

    Raises
    ------
    AnalysisError
        If encoding is not a valid BaseEncoding instance or if
        resource counting fails.
    ValidationError
        If x is required but not provided (for data-dependent encodings),
        or if x has invalid shape or values.

    Examples
    --------
    Basic resource counting:

    >>> from encoding_atlas import AngleEncoding, IQPEncoding
    >>> from encoding_atlas.analysis import count_resources
    >>>
    >>> enc = AngleEncoding(n_features=4)
    >>> res = count_resources(enc)
    >>> print(f"Qubits: {res['n_qubits']}, Depth: {res['depth']}")
    Qubits: 4, Depth: 1

    Comparing two encodings:

    >>> enc_simple = AngleEncoding(n_features=4)
    >>> enc_complex = IQPEncoding(n_features=4, reps=2)
    >>>
    >>> res_simple = count_resources(enc_simple)
    >>> res_complex = count_resources(enc_complex)
    >>>
    >>> print(f"Simple: {res_simple['gate_count']} gates")
    >>> print(f"Complex: {res_complex['gate_count']} gates")

    Detailed breakdown:

    >>> breakdown = count_resources(enc_complex, detailed=True)
    >>> print(f"CNOT gates: {breakdown['cnot']}")
    >>> print(f"RZ gates: {breakdown['rz']}")

    Data-dependent encoding:

    >>> from encoding_atlas import BasisEncoding
    >>> import numpy as np
    >>>
    >>> enc = BasisEncoding(n_features=4)
    >>> x = np.array([0.1, 0.9, 0.3, 0.8])  # Will be binarized
    >>> res = count_resources(enc, x=x)
    >>> print(f"X gates (depends on input): {res['gate_count']}")

    Notes
    -----
    For most encodings, resource counts are computed analytically from
    encoding parameters (n_features, reps, entanglement pattern, etc.).
    This is fast and deterministic.

    For data-dependent encodings like BasisEncoding, the gate count
    depends on the specific input values (e.g., number of 1s in binary
    input determines number of X gates).

    See Also
    --------
    get_resource_summary : Get summary from encoding properties.
    get_gate_breakdown : Get detailed gate-by-gate breakdown.
    compare_resources : Compare resources between multiple encodings.
    check_simulability : Check if encoding is classically simulable.
    """
    # Validate encoding
    _validate_encoding(encoding)

    _logger.debug(
        "Counting resources for %s (detailed=%s)",
        encoding.__class__.__name__,
        detailed,
    )

    # Check if encoding requires data for resource counting
    is_data_dependent = is_data_dependent_resource_analyzable(encoding)

    if is_data_dependent:
        if x is None:
            raise ValidationError(
                f"{encoding.__class__.__name__} is data-dependent. "
                "Please provide input data 'x' for resource counting. "
                "For worst-case estimates, use encoding.properties instead."
            )
        # Validate input data
        x_validated = _validate_input_data(x, encoding.n_features)
        return _count_data_dependent_resources(encoding, x_validated, detailed)

    # Check if encoding implements ResourceAnalyzable protocol
    if is_resource_analyzable(encoding):
        _logger.debug(
            "Using ResourceAnalyzable protocol for %s",
            encoding.__class__.__name__,
        )
        if detailed:
            return _get_detailed_breakdown_from_protocol(encoding)
        else:
            return _get_summary_from_protocol(encoding)

    # Fallback: use encoding properties
    _logger.debug(
        "Using properties fallback for %s",
        encoding.__class__.__name__,
    )
    return _get_summary_from_properties(encoding, detailed)


def get_resource_summary(encoding: BaseEncoding) -> ResourceCountSummary:
    """Get a quick resource summary from encoding properties.

    This is a lightweight alternative to :func:`count_resources` that
    uses cached encoding properties when available. It always returns
    theoretical (worst-case) values, even for data-dependent encodings.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to summarize.

    Returns
    -------
    ResourceCountSummary
        Dictionary with resource metrics. See **Warnings** section for
        details on which fields contain approximations.

    Raises
    ------
    AnalysisError
        If encoding is not a valid BaseEncoding instance.

    Warnings
    --------
    **Approximated Fields**: The following fields are approximations based on
    aggregate property values, not actual gate-by-gate circuit analysis:

    - ``cnot_count``: Approximated as ``two_qubit_gates``. This assumes all
      two-qubit gates are CNOTs, which may overcount if the encoding uses
      CZ or SWAP gates instead.
    - ``cz_count``: Always returns 0 (requires detailed circuit analysis).
    - ``t_gate_count``: Always returns 0 (requires detailed circuit analysis).
    - ``hadamard_count``: Always returns 0 (requires detailed circuit analysis).
    - ``rotation_gates``: Approximated as ``single_qubit_gates``. This assumes
      all single-qubit gates are rotation gates (RX, RY, RZ), which may
      overcount if the encoding uses non-rotation gates like H, X, Y, Z, S, T.

    For **accurate gate-type counts**, use :func:`count_resources` with
    ``detailed=True``, which performs actual gate-by-gate analysis via
    the encoding's ``gate_count_breakdown()`` method when available.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.analysis import get_resource_summary
    >>>
    >>> enc = IQPEncoding(n_features=4, reps=2)
    >>> summary = get_resource_summary(enc)
    >>> print(f"Qubits: {summary['n_qubits']}, Depth: {summary['depth']}")

    For accurate gate counts, prefer count_resources with detailed=True:

    >>> from encoding_atlas.analysis import count_resources
    >>> detailed = count_resources(enc, detailed=True)
    >>> print(f"Actual Hadamard gates: {detailed['h']}")

    Notes
    -----
    This function uses the ``properties`` attribute of the encoding,
    which is typically cached after first access. This makes it very
    fast for repeated calls.

    For data-dependent encodings, this returns worst-case (maximum)
    gate counts. Use :func:`count_resources` with input data for
    actual counts.

    See Also
    --------
    count_resources : Full resource counting with detailed options.
    get_gate_breakdown : Get detailed gate-by-gate breakdown.
    """
    _validate_encoding(encoding)

    props = encoding.properties

    gate_count = props.gate_count
    n_qubits = props.n_qubits

    # Note: Some fields below are approximations based on aggregate property
    # values. See docstring Warnings section for details. For accurate
    # gate-type counts, use count_resources() with detailed=True.
    return {
        "n_qubits": n_qubits,
        "depth": props.depth,
        "gate_count": gate_count,
        "single_qubit_gates": props.single_qubit_gates,
        "two_qubit_gates": props.two_qubit_gates,
        "parameter_count": props.parameter_count,
        # Approximation: assumes all 2Q gates are CNOTs (see docstring Warnings)
        "cnot_count": props.two_qubit_gates,
        # Cannot determine without detailed circuit analysis
        "cz_count": 0,
        "t_gate_count": 0,
        "hadamard_count": 0,
        # Approximation: assumes all 1Q gates are rotations (see docstring Warnings)
        "rotation_gates": props.single_qubit_gates,
        "two_qubit_ratio": _safe_divide(props.two_qubit_gates, gate_count),
        "gates_per_qubit": _safe_divide(gate_count, n_qubits),
        "encoding_name": encoding.__class__.__name__,
        "is_data_dependent": is_data_dependent_resource_analyzable(encoding),
    }


def get_gate_breakdown(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]] | None = None,
) -> DetailedGateBreakdown:
    """Get detailed gate-by-gate breakdown for an encoding.

    This is a convenience function that calls :func:`count_resources`
    with ``detailed=True``.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    x : NDArray[np.floating], optional
        Input data for data-dependent encodings. Required for encodings
        like BasisEncoding where gate count depends on input values.

    Returns
    -------
    DetailedGateBreakdown
        Dictionary with counts for each gate type.

    Raises
    ------
    AnalysisError
        If encoding is invalid.
    ValidationError
        If x is required but not provided or has invalid values.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.analysis import get_gate_breakdown
    >>>
    >>> enc = IQPEncoding(n_features=4, reps=2)
    >>> breakdown = get_gate_breakdown(enc)
    >>> print(f"CNOT gates: {breakdown['cnot']}")
    >>> print(f"Hadamard gates: {breakdown['h']}")
    >>> print(f"Total: {breakdown['total']}")

    See Also
    --------
    count_resources : Full resource counting with summary option.
    get_resource_summary : Quick summary from cached properties.
    """
    result = count_resources(encoding, x=x, detailed=True)
    # Runtime type narrowing (cannot use assert — stripped by python -O)
    if "rx" not in result:  # pragma: no cover
        raise AnalysisError(
            "count_resources(detailed=True) did not return a "
            "DetailedGateBreakdown. This is an internal error.",
        )
    return result


def compare_resources(
    encodings: list[BaseEncoding],
    metrics: list[str] | None = None,
    *,
    include_names: bool = True,
) -> dict[str, list[Any]]:
    """Compare resources across multiple encodings.

    Provides side-by-side comparison of resource metrics for a list
    of encodings. Useful for encoding selection and benchmarking.

    Parameters
    ----------
    encodings : list[BaseEncoding]
        List of encodings to compare. Must all be valid BaseEncoding
        instances.
    metrics : list[str], optional
        Specific metrics to compare. If None, compares all standard metrics.
        Available metrics:

        - ``"n_qubits"``: Number of qubits
        - ``"depth"``: Circuit depth
        - ``"gate_count"``: Total gates
        - ``"single_qubit_gates"``: Single-qubit gate count
        - ``"two_qubit_gates"``: Two-qubit gate count
        - ``"parameter_count"``: Parameterized gate count
        - ``"two_qubit_ratio"``: Ratio of two-qubit gates
        - ``"gates_per_qubit"``: Average gates per qubit

    include_names : bool, default=True
        If True, include encoding names in the result.

    Returns
    -------
    dict[str, list[Any]]
        Dictionary mapping metric names to lists of values
        (one per encoding). If ``include_names=True``, includes
        an ``"encoding_name"`` key with encoding class names.

    Raises
    ------
    AnalysisError
        If any encoding is invalid.
    ValueError
        If encodings list is empty.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding, IQPEncoding
    >>> from encoding_atlas.analysis import compare_resources
    >>>
    >>> encodings = [
    ...     AngleEncoding(n_features=4),
    ...     IQPEncoding(n_features=4, reps=1),
    ...     IQPEncoding(n_features=4, reps=2),
    ... ]
    >>> comparison = compare_resources(encodings)
    >>> print(comparison['encoding_name'])
    ['AngleEncoding', 'IQPEncoding', 'IQPEncoding']
    >>> print(comparison['gate_count'])
    [4, 26, 52]

    Compare specific metrics:

    >>> comparison = compare_resources(
    ...     encodings,
    ...     metrics=['gate_count', 'two_qubit_gates'],
    ... )
    >>> print(comparison.keys())
    dict_keys(['encoding_name', 'gate_count', 'two_qubit_gates'])

    Notes
    -----
    For data-dependent encodings, this function returns worst-case
    (maximum) values from encoding properties. Use :func:`count_resources`
    with specific input data for actual counts.

    See Also
    --------
    count_resources : Get resources for a single encoding.
    get_resource_summary : Quick summary for a single encoding.
    """
    if not encodings:
        raise ValueError("encodings list cannot be empty")

    _VALID_METRICS = {
        "n_qubits",
        "depth",
        "gate_count",
        "single_qubit_gates",
        "two_qubit_gates",
        "parameter_count",
        "cnot_count",
        "cz_count",
        "t_gate_count",
        "hadamard_count",
        "rotation_gates",
        "two_qubit_ratio",
        "gates_per_qubit",
        "encoding_name",
        "is_data_dependent",
    }

    if metrics is None:
        metrics = [
            "n_qubits",
            "depth",
            "gate_count",
            "single_qubit_gates",
            "two_qubit_gates",
            "parameter_count",
            "two_qubit_ratio",
            "gates_per_qubit",
        ]
    else:
        unknown = set(metrics) - _VALID_METRICS
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {sorted(unknown)}. "
                f"Valid metrics: {sorted(_VALID_METRICS)}"
            )

    results: dict[str, list[Any]] = {m: [] for m in metrics}

    if include_names:
        results["encoding_name"] = []

    for enc in encodings:
        # Use get_resource_summary for worst-case values (doesn't require x)
        summary = get_resource_summary(enc)

        if include_names:
            results["encoding_name"].append(enc.__class__.__name__)

        for metric in metrics:
            # Skip encoding_name if already populated by include_names
            # to avoid duplicating entries in the same list.
            if metric == "encoding_name" and include_names:
                continue
            value = summary.get(metric)
            if value is None:
                _logger.warning(
                    "Metric '%s' not available for %s",
                    metric,
                    enc.__class__.__name__,
                )
            results[metric].append(value)

    _logger.debug(
        "Compared %d encodings across %d metrics",
        len(encodings),
        len(metrics),
    )

    return results


def estimate_execution_time(
    encoding: BaseEncoding,
    *,
    single_qubit_gate_time_us: float = _DEFAULT_SINGLE_QUBIT_GATE_TIME_US,
    two_qubit_gate_time_us: float = _DEFAULT_TWO_QUBIT_GATE_TIME_US,
    measurement_time_us: float = _DEFAULT_MEASUREMENT_TIME_US,
    include_measurement: bool = True,
    parallelization_factor: float = 0.5,
) -> dict[str, float]:
    """Estimate execution time for an encoding circuit.

    Provides a rough estimate of circuit execution time based on
    gate counts and typical gate times. This is useful for:

    - Comparing relative execution times between encodings
    - Planning batch processing
    - Evaluating hardware constraints

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    single_qubit_gate_time_us : float, default=0.02
        Time for single-qubit gates in microseconds. Default is 20 ns,
        typical for superconducting qubits.
    two_qubit_gate_time_us : float, default=0.2
        Time for two-qubit gates in microseconds. Default is 200 ns,
        typical for superconducting qubits.
    measurement_time_us : float, default=1.0
        Time for measurement in microseconds.
    include_measurement : bool, default=True
        Whether to include measurement time in the estimate.
    parallelization_factor : float, default=0.5
        Fraction of gates assumed to run in parallel. 0 = fully serial,
        1 = fully parallel. Default of 0.5 assumes moderate parallelism.

    Returns
    -------
    dict[str, float]
        Dictionary containing:

        - ``serial_time_us``: Time assuming no parallelization
        - ``estimated_time_us``: Time with parallelization factor
        - ``single_qubit_time_us``: Time for single-qubit gates only
        - ``two_qubit_time_us``: Time for two-qubit gates only
        - ``measurement_time_us``: Measurement time (if included)
        - ``parallelization_factor``: The factor used

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.analysis import estimate_execution_time
    >>>
    >>> enc = IQPEncoding(n_features=4, reps=2)
    >>> times = estimate_execution_time(enc)
    >>> print(f"Estimated time: {times['estimated_time_us']:.2f} μs")

    With custom gate times (trapped ions):

    >>> times = estimate_execution_time(
    ...     enc,
    ...     single_qubit_gate_time_us=1.0,  # 1 μs
    ...     two_qubit_gate_time_us=100.0,   # 100 μs
    ... )

    Notes
    -----
    This is a rough estimate based on gate counts and does not account for:

    - Circuit topology and routing overhead
    - Hardware-specific gate implementations
    - Error correction overhead
    - Classical control latency

    For accurate timing, run the circuit on actual hardware or use
    a detailed hardware simulator.

    The parallelization factor accounts for the fact that independent
    gates can run in parallel. A factor of 0.5 means the effective time
    is 50% of the fully serial time.
    """
    _validate_encoding(encoding)

    # Validate timing parameters
    if not 0.0 <= parallelization_factor <= 1.0:
        raise ValidationError(
            f"parallelization_factor must be between 0.0 and 1.0, "
            f"got {parallelization_factor}"
        )
    if single_qubit_gate_time_us < 0:
        raise ValidationError(
            f"single_qubit_gate_time_us must be non-negative, "
            f"got {single_qubit_gate_time_us}"
        )
    if two_qubit_gate_time_us < 0:
        raise ValidationError(
            f"two_qubit_gate_time_us must be non-negative, "
            f"got {two_qubit_gate_time_us}"
        )
    if measurement_time_us < 0:
        raise ValidationError(
            f"measurement_time_us must be non-negative, " f"got {measurement_time_us}"
        )

    summary = get_resource_summary(encoding)

    single_qubit_gates = summary["single_qubit_gates"]
    two_qubit_gates = summary["two_qubit_gates"]

    # Calculate times
    single_qubit_time = single_qubit_gates * single_qubit_gate_time_us
    two_qubit_time = two_qubit_gates * two_qubit_gate_time_us
    meas_time = measurement_time_us if include_measurement else 0.0

    serial_time = single_qubit_time + two_qubit_time + meas_time

    # Apply parallelization factor
    # Factor of 0.5 means we take half the serial time
    # (assuming 50% of gates can run in parallel)
    gate_time = single_qubit_time + two_qubit_time
    estimated_time = gate_time * (1.0 - parallelization_factor) + meas_time

    # But estimated time shouldn't be less than the critical path
    # (depth * slowest gate time per layer).
    # Use the appropriate per-layer gate time: if the circuit has two-qubit
    # gates, assume layers may contain them; otherwise use single-qubit time.
    depth = summary["depth"]
    if two_qubit_gates > 0:
        per_layer_time = max(single_qubit_gate_time_us, two_qubit_gate_time_us)
    else:
        per_layer_time = single_qubit_gate_time_us
    critical_path_time = depth * per_layer_time + meas_time
    estimated_time = max(estimated_time, critical_path_time)

    _logger.debug(
        "Execution time estimate for %s: %.2f μs (serial: %.2f μs)",
        encoding.__class__.__name__,
        estimated_time,
        serial_time,
    )

    return {
        "serial_time_us": serial_time,
        "estimated_time_us": estimated_time,
        "single_qubit_time_us": single_qubit_time,
        "two_qubit_time_us": two_qubit_time,
        "measurement_time_us": meas_time,
        "parallelization_factor": parallelization_factor,
    }


# =============================================================================
# Private Helper Functions
# =============================================================================


def _validate_encoding(encoding: Any) -> None:
    """Validate that encoding is a valid BaseEncoding instance.

    Performs comprehensive validation to ensure the encoding object is valid
    and has consistent, usable properties before any analysis operations.

    Parameters
    ----------
    encoding : Any
        Object to validate. Should be a ``BaseEncoding`` instance.

    Raises
    ------
    AnalysisError
        If any of the following conditions are met:

        - ``encoding`` is not a ``BaseEncoding`` instance
        - Required properties (``n_features``, ``n_qubits``, ``depth``) are
          not accessible
        - ``n_qubits < 1`` (invalid quantum circuit)
        - ``n_features < 1`` (no features to encode)
        - ``depth < 0`` (invalid circuit depth)

    Notes
    -----
    This validation is performed at the start of all public analysis functions
    to provide clear, early error messages. The validation checks are ordered
    from most common errors (wrong type) to more specific issues (invalid
    property values).

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> _validate_encoding(enc)  # No exception raised

    >>> _validate_encoding("not an encoding")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    AnalysisError: Expected BaseEncoding instance, got str...
    """
    # Check type first - most common error
    if not isinstance(encoding, BaseEncoding):
        raise AnalysisError(
            f"Expected BaseEncoding instance, got {type(encoding).__name__}. "
            "Please provide a valid encoding from encoding_atlas.",
            details={"actual_type": type(encoding).__name__},
        )

    # Validate basic properties are accessible
    try:
        n_features = encoding.n_features
        n_qubits = encoding.n_qubits
        depth = encoding.depth
    except Exception as e:
        raise AnalysisError(
            f"Encoding {encoding.__class__.__name__} has invalid properties: {e}",
            details={"error": str(e)},
        ) from e

    # Validate property values are within valid ranges
    # These checks catch edge cases that could cause division by zero or
    # meaningless results in downstream computations.
    if n_qubits < 1:
        raise AnalysisError(
            f"Encoding {encoding.__class__.__name__} has invalid n_qubits={n_qubits}. "
            "A quantum circuit must have at least 1 qubit.",
            details={
                "encoding": encoding.__class__.__name__,
                "n_qubits": n_qubits,
            },
        )

    if n_features < 1:
        raise AnalysisError(
            f"Encoding {encoding.__class__.__name__} has invalid n_features={n_features}. "
            "An encoding must have at least 1 feature to encode.",
            details={
                "encoding": encoding.__class__.__name__,
                "n_features": n_features,
            },
        )

    if depth < 0:
        raise AnalysisError(
            f"Encoding {encoding.__class__.__name__} has invalid depth={depth}. "
            "Circuit depth cannot be negative.",
            details={
                "encoding": encoding.__class__.__name__,
                "depth": depth,
            },
        )


def _validate_input_data(
    x: NDArray[np.floating[Any]] | Any,
    n_features: int,
) -> NDArray[np.floating[Any]]:
    """Validate input data for data-dependent resource counting.

    Parameters
    ----------
    x : array-like
        Input data to validate.
    n_features : int
        Expected number of features.

    Returns
    -------
    NDArray[np.floating]
        Validated input array.

    Raises
    ------
    ValidationError
        If input is invalid.
    """
    try:
        x_array = np.asarray(x, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert input to numpy array: {e}") from e

    # Handle 2D input (single sample)
    if x_array.ndim == 2:
        if x_array.shape[0] != 1:
            raise ValidationError(
                f"For batch input, use a single sample. Got shape {x_array.shape}"
            )
        x_array = x_array[0]

    if x_array.ndim != 1:
        raise ValidationError(f"Input must be 1D array, got shape {x_array.shape}")

    if x_array.shape[0] != n_features:
        raise ValidationError(f"Expected {n_features} features, got {x_array.shape[0]}")

    if np.any(np.isnan(x_array)) or np.any(np.isinf(x_array)):
        raise ValidationError("Input contains NaN or infinite values")

    return x_array


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Safely compute ratio with zero handling.

    Parameters
    ----------
    numerator : int or float
        The numerator.
    denominator : int or float
        The denominator.

    Returns
    -------
    float
        The ratio, or 0.0 if denominator is zero.
    """
    if denominator == 0 or abs(denominator) < _EPSILON:
        return 0.0
    return float(numerator) / float(denominator)


def _get_summary_from_protocol(encoding: BaseEncoding) -> ResourceCountSummary:
    """Get resource summary from ResourceAnalyzable protocol.

    This function extracts resource information from encodings that implement
    the ``ResourceAnalyzable`` protocol, providing a standardized summary
    regardless of the specific encoding implementation details.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding that implements ResourceAnalyzable protocol. The encoding
        must provide both ``resource_summary()`` and ``gate_count_breakdown()``
        methods.

    Returns
    -------
    ResourceCountSummary
        Complete resource summary with all standard fields populated.

    Notes
    -----
    **Protocol Structure Flexibility**

    Different encodings may structure their ``resource_summary()`` return values
    differently. This function handles the following variations:

    - **parameter_count**: May be at the top level of ``resource_summary()`` or
      may need to be retrieved from ``encoding.properties.parameter_count`` as
      a fallback. Both approaches are supported.

    - **Rotation gates**: Encodings may use different naming conventions:
      - ``rx``, ``ry``, ``rz`` (e.g., AngleEncoding)
      - ``rz_single``, ``rz_zz`` (e.g., IQPEncoding for ZZ interactions)

      All variants are aggregated without double-counting.

    - **Gate naming**: Supports both ``hadamard``/``h`` and ``cnot``/``cx``
      naming conventions with automatic fallback.

    See Also
    --------
    ResourceAnalyzable : Protocol defining the expected interface.
    _get_detailed_breakdown_from_protocol : For detailed gate-by-gate breakdown.
    """
    # Get data from protocol methods
    resource_dict = encoding.resource_summary()  # type: ignore[attr-defined]
    gate_breakdown = encoding.gate_count_breakdown()  # type: ignore[attr-defined]

    total_gates = gate_breakdown.get("total", 0)
    total_single = gate_breakdown.get("total_single_qubit", 0)
    total_two = gate_breakdown.get("total_two_qubit", 0)

    # Extract rotation gate counts
    # Note: Encodings use different naming conventions for rotation gates:
    # - AngleEncoding uses: rx, ry, rz
    # - IQPEncoding uses: rz_single (single-qubit RZ), rz_zz (RZ in ZZ blocks)
    # We aggregate all variants - they are mutually exclusive per encoding.
    rx_count = gate_breakdown.get("rx", 0)
    ry_count = gate_breakdown.get("ry", 0)
    rz_count = gate_breakdown.get("rz", 0)
    rz_single = gate_breakdown.get("rz_single", 0)
    rz_zz = gate_breakdown.get("rz_zz", 0)

    rotation_gates = rx_count + ry_count + rz_count + rz_single + rz_zz

    # Extract specific gate counts with fallback naming conventions
    # Some encodings use "hadamard", others use "h"
    hadamard_count = gate_breakdown.get("hadamard", gate_breakdown.get("h", 0))
    # Some encodings use "cnot", others use "cx"
    cnot_count = gate_breakdown.get("cnot", gate_breakdown.get("cx", 0))
    cz_count = gate_breakdown.get("cz", 0)
    t_count = gate_breakdown.get("t", 0)

    n_qubits = encoding.n_qubits

    # Parameter count: try resource_dict first, fall back to properties
    # Note: Some encodings store parameter_count at top level of resource_summary(),
    # while others (e.g., IQPEncoding) store it within nested structures or rely
    # on properties. The fallback ensures consistent behavior across all encodings.
    parameter_count = resource_dict.get(
        "parameter_count",
        encoding.properties.parameter_count,
    )

    return {
        "n_qubits": n_qubits,
        "depth": encoding.depth,
        "gate_count": total_gates,
        "single_qubit_gates": total_single,
        "two_qubit_gates": total_two,
        "parameter_count": parameter_count,
        "cnot_count": cnot_count,
        "cz_count": cz_count,
        "t_gate_count": t_count,
        "hadamard_count": hadamard_count,
        "rotation_gates": rotation_gates,
        "two_qubit_ratio": _safe_divide(total_two, total_gates),
        "gates_per_qubit": _safe_divide(total_gates, n_qubits),
        "encoding_name": encoding.__class__.__name__,
        "is_data_dependent": False,
    }


def _get_detailed_breakdown_from_protocol(
    encoding: BaseEncoding,
) -> DetailedGateBreakdown:
    """Get detailed breakdown from ResourceAnalyzable protocol.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding that implements ResourceAnalyzable.

    Returns
    -------
    DetailedGateBreakdown
        Detailed gate-by-gate breakdown.
    """
    breakdown = encoding.gate_count_breakdown()  # type: ignore[attr-defined]

    # Map various gate naming conventions
    rz_total = (
        breakdown.get("rz", 0)
        + breakdown.get("rz_single", 0)
        + breakdown.get("rz_zz", 0)
    )

    return {
        "rx": breakdown.get("rx", 0),
        "ry": breakdown.get("ry", 0),
        "rz": rz_total,
        "h": breakdown.get("hadamard", breakdown.get("h", 0)),
        "x": breakdown.get("x", 0),
        "y": breakdown.get("y", 0),
        "z": breakdown.get("z", 0),
        "s": breakdown.get("s", 0),
        "t": breakdown.get("t", 0),
        "cnot": breakdown.get("cnot", breakdown.get("cx", 0)),
        "cx": breakdown.get("cx", breakdown.get("cnot", 0)),
        "cz": breakdown.get("cz", 0),
        "swap": breakdown.get("swap", 0),
        "total_single_qubit": breakdown.get("total_single_qubit", 0),
        "total_two_qubit": breakdown.get("total_two_qubit", 0),
        "total": breakdown.get("total", 0),
        "encoding_name": encoding.__class__.__name__,
    }


def _get_summary_from_properties(
    encoding: BaseEncoding,
    detailed: bool,
) -> ResourceCountSummary | DetailedGateBreakdown:
    """Fallback: get resources from encoding properties.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    detailed : bool
        Whether to return detailed breakdown.

    Returns
    -------
    ResourceCountSummary or DetailedGateBreakdown
        Resource information from properties.
    """
    props = encoding.properties

    if detailed:
        # Best-effort detailed breakdown from properties
        # We don't know exact gate types, so make reasonable assumptions
        return {
            "rx": 0,
            "ry": props.single_qubit_gates,  # Assume rotation gates
            "rz": 0,
            "h": 0,
            "x": 0,
            "y": 0,
            "z": 0,
            "s": 0,
            "t": 0,
            "cnot": props.two_qubit_gates,  # Assume CNOTs
            "cx": props.two_qubit_gates,
            "cz": 0,
            "swap": 0,
            "total_single_qubit": props.single_qubit_gates,
            "total_two_qubit": props.two_qubit_gates,
            "total": props.gate_count,
            "encoding_name": encoding.__class__.__name__,
        }

    return get_resource_summary(encoding)


def _count_data_dependent_resources(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    detailed: bool,
) -> ResourceCountSummary | DetailedGateBreakdown:
    """Count resources for data-dependent encodings.

    This function handles encodings where the circuit structure (specifically
    gate count) depends on the actual input data values. The canonical example
    is ``BasisEncoding``, which applies X gates only where input values exceed
    a threshold.

    Parameters
    ----------
    encoding : BaseEncoding
        Data-dependent encoding that implements the
        ``DataDependentResourceAnalyzable`` protocol. Must provide an
        ``actual_gate_count(x)`` method.
    x : NDArray[np.floating]
        Input data for resource counting. Shape must be ``(n_features,)``.
    detailed : bool
        If True, return detailed gate-by-gate breakdown.
        If False, return summary with totals.

    Returns
    -------
    ResourceCountSummary or DetailedGateBreakdown
        Resource information computed for the specific input data.

    Notes
    -----
    **Current Implementation Assumptions**

    This function currently assumes the following characteristics for
    data-dependent encodings (based on ``BasisEncoding``):

    1. **Single-qubit gates only**: All data-dependent gates are single-qubit
       gates (specifically X gates for BasisEncoding). No two-qubit gates
       vary with input data.

    2. **No parameterized gates**: Data-dependent encodings are assumed to
       have ``parameter_count = 0`` since they don't use rotation angles
       from input data (unlike angle encodings).

    3. **Depth is data-independent**: Circuit depth is assumed to be constant
       regardless of input (depth=1 for BasisEncoding).

    4. **Gate type is X**: For detailed breakdown, all data-dependent gates
       are assumed to be Pauli-X gates.

    **Extending for Future Encodings**

    If new data-dependent encodings are added that don't fit these assumptions
    (e.g., encodings with data-dependent two-qubit gates), this function
    should be extended to:

    1. Query the encoding for gate type information via protocol methods
    2. Support data-dependent two-qubit gate counts
    3. Handle variable depth based on input data

    The ``DataDependentResourceAnalyzable`` protocol should be extended
    accordingly to provide this additional information.

    See Also
    --------
    DataDependentResourceAnalyzable : Protocol for data-dependent encodings.
    BasisEncoding : Primary use case for this function.

    Examples
    --------
    >>> from encoding_atlas import BasisEncoding
    >>> import numpy as np
    >>> enc = BasisEncoding(n_features=4, threshold=0.5)
    >>> x = np.array([0.1, 0.9, 0.3, 0.8])  # 2 values > threshold
    >>> resources = _count_data_dependent_resources(enc, x, detailed=False)
    >>> resources['gate_count']  # Only 2 X gates applied
    2
    """
    # Get data-dependent resource summary if available
    resource_dict: dict[str, Any] = {}
    if hasattr(encoding, "resource_summary"):
        try:
            # Call with input data for data-dependent encodings
            resource_dict = encoding.resource_summary(x)
        except TypeError:
            # Fallback if method doesn't accept x
            try:
                resource_dict = encoding.resource_summary()
            except Exception as e:
                _logger.warning(
                    "Failed to get resource_summary for %s: %s",
                    encoding.__class__.__name__,
                    e,
                )
                resource_dict = {}

    # Get actual gate count for specific input
    actual_gates = 0
    if hasattr(encoding, "actual_gate_count"):
        try:
            actual_gates = encoding.actual_gate_count(x)
        except Exception as e:
            _logger.warning(
                "Failed to get actual_gate_count for %s: %s",
                encoding.__class__.__name__,
                e,
            )
            # Fall back to worst-case from properties
            actual_gates = encoding.properties.gate_count

    n_qubits = encoding.n_qubits
    depth = resource_dict.get("depth", encoding.properties.depth)

    if detailed:
        # Current assumption: Data-dependent gates are X gates (BasisEncoding).
        # For future encodings with different gate types, this should query
        # the encoding for gate type information via an extended protocol.
        return {
            "rx": 0,
            "ry": 0,
            "rz": 0,
            "h": 0,
            "x": actual_gates,  # Data-dependent X gate count (BasisEncoding)
            "y": 0,
            "z": 0,
            "s": 0,
            "t": 0,
            "cnot": 0,
            "cx": 0,
            "cz": 0,
            "swap": 0,
            "total_single_qubit": actual_gates,
            "total_two_qubit": 0,  # Assumption: no data-dependent 2Q gates
            "total": actual_gates,
            "encoding_name": encoding.__class__.__name__,
        }

    # Summary format
    # Note: two_qubit_gates=0 and parameter_count=0 based on current
    # assumptions for data-dependent encodings. See docstring for details.
    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "gate_count": actual_gates,
        "single_qubit_gates": actual_gates,
        "two_qubit_gates": 0,
        "parameter_count": 0,
        "cnot_count": 0,
        "cz_count": 0,
        "t_gate_count": 0,
        "hadamard_count": 0,
        "rotation_gates": 0,
        "two_qubit_ratio": 0.0,
        "gates_per_qubit": _safe_divide(actual_gates, n_qubits),
        "encoding_name": encoding.__class__.__name__,
        "is_data_dependent": True,
    }
