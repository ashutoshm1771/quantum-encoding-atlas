"""Custom exceptions for encoding_atlas.

This module defines the exception hierarchy for the encoding_atlas package.
All custom exceptions inherit from :class:`EncodingError`, which itself
inherits from Python's built-in :class:`Exception`.

Exception Hierarchy
-------------------
::

    Exception
    └── EncodingError (base for all encoding_atlas exceptions)
        ├── ValidationError (input validation failures)
        ├── BackendError (quantum backend issues)
        ├── RegistryError (encoding registry issues)
        └── AnalysisError (analysis operation failures)
            ├── SimulationError (circuit simulation failures)
            ├── ConvergenceError (iterative computation failures)
            ├── NumericalInstabilityError (numerical issues)
            └── InsufficientSamplesError (sampling issues)

Examples
--------
Catching all encoding-related exceptions:

>>> try:
...     result = some_encoding_operation()
... except EncodingError as e:
...     print(f"Encoding operation failed: {e}")

Catching specific analysis exceptions:

>>> from encoding_atlas.core.exceptions import SimulationError
>>> try:
...     statevector = simulate_encoding(encoding, x)
... except SimulationError as e:
...     print(f"Simulation failed: {e}")
"""

from __future__ import annotations

__all__ = [
    # Base exceptions
    "EncodingError",
    "ValidationError",
    "BackendError",
    "RegistryError",
    # Analysis exceptions
    "AnalysisError",
    "SimulationError",
    "ConvergenceError",
    "NumericalInstabilityError",
    "InsufficientSamplesError",
]


# =============================================================================
# Base Exceptions
# =============================================================================


class EncodingError(Exception):
    """Base exception for all encoding_atlas errors.

    This is the root exception class for the encoding_atlas package.
    All custom exceptions in this package inherit from this class,
    allowing users to catch all encoding-related errors with a single
    except clause.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    *args : tuple
        Additional positional arguments passed to Exception.

    Examples
    --------
    >>> raise EncodingError("Something went wrong")
    Traceback (most recent call last):
        ...
    EncodingError: Something went wrong
    """

    pass


class ValidationError(EncodingError):
    """Raised when input validation fails.

    This exception indicates that input data does not meet the
    requirements of an encoding or analysis function. Common causes
    include incorrect shapes, invalid data types, or out-of-range values.

    Examples
    --------
    >>> raise ValidationError("Expected 4 features, got 3")
    Traceback (most recent call last):
        ...
    ValidationError: Expected 4 features, got 3
    """

    pass


class BackendError(EncodingError):
    """Raised when quantum backend operations fail.

    This exception indicates a failure in the underlying quantum
    computing framework (PennyLane, Qiskit, or Cirq). Common causes
    include missing backend installations, invalid circuit operations,
    or backend-specific limitations.

    Examples
    --------
    >>> raise BackendError("Qiskit backend not available")
    Traceback (most recent call last):
        ...
    BackendError: Qiskit backend not available
    """

    pass


class RegistryError(EncodingError):
    """Raised when encoding registry operations fail.

    This exception indicates a failure in the encoding registration
    or lookup system. Common causes include duplicate registrations
    or requests for non-existent encodings.

    Examples
    --------
    >>> raise RegistryError("Encoding 'CustomEncoding' not found in registry")
    Traceback (most recent call last):
        ...
    RegistryError: Encoding 'CustomEncoding' not found in registry
    """

    pass


# =============================================================================
# Analysis Exceptions
# =============================================================================


class AnalysisError(EncodingError):
    """Base exception for analysis operations.

    This exception class serves as the base for all analysis-related
    errors in the encoding_atlas.analysis module. It inherits from
    :class:`EncodingError` to maintain the package's exception hierarchy.

    Use this exception for general analysis failures that don't fit
    into more specific categories. For specific failure modes, use
    the specialized subclasses.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    details : dict, optional
        Additional context about the error (e.g., parameter values,
        intermediate results).

    Attributes
    ----------
    details : dict
        Additional context about the error, if provided.

    Examples
    --------
    >>> raise AnalysisError("Analysis failed for encoding")
    Traceback (most recent call last):
        ...
    AnalysisError: Analysis failed for encoding

    With additional details:

    >>> raise AnalysisError(
    ...     "Invalid encoding for analysis",
    ...     details={"encoding_type": "CustomEncoding", "n_qubits": 0}
    ... )
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details: dict = details or {}


class SimulationError(AnalysisError):
    """Raised when quantum circuit simulation fails.

    This exception indicates that a quantum circuit could not be
    successfully simulated. Common causes include:

    - Missing backend dependencies (PennyLane, Qiskit not installed)
    - Invalid circuit structure
    - Memory limitations for large qubit counts
    - Backend-specific errors

    Parameters
    ----------
    message : str
        Human-readable description of the simulation failure.
    backend : str, optional
        The backend that failed (e.g., "pennylane", "qiskit").
    details : dict, optional
        Additional context about the failure.

    Attributes
    ----------
    backend : str or None
        The backend that failed, if specified.
    details : dict
        Additional context about the failure.

    Examples
    --------
    >>> raise SimulationError("PennyLane simulation failed: device error")
    Traceback (most recent call last):
        ...
    SimulationError: PennyLane simulation failed: device error

    With backend information:

    >>> raise SimulationError(
    ...     "Simulation failed",
    ...     backend="qiskit",
    ...     details={"n_qubits": 20, "error": "memory exceeded"}
    ... )
    """

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details)
        self.backend: str | None = backend


class ConvergenceError(AnalysisError):
    """Raised when iterative computation fails to converge.

    This exception indicates that an iterative algorithm did not
    reach the desired convergence criteria within the allowed
    iterations or time. Common causes include:

    - Insufficient sample count
    - Poor initial conditions
    - Numerically unstable problem
    - Convergence criteria too strict

    Parameters
    ----------
    message : str
        Human-readable description of the convergence failure.
    iterations : int, optional
        Number of iterations attempted before failure.
    target_tolerance : float, optional
        The convergence tolerance that was not achieved.
    achieved_tolerance : float, optional
        The best tolerance achieved before stopping.
    details : dict, optional
        Additional context about the failure.

    Attributes
    ----------
    iterations : int or None
        Number of iterations attempted.
    target_tolerance : float or None
        The target convergence tolerance.
    achieved_tolerance : float or None
        The best tolerance achieved.
    details : dict
        Additional context about the failure.

    Examples
    --------
    >>> raise ConvergenceError(
    ...     "Expressibility estimation did not converge",
    ...     iterations=10000,
    ...     target_tolerance=1e-4,
    ...     achieved_tolerance=1e-2
    ... )
    """

    def __init__(
        self,
        message: str,
        iterations: int | None = None,
        target_tolerance: float | None = None,
        achieved_tolerance: float | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details)
        self.iterations: int | None = iterations
        self.target_tolerance: float | None = target_tolerance
        self.achieved_tolerance: float | None = achieved_tolerance


class NumericalInstabilityError(AnalysisError):
    """Raised when numerical instability is detected.

    This exception indicates that a computation encountered numerical
    issues that compromise the accuracy or validity of results. Common
    causes include:

    - Division by near-zero values
    - Overflow or underflow in intermediate calculations
    - Loss of precision in floating-point operations
    - Ill-conditioned matrices

    Parameters
    ----------
    message : str
        Human-readable description of the numerical issue.
    value : float, optional
        The problematic value that triggered the error.
    operation : str, optional
        The operation that encountered the issue.
    details : dict, optional
        Additional context about the numerical issue.

    Attributes
    ----------
    value : float or None
        The problematic value, if applicable.
    operation : str or None
        The operation that failed, if specified.
    details : dict
        Additional context about the failure.

    Examples
    --------
    >>> raise NumericalInstabilityError(
    ...     "Division by near-zero in fidelity computation",
    ...     value=1e-320,
    ...     operation="compute_fidelity"
    ... )

    Notes
    -----
    This exception is distinct from Python's built-in arithmetic
    exceptions (ZeroDivisionError, OverflowError) because it catches
    cases where the computation technically succeeds but produces
    unreliable results due to floating-point limitations.
    """

    def __init__(
        self,
        message: str,
        value: float | None = None,
        operation: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details)
        self.value: float | None = value
        self.operation: str | None = operation


class InsufficientSamplesError(AnalysisError):
    """Raised when sample count is too low for reliable results.

    This exception indicates that a statistical analysis cannot
    provide reliable results with the given number of samples.
    This is distinct from a convergence failure — here, the
    algorithm recognizes upfront that the sample count is
    inadequate.

    Parameters
    ----------
    message : str
        Human-readable description of the sampling issue.
    requested_samples : int, optional
        The number of samples requested by the user.
    minimum_samples : int, optional
        The minimum number of samples required for reliable results.
    metric : str, optional
        The metric being computed (e.g., "expressibility").
    details : dict, optional
        Additional context about the sampling requirements.

    Attributes
    ----------
    requested_samples : int or None
        The number of samples requested.
    minimum_samples : int or None
        The minimum required samples.
    metric : str or None
        The metric being computed.
    details : dict
        Additional context about the failure.

    Examples
    --------
    >>> raise InsufficientSamplesError(
    ...     "Too few samples for expressibility estimation",
    ...     requested_samples=10,
    ...     minimum_samples=100,
    ...     metric="expressibility"
    ... )

    Notes
    -----
    When this exception is raised, the recommended action is to
    increase the sample count. The ``minimum_samples`` attribute
    provides guidance on the minimum viable sample count.
    """

    def __init__(
        self,
        message: str,
        requested_samples: int | None = None,
        minimum_samples: int | None = None,
        metric: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details)
        self.requested_samples: int | None = requested_samples
        self.minimum_samples: int | None = minimum_samples
        self.metric: str | None = metric
