"""Base encoding class and protocols."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

if TYPE_CHECKING:
    pass


class BaseEncoding(ABC):
    """Abstract base class for all quantum data encodings.

    All encoding implementations must inherit from this class and implement
    the required abstract methods.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode.
    **kwargs : Any
        Additional encoding-specific parameters.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> encoding = AngleEncoding(n_features=4, rotation='Y')
    >>> encoding.n_qubits
    4

    Notes
    -----
    This class uses __slots__ for memory efficiency. Subclasses should also
    define __slots__ listing their own instance attributes to maintain this
    optimization. If a subclass does not define __slots__, it will have a
    __dict__ and the memory benefit is partially lost.
    """

    __slots__ = (
        "_n_features",
        "_config",
        "_properties",
        "_properties_lock",
    )

    def __init__(self, n_features: int, **kwargs: Any) -> None:
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError(f"n_features must be a positive integer, got {n_features}")

        self._n_features = n_features
        self._config = kwargs
        self._properties: EncodingProperties | None = None
        # Thread lock for safe lazy initialization of properties
        # Uses double-checked locking pattern for efficiency
        self._properties_lock: threading.Lock = threading.Lock()

    @property
    def n_features(self) -> int:
        """Number of classical features this encoding accepts."""
        return self._n_features

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Number of qubits required for this encoding."""
        ...

    @property
    @abstractmethod
    def depth(self) -> int:
        """Circuit depth of the encoding."""
        ...

    @property
    def config(self) -> dict[str, Any]:
        """Encoding configuration parameters."""
        return self._config.copy()

    @property
    def properties(self) -> EncodingProperties:
        """Compute and return encoding properties.

        This property uses thread-safe lazy initialization with double-checked
        locking to ensure safe access in multi-threaded environments while
        minimizing lock contention for subsequent accesses.

        Returns
        -------
        EncodingProperties
            Computed properties of this encoding.

        Notes
        -----
        Thread Safety:
            The first access in a multi-threaded context will acquire a lock
            to ensure only one thread computes the properties. Subsequent
            accesses bypass the lock entirely for optimal performance.

        The double-checked locking pattern:
            1. First check without lock (fast path for initialized case)
            2. Acquire lock
            3. Second check inside lock (handles race condition)
            4. Compute if still None
        """
        # Fast path: already initialized (no lock needed)
        if self._properties is not None:
            return self._properties

        # Slow path: need to initialize (thread-safe)
        with self._properties_lock:
            # Double-check inside lock to handle race condition
            # Another thread may have initialized while we waited for lock
            if self._properties is None:
                self._properties = self._compute_properties()

        return self._properties

    @abstractmethod
    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for the given data.

        Parameters
        ----------
        x : ArrayLike
            Input data of shape (n_features,) for a single sample.
        backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.
        """
        ...

    @abstractmethod
    def get_circuits(
        self,
        X: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> list[CircuitType]:
        """Generate quantum circuits for multiple data samples.

        Parameters
        ----------
        X : ArrayLike
            Input data of shape (n_samples, n_features).
        backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
            Target quantum computing framework.

        Returns
        -------
        list[CircuitType]
            List of circuits, one per sample.
        """
        ...

    @abstractmethod
    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding."""
        ...

    def _validate_input(
        self,
        x: ArrayLike,
        ensure_copy: bool = True,
        make_immutable: bool = True,
    ) -> NDArray[np.floating[Any]]:
        """Validate and preprocess input data with thread-safety guarantees.

        This method validates input data shape and values, and optionally
        ensures the returned array is an owned copy that cannot be modified.

        Parameters
        ----------
        x : ArrayLike
            Input data to validate. Can be a list, tuple, or numpy array.
        ensure_copy : bool, default=True
            If True, always return an array that owns its data (not a view).
            This prevents issues where the caller modifies the original array
            while the encoding is still using it.

            Set to False only if you are certain the input will not be
            modified and you need maximum performance.
        make_immutable : bool, default=True
            If True, set the returned array's writeable flag to False.
            This catches accidental mutations early with a clear error.

            Only effective when ensure_copy is True (cannot make a view
            immutable without affecting the original).

        Returns
        -------
        ndarray of float64
            Validated input array with shape (n_features,) or
            (n_samples, n_features).

        Raises
        ------
        ValueError
            If the input shape doesn't match n_features, or if the input
            contains NaN or infinite values.

        Notes
        -----
        **Thread Safety:**

        When ``ensure_copy=True`` (the default), this method guarantees that
        the returned array is independent of the input. This means:

        1. Concurrent calls to ``get_circuit`` with the same input array
           are safe, even if another thread modifies the array.

        2. The encoding's internal computations cannot be affected by
           external modifications to the input.

        **Performance:**

        For performance-critical code where thread safety is not a concern
        (e.g., single-threaded batch processing), you can set
        ``ensure_copy=False`` to avoid the copy overhead. However, this is
        generally not recommended unless profiling shows it's necessary.

        **Immutability:**

        When ``make_immutable=True``, any attempt to modify the returned
        array will raise a ValueError with a clear message. This helps
        catch bugs where code accidentally mutates the validated input.

        Examples
        --------
        >>> enc = SomeEncoding(n_features=4)
        >>> x = np.array([1.0, 2.0, 3.0, 4.0])
        >>> x_validated = enc._validate_input(x)
        >>> x_validated.flags.owndata  # Always owns its data
        True
        >>> x_validated.flags.writeable  # Immutable by default
        False
        >>> x[0] = 999  # Original can still be modified
        >>> x_validated[0]  # But validated copy is unchanged
        1.0
        """
        # =======================================================================
        # TYPE VALIDATION (before numpy conversion)
        # =======================================================================
        # Check for string inputs which numpy would silently convert
        # This catches common user mistakes like passing ["0.5", "0.3"] instead of [0.5, 0.3]
        if isinstance(x, (list, tuple)):
            if len(x) > 0 and isinstance(x[0], str):
                raise TypeError(
                    f"Input contains string values. Expected numeric data, got {type(x[0]).__name__}. "
                    "Convert strings to floats before encoding."
                )
        elif isinstance(x, np.ndarray) and x.dtype.kind in (
            "U",
            "S",
            "O",
        ):  # Unicode, byte string, or object dtype
            raise TypeError(
                f"Input array has non-numeric dtype '{x.dtype}'. "
                "Expected numeric data (float or int)."
            )

        # =======================================================================
        # COMPLEX NUMBER VALIDATION
        # =======================================================================
        # Convert to array first WITHOUT forcing dtype to detect complex values.
        # np.asarray(complex_data, dtype=np.float64) silently discards imaginary
        # parts, which is silent data corruption. We must catch this explicitly.
        x_temp = np.asarray(x)
        if np.issubdtype(x_temp.dtype, np.complexfloating):
            raise TypeError(
                f"Input contains complex values (dtype: {x_temp.dtype}). "
                "Complex numbers are not supported. Use real-valued data only."
            )

        # Convert to numpy array with float64 dtype
        # Now safe since we've verified no complex values
        x_array = x_temp.astype(np.float64, copy=False)

        # =======================================================================
        # SHAPE VALIDATION
        # =======================================================================
        if x_array.ndim == 1:
            if x_array.shape[0] != self.n_features:
                raise ValueError(
                    f"Expected {self.n_features} features, got {x_array.shape[0]}"
                )
        elif x_array.ndim == 2:
            if x_array.shape[1] != self.n_features:
                raise ValueError(
                    f"Expected {self.n_features} features, got {x_array.shape[1]}"
                )
        else:
            raise ValueError(f"Input must be 1D or 2D array, got {x_array.ndim}D")

        # =======================================================================
        # VALUE VALIDATION
        # =======================================================================
        if np.any(np.isnan(x_array)) or np.any(np.isinf(x_array)):
            raise ValueError("Input contains NaN or infinite values")

        # =======================================================================
        # DEFENSIVE COPY (Thread Safety)
        # =======================================================================
        # IMPORTANT: np.asarray may return the SAME object if input is already
        # a float64 ndarray. The `owndata` flag is True for the original array
        # (since it owns its data), so checking owndata alone is NOT sufficient.
        #
        # To ensure complete isolation from the caller's array:
        # 1. We always copy when ensure_copy=True
        # 2. This guarantees the caller can modify their original array without
        #    affecting our computations
        # 3. This also ensures setting writeable=False doesn't affect the original
        #
        # The copy overhead is minimal for typical input sizes (4-100 features).
        if ensure_copy:
            # Always create an owned copy for complete isolation
            x_array = x_array.copy()

            # Make immutable to catch accidental internal mutations
            # This is a debugging aid that catches bugs early
            if make_immutable:
                x_array.flags.writeable = False

        return x_array

    def __repr__(self) -> str:
        config_str = ", ".join(f"{k}={v!r}" for k, v in self._config.items())
        if config_str:
            return (
                f"{self.__class__.__name__}(n_features={self.n_features}, {config_str})"
            )
        return f"{self.__class__.__name__}(n_features={self.n_features})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEncoding):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.n_features == other.n_features
            and self._config == other._config
        )

    def __hash__(self) -> int:
        def _make_hashable(obj: Any) -> Any:
            """Convert unhashable types to hashable equivalents."""
            if isinstance(obj, list):
                return tuple(_make_hashable(item) for item in obj)
            elif isinstance(obj, dict):
                return tuple(
                    sorted(
                        (_make_hashable(k), _make_hashable(v)) for k, v in obj.items()
                    )
                )
            elif isinstance(obj, set):
                return frozenset(_make_hashable(item) for item in obj)
            return obj

        hashable_config = tuple(
            sorted((k, _make_hashable(v)) for k, v in self._config.items())
        )
        return hash((self.__class__.__name__, self.n_features, hashable_config))

    def __getstate__(self) -> dict[str, Any]:
        """Prepare instance state for pickling.

        This method enables pickle serialization for encoding objects by
        excluding the unpicklable thread lock. The lock is recreated during
        unpickling via __setstate__.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'slots' and optionally 'dict' keys with
            the respective attribute values.

        Notes
        -----
        This implementation handles both:
        1. Slot attributes from the base class hierarchy (traversing MRO)
        2. Instance __dict__ for subclasses that don't define __slots__

        The cached `_properties` value is preserved if it was computed,
        avoiding recomputation after unpickling.
        """
        state: dict[str, Any] = {"slots": {}, "dict": None}

        # Collect slot attributes from the entire class hierarchy
        for cls in type(self).__mro__:
            if hasattr(cls, "__slots__"):
                for slot in cls.__slots__:
                    # Skip the lock - it cannot be pickled
                    if slot == "_properties_lock":
                        continue
                    # Only include slots that have been set
                    if hasattr(self, slot):
                        state["slots"][slot] = getattr(self, slot)

        # Handle __dict__ for subclasses that don't define __slots__
        # (they inherit slots from BaseEncoding but store their own attrs in __dict__)
        if hasattr(self, "__dict__") and self.__dict__:
            state["dict"] = self.__dict__.copy()

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state after unpickling.

        This method restores all serialized slot values and dict attributes,
        then creates a new thread lock for thread-safe property access.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary from __getstate__ containing 'slots' and 'dict' keys.

        Notes
        -----
        The thread lock is recreated fresh, ensuring the unpickled object
        is fully functional for concurrent access.
        """
        # Restore slot values
        for slot, value in state.get("slots", {}).items():
            setattr(self, slot, value)

        # Restore __dict__ attributes if present
        if state.get("dict"):
            if not hasattr(self, "__dict__"):
                # This shouldn't happen, but handle gracefully
                object.__setattr__(self, "__dict__", {})
            self.__dict__.update(state["dict"])

        # Recreate the thread lock
        self._properties_lock = threading.Lock()
