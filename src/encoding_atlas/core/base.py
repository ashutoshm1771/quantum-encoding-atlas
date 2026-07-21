"""Base encoding class and protocols."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-pool worker globals
# ---------------------------------------------------------------------------
# ProcessPoolExecutor pickles arguments per-task by default. To avoid shipping
# the full encoding instance with every sample, we use the standard
# initializer/initargs pattern: each worker process unpickles the encoding
# once at startup, then `_process_worker_generate` reads from module globals.
_WORKER_ENCODING: BaseEncoding | None = None
_WORKER_BACKEND: BackendType | None = None


def _process_worker_init(encoding: BaseEncoding, backend: BackendType) -> None:
    """Initializer for ProcessPoolExecutor workers (top-level for picklability)."""
    global _WORKER_ENCODING, _WORKER_BACKEND
    _WORKER_ENCODING = encoding
    _WORKER_BACKEND = backend


def _process_worker_generate(x: NDArray[np.floating[Any]]) -> CircuitType:
    """Worker entrypoint that reads encoding/backend from module globals."""
    assert (
        _WORKER_ENCODING is not None and _WORKER_BACKEND is not None
    ), "Process pool worker invoked before initializer ran"
    return _WORKER_ENCODING._get_circuit_from_validated(x, _WORKER_BACKEND)


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
        "_circuit_cache",
        "_circuit_cache_maxsize",
        "_circuit_cache_lock",
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

        # Optional LRU cache for circuit construction (disabled by default).
        # Enabled via :meth:`enable_cache`. ``None`` means caching is off.
        self._circuit_cache: OrderedDict[tuple[bytes, str], CircuitType] | None = None
        self._circuit_cache_maxsize: int = 0
        self._circuit_cache_lock: threading.Lock = threading.Lock()

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

    def get_circuit(
        self,
        x: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> CircuitType:
        """Generate quantum circuit for the given data.

        This is a template method that validates input and delegates to the
        encoding-specific :meth:`_get_circuit_from_validated`. Subclasses
        should normally override :meth:`_get_circuit_from_validated` rather
        than this method.

        If circuit caching has been enabled via :meth:`enable_cache`, the
        result for repeated calls with byte-identical inputs is returned
        from cache.

        Parameters
        ----------
        x : ArrayLike
            Input data of shape (n_features,) or (1, n_features) for a single
            sample. If 2D with a single row, the row is unwrapped before
            dispatch.
        backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.

        Raises
        ------
        ValueError
            If input shape doesn't match n_features, contains NaN/Inf values,
            or backend is not recognized.
        """
        x_validated = self._validate_input(x)
        if x_validated.ndim == 2:
            x_validated = x_validated[0]
        return self._cached_dispatch(x_validated, backend)

    def get_circuits(
        self,
        X: ArrayLike,
        backend: BackendType = "pennylane",
        *,
        parallel: Union[bool, Literal["thread", "process"]] = False,
        max_workers: int | None = None,
    ) -> list[CircuitType]:
        """Generate quantum circuits for multiple data samples.

        This is a template method that validates the batch and dispatches
        each sample through :meth:`_get_circuit_from_validated`. The batch
        is validated once; per-sample validation is skipped for performance.

        Parameters
        ----------
        X : ArrayLike
            Input data of shape (n_samples, n_features) or (n_features,).
            If 1D, treated as a single sample.
        backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
            Target quantum computing framework.
        parallel : bool or {'thread', 'process'}, default=False
            Parallelization mode for circuit generation:

            - ``False`` (default) — sequential, no executor overhead.
            - ``True`` or ``'thread'`` — :class:`ThreadPoolExecutor`.
              Best for I/O-bound work or PennyLane closures. Limited by
              the CPython GIL for pure-Python work.
            - ``'process'`` — :class:`ProcessPoolExecutor` with the
              standard initializer/initargs pattern (encoding pickled
              once per worker). Use for CPU-bound circuit construction
              such as Cirq's amplitude unitary materialization on large
              batches. Has process-startup overhead, so only worthwhile
              for batches of roughly 100+ samples. **Not supported with
              ``backend='pennylane'``** — PennyLane returns circuits as
              local closures which cannot be pickled across processes;
              use ``'thread'`` instead.

            Order of results is preserved across all modes.
        max_workers : int or None, default=None
            Maximum number of workers when ``parallel`` is enabled. If
            ``None``, the executor chooses a default based on CPU count.

        Returns
        -------
        list[CircuitType]
            List of circuits, one per sample, in input order.

        Raises
        ------
        ValueError
            If ``parallel`` is not one of the accepted values.

        Notes
        -----
        Thread safety: each call to :meth:`_get_circuit_from_validated`
        operates on an immutable validated copy of its input and does not
        mutate instance state, so concurrent invocation is safe.

        Per-sample caching (see :meth:`enable_cache`) is bypassed in
        batch mode — caching is intended for repeated single-sample
        calls and would add lock contention to batch hot paths.
        """
        # Resolve the parallel mode early so we can fail fast on bad input
        # before doing any expensive validation.
        mode = self._resolve_parallel_mode(parallel)

        # PennyLane returns circuits as local closures (qfuncs) that are not
        # picklable, so they cannot cross the process boundary. Reject the
        # combination up front with a helpful message instead of letting the
        # ProcessPoolExecutor fail mid-batch with a cryptic pickling error.
        if mode == "process" and backend == "pennylane":
            raise ValueError(
                "parallel='process' is not supported with backend='pennylane' "
                "because PennyLane circuits are local closures which cannot "
                "be pickled across process boundaries. Use parallel='thread' "
                "for PennyLane, or switch to backend='qiskit' or "
                "backend='cirq' for process-pool parallelism."
            )

        X_validated = self._validate_input(X)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)

        n_samples = X_validated.shape[0]

        _logger.debug(
            "Batch circuit generation: encoding=%s, n_samples=%d, "
            "backend=%r, parallel=%s, max_workers=%s",
            type(self).__name__,
            n_samples,
            backend,
            mode,
            max_workers,
        )

        # Sequential fast path (also used when only one sample is present).
        if mode == "sequential" or n_samples <= 1:
            return [self._get_circuit_from_validated(x, backend) for x in X_validated]

        if mode == "thread":
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(
                    executor.map(
                        lambda x: self._get_circuit_from_validated(x, backend),
                        X_validated,
                    )
                )

        # mode == "process"
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_process_worker_init,
            initargs=(self, backend),
        ) as executor:
            return list(executor.map(_process_worker_generate, X_validated))

    def iter_circuits(
        self,
        X: ArrayLike,
        backend: BackendType = "pennylane",
    ) -> Iterator[CircuitType]:
        """Yield quantum circuits one sample at a time (memory-efficient).

        Generator alternative to :meth:`get_circuits`: validates the batch
        once, then yields a single circuit per sample without ever
        materializing the full list. Use for datasets that don't fit in
        memory or when circuits can be processed and discarded as they
        come.

        Parameters
        ----------
        X : ArrayLike
            Input data of shape (n_samples, n_features) or (n_features,).
            If 1D, treated as a single sample.
        backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
            Target quantum computing framework.

        Yields
        ------
        CircuitType
            Circuit for each sample, in input order.

        Examples
        --------
        Stream circuits without loading the full list:

        >>> from itertools import islice
        >>> first_ten = list(islice(enc.iter_circuits(X_huge), 10))

        Notes
        -----
        Each yielded circuit holds only its own state. The generator
        itself is not thread-safe; for parallelism, use
        :meth:`get_circuits` with ``parallel='thread'`` or
        ``parallel='process'`` instead.
        """
        X_validated = self._validate_input(X)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)
        for x in X_validated:
            yield self._get_circuit_from_validated(x, backend)

    @staticmethod
    def _resolve_parallel_mode(
        parallel: Union[bool, Literal["thread", "process"]],
    ) -> Literal["sequential", "thread", "process"]:
        """Map the public ``parallel`` argument to an internal mode label."""
        if parallel is False:
            return "sequential"
        if parallel is True or parallel == "thread":
            return "thread"
        if parallel == "process":
            return "process"
        raise ValueError(
            f"parallel must be False, True, 'thread', or 'process', "
            f"got {parallel!r}"
        )

    # ------------------------------------------------------------------
    # Optional circuit cache (LRU, opt-in)
    # ------------------------------------------------------------------

    def enable_cache(self, maxsize: int = 128) -> None:
        """Enable LRU caching of single-sample circuit construction.

        When enabled, :meth:`get_circuit` returns cached circuits for
        byte-identical inputs (same ``backend`` and same
        ``x.tobytes()``). Useful for training loops that revisit the
        same data points across epochs.

        Parameters
        ----------
        maxsize : int, default=128
            Maximum number of entries before least-recently-used eviction.
            Must be a positive integer.

        Raises
        ------
        ValueError
            If ``maxsize`` is not a positive integer.

        Notes
        -----
        - The cache holds backend-specific circuit objects. PennyLane
          callables, Qiskit ``QuantumCircuit`` instances, and Cirq
          ``Circuit`` objects can all consume non-trivial memory; choose
          ``maxsize`` accordingly.
        - The cache is bypassed in :meth:`get_circuits` (batch mode) and
          :meth:`iter_circuits` to avoid lock contention.
        - The cache is dropped on pickle and must be re-enabled on the
          unpickled instance if desired.
        """
        if not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError(f"maxsize must be a positive integer, got {maxsize!r}")
        with self._circuit_cache_lock:
            self._circuit_cache_maxsize = maxsize
            if self._circuit_cache is None:
                self._circuit_cache = OrderedDict()
            elif len(self._circuit_cache) > maxsize:
                # Trim down to the new (smaller) limit.
                while len(self._circuit_cache) > maxsize:
                    self._circuit_cache.popitem(last=False)

    def disable_cache(self) -> None:
        """Disable circuit caching and discard all cached entries."""
        with self._circuit_cache_lock:
            self._circuit_cache = None
            self._circuit_cache_maxsize = 0

    def clear_cache(self) -> None:
        """Discard cached circuits while leaving caching enabled."""
        with self._circuit_cache_lock:
            if self._circuit_cache is not None:
                self._circuit_cache.clear()

    def cache_info(self) -> dict[str, int | bool]:
        """Return introspection data about the circuit cache.

        Returns
        -------
        dict
            ``enabled`` (bool), ``size`` (current entries), ``maxsize``
            (configured limit, 0 if disabled).
        """
        with self._circuit_cache_lock:
            return {
                "enabled": self._circuit_cache is not None,
                "size": (
                    len(self._circuit_cache) if self._circuit_cache is not None else 0
                ),
                "maxsize": self._circuit_cache_maxsize,
            }

    def _cached_dispatch(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Cache-aware wrapper around :meth:`_get_circuit_from_validated`.

        On a cache miss, computes outside the lock to avoid blocking
        other threads, then re-checks under the lock before inserting.
        """
        if self._circuit_cache is None:
            return self._get_circuit_from_validated(x, backend)

        cache_key = (x.tobytes(), backend)
        with self._circuit_cache_lock:
            cached = self._circuit_cache.get(cache_key)
            if cached is not None:
                self._circuit_cache.move_to_end(cache_key)
                return cached

        result = self._get_circuit_from_validated(x, backend)

        with self._circuit_cache_lock:
            # Read the attribute once under the lock; casting back to the
            # Optional type lets mypy see that a concurrent ``disable`` (which
            # sets the cache to ``None``) is still possible here.
            cache = cast(
                "OrderedDict[tuple[bytes, str], CircuitType] | None",
                self._circuit_cache,
            )
            if cache is None:
                return result
            # Re-check: another thread may have inserted while we computed.
            existing = cache.get(cache_key)
            if existing is not None:
                cache.move_to_end(cache_key)
                return existing
            cache[cache_key] = result
            if len(cache) > self._circuit_cache_maxsize:
                cache.popitem(last=False)
            return result

    @abstractmethod
    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input (subclass hook).

        This is the encoding-specific seam called by :meth:`get_circuit` and
        :meth:`get_circuits` after input has been validated. Subclasses
        implement encoding-specific preprocessing (padding, normalization,
        thresholding, etc.) and dispatch to backend-specific methods here.

        Parameters
        ----------
        x : NDArray
            Pre-validated 1D input features of shape (n_features,). Caller
            guarantees: dtype is float64, no NaN/Inf, length matches
            ``n_features``.
        backend : BackendType
            Target quantum computing framework.

        Returns
        -------
        CircuitType
            Circuit in the specified backend's format.

        Notes
        -----
        Internal method. External callers should use :meth:`get_circuit`,
        which performs full input validation. Implementations must be
        thread-safe (no instance mutation).
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

        # Slots that must NOT be pickled:
        # - threading.Lock instances cannot be pickled at all.
        # - The circuit cache may hold backend-specific closures
        #   (PennyLane qfuncs in particular) that are not pickle-safe; we
        #   drop the cache rather than risk a confusing pickle error.
        _NON_PICKLED_SLOTS = frozenset(
            {
                "_properties_lock",
                "_circuit_cache",
                "_circuit_cache_lock",
            }
        )

        # Collect slot attributes from the entire class hierarchy
        for cls in type(self).__mro__:
            if hasattr(cls, "__slots__"):
                for slot in cls.__slots__:
                    if slot in _NON_PICKLED_SLOTS:
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

        # Recreate non-picklable state. The circuit cache is intentionally
        # dropped on pickle (see __getstate__) — the unpickled instance
        # starts with caching disabled and can re-enable via enable_cache.
        self._properties_lock = threading.Lock()
        self._circuit_cache_lock = threading.Lock()
        self._circuit_cache = None
        self._circuit_cache_maxsize = 0
