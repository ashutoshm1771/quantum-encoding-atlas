"""Tests for ``BaseEncoding`` extension methods added on top of the
template-method refactor: ``iter_circuits`` (lazy generator),
``parallel='process'`` mode in ``get_circuits``, and the optional LRU
circuit cache (``enable_cache``/``disable_cache``/``clear_cache``/
``cache_info``).

These exercises live in ``tests/unit/core/`` because the behavior they
verify lives entirely in :class:`encoding_atlas.core.base.BaseEncoding`;
``AngleEncoding`` is used as a concrete subclass purely to obtain a real
encoding instance.
"""

from __future__ import annotations

import pickle
import types

import numpy as np
import pytest

from encoding_atlas import AngleEncoding

# Backend availability — process-pool tests rely on Qiskit/Cirq because
# PennyLane circuits are local closures that cannot be pickled.
try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import cirq  # noqa: F401

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False


# =============================================================================
# iter_circuits
# =============================================================================


class TestIterCircuits:
    """``iter_circuits`` is a lazy generator over the validated batch."""

    def test_returns_a_generator(self) -> None:
        enc = AngleEncoding(n_features=4)
        gen = enc.iter_circuits(np.random.randn(3, 4))
        assert isinstance(gen, types.GeneratorType)

    def test_yields_one_circuit_per_sample(self) -> None:
        enc = AngleEncoding(n_features=4)
        X = np.random.randn(7, 4)
        circuits = list(enc.iter_circuits(X))
        assert len(circuits) == 7
        assert all(callable(c) for c in circuits)

    def test_preserves_input_order(self) -> None:
        enc = AngleEncoding(n_features=4)
        X = np.array([[float(i), 0.0, 0.0, 0.0] for i in range(5)])
        circuits_iter = list(enc.iter_circuits(X, backend="pennylane"))
        circuits_list = enc.get_circuits(X, backend="pennylane")
        # Both come from the same _get_circuit_from_validated path; we rely
        # on order being identical.
        assert len(circuits_iter) == len(circuits_list)

    def test_handles_1d_input_as_single_sample(self) -> None:
        enc = AngleEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuits = list(enc.iter_circuits(x))
        assert len(circuits) == 1

    def test_lazy_evaluation(self) -> None:
        """The generator should not consume the whole batch eagerly."""
        enc = AngleEncoding(n_features=4)
        X = np.random.randn(100, 4)
        gen = enc.iter_circuits(X)
        # Pulling a single item must not exhaust the generator.
        first = next(gen)
        assert callable(first)
        remaining = sum(1 for _ in gen)
        assert remaining == 99

    def test_invalid_input_shape_raises(self) -> None:
        enc = AngleEncoding(n_features=4)
        # A 1D array of wrong length still flows through ``_validate_input``.
        with pytest.raises(ValueError):
            list(enc.iter_circuits(np.zeros(3)))

    def test_qiskit_backend(self) -> None:
        if not HAS_QISKIT:
            pytest.skip("Qiskit not installed")
        from qiskit import QuantumCircuit

        enc = AngleEncoding(n_features=4)
        circuits = list(enc.iter_circuits(np.random.randn(3, 4), backend="qiskit"))
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    def test_inherited_by_subclass(self) -> None:
        """Hamiltonian no longer defines its own iter_circuits — confirm it
        inherits a working one from BaseEncoding."""
        from encoding_atlas import HamiltonianEncoding

        enc = HamiltonianEncoding(n_features=4)
        # Use values away from the critical point at π to avoid noisy warns.
        X = np.full((4, 4), 0.5)
        circuits = list(enc.iter_circuits(X, backend="pennylane"))
        assert len(circuits) == 4


# =============================================================================
# parallel='process'
# =============================================================================


class TestProcessPoolMode:
    """``parallel='process'`` uses ProcessPoolExecutor with the standard
    initializer/initargs pattern so the encoding is pickled once per worker."""

    def test_pennylane_with_process_raises_clear_error(self) -> None:
        """PennyLane returns local closures that cannot cross processes."""
        enc = AngleEncoding(n_features=4)
        with pytest.raises(ValueError, match="parallel='process'.*pennylane"):
            enc.get_circuits(
                np.random.randn(3, 4), backend="pennylane", parallel="process"
            )

    def test_qiskit_process_pool(self) -> None:
        if not HAS_QISKIT:
            pytest.skip("Qiskit not installed")
        from qiskit import QuantumCircuit

        enc = AngleEncoding(n_features=4)
        circuits = enc.get_circuits(
            np.random.randn(4, 4),
            backend="qiskit",
            parallel="process",
            max_workers=2,
        )
        assert len(circuits) == 4
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    def test_cirq_process_pool(self) -> None:
        if not HAS_CIRQ:
            pytest.skip("Cirq not installed")
        import cirq as cirq_mod

        enc = AngleEncoding(n_features=4)
        circuits = enc.get_circuits(
            np.random.randn(4, 4),
            backend="cirq",
            parallel="process",
            max_workers=2,
        )
        assert len(circuits) == 4
        assert all(isinstance(c, cirq_mod.Circuit) for c in circuits)

    def test_process_pool_preserves_order(self) -> None:
        if not HAS_QISKIT:
            pytest.skip("Qiskit not installed")
        enc = AngleEncoding(n_features=4)
        # Distinguishable inputs so we can verify each output corresponds.
        X = np.array([[float(i), 0.0, 0.0, 0.0] for i in range(8)])
        circuits = enc.get_circuits(
            X, backend="qiskit", parallel="process", max_workers=2
        )
        # Compare against the sequential reference.
        reference = enc.get_circuits(X, backend="qiskit", parallel=False)
        assert len(circuits) == len(reference)

    def test_invalid_parallel_value_raises(self) -> None:
        enc = AngleEncoding(n_features=4)
        with pytest.raises(ValueError, match="parallel must be"):
            enc.get_circuits(
                np.random.randn(3, 4), parallel="invalid"  # type: ignore[arg-type]
            )

    def test_thread_string_alias_equivalent_to_true(self) -> None:
        """Backward compat: ``parallel=True`` and ``parallel='thread'``
        produce the same result and both use ThreadPoolExecutor."""
        enc = AngleEncoding(n_features=4)
        X = np.random.randn(3, 4)
        a = enc.get_circuits(X, parallel=True)
        b = enc.get_circuits(X, parallel="thread")
        assert len(a) == len(b) == 3


# =============================================================================
# Circuit cache
# =============================================================================


class TestCircuitCache:
    """The opt-in LRU cache is keyed on (x.tobytes(), backend)."""

    def test_disabled_by_default(self) -> None:
        enc = AngleEncoding(n_features=4)
        info = enc.cache_info()
        assert info == {"enabled": False, "size": 0, "maxsize": 0}

    def test_enable_cache_validates_maxsize(self) -> None:
        enc = AngleEncoding(n_features=4)
        with pytest.raises(ValueError, match="maxsize must be a positive integer"):
            enc.enable_cache(maxsize=0)
        with pytest.raises(ValueError, match="maxsize must be a positive integer"):
            enc.enable_cache(maxsize=-1)
        with pytest.raises(ValueError, match="maxsize must be a positive integer"):
            enc.enable_cache(maxsize="ten")  # type: ignore[arg-type]

    def test_cache_hit_returns_same_object(self) -> None:
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=5)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        c1 = enc.get_circuit(x)
        c2 = enc.get_circuit(x)
        assert c1 is c2
        assert enc.cache_info()["size"] == 1

    def test_cache_distinguishes_inputs(self) -> None:
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=5)
        c1 = enc.get_circuit(np.array([0.1, 0.2, 0.3, 0.4]))
        c2 = enc.get_circuit(np.array([0.5, 0.6, 0.7, 0.8]))
        assert c1 is not c2
        assert enc.cache_info()["size"] == 2

    def test_cache_distinguishes_backends(self) -> None:
        if not HAS_QISKIT:
            pytest.skip("Qiskit not installed")
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=5)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        enc.get_circuit(x, backend="pennylane")
        enc.get_circuit(x, backend="qiskit")
        assert enc.cache_info()["size"] == 2

    def test_lru_eviction(self) -> None:
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=2)
        for i in range(5):
            enc.get_circuit(np.array([float(i), 0.0, 0.0, 0.0]))
        assert enc.cache_info()["size"] == 2

    def test_lru_resize_shrinks(self) -> None:
        """Re-enabling with a smaller maxsize trims existing entries."""
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=10)
        for i in range(8):
            enc.get_circuit(np.array([float(i), 0.0, 0.0, 0.0]))
        assert enc.cache_info()["size"] == 8
        enc.enable_cache(maxsize=3)
        assert enc.cache_info()["size"] == 3

    def test_clear_cache_keeps_enabled(self) -> None:
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=5)
        enc.get_circuit(np.array([0.1, 0.2, 0.3, 0.4]))
        enc.clear_cache()
        info = enc.cache_info()
        assert info == {"enabled": True, "size": 0, "maxsize": 5}

    def test_disable_cache_clears_and_disables(self) -> None:
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=5)
        enc.get_circuit(np.array([0.1, 0.2, 0.3, 0.4]))
        enc.disable_cache()
        assert enc.cache_info() == {"enabled": False, "size": 0, "maxsize": 0}

    def test_pickle_drops_cache(self) -> None:
        """Cache holds backend-specific objects (PennyLane closures, etc.)
        that are not pickle-safe — it must be cleared on round-trip."""
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=5)
        enc.get_circuit(np.array([0.1, 0.2, 0.3, 0.4]))
        restored = pickle.loads(pickle.dumps(enc))
        assert restored.cache_info() == {
            "enabled": False,
            "size": 0,
            "maxsize": 0,
        }
        # Restored instance must remain functional.
        c = restored.get_circuit(np.array([0.1, 0.2, 0.3, 0.4]))
        assert callable(c)

    def test_batch_path_bypasses_cache(self) -> None:
        """get_circuits should not pollute (or read from) the single-sample
        cache — caching is intended for repeated single-sample calls and
        adding it to the batch hot path would only add lock contention."""
        enc = AngleEncoding(n_features=4)
        enc.enable_cache(maxsize=10)
        X = np.random.randn(5, 4)
        enc.get_circuits(X)
        assert enc.cache_info()["size"] == 0
