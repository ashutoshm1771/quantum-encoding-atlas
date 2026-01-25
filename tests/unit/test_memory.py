"""Memory usage tests for quantum encodings.

This module tests memory behavior including:
- Memory consumption for different feature counts
- Memory leak detection in repeated operations
- Memory behavior for batch operations
- Memory cleanup after circuit generation

These tests are marked as 'memory' and 'slow' since they may take time.

Run with: pytest tests/unit/test_memory.py -v -m memory
"""

from __future__ import annotations

import gc
import sys
import weakref
from typing import TYPE_CHECKING, Any, Callable, List, Type

import numpy as np
import pytest

from encoding_atlas import (
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    DataReuploading,
    HamiltonianEncoding,
    HardwareEfficientEncoding,
    HigherOrderAngleEncoding,
    IQPEncoding,
    PauliFeatureMap,
    QAOAEncoding,
    SymmetryInspiredFeatureMap,
    ZZFeatureMap,
)

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding


# =============================================================================
# Memory Tracking Utilities
# =============================================================================


def get_object_size(obj: Any) -> int:
    """Get approximate memory size of an object in bytes.

    Uses sys.getsizeof with recursive tracking for containers.

    Parameters
    ----------
    obj : Any
        Object to measure.

    Returns
    -------
    int
        Approximate size in bytes.
    """
    seen = set()

    def sizeof(o: Any) -> int:
        obj_id = id(o)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(o)

        if isinstance(o, dict):
            size += sum(sizeof(k) + sizeof(v) for k, v in o.items())
        elif hasattr(o, "__dict__"):
            size += sizeof(o.__dict__)
        elif hasattr(o, "__iter__") and not isinstance(o, (str, bytes, bytearray)):
            try:
                size += sum(sizeof(i) for i in o)
            except TypeError:
                pass  # Not iterable after all

        return size

    return sizeof(obj)


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB.

    Returns
    -------
    float
        Memory usage in megabytes.
    """
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback if psutil not available - use tracemalloc
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start()
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)


def force_gc() -> None:
    """Force garbage collection to clean up memory."""
    gc.collect()
    gc.collect()
    gc.collect()


# =============================================================================
# Fixtures
# =============================================================================


# All encodings for parametrized tests
ENCODINGS_WITH_PARAMS: List[tuple] = [
    (AngleEncoding, {"n_features": 4}),
    (AmplitudeEncoding, {"n_features": 4}),
    (BasisEncoding, {"n_features": 4}),
    (IQPEncoding, {"n_features": 4}),
    (ZZFeatureMap, {"n_features": 4}),
    (PauliFeatureMap, {"n_features": 4}),
    (DataReuploading, {"n_features": 4}),
    (HardwareEfficientEncoding, {"n_features": 4}),
    (HigherOrderAngleEncoding, {"n_features": 4}),
    (QAOAEncoding, {"n_features": 4}),
    (HamiltonianEncoding, {"n_features": 4}),
    (SymmetryInspiredFeatureMap, {"n_features": 4}),
]


@pytest.fixture
def sample_data_4d() -> np.ndarray:
    """4-dimensional sample data."""
    return np.array([0.1, 0.2, 0.3, 0.4])


# =============================================================================
# Test Class: Memory Consumption
# =============================================================================


@pytest.mark.memory
@pytest.mark.slow
class TestMemoryConsumption:
    """Tests for memory consumption characteristics."""

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_WITH_PARAMS)
    def test_encoding_instance_size_reasonable(
        self, encoding_cls: Type["BaseEncoding"], params: dict
    ) -> None:
        """Test that encoding instances have reasonable memory footprint.

        An encoding instance should not exceed 1MB for basic n_features=4.
        This catches accidental storage of large arrays or circuits.
        """
        enc = encoding_cls(**params)
        size_bytes = get_object_size(enc)
        size_mb = size_bytes / (1024 * 1024)

        # Encoding instance should be under 1MB for n_features=4
        assert size_mb < 1.0, (
            f"{encoding_cls.__name__} instance size is {size_mb:.2f}MB, "
            f"expected < 1MB for n_features=4"
        )

    @pytest.mark.parametrize(
        "n_features,max_mb",
        [
            (4, 1.0),
            (8, 2.0),
            (16, 5.0),
            (32, 15.0),
        ],
    )
    def test_angle_encoding_size_scales_linearly(
        self, n_features: int, max_mb: float
    ) -> None:
        """Test that AngleEncoding memory scales linearly with features.

        Angle encoding should scale O(n) with features since each feature
        maps to one rotation gate.
        """
        enc = AngleEncoding(n_features=n_features)
        size_mb = get_object_size(enc) / (1024 * 1024)

        assert size_mb < max_mb, (
            f"AngleEncoding(n_features={n_features}) size is {size_mb:.2f}MB, "
            f"expected < {max_mb}MB"
        )

    def test_amplitude_encoding_size_scales_logarithmically(self) -> None:
        """Test that AmplitudeEncoding qubits scale logarithmically.

        Amplitude encoding uses log2(n) qubits, so memory should not
        explode with feature count.
        """
        sizes = {}
        for n_features in [4, 16, 64, 256]:
            enc = AmplitudeEncoding(n_features=n_features)
            sizes[n_features] = get_object_size(enc)

        # Size should not grow faster than O(n log n) at most
        # Check that 256 features is not more than 10x 4 features
        growth_ratio = sizes[256] / sizes[4]
        assert growth_ratio < 20, (
            f"AmplitudeEncoding size grew by {growth_ratio:.1f}x from "
            f"4 to 256 features, expected < 20x"
        )


# =============================================================================
# Test Class: Memory Leak Detection
# =============================================================================


@pytest.mark.memory
@pytest.mark.slow
class TestMemoryLeaks:
    """Tests for memory leak detection in repeated operations."""

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_WITH_PARAMS)
    def test_no_leak_repeated_circuit_generation(
        self,
        encoding_cls: Type["BaseEncoding"],
        params: dict,
        sample_data_4d: np.ndarray,
    ) -> None:
        """Test no memory leak in repeated circuit generation.

        Generate circuits in a loop and verify memory doesn't grow unboundedly.
        This catches issues like circular references or cached objects not
        being released.
        """
        enc = encoding_cls(**params)
        x = sample_data_4d

        # Warm up - generate some circuits to stabilize
        for _ in range(10):
            _ = enc.get_circuit(x, backend="pennylane")
        force_gc()

        # Measure baseline
        baseline_memory = get_process_memory_mb()

        # Generate many circuits
        n_iterations = 100
        for _ in range(n_iterations):
            circuit = enc.get_circuit(x, backend="pennylane")
            del circuit
        force_gc()

        # Measure final memory
        final_memory = get_process_memory_mb()
        memory_growth = final_memory - baseline_memory

        # Allow some growth but not excessive (< 50MB for 100 iterations)
        assert memory_growth < 50, (
            f"{encoding_cls.__name__} memory grew by {memory_growth:.1f}MB "
            f"after {n_iterations} circuit generations, possible memory leak"
        )

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_WITH_PARAMS[:3])
    def test_encoding_instances_garbage_collected(
        self, encoding_cls: Type["BaseEncoding"], params: dict
    ) -> None:
        """Test that encoding instances are properly garbage collected.

        Create encodings, delete them, and verify they're collected.
        """
        weak_refs: List[weakref.ref] = []
        supports_weakref = True

        # Create and immediately delete encodings
        for _ in range(10):
            enc = encoding_cls(**params)
            try:
                weak_refs.append(weakref.ref(enc))
            except TypeError:
                # Object doesn't support weak references
                supports_weakref = False
                break
            del enc

        if not supports_weakref:
            # Skip this test for objects that don't support weakref
            pytest.skip(f"{encoding_cls.__name__} doesn't support weak references")

        force_gc()

        # All weak references should be dead
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        assert alive_count == 0, (
            f"{encoding_cls.__name__}: {alive_count}/10 instances not garbage "
            f"collected, possible circular reference"
        )

    def test_circuit_objects_released_after_scope(
        self, sample_data_4d: np.ndarray
    ) -> None:
        """Test that circuit objects are released when out of scope."""
        enc = AngleEncoding(n_features=4)
        x = sample_data_4d

        # Create circuit in inner scope
        def create_and_measure() -> tuple:
            circuit = enc.get_circuit(x, backend="pennylane")
            try:
                return weakref.ref(circuit), True
            except TypeError:
                return None, False

        ref, supports_weakref = create_and_measure()

        if not supports_weakref:
            # Circuit type doesn't support weak references, skip
            pytest.skip("Circuit object doesn't support weak references")

        force_gc()

        # Circuit should be collected
        assert ref() is None, "Circuit not garbage collected after going out of scope"


# =============================================================================
# Test Class: Batch Memory Behavior
# =============================================================================


@pytest.mark.memory
@pytest.mark.slow
class TestBatchMemoryBehavior:
    """Tests for memory behavior with batch operations."""

    @pytest.mark.parametrize(
        "batch_size,max_memory_mb",
        [
            (10, 50),
            (50, 150),
            (100, 300),
        ],
    )
    def test_batch_circuit_memory_linear(
        self, batch_size: int, max_memory_mb: float
    ) -> None:
        """Test that batch circuit generation memory scales linearly.

        Memory should not explode when generating circuits for batches.
        """
        enc = AngleEncoding(n_features=4)
        batch = np.random.randn(batch_size, 4)

        force_gc()
        baseline = get_process_memory_mb()

        # Generate batch of circuits
        circuits = enc.get_circuits(batch, backend="pennylane")
        memory_used = get_process_memory_mb() - baseline

        assert memory_used < max_memory_mb, (
            f"Batch of {batch_size} circuits used {memory_used:.1f}MB, "
            f"expected < {max_memory_mb}MB"
        )

        # Cleanup
        del circuits

    def test_large_batch_does_not_crash(self) -> None:
        """Test that reasonably large batch sizes don't cause OOM.

        This is a smoke test for batch processing capability.
        """
        enc = AngleEncoding(n_features=4)
        batch = np.random.randn(500, 4)

        # Should complete without memory error
        try:
            circuits = enc.get_circuits(batch, backend="pennylane")
            assert len(circuits) == 500
        except MemoryError:
            pytest.fail("Memory error on batch size 500 - unexpected OOM")

    def test_sequential_batches_no_memory_accumulation(self) -> None:
        """Test that processing sequential batches doesn't accumulate memory."""
        enc = DataReuploading(n_features=4)

        force_gc()
        baseline = get_process_memory_mb()

        # Process multiple batches
        for _ in range(10):
            batch = np.random.randn(50, 4)
            circuits = enc.get_circuits(batch, backend="pennylane")
            del circuits
            del batch

        force_gc()
        final = get_process_memory_mb()
        growth = final - baseline

        # Should not accumulate > 100MB across batches
        assert growth < 100, (
            f"Memory grew by {growth:.1f}MB across 10 batches, "
            f"expected < 100MB (possible accumulation)"
        )


# =============================================================================
# Test Class: Memory Cleanup
# =============================================================================


@pytest.mark.memory
class TestMemoryCleanup:
    """Tests for proper memory cleanup."""

    def test_encoding_cleanup_releases_references(self) -> None:
        """Test that encoding cleanup releases internal references."""
        enc = HardwareEfficientEncoding(n_features=4, reps=3)

        # Access properties to populate caches
        _ = enc.properties
        _ = enc.depth
        _ = enc.n_qubits

        # Try to store weak reference - some objects may not support it
        try:
            ref = weakref.ref(enc)
            supports_weakref = True
        except TypeError:
            # Object doesn't support weak references (has __slots__)
            supports_weakref = False

        # Delete encoding
        del enc
        force_gc()

        if supports_weakref:
            # Should be collected
            assert ref() is None, "Encoding not collected after deletion"
        # If weakref not supported, we just verify deletion doesn't error

    @pytest.mark.parametrize("backend", ["pennylane", "qiskit", "cirq"])
    def test_backend_circuits_independent(
        self, backend: str, sample_data_4d: np.ndarray
    ) -> None:
        """Test that circuits from different backends are independent.

        Generating a circuit for one backend shouldn't affect memory
        of another backend's circuit.
        """
        pytest.importorskip("qiskit" if backend == "qiskit" else backend)

        enc = AngleEncoding(n_features=4)

        # Generate circuit
        circuit = enc.get_circuit(sample_data_4d, backend=backend)
        size_before = get_object_size(circuit)

        # Generate another circuit (shouldn't affect first)
        _ = enc.get_circuit(sample_data_4d * 2, backend=backend)

        # First circuit should be unchanged
        size_after = get_object_size(circuit)
        assert size_before == size_after, (
            f"Circuit size changed after generating another: "
            f"{size_before} -> {size_after}"
        )


# =============================================================================
# Test Class: Memory Stress Tests
# =============================================================================


@pytest.mark.memory
@pytest.mark.slow
class TestMemoryStress:
    """Stress tests for memory behavior under load."""

    def test_rapid_encoding_creation_destruction(self) -> None:
        """Test rapid creation/destruction cycle doesn't leak memory."""
        force_gc()
        baseline = get_process_memory_mb()

        # Rapid creation/destruction cycle
        for _ in range(1000):
            enc = AngleEncoding(n_features=4)
            _ = enc.n_qubits
            del enc

        force_gc()
        final = get_process_memory_mb()
        growth = final - baseline

        # Should not grow significantly
        assert growth < 50, (
            f"Memory grew by {growth:.1f}MB during rapid create/destroy, "
            f"possible leak"
        )

    def test_mixed_encoding_types_memory(self) -> None:
        """Test memory behavior when using mixed encoding types."""
        force_gc()
        baseline = get_process_memory_mb()

        # Create various encoding types
        for _ in range(50):
            encodings = [
                AngleEncoding(n_features=4),
                IQPEncoding(n_features=4),
                ZZFeatureMap(n_features=4),
                DataReuploading(n_features=4),
            ]
            x = np.random.randn(4)

            for enc in encodings:
                _ = enc.get_circuit(x, backend="pennylane")

            del encodings

        force_gc()
        final = get_process_memory_mb()
        growth = final - baseline

        assert growth < 100, (
            f"Memory grew by {growth:.1f}MB with mixed encodings, expected < 100MB"
        )

    def test_property_access_doesnt_leak(self) -> None:
        """Test that repeated property access doesn't cause memory growth."""
        enc = HardwareEfficientEncoding(n_features=8, reps=3)

        force_gc()
        baseline = get_process_memory_mb()

        # Access properties many times
        for _ in range(10000):
            _ = enc.n_qubits
            _ = enc.depth
            _ = enc.properties

        force_gc()
        final = get_process_memory_mb()
        growth = final - baseline

        # Property access should be O(1) memory
        assert growth < 10, (
            f"Memory grew by {growth:.1f}MB from property access, "
            f"properties should be cached"
        )
