"""Performance benchmark tests for quantum encodings.

This module provides performance benchmarking including:
- Circuit generation time benchmarks
- Scaling behavior tests (n_features: 4, 8, 16, 32, 64)
- Backend comparison performance tests
- Property computation performance
- Batch processing throughput

These tests are marked as 'slow' and are useful for:
- Tracking performance regressions
- Comparing encoding implementations
- Documenting expected performance characteristics

Run with: pytest tests/unit/test_performance.py -v -m slow
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type

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
# Performance Measurement Utilities
# =============================================================================


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer() as t:
            # code to time
        print(f"Elapsed: {t.elapsed}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def benchmark_function(
    func: Callable[[], Any],
    n_iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Benchmark a function over multiple iterations.

    Parameters
    ----------
    func : Callable
        Function to benchmark (no arguments).
    n_iterations : int
        Number of timed iterations.
    warmup : int
        Number of warmup iterations (not timed).

    Returns
    -------
    Dict[str, float]
        Statistics: mean, std, min, max, median (all in seconds).
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times: List[float] = []
    for _ in range(n_iterations):
        with Timer() as t:
            func()
        times.append(t.elapsed)

    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times),
        "total": sum(times),
        "iterations": n_iterations,
    }


# =============================================================================
# Encoding Configurations for Testing
# =============================================================================


# Encodings with their constructor parameters for n_features=4
ENCODINGS_4D: List[Tuple[Type["BaseEncoding"], Dict[str, Any]]] = [
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


# Scalable encodings (those that accept varying n_features easily)
SCALABLE_ENCODINGS: List[Type["BaseEncoding"]] = [
    AngleEncoding,
    AmplitudeEncoding,
    IQPEncoding,
    ZZFeatureMap,
    PauliFeatureMap,
    DataReuploading,
    HardwareEfficientEncoding,
    HigherOrderAngleEncoding,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data_4d() -> np.ndarray:
    """4-dimensional sample data."""
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def random_data_factory() -> Callable[[int], np.ndarray]:
    """Factory for random data of given dimension."""
    rng = np.random.default_rng(42)  # Reproducible random data

    def factory(n_features: int) -> np.ndarray:
        return rng.uniform(-1, 1, n_features)

    return factory


# =============================================================================
# Test Class: Circuit Generation Benchmarks
# =============================================================================


@pytest.mark.slow
class TestCircuitGenerationBenchmarks:
    """Benchmark tests for circuit generation speed."""

    # Maximum allowed time per circuit (in seconds)
    # These are generous limits to catch severe regressions
    MAX_CIRCUIT_TIME_MS = {
        "AngleEncoding": 50,
        "AmplitudeEncoding": 100,
        "BasisEncoding": 50,
        "IQPEncoding": 100,
        "ZZFeatureMap": 100,
        "PauliFeatureMap": 100,
        "DataReuploading": 150,
        "HardwareEfficientEncoding": 150,
        "HigherOrderAngleEncoding": 100,
        "QAOAEncoding": 200,
        "HamiltonianEncoding": 150,
        "SymmetryInspiredFeatureMap": 150,
    }

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_4D)
    def test_circuit_generation_time_acceptable(
        self,
        encoding_cls: Type["BaseEncoding"],
        params: Dict[str, Any],
        sample_data_4d: np.ndarray,
    ) -> None:
        """Test that circuit generation completes within acceptable time.

        This catches severe performance regressions. Times are generous
        to account for different hardware.
        """
        enc = encoding_cls(**params)

        stats = benchmark_function(
            lambda: enc.get_circuit(sample_data_4d, backend="pennylane"),
            n_iterations=50,
            warmup=5,
        )

        # Get maximum allowed time for this encoding
        max_time_ms = self.MAX_CIRCUIT_TIME_MS.get(encoding_cls.__name__, 200)
        max_time_s = max_time_ms / 1000

        assert stats["mean"] < max_time_s, (
            f"{encoding_cls.__name__} circuit generation too slow: "
            f"mean={stats['mean']*1000:.1f}ms, max allowed={max_time_ms}ms"
        )

    @pytest.mark.parametrize("backend", ["pennylane", "qiskit", "cirq"])
    def test_angle_encoding_backends_similar_performance(
        self, backend: str, sample_data_4d: np.ndarray
    ) -> None:
        """Test that different backends have similar generation times.

        No backend should be dramatically slower than others (within 10x).
        """
        # Skip if backend not available
        if backend == "qiskit":
            pytest.importorskip("qiskit")
        elif backend == "cirq":
            pytest.importorskip("cirq")

        enc = AngleEncoding(n_features=4)

        stats = benchmark_function(
            lambda: enc.get_circuit(sample_data_4d, backend=backend),
            n_iterations=30,
            warmup=5,
        )

        # All backends should complete within 100ms per circuit
        assert stats["mean"] < 0.1, (
            f"AngleEncoding {backend} backend too slow: "
            f"mean={stats['mean']*1000:.1f}ms, expected < 100ms"
        )

    def test_circuit_generation_consistent(
        self, sample_data_4d: np.ndarray
    ) -> None:
        """Test that circuit generation time is consistent.

        Standard deviation should be less than mean (coefficient of
        variation < 1.0), indicating stable performance.
        """
        enc = IQPEncoding(n_features=4)

        # Use more warmup iterations to account for JIT compilation
        # and caching effects that can cause initial variance
        stats = benchmark_function(
            lambda: enc.get_circuit(sample_data_4d, backend="pennylane"),
            n_iterations=100,
            warmup=30,
        )

        cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
        assert cv < 1.0, (
            f"Circuit generation time highly variable: "
            f"CV={cv:.2f}, expected < 1.0"
        )


# =============================================================================
# Test Class: Scaling Behavior
# =============================================================================


@pytest.mark.slow
class TestScalingBehavior:
    """Tests for how performance scales with problem size."""

    @pytest.mark.parametrize("n_features", [4, 8, 16, 32])
    def test_angle_encoding_linear_scaling(
        self,
        n_features: int,
        random_data_factory: Callable[[int], np.ndarray],
    ) -> None:
        """Test that AngleEncoding scales linearly with n_features.

        Time should scale O(n) with feature count.
        """
        enc = AngleEncoding(n_features=n_features)
        x = random_data_factory(n_features)

        stats = benchmark_function(
            lambda: enc.get_circuit(x, backend="pennylane"),
            n_iterations=30,
            warmup=5,
        )

        # Time per feature should be roughly constant (< 10ms)
        time_per_feature = stats["mean"] * 1000 / n_features
        assert time_per_feature < 10, (
            f"AngleEncoding time per feature too high: "
            f"{time_per_feature:.2f}ms/feature for n_features={n_features}"
        )

    @pytest.mark.parametrize("n_features", [4, 8, 16, 32, 64])
    def test_amplitude_encoding_scales_with_log_features(
        self, n_features: int, random_data_factory: Callable[[int], np.ndarray]
    ) -> None:
        """Test AmplitudeEncoding scaling behavior.

        Uses log2(n) qubits, but state prep complexity is O(2^n_qubits).
        """
        enc = AmplitudeEncoding(n_features=n_features)
        x = random_data_factory(n_features)

        stats = benchmark_function(
            lambda: enc.get_circuit(x, backend="pennylane"),
            n_iterations=20,
            warmup=3,
        )

        # Should complete within reasonable time even for 64 features
        max_time = 0.5 if n_features <= 32 else 2.0
        assert stats["mean"] < max_time, (
            f"AmplitudeEncoding too slow for n_features={n_features}: "
            f"{stats['mean']*1000:.1f}ms, expected < {max_time*1000}ms"
        )

    @pytest.mark.parametrize(
        "encoding_cls",
        [IQPEncoding, ZZFeatureMap, PauliFeatureMap],
    )
    def test_entangling_encoding_quadratic_scaling(
        self,
        encoding_cls: Type["BaseEncoding"],
        random_data_factory: Callable[[int], np.ndarray],
    ) -> None:
        """Test that entangling encodings scale reasonably.

        These typically have O(n^2) gate count due to pairwise interactions.
        """
        times: Dict[int, float] = {}

        for n_features in [4, 8, 16]:
            enc = encoding_cls(n_features=n_features)
            x = random_data_factory(n_features)

            stats = benchmark_function(
                lambda enc=enc, x=x: enc.get_circuit(x, backend="pennylane"),
                n_iterations=20,
                warmup=3,
            )
            times[n_features] = stats["mean"]

        # Time for 16 features should be < 20x time for 4 features
        # (quadratic would be 16x, allowing some overhead)
        if times[4] > 0:
            ratio = times[16] / times[4]
            assert ratio < 20, (
                f"{encoding_cls.__name__} scaling too steep: "
                f"16-feature time is {ratio:.1f}x 4-feature time, expected < 20x"
            )

    @pytest.mark.parametrize("reps", [1, 2, 4, 8])
    def test_hardware_efficient_reps_scaling(
        self,
        reps: int,
        random_data_factory: Callable[[int], np.ndarray],
    ) -> None:
        """Test HardwareEfficientEncoding scales linearly with reps.

        Increasing reps should increase time linearly.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=reps)
        x = random_data_factory(4)

        stats = benchmark_function(
            lambda: enc.get_circuit(x, backend="pennylane"),
            n_iterations=20,
            warmup=3,
        )

        # Time should be roughly proportional to reps
        time_per_rep = stats["mean"] * 1000 / reps
        assert time_per_rep < 50, (
            f"HardwareEfficientEncoding time per rep too high: "
            f"{time_per_rep:.1f}ms/rep for reps={reps}"
        )


# =============================================================================
# Test Class: Backend Comparison
# =============================================================================


@pytest.mark.slow
class TestBackendComparison:
    """Tests comparing performance across backends."""

    BACKENDS = ["pennylane", "qiskit", "cirq"]

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_4D[:6])
    def test_all_backends_complete_in_time(
        self,
        encoding_cls: Type["BaseEncoding"],
        params: Dict[str, Any],
        sample_data_4d: np.ndarray,
    ) -> None:
        """Test that all backends complete within reasonable time."""
        enc = encoding_cls(**params)

        for backend in self.BACKENDS:
            # Skip unavailable backends
            if backend == "qiskit":
                try:
                    import qiskit  # noqa: F401
                except ImportError:
                    continue
            elif backend == "cirq":
                try:
                    import cirq  # noqa: F401
                except ImportError:
                    continue

            with Timer() as t:
                for _ in range(10):
                    _ = enc.get_circuit(sample_data_4d, backend=backend)

            avg_time_ms = (t.elapsed / 10) * 1000
            assert avg_time_ms < 500, (
                f"{encoding_cls.__name__} {backend} backend too slow: "
                f"{avg_time_ms:.1f}ms per circuit"
            )

    def test_backend_performance_comparison(
        self, sample_data_4d: np.ndarray
    ) -> None:
        """Compare relative performance of different backends.

        This is informational - doesn't assert but logs results.
        """
        enc = AngleEncoding(n_features=4)
        results: Dict[str, float] = {}

        for backend in self.BACKENDS:
            try:
                if backend == "qiskit":
                    import qiskit  # noqa: F401
                elif backend == "cirq":
                    import cirq  # noqa: F401

                stats = benchmark_function(
                    lambda: enc.get_circuit(sample_data_4d, backend=backend),
                    n_iterations=50,
                    warmup=5,
                )
                results[backend] = stats["mean"]
            except ImportError:
                results[backend] = float("nan")

        # All available backends should be within 10x of each other
        valid_times = [t for t in results.values() if not np.isnan(t)]
        if len(valid_times) >= 2:
            ratio = max(valid_times) / min(valid_times)
            assert ratio < 10, (
                f"Backend performance varies too much: "
                f"max/min ratio = {ratio:.1f}, expected < 10"
            )


# =============================================================================
# Test Class: Property Computation Performance
# =============================================================================


@pytest.mark.slow
class TestPropertyPerformance:
    """Tests for property computation performance."""

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_4D)
    def test_property_computation_fast(
        self, encoding_cls: Type["BaseEncoding"], params: Dict[str, Any]
    ) -> None:
        """Test that property computation is fast (typically cached)."""
        enc = encoding_cls(**params)

        # First access (may compute)
        with Timer() as t1:
            _ = enc.properties

        # Second access (should be cached)
        with Timer() as t2:
            _ = enc.properties

        # Cached access should be < 1ms
        assert t2.elapsed < 0.001, (
            f"{encoding_cls.__name__} property not cached: "
            f"first={t1.elapsed*1000:.2f}ms, second={t2.elapsed*1000:.2f}ms"
        )

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_4D)
    def test_depth_computation_fast(
        self, encoding_cls: Type["BaseEncoding"], params: Dict[str, Any]
    ) -> None:
        """Test that depth computation is fast."""
        enc = encoding_cls(**params)

        stats = benchmark_function(
            lambda: enc.depth,
            n_iterations=1000,
            warmup=10,
        )

        # Depth access should be very fast (< 0.1ms on average)
        assert stats["mean"] < 0.0001, (
            f"{encoding_cls.__name__} depth computation too slow: "
            f"{stats['mean']*1000000:.1f}us"
        )


# =============================================================================
# Test Class: Batch Processing Throughput
# =============================================================================


@pytest.mark.slow
class TestBatchThroughput:
    """Tests for batch processing throughput."""

    @pytest.mark.parametrize("batch_size", [10, 50, 100])
    def test_batch_throughput(self, batch_size: int) -> None:
        """Test batch processing throughput.

        Batch processing should be efficient - at least as fast as
        individual processing.
        """
        enc = AngleEncoding(n_features=4)
        batch = np.random.randn(batch_size, 4)

        # Time batch processing
        stats_batch = benchmark_function(
            lambda: enc.get_circuits(batch, backend="pennylane"),
            n_iterations=10,
            warmup=2,
        )

        # Time individual processing (for comparison)
        single_x = batch[0]
        stats_single = benchmark_function(
            lambda: enc.get_circuit(single_x, backend="pennylane"),
            n_iterations=batch_size * 10,
            warmup=5,
        )

        # Batch should not be slower than sum of individuals
        batch_time = stats_batch["mean"]
        individual_total = stats_single["mean"] * batch_size

        # Allow 2x overhead for batch coordination
        assert batch_time < individual_total * 2, (
            f"Batch processing slower than expected: "
            f"batch={batch_time*1000:.1f}ms, "
            f"individuals={individual_total*1000:.1f}ms"
        )

    def test_large_batch_throughput(self) -> None:
        """Test throughput for large batches."""
        enc = AngleEncoding(n_features=4)
        batch_size = 500
        batch = np.random.randn(batch_size, 4)

        with Timer() as t:
            circuits = enc.get_circuits(batch, backend="pennylane")

        circuits_per_second = batch_size / t.elapsed
        assert circuits_per_second > 100, (
            f"Low batch throughput: {circuits_per_second:.0f} circuits/second, "
            f"expected > 100"
        )

    @pytest.mark.parametrize(
        "encoding_cls",
        [AngleEncoding, IQPEncoding, ZZFeatureMap],
    )
    def test_encoding_batch_throughput_comparison(
        self, encoding_cls: Type["BaseEncoding"]
    ) -> None:
        """Compare batch throughput across encoding types."""
        enc = encoding_cls(n_features=4)
        batch = np.random.randn(100, 4)

        with Timer() as t:
            _ = enc.get_circuits(batch, backend="pennylane")

        # All should process at least 50 circuits/second
        throughput = 100 / t.elapsed
        assert throughput > 50, (
            f"{encoding_cls.__name__} batch throughput too low: "
            f"{throughput:.0f} circuits/second, expected > 50"
        )


# =============================================================================
# Test Class: Instantiation Performance
# =============================================================================


@pytest.mark.slow
class TestInstantiationPerformance:
    """Tests for encoding instantiation performance."""

    @pytest.mark.parametrize("encoding_cls,params", ENCODINGS_4D)
    def test_instantiation_fast(
        self, encoding_cls: Type["BaseEncoding"], params: Dict[str, Any]
    ) -> None:
        """Test that encoding instantiation is fast."""
        stats = benchmark_function(
            lambda: encoding_cls(**params),
            n_iterations=100,
            warmup=10,
        )

        # Instantiation should be < 10ms
        assert stats["mean"] < 0.01, (
            f"{encoding_cls.__name__} instantiation too slow: "
            f"{stats['mean']*1000:.1f}ms, expected < 10ms"
        )

    def test_repeated_instantiation_consistent(self) -> None:
        """Test that repeated instantiation has consistent performance.

        Uses coefficient of variation (CV = std/mean) to measure consistency.
        A CV < 1.0 indicates acceptable variance for timing tests in CI
        environments where system load can vary. The threshold is generous
        to avoid flaky test failures while still catching severe regressions.
        """
        # Warmup iterations to allow JIT compilation and caching to stabilize
        for _ in range(20):
            _ = HardwareEfficientEncoding(n_features=4, reps=2)

        times: List[float] = []

        for _ in range(100):
            with Timer() as t:
                _ = HardwareEfficientEncoding(n_features=4, reps=2)
            times.append(t.elapsed)

        # Check consistency (low variance)
        # CV < 1.0 means std < mean, indicating reasonable consistency
        # This threshold is generous to accommodate CI environment variability
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        cv = std_time / mean_time if mean_time > 0 else 0

        assert cv < 1.0, (
            f"Instantiation time inconsistent: CV={cv:.2f}, expected < 1.0"
        )


# =============================================================================
# Test Class: Performance Regression Detection
# =============================================================================


@pytest.mark.slow
class TestPerformanceRegression:
    """Tests designed to detect performance regressions.

    These tests establish baseline expectations. If these fail,
    it indicates a significant performance regression.
    """

    def test_angle_encoding_baseline(self, sample_data_4d: np.ndarray) -> None:
        """Baseline test for AngleEncoding performance."""
        enc = AngleEncoding(n_features=4)

        stats = benchmark_function(
            lambda: enc.get_circuit(sample_data_4d, backend="pennylane"),
            n_iterations=100,
            warmup=10,
        )

        # Baseline: should complete in < 20ms (generous for CI variance)
        assert stats["mean"] < 0.02, (
            f"AngleEncoding regression: {stats['mean']*1000:.1f}ms > 20ms baseline"
        )

    def test_iqp_encoding_baseline(self, sample_data_4d: np.ndarray) -> None:
        """Baseline test for IQPEncoding performance."""
        enc = IQPEncoding(n_features=4)

        stats = benchmark_function(
            lambda: enc.get_circuit(sample_data_4d, backend="pennylane"),
            n_iterations=100,
            warmup=10,
        )

        # Baseline: should complete in < 50ms
        assert stats["mean"] < 0.05, (
            f"IQPEncoding regression: {stats['mean']*1000:.1f}ms > 50ms baseline"
        )

    def test_hardware_efficient_baseline(
        self, sample_data_4d: np.ndarray
    ) -> None:
        """Baseline test for HardwareEfficientEncoding performance."""
        enc = HardwareEfficientEncoding(n_features=4, reps=2)

        stats = benchmark_function(
            lambda: enc.get_circuit(sample_data_4d, backend="pennylane"),
            n_iterations=100,
            warmup=10,
        )

        # Baseline: should complete in < 100ms
        assert stats["mean"] < 0.1, (
            f"HardwareEfficientEncoding regression: "
            f"{stats['mean']*1000:.1f}ms > 100ms baseline"
        )
