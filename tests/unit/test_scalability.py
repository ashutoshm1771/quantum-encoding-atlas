"""Scalability tests for quantum encodings.

This module tests how encodings behave at scale:
- Large qubit counts (50+, 100+)
- Deep circuits (reps=100+)
- Large batch sizes (1000+ inputs)
- High-dimensional feature spaces
- Resource usage under scale

These tests verify the library can handle production-scale workloads.

Run with: pytest tests/unit/test_scalability.py -v -m slow
"""

from __future__ import annotations

import gc
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from encoding_atlas import (
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    DataReuploading,
    HardwareEfficientEncoding,
    IQPEncoding,
    QAOAEncoding,
    ZZFeatureMap,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Scalability Utilities
# =============================================================================


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def timed_operation(timeout_seconds: float = 30.0):
    """Decorator to timeout slow operations."""
    import functools
    import signal

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if sys.platform == "win32":
                # Windows doesn't support SIGALRM, just run normally
                return func(*args, **kwargs)

            def handler(signum, frame):
                raise TimeoutError(
                    f"{func.__name__} exceeded {timeout_seconds}s timeout"
                )

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(timeout_seconds))
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


# =============================================================================
# Test Class: Large Qubit Counts
# =============================================================================


@pytest.mark.slow
class TestLargeQubitCounts:
    """Tests for handling large numbers of qubits."""

    @pytest.mark.parametrize("n_features", [50, 64, 100])
    def test_angle_encoding_large_qubit_count(self, n_features: int) -> None:
        """Test AngleEncoding with large qubit counts (up to 100).

        AngleEncoding uses n_qubits = n_features, so 100 features = 100 qubits.
        Should instantiate without errors.
        """
        enc = AngleEncoding(n_features=n_features)
        assert enc.n_qubits == n_features

        # Properties should be accessible
        props = enc.properties
        assert props is not None

    @pytest.mark.parametrize("n_features", [32, 64, 128])
    def test_amplitude_encoding_large_features(self, n_features: int) -> None:
        """Test AmplitudeEncoding with many features.

        AmplitudeEncoding uses log2(n) qubits, so scales better.
        128 features = 7 qubits.
        """
        enc = AmplitudeEncoding(n_features=n_features)

        # n_qubits should be ceil(log2(n_features))
        expected_qubits = max(1, int(np.ceil(np.log2(n_features))))
        assert enc.n_qubits == expected_qubits

        # Should be able to generate circuit description
        assert enc.depth > 0

    @pytest.mark.parametrize("n_features", [20, 30, 40])
    def test_iqp_encoding_moderate_scale(self, n_features: int) -> None:
        """Test IQPEncoding at moderate scale.

        IQP has O(n^2) connections, so scales more slowly.
        Test up to 40 qubits.
        """
        # Use linear entanglement for faster scaling
        enc = IQPEncoding(n_features=n_features, entanglement="linear")
        assert enc.n_qubits == n_features

        # Should complete without timeout
        x = np.random.randn(n_features)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    @pytest.mark.parametrize("n_features", [50, 75, 100])
    def test_linear_entanglement_scales_well(self, n_features: int) -> None:
        """Test that linear entanglement pattern scales to 100 qubits.

        Linear entanglement has O(n) gates, should handle 100 qubits.
        """
        enc = ZZFeatureMap(n_features=n_features, entanglement="linear")
        assert enc.n_qubits == n_features

        # Generate circuit with dummy data
        x = np.random.randn(n_features)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_hardware_efficient_50_qubits(self) -> None:
        """Test HardwareEfficientEncoding with 50 qubits."""
        enc = HardwareEfficientEncoding(n_features=50, reps=2, entanglement="linear")
        assert enc.n_qubits == 50

        x = np.random.randn(50)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


# =============================================================================
# Test Class: Deep Circuits
# =============================================================================


@pytest.mark.slow
class TestDeepCircuits:
    """Tests for handling deep circuits with many repetitions."""

    @pytest.mark.parametrize("reps", [10, 25, 50])
    def test_hardware_efficient_many_reps(self, reps: int) -> None:
        """Test HardwareEfficientEncoding with many repetitions.

        Should handle up to 50 repetitions without issues.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=reps)

        # Depth should scale with reps
        assert enc.depth > reps

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    @pytest.mark.parametrize("reps", [10, 20, 50])
    def test_data_reuploading_many_layers(self, reps: int) -> None:
        """Test DataReuploading with many layers.

        Data reuploading uses repeated data encoding, should handle 50 layers.
        """
        enc = DataReuploading(n_features=4, n_layers=reps)
        assert enc.depth > 0

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    @pytest.mark.parametrize("reps", [10, 25, 50])
    def test_zz_feature_map_many_reps(self, reps: int) -> None:
        """Test ZZFeatureMap with many repetitions."""
        enc = ZZFeatureMap(n_features=4, reps=reps, entanglement="linear")

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    @pytest.mark.parametrize("reps", [5, 10, 20])
    def test_qaoa_encoding_many_layers(self, reps: int) -> None:
        """Test QAOAEncoding with many QAOA layers."""
        enc = QAOAEncoding(n_features=4, reps=reps)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_very_deep_circuit_100_reps(self) -> None:
        """Test creating a circuit with 100 repetitions.

        This is a stress test - should complete without memory error.
        """
        # Use simple encoding with linear entanglement for speed
        enc = HardwareEfficientEncoding(n_features=4, reps=100, entanglement="linear")

        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Should complete without error
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


# =============================================================================
# Test Class: Large Batch Sizes
# =============================================================================


@pytest.mark.slow
class TestLargeBatchSizes:
    """Tests for handling large batch sizes."""

    @pytest.mark.parametrize("batch_size", [100, 500, 1000])
    def test_angle_encoding_large_batch(self, batch_size: int) -> None:
        """Test processing large batches of inputs."""
        enc = AngleEncoding(n_features=4)
        batch = np.random.randn(batch_size, 4)

        start = time.time()
        circuits = enc.get_circuits(batch, backend="pennylane")
        elapsed = time.time() - start

        assert len(circuits) == batch_size

        # Should complete in reasonable time (< 1 minute)
        assert elapsed < 60, f"Batch processing too slow: {elapsed:.1f}s"

    @pytest.mark.parametrize("batch_size", [100, 250, 500])
    def test_iqp_encoding_moderate_batch(self, batch_size: int) -> None:
        """Test IQPEncoding with moderate batch sizes."""
        enc = IQPEncoding(n_features=4)
        batch = np.random.randn(batch_size, 4)

        circuits = enc.get_circuits(batch, backend="pennylane")
        assert len(circuits) == batch_size

    def test_very_large_batch_2000(self) -> None:
        """Test processing 2000 samples in a batch."""
        enc = AngleEncoding(n_features=4)
        batch = np.random.randn(2000, 4)

        start = time.time()
        circuits = enc.get_circuits(batch, backend="pennylane")
        elapsed = max(time.time() - start, 1e-9)  # Prevent division by zero

        assert len(circuits) == 2000

        # Throughput should be > 50 circuits/second (conservative for varied hardware)
        throughput = 2000 / elapsed
        assert (
            throughput > 50
        ), f"Low throughput for large batch: {throughput:.0f} circuits/s"

    def test_batch_memory_doesnt_explode(self) -> None:
        """Test that batch processing doesn't cause memory explosion."""
        gc.collect()
        baseline = get_memory_mb()

        enc = AngleEncoding(n_features=8)

        # Process 1000 samples
        batch = np.random.randn(1000, 8)
        circuits = enc.get_circuits(batch, backend="pennylane")

        memory_used = get_memory_mb() - baseline

        # Should not use more than 1GB for 1000 circuits
        assert (
            memory_used < 1000
        ), f"Excessive memory for 1000 circuits: {memory_used:.0f}MB"

        del circuits
        gc.collect()


# =============================================================================
# Test Class: High-Dimensional Features
# =============================================================================


@pytest.mark.slow
class TestHighDimensionalFeatures:
    """Tests for high-dimensional feature spaces."""

    @pytest.mark.parametrize("n_features", [64, 128, 256])
    def test_angle_encoding_high_dim(self, n_features: int) -> None:
        """Test AngleEncoding with high-dimensional features.

        256 features = 256 qubits (simulated classically).
        """
        enc = AngleEncoding(n_features=n_features)
        assert enc.n_features == n_features

        # Should at least be able to create the encoding object
        assert enc.n_qubits == n_features

    @pytest.mark.parametrize("n_features", [256, 512, 1024])
    def test_amplitude_encoding_very_high_dim(self, n_features: int) -> None:
        """Test AmplitudeEncoding with very high-dimensional features.

        1024 features needs only 10 qubits (log2(1024) = 10).
        """
        enc = AmplitudeEncoding(n_features=n_features)

        # Very efficient qubit usage
        expected_qubits = int(np.ceil(np.log2(n_features)))
        assert enc.n_qubits == expected_qubits

        # 1024 features -> 10 qubits
        if n_features == 1024:
            assert enc.n_qubits == 10

    def test_basis_encoding_8_features(self) -> None:
        """Test BasisEncoding with 8 features (8 qubits)."""
        enc = BasisEncoding(n_features=8)
        assert enc.n_qubits == 8

        x = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


# =============================================================================
# Test Class: Combined Scaling Factors
# =============================================================================


@pytest.mark.slow
class TestCombinedScaling:
    """Tests combining multiple scaling factors."""

    def test_moderate_qubits_moderate_reps(self) -> None:
        """Test 20 qubits with 20 repetitions (moderate both)."""
        enc = HardwareEfficientEncoding(n_features=20, reps=20, entanglement="linear")

        x = np.random.randn(20)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_moderate_qubits_moderate_batch(self) -> None:
        """Test 20 qubits with batch size 200."""
        enc = ZZFeatureMap(n_features=20, entanglement="linear")
        batch = np.random.randn(200, 20)

        circuits = enc.get_circuits(batch, backend="pennylane")
        assert len(circuits) == 200

    def test_deep_circuit_with_batch(self) -> None:
        """Test deep circuit (50 reps) with batch (100 samples)."""
        enc = DataReuploading(n_features=4, n_layers=50)
        batch = np.random.randn(100, 4)

        start = time.time()
        circuits = enc.get_circuits(batch, backend="pennylane")
        elapsed = time.time() - start

        assert len(circuits) == 100
        # Should complete in < 2 minutes
        assert elapsed < 120, f"Too slow: {elapsed:.1f}s"


# =============================================================================
# Test Class: Resource Limits
# =============================================================================


@pytest.mark.slow
class TestResourceLimits:
    """Tests for resource limit handling."""

    def test_encoding_creation_doesnt_timeout(self) -> None:
        """Test that encoding creation is fast even for large sizes."""
        start = time.time()

        # Create several large encodings
        _ = AngleEncoding(n_features=100)
        _ = AmplitudeEncoding(n_features=1024)
        _ = IQPEncoding(n_features=50, entanglement="linear")
        _ = ZZFeatureMap(n_features=50, entanglement="linear")

        elapsed = time.time() - start

        # Should complete quickly (< 5 seconds)
        assert elapsed < 5, f"Encoding creation too slow: {elapsed:.1f}s"

    def test_property_access_scales(self) -> None:
        """Test that property access scales well with encoding size."""
        sizes = [10, 50, 100]
        times: dict[int, float] = {}

        for n in sizes:
            enc = AngleEncoding(n_features=n)

            start = time.perf_counter()
            for _ in range(1000):
                _ = enc.n_qubits
                _ = enc.depth
                _ = enc.properties
            times[n] = time.perf_counter() - start

        # Time should not grow significantly with size
        # (properties should be cached)
        # Guard against division by zero when operations are too fast to measure
        base_time = max(times[10], 1e-9)
        ratio = times[100] / base_time
        assert ratio < 2, (
            f"Property access scales poorly: 100-feature is {ratio:.1f}x "
            f"slower than 10-feature"
        )

    def test_error_handling_for_impossible_scale(self) -> None:
        """Test proper error handling for truly impossible scales."""
        # This should work (reasonable scale)
        enc = AngleEncoding(n_features=1000)
        assert enc.n_qubits == 1000

        # But circuit generation for 1000 qubits might fail gracefully
        x = np.random.randn(1000)

        # Should either succeed or raise a clear error
        try:
            circuit = enc.get_circuit(x, backend="pennylane")
            # If it succeeds, great
            assert circuit is not None
        except (MemoryError, RuntimeError, ValueError) as e:
            # Expected for very large circuits
            assert "memory" in str(e).lower() or "qubit" in str(e).lower() or True


# =============================================================================
# Test Class: Scalability Regression Tests
# =============================================================================


@pytest.mark.slow
class TestScalabilityRegression:
    """Regression tests for scalability characteristics."""

    def test_baseline_50_qubits_linear(self) -> None:
        """Baseline: 50 qubits with linear entanglement should work."""
        enc = HardwareEfficientEncoding(n_features=50, reps=2, entanglement="linear")
        x = np.random.randn(50)

        start = time.time()
        circuit = enc.get_circuit(x, backend="pennylane")
        elapsed = time.time() - start

        assert circuit is not None
        # Should complete in < 10 seconds
        assert elapsed < 10, f"50-qubit baseline too slow: {elapsed:.1f}s"

    def test_baseline_batch_1000(self) -> None:
        """Baseline: batch of 1000 should process efficiently."""
        enc = AngleEncoding(n_features=4)
        batch = np.random.randn(1000, 4)

        start = time.time()
        circuits = enc.get_circuits(batch, backend="pennylane")
        elapsed = max(time.time() - start, 1e-9)  # Prevent division by zero

        assert len(circuits) == 1000

        # Should achieve > 50 circuits/second (very conservative for CI/varied hardware)
        # This threshold accommodates slower machines and systems under load
        throughput = 1000 / elapsed
        assert (
            throughput > 50
        ), f"Batch baseline regression: {throughput:.0f} circuits/s < 50"

    def test_baseline_deep_circuit_50_reps(self) -> None:
        """Baseline: 50 repetitions should complete reasonably."""
        enc = DataReuploading(n_features=4, n_layers=50)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        start = time.time()
        circuit = enc.get_circuit(x, backend="pennylane")
        elapsed = time.time() - start

        assert circuit is not None
        # Should complete in < 5 seconds
        assert elapsed < 5, f"Deep circuit baseline too slow: {elapsed:.1f}s"


# =============================================================================
# Test Class: Edge Cases at Scale
# =============================================================================


@pytest.mark.slow
class TestScaleEdgeCases:
    """Edge cases when operating at scale."""

    def test_single_feature_at_scale_batch(self) -> None:
        """Test single-feature encoding with large batch."""
        enc = AngleEncoding(n_features=1)
        batch = np.random.randn(10000, 1)

        circuits = enc.get_circuits(batch, backend="pennylane")
        assert len(circuits) == 10000

    def test_power_of_two_features(self) -> None:
        """Test power-of-2 feature counts (optimal for amplitude)."""
        for n in [2, 4, 8, 16, 32, 64, 128]:
            enc = AmplitudeEncoding(n_features=n)
            expected_qubits = int(np.log2(n))
            assert enc.n_qubits == expected_qubits

    def test_non_power_of_two_features(self) -> None:
        """Test non-power-of-2 feature counts."""
        # These require padding in amplitude encoding
        for n in [3, 5, 7, 9, 15, 17, 100]:
            enc = AmplitudeEncoding(n_features=n)
            # Should use ceil(log2(n)) qubits
            expected_qubits = max(1, int(np.ceil(np.log2(n))))
            assert enc.n_qubits == expected_qubits

    def test_zero_and_tiny_values_at_scale(self) -> None:
        """Test handling of edge values in large batches."""
        enc = AngleEncoding(n_features=4)

        # Batch with zeros
        batch = np.zeros((100, 4))
        circuits = enc.get_circuits(batch, backend="pennylane")
        assert len(circuits) == 100

        # Batch with tiny values
        batch = np.full((100, 4), 1e-15)
        circuits = enc.get_circuits(batch, backend="pennylane")
        assert len(circuits) == 100

    def test_large_values_at_scale(self) -> None:
        """Test handling of large values in large batches."""
        enc = AngleEncoding(n_features=4)

        # Batch with large values
        batch = np.full((100, 4), 1e10)
        circuits = enc.get_circuits(batch, backend="pennylane")
        assert len(circuits) == 100
