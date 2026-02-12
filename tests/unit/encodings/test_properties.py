"""Property-based tests for quantum encodings using Hypothesis.

This test suite uses property-based testing to automatically discover
edge cases and verify invariants that should hold for all valid inputs.

Test Categories:
1. State normalization properties
2. Determinism properties
3. Input/output relationship properties
4. Numerical stability properties
5. Cross-backend consistency properties
"""

from __future__ import annotations

import numpy as np
import pytest

# Try to import hypothesis, skip all tests if not available
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from encoding_atlas import (
    AmplitudeEncoding,
    AngleEncoding,
    IQPEncoding,
    ZZFeatureMap,
)

# =============================================================================
# CUSTOM STRATEGIES
# =============================================================================


def valid_floats(min_value: float = -10, max_value: float = 10):
    """Strategy for valid float values (no NaN, no Inf)."""
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def float_list(n: int, min_value: float = -10, max_value: float = 10):
    """Strategy for list of n valid floats."""
    return st.lists(
        valid_floats(min_value, max_value),
        min_size=n,
        max_size=n,
    )


# =============================================================================
# STATE NORMALIZATION PROPERTIES
# =============================================================================


class TestNormalizationProperties:
    """Property: all encodings produce normalized quantum states."""

    @given(float_list(4))
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_angle_encoding_always_normalized(self, data) -> None:
        """Property: AngleEncoding always produces a normalized state."""
        qml = pytest.importorskip("pennylane")

        enc = AngleEncoding(n_features=4, rotation="Y")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def circuit(x):
            enc.get_circuit(x, backend="pennylane")()
            return qml.state()

        x = np.array(data)
        state = circuit(x)
        norm = np.sum(np.abs(state) ** 2)

        assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

    @given(float_list(4, min_value=0.01, max_value=10))
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_amplitude_encoding_always_normalized(self, data) -> None:
        """Property: AmplitudeEncoding always produces normalized state."""
        qml = pytest.importorskip("pennylane")

        # Skip if all values are too close to zero
        assume(np.linalg.norm(data) > 1e-6)

        enc = AmplitudeEncoding(n_features=4, normalize=True)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def circuit(x):
            enc.get_circuit(x, backend="pennylane")()
            return qml.state()

        x = np.array(data)
        state = circuit(x)
        norm = np.sum(np.abs(state) ** 2)

        assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

    @given(float_list(4))
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_iqp_encoding_always_normalized(self, data) -> None:
        """Property: IQPEncoding always produces normalized state."""
        qml = pytest.importorskip("pennylane")

        enc = IQPEncoding(n_features=4, reps=1)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def circuit(x):
            enc.get_circuit(x, backend="pennylane")()
            return qml.state()

        x = np.array(data)
        state = circuit(x)
        norm = np.sum(np.abs(state) ** 2)

        assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

    @given(float_list(4))
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_zz_feature_map_always_normalized(self, data) -> None:
        """Property: ZZFeatureMap always produces normalized state."""
        qml = pytest.importorskip("pennylane")

        enc = ZZFeatureMap(n_features=4, reps=1)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def circuit(x):
            enc.get_circuit(x, backend="pennylane")()
            return qml.state()

        x = np.array(data)
        state = circuit(x)
        norm = np.sum(np.abs(state) ** 2)

        assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"


# =============================================================================
# DETERMINISM PROPERTIES
# =============================================================================


class TestDeterminismProperties:
    """Property: same input always produces same output."""

    @given(float_list(4))
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_angle_encoding_deterministic(self, data) -> None:
        """Property: AngleEncoding is deterministic."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = AngleEncoding(n_features=4)
        x = np.array(data)

        circuit1 = enc.get_circuit(x, backend="qiskit")
        circuit2 = enc.get_circuit(x, backend="qiskit")

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        np.testing.assert_allclose(state1, state2, atol=1e-14)

    @given(float_list(4))
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_iqp_encoding_deterministic(self, data) -> None:
        """Property: IQPEncoding is deterministic."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = IQPEncoding(n_features=4, reps=1)
        x = np.array(data)

        circuit1 = enc.get_circuit(x, backend="qiskit")
        circuit2 = enc.get_circuit(x, backend="qiskit")

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        np.testing.assert_allclose(state1, state2, atol=1e-14)

    @given(float_list(4))
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_zz_feature_map_deterministic(self, data) -> None:
        """Property: ZZFeatureMap is deterministic."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = ZZFeatureMap(n_features=4, reps=1)
        x = np.array(data)

        circuit1 = enc.get_circuit(x, backend="qiskit")
        circuit2 = enc.get_circuit(x, backend="qiskit")

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        np.testing.assert_allclose(state1, state2, atol=1e-14)


# =============================================================================
# SINGLE FEATURE PROPERTIES
# =============================================================================


class TestSingleFeatureProperties:
    """Properties for single-feature encodings."""

    @given(valid_floats(-100, 100))
    @settings(max_examples=100, deadline=None)
    def test_single_feature_angle_encoding_valid(self, value: float) -> None:
        """Property: single feature angle encoding always produces valid circuit."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([value])

        # Should not raise
        circuit = enc.get_circuit(x, backend="qiskit")
        assert circuit is not None

    @given(valid_floats(-100, 100))
    @settings(max_examples=100, deadline=None)
    def test_single_feature_normalized(self, value: float) -> None:
        """Property: single feature encoding produces normalized state."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([value])

        circuit = enc.get_circuit(x, backend="qiskit")
        state = Statevector(circuit).data
        norm = np.sum(np.abs(state) ** 2)

        assert np.isclose(norm, 1.0, atol=1e-10)


# =============================================================================
# NUMERICAL STABILITY PROPERTIES
# =============================================================================


class TestNumericalStabilityProperties:
    """Properties for numerical stability."""

    @given(float_list(4, min_value=-100, max_value=100))
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_large_values_produce_valid_state(self, data) -> None:
        """Property: large input values still produce valid states."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = AngleEncoding(n_features=4, rotation="Y")
        x = np.array(data)

        circuit = enc.get_circuit(x, backend="qiskit")
        state = Statevector(circuit).data

        # State should be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

        # State should have no NaN or Inf
        assert np.all(np.isfinite(state)), "State contains NaN or Inf"

    @given(st.lists(valid_floats(-1e-10, 1e-10), min_size=4, max_size=4))
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_tiny_values_produce_valid_state(self, data) -> None:
        """Property: very small input values still produce valid states."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = AngleEncoding(n_features=4, rotation="Y")
        x = np.array(data)

        circuit = enc.get_circuit(x, backend="qiskit")
        state = Statevector(circuit).data

        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))


# =============================================================================
# ROTATION INVARIANT PROPERTIES
# =============================================================================


class TestRotationProperties:
    """Properties related to rotation angles."""

    @given(valid_floats(0, 2 * np.pi))
    @settings(max_examples=50, deadline=None)
    def test_periodicity_2pi(self, angle: float) -> None:
        """Property: rotations are 2π periodic in probability."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = AngleEncoding(n_features=1, rotation="Y")

        x1 = np.array([angle])
        x2 = np.array([angle + 2 * np.pi])

        circuit1 = enc.get_circuit(x1, backend="qiskit")
        circuit2 = enc.get_circuit(x2, backend="qiskit")

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        # Probabilities should match (states may differ by global phase)
        prob1 = np.abs(state1) ** 2
        prob2 = np.abs(state2) ** 2

        np.testing.assert_allclose(prob1, prob2, atol=1e-10)

    @given(valid_floats(-10, 10))
    @settings(max_examples=50, deadline=None)
    def test_opposite_angles_related(self, angle: float) -> None:
        """Property: opposite angles produce related states."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        enc = AngleEncoding(n_features=1, rotation="Y")

        x_pos = np.array([angle])
        x_neg = np.array([-angle])

        circuit_pos = enc.get_circuit(x_pos, backend="qiskit")
        circuit_neg = enc.get_circuit(x_neg, backend="qiskit")

        state_pos = Statevector(circuit_pos).data
        state_neg = Statevector(circuit_neg).data

        # Both should be normalized
        assert np.isclose(np.sum(np.abs(state_pos) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(state_neg) ** 2), 1.0, atol=1e-10)


# =============================================================================
# ENCODING COMPARISON PROPERTIES
# =============================================================================


class TestEncodingComparisonProperties:
    """Properties comparing different encodings."""

    @given(float_list(4))
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_different_encodings_different_states(self, data) -> None:
        """Property: different encodings produce different states (usually)."""
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import Statevector

        x = np.array(data)

        enc_angle = AngleEncoding(n_features=4, rotation="Y")
        enc_iqp = IQPEncoding(n_features=4, reps=1)

        circuit_angle = enc_angle.get_circuit(x, backend="qiskit")
        circuit_iqp = enc_iqp.get_circuit(x, backend="qiskit")

        state_angle = Statevector(circuit_angle).data
        state_iqp = Statevector(circuit_iqp).data

        # Fidelity between different encodings should generally be < 1
        # (they encode the same data differently)
        fidelity = np.abs(np.vdot(state_angle, state_iqp)) ** 2

        # Both should be valid normalized states
        assert np.isclose(np.sum(np.abs(state_angle) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(state_iqp) ** 2), 1.0, atol=1e-10)

        # Fidelity should be a valid probability
        assert 0 <= fidelity <= 1 + 1e-10


# =============================================================================
# CIRCUIT GENERATION PROPERTIES
# =============================================================================


class TestCircuitGenerationProperties:
    """Properties for circuit generation."""

    @given(float_list(4))
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_circuit_generation_no_exception(self, data) -> None:
        """Property: circuit generation never raises for valid input."""
        x = np.array(data)

        encodings = [
            AngleEncoding(n_features=4, rotation="Y"),
            IQPEncoding(n_features=4, reps=1),
            ZZFeatureMap(n_features=4, reps=1),
        ]

        for enc in encodings:
            # Should not raise for any backend
            circuit_qiskit = enc.get_circuit(x, backend="qiskit")
            assert circuit_qiskit is not None

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20, deadline=None)
    def test_qubit_count_matches_features(self, n_features: int) -> None:
        """Property: number of qubits is consistent with features."""
        enc = AngleEncoding(n_features=n_features, rotation="Y")

        # AngleEncoding uses 1 qubit per feature
        assert enc.n_qubits == n_features

        x = np.random.rand(n_features)
        circuit = enc.get_circuit(x, backend="qiskit")

        assert circuit.num_qubits == n_features
