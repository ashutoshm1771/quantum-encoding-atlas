"""Comprehensive tests for ZZFeatureMap.

This test module provides complete coverage of the ZZFeatureMap class,
which implements the ZZ feature map encoding with pairwise ZZ entangling gates.
It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, entanglement patterns)
- Entanglement pair behavior (unique to ZZFeatureMap)
- Circuit generation for all backends (PennyLane, Qiskit, Cirq)
- Mathematical correctness verification
- Edge cases and boundary conditions
- Numerical stability tests
- Equality and hashing
- String representation
- Backend error handling
- Serialization (pickle roundtrip)
- Concurrent access / thread safety
- ZZFeatureMap-specific tests (gate counts, caching, logging)
- Slow simulation tests (cross-backend state fidelity)

Run with: pytest tests/unit/encodings/test_zz_feature_map.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_zz_feature_map.py -v -m "not slow"

References
----------
.. [1] Havlicek et al., "Supervised learning with quantum-enhanced feature spaces"
       Nature 567, 209-212 (2019)
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas import ZZFeatureMap
from encoding_atlas.core.properties import EncodingProperties

if TYPE_CHECKING:
    from typing import Any


# =============================================================================
# Backend Availability Checks
# =============================================================================

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import cirq

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data_4d() -> NDArray[np.floating]:
    """4-dimensional sample data for testing.

    Values chosen to exercise typical encoding behavior.
    """
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def sample_data_2d() -> NDArray[np.floating]:
    """2-dimensional sample data for testing."""
    return np.array([0.5, 1.0])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (typical values)
    - [0.9, 1.0, 1.1, 1.2] (typical values)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ])


@pytest.fixture
def default_encoding() -> ZZFeatureMap:
    """Default ZZFeatureMap with 4 features."""
    return ZZFeatureMap(n_features=4)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for ZZFeatureMap instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = ZZFeatureMap(n_features=4)
        assert enc.n_features == 4
        assert enc.n_qubits == 4
        assert enc.reps == 2
        assert enc.entanglement == "full"

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = ZZFeatureMap(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16, 32]:
            enc = ZZFeatureMap(n_features=n)
            assert enc.n_features == n

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = ZZFeatureMap(n_features=4, reps=5)
        assert enc.reps == 5

    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_entanglement_patterns(self, entanglement: str) -> None:
        """Test all valid entanglement patterns."""
        enc = ZZFeatureMap(n_features=4, entanglement=entanglement)  # type: ignore
        assert enc.entanglement == entanglement

    def test_config_stored_correctly(self) -> None:
        """Test that configuration is stored in config dict."""
        enc = ZZFeatureMap(n_features=4, reps=3, entanglement="linear")
        assert enc.config["reps"] == 3
        assert enc.config["entanglement"] == "linear"

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = ZZFeatureMap(n_features=100, entanglement="linear")
        assert enc.n_features == 100
        # Should be fast since no circuit is pre-computed


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    def test_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            ZZFeatureMap(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            ZZFeatureMap(n_features=-1)

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            ZZFeatureMap(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            ZZFeatureMap(n_features=4, reps=-1)

    def test_invalid_reps_float(self) -> None:
        """Test that float reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            ZZFeatureMap(n_features=4, reps=2.5)  # type: ignore

    def test_invalid_reps_boolean(self) -> None:
        """Test that boolean reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            ZZFeatureMap(n_features=4, reps=True)  # type: ignore

    def test_invalid_entanglement(self) -> None:
        """Test that invalid entanglement raises ValueError."""
        with pytest.raises(ValueError, match="entanglement must be one of"):
            ZZFeatureMap(n_features=4, entanglement="invalid")  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of ZZFeatureMap."""

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features."""
        for n in [2, 4, 8, 16]:
            enc = ZZFeatureMap(n_features=n)
            assert enc.n_qubits == n

    def test_depth_calculation(self) -> None:
        """Test depth calculation formula.

        For n=4, full entanglement, reps=2:
        - chromatic_index = n-1 = 3 (n=4 is even)
        - depth_per_rep = 2 (single-qubit) + 3*3 (ZZ layer) = 11
        - total depth = 2 * 11 = 22
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        assert enc.depth == 22  # 2 * (2 + 3*3)

    def test_depth_linear_entanglement(self) -> None:
        """Test depth with linear entanglement.

        For n=4, linear entanglement, reps=1:
        - n_pairs = n-1 = 3
        - zz_depth = 3 * 3 = 9
        - depth_per_rep = 2 + 9 = 11
        """
        enc = ZZFeatureMap(n_features=4, reps=1, entanglement="linear")
        assert enc.depth == 11

    def test_depth_circular_entanglement(self) -> None:
        """Test depth with circular entanglement.

        For n=4, circular entanglement, reps=1:
        - n_pairs = n = 4 (for n > 2)
        - zz_depth = 3 * 4 = 12
        - depth_per_rep = 2 + 12 = 14
        """
        enc = ZZFeatureMap(n_features=4, reps=1, entanglement="circular")
        assert enc.depth == 14

    def test_depth_full_even_qubits(self) -> None:
        """Test depth with full entanglement and even qubit count.

        For n=6 (even), full entanglement, reps=1:
        - chromatic_index = n-1 = 5
        - zz_depth = 3 * 5 = 15
        - depth_per_rep = 2 + 15 = 17
        """
        enc = ZZFeatureMap(n_features=6, reps=1, entanglement="full")
        assert enc.depth == 17

    def test_depth_full_odd_qubits(self) -> None:
        """Test depth with full entanglement and odd qubit count.

        For n=5 (odd), full entanglement, reps=1:
        - chromatic_index = n = 5
        - zz_depth = 3 * 5 = 15
        - depth_per_rep = 2 + 15 = 17
        """
        enc = ZZFeatureMap(n_features=5, reps=1, entanglement="full")
        assert enc.depth == 17

    def test_depth_circular_n2(self) -> None:
        """Test depth with circular entanglement and n=2.

        For n=2, circular = linear (no wrap-around to avoid duplicate):
        - n_pairs = 1
        - zz_depth = 3
        - depth_per_rep = 2 + 3 = 5
        """
        enc = ZZFeatureMap(n_features=2, reps=1, entanglement="circular")
        assert enc.depth == 5

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = ZZFeatureMap(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = ZZFeatureMap(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_is_entangling(self) -> None:
        """Test is_entangling is True."""
        enc = ZZFeatureMap(n_features=4)
        assert enc.properties.is_entangling is True

    def test_not_simulable(self) -> None:
        """Test simulability with entanglement."""
        enc = ZZFeatureMap(n_features=4)
        assert enc.properties.simulability == "not_simulable"

    def test_gate_count_consistency(self) -> None:
        """Test that gate counts are consistent."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        props = enc.properties
        assert props.gate_count == props.single_qubit_gates + props.two_qubit_gates


# =============================================================================
# Test Class: Entanglement Behavior (ZZFeatureMap-Specific)
# =============================================================================


class TestEntanglementBehavior:
    """Tests for entanglement pair computation (unique to ZZFeatureMap)."""

    def test_full_entanglement_4_qubits(self) -> None:
        """Test full entanglement with 4 qubits."""
        enc = ZZFeatureMap(n_features=4, entanglement="full")
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert enc._get_entanglement_pairs() == expected

    def test_linear_entanglement_4_qubits(self) -> None:
        """Test linear entanglement with 4 qubits."""
        enc = ZZFeatureMap(n_features=4, entanglement="linear")
        expected = [(0, 1), (1, 2), (2, 3)]
        assert enc._get_entanglement_pairs() == expected

    def test_circular_entanglement_4_qubits(self) -> None:
        """Test circular entanglement with 4 qubits."""
        enc = ZZFeatureMap(n_features=4, entanglement="circular")
        expected = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert enc._get_entanglement_pairs() == expected

    def test_full_entanglement_count(self) -> None:
        """Test that full entanglement has n*(n-1)/2 pairs."""
        for n in [2, 4, 6, 8]:
            enc = ZZFeatureMap(n_features=n, entanglement="full")
            expected_pairs = n * (n - 1) // 2
            assert len(enc._get_entanglement_pairs()) == expected_pairs


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_1d_input(self, default_encoding: ZZFeatureMap) -> None:
        """Test that valid 1D input passes validation."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        validated = default_encoding._validate_input(x)
        assert validated.shape == (4,)

    def test_wrong_feature_count(self, default_encoding: ZZFeatureMap) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2])  # Only 2 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_nan_values_rejected(self, default_encoding: ZZFeatureMap) -> None:
        """Test that NaN values are rejected."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding._validate_input(x)

    def test_inf_values_rejected(self, default_encoding: ZZFeatureMap) -> None:
        """Test that infinite values are rejected."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_list_input_accepted(self, default_encoding: ZZFeatureMap) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]  # list, not array
        validated = default_encoding._validate_input(x)
        assert isinstance(validated, np.ndarray)


# =============================================================================
# Test Class: PennyLane Backend
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.backend_pennylane
class TestPennyLaneBackend:
    """Tests for PennyLane circuit generation."""

    def test_circuit_is_callable(
        self,
        default_encoding: ZZFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that generated circuit executes correctly in QNode context."""
        enc = ZZFeatureMap(n_features=4, reps=1)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be normalized
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_batch_circuits(
        self,
        default_encoding: ZZFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)


# =============================================================================
# Test Class: Qiskit Backend
# =============================================================================


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.mark.backend_qiskit
class TestQiskitBackend:
    """Tests for Qiskit circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: ZZFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: ZZFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_batch_circuits(
        self,
        default_encoding: ZZFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)


# =============================================================================
# Test Class: Cirq Backend
# =============================================================================


@pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
@pytest.mark.backend_cirq
class TestCirqBackend:
    """Tests for Cirq circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: ZZFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: ZZFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: ZZFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    def test_batch_circuits(
        self,
        default_encoding: ZZFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)

    def test_linear_entanglement(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test Cirq circuit with linear entanglement."""
        enc = ZZFeatureMap(n_features=4, entanglement="linear", reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circular_entanglement(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test Cirq circuit with circular entanglement."""
        enc = ZZFeatureMap(n_features=4, entanglement="circular", reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_rep(self) -> None:
        """Test encoding with single repetition."""
        enc = ZZFeatureMap(n_features=4, reps=1)
        assert enc.reps == 1

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions.

        For n=4, full entanglement, reps=10:
        - depth_per_rep = 2 + 3*3 = 11
        - total depth = 10 * 11 = 110
        """
        enc = ZZFeatureMap(n_features=4, reps=10)
        assert enc.reps == 10
        assert enc.depth == 110  # 10 * (2 + 3*3)

    def test_large_feature_count(self) -> None:
        """Test encoding with large feature count."""
        enc = ZZFeatureMap(n_features=20, entanglement="linear")
        assert enc.n_qubits == 20
        # Linear entanglement should have n-1 pairs
        assert len(enc._get_entanglement_pairs()) == 19

    def test_zero_valued_input(self, default_encoding: ZZFeatureMap) -> None:
        """Test with all-zero input."""
        x = np.zeros(4)
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_large_valued_input(self, default_encoding: ZZFeatureMap) -> None:
        """Test with large input values."""
        x = np.array([100.0, 200.0, 300.0, 400.0])
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_minimum_qubits_for_entanglement(self) -> None:
        """Test minimum qubit count (2) for meaningful entanglement."""
        enc = ZZFeatureMap(n_features=2, reps=1)
        assert enc.n_qubits == 2
        # Should have exactly 1 entangling pair
        assert len(enc._get_entanglement_pairs()) == 1


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values.

    These tests ensure the encoding handles edge cases in numerical
    precision without producing NaN, Inf, or denormalized states.
    Production systems may encounter unexpected input ranges.

    ZZFeatureMap is particularly sensitive to numerical issues because:
    - Phase gates use 2*x_i directly
    - ZZ interactions use 2*(pi-x_i)*(pi-x_j) which can have large magnitudes
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_small_values(self) -> None:
        """Test encoding with very small input values.

        Very small values should produce states close to uniform superposition
        (dominated by Hadamards with minimal phase rotation).
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

        # No NaN or Inf values
        assert not np.any(np.isnan(state)), "State contains NaN"
        assert not np.any(np.isinf(state)), "State contains Inf"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_large_values(self) -> None:
        """Test encoding with very large input values.

        Large values are valid (rotations are periodic).
        Should not cause overflow.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([1e10, 1e11, 1e12, 1e13])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

        # No NaN or Inf values
        assert not np.any(np.isnan(state)), "State contains NaN"
        assert not np.any(np.isinf(state)), "State contains Inf"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_mixed_magnitude_values(self) -> None:
        """Test encoding with mixed magnitude input values.

        Real-world data often has features at different scales.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([1e-10, 1e10, 1e-5, 1e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

        # No NaN or Inf values
        assert not np.any(np.isnan(state)), "State contains NaN"
        assert not np.any(np.isinf(state)), "State contains Inf"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_negative_values(self) -> None:
        """Test encoding with negative input values.

        Negative values affect both single-qubit phases and ZZ interactions.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([-0.5, -1.0, -1.5, -2.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_values_near_pi(self) -> None:
        """Test encoding with values near pi.

        When x_i is approximately pi, the ZZ interaction angle (pi-x_i)
        approaches zero, which can reveal numerical edge cases.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        eps = 1e-14
        x = np.array([
            np.pi - eps,
            np.pi + eps,
            np.pi / 2,
            np.pi * 2,
        ])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_alternating_signs(self) -> None:
        """Test encoding with alternating sign pattern.

        Common in normalized/centered data.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([-1.0, 1.0, -1.0, 1.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_subnormal_values(self) -> None:
        """Test encoding with subnormal (denormalized) floating-point values.

        Subnormal values can cause precision issues in some implementations.
        """
        enc = ZZFeatureMap(n_features=4, reps=1)
        # Subnormal values for double precision (< 2.2e-308)
        x = np.array([1e-310, 1e-315, 1e-320, 1e-308])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should still be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_numerical_stability_qiskit(self) -> None:
        """Test numerical stability specifically with Qiskit backend."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        enc = ZZFeatureMap(n_features=4, reps=2)

        test_inputs = [
            np.array([1e-15, 1e-14, 1e-13, 1e-12]),
            np.array([1e8, 1e9, 1e10, 1e11]),
            np.array([-1e5, 1e5, -1e-5, 1e-5]),
            np.array([np.pi, np.pi/2, np.pi/4, np.pi/8]),
        ]

        simulator = AerSimulator(method="statevector")

        for x in test_inputs:
            circuit = enc.get_circuit(x, backend="qiskit")
            circuit.save_statevector()

            compiled = transpile(circuit, simulator)
            result = simulator.run(compiled).result()
            state = np.array(result.get_statevector().data)

            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)
            assert not np.any(np.isnan(state))
            assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_numerical_stability_cirq(self) -> None:
        """Test numerical stability specifically with Cirq backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)

        test_inputs = [
            np.array([1e-15, 1e-14, 1e-13, 1e-12]),
            np.array([1e8, 1e9, 1e10, 1e11]),
            np.array([-1e5, 1e5, -1e-5, 1e-5]),
            np.array([np.pi, np.pi/2, np.pi/4, np.pi/8]),
        ]

        simulator = cirq.Simulator()

        for x in test_inputs:
            circuit = enc.get_circuit(x, backend="cirq")
            result = simulator.simulate(circuit)
            state = np.array(result.final_state_vector)

            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)
            assert not np.any(np.isnan(state))
            assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_many_reps_numerical_stability(self) -> None:
        """Test numerical stability with high repetition count.

        Deep circuits can accumulate numerical errors.
        """
        enc = ZZFeatureMap(n_features=4, reps=15, entanglement="linear")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-9), f"Norm after 15 reps: {norm}"
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zz_interaction_large_product(self) -> None:
        """Test ZZ interaction with values that create large angle products.

        The ZZ angle is 2*(pi-x_i)*(pi-x_j). When x_i and x_j are far from pi,
        this can produce large angles. Test this specific scenario.
        """
        enc = ZZFeatureMap(n_features=4, reps=1)
        # Values far from pi to create large ZZ angles
        x = np.array([0.0, 0.0, 0.0, 0.0])  # Angle = 2*pi*pi approx 19.7

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        enc2 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        enc2 = ZZFeatureMap(n_features=8, reps=2, entanglement="full")
        assert enc1 != enc2

    def test_equality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        enc2 = ZZFeatureMap(n_features=4, reps=3, entanglement="full")
        assert enc1 != enc2

    def test_equality_different_entanglement(self) -> None:
        """Test that encodings with different entanglement are not equal."""
        enc1 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        enc2 = ZZFeatureMap(n_features=4, reps=2, entanglement="linear")
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        enc2 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        enc2 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")  # Same as enc1
        enc3 = ZZFeatureMap(n_features=4, reps=3, entanglement="full")

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        enc2 = ZZFeatureMap(n_features=4, reps=2, entanglement="full")

        d = {enc1: "value1"}
        d[enc2] = "value2"  # Should update same key since enc1 == enc2

        assert len(d) == 1
        assert d[enc1] == "value2"


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ string representation."""

    def test_repr_contains_class_name(self) -> None:
        """Test that repr contains the class name."""
        enc = ZZFeatureMap(n_features=4)
        repr_str = repr(enc)
        assert "ZZFeatureMap" in repr_str

    def test_repr_contains_n_features(self) -> None:
        """Test that repr contains n_features."""
        enc = ZZFeatureMap(n_features=4)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str

    def test_repr_contains_custom_parameters(self) -> None:
        """Test that repr contains custom parameters."""
        enc = ZZFeatureMap(n_features=8, reps=3, entanglement="linear")
        repr_str = repr(enc)
        assert "n_features=8" in repr_str
        assert "reps=3" in repr_str


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for backend availability and error handling."""

    def test_invalid_backend_raises_error(
        self,
        default_encoding: ZZFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_get_circuit_from_validated_invalid_backend(self) -> None:
        """Test _get_circuit_from_validated with invalid backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        x_validated = enc._validate_input(x)

        with pytest.raises(ValueError, match="Unknown backend"):
            enc._get_circuit_from_validated(x_validated, "invalid")  # type: ignore


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = ZZFeatureMap(n_features=4, reps=3, entanglement="linear")

        # Compute properties to populate cache
        _ = enc.properties
        _ = enc._get_entanglement_pairs()

        # Pickle and unpickle
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.reps == enc.reps
        assert restored.entanglement == enc.entanglement
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Get state before pickling
        circuit_before = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def run_before():
            circuit_before()
            return qml.state()

        state_before = run_before()

        # Pickle and unpickle
        restored = pickle.loads(pickle.dumps(enc))

        # Get state after unpickling
        circuit_after = restored.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def run_after():
            circuit_after()
            return qml.state()

        state_after = run_after()

        # States should match
        np.testing.assert_allclose(state_before, state_after, atol=1e-12)

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        props = restored.properties
        assert isinstance(props, EncodingProperties)
        assert props.n_qubits == enc.n_qubits


# =============================================================================
# Test Class: Concurrent Access
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
                    circuit = enc.get_circuit(x, backend="pennylane")
                    circuits.append(circuit)
            except Exception as e:
                errors.append(e)
            return circuits

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(generate_circuits, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"
        # All circuits should have been generated
        total_circuits = sum(len(r) for r in results)
        assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        num_threads = 20

        results: list[EncodingProperties] = []
        errors: list[Exception] = []

        def access_properties() -> None:
            try:
                props = enc.properties
                results.append(props)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(access_properties) for _ in range(num_threads)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0
        assert len(results) == num_threads
        # All should return the same cached object
        assert all(r is results[0] for r in results)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_concurrent_batch_generation(self) -> None:
        """Test concurrent calls to get_circuits from multiple threads."""
        enc = ZZFeatureMap(n_features=4, reps=2)

        def batch_generate(batch_id: int) -> tuple[int, int]:
            X = np.random.randn(10, 4)
            circuits = enc.get_circuits(X, backend="pennylane")
            return (batch_id, len(circuits))

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(batch_generate, i) for i in range(10)]

            for future in as_completed(futures):
                results.append(future.result())

        # All batches should complete successfully
        assert len(results) == 10
        assert all(count == 10 for _, count in results)


# =============================================================================
# Test Class: ZZFeatureMap Specific - Entanglement Pairs
# =============================================================================


class TestZZFeatureMapEntanglementPairs:
    """Tests for the public get_entanglement_pairs() method."""

    def test_full_entanglement_4_qubits(self) -> None:
        """Test get_entanglement_pairs with full entanglement."""
        enc = ZZFeatureMap(n_features=4, entanglement="full")
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert pairs == expected

    def test_linear_entanglement_4_qubits(self) -> None:
        """Test get_entanglement_pairs with linear entanglement."""
        enc = ZZFeatureMap(n_features=4, entanglement="linear")
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (1, 2), (2, 3)]
        assert pairs == expected

    def test_circular_entanglement_4_qubits(self) -> None:
        """Test get_entanglement_pairs with circular entanglement."""
        enc = ZZFeatureMap(n_features=4, entanglement="circular")
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert pairs == expected

    def test_circular_entanglement_2_qubits(self) -> None:
        """Test circular entanglement with 2 qubits (edge case)."""
        enc = ZZFeatureMap(n_features=2, entanglement="circular")
        pairs = enc.get_entanglement_pairs()

        # For n=2, circular should be same as linear (no wrap-around)
        expected = [(0, 1)]
        assert pairs == expected

    def test_circular_entanglement_3_qubits(self) -> None:
        """Test circular entanglement with 3 qubits."""
        enc = ZZFeatureMap(n_features=3, entanglement="circular")
        pairs = enc.get_entanglement_pairs()

        # For n=3, circular adds wrap-around
        expected = [(0, 1), (1, 2), (2, 0)]
        assert pairs == expected

    def test_full_entanglement_pair_count(self) -> None:
        """Test that full entanglement has n*(n-1)/2 pairs."""
        for n in [2, 3, 5, 8, 10]:
            enc = ZZFeatureMap(n_features=n, entanglement="full")
            pairs = enc.get_entanglement_pairs()
            expected_count = n * (n - 1) // 2
            assert len(pairs) == expected_count

    def test_linear_entanglement_pair_count(self) -> None:
        """Test that linear entanglement has n-1 pairs."""
        for n in [2, 3, 5, 8, 10]:
            enc = ZZFeatureMap(n_features=n, entanglement="linear")
            pairs = enc.get_entanglement_pairs()
            expected_count = n - 1
            assert len(pairs) == expected_count

    def test_circular_entanglement_pair_count(self) -> None:
        """Test circular entanglement pair count (n for n>2, n-1 for n<=2)."""
        # n=2: no wrap-around
        enc2 = ZZFeatureMap(n_features=2, entanglement="circular")
        assert len(enc2.get_entanglement_pairs()) == 1

        # n>2: adds wrap-around
        for n in [3, 4, 5, 8]:
            enc = ZZFeatureMap(n_features=n, entanglement="circular")
            pairs = enc.get_entanglement_pairs()
            expected_count = n  # n pairs for circular with n>2
            assert len(pairs) == expected_count

    def test_returns_list_of_tuples(self) -> None:
        """Test that return type is list of tuples."""
        enc = ZZFeatureMap(n_features=4, entanglement="full")
        pairs = enc.get_entanglement_pairs()

        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) for p in pairs)
        assert all(len(p) == 2 for p in pairs)
        assert all(isinstance(p[0], int) and isinstance(p[1], int) for p in pairs)

    def test_public_method_matches_private(self) -> None:
        """Test that public method returns same result as private method."""
        enc = ZZFeatureMap(n_features=5, entanglement="full")

        public_result = enc.get_entanglement_pairs()
        private_result = enc._get_entanglement_pairs()

        assert public_result == private_result


# =============================================================================
# Test Class: ZZFeatureMap Specific - Entanglement Pairs Caching
# =============================================================================


class TestZZFeatureMapEntanglementPairsCaching:
    """Tests for entanglement pairs computed at initialization."""

    def test_pairs_computed_at_init(self) -> None:
        """Test that entanglement pairs are computed at initialization."""
        enc = ZZFeatureMap(n_features=4, entanglement="full")

        # Pairs should be computed immediately at init, not lazily
        assert enc._entanglement_pairs is not None
        assert len(enc._entanglement_pairs) == 6  # 4*3/2 = 6 pairs

        # Accessor returns the same cached list
        pairs1 = enc._get_entanglement_pairs()
        pairs2 = enc._get_entanglement_pairs()
        assert pairs1 is pairs2  # Same object, not just equal
        assert pairs1 is enc._entanglement_pairs

    def test_cache_content_correct(self) -> None:
        """Test that cached pairs have correct content."""
        enc = ZZFeatureMap(n_features=3, entanglement="full")

        pairs = enc._get_entanglement_pairs()
        expected = [(0, 1), (0, 2), (1, 2)]

        assert pairs == expected
        assert enc._entanglement_pairs == expected

    def test_cache_independent_per_instance(self) -> None:
        """Test that caching is independent per instance."""
        enc1 = ZZFeatureMap(n_features=3, entanglement="full")
        enc2 = ZZFeatureMap(n_features=3, entanglement="linear")

        pairs1 = enc1._get_entanglement_pairs()
        pairs2 = enc2._get_entanglement_pairs()

        assert pairs1 != pairs2
        assert enc1._entanglement_pairs is pairs1
        assert enc2._entanglement_pairs is pairs2


# =============================================================================
# Test Class: ZZFeatureMap Specific - Gate Count Breakdown
# =============================================================================


class TestZZFeatureMapGateCountBreakdown:
    """Tests for the gate_count_breakdown() method."""

    def test_returns_typed_dict(self) -> None:
        """Test that gate_count_breakdown returns correct type."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        breakdown = enc.gate_count_breakdown()

        # Should have all expected keys
        expected_keys = {
            'hadamard', 'phase_single', 'phase_zz', 'cnot',
            'total_single_qubit', 'total_two_qubit', 'total'
        }
        assert set(breakdown.keys()) == expected_keys

        # All values should be integers
        assert all(isinstance(v, int) for v in breakdown.values())

    def test_hadamard_count(self) -> None:
        """Test Hadamard gate count: n * reps."""
        for n, reps in [(4, 2), (6, 3), (3, 1), (8, 4)]:
            enc = ZZFeatureMap(n_features=n, reps=reps)
            breakdown = enc.gate_count_breakdown()

            expected = n * reps
            assert breakdown['hadamard'] == expected

    def test_phase_single_count(self) -> None:
        """Test single-qubit phase gate count: n * reps."""
        for n, reps in [(4, 2), (6, 3), (3, 1), (8, 4)]:
            enc = ZZFeatureMap(n_features=n, reps=reps)
            breakdown = enc.gate_count_breakdown()

            expected = n * reps
            assert breakdown['phase_single'] == expected

    def test_cnot_count_full_entanglement(self) -> None:
        """Test CNOT count with full entanglement: 2 * n_pairs * reps."""
        for n, reps in [(4, 2), (6, 1), (3, 3)]:
            enc = ZZFeatureMap(n_features=n, reps=reps, entanglement="full")
            breakdown = enc.gate_count_breakdown()

            n_pairs = n * (n - 1) // 2
            expected = 2 * n_pairs * reps
            assert breakdown['cnot'] == expected

    def test_cnot_count_linear_entanglement(self) -> None:
        """Test CNOT count with linear entanglement: 2 * (n-1) * reps."""
        for n, reps in [(4, 2), (6, 1), (3, 3)]:
            enc = ZZFeatureMap(n_features=n, reps=reps, entanglement="linear")
            breakdown = enc.gate_count_breakdown()

            n_pairs = n - 1
            expected = 2 * n_pairs * reps
            assert breakdown['cnot'] == expected

    def test_phase_zz_count(self) -> None:
        """Test ZZ phase gate count: n_pairs * reps."""
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        breakdown = enc.gate_count_breakdown()

        n_pairs = 6  # 4 * 3 / 2
        expected = n_pairs * 2  # n_pairs * reps
        assert breakdown['phase_zz'] == expected

    def test_total_single_qubit_gates(self) -> None:
        """Test total single-qubit gates sum."""
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        breakdown = enc.gate_count_breakdown()

        expected = breakdown['hadamard'] + breakdown['phase_single'] + breakdown['phase_zz']
        assert breakdown['total_single_qubit'] == expected

    def test_total_two_qubit_gates(self) -> None:
        """Test total two-qubit gates equals CNOT count."""
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        breakdown = enc.gate_count_breakdown()

        assert breakdown['total_two_qubit'] == breakdown['cnot']

    def test_total_gate_count(self) -> None:
        """Test total gate count is sum of all gates."""
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        breakdown = enc.gate_count_breakdown()

        expected = breakdown['total_single_qubit'] + breakdown['total_two_qubit']
        assert breakdown['total'] == expected

    def test_specific_example_4_qubits_2_reps_full(self) -> None:
        """Test specific known values for n=4, reps=2, full entanglement.

        Calculations:
        - n=4, reps=2, full entanglement: 6 pairs
        - Hadamard: 4 * 2 = 8
        - Phase single: 4 * 2 = 8
        - Phase ZZ: 6 * 2 = 12
        - CNOT: 2 * 6 * 2 = 24
        - Total single: 8 + 8 + 12 = 28
        - Total two: 24
        - Total: 52
        """
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")
        breakdown = enc.gate_count_breakdown()

        assert breakdown['hadamard'] == 8
        assert breakdown['phase_single'] == 8
        assert breakdown['phase_zz'] == 12
        assert breakdown['cnot'] == 24
        assert breakdown['total_single_qubit'] == 28
        assert breakdown['total_two_qubit'] == 24
        assert breakdown['total'] == 52

    def test_specific_example_4_qubits_2_reps_linear(self) -> None:
        """Test specific known values for n=4, reps=2, linear entanglement.

        Calculations:
        - n=4, reps=2, linear entanglement: 3 pairs
        - Hadamard: 4 * 2 = 8
        - Phase single: 4 * 2 = 8
        - Phase ZZ: 3 * 2 = 6
        - CNOT: 2 * 3 * 2 = 12
        - Total single: 8 + 8 + 6 = 22
        - Total two: 12
        - Total: 34
        """
        enc = ZZFeatureMap(n_features=4, reps=2, entanglement="linear")
        breakdown = enc.gate_count_breakdown()

        assert breakdown['hadamard'] == 8
        assert breakdown['phase_single'] == 8
        assert breakdown['phase_zz'] == 6
        assert breakdown['cnot'] == 12
        assert breakdown['total_single_qubit'] == 22
        assert breakdown['total_two_qubit'] == 12
        assert breakdown['total'] == 34

    def test_breakdown_matches_properties(self) -> None:
        """Test that breakdown matches properties gate counts."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        breakdown = enc.gate_count_breakdown()
        props = enc.properties

        assert breakdown['total'] == props.gate_count
        assert breakdown['total_single_qubit'] == props.single_qubit_gates
        assert breakdown['total_two_qubit'] == props.two_qubit_gates

    def test_different_entanglement_topologies(self) -> None:
        """Test breakdown for all entanglement topologies."""
        n = 5
        reps = 2

        enc_full = ZZFeatureMap(n_features=n, reps=reps, entanglement="full")
        enc_linear = ZZFeatureMap(n_features=n, reps=reps, entanglement="linear")
        enc_circular = ZZFeatureMap(n_features=n, reps=reps, entanglement="circular")

        bd_full = enc_full.gate_count_breakdown()
        bd_linear = enc_linear.gate_count_breakdown()
        bd_circular = enc_circular.gate_count_breakdown()

        # Full has most gates (most pairs)
        assert bd_full['total'] > bd_linear['total']
        assert bd_full['total'] > bd_circular['total']

        # Circular has one more pair than linear for n > 2
        assert bd_circular['cnot'] > bd_linear['cnot']

        # Hadamard and phase_single should be same for all topologies
        assert bd_full['hadamard'] == bd_linear['hadamard'] == bd_circular['hadamard']
        assert bd_full['phase_single'] == bd_linear['phase_single'] == bd_circular['phase_single']


# =============================================================================
# Test Class: ZZFeatureMap Specific - Parallel Batch Processing
# =============================================================================


class TestZZFeatureMapParallelBatchProcessing:
    """Tests for parallel circuit generation in get_circuits()."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_parallel_processing_pennylane(self) -> None:
        """Test parallel batch processing with PennyLane backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        X = np.random.randn(20, 4)

        # Generate circuits in parallel
        circuits_parallel = enc.get_circuits(X, backend="pennylane", parallel=True)

        # Generate circuits sequentially
        circuits_sequential = enc.get_circuits(X, backend="pennylane", parallel=False)

        # Both should have same length
        assert len(circuits_parallel) == len(circuits_sequential) == 20
        assert all(callable(c) for c in circuits_parallel)
        assert all(callable(c) for c in circuits_sequential)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_parallel_processing_qiskit(self) -> None:
        """Test parallel batch processing with Qiskit backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        X = np.random.randn(20, 4)

        # Generate circuits in parallel
        circuits_parallel = enc.get_circuits(X, backend="qiskit", parallel=True)

        # Generate circuits sequentially
        circuits_sequential = enc.get_circuits(X, backend="qiskit", parallel=False)

        # Both should have same length
        assert len(circuits_parallel) == len(circuits_sequential) == 20
        assert all(isinstance(c, QuantumCircuit) for c in circuits_parallel)
        assert all(isinstance(c, QuantumCircuit) for c in circuits_sequential)

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_parallel_processing_cirq(self) -> None:
        """Test parallel batch processing with Cirq backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        X = np.random.randn(20, 4)

        # Generate circuits in parallel
        circuits_parallel = enc.get_circuits(X, backend="cirq", parallel=True)

        # Generate circuits sequentially
        circuits_sequential = enc.get_circuits(X, backend="cirq", parallel=False)

        # Both should have same length
        assert len(circuits_parallel) == len(circuits_sequential) == 20
        assert all(isinstance(c, cirq.Circuit) for c in circuits_parallel)
        assert all(isinstance(c, cirq.Circuit) for c in circuits_sequential)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_parallel_preserves_order(self) -> None:
        """Test that parallel processing preserves sample order."""
        enc = ZZFeatureMap(n_features=4, reps=1)

        # Create distinct samples that would produce different circuits
        X = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0],
        ])

        circuits_parallel = enc.get_circuits(X, backend="pennylane", parallel=True)
        circuits_sequential = enc.get_circuits(X, backend="pennylane", parallel=False)

        # Execute circuits and compare states
        dev = qml.device("default.qubit", wires=4)

        for i in range(len(X)):
            @qml.qnode(dev)
            def run_parallel():
                circuits_parallel[i]()
                return qml.state()

            @qml.qnode(dev)
            def run_sequential():
                circuits_sequential[i]()
                return qml.state()

            state_p = run_parallel()
            state_s = run_sequential()

            # States should be identical
            np.testing.assert_allclose(state_p, state_s, atol=1e-12)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_parallel_with_max_workers(self) -> None:
        """Test parallel processing with custom max_workers."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        X = np.random.randn(10, 4)

        # Test with explicit max_workers
        circuits = enc.get_circuits(
            X, backend="pennylane", parallel=True, max_workers=2
        )

        assert len(circuits) == 10
        assert all(callable(c) for c in circuits)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_single_sample_no_parallel(self) -> None:
        """Test that single sample doesn't use parallel processing."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        X = np.array([[0.1, 0.2, 0.3, 0.4]])

        # Even with parallel=True, single sample should work correctly
        circuits = enc.get_circuits(X, backend="pennylane", parallel=True)

        assert len(circuits) == 1
        assert callable(circuits[0])


# =============================================================================
# Test Class: ZZFeatureMap Specific - Module Configuration
# =============================================================================


class TestZZFeatureMapModuleConfiguration:
    """Tests for module-level configuration."""

    def test_default_reps_constant(self) -> None:
        """Test default reps constant matches class default."""
        from encoding_atlas.encodings.zz_feature_map import _DEFAULT_REPS

        enc = ZZFeatureMap(n_features=4)
        assert enc.reps == _DEFAULT_REPS
        assert _DEFAULT_REPS == 2

    def test_default_entanglement_constant(self) -> None:
        """Test default entanglement constant matches class default."""
        from encoding_atlas.encodings.zz_feature_map import _DEFAULT_ENTANGLEMENT

        enc = ZZFeatureMap(n_features=4)
        assert enc.entanglement == _DEFAULT_ENTANGLEMENT
        assert _DEFAULT_ENTANGLEMENT == "full"

    def test_valid_entanglements_constant(self) -> None:
        """Test valid entanglements constant contains all valid options."""
        from encoding_atlas.encodings.zz_feature_map import _VALID_ENTANGLEMENTS

        assert _VALID_ENTANGLEMENTS == frozenset({"full", "linear", "circular"})

        # All valid options should work
        for ent in _VALID_ENTANGLEMENTS:
            enc = ZZFeatureMap(n_features=4, entanglement=ent)  # type: ignore
            assert enc.entanglement == ent

    def test_all_export(self) -> None:
        """Test __all__ export list."""
        from encoding_atlas.encodings import zz_feature_map

        assert hasattr(zz_feature_map, "__all__")
        assert "ZZFeatureMap" in zz_feature_map.__all__

    def test_logger_exists(self) -> None:
        """Test that module logger is configured."""
        from encoding_atlas.encodings.zz_feature_map import _logger

        assert _logger is not None
        assert _logger.name == "encoding_atlas.encodings.zz_feature_map"


# =============================================================================
# Test Class: ZZFeatureMap Specific - Internal Optimizations
# =============================================================================


class TestZZFeatureMapInternalOptimizations:
    """Tests for internal optimization methods."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_get_circuit_from_validated_pennylane(self) -> None:
        """Test _get_circuit_from_validated with PennyLane backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Validate input first
        x_validated = enc._validate_input(x)

        # Use internal method
        circuit = enc._get_circuit_from_validated(x_validated, "pennylane")

        assert callable(circuit)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_get_circuit_from_validated_qiskit(self) -> None:
        """Test _get_circuit_from_validated with Qiskit backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Validate input first
        x_validated = enc._validate_input(x)

        # Use internal method
        circuit = enc._get_circuit_from_validated(x_validated, "qiskit")

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_get_circuit_from_validated_cirq(self) -> None:
        """Test _get_circuit_from_validated with Cirq backend."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Validate input first
        x_validated = enc._validate_input(x)

        # Use internal method
        circuit = enc._get_circuit_from_validated(x_validated, "cirq")

        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 4

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_internal_method_matches_public(self) -> None:
        """Test that internal method produces same result as public method."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.5, 1.0, 1.5, 2.0])

        # Get circuits both ways
        x_validated = enc._validate_input(x)
        circuit_internal = enc._get_circuit_from_validated(x_validated, "pennylane")
        circuit_public = enc.get_circuit(x, backend="pennylane")

        # Execute both and compare states
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def run_internal():
            circuit_internal()
            return qml.state()

        @qml.qnode(dev)
        def run_public():
            circuit_public()
            return qml.state()

        state_internal = run_internal()
        state_public = run_public()

        np.testing.assert_allclose(state_internal, state_public, atol=1e-12)


# =============================================================================
# Test Class: ZZFeatureMap Specific - Full Entanglement Warning
# =============================================================================


class TestZZFeatureMapFullEntanglementWarning:
    """Tests for the UserWarning with large n_features and full entanglement.

    ZZFeatureMap emits a UserWarning when full entanglement is used with
    n_features > 10 to alert users about O(n²) gate scaling. This matches
    the pattern used by IQPEncoding and other encodings.
    """

    def test_threshold_constant_exists(self) -> None:
        """Test that the threshold constant exists."""
        from encoding_atlas.encodings.zz_feature_map import _FULL_ENTANGLEMENT_WARNING_THRESHOLD

        assert isinstance(_FULL_ENTANGLEMENT_WARNING_THRESHOLD, int)
        assert _FULL_ENTANGLEMENT_WARNING_THRESHOLD == 10

    def test_no_warning_at_threshold(self) -> None:
        """Test that no warning is emitted at threshold (n=10)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ZZFeatureMap(n_features=10, entanglement="full")

        # Filter for UserWarnings about full entanglement
        entanglement_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "Full entanglement" in str(x.message)
        ]
        assert len(entanglement_warnings) == 0

    def test_warning_above_threshold(self) -> None:
        """Test that UserWarning is emitted above threshold."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ZZFeatureMap(n_features=12, entanglement="full")

        # Filter for UserWarnings about full entanglement
        entanglement_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "Full entanglement" in str(x.message)
        ]
        assert len(entanglement_warnings) == 1

        # Verify message content
        msg = str(entanglement_warnings[0].message)
        assert "12" in msg  # n_features
        assert "66" in msg  # n_pairs = 12*11/2 = 66
        assert "linear" in msg.lower()  # Suggestion for linear

    def test_no_warning_for_linear_entanglement(self) -> None:
        """Test that no warning is emitted for linear entanglement regardless of n."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ZZFeatureMap(n_features=20, entanglement="linear")

        entanglement_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "Full entanglement" in str(x.message)
        ]
        assert len(entanglement_warnings) == 0

    def test_no_warning_for_circular_entanglement(self) -> None:
        """Test that no warning is emitted for circular entanglement regardless of n."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ZZFeatureMap(n_features=20, entanglement="circular")

        entanglement_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "Full entanglement" in str(x.message)
        ]
        assert len(entanglement_warnings) == 0

    def test_warning_contains_cnot_count(self) -> None:
        """Test that the warning message contains CNOT count."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ZZFeatureMap(n_features=15, reps=2, entanglement="full")

        entanglement_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "Full entanglement" in str(x.message)
        ]
        assert len(entanglement_warnings) == 1

        msg = str(entanglement_warnings[0].message)
        # n_pairs = 15*14/2 = 105
        # CNOTs = 2 * 105 * 2 = 420
        assert "420" in msg
        assert "CNOT" in msg

    def test_warning_stacklevel_points_to_caller(self) -> None:
        """Test that warning stacklevel points to the caller's code, not __init__."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ZZFeatureMap(n_features=15, entanglement="full")

        entanglement_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "Full entanglement" in str(x.message)
        ]
        assert len(entanglement_warnings) == 1

        # The warning should point to this test file, not zz_feature_map.py
        warning_filename = entanglement_warnings[0].filename
        assert "test_zz_feature_map" in warning_filename

    def test_logger_warning_also_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that _logger.warning is also called alongside warnings.warn."""
        import logging
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with caplog.at_level(logging.WARNING, logger="encoding_atlas.encodings.zz_feature_map"):
                _ = ZZFeatureMap(n_features=12, entanglement="full")

        # Check that WARNING was logged (in addition to UserWarning)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1
        assert "Large feature count" in warning_records[0].message
        assert "12" in warning_records[0].message


# =============================================================================
# Test Class: ZZFeatureMap Specific - GateCountBreakdown TypedDict
# =============================================================================


class TestZZFeatureMapGateCountBreakdownTypedDict:
    """Tests for the GateCountBreakdown TypedDict structure."""

    def test_import_typeddict(self) -> None:
        """Test that GateCountBreakdown can be imported."""
        from encoding_atlas.encodings.zz_feature_map import GateCountBreakdown

        # Verify it's a TypedDict by checking __annotations__
        assert hasattr(GateCountBreakdown, '__annotations__')
        expected_keys = {
            'hadamard', 'phase_single', 'phase_zz', 'cnot',
            'total_single_qubit', 'total_two_qubit', 'total'
        }
        assert set(GateCountBreakdown.__annotations__.keys()) == expected_keys

    def test_all_annotations_are_int(self) -> None:
        """Test that all TypedDict fields are annotated as int."""
        from typing import get_type_hints
        from encoding_atlas.encodings.zz_feature_map import GateCountBreakdown

        # Use get_type_hints to resolve forward references
        hints = get_type_hints(GateCountBreakdown)
        for key, value_type in hints.items():
            assert value_type is int, f"Field {key} should be int, got {value_type}"


# =============================================================================
# Test Class: Slow Simulation Tests
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    They require all backends (PennyLane, Qiskit, Cirq) to be installed.
    """

    @staticmethod
    def _get_pennylane_state(
        enc: ZZFeatureMap,
        x: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Execute PennyLane circuit and return state vector."""
        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        return np.array(full_circuit())

    @staticmethod
    def _get_qiskit_state(
        enc: ZZFeatureMap,
        x: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Execute Qiskit circuit and return state vector."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        circuit = enc.get_circuit(x, backend="qiskit")
        circuit.save_statevector()

        simulator = AerSimulator(method="statevector")
        compiled = transpile(circuit, simulator)
        result = simulator.run(compiled).result()

        return np.array(result.get_statevector().data)

    @staticmethod
    def _get_cirq_state(
        enc: ZZFeatureMap,
        x: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Execute Cirq circuit and return state vector."""
        circuit = enc.get_circuit(x, backend="cirq")
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        return np.array(result.final_state_vector)

    def _assert_states_equivalent(
        self,
        state1: NDArray[np.complexfloating],
        state2: NDArray[np.complexfloating],
        name1: str = "state1",
        name2: str = "state2",
        atol: float = 1e-6,
    ) -> None:
        """Assert two quantum states are equivalent up to qubit ordering.

        Compares sorted probability distributions to account for different
        qubit ordering conventions between backends.
        """
        # Both states must have same dimension
        assert len(state1) == len(state2), (
            f"{name1} and {name2} have different dimensions: "
            f"{len(state1)} vs {len(state2)}"
        )

        # Both states must be normalized
        norm1 = np.sum(np.abs(state1) ** 2)
        norm2 = np.sum(np.abs(state2) ** 2)
        assert np.isclose(norm1, 1.0, atol=1e-10), (
            f"{name1} is not normalized: |norm|^2 = {norm1}"
        )
        assert np.isclose(norm2, 1.0, atol=1e-10), (
            f"{name2} is not normalized: |norm|^2 = {norm2}"
        )

        # Compare sorted probability distributions
        probs1 = sorted(np.abs(state1) ** 2)
        probs2 = sorted(np.abs(state2) ** 2)

        np.testing.assert_allclose(
            probs1,
            probs2,
            atol=atol,
            err_msg=(
                f"Probability distributions differ between {name1} and {name2}"
            ),
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane
        pl_state = self._get_pennylane_state(enc, x)
        assert np.isclose(np.sum(np.abs(pl_state) ** 2), 1.0, atol=1e-10)

        # Qiskit
        qk_state = self._get_qiskit_state(enc, x)
        assert np.isclose(np.sum(np.abs(qk_state) ** 2), 1.0, atol=1e-10)

        # Cirq
        cirq_state = self._get_cirq_state(enc, x)
        assert np.isclose(np.sum(np.abs(cirq_state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_equivalent_states(self) -> None:
        """Test that all backends produce mathematically equivalent states.

        This is the core cross-backend fidelity test. It verifies that
        PennyLane, Qiskit, and Cirq all encode the same input data into
        quantum states with identical probability distributions.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # All states should have correct dimension: 2^n_qubits = 16
        expected_dim = 2 ** enc.n_qubits
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        # Cross-compare all pairs
        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
        self._assert_states_equivalent(qk_state, cirq_state, "Qiskit", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("reps", [1, 2, 3])
    def test_cross_backend_with_different_reps(self, reps: int) -> None:
        """Test cross-backend equivalence with different repetition counts.

        ZZFeatureMap uses repeated application of the encoding circuit.
        This tests consistency across different layer depths.
        """
        enc = ZZFeatureMap(n_features=4, reps=reps)
        x = np.array([0.5, 1.0, 1.5, 2.0])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(
            pl_state, qk_state,
            f"PennyLane (reps={reps})",
            f"Qiskit (reps={reps})"
        )
        self._assert_states_equivalent(
            pl_state, cirq_state,
            f"PennyLane (reps={reps})",
            f"Cirq (reps={reps})"
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_cross_backend_with_different_entanglement(
        self, entanglement: str
    ) -> None:
        """Test cross-backend equivalence with different entanglement patterns.

        ZZFeatureMap supports full, linear, and circular entanglement.
        Each pattern should produce consistent states across backends.
        """
        enc = ZZFeatureMap(
            n_features=4,
            reps=2,
            entanglement=entanglement,  # type: ignore
        )
        x = np.array([0.2, 0.4, 0.6, 0.8])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(
            pl_state, qk_state,
            f"PennyLane (entanglement={entanglement})",
            f"Qiskit (entanglement={entanglement})"
        )
        self._assert_states_equivalent(
            pl_state, cirq_state,
            f"PennyLane (entanglement={entanglement})",
            f"Cirq (entanglement={entanglement})"
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_with_2_qubits(self) -> None:
        """Test cross-backend equivalence with minimal qubit count.

        2 qubits is the minimum for meaningful entanglement testing.
        With full entanglement, there's only 1 ZZ pair.
        """
        enc = ZZFeatureMap(n_features=2, reps=2)
        x = np.array([0.7, 1.4])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # Dimension: 2^2 = 4
        expected_dim = 4
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_with_6_qubits(self) -> None:
        """Test cross-backend equivalence with larger qubit count.

        Tests scaling behavior and consistency for moderately large systems.
        With full entanglement, 6 qubits have 15 ZZ pairs.
        """
        # Use linear entanglement for faster execution
        enc = ZZFeatureMap(n_features=6, reps=1, entanglement="linear")
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # Dimension: 2^6 = 64
        expected_dim = 64
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        inputs = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]

        states = []
        for x in inputs:
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit():
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # At least some states should be different
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                fidelity = np.abs(np.vdot(states[i], states[j])) ** 2
                # Different inputs should generally produce different states
                # (fidelity < 1 means states are different)
                assert fidelity < 0.9999 or np.allclose(inputs[i], inputs[j])

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input always produces same state."""
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        states = []
        for _ in range(5):
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit():
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # All states should be identical
        for i in range(1, len(states)):
            assert np.allclose(states[0], states[i], atol=1e-10)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_reproducibility(self) -> None:
        """Test that repeated executions produce identical results.

        Each backend should produce exactly the same state when run
        multiple times with the same input.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([0.3, 0.5, 0.7, 0.9])

        # Run each backend twice
        pl_state1 = self._get_pennylane_state(enc, x)
        pl_state2 = self._get_pennylane_state(enc, x)
        np.testing.assert_allclose(
            pl_state1, pl_state2, atol=1e-14,
            err_msg="PennyLane states differ between runs",
        )

        qk_state1 = self._get_qiskit_state(enc, x)
        qk_state2 = self._get_qiskit_state(enc, x)
        np.testing.assert_allclose(
            qk_state1, qk_state2, atol=1e-14,
            err_msg="Qiskit states differ between runs",
        )

        cirq_state1 = self._get_cirq_state(enc, x)
        cirq_state2 = self._get_cirq_state(enc, x)
        np.testing.assert_allclose(
            cirq_state1, cirq_state2, atol=1e-14,
            err_msg="Cirq states differ between runs",
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_different_inputs_different_states(self) -> None:
        """Test that different inputs produce different states across backends.

        This ensures the encoding is actually data-dependent and not
        trivially constant.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x1 = np.array([0.1, 0.2, 0.3, 0.4])
        x2 = np.array([0.5, 0.6, 0.7, 0.8])

        # PennyLane states should differ
        pl_state1 = self._get_pennylane_state(enc, x1)
        pl_state2 = self._get_pennylane_state(enc, x2)
        probs1 = sorted(np.abs(pl_state1) ** 2)
        probs2 = sorted(np.abs(pl_state2) ** 2)
        assert not np.allclose(probs1, probs2, atol=1e-3), (
            "PennyLane: Different inputs produced identical states"
        )

        # Qiskit states should differ
        qk_state1 = self._get_qiskit_state(enc, x1)
        qk_state2 = self._get_qiskit_state(enc, x2)
        probs1 = sorted(np.abs(qk_state1) ** 2)
        probs2 = sorted(np.abs(qk_state2) ** 2)
        assert not np.allclose(probs1, probs2, atol=1e-3), (
            "Qiskit: Different inputs produced identical states"
        )

        # Cirq states should differ
        cirq_state1 = self._get_cirq_state(enc, x1)
        cirq_state2 = self._get_cirq_state(enc, x2)
        probs1 = sorted(np.abs(cirq_state1) ** 2)
        probs2 = sorted(np.abs(cirq_state2) ** 2)
        assert not np.allclose(probs1, probs2, atol=1e-3), (
            "Cirq: Different inputs produced identical states"
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_full_parameter_combinations(self) -> None:
        """Test cross-backend equivalence with comprehensive parameter combinations.

        Exhaustive test covering multiple reps and entanglement patterns
        to ensure robust cross-backend behavior.
        """
        test_configs = [
            {"reps": 1, "entanglement": "full"},
            {"reps": 2, "entanglement": "linear"},
            {"reps": 1, "entanglement": "circular"},
            {"reps": 3, "entanglement": "linear"},
        ]

        for config in test_configs:
            enc = ZZFeatureMap(
                n_features=4,
                reps=config["reps"],
                entanglement=config["entanglement"],  # type: ignore
            )
            x = np.array([0.25, 0.5, 0.75, 1.0])

            pl_state = self._get_pennylane_state(enc, x)
            qk_state = self._get_qiskit_state(enc, x)
            cirq_state = self._get_cirq_state(enc, x)

            config_str = f"reps={config['reps']}, ent={config['entanglement']}"
            self._assert_states_equivalent(
                pl_state, qk_state,
                f"PennyLane ({config_str})",
                f"Qiskit ({config_str})"
            )
            self._assert_states_equivalent(
                pl_state, cirq_state,
                f"PennyLane ({config_str})",
                f"Cirq ({config_str})"
            )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_zero_input(self) -> None:
        """Test cross-backend equivalence with zero input values.

        Edge case: all-zero input tests the Hadamard initialization
        and specific ZZ angle computation with x_i = 0.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.zeros(4)

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_pi_inputs(self) -> None:
        """Test cross-backend equivalence with pi values as input.

        When x_i = pi, the ZZ interaction angle becomes 2*(pi-pi)*(pi-x_j) = 0,
        which is an interesting special case.
        """
        enc = ZZFeatureMap(n_features=4, reps=1)
        x = np.array([np.pi, np.pi / 2, np.pi / 4, np.pi / 3])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_negative_inputs(self) -> None:
        """Test cross-backend equivalence with negative input values.

        Negative values should be handled correctly by all backends.
        """
        enc = ZZFeatureMap(n_features=4, reps=2)
        x = np.array([-0.5, -1.0, 0.5, 1.0])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
