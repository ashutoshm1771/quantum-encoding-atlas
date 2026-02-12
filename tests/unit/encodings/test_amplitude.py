"""Comprehensive tests for AmplitudeEncoding.

This test module provides complete coverage of the AmplitudeEncoding class,
which maps classical data to quantum state amplitudes for exponential data
compression. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, exponential compression)
- Normalization and zero-norm vector handling (unique to AmplitudeEncoding)
- Circuit generation for all backends (PennyLane, Qiskit, Cirq)
- Mathematical correctness verification
- Edge cases and boundary conditions
- Numerical stability tests
- Equality and hashing
- String representation
- Backend error handling
- Serialization (pickle roundtrip)
- Concurrent access / thread safety
- Slow simulation tests (cross-backend state fidelity)

Run with: pytest tests/unit/encodings/test_amplitude.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_amplitude.py -v -m "not slow"

References
----------
.. [1] Schuld, M., & Petruccione, F. (2018). Supervised Learning with Quantum
       Computers. Springer.
"""

from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas import AmplitudeEncoding
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

    Values chosen to exercise typical amplitude encoding behavior with
    equal magnitudes for uniform superposition.
    """
    return np.array([0.5, 0.5, 0.5, 0.5])


@pytest.fixture
def sample_data_8d() -> NDArray[np.floating]:
    """8-dimensional sample data for testing.

    Contains varied magnitudes for testing normalization behavior.
    """
    return np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.3, 0.2])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.5, 0.5, 0.5, 0.5] (uniform values)
    - [0.1, 0.2, 0.3, 0.4] (varied values)
    - [0.8, 0.6, 0.4, 0.2] (decreasing values)
    """
    return np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.1, 0.2, 0.3, 0.4],
            [0.8, 0.6, 0.4, 0.2],
        ]
    )


@pytest.fixture
def default_encoding() -> AmplitudeEncoding:
    """Default AmplitudeEncoding with 4 features."""
    return AmplitudeEncoding(n_features=4)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for AmplitudeEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = AmplitudeEncoding(n_features=4)
        assert enc.n_features == 4
        # n_qubits = ceil(log2(4)) = 2
        assert enc.n_qubits == 2
        assert enc.normalize is True

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case).

        Special case: n_features=1 yields log2(1)=0, but we need at least 1 qubit.
        """
        enc = AmplitudeEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        test_cases = [
            (1, 1),  # log2(1) = 0, but min is 1 qubit
            (2, 1),  # log2(2) = 1
            (3, 2),  # log2(3) = 1.58... -> 2
            (4, 2),  # log2(4) = 2
            (5, 3),  # log2(5) = 2.32... -> 3
            (8, 3),  # log2(8) = 3
            (16, 4),  # log2(16) = 4
        ]
        for n_features, expected_qubits in test_cases:
            enc = AmplitudeEncoding(n_features=n_features)
            assert enc.n_qubits == expected_qubits, (
                f"n_features={n_features}: expected {expected_qubits} qubits, "
                f"got {enc.n_qubits}"
            )

    def test_custom_normalize(self) -> None:
        """Test instantiation with custom normalization setting."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        assert enc.normalize is False

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = AmplitudeEncoding(n_features=100)
        assert enc.n_features == 100
        # Should be fast since no circuit is pre-computed

    def test_config_stored_correctly(self) -> None:
        """Test that configuration is stored in config dict."""
        enc = AmplitudeEncoding(n_features=8, normalize=False)
        assert enc.config["normalize"] is False


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    def test_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            AmplitudeEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            AmplitudeEncoding(n_features=-1)

    def test_normalize_string_raises_type_error(self) -> None:
        """Test that normalize='True' (string) raises TypeError.

        This is a common mistake where users pass string "True" instead of
        boolean True. The validation should catch this early with a clear message.
        """
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize="True")  # type: ignore

    def test_normalize_string_false_raises_type_error(self) -> None:
        """Test that normalize='False' (string) raises TypeError."""
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize="False")  # type: ignore

    def test_normalize_integer_one_raises_type_error(self) -> None:
        """Test that normalize=1 (integer) raises TypeError.

        In Python, 1 is truthy but not the same as True for strict boolean checks.
        We require explicit True/False for clarity.
        """
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize=1)  # type: ignore

    def test_normalize_integer_zero_raises_type_error(self) -> None:
        """Test that normalize=0 (integer) raises TypeError."""
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize=0)  # type: ignore

    def test_normalize_none_raises_type_error(self) -> None:
        """Test that normalize=None raises TypeError.

        None is falsy but not a boolean. Users should explicitly specify
        True or False.
        """
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize=None)  # type: ignore

    def test_normalize_list_raises_type_error(self) -> None:
        """Test that normalize=[] (empty list) raises TypeError."""
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize=[])  # type: ignore

    def test_normalize_dict_raises_type_error(self) -> None:
        """Test that normalize={} (empty dict) raises TypeError."""
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize={})  # type: ignore

    def test_normalize_float_raises_type_error(self) -> None:
        """Test that normalize=1.0 (float) raises TypeError."""
        with pytest.raises(TypeError, match="normalize must be a boolean"):
            AmplitudeEncoding(n_features=4, normalize=1.0)  # type: ignore

    def test_normalize_true_valid(self) -> None:
        """Test that normalize=True (boolean) is accepted."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        assert enc.normalize is True

    def test_normalize_false_valid(self) -> None:
        """Test that normalize=False (boolean) is accepted."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        assert enc.normalize is False

    def test_normalize_error_message_includes_type_and_value(self) -> None:
        """Test that TypeError message includes helpful debugging info.

        The error message should tell users:
        1. What type they provided
        2. What value they provided
        3. What the valid options are
        """
        with pytest.raises(TypeError) as excinfo:
            AmplitudeEncoding(n_features=4, normalize="yes")  # type: ignore

        error_msg = str(excinfo.value)

        # Should mention the expected type
        assert "boolean" in error_msg.lower()
        # Should mention what was provided
        assert "str" in error_msg
        # Should mention the actual value
        assert "yes" in error_msg
        # Should mention valid options
        assert "True" in error_msg
        assert "False" in error_msg


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of AmplitudeEncoding."""

    def test_n_qubits(self) -> None:
        """Test n_qubits property calculation.

        n_qubits = max(1, ceil(log2(n_features)))
        """
        enc = AmplitudeEncoding(n_features=4)
        assert enc.n_qubits == 2

    def test_exponential_compression(self) -> None:
        """Test that amplitude encoding provides exponential compression.

        2^n states can be encoded with n qubits.
        """
        for n_qubits in [2, 3, 4, 5]:
            n_features = 2**n_qubits
            enc = AmplitudeEncoding(n_features=n_features)
            assert enc.n_qubits == n_qubits

    def test_depth(self) -> None:
        """Test circuit depth calculation.

        Depth = 2^n_qubits for state preparation.
        """
        enc = AmplitudeEncoding(n_features=8)
        # depth = 2^n_qubits = 2^3 = 8
        assert enc.depth == 2**enc.n_qubits

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = AmplitudeEncoding(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = AmplitudeEncoding(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_is_entangling(self) -> None:
        """Test that amplitude encoding creates entanglement."""
        enc = AmplitudeEncoding(n_features=4)
        assert enc.properties.is_entangling is True

    def test_not_simulable(self) -> None:
        """Test that amplitude encoding is not efficiently simulable."""
        enc = AmplitudeEncoding(n_features=4)
        assert enc.properties.simulability == "not_simulable"


# =============================================================================
# Test Class: Normalization Behavior (Unique to AmplitudeEncoding)
# =============================================================================


class TestNormalizationBehavior:
    """Tests for normalization behavior unique to AmplitudeEncoding.

    When normalize=True (default), input is L2-normalized before encoding.
    When normalize=False, user must provide pre-normalized data.
    """

    @staticmethod
    def _normalize_vector(x: np.ndarray) -> np.ndarray:
        """Manually normalize a vector to unit L2 norm."""
        norm = np.linalg.norm(x)
        return x / norm

    def test_prenormalized_input_produces_valid_state(self) -> None:
        """Test that pre-normalized input with normalize=False produces valid state."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x_raw = np.array([1.0, 2.0, 3.0, 4.0])
        x_normalized = self._normalize_vector(x_raw)

        if HAS_PENNYLANE:
            circuit_fn = enc.get_circuit(x_normalized, backend="pennylane")
            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit() -> np.ndarray:
                circuit_fn()
                return qml.state()

            state = full_circuit()
            state_norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(state_norm, 1.0, atol=1e-10)

    def test_normalize_true_rescales_unnormalized_input(self) -> None:
        """Test that normalize=True rescales unnormalized input correctly."""
        x_unnorm = np.array([3.0, 4.0, 0.0, 0.0])  # norm = 5
        x_expected_normalized = x_unnorm / np.linalg.norm(x_unnorm)

        enc = AmplitudeEncoding(n_features=4, normalize=True)

        if HAS_PENNYLANE:
            circuit_fn = enc.get_circuit(x_unnorm, backend="pennylane")
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def full_circuit() -> np.ndarray:
                circuit_fn()
                return qml.state()

            state = full_circuit()
            actual_probs = np.abs(state) ** 2
            expected_probs = np.abs(x_expected_normalized) ** 2

            np.testing.assert_allclose(
                actual_probs,
                expected_probs,
                atol=1e-10,
            )

    def test_zero_vector_raises_with_normalize_true(self) -> None:
        """Test that an all-zero vector raises ValueError when normalize=True.

        An all-zero input cannot be normalized to a valid quantum state.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="zero.*norm"):
            enc.get_circuit(x, backend="pennylane")

    def test_zero_vector_raises_with_normalize_false(self) -> None:
        """Test that an all-zero vector raises ValueError even when normalize=False."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([0.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="zero.*norm"):
            enc.get_circuit(x, backend="pennylane")

    def test_near_zero_vector_raises_error(self) -> None:
        """Test that near-zero vectors (numerically unstable) raise ValueError."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1e-20, 1e-20, 1e-20, 1e-20])

        with pytest.raises(ValueError, match="zero.*norm"):
            enc.get_circuit(x, backend="pennylane")

    def test_partial_zero_vector_is_valid(self) -> None:
        """Test that vectors with some zeros but non-zero norm are valid."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1.0, 0.0, 0.0, 0.0])

        if HAS_PENNYLANE:
            circuit_fn = enc.get_circuit(x, backend="pennylane")
            assert callable(circuit_fn)


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        validated = default_encoding._validate_input(sample_data_4d)
        assert validated.shape == (4,)

    def test_wrong_feature_count(self, default_encoding: AmplitudeEncoding) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2])  # Only 2 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_nan_input_rejected(self, default_encoding: AmplitudeEncoding) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.5, np.nan, 0.5, 0.5])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding._validate_input(x)

    def test_inf_input_rejected(self, default_encoding: AmplitudeEncoding) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.5, np.inf, 0.5, 0.5])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_list_input_accepted(self, default_encoding: AmplitudeEncoding) -> None:
        """Test that list input is converted to array."""
        x = [0.5, 0.5, 0.5, 0.5]  # list, not array
        if HAS_QISKIT:
            circuit = default_encoding.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_batch_input_shape(
        self,
        default_encoding: AmplitudeEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that batch input generates multiple circuits."""
        if HAS_PENNYLANE:
            circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
            assert len(circuits) == 3


# =============================================================================
# Test Class: PennyLane Backend
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.backend_pennylane
class TestPennyLaneBackend:
    """Tests for PennyLane circuit generation."""

    def test_circuit_is_callable(
        self,
        default_encoding: AmplitudeEncoding,
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
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be normalized
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit uses correct number of qubits."""
        circuit_fn = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # 2 qubits = 4 states
        assert len(state) == 4

    def test_batch_circuits(
        self,
        default_encoding: AmplitudeEncoding,
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
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        # 4 features = 2 qubits
        assert circuit.num_qubits == 2

    def test_batch_circuits(
        self,
        default_encoding: AmplitudeEncoding,
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
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        # 4 features = 2 qubits
        assert len(circuit.all_qubits()) == 2

    def test_circuit_with_8_features(
        self,
        sample_data_8d: NDArray[np.floating],
    ) -> None:
        """Test Cirq circuit with 8 features."""
        enc = AmplitudeEncoding(n_features=8)
        circuit = enc.get_circuit(sample_data_8d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        # log2(8) = 3
        assert len(circuit.all_qubits()) == 3

    def test_batch_circuits(
        self,
        default_encoding: AmplitudeEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of amplitude encoding."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_uniform_input_produces_uniform_superposition(self) -> None:
        """Test that uniform input produces uniform probability distribution."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1.0, 1.0, 1.0, 1.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        probs = np.abs(state) ** 2

        # All probabilities should be 0.25
        np.testing.assert_allclose(probs, np.full(4, 0.25), atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_basis_state_encoding(self) -> None:
        """Test encoding computational basis states.

        Pre-normalized basis state [1, 0, 0, 0] should produce |00⟩.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x_basis = np.array([1.0, 0.0, 0.0, 0.0])

        circuit_fn = enc.get_circuit(x_basis, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        probs = np.abs(state) ** 2

        # Should have probability 1.0 on |00⟩
        expected_probs = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(probs, expected_probs, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_probability_distribution_matches_input_squared(self) -> None:
        """Test that output probabilities equal normalized input amplitudes squared.

        For a pre-normalized input x, P(i) = |x_i|^2.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=False)

        target_probs = np.array([0.1, 0.2, 0.3, 0.4])
        x_normalized = np.sqrt(target_probs)

        circuit_fn = enc.get_circuit(x_normalized, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        actual_probs = np.abs(state) ** 2

        np.testing.assert_allclose(actual_probs, target_probs, atol=1e-10)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_power_of_two_features(self) -> None:
        """Test with power-of-two number of features."""
        for n in [2, 4, 8, 16]:
            enc = AmplitudeEncoding(n_features=n)
            assert enc.n_qubits == int(np.log2(n))

    def test_non_power_of_two_features(self) -> None:
        """Test with non-power-of-two features (requires padding)."""
        enc = AmplitudeEncoding(n_features=5)
        # ceil(log2(5)) = 3
        assert enc.n_qubits == 3

    def test_large_feature_count(self) -> None:
        """Test encoding with large feature count."""
        enc = AmplitudeEncoding(n_features=100)
        # ceil(log2(100)) = 7
        assert enc.n_qubits == 7

    def test_single_feature_requires_one_qubit(self) -> None:
        """Test that n_features=1 correctly uses 1 qubit, not 0.

        This verifies the fix for the edge case where log2(1) = 0.
        """
        enc = AmplitudeEncoding(n_features=1)
        assert enc.n_qubits == 1
        assert enc.n_features == 1
        # depth = 2^1 = 2
        assert enc.depth == 2

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_single_feature_pennylane_circuit(self) -> None:
        """Test PennyLane circuit generation with single feature.

        The single feature [x] is padded to [x, 0] and normalized to [1, 0],
        which should produce the |0⟩ state.
        """
        enc = AmplitudeEncoding(n_features=1, normalize=True)
        x = np.array([1.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert len(state) == 2
        assert np.isclose(np.abs(state[0]) ** 2, 1.0, atol=1e-10)
        assert np.isclose(np.abs(state[1]) ** 2, 0.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_single_feature_qiskit_circuit(self) -> None:
        """Test Qiskit circuit generation with single feature."""
        enc = AmplitudeEncoding(n_features=1, normalize=True)
        x = np.array([1.0])

        circuit = enc.get_circuit(x, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 1

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_single_feature_cirq_circuit(self) -> None:
        """Test Cirq circuit generation with single feature."""
        enc = AmplitudeEncoding(n_features=1, normalize=True)
        x = np.array([1.0])

        circuit = enc.get_circuit(x, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 1

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_normalization_behavior(self) -> None:
        """Test that normalization works correctly."""
        x = np.array([3.0, 4.0, 0.0, 0.0])  # norm = 5
        enc = AmplitudeEncoding(n_features=4, normalize=True)

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_small_values(self) -> None:
        """Test encoding with very small input values.

        Values close to zero should not cause numerical issues.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_large_values(self) -> None:
        """Test encoding with very large input values.

        Large values should not cause overflow.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1e10, 2e10, 3e10, 4e10])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1e-10, 1e10, 1e-5, 1e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_near_zero_total_amplitude(self) -> None:
        """Test encoding when amplitudes nearly cancel out.

        This tests numerical stability when the norm is very small but non-zero.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1e-10, -1e-10, 1e-10, -1e-10])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        assert callable(circuit_fn)


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = AmplitudeEncoding(n_features=4, normalize=True)
        enc2 = AmplitudeEncoding(n_features=4, normalize=True)
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = AmplitudeEncoding(n_features=4)
        enc2 = AmplitudeEncoding(n_features=8)
        assert enc1 != enc2

    def test_equality_different_normalize(self) -> None:
        """Test that encodings with different normalize are not equal."""
        enc1 = AmplitudeEncoding(n_features=4, normalize=True)
        enc2 = AmplitudeEncoding(n_features=4, normalize=False)
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = AmplitudeEncoding(n_features=4, normalize=True)
        enc2 = AmplitudeEncoding(n_features=4, normalize=True)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = AmplitudeEncoding(n_features=4)
        enc2 = AmplitudeEncoding(n_features=4)  # Same as enc1
        enc3 = AmplitudeEncoding(n_features=8)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = AmplitudeEncoding(n_features=4)
        enc2 = AmplitudeEncoding(n_features=4)

        d = {enc1: "four features"}
        assert d[enc2] == "four features"


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ contains all parameters."""

    def test_repr_contains_class_name(self) -> None:
        """Test that repr contains the class name."""
        enc = AmplitudeEncoding(n_features=4)
        repr_str = repr(enc)
        assert "AmplitudeEncoding" in repr_str

    def test_repr_contains_n_features(self) -> None:
        """Test that repr contains n_features."""
        enc = AmplitudeEncoding(n_features=4)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str

    def test_repr_with_custom_parameters(self) -> None:
        """Test repr with custom parameters."""
        enc = AmplitudeEncoding(n_features=8, normalize=False)
        repr_str = repr(enc)
        assert "n_features=8" in repr_str

    def test_repr_single_feature(self) -> None:
        """Test string representation with single feature."""
        enc = AmplitudeEncoding(n_features=1)
        repr_str = repr(enc)
        assert "AmplitudeEncoding" in repr_str
        assert "n_features=1" in repr_str


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for invalid backend names and missing backends."""

    def test_invalid_backend_raises_error(
        self,
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_empty_backend_raises_error(
        self,
        default_encoding: AmplitudeEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that empty backend string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="")  # type: ignore


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = AmplitudeEncoding(n_features=8, normalize=False)
        # Force properties to be computed
        _ = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.normalize == enc.normalize
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = AmplitudeEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = AmplitudeEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.5, 0.3, 0.7, 0.1])
        if HAS_PENNYLANE:
            circuit = restored.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = AmplitudeEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        props = restored.properties
        assert isinstance(props, EncodingProperties)
        assert props.n_qubits == enc.n_qubits


# =============================================================================
# Test Class: Concurrent Access / Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        enc = AmplitudeEncoding(n_features=4)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for _i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
                    if HAS_PENNYLANE:
                        circuit = enc.get_circuit(x, backend="pennylane")
                        circuits.append(circuit)
            except Exception as e:
                errors.append(e)
            return circuits

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(generate_circuits, i) for i in range(num_threads)
            ]
            results = [f.result() for f in as_completed(futures)]

        assert len(errors) == 0, f"Thread errors: {errors}"
        if HAS_PENNYLANE:
            total_circuits = sum(len(r) for r in results)
            assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = AmplitudeEncoding(n_features=4)
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

    def test_concurrent_circuit_generation_threading(self) -> None:
        """Test that concurrent circuit generation is thread-safe using threading."""
        enc = AmplitudeEncoding(n_features=4)
        results: list[Any] = []
        errors: list[Exception] = []

        inputs = [np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]) for i in range(1, 11)]

        def generate_circuit(x: np.ndarray) -> None:
            try:
                if HAS_PENNYLANE:
                    circuit = enc.get_circuit(x, backend="pennylane")
                    results.append(circuit)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=generate_circuit, args=(x,)) for x in inputs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent generation: {errors}"
        if HAS_PENNYLANE:
            assert len(results) == len(inputs)


# =============================================================================
# Test Class: AmplitudeEncoding Specific Tests
# =============================================================================


class TestAmplitudeSpecific:
    """Additional tests specific to AmplitudeEncoding.

    These tests cover unique behaviors and internal implementation details
    that are specific to the amplitude encoding approach.
    """

    def test_padding_adds_zeros(self) -> None:
        """Test that padding adds zeros to reach power of 2.

        Uses normalize=True (default) so the input [1.0, 2.0, 3.0] is
        automatically normalized before being padded to 4 elements.
        """
        enc = AmplitudeEncoding(n_features=3, normalize=True)
        x = np.array([1.0, 2.0, 3.0])

        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_no_padding_for_power_of_two(self) -> None:
        """Test that no padding occurs for power-of-two features."""
        enc = AmplitudeEncoding(n_features=4)
        x = np.array([0.5, 0.5, 0.5, 0.5])

        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_cirq_determinism(self) -> None:
        """Test that Cirq circuit generation is deterministic.

        Repeated calls with the same input should produce identical unitaries.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.5, 0.3, 0.7, 0.1])

        circuits = [enc.get_circuit(x, backend="cirq") for _ in range(5)]
        unitaries = [cirq.unitary(c) for c in circuits]

        reference = unitaries[0]
        for i, unitary in enumerate(unitaries[1:], start=2):
            np.testing.assert_array_equal(
                unitary,
                reference,
                err_msg=f"Cirq circuit {i} differs from first circuit.",
            )

    def test_zero_vector_error_message_quality(self) -> None:
        """Test that the error message provides helpful guidance."""
        enc = AmplitudeEncoding(n_features=4)
        x = np.zeros(4)

        with pytest.raises(ValueError) as excinfo:
            enc.get_circuit(x, backend="pennylane")

        error_msg = str(excinfo.value)
        assert "zero" in error_msg.lower() or "near-zero" in error_msg.lower()
        assert "quantum state" in error_msg.lower()

    def test_single_feature_equality(self) -> None:
        """Test equality comparison for single-feature encodings."""
        enc1 = AmplitudeEncoding(n_features=1, normalize=True)
        enc2 = AmplitudeEncoding(n_features=1, normalize=True)
        enc3 = AmplitudeEncoding(n_features=1, normalize=False)

        assert enc1 == enc2
        assert enc1 != enc3
        assert hash(enc1) == hash(enc2)

    def test_memory_size_formatting(self) -> None:
        """Test the _format_memory_size helper function."""
        from encoding_atlas.encodings.amplitude import _format_memory_size

        assert _format_memory_size(0) == "0 bytes"
        assert _format_memory_size(1024) == "1.0 KB"
        assert _format_memory_size(1024**2) == "1.0 MB"
        assert _format_memory_size(1024**3) == "1.0 GB"

    def test_performance_instantiation_fast(self) -> None:
        """Test that instantiation is fast (< 1ms for small feature counts)."""
        n_iterations = 100
        feature_counts = [4, 8, 16, 32]

        for n_features in feature_counts:
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = AmplitudeEncoding(n_features=n_features)
            elapsed = time.perf_counter() - start
            avg_time_ms = (elapsed / n_iterations) * 1000

            assert avg_time_ms < 1.0, (
                f"Instantiation too slow for n_features={n_features}: "
                f"{avg_time_ms:.3f}ms"
            )


# =============================================================================
# Test Class: Slow Simulation Tests
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    """

    @staticmethod
    def _get_pennylane_state(
        enc: AmplitudeEncoding,
        x: np.ndarray,
    ) -> np.ndarray:
        """Execute PennyLane circuit and return state vector."""
        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> np.ndarray:
            circuit_fn()
            return qml.state()

        return np.array(full_circuit())

    @staticmethod
    def _get_qiskit_state(
        enc: AmplitudeEncoding,
        x: np.ndarray,
    ) -> np.ndarray:
        """Execute Qiskit circuit and return state vector."""
        circuit = enc.get_circuit(x, backend="qiskit")
        sv = Statevector(circuit)
        return np.array(sv.data)

    @staticmethod
    def _get_cirq_state(
        enc: AmplitudeEncoding,
        x: np.ndarray,
    ) -> np.ndarray:
        """Execute Cirq circuit and return state vector."""
        circuit = enc.get_circuit(x, backend="cirq")
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        return np.array(result.final_state_vector)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.5, 0.3, 0.7, 0.1])

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
    def test_cross_backend_probability_equivalence(self) -> None:
        """Test that all backends produce equivalent probability distributions.

        Compares sorted probability distributions as a basic sanity check.
        See also ``test_cross_backend_statevector_index_consistency`` for
        the stronger per-index comparison via the simulation utility.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.5, 0.3, 0.7, 0.1])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(pl_probs, qk_probs, atol=1e-6)
        np.testing.assert_allclose(pl_probs, cirq_probs, atol=1e-6)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_statevector_index_consistency(self) -> None:
        """Test per-index statevector consistency through the simulation utility.

        Unlike ``test_cross_backend_probability_equivalence`` (which sorts
        probabilities), this test verifies that the simulation utility
        returns statevectors with matching amplitude *positions* across
        all backends.  This ensures that ``_to_qiskit()``'s MSB→LSB
        index conversion is correct.
        """
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.5, 0.3, 0.7, 0.1])

        sv_pl = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qk = simulate_encoding_statevector(enc, x, backend="qiskit")
        sv_cirq = simulate_encoding_statevector(enc, x, backend="cirq")

        # Fidelity should be ~1.0 (not just same probability set)
        fidelity_qk = np.abs(np.vdot(sv_pl, sv_qk)) ** 2
        fidelity_cirq = np.abs(np.vdot(sv_pl, sv_cirq)) ** 2

        assert fidelity_qk > 0.9999, (
            f"PennyLane–Qiskit fidelity {fidelity_qk:.6f}: "
            f"amplitude index ordering may be mismatched"
        )
        assert fidelity_cirq > 0.9999, (
            f"PennyLane–Cirq fidelity {fidelity_cirq:.6f}: "
            f"amplitude index ordering may be mismatched"
        )

        # Per-index probability match
        np.testing.assert_allclose(
            np.abs(sv_pl) ** 2,
            np.abs(sv_qk) ** 2,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.abs(sv_pl) ** 2,
            np.abs(sv_cirq) ** 2,
            atol=1e-6,
        )

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        inputs = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]

        states = []
        for x in inputs:
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit() -> np.ndarray:
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # At least some states should be different
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                fidelity = np.abs(np.vdot(states[i], states[j])) ** 2
                assert fidelity < 0.9999 or np.allclose(inputs[i], inputs[j])

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input always produces same state."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        states = []
        for _ in range(5):
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit() -> np.ndarray:
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # All states should be identical
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_with_8_features(self) -> None:
        """Test cross-backend equivalence with 8 features (3 qubits)."""
        enc = AmplitudeEncoding(n_features=8, normalize=True)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # All states should have dimension 2^3 = 8
        assert len(pl_state) == 8
        assert len(qk_state) == 8
        assert len(cirq_state) == 8

        # Compare sorted probability distributions
        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(pl_probs, qk_probs, atol=1e-6)
        np.testing.assert_allclose(pl_probs, cirq_probs, atol=1e-6)

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    @pytest.mark.timeout(120)
    def test_cirq_memory_warning_at_threshold(self) -> None:
        """Test that a warning is emitted when n_qubits equals the threshold (12).

        At exactly 12 qubits (4096 features), the unitary matrix is 4096x4096
        complex128 elements = 256 MB, which triggers the warning.
        """

        enc = AmplitudeEncoding(n_features=4096)
        assert enc.n_qubits == 12

        x = np.random.randn(4096)

        with pytest.warns(UserWarning, match="Cirq backend memory warning"):
            enc.get_circuit(x, backend="cirq")

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    @pytest.mark.timeout(60)
    def test_cirq_no_warning_below_threshold(self) -> None:
        """Test that no warning is emitted when n_qubits is below threshold.

        At 11 qubits (2048 features), the memory requirement is 64 MB,
        which is below the 256 MB threshold.
        """
        import warnings as warnings_module

        enc = AmplitudeEncoding(n_features=2048)
        assert enc.n_qubits == 11

        x = np.random.randn(2048)

        with warnings_module.catch_warnings(record=True) as caught_warnings:
            warnings_module.simplefilter("always")
            enc.get_circuit(x, backend="cirq")

            memory_warnings = [
                w
                for w in caught_warnings
                if issubclass(w.category, UserWarning)
                and "Cirq backend memory warning" in str(w.message)
            ]
            assert len(memory_warnings) == 0


# =============================================================================
# Test Class: transform_input (DataTransformable Protocol)
# =============================================================================


class TestTransformInput:
    """Tests for the transform_input() method (DataTransformable protocol).

    This method exposes the encoding's internal preprocessing:
    validation, power-of-2 padding, and optional L2 normalization.
    """

    def test_normalizes_to_unit_norm(self) -> None:
        """Test that transform_input produces a unit-norm vector."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([3.0, 4.0, 0.0, 0.0])  # norm = 5

        result = enc.transform_input(x)

        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-15)

    def test_normalized_values_are_correct(self) -> None:
        """Test that normalized amplitudes match manual computation."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([3.0, 4.0, 0.0, 0.0])
        expected = x / np.linalg.norm(x)  # [0.6, 0.8, 0.0, 0.0]

        result = enc.transform_input(x)

        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_pads_non_power_of_two_features(self) -> None:
        """Test that non-power-of-2 features are zero-padded."""
        enc = AmplitudeEncoding(n_features=3)  # n_qubits=2, state_dim=4
        x = np.array([1.0, 0.0, 0.0])

        result = enc.transform_input(x)

        assert len(result) == 4  # padded to 2^2
        # [1, 0, 0] normalized -> [1, 0, 0], padded -> [1, 0, 0, 0]
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_no_padding_for_power_of_two(self) -> None:
        """Test that power-of-2 features are not padded."""
        enc = AmplitudeEncoding(n_features=4)
        x = np.array([0.5, 0.5, 0.5, 0.5])

        result = enc.transform_input(x)

        assert len(result) == 4

    def test_normalize_false_skips_normalization(self) -> None:
        """Test that normalize=False returns padded but unnormalized data."""
        enc = AmplitudeEncoding(n_features=3, normalize=False)
        # Pre-normalize a 3-feature vector; padding won't change its norm
        x_raw = np.array([1.0, 2.0, 2.0])
        x_pre = x_raw / np.linalg.norm(x_raw)

        result = enc.transform_input(x_pre)

        assert len(result) == 4  # padded to 2^2
        # First 3 elements unchanged, 4th is zero
        np.testing.assert_allclose(result[:3], x_pre, atol=1e-15)
        assert result[3] == 0.0

    def test_zero_vector_raises_error(self) -> None:
        """Test that a zero vector raises ValueError."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.zeros(4)

        with pytest.raises(ValueError, match="zero.*norm"):
            enc.transform_input(x)

    def test_wrong_shape_raises_error(self) -> None:
        """Test that wrong feature count raises ValueError."""
        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 2.0])  # only 2 features

        with pytest.raises(ValueError, match="Expected 4 features"):
            enc.transform_input(x)

    def test_batch_input_raises_error(self) -> None:
        """Test that multi-sample 2D input raises ValueError."""
        enc = AmplitudeEncoding(n_features=4)
        X = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        with pytest.raises(ValueError, match="single sample"):
            enc.transform_input(X)

    def test_2d_single_sample_accepted(self) -> None:
        """Test that (1, n_features) shaped input is accepted."""
        enc = AmplitudeEncoding(n_features=4)
        x = np.array([[3.0, 4.0, 0.0, 0.0]])

        result = enc.transform_input(x)

        assert result.ndim == 1
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-15)

    def test_list_input_accepted(self) -> None:
        """Test that plain list input is accepted and processed."""
        enc = AmplitudeEncoding(n_features=4)
        result = enc.transform_input([3.0, 4.0, 0.0, 0.0])

        assert isinstance(result, np.ndarray)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-15)

    def test_idempotency_on_normalized_data(self) -> None:
        """Test approximate idempotency for already-normalized data."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.5, 0.5, 0.5, 0.5])

        first = enc.transform_input(x)
        # Make writeable copy for second call (transform_input returns immutable)
        second = enc.transform_input(first.copy())

        np.testing.assert_allclose(first, second, atol=1e-14)

    def test_satisfies_data_transformable_protocol(self) -> None:
        """Test that AmplitudeEncoding satisfies the DataTransformable protocol."""
        from encoding_atlas.core.protocols import DataTransformable

        enc = AmplitudeEncoding(n_features=4)

        assert isinstance(enc, DataTransformable)


# =============================================================================
# Test Class: gate_count_breakdown (ResourceAnalyzable Protocol)
# =============================================================================


class TestGateCountBreakdown:
    """Tests for the gate_count_breakdown() method.

    Gate counts are theoretical estimates based on the Möttönen et al.
    decomposition: ~2^n rotations and ~2^n - 2 CNOTs for n qubits.
    """

    def test_return_type_is_dict(self) -> None:
        """Test that gate_count_breakdown returns a dict."""
        enc = AmplitudeEncoding(n_features=4)
        breakdown = enc.gate_count_breakdown()

        assert isinstance(breakdown, dict)

    def test_required_keys_present(self) -> None:
        """Test that all documented keys are present."""
        enc = AmplitudeEncoding(n_features=8)
        breakdown = enc.gate_count_breakdown()

        expected_keys = {
            "rotation_gates",
            "cnot",
            "total_single_qubit",
            "total_two_qubit",
            "total",
            "state_dimension",
            "is_estimate",
        }
        assert expected_keys == set(breakdown.keys())

    def test_is_estimate_always_true(self) -> None:
        """Test that is_estimate is always True for amplitude encoding."""
        for n in [2, 4, 8, 16]:
            enc = AmplitudeEncoding(n_features=n)
            assert enc.gate_count_breakdown()["is_estimate"] is True

    def test_4_features_gate_counts(self) -> None:
        """Test gate counts for 4 features (2 qubits, state_dim=4).

        rotation_gates = 4, cnot = 4-2 = 2, total = 6.
        """
        enc = AmplitudeEncoding(n_features=4)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["rotation_gates"] == 4
        assert breakdown["cnot"] == 2
        assert breakdown["total_single_qubit"] == 4
        assert breakdown["total_two_qubit"] == 2
        assert breakdown["total"] == 6
        assert breakdown["state_dimension"] == 4

    def test_8_features_gate_counts(self) -> None:
        """Test gate counts for 8 features (3 qubits, state_dim=8).

        rotation_gates = 8, cnot = 8-2 = 6, total = 14.
        """
        enc = AmplitudeEncoding(n_features=8)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["rotation_gates"] == 8
        assert breakdown["cnot"] == 6
        assert breakdown["total"] == 14
        assert breakdown["state_dimension"] == 8

    def test_16_features_gate_counts(self) -> None:
        """Test gate counts for 16 features (4 qubits, state_dim=16).

        rotation_gates = 16, cnot = 16-2 = 14, total = 30.
        """
        enc = AmplitudeEncoding(n_features=16)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["rotation_gates"] == 16
        assert breakdown["cnot"] == 14
        assert breakdown["total"] == 30

    def test_single_feature_gate_counts(self) -> None:
        """Test gate counts for 1 feature (1 qubit, state_dim=2).

        rotation_gates = 2, cnot = max(0, 2-2) = 0, total = 2.
        Single-qubit state prep requires no entanglement.
        """
        enc = AmplitudeEncoding(n_features=1)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["rotation_gates"] == 2
        assert breakdown["cnot"] == 0
        assert breakdown["total_two_qubit"] == 0
        assert breakdown["total"] == 2

    def test_total_equals_single_plus_two_qubit(self) -> None:
        """Test that total = total_single_qubit + total_two_qubit."""
        for n in [1, 2, 3, 4, 5, 8, 16, 32]:
            enc = AmplitudeEncoding(n_features=n)
            bd = enc.gate_count_breakdown()
            assert bd["total"] == bd["total_single_qubit"] + bd["total_two_qubit"], (
                f"n_features={n}: {bd['total']} != "
                f"{bd['total_single_qubit']} + {bd['total_two_qubit']}"
            )

    def test_non_power_of_two_uses_padded_state_dim(self) -> None:
        """Test that non-power-of-2 features use the padded state dimension.

        5 features → 3 qubits → state_dim=8, same counts as 8 features.
        """
        enc_5 = AmplitudeEncoding(n_features=5)
        enc_8 = AmplitudeEncoding(n_features=8)

        assert enc_5.gate_count_breakdown() == enc_8.gate_count_breakdown()

    def test_exponential_scaling(self) -> None:
        """Test that gate counts scale exponentially with qubits."""
        counts = []
        for n_qubits in range(1, 6):
            enc = AmplitudeEncoding(n_features=2**n_qubits)
            counts.append(enc.gate_count_breakdown()["total"])

        # Each doubling of state dim should roughly double the gate count
        for i in range(1, len(counts)):
            # total = 2 * state_dim - 2, so ratio is ~2 for large state_dim
            ratio = counts[i] / counts[i - 1]
            assert (
                ratio > 1.5
            ), f"Gate count did not scale: {counts[i-1]} -> {counts[i]}"


# =============================================================================
# Test Class: resource_summary (ResourceAnalyzable Protocol)
# =============================================================================


class TestResourceSummary:
    """Tests for the resource_summary() method.

    Provides a comprehensive breakdown of circuit resources including
    qubit requirements, compression ratio, memory estimates, etc.
    """

    def test_return_type_is_dict(self) -> None:
        """Test that resource_summary returns a dict."""
        enc = AmplitudeEncoding(n_features=4)
        summary = enc.resource_summary()

        assert isinstance(summary, dict)

    def test_required_keys_present(self) -> None:
        """Test that all documented keys are present."""
        enc = AmplitudeEncoding(n_features=8)
        summary = enc.resource_summary()

        expected_keys = {
            "n_features",
            "n_qubits",
            "state_dimension",
            "compression_ratio",
            "padding_zeros",
            "depth",
            "normalize",
            "theoretical_gate_count",
            "theoretical_single_qubit_gates",
            "theoretical_two_qubit_gates",
            "is_entangling",
            "simulability",
            "cirq_unitary_memory_bytes",
            "cirq_unitary_memory_human",
            "backend_notes",
        }
        assert expected_keys == set(summary.keys())

    def test_basic_structure_4_features(self) -> None:
        """Test basic structure values for 4 features (2 qubits)."""
        enc = AmplitudeEncoding(n_features=4)
        summary = enc.resource_summary()

        assert summary["n_features"] == 4
        assert summary["n_qubits"] == 2
        assert summary["state_dimension"] == 4
        assert summary["padding_zeros"] == 0
        assert summary["depth"] == 4  # 2^2

    def test_compression_ratio(self) -> None:
        """Test compression ratio calculation (n_features / n_qubits)."""
        enc = AmplitudeEncoding(n_features=16)
        summary = enc.resource_summary()

        # 16 features / 4 qubits = 4.0
        assert summary["compression_ratio"] == 4.0

    def test_compression_ratio_non_power_of_two(self) -> None:
        """Test compression ratio for non-power-of-2 features."""
        enc = AmplitudeEncoding(n_features=8)
        summary = enc.resource_summary()

        # 8 features / 3 qubits ≈ 2.667
        expected = 8.0 / 3.0
        assert np.isclose(summary["compression_ratio"], expected)

    def test_padding_zeros_power_of_two(self) -> None:
        """Test that power-of-2 features have zero padding."""
        enc = AmplitudeEncoding(n_features=8)
        summary = enc.resource_summary()

        assert summary["padding_zeros"] == 0

    def test_padding_zeros_non_power_of_two(self) -> None:
        """Test padding count for non-power-of-2 features."""
        enc = AmplitudeEncoding(n_features=5)
        summary = enc.resource_summary()

        # 5 features → 3 qubits → state_dim=8, padding=3
        assert summary["padding_zeros"] == 3

    def test_theoretical_gate_counts(self) -> None:
        """Test theoretical gate count formulas (2n-2 total)."""
        enc = AmplitudeEncoding(n_features=8)
        summary = enc.resource_summary()

        state_dim = 8
        assert summary["theoretical_gate_count"] == 2 * state_dim - 2  # 14
        assert summary["theoretical_single_qubit_gates"] == state_dim  # 8
        assert summary["theoretical_two_qubit_gates"] == state_dim - 2  # 6

    def test_encoding_characteristics(self) -> None:
        """Test fixed encoding characteristics."""
        enc = AmplitudeEncoding(n_features=4)
        summary = enc.resource_summary()

        assert summary["is_entangling"] is True
        assert summary["simulability"] == "not_simulable"

    def test_normalize_flag_reflected(self) -> None:
        """Test that the normalize setting is reflected in summary."""
        enc_true = AmplitudeEncoding(n_features=4, normalize=True)
        enc_false = AmplitudeEncoding(n_features=4, normalize=False)

        assert enc_true.resource_summary()["normalize"] is True
        assert enc_false.resource_summary()["normalize"] is False

    def test_cirq_memory_calculation(self) -> None:
        """Test Cirq unitary memory estimation (2^n × 2^n × 16 bytes)."""
        enc = AmplitudeEncoding(n_features=4)
        summary = enc.resource_summary()

        # state_dim=4, memory = 4*4*16 = 256 bytes
        assert summary["cirq_unitary_memory_bytes"] == 256

    def test_cirq_memory_human_readable(self) -> None:
        """Test human-readable memory string for larger encodings."""
        enc = AmplitudeEncoding(n_features=1024)
        summary = enc.resource_summary()

        # 10 qubits, state_dim=1024, memory = 1024^2 * 16 = 16 MB
        assert summary["cirq_unitary_memory_human"] == "16.0 MB"

    def test_backend_notes_has_all_backends(self) -> None:
        """Test that backend_notes includes all three backends."""
        enc = AmplitudeEncoding(n_features=4)
        notes = enc.resource_summary()["backend_notes"]

        assert "pennylane" in notes
        assert "qiskit" in notes
        assert "cirq" in notes

    def test_depth_matches_property(self) -> None:
        """Test that depth in summary matches the depth property."""
        enc = AmplitudeEncoding(n_features=16)
        summary = enc.resource_summary()

        assert summary["depth"] == enc.depth

    def test_single_feature_summary(self) -> None:
        """Test resource summary for the edge case of 1 feature."""
        enc = AmplitudeEncoding(n_features=1)
        summary = enc.resource_summary()

        assert summary["n_features"] == 1
        assert summary["n_qubits"] == 1
        assert summary["state_dimension"] == 2
        assert summary["padding_zeros"] == 1  # 1 feature padded to 2
        assert summary["compression_ratio"] == 1.0


# =============================================================================
# Test Class: normalize=False Rejection (Non-Unit Norm)
# =============================================================================


class TestNormalizeFalseRejection:
    """Tests for the normalize=False norm validation in get_circuit().

    When normalize=False, the encoding validates that the input vector
    has L2 norm close to 1.0 (within 1% relative tolerance). This
    catches the common mistake of forgetting to pre-normalize data.
    """

    def test_unnormalized_data_rejected(self) -> None:
        """Test that raw unnormalized data raises ValueError."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([1.0, 2.0, 3.0, 4.0])  # norm ≈ 5.48

        with pytest.raises(ValueError, match="normalize=False"):
            enc.get_circuit(x, backend="pennylane")

    def test_error_message_includes_actual_norm(self) -> None:
        """Test that the error message reports the actual norm value."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError) as excinfo:
            enc.get_circuit(x, backend="pennylane")

        error_msg = str(excinfo.value)
        # Should contain the norm value (5.477...)
        assert "5.4" in error_msg
        # Should suggest remediation options
        assert "normalize=True" in error_msg
        assert "np.linalg.norm" in error_msg

    def test_exactly_unit_norm_accepted(self) -> None:
        """Test that a vector with exact unit norm is accepted."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([0.5, 0.5, 0.5, 0.5])  # norm = 1.0

        # Should not raise
        if HAS_PENNYLANE:
            circuit = enc.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_slightly_off_unit_norm_accepted(self) -> None:
        """Test that small numerical deviations from unit norm pass.

        The tolerance is 1% relative, so norm=1.005 should be accepted.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([0.5, 0.5, 0.5, 0.5])
        # Introduce tiny deviation (within 1% rtol)
        x_slightly_off = x * 1.003  # norm ≈ 1.003

        if HAS_PENNYLANE:
            circuit = enc.get_circuit(x_slightly_off, backend="pennylane")
            assert callable(circuit)

    def test_beyond_tolerance_rejected(self) -> None:
        """Test that norm deviations beyond 1% are rejected."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([0.5, 0.5, 0.5, 0.5])
        x_too_far = x * 1.02  # norm ≈ 1.02, >1% off

        with pytest.raises(ValueError, match="normalize=False"):
            enc.get_circuit(x_too_far, backend="pennylane")

    def test_subnormalized_data_rejected(self) -> None:
        """Test that vectors with norm significantly below 1 are rejected."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([0.1, 0.1, 0.1, 0.1])  # norm = 0.2

        with pytest.raises(ValueError, match="normalize=False"):
            enc.get_circuit(x, backend="pennylane")

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_rejection_works_for_qiskit_backend(self) -> None:
        """Test that norm rejection also triggers for Qiskit backend."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="normalize=False"):
            enc.get_circuit(x, backend="qiskit")

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_rejection_works_for_cirq_backend(self) -> None:
        """Test that norm rejection also triggers for Cirq backend."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="normalize=False"):
            enc.get_circuit(x, backend="cirq")

    def test_rejection_in_get_circuit_from_validated(self) -> None:
        """Test that _get_circuit_from_validated also enforces norm check.

        This internal method is used by get_circuits() for batch processing.
        """
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x_unnorm = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="normalize=False"):
            enc._get_circuit_from_validated(x_unnorm, "pennylane")

    def test_prenormalized_data_accepted_all_backends(self) -> None:
        """Test that properly pre-normalized data works across all backends."""
        enc = AmplitudeEncoding(n_features=4, normalize=False)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        x_norm = x / np.linalg.norm(x)

        if HAS_PENNYLANE:
            assert callable(enc.get_circuit(x_norm, backend="pennylane"))
        if HAS_QISKIT:
            circuit = enc.get_circuit(x_norm, backend="qiskit")
            assert circuit.num_qubits == 2
        if HAS_CIRQ:
            circuit = enc.get_circuit(x_norm, backend="cirq")
            assert len(circuit.all_qubits()) == 2
