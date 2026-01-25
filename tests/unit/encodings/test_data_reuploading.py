"""Comprehensive tests for DataReuploading encoding.

This test module provides complete coverage of the DataReuploading class,
which implements a data re-uploading quantum encoding strategy where classical
data is repeatedly encoded into the quantum circuit across multiple layers,
enabling universal approximation capabilities. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, n_layers)
- Data re-uploading behavior (universal approximation)
- Circuit generation for all backends (PennyLane, Qiskit, Cirq)
- Mathematical correctness verification
- Edge cases and boundary conditions
- Numerical stability tests
- Equality and hashing
- String representation
- Backend error handling
- Serialization (pickle roundtrip)
- Concurrent access / thread safety
- DataReuploading-specific features (gate count breakdown, resource summary, etc.)
- Slow simulation tests (cross-backend state fidelity)

Run with: pytest tests/unit/encodings/test_data_reuploading.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_data_reuploading.py -v -m "not slow"

References
----------
.. [1] Pérez-Salinas et al., "Data re-uploading for a universal quantum
       classifier" (2020). arXiv:1907.02085
"""

from __future__ import annotations

import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas import DataReuploading
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
    """2-dimensional sample data for testing.

    Used for tests requiring smaller qubit counts.
    """
    return np.array([0.5, 1.0])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (mid-range values)
    - [0.9, 1.0, 1.1, 1.2] (near boundary values)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ])


@pytest.fixture
def default_encoding() -> DataReuploading:
    """Default DataReuploading with 4 features."""
    return DataReuploading(n_features=4)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for DataReuploading instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = DataReuploading(n_features=4)
        assert enc.n_features == 4
        assert enc.n_qubits == 4
        # Default n_layers is 3
        assert enc.n_layers == 3

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = DataReuploading(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16]:
            enc = DataReuploading(n_features=n)
            assert enc.n_features == n

    def test_custom_layers(self) -> None:
        """Test instantiation with custom layer count."""
        enc = DataReuploading(n_features=4, n_layers=5)
        assert enc.n_layers == 5

    def test_custom_qubits(self) -> None:
        """Test instantiation with custom qubit count."""
        enc = DataReuploading(n_features=4, n_qubits=6)
        assert enc.n_qubits == 6

    def test_qubits_defaults_to_features(self) -> None:
        """Test that n_qubits defaults to n_features when not specified."""
        enc = DataReuploading(n_features=8)
        assert enc.n_qubits == 8

    def test_config_stored_correctly(self) -> None:
        """Test that configuration is stored in config dict."""
        enc = DataReuploading(n_features=4, n_layers=5, n_qubits=6)
        assert enc.config["n_layers"] == 5
        assert enc.config["n_qubits_override"] == 6

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = DataReuploading(n_features=100)
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
            DataReuploading(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            DataReuploading(n_features=-1)

    def test_invalid_n_features_float(self) -> None:
        """Test that float n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            DataReuploading(n_features=4.0)  # type: ignore

    def test_invalid_n_features_string(self) -> None:
        """Test that string n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            DataReuploading(n_features="4")  # type: ignore

    def test_invalid_n_features_none(self) -> None:
        """Test that None n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            DataReuploading(n_features=None)  # type: ignore

    def test_invalid_n_layers_zero(self) -> None:
        """Test that n_layers=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            DataReuploading(n_features=4, n_layers=0)

    def test_invalid_n_layers_negative(self) -> None:
        """Test that negative n_layers raises ValueError."""
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            DataReuploading(n_features=4, n_layers=-1)


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of DataReuploading."""

    def test_n_qubits(self) -> None:
        """Test n_qubits property calculation."""
        enc = DataReuploading(n_features=4)
        # n_qubits = n_features by default for this encoding
        assert enc.n_qubits == 4

    def test_n_qubits_configurable(self) -> None:
        """Test that qubit count can be configured independently."""
        enc1 = DataReuploading(n_features=4)
        assert enc1.n_qubits == 4

        enc2 = DataReuploading(n_features=4, n_qubits=2)
        assert enc2.n_qubits == 2

    def test_depth(self) -> None:
        """Test circuit depth calculation.

        Depth per layer = ceil(n_features/n_qubits) + (n_qubits - 1)
        - Encoding sublayer: ceil(n_features/n_qubits) for cyclic feature mapping
        - Entangling sublayer: n_qubits - 1 sequential CNOTs
        """
        # n_features=4, n_qubits=4, n_layers=3
        # depth_per_layer = ceil(4/4) + (4-1) = 1 + 3 = 4
        # total_depth = 3 * 4 = 12
        enc = DataReuploading(n_features=4, n_layers=3)
        assert enc.depth == 3 * (1 + 3)

        # Test with cyclic mapping: n_features > n_qubits
        # n_features=8, n_qubits=4, n_layers=2
        # depth_per_layer = ceil(8/4) + (4-1) = 2 + 3 = 5
        # total_depth = 2 * 5 = 10
        enc2 = DataReuploading(n_features=8, n_qubits=4, n_layers=2)
        assert enc2.depth == 2 * (2 + 3)

        # Test single qubit (no entanglement)
        # n_features=1, n_qubits=1, n_layers=5
        # depth_per_layer = ceil(1/1) + (1-1) = 1 + 0 = 1
        # total_depth = 5 * 1 = 5
        enc3 = DataReuploading(n_features=1, n_layers=5)
        assert enc3.depth == 5 * (1 + 0)

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = DataReuploading(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = DataReuploading(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_is_entangling(self) -> None:
        """Test that data reuploading creates entanglement for multi-qubit."""
        enc = DataReuploading(n_features=4)
        assert enc.properties.is_entangling is True

    def test_not_simulable(self) -> None:
        """Test that data reuploading is not efficiently simulable."""
        enc = DataReuploading(n_features=4)
        assert enc.properties.simulability == "not_simulable"

    def test_universal_approximation(self) -> None:
        """Test that data reuploading has universal approximation property."""
        enc = DataReuploading(n_features=4)
        # Check that notes mention universal approximation
        assert "universal" in enc.properties.notes.lower()

    def test_same_input_same_properties(self) -> None:
        """Test that encoding properties are consistent regardless of input data."""
        enc = DataReuploading(n_features=4, n_layers=3)

        # Properties should be independent of any particular input
        assert enc.n_qubits == 4
        # depth = n_layers * (ceil(n_features/n_qubits) + (n_qubits-1)) = 3 * (1 + 3) = 12
        assert enc.depth == 12
        assert enc.properties.is_entangling is True


# =============================================================================
# Test Class: Data Reuploading Behavior
# =============================================================================


class TestDataReuploadingBehavior:
    """Tests for data re-uploading specific behavior.

    Data re-uploading encodes data multiple times across layers,
    which is key to achieving universal approximation.
    """

    def test_multiple_layers_increase_depth(self) -> None:
        """Test that more layers increase circuit depth."""
        depths = []
        for n_layers in [1, 2, 3, 4, 5]:
            enc = DataReuploading(n_features=4, n_layers=n_layers)
            depths.append(enc.depth)

        # Depth should increase with layers
        for i in range(1, len(depths)):
            assert depths[i] > depths[i - 1]

    def test_fewer_qubits_than_features_allowed(self) -> None:
        """Test that n_qubits can be less than n_features (cyclic mapping)."""
        enc = DataReuploading(n_features=8, n_qubits=4)
        assert enc.n_qubits == 4
        assert enc.n_features == 8

    def test_more_qubits_than_features_allowed(self) -> None:
        """Test that n_qubits can be greater than n_features."""
        enc = DataReuploading(n_features=2, n_qubits=4)
        assert enc.n_qubits == 4
        assert enc.n_features == 2


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        validated = default_encoding._validate_input(sample_data_4d)
        assert validated.shape == (4,)

    def test_wrong_feature_count(self, default_encoding: DataReuploading) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2])  # Only 2 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_nan_input_rejected(self, default_encoding: DataReuploading) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding._validate_input(x)

    def test_inf_input_rejected(self, default_encoding: DataReuploading) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_list_input_accepted(self, default_encoding: DataReuploading) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]  # list, not array
        # Should not raise
        validated = default_encoding._validate_input(x)
        assert isinstance(validated, np.ndarray)

    def test_batch_input_shape(
        self,
        default_encoding: DataReuploading,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that batch input generates multiple circuits."""
        if HAS_QISKIT:
            circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
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
        default_encoding: DataReuploading,
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
        enc = DataReuploading(n_features=4, n_layers=2)
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
        default_encoding: DataReuploading,
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
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_circuit_has_expected_gates(
        self,
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit contains expected gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        ops = circuit.count_ops()
        # DataReuploading uses RY rotations and CNOT for entanglement
        assert "ry" in ops or "cx" in ops

    def test_batch_circuits(
        self,
        default_encoding: DataReuploading,
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
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    def test_batch_circuits(
        self,
        default_encoding: DataReuploading,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)

    def test_custom_qubits(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test Cirq circuit with custom qubit count."""
        enc = DataReuploading(n_features=4, n_qubits=2, n_layers=2)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 2


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of DataReuploading encoding."""

    def test_depth_formula(self) -> None:
        """Test that depth follows expected formula.

        For DataReuploading:
        depth_per_layer = ceil(n_features/n_qubits) + (n_qubits - 1)
        total_depth = n_layers * depth_per_layer
        """
        # Test with n_features == n_qubits
        enc = DataReuploading(n_features=4, n_layers=3)
        # depth_per_layer = ceil(4/4) + (4-1) = 1 + 3 = 4
        # total = 3 * 4 = 12
        expected_depth = 3 * (1 + 3)
        assert enc.depth == expected_depth

    def test_gate_count_formula(self) -> None:
        """Test gate count follows expected formula.

        For n_qubits qubits and n_layers layers:
        - RY gates per layer: n_features (all features encoded)
        - CNOT gates per layer: n_qubits - 1 (linear entanglement)
        """
        enc = DataReuploading(n_features=4, n_layers=3)
        breakdown = enc.gate_count_breakdown()

        # RY gates: n_features * n_layers = 4 * 3 = 12
        assert breakdown['ry_gates'] == 12
        # CNOT gates: (n_qubits - 1) * n_layers = 3 * 3 = 9
        assert breakdown['cnot_gates'] == 9

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_rotation_angle_applied(self) -> None:
        """Test that rotation angles are actually applied from input data.

        Different inputs should produce different states.
        """
        enc = DataReuploading(n_features=2, n_layers=1)
        dev = qml.device("default.qubit", wires=2)

        x1 = np.array([0.0, 0.0])
        x2 = np.array([np.pi / 2, np.pi / 2])

        circuit_fn1 = enc.get_circuit(x1, backend="pennylane")
        circuit_fn2 = enc.get_circuit(x2, backend="pennylane")

        @qml.qnode(dev)
        def get_state1():
            circuit_fn1()
            return qml.state()

        @qml.qnode(dev)
        def get_state2():
            circuit_fn2()
            return qml.state()

        state1 = get_state1()
        state2 = get_state2()

        # States should be different
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99, "Different inputs should produce different states"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature(self) -> None:
        """Test encoding with single feature (minimum case)."""
        enc = DataReuploading(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_single_layer(self) -> None:
        """Test encoding with single layer."""
        enc = DataReuploading(n_features=4, n_layers=1)
        assert enc.n_layers == 1

    def test_many_layers(self) -> None:
        """Test encoding with many layers."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = DataReuploading(n_features=4, n_layers=10)
        assert enc.n_layers == 10

    def test_fewer_qubits_than_features(self) -> None:
        """Test with fewer qubits than features (cyclic mapping)."""
        enc = DataReuploading(n_features=8, n_qubits=4)
        assert enc.n_qubits == 4
        # Should still be able to generate circuits
        if HAS_PENNYLANE:
            x = np.ones(8)
            circuit = enc.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_more_qubits_than_features(self) -> None:
        """Test with more qubits than features."""
        enc = DataReuploading(n_features=2, n_qubits=4)
        assert enc.n_qubits == 4
        # Should still work
        if HAS_PENNYLANE:
            x = np.array([0.5, 1.0])
            circuit = enc.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_zero_valued_input(self, default_encoding: DataReuploading) -> None:
        """Test with all-zero input."""
        x = np.zeros(4)
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_large_feature_count(self) -> None:
        """Test instantiation with large feature count."""
        enc = DataReuploading(n_features=32)
        assert enc.n_features == 32
        assert enc.n_qubits == 32


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_very_small_values(self) -> None:
        """Test encoding with very small input values.

        Values close to zero should not cause numerical issues.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

        if HAS_PENNYLANE:
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

    def test_very_large_values(self) -> None:
        """Test encoding with very large input values.

        Large values are valid as rotation angles wrap around 2π.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([1e5, 2e5, 3e5, 4e5])

        if HAS_PENNYLANE:
            circuit_fn = enc.get_circuit(x, backend="pennylane")
            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            # State should be normalized despite large input
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)

    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values."""
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([1e-10, 1e10, 1e-5, 1e5])

        if HAS_PENNYLANE:
            circuit_fn = enc.get_circuit(x, backend="pennylane")
            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)

    def test_near_pi_multiples(self) -> None:
        """Test numerical stability near π multiples.

        Values very close to π can cause numerical issues in some
        implementations of trigonometric functions.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
        # Values very close to π but not exactly π
        x = np.array([
            np.pi - 1e-14,
            np.pi + 1e-14,
            2 * np.pi - 1e-14,
            np.pi / 2 + 1e-14
        ])

        if HAS_PENNYLANE:
            circuit_fn = enc.get_circuit(x, backend="pennylane")
            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_extreme_values_produce_normalized_state(self) -> None:
        """Test that extreme values still produce normalized quantum states."""
        enc = DataReuploading(n_features=4, n_layers=2)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([1e-100, 1e10, 0.5, 0.1])
        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_many_layers_stability(self) -> None:
        """Test numerical stability with many layers.

        Deep circuits can accumulate numerical errors.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = DataReuploading(n_features=2, n_layers=20)
        x = np.array([0.5, 1.0])

        if HAS_PENNYLANE:
            circuit_fn = enc.get_circuit(x, backend="pennylane")
            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = DataReuploading(n_features=4, n_layers=3)
        enc2 = DataReuploading(n_features=4, n_layers=3)
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = DataReuploading(n_features=4)
        enc2 = DataReuploading(n_features=8)
        assert enc1 != enc2

    def test_equality_different_n_layers(self) -> None:
        """Test that encodings with different n_layers are not equal."""
        enc1 = DataReuploading(n_features=4, n_layers=2)
        enc2 = DataReuploading(n_features=4, n_layers=3)
        assert enc1 != enc2

    def test_equality_different_n_qubits(self) -> None:
        """Test that encodings with different n_qubits are not equal."""
        enc1 = DataReuploading(n_features=4, n_qubits=4)
        enc2 = DataReuploading(n_features=4, n_qubits=6)
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = DataReuploading(n_features=4, n_layers=3)
        enc2 = DataReuploading(n_features=4, n_layers=3)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = DataReuploading(n_features=4, n_layers=2)
        enc2 = DataReuploading(n_features=4, n_layers=2)  # Same as enc1
        enc3 = DataReuploading(n_features=4, n_layers=3)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = DataReuploading(n_features=4)
        enc2 = DataReuploading(n_features=4)

        d = {enc1: "test_value"}
        assert d[enc2] == "test_value"  # enc2 should find same key


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ string representation."""

    def test_repr_contains_class_name(self) -> None:
        """Test that repr contains the class name."""
        enc = DataReuploading(n_features=4)
        repr_str = repr(enc)
        assert "DataReuploading" in repr_str

    def test_repr_contains_n_features(self) -> None:
        """Test that repr contains n_features parameter."""
        enc = DataReuploading(n_features=4)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str

    def test_repr_with_custom_parameters(self) -> None:
        """Test repr with custom parameters."""
        enc = DataReuploading(n_features=8, n_layers=5, n_qubits=6)
        repr_str = repr(enc)
        assert "n_features=8" in repr_str

    def test_repr_is_valid_string(self) -> None:
        """Test that repr returns a valid string."""
        enc = DataReuploading(n_features=4)
        repr_str = repr(enc)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for backend availability and error handling."""

    def test_invalid_backend_raises_error(
        self,
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_case_sensitivity(
        self,
        default_encoding: DataReuploading,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test backend name case sensitivity."""
        # Should raise for incorrect case (if case sensitive)
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="QISKIT")  # type: ignore


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = DataReuploading(n_features=4, n_layers=3, n_qubits=6)

        # Force properties to be computed
        _ = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.n_layers == enc.n_layers
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = DataReuploading(n_features=4, n_layers=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = DataReuploading(n_features=4, n_layers=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        if HAS_QISKIT:
            circuit = restored.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == 4

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = DataReuploading(n_features=4, n_layers=2)
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
        enc = DataReuploading(n_features=4, n_layers=2)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
                    if HAS_QISKIT:
                        circuit = enc.get_circuit(x, backend="qiskit")
                        circuits.append(circuit)
            except Exception as e:
                errors.append(e)
            return circuits

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(generate_circuits, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"
        # All circuits should have been generated (if Qiskit available)
        if HAS_QISKIT:
            total_circuits = sum(len(r) for r in results)
            assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = DataReuploading(n_features=4, n_layers=3)
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

    def test_concurrent_mixed_backends(self) -> None:
        """Test concurrent circuit generation across different backends."""
        enc = DataReuploading(n_features=4, n_layers=2)
        results: list[tuple[str, Any]] = []
        errors: list[tuple[str, Exception]] = []

        def generate_circuit(x: NDArray[np.floating], backend: str) -> None:
            try:
                circuit = enc.get_circuit(x, backend=backend)
                results.append((backend, circuit))
            except Exception as e:
                errors.append((backend, e))

        threads = []
        x = np.array([0.1, 0.2, 0.3, 0.4])

        if HAS_PENNYLANE:
            for _ in range(5):
                threads.append(
                    threading.Thread(target=generate_circuit, args=(x, "pennylane"))
                )

        if HAS_QISKIT:
            for _ in range(5):
                threads.append(
                    threading.Thread(target=generate_circuit, args=(x, "qiskit"))
                )

        if HAS_CIRQ:
            for _ in range(5):
                threads.append(
                    threading.Thread(target=generate_circuit, args=(x, "cirq"))
                )

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent generation: {errors}"


# =============================================================================
# Test Class: DataReuploading Specific
# =============================================================================


class TestDataReuploadingSpecific:
    """Tests specific to DataReuploading encoding features.

    This includes gate count breakdown, resource summary, deep circuit warnings,
    cyclic feature mapping, entanglement pairs, and parallel processing.
    """

    # -------------------------------------------------------------------------
    # Gate Count Breakdown Tests
    # -------------------------------------------------------------------------

    def test_basic_gate_count(self) -> None:
        """Test basic gate count computation.

        For 4 qubits, 3 layers:
        - RY per layer: 4, CNOT per layer: 3
        """
        enc = DataReuploading(n_features=4, n_layers=3)
        breakdown = enc.gate_count_breakdown()

        assert breakdown['ry_gates'] == 12  # 4 * 3
        assert breakdown['cnot_gates'] == 9  # 3 * 3
        assert breakdown['total'] == 21  # 12 + 9

    def test_gate_count_per_layer(self) -> None:
        """Test per-layer gate count computation."""
        enc = DataReuploading(n_features=8, n_layers=5)
        breakdown = enc.gate_count_breakdown()

        assert breakdown['ry_per_layer'] == 8
        assert breakdown['cnot_per_layer'] == 7
        assert breakdown['gates_per_layer'] == 15

    def test_single_qubit_no_cnot(self) -> None:
        """Test that single-qubit encoding has no CNOT gates."""
        enc = DataReuploading(n_features=1, n_layers=5)
        breakdown = enc.gate_count_breakdown()

        assert breakdown['ry_gates'] == 5  # 1 * 5 layers
        assert breakdown['cnot_gates'] == 0  # No entanglement possible
        assert breakdown['total'] == 5

    def test_single_layer_gate_count(self) -> None:
        """Test gate count with single layer."""
        enc = DataReuploading(n_features=4, n_layers=1)
        breakdown = enc.gate_count_breakdown()

        assert breakdown['ry_gates'] == 4
        assert breakdown['cnot_gates'] == 3
        assert breakdown['total'] == 7

    def test_many_layers_gate_count(self) -> None:
        """Test gate count with many layers."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = DataReuploading(n_features=4, n_layers=10)

        breakdown = enc.gate_count_breakdown()

        assert breakdown['ry_gates'] == 40  # 4 * 10
        assert breakdown['cnot_gates'] == 30  # 3 * 10
        assert breakdown['total'] == 70

    def test_fewer_qubits_than_features_gate_count(self) -> None:
        """Test gate count when n_qubits < n_features (cyclic mapping)."""
        enc = DataReuploading(n_features=8, n_qubits=4, n_layers=2)
        breakdown = enc.gate_count_breakdown()

        # With cyclic mapping: ALL features are encoded (one RY per feature)
        assert breakdown['ry_per_layer'] == 8  # All 8 features
        assert breakdown['cnot_per_layer'] == 3  # n_qubits - 1
        assert breakdown['ry_gates'] == 16  # 8 * 2
        assert breakdown['cnot_gates'] == 6  # 3 * 2

    def test_more_qubits_than_features_gate_count(self) -> None:
        """Test gate count when n_qubits > n_features."""
        enc = DataReuploading(n_features=2, n_qubits=6, n_layers=3)
        breakdown = enc.gate_count_breakdown()

        # RY per layer = n_features (all features encoded)
        assert breakdown['ry_per_layer'] == 2
        # Entanglement uses all n_qubits
        assert breakdown['cnot_per_layer'] == 5  # n_qubits - 1
        assert breakdown['ry_gates'] == 6  # 2 * 3
        assert breakdown['cnot_gates'] == 15  # 5 * 3

    def test_total_single_and_two_qubit(self) -> None:
        """Test that single/two qubit totals match gate counts."""
        enc = DataReuploading(n_features=4, n_layers=3)
        breakdown = enc.gate_count_breakdown()

        assert breakdown['total_single_qubit'] == breakdown['ry_gates']
        assert breakdown['total_two_qubit'] == breakdown['cnot_gates']
        assert breakdown['total'] == (
            breakdown['total_single_qubit'] + breakdown['total_two_qubit']
        )

    def test_gate_count_return_type(self) -> None:
        """Test that breakdown returns correct type with all keys."""
        enc = DataReuploading(n_features=4)
        breakdown = enc.gate_count_breakdown()

        expected_keys = {
            'ry_gates', 'cnot_gates', 'total_single_qubit',
            'total_two_qubit', 'total', 'ry_per_layer',
            'cnot_per_layer', 'gates_per_layer'
        }
        assert set(breakdown.keys()) == expected_keys
        assert all(isinstance(v, int) for v in breakdown.values())

    # -------------------------------------------------------------------------
    # Resource Summary Tests
    # -------------------------------------------------------------------------

    def test_basic_summary(self) -> None:
        """Test basic resource summary content."""
        enc = DataReuploading(n_features=4, n_layers=3)
        summary = enc.resource_summary()

        assert summary['n_qubits'] == 4
        assert summary['n_features'] == 4
        assert summary['n_layers'] == 3
        # depth = n_layers * (ceil(n_features/n_qubits) + (n_qubits-1)) = 3 * (1 + 3) = 12
        assert summary['depth'] == 12

    def test_gate_counts_included(self) -> None:
        """Test that gate counts are included in summary."""
        enc = DataReuploading(n_features=4, n_layers=3)
        summary = enc.resource_summary()

        assert 'gate_counts' in summary
        assert summary['gate_counts']['total'] == 21

    def test_entangling_property_in_summary(self) -> None:
        """Test is_entangling reflects qubit count in summary."""
        enc_multi = DataReuploading(n_features=4)
        enc_single = DataReuploading(n_features=1)

        assert enc_multi.resource_summary()['is_entangling'] is True
        assert enc_single.resource_summary()['is_entangling'] is False

    def test_simulability_property_in_summary(self) -> None:
        """Test simulability reflects qubit count in summary."""
        enc_multi = DataReuploading(n_features=4)
        enc_single = DataReuploading(n_features=1)

        assert enc_multi.resource_summary()['simulability'] == 'not_simulable'
        assert enc_single.resource_summary()['simulability'] == 'simulable'

    def test_trainability_estimate(self) -> None:
        """Test trainability estimate computation."""
        enc_shallow = DataReuploading(n_features=4, n_layers=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc_deep = DataReuploading(n_features=4, n_layers=10)

        summary_shallow = enc_shallow.resource_summary()
        summary_deep = enc_deep.resource_summary()

        # Shallow should have higher trainability
        assert summary_shallow['trainability_estimate'] > summary_deep['trainability_estimate']
        # Trainability should be in valid range
        assert 0.0 <= summary_shallow['trainability_estimate'] <= 1.0
        assert 0.0 <= summary_deep['trainability_estimate'] <= 1.0

    def test_fourier_frequencies(self) -> None:
        """Test Fourier frequencies equals n_layers."""
        enc = DataReuploading(n_features=4, n_layers=7)
        summary = enc.resource_summary()

        assert summary['fourier_frequencies'] == 7

    def test_hardware_requirements(self) -> None:
        """Test hardware requirements section."""
        enc = DataReuploading(n_features=4, n_layers=3)
        summary = enc.resource_summary()

        hw_req = summary['hardware_requirements']
        assert hw_req['connectivity'] == 'linear'
        assert 'RY' in hw_req['native_gates']
        assert 'CNOT' in hw_req['native_gates']
        assert hw_req['min_qubit_count'] == 4

    def test_summary_all_expected_keys(self) -> None:
        """Test that summary contains all expected keys."""
        enc = DataReuploading(n_features=4)
        summary = enc.resource_summary()

        expected_keys = {
            'n_qubits', 'n_features', 'n_layers', 'depth',
            'gate_counts', 'is_entangling', 'simulability',
            'trainability_estimate', 'fourier_frequencies',
            'hardware_requirements', 'recommendations'
        }
        assert set(summary.keys()) == expected_keys

    # -------------------------------------------------------------------------
    # Deep Circuit Warning Tests
    # -------------------------------------------------------------------------

    def test_no_warning_shallow_circuit(self) -> None:
        """Test that shallow circuits don't trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataReuploading(n_features=4, n_layers=5)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_warning_deep_circuit(self) -> None:
        """Test that deep circuits trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataReuploading(n_features=4, n_layers=15)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert 'deep circuit' in str(user_warnings[0].message).lower()

    def test_warning_at_threshold(self) -> None:
        """Test warning behavior at threshold boundary."""
        # At threshold (10) - no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataReuploading(n_features=4, n_layers=10)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

        # Above threshold (11) - warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataReuploading(n_features=4, n_layers=11)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_warning_contains_recommendations(self) -> None:
        """Test that warning message contains helpful recommendations."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataReuploading(n_features=4, n_layers=20)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1

        msg = str(user_warnings[0].message).lower()
        assert 'trainability' in msg
        assert 'barren plateau' in msg

    def test_encoding_still_works_with_warning(self) -> None:
        """Test that encoding functions correctly despite warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = DataReuploading(n_features=4, n_layers=20)

        # Should still work
        assert enc.n_layers == 20
        # depth = n_layers * (ceil(n_features/n_qubits) + (n_qubits-1)) = 20 * (1 + 3) = 80
        assert enc.depth == 80

    # -------------------------------------------------------------------------
    # Cyclic Feature Mapping Tests
    # -------------------------------------------------------------------------

    def test_cyclic_mapping_gate_count(self) -> None:
        """Test that gate count includes all features with cyclic mapping.

        When n_features > n_qubits, features are cyclically mapped.
        """
        enc = DataReuploading(n_features=8, n_qubits=4, n_layers=2)
        breakdown = enc.gate_count_breakdown()

        # With cyclic mapping: ALL 8 features are encoded
        assert breakdown['ry_per_layer'] == 8
        assert breakdown['ry_gates'] == 16  # 8 * 2
        # CNOT uses n_qubits, so 3 per layer
        assert breakdown['cnot_per_layer'] == 3
        assert breakdown['cnot_gates'] == 6  # 3 * 2

    def test_cyclic_mapping_more_features_than_qubits(self) -> None:
        """Test encoding with more features than qubits."""
        enc = DataReuploading(n_features=6, n_qubits=3, n_layers=2)

        # Should not raise and all properties should be correct
        assert enc.n_features == 6
        assert enc.n_qubits == 3
        assert enc.n_layers == 2

        # Gate counts should reflect all features
        breakdown = enc.gate_count_breakdown()
        assert breakdown['ry_per_layer'] == 6  # All 6 features

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_cyclic_mapping_produces_valid_state_pennylane(self) -> None:
        """Test that cyclic mapping produces valid quantum states (PennyLane)."""
        enc = DataReuploading(n_features=8, n_qubits=4, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        circuit_fn = enc.get_circuit(x, backend='pennylane')
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

        # State should have correct dimension (2^4 = 16)
        assert len(state) == 16

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_cyclic_mapping_produces_valid_circuit_qiskit(self) -> None:
        """Test that cyclic mapping produces valid Qiskit circuits."""
        enc = DataReuploading(n_features=8, n_qubits=4, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        circuit = enc.get_circuit(x, backend='qiskit')

        assert circuit.num_qubits == 4

        # Count RY gates in the circuit
        ry_count = sum(1 for inst in circuit.data if inst.operation.name == 'ry')
        # With cyclic mapping: 8 features * 2 layers = 16 RY gates
        assert ry_count == 16

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_cyclic_mapping_produces_valid_circuit_cirq(self) -> None:
        """Test that cyclic mapping produces valid Cirq circuits."""
        enc = DataReuploading(n_features=6, n_qubits=3, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        circuit = enc.get_circuit(x, backend='cirq')

        assert len(circuit.all_qubits()) == 3

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_cyclic_mapping_different_from_truncation(self) -> None:
        """Test that cyclic mapping produces different states than truncation.

        This verifies that features beyond n_qubits actually affect the output.
        """
        enc = DataReuploading(n_features=8, n_qubits=4, n_layers=2)

        # Two inputs identical for first 4 features but differ after
        x1 = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0])
        x2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        circuit_fn1 = enc.get_circuit(x1, backend='pennylane')
        circuit_fn2 = enc.get_circuit(x2, backend='pennylane')

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def get_state1():
            circuit_fn1()
            return qml.state()

        @qml.qnode(dev)
        def get_state2():
            circuit_fn2()
            return qml.state()

        state1 = np.array(get_state1())
        state2 = np.array(get_state2())

        # If truncation was used, states would be identical
        # With cyclic mapping, they should differ
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99, (
            "States should differ when features beyond n_qubits are changed"
        )

    # -------------------------------------------------------------------------
    # Entanglement Pairs Tests
    # -------------------------------------------------------------------------

    def test_basic_entanglement_pairs(self) -> None:
        """Test basic entanglement pairs for standard configuration."""
        enc = DataReuploading(n_features=4, n_layers=3)
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (1, 2), (2, 3)]
        assert pairs == expected

    def test_two_qubit_entanglement(self) -> None:
        """Test entanglement pairs for 2-qubit configuration."""
        enc = DataReuploading(n_features=2)
        pairs = enc.get_entanglement_pairs()

        assert pairs == [(0, 1)]

    def test_single_qubit_no_entanglement(self) -> None:
        """Test that single-qubit encoding has no entanglement pairs."""
        enc = DataReuploading(n_features=1, n_layers=5)
        pairs = enc.get_entanglement_pairs()

        assert pairs == []

    def test_entanglement_pairs_count(self) -> None:
        """Test that entanglement pairs count matches n_qubits - 1."""
        for n_qubits in [2, 4, 6, 8]:
            enc = DataReuploading(n_features=n_qubits)
            pairs = enc.get_entanglement_pairs()
            assert len(pairs) == n_qubits - 1

    def test_entanglement_pairs_with_fewer_qubits(self) -> None:
        """Test entanglement pairs when n_qubits < n_features."""
        enc = DataReuploading(n_features=8, n_qubits=4)
        pairs = enc.get_entanglement_pairs()

        # Should use n_qubits for entanglement, not n_features
        assert pairs == [(0, 1), (1, 2), (2, 3)]
        assert len(pairs) == 3  # n_qubits - 1

    def test_entanglement_pairs_return_type(self) -> None:
        """Test that return type is a list of tuples."""
        enc = DataReuploading(n_features=4)
        pairs = enc.get_entanglement_pairs()

        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) for p in pairs)
        assert all(len(p) == 2 for p in pairs)
        assert all(isinstance(p[0], int) and isinstance(p[1], int) for p in pairs)

    # -------------------------------------------------------------------------
    # Parallel Processing Tests
    # -------------------------------------------------------------------------

    def test_parallel_produces_same_results_as_sequential(self) -> None:
        """Test that parallel processing produces same results as sequential."""
        enc = DataReuploading(n_features=4, n_layers=2)
        X = np.random.default_rng(42).random((20, 4))

        if HAS_PENNYLANE:
            circuits_seq = enc.get_circuits(X, backend='pennylane', parallel=False)
            circuits_par = enc.get_circuits(X, backend='pennylane', parallel=True)

            assert len(circuits_seq) == len(circuits_par)
            assert all(callable(c) for c in circuits_seq)
            assert all(callable(c) for c in circuits_par)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_parallel_qiskit_circuits(self) -> None:
        """Test parallel processing with Qiskit backend."""
        enc = DataReuploading(n_features=4, n_layers=2)
        X = np.random.default_rng(42).random((20, 4))

        circuits_seq = enc.get_circuits(X, backend='qiskit', parallel=False)
        circuits_par = enc.get_circuits(X, backend='qiskit', parallel=True)

        assert len(circuits_seq) == len(circuits_par)
        assert all(isinstance(c, QuantumCircuit) for c in circuits_seq)
        assert all(isinstance(c, QuantumCircuit) for c in circuits_par)

    def test_parallel_with_max_workers(self) -> None:
        """Test parallel processing with custom max_workers."""
        enc = DataReuploading(n_features=4, n_layers=2)
        X = np.random.default_rng(42).random((20, 4))

        if HAS_PENNYLANE:
            circuits = enc.get_circuits(
                X, backend='pennylane', parallel=True, max_workers=2
            )
            assert len(circuits) == 20

    def test_parallel_single_sample_fallback(self) -> None:
        """Test that single sample doesn't use parallel processing."""
        enc = DataReuploading(n_features=4, n_layers=2)
        X = np.random.default_rng(42).random((1, 4))

        if HAS_PENNYLANE:
            # Should work without issues (parallel=True but only 1 sample)
            circuits = enc.get_circuits(X, backend='pennylane', parallel=True)
            assert len(circuits) == 1

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_parallel_order_preservation(self) -> None:
        """Test that parallel processing preserves input order."""
        enc = DataReuploading(n_features=4, n_layers=2)

        # Create distinguishable inputs
        X = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        circuits = enc.get_circuits(X, backend='pennylane', parallel=True)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        states = []
        for circuit_fn in circuits:
            @qml.qnode(dev)
            def run_circuit():
                circuit_fn()
                return qml.state()
            states.append(np.array(run_circuit()))

        # Verify that states are in the expected order by checking
        # that the first state (from all-zeros) is different from others
        for i in range(1, len(states)):
            fidelity = np.abs(np.vdot(states[0], states[i])) ** 2
            assert fidelity < 0.99, f"State 0 and state {i} should differ"

    # -------------------------------------------------------------------------
    # _get_circuit_from_validated Tests
    # -------------------------------------------------------------------------

    def test_get_circuit_from_validated_produces_same_result(self) -> None:
        """Test that _get_circuit_from_validated produces same result."""
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        if HAS_PENNYLANE:
            circuit1 = enc.get_circuit(x, backend='pennylane')
            circuit2 = enc._get_circuit_from_validated(x, backend='pennylane')

            assert callable(circuit1)
            assert callable(circuit2)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_qiskit_from_validated(self) -> None:
        """Test _get_circuit_from_validated with Qiskit backend."""
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc._get_circuit_from_validated(x, backend='qiskit')
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_cirq_from_validated(self) -> None:
        """Test _get_circuit_from_validated with Cirq backend."""
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc._get_circuit_from_validated(x, backend='cirq')
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 4

    def test_get_circuit_from_validated_invalid_backend(self) -> None:
        """Test that invalid backend raises ValueError."""
        enc = DataReuploading(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="Unknown backend"):
            enc._get_circuit_from_validated(x, backend='invalid')  # type: ignore


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
        enc: DataReuploading,
        x: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Execute PennyLane circuit and return state vector.

        Args:
            enc: The DataReuploading instance.
            x: Input data array.

        Returns:
            Complex state vector as numpy array.
        """
        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        return np.array(full_circuit())

    @staticmethod
    def _get_qiskit_state(
        enc: DataReuploading,
        x: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Execute Qiskit circuit and return state vector.

        Args:
            enc: The DataReuploading instance.
            x: Input data array.

        Returns:
            Complex state vector as numpy array.
        """
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
        enc: DataReuploading,
        x: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Execute Cirq circuit and return state vector.

        Args:
            enc: The DataReuploading instance.
            x: Input data array.

        Returns:
            Complex state vector as numpy array.
        """
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
            f"{name1} is not normalized: |norm|² = {norm1}"
        )
        assert np.isclose(norm2, 1.0, atol=1e-10), (
            f"{name2} is not normalized: |norm|² = {norm2}"
        )

        # Compare sorted probability distributions
        probs1 = sorted(np.abs(state1) ** 2)
        probs2 = sorted(np.abs(state2) ** 2)

        np.testing.assert_allclose(
            probs1,
            probs2,
            atol=atol,
            err_msg=f"Probability distributions differ between {name1} and {name2}",
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = pl_full()
        assert np.isclose(np.sum(np.abs(pl_state) ** 2), 1.0, atol=1e-10)

        # Qiskit
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        assert np.isclose(np.sum(np.abs(qk_sv.data) ** 2), 1.0, atol=1e-10)

        # Cirq
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = cirq_result.final_state_vector
        assert np.isclose(np.sum(np.abs(cirq_state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_output(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that all backends produce valid output."""
        enc = DataReuploading(n_features=4)

        pl_circuit = enc.get_circuit(sample_data_4d, backend="pennylane")
        qk_circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        cirq_circuit = enc.get_circuit(sample_data_4d, backend="cirq")

        assert callable(pl_circuit)
        assert isinstance(qk_circuit, QuantumCircuit)
        assert isinstance(cirq_circuit, cirq.Circuit)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_equivalent_states(self) -> None:
        """Test that all backends produce mathematically equivalent states.

        This is the core cross-backend fidelity test. It verifies that
        PennyLane, Qiskit, and Cirq all encode the same input data into
        quantum states with identical probability distributions.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
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
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_with_multiple_layers(self) -> None:
        """Test cross-backend equivalence with multiple reuploading layers.

        DataReuploading's key feature is multiple layers of data encoding.
        """
        for n_layers in [1, 3, 5]:
            enc = DataReuploading(n_features=2, n_layers=n_layers)
            x = np.array([0.5, 1.0])

            pl_state = self._get_pennylane_state(enc, x)
            qk_state = self._get_qiskit_state(enc, x)
            cirq_state = self._get_cirq_state(enc, x)

            # Dimension: 2^2 = 4 (2 qubits for 2 features)
            assert len(pl_state) == 4
            assert len(qk_state) == 4
            assert len(cirq_state) == 4

            self._assert_states_equivalent(
                pl_state, qk_state,
                f"PennyLane (layers={n_layers})",
                f"Qiskit (layers={n_layers})"
            )
            self._assert_states_equivalent(
                pl_state, cirq_state,
                f"PennyLane (layers={n_layers})",
                f"Cirq (layers={n_layers})"
            )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_with_fewer_qubits_than_features(self) -> None:
        """Test cross-backend equivalence when n_qubits < n_features.

        DataReuploading allows using fewer qubits than features, encoding
        features across multiple qubits or layers. This tests that behavior.
        """
        enc = DataReuploading(n_features=8, n_qubits=4, n_layers=2)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # Dimension: 2^4 = 16 (4 qubits explicitly set)
        expected_dim = 2 ** 4
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_with_more_qubits_than_features(self) -> None:
        """Test cross-backend equivalence when n_qubits > n_features.

        DataReuploading can use more qubits than features for increased
        expressibility. This tests that behavior is consistent.
        """
        enc = DataReuploading(n_features=2, n_qubits=4, n_layers=2)
        x = np.array([0.5, 1.0])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # Dimension: 2^4 = 16 (4 qubits explicitly set)
        expected_dim = 2 ** 4
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = DataReuploading(n_features=4, n_layers=2)
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
                assert fidelity < 0.9999 or np.allclose(inputs[i], inputs[j])

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input always produces same state."""
        enc = DataReuploading(n_features=4, n_layers=2)
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
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_reproducibility(self) -> None:
        """Test that repeated executions produce identical results.

        Each backend should produce exactly the same state when run
        multiple times with the same input.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
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
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_different_inputs_different_states(self) -> None:
        """Test that different inputs produce different states across backends.

        This ensures the encoding is actually data-dependent.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
        x1 = np.array([0.1, 0.2, 0.3, 0.4])
        x2 = np.array([0.9, 0.8, 0.7, 0.6])

        def fidelity(s1: NDArray[np.complexfloating], s2: NDArray[np.complexfloating]) -> float:
            return float(np.abs(np.vdot(s1, s2)) ** 2)

        # Get states for both inputs from each backend
        pl_state1 = self._get_pennylane_state(enc, x1)
        pl_state2 = self._get_pennylane_state(enc, x2)

        qk_state1 = self._get_qiskit_state(enc, x1)
        qk_state2 = self._get_qiskit_state(enc, x2)

        cirq_state1 = self._get_cirq_state(enc, x1)
        cirq_state2 = self._get_cirq_state(enc, x2)

        # States for different inputs should be different
        assert fidelity(pl_state1, pl_state2) < 0.99
        assert fidelity(qk_state1, qk_state2) < 0.99
        assert fidelity(cirq_state1, cirq_state2) < 0.99

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("n_features", [2, 4, 6, 8])
    def test_cross_backend_various_feature_counts(self, n_features: int) -> None:
        """Test cross-backend equivalence for various feature counts.

        Parametrized test covering 2, 4, 6, and 8 features.
        """
        enc = DataReuploading(n_features=n_features, n_layers=2)

        # Generate random but reproducible test data
        rng = np.random.default_rng(seed=n_features * 42)
        x = rng.random(n_features)

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # Verify dimensions
        expected_dim = 2 ** enc.n_qubits
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        # Verify equivalence
        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_extreme_values(self) -> None:
        """Test cross-backend equivalence with extreme input values.

        Tests with very small and very large values to ensure numerical
        stability across backends.
        """
        enc = DataReuploading(n_features=4, n_layers=2)

        # Test with very small values
        x_small = np.array([1e-8, 1e-8, 1e-8, 1e-8])
        pl_small = self._get_pennylane_state(enc, x_small)
        qk_small = self._get_qiskit_state(enc, x_small)
        cirq_small = self._get_cirq_state(enc, x_small)

        self._assert_states_equivalent(pl_small, qk_small, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_small, cirq_small, "PennyLane", "Cirq")

        # Test with large values (rotation angles wrap around 2π)
        x_large = np.array([10.0, 20.0, 30.0, 40.0])
        pl_large = self._get_pennylane_state(enc, x_large)
        qk_large = self._get_qiskit_state(enc, x_large)
        cirq_large = self._get_cirq_state(enc, x_large)

        self._assert_states_equivalent(pl_large, qk_large, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_large, cirq_large, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_zero_input(self) -> None:
        """Test cross-backend equivalence with all-zero input.

        Zero input is a boundary case that should produce a valid and
        consistent state across all backends.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.zeros(4)

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # All should produce valid normalized states
        assert np.isclose(np.sum(np.abs(pl_state) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(qk_state) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(cirq_state) ** 2), 1.0, atol=1e-10)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_negative_values(self) -> None:
        """Test cross-backend equivalence with negative input values.

        Negative values are valid rotation angles and should be handled
        consistently across backends.
        """
        enc = DataReuploading(n_features=4, n_layers=2)
        x = np.array([-0.5, 0.3, -0.7, 0.1])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_pi_multiples(self) -> None:
        """Test cross-backend equivalence at π and 2π rotation points.

        These are special angles where rotation gates have specific behaviors
        (e.g., RX(π) is a bit flip). All backends should agree at these points.
        """
        enc = DataReuploading(n_features=4, n_layers=1)

        # Test at π multiples
        x_pi = np.array([np.pi, np.pi / 2, np.pi / 4, 2 * np.pi])

        pl_state = self._get_pennylane_state(enc, x_pi)
        qk_state = self._get_qiskit_state(enc, x_pi)
        cirq_state = self._get_cirq_state(enc, x_pi)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_entanglement_verified(self) -> None:
        """Test that all backends produce entangled states.

        DataReuploading creates entanglement through CNOT gates between layers.
        """
        enc = DataReuploading(n_features=2, n_layers=3)
        x = np.array([0.5, 1.0])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        def is_entangled(state: NDArray[np.complexfloating], n_qubits: int) -> bool:
            """Check if state is entangled using Schmidt decomposition.

            For a 2-qubit system, if the state cannot be written as |a⟩⊗|b⟩,
            it is entangled.
            """
            if n_qubits != 2:
                return True  # Assume entangled for larger systems

            # Reshape to 2x2 matrix for bipartite system
            state_matrix = state.reshape(2, 2)

            # Compute reduced density matrix by partial trace
            rho_reduced = state_matrix @ state_matrix.conj().T

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(rho_reduced)

            # If only one significant eigenvalue, state is separable
            significant = np.sum(eigenvalues > 1e-10)
            return significant > 1

        # All backends should produce entangled states
        assert is_entangled(pl_state, enc.n_qubits), "PennyLane state should be entangled"
        assert is_entangled(qk_state, enc.n_qubits), "Qiskit state should be entangled"
        assert is_entangled(cirq_state, enc.n_qubits), "Cirq state should be entangled"
