"""Comprehensive tests for AngleEncoding.

This test module provides complete coverage of the AngleEncoding class,
which encodes classical data into quantum states using single-qubit rotations.
It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, rotation type)
- Rotation axis behavior (X, Y, Z)
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

Run with: pytest tests/unit/encodings/test_angle.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_angle.py -v -m "not slow"

References
----------
.. [1] Schuld, M., & Petruccione, F. (2018). Supervised Learning with
       Quantum Computers. Springer.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest

from encoding_atlas import AngleEncoding
from encoding_atlas.core.properties import EncodingProperties

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


# =============================================================================
# Backend Availability Checks
# =============================================================================

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except (ImportError, AttributeError):
    # AttributeError: autoray compatibility issue on Python 3.9
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
    return np.array([0.5, 0.5])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (mid-range values)
    - [0.0, 0.0, 0.0, 0.0] (edge case: zeros)
    """
    return np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture
def default_encoding() -> AngleEncoding:
    """Default AngleEncoding with 4 features and Y rotation."""
    return AngleEncoding(n_features=4)


@pytest.fixture
def encoding_x() -> AngleEncoding:
    """AngleEncoding with X rotation."""
    return AngleEncoding(n_features=4, rotation="X")


@pytest.fixture
def encoding_z() -> AngleEncoding:
    """AngleEncoding with Z rotation."""
    return AngleEncoding(n_features=4, rotation="Z")


@pytest.fixture
def encoding_multi_reps() -> AngleEncoding:
    """AngleEncoding with multiple repetitions."""
    return AngleEncoding(n_features=4, reps=3)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for AngleEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test instantiation with default parameters."""
        enc = AngleEncoding(n_features=4)

        assert enc.n_features == 4
        assert enc.rotation == "Y"
        assert enc.reps == 1

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = AngleEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_custom_rotation_x(self) -> None:
        """Test instantiation with X rotation."""
        enc = AngleEncoding(n_features=4, rotation="X")
        assert enc.rotation == "X"

    def test_custom_rotation_y(self) -> None:
        """Test instantiation with Y rotation (explicit)."""
        enc = AngleEncoding(n_features=4, rotation="Y")
        assert enc.rotation == "Y"

    def test_custom_rotation_z(self) -> None:
        """Test instantiation with Z rotation."""
        enc = AngleEncoding(n_features=4, rotation="Z")
        assert enc.rotation == "Z"

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = AngleEncoding(n_features=4, reps=5)
        assert enc.reps == 5

    def test_config_contains_rotation(self) -> None:
        """Test that config dictionary contains rotation parameter."""
        enc = AngleEncoding(n_features=4, rotation="X")
        assert enc.config["rotation"] == "X"

    def test_config_contains_reps(self) -> None:
        """Test that config dictionary contains reps parameter."""
        enc = AngleEncoding(n_features=4, reps=3)
        assert enc.config["reps"] == 3

    def test_all_custom_parameters(self) -> None:
        """Test all parameters customized together."""
        enc = AngleEncoding(n_features=8, rotation="Z", reps=4)

        assert enc.n_features == 8
        assert enc.rotation == "Z"
        assert enc.reps == 4

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotation_variants(self, rotation: str) -> None:
        """Test all rotation axis variants."""
        enc = AngleEncoding(n_features=4, rotation=rotation)
        assert enc.config["rotation"] == rotation

    @pytest.mark.parametrize("n_features", [1, 2, 4, 8, 16])
    def test_various_feature_counts(self, n_features: int) -> None:
        """Test instantiation with various feature counts."""
        enc = AngleEncoding(n_features=n_features)
        assert enc.n_features == n_features

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = AngleEncoding(n_features=100)
        assert enc.n_features == 100
        # Should be fast since no circuit is pre-computed


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    def test_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises ValueError."""
        with pytest.raises(ValueError):
            AngleEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError):
            AngleEncoding(n_features=-1)

    def test_invalid_n_features_float(self) -> None:
        """Test that float n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            AngleEncoding(n_features=4.5)  # type: ignore

    def test_invalid_n_features_string(self) -> None:
        """Test that string n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            AngleEncoding(n_features="four")  # type: ignore

    def test_invalid_n_features_none(self) -> None:
        """Test that None n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            AngleEncoding(n_features=None)  # type: ignore

    def test_invalid_rotation(self) -> None:
        """Test that invalid rotation raises error."""
        with pytest.raises(ValueError, match="rotation must be"):
            AngleEncoding(n_features=4, rotation="W")

    def test_invalid_rotation_lowercase(self) -> None:
        """Test that lowercase rotation raises error (case-sensitive)."""
        with pytest.raises(ValueError, match="rotation must be"):
            AngleEncoding(n_features=4, rotation="y")

    def test_invalid_rotation_empty(self) -> None:
        """Test that empty rotation raises error."""
        with pytest.raises(ValueError, match="rotation must be"):
            AngleEncoding(n_features=4, rotation="")

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises error."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            AngleEncoding(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises error."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            AngleEncoding(n_features=4, reps=-1)

    def test_invalid_reps_type_float(self) -> None:
        """Test that float reps raises error."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            AngleEncoding(n_features=4, reps=2.5)  # type: ignore

    def test_invalid_reps_type_bool_true(self) -> None:
        """Test that True as reps raises error (bool is subclass of int)."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            AngleEncoding(n_features=4, reps=True)  # type: ignore

    def test_invalid_reps_type_bool_false(self) -> None:
        """Test that False as reps raises error."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            AngleEncoding(n_features=4, reps=False)  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of AngleEncoding."""

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features (one qubit per feature)."""
        for n in [1, 2, 4, 8, 16]:
            enc = AngleEncoding(n_features=n)
            assert enc.n_qubits == n

    def test_depth(self) -> None:
        """Test circuit depth calculation.

        Depth formula: depth = reps (one rotation per layer)
        For n_features=4, reps=2: depth = 2
        """
        enc = AngleEncoding(n_features=4, reps=2)
        assert enc.depth == 2

    def test_depth_equals_reps(self) -> None:
        """Test that depth equals number of repetitions."""
        for reps in [1, 2, 3, 5]:
            enc = AngleEncoding(n_features=4, reps=reps)
            assert enc.depth == reps

    def test_depth_single_layer(self) -> None:
        """Test that single-layer angle encoding has depth 1."""
        enc = AngleEncoding(n_features=4, reps=1)
        assert enc.depth == 1

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = AngleEncoding(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = AngleEncoding(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_properties_gate_count(self) -> None:
        """Test gate count calculation.

        Gate count = n_features * reps (one rotation gate per feature per rep)
        For n_features=4, reps=1: gate_count = 4
        """
        enc = AngleEncoding(n_features=4)
        props = enc.properties
        # gate_count = n_features * reps = 4 * 1 = 4
        assert props.gate_count == 4
        assert props.single_qubit_gates == 4
        assert props.two_qubit_gates == 0
        assert props.parameter_count == 4

    def test_properties_with_multiple_reps(self) -> None:
        """Test properties with multiple repetitions.

        With 3 reps and 4 features: 4 * 3 = 12 gates
        """
        enc = AngleEncoding(n_features=4, reps=3)
        props = enc.properties

        assert props.gate_count == 12
        assert props.single_qubit_gates == 12
        assert props.two_qubit_gates == 0
        assert props.parameter_count == 12
        assert props.depth == 3

    def test_not_entangling(self) -> None:
        """Test that pure angle encoding creates no entanglement."""
        enc = AngleEncoding(n_features=4)
        assert enc.properties.is_entangling is False

    def test_classically_simulable(self) -> None:
        """Test that product states are classically simulable."""
        enc = AngleEncoding(n_features=4)
        assert enc.properties.simulability == "simulable"

    def test_gate_count_consistency(self, default_encoding: AngleEncoding) -> None:
        """Test that single + two qubit gates equals total."""
        props = default_encoding.properties
        assert props.single_qubit_gates + props.two_qubit_gates == props.gate_count

    def test_properties_notes_mention_product_states(self) -> None:
        """Test that notes mention product states."""
        enc = AngleEncoding(n_features=4)
        props = enc.properties
        assert "product" in props.notes.lower() or "entanglement" in props.notes.lower()

    def test_trainability_estimate(self) -> None:
        """Test that trainability estimate is reasonable.

        Angle encoding should have high trainability (no barren plateaus).
        """
        enc = AngleEncoding(n_features=4)
        props = enc.properties
        assert props.trainability_estimate >= 0.5


# =============================================================================
# Test Class: Rotation Behavior (AngleEncoding-specific)
# =============================================================================


class TestRotationBehavior:
    """Tests for rotation axis-specific behavior unique to AngleEncoding."""

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_rotation_x_uses_rx_gates(self) -> None:
        """Test that X rotation uses RX gates in generated circuit."""
        enc = AngleEncoding(n_features=4, rotation="X")
        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="qiskit")

        for inst in circuit.data:
            assert inst.operation.name == "rx"

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_rotation_y_uses_ry_gates(self) -> None:
        """Test that Y rotation uses RY gates in generated circuit."""
        enc = AngleEncoding(n_features=4, rotation="Y")
        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="qiskit")

        for inst in circuit.data:
            assert inst.operation.name == "ry"

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_rotation_z_uses_rz_gates(self) -> None:
        """Test that Z rotation uses RZ gates in generated circuit."""
        enc = AngleEncoding(n_features=4, rotation="Z")
        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="qiskit")

        for inst in circuit.data:
            assert inst.operation.name == "rz"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations_produce_valid_states(self, rotation: str) -> None:
        """Test that all rotation types produce valid normalized states."""
        enc = AngleEncoding(n_features=4, rotation=rotation)
        x = np.array([0.3, 0.5, 0.7, 0.9])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert circuit is not None

    def test_valid_2d_input_single_sample(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 2D input with single sample is accepted."""
        x = sample_data_4d.reshape(1, -1)
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_wrong_feature_count(self, default_encoding: AngleEncoding) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2, 0.3])  # 3 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding.get_circuit(x, backend="qiskit")

    def test_nan_input_rejected(self, default_encoding: AngleEncoding) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding.get_circuit(x, backend="qiskit")

    def test_inf_input_rejected(self, default_encoding: AngleEncoding) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.get_circuit(x, backend="qiskit")

    def test_negative_inf_input_rejected(self, default_encoding: AngleEncoding) -> None:
        """Test that negative infinite values are rejected."""
        x = np.array([0.1, -np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.get_circuit(x, backend="qiskit")

    def test_list_input_accepted(self, default_encoding: AngleEncoding) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]
        circuit = default_encoding.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_tuple_input_accepted(self, default_encoding: AngleEncoding) -> None:
        """Test that tuple input is converted to numpy array."""
        x = (0.1, 0.2, 0.3, 0.4)
        circuit = default_encoding.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_empty_input_rejected(self) -> None:
        """Test that empty input raises error."""
        enc = AngleEncoding(n_features=4)
        with pytest.raises(ValueError):
            enc.get_circuit(np.array([]))

    def test_batch_input_shape(
        self,
        default_encoding: AngleEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that batch input generates multiple circuits."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    def test_invalid_complex_array(self) -> None:
        """Test that complex numpy array is rejected."""
        enc = AngleEncoding(n_features=4)
        x = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
        with pytest.raises(TypeError, match="complex"):
            enc.get_circuit(x)

    def test_invalid_complex_list(self) -> None:
        """Test that list with complex values is rejected."""
        enc = AngleEncoding(n_features=4)
        x = [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]
        with pytest.raises(TypeError, match="complex"):
            enc.get_circuit(x)

    def test_invalid_complex_mixed(self) -> None:
        """Test that mixed real/complex values are rejected."""
        enc = AngleEncoding(n_features=4)
        x = np.array([1.0, 2.0, 3 + 1j, 4.0])
        with pytest.raises(TypeError, match="complex"):
            enc.get_circuit(x)

    def test_invalid_complex_batch(self) -> None:
        """Test that batch with complex values is rejected."""
        enc = AngleEncoding(n_features=4)
        X = np.array(
            [
                [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j],
                [5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j],
            ]
        )
        with pytest.raises(TypeError, match="complex"):
            enc.get_circuits(X)


# =============================================================================
# Test Class: PennyLane Backend
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.backend_pennylane
class TestPennyLaneBackend:
    """Tests for PennyLane circuit generation."""

    def test_circuit_is_callable(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that generated circuit executes correctly in QNode context."""
        circuit_fn = default_encoding.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=default_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None
        assert len(state) == 2**default_encoding.n_qubits

    def test_state_normalized(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that resulting state is normalized."""
        circuit_fn = default_encoding.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=default_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations_execute(
        self, rotation: str, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all rotation types execute successfully."""
        enc = AngleEncoding(n_features=4, rotation=rotation)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_multiple_reps(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test circuit with multiple repetitions."""
        enc = AngleEncoding(n_features=4, reps=5)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_batch_circuits(
        self,
        default_encoding: AngleEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    def test_single_feature(self) -> None:
        """Test PennyLane with single feature."""
        enc = AngleEncoding(n_features=1)
        x = np.array([0.5])
        circuit_fn = enc.get_circuit(x, backend="pennylane")

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert len(state) == 2


# =============================================================================
# Test Class: Qiskit Backend
# =============================================================================


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.mark.backend_qiskit
class TestQiskitBackend:
    """Tests for Qiskit circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == default_encoding.n_qubits

    def test_circuit_has_expected_gates(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit contains expected gates.

        Should have exactly n_features gates (one per qubit).
        """
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert len(circuit.data) == default_encoding.n_features

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations_generate_circuits(
        self, rotation: str, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all rotation types generate valid circuits."""
        enc = AngleEncoding(n_features=4, rotation=rotation)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_multiple_reps_gate_count(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that multiple reps multiply gate count.

        4 features * 3 reps = 12 gates.
        """
        enc = AngleEncoding(n_features=4, reps=3)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        assert len(circuit.data) == 12

    def test_batch_circuits(
        self,
        default_encoding: AngleEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    def test_single_feature(self) -> None:
        """Test Qiskit with single feature."""
        enc = AngleEncoding(n_features=1)
        x = np.array([0.5])
        circuit = enc.get_circuit(x, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 1


# =============================================================================
# Test Class: Cirq Backend
# =============================================================================


@pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
@pytest.mark.backend_cirq
class TestCirqBackend:
    """Tests for Cirq circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: AngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == default_encoding.n_qubits

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations_generate_circuits(
        self, rotation: str, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all rotation types generate valid circuits."""
        enc = AngleEncoding(n_features=4, rotation=rotation)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 4

    def test_operation_count_matches_features(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that operation count matches n_features."""
        enc = AngleEncoding(n_features=4, reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        ops = list(circuit.all_operations())
        assert len(ops) == 4

    def test_multiple_reps_operation_count(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that multiple reps multiply operation count.

        4 features * 3 reps = 12 operations.
        """
        enc = AngleEncoding(n_features=4, reps=3)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        ops = list(circuit.all_operations())
        assert len(ops) == 12

    def test_batch_circuits(
        self,
        default_encoding: AngleEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)

    def test_single_feature(self) -> None:
        """Test Cirq with single feature."""
        enc = AngleEncoding(n_features=1)
        x = np.array([0.5])
        circuit = enc.get_circuit(x, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 1


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of angle encoding."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zero_angle_produces_zero_state(self) -> None:
        """Test that zero input produces |0...0> state for Y rotation."""
        enc = AngleEncoding(n_features=2, rotation="Y")
        x = np.array([0.0, 0.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # |00> state should have amplitude 1 at index 0
        assert np.isclose(np.abs(state[0]), 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_pi_rotation_y_flips_qubit(self) -> None:
        """Test that pi rotation on Y axis flips qubit to |1>."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([np.pi])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # |1> state should have amplitude ~1 at index 1
        assert np.isclose(np.abs(state[1]), 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_pi_half_rotation_creates_superposition(self) -> None:
        """Test that pi/2 rotation creates equal superposition."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([np.pi / 2])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # Should be in superposition: |0> + |1>
        prob_0 = np.abs(state[0]) ** 2
        prob_1 = np.abs(state[1]) ** 2
        assert np.isclose(prob_0, 0.5, atol=1e-10)
        assert np.isclose(prob_1, 0.5, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_x_rotation_evolution(self) -> None:
        """Test X rotation produces correct evolution.

        RX(pi)|0> = -i|1>, so probability of |1> should be 1.
        """
        enc = AngleEncoding(n_features=1, rotation="X")
        x = np.array([np.pi])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.abs(state[1]) ** 2, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_z_rotation_phase_only(self) -> None:
        """Test Z rotation only adds phase to |0> state.

        RZ on |0> only adds global phase, stays in |0>.
        """
        enc = AngleEncoding(n_features=1, rotation="Z")
        x = np.array([np.pi])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.abs(state[0]) ** 2, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_product_state_structure(self) -> None:
        """Test that multiple qubits create product states.

        First qubit: pi/2 rotation (equal superposition)
        Second qubit: 0 rotation (stays |0>)
        Result: (|0> + |1>) / sqrt(2) tensor |0> = (|00> + |10>) / sqrt(2)
        """
        enc = AngleEncoding(n_features=2, rotation="Y")
        x = np.array([np.pi / 2, 0.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # PennyLane ordering: |00> is index 0, |10> is index 2
        prob_00 = np.abs(state[0]) ** 2
        prob_10 = np.abs(state[2]) ** 2

        assert np.isclose(prob_00, 0.5, atol=1e-10)
        assert np.isclose(prob_10, 0.5, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reps_apply_repeatedly(self) -> None:
        """Test that multiple reps apply rotation repeatedly.

        With 2 reps of pi/2, should get pi rotation total, flipping to |1>.
        """
        enc = AngleEncoding(n_features=1, rotation="Y", reps=2)
        x = np.array([np.pi / 2])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # 2 * RY(pi/2) = RY(pi), should flip to |1>
        assert np.isclose(np.abs(state[1]) ** 2, 1.0, atol=1e-10)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature(self) -> None:
        """Test encoding with single feature (minimum case)."""
        enc = AngleEncoding(n_features=1)
        x = np.array([0.5])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None
        assert enc.n_qubits == 1

    def test_large_feature_count(self) -> None:
        """Test encoding with many features."""
        enc = AngleEncoding(n_features=16)

        assert enc.n_qubits == 16
        x = np.random.randn(16)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_extreme_positive_values(self) -> None:
        """Test encoding with large positive input values."""
        enc = AngleEncoding(n_features=4)
        x_large = np.array([100.0, 200.0, 300.0, 400.0])
        circuit = enc.get_circuit(x_large, backend="pennylane")
        assert circuit is not None

    def test_extreme_negative_values(self) -> None:
        """Test encoding with large negative input values."""
        enc = AngleEncoding(n_features=4)
        x_neg = np.array([-100.0, -200.0, -300.0, -400.0])
        circuit = enc.get_circuit(x_neg, backend="pennylane")
        assert circuit is not None

    def test_mixed_sign_values(self) -> None:
        """Test encoding with mixed positive and negative values."""
        enc = AngleEncoding(n_features=4)
        x = np.array([-1.0, 2.0, -3.0, 4.0])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_very_small_values(self) -> None:
        """Test encoding with very small values."""
        enc = AngleEncoding(n_features=4)
        x_small = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        circuit = enc.get_circuit(x_small, backend="pennylane")
        assert circuit is not None

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions.

        With 50 reps: depth = 50, gate_count = 4 * 50 = 200.
        """
        enc = AngleEncoding(n_features=4, reps=50)

        assert enc.depth == 50
        props = enc.properties
        assert props.gate_count == 200

    def test_all_zeros_input(self) -> None:
        """Test encoding with all zeros."""
        enc = AngleEncoding(n_features=4)
        x = np.zeros(4)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_multiples_of_2pi(self) -> None:
        """Test encoding with multiples of 2*pi (should wrap around)."""
        enc = AngleEncoding(n_features=4)
        x = np.array([2 * np.pi, 4 * np.pi, 6 * np.pi, 8 * np.pi])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


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
        enc = AngleEncoding(n_features=4)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
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
        enc = AngleEncoding(n_features=4)
        x = np.array([1e10, 1e11, 1e12, 1e13])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values."""
        enc = AngleEncoding(n_features=4)
        x = np.array([1e-10, 1e10, 1e-5, 1e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_extreme_values_produce_normalized_state(self) -> None:
        """Test that extreme values still produce normalized quantum states."""
        enc = AngleEncoding(n_features=4)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([1e-100, 1e100, 0.5, 0.1])
        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_no_overflow_large_angles(self) -> None:
        """Test that large angles don't cause overflow."""
        enc = AngleEncoding(n_features=2)
        x = np.array([1e15, 1e15])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.all(np.isfinite(state))
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_precision_at_machine_epsilon(self) -> None:
        """Test encoding at machine epsilon scale."""
        enc = AngleEncoding(n_features=2)
        eps = np.finfo(float).eps
        x = np.array([eps, eps])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = AngleEncoding(n_features=4, rotation="Y", reps=2)
        enc2 = AngleEncoding(n_features=4, rotation="Y", reps=2)
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = AngleEncoding(n_features=4)
        enc2 = AngleEncoding(n_features=8)
        assert enc1 != enc2

    def test_equality_different_rotation(self) -> None:
        """Test that encodings with different rotation are not equal."""
        enc1 = AngleEncoding(n_features=4, rotation="X")
        enc2 = AngleEncoding(n_features=4, rotation="Y")
        assert enc1 != enc2

    def test_equality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = AngleEncoding(n_features=4, reps=1)
        enc2 = AngleEncoding(n_features=4, reps=2)
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = AngleEncoding(n_features=4, rotation="Z", reps=3)
        enc2 = AngleEncoding(n_features=4, rotation="Z", reps=3)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = AngleEncoding(n_features=4, rotation="Y")
        enc2 = AngleEncoding(n_features=4, rotation="Y")  # Same as enc1
        enc3 = AngleEncoding(n_features=4, rotation="X")

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = AngleEncoding(n_features=4)
        enc2 = AngleEncoding(n_features=4)

        d = {enc1: "four features"}
        assert d[enc2] == "four features"

    def test_inequality_with_none(self) -> None:
        """Test that encoding is not equal to None."""
        enc = AngleEncoding(n_features=4)
        assert enc != None  # noqa: E711

    def test_inequality_with_different_type(self) -> None:
        """Test that encoding is not equal to different types."""
        enc = AngleEncoding(n_features=4)
        assert enc != "AngleEncoding"
        assert enc != 4


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ string representation."""

    def test_repr_contains_class_name(self, default_encoding: AngleEncoding) -> None:
        """Test that repr contains class name."""
        assert "AngleEncoding" in repr(default_encoding)

    def test_repr_contains_n_features(self, default_encoding: AngleEncoding) -> None:
        """Test that repr contains n_features."""
        assert "n_features=4" in repr(default_encoding)

    def test_repr_contains_rotation(self) -> None:
        """Test that repr contains rotation parameter."""
        enc = AngleEncoding(n_features=4, rotation="X")
        r = repr(enc)
        assert "rotation" in r.lower() or "X" in r

    def test_repr_contains_reps(self) -> None:
        """Test that repr contains reps parameter."""
        enc = AngleEncoding(n_features=4, reps=5)
        r = repr(enc)
        assert "reps" in r.lower() or "5" in r

    def test_repr_all_custom_parameters(self) -> None:
        """Test that repr contains all custom parameters."""
        enc = AngleEncoding(n_features=8, rotation="Z", reps=3)
        r = repr(enc)
        assert "n_features=8" in r or "8" in r


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for invalid backend names and missing backends."""

    def test_invalid_backend(
        self, default_encoding: AngleEncoding, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_invalid_backend_empty(
        self, default_encoding: AngleEncoding, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that empty backend raises error."""
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
        enc = AngleEncoding(n_features=8, rotation="Z", reps=3)

        # Force properties to be computed
        _ = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.rotation == enc.rotation
        assert restored.reps == enc.reps
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = AngleEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = AngleEncoding(n_features=4, rotation="X")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit_before = enc.get_circuit(x, backend="pennylane")

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        circuit_after = restored.get_circuit(x, backend="pennylane")

        assert callable(circuit_before)
        assert callable(circuit_after)

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = AngleEncoding(n_features=4)
        original_props = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        restored_props = restored.properties

        assert isinstance(restored_props, EncodingProperties)
        assert restored_props.n_qubits == original_props.n_qubits
        assert restored_props.depth == original_props.depth
        assert restored_props.gate_count == original_props.gate_count


# =============================================================================
# Test Class: Concurrent Access / Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        enc = AngleEncoding(n_features=4)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for _i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
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
        total_circuits = sum(len(r) for r in results)
        assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = AngleEncoding(n_features=4)
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


# =============================================================================
# Test Class: Gate Count Breakdown (AngleEncoding-specific)
# =============================================================================


class TestGateCountBreakdown:
    """Tests for the gate_count_breakdown() method."""

    def test_returns_typed_dict(self) -> None:
        """Test that gate_count_breakdown returns correct structure."""
        enc = AngleEncoding(n_features=4)
        breakdown = enc.gate_count_breakdown()

        expected_keys = {
            "rx",
            "ry",
            "rz",
            "total_single_qubit",
            "total_two_qubit",
            "total",
        }
        assert set(breakdown.keys()) == expected_keys

    def test_y_rotation_gate_counts(self) -> None:
        """Test gate counts for Y rotation (default).

        With n_features=4, reps=1: 4 RY gates total.
        """
        enc = AngleEncoding(n_features=4, rotation="Y", reps=1)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["rx"] == 0
        assert breakdown["ry"] == 4
        assert breakdown["rz"] == 0
        assert breakdown["total_single_qubit"] == 4
        assert breakdown["total_two_qubit"] == 0
        assert breakdown["total"] == 4

    def test_x_rotation_gate_counts(self) -> None:
        """Test gate counts for X rotation."""
        enc = AngleEncoding(n_features=4, rotation="X", reps=1)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["rx"] == 4
        assert breakdown["ry"] == 0
        assert breakdown["rz"] == 0
        assert breakdown["total_single_qubit"] == 4
        assert breakdown["total_two_qubit"] == 0
        assert breakdown["total"] == 4

    def test_z_rotation_gate_counts(self) -> None:
        """Test gate counts for Z rotation."""
        enc = AngleEncoding(n_features=4, rotation="Z", reps=1)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["rx"] == 0
        assert breakdown["ry"] == 0
        assert breakdown["rz"] == 4
        assert breakdown["total_single_qubit"] == 4
        assert breakdown["total_two_qubit"] == 0
        assert breakdown["total"] == 4

    def test_multiple_reps_gate_counts(self) -> None:
        """Test gate counts with multiple repetitions.

        4 features * 3 reps = 12 RY gates.
        """
        enc = AngleEncoding(n_features=4, rotation="Y", reps=3)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["ry"] == 12
        assert breakdown["total_single_qubit"] == 12
        assert breakdown["total"] == 12

    def test_single_feature_gate_counts(self) -> None:
        """Test gate counts with single feature."""
        enc = AngleEncoding(n_features=1, rotation="Y", reps=1)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["ry"] == 1
        assert breakdown["total"] == 1

    @pytest.mark.parametrize("n_features", [1, 2, 4, 8, 16])
    def test_gate_counts_scale_linearly(self, n_features: int) -> None:
        """Test that gate counts scale linearly with n_features."""
        enc = AngleEncoding(n_features=n_features, rotation="Y", reps=1)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["total"] == n_features
        assert breakdown["total_single_qubit"] == n_features

    @pytest.mark.parametrize("reps", [1, 2, 3, 5])
    def test_gate_counts_scale_with_reps(self, reps: int) -> None:
        """Test that gate counts scale linearly with reps."""
        enc = AngleEncoding(n_features=4, rotation="Y", reps=reps)
        breakdown = enc.gate_count_breakdown()

        expected_total = 4 * reps
        assert breakdown["total"] == expected_total
        assert breakdown["ry"] == expected_total

    def test_two_qubit_gates_always_zero(self) -> None:
        """Test that two_qubit_gates is always 0 for angle encoding."""
        for rotation in ["X", "Y", "Z"]:
            for reps in [1, 2, 3]:
                enc = AngleEncoding(n_features=4, rotation=rotation, reps=reps)
                breakdown = enc.gate_count_breakdown()
                assert breakdown["total_two_qubit"] == 0

    def test_consistency_with_properties(self) -> None:
        """Test that breakdown total matches properties.gate_count."""
        enc = AngleEncoding(n_features=4, rotation="Y", reps=3)
        breakdown = enc.gate_count_breakdown()
        props = enc.properties

        assert breakdown["total"] == props.gate_count
        assert breakdown["total_single_qubit"] == props.single_qubit_gates
        assert breakdown["total_two_qubit"] == props.two_qubit_gates


# =============================================================================
# Test Class: Resource Summary (AngleEncoding-specific)
# =============================================================================


class TestResourceSummary:
    """Tests for the resource_summary() method."""

    def test_returns_dict(self) -> None:
        """Test that resource_summary returns a dictionary."""
        enc = AngleEncoding(n_features=4)
        summary = enc.resource_summary()
        assert isinstance(summary, dict)

    def test_circuit_structure_fields(self) -> None:
        """Test circuit structure fields in resource summary."""
        enc = AngleEncoding(n_features=4, rotation="Y", reps=2)
        summary = enc.resource_summary()

        assert summary["n_qubits"] == 4
        assert summary["n_features"] == 4
        assert summary["depth"] == 2
        assert summary["reps"] == 2
        assert summary["rotation"] == "Y"

    def test_gate_counts_included(self) -> None:
        """Test that gate counts are included in summary."""
        enc = AngleEncoding(n_features=4, rotation="Y", reps=1)
        summary = enc.resource_summary()

        assert "gate_counts" in summary
        assert summary["gate_counts"]["ry"] == 4
        assert summary["gate_counts"]["total"] == 4

    def test_encoding_characteristics(self) -> None:
        """Test encoding characteristics in summary."""
        enc = AngleEncoding(n_features=4)
        summary = enc.resource_summary()

        assert summary["is_entangling"] is False
        assert summary["simulability"] == "simulable"
        assert "trainability_estimate" in summary
        assert 0.0 <= summary["trainability_estimate"] <= 1.0

    def test_hardware_requirements(self) -> None:
        """Test hardware requirements in summary."""
        enc = AngleEncoding(n_features=4, rotation="Y")
        summary = enc.resource_summary()

        assert "hardware_requirements" in summary
        hw = summary["hardware_requirements"]

        assert hw["connectivity"] == "none"
        assert hw["native_gates"] == ["RY"]

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_native_gates_match_rotation(self, rotation: str) -> None:
        """Test that native_gates matches the rotation parameter."""
        enc = AngleEncoding(n_features=4, rotation=rotation)
        summary = enc.resource_summary()

        expected_gate = f"R{rotation}"
        assert summary["hardware_requirements"]["native_gates"] == [expected_gate]

    def test_consistency_with_properties(self) -> None:
        """Test consistency between summary and properties."""
        enc = AngleEncoding(n_features=4, rotation="Y", reps=2)
        summary = enc.resource_summary()
        props = enc.properties

        assert summary["n_qubits"] == props.n_qubits
        assert summary["depth"] == props.depth
        assert summary["is_entangling"] == props.is_entangling
        assert summary["simulability"] == props.simulability
        assert summary["trainability_estimate"] == props.trainability_estimate

    def test_consistency_with_gate_count_breakdown(self) -> None:
        """Test that summary gate_counts matches gate_count_breakdown."""
        enc = AngleEncoding(n_features=4, rotation="Y", reps=2)
        summary = enc.resource_summary()
        breakdown = enc.gate_count_breakdown()

        assert summary["gate_counts"] == breakdown

    @pytest.mark.parametrize("n_features,reps", [(1, 1), (4, 2), (8, 3)])
    def test_various_configurations(self, n_features: int, reps: int) -> None:
        """Test resource summary for various configurations."""
        enc = AngleEncoding(n_features=n_features, rotation="Y", reps=reps)
        summary = enc.resource_summary()

        assert summary["n_qubits"] == n_features
        assert summary["n_features"] == n_features
        assert summary["depth"] == reps
        assert summary["gate_counts"]["total"] == n_features * reps


# =============================================================================
# Test Class: Input Range Debug Logging (AngleEncoding-specific)
# =============================================================================


class TestInputRangeDebugLogging:
    """Tests for the debug logging for input value ranges."""

    def test_normal_values_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that normal input values don't trigger debug warning."""
        import logging

        enc = AngleEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        with caplog.at_level(logging.DEBUG, logger="encoding_atlas.encodings.angle"):
            enc.get_circuit(x, backend="pennylane")

        assert "outside typical range" not in caplog.text

    def test_large_values_trigger_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that large input values trigger debug warning."""
        import logging

        enc = AngleEncoding(n_features=4)
        x = np.array([100.0, 200.0, 300.0, 400.0])

        with caplog.at_level(logging.DEBUG, logger="encoding_atlas.encodings.angle"):
            enc.get_circuit(x, backend="pennylane")

        assert "outside typical range" in caplog.text

    def test_negative_large_values_trigger_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that large negative input values trigger debug warning."""
        import logging

        enc = AngleEncoding(n_features=4)
        x = np.array([-100.0, -200.0, -300.0, -400.0])

        with caplog.at_level(logging.DEBUG, logger="encoding_atlas.encodings.angle"):
            enc.get_circuit(x, backend="pennylane")

        assert "outside typical range" in caplog.text

    def test_boundary_values_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that values within threshold don't trigger warning.

        Values within 4*pi threshold (~12.57) should not warn.
        """
        import logging

        enc = AngleEncoding(n_features=4)
        x = np.array([10.0, 10.0, 10.0, 10.0])

        with caplog.at_level(logging.DEBUG, logger="encoding_atlas.encodings.angle"):
            enc.get_circuit(x, backend="pennylane")

        assert "outside typical range" not in caplog.text

    def test_threshold_exceeded_trigger_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that values just above threshold trigger warning.

        4*pi ~ 12.57, so 13.0 should trigger warning.
        """
        import logging

        enc = AngleEncoding(n_features=4)
        x = np.array([13.0, 0.0, 0.0, 0.0])

        with caplog.at_level(logging.DEBUG, logger="encoding_atlas.encodings.angle"):
            enc.get_circuit(x, backend="pennylane")

        assert "outside typical range" in caplog.text

    def test_debug_logging_does_not_affect_circuit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that debug logging doesn't affect circuit generation."""
        import logging

        enc = AngleEncoding(n_features=4)
        x_normal = np.array([0.1, 0.2, 0.3, 0.4])
        x_large = np.array([100.0, 200.0, 300.0, 400.0])

        with caplog.at_level(logging.DEBUG, logger="encoding_atlas.encodings.angle"):
            circuit_normal = enc.get_circuit(x_normal, backend="pennylane")
            circuit_large = enc.get_circuit(x_large, backend="pennylane")

        assert callable(circuit_normal)
        assert callable(circuit_large)


# =============================================================================
# Test Class: Slow Simulation Tests
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = AngleEncoding(n_features=4)

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        def get_state(x: np.ndarray) -> np.ndarray:
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            return full_circuit()

        x1 = np.array([0.1, 0.2, 0.3, 0.4])
        x2 = np.array([0.5, 0.6, 0.7, 0.8])

        state1 = get_state(x1)
        state2 = get_state(x2)

        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input produces same state."""
        enc = AngleEncoding(n_features=4)
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

        for i in range(1, len(states)):
            assert np.allclose(states[0], states[i], atol=1e-10)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states.

        Note: Different backends may use different qubit ordering conventions,
        so we verify states are valid and normalized rather than comparing
        exact probability distributions.
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        enc = AngleEncoding(n_features=4, rotation="Y", reps=1)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = pl_full()

        # Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_circuit.save_statevector()
        simulator = AerSimulator(method="statevector")
        compiled = transpile(qk_circuit, simulator)
        result = simulator.run(compiled).result()
        qk_state = result.get_statevector().data

        # Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_simulator = cirq.Simulator()
        cirq_result = cirq_simulator.simulate(cirq_circuit)
        cirq_state = cirq_result.final_state_vector

        # All states should be normalized
        assert np.isclose(np.sum(np.abs(pl_state) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(qk_state) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(cirq_state) ** 2), 1.0, atol=1e-10)

        # All states should have the same dimension
        assert len(pl_state) == len(qk_state) == len(cirq_state) == 2**enc.n_qubits

        # Probability distributions should have same values (accounting for ordering)
        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(pl_probs, qk_probs, atol=1e-6)
        np.testing.assert_allclose(pl_probs, cirq_probs, atol=1e-6)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations_produce_valid_states(self, rotation: str) -> None:
        """Test that all rotation types produce valid states across backends."""
        enc = AngleEncoding(n_features=4, rotation=rotation)
        x = np.array([0.3, 0.5, 0.7, 0.9])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_large_scale_simulation(self) -> None:
        """Test simulation with larger qubit count."""
        enc = AngleEncoding(n_features=10, reps=2)
        x = np.random.randn(10)

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        assert len(state) == 2**10
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_batch_simulation_consistency(self) -> None:
        """Test that batch generation produces consistent results."""
        enc = AngleEncoding(n_features=4)
        batch = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
        )

        circuits = enc.get_circuits(batch, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        states = []
        for circuit_fn in circuits:

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            states.append(full_circuit())

        # Both states should be normalized
        for state in states:
            assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

        # States should be different
        fidelity = np.abs(np.vdot(states[0], states[1])) ** 2
        assert fidelity < 0.99


# =============================================================================
# Test Class: Cross-Backend State Fidelity (Slow)
# =============================================================================


@pytest.mark.slow
@pytest.mark.cross_backend
class TestCrossBackendStateFidelity:
    """Tests that verify state equivalence across all quantum backends.

    These tests ensure that PennyLane, Qiskit, and Cirq backends produce
    mathematically equivalent quantum states for the same input data.
    This is critical for production use where backend choice should not
    affect the encoded quantum state.

    Note: Different backends may use different qubit ordering conventions
    (little-endian vs big-endian), so we compare sorted probability
    distributions rather than raw state vectors.
    """

    @staticmethod
    def _get_pennylane_state(
        enc: AngleEncoding,
        x: np.ndarray,
    ) -> np.ndarray:
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
        enc: AngleEncoding,
        x: np.ndarray,
    ) -> np.ndarray:
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
        enc: AngleEncoding,
        x: np.ndarray,
    ) -> np.ndarray:
        """Execute Cirq circuit and return state vector."""
        circuit = enc.get_circuit(x, backend="cirq")
        simulator = cirq.Simulator(dtype=np.complex128)
        result = simulator.simulate(circuit)

        return np.array(result.final_state_vector)

    def _assert_states_equivalent(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
        name1: str = "state1",
        name2: str = "state2",
        atol: float = 1e-6,
    ) -> None:
        """Assert two quantum states are equivalent up to qubit ordering.

        Compares sorted probability distributions to account for different
        qubit ordering conventions between backends.
        """
        assert len(state1) == len(state2), (
            f"{name1} and {name2} have different dimensions: "
            f"{len(state1)} vs {len(state2)}"
        )

        norm1 = np.sum(np.abs(state1) ** 2)
        norm2 = np.sum(np.abs(state2) ** 2)
        assert np.isclose(
            norm1, 1.0, atol=1e-10
        ), f"{name1} is not normalized: |norm|^2 = {norm1}"
        assert np.isclose(
            norm2, 1.0, atol=1e-10
        ), f"{name2} is not normalized: |norm|^2 = {norm2}"

        probs1 = sorted(np.abs(state1) ** 2)
        probs2 = sorted(np.abs(state2) ** 2)

        np.testing.assert_allclose(
            probs1,
            probs2,
            atol=atol,
            err_msg=(f"Probability distributions differ between {name1} and {name2}"),
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_all_backends_produce_equivalent_states(self) -> None:
        """Test that all backends produce mathematically equivalent states.

        This is the core cross-backend fidelity test for AngleEncoding.
        """
        enc = AngleEncoding(n_features=4, rotation="Y", reps=1)
        x = np.array([0.5, 0.3, 0.7, 0.1])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # All states should have correct dimension: 2^n_qubits = 16
        assert len(pl_state) == 16
        assert len(qk_state) == 16
        assert len(cirq_state) == 16

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
        self._assert_states_equivalent(qk_state, cirq_state, "Qiskit", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_cross_backend_all_rotations(self, rotation: str) -> None:
        """Test cross-backend equivalence for all rotation types."""
        enc = AngleEncoding(n_features=4, rotation=rotation, reps=1)
        x = np.array([0.3, 0.5, 0.7, 0.9])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.parametrize("n_features", [1, 2, 4, 8])
    def test_cross_backend_various_sizes(self, n_features: int) -> None:
        """Test cross-backend equivalence for various feature counts."""
        enc = AngleEncoding(n_features=n_features, rotation="Y")

        rng = np.random.default_rng(seed=n_features * 42)
        x = rng.random(n_features) * np.pi

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        expected_dim = 2**enc.n_qubits
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.parametrize("reps", [1, 2, 3])
    def test_cross_backend_multiple_reps(self, reps: int) -> None:
        """Test cross-backend equivalence with multiple repetitions."""
        enc = AngleEncoding(n_features=4, rotation="Y", reps=reps)
        x = np.array([0.2, 0.4, 0.6, 0.8])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_single_feature(self) -> None:
        """Test cross-backend equivalence with single feature (1 qubit)."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([np.pi / 4])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        assert len(pl_state) == 2
        assert len(qk_state) == 2
        assert len(cirq_state) == 2

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_zero_angles(self) -> None:
        """Test cross-backend equivalence with zero angles.

        Zero rotation should leave all qubits in |0> state for Y rotation.
        """
        enc = AngleEncoding(n_features=4, rotation="Y")
        x = np.zeros(4)

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        for state, name in [
            (pl_state, "PennyLane"),
            (qk_state, "Qiskit"),
            (cirq_state, "Cirq"),
        ]:
            max_prob = np.max(np.abs(state) ** 2)
            assert np.isclose(
                max_prob, 1.0, atol=1e-10
            ), f"{name}: max probability should be 1.0 for zero angles"

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_pi_rotation(self) -> None:
        """Test cross-backend equivalence with pi rotation.

        Pi rotation on Y axis should flip qubits from |0> to |1>.
        """
        enc = AngleEncoding(n_features=2, rotation="Y")
        x = np.array([np.pi, np.pi])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        for state, name in [
            (pl_state, "PennyLane"),
            (qk_state, "Qiskit"),
            (cirq_state, "Cirq"),
        ]:
            max_prob = np.max(np.abs(state) ** 2)
            assert np.isclose(
                max_prob, 1.0, atol=1e-10
            ), f"{name}: max probability should be 1.0 for pi rotation"

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_superposition(self) -> None:
        """Test cross-backend equivalence for superposition states.

        Pi/2 rotation creates equal superposition between |0> and |1>.
        """
        enc = AngleEncoding(n_features=2, rotation="Y")
        x = np.array([np.pi / 2, np.pi / 2])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        # For 2 qubits in equal superposition, all 4 probabilities should be 0.25
        for state, name in [
            (pl_state, "PennyLane"),
            (qk_state, "Qiskit"),
            (cirq_state, "Cirq"),
        ]:
            probs = np.abs(state) ** 2
            np.testing.assert_allclose(
                sorted(probs),
                [0.25, 0.25, 0.25, 0.25],
                atol=1e-10,
                err_msg=f"{name} does not produce uniform superposition",
            )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_reproducibility(self) -> None:
        """Test that repeated executions produce identical results."""
        enc = AngleEncoding(n_features=4, rotation="Y")
        x = np.array([0.3, 0.5, 0.7, 0.9])

        pl_state1 = self._get_pennylane_state(enc, x)
        pl_state2 = self._get_pennylane_state(enc, x)
        np.testing.assert_allclose(pl_state1, pl_state2, atol=1e-14)

        qk_state1 = self._get_qiskit_state(enc, x)
        qk_state2 = self._get_qiskit_state(enc, x)
        np.testing.assert_allclose(qk_state1, qk_state2, atol=1e-14)

        cirq_state1 = self._get_cirq_state(enc, x)
        cirq_state2 = self._get_cirq_state(enc, x)
        np.testing.assert_allclose(cirq_state1, cirq_state2, atol=1e-14)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_different_inputs_different_states(self) -> None:
        """Test that different inputs produce different states across backends."""
        enc = AngleEncoding(n_features=4, rotation="Y")
        x1 = np.array([0.1, 0.2, 0.3, 0.4])
        x2 = np.array([0.9, 0.8, 0.7, 0.6])

        pl_state1 = self._get_pennylane_state(enc, x1)
        pl_state2 = self._get_pennylane_state(enc, x2)

        qk_state1 = self._get_qiskit_state(enc, x1)
        qk_state2 = self._get_qiskit_state(enc, x2)

        cirq_state1 = self._get_cirq_state(enc, x1)
        cirq_state2 = self._get_cirq_state(enc, x2)

        def fidelity(s1: np.ndarray, s2: np.ndarray) -> float:
            return float(np.abs(np.vdot(s1, s2)) ** 2)

        assert fidelity(pl_state1, pl_state2) < 0.99
        assert fidelity(qk_state1, qk_state2) < 0.99
        assert fidelity(cirq_state1, cirq_state2) < 0.99

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_extreme_angles(self) -> None:
        """Test cross-backend equivalence with extreme angle values.

        Angles much larger than 2*pi should still produce consistent results
        due to periodicity of rotations.
        """
        enc = AngleEncoding(n_features=4, rotation="Y")
        x = np.array([100.0, -100.0, 1000.0, -1000.0])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        assert np.isclose(np.sum(np.abs(pl_state) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(qk_state) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(cirq_state) ** 2), 1.0, atol=1e-10)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    def test_cross_backend_negative_angles(self) -> None:
        """Test cross-backend equivalence with negative angles."""
        enc = AngleEncoding(n_features=4, rotation="Y")
        x = np.array([-0.5, -0.3, -0.7, -0.1])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
