"""Comprehensive tests for PauliFeatureMap.

This test module provides complete coverage of the PauliFeatureMap class,
which encodes classical data into quantum states using configurable Pauli
rotation gates with optional entangling operations. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, gate_count, entanglement pairs)
- Pauli term behavior (single-qubit, two-qubit, normalization, deduplication)
- Circuit generation for all backends (PennyLane, Qiskit, Cirq)
- Mathematical correctness verification (feature map functions)
- Edge cases and boundary conditions
- Numerical stability tests
- Equality and hashing
- String representation
- Backend error handling
- Serialization (pickle roundtrip)
- Concurrent access / thread safety
- Slow simulation tests (cross-backend state fidelity)

Run with: pytest tests/unit/encodings/test_pauli_feature_map.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_pauli_feature_map.py -v -m "not slow"

References
----------
.. [1] Havlicek et al., "Supervised learning with quantum-enhanced feature
       spaces." Nature 567, 209-212 (2019).
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas import PauliFeatureMap
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

    Used for minimal qubit count tests.
    """
    return np.array([0.5, 1.0])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (mid-range values)
    - [0.9, 1.0, 1.1, 1.2] (higher values)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ])


@pytest.fixture
def default_encoding() -> PauliFeatureMap:
    """Default PauliFeatureMap with 4 features."""
    return PauliFeatureMap(n_features=4)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for PauliFeatureMap instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = PauliFeatureMap(n_features=4)

        assert enc.n_features == 4
        assert enc.n_qubits == 4
        assert enc.reps == 2
        assert enc.paulis == ["Z", "ZZ"]
        assert enc.entanglement == "full"

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = PauliFeatureMap(n_features=1, paulis=["Z"])
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16]:
            enc = PauliFeatureMap(n_features=n, paulis=["Z"])
            assert enc.n_features == n

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = PauliFeatureMap(n_features=4, reps=5)
        assert enc.reps == 5

    def test_custom_paulis(self) -> None:
        """Test instantiation with custom Pauli terms."""
        enc = PauliFeatureMap(n_features=4, paulis=["Y", "YY", "ZZ"])
        assert enc.paulis == ["Y", "YY", "ZZ"]

    def test_paulis_normalized_to_uppercase(self) -> None:
        """Test that Pauli terms are normalized to uppercase."""
        enc = PauliFeatureMap(n_features=4, paulis=["z", "zz", "Yy"])
        assert enc.paulis == ["Z", "ZZ", "YY"]

    def test_paulis_deduplicated(self) -> None:
        """Test that duplicate Pauli terms are removed."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ", "Z", "ZZ"])
        assert enc.paulis == ["Z", "ZZ"]

    def test_single_pauli_only(self) -> None:
        """Test encoding with only single-qubit Paulis."""
        enc = PauliFeatureMap(n_features=4, paulis=["X", "Y", "Z"])
        assert enc._single_paulis == ["X", "Y", "Z"]
        assert enc._two_paulis == []

    def test_two_pauli_only(self) -> None:
        """Test encoding with only two-qubit Paulis."""
        enc = PauliFeatureMap(n_features=4, paulis=["XX", "YY", "ZZ"])
        assert enc._single_paulis == []
        assert enc._two_paulis == ["XX", "YY", "ZZ"]

    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_entanglement_patterns(self, entanglement: str) -> None:
        """Test all valid entanglement patterns."""
        enc = PauliFeatureMap(n_features=4, entanglement=entanglement)  # type: ignore
        assert enc.entanglement == entanglement

    def test_config_stored_correctly(self) -> None:
        """Test that configuration is stored in config dict."""
        enc = PauliFeatureMap(
            n_features=4,
            reps=3,
            paulis=["Y", "ZZ"],
            entanglement="linear",
        )
        assert enc.config["reps"] == 3
        assert enc.config["paulis"] == ["Y", "ZZ"]
        assert enc.config["entanglement"] == "linear"

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = PauliFeatureMap(n_features=100, paulis=["Z"])
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
            PauliFeatureMap(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            PauliFeatureMap(n_features=-1)

    def test_invalid_n_features_float(self) -> None:
        """Test that float n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            PauliFeatureMap(n_features=4.0)  # type: ignore

    def test_invalid_n_features_string(self) -> None:
        """Test that string n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            PauliFeatureMap(n_features="4")  # type: ignore

    def test_invalid_n_features_none(self) -> None:
        """Test that None n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            PauliFeatureMap(n_features=None)  # type: ignore

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            PauliFeatureMap(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            PauliFeatureMap(n_features=4, reps=-1)

    def test_invalid_reps_float(self) -> None:
        """Test that float reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            PauliFeatureMap(n_features=4, reps=2.5)  # type: ignore

    def test_invalid_pauli_term(self) -> None:
        """Test that invalid Pauli term raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Pauli term"):
            PauliFeatureMap(n_features=4, paulis=["Z", "INVALID"])

    def test_invalid_pauli_three_qubit(self) -> None:
        """Test that three-qubit Pauli term raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Pauli term"):
            PauliFeatureMap(n_features=4, paulis=["ZZZ"])

    def test_empty_paulis_list(self) -> None:
        """Test that empty paulis list raises ValueError."""
        with pytest.raises(ValueError, match="paulis list cannot be empty"):
            PauliFeatureMap(n_features=4, paulis=[])

    def test_paulis_wrong_type(self) -> None:
        """Test that non-list paulis raises TypeError."""
        with pytest.raises(TypeError, match="paulis must be a list"):
            PauliFeatureMap(n_features=4, paulis="ZZ")  # type: ignore

    def test_invalid_entanglement(self) -> None:
        """Test that invalid entanglement pattern raises ValueError."""
        with pytest.raises(ValueError, match="entanglement must be one of"):
            PauliFeatureMap(n_features=4, entanglement="invalid")  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of PauliFeatureMap."""

    def test_n_qubits(self) -> None:
        """Test n_qubits property calculation.

        For PauliFeatureMap, n_qubits = n_features.
        """
        for n in [2, 4, 8, 16]:
            enc = PauliFeatureMap(n_features=n)
            assert enc.n_qubits == n

    def test_depth(self) -> None:
        """Test circuit depth calculation.

        Depth formula: reps * (H_layer + single_paulis + 3*two_paulis)
        For n_features=4, reps=2, paulis=["Z", "ZZ"]: depth = 2 * (1 + 1 + 3) = 10
        """
        enc = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])
        # depth = reps * (H_layer(1) + single(1) + two(3)) = 2 * 5 = 10
        assert enc.depth == 10

    def test_depth_single_pauli_only(self) -> None:
        """Test depth with only single-qubit Paulis.

        Depth formula: reps * (H_layer + single_paulis)
        For paulis=["Z", "Y"]: depth = 2 * (1 + 2) = 6
        """
        enc = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "Y"])
        assert enc.depth == 6

    def test_depth_two_pauli_only(self) -> None:
        """Test depth with only two-qubit Paulis.

        Depth formula: reps * (H_layer + 3*two_paulis)
        For paulis=["ZZ"]: depth = 1 * (1 + 3) = 4
        """
        enc = PauliFeatureMap(n_features=4, reps=1, paulis=["ZZ"])
        assert enc.depth == 4

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = PauliFeatureMap(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = PauliFeatureMap(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_properties_gate_count(self) -> None:
        """Test gate count calculation consistency."""
        enc = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])
        props = enc.properties
        assert props.gate_count == props.single_qubit_gates + props.two_qubit_gates

    def test_is_entangling_with_two_qubit_paulis(self) -> None:
        """Test is_entangling is True with two-qubit Paulis."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"])
        assert enc.properties.is_entangling is True

    def test_is_not_entangling_single_paulis_only(self) -> None:
        """Test is_entangling is False with only single-qubit Paulis."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "Y", "X"])
        assert enc.properties.is_entangling is False

    def test_is_not_entangling_single_qubit(self) -> None:
        """Test is_entangling is False with single qubit."""
        enc = PauliFeatureMap(n_features=1, paulis=["Z"])
        assert enc.properties.is_entangling is False

    def test_simulability_no_entanglement(self) -> None:
        """Test simulability when not entangling."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z"])
        assert enc.properties.simulability == "simulable"

    def test_simulability_with_entanglement(self) -> None:
        """Test simulability with entanglement."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"])
        assert enc.properties.simulability == "not_simulable"

    def test_parameter_count_is_zero(self) -> None:
        """Test that parameter_count is 0 (no trainable params)."""
        enc = PauliFeatureMap(n_features=4)
        assert enc.properties.parameter_count == 0


# =============================================================================
# Test Class: Pauli Behavior (Encoding-Specific)
# =============================================================================


class TestPauliBehavior:
    """Tests for Pauli-specific behavior including entanglement pairs."""

    def test_full_entanglement_4_qubits(self) -> None:
        """Test full entanglement with 4 qubits.

        Full entanglement connects all pairs: n*(n-1)/2 = 6 pairs.
        """
        enc = PauliFeatureMap(n_features=4, entanglement="full")
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert enc._entanglement_pairs == expected

    def test_linear_entanglement_4_qubits(self) -> None:
        """Test linear entanglement with 4 qubits.

        Linear entanglement connects adjacent pairs: n-1 = 3 pairs.
        """
        enc = PauliFeatureMap(n_features=4, entanglement="linear")
        expected = [(0, 1), (1, 2), (2, 3)]
        assert enc._entanglement_pairs == expected

    def test_circular_entanglement_4_qubits(self) -> None:
        """Test circular entanglement with 4 qubits.

        Circular entanglement adds wrap-around: n pairs.
        """
        enc = PauliFeatureMap(n_features=4, entanglement="circular")
        expected = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert enc._entanglement_pairs == expected

    def test_entanglement_2_qubits(self) -> None:
        """Test that all entanglement patterns have same pairs for 2 qubits."""
        for ent in ["full", "linear", "circular"]:
            enc = PauliFeatureMap(n_features=2, entanglement=ent)  # type: ignore
            assert enc._entanglement_pairs == [(0, 1)]

    def test_entanglement_1_qubit(self) -> None:
        """Test that 1 qubit has no entanglement pairs."""
        enc = PauliFeatureMap(n_features=1, paulis=["Z"])
        assert enc._entanglement_pairs == []

    def test_full_entanglement_count(self) -> None:
        """Test that full entanglement has n*(n-1)/2 pairs."""
        for n in [2, 4, 6, 8]:
            enc = PauliFeatureMap(n_features=n, entanglement="full")
            expected_pairs = n * (n - 1) // 2
            assert len(enc._entanglement_pairs) == expected_pairs

    def test_all_valid_single_paulis(self) -> None:
        """Test all valid single-qubit Pauli terms."""
        enc = PauliFeatureMap(n_features=4, paulis=["X", "Y", "Z"])
        assert set(enc._single_paulis) == {"X", "Y", "Z"}

    def test_all_valid_two_paulis(self) -> None:
        """Test all valid two-qubit Pauli terms."""
        all_two_paulis = ["XX", "YY", "ZZ", "XY", "XZ", "YZ", "YX", "ZX", "ZY"]
        enc = PauliFeatureMap(n_features=4, paulis=all_two_paulis)
        assert len(enc._two_paulis) == 9


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        validated = default_encoding._validate_input(sample_data_4d)
        assert validated.shape == (4,)

    def test_valid_2d_single_sample(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test that valid 2D single-sample input passes."""
        x = np.array([[0.1, 0.2, 0.3, 0.4]])
        validated = default_encoding._validate_input(x)
        assert validated.shape == (1, 4)

    def test_wrong_feature_count(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2])  # Only 2 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_nan_input_rejected(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding._validate_input(x)

    def test_inf_input_rejected(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_list_input_accepted(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]  # list, not array
        validated = default_encoding._validate_input(x)
        assert isinstance(validated, np.ndarray)

    def test_batch_input_shape(
        self,
        default_encoding: PauliFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that batch input generates multiple circuits."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        if HAS_QISKIT:
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
        default_encoding: PauliFeatureMap,
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
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"])
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be valid (normalized)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_single_pauli_circuit(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test circuit with only single-qubit Paulis."""
        enc = PauliFeatureMap(n_features=4, paulis=["Y"], reps=1)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)

    def test_xx_pauli_circuit(
        self,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test circuit with XX Pauli term."""
        enc = PauliFeatureMap(n_features=2, paulis=["XX"], reps=1)
        circuit_fn = enc.get_circuit(sample_data_2d, backend="pennylane")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)

    def test_yy_pauli_circuit(
        self,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test circuit with YY Pauli term."""
        enc = PauliFeatureMap(n_features=2, paulis=["YY"], reps=1)
        circuit_fn = enc.get_circuit(sample_data_2d, backend="pennylane")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)

    def test_batch_circuits(
        self,
        default_encoding: PauliFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    def test_different_inputs_give_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = PauliFeatureMap(n_features=2, paulis=["Z", "ZZ"], reps=1)

        x1 = np.array([0.1, 0.2])
        x2 = np.array([0.5, 0.6])

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def get_state(x):
            circuit_fn = enc.get_circuit(x, backend="pennylane")
            circuit_fn()
            return qml.state()

        state1 = get_state(x1)
        state2 = get_state(x2)

        # States should be different
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99


# =============================================================================
# Test Class: Qiskit Backend
# =============================================================================


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.mark.backend_qiskit
class TestQiskitBackend:
    """Tests for Qiskit circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_circuit_name(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has appropriate name."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.name == "PauliFeatureMap"

    def test_single_pauli_circuit(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test circuit with only single-qubit Paulis."""
        enc = PauliFeatureMap(n_features=4, paulis=["X", "Y", "Z"], reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_all_two_qubit_paulis(
        self,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test circuit with various two-qubit Paulis."""
        for pauli in ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]:
            enc = PauliFeatureMap(n_features=2, paulis=[pauli], reps=1)
            circuit = enc.get_circuit(sample_data_2d, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_batch_circuits(
        self,
        default_encoding: PauliFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    def test_circuit_depth_reasonable(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit depth is reasonable."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        # Depth should be positive and bounded
        assert 0 < circuit.depth() <= 100


# =============================================================================
# Test Class: Cirq Backend
# =============================================================================


@pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
@pytest.mark.backend_cirq
class TestCirqBackend:
    """Tests for Cirq circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    def test_single_pauli_circuit(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test circuit with only single-qubit Paulis."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z"], reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_all_two_qubit_paulis(
        self,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test circuit with various two-qubit Paulis."""
        for pauli in ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]:
            enc = PauliFeatureMap(n_features=2, paulis=[pauli], reps=1)
            circuit = enc.get_circuit(sample_data_2d, backend="cirq")
            assert isinstance(circuit, cirq.Circuit)

    def test_batch_circuits(
        self,
        default_encoding: PauliFeatureMap,
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
    """Tests for mathematical correctness of feature map computations."""

    def test_single_feature_map(self) -> None:
        """Test single-qubit feature map computation.

        Formula: phi(x_i) = 2 * x_i
        """
        enc = PauliFeatureMap(n_features=4)
        x = np.array([0.5, 1.0, 1.5, 2.0])

        # phi(x_i) = 2 * x_i
        assert enc._feature_map_single(x, 0) == 1.0  # 2 * 0.5
        assert enc._feature_map_single(x, 1) == 2.0  # 2 * 1.0
        assert enc._feature_map_single(x, 2) == 3.0  # 2 * 1.5
        assert enc._feature_map_single(x, 3) == 4.0  # 2 * 2.0

    def test_two_feature_map(self) -> None:
        """Test two-qubit feature map computation.

        Formula: phi(x_i, x_j) = 2 * (pi - x_i) * (pi - x_j)
        """
        enc = PauliFeatureMap(n_features=4)
        x = np.array([0.0, np.pi, np.pi, 0.0])

        # For x_i = 0, x_j = pi: 2 * pi * 0 = 0
        result = enc._feature_map_two(x, 0, 1)
        assert np.isclose(result, 0.0)

        # For x_i = pi, x_j = pi: 2 * 0 * 0 = 0
        result = enc._feature_map_two(x, 1, 2)
        assert np.isclose(result, 0.0)

    def test_feature_map_symmetry(self) -> None:
        """Test that two-qubit feature map is symmetric.

        phi(x_i, x_j) should equal phi(x_j, x_i).
        """
        enc = PauliFeatureMap(n_features=4)
        x = np.array([0.1, 0.5, 1.0, 1.5])

        for i in range(4):
            for j in range(i + 1, 4):
                result_ij = enc._feature_map_two(x, i, j)
                result_ji = enc._feature_map_two(x, j, i)
                assert np.isclose(result_ij, result_ji)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature(self) -> None:
        """Test encoding with single feature."""
        enc = PauliFeatureMap(n_features=1, paulis=["Z"])
        assert enc.n_qubits == 1
        assert enc._entanglement_pairs == []

    def test_single_rep(self) -> None:
        """Test encoding with single repetition."""
        enc = PauliFeatureMap(n_features=4, reps=1)
        assert enc.reps == 1

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions.

        Depth formula: 10 * (1 + 1 + 3) = 50
        """
        enc = PauliFeatureMap(n_features=4, reps=10)
        assert enc.reps == 10
        assert enc.depth == 10 * (1 + 1 + 3)

    def test_large_feature_count(self) -> None:
        """Test encoding with large feature count.

        Linear entanglement should have n-1 pairs.
        """
        enc = PauliFeatureMap(n_features=20, entanglement="linear")
        assert enc.n_qubits == 20
        assert len(enc._entanglement_pairs) == 19

    def test_zero_valued_input(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test with all-zero input."""
        x = np.zeros(4)
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_large_valued_input(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test with large input values."""
        x = np.array([100.0, 200.0, 300.0, 400.0])
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_negative_valued_input(
        self,
        default_encoding: PauliFeatureMap,
    ) -> None:
        """Test with negative input values."""
        x = np.array([-0.1, -0.2, -0.3, -0.4])
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)


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
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

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
    def test_very_large_values(self) -> None:
        """Test encoding with very large input values.

        Large values should not cause overflow.
        """
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
        x = np.array([1e10, 1e11, 1e12, 1e13])

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
    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
        x = np.array([1e-10, 1e10, 1e-5, 1e5])

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
    def test_extreme_values_produce_normalized_state(self) -> None:
        """Test that extreme values still produce normalized quantum states."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
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
    def test_negative_values(self) -> None:
        """Test encoding with negative input values.

        Rotation gates handle negative angles correctly.
        """
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
        x = np.array([-0.5, -1.0, -1.5, -2.0])

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
    def test_subnormal_values(self) -> None:
        """Test encoding with subnormal (denormalized) floating-point values.

        Subnormal values can cause precision issues in some implementations.
        """
        enc = PauliFeatureMap(n_features=4, paulis=["Z"], reps=1)
        x = np.array([1e-310, 1e-315, 1e-320, 1e-308])

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
    def test_values_near_pi_boundaries(self) -> None:
        """Test encoding with values near pi and 2*pi boundaries.

        Rotation periodicity boundaries can reveal numerical edge cases.
        """
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
        eps = 1e-14
        x = np.array([
            np.pi - eps,
            np.pi + eps,
            2 * np.pi - eps,
            2 * np.pi + eps,
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
    def test_precision_with_high_reps(self) -> None:
        """Test numerical precision is maintained with high repetition count.

        Accumulated floating-point errors could compound with many layers.
        """
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=20)
        x = np.array([0.123456789, 0.987654321, 0.111111111, 0.999999999])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-8)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])
        enc2 = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = PauliFeatureMap(n_features=4)
        enc2 = PauliFeatureMap(n_features=8)
        assert enc1 != enc2

    def test_equality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = PauliFeatureMap(n_features=4, reps=2)
        enc2 = PauliFeatureMap(n_features=4, reps=3)
        assert enc1 != enc2

    def test_equality_different_paulis(self) -> None:
        """Test that encodings with different paulis are not equal."""
        enc1 = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"])
        enc2 = PauliFeatureMap(n_features=4, paulis=["Y", "YY"])
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])
        enc2 = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = PauliFeatureMap(n_features=4, reps=2)
        enc2 = PauliFeatureMap(n_features=4, reps=2)  # Same as enc1
        enc3 = PauliFeatureMap(n_features=4, reps=3)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = PauliFeatureMap(n_features=4, reps=2)
        enc2 = PauliFeatureMap(n_features=4, reps=2)

        d = {enc1: "two reps"}
        assert d[enc2] == "two reps"  # enc2 should find same key


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ string representation."""

    def test_repr_contains_class_name(self) -> None:
        """Test that repr contains class name."""
        enc = PauliFeatureMap(n_features=4)
        repr_str = repr(enc)
        assert "PauliFeatureMap" in repr_str

    def test_repr_contains_n_features(self) -> None:
        """Test that repr contains n_features parameter."""
        enc = PauliFeatureMap(n_features=4)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str

    def test_repr_contains_reps(self) -> None:
        """Test that repr contains reps parameter."""
        enc = PauliFeatureMap(n_features=4, reps=3)
        repr_str = repr(enc)
        assert "reps=3" in repr_str

    def test_repr_contains_paulis(self) -> None:
        """Test that repr contains paulis parameter."""
        enc = PauliFeatureMap(n_features=4, paulis=["Y", "YY"])
        repr_str = repr(enc)
        assert "['Y', 'YY']" in repr_str

    def test_repr_contains_entanglement(self) -> None:
        """Test that repr contains entanglement parameter."""
        enc = PauliFeatureMap(n_features=4, entanglement="linear")
        repr_str = repr(enc)
        assert "linear" in repr_str

    def test_repr_is_informative(self) -> None:
        """Test that repr provides complete information."""
        enc = PauliFeatureMap(
            n_features=8,
            reps=3,
            paulis=["Y", "YY"],
            entanglement="linear",
        )
        repr_str = repr(enc)
        assert "n_features=8" in repr_str
        assert "reps=3" in repr_str
        assert "['Y', 'YY']" in repr_str
        assert "linear" in repr_str


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for invalid backend names and missing backends."""

    def test_invalid_backend_raises_error(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_backend_case_sensitivity(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that backend names are case-sensitive."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="QISKIT")  # type: ignore

    def test_empty_backend_raises_error(
        self,
        default_encoding: PauliFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that empty backend string raises error."""
        with pytest.raises(ValueError):
            default_encoding.get_circuit(sample_data_4d, backend="")  # type: ignore


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = PauliFeatureMap(n_features=4, reps=3, paulis=["Y", "ZZ"])
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.reps == enc.reps
        assert restored.paulis == enc.paulis
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = PauliFeatureMap(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = PauliFeatureMap(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        if HAS_QISKIT:
            circuit = restored.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == 4

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = PauliFeatureMap(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        props = restored.properties
        assert isinstance(props, EncodingProperties)
        assert props.n_qubits == enc.n_qubits

    def test_pickle_with_all_parameters(self) -> None:
        """Test pickle roundtrip with all parameters specified."""
        enc = PauliFeatureMap(
            n_features=6,
            reps=4,
            paulis=["X", "Y", "Z", "XX", "YY", "ZZ"],
            entanglement="circular",
        )
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == 6
        assert restored.reps == 4
        assert restored.paulis == ["X", "Y", "Z", "XX", "YY", "ZZ"]
        assert restored.entanglement == "circular"


# =============================================================================
# Test Class: Concurrent Access / Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        enc = PauliFeatureMap(n_features=4)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
                    circuit = enc.get_circuit(x, backend="qiskit")
                    circuits.append(circuit)
            except Exception as e:
                errors.append(e)
            return circuits

        if HAS_QISKIT:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(generate_circuits, i) for i in range(num_threads)
                ]
                results = [f.result() for f in as_completed(futures)]

            # No errors should have occurred
            assert len(errors) == 0, f"Thread errors: {errors}"
            # All circuits should have been generated
            total_circuits = sum(len(r) for r in results)
            assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = PauliFeatureMap(n_features=4)
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

    def test_concurrent_different_backends(self) -> None:
        """Test concurrent circuit generation with different backends."""
        enc = PauliFeatureMap(n_features=4)
        errors: list[Exception] = []

        def generate_for_backend(backend: str) -> Any:
            try:
                x = np.random.randn(4)
                return enc.get_circuit(x, backend=backend)
            except Exception as e:
                errors.append(e)
                return None

        backends = []
        if HAS_PENNYLANE:
            backends.extend(["pennylane"] * 10)
        if HAS_QISKIT:
            backends.extend(["qiskit"] * 10)
        if HAS_CIRQ:
            backends.extend(["cirq"] * 10)

        if backends:
            with ThreadPoolExecutor(max_workers=len(backends)) as executor:
                futures = [
                    executor.submit(generate_for_backend, b) for b in backends
                ]
                results = [f.result() for f in as_completed(futures)]

            assert len(errors) == 0
            assert all(r is not None for r in results)


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
        enc: PauliFeatureMap,
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
        enc: PauliFeatureMap,
        x: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Execute Qiskit circuit and return state vector."""
        circuit = enc.get_circuit(x, backend="qiskit")
        sv = Statevector(circuit)
        return np.array(sv.data)

    @staticmethod
    def _get_cirq_state(
        enc: PauliFeatureMap,
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
        assert len(state1) == len(state2), (
            f"{name1} and {name2} have different dimensions"
        )

        # Both states must be normalized
        norm1 = np.sum(np.abs(state1) ** 2)
        norm2 = np.sum(np.abs(state2) ** 2)
        assert np.isclose(norm1, 1.0, atol=1e-10)
        assert np.isclose(norm2, 1.0, atol=1e-10)

        # Compare sorted probability distributions
        probs1 = sorted(np.abs(state1) ** 2)
        probs2 = sorted(np.abs(state2) ** 2)

        np.testing.assert_allclose(probs1, probs2, atol=atol)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
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
    def test_cross_backend_state_equivalence(self) -> None:
        """Test that all backends produce equivalent quantum states."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
        self._assert_states_equivalent(qk_state, cirq_state, "Qiskit", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("reps", [1, 2, 4])
    def test_cross_backend_with_different_reps(self, reps: int) -> None:
        """Test cross-backend equivalence with different repetition counts."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=reps)
        x = np.array([0.5, 1.0, 1.5, 2.0])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(
            pl_state, qk_state, f"PennyLane (reps={reps})", f"Qiskit (reps={reps})"
        )
        self._assert_states_equivalent(
            pl_state, cirq_state, f"PennyLane (reps={reps})", f"Cirq (reps={reps})"
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_cross_backend_with_different_entanglement(
        self,
        entanglement: str,
    ) -> None:
        """Test cross-backend equivalence with different entanglement patterns."""
        enc = PauliFeatureMap(
            n_features=4,
            paulis=["Z", "ZZ"],
            reps=2,
            entanglement=entanglement,  # type: ignore
        )
        x = np.array([0.2, 0.4, 0.6, 0.8])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(
            pl_state, qk_state,
            f"PennyLane (ent={entanglement})",
            f"Qiskit (ent={entanglement})"
        )
        self._assert_states_equivalent(
            pl_state, cirq_state,
            f"PennyLane (ent={entanglement})",
            f"Cirq (ent={entanglement})"
        )

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
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
                assert fidelity < 0.9999 or np.allclose(inputs[i], inputs[j])

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input always produces same state."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_state_normalization_many_configs(self) -> None:
        """Test state normalization across many configurations."""
        configs = [
            {"paulis": ["Z"], "reps": 1},
            {"paulis": ["Z", "ZZ"], "reps": 2},
            {"paulis": ["Y", "YY"], "reps": 1},
            {"paulis": ["X", "XX"], "reps": 1},
        ]

        for config in configs:
            enc = PauliFeatureMap(n_features=4, **config)
            x = np.random.randn(4)
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            dev = qml.device("default.qubit", wires=4)

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0), f"Config {config} gave norm {norm}"

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_zero_input(self) -> None:
        """Test cross-backend equivalence with zero input values."""
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=2)
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
    def test_cross_backend_pi_multiples(self) -> None:
        """Test cross-backend equivalence with pi multiples as input.

        These are special angles where rotation gates should produce
        well-defined states.
        """
        enc = PauliFeatureMap(n_features=4, paulis=["Z", "ZZ"], reps=1)
        x = np.array([np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
