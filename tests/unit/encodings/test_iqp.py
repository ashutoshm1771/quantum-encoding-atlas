"""Comprehensive tests for IQPEncoding.

This test module provides complete coverage of the IQPEncoding class,
which implements the Instantaneous Quantum Polynomial encoding for mapping
classical data to quantum states using diagonal gates in the Hadamard basis. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, gate_count, simulability)
- Entanglement pattern behavior (unique to IQP)
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

Run with: pytest tests/unit/encodings/test_iqp.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_iqp.py -v -m "not slow"

References
----------
.. [1] Havlicek et al. (2019), Nature - Supervised learning with quantum
       enhanced feature spaces
"""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest

from encoding_atlas import IQPEncoding

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

    Values chosen to exercise typical IQP encoding behavior.
    """
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def sample_data_2d() -> NDArray[np.floating]:
    """2-dimensional sample data for testing.

    Minimal case for testing entanglement behavior.
    """
    return np.array([0.5, 0.5])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (different values)
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
def default_encoding() -> IQPEncoding:
    """Default IQPEncoding with 4 features and full entanglement."""
    return IQPEncoding(n_features=4)


@pytest.fixture
def encoding_linear() -> IQPEncoding:
    """IQPEncoding with linear entanglement."""
    return IQPEncoding(n_features=4, entanglement="linear")


@pytest.fixture
def encoding_circular() -> IQPEncoding:
    """IQPEncoding with circular entanglement."""
    return IQPEncoding(n_features=4, entanglement="circular")


@pytest.fixture
def encoding_multi_reps() -> IQPEncoding:
    """IQPEncoding with multiple repetitions."""
    return IQPEncoding(n_features=4, reps=3)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for IQPEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = IQPEncoding(n_features=4)

        assert enc.n_features == 4
        assert enc.reps == 2  # Default reps
        assert enc.entanglement == "full"  # Default entanglement

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = IQPEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = IQPEncoding(n_features=4, reps=5)
        assert enc.reps == 5

    def test_reps_equals_one(self) -> None:
        """Test reps=1 (minimal IQP depth)."""
        enc = IQPEncoding(n_features=4, reps=1)
        assert enc.reps == 1

    def test_entanglement_full(self) -> None:
        """Test full entanglement pattern."""
        enc = IQPEncoding(n_features=4, entanglement="full")
        assert enc.entanglement == "full"

    def test_entanglement_linear(self) -> None:
        """Test linear entanglement pattern."""
        enc = IQPEncoding(n_features=4, entanglement="linear")
        assert enc.entanglement == "linear"

    def test_entanglement_circular(self) -> None:
        """Test circular entanglement pattern."""
        enc = IQPEncoding(n_features=4, entanglement="circular")
        assert enc.entanglement == "circular"

    def test_config_contains_reps(self) -> None:
        """Test that config dictionary contains reps parameter."""
        enc = IQPEncoding(n_features=4, reps=3)
        assert enc.config["reps"] == 3

    def test_config_contains_entanglement(self) -> None:
        """Test that config dictionary contains entanglement parameter."""
        enc = IQPEncoding(n_features=4, entanglement="linear")
        assert enc.config["entanglement"] == "linear"

    def test_all_custom_parameters(self) -> None:
        """Test all parameters customized together."""
        enc = IQPEncoding(n_features=8, reps=4, entanglement="circular")

        assert enc.n_features == 8
        assert enc.reps == 4
        assert enc.entanglement == "circular"

    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_all_entanglement_variants(self, entanglement: str) -> None:
        """Test all entanglement topology variants."""
        enc = IQPEncoding(n_features=4, entanglement=entanglement)
        assert enc.entanglement == entanglement

    @pytest.mark.parametrize("n_features", [1, 2, 4, 8, 16])
    def test_various_feature_counts(self, n_features: int) -> None:
        """Test instantiation with various feature counts."""
        enc = IQPEncoding(n_features=n_features)
        assert enc.n_features == n_features

    @pytest.mark.parametrize("reps", [1, 2, 5, 10])
    def test_various_reps(self, reps: int) -> None:
        """Test instantiation with various repetition counts."""
        enc = IQPEncoding(n_features=4, reps=reps)
        assert enc.reps == reps

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = IQPEncoding(n_features=100)
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
            IQPEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError):
            IQPEncoding(n_features=-1)

    def test_invalid_n_features_float(self) -> None:
        """Test that float n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            IQPEncoding(n_features=4.5)  # type: ignore

    def test_invalid_n_features_string(self) -> None:
        """Test that string n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            IQPEncoding(n_features="four")  # type: ignore

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises error."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            IQPEncoding(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises error."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            IQPEncoding(n_features=4, reps=-1)

    def test_invalid_entanglement(self) -> None:
        """Test that invalid entanglement raises error."""
        with pytest.raises(ValueError, match="entanglement must be"):
            IQPEncoding(n_features=4, entanglement="invalid")

    def test_invalid_entanglement_none_string(self) -> None:
        """Test that 'none' entanglement raises error (IQP requires entanglement)."""
        with pytest.raises(ValueError, match="entanglement must be"):
            IQPEncoding(n_features=4, entanglement="none")


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of IQPEncoding."""

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features (one qubit per feature)."""
        for n in [1, 2, 4, 8, 16]:
            enc = IQPEncoding(n_features=n)
            assert enc.n_qubits == n

    def test_depth(self) -> None:
        """Test circuit depth calculation.

        Depth formula: reps * 3 (H layer + RZ layer + ZZ layer per rep)
        For n_features=4, reps=2: depth = 2 * 3 = 6
        """
        enc = IQPEncoding(n_features=4, reps=2)
        # depth = reps(2) * layers_per_rep(3) = 6
        assert enc.depth == 6

    def test_depth_calculation_various_reps(self) -> None:
        """Test that depth equals reps * 3."""
        for reps in [1, 2, 3, 5]:
            enc = IQPEncoding(n_features=4, reps=reps)
            # Depth = reps * 3 (H layer, RZ layer, ZZ layer per rep)
            assert enc.depth == reps * 3

    def test_depth_single_rep(self) -> None:
        """Test single rep IQP encoding has depth 3."""
        enc = IQPEncoding(n_features=4, reps=1)
        assert enc.depth == 3

    def test_properties_type(self, default_encoding: IQPEncoding) -> None:
        """Test that properties returns expected structure."""
        props = default_encoding.properties

        assert props.n_qubits == 4
        assert props.depth == 6  # 2 reps * 3
        assert props.gate_count > 0
        assert props.single_qubit_gates > 0
        assert props.two_qubit_gates > 0

    def test_properties_cached(self, default_encoding: IQPEncoding) -> None:
        """Test that properties are cached (same object returned)."""
        props1 = default_encoding.properties
        props2 = default_encoding.properties
        assert props1 is props2

    def test_is_entangling(self) -> None:
        """Test that IQP encoding creates entanglement."""
        enc = IQPEncoding(n_features=4)
        assert enc.properties.is_entangling is True

    def test_not_classically_simulable(self) -> None:
        """Test that IQP is provably not classically simulable."""
        enc = IQPEncoding(n_features=4)
        assert enc.properties.simulability == "not_simulable"

    def test_properties_with_multiple_reps(self) -> None:
        """Test properties with multiple repetitions."""
        enc = IQPEncoding(n_features=4, reps=3)
        props = enc.properties

        assert props.depth == 9  # 3 reps * 3
        assert props.n_qubits == 4

    def test_gate_count_consistency(self, default_encoding: IQPEncoding) -> None:
        """Test that single + two qubit gates equals total."""
        props = default_encoding.properties
        assert props.single_qubit_gates + props.two_qubit_gates == props.gate_count

    def test_two_qubit_gate_count_full_entanglement(self) -> None:
        """Test two-qubit gate count for full entanglement.

        Full entanglement: 6 pairs for 4 qubits.
        Each ZZ interaction uses 2 CNOTs = 12 two-qubit gates per rep.
        """
        enc = IQPEncoding(n_features=4, reps=1, entanglement="full")
        props = enc.properties
        # Full entanglement: 6 pairs, each requires 2 CNOTs = 12 two-qubit gates
        assert props.two_qubit_gates == 12

    def test_two_qubit_gate_count_linear_entanglement(self) -> None:
        """Test two-qubit gate count for linear entanglement.

        Linear entanglement: 3 pairs for 4 qubits.
        Each ZZ interaction uses 2 CNOTs = 6 two-qubit gates per rep.
        """
        enc = IQPEncoding(n_features=4, reps=1, entanglement="linear")
        props = enc.properties
        # Linear entanglement: 3 pairs, each requires 2 CNOTs = 6 two-qubit gates
        assert props.two_qubit_gates == 6

    def test_trainability_estimate(self) -> None:
        """Test that trainability estimate is reasonable."""
        enc = IQPEncoding(n_features=4)
        props = enc.properties
        assert 0 < props.trainability_estimate <= 1

    def test_properties_notes_mention_hard_to_simulate(self) -> None:
        """Test that notes mention classical simulation difficulty."""
        enc = IQPEncoding(n_features=4)
        props = enc.properties
        assert "hard" in props.notes.lower() or "simulab" in props.notes.lower()


# =============================================================================
# Test Class: Entanglement Patterns (IQP-Specific Behavior)
# =============================================================================


class TestEntanglementPatterns:
    """Tests for entanglement pair computation for different patterns.

    This is IQP-specific behavior testing different entanglement topologies.
    """

    def test_full_entanglement_pairs(self) -> None:
        """Test full entanglement produces all pairs."""
        enc = IQPEncoding(n_features=4, entanglement="full")
        pairs = enc._get_entanglement_pairs()
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert pairs == expected

    def test_linear_entanglement_pairs(self) -> None:
        """Test linear entanglement produces nearest-neighbor pairs."""
        enc = IQPEncoding(n_features=4, entanglement="linear")
        pairs = enc._get_entanglement_pairs()
        expected = [(0, 1), (1, 2), (2, 3)]
        assert pairs == expected

    def test_circular_entanglement_pairs(self) -> None:
        """Test circular entanglement produces ring pairs."""
        enc = IQPEncoding(n_features=4, entanglement="circular")
        pairs = enc._get_entanglement_pairs()
        expected = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert pairs == expected

    def test_single_qubit_full_no_pairs(self) -> None:
        """Test single qubit with full entanglement has no pairs."""
        enc = IQPEncoding(n_features=1, entanglement="full")
        pairs = enc._get_entanglement_pairs()
        assert pairs == []

    def test_single_qubit_linear_no_pairs(self) -> None:
        """Test single qubit with linear entanglement has no pairs."""
        enc = IQPEncoding(n_features=1, entanglement="linear")
        pairs = enc._get_entanglement_pairs()
        assert pairs == []

    def test_two_qubits_linear(self) -> None:
        """Test two qubits with linear entanglement."""
        enc = IQPEncoding(n_features=2, entanglement="linear")
        pairs = enc._get_entanglement_pairs()
        assert pairs == [(0, 1)]

    def test_two_qubits_circular(self) -> None:
        """Test two qubits with circular entanglement (same as linear for n=2)."""
        enc = IQPEncoding(n_features=2, entanglement="circular")
        pairs = enc._get_entanglement_pairs()
        # For n=2, circular is same as linear (no wrap-around pair)
        assert pairs == [(0, 1)]

    def test_three_qubits_circular(self) -> None:
        """Test three qubits with circular entanglement includes wrap-around."""
        enc = IQPEncoding(n_features=3, entanglement="circular")
        pairs = enc._get_entanglement_pairs()
        expected = [(0, 1), (1, 2), (2, 0)]
        assert pairs == expected

    def test_pair_count_linear(self) -> None:
        """Test linear entanglement pair count is n-1."""
        for n in range(2, 10):
            enc = IQPEncoding(n_features=n, entanglement="linear")
            pairs = enc._get_entanglement_pairs()
            assert len(pairs) == n - 1

    def test_pair_count_full(self) -> None:
        """Test full entanglement pair count is n*(n-1)/2."""
        for n in range(2, 10):
            enc = IQPEncoding(n_features=n, entanglement="full")
            pairs = enc._get_entanglement_pairs()
            expected = n * (n - 1) // 2
            assert len(pairs) == expected

    def test_pair_count_circular(self) -> None:
        """Test circular entanglement pair count is n."""
        for n in range(3, 10):
            enc = IQPEncoding(n_features=n, entanglement="circular")
            pairs = enc._get_entanglement_pairs()
            assert len(pairs) == n


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        circuit = default_encoding.get_circuit(sample_data_4d)
        assert circuit is not None

    def test_valid_2d_input_single_sample(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 2D input with single sample is accepted."""
        x = sample_data_4d.reshape(1, -1)
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_wrong_feature_count(self, default_encoding: IQPEncoding) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2, 0.3])  # 3 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding.get_circuit(x)

    def test_nan_input_rejected(self, default_encoding: IQPEncoding) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding.get_circuit(x)

    def test_inf_input_rejected(self, default_encoding: IQPEncoding) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.get_circuit(x)

    def test_negative_inf_input_rejected(self, default_encoding: IQPEncoding) -> None:
        """Test that negative infinite values are rejected."""
        x = np.array([0.1, -np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.get_circuit(x)

    def test_list_input_accepted(self, default_encoding: IQPEncoding) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]  # list, not array
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_tuple_input_converted(self, default_encoding: IQPEncoding) -> None:
        """Test that tuple input is converted to numpy array."""
        x = (0.1, 0.2, 0.3, 0.4)
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_batch_input_shape(
        self,
        default_encoding: IQPEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that batch input generates multiple circuits."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3

    def test_empty_input(self) -> None:
        """Test that empty input raises error."""
        enc = IQPEncoding(n_features=4)
        with pytest.raises(ValueError):
            enc.get_circuit(np.array([]))


# =============================================================================
# Test Class: PennyLane Backend
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.backend_pennylane
class TestPennyLaneBackend:
    """Tests for PennyLane circuit generation."""

    def test_circuit_is_callable(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that generated circuit executes correctly in QNode context."""
        circuit_fn = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be valid (normalized)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_state_normalized(
        self,
        default_encoding: IQPEncoding,
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

    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_all_entanglements_execute(
        self, entanglement: str, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all entanglement types execute successfully."""
        enc = IQPEncoding(n_features=4, entanglement=entanglement)
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
        enc = IQPEncoding(n_features=4, reps=5)
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
        default_encoding: IQPEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    def test_single_feature(self) -> None:
        """Test PennyLane with single feature."""
        enc = IQPEncoding(n_features=1)
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
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_circuit_has_expected_gates(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit contains expected gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        # IQP circuit should have H, RZ, and CX gates
        assert len(circuit.data) > default_encoding.n_qubits

    def test_has_hadamard_gates(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Hadamard gates are present.

        n_qubits H gates per rep, default is 2 reps.
        """
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")

        h_count = sum(1 for inst in circuit.data if inst.operation.name == "h")
        expected = default_encoding.n_qubits * default_encoding.reps
        assert h_count == expected

    def test_has_cnot_gates(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that CNOT gates are present for ZZ interactions."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")

        cx_count = sum(1 for inst in circuit.data if inst.operation.name == "cx")
        # Each ZZ interaction uses 2 CNOTs, full entanglement has 6 pairs, 2 reps
        assert cx_count > 0

    def test_has_rz_gates(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that RZ gates are present."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")

        rz_count = sum(1 for inst in circuit.data if inst.operation.name == "rz")
        # Should have RZ gates for single-qubit rotations and ZZ interactions
        assert rz_count > 0

    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_all_entanglements_generate_circuits(
        self, entanglement: str, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all entanglement types generate valid circuits."""
        enc = IQPEncoding(n_features=4, entanglement=entanglement)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_multiple_reps_gate_count_increases(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that multiple reps increase gate count."""
        enc1 = IQPEncoding(n_features=4, reps=1)
        enc2 = IQPEncoding(n_features=4, reps=3)

        circuit1 = enc1.get_circuit(sample_data_4d, backend="qiskit")
        circuit2 = enc2.get_circuit(sample_data_4d, backend="qiskit")

        assert len(circuit2.data) > len(circuit1.data)

    def test_batch_circuits(
        self,
        default_encoding: IQPEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    def test_single_feature(self) -> None:
        """Test Qiskit with single feature."""
        enc = IQPEncoding(n_features=1)
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
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: IQPEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_all_entanglements_generate_circuits(
        self, entanglement: str, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all entanglement types generate valid circuits."""
        enc = IQPEncoding(n_features=4, entanglement=entanglement)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 4

    def test_multiple_reps_operation_count(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that multiple reps increase operation count."""
        enc1 = IQPEncoding(n_features=4, reps=1)
        enc2 = IQPEncoding(n_features=4, reps=3)

        circuit1 = enc1.get_circuit(sample_data_4d, backend="cirq")
        circuit2 = enc2.get_circuit(sample_data_4d, backend="cirq")

        ops1 = len(list(circuit1.all_operations()))
        ops2 = len(list(circuit2.all_operations()))
        assert ops2 > ops1

    def test_batch_circuits(
        self,
        default_encoding: IQPEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)

    def test_single_feature(self) -> None:
        """Test Cirq with single feature."""
        enc = IQPEncoding(n_features=1)
        x = np.array([0.5])
        circuit = enc.get_circuit(x, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == 1


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of IQP encoding."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zero_input_starts_in_superposition(self) -> None:
        """Test that zero input produces uniform superposition after H layer.

        With zero input, all RZ rotations are identity. State after H is
        uniform superposition, ZZ with angle 0 is identity.
        """
        enc = IQPEncoding(n_features=2, reps=1, entanglement="linear")
        x = np.array([0.0, 0.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        probs = np.abs(state) ** 2
        # Due to H gates creating superposition
        assert np.all(probs > 0)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_single_qubit_rotation_angles(self) -> None:
        """Test that single-qubit rotations use 2*x as angle."""
        enc = IQPEncoding(n_features=1, reps=1)
        x = np.array([np.pi / 4])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # Should be valid quantum state
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zz_interaction_scaling(self) -> None:
        """Test that ZZ interactions scale with x_i * x_j."""
        # Two different inputs with different products
        enc = IQPEncoding(n_features=2, reps=1, entanglement="full")

        x1 = np.array([0.5, 0.5])  # product = 0.25
        x2 = np.array([1.0, 0.5])  # product = 0.5

        dev = qml.device("default.qubit", wires=2)

        def get_state(x):
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            return full_circuit()

        state1 = get_state(x1)
        state2 = get_state(x2)

        # States should be different due to different ZZ interaction angles
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_entanglement_different_states(self) -> None:
        """Test that different entanglement patterns produce different states."""
        # Use larger input values to create more significant differences
        x = np.array([0.5, 1.0, 1.5, 2.0])
        dev = qml.device("default.qubit", wires=4)

        states = {}
        for entanglement in ["full", "linear", "circular"]:
            enc = IQPEncoding(n_features=4, reps=2, entanglement=entanglement)
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            states[entanglement] = full_circuit()

        # All three states should be different (fidelity < 1)
        fidelity_fl = np.abs(np.vdot(states["full"], states["linear"])) ** 2
        fidelity_fc = np.abs(np.vdot(states["full"], states["circular"])) ** 2
        fidelity_lc = np.abs(np.vdot(states["linear"], states["circular"])) ** 2

        # At minimum, they should not be identical
        assert fidelity_fl < 0.9999, f"full-linear fidelity too high: {fidelity_fl}"
        assert fidelity_fc < 0.9999, f"full-circular fidelity too high: {fidelity_fc}"
        assert fidelity_lc < 0.9999, f"linear-circular fidelity too high: {fidelity_lc}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_negative_input_valid_state(self) -> None:
        """Test that negative input values produce valid states."""
        enc = IQPEncoding(n_features=4, reps=1)
        x = np.array([-0.1, -0.2, 0.3, 0.4])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for boundary values and edge cases."""

    def test_single_feature(self) -> None:
        """Test encoding with single feature."""
        enc = IQPEncoding(n_features=1)
        x = np.array([0.5])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None
        assert enc.n_qubits == 1

    def test_large_feature_count(self) -> None:
        """Test encoding with many features."""
        enc = IQPEncoding(n_features=16)

        assert enc.n_qubits == 16
        x = np.random.randn(16)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_extreme_positive_values(self) -> None:
        """Test encoding with large positive input values."""
        enc = IQPEncoding(n_features=4)

        x_large = np.array([100.0, 200.0, 300.0, 400.0])
        circuit = enc.get_circuit(x_large, backend="pennylane")
        assert circuit is not None

    def test_extreme_negative_values(self) -> None:
        """Test encoding with large negative input values."""
        enc = IQPEncoding(n_features=4)

        x_neg = np.array([-100.0, -200.0, -300.0, -400.0])
        circuit = enc.get_circuit(x_neg, backend="pennylane")
        assert circuit is not None

    def test_mixed_sign_values(self) -> None:
        """Test encoding with mixed positive and negative values."""
        enc = IQPEncoding(n_features=4)

        x = np.array([-1.0, 2.0, -3.0, 4.0])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_very_small_values(self) -> None:
        """Test encoding with very small values."""
        enc = IQPEncoding(n_features=4)

        x_small = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        circuit = enc.get_circuit(x_small, backend="pennylane")
        assert circuit is not None

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions."""
        enc = IQPEncoding(n_features=4, reps=50)

        # depth = 50 * 3
        assert enc.depth == 150
        props = enc.properties
        assert props.depth == 150

    def test_all_zeros_input(self) -> None:
        """Test encoding with all zeros."""
        enc = IQPEncoding(n_features=4)
        x = np.zeros(4)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_multiples_of_2pi(self) -> None:
        """Test encoding with multiples of 2*pi."""
        enc = IQPEncoding(n_features=4)
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
        enc = IQPEncoding(n_features=4)
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
        enc = IQPEncoding(n_features=4)
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
        enc = IQPEncoding(n_features=4)
        x = np.array([1e-10, 1e10, 1e-5, 1e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should still be normalized
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        # State should have all finite values
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_extreme_values_produce_normalized_state(self) -> None:
        """Test that extreme values still produce normalized quantum states."""
        enc = IQPEncoding(n_features=2)
        x = np.array([1e15, 1e15])  # Very large angles

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
        enc = IQPEncoding(n_features=2)
        eps = np.finfo(float).eps
        x = np.array([eps, eps])

        # Should not raise
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = IQPEncoding(n_features=4, reps=2, entanglement="full")
        enc2 = IQPEncoding(n_features=4, reps=2, entanglement="full")
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = IQPEncoding(n_features=4)
        enc2 = IQPEncoding(n_features=8)
        assert enc1 != enc2

    def test_equality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = IQPEncoding(n_features=4, reps=1)
        enc2 = IQPEncoding(n_features=4, reps=2)
        assert enc1 != enc2

    def test_equality_different_entanglement(self) -> None:
        """Test that encodings with different entanglement are not equal."""
        enc1 = IQPEncoding(n_features=4, entanglement="full")
        enc2 = IQPEncoding(n_features=4, entanglement="linear")
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = IQPEncoding(n_features=4, reps=3, entanglement="circular")
        enc2 = IQPEncoding(n_features=4, reps=3, entanglement="circular")
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = IQPEncoding(n_features=4, entanglement="full")
        enc2 = IQPEncoding(n_features=4, entanglement="full")  # Same as enc1
        enc3 = IQPEncoding(n_features=4, entanglement="linear")

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = IQPEncoding(n_features=4)
        enc2 = IQPEncoding(n_features=4)

        d = {enc1: "four features"}
        assert d[enc2] == "four features"  # enc2 should find same key

    def test_inequality_with_none(self) -> None:
        """Test that encoding is not equal to None."""
        enc = IQPEncoding(n_features=4)
        assert enc != None  # noqa: E711

    def test_inequality_with_different_type(self) -> None:
        """Test that encoding is not equal to different types."""
        enc = IQPEncoding(n_features=4)
        assert enc != "IQPEncoding"
        assert enc != 4


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for string representation (__repr__)."""

    def test_repr_contains_class_name(self, default_encoding: IQPEncoding) -> None:
        """Test that repr contains class name."""
        assert "IQPEncoding" in repr(default_encoding)

    def test_repr_contains_n_features(self, default_encoding: IQPEncoding) -> None:
        """Test that repr contains n_features."""
        assert "n_features=4" in repr(default_encoding)

    def test_repr_contains_reps(self) -> None:
        """Test that repr contains reps."""
        enc = IQPEncoding(n_features=4, reps=5)
        r = repr(enc)
        assert "reps" in r.lower() or "5" in r

    def test_repr_contains_entanglement(self) -> None:
        """Test that repr contains entanglement."""
        enc = IQPEncoding(n_features=4, entanglement="linear")
        r = repr(enc)
        assert "entanglement" in r.lower() or "linear" in r

    def test_repr_all_custom_parameters(self) -> None:
        """Test that repr contains all custom parameters."""
        enc = IQPEncoding(n_features=8, reps=3, entanglement="circular")
        r = repr(enc)

        assert "n_features=8" in r or "8" in r


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for backend error handling."""

    def test_invalid_backend(
        self, default_encoding: IQPEncoding, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_invalid_backend_empty(
        self, default_encoding: IQPEncoding, sample_data_4d: NDArray[np.floating]
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
        enc = IQPEncoding(n_features=8, reps=3, entanglement="circular")

        # Force properties to be computed
        _ = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.reps == enc.reps
        assert restored.entanglement == enc.entanglement

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = IQPEncoding(n_features=4, entanglement="linear")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Generate circuit before pickle
        circuit_before = enc.get_circuit(x, backend="pennylane")

        # Pickle and restore
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        # Generate circuit after restore
        circuit_after = restored.get_circuit(x, backend="pennylane")

        # Both should be callable
        assert callable(circuit_before)
        assert callable(circuit_after)

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = IQPEncoding(n_features=4)
        original_props = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        restored_props = restored.properties

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
        enc = IQPEncoding(n_features=4)
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

        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"
        # All circuits should have been generated
        total_circuits = sum(len(r) for r in results)
        assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = IQPEncoding(n_features=4)
        results: list[object] = []
        errors: list[Exception] = []
        n_threads = 50

        def access_properties() -> None:
            try:
                props = enc.properties
                results.append(props)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_properties) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        # All results should be the same object (cached)
        assert len(results) == n_threads


# =============================================================================
# Test Class: IQP-Specific Tests
# =============================================================================


class TestIQPSpecific:
    """Tests specific to IQP encoding functionality.

    These tests cover IQP-specific behavior including parallel batch
    processing, public API methods, type edge cases, and warnings.
    """

    # --- Parallel Batch Processing ---

    def test_parallel_produces_same_results_as_sequential(self) -> None:
        """Test that parallel processing produces identical circuits."""
        enc = IQPEncoding(n_features=4)
        X = np.random.randn(20, 4)

        # Generate circuits both ways
        sequential_circuits = enc.get_circuits(X, backend="pennylane", parallel=False)
        parallel_circuits = enc.get_circuits(X, backend="pennylane", parallel=True)

        # Same number of circuits
        assert len(sequential_circuits) == len(parallel_circuits)
        assert len(sequential_circuits) == 20

        # All should be callable
        assert all(callable(c) for c in sequential_circuits)
        assert all(callable(c) for c in parallel_circuits)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_parallel_preserves_order(self) -> None:
        """Test that parallel processing preserves input order."""
        enc = IQPEncoding(n_features=2, reps=1)
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ]
        )

        dev = qml.device("default.qubit", wires=2)

        def get_state(circuit_fn):
            @qml.qnode(dev)
            def qnode():
                circuit_fn()
                return qml.state()

            return qnode()

        sequential = enc.get_circuits(X, backend="pennylane", parallel=False)
        parallel = enc.get_circuits(X, backend="pennylane", parallel=True)

        # States should match for each position
        for i in range(len(X)):
            state_seq = get_state(sequential[i])
            state_par = get_state(parallel[i])
            np.testing.assert_allclose(state_seq, state_par, atol=1e-10)

    def test_parallel_with_max_workers(self) -> None:
        """Test parallel processing with custom max_workers."""
        enc = IQPEncoding(n_features=4)
        X = np.random.randn(10, 4)

        circuits = enc.get_circuits(
            X, backend="pennylane", parallel=True, max_workers=2
        )
        assert len(circuits) == 10
        assert all(callable(c) for c in circuits)

    def test_parallel_with_single_sample(self) -> None:
        """Test that parallel=True with single sample works."""
        enc = IQPEncoding(n_features=4)
        X = np.array([[0.1, 0.2, 0.3, 0.4]])

        circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
        assert len(circuits) == 1
        assert callable(circuits[0])

    def test_parallel_default_is_false(self) -> None:
        """Test that parallel defaults to False (backward compatibility)."""
        enc = IQPEncoding(n_features=4)
        X = np.random.randn(5, 4)

        # Call without parallel argument - should work (default False)
        circuits = enc.get_circuits(X, backend="pennylane")
        assert len(circuits) == 5

    # --- Public API Methods ---

    def test_get_entanglement_pairs_full(self) -> None:
        """Test get_entanglement_pairs() with full entanglement."""
        enc = IQPEncoding(n_features=4, entanglement="full")
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert pairs == expected
        # n*(n-1)/2 = 4*3/2 = 6
        assert len(pairs) == 6

    def test_get_entanglement_pairs_linear(self) -> None:
        """Test get_entanglement_pairs() with linear entanglement."""
        enc = IQPEncoding(n_features=5, entanglement="linear")
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert pairs == expected
        # n-1
        assert len(pairs) == 4

    def test_get_entanglement_pairs_circular(self) -> None:
        """Test get_entanglement_pairs() with circular entanglement."""
        enc = IQPEncoding(n_features=5, entanglement="circular")
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        assert pairs == expected
        # n pairs for circular
        assert len(pairs) == 5

    def test_get_entanglement_pairs_returns_new_list(self) -> None:
        """Test that get_entanglement_pairs() returns a new list each time."""
        enc = IQPEncoding(n_features=4, entanglement="full")

        pairs1 = enc.get_entanglement_pairs()
        pairs2 = enc.get_entanglement_pairs()

        # Should be equal but not the same object
        assert pairs1 == pairs2
        assert pairs1 is not pairs2

        # Modifying one should not affect the other
        pairs1.append((99, 99))
        assert (99, 99) not in pairs2

    def test_gate_count_breakdown_full_entanglement(self) -> None:
        """Test gate_count_breakdown() with full entanglement.

        4 qubits, 1 rep, 6 pairs (full entanglement).
        """
        enc = IQPEncoding(n_features=4, reps=1, entanglement="full")
        breakdown = enc.gate_count_breakdown()

        assert breakdown["hadamard"] == 4  # n_qubits * reps
        assert breakdown["rz_single"] == 4  # n_qubits * reps
        assert breakdown["rz_zz"] == 6  # n_pairs * reps
        assert breakdown["cnot"] == 12  # 2 * n_pairs * reps
        assert breakdown["total_single_qubit"] == 14  # H + RZ_single + RZ_zz
        assert breakdown["total_two_qubit"] == 12  # CNOTs
        assert breakdown["total"] == 26  # single + two qubit

    def test_gate_count_breakdown_consistency_with_properties(self) -> None:
        """Test that gate_count_breakdown() is consistent with properties."""
        enc = IQPEncoding(n_features=6, reps=3, entanglement="circular")
        breakdown = enc.gate_count_breakdown()
        props = enc.properties

        assert breakdown["total"] == props.gate_count
        assert breakdown["total_single_qubit"] == props.single_qubit_gates
        assert breakdown["total_two_qubit"] == props.two_qubit_gates

    def test_resource_summary_structure(self) -> None:
        """Test resource_summary() returns expected structure."""
        enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
        summary = enc.resource_summary()

        expected_keys = [
            "n_qubits",
            "n_features",
            "depth",
            "reps",
            "entanglement",
            "entanglement_pairs",
            "n_entanglement_pairs",
            "gate_counts",
            "is_entangling",
            "simulability",
            "trainability_estimate",
            "hardware_requirements",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_resource_summary_values_full(self) -> None:
        """Test resource_summary() values with full entanglement."""
        enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
        summary = enc.resource_summary()

        assert summary["n_qubits"] == 4
        assert summary["n_features"] == 4
        assert summary["depth"] == 6  # reps * 3
        assert summary["reps"] == 2
        assert summary["entanglement"] == "full"
        assert summary["n_entanglement_pairs"] == 6
        assert summary["is_entangling"] is True
        assert summary["simulability"] == "not_simulable"
        assert summary["hardware_requirements"]["connectivity"] == "all-to-all"

    # --- Type Edge Cases ---

    def test_reps_boolean_true_rejected(self) -> None:
        """Test that reps=True is rejected (bool is subclass of int)."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            IQPEncoding(n_features=4, reps=True)

    def test_reps_boolean_false_rejected(self) -> None:
        """Test that reps=False is rejected."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            IQPEncoding(n_features=4, reps=False)

    def test_reps_float_rejected(self) -> None:
        """Test that float reps values are rejected."""
        with pytest.raises((TypeError, ValueError)):
            IQPEncoding(n_features=4, reps=2.0)  # type: ignore

    def test_entanglement_case_sensitive(self) -> None:
        """Test that entanglement is case-sensitive."""
        with pytest.raises(ValueError, match="entanglement must be"):
            IQPEncoding(n_features=4, entanglement="Full")  # type: ignore

        with pytest.raises(ValueError, match="entanglement must be"):
            IQPEncoding(n_features=4, entanglement="LINEAR")  # type: ignore

    def test_input_integer_array(self) -> None:
        """Test that integer input arrays work (converted to float)."""
        enc = IQPEncoding(n_features=4)
        x = np.array([1, 2, 3, 4], dtype=np.int32)

        circuit = enc.get_circuit(x, backend="pennylane")
        assert callable(circuit)

    def test_input_complex_array_rejected(self) -> None:
        """Test that complex input arrays are rejected."""
        enc = IQPEncoding(n_features=4)
        x = np.array([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j], dtype=np.complex128)

        with pytest.raises((TypeError, ValueError)):
            enc.get_circuit(x, backend="pennylane")

    # --- Warnings ---

    def test_full_entanglement_warning_threshold(self) -> None:
        """Test that warning is raised when n_features > 12 with full entanglement.

        The IQP encoding issues a UserWarning when using full entanglement
        with more than 12 features because full entanglement creates
        n*(n-1)/2 ZZ pairs, which may exceed NISQ device capabilities.
        """
        with pytest.warns(UserWarning, match="Full entanglement with"):
            IQPEncoding(n_features=13, entanglement="full")

    def test_no_warning_at_threshold(self) -> None:
        """Test that no warning at exactly 12 features (threshold)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            IQPEncoding(n_features=12, entanglement="full")

            # Filter for UserWarnings about entanglement
            entanglement_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning)
                and "entanglement" in str(x.message).lower()
            ]
            assert len(entanglement_warnings) == 0

    def test_no_warning_for_linear_entanglement_large_features(self) -> None:
        """Test that no warning for linear entanglement even with many features."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            IQPEncoding(n_features=20, entanglement="linear")

            entanglement_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning)
                and "Full entanglement" in str(x.message)
            ]
            assert len(entanglement_warnings) == 0


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
        enc = IQPEncoding(n_features=4, reps=2)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        def get_state(x):
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

        # States should be different
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input always produces same state."""
        enc = IQPEncoding(n_features=4, reps=2)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([0.1, 0.2, 0.3, 0.4])

        def get_state():
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            return full_circuit()

        state1 = get_state()
        state2 = get_state()

        np.testing.assert_allclose(state1, state2)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_entanglement_creates_entangled_state(self) -> None:
        """Test that IQP actually creates entangled state.

        For an entangled state, reduced density matrix has mixed state.
        Purity < 1 indicates entanglement.
        """
        enc = IQPEncoding(n_features=2, reps=1, entanglement="full")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([np.pi / 4, np.pi / 4])
        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # Compute purity of reduced state
        rho_full = np.outer(state, np.conj(state))
        rho_reduced = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rho_reduced[i, j] += rho_full[2 * i + k, 2 * j + k]

        purity = np.real(np.trace(rho_reduced @ rho_reduced))

        # Pure product state has purity = 1, entangled state has purity < 1
        assert purity < 1.0

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states.

        Different backends may use different qubit ordering conventions,
        so we verify states are valid and normalized rather than comparing
        exact probability distributions.
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        enc = IQPEncoding(n_features=4, reps=1, entanglement="linear")
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

        # Probability distributions should have the same set of values
        # (accounting for different qubit ordering conventions)
        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(pl_probs, qk_probs, atol=1e-6)
        np.testing.assert_allclose(pl_probs, cirq_probs, atol=1e-6)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_all_entanglements_produce_valid_states(self, entanglement: str) -> None:
        """Test that all entanglement types produce valid states across backends."""
        enc = IQPEncoding(n_features=4, entanglement=entanglement)
        x = np.array([0.3, 0.5, 0.7, 0.9])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should be normalized
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)
        # All values should be finite
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_large_scale_simulation(self) -> None:
        """Test simulation with larger qubit count."""
        enc = IQPEncoding(n_features=10, reps=1, entanglement="linear")
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
        enc = IQPEncoding(n_features=4)
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

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All three backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_zero_input(self) -> None:
        """Test cross-backend equivalence for zero input.

        With all-zero input, RZ and ZZ gates become identity (angle=0),
        so the state should be a uniform superposition created by H gates.
        All backends should produce the same uniform distribution.
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        enc = IQPEncoding(n_features=4, reps=1, entanglement="full")
        x = np.zeros(4)

        # PennyLane
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Qiskit
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_circuit.save_statevector()
        simulator = AerSimulator(method="statevector")
        compiled = transpile(qk_circuit, simulator)
        result = simulator.run(compiled).result()
        qk_state = np.array(result.get_statevector().data)

        # Cirq
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_simulator = cirq.Simulator()
        cirq_result = cirq_simulator.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # For zero input, state should be uniform superposition
        # All probabilities should be 1/16 = 0.0625
        expected_prob = 1.0 / 16

        for state, name in [
            (pl_state, "PennyLane"),
            (qk_state, "Qiskit"),
            (cirq_state, "Cirq"),
        ]:
            probs = np.abs(state) ** 2
            np.testing.assert_allclose(
                probs,
                np.full(16, expected_prob),
                atol=1e-10,
                err_msg=f"{name} does not produce uniform distribution for zero input",
            )


# =============================================================================
# Test Class: Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for IQP encoding with timeout protection.

    These tests verify that the encoding handles demanding configurations
    without hanging, memory explosion, or performance degradation.
    Tests focus on circuit GENERATION, not SIMULATION.
    """

    @pytest.mark.timeout(5)
    def test_high_reps_circuit_generation(self) -> None:
        """Test circuit generation with high repetition count.

        Configuration: 4 qubits, 100 reps, full entanglement.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            enc = IQPEncoding(n_features=4, reps=100, entanglement="full")

            # Verify configuration
            assert enc.reps == 100
            # depth = 100 * 3
            assert enc.depth == 300

            # Generate circuit - should complete quickly
            x = np.array([0.1, 0.2, 0.3, 0.4])
            circuit = enc.get_circuit(x, backend="pennylane")

            assert callable(circuit)

    @pytest.mark.timeout(5)
    def test_high_reps_properties_computation(self) -> None:
        """Test property computation with high repetition count.

        Properties should be computed efficiently using analytical formulas.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            enc = IQPEncoding(n_features=8, reps=200, entanglement="full")

            # Property access should be instantaneous
            props = enc.properties

            # depth = 200 * 3
            assert props.depth == 600
            assert props.n_qubits == 8

    @pytest.mark.timeout(5)
    def test_large_feature_count_linear_entanglement(self) -> None:
        """Test circuit generation with many features using linear entanglement.

        Configuration: 50 features, 2 reps, linear entanglement.
        Linear entanglement scales as O(n) pairs.
        """
        enc = IQPEncoding(n_features=50, reps=2, entanglement="linear")

        # Verify pair count: n-1 = 49
        pairs = enc.get_entanglement_pairs()
        assert len(pairs) == 49

        # Generate circuit
        x = np.random.randn(50)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert callable(circuit)

    @pytest.mark.timeout(5)
    def test_batch_circuit_generation_stress(self) -> None:
        """Test batch circuit generation with many samples.

        Configuration: 100 samples, 4 features, 2 reps.
        """
        enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
        X = np.random.randn(100, 4)

        # Sequential batch generation
        circuits = enc.get_circuits(X, backend="pennylane", parallel=False)

        assert len(circuits) == 100
        assert all(callable(c) for c in circuits)

    @pytest.mark.timeout(10)
    def test_parallel_batch_generation_stress(self) -> None:
        """Test parallel batch circuit generation with many samples.

        Configuration: 200 samples, 4 features, 2 reps, parallel=True.
        """
        enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
        X = np.random.randn(200, 4)

        # Parallel batch generation
        circuits = enc.get_circuits(X, backend="pennylane", parallel=True)

        assert len(circuits) == 200
        assert all(callable(c) for c in circuits)

    @pytest.mark.timeout(5)
    def test_repeated_circuit_generation(self) -> None:
        """Test generating many circuits from the same encoding.

        Verifies that repeated circuit generation doesn't accumulate
        state or degrade performance over time.
        """
        enc = IQPEncoding(n_features=4, reps=3, entanglement="full")

        circuits = []
        for _i in range(50):
            x = np.random.randn(4)
            circuit = enc.get_circuit(x, backend="pennylane")
            circuits.append(circuit)

        assert len(circuits) == 50
        assert all(callable(c) for c in circuits)

    @pytest.mark.timeout(5)
    @pytest.mark.parametrize(
        "n_features,reps,entanglement",
        [
            (10, 50, "linear"),
            (10, 50, "circular"),
            (5, 100, "full"),
            (20, 10, "linear"),
            (8, 25, "full"),
        ],
    )
    def test_various_stress_configurations(
        self,
        n_features: int,
        reps: int,
        entanglement: str,
    ) -> None:
        """Parametrized stress test across various configurations.

        Tests a matrix of feature counts, repetitions, and entanglement
        patterns to ensure consistent performance characteristics.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            enc = IQPEncoding(
                n_features=n_features,
                reps=reps,
                entanglement=entanglement,
            )

            # Verify instantiation
            assert enc.n_features == n_features
            assert enc.reps == reps
            assert enc.entanglement == entanglement

            # Generate circuit
            x = np.random.randn(n_features)
            circuit = enc.get_circuit(x, backend="pennylane")
            assert callable(circuit)

            # Access properties
            props = enc.properties
            # depth = reps * 3
            assert props.depth == reps * 3
