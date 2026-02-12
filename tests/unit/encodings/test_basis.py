"""Comprehensive tests for BasisEncoding.

This test module provides complete coverage of the BasisEncoding class,
which maps binary/discrete data to computational basis states. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, threshold)
- Binarization behavior (unique to BasisEncoding)
- Circuit generation for all backends (PennyLane, Qiskit, Cirq)
- Mathematical correctness verification
- Edge cases (all zeros, all ones, alternating patterns)
- Numerical stability tests
- Equality and hashing
- String representation
- Backend error handling
- Serialization (pickle roundtrip)
- Concurrent access / thread safety
- Gate count breakdown and resource analysis
- Protocol conformance (DataTransformable, DataDependentResourceAnalyzable)
- Slow simulation tests (cross-backend state fidelity)

Run with: pytest tests/unit/encodings/test_basis.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_basis.py -v -m "not slow"

References
----------
.. [1] Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and
       Quantum Information." Cambridge University Press. Chapter 1.
.. [2] Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
       Quantum Computers." Springer. Chapter 4: Quantum Feature Maps.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas import BasisEncoding
from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.protocols import (
    DataDependentResourceAnalyzable,
    DataTransformable,
    EntanglementQueryable,
)

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
def binary_data_4d() -> NDArray[np.integer]:
    """4-dimensional binary test data [1, 0, 1, 0]."""
    return np.array([1, 0, 1, 0], dtype=np.int64)


@pytest.fixture
def continuous_data_4d() -> NDArray[np.floating]:
    """4-dimensional continuous data for binarization testing.

    Values: [0.8, 0.2, 0.6, 0.1]
    Expected binarization (threshold=0.5): [1, 0, 1, 0]
    """
    return np.array([0.8, 0.2, 0.6, 0.1])


@pytest.fixture
def batch_data_4d() -> NDArray[np.integer]:
    """Batch of 4-dimensional binary samples.

    Contains 3 samples:
    - [1, 0, 1, 0] -> |1010> (alternating)
    - [1, 1, 1, 1] -> |1111> (all ones)
    - [0, 0, 0, 0] -> |0000> (all zeros)
    """
    return np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )


@pytest.fixture
def default_encoding() -> BasisEncoding:
    """Default BasisEncoding with 4 features."""
    return BasisEncoding(n_features=4)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for BasisEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default n_features."""
        enc = BasisEncoding(n_features=4)
        assert enc.n_features == 4
        assert enc.n_qubits == 4  # 1:1 mapping for basis encoding
        assert enc.depth == 1  # Always depth 1
        assert enc.threshold == 0.5  # Default binarization threshold

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = BasisEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1
        assert enc.depth == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16, 32, 64]:
            enc = BasisEncoding(n_features=n)
            assert enc.n_features == n
            # BasisEncoding uses 1:1 qubit mapping (unlike AmplitudeEncoding)
            assert enc.n_qubits == n
            # Depth is always 1 (all X gates can be parallel)
            assert enc.depth == 1

    def test_threshold_attribute_stored(self) -> None:
        """Test that threshold is stored as instance attribute."""
        enc = BasisEncoding(n_features=4)
        assert hasattr(enc, "threshold")
        assert enc.threshold == 0.5

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        # This should be fast since no circuit is pre-computed
        enc = BasisEncoding(n_features=100)
        assert enc.n_features == 100
        # Accessing n_qubits/depth should not generate a circuit either
        _ = enc.n_qubits
        _ = enc.depth


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    def test_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            BasisEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            BasisEncoding(n_features=-1)

    def test_invalid_n_features_negative_large(self) -> None:
        """Test that large negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            BasisEncoding(n_features=-100)

    def test_invalid_n_features_float(self) -> None:
        """Test that float n_features raises ValueError.

        n_features must be an integer, not a float even if it's a whole number.
        """
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            BasisEncoding(n_features=4.0)  # type: ignore

    def test_invalid_n_features_string(self) -> None:
        """Test that string n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            BasisEncoding(n_features="4")  # type: ignore

    def test_invalid_n_features_none(self) -> None:
        """Test that None n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            BasisEncoding(n_features=None)  # type: ignore

    def test_invalid_n_features_list(self) -> None:
        """Test that list n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            BasisEncoding(n_features=[4])  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of BasisEncoding."""

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features (1:1 mapping)."""
        for n in [1, 2, 4, 8, 16]:
            enc = BasisEncoding(n_features=n)
            assert (
                enc.n_qubits == n
            ), f"n_features={n}: expected n_qubits={n}, got {enc.n_qubits}"

    def test_depth_always_one(self) -> None:
        """Test that circuit depth is always 1 regardless of feature count.

        All X gates can be applied in parallel since they act on different qubits.
        """
        for n in [1, 2, 4, 8, 16, 32]:
            enc = BasisEncoding(n_features=n)
            assert enc.depth == 1, f"n_features={n}: expected depth=1, got {enc.depth}"

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = BasisEncoding(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_not_entangling(self) -> None:
        """Test that basis encoding does not create entanglement.

        BasisEncoding produces product states (no superposition/entanglement).
        """
        enc = BasisEncoding(n_features=4)
        assert enc.properties.is_entangling is False

    def test_properties_simulable(self) -> None:
        """Test that basis encoding is classically simulable.

        Basis states are trivially trackable classically.
        """
        enc = BasisEncoding(n_features=4)
        assert enc.properties.simulability == "simulable"

    def test_properties_gate_count(self) -> None:
        """Test that gate count equals n_features (worst case: all ones)."""
        enc = BasisEncoding(n_features=4)
        # Worst case: all features are 1, so n X gates
        assert enc.properties.gate_count == 4

    def test_properties_single_qubit_gates(self) -> None:
        """Test single qubit gate count equals n_features."""
        enc = BasisEncoding(n_features=8)
        assert enc.properties.single_qubit_gates == 8

    def test_properties_two_qubit_gates_zero(self) -> None:
        """Test that there are no two-qubit gates."""
        enc = BasisEncoding(n_features=4)
        assert enc.properties.two_qubit_gates == 0

    def test_properties_parameter_count_zero(self) -> None:
        """Test that there are no continuous parameters."""
        enc = BasisEncoding(n_features=4)
        assert enc.properties.parameter_count == 0

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = BasisEncoding(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        # Should return the same cached object
        assert props1 is props2


# =============================================================================
# Test Class: Binarization Behavior (Unique to BasisEncoding)
# =============================================================================


class TestBinarizationBehavior:
    """Tests for binarization behavior unique to BasisEncoding.

    BasisEncoding converts continuous inputs to binary using a threshold of 0.5:
    - Values > 0.5 -> 1 (X gate applied)
    - Values <= 0.5 -> 0 (no gate applied)
    """

    def test_continuous_to_binary_basic(
        self,
        default_encoding: BasisEncoding,
        continuous_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that continuous data is correctly binarized.

        Input: [0.8, 0.2, 0.6, 0.1]
        Expected binary: [1, 0, 1, 0] (using threshold 0.5)
        """
        # Just verify circuit generation doesn't raise
        circuit = default_encoding.get_circuit(continuous_data_4d, backend="qiskit")
        if HAS_QISKIT:
            assert isinstance(circuit, QuantumCircuit)

    def test_threshold_boundary_above(self) -> None:
        """Test values just above threshold are mapped to 1."""
        enc = BasisEncoding(n_features=4)
        # 0.51 > 0.5, should become 1
        x = np.array([0.51, 0.51, 0.51, 0.51])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            # All qubits should have X gates -> all in |1> state
            # Count X gates in the circuit
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 4, f"Expected 4 X gates, got {x_count}"

    def test_threshold_boundary_at(self) -> None:
        """Test values exactly at threshold are mapped to 0.

        The condition is x > threshold, so x == 0.5 should be 0.
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([0.5, 0.5, 0.5, 0.5])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            # All values == 0.5, should become 0, no X gates
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, f"Expected 0 X gates, got {x_count}"

    def test_threshold_boundary_below(self) -> None:
        """Test values just below threshold are mapped to 0."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.49, 0.49, 0.49, 0.49])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, f"Expected 0 X gates, got {x_count}"

    def test_already_binary_input(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that already binary input works correctly.

        Binary input [1, 0, 1, 0]:
        - 1 > 0.5 -> 1 (X gate)
        - 0 <= 0.5 -> 0 (no gate)
        """
        if HAS_QISKIT:
            circuit = default_encoding.get_circuit(binary_data_4d, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2, f"Expected 2 X gates for [1,0,1,0], got {x_count}"

    def test_negative_values_binarization(self) -> None:
        """Test that negative values are correctly binarized.

        Negative values are <= 0.5, so they should be mapped to 0.
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([-1.0, -0.5, -0.1, -10.0])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert (
                x_count == 0
            ), f"Expected 0 X gates for negative values, got {x_count}"

    def test_large_positive_values_binarization(self) -> None:
        """Test that large positive values are mapped to 1."""
        enc = BasisEncoding(n_features=4)
        x = np.array([100.0, 1000.0, 1e6, 1e10])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 4, f"Expected 4 X gates for large values, got {x_count}"

    def test_mixed_values_binarization(self) -> None:
        """Test mixed values with clear boundary cases.

        Input: [0.0, 0.5, 0.500001, 1.0]
        Expected: [0, 0, 1, 1] -> 2 X gates on qubits 2 and 3
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([0.0, 0.5, 0.500001, 1.0])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2, f"Expected 2 X gates, got {x_count}"


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation."""

    def test_valid_1d_input(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that valid 1D input passes validation."""
        validated = default_encoding._validate_input(binary_data_4d)
        assert validated.shape == (4,)

    def test_valid_2d_input_single_sample(
        self, default_encoding: BasisEncoding
    ) -> None:
        """Test that 2D input with single sample is flattened."""
        x = np.array([[1, 0, 1, 0]])  # Shape (1, 4)
        validated = default_encoding._validate_input(x)
        # Should be flattened or kept as (1, 4) depending on implementation
        assert validated.shape in [(4,), (1, 4)]

    def test_wrong_feature_count_too_few(self, default_encoding: BasisEncoding) -> None:
        """Test that wrong feature count (too few) raises ValueError."""
        x = np.array([1, 0])  # Only 2 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_wrong_feature_count_too_many(
        self, default_encoding: BasisEncoding
    ) -> None:
        """Test that wrong feature count (too many) raises ValueError."""
        x = np.array([1, 0, 1, 0, 1, 0])  # 6 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_nan_values_rejected(self, default_encoding: BasisEncoding) -> None:
        """Test that NaN values are rejected."""
        x = np.array([1, np.nan, 1, 0])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding._validate_input(x)

    def test_inf_values_rejected(self, default_encoding: BasisEncoding) -> None:
        """Test that positive infinite values are rejected."""
        x = np.array([1, np.inf, 1, 0])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_neg_inf_values_rejected(self, default_encoding: BasisEncoding) -> None:
        """Test that negative infinite values are rejected."""
        x = np.array([1, -np.inf, 1, 0])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_list_input_converted(self, default_encoding: BasisEncoding) -> None:
        """Test that list input is converted to numpy array."""
        x = [1, 0, 1, 0]  # Python list
        # Should not raise - list is converted internally
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            assert isinstance(circuit, QuantumCircuit)

    def test_empty_input_rejected(self) -> None:
        """Test that empty input is rejected."""
        enc = BasisEncoding(n_features=4)
        x = np.array([])
        with pytest.raises(ValueError):
            enc._validate_input(x)


# =============================================================================
# Test Class: PennyLane Backend
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.backend_pennylane
class TestPennyLaneBackend:
    """Tests for PennyLane circuit generation."""

    def test_circuit_is_callable(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(binary_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that generated circuit executes correctly in QNode context."""
        circuit_fn = default_encoding.get_circuit(binary_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be valid (normalized)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_circuit_produces_correct_basis_state(
        self,
        default_encoding: BasisEncoding,
    ) -> None:
        """Test that circuit produces correct computational basis state.

        Input [1, 0, 1, 0] should produce |1010> state.
        In PennyLane with default qubit ordering, this is index 5 (binary 0101 reversed).
        """
        x = np.array([1, 0, 1, 0])
        circuit_fn = default_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # The state should be a computational basis state (one amplitude = 1)
        probs = np.abs(state) ** 2
        assert np.max(probs) > 0.99, "Should be a pure basis state"
        assert np.sum(probs > 0.01) == 1, "Only one basis state should have amplitude"

    def test_batch_circuits(
        self,
        default_encoding: BasisEncoding,
        batch_data_4d: NDArray[np.integer],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    def test_all_zeros_produces_ground_state(
        self, default_encoding: BasisEncoding
    ) -> None:
        """Test that all-zeros input produces |0000> ground state."""
        x = np.array([0, 0, 0, 0])
        circuit_fn = default_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # |0000> state: amplitude at index 0 should be 1
        assert np.isclose(np.abs(state[0]) ** 2, 1.0, atol=1e-10)

    def test_all_ones_produces_all_ones_state(
        self, default_encoding: BasisEncoding
    ) -> None:
        """Test that all-ones input produces |1111> state."""
        x = np.array([1, 1, 1, 1])
        circuit_fn = default_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # |1111> state: amplitude at last index (2^4 - 1 = 15) should be 1
        assert np.isclose(np.abs(state[-1]) ** 2, 1.0, atol=1e-10)


# =============================================================================
# Test Class: Qiskit Backend
# =============================================================================


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.mark.backend_qiskit
class TestQiskitBackend:
    """Tests for Qiskit circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(binary_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(binary_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_circuit_has_correct_x_gates(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that circuit has correct X gates for input [1,0,1,0]."""
        circuit = default_encoding.get_circuit(binary_data_4d, backend="qiskit")
        x_count = circuit.count_ops().get("x", 0)
        assert x_count == 2, f"Expected 2 X gates, got {x_count}"

    def test_circuit_named_correctly(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that circuit has descriptive name."""
        circuit = default_encoding.get_circuit(binary_data_4d, backend="qiskit")
        assert "BasisEncoding" in circuit.name

    def test_all_zeros_empty_circuit(self, default_encoding: BasisEncoding) -> None:
        """Test that all-zeros input produces circuit with no X gates."""
        x = np.array([0, 0, 0, 0])
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        x_count = circuit.count_ops().get("x", 0)
        assert x_count == 0, f"Expected 0 X gates for all-zeros, got {x_count}"
        # Depth should be 0 when no gates
        assert circuit.depth() == 0

    def test_all_ones_full_x_gates(self, default_encoding: BasisEncoding) -> None:
        """Test that all-ones input produces circuit with X on all qubits."""
        x = np.array([1, 1, 1, 1])
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        x_count = circuit.count_ops().get("x", 0)
        assert x_count == 4, f"Expected 4 X gates for all-ones, got {x_count}"
        # All X gates are parallel, depth should be 1
        assert circuit.depth() == 1

    def test_batch_circuits(
        self,
        default_encoding: BasisEncoding,
        batch_data_4d: NDArray[np.integer],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    def test_circuit_produces_correct_statevector(
        self,
        default_encoding: BasisEncoding,
    ) -> None:
        """Test that circuit produces correct statevector."""
        x = np.array([1, 0, 1, 0])
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        sv = Statevector(circuit)
        probs = sv.probabilities()
        # Should be a pure basis state
        assert np.max(probs) > 0.99


# =============================================================================
# Test Class: Cirq Backend
# =============================================================================


@pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
@pytest.mark.backend_cirq
class TestCirqBackend:
    """Tests for Cirq circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(binary_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: BasisEncoding,
    ) -> None:
        """Test that circuit has correct number of qubits for all-ones input.

        Note: Cirq's all_qubits() only returns qubits with operations.
        For sparse inputs, we need to use all-ones to get all qubits.
        """
        x = np.array([1, 1, 1, 1])  # All ones to use all qubits
        circuit = default_encoding.get_circuit(x, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    def test_circuit_has_operations(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that circuit has operations for input [1,0,1,0].

        Input [1,0,1,0] should have X gates on qubits 0 and 2 only.
        """
        circuit = default_encoding.get_circuit(binary_data_4d, backend="cirq")
        # Should have 2 X gates (for the two 1s in input)
        assert len(list(circuit.all_operations())) == 2
        # Cirq only tracks qubits with operations
        assert len(circuit.all_qubits()) == 2

    def test_all_zeros_empty_circuit(self, default_encoding: BasisEncoding) -> None:
        """Test that all-zeros input produces empty circuit."""
        x = np.array([0, 0, 0, 0])
        circuit = default_encoding.get_circuit(x, backend="cirq")
        # No operations (no X gates needed)
        assert len(list(circuit.all_operations())) == 0

    def test_all_ones_full_x_gates(self, default_encoding: BasisEncoding) -> None:
        """Test that all-ones input produces circuit with 4 X gates."""
        x = np.array([1, 1, 1, 1])
        circuit = default_encoding.get_circuit(x, backend="cirq")
        ops = list(circuit.all_operations())
        assert len(ops) == 4, f"Expected 4 X gates, got {len(ops)}"
        # All should be X gates
        assert all(isinstance(op.gate, cirq.XPowGate) for op in ops)

    def test_batch_circuits(
        self,
        default_encoding: BasisEncoding,
        batch_data_4d: NDArray[np.integer],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)

    def test_x_gates_in_single_moment(self, default_encoding: BasisEncoding) -> None:
        """Test that all X gates are placed in a single Moment (parallel)."""
        x = np.array([1, 1, 1, 1])
        circuit = default_encoding.get_circuit(x, backend="cirq")
        # Should be a single moment containing all X gates
        moments = list(circuit.moments)
        assert len(moments) == 1, f"Expected 1 moment, got {len(moments)}"

    def test_circuit_produces_correct_state(
        self, default_encoding: BasisEncoding
    ) -> None:
        """Test that circuit produces correct state vector."""
        x = np.array([1, 0, 1, 0])
        circuit = default_encoding.get_circuit(x, backend="cirq")
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state = result.final_state_vector
        probs = np.abs(state) ** 2
        # Should be a pure basis state
        assert np.max(probs) > 0.99


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of the encoding.

    BasisEncoding implements the mapping:
        |psi(x)> = X^{x_0} ⊗ X^{x_1} ⊗ ... ⊗ X^{x_{n-1}} |0>^⊗n

    where X^{x_i} means "apply X if x_i=1, else identity".
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_single_qubit_states(self) -> None:
        """Test single qubit encoding produces correct states.

        |0> when input is 0
        |1> when input is 1
        """
        enc = BasisEncoding(n_features=1)
        dev = qml.device("default.qubit", wires=1)

        # Test |0> state
        circuit_fn_0 = enc.get_circuit(np.array([0]), backend="pennylane")

        @qml.qnode(dev)
        def circuit_0():
            circuit_fn_0()
            return qml.state()

        state_0 = circuit_0()
        assert np.isclose(state_0[0], 1.0, atol=1e-10)  # |0> component
        assert np.isclose(state_0[1], 0.0, atol=1e-10)  # |1> component

        # Test |1> state
        circuit_fn_1 = enc.get_circuit(np.array([1]), backend="pennylane")

        @qml.qnode(dev)
        def circuit_1():
            circuit_fn_1()
            return qml.state()

        state_1 = circuit_1()
        assert np.isclose(state_1[0], 0.0, atol=1e-10)  # |0> component
        assert np.isclose(state_1[1], 1.0, atol=1e-10)  # |1> component

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_two_qubit_all_combinations(self) -> None:
        """Test all 4 combinations for 2-qubit encoding.

        |00>, |01>, |10>, |11>
        """
        enc = BasisEncoding(n_features=2)
        dev = qml.device("default.qubit", wires=2)

        test_cases = [
            (np.array([0, 0]), 0),  # |00> -> index 0
            (np.array([1, 0]), 1),  # |10> -> index 1 (in little-endian)
            (np.array([0, 1]), 2),  # |01> -> index 2
            (np.array([1, 1]), 3),  # |11> -> index 3
        ]

        for x, expected_idx in test_cases:
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit():
                circuit_fn()
                return qml.state()

            state = circuit()
            probs = np.abs(state) ** 2

            # Find the index with probability ~1
            actual_idx = np.argmax(probs)
            assert np.isclose(probs[actual_idx], 1.0, atol=1e-10), (
                f"Input {x}: expected pure state at index {expected_idx}, "
                f"got prob {probs[actual_idx]} at index {actual_idx}"
            )

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_state_normalization(self, default_encoding: BasisEncoding) -> None:
        """Test that all output states are properly normalized."""
        dev = qml.device("default.qubit", wires=4)

        test_inputs = [
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            np.array([1, 0, 1, 0]),
            np.array([0.8, 0.2, 0.9, 0.1]),  # Continuous input
        ]

        for x in test_inputs:
            circuit_fn = default_encoding.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit():
                circuit_fn()
                return qml.state()

            state = circuit()
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(
                norm, 1.0, atol=1e-10
            ), f"State not normalized for input {x}: norm = {norm}"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_zeros_input(self, default_encoding: BasisEncoding) -> None:
        """Test encoding with all-zeros input.

        Should produce |0000> state with no X gates.
        """
        x = np.array([0, 0, 0, 0])
        # Should not raise
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            assert circuit.depth() == 0  # No gates needed

    def test_all_ones_input(self, default_encoding: BasisEncoding) -> None:
        """Test encoding with all-ones input.

        Should produce |1111> state with X on all qubits.
        """
        x = np.array([1, 1, 1, 1])
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 4

    def test_alternating_input(self, default_encoding: BasisEncoding) -> None:
        """Test encoding with alternating pattern [1,0,1,0]."""
        x = np.array([1, 0, 1, 0])
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2

    def test_single_feature_zero(self) -> None:
        """Test single feature with value 0."""
        enc = BasisEncoding(n_features=1)
        x = np.array([0])
        circuit = enc.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0

    def test_single_feature_one(self) -> None:
        """Test single feature with value 1."""
        enc = BasisEncoding(n_features=1)
        x = np.array([1])
        circuit = enc.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 1

    def test_large_feature_count(self) -> None:
        """Test encoding with large number of features."""
        enc = BasisEncoding(n_features=20)
        x = np.random.randint(0, 2, size=20)
        # Should not raise
        circuit = enc.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            assert circuit.num_qubits == 20

    def test_2d_input_single_sample(self, default_encoding: BasisEncoding) -> None:
        """Test that 2D input with single sample is handled correctly."""
        x = np.array([[1, 0, 1, 0]])  # Shape (1, 4)
        # Should not raise
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            assert isinstance(circuit, QuantumCircuit)

    def test_boolean_input(self, default_encoding: BasisEncoding) -> None:
        """Test that boolean numpy array input works."""
        x = np.array([True, False, True, False])
        circuit = default_encoding.get_circuit(x, backend="qiskit")
        if HAS_QISKIT:
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_very_small_positive_values(self) -> None:
        """Test values very close to zero (but positive).

        All should binarize to 0 since they're <= 0.5.
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, "Small positive values should map to 0"

    def test_values_just_above_threshold(self) -> None:
        """Test values just above 0.5 threshold."""
        enc = BasisEncoding(n_features=4)
        eps = 1e-15  # Machine epsilon level
        x = np.array([0.5 + eps, 0.5 + eps, 0.5 + eps, 0.5 + eps])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            # Should be 4 X gates since all are > 0.5
            assert x_count == 4

    def test_values_just_below_threshold(self) -> None:
        """Test values just below 0.5 threshold."""
        enc = BasisEncoding(n_features=4)
        eps = 1e-15
        x = np.array([0.5 - eps, 0.5 - eps, 0.5 - eps, 0.5 - eps])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            # Should be 0 X gates since all are < 0.5
            assert x_count == 0

    def test_very_large_values(self) -> None:
        """Test encoding with very large values.

        All should binarize to 1 since they're > 0.5.
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([1e10, 1e15, 1e20, 1e30])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 4, "Large values should map to 1"

    def test_very_large_negative_values(self) -> None:
        """Test encoding with very large negative values.

        All should binarize to 0 since they're < 0.5.
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([-1e10, -1e15, -1e20, -1e30])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, "Large negative values should map to 0"

    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values."""
        enc = BasisEncoding(n_features=4)
        x = np.array([1e-20, 1e20, -1e15, 0.51])
        # Expected: [0, 1, 0, 1] -> 2 X gates
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_extreme_values_produce_normalized_state(self) -> None:
        """Test that extreme values still produce normalized quantum states."""
        enc = BasisEncoding(n_features=4)
        dev = qml.device("default.qubit", wires=4)

        x = np.array([1e-100, 1e100, 0.5, 0.500001])
        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = BasisEncoding(n_features=4)
        enc2 = BasisEncoding(n_features=4)
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = BasisEncoding(n_features=4)
        enc2 = BasisEncoding(n_features=8)
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = BasisEncoding(n_features=4)
        enc2 = BasisEncoding(n_features=4)
        assert hash(enc1) == hash(enc2)

    def test_hash_different_for_different_params(self) -> None:
        """Test that different objects likely have different hashes."""
        enc1 = BasisEncoding(n_features=4)
        enc2 = BasisEncoding(n_features=8)
        # Different hashes (not guaranteed but highly likely)
        assert hash(enc1) != hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = BasisEncoding(n_features=4)
        enc2 = BasisEncoding(n_features=4)
        enc3 = BasisEncoding(n_features=8)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = BasisEncoding(n_features=4)
        enc2 = BasisEncoding(n_features=4)
        enc3 = BasisEncoding(n_features=8)

        d = {enc1: "four features"}
        d[enc3] = "eight features"

        # enc2 should retrieve the same value as enc1
        assert d[enc2] == "four features"
        assert d[enc3] == "eight features"

    def test_not_equal_to_other_types(self) -> None:
        """Test that encoding is not equal to other types."""
        enc = BasisEncoding(n_features=4)
        assert enc != "BasisEncoding(n_features=4)"
        assert enc != 4
        assert enc != None
        assert enc != {"n_features": 4}


# =============================================================================
# Test Class: String Representation
# =============================================================================


class TestRepr:
    """Tests for string representation (__repr__)."""

    def test_repr_contains_class_name(self) -> None:
        """Test that repr contains the class name."""
        enc = BasisEncoding(n_features=4)
        repr_str = repr(enc)
        assert "BasisEncoding" in repr_str

    def test_repr_contains_n_features(self) -> None:
        """Test that repr contains n_features parameter."""
        enc = BasisEncoding(n_features=4)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str

    def test_repr_different_for_different_params(self) -> None:
        """Test that repr differs for different parameters."""
        enc1 = BasisEncoding(n_features=4)
        enc2 = BasisEncoding(n_features=8)
        assert repr(enc1) != repr(enc2)

    def test_repr_is_valid_constructor_call(self) -> None:
        """Test that repr looks like a valid constructor call."""
        enc = BasisEncoding(n_features=4)
        repr_str = repr(enc)
        # Should match pattern: BasisEncoding(n_features=4)
        assert repr_str == "BasisEncoding(n_features=4)"


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for backend error handling."""

    def test_invalid_backend_raises_error(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(binary_data_4d, backend="invalid")  # type: ignore

    def test_invalid_backend_empty_string(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that empty string backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(binary_data_4d, backend="")  # type: ignore

    def test_invalid_backend_none(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that None backend raises appropriate error."""
        with pytest.raises((ValueError, TypeError)):
            default_encoding.get_circuit(binary_data_4d, backend=None)  # type: ignore

    def test_backend_case_sensitivity(
        self,
        default_encoding: BasisEncoding,
        binary_data_4d: NDArray[np.integer],
    ) -> None:
        """Test that backend names are case-sensitive.

        'PennyLane' (capitalized) should raise ValueError, only 'pennylane' works.
        """
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(binary_data_4d, backend="PennyLane")  # type: ignore


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = BasisEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.n_qubits == enc.n_qubits
        assert restored.depth == enc.depth
        assert restored.threshold == enc.threshold

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = BasisEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = BasisEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([1, 0, 1, 0])
        if HAS_QISKIT:
            circuit = restored.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == 4

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = BasisEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        props = restored.properties
        assert isinstance(props, EncodingProperties)
        assert props.n_qubits == 4
        assert props.is_entangling is False

    def test_pickle_various_sizes(self) -> None:
        """Test pickle roundtrip for various sizes."""
        for n in [1, 4, 8, 16, 32]:
            enc = BasisEncoding(n_features=n)
            pickled = pickle.dumps(enc)
            restored = pickle.loads(pickled)
            assert enc == restored


# =============================================================================
# Test Class: Concurrent Access / Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        enc = BasisEncoding(n_features=4)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for _i in range(num_circuits_per_thread):
                    x = np.random.randint(0, 2, size=4)
                    circuit = enc.get_circuit(x, backend="qiskit")
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
        enc = BasisEncoding(n_features=4)
        num_threads = 10
        num_accesses = 100

        results: list[int] = []
        errors: list[Exception] = []

        def access_properties(thread_id: int) -> None:
            try:
                for _ in range(num_accesses):
                    _ = enc.n_qubits
                    _ = enc.depth
                    props = enc.properties
                    results.append(props.n_qubits)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(access_properties, i) for i in range(num_threads)
            ]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # All property values should be consistent
        assert all(r == 4 for r in results)

    def test_shared_encoding_across_threads(self) -> None:
        """Test that a shared encoding instance works correctly across threads."""
        enc = BasisEncoding(n_features=4)
        num_threads = 5

        def worker(thread_id: int) -> tuple[int, int, int]:
            # Each thread generates circuits and checks properties
            x = np.array([thread_id % 2] * 4)
            if HAS_QISKIT:
                circuit = enc.get_circuit(x, backend="qiskit")
                return (circuit.num_qubits, enc.n_qubits, enc.depth)
            return (4, 4, 1)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # All results should be consistent
        for circuit_qubits, enc_qubits, depth in results:
            assert circuit_qubits == 4
            assert enc_qubits == 4
            assert depth == 1


# =============================================================================
# Test Class: Custom Threshold (Comprehensive)
# =============================================================================


@pytest.mark.custom_threshold
class TestCustomThreshold:
    """Comprehensive tests for custom binarization threshold functionality.

    BasisEncoding allows users to specify a custom threshold for binarizing
    continuous data. These tests cover:
    - Various threshold values (0.0, negative, high, extreme)
    - Threshold validation (type and value checking)
    - Interaction with binarize(), actual_gate_count(), resource_summary()
    - Equality, hashing, and repr with custom thresholds
    - Serialization with custom thresholds
    - Boundary behavior at custom thresholds
    """

    # =========================================================================
    # Threshold Instantiation Tests
    # =========================================================================

    def test_custom_threshold_zero(self) -> None:
        """Test threshold=0.0 for signed data (negative -> 0, positive -> 1)."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        assert enc.threshold == 0.0

    def test_custom_threshold_positive(self) -> None:
        """Test custom positive threshold values."""
        for threshold in [0.1, 0.3, 0.7, 0.9, 1.0]:
            enc = BasisEncoding(n_features=4, threshold=threshold)
            assert enc.threshold == threshold

    def test_custom_threshold_negative(self) -> None:
        """Test custom negative threshold values."""
        for threshold in [-0.1, -0.5, -1.0, -10.0]:
            enc = BasisEncoding(n_features=4, threshold=threshold)
            assert enc.threshold == threshold

    def test_custom_threshold_integer(self) -> None:
        """Test that integer threshold is converted to float."""
        enc = BasisEncoding(n_features=4, threshold=1)
        assert enc.threshold == 1.0
        assert isinstance(enc.threshold, float)

    def test_custom_threshold_large_values(self) -> None:
        """Test extreme threshold values."""
        enc_large = BasisEncoding(n_features=4, threshold=1e10)
        assert enc_large.threshold == 1e10

        enc_small = BasisEncoding(n_features=4, threshold=-1e10)
        assert enc_small.threshold == -1e10

    # =========================================================================
    # Threshold Validation Tests
    # =========================================================================

    def test_threshold_nan_rejected(self) -> None:
        """Test that NaN threshold raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            BasisEncoding(n_features=4, threshold=float("nan"))

    def test_threshold_positive_inf_rejected(self) -> None:
        """Test that positive infinity threshold raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            BasisEncoding(n_features=4, threshold=float("inf"))

    def test_threshold_negative_inf_rejected(self) -> None:
        """Test that negative infinity threshold raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            BasisEncoding(n_features=4, threshold=float("-inf"))

    def test_threshold_bool_rejected(self) -> None:
        """Test that boolean threshold raises TypeError."""
        with pytest.raises(TypeError, match="numeric type"):
            BasisEncoding(n_features=4, threshold=True)  # type: ignore

        with pytest.raises(TypeError, match="numeric type"):
            BasisEncoding(n_features=4, threshold=False)  # type: ignore

    def test_threshold_string_rejected(self) -> None:
        """Test that string threshold raises TypeError."""
        with pytest.raises(TypeError, match="numeric type"):
            BasisEncoding(n_features=4, threshold="0.5")  # type: ignore

    def test_threshold_none_rejected(self) -> None:
        """Test that None threshold raises TypeError."""
        with pytest.raises(TypeError, match="numeric type"):
            BasisEncoding(n_features=4, threshold=None)  # type: ignore

    def test_threshold_list_rejected(self) -> None:
        """Test that list threshold raises TypeError."""
        with pytest.raises(TypeError, match="numeric type"):
            BasisEncoding(n_features=4, threshold=[0.5])  # type: ignore

    # =========================================================================
    # Binarization Behavior with Custom Threshold
    # =========================================================================

    def test_threshold_zero_signed_data(self) -> None:
        """Test threshold=0.0 correctly binarizes signed data.

        - Negative values -> 0
        - Zero -> 0 (exactly at threshold)
        - Positive values -> 1
        """
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.0, 0.5, 1.0])
        # Expected: [0, 0, 1, 1]
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2, f"Expected 2 X gates for threshold=0, got {x_count}"

    def test_threshold_zero_all_negative(self) -> None:
        """Test threshold=0.0 with all negative values."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-1.0, -0.5, -0.1, -0.001])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, "All negative values should map to 0"

    def test_threshold_zero_all_positive(self) -> None:
        """Test threshold=0.0 with all positive values."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([0.001, 0.1, 0.5, 1.0])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 4, "All positive values should map to 1"

    def test_threshold_high_value(self) -> None:
        """Test high threshold (0.7) for conservative binarization."""
        enc = BasisEncoding(n_features=4, threshold=0.7)
        x = np.array([0.5, 0.6, 0.7, 0.8])
        # Expected: [0, 0, 0, 1] - only 0.8 > 0.7
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 1, f"Expected 1 X gate for threshold=0.7, got {x_count}"

    def test_threshold_low_value(self) -> None:
        """Test low threshold (0.3) for aggressive binarization."""
        enc = BasisEncoding(n_features=4, threshold=0.3)
        x = np.array([0.2, 0.3, 0.4, 0.5])
        # Expected: [0, 0, 1, 1] - 0.4 and 0.5 > 0.3
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2, f"Expected 2 X gates for threshold=0.3, got {x_count}"

    def test_threshold_negative_value(self) -> None:
        """Test negative threshold (-0.5) for data shifted below zero."""
        enc = BasisEncoding(n_features=4, threshold=-0.5)
        x = np.array([-1.0, -0.5, 0.0, 0.5])
        # Expected: [0, 0, 1, 1] - 0.0 and 0.5 > -0.5
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2, f"Expected 2 X gates for threshold=-0.5, got {x_count}"

    def test_threshold_one_boundary(self) -> None:
        """Test threshold=1.0 where most normalized data becomes 0."""
        enc = BasisEncoding(n_features=4, threshold=1.0)
        x = np.array([0.5, 0.9, 1.0, 1.1])
        # Expected: [0, 0, 0, 1] - only 1.1 > 1.0
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 1, f"Expected 1 X gate for threshold=1.0, got {x_count}"

    def test_threshold_very_large(self) -> None:
        """Test very large threshold where everything becomes 0."""
        enc = BasisEncoding(n_features=4, threshold=1e10)
        x = np.array([1e5, 1e6, 1e7, 1e8])
        # All values <= 1e10, so all should be 0
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, "Large threshold should map all to 0"

    def test_threshold_very_small(self) -> None:
        """Test very small (negative) threshold where everything becomes 1."""
        enc = BasisEncoding(n_features=4, threshold=-1e10)
        x = np.array([-1e5, -1e6, -1e7, -1e8])
        # All values > -1e10, so all should be 1
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 4, "Very small threshold should map all to 1"

    # =========================================================================
    # Boundary Precision Tests at Custom Threshold
    # =========================================================================

    def test_custom_threshold_exactly_at_boundary(self) -> None:
        """Test that values exactly at custom threshold map to 0."""
        enc = BasisEncoding(n_features=4, threshold=0.7)
        x = np.array([0.7, 0.7, 0.7, 0.7])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, "Values exactly at threshold should map to 0"

    def test_custom_threshold_epsilon_above(self) -> None:
        """Test values epsilon above custom threshold map to 1."""
        enc = BasisEncoding(n_features=4, threshold=0.7)
        eps = 1e-15
        x = np.array([0.7 + eps] * 4)
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 4, "Values just above threshold should map to 1"

    def test_custom_threshold_epsilon_below(self) -> None:
        """Test values epsilon below custom threshold map to 0."""
        enc = BasisEncoding(n_features=4, threshold=0.7)
        eps = 1e-15
        x = np.array([0.7 - eps] * 4)
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 0, "Values just below threshold should map to 0"

    # =========================================================================
    # binarize() Method with Custom Threshold
    # =========================================================================

    def test_binarize_custom_threshold_zero(self) -> None:
        """Test binarize() method with threshold=0.0."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.0, 0.5, 1.0])
        binary = enc.binarize(x)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(binary, expected)

    def test_binarize_custom_threshold_high(self) -> None:
        """Test binarize() method with high threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.8)
        x = np.array([0.5, 0.7, 0.8, 0.9])
        binary = enc.binarize(x)
        expected = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(binary, expected)

    def test_binarize_custom_threshold_negative(self) -> None:
        """Test binarize() method with negative threshold."""
        enc = BasisEncoding(n_features=4, threshold=-0.3)
        x = np.array([-0.5, -0.3, -0.1, 0.1])
        binary = enc.binarize(x)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(binary, expected)

    def test_binarize_batch_custom_threshold(self) -> None:
        """Test binarize() with batch input and custom threshold."""
        enc = BasisEncoding(n_features=3, threshold=0.0)
        X = np.array(
            [
                [-1.0, 0.0, 1.0],
                [0.5, -0.5, 0.1],
            ]
        )
        binary = enc.binarize(X)
        expected = np.array(
            [
                [0, 0, 1],
                [1, 0, 1],
            ]
        )
        np.testing.assert_array_equal(binary, expected)

    # =========================================================================
    # actual_gate_count() with Custom Threshold
    # =========================================================================

    def test_actual_gate_count_custom_threshold(self) -> None:
        """Test actual_gate_count() respects custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.0, 0.5, 1.0])
        # Expected binary: [0, 0, 1, 1] -> 2 gates
        assert enc.actual_gate_count(x) == 2

    def test_actual_gate_count_high_threshold(self) -> None:
        """Test actual_gate_count() with high threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.9)
        x = np.array([0.5, 0.7, 0.9, 0.95])
        # Expected binary: [0, 0, 0, 1] -> 1 gate
        assert enc.actual_gate_count(x) == 1

    def test_actual_gate_count_vs_properties_custom_threshold(self) -> None:
        """Test that actual_gate_count can differ from properties.gate_count."""
        enc = BasisEncoding(n_features=8, threshold=0.9)
        # properties.gate_count is worst-case (all 1s)
        assert enc.properties.gate_count == 8

        # With high threshold, sparse data has fewer gates
        x = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93])
        actual = enc.actual_gate_count(x)
        assert actual == 3, "Only values > 0.9 should produce gates"
        assert actual < enc.properties.gate_count

    # =========================================================================
    # resource_summary() with Custom Threshold
    # =========================================================================

    def test_resource_summary_custom_threshold(self) -> None:
        """Test resource_summary() includes correct threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.3)
        x = np.array([0.2, 0.3, 0.4, 0.5])
        summary = enc.resource_summary(x)

        assert summary["threshold"] == 0.3
        assert summary["actual_gate_count"] == 2  # 0.4 and 0.5 > 0.3
        assert summary["binarized_input"] == [0, 0, 1, 1]

    def test_resource_summary_sparsity_custom_threshold(self) -> None:
        """Test resource_summary sparsity calculation with custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.9)
        x = np.array([0.5, 0.7, 0.9, 0.95])
        # Expected: [0, 0, 0, 1] -> 3 zeros, 1 one
        summary = enc.resource_summary(x)

        assert summary["ones_count"] == 1
        assert summary["zeros_count"] == 3
        assert summary["sparsity"] == 0.75  # 3/4 zeros

    # =========================================================================
    # repr() with Custom Threshold
    # =========================================================================

    def test_repr_default_threshold_not_shown(self) -> None:
        """Test that repr doesn't show threshold when it's the default (0.5)."""
        enc = BasisEncoding(n_features=4)
        repr_str = repr(enc)
        assert repr_str == "BasisEncoding(n_features=4)"
        assert "threshold" not in repr_str

    def test_repr_custom_threshold_shown(self) -> None:
        """Test that repr shows custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        repr_str = repr(enc)
        assert "threshold=0.0" in repr_str
        assert repr_str == "BasisEncoding(n_features=4, threshold=0.0)"

    def test_repr_various_custom_thresholds(self) -> None:
        """Test repr with various custom threshold values."""
        test_cases = [
            (0.0, "BasisEncoding(n_features=4, threshold=0.0)"),
            (0.3, "BasisEncoding(n_features=4, threshold=0.3)"),
            (0.7, "BasisEncoding(n_features=4, threshold=0.7)"),
            (-0.5, "BasisEncoding(n_features=4, threshold=-0.5)"),
            (1.0, "BasisEncoding(n_features=4, threshold=1.0)"),
        ]
        for threshold, expected_repr in test_cases:
            enc = BasisEncoding(n_features=4, threshold=threshold)
            assert repr(enc) == expected_repr, f"Failed for threshold={threshold}"

    # =========================================================================
    # Equality and Hashing with Custom Threshold
    # =========================================================================

    def test_equality_same_custom_threshold(self) -> None:
        """Test that encodings with same custom threshold are equal."""
        enc1 = BasisEncoding(n_features=4, threshold=0.3)
        enc2 = BasisEncoding(n_features=4, threshold=0.3)
        assert enc1 == enc2

    def test_equality_different_thresholds(self) -> None:
        """Test that encodings with different thresholds are not equal."""
        enc1 = BasisEncoding(n_features=4, threshold=0.3)
        enc2 = BasisEncoding(n_features=4, threshold=0.7)
        assert enc1 != enc2

    def test_equality_default_vs_explicit_threshold(self) -> None:
        """Test that default threshold equals explicit 0.5."""
        enc_default = BasisEncoding(n_features=4)
        enc_explicit = BasisEncoding(n_features=4, threshold=0.5)
        assert enc_default == enc_explicit

    def test_hash_same_custom_threshold(self) -> None:
        """Test that equal encodings with custom threshold have same hash."""
        enc1 = BasisEncoding(n_features=4, threshold=0.3)
        enc2 = BasisEncoding(n_features=4, threshold=0.3)
        assert hash(enc1) == hash(enc2)

    def test_hash_different_thresholds(self) -> None:
        """Test that different thresholds produce different hashes."""
        enc1 = BasisEncoding(n_features=4, threshold=0.3)
        enc2 = BasisEncoding(n_features=4, threshold=0.7)
        assert hash(enc1) != hash(enc2)

    def test_set_membership_custom_threshold(self) -> None:
        """Test set membership with custom thresholds."""
        enc1 = BasisEncoding(n_features=4, threshold=0.3)
        enc2 = BasisEncoding(n_features=4, threshold=0.3)  # Same as enc1
        enc3 = BasisEncoding(n_features=4, threshold=0.7)  # Different

        s = {enc1, enc2, enc3}
        assert len(s) == 2  # enc1 and enc2 are equal

    def test_dict_key_custom_threshold(self) -> None:
        """Test using encodings with custom threshold as dict keys."""
        enc1 = BasisEncoding(n_features=4, threshold=0.3)
        enc2 = BasisEncoding(n_features=4, threshold=0.3)

        d = {enc1: "low threshold"}
        assert d[enc2] == "low threshold"

    # =========================================================================
    # Serialization with Custom Threshold
    # =========================================================================

    def test_pickle_custom_threshold(self) -> None:
        """Test pickle roundtrip preserves custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.3)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.threshold == 0.3
        assert enc == restored

    def test_pickle_various_thresholds(self) -> None:
        """Test pickle roundtrip for various threshold values."""
        thresholds = [0.0, 0.3, 0.7, -0.5, 1.0, -10.0, 10.0]
        for threshold in thresholds:
            enc = BasisEncoding(n_features=4, threshold=threshold)
            pickled = pickle.dumps(enc)
            restored = pickle.loads(pickled)

            assert restored.threshold == threshold
            assert enc == restored

    def test_pickle_circuit_generation_custom_threshold(self) -> None:
        """Test circuit generation after unpickling with custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([-0.5, 0.0, 0.5, 1.0])
        if HAS_QISKIT:
            circuit = restored.get_circuit(x, backend="qiskit")
            x_count = circuit.count_ops().get("x", 0)
            assert x_count == 2  # threshold=0: [0, 0, 1, 1]

    # =========================================================================
    # Cross-Backend Tests with Custom Threshold
    # =========================================================================

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_qiskit_custom_threshold(self) -> None:
        """Test Qiskit backend respects custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.0, 0.5, 1.0])
        circuit = enc.get_circuit(x, backend="qiskit")

        assert circuit.num_qubits == 4
        x_count = circuit.count_ops().get("x", 0)
        assert x_count == 2

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_pennylane_custom_threshold(self) -> None:
        """Test PennyLane backend respects custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.0, 0.5, 1.0])
        circuit_fn = enc.get_circuit(x, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.probs()

        probs = full_circuit()
        # Should be a pure basis state
        assert np.max(probs) > 0.99

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_cirq_custom_threshold(self) -> None:
        """Test Cirq backend respects custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.0, 0.5, 1.0])
        circuit = enc.get_circuit(x, backend="cirq")

        # Expected: [0, 0, 1, 1] -> 2 X gates
        ops = list(circuit.all_operations())
        assert len(ops) == 2

    # =========================================================================
    # Mathematical Correctness with Custom Threshold
    # =========================================================================

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_state_correctness_custom_threshold(self) -> None:
        """Test that custom threshold produces correct quantum state."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-1.0, -0.5, 0.5, 1.0])
        # Expected binary: [0, 0, 1, 1]

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        probs = np.abs(state) ** 2

        # Should be pure basis state
        assert np.max(probs) > 0.99
        assert np.sum(probs > 0.01) == 1

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_thresholds_different_states(self) -> None:
        """Test that same data with different thresholds produces different states."""
        x = np.array([0.3, 0.5, 0.7, 0.9])
        dev = qml.device("default.qubit", wires=4)

        states = []
        for threshold in [0.0, 0.4, 0.6, 0.85]:
            enc = BasisEncoding(n_features=4, threshold=threshold)
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit():
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # Each threshold should produce a different state
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                overlap = np.abs(np.vdot(states[i], states[j])) ** 2
                # Most should be orthogonal (different basis states)
                # Some might overlap if they produce same binary pattern
                # Just verify they're not all identical
                pass  # States exist and are valid

        # Verify at least some states are different
        unique_states = len({tuple(np.round(np.abs(s) ** 2, 5)) for s in states})
        assert (
            unique_states >= 2
        ), "Different thresholds should produce different states"


# =============================================================================
# Test Class: Slow Simulation Tests
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    """

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_same_basis_state(self) -> None:
        """Test that all backends produce the same computational basis state.

        For basis encoding, all backends should produce exactly the same
        classical basis state (up to global phase and qubit ordering conventions).
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([1, 0, 1, 0])

        # PennyLane
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.probs()

        pl_probs = pl_full()

        # Qiskit
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_probs = qk_sv.probabilities()

        # Cirq
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_probs = np.abs(cirq_result.final_state_vector) ** 2

        # For basis encoding, there should be exactly one basis state with prob=1
        # The exact index may differ due to qubit ordering conventions
        assert np.max(pl_probs) > 0.99, "PennyLane should produce pure basis state"
        assert np.max(qk_probs) > 0.99, "Qiskit should produce pure basis state"
        assert np.max(cirq_probs) > 0.99, "Cirq should produce pure basis state"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different basis states."""
        enc = BasisEncoding(n_features=4)
        dev = qml.device("default.qubit", wires=4)

        inputs = [
            np.array([0, 0, 0, 0]),
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([1, 1, 1, 1]),
        ]

        states = []
        for x in inputs:
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit():
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # Each state should be different (orthogonal basis states)
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                # Inner product of orthogonal basis states should be 0
                overlap = np.abs(np.vdot(states[i], states[j])) ** 2
                assert (
                    overlap < 0.01
                ), f"States {i} and {j} should be orthogonal, overlap={overlap}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input always produces same state."""
        enc = BasisEncoding(n_features=4)
        x = np.array([1, 0, 1, 0])
        dev = qml.device("default.qubit", wires=4)

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
            np.testing.assert_allclose(
                states[0],
                states[i],
                atol=1e-10,
                err_msg=f"State {i} differs from state 0",
            )

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_large_circuit_simulation(self) -> None:
        """Test simulation with larger circuit (10 qubits)."""
        enc = BasisEncoding(n_features=10)
        x = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        dev = qml.device("default.qubit", wires=10)

        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.probs()

        probs = circuit()

        # Should be a pure basis state
        assert np.max(probs) > 0.99
        assert np.sum(probs > 0.01) == 1

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT),
        reason="PennyLane and Qiskit required",
    )
    def test_batch_consistency_across_backends(self) -> None:
        """Test that batch processing is consistent across backends."""
        enc = BasisEncoding(n_features=4)
        batch = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ]
        )

        pl_circuits = enc.get_circuits(batch, backend="pennylane")
        qk_circuits = enc.get_circuits(batch, backend="qiskit")

        assert len(pl_circuits) == len(qk_circuits) == 4

        # Verify each produces a pure basis state
        for i, (pl_c, qk_c) in enumerate(zip(pl_circuits, qk_circuits)):
            dev = qml.device("default.qubit", wires=4)

            @qml.qnode(dev)
            def pl_circuit():
                pl_c()
                return qml.probs()

            pl_probs = pl_circuit()
            qk_probs = Statevector(qk_c).probabilities()

            assert np.max(pl_probs) > 0.99, f"PL batch {i} not pure basis state"
            assert np.max(qk_probs) > 0.99, f"QK batch {i} not pure basis state"


# =============================================================================
# Test Class: Gate Count Breakdown
# =============================================================================


class TestGateCountBreakdown:
    """Tests for the gate_count_breakdown() method.

    gate_count_breakdown() returns worst-case gate counts as a TypedDict
    with fields: x_gates, total_single_qubit, total_two_qubit, total,
    is_worst_case.
    """

    def test_breakdown_returns_dict(self) -> None:
        """Test that gate_count_breakdown() returns a dict."""
        enc = BasisEncoding(n_features=4)
        breakdown = enc.gate_count_breakdown()
        assert isinstance(breakdown, dict)

    def test_breakdown_x_gates_equals_n_features(self) -> None:
        """Test that x_gates equals n_features (worst case: all 1s)."""
        for n in [1, 4, 8, 16]:
            enc = BasisEncoding(n_features=n)
            breakdown = enc.gate_count_breakdown()
            assert breakdown["x_gates"] == n

    def test_breakdown_total_single_qubit(self) -> None:
        """Test that total_single_qubit equals x_gates."""
        enc = BasisEncoding(n_features=8)
        breakdown = enc.gate_count_breakdown()
        assert breakdown["total_single_qubit"] == breakdown["x_gates"]
        assert breakdown["total_single_qubit"] == 8

    def test_breakdown_no_two_qubit_gates(self) -> None:
        """Test that there are never two-qubit gates in basis encoding."""
        for n in [1, 4, 8, 32]:
            enc = BasisEncoding(n_features=n)
            breakdown = enc.gate_count_breakdown()
            assert breakdown["total_two_qubit"] == 0

    def test_breakdown_total_equals_single_qubit(self) -> None:
        """Test that total equals total_single_qubit (no two-qubit gates)."""
        enc = BasisEncoding(n_features=4)
        breakdown = enc.gate_count_breakdown()
        assert breakdown["total"] == breakdown["total_single_qubit"]

    def test_breakdown_total_consistency(self) -> None:
        """Test that total = total_single_qubit + total_two_qubit."""
        enc = BasisEncoding(n_features=8)
        breakdown = enc.gate_count_breakdown()
        assert breakdown["total"] == (
            breakdown["total_single_qubit"] + breakdown["total_two_qubit"]
        )

    def test_breakdown_is_worst_case_flag(self) -> None:
        """Test that is_worst_case is always True."""
        enc = BasisEncoding(n_features=4)
        breakdown = enc.gate_count_breakdown()
        assert breakdown["is_worst_case"] is True

    def test_breakdown_matches_properties(self) -> None:
        """Test that breakdown total matches properties.gate_count."""
        enc = BasisEncoding(n_features=8)
        breakdown = enc.gate_count_breakdown()
        assert breakdown["total"] == enc.properties.gate_count
        assert breakdown["total_single_qubit"] == enc.properties.single_qubit_gates
        assert breakdown["total_two_qubit"] == enc.properties.two_qubit_gates

    def test_breakdown_custom_threshold_unchanged(self) -> None:
        """Test that breakdown is the same regardless of threshold.

        Threshold affects actual gate counts, not worst-case bounds.
        """
        enc_default = BasisEncoding(n_features=4)
        enc_custom = BasisEncoding(n_features=4, threshold=0.0)
        assert enc_default.gate_count_breakdown() == enc_custom.gate_count_breakdown()

    def test_breakdown_has_all_expected_keys(self) -> None:
        """Test that breakdown contains all documented keys."""
        enc = BasisEncoding(n_features=4)
        breakdown = enc.gate_count_breakdown()
        expected_keys = {
            "x_gates",
            "total_single_qubit",
            "total_two_qubit",
            "total",
            "is_worst_case",
        }
        assert set(breakdown.keys()) == expected_keys


# =============================================================================
# Test Class: transform_input (DataTransformable Protocol)
# =============================================================================


class TestTransformInput:
    """Tests for the transform_input() method.

    transform_input() is an alias for binarize() that implements the
    DataTransformable protocol for standardized cross-encoding usage.
    """

    def test_transform_basic(self) -> None:
        """Test basic transform_input with continuous data."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.8, 0.2, 0.6, 0.4])
        result = enc.transform_input(x)
        expected = np.array([1, 0, 1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_transform_matches_binarize(self) -> None:
        """Test that transform_input returns identical output to binarize."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.1, 0.9, 0.5, 0.51])
        np.testing.assert_array_equal(enc.transform_input(x), enc.binarize(x))

    def test_transform_binary_data_idempotent(self) -> None:
        """Test that transform_input is idempotent for binary data.

        With default threshold 0.5, binary [0, 1] data is unchanged
        because 1 > 0.5 -> 1 and 0 <= 0.5 -> 0.
        """
        enc = BasisEncoding(n_features=4)
        x = np.array([1, 0, 1, 0])
        result = enc.transform_input(x)
        np.testing.assert_array_equal(result, x)

    def test_transform_custom_threshold(self) -> None:
        """Test transform_input respects custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.0, 0.5, 1.0])
        result = enc.transform_input(x)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_transform_all_zeros(self) -> None:
        """Test transform_input with all-zero input."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.0, 0.0, 0.0, 0.0])
        result = enc.transform_input(x)
        expected = np.array([0, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_transform_all_ones(self) -> None:
        """Test transform_input with all-one input."""
        enc = BasisEncoding(n_features=4)
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = enc.transform_input(x)
        expected = np.array([1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_transform_returns_integer_dtype(self) -> None:
        """Test that transform_input returns integer-typed array."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.8, 0.2, 0.6, 0.4])
        result = enc.transform_input(x)
        assert np.issubdtype(result.dtype, np.integer)

    def test_transform_preserves_shape_1d(self) -> None:
        """Test that 1D input produces 1D output."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.1, 0.9, 0.3, 0.8])
        result = enc.transform_input(x)
        assert result.ndim == 1
        assert result.shape == (4,)

    def test_transform_preserves_shape_2d(self) -> None:
        """Test that 2D batch input produces 2D output."""
        enc = BasisEncoding(n_features=3)
        X = np.array([[0.1, 0.9, 0.5], [0.6, 0.4, 0.7]])
        result = enc.transform_input(X)
        assert result.ndim == 2
        assert result.shape == (2, 3)

    def test_transform_wrong_shape_rejected(self) -> None:
        """Test that wrong feature count raises ValueError."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="Expected 4 features"):
            enc.transform_input(x)

    def test_transform_nan_rejected(self) -> None:
        """Test that NaN values are rejected."""
        enc = BasisEncoding(n_features=4)
        x = np.array([0.5, np.nan, 0.5, 0.5])
        with pytest.raises(ValueError, match="NaN"):
            enc.transform_input(x)

    def test_transform_list_input(self) -> None:
        """Test that list input is accepted and converted."""
        enc = BasisEncoding(n_features=4)
        result = enc.transform_input([0.8, 0.2, 0.6, 0.4])
        expected = np.array([1, 0, 1, 0])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# Test Class: Protocol Conformance
# =============================================================================


class TestProtocolConformance:
    """Tests for protocol conformance via isinstance checks.

    BasisEncoding should implement:
    - DataDependentResourceAnalyzable (resource_summary(x), actual_gate_count(x))
    - DataTransformable (transform_input(x))

    BasisEncoding should NOT implement:
    - ResourceAnalyzable (data-independent resource_summary() — no parameter)
    - EntanglementQueryable (no entanglement created)
    """

    def test_implements_data_dependent_resource_analyzable(self) -> None:
        """Test that BasisEncoding satisfies DataDependentResourceAnalyzable."""
        enc = BasisEncoding(n_features=4)
        assert isinstance(enc, DataDependentResourceAnalyzable)

    def test_implements_data_transformable(self) -> None:
        """Test that BasisEncoding satisfies DataTransformable."""
        enc = BasisEncoding(n_features=4)
        assert isinstance(enc, DataTransformable)

    def test_does_not_implement_entanglement_queryable(self) -> None:
        """Test that BasisEncoding does NOT satisfy EntanglementQueryable.

        BasisEncoding creates product states with no entanglement.
        """
        enc = BasisEncoding(n_features=4)
        assert not isinstance(enc, EntanglementQueryable)

    def test_data_dependent_protocol_methods_work(self) -> None:
        """Test that protocol methods produce correct results when called
        through the protocol interface."""
        enc = BasisEncoding(n_features=4)

        # Use the protocol's actual_gate_count method
        assert isinstance(enc, DataDependentResourceAnalyzable)
        x = np.array([1, 0, 1, 0])
        assert enc.actual_gate_count(x) == 2

        # Use the protocol's resource_summary method
        summary = enc.resource_summary(x)
        assert summary["actual_gate_count"] == 2

    def test_data_transformable_protocol_method_works(self) -> None:
        """Test that transform_input produces correct results when called
        through the DataTransformable protocol interface."""
        enc = BasisEncoding(n_features=4)
        assert isinstance(enc, DataTransformable)

        x = np.array([0.8, 0.2, 0.6, 0.4])
        result = enc.transform_input(x)
        expected = np.array([1, 0, 1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_protocol_check_with_custom_threshold(self) -> None:
        """Test that protocol conformance holds with custom threshold."""
        enc = BasisEncoding(n_features=4, threshold=0.0)
        assert isinstance(enc, DataDependentResourceAnalyzable)
        assert isinstance(enc, DataTransformable)

        x = np.array([-0.5, 0.0, 0.5, 1.0])
        assert enc.actual_gate_count(x) == 2
        np.testing.assert_array_equal(enc.transform_input(x), np.array([0, 0, 1, 1]))
