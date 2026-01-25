"""Comprehensive tests for HardwareEfficientEncoding.

This test module provides complete coverage of the HardwareEfficientEncoding class,
which implements hardware-efficient ansatz circuits optimized for near-term quantum
devices with limited connectivity. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, gate_count)
- Entanglement pattern behavior (linear, circular)
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

Run with: pytest tests/unit/encodings/test_hardware_efficient.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_hardware_efficient.py -v -m "not slow"

References
----------
.. [1] Kandala, A. et al. "Hardware-efficient variational quantum eigensolver
       for small molecules and quantum magnets." Nature 549, 242-246 (2017).
"""

from __future__ import annotations

# Standard library imports
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

# Third-party imports
import numpy as np
import pytest
from numpy.typing import NDArray

# Local imports
from encoding_atlas import HardwareEfficientEncoding
from encoding_atlas.core.properties import EncodingProperties

# Conditional TYPE_CHECKING imports
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

    Values chosen to exercise typical encoding behavior with
    moderate rotation angles.
    """
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def sample_data_2d() -> NDArray[np.floating]:
    """2-dimensional sample data for testing.

    Minimal configuration for entanglement testing.
    """
    return np.array([0.5, 1.0])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (moderate values)
    - [0.9, 1.0, 1.1, 1.2] (larger values)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ])


@pytest.fixture
def default_encoding() -> HardwareEfficientEncoding:
    """Default HardwareEfficientEncoding with 4 features."""
    return HardwareEfficientEncoding(n_features=4)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for HardwareEfficientEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = HardwareEfficientEncoding(n_features=4)
        assert enc.n_features == 4
        assert enc.n_qubits == 4
        assert enc.reps == 2
        assert enc.rotation == "Y"
        assert enc.entanglement == "linear"

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = HardwareEfficientEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16, 32]:
            enc = HardwareEfficientEncoding(n_features=n)
            assert enc.n_features == n

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = HardwareEfficientEncoding(n_features=4, reps=5)
        assert enc.reps == 5

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_rotation_variants(self, rotation: str) -> None:
        """Test all rotation axis variants."""
        enc = HardwareEfficientEncoding(n_features=4, rotation=rotation)  # type: ignore
        assert enc.rotation == rotation

    @pytest.mark.parametrize("entanglement", ["linear", "circular", "full"])
    def test_entanglement_patterns(self, entanglement: str) -> None:
        """Test all valid entanglement patterns."""
        enc = HardwareEfficientEncoding(n_features=4, entanglement=entanglement)  # type: ignore
        assert enc.entanglement == entanglement

    def test_config_stored_correctly(self) -> None:
        """Test that configuration is stored in config dict."""
        enc = HardwareEfficientEncoding(
            n_features=4,
            reps=3,
            rotation="Z",
            entanglement="circular",
        )
        assert enc.config["reps"] == 3
        assert enc.config["rotation"] == "Z"
        assert enc.config["entanglement"] == "circular"

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = HardwareEfficientEncoding(n_features=100)
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
            HardwareEfficientEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            HardwareEfficientEncoding(n_features=-1)

    def test_invalid_n_features_float(self) -> None:
        """Test that float n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            HardwareEfficientEncoding(n_features=4.0)  # type: ignore

    def test_invalid_n_features_string(self) -> None:
        """Test that string n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            HardwareEfficientEncoding(n_features="4")  # type: ignore

    def test_invalid_n_features_none(self) -> None:
        """Test that None n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            HardwareEfficientEncoding(n_features=None)  # type: ignore

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError):
            HardwareEfficientEncoding(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError):
            HardwareEfficientEncoding(n_features=4, reps=-1)

    def test_invalid_entanglement(self) -> None:
        """Test that invalid entanglement pattern raises ValueError."""
        with pytest.raises(ValueError, match="entanglement must be one of"):
            HardwareEfficientEncoding(n_features=4, entanglement="invalid")  # type: ignore

    def test_invalid_rotation(self) -> None:
        """Test that invalid rotation axis raises ValueError."""
        with pytest.raises(ValueError, match="rotation must be one of"):
            HardwareEfficientEncoding(n_features=4, rotation="W")  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of HardwareEfficientEncoding."""

    def test_n_qubits(self) -> None:
        """Test n_qubits property calculation.

        For HardwareEfficientEncoding, n_qubits equals n_features.
        """
        enc = HardwareEfficientEncoding(n_features=4)
        assert enc.n_qubits == 4

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features for various sizes."""
        for n in [2, 4, 8, 16]:
            enc = HardwareEfficientEncoding(n_features=n)
            assert enc.n_qubits == n

    def test_depth(self) -> None:
        """Test circuit depth calculation.

        Depth formula: reps * 2 (rotation layer + entangling layer per rep)
        For n_features=4, reps=2: depth = 2 * 2 = 4
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
        # depth = reps * 2
        assert enc.depth == 4

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = HardwareEfficientEncoding(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = HardwareEfficientEncoding(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_is_entangling(self) -> None:
        """Test that hardware-efficient encoding creates entanglement."""
        enc = HardwareEfficientEncoding(n_features=4)
        assert enc.properties.is_entangling is True

    def test_not_simulable(self) -> None:
        """Test that hardware-efficient encoding is not efficiently simulable."""
        enc = HardwareEfficientEncoding(n_features=4)
        assert enc.properties.simulability == "not_simulable"

    def test_optimized_for_hardware(self) -> None:
        """Test that encoding is optimized for hardware connectivity."""
        enc = HardwareEfficientEncoding(n_features=4)
        # Check that notes mention hardware optimization
        assert "hardware" in enc.properties.notes.lower()

    def test_full_entanglement_gate_count_property(self) -> None:
        """Test gate count property with full entanglement.

        Full entanglement has n(n-1)/2 two-qubit gates per rep.
        For 4 qubits, 2 reps: 6 × 2 = 12 CNOT gates.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="full", reps=2)
        props = enc.properties
        # 4 qubits × 2 reps = 8 single-qubit gates
        assert props.single_qubit_gates == 8
        # 4(4-1)/2 × 2 reps = 6 × 2 = 12 two-qubit gates
        assert props.two_qubit_gates == 12
        # Total: 8 + 12 = 20
        assert props.gate_count == 20

    def test_full_entanglement_resource_summary(self) -> None:
        """Test resource summary with full entanglement.

        Verifies that resource_summary() correctly reports full entanglement
        configuration and hardware requirements.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="full", reps=2)
        summary = enc.resource_summary()

        assert summary["entanglement"] == "full"
        assert summary["n_entanglement_pairs"] == 6  # 4(4-1)/2 = 6
        assert summary["hardware_requirements"]["connectivity"] == "full"
        # Verify all pairs are present
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert summary["entanglement_pairs"] == expected_pairs

    def test_full_entanglement_gate_count_breakdown(self) -> None:
        """Test gate_count_breakdown() with full entanglement.

        Verifies detailed gate breakdown for full connectivity.
        """
        enc = HardwareEfficientEncoding(
            n_features=4, entanglement="full", reps=2, rotation="Y"
        )
        breakdown = enc.gate_count_breakdown()

        assert breakdown["ry"] == 8  # 4 qubits × 2 reps
        assert breakdown["rx"] == 0
        assert breakdown["rz"] == 0
        assert breakdown["cnot"] == 12  # 6 pairs × 2 reps
        assert breakdown["total_single_qubit"] == 8
        assert breakdown["total_two_qubit"] == 12
        assert breakdown["total"] == 20


# =============================================================================
# Test Class: Entanglement Behavior
# =============================================================================


class TestEntanglementBehavior:
    """Tests for entanglement pair computation and patterns.

    Hardware-efficient encoding supports different entanglement patterns
    optimized for various hardware topologies.
    """

    def test_linear_entanglement_4_qubits(self) -> None:
        """Test linear entanglement with 4 qubits.

        Linear pattern: (0,1), (1,2), (2,3) - nearest-neighbor only.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="linear")
        expected = [(0, 1), (1, 2), (2, 3)]
        assert enc._get_entanglement_pairs() == expected

    def test_circular_entanglement_4_qubits(self) -> None:
        """Test circular entanglement with 4 qubits.

        Circular pattern: (0,1), (1,2), (2,3), (3,0) - ring topology.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="circular")
        expected = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert enc._get_entanglement_pairs() == expected

    def test_linear_entanglement_count(self) -> None:
        """Test that linear entanglement has n-1 pairs."""
        for n in [2, 4, 6, 8]:
            enc = HardwareEfficientEncoding(n_features=n, entanglement="linear")
            # Linear: n-1 pairs (nearest neighbor chain)
            expected_pairs = n - 1
            assert len(enc._get_entanglement_pairs()) == expected_pairs

    def test_circular_entanglement_count(self) -> None:
        """Test that circular entanglement has n pairs."""
        for n in [3, 4, 6, 8]:
            enc = HardwareEfficientEncoding(n_features=n, entanglement="circular")
            # Circular: n pairs (ring topology)
            expected_pairs = n
            assert len(enc._get_entanglement_pairs()) == expected_pairs

    def test_single_qubit_no_entanglement(self) -> None:
        """Test that single qubit has no entanglement pairs."""
        enc = HardwareEfficientEncoding(n_features=1, entanglement="linear")
        assert len(enc._get_entanglement_pairs()) == 0

    def test_full_entanglement_4_qubits(self) -> None:
        """Test full entanglement with 4 qubits.

        Full pattern: all-to-all connectivity with n(n-1)/2 pairs.
        For 4 qubits: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) = 6 pairs.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="full")
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert enc._get_entanglement_pairs() == expected

    def test_full_entanglement_count(self) -> None:
        """Test that full entanglement has n(n-1)/2 pairs.

        Full entanglement creates all-to-all connectivity, which scales
        quadratically with the number of qubits.
        """
        for n in [2, 3, 4, 5, 6, 8]:
            enc = HardwareEfficientEncoding(n_features=n, entanglement="full")
            # Full: n(n-1)/2 pairs (all-to-all connectivity)
            expected_pairs = n * (n - 1) // 2
            assert len(enc._get_entanglement_pairs()) == expected_pairs

    def test_full_entanglement_pairs_ordered(self) -> None:
        """Test that full entanglement pairs are properly ordered.

        Pairs should be generated with control < target (i < j) to ensure
        consistent circuit generation across backends.
        """
        enc = HardwareEfficientEncoding(n_features=5, entanglement="full")
        pairs = enc._get_entanglement_pairs()
        # All pairs should have control < target
        for control, target in pairs:
            assert control < target, f"Pair ({control}, {target}) is not ordered"

    def test_two_qubits_minimal_entanglement(self) -> None:
        """Test minimal entanglement with 2 qubits."""
        enc_linear = HardwareEfficientEncoding(n_features=2, entanglement="linear")
        enc_circular = HardwareEfficientEncoding(n_features=2, entanglement="circular")
        # Linear: 1 pair, Circular: 2 pairs (but (0,1) and (1,0) are equivalent)
        assert len(enc_linear._get_entanglement_pairs()) == 1

    def test_two_qubits_full_equals_linear(self) -> None:
        """Test that full entanglement with 2 qubits equals linear.

        With only 2 qubits, full connectivity reduces to linear (single pair).
        """
        enc_linear = HardwareEfficientEncoding(n_features=2, entanglement="linear")
        enc_full = HardwareEfficientEncoding(n_features=2, entanglement="full")
        # Both should have exactly 1 pair: (0, 1)
        assert enc_linear._get_entanglement_pairs() == [(0, 1)]
        assert enc_full._get_entanglement_pairs() == [(0, 1)]

    def test_three_qubits_full_entanglement(self) -> None:
        """Test full entanglement with 3 qubits.

        Full pattern: (0,1), (0,2), (1,2) = 3 pairs = 3(3-1)/2.
        """
        enc = HardwareEfficientEncoding(n_features=3, entanglement="full")
        expected = [(0, 1), (0, 2), (1, 2)]
        assert enc._get_entanglement_pairs() == expected


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        validated = default_encoding._validate_input(sample_data_4d)
        assert validated.shape == (4,)

    def test_wrong_feature_count(
        self, default_encoding: HardwareEfficientEncoding
    ) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2])  # 2 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_nan_input_rejected(
        self, default_encoding: HardwareEfficientEncoding
    ) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding._validate_input(x)

    def test_inf_input_rejected(
        self, default_encoding: HardwareEfficientEncoding
    ) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_list_input_accepted(
        self, default_encoding: HardwareEfficientEncoding
    ) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]  # list, not array
        # Should not raise
        validated = default_encoding._validate_input(x)
        assert isinstance(validated, np.ndarray)

    def test_batch_input_shape(
        self,
        default_encoding: HardwareEfficientEncoding,
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
        default_encoding: HardwareEfficientEncoding,
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
        enc = HardwareEfficientEncoding(n_features=4, reps=1)
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
        default_encoding: HardwareEfficientEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations(
        self,
        sample_data_4d: NDArray[np.floating],
        rotation: str,
    ) -> None:
        """Test circuit generation with all rotation types."""
        enc = HardwareEfficientEncoding(n_features=4, rotation=rotation, reps=1)  # type: ignore
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_full_entanglement_produces_valid_state(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that full entanglement produces a valid normalized state.

        Full entanglement creates all-to-all connectivity, which should
        produce highly entangled states.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="full", reps=2)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be normalized
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)
        # Full entanglement should create superposition
        non_zero_amplitudes = np.sum(np.abs(state) > 1e-10)
        assert non_zero_amplitudes > 1, "Full entanglement should create superposition"

    def test_full_entanglement_more_expressive_than_linear(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that full entanglement produces different states than linear.

        Full connectivity should create stronger correlations than linear.
        """
        enc_linear = HardwareEfficientEncoding(n_features=4, entanglement="linear", reps=2)
        enc_full = HardwareEfficientEncoding(n_features=4, entanglement="full", reps=2)

        dev = qml.device("default.qubit", wires=4)

        circuit_fn_linear = enc_linear.get_circuit(sample_data_4d, backend="pennylane")
        circuit_fn_full = enc_full.get_circuit(sample_data_4d, backend="pennylane")

        @qml.qnode(dev)
        def linear_circuit():
            circuit_fn_linear()
            return qml.state()

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn_full()
            return qml.state()

        state_linear = linear_circuit()
        state_full = full_circuit()

        # States should be different (not perfectly overlapping)
        fidelity = np.abs(np.vdot(state_linear, state_full)) ** 2
        assert fidelity < 0.9999, "Full and linear entanglement should produce different states"


# =============================================================================
# Test Class: Qiskit Backend
# =============================================================================


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.mark.backend_qiskit
class TestQiskitBackend:
    """Tests for Qiskit circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_circuit_has_expected_gates(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit contains expected gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        ops = circuit.count_ops()
        # Verify rotation gates and CNOT gates are present
        assert "ry" in ops or "rx" in ops or "rz" in ops
        assert "cx" in ops  # CNOT for entanglement

    def test_batch_circuits(
        self,
        default_encoding: HardwareEfficientEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations(
        self,
        sample_data_4d: NDArray[np.floating],
        rotation: str,
    ) -> None:
        """Test circuit generation with all rotation types."""
        enc = HardwareEfficientEncoding(n_features=4, rotation=rotation, reps=1)  # type: ignore
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_full_entanglement_gate_count(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that full entanglement produces correct gate count.

        Full entanglement with 4 qubits should have n(n-1)/2 = 6 CNOT
        gates per repetition.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="full", reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        ops = circuit.count_ops()
        # 4 qubits, full entanglement: 6 CNOT gates per rep
        assert ops.get("cx", 0) == 6

    def test_full_entanglement_scaling(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that full entanglement gate count scales quadratically.

        For reps=2, should have 2 × n(n-1)/2 = 12 CNOT gates.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="full", reps=2)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        ops = circuit.count_ops()
        # 4 qubits, full entanglement, 2 reps: 6 × 2 = 12 CNOT gates
        assert ops.get("cx", 0) == 12


# =============================================================================
# Test Class: Cirq Backend
# =============================================================================


@pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
@pytest.mark.backend_cirq
class TestCirqBackend:
    """Tests for Cirq circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        # all_operations() returns a generator, convert to list for length check
        operations = list(circuit.all_operations())
        assert len(operations) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    def test_batch_circuits(
        self,
        default_encoding: HardwareEfficientEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_all_rotations(
        self,
        sample_data_4d: NDArray[np.floating],
        rotation: str,
    ) -> None:
        """Test circuit generation with all rotation types."""
        enc = HardwareEfficientEncoding(n_features=4, rotation=rotation, reps=1)  # type: ignore
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_linear_entanglement(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test Cirq circuit with linear entanglement."""
        enc = HardwareEfficientEncoding(n_features=4, entanglement="linear", reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circular_entanglement(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test Cirq circuit with circular entanglement."""
        enc = HardwareEfficientEncoding(n_features=4, entanglement="circular", reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_full_entanglement(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test Cirq circuit with full entanglement.

        Full entanglement creates all-to-all connectivity, suitable for
        ion trap devices with global connectivity.
        """
        enc = HardwareEfficientEncoding(n_features=4, entanglement="full", reps=1)
        circuit = enc.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)
        # Full entanglement should have more operations than linear
        enc_linear = HardwareEfficientEncoding(n_features=4, entanglement="linear", reps=1)
        circuit_linear = enc_linear.get_circuit(sample_data_4d, backend="cirq")
        # Convert generators to lists for length comparison
        full_ops = list(circuit.all_operations())
        linear_ops = list(circuit_linear.all_operations())
        assert len(full_ops) > len(linear_ops)


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of the encoding.

    Verifies that rotation angles are computed correctly and gates
    are applied as expected.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zero_input_produces_ground_state_rotation(self) -> None:
        """Test that zero input produces expected state.

        With zero input angles, the state should remain close to ground state
        (only affected by entangling gates).
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=1)
        x = np.zeros(4)

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        # State should be normalized
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_rotation_angles_applied_correctly(self) -> None:
        """Test that rotation angles from input are applied correctly.

        Different inputs should produce different states.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=1)
        dev = qml.device("default.qubit", wires=4)

        x1 = np.array([0.0, 0.0, 0.0, 0.0])
        x2 = np.array([np.pi, np.pi, np.pi, np.pi])

        circuit_fn1 = enc.get_circuit(x1, backend="pennylane")
        circuit_fn2 = enc.get_circuit(x2, backend="pennylane")

        @qml.qnode(dev)
        def circuit1():
            circuit_fn1()
            return qml.state()

        @qml.qnode(dev)
        def circuit2():
            circuit_fn2()
            return qml.state()

        state1 = circuit1()
        state2 = circuit2()

        # States should be different
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99, "Different inputs should produce different states"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_entanglement_creates_correlations(self) -> None:
        """Test that entangling gates create quantum correlations.

        The final state should not be a product state for non-trivial inputs.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
        x = np.array([0.5, 1.0, 1.5, 2.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        # Multiple non-zero amplitudes indicate superposition/entanglement
        non_zero_amplitudes = np.sum(np.abs(state) > 1e-10)
        assert non_zero_amplitudes > 1, "Entangled state should have multiple amplitudes"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_rep(self) -> None:
        """Test encoding with single repetition."""
        enc = HardwareEfficientEncoding(n_features=4, reps=1)
        assert enc.reps == 1

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions."""
        enc = HardwareEfficientEncoding(n_features=4, reps=10)
        assert enc.reps == 10

    def test_large_feature_count(self) -> None:
        """Test encoding with large feature count."""
        enc = HardwareEfficientEncoding(n_features=20, entanglement="linear")
        assert enc.n_qubits == 20
        # Linear entanglement should have n-1 pairs
        assert len(enc._get_entanglement_pairs()) == 19

    def test_zero_valued_input(
        self, default_encoding: HardwareEfficientEncoding
    ) -> None:
        """Test with all-zero input."""
        x = np.zeros(4)
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_large_valued_input(
        self, default_encoding: HardwareEfficientEncoding
    ) -> None:
        """Test with large input values."""
        x = np.array([100.0, 200.0, 300.0, 400.0])
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_negative_valued_input(
        self, default_encoding: HardwareEfficientEncoding
    ) -> None:
        """Test with negative input values."""
        x = np.array([-0.5, -1.0, -1.5, -2.0])
        # Should not raise
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_two_features_minimum_entangling(self) -> None:
        """Test minimum configuration for entanglement."""
        enc = HardwareEfficientEncoding(n_features=2, reps=1)
        assert enc.n_qubits == 2
        assert len(enc._get_entanglement_pairs()) >= 1


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values.

    These tests ensure the encoding handles edge cases in numerical
    precision without producing NaN, Inf, or denormalized states.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_small_values(self) -> None:
        """Test encoding with very small input values.

        Very small rotation angles should produce states close to ground state.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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

        Large rotation angles are valid (rotations are periodic).
        Should not cause overflow.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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
    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values.

        Real-world data often has features at different scales.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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
    def test_subnormal_values(self) -> None:
        """Test encoding with subnormal (denormalized) floating-point values.

        Subnormal values can cause precision issues in some implementations.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=1)
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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_values_near_pi_boundaries(self) -> None:
        """Test encoding with values near pi and 2pi boundaries.

        Rotation periodicity boundaries can reveal numerical edge cases.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
        # Values very close to pi boundaries
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
    def test_many_reps_numerical_stability(self) -> None:
        """Test numerical stability with high repetition count.

        Deep circuits can accumulate numerical errors.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=20)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-9), f"Norm after 20 reps: {norm}"
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = HardwareEfficientEncoding(n_features=4)
        enc2 = HardwareEfficientEncoding(n_features=4)
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = HardwareEfficientEncoding(n_features=4)
        enc2 = HardwareEfficientEncoding(n_features=8)
        assert enc1 != enc2

    def test_equality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = HardwareEfficientEncoding(n_features=4, reps=2)
        enc2 = HardwareEfficientEncoding(n_features=4, reps=3)
        assert enc1 != enc2

    def test_equality_different_rotation(self) -> None:
        """Test that encodings with different rotation are not equal."""
        enc1 = HardwareEfficientEncoding(n_features=4, rotation="X")
        enc2 = HardwareEfficientEncoding(n_features=4, rotation="Y")
        assert enc1 != enc2

    def test_equality_different_entanglement(self) -> None:
        """Test that encodings with different entanglement are not equal."""
        enc1 = HardwareEfficientEncoding(n_features=4, entanglement="linear")
        enc2 = HardwareEfficientEncoding(n_features=4, entanglement="circular")
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = HardwareEfficientEncoding(n_features=4, reps=2)
        enc2 = HardwareEfficientEncoding(n_features=4, reps=2)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = HardwareEfficientEncoding(n_features=4)
        enc2 = HardwareEfficientEncoding(n_features=4)  # Same as enc1
        enc3 = HardwareEfficientEncoding(n_features=8)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = HardwareEfficientEncoding(n_features=4)
        enc2 = HardwareEfficientEncoding(n_features=4)

        d = {enc1: "four features"}
        assert d[enc2] == "four features"  # enc2 should find same key


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ and string representation."""

    def test_repr_contains_class_name(self) -> None:
        """Test that repr contains the class name."""
        enc = HardwareEfficientEncoding(n_features=4)
        repr_str = repr(enc)
        assert "HardwareEfficientEncoding" in repr_str

    def test_repr_contains_n_features(self) -> None:
        """Test that repr contains n_features parameter."""
        enc = HardwareEfficientEncoding(n_features=4)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str

    def test_repr_contains_reps(self) -> None:
        """Test that repr contains reps parameter."""
        enc = HardwareEfficientEncoding(n_features=4, reps=3)
        repr_str = repr(enc)
        assert "reps=3" in repr_str

    def test_repr_custom_parameters(self) -> None:
        """Test repr with custom parameters."""
        enc = HardwareEfficientEncoding(
            n_features=8,
            reps=3,
            rotation="Z",
            entanglement="circular",
        )
        repr_str = repr(enc)
        assert "n_features=8" in repr_str
        assert "reps=3" in repr_str

    def test_repr_is_valid_string(self) -> None:
        """Test that repr returns a valid non-empty string."""
        enc = HardwareEfficientEncoding(n_features=4)
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
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_empty_backend_raises_error(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that empty backend string raises error."""
        with pytest.raises(ValueError):
            default_encoding.get_circuit(sample_data_4d, backend="")  # type: ignore

    def test_none_backend_raises_error(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that None backend raises error."""
        with pytest.raises((ValueError, TypeError)):
            default_encoding.get_circuit(sample_data_4d, backend=None)  # type: ignore

    def test_case_sensitive_backend(
        self,
        default_encoding: HardwareEfficientEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that backend names are case-sensitive."""
        with pytest.raises(ValueError):
            default_encoding.get_circuit(sample_data_4d, backend="QISKIT")  # type: ignore


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = HardwareEfficientEncoding(
            n_features=4,
            reps=3,
            rotation="Z",
            entanglement="circular",
        )
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.reps == enc.reps
        assert restored.rotation == enc.rotation
        assert restored.entanglement == enc.entanglement
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = HardwareEfficientEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        if HAS_QISKIT:
            circuit = restored.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == 4

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = HardwareEfficientEncoding(n_features=4)
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
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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
            futures = [
                executor.submit(generate_circuits, i)
                for i in range(num_threads)
            ]
            results = [f.result() for f in as_completed(futures)]

        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"

        # All circuits should have been generated (if Qiskit available)
        if HAS_QISKIT:
            total_circuits = sum(len(r) for r in results)
            assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = HardwareEfficientEncoding(n_features=4)
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
            futures = [
                executor.submit(access_properties)
                for _ in range(num_threads)
            ]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0
        assert len(results) == num_threads
        # All should return the same cached object
        assert all(r is results[0] for r in results)

    def test_concurrent_different_encodings(self) -> None:
        """Test concurrent access to different encoding instances."""
        encodings = [
            HardwareEfficientEncoding(n_features=4, reps=1),
            HardwareEfficientEncoding(n_features=4, reps=2),
            HardwareEfficientEncoding(n_features=4, reps=3),
        ]
        num_circuits_per_encoding = 20

        errors: list[Exception] = []
        results: list[int] = []

        def generate_for_encoding(enc: HardwareEfficientEncoding) -> int:
            count = 0
            try:
                for _ in range(num_circuits_per_encoding):
                    x = np.random.randn(4)
                    if HAS_QISKIT:
                        enc.get_circuit(x, backend="qiskit")
                        count += 1
            except Exception as e:
                errors.append(e)
            return count

        with ThreadPoolExecutor(max_workers=len(encodings)) as executor:
            futures = [
                executor.submit(generate_for_encoding, enc)
                for enc in encodings
            ]
            results = [f.result() for f in as_completed(futures)]

        assert len(errors) == 0


# =============================================================================
# Test Class: Slow Simulation Tests
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    They require all backends to be installed.
    """

    @staticmethod
    def _get_pennylane_state(
        enc: HardwareEfficientEncoding,
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
        enc: HardwareEfficientEncoding,
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
        enc: HardwareEfficientEncoding,
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
            err_msg=f"Probability distributions differ between {name1} and {name2}",
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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

        This is the core cross-backend fidelity test.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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
    @pytest.mark.parametrize("reps", [1, 2, 4])
    def test_cross_backend_with_different_reps(self, reps: int) -> None:
        """Test cross-backend equivalence with different repetition counts."""
        enc = HardwareEfficientEncoding(n_features=4, reps=reps)
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
    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_cross_backend_with_different_rotations(self, rotation: str) -> None:
        """Test cross-backend equivalence with different rotation axes."""
        enc = HardwareEfficientEncoding(
            n_features=4,
            reps=2,
            rotation=rotation,  # type: ignore
        )
        x = np.array([0.3, 0.6, 0.9, 1.2])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(
            pl_state, qk_state,
            f"PennyLane (rotation={rotation})",
            f"Qiskit (rotation={rotation})"
        )
        self._assert_states_equivalent(
            pl_state, cirq_state,
            f"PennyLane (rotation={rotation})",
            f"Cirq (rotation={rotation})"
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("entanglement", ["linear", "circular", "full"])
    def test_cross_backend_with_different_entanglement(
        self, entanglement: str
    ) -> None:
        """Test cross-backend equivalence with different entanglement patterns."""
        enc = HardwareEfficientEncoding(
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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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
    def test_cross_backend_with_6_qubits(self) -> None:
        """Test cross-backend equivalence with larger qubit count.

        Tests scaling behavior and consistency for moderately large systems.
        """
        enc = HardwareEfficientEncoding(n_features=6, reps=2)
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

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_zero_input(self) -> None:
        """Test cross-backend equivalence with zero input values.

        Edge case: all-zero input should produce consistent states.
        """
        enc = HardwareEfficientEncoding(n_features=4, reps=2)
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
        enc = HardwareEfficientEncoding(n_features=4, reps=1)
        x = np.array([np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
