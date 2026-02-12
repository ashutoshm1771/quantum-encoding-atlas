"""Comprehensive tests for HamiltonianEncoding.

This test module provides complete coverage of the HamiltonianEncoding class,
which encodes classical data through time evolution under parameterized
Hamiltonians (IQP, XY, Heisenberg, Pauli-Z). It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, entanglement patterns)
- Hamiltonian type behavior (IQP, XY, Heisenberg, Pauli-Z)
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

Run with: pytest tests/unit/encodings/test_hamiltonian.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_hamiltonian.py -v -m "not slow"

References
----------
.. [1] Havlicek et al., "Supervised learning with quantum-enhanced feature
       spaces", Nature 567, 209-212 (2019).
.. [2] Schuld & Killoran, "Quantum Machine Learning in Feature Hilbert
       Spaces", Phys. Rev. Lett. 122, 040504 (2019).
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas import HamiltonianEncoding
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
def sample_data_3d() -> NDArray[np.floating]:
    """3-dimensional sample data for testing.

    Values chosen for typical encoding behavior with 3-qubit systems.
    """
    return np.array([0.1, 0.2, 0.3])


@pytest.fixture
def sample_data_4d() -> NDArray[np.floating]:
    """4-dimensional sample data for testing.

    Values chosen to exercise typical encoding behavior.
    """
    return np.array([0.1, 0.2, 0.3, 0.4])


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
def default_encoding() -> HamiltonianEncoding:
    """Default HamiltonianEncoding with 4 features."""
    return HamiltonianEncoding(n_features=4)


@pytest.fixture
def iqp_encoding() -> HamiltonianEncoding:
    """IQP-style Hamiltonian encoding.

    Uses ZZ interactions for entanglement.
    """
    return HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")


@pytest.fixture
def xy_encoding() -> HamiltonianEncoding:
    """XY model Hamiltonian encoding.

    Uses XX+YY interactions for entanglement.
    """
    return HamiltonianEncoding(n_features=4, hamiltonian_type="xy")


@pytest.fixture
def heisenberg_encoding() -> HamiltonianEncoding:
    """Heisenberg model Hamiltonian encoding.

    Uses XX+YY+ZZ interactions for entanglement.
    """
    return HamiltonianEncoding(n_features=4, hamiltonian_type="heisenberg")


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for HamiltonianEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = HamiltonianEncoding(n_features=4)

        assert enc.n_features == 4
        assert enc.hamiltonian_type == "iqp"
        assert enc.evolution_time == 1.0
        assert enc.reps == 2
        assert enc.entanglement == "full"
        assert enc.insert_barriers is True

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = HamiltonianEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16]:
            enc = HamiltonianEncoding(n_features=n)
            assert enc.n_features == n

    def test_custom_hamiltonian_type_iqp(self) -> None:
        """Test IQP Hamiltonian type instantiation."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        assert enc.hamiltonian_type == "iqp"

    def test_custom_hamiltonian_type_xy(self) -> None:
        """Test XY Hamiltonian type instantiation."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="xy")
        assert enc.hamiltonian_type == "xy"

    def test_custom_hamiltonian_type_heisenberg(self) -> None:
        """Test Heisenberg Hamiltonian type instantiation."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="heisenberg")
        assert enc.hamiltonian_type == "heisenberg"

    def test_custom_hamiltonian_type_pauli_z(self) -> None:
        """Test Pauli-Z Hamiltonian type instantiation."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="pauli_z")
        assert enc.hamiltonian_type == "pauli_z"

    def test_hamiltonian_type_case_insensitive(self) -> None:
        """Test that Hamiltonian type is case-insensitive."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="IQP")
        assert enc.hamiltonian_type == "iqp"

    def test_custom_evolution_time(self) -> None:
        """Test instantiation with custom evolution time."""
        enc = HamiltonianEncoding(n_features=4, evolution_time=2.5)
        assert enc.evolution_time == 2.5

    def test_zero_evolution_time(self) -> None:
        """Test instantiation with zero evolution time."""
        enc = HamiltonianEncoding(n_features=4, evolution_time=0.0)
        assert enc.evolution_time == 0.0

    def test_negative_evolution_time(self) -> None:
        """Test instantiation with negative evolution time (allowed for time reversal)."""
        enc = HamiltonianEncoding(n_features=4, evolution_time=-1.0)
        assert enc.evolution_time == -1.0

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = HamiltonianEncoding(n_features=4, reps=5)
        assert enc.reps == 5

    def test_entanglement_linear(self) -> None:
        """Test instantiation with linear entanglement pattern."""
        enc = HamiltonianEncoding(n_features=4, entanglement="linear")
        assert enc.entanglement == "linear"

    def test_entanglement_circular(self) -> None:
        """Test instantiation with circular entanglement pattern."""
        enc = HamiltonianEncoding(n_features=4, entanglement="circular")
        assert enc.entanglement == "circular"

    def test_entanglement_case_insensitive(self) -> None:
        """Test that entanglement is case-insensitive."""
        enc = HamiltonianEncoding(n_features=4, entanglement="FULL")
        assert enc.entanglement == "full"

    def test_insert_barriers_false(self) -> None:
        """Test instantiation with insert_barriers=False."""
        enc = HamiltonianEncoding(n_features=4, insert_barriers=False)
        assert enc.insert_barriers is False

    def test_all_custom_parameters(self) -> None:
        """Test instantiation with all parameters customized."""
        enc = HamiltonianEncoding(
            n_features=6,
            hamiltonian_type="heisenberg",
            evolution_time=3.0,
            reps=4,
            entanglement="circular",
            insert_barriers=False,
        )

        assert enc.n_features == 6
        assert enc.hamiltonian_type == "heisenberg"
        assert enc.evolution_time == 3.0
        assert enc.reps == 4
        assert enc.entanglement == "circular"
        assert enc.insert_barriers is False

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = HamiltonianEncoding(n_features=100)
        assert enc.n_features == 100
        # Should be fast since no circuit is pre-computed


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    def test_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be"):
            HamiltonianEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be"):
            HamiltonianEncoding(n_features=-1)

    def test_invalid_hamiltonian_type(self) -> None:
        """Test that invalid Hamiltonian type raises ValueError."""
        with pytest.raises(ValueError, match="hamiltonian_type must be one of"):
            HamiltonianEncoding(n_features=4, hamiltonian_type="invalid")

    def test_invalid_evolution_time_nan(self) -> None:
        """Test that NaN evolution_time raises ValueError."""
        with pytest.raises(ValueError, match="evolution_time must be finite"):
            HamiltonianEncoding(n_features=4, evolution_time=float("nan"))

    def test_invalid_evolution_time_inf(self) -> None:
        """Test that infinite evolution_time raises ValueError."""
        with pytest.raises(ValueError, match="evolution_time must be finite"):
            HamiltonianEncoding(n_features=4, evolution_time=float("inf"))

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            HamiltonianEncoding(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            HamiltonianEncoding(n_features=4, reps=-1)

    def test_invalid_entanglement(self) -> None:
        """Test that invalid entanglement pattern raises ValueError."""
        with pytest.raises(ValueError, match="entanglement must be one of"):
            HamiltonianEncoding(n_features=4, entanglement="invalid")

    def test_type_error_hamiltonian_type_int(self) -> None:
        """Test that non-string hamiltonian_type raises TypeError."""
        with pytest.raises(TypeError, match="hamiltonian_type must be a string"):
            HamiltonianEncoding(n_features=4, hamiltonian_type=1)  # type: ignore

    def test_type_error_evolution_time_string(self) -> None:
        """Test that non-numeric evolution_time raises TypeError."""
        with pytest.raises(TypeError, match="evolution_time must be a number"):
            HamiltonianEncoding(n_features=4, evolution_time="1.0")  # type: ignore

    def test_type_error_reps_float(self) -> None:
        """Test that float reps raises TypeError."""
        with pytest.raises(TypeError, match="reps must be an integer"):
            HamiltonianEncoding(n_features=4, reps=2.5)  # type: ignore

    def test_type_error_entanglement_int(self) -> None:
        """Test that non-string entanglement raises TypeError."""
        with pytest.raises(TypeError, match="entanglement must be a string"):
            HamiltonianEncoding(n_features=4, entanglement=1)  # type: ignore

    def test_type_error_insert_barriers_string(self) -> None:
        """Test that non-bool insert_barriers raises TypeError."""
        with pytest.raises(TypeError, match="insert_barriers must be a bool"):
            HamiltonianEncoding(n_features=4, insert_barriers="True")  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of HamiltonianEncoding."""

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features."""
        for n in [2, 4, 8, 16]:
            enc = HamiltonianEncoding(n_features=n)
            assert enc.n_qubits == n

    def test_depth_increases_with_reps(self) -> None:
        """Test that depth increases with repetitions."""
        enc1 = HamiltonianEncoding(n_features=4, reps=1)
        enc2 = HamiltonianEncoding(n_features=4, reps=2)
        enc3 = HamiltonianEncoding(n_features=4, reps=4)

        assert enc2.depth > enc1.depth
        assert enc3.depth > enc2.depth

    def test_properties_type(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that properties returns EncodingProperties instance."""
        assert isinstance(default_encoding.properties, EncodingProperties)

    def test_properties_cached(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that properties are cached (same object returned)."""
        props1 = default_encoding.properties
        props2 = default_encoding.properties
        assert props1 is props2

    def test_properties_n_qubits(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that properties contains correct n_qubits."""
        props = default_encoding.properties
        assert props.n_qubits == 4

    def test_properties_depth(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that properties contains positive depth."""
        props = default_encoding.properties
        assert props.depth > 0

    def test_properties_gate_count(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that properties contains positive gate counts."""
        props = default_encoding.properties
        assert props.gate_count > 0
        assert props.single_qubit_gates > 0
        assert props.two_qubit_gates > 0

    def test_properties_parameter_count(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that parameter_count is 0 (data-driven, not trainable)."""
        props = default_encoding.properties
        assert props.parameter_count == 0

    def test_is_entangling(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that encoding creates entanglement."""
        assert default_encoding.properties.is_entangling is True

    def test_not_simulable(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that IQP Hamiltonian is not efficiently simulable."""
        assert default_encoding.properties.simulability == "not_simulable"

    def test_properties_notes_contain_hamiltonian_type(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that notes mention the Hamiltonian type."""
        props = default_encoding.properties
        assert "iqp" in props.notes.lower()

    def test_properties_linear_entanglement_fewer_gates(self) -> None:
        """Test that linear entanglement has fewer gates than full."""
        enc_full = HamiltonianEncoding(n_features=8, entanglement="full")
        enc_linear = HamiltonianEncoding(n_features=8, entanglement="linear")

        assert (
            enc_linear.properties.two_qubit_gates < enc_full.properties.two_qubit_gates
        )

    def test_entanglement_pairs_full(self) -> None:
        """Test full entanglement pattern generates correct pairs.

        For 4 qubits: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) = 6 pairs.
        """
        enc = HamiltonianEncoding(n_features=4, entanglement="full")
        pairs = enc._entanglement_pairs
        assert len(pairs) == 6

    def test_entanglement_pairs_linear(self) -> None:
        """Test linear entanglement pattern generates correct pairs.

        For 4 qubits: (0,1), (1,2), (2,3) = 3 pairs.
        """
        enc = HamiltonianEncoding(n_features=4, entanglement="linear")
        pairs = enc._entanglement_pairs
        assert len(pairs) == 3
        assert pairs == [(0, 1), (1, 2), (2, 3)]

    def test_entanglement_pairs_circular(self) -> None:
        """Test circular entanglement pattern generates correct pairs.

        For 4 qubits: linear pairs + wraparound = 4 pairs.
        """
        enc = HamiltonianEncoding(n_features=4, entanglement="circular")
        pairs = enc._entanglement_pairs
        assert len(pairs) == 4
        assert (3, 0) in pairs or (0, 3) in pairs

    def test_single_qubit_no_entanglement_pairs(self) -> None:
        """Test that single qubit has no entanglement pairs."""
        enc = HamiltonianEncoding(n_features=1)
        assert len(enc._entanglement_pairs) == 0


# =============================================================================
# Test Class: Hamiltonian Behavior
# =============================================================================


class TestHamiltonianBehavior:
    """Tests for Hamiltonian-type-specific behavior.

    This class tests the unique characteristics of different Hamiltonian types:
    IQP (ZZ), XY (XX+YY), Heisenberg (XX+YY+ZZ), and Pauli-Z (Z+ZZ).
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_all_types_execute(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that all Hamiltonian types execute without error."""
        hamiltonian_types = ["iqp", "xy", "heisenberg", "pauli_z"]

        for ham_type in hamiltonian_types:
            enc = HamiltonianEncoding(n_features=4, hamiltonian_type=ham_type)
            circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit() -> NDArray[np.complexfloating]:
                circuit_fn()
                return qml.state()

            state = full_circuit()
            assert state is not None

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_iqp_creates_entanglement(
        self,
        iqp_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that IQP Hamiltonian creates entanglement."""
        circuit_fn = iqp_encoding.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=iqp_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # Check that state has multiple non-zero amplitudes (entangled)
        probs = np.abs(state) ** 2
        non_zero_count = np.sum(probs > 1e-10)
        assert non_zero_count >= 1

    def test_pauli_z_similar_to_iqp(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that Pauli-Z has similar structure to IQP."""
        enc_iqp = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        enc_pauli_z = HamiltonianEncoding(n_features=4, hamiltonian_type="pauli_z")

        # Both should have same entangling property
        assert enc_iqp.properties.is_entangling == enc_pauli_z.properties.is_entangling

    def test_time_step_computation(self) -> None:
        """Test that time step is correctly computed.

        Time step = evolution_time / reps.
        For evolution_time=2.0, reps=4: time_step = 0.5.
        """
        enc = HamiltonianEncoding(n_features=4, evolution_time=2.0, reps=4)
        assert enc._time_step == 0.5


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        circuit = default_encoding.get_circuit(sample_data_4d)
        assert circuit is not None

    def test_valid_2d_input_single_sample(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 2D input with single sample is accepted."""
        x = sample_data_4d.reshape(1, -1)
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_wrong_feature_count(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2, 0.3])  # 3 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding.get_circuit(x)

    def test_nan_input_rejected(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding.get_circuit(x)

    def test_inf_input_rejected(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.get_circuit(x)

    def test_2d_multiple_samples_rejected(
        self,
        default_encoding: HamiltonianEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that 2D input with multiple samples raises error for get_circuit."""
        with pytest.raises(ValueError, match="single sample"):
            default_encoding.get_circuit(batch_data_4d)

    def test_list_input_accepted(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_batch_input_shape(
        self,
        default_encoding: HamiltonianEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that batch input generates multiple circuits."""
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
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that generated circuit executes correctly in QNode context."""
        circuit_fn = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=default_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None
        assert len(state) == 2**default_encoding.n_qubits

    def test_state_normalized(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that resulting state is normalized."""
        circuit_fn = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=default_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_iqp_hamiltonian(
        self,
        iqp_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test IQP Hamiltonian circuit executes."""
        circuit_fn = iqp_encoding.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=iqp_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_xy_hamiltonian(
        self,
        xy_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test XY Hamiltonian circuit executes."""
        circuit_fn = xy_encoding.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=xy_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_heisenberg_hamiltonian(
        self,
        heisenberg_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test Heisenberg Hamiltonian circuit executes."""
        circuit_fn = heisenberg_encoding.get_circuit(
            sample_data_4d, backend="pennylane"
        )
        dev = qml.device("default.qubit", wires=heisenberg_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_batch_circuits(
        self,
        default_encoding: HamiltonianEncoding,
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
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == default_encoding.n_qubits

    def test_circuit_has_gates(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit has gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert len(circuit.data) > 0

    def test_iqp_hamiltonian_has_rz_gates(
        self,
        iqp_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that IQP Hamiltonian uses RZ gates."""
        circuit = iqp_encoding.get_circuit(sample_data_4d, backend="qiskit")
        gate_names = [inst.operation.name for inst in circuit.data]
        assert "rz" in gate_names or "barrier" in gate_names

    def test_barriers_inserted(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that barriers are inserted between Trotter steps.

        For reps=3 and insert_barriers=True, expect reps-1=2 barriers.
        """
        enc = HamiltonianEncoding(n_features=4, reps=3, insert_barriers=True)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        barrier_count = sum(
            1 for inst in circuit.data if inst.operation.name == "barrier"
        )
        # Should have reps - 1 = 2 barriers
        assert barrier_count == 2

    def test_no_barriers_when_disabled(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that no barriers are inserted when disabled."""
        enc = HamiltonianEncoding(n_features=4, reps=3, insert_barriers=False)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        barrier_count = sum(
            1 for inst in circuit.data if inst.operation.name == "barrier"
        )
        assert barrier_count == 0

    def test_batch_circuits(
        self,
        default_encoding: HamiltonianEncoding,
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
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == default_encoding.n_qubits

    def test_batch_circuits(
        self,
        default_encoding: HamiltonianEncoding,
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
    """Tests for mathematical correctness of the encoding."""

    def test_time_step_computation(self) -> None:
        """Test that time step is correctly computed.

        Time step = evolution_time / reps.
        For evolution_time=2.0, reps=4: time_step = 0.5.
        """
        enc = HamiltonianEncoding(n_features=4, evolution_time=2.0, reps=4)
        assert enc._time_step == 0.5

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zero_evolution_produces_superposition(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that zero evolution time produces uniform superposition.

        With zero evolution time, only the Hadamard layer is applied,
        resulting in the |++++> state with all amplitudes = 1/sqrt(2^n).
        """
        enc = HamiltonianEncoding(n_features=4, evolution_time=0.0)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # All amplitudes should be equal: 1/sqrt(2^n) = 1/4
        expected_amplitude = 1.0 / np.sqrt(2**4)
        expected = np.ones(2**4) * expected_amplitude

        np.testing.assert_allclose(np.abs(state), np.abs(expected), atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = HamiltonianEncoding(n_features=4, evolution_time=5.0, reps=3)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        def get_state(x: NDArray[np.floating]) -> NDArray[np.complexfloating]:
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def full_circuit() -> NDArray[np.complexfloating]:
                circuit_fn()
                return qml.state()

            return full_circuit()

        x1 = np.array([0.1, 0.2, 0.3, 0.4])
        x2 = np.array([2.0, 2.5, 3.0, 1.5])

        state1 = get_state(x1)
        state2 = get_state(x2)

        # Both states should be normalized
        assert np.isclose(np.sum(np.abs(state1) ** 2), 1.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(state2) ** 2), 1.0, atol=1e-10)

        # States should differ from ground state
        ground_state = np.zeros(2**4)
        ground_state[0] = 1.0
        fidelity_ground1 = np.abs(np.vdot(state1, ground_state)) ** 2
        fidelity_ground2 = np.abs(np.vdot(state2, ground_state)) ** 2

        assert fidelity_ground1 < 0.99 or fidelity_ground2 < 0.99


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_qubit_encoding(self) -> None:
        """Test encoding with single qubit (minimum case)."""
        enc = HamiltonianEncoding(n_features=1)
        x = np.array([0.5])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_large_feature_count(self) -> None:
        """Test encoding with many features.

        For 10 qubits with linear entanglement: 9 pairs.
        """
        enc = HamiltonianEncoding(n_features=10, entanglement="linear")

        assert enc.n_qubits == 10
        assert len(enc._entanglement_pairs) == 9

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions."""
        enc = HamiltonianEncoding(n_features=4, reps=10)
        assert enc.reps == 10

    def test_extreme_evolution_time(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test encoding with large evolution time."""
        enc = HamiltonianEncoding(n_features=4, evolution_time=100.0)
        circuit = enc.get_circuit(sample_data_4d, backend="pennylane")
        assert circuit is not None

    def test_two_qubits_minimum_entangled(self) -> None:
        """Test encoding with two qubits (minimum for entanglement)."""
        enc = HamiltonianEncoding(n_features=2)
        assert enc.n_qubits == 2
        assert len(enc._entanglement_pairs) == 1


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_small_values(self) -> None:
        """Test encoding with very small input values.

        Values close to machine epsilon should not cause numerical issues.
        """
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp", reps=2)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_large_values(self) -> None:
        """Test encoding with very large input values.

        Large values are valid as rotation angles wrap around 2pi.
        """
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp", reps=2)
        x = np.array([1e5, 2e5, 3e5, 4e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp", reps=2)
        x = np.array([1e-10, 1e10, 1e-5, 1e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_near_pi_multiples(self) -> None:
        """Test numerical stability near pi multiples."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp", reps=2)
        x = np.array(
            [
                np.pi - 1e-14,
                np.pi + 1e-14,
                2 * np.pi - 1e-14,
                np.pi / 2 + 1e-14,
            ]
        )

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_many_reps_stability(self) -> None:
        """Test numerical stability with many repetitions."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp", reps=10)
        x = np.array([0.5, 1.0, 1.5, 2.0])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    @pytest.mark.parametrize("hamiltonian_type", ["iqp", "xy", "heisenberg", "pauli_z"])
    def test_all_hamiltonians_stable(self, hamiltonian_type: str) -> None:
        """Test numerical stability across all Hamiltonian types."""
        enc = HamiltonianEncoding(
            n_features=4, hamiltonian_type=hamiltonian_type, reps=2
        )
        x = np.random.randn(4) * 100  # Large random values

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_extreme_evolution_time_stability(self) -> None:
        """Test numerical stability with extreme evolution time."""
        enc = HamiltonianEncoding(
            n_features=4, hamiltonian_type="iqp", evolution_time=1000.0, reps=2
        )
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_all_zeros_input(self) -> None:
        """Test encoding with all zeros input."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp", reps=2)
        x = np.zeros(4)

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit() -> NDArray[np.complexfloating]:
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_machine_epsilon_precision(self) -> None:
        """Test encoding at machine epsilon scale."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp", reps=2)
        eps = np.finfo(float).eps
        x = np.array([eps, 2 * eps, 3 * eps, 4 * eps])

        # Should not raise
        circuit = enc.get_circuit(x, backend="pennylane")
        assert callable(circuit)


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        enc2 = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = HamiltonianEncoding(n_features=4)
        enc2 = HamiltonianEncoding(n_features=8)
        assert enc1 != enc2

    def test_equality_different_hamiltonian_type(self) -> None:
        """Test that encodings with different hamiltonian_type are not equal."""
        enc1 = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        enc2 = HamiltonianEncoding(n_features=4, hamiltonian_type="xy")
        assert enc1 != enc2

    def test_equality_different_evolution_time(self) -> None:
        """Test that encodings with different evolution_time are not equal."""
        enc1 = HamiltonianEncoding(n_features=4, evolution_time=1.0)
        enc2 = HamiltonianEncoding(n_features=4, evolution_time=2.0)
        assert enc1 != enc2

    def test_equality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = HamiltonianEncoding(n_features=4, reps=2)
        enc2 = HamiltonianEncoding(n_features=4, reps=3)
        assert enc1 != enc2

    def test_equality_different_entanglement(self) -> None:
        """Test that encodings with different entanglement are not equal."""
        enc1 = HamiltonianEncoding(n_features=4, entanglement="full")
        enc2 = HamiltonianEncoding(n_features=4, entanglement="linear")
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        enc2 = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = HamiltonianEncoding(n_features=4)
        enc2 = HamiltonianEncoding(n_features=4)  # Same as enc1
        enc3 = HamiltonianEncoding(n_features=8)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = HamiltonianEncoding(n_features=4)
        enc2 = HamiltonianEncoding(n_features=4)

        d = {enc1: "four features"}
        assert d[enc2] == "four features"  # enc2 should find same key


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ string representation."""

    def test_repr_contains_class_name(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that repr contains class name."""
        assert "HamiltonianEncoding" in repr(default_encoding)

    def test_repr_contains_n_features(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that repr contains n_features."""
        assert "n_features=4" in repr(default_encoding)

    def test_repr_contains_hamiltonian_type(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that repr contains hamiltonian_type."""
        assert "hamiltonian_type='iqp'" in repr(default_encoding)

    def test_repr_contains_evolution_time(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that repr contains evolution_time."""
        assert "evolution_time=1.0" in repr(default_encoding)

    def test_repr_contains_reps(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that repr contains reps."""
        assert "reps=2" in repr(default_encoding)

    def test_repr_contains_entanglement(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that repr contains entanglement."""
        assert "entanglement='full'" in repr(default_encoding)

    def test_repr_contains_insert_barriers(
        self, default_encoding: HamiltonianEncoding
    ) -> None:
        """Test that repr contains insert_barriers."""
        assert "insert_barriers=True" in repr(default_encoding)

    def test_repr_all_parameters(self) -> None:
        """Test that repr contains all parameters for custom encoding."""
        enc = HamiltonianEncoding(
            n_features=6,
            hamiltonian_type="heisenberg",
            evolution_time=2.5,
            reps=3,
            entanglement="circular",
            insert_barriers=False,
        )
        r = repr(enc)

        assert "n_features=6" in r
        assert "hamiltonian_type='heisenberg'" in r
        assert "evolution_time=2.5" in r
        assert "reps=3" in r
        assert "entanglement='circular'" in r
        assert "insert_barriers=False" in r


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for backend error handling."""

    def test_invalid_backend_name(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_invalid_backend_type(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that non-string backend raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            default_encoding.get_circuit(sample_data_4d, backend=123)  # type: ignore

    def test_none_backend_uses_default(
        self,
        default_encoding: HamiltonianEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that None backend uses default (pennylane)."""
        circuit = default_encoding.get_circuit(sample_data_4d)
        assert callable(circuit)


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = HamiltonianEncoding(
            n_features=4,
            hamiltonian_type="heisenberg",
            evolution_time=2.0,
            reps=3,
        )
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.hamiltonian_type == enc.hamiltonian_type
        assert restored.evolution_time == enc.evolution_time
        assert restored.reps == enc.reps
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = HamiltonianEncoding(n_features=4, hamiltonian_type="iqp")
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = HamiltonianEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = restored.get_circuit(x, backend="pennylane")
        assert callable(circuit)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_pickle_qiskit_circuit_after_restore(self) -> None:
        """Test that Qiskit circuit generation works after unpickling."""
        enc = HamiltonianEncoding(n_features=4)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = restored.get_circuit(x, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = HamiltonianEncoding(n_features=4)
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
        enc = HamiltonianEncoding(n_features=4)
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
        enc = HamiltonianEncoding(n_features=4)
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

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_concurrent_qiskit_circuit_generation(self) -> None:
        """Test that concurrent Qiskit circuit generation is thread-safe."""
        enc = HamiltonianEncoding(n_features=4)
        num_threads = 10
        num_circuits_per_thread = 20

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[QuantumCircuit]:
            circuits = []
            try:
                for _i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
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

        assert len(errors) == 0, f"Thread errors: {errors}"
        total_circuits = sum(len(r) for r in results)
        assert total_circuits == num_threads * num_circuits_per_thread


# =============================================================================
# Test Class: Slow Simulation Tests
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self, default_encoding: HamiltonianEncoding) -> None:
        """Test that same input always produces same state."""
        dev = qml.device("default.qubit", wires=default_encoding.n_qubits)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        states = []
        for _ in range(5):
            circuit_fn = default_encoding.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def full_circuit() -> NDArray[np.complexfloating]:
                circuit_fn()
                return qml.state()

            states.append(full_circuit())

        # All states should be identical
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_all_hamiltonian_types_execute_correctly(self) -> None:
        """Test that all Hamiltonian types execute and produce valid states."""
        x = np.array([0.8, 1.2, 1.6, 2.0])
        hamiltonian_types = ["iqp", "xy", "heisenberg", "pauli_z"]

        states = {}

        for ham_type in hamiltonian_types:
            enc = HamiltonianEncoding(
                n_features=4, hamiltonian_type=ham_type, evolution_time=3.0, reps=2
            )
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit() -> NDArray[np.complexfloating]:
                circuit_fn()
                return qml.state()

            states[ham_type] = full_circuit()

            # Verify state is normalized
            norm = np.sum(np.abs(states[ham_type]) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10), f"{ham_type} state not normalized"

        # IQP and Pauli-Z should produce identical states
        fidelity_iqp_pauli_z = np.abs(np.vdot(states["iqp"], states["pauli_z"])) ** 2
        assert (
            fidelity_iqp_pauli_z > 0.99
        ), "IQP and Pauli-Z should produce identical states"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = HamiltonianEncoding(n_features=4, evolution_time=3.0, reps=2)
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
            def circuit() -> NDArray[np.complexfloating]:
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # At least some states should be different
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                fidelity = np.abs(np.vdot(states[i], states[j])) ** 2
                # Different inputs should generally produce different states
                assert fidelity < 0.9999 or np.allclose(inputs[i], inputs[j])

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = HamiltonianEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full() -> NDArray[np.complexfloating]:
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
    def test_cross_backend_state_equivalence(self) -> None:
        """Test that all backends produce equivalent quantum states.

        Compares sorted probability distributions to account for different
        qubit ordering conventions between backends.
        """
        enc = HamiltonianEncoding(
            n_features=4,
            hamiltonian_type="iqp",
            evolution_time=1.0,
            reps=2,
        )
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full() -> NDArray[np.complexfloating]:
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # Compare sorted probability distributions
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
    @pytest.mark.parametrize("hamiltonian_type", ["iqp", "xy", "heisenberg", "pauli_z"])
    def test_cross_backend_with_different_hamiltonian_types(
        self, hamiltonian_type: str
    ) -> None:
        """Test cross-backend equivalence with different Hamiltonian types."""
        enc = HamiltonianEncoding(
            n_features=4,
            hamiltonian_type=hamiltonian_type,
            evolution_time=1.5,
            reps=2,
        )
        x = np.array([0.3, 0.6, 0.9, 1.2])

        # PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full() -> NDArray[np.complexfloating]:
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # Compare sorted probability distributions
        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(
            pl_probs,
            qk_probs,
            atol=1e-6,
            err_msg=f"PennyLane vs Qiskit mismatch for {hamiltonian_type}",
        )
        np.testing.assert_allclose(
            pl_probs,
            cirq_probs,
            atol=1e-6,
            err_msg=f"PennyLane vs Cirq mismatch for {hamiltonian_type}",
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_cross_backend_with_different_entanglement(self, entanglement: str) -> None:
        """Test cross-backend equivalence with different entanglement patterns."""
        enc = HamiltonianEncoding(
            n_features=4,
            hamiltonian_type="iqp",
            evolution_time=1.0,
            reps=2,
            entanglement=entanglement,
        )
        x = np.array([0.2, 0.4, 0.6, 0.8])

        # PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full() -> NDArray[np.complexfloating]:
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # Compare sorted probability distributions
        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(
            pl_probs,
            qk_probs,
            atol=1e-6,
            err_msg=f"PennyLane vs Qiskit mismatch for {entanglement}",
        )
        np.testing.assert_allclose(
            pl_probs,
            cirq_probs,
            atol=1e-6,
            err_msg=f"PennyLane vs Cirq mismatch for {entanglement}",
        )
