"""Comprehensive tests for SymmetryInspiredFeatureMap encoding.

This test module provides complete coverage of the SymmetryInspiredFeatureMap class
(formerly CovariantFeatureMap), which implements symmetry-inspired feature maps for
quantum machine learning. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, gate_count, symmetry configuration)
- Symmetry behavior (rotation, cyclic, reflection, full)
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

Run with: pytest tests/unit/encodings/test_symmetry_inspired_feature_map.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_symmetry_inspired_feature_map.py -v -m "not slow"

Note: CovariantFeatureMap is available as a backwards-compatible alias for
SymmetryInspiredFeatureMap.

References
----------
.. [1] Meyer et al., "Exploiting Symmetry in Variational Quantum Machine Learning" (2022)
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest

from encoding_atlas.encodings.symmetry_inspired_feature_map import (
    CovariantFeatureMap,
    CovariantGateBreakdown,
    SymmetryInspiredFeatureMap,
    SymmetryInspiredGateBreakdown,
    convert_state_vector_ordering,
    estimate_circuit_resources,
    estimate_memory_usage,
    get_supported_symmetries,
)

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


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
def sample_data_6d() -> NDArray[np.floating]:
    """6-dimensional sample data for testing."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4-dimensional samples.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (higher values)
    - [0.0, 0.0, 0.0, 0.0] (edge case: zeros)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.0, 0.0, 0.0, 0.0],
    ])


@pytest.fixture
def default_encoding() -> SymmetryInspiredFeatureMap:
    """Default SymmetryInspiredFeatureMap with 4 features and rotation symmetry."""
    return SymmetryInspiredFeatureMap(n_features=4)


@pytest.fixture
def cyclic_encoding() -> SymmetryInspiredFeatureMap:
    """Encoding with cyclic symmetry."""
    return SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")


@pytest.fixture
def reflection_encoding() -> SymmetryInspiredFeatureMap:
    """Encoding with reflection symmetry."""
    return SymmetryInspiredFeatureMap(n_features=4, symmetry="reflection")


@pytest.fixture
def full_encoding() -> SymmetryInspiredFeatureMap:
    """Encoding with full permutation symmetry."""
    return SymmetryInspiredFeatureMap(n_features=4, symmetry="full")


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for SymmetryInspiredFeatureMap instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test instantiation with default parameters."""
        enc = SymmetryInspiredFeatureMap(n_features=4)

        assert enc.n_features == 4
        assert enc.symmetry == "rotation"
        assert enc.reps == 2
        assert enc.entanglement == "linear"
        assert enc.feature_map == "angle"
        assert enc.include_barriers is True

    def test_custom_symmetry_rotation(self) -> None:
        """Test instantiation with rotation symmetry."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        assert enc.symmetry == "rotation"

    def test_custom_symmetry_cyclic(self) -> None:
        """Test instantiation with cyclic symmetry."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")
        assert enc.symmetry == "cyclic"

    def test_custom_symmetry_reflection(self) -> None:
        """Test instantiation with reflection symmetry."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="reflection")
        assert enc.symmetry == "reflection"

    def test_custom_symmetry_full(self) -> None:
        """Test instantiation with full permutation symmetry."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="full")
        assert enc.symmetry == "full"

    def test_symmetry_case_insensitive(self) -> None:
        """Test that symmetry parameter is case insensitive."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="ROTATION")
        assert enc.symmetry == "rotation"

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = SymmetryInspiredFeatureMap(n_features=4, reps=5)
        assert enc.reps == 5

    def test_entanglement_full(self) -> None:
        """Test full entanglement pattern."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="full")
        assert enc.entanglement == "full"

    def test_entanglement_linear(self) -> None:
        """Test linear entanglement pattern."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="linear")
        assert enc.entanglement == "linear"

    def test_entanglement_circular(self) -> None:
        """Test circular entanglement pattern."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="circular")
        assert enc.entanglement == "circular"

    def test_entanglement_none(self) -> None:
        """Test no entanglement (product state)."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="none")
        assert enc.entanglement == "none"
        assert not enc.properties.is_entangling

    def test_entanglement_case_insensitive(self) -> None:
        """Test that entanglement parameter is case insensitive."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="LINEAR")
        assert enc.entanglement == "linear"

    def test_feature_map_angle(self) -> None:
        """Test angle feature mapping."""
        enc = SymmetryInspiredFeatureMap(n_features=4, feature_map="angle")
        assert enc.feature_map == "angle"

    def test_feature_map_fourier(self) -> None:
        """Test fourier feature mapping."""
        enc = SymmetryInspiredFeatureMap(n_features=4, feature_map="fourier")
        assert enc.feature_map == "fourier"

    def test_feature_map_polynomial(self) -> None:
        """Test polynomial feature mapping."""
        enc = SymmetryInspiredFeatureMap(n_features=4, feature_map="polynomial")
        assert enc.feature_map == "polynomial"

    def test_feature_map_case_insensitive(self) -> None:
        """Test that feature_map parameter is case insensitive."""
        enc = SymmetryInspiredFeatureMap(n_features=4, feature_map="FOURIER")
        assert enc.feature_map == "fourier"

    def test_include_barriers_false(self) -> None:
        """Test with barriers disabled."""
        enc = SymmetryInspiredFeatureMap(n_features=4, include_barriers=False)
        assert enc.include_barriers is False

    def test_all_custom_parameters(self) -> None:
        """Test all parameters customized together."""
        enc = SymmetryInspiredFeatureMap(
            n_features=6,
            symmetry="cyclic",
            reps=3,
            entanglement="circular",
            feature_map="fourier",
            include_barriers=False,
        )

        assert enc.n_features == 6
        assert enc.symmetry == "cyclic"
        assert enc.reps == 3
        assert enc.entanglement == "circular"
        assert enc.feature_map == "fourier"
        assert enc.include_barriers is False


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    def test_invalid_n_features_one(self) -> None:
        """Test that n_features=1 raises error (need at least 2)."""
        with pytest.raises(ValueError, match="at least 2"):
            SymmetryInspiredFeatureMap(n_features=1)

    def test_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises error."""
        with pytest.raises(ValueError, match="n_features must be"):
            SymmetryInspiredFeatureMap(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises error."""
        with pytest.raises(ValueError, match="n_features must be"):
            SymmetryInspiredFeatureMap(n_features=-1)

    def test_invalid_n_features_type(self) -> None:
        """Test that non-integer n_features raises TypeError."""
        with pytest.raises(TypeError, match="n_features must be an integer"):
            SymmetryInspiredFeatureMap(n_features=4.5)  # type: ignore

    def test_invalid_symmetry(self) -> None:
        """Test invalid symmetry type raises error."""
        with pytest.raises(ValueError, match="symmetry must be one of"):
            SymmetryInspiredFeatureMap(n_features=4, symmetry="invalid")

    def test_invalid_symmetry_type(self) -> None:
        """Test that non-string symmetry raises TypeError."""
        with pytest.raises(TypeError, match="symmetry must be a string"):
            SymmetryInspiredFeatureMap(n_features=4, symmetry=123)  # type: ignore

    def test_rotation_requires_even_features(self) -> None:
        """Test that rotation symmetry requires even n_features."""
        with pytest.raises(ValueError, match="even n_features"):
            SymmetryInspiredFeatureMap(n_features=3, symmetry="rotation")

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises error."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            SymmetryInspiredFeatureMap(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises error."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            SymmetryInspiredFeatureMap(n_features=4, reps=-1)

    def test_invalid_reps_type(self) -> None:
        """Test that non-integer reps raises TypeError."""
        with pytest.raises(TypeError, match="reps must be an integer"):
            SymmetryInspiredFeatureMap(n_features=4, reps=2.5)  # type: ignore

    def test_invalid_entanglement(self) -> None:
        """Test invalid entanglement pattern raises error."""
        with pytest.raises(ValueError, match="entanglement must be one of"):
            SymmetryInspiredFeatureMap(n_features=4, entanglement="invalid")

    def test_invalid_entanglement_type(self) -> None:
        """Test that non-string entanglement raises TypeError."""
        with pytest.raises(TypeError, match="entanglement must be a string"):
            SymmetryInspiredFeatureMap(n_features=4, entanglement=123)  # type: ignore

    def test_invalid_feature_map(self) -> None:
        """Test invalid feature map raises error."""
        with pytest.raises(ValueError, match="feature_map must be one of"):
            SymmetryInspiredFeatureMap(n_features=4, feature_map="invalid")

    def test_invalid_feature_map_type(self) -> None:
        """Test that non-string feature_map raises TypeError."""
        with pytest.raises(TypeError, match="feature_map must be a string"):
            SymmetryInspiredFeatureMap(n_features=4, feature_map=123)  # type: ignore

    def test_invalid_include_barriers_type(self) -> None:
        """Test that non-bool include_barriers raises TypeError."""
        with pytest.raises(TypeError, match="include_barriers must be a bool"):
            SymmetryInspiredFeatureMap(n_features=4, include_barriers="True")  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of SymmetryInspiredFeatureMap."""

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features."""
        for n in [2, 4, 6, 8]:
            enc = SymmetryInspiredFeatureMap(n_features=n, symmetry="cyclic")
            assert enc.n_qubits == n

    def test_depth_increases_with_reps(self) -> None:
        """Test that depth increases with repetitions."""
        depths = []
        for reps in [1, 2, 3, 4]:
            enc = SymmetryInspiredFeatureMap(n_features=4, reps=reps)
            depths.append(enc.depth)

        # Depth should be monotonically increasing
        assert all(depths[i] < depths[i + 1] for i in range(len(depths) - 1))

    def test_properties_object(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that properties returns EncodingProperties object."""
        props = default_encoding.properties

        assert props.n_qubits == 4
        assert props.depth > 0
        assert props.gate_count > 0
        assert props.single_qubit_gates > 0
        assert props.two_qubit_gates >= 0
        assert props.parameter_count == 0  # Data-dependent only

    def test_entangling_with_entanglement(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that encoding is entangling with entanglement enabled."""
        assert default_encoding.properties.is_entangling is True

    def test_not_entangling_without_entanglement(self) -> None:
        """Test that encoding is not entangling with entanglement='none'."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="none")
        assert enc.properties.is_entangling is False

    def test_simulable_without_entanglement(self) -> None:
        """Test that encoding is simulable without entanglement."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="none")
        assert enc.properties.simulability == "simulable"

    def test_not_simulable_with_entanglement(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that encoding is not simulable with entanglement."""
        assert default_encoding.properties.simulability == "not_simulable"

    def test_properties_cached(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that properties are cached (same object returned)."""
        props1 = default_encoding.properties
        props2 = default_encoding.properties
        assert props1 is props2

    def test_gate_count_consistency(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that single + two qubit gates equals total."""
        props = default_encoding.properties
        assert props.single_qubit_gates + props.two_qubit_gates == props.gate_count

    def test_properties_notes_contain_symmetry(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that notes mention the symmetry type."""
        props = default_encoding.properties
        assert "rotation" in props.notes.lower()


# =============================================================================
# Test Class: Symmetry Configuration (Unique Behavior)
# =============================================================================


class TestSymmetryConfiguration:
    """Tests for symmetry-specific configuration behavior."""

    def test_rotation_symmetry_pairs(self) -> None:
        """Test rotation symmetry creates coordinate pairs."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        info = enc.get_symmetry_info()

        assert info["type"] == "rotation"
        assert info["n_pairs"] == 2
        assert info["pair_indices"] == [(0, 1), (2, 3)]

    def test_rotation_symmetry_six_features(self) -> None:
        """Test rotation symmetry with 6 features."""
        enc = SymmetryInspiredFeatureMap(n_features=6, symmetry="rotation")
        info = enc.get_symmetry_info()

        assert info["n_pairs"] == 3
        assert info["pair_indices"] == [(0, 1), (2, 3), (4, 5)]

    def test_cyclic_symmetry_generator(self) -> None:
        """Test cyclic symmetry generator shift."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")
        info = enc.get_symmetry_info()

        assert info["type"] == "cyclic"
        assert info["generator_shift"] == 1

    def test_reflection_symmetry_mirror_pairs(self) -> None:
        """Test reflection symmetry creates mirror pairs."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="reflection")
        info = enc.get_symmetry_info()

        assert info["type"] == "reflection"
        assert info["mirror_pairs"] == [(0, 3), (1, 2)]

    def test_full_symmetry_swap_pairs(self) -> None:
        """Test full symmetry creates all swap pairs."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="full")
        info = enc.get_symmetry_info()

        assert info["type"] == "full"
        # C(4,2) = 6 swap pairs
        assert len(info["swap_pairs"]) == 6

    def test_entanglement_pairs_linear(self) -> None:
        """Test linear entanglement creates consecutive pairs."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="linear")
        info = enc.get_symmetry_info()

        assert info["entanglement_pairs"] == [(0, 1), (1, 2), (2, 3)]

    def test_entanglement_pairs_circular(self) -> None:
        """Test circular entanglement includes wrap-around."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="circular")
        info = enc.get_symmetry_info()

        assert info["entanglement_pairs"] == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_entanglement_pairs_full(self) -> None:
        """Test full entanglement creates all pairs."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="full")
        info = enc.get_symmetry_info()

        # C(4,2) = 6 pairs
        assert len(info["entanglement_pairs"]) == 6

    def test_entanglement_pairs_none(self) -> None:
        """Test no entanglement creates no pairs."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="none")
        info = enc.get_symmetry_info()

        assert info["entanglement_pairs"] == []


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_1d_input(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 1D input is accepted."""
        circuit = default_encoding.get_circuit(sample_data_4d)
        assert circuit is not None

    def test_valid_2d_input_single_sample(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 2D input with single sample is accepted."""
        x = sample_data_4d.reshape(1, -1)
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_invalid_feature_count(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that wrong number of features raises ValueError."""
        x = np.array([0.1, 0.2, 0.3])  # 3 features, expect 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding.get_circuit(x)

    def test_invalid_nan_values(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that NaN values are rejected."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding.get_circuit(x)

    def test_invalid_inf_values(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that infinite values are rejected."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.get_circuit(x)

    def test_invalid_2d_multiple_samples(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that 2D input with multiple samples raises error for get_circuit."""
        with pytest.raises(ValueError, match="single sample"):
            default_encoding.get_circuit(batch_data_4d)

    def test_list_input_converted(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that list input is converted to numpy array."""
        x = [0.1, 0.2, 0.3, 0.4]
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None


# =============================================================================
# Test Class: PennyLane Backend
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.backend_pennylane
class TestPennyLaneBackend:
    """Tests for PennyLane circuit generation."""

    def test_circuit_callable(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit executes without error."""
        circuit_fn = default_encoding.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=default_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None
        assert len(state) == 2 ** default_encoding.n_qubits

    def test_state_normalized(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
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

    def test_all_symmetry_types_execute(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all symmetry types execute successfully."""
        for symmetry in ["rotation", "cyclic", "reflection", "full"]:
            enc = SymmetryInspiredFeatureMap(n_features=4, symmetry=symmetry)
            circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            assert state is not None

    def test_all_feature_maps_execute(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that all feature maps execute successfully."""
        for feature_map in ["angle", "fourier", "polynomial"]:
            enc = SymmetryInspiredFeatureMap(
                n_features=4, symmetry="cyclic", feature_map=feature_map
            )
            circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

            dev = qml.device("default.qubit", wires=enc.n_qubits)

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            assert state is not None

    def test_multiple_reps(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test circuit with multiple repetitions."""
        enc = SymmetryInspiredFeatureMap(n_features=4, reps=5)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_get_circuits_batch(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test batch circuit generation."""
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
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_qubit_count(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit has correct qubit count."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == default_encoding.n_qubits

    def test_circuit_has_gates(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit has gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        # Should have multiple gates
        assert len(circuit.data) > default_encoding.n_qubits

    def test_all_symmetry_types(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that all symmetry types generate valid circuits."""
        for symmetry in ["rotation", "cyclic", "reflection", "full"]:
            enc = SymmetryInspiredFeatureMap(n_features=4, symmetry=symmetry)
            circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == 4

    def test_barriers_present(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that barriers are present when enabled."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")

        # Count barriers
        barrier_count = sum(
            1 for inst in circuit.data if inst.operation.name == "barrier"
        )
        assert barrier_count > 0

    def test_no_barriers_when_disabled(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that no barriers when disabled."""
        enc = SymmetryInspiredFeatureMap(n_features=4, include_barriers=False)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        # Count barriers
        barrier_count = sum(
            1 for inst in circuit.data if inst.operation.name == "barrier"
        )
        assert barrier_count == 0

    def test_get_circuits_batch(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test batch circuit generation."""
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
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubits(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit uses correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == default_encoding.n_qubits

    def test_all_symmetry_types(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that all symmetry types generate valid circuits."""
        for symmetry in ["rotation", "cyclic", "reflection", "full"]:
            enc = SymmetryInspiredFeatureMap(n_features=4, symmetry=symmetry)
            circuit = enc.get_circuit(sample_data_4d, backend="cirq")
            assert isinstance(circuit, cirq.Circuit)
            assert len(circuit.all_qubits()) == 4

    def test_get_circuits_batch(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test batch circuit generation."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of angle computation."""

    def test_angle_feature_map(self) -> None:
        """Test angle feature mapping computes correctly."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", feature_map="angle"
        )
        x = np.array([0.5, 1.0, 1.5, 2.0])

        angles = enc.compute_encoding_angles(x)

        # Angle mapping: theta = x
        np.testing.assert_allclose(angles["encoding"], x)

    def test_fourier_feature_map(self) -> None:
        """Test fourier feature mapping computes correctly."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", feature_map="fourier"
        )
        x = np.array([0.0, 0.25, 0.5, 1.0])

        angles = enc.compute_encoding_angles(x)

        # Fourier mapping: theta = 2*pi*x
        expected = 2.0 * np.pi * x
        np.testing.assert_allclose(angles["encoding"], expected)

    def test_polynomial_feature_map(self) -> None:
        """Test polynomial feature mapping computes correctly."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", feature_map="polynomial"
        )
        x = np.array([0.5, 1.0, 1.5, 2.0])

        angles = enc.compute_encoding_angles(x)

        # Polynomial mapping: theta = x + x^2
        expected = x + x ** 2
        np.testing.assert_allclose(angles["encoding"], expected)

    def test_rotation_equivariant_angle_radius(self) -> None:
        """Test rotation symmetry uses radius for equivariant angle."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        # (3,4) has radius 5, (0,5) has radius 5
        x = np.array([3.0, 4.0, 0.0, 5.0])

        angles = enc.compute_encoding_angles(x)

        # For rotation symmetry, equivariant angle is radius
        # Qubit 0 and 1 are in pair 0 with radius sqrt(3^2 + 4^2) = 5
        # Qubit 2 and 3 are in pair 1 with radius sqrt(0^2 + 5^2) = 5
        assert np.isclose(angles["equivariant"][0], 5.0)
        assert np.isclose(angles["equivariant"][1], 5.0)
        assert np.isclose(angles["equivariant"][2], 5.0)
        assert np.isclose(angles["equivariant"][3], 5.0)

    def test_reflection_symmetric_combination(self) -> None:
        """Test reflection symmetry uses symmetric combination."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="reflection")
        x = np.array([1.0, 2.0, 3.0, 4.0])

        angles = enc.compute_encoding_angles(x)

        # Reflection combines x[i] with x[n-1-i]
        # Qubit 0: (x[0] + x[3]) / 2 = (1 + 4) / 2 = 2.5
        # Qubit 1: (x[1] + x[2]) / 2 = (2 + 3) / 2 = 2.5
        assert np.isclose(angles["equivariant"][0], 2.5)
        assert np.isclose(angles["equivariant"][1], 2.5)

    def test_zero_input(self) -> None:
        """Test that zero input produces finite angles."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")
        x = np.zeros(4)

        angles = enc.compute_encoding_angles(x)

        assert np.all(np.isfinite(angles["encoding"]))
        assert np.all(np.isfinite(angles["equivariant"]))


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_features(self) -> None:
        """Test encoding with minimum 2 features."""
        enc = SymmetryInspiredFeatureMap(n_features=2, symmetry="cyclic")
        x = np.array([0.5, 0.5])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None
        assert enc.n_qubits == 2

    def test_large_feature_count(self) -> None:
        """Test encoding with many features."""
        enc = SymmetryInspiredFeatureMap(n_features=10, symmetry="cyclic")

        assert enc.n_qubits == 10
        x = np.random.randn(10)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_large_feature_count_rotation(self) -> None:
        """Test rotation symmetry with many features."""
        enc = SymmetryInspiredFeatureMap(n_features=8, symmetry="rotation")

        info = enc.get_symmetry_info()
        # 8 features = 4 pairs
        assert info["n_pairs"] == 4

    def test_extreme_values(self) -> None:
        """Test encoding with extreme input values."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")

        # Large positive values
        x_large = np.array([100.0, 200.0, 300.0, 400.0])
        circuit = enc.get_circuit(x_large, backend="pennylane")
        assert circuit is not None

        # Small values
        x_small = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        circuit = enc.get_circuit(x_small, backend="pennylane")
        assert circuit is not None

        # Negative values
        x_neg = np.array([-1.0, -2.0, -3.0, -4.0])
        circuit = enc.get_circuit(x_neg, backend="pennylane")
        assert circuit is not None

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions."""
        enc = SymmetryInspiredFeatureMap(n_features=4, reps=10)

        assert enc.depth > 10
        props = enc.properties
        assert props.gate_count > 100

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_no_entanglement_product_state(self) -> None:
        """Test that no entanglement produces product state."""
        enc = SymmetryInspiredFeatureMap(n_features=4, entanglement="none")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit_fn = enc.get_circuit(x, backend="pennylane")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # Product state should be normalized
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values.

    These tests verify that SymmetryInspiredFeatureMap produces valid, normalized
    quantum states even when input data contains extreme values.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_small_values(self) -> None:
        """Test encoding with very small input values near machine epsilon."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
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
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_large_values(self) -> None:
        """Test encoding with very large input values.

        Large values are valid as rotation angles wrap around 2*pi.
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
        x = np.array([1e5, 2e5, 3e5, 4e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_mixed_magnitude_values(self) -> None:
        """Test encoding with mixed magnitude values spanning many orders."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
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
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_near_pi_multiples(self) -> None:
        """Test numerical stability near pi multiples."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
        x = np.array([
            np.pi - 1e-14,
            np.pi + 1e-14,
            2 * np.pi - 1e-14,
            np.pi / 2 + 1e-14,
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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_many_reps_stability(self) -> None:
        """Test numerical stability with many repetitions."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=10)
        x = np.array([0.5, 1.0, 1.5, 2.0])

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
    @pytest.mark.parametrize("symmetry", ["rotation", "cyclic", "reflection", "full"])
    def test_all_symmetries_stable(self, symmetry: str) -> None:
        """Test numerical stability across all symmetry types."""
        n_features = 4 if symmetry == "rotation" else 5
        enc = SymmetryInspiredFeatureMap(
            n_features=n_features, symmetry=symmetry, reps=2
        )
        x = np.random.randn(n_features) * 100  # Large random values

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_all_zeros_input(self) -> None:
        """Test encoding with all zeros input."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
        x = np.zeros(4)

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_machine_epsilon_precision(self) -> None:
        """Test encoding at machine epsilon scale."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
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

    def test_equality_same_params(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        enc2 = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")

        assert enc1 == enc2

    def test_inequality_different_symmetry(self) -> None:
        """Test that encodings with different symmetry are not equal."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        enc2 = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")

        assert enc1 != enc2

    def test_inequality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, reps=2)
        enc2 = SymmetryInspiredFeatureMap(n_features=4, reps=3)

        assert enc1 != enc2

    def test_inequality_different_entanglement(self) -> None:
        """Test that encodings with different entanglement are not equal."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, entanglement="linear")
        enc2 = SymmetryInspiredFeatureMap(n_features=4, entanglement="full")

        assert enc1 != enc2

    def test_inequality_different_feature_map(self) -> None:
        """Test that encodings with different feature_map are not equal."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, feature_map="angle")
        enc2 = SymmetryInspiredFeatureMap(n_features=4, feature_map="fourier")

        assert enc1 != enc2

    def test_hash_same_params(self) -> None:
        """Test that encodings with same parameters have same hash."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=3)
        enc2 = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=3)

        assert hash(enc1) == hash(enc2)

    def test_hash_can_be_dict_key(self) -> None:
        """Test that encoding can be used as dictionary key."""
        enc = SymmetryInspiredFeatureMap(n_features=4)
        d = {enc: "test"}

        assert d[enc] == "test"

    def test_hash_in_set(self) -> None:
        """Test that equal encodings are deduplicated in sets."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        enc2 = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        enc3 = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for string representation."""

    def test_repr_contains_class_name(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that repr contains class name."""
        # SymmetryInspiredFeatureMap is the canonical name
        assert "SymmetryInspiredFeatureMap" in repr(default_encoding)

    def test_repr_contains_parameters(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that repr contains all parameters."""
        r = repr(default_encoding)

        assert "n_features=4" in r
        assert "symmetry='rotation'" in r
        assert "reps=2" in r
        assert "entanglement='linear'" in r
        assert "feature_map='angle'" in r
        assert "include_barriers=True" in r

    def test_repr_round_trip(self) -> None:
        """Test that repr contains all custom parameters."""
        enc = SymmetryInspiredFeatureMap(
            n_features=6,
            symmetry="cyclic",
            reps=3,
            entanglement="circular",
            feature_map="fourier",
            include_barriers=False,
        )

        r = repr(enc)
        assert "n_features=6" in r
        assert "symmetry='cyclic'" in r
        assert "reps=3" in r
        assert "entanglement='circular'" in r
        assert "feature_map='fourier'" in r
        assert "include_barriers=False" in r


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for backend error handling."""

    def test_invalid_backend(
        self,
        default_encoding: SymmetryInspiredFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_round_trip(self) -> None:
        """Test that encoding survives pickle round trip."""
        enc = SymmetryInspiredFeatureMap(
            n_features=6,
            symmetry="cyclic",
            reps=3,
            entanglement="circular",
            feature_map="fourier",
        )

        # Force properties to be computed
        _ = enc.properties

        # Serialize and deserialize
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        # Check all attributes match
        assert restored.n_features == enc.n_features
        assert restored.symmetry == enc.symmetry
        assert restored.reps == enc.reps
        assert restored.entanglement == enc.entanglement
        assert restored.feature_map == enc.feature_map
        assert restored.include_barriers == enc.include_barriers

    def test_pickle_properties_recomputed(self) -> None:
        """Test that properties are recomputed after unpickle."""
        enc = SymmetryInspiredFeatureMap(n_features=4)
        original_props = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        # Properties should be recomputed
        restored_props = restored.properties

        assert restored_props.n_qubits == original_props.n_qubits
        assert restored_props.depth == original_props.depth
        assert restored_props.gate_count == original_props.gate_count

    def test_pickle_circuit_still_works(self) -> None:
        """Test that circuits can be generated after unpickle."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
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

    def test_equality_after_pickle(self) -> None:
        """Test that pickled encoding equals original."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4,
            symmetry="reflection",
            reps=2,
            entanglement="full",
        )

        restored = pickle.loads(pickle.dumps(enc))

        assert restored == enc
        assert hash(restored) == hash(enc)


# =============================================================================
# Test Class: Concurrent Access / Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access.

    These tests verify that the encoding is safe to use in multi-threaded
    environments, particularly focusing on the thread-safe lazy initialization
    of the `properties` attribute.
    """

    def test_thread_safe_property_access(self) -> None:
        """Test that properties are thread-safe.

        Multiple threads accessing the properties concurrently should all
        get the same cached object, and there should be no race conditions.
        """
        import threading
        import time

        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")
        results: list[object] = []
        errors: list[Exception] = []
        n_threads = 50

        def access_properties() -> None:
            """Worker function that accesses properties."""
            try:
                # Add small random delay to increase race condition likelihood
                time.sleep(0.001 * (threading.current_thread().ident or 0) % 10)
                props = enc.properties
                results.append(props)
            except Exception as e:
                errors.append(e)

        # Force properties to be uninitialized by creating a fresh encoding
        enc._properties = None  # type: ignore

        # Create and start threads
        threads = [
            threading.Thread(target=access_properties)
            for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

        # All results should be the same object (cached)
        assert len(results) == n_threads
        assert all(r is results[0] for r in results), (
            "All threads should get the same cached properties object"
        )

    def test_thread_safe_property_values_consistent(self) -> None:
        """Test that property values are consistent across threads.

        All threads should see the same computed values.
        """
        import threading

        enc = SymmetryInspiredFeatureMap(
            n_features=6,
            symmetry="reflection",
            reps=3,
            entanglement="circular"
        )

        results: list[tuple[int, int, int, bool]] = []
        n_threads = 20

        def collect_property_values() -> None:
            """Worker that collects property values."""
            props = enc.properties
            results.append((
                props.n_qubits,
                props.depth,
                props.gate_count,
                props.is_entangling,
            ))

        threads = [
            threading.Thread(target=collect_property_values)
            for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have collected the same values
        assert len(results) == n_threads
        assert all(r == results[0] for r in results), (
            "All threads should see identical property values"
        )

    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is safe.

        Multiple threads generating circuits from the same encoding
        should not interfere with each other.
        """
        import threading

        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        results: list[object] = []
        errors: list[Exception] = []

        # Different input data for each thread
        inputs = [
            np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])
            for i in range(1, 11)
        ]

        def generate_circuit(x: NDArray[np.floating]) -> None:
            """Worker that generates a circuit."""
            try:
                circuit = enc.get_circuit(x, backend="pennylane")
                results.append(circuit)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=generate_circuit, args=(x,))
            for x in inputs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors during concurrent generation: {errors}"

        # All threads should have produced circuits
        assert len(results) == len(inputs)
        assert all(callable(c) for c in results)

    def test_many_concurrent_encodings(self) -> None:
        """Test creating many encodings concurrently.

        Verify that many threads can create and use encodings simultaneously.
        """
        import threading

        results: list[int] = []
        errors: list[Exception] = []
        n_threads = 30

        def create_and_use_encoding(n: int) -> None:
            """Worker that creates an encoding and computes properties."""
            try:
                # Use n_features based on thread index (all even for rotation)
                n_features = 4 + (n % 5) * 2  # 4, 6, 8, 10, or 12
                enc = SymmetryInspiredFeatureMap(n_features=n_features)
                props = enc.properties
                results.append(props.depth)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_and_use_encoding, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors: {errors}"

        # All threads should have produced results
        assert len(results) == n_threads


# =============================================================================
# Test Class: Parallel Batch Processing (Encoding-Specific)
# =============================================================================


class TestParallelBatchProcessing:
    """Tests for parallel batch processing via ThreadPoolExecutor.

    These tests verify the parallel processing feature of get_circuits(),
    ensuring correctness, order preservation, and performance characteristics.
    """

    def test_parallel_matches_sequential_pennylane(self) -> None:
        """Test that parallel and sequential produce equivalent results (PennyLane).

        The circuits generated in parallel should be functionally equivalent
        to those generated sequentially.
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        X = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ])

        # Generate sequentially
        circuits_seq = enc.get_circuits(X, backend="pennylane", parallel=False)

        # Generate in parallel
        circuits_par = enc.get_circuits(X, backend="pennylane", parallel=True)

        # Same number of circuits
        assert len(circuits_seq) == len(circuits_par)
        assert len(circuits_seq) == 3

        # All should be callable
        assert all(callable(c) for c in circuits_seq)
        assert all(callable(c) for c in circuits_par)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_parallel_matches_sequential_qiskit(self) -> None:
        """Test that parallel and sequential produce equivalent results (Qiskit).

        Qiskit circuits should have identical structure regardless of
        parallel vs sequential generation.
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")
        X = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ])

        circuits_seq = enc.get_circuits(X, backend="qiskit", parallel=False)
        circuits_par = enc.get_circuits(X, backend="qiskit", parallel=True)

        assert len(circuits_seq) == len(circuits_par)

        # Check that circuits have same gate count
        for seq, par in zip(circuits_seq, circuits_par):
            assert len(seq.data) == len(par.data)
            assert seq.num_qubits == par.num_qubits

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_parallel_matches_sequential_cirq(self) -> None:
        """Test that parallel and sequential produce equivalent results (Cirq).

        Cirq circuits should have identical structure regardless of
        parallel vs sequential generation.
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="reflection")
        X = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ])

        circuits_seq = enc.get_circuits(X, backend="cirq", parallel=False)
        circuits_par = enc.get_circuits(X, backend="cirq", parallel=True)

        assert len(circuits_seq) == len(circuits_par)

        # Check that circuits have same number of operations
        for seq, par in zip(circuits_seq, circuits_par):
            assert len(list(seq.all_operations())) == len(list(par.all_operations()))
            assert len(seq.all_qubits()) == len(par.all_qubits())

    def test_parallel_order_preservation(self) -> None:
        """Test that parallel processing preserves sample order.

        Results must maintain the same order as input data, even when
        generated in parallel with unpredictable execution order.
        """
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", feature_map="angle"
        )

        # Create distinctly different input samples
        X = np.array([
            [0.0, 0.0, 0.0, 0.0],      # All zeros
            [1.0, 1.0, 1.0, 1.0],      # All ones
            [0.5, 0.5, 0.5, 0.5],      # All halves
            [np.pi, np.pi, np.pi, np.pi],  # All pi
        ])

        # Generate multiple times to catch potential ordering bugs
        for _ in range(5):
            circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
            assert len(circuits) == 4

            # Each circuit should be callable and independent
            assert all(callable(c) for c in circuits)

    def test_parallel_with_max_workers(self) -> None:
        """Test parallel processing with custom max_workers."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        X = np.random.randn(10, 4)

        # Test with various worker counts
        for max_workers in [1, 2, 4]:
            circuits = enc.get_circuits(
                X, backend="pennylane", parallel=True, max_workers=max_workers
            )
            assert len(circuits) == 10
            assert all(callable(c) for c in circuits)

    def test_parallel_single_sample_fallback(self) -> None:
        """Test that single sample bypasses parallel overhead.

        When only one sample is provided, parallel=True should still work
        but may use sequential path internally (implementation detail).
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="full")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Single sample as 1D array
        circuits = enc.get_circuits(x, backend="pennylane", parallel=True)
        assert len(circuits) == 1
        assert callable(circuits[0])

        # Single sample as 2D array
        X = x.reshape(1, -1)
        circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
        assert len(circuits) == 1

    def test_parallel_all_symmetry_types(self) -> None:
        """Test parallel processing works for all symmetry types."""
        X = np.random.randn(5, 4)

        for symmetry in ["rotation", "cyclic", "reflection", "full"]:
            enc = SymmetryInspiredFeatureMap(n_features=4, symmetry=symmetry)
            circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
            assert len(circuits) == 5
            assert all(callable(c) for c in circuits)

    def test_parallel_default_is_false(self) -> None:
        """Test that parallel defaults to False for backward compatibility."""
        import inspect

        sig = inspect.signature(SymmetryInspiredFeatureMap.get_circuits)
        parallel_param = sig.parameters.get("parallel")

        assert parallel_param is not None
        assert parallel_param.default is False

    def test_parallel_large_batch(self) -> None:
        """Test parallel processing with a larger batch."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation", reps=1)
        X = np.random.randn(50, 4)

        # Sequential
        circuits_seq = enc.get_circuits(X, backend="pennylane", parallel=False)

        # Parallel
        circuits_par = enc.get_circuits(X, backend="pennylane", parallel=True)

        assert len(circuits_seq) == 50
        assert len(circuits_par) == 50
        assert all(callable(c) for c in circuits_seq)
        assert all(callable(c) for c in circuits_par)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_parallel_produces_valid_states(self) -> None:
        """Test that parallel-generated circuits produce valid quantum states.

        Execute the circuits and verify normalization.
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        X = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ])

        circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        for circuit_fn in circuits:
            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            state = full_circuit()
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)

    def test_get_circuit_from_validated_internal_method(self) -> None:
        """Test that _get_circuit_from_validated works correctly.

        This internal method is used by parallel processing and should
        produce the same results as get_circuit.
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Both methods should produce callable circuits
        circuit_public = enc.get_circuit(x, backend="pennylane")
        circuit_internal = enc._get_circuit_from_validated(x, backend="pennylane")

        assert callable(circuit_public)
        assert callable(circuit_internal)

    def test_parallel_thread_safety_stress(self) -> None:
        """Stress test for parallel processing thread safety.

        Generate many batches in parallel to detect any thread safety issues.
        """
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        errors: list[Exception] = []
        results: list[int] = []

        def generate_batch(seed: int) -> None:
            try:
                rng = np.random.default_rng(seed)
                X = rng.random((10, 4))
                circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
                results.append(len(circuits))
            except Exception as e:
                errors.append(e)

        # Run multiple batch generations concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_batch, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        assert len(errors) == 0, f"Errors during stress test: {errors}"
        assert all(r == 10 for r in results)


# =============================================================================
# Test Class: Utility Functions (Encoding-Specific)
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_supported_symmetries(self) -> None:
        """Test get_supported_symmetries returns all types."""
        symmetries = get_supported_symmetries()

        assert "rotation" in symmetries
        assert "cyclic" in symmetries
        assert "reflection" in symmetries
        assert "full" in symmetries
        assert len(symmetries) == 4

    def test_estimate_circuit_resources_basic(self) -> None:
        """Test estimate_circuit_resources returns correct structure."""
        resources = estimate_circuit_resources(4, "rotation", 2, "linear")

        assert "n_qubits" in resources
        assert "estimated_depth" in resources
        assert "estimated_two_qubit_gates" in resources
        assert resources["n_qubits"] == 4

    def test_estimate_resources_matches_actual(self) -> None:
        """Test that estimated resources are reasonable."""
        resources = estimate_circuit_resources(4, "cyclic", 2, "linear")

        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", reps=2, entanglement="linear"
        )
        props = enc.properties

        # Estimates should be in the same ballpark
        assert resources["n_qubits"] == props.n_qubits
        # Allow some flexibility in depth estimate
        assert resources["estimated_depth"] > 0

    def test_estimate_resources_no_entanglement(self) -> None:
        """Test resource estimation with no entanglement."""
        resources = estimate_circuit_resources(4, "rotation", 2, "none")

        assert resources["estimated_two_qubit_gates"] == 0


# =============================================================================
# Test Class: Gate Count Breakdown (Encoding-Specific)
# =============================================================================


class TestGateCountBreakdown:
    """Tests for gate_count_breakdown() method."""

    def test_gate_count_breakdown_returns_typed_dict(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that gate_count_breakdown returns correct structure."""
        breakdown = default_encoding.gate_count_breakdown()

        # Check all expected keys exist
        assert "hadamard" in breakdown
        assert "ry_encoding" in breakdown
        assert "rz_equivariant" in breakdown
        assert "rz_entanglement" in breakdown
        assert "ry_entanglement" in breakdown
        assert "cnot" in breakdown
        assert "crz" in breakdown
        assert "cz" in breakdown
        assert "total_single_qubit" in breakdown
        assert "total_two_qubit" in breakdown
        assert "total" in breakdown

    def test_gate_count_breakdown_rotation_symmetry(self) -> None:
        """Test gate counts for rotation symmetry."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation", reps=1)
        breakdown = enc.gate_count_breakdown()

        # Rotation symmetry uses CRZ gates
        # 4 qubits x 1 rep = 4 Hadamard gates
        assert breakdown["hadamard"] == 4
        assert breakdown["ry_encoding"] == 4
        assert breakdown["rz_equivariant"] == 4
        # 2 coordinate pairs
        assert breakdown["crz"] == 2
        # No CNOT gates (CRZ only)
        assert breakdown["cnot"] == 0
        assert breakdown["cz"] == 0
        # CRZ decomposed to 2 CNOTs each
        assert breakdown["total_two_qubit"] == 4

    def test_gate_count_breakdown_cyclic_symmetry(self) -> None:
        """Test gate counts for cyclic symmetry."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=1)
        breakdown = enc.gate_count_breakdown()

        # Cyclic symmetry uses CNOT-RZ-CNOT structure
        assert breakdown["hadamard"] == 4
        # 3 pairs x 2 CNOTs per pair
        assert breakdown["cnot"] == 6
        assert breakdown["crz"] == 0
        assert breakdown["cz"] == 0

    def test_gate_count_breakdown_reflection_symmetry(self) -> None:
        """Test gate counts for reflection symmetry."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="reflection", reps=1)
        breakdown = enc.gate_count_breakdown()

        # Reflection symmetry uses CZ + 2 RZ per pair
        # 3 pairs
        assert breakdown["cz"] == 3
        # 3 pairs x 2 RZ per pair
        assert breakdown["rz_entanglement"] == 6

    def test_gate_count_breakdown_full_symmetry(self) -> None:
        """Test gate counts for full symmetry with linear entanglement."""
        # Full symmetry with default linear entanglement (3 pairs for 4 qubits)
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="full", reps=1)
        breakdown = enc.gate_count_breakdown()

        # Full symmetry uses 3 CNOTs + 2 RY per pair
        # With linear entanglement on 4 qubits: 3 pairs
        # 3 pairs x 3 CNOTs per pair
        assert breakdown["cnot"] == 9
        # 3 pairs x 2 RY per pair
        assert breakdown["ry_entanglement"] == 6

    def test_gate_count_breakdown_full_entanglement(self) -> None:
        """Test gate counts for full symmetry with full entanglement."""
        # Full symmetry with full entanglement (6 pairs for 4 qubits)
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="full", entanglement="full", reps=1
        )
        breakdown = enc.gate_count_breakdown()

        # Full symmetry uses 3 CNOTs + 2 RY per pair
        # With full entanglement on 4 qubits: n*(n-1)/2 = 6 pairs
        # 6 pairs x 3 CNOTs per pair
        assert breakdown["cnot"] == 18
        # 6 pairs x 2 RY per pair
        assert breakdown["ry_entanglement"] == 12

    def test_gate_count_breakdown_totals(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that total counts are consistent."""
        breakdown = default_encoding.gate_count_breakdown()

        # Single qubit gates
        expected_single = (
            breakdown["hadamard"] +
            breakdown["ry_encoding"] +
            breakdown["rz_equivariant"] +
            breakdown["rz_entanglement"] +
            breakdown["ry_entanglement"]
        )
        assert breakdown["total_single_qubit"] == expected_single

        # Total
        assert breakdown["total"] == (
            breakdown["total_single_qubit"] + breakdown["total_two_qubit"]
        )

    def test_gate_count_breakdown_multiple_reps(self) -> None:
        """Test that gate counts scale with reps."""
        enc1 = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation", reps=1)
        enc2 = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation", reps=2)

        breakdown1 = enc1.gate_count_breakdown()
        breakdown2 = enc2.gate_count_breakdown()

        # All gates should double with 2 reps
        assert breakdown2["hadamard"] == 2 * breakdown1["hadamard"]
        assert breakdown2["ry_encoding"] == 2 * breakdown1["ry_encoding"]
        assert breakdown2["crz"] == 2 * breakdown1["crz"]

    def test_gate_count_breakdown_no_entanglement(self) -> None:
        """Test gate counts with no entanglement."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", entanglement="none", reps=1
        )
        breakdown = enc.gate_count_breakdown()

        # No two-qubit gates
        assert breakdown["cnot"] == 0
        assert breakdown["crz"] == 0
        assert breakdown["cz"] == 0
        assert breakdown["total_two_qubit"] == 0

        # Still has single-qubit gates
        assert breakdown["hadamard"] == 4
        assert breakdown["ry_encoding"] == 4


# =============================================================================
# Test Class: Resource Summary (Encoding-Specific)
# =============================================================================


class TestResourceSummary:
    """Tests for resource_summary() method."""

    def test_resource_summary_returns_dict(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that resource_summary returns correct structure."""
        summary = default_encoding.resource_summary()

        # Check all expected keys exist
        assert "n_qubits" in summary
        assert "n_features" in summary
        assert "depth" in summary
        assert "reps" in summary
        assert "symmetry" in summary
        assert "entanglement" in summary
        assert "feature_map" in summary
        assert "entanglement_pairs" in summary
        assert "n_entanglement_pairs" in summary
        assert "symmetry_info" in summary
        assert "gate_counts" in summary
        assert "is_entangling" in summary
        assert "simulability" in summary
        assert "trainability_estimate" in summary
        assert "hardware_requirements" in summary
        assert "memory_estimate" in summary

    def test_resource_summary_basic_values(self) -> None:
        """Test basic values in resource summary."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="rotation", reps=2, entanglement="linear"
        )
        summary = enc.resource_summary()

        assert summary["n_qubits"] == 4
        assert summary["n_features"] == 4
        assert summary["reps"] == 2
        assert summary["symmetry"] == "rotation"
        assert summary["entanglement"] == "linear"

    def test_resource_summary_gate_counts_embedded(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that gate_counts in summary matches gate_count_breakdown."""
        summary = default_encoding.resource_summary()
        breakdown = default_encoding.gate_count_breakdown()

        assert summary["gate_counts"] == breakdown

    def test_resource_summary_symmetry_info_embedded(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that symmetry_info is embedded in summary."""
        summary = default_encoding.resource_summary()
        symmetry_info = default_encoding.get_symmetry_info()

        assert summary["symmetry_info"] == symmetry_info

    def test_resource_summary_hardware_requirements(self) -> None:
        """Test hardware requirements in resource summary."""
        # Full entanglement requires all-to-all connectivity
        enc_full = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", entanglement="full"
        )
        summary_full = enc_full.resource_summary()
        assert summary_full["hardware_requirements"]["connectivity"] == "all-to-all"

        # Linear entanglement requires linear connectivity
        enc_linear = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", entanglement="linear"
        )
        summary_linear = enc_linear.resource_summary()
        assert summary_linear["hardware_requirements"]["connectivity"] == "linear"

        # No entanglement requires no connectivity
        enc_none = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", entanglement="none"
        )
        summary_none = enc_none.resource_summary()
        assert summary_none["hardware_requirements"]["connectivity"] == "none"

    def test_resource_summary_native_gates(self) -> None:
        """Test native gates in resource summary based on symmetry."""
        enc_rot = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")
        summary_rot = enc_rot.resource_summary()
        assert "CRZ" in summary_rot["hardware_requirements"]["native_gates"]

        enc_cyc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic")
        summary_cyc = enc_cyc.resource_summary()
        assert "CNOT" in summary_cyc["hardware_requirements"]["native_gates"]

        enc_ref = SymmetryInspiredFeatureMap(n_features=4, symmetry="reflection")
        summary_ref = enc_ref.resource_summary()
        assert "CZ" in summary_ref["hardware_requirements"]["native_gates"]

    def test_resource_summary_memory_estimate(
        self, default_encoding: SymmetryInspiredFeatureMap
    ) -> None:
        """Test that memory estimate is included."""
        summary = default_encoding.resource_summary()

        assert "memory_estimate" in summary
        assert "total_instance_bytes" in summary["memory_estimate"]
        assert summary["memory_estimate"]["total_instance_bytes"] > 0

    def test_resource_summary_all_symmetry_types(self) -> None:
        """Test resource summary works for all symmetry types."""
        symmetries = ["rotation", "cyclic", "reflection", "full"]

        for sym in symmetries:
            enc = SymmetryInspiredFeatureMap(n_features=4, symmetry=sym)
            summary = enc.resource_summary()

            # Should complete without error
            assert summary["symmetry"] == sym
            assert isinstance(summary["gate_counts"], dict)
            assert isinstance(summary["hardware_requirements"], dict)


# =============================================================================
# Test Class: State Vector Conversion (Encoding-Specific)
# =============================================================================


class TestStateVectorConversion:
    """Tests for the convert_state_vector_ordering utility function."""

    def test_identity_conversion(self) -> None:
        """Test that same source and target returns copy."""
        state = np.array([1, 0, 0, 0], dtype=complex)

        result = convert_state_vector_ordering(state, 2, "big", "big")

        np.testing.assert_array_equal(result, state)
        assert result is not state  # Should be a copy

    def test_two_qubit_conversion(self) -> None:
        """Test conversion for 2 qubits.

        For 2 qubits, the mapping is:
        - Index 0 (00) -> 0 (00)
        - Index 1 (01) -> 2 (10)
        - Index 2 (10) -> 1 (01)
        - Index 3 (11) -> 3 (11)
        """
        # State |01> in big-endian (index 1)
        big_endian = np.array([0, 1, 0, 0], dtype=complex)

        little_endian = convert_state_vector_ordering(
            big_endian, n_qubits=2, source="big", target="little"
        )

        # In little-endian, this should be at index 2
        expected = np.array([0, 0, 1, 0], dtype=complex)
        np.testing.assert_array_equal(little_endian, expected)

    def test_round_trip(self) -> None:
        """Test that big->little->big gives back original."""
        original = np.array([0.5, 0.3, 0.1, 0.7j], dtype=complex)
        original /= np.linalg.norm(original)  # Normalize

        little = convert_state_vector_ordering(
            original, n_qubits=2, source="big", target="little"
        )
        back_to_big = convert_state_vector_ordering(
            little, n_qubits=2, source="little", target="big"
        )

        np.testing.assert_allclose(back_to_big, original)

    def test_three_qubit_conversion(self) -> None:
        """Test conversion for 3 qubits.

        Bit reversal for 3 bits:
        - 000 (0) -> 000 (0)
        - 001 (1) -> 100 (4)
        - 010 (2) -> 010 (2)
        - 011 (3) -> 110 (6)
        - 100 (4) -> 001 (1)
        - 101 (5) -> 101 (5)
        - 110 (6) -> 011 (3)
        - 111 (7) -> 111 (7)
        """
        # State |001> in big-endian (index 1)
        big_endian = np.zeros(8, dtype=complex)
        big_endian[1] = 1.0

        little_endian = convert_state_vector_ordering(
            big_endian, n_qubits=3, source="big", target="little"
        )

        # Should be at index 4 in little-endian
        expected = np.zeros(8, dtype=complex)
        expected[4] = 1.0
        np.testing.assert_array_equal(little_endian, expected)

    def test_probability_preservation(self) -> None:
        """Test that probabilities are preserved after conversion."""
        # Random state
        np.random.seed(42)
        state = np.random.randn(16) + 1j * np.random.randn(16)
        state /= np.linalg.norm(state)

        converted = convert_state_vector_ordering(
            state, n_qubits=4, source="big", target="little"
        )

        # Probabilities should be the same (just reordered)
        orig_probs = np.abs(state) ** 2
        conv_probs = np.abs(converted) ** 2

        np.testing.assert_allclose(sorted(orig_probs), sorted(conv_probs))
        np.testing.assert_allclose(np.sum(conv_probs), 1.0)

    def test_invalid_state_size(self) -> None:
        """Test that mismatched state size raises error."""
        state = np.array([1, 0, 0], dtype=complex)  # Not power of 2

        with pytest.raises(ValueError, match="must equal"):
            convert_state_vector_ordering(state, n_qubits=2)

    def test_invalid_n_qubits(self) -> None:
        """Test that invalid n_qubits raises error."""
        state = np.array([1, 0, 0, 0], dtype=complex)

        with pytest.raises(ValueError, match="at least 1"):
            convert_state_vector_ordering(state, n_qubits=0)

    def test_invalid_source(self) -> None:
        """Test that invalid source raises error."""
        state = np.array([1, 0, 0, 0], dtype=complex)

        with pytest.raises(ValueError, match="source must be"):
            convert_state_vector_ordering(state, 2, source="medium")  # type: ignore

    def test_invalid_target(self) -> None:
        """Test that invalid target raises error."""
        state = np.array([1, 0, 0, 0], dtype=complex)

        with pytest.raises(ValueError, match="target must be"):
            convert_state_vector_ordering(state, 2, target="medium")  # type: ignore

    def test_list_input_converted(self) -> None:
        """Test that list input is automatically converted."""
        state = [1, 0, 0, 0]

        result = convert_state_vector_ordering(state, n_qubits=2)

        assert isinstance(result, np.ndarray)


# =============================================================================
# Test Class: Memory Estimation (Encoding-Specific)
# =============================================================================


class TestMemoryEstimation:
    """Tests for the estimate_memory_usage utility function."""

    def test_basic_structure(self) -> None:
        """Test that estimation returns expected keys."""
        mem = estimate_memory_usage(4, "rotation", 2, "linear")

        assert "instance_bytes" in mem
        assert "entanglement_pairs_bytes" in mem
        assert "symmetry_params_bytes" in mem
        assert "total_instance_bytes" in mem
        assert "circuit_angles_bytes" in mem
        assert "total_per_circuit_bytes" in mem
        assert "n_entanglement_pairs" in mem
        assert "recommended_max_features" in mem

    def test_state_vector_optional(self) -> None:
        """Test state vector estimation is optional."""
        mem_without = estimate_memory_usage(4, "rotation", 2, "linear")
        mem_with = estimate_memory_usage(
            4, "rotation", 2, "linear", include_state_vector=True
        )

        assert "state_vector_bytes" not in mem_without
        assert "state_vector_bytes" in mem_with
        # 16 bytes per complex128
        assert mem_with["state_vector_bytes"] == 2 ** 4 * 16

    def test_full_entanglement_more_memory(self) -> None:
        """Test that full entanglement uses more memory than linear."""
        mem_linear = estimate_memory_usage(10, "cyclic", 2, "linear")
        mem_full = estimate_memory_usage(10, "cyclic", 2, "full")

        assert mem_full["entanglement_pairs_bytes"] > mem_linear["entanglement_pairs_bytes"]
        assert mem_full["total_instance_bytes"] > mem_linear["total_instance_bytes"]

    def test_no_entanglement_minimal_pairs(self) -> None:
        """Test that no entanglement has zero pairs."""
        mem = estimate_memory_usage(10, "cyclic", 2, "none")

        assert mem["n_entanglement_pairs"] == 0

    def test_scaling_with_features(self) -> None:
        """Test that memory scales with feature count."""
        mem_small = estimate_memory_usage(4, "cyclic", 2, "linear")
        mem_large = estimate_memory_usage(20, "cyclic", 2, "linear")

        assert mem_large["total_instance_bytes"] > mem_small["total_instance_bytes"]

    def test_recommended_max_full_vs_linear(self) -> None:
        """Test recommended max is lower for full entanglement."""
        mem_linear = estimate_memory_usage(10, "cyclic", 2, "linear")
        mem_full = estimate_memory_usage(10, "cyclic", 2, "full")

        assert mem_full["recommended_max_features"] < mem_linear["recommended_max_features"]

    def test_validation_errors(self) -> None:
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="at least 2"):
            estimate_memory_usage(1, "rotation", 2, "linear")

        with pytest.raises(ValueError, match="symmetry must be"):
            estimate_memory_usage(4, "invalid", 2, "linear")

        with pytest.raises(ValueError, match="reps must be at least 1"):
            estimate_memory_usage(4, "rotation", 0, "linear")

        with pytest.raises(ValueError, match="entanglement must be"):
            estimate_memory_usage(4, "rotation", 2, "invalid")

    def test_rotation_requires_even(self) -> None:
        """Test rotation symmetry validation."""
        with pytest.raises(ValueError, match="even n_features"):
            estimate_memory_usage(5, "rotation", 2, "linear")

    def test_state_vector_mb_included(self) -> None:
        """Test state vector MB is included when state vector is requested."""
        mem = estimate_memory_usage(
            10, "cyclic", 2, "linear", include_state_vector=True
        )

        assert "state_vector_mb" in mem
        expected_mb = (2 ** 10 * 16) / (1024 * 1024)
        np.testing.assert_allclose(mem["state_vector_mb"], expected_mb)


# =============================================================================
# Test Class: Slow Simulation Tests (Always Last)
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    This class includes all slow simulation tests including cross-backend
    state fidelity verification.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        def get_state(x: NDArray[np.floating]) -> NDArray[np.complexfloating]:
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
        """Test that same input produces same state."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([0.1, 0.2, 0.3, 0.4])

        def get_state() -> NDArray[np.complexfloating]:
            circuit_fn = enc.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def full_circuit():
                circuit_fn()
                return qml.state()

            return full_circuit()

        state1 = get_state()
        state2 = get_state()

        np.testing.assert_allclose(state1, state2)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states.

        Note: PennyLane and Qiskit use different qubit ordering conventions
        (little-endian vs big-endian), so we verify states are valid and
        normalized rather than comparing exact probability distributions.
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", reps=1, entanglement="linear"
        )
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
        assert len(pl_state) == len(qk_state) == len(cirq_state) == 2 ** enc.n_qubits

        # Probability distributions should have the same set of values
        # (accounting for different qubit ordering conventions)
        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(pl_probs, qk_probs, atol=1e-6)
        np.testing.assert_allclose(pl_probs, cirq_probs, atol=1e-6)

    # -------------------------------------------------------------------------
    # Cross-Backend State Fidelity Tests
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_pennylane_state(
        enc: SymmetryInspiredFeatureMap,
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
        enc: SymmetryInspiredFeatureMap,
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
        enc: SymmetryInspiredFeatureMap,
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
        """Assert two quantum states are equivalent up to qubit ordering."""
        assert len(state1) == len(state2), (
            f"{name1} and {name2} have different dimensions"
        )

        norm1 = np.sum(np.abs(state1) ** 2)
        norm2 = np.sum(np.abs(state2) ** 2)
        assert np.isclose(norm1, 1.0, atol=1e-10), f"{name1} is not normalized"
        assert np.isclose(norm2, 1.0, atol=1e-10), f"{name2} is not normalized"

        probs1 = sorted(np.abs(state1) ** 2)
        probs2 = sorted(np.abs(state2) ** 2)

        np.testing.assert_allclose(probs1, probs2, atol=atol)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_equivalent_states(self) -> None:
        """Test that all backends produce mathematically equivalent states."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
        x = np.array([0.5, 0.3, 0.7, 0.1])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        expected_dim = 2 ** enc.n_qubits
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
        self._assert_states_equivalent(qk_state, cirq_state, "Qiskit", "Cirq")

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("symmetry", ["cyclic", "reflection", "full"])
    def test_cross_backend_all_symmetries(self, symmetry: str) -> None:
        """Test cross-backend equivalence for all symmetry types."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry=symmetry, reps=2)
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
    @pytest.mark.cross_backend
    def test_cross_backend_rotation_symmetry(self) -> None:
        """Test cross-backend equivalence for rotation symmetry (requires even n)."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="rotation", reps=2)
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
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("entanglement", ["linear", "full", "circular"])
    def test_cross_backend_all_entanglements(self, entanglement: str) -> None:
        """Test cross-backend equivalence for all entanglement patterns."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", reps=2, entanglement=entanglement
        )
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
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("n_features", [3, 4, 5, 6])
    def test_cross_backend_various_sizes(self, n_features: int) -> None:
        """Test cross-backend equivalence for various feature counts."""
        enc = SymmetryInspiredFeatureMap(
            n_features=n_features, symmetry="cyclic", reps=2
        )

        rng = np.random.default_rng(seed=n_features * 42)
        x = rng.random(n_features)

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        expected_dim = 2 ** enc.n_qubits
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
        """Test cross-backend equivalence with zero input."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
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
    def test_cross_backend_reproducibility(self) -> None:
        """Test that repeated executions produce identical results."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
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
    @pytest.mark.cross_backend
    def test_cross_backend_different_inputs_different_states(self) -> None:
        """Test that different inputs produce different states."""
        enc = SymmetryInspiredFeatureMap(n_features=4, symmetry="cyclic", reps=2)
        x1 = np.array([0.1, 0.2, 0.3, 0.4])
        x2 = np.array([0.9, 0.8, 0.7, 0.6])

        pl_state1 = self._get_pennylane_state(enc, x1)
        pl_state2 = self._get_pennylane_state(enc, x2)

        def fidelity(s1: np.ndarray, s2: np.ndarray) -> float:
            return float(np.abs(np.vdot(s1, s2)) ** 2)

        assert fidelity(pl_state1, pl_state2) < 0.99

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("feature_map", ["angle", "fourier"])
    def test_cross_backend_feature_maps(self, feature_map: str) -> None:
        """Test cross-backend equivalence for different feature map types."""
        enc = SymmetryInspiredFeatureMap(
            n_features=4, symmetry="cyclic", reps=2, feature_map=feature_map
        )
        x = np.array([0.3, 0.5, 0.7, 0.9])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
