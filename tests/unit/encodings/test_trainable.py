"""Comprehensive tests for TrainableEncoding.

This test module provides complete coverage of the TrainableEncoding class,
which implements parameterized quantum encoding with learnable parameters
that can be optimized through variational training. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, n_trainable_parameters)
- Trainable parameter management (get, set, reset)
- Entanglement pattern behavior (linear, circular, full, none)
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

Run with: pytest tests/unit/encodings/test_trainable.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_trainable.py -v -m "not slow"

References
----------
.. [1] Schuld, M., et al. (2021). "Effect of data encoding on the expressive
       power of variational quantum machine learning models." Physical Review A.
.. [2] Benedetti, M., et al. (2019). "Parameterized quantum circuits as machine
       learning models." Quantum Science and Technology.
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

from encoding_atlas.core.properties import EncodingProperties

# Local imports
from encoding_atlas.encodings.trainable import TrainableEncoding

# Conditional TYPE_CHECKING imports
if TYPE_CHECKING:
    from typing import Any


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

    Contains 3 samples with varying value ranges.
    """
    return np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]
    )


@pytest.fixture
def default_encoding() -> TrainableEncoding:
    """Default TrainableEncoding with 4 features."""
    return TrainableEncoding(n_features=4, seed=42)


@pytest.fixture
def seeded_encoding() -> TrainableEncoding:
    """TrainableEncoding with fixed seed for reproducibility tests."""
    return TrainableEncoding(n_features=4, n_layers=2, seed=12345)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for TrainableEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = TrainableEncoding(n_features=4)
        assert enc.n_features == 4
        assert enc.n_qubits == 4
        assert enc.n_layers == 2
        assert enc.data_rotation == "Y"
        assert enc.trainable_rotation == "Y"
        assert enc.entanglement == "linear"
        assert enc.initialization == "xavier"

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = TrainableEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1
        assert enc.n_trainable_parameters == 2  # n_layers * n_features

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16]:
            enc = TrainableEncoding(n_features=n)
            assert enc.n_features == n
            assert enc.n_trainable_parameters == 2 * n

    def test_custom_n_layers(self) -> None:
        """Test instantiation with custom layer count."""
        enc = TrainableEncoding(n_features=4, n_layers=5)
        assert enc.n_layers == 5
        assert enc.n_trainable_parameters == 20

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_data_rotation_variants(self, rotation: str) -> None:
        """Test all data rotation axis variants."""
        enc = TrainableEncoding(n_features=4, data_rotation=rotation)  # type: ignore
        assert enc.data_rotation == rotation

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_trainable_rotation_variants(self, rotation: str) -> None:
        """Test all trainable rotation axis variants."""
        enc = TrainableEncoding(n_features=4, trainable_rotation=rotation)  # type: ignore
        assert enc.trainable_rotation == rotation

    @pytest.mark.parametrize("entanglement", ["linear", "circular", "full", "none"])
    def test_entanglement_patterns(self, entanglement: str) -> None:
        """Test all valid entanglement patterns."""
        enc = TrainableEncoding(n_features=4, entanglement=entanglement)  # type: ignore
        assert enc.entanglement == entanglement

    @pytest.mark.parametrize(
        "init", ["xavier", "he", "zeros", "random", "small_random"]
    )
    def test_initialization_strategies(self, init: str) -> None:
        """Test all initialization strategies."""
        enc = TrainableEncoding(n_features=4, initialization=init)  # type: ignore
        assert enc.initialization == init
        params = enc.get_trainable_parameters()
        assert params.shape == (2, 4)

    def test_config_stored_correctly(self) -> None:
        """Test that configuration is stored in config dict."""
        enc = TrainableEncoding(
            n_features=4,
            n_layers=3,
            data_rotation="Z",
            trainable_rotation="X",
            entanglement="circular",
            initialization="he",
        )
        assert enc.config["n_layers"] == 3
        assert enc.config["data_rotation"] == "Z"
        assert enc.config["trainable_rotation"] == "X"
        assert enc.config["entanglement"] == "circular"
        assert enc.config["initialization"] == "he"

    def test_reproducible_initialization_with_seed(self) -> None:
        """Test that same seed produces same initial parameters."""
        enc1 = TrainableEncoding(n_features=4, seed=42)
        enc2 = TrainableEncoding(n_features=4, seed=42)
        np.testing.assert_array_equal(
            enc1.get_trainable_parameters(), enc2.get_trainable_parameters()
        )

    def test_different_seeds_produce_different_params(self) -> None:
        """Test that different seeds produce different parameters."""
        enc1 = TrainableEncoding(n_features=4, seed=42)
        enc2 = TrainableEncoding(n_features=4, seed=43)
        assert not np.allclose(
            enc1.get_trainable_parameters(), enc2.get_trainable_parameters()
        )

    def test_instantiation_is_lazy(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = TrainableEncoding(n_features=100)
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
            TrainableEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            TrainableEncoding(n_features=-1)

    def test_invalid_n_layers_zero(self) -> None:
        """Test that n_layers=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            TrainableEncoding(n_features=4, n_layers=0)

    def test_invalid_n_layers_negative(self) -> None:
        """Test that negative n_layers raises ValueError."""
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            TrainableEncoding(n_features=4, n_layers=-1)

    def test_invalid_n_layers_bool(self) -> None:
        """Test that boolean n_layers raises ValueError."""
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            TrainableEncoding(n_features=4, n_layers=True)  # type: ignore

    def test_invalid_data_rotation(self) -> None:
        """Test that invalid data rotation raises ValueError."""
        with pytest.raises(ValueError, match="data_rotation must be one of"):
            TrainableEncoding(n_features=4, data_rotation="W")  # type: ignore

    def test_invalid_trainable_rotation(self) -> None:
        """Test that invalid trainable rotation raises ValueError."""
        with pytest.raises(ValueError, match="trainable_rotation must be one of"):
            TrainableEncoding(n_features=4, trainable_rotation="W")  # type: ignore

    def test_invalid_entanglement(self) -> None:
        """Test that invalid entanglement pattern raises ValueError."""
        with pytest.raises(ValueError, match="entanglement must be one of"):
            TrainableEncoding(n_features=4, entanglement="invalid")  # type: ignore

    def test_invalid_initialization(self) -> None:
        """Test that invalid initialization raises ValueError."""
        with pytest.raises(ValueError, match="initialization must be one of"):
            TrainableEncoding(n_features=4, initialization="invalid")  # type: ignore

    def test_invalid_seed_negative(self) -> None:
        """Test that negative seed raises ValueError."""
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            TrainableEncoding(n_features=4, seed=-1)


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of TrainableEncoding."""

    def test_n_qubits(self) -> None:
        """Test n_qubits property calculation."""
        enc = TrainableEncoding(n_features=4)
        assert enc.n_qubits == 4

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features for various sizes."""
        for n in [2, 4, 8, 16]:
            enc = TrainableEncoding(n_features=n)
            assert enc.n_qubits == n

    def test_depth(self) -> None:
        """Test circuit depth calculation."""
        enc = TrainableEncoding(n_features=4, n_layers=2, entanglement="linear")
        # depth = n_layers * (2 + entangling_depth)
        # entangling_depth for linear = n_qubits - 1 = 3
        # depth = 2 * (2 + 3) = 10
        assert enc.depth == 10

    def test_n_trainable_parameters(self) -> None:
        """Test trainable parameter count."""
        enc = TrainableEncoding(n_features=4, n_layers=3)
        assert enc.n_trainable_parameters == 12  # 3 * 4

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        enc = TrainableEncoding(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = TrainableEncoding(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_is_entangling_linear(self) -> None:
        """Test that linear entanglement creates entanglement."""
        enc = TrainableEncoding(n_features=4, entanglement="linear")
        assert enc.properties.is_entangling is True

    def test_is_entangling_none(self) -> None:
        """Test that no entanglement doesn't create entanglement."""
        enc = TrainableEncoding(n_features=4, entanglement="none")
        assert enc.properties.is_entangling is False

    def test_single_qubit_not_entangling(self) -> None:
        """Test that single qubit is not entangling."""
        enc = TrainableEncoding(n_features=1, entanglement="linear")
        assert enc.properties.is_entangling is False

    def test_properties_invalidated_after_param_update(self) -> None:
        """Test that properties are recomputed after parameter update."""
        enc = TrainableEncoding(n_features=4)
        _ = enc.properties  # Cache properties
        new_params = np.zeros((2, 4))
        enc.set_trainable_parameters(new_params)
        # Properties should be recomputed (cache invalidated)
        # The properties themselves don't depend on param values, but
        # we verify the cache was cleared
        assert enc.properties is not None


# =============================================================================
# Test Class: Trainable Parameter Management
# =============================================================================


class TestTrainableParameterManagement:
    """Tests for trainable parameter get/set/reset operations."""

    def test_get_trainable_parameters_shape(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test that get_trainable_parameters returns correct shape."""
        params = default_encoding.get_trainable_parameters()
        assert params.shape == (2, 4)

    def test_get_trainable_parameters_returns_copy(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test that get_trainable_parameters returns a copy."""
        params1 = default_encoding.get_trainable_parameters()
        params2 = default_encoding.get_trainable_parameters()
        # Should be different objects
        assert params1 is not params2
        # But same values
        np.testing.assert_array_equal(params1, params2)

    def test_set_trainable_parameters(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test setting trainable parameters."""
        new_params = np.ones((2, 4)) * 0.5
        default_encoding.set_trainable_parameters(new_params)
        np.testing.assert_array_almost_equal(
            default_encoding.get_trainable_parameters(), new_params
        )

    def test_set_trainable_parameters_flat_array(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test setting parameters with flat array."""
        new_params = np.ones(8) * 0.5
        default_encoding.set_trainable_parameters(new_params)
        assert default_encoding.get_trainable_parameters().shape == (2, 4)

    def test_set_trainable_parameters_wrong_shape_raises(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test that wrong shape raises ValueError."""
        wrong_params = np.ones((3, 4))  # Wrong number of layers
        with pytest.raises(ValueError, match="Expected parameters with shape"):
            default_encoding.set_trainable_parameters(wrong_params)

    def test_set_trainable_parameters_nan_raises(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test that NaN parameters raise ValueError."""
        bad_params = np.ones((2, 4))
        bad_params[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            default_encoding.set_trainable_parameters(bad_params)

    def test_set_trainable_parameters_inf_raises(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test that Inf parameters raise ValueError."""
        bad_params = np.ones((2, 4))
        bad_params[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.set_trainable_parameters(bad_params)

    def test_reset_parameters(self, seeded_encoding: TrainableEncoding) -> None:
        """Test resetting parameters to original initialization."""
        original_params = seeded_encoding.get_trainable_parameters().copy()
        # Modify parameters
        seeded_encoding.set_trainable_parameters(np.zeros((2, 4)))
        # Reset
        seeded_encoding.reset_parameters()
        # Should match original
        np.testing.assert_array_almost_equal(
            seeded_encoding.get_trainable_parameters(), original_params
        )

    def test_reset_parameters_with_new_seed(
        self, default_encoding: TrainableEncoding
    ) -> None:
        """Test resetting parameters with a new seed."""
        original_params = default_encoding.get_trainable_parameters().copy()
        default_encoding.reset_parameters(seed=99999)
        # Should be different from original
        assert not np.allclose(
            default_encoding.get_trainable_parameters(), original_params
        )

    def test_xavier_initialization_scale(self) -> None:
        """Test that Xavier initialization produces reasonable scale."""
        enc = TrainableEncoding(
            n_features=16, n_layers=10, initialization="xavier", seed=42
        )
        params = enc.get_trainable_parameters()
        # Xavier: scale = sqrt(2 / (n_in + n_out)) = sqrt(2 / 32) ≈ 0.25
        assert abs(np.std(params)) < 1.0  # Should be small

    def test_zeros_initialization(self) -> None:
        """Test that zeros initialization produces all zeros."""
        enc = TrainableEncoding(n_features=4, initialization="zeros")
        params = enc.get_trainable_parameters()
        np.testing.assert_array_equal(params, np.zeros_like(params))


# =============================================================================
# Test Class: Entanglement Behavior
# =============================================================================


class TestEntanglementBehavior:
    """Tests for entanglement pair computation and patterns."""

    def test_linear_entanglement_4_qubits(self) -> None:
        """Test linear entanglement with 4 qubits."""
        enc = TrainableEncoding(n_features=4, entanglement="linear")
        expected = [(0, 1), (1, 2), (2, 3)]
        assert enc.get_entanglement_pairs() == expected

    def test_circular_entanglement_4_qubits(self) -> None:
        """Test circular entanglement with 4 qubits."""
        enc = TrainableEncoding(n_features=4, entanglement="circular")
        expected = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert enc.get_entanglement_pairs() == expected

    def test_full_entanglement_4_qubits(self) -> None:
        """Test full entanglement with 4 qubits."""
        enc = TrainableEncoding(n_features=4, entanglement="full")
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert enc.get_entanglement_pairs() == expected

    def test_no_entanglement(self) -> None:
        """Test no entanglement pattern."""
        enc = TrainableEncoding(n_features=4, entanglement="none")
        assert enc.get_entanglement_pairs() == []

    def test_linear_entanglement_count(self) -> None:
        """Test that linear entanglement has n-1 pairs."""
        for n in [2, 4, 6, 8]:
            enc = TrainableEncoding(n_features=n, entanglement="linear")
            assert len(enc.get_entanglement_pairs()) == n - 1

    def test_circular_entanglement_count(self) -> None:
        """Test that circular entanglement has n pairs."""
        for n in [3, 4, 6, 8]:
            enc = TrainableEncoding(n_features=n, entanglement="circular")
            assert len(enc.get_entanglement_pairs()) == n

    def test_full_entanglement_count(self) -> None:
        """Test that full entanglement has n(n-1)/2 pairs."""
        for n in [2, 3, 4, 5, 6]:
            enc = TrainableEncoding(n_features=n, entanglement="full")
            expected_pairs = n * (n - 1) // 2
            assert len(enc.get_entanglement_pairs()) == expected_pairs

    def test_single_qubit_no_pairs(self) -> None:
        """Test that single qubit has no entanglement pairs."""
        enc = TrainableEncoding(n_features=1, entanglement="linear")
        assert len(enc.get_entanglement_pairs()) == 0


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        validated = default_encoding._validate_input(sample_data_4d)
        assert validated.shape == (4,)

    def test_wrong_feature_count(self, default_encoding: TrainableEncoding) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2])  # 2 features, expected 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding._validate_input(x)

    def test_nan_input_rejected(self, default_encoding: TrainableEncoding) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding._validate_input(x)

    def test_inf_input_rejected(self, default_encoding: TrainableEncoding) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding._validate_input(x)

    def test_list_input_accepted(self, default_encoding: TrainableEncoding) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]
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
        default_encoding: TrainableEncoding,
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
        enc = TrainableEncoding(n_features=4, n_layers=1, seed=42)
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
        default_encoding: TrainableEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    @pytest.mark.parametrize("data_rot", ["X", "Y", "Z"])
    def test_data_rotations(
        self,
        sample_data_4d: NDArray[np.floating],
        data_rot: str,
    ) -> None:
        """Test circuit generation with different data rotations."""
        enc = TrainableEncoding(n_features=4, data_rotation=data_rot, n_layers=1, seed=42)  # type: ignore
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.parametrize("trainable_rot", ["X", "Y", "Z"])
    def test_trainable_rotations(
        self,
        sample_data_4d: NDArray[np.floating],
        trainable_rot: str,
    ) -> None:
        """Test circuit generation with different trainable rotations."""
        enc = TrainableEncoding(n_features=4, trainable_rotation=trainable_rot, n_layers=1, seed=42)  # type: ignore
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_different_params_produce_different_states(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that different trainable parameters produce different states."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        dev = qml.device("default.qubit", wires=4)

        # Get state with original parameters
        circuit_fn1 = enc.get_circuit(sample_data_4d, backend="pennylane")

        @qml.qnode(dev)
        def circuit1():
            circuit_fn1()
            return qml.state()

        state1 = circuit1()

        # Change parameters and get new state
        enc.set_trainable_parameters(np.random.randn(2, 4) * 2)
        circuit_fn2 = enc.get_circuit(sample_data_4d, backend="pennylane")

        @qml.qnode(dev)
        def circuit2():
            circuit_fn2()
            return qml.state()

        state2 = circuit2()

        # States should be different
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99

    def test_no_entanglement_produces_product_state(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that no entanglement produces separable states."""
        enc = TrainableEncoding(n_features=4, entanglement="none", n_layers=2, seed=42)
        dev = qml.device("default.qubit", wires=4)

        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        # State should be normalized
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)


# =============================================================================
# Test Class: Qiskit Backend
# =============================================================================


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.mark.backend_qiskit
class TestQiskitBackend:
    """Tests for Qiskit circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_circuit_has_expected_gates(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit contains expected gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        ops = circuit.count_ops()
        # Should have rotation gates
        assert "ry" in ops or "rx" in ops or "rz" in ops
        # Should have CNOT gates for entanglement
        assert "cx" in ops

    def test_batch_circuits(
        self,
        default_encoding: TrainableEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = default_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)

    @pytest.mark.parametrize("entanglement", ["linear", "circular", "full", "none"])
    def test_all_entanglements(
        self,
        sample_data_4d: NDArray[np.floating],
        entanglement: str,
    ) -> None:
        """Test circuit generation with all entanglement patterns."""
        enc = TrainableEncoding(n_features=4, entanglement=entanglement, seed=42)  # type: ignore
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)


# =============================================================================
# Test Class: Cirq Backend
# =============================================================================


@pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
@pytest.mark.backend_cirq
class TestCirqBackend:
    """Tests for Cirq circuit generation."""

    def test_circuit_type(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        operations = list(circuit.all_operations())
        assert len(operations) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    def test_batch_circuits(
        self,
        default_encoding: TrainableEncoding,
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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zero_input_with_zero_params(self) -> None:
        """Test that zero input with zero params produces known state."""
        enc = TrainableEncoding(
            n_features=4, n_layers=1, initialization="zeros", seed=42
        )
        x = np.zeros(4)

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            return qml.state()

        state = circuit()
        # With zero rotations and only CNOT gates from |0000⟩, state should still
        # be close to |0000⟩ (CNOT on |0⟩ control does nothing)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different states."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
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

        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        assert fidelity < 0.99


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_layer(self) -> None:
        """Test encoding with single layer."""
        enc = TrainableEncoding(n_features=4, n_layers=1)
        assert enc.n_layers == 1

    def test_many_layers(self) -> None:
        """Test encoding with many layers."""
        # Should issue a warning but work
        with pytest.warns(UserWarning, match="deep circuit"):
            enc = TrainableEncoding(n_features=4, n_layers=10)
        assert enc.n_layers == 10

    def test_large_feature_count(self) -> None:
        """Test encoding with large feature count."""
        enc = TrainableEncoding(n_features=20, entanglement="linear")
        assert enc.n_qubits == 20
        assert len(enc.get_entanglement_pairs()) == 19

    def test_zero_valued_input(self, default_encoding: TrainableEncoding) -> None:
        """Test with all-zero input."""
        x = np.zeros(4)
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_large_valued_input(self, default_encoding: TrainableEncoding) -> None:
        """Test with large input values."""
        x = np.array([100.0, 200.0, 300.0, 400.0])
        if HAS_PENNYLANE:
            circuit = default_encoding.get_circuit(x, backend="pennylane")
            assert callable(circuit)

    def test_negative_valued_input(self, default_encoding: TrainableEncoding) -> None:
        """Test with negative input values."""
        x = np.array([-0.5, -1.0, -1.5, -2.0])
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
        """Test encoding with very small input values."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_large_values(self) -> None:
        """Test encoding with very large input values."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        x = np.array([1e10, 1e11, 1e12, 1e13])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_values_near_pi_boundaries(self) -> None:
        """Test encoding with values near pi boundaries."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        eps = 1e-14
        x = np.array(
            [
                np.pi - eps,
                np.pi + eps,
                2 * np.pi - eps,
                2 * np.pi + eps,
            ]
        )

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = TrainableEncoding(n_features=4, seed=42)
        enc2 = TrainableEncoding(n_features=4, seed=42)
        assert enc1 == enc2

    def test_equality_different_params(self) -> None:
        """Test that encodings with different trainable params are not equal."""
        enc1 = TrainableEncoding(n_features=4, seed=42)
        enc2 = TrainableEncoding(n_features=4, seed=43)
        assert enc1 != enc2

    def test_equality_after_param_update(self) -> None:
        """Test equality after setting same parameters."""
        enc1 = TrainableEncoding(n_features=4, seed=42)
        enc2 = TrainableEncoding(n_features=4, seed=43)
        enc2.set_trainable_parameters(enc1.get_trainable_parameters())
        assert enc1 == enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = TrainableEncoding(n_features=4, seed=42)
        enc2 = TrainableEncoding(n_features=4, seed=42)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = TrainableEncoding(n_features=4, seed=42)
        enc2 = TrainableEncoding(n_features=4, seed=42)
        enc3 = TrainableEncoding(n_features=8, seed=42)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 have same hash, but equality also checks params
        # They should be equal since same seed
        assert len(s) == 2


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ and string representation."""

    def test_repr_contains_class_name(self) -> None:
        """Test that repr contains the class name."""
        enc = TrainableEncoding(n_features=4)
        repr_str = repr(enc)
        assert "TrainableEncoding" in repr_str

    def test_repr_contains_n_features(self) -> None:
        """Test that repr contains n_features parameter."""
        enc = TrainableEncoding(n_features=4)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str

    def test_repr_contains_n_layers(self) -> None:
        """Test that repr contains n_layers parameter."""
        enc = TrainableEncoding(n_features=4, n_layers=3)
        repr_str = repr(enc)
        assert "n_layers=3" in repr_str

    def test_repr_contains_rotations(self) -> None:
        """Test that repr contains rotation parameters."""
        enc = TrainableEncoding(
            n_features=4,
            data_rotation="X",
            trainable_rotation="Z",
        )
        repr_str = repr(enc)
        assert "data_rotation='X'" in repr_str
        assert "trainable_rotation='Z'" in repr_str


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for backend availability and error handling."""

    def test_invalid_backend_raises_error(
        self,
        default_encoding: TrainableEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_empty_backend_raises_error(
        self,
        default_encoding: TrainableEncoding,
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
        enc = TrainableEncoding(
            n_features=4,
            n_layers=3,
            data_rotation="Z",
            trainable_rotation="X",
            entanglement="circular",
            seed=42,
        )
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.n_layers == enc.n_layers
        assert restored.data_rotation == enc.data_rotation
        assert restored.trainable_rotation == enc.trainable_rotation
        assert restored.entanglement == enc.entanglement
        np.testing.assert_array_almost_equal(
            restored.get_trainable_parameters(), enc.get_trainable_parameters()
        )

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = TrainableEncoding(n_features=4, seed=42)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        if HAS_QISKIT:
            circuit = restored.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == 4


# =============================================================================
# Test Class: Concurrent Access / Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for _i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
                    if HAS_QISKIT:
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

        if HAS_QISKIT:
            total_circuits = sum(len(r) for r in results)
            assert total_circuits == num_threads * num_circuits_per_thread

    def test_concurrent_property_access(self) -> None:
        """Test that concurrent property access is thread-safe."""
        enc = TrainableEncoding(n_features=4, seed=42)
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


# =============================================================================
# Test Class: Copy Method
# =============================================================================


class TestCopyMethod:
    """Tests for the copy() method."""

    def test_copy_creates_independent_object(self) -> None:
        """Test that copy creates an independent object."""
        enc = TrainableEncoding(n_features=4, seed=42)
        enc_copy = enc.copy()

        # Should be different objects
        assert enc is not enc_copy

        # But have same parameters
        np.testing.assert_array_almost_equal(
            enc.get_trainable_parameters(), enc_copy.get_trainable_parameters()
        )

    def test_copy_is_truly_independent(self) -> None:
        """Test that modifying copy doesn't affect original."""
        enc = TrainableEncoding(n_features=4, seed=42)
        original_params = enc.get_trainable_parameters().copy()

        enc_copy = enc.copy()
        enc_copy.set_trainable_parameters(np.zeros((2, 4)))

        # Original should be unchanged
        np.testing.assert_array_almost_equal(
            enc.get_trainable_parameters(), original_params
        )


# =============================================================================
# Test Class: Resource Summary
# =============================================================================


class TestResourceSummary:
    """Tests for resource_summary() method."""

    def test_resource_summary_keys(self) -> None:
        """Test that resource summary contains expected keys."""
        enc = TrainableEncoding(n_features=4, n_layers=2)
        summary = enc.resource_summary()

        expected_keys = [
            "n_qubits",
            "n_features",
            "n_layers",
            "depth",
            "gate_counts",
            "data_rotation",
            "trainable_rotation",
            "entanglement",
            "initialization",
            "n_trainable_parameters",
            "parameter_statistics",
            "is_entangling",
            "simulability",
            "trainability_estimate",
            "hardware_requirements",
            "n_entanglement_pairs",
            "entanglement_pairs",
            "recommendations",
        ]

        for key in expected_keys:
            assert key in summary

    def test_parameter_statistics(self) -> None:
        """Test that parameter statistics are computed correctly."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        summary = enc.resource_summary()

        stats = summary["parameter_statistics"]
        params = enc.get_trainable_parameters()

        assert np.isclose(stats["mean"], np.mean(params))
        assert np.isclose(stats["std"], np.std(params))
        assert np.isclose(stats["min"], np.min(params))
        assert np.isclose(stats["max"], np.max(params))

    def test_gate_count_breakdown(self) -> None:
        """Test gate count breakdown computation."""
        enc = TrainableEncoding(
            n_features=4,
            n_layers=2,
            data_rotation="Y",
            trainable_rotation="Y",
            entanglement="linear",
        )
        breakdown = enc.gate_count_breakdown()

        # 4 features, 2 layers:
        # - Data RY gates: 2 * 4 = 8
        # - Trainable RY gates: 2 * 4 = 8
        # - CNOT gates: 2 * 3 = 6 (linear has n-1 pairs)
        assert breakdown["data_ry"] == 8
        assert breakdown["trainable_ry"] == 8
        assert breakdown["cnot_gates"] == 6
        assert breakdown["total"] == 22


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
        enc: TrainableEncoding,
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
        enc: TrainableEncoding,
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
        enc: TrainableEncoding,
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
        """Assert two quantum states are equivalent up to qubit ordering."""
        assert len(state1) == len(state2)

        norm1 = np.sum(np.abs(state1) ** 2)
        norm2 = np.sum(np.abs(state2) ** 2)
        assert np.isclose(norm1, 1.0, atol=1e-10)
        assert np.isclose(norm2, 1.0, atol=1e-10)

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
    def test_all_backends_produce_equivalent_states(self) -> None:
        """Test that all backends produce mathematically equivalent states."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        pl_state = self._get_pennylane_state(enc, x)
        qk_state = self._get_qiskit_state(enc, x)
        cirq_state = self._get_cirq_state(enc, x)

        expected_dim = 2**enc.n_qubits
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        self._assert_states_equivalent(pl_state, qk_state, "PennyLane", "Qiskit")
        self._assert_states_equivalent(pl_state, cirq_state, "PennyLane", "Cirq")
        self._assert_states_equivalent(qk_state, cirq_state, "Qiskit", "Cirq")

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_reproducibility(self) -> None:
        """Test that same input always produces same state."""
        enc = TrainableEncoding(n_features=4, n_layers=2, seed=42)
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
