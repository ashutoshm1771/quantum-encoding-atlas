"""Comprehensive tests for Equivariant Feature Map encodings.

This test module provides complete coverage of the equivariant encoding classes:
- SO2EquivariantFeatureMap (2D rotation equivariance)
- CyclicEquivariantFeatureMap (cyclic shift equivariance)
- SwapEquivariantFeatureMap (pair swap equivariance)

It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, gate_count)
- Mathematical equivariance verification
- Group action and unitary representation correctness
- Circuit generation for all backends (PennyLane, Qiskit, Cirq)
- Edge cases and boundary conditions
- Numerical stability tests
- Equality and hashing
- String representation
- Backend error handling
- Serialization (pickle roundtrip)
- Concurrent access / thread safety
- Slow simulation tests (cross-backend state fidelity)

Run with: pytest tests/unit/encodings/test_equivariant_feature_map.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_equivariant_feature_map.py -v -m "not slow"

References
----------
.. [1] Larocca, M., et al. (2022). "Group-Invariant Quantum Machine Learning."
       PRX Quantum, 3, 030341.
.. [2] Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational Quantum
       Machine Learning." PRX Quantum, 4, 010328.
.. [3] Schatzki, L., et al. (2022). "Theoretical Guarantees for
       Permutation-Equivariant Quantum Neural Networks." npj Quantum
       Information, 8, 74.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas.encodings.equivariant_feature_map import (
    CyclicEquivariantFeatureMap,
    SO2EquivariantFeatureMap,
    SwapEquivariantFeatureMap,
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
def sample_data_2d() -> NDArray[np.floating]:
    """2-dimensional sample data for SO2 testing."""
    return np.array([0.5, 0.3])


@pytest.fixture
def batch_data_2d() -> NDArray[np.floating]:
    """Batch of 2D samples for SO2 testing.

    Contains 4 samples:
    - [0.5, 0.3] (typical values)
    - [1.0, 0.0] (on x-axis)
    - [0.0, 1.0] (on y-axis)
    - [-0.5, 0.5] (negative x)
    """
    return np.array([
        [0.5, 0.3],
        [1.0, 0.0],
        [0.0, 1.0],
        [-0.5, 0.5],
    ])


@pytest.fixture
def sample_data_4d() -> NDArray[np.floating]:
    """4-dimensional sample data for cyclic/swap testing."""
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def batch_data_4d() -> NDArray[np.floating]:
    """Batch of 4D samples for cyclic/swap testing.

    Contains 3 samples:
    - [0.1, 0.2, 0.3, 0.4] (typical values)
    - [0.5, 0.6, 0.7, 0.8] (larger values)
    - [0.0, 0.0, 0.0, 0.0] (edge case: zeros)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.0, 0.0, 0.0, 0.0],
    ])


@pytest.fixture
def so2_encoding() -> SO2EquivariantFeatureMap:
    """Default SO(2) equivariant encoding."""
    return SO2EquivariantFeatureMap(max_angular_momentum=1)


@pytest.fixture
def cyclic_encoding() -> CyclicEquivariantFeatureMap:
    """Default cyclic equivariant encoding with 4 features."""
    return CyclicEquivariantFeatureMap(n_features=4)


@pytest.fixture
def swap_encoding() -> SwapEquivariantFeatureMap:
    """Default swap equivariant encoding with 4 features."""
    return SwapEquivariantFeatureMap(n_features=4)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for equivariant encoding instantiation and parameter handling."""

    # --- SO2EquivariantFeatureMap ---

    def test_so2_default_parameters(self) -> None:
        """Test creating SO2 encoding with default parameters."""
        enc = SO2EquivariantFeatureMap()
        assert enc.n_features == 2
        assert enc.max_angular_momentum == 1
        assert enc.radial_function == "gaussian"
        assert enc.radial_sigma == 1.0

    def test_so2_custom_max_angular_momentum(self) -> None:
        """Test SO2 with custom max angular momentum."""
        enc = SO2EquivariantFeatureMap(max_angular_momentum=2)
        assert enc.max_angular_momentum == 2
        # max_m=2 => 5 states (-2, -1, 0, 1, 2)
        assert len(enc.angular_momenta) == 5

    def test_so2_uniform_radial_function(self) -> None:
        """Test SO2 with uniform radial function."""
        enc = SO2EquivariantFeatureMap(radial_function="uniform")
        assert enc.radial_function == "uniform"

    def test_so2_custom_radial_sigma(self) -> None:
        """Test SO2 with custom radial sigma."""
        enc = SO2EquivariantFeatureMap(radial_sigma=2.0)
        assert enc.radial_sigma == 2.0

    # --- CyclicEquivariantFeatureMap ---

    def test_cyclic_default_parameters(self) -> None:
        """Test creating cyclic encoding with default parameters."""
        enc = CyclicEquivariantFeatureMap(n_features=4)
        assert enc.n_features == 4
        assert enc.reps == 2
        assert enc.coupling_strength == np.pi / 4

    def test_cyclic_custom_reps(self) -> None:
        """Test cyclic with custom repetitions."""
        enc = CyclicEquivariantFeatureMap(n_features=4, reps=3)
        assert enc.reps == 3

    def test_cyclic_custom_coupling_strength(self) -> None:
        """Test cyclic with custom coupling strength."""
        enc = CyclicEquivariantFeatureMap(n_features=4, coupling_strength=np.pi / 2)
        assert enc.coupling_strength == np.pi / 2

    def test_cyclic_various_feature_counts(self) -> None:
        """Test cyclic instantiation with various feature counts."""
        for n in [2, 3, 4, 8, 16]:
            enc = CyclicEquivariantFeatureMap(n_features=n)
            assert enc.n_features == n

    # --- SwapEquivariantFeatureMap ---

    def test_swap_default_parameters(self) -> None:
        """Test creating swap encoding with default parameters."""
        enc = SwapEquivariantFeatureMap(n_features=4)
        assert enc.n_features == 4
        # n_pairs = n_features // 2
        assert enc.n_pairs == 2
        assert enc.reps == 2

    def test_swap_custom_reps(self) -> None:
        """Test swap with custom repetitions."""
        enc = SwapEquivariantFeatureMap(n_features=4, reps=3)
        assert enc.reps == 3

    def test_swap_various_feature_counts(self) -> None:
        """Test swap instantiation with various even feature counts."""
        for n in [2, 4, 6, 8]:
            enc = SwapEquivariantFeatureMap(n_features=n)
            assert enc.n_features == n
            assert enc.n_pairs == n // 2

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        # Large angular momentum to ensure this is fast (no pre-computation)
        enc = SO2EquivariantFeatureMap(max_angular_momentum=10)
        assert enc.max_angular_momentum == 10


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    # --- SO2EquivariantFeatureMap ---

    def test_so2_invalid_max_angular_momentum_negative(self) -> None:
        """Test that negative max_angular_momentum raises ValueError."""
        with pytest.raises(ValueError, match="max_angular_momentum must be"):
            SO2EquivariantFeatureMap(max_angular_momentum=-1)

    def test_so2_invalid_radial_function(self) -> None:
        """Test that invalid radial function raises ValueError."""
        with pytest.raises(ValueError, match="radial_function must be one of"):
            SO2EquivariantFeatureMap(radial_function="invalid")

    def test_so2_invalid_radial_sigma_zero(self) -> None:
        """Test that zero radial_sigma raises ValueError."""
        with pytest.raises(ValueError, match="radial_sigma must be positive"):
            SO2EquivariantFeatureMap(radial_sigma=0.0)

    def test_so2_invalid_radial_sigma_negative(self) -> None:
        """Test that negative radial_sigma raises ValueError."""
        with pytest.raises(ValueError, match="radial_sigma must be positive"):
            SO2EquivariantFeatureMap(radial_sigma=-1.0)

    def test_so2_invalid_radial_sigma_inf(self) -> None:
        """Test that infinite radial_sigma raises ValueError."""
        with pytest.raises(ValueError, match="radial_sigma must be finite"):
            SO2EquivariantFeatureMap(radial_sigma=np.inf)

    def test_so2_invalid_radial_sigma_nan(self) -> None:
        """Test that NaN radial_sigma raises ValueError."""
        with pytest.raises(ValueError, match="radial_sigma must be finite"):
            SO2EquivariantFeatureMap(radial_sigma=np.nan)

    def test_so2_invalid_radial_sigma_type(self) -> None:
        """Test that non-numeric radial_sigma raises TypeError."""
        with pytest.raises(TypeError, match="radial_sigma must be"):
            SO2EquivariantFeatureMap(radial_sigma="1.0")  # type: ignore[arg-type]

    def test_so2_invalid_n_features_not_2(self) -> None:
        """Test that n_features != 2 raises ValueError for SO(2) encoding.

        SO(2) rotation equivariance is only defined for 2D Cartesian coordinates,
        so n_features must be exactly 2.
        """
        with pytest.raises(ValueError, match="requires n_features=2"):
            SO2EquivariantFeatureMap(n_features=3)

    def test_so2_invalid_n_features_1(self) -> None:
        """Test that n_features=1 raises ValueError for SO(2) encoding."""
        with pytest.raises(ValueError, match="requires n_features=2"):
            SO2EquivariantFeatureMap(n_features=1)

    def test_so2_invalid_n_features_4(self) -> None:
        """Test that n_features=4 raises ValueError for SO(2) encoding."""
        with pytest.raises(ValueError, match="requires n_features=2"):
            SO2EquivariantFeatureMap(n_features=4)

    def test_so2_invalid_n_features_type_float(self) -> None:
        """Test that float n_features raises TypeError for SO(2) encoding."""
        with pytest.raises(TypeError, match="n_features must be an integer"):
            SO2EquivariantFeatureMap(n_features=2.0)  # type: ignore[arg-type]

    def test_so2_invalid_n_features_type_bool(self) -> None:
        """Test that bool n_features raises TypeError for SO(2) encoding.

        Python's bool is a subclass of int, but should be rejected for clarity.
        """
        with pytest.raises(TypeError, match="n_features must be an integer"):
            SO2EquivariantFeatureMap(n_features=True)  # type: ignore[arg-type]

    def test_so2_invalid_max_angular_momentum_type_float(self) -> None:
        """Test that float max_angular_momentum raises TypeError."""
        with pytest.raises(TypeError, match="max_angular_momentum must be an integer"):
            SO2EquivariantFeatureMap(max_angular_momentum=1.5)  # type: ignore[arg-type]

    def test_so2_invalid_max_angular_momentum_type_bool(self) -> None:
        """Test that bool max_angular_momentum raises TypeError."""
        with pytest.raises(TypeError, match="max_angular_momentum must be an integer"):
            SO2EquivariantFeatureMap(max_angular_momentum=True)  # type: ignore[arg-type]

    # --- CyclicEquivariantFeatureMap ---

    def test_cyclic_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            CyclicEquivariantFeatureMap(n_features=4, reps=0)

    def test_cyclic_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            CyclicEquivariantFeatureMap(n_features=4, reps=-1)

    def test_cyclic_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be"):
            CyclicEquivariantFeatureMap(n_features=0)

    def test_cyclic_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be"):
            CyclicEquivariantFeatureMap(n_features=-1)

    # --- SwapEquivariantFeatureMap ---

    def test_swap_invalid_n_features_odd(self) -> None:
        """Test that odd n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be even"):
            SwapEquivariantFeatureMap(n_features=5)

    def test_swap_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            SwapEquivariantFeatureMap(n_features=4, reps=0)

    def test_swap_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be a positive integer"):
            SwapEquivariantFeatureMap(n_features=4, reps=-1)


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of equivariant encodings."""

    # --- SO2EquivariantFeatureMap ---

    def test_so2_n_features_fixed_at_2(self, so2_encoding: SO2EquivariantFeatureMap) -> None:
        """Test that n_features is always 2 for SO(2) encoding."""
        assert so2_encoding.n_features == 2

    def test_so2_n_qubits(self, so2_encoding: SO2EquivariantFeatureMap) -> None:
        """Test n_qubits calculation for SO2.

        max_m=1 => 3 states (-1, 0, 1) => ceil(log2(3)) = 2 qubits
        """
        assert so2_encoding.n_qubits == 2

    def test_so2_n_qubits_larger_max_m(self) -> None:
        """Test n_qubits with larger max angular momentum.

        max_m=3 => 7 states (-3,-2,-1,0,1,2,3) => ceil(log2(7)) = 3 qubits
        """
        enc = SO2EquivariantFeatureMap(max_angular_momentum=3)
        assert enc.n_qubits == 3

    def test_so2_depth(self, so2_encoding: SO2EquivariantFeatureMap) -> None:
        """Test circuit depth is positive integer."""
        assert so2_encoding.depth > 0
        assert isinstance(so2_encoding.depth, int)

    def test_so2_properties_object(self, so2_encoding: SO2EquivariantFeatureMap) -> None:
        """Test that properties object is computed correctly."""
        props = so2_encoding.properties
        assert props.n_qubits == 2
        assert props.is_entangling is True
        assert props.simulability == "not_simulable"

    # --- CyclicEquivariantFeatureMap ---

    def test_cyclic_n_qubits_equals_n_features(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that n_qubits equals n_features for cyclic encoding."""
        assert cyclic_encoding.n_qubits == cyclic_encoding.n_features

    def test_cyclic_depth_calculation(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test depth calculation: 3 * reps."""
        # depth = 3 * reps (H layer + RZZ layer + RX layer per rep)
        assert cyclic_encoding.depth == 3 * cyclic_encoding.reps

    def test_cyclic_gate_count_breakdown(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test gate count breakdown for cyclic encoding."""
        breakdown = cyclic_encoding.gate_count_breakdown()
        n = cyclic_encoding.n_features
        reps = cyclic_encoding.reps

        # Per rep: n RY gates, n RZZ gates, n RX gates
        assert breakdown["ry"] == n * reps
        assert breakdown["rzz"] == n * reps
        assert breakdown["rx"] == n * reps
        assert breakdown["total_single_qubit"] == breakdown["ry"] + breakdown["rx"]
        assert breakdown["total_two_qubit"] == breakdown["rzz"]

    # --- SwapEquivariantFeatureMap ---

    def test_swap_n_qubits_equals_n_features(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test that n_qubits equals n_features for swap encoding."""
        assert swap_encoding.n_qubits == swap_encoding.n_features

    def test_swap_n_pairs_calculation(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test n_pairs is half of n_features."""
        assert swap_encoding.n_pairs == swap_encoding.n_features // 2

    def test_swap_depth_calculation(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test depth calculation: 3 * reps."""
        assert swap_encoding.depth == 3 * swap_encoding.reps

    def test_swap_gate_count_breakdown(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test gate count breakdown for swap encoding.

        The swap encoding uses CZ gates (not CNOT) for symmetric
        pair entanglement that preserves equivariance.
        """
        breakdown = swap_encoding.gate_count_breakdown()
        n = swap_encoding.n_features
        reps = swap_encoding.reps
        n_pairs = swap_encoding.n_pairs

        assert breakdown["ry"] == n * reps
        assert breakdown["hadamard"] == n * reps
        assert breakdown["cz"] == n_pairs * reps
        assert breakdown["total_single_qubit"] == breakdown["ry"] + breakdown["hadamard"]
        assert breakdown["total_two_qubit"] == breakdown["cz"]
        assert breakdown["total"] == breakdown["total_single_qubit"] + breakdown["total_two_qubit"]

    def test_properties_cached(self, so2_encoding: SO2EquivariantFeatureMap) -> None:
        """Test that properties are cached (same object returned)."""
        props1 = so2_encoding.properties
        props2 = so2_encoding.properties
        assert props1 is props2


# =============================================================================
# Test Class: Equivariance Behavior (Unique to Equivariant Encodings)
# =============================================================================


class TestEquivarianceBehavior:
    """Tests for mathematical equivariance (unique to equivariant feature maps)."""

    # --- SO2 Group Action ---

    def test_so2_group_action_rotation(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test that group action correctly rotates 2D points."""
        x = np.array([1.0, 0.0])
        phi = np.pi / 2  # 90-degree rotation

        rotated = so2_encoding.group_action(phi, x)

        # Should rotate (1, 0) to approximately (0, 1)
        assert np.isclose(rotated[0], 0.0, atol=1e-10)
        assert np.isclose(rotated[1], 1.0, atol=1e-10)

    def test_so2_group_action_preserves_radius(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test that rotation preserves radius."""
        x = np.array([0.5, 0.3])
        phi = np.pi / 4

        r_original = np.sqrt(x[0] ** 2 + x[1] ** 2)
        rotated = so2_encoding.group_action(phi, x)
        r_rotated = np.sqrt(rotated[0] ** 2 + rotated[1] ** 2)

        assert np.isclose(r_original, r_rotated, atol=1e-10)

    # --- Cyclic Group Action ---

    def test_cyclic_group_action_shift_by_one(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic shift by 1."""
        x = np.array([1, 2, 3, 4])
        shifted = cyclic_encoding.group_action(1, x)
        np.testing.assert_array_equal(shifted, np.array([2, 3, 4, 1]))

    def test_cyclic_group_action_shift_by_two(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic shift by 2."""
        x = np.array([1, 2, 3, 4])
        shifted = cyclic_encoding.group_action(2, x)
        np.testing.assert_array_equal(shifted, np.array([3, 4, 1, 2]))

    def test_cyclic_group_action_full_cycle(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that full cycle returns to original."""
        x = np.array([1, 2, 3, 4])
        shifted = cyclic_encoding.group_action(4, x)
        np.testing.assert_array_equal(shifted, x)

    # --- Swap Group Action ---

    def test_swap_group_action_swap_first_pair(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swapping first pair only."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        swaps = [True, False]
        result = swap_encoding.group_action(swaps, x)
        np.testing.assert_array_equal(result, np.array([2.0, 1.0, 3.0, 4.0]))

    def test_swap_group_action_swap_both_pairs(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swapping both pairs."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        swaps = [True, True]
        result = swap_encoding.group_action(swaps, x)
        np.testing.assert_array_equal(result, np.array([2.0, 1.0, 4.0, 3.0]))

    def test_swap_group_action_no_swaps(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test no swaps (identity)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        swaps = [False, False]
        result = swap_encoding.group_action(swaps, x)
        np.testing.assert_array_equal(result, x)


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_so2_wrong_dimension(self, so2_encoding: SO2EquivariantFeatureMap) -> None:
        """Test SO2 with wrong input dimension (expected 2)."""
        with pytest.raises(ValueError, match="Expected 2 features"):
            so2_encoding.get_circuit(np.array([0.1, 0.2, 0.3]))

    def test_cyclic_wrong_dimension(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic with wrong input dimension (expected 4)."""
        with pytest.raises(ValueError, match="Expected 4 features"):
            cyclic_encoding.get_circuit(np.array([0.1, 0.2]))

    def test_swap_wrong_dimension(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swap with wrong input dimension (expected 4)."""
        with pytest.raises(ValueError, match="Expected 4 features"):
            swap_encoding.get_circuit(np.array([0.1, 0.2]))

    def test_list_input_accepted(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]  # list, not array
        # Should not raise
        if HAS_QISKIT:
            circuit = cyclic_encoding.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_batch_input_shape(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that batch input generates multiple circuits."""
        circuits = cyclic_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3


# =============================================================================
# Test Class: PennyLane Backend
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.backend_pennylane
class TestPennyLaneBackend:
    """Tests for PennyLane circuit generation."""

    def test_so2_circuit_is_callable(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test that SO2 PennyLane circuit is a callable function."""
        circuit = so2_encoding.get_circuit(sample_data_2d, backend="pennylane")
        assert callable(circuit)

    def test_cyclic_circuit_is_callable(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that cyclic PennyLane circuit is a callable function."""
        circuit = cyclic_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_swap_circuit_is_callable(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that swap PennyLane circuit is a callable function."""
        circuit = swap_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test that generated circuit executes correctly in QNode context."""
        circuit_fn = so2_encoding.get_circuit(sample_data_2d, backend="pennylane")
        dev = qml.device("default.qubit", wires=so2_encoding.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be valid (normalized)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_so2_batch_circuits(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        batch_data_2d: NDArray[np.floating],
    ) -> None:
        """Test generating SO2 circuits for batch of samples."""
        circuits = so2_encoding.get_circuits(batch_data_2d, backend="pennylane")
        assert len(circuits) == len(batch_data_2d)
        assert all(callable(c) for c in circuits)

    def test_cyclic_batch_circuits(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating cyclic circuits for batch of samples."""
        circuits = cyclic_encoding.get_circuits(batch_data_4d, backend="pennylane")
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)


# =============================================================================
# Test Class: Qiskit Backend
# =============================================================================


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
@pytest.mark.backend_qiskit
class TestQiskitBackend:
    """Tests for Qiskit circuit generation."""

    def test_so2_circuit_type(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test that SO2 Qiskit circuit is a QuantumCircuit."""
        circuit = so2_encoding.get_circuit(sample_data_2d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_so2_circuit_has_correct_qubit_count(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test that SO2 circuit has correct number of qubits."""
        circuit = so2_encoding.get_circuit(sample_data_2d, backend="qiskit")
        assert circuit.num_qubits == 2

    def test_cyclic_circuit_type(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that cyclic Qiskit circuit is a QuantumCircuit."""
        circuit = cyclic_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_cyclic_circuit_has_correct_qubit_count(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that cyclic circuit has correct number of qubits."""
        circuit = cyclic_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == cyclic_encoding.n_qubits

    def test_swap_circuit_type(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that swap Qiskit circuit is a QuantumCircuit."""
        circuit = swap_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_swap_circuit_has_correct_qubit_count(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that swap circuit has correct number of qubits."""
        circuit = swap_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == swap_encoding.n_qubits

    def test_batch_circuits(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = cyclic_encoding.get_circuits(batch_data_4d, backend="qiskit")
        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuit) for c in circuits)


# =============================================================================
# Test Class: Cirq Backend
# =============================================================================


@pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
@pytest.mark.backend_cirq
class TestCirqBackend:
    """Tests for Cirq circuit generation."""

    def test_cyclic_circuit_type(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that cyclic Cirq circuit is a cirq.Circuit."""
        circuit = cyclic_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_cyclic_circuit_has_operations(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that cyclic circuit has operations."""
        circuit = cyclic_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_cyclic_circuit_qubit_count(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that cyclic circuit has correct number of qubits."""
        circuit = cyclic_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == cyclic_encoding.n_qubits

    def test_swap_circuit_type(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that swap Cirq circuit is a cirq.Circuit."""
        circuit = swap_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_swap_circuit_qubit_count(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that swap circuit has correct number of qubits."""
        circuit = swap_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == swap_encoding.n_qubits

    def test_batch_circuits(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test generating circuits for batch of samples."""
        circuits = cyclic_encoding.get_circuits(batch_data_4d, backend="cirq")
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)


# =============================================================================
# Test Class: Mathematical Correctness
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for unitary representation and mathematical properties."""

    # --- SO2 Unitary Representation ---

    def test_so2_unitary_is_unitary(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test that SO2 unitary representation is actually unitary."""
        phi = np.pi / 4
        U = so2_encoding.unitary_representation(phi)

        # Check U†U = I
        U_dagger_U = U.conj().T @ U
        identity = np.eye(U.shape[0])

        assert np.allclose(U_dagger_U, identity, atol=1e-10)

    def test_so2_unitary_composition(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test that U(phi1)U(phi2) = U(phi1 + phi2)."""
        phi1 = np.pi / 6
        phi2 = np.pi / 3

        U1 = so2_encoding.unitary_representation(phi1)
        U2 = so2_encoding.unitary_representation(phi2)
        U_composed = so2_encoding.unitary_representation(phi1 + phi2)

        assert np.allclose(U1 @ U2, U_composed, atol=1e-10)

    # --- Cyclic Unitary Representation ---

    def test_cyclic_unitary_is_permutation(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that cyclic unitary is a permutation matrix."""
        U = cyclic_encoding.unitary_representation(1)

        # Permutation matrix: each row and column has exactly one 1
        row_sums = np.sum(np.abs(U), axis=1)
        col_sums = np.sum(np.abs(U), axis=0)

        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_cyclic_unitary_composition(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that U(k1)U(k2) = U(k1 + k2 mod n)."""
        n = cyclic_encoding.n_features
        k1, k2 = 1, 2

        U1 = cyclic_encoding.unitary_representation(k1)
        U2 = cyclic_encoding.unitary_representation(k2)
        U_composed = cyclic_encoding.unitary_representation((k1 + k2) % n)

        np.testing.assert_allclose(U1 @ U2, U_composed, atol=1e-10)

    # --- Swap Unitary Representation ---

    def test_swap_unitary_is_unitary(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test that swap unitary representation is unitary."""
        swaps = [True, False]
        U = swap_encoding.unitary_representation(swaps)

        U_dagger_U = U.conj().T @ U
        identity = np.eye(U.shape[0])

        np.testing.assert_allclose(U_dagger_U, identity, atol=1e-10)

    def test_swap_unitary_involutive(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test that SWAP^2 = I (swapping twice gives identity)."""
        swaps = [True, True]
        U = swap_encoding.unitary_representation(swaps)

        U_squared = U @ U
        identity = np.eye(U.shape[0])

        np.testing.assert_allclose(U_squared, identity, atol=1e-10)

    # --- Equivariance Verification ---

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_so2_equivariance_zero_rotation(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test SO2 equivariance for identity element (zero rotation)."""
        assert so2_encoding.verify_equivariance(sample_data_2d, 0.0)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_so2_equivariance_multiple_angles(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test SO2 equivariance for multiple rotation angles."""
        angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]

        for phi in angles:
            assert so2_encoding.verify_equivariance(
                sample_data_2d, phi, atol=1e-6
            ), f"Equivariance failed for angle {phi}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_cyclic_equivariance_identity(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test cyclic equivariance for identity element (shift by 0)."""
        assert cyclic_encoding.verify_equivariance(sample_data_4d, 0, atol=1e-6)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_cyclic_equivariance_all_shifts(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test cyclic equivariance for all possible cyclic shifts."""
        n = cyclic_encoding.n_features

        for k in range(n):
            assert cyclic_encoding.verify_equivariance(
                sample_data_4d, k, atol=1e-6
            ), f"Equivariance failed for shift k={k}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_identity(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test swap equivariance for identity (no swaps)."""
        swaps = [False, False]
        assert swap_encoding.verify_equivariance(sample_data_4d, swaps, atol=1e-6)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_single_pair(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test swap equivariance for swapping a single pair."""
        swaps = [True, False]
        assert swap_encoding.verify_equivariance(sample_data_4d, swaps, atol=1e-6)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_all_pairs(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test swap equivariance when all pairs are swapped."""
        swaps = [True, True]
        assert swap_encoding.verify_equivariance(sample_data_4d, swaps, atol=1e-6)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_on_generators(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test swap equivariance on all group generators."""
        assert swap_encoding.verify_equivariance_on_generators(
            sample_data_4d, atol=1e-6
        )

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_exhaustive_group_elements(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
    ) -> None:
        """Test swap equivariance on all group elements for random inputs."""
        from itertools import product as iter_product

        rng = np.random.default_rng(42)
        x = rng.uniform(-np.pi, np.pi, size=4)
        n_pairs = swap_encoding.n_pairs

        for swaps_tuple in iter_product([False, True], repeat=n_pairs):
            swaps = list(swaps_tuple)
            assert swap_encoding.verify_equivariance(
                x, swaps, atol=1e-6
            ), f"Equivariance failed for swaps={swaps}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_multiple_random_inputs(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
    ) -> None:
        """Test swap equivariance on multiple random inputs."""
        rng = np.random.default_rng(123)

        for _ in range(10):
            x = rng.uniform(-np.pi, np.pi, size=4)
            for swaps in [[True, False], [False, True], [True, True]]:
                assert swap_encoding.verify_equivariance(
                    x, swaps, atol=1e-6
                ), f"Equivariance failed for x={x}, swaps={swaps}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_various_reps(self) -> None:
        """Test swap equivariance holds across different repetition counts."""
        x = np.array([0.3, 0.7, 1.1, 0.5])
        swaps = [True, True]

        for reps in [1, 2, 3, 4]:
            enc = SwapEquivariantFeatureMap(n_features=4, reps=reps)
            assert enc.verify_equivariance(
                x, swaps, atol=1e-6
            ), f"Equivariance failed for reps={reps}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_swap_equivariance_six_features(self) -> None:
        """Test swap equivariance with 6 features (3 pairs)."""
        enc = SwapEquivariantFeatureMap(n_features=6, reps=2)
        x = np.array([0.1, 0.9, 0.4, 0.6, 0.2, 0.8])

        for g in enc.group_generators():
            assert enc.verify_equivariance(
                x, g, atol=1e-6
            ), f"Equivariance failed for generator {g}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_equivariance_on_generators(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test SO2 equivariance on all generators."""
        assert so2_encoding.verify_equivariance_on_generators(sample_data_2d, atol=1e-6)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_so2_equivariance_detailed_result(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test SO2 detailed equivariance verification."""
        phi = np.pi / 4
        result = so2_encoding.verify_equivariance_detailed(
            sample_data_2d, phi, atol=1e-6
        )

        assert result["equivariant"] is True
        assert result["overlap"] > 0.999999
        assert result["tolerance"] == 1e-6
        assert result["group_element"] == phi


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_so2_max_angular_momentum_zero(self) -> None:
        """Test SO2 with max_angular_momentum=0 (single state)."""
        enc = SO2EquivariantFeatureMap(max_angular_momentum=0)
        assert len(enc.angular_momenta) == 1  # Only m=0
        assert enc.n_qubits >= 1

    def test_cyclic_single_qubit(self) -> None:
        """Test cyclic encoding with single feature (trivial case)."""
        enc = CyclicEquivariantFeatureMap(n_features=1)
        assert enc.n_qubits == 1
        assert enc.n_features == 1

    def test_cyclic_two_features(self) -> None:
        """Test cyclic encoding with minimum non-trivial case."""
        enc = CyclicEquivariantFeatureMap(n_features=2)
        assert enc.n_qubits == 2

    def test_swap_minimum_features(self) -> None:
        """Test swap encoding with minimum features (2)."""
        enc = SwapEquivariantFeatureMap(n_features=2)
        assert enc.n_features == 2
        assert enc.n_pairs == 1

    def test_so2_zero_input(self, so2_encoding: SO2EquivariantFeatureMap) -> None:
        """Test SO2 with zero input (origin)."""
        x = np.array([0.0, 0.0])
        if HAS_QISKIT:
            circuit = so2_encoding.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_cyclic_zero_input(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic with all-zero input."""
        x = np.array([0.0, 0.0, 0.0, 0.0])
        if HAS_QISKIT:
            circuit = cyclic_encoding.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_cyclic_ones_input(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic with all-ones input."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        if HAS_QISKIT:
            circuit = cyclic_encoding.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_so2_very_small_radius(self) -> None:
        """Test SO2 encoding with very small radius values."""
        enc = SO2EquivariantFeatureMap()
        x = np.array([1e-15, 1e-16])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_so2_very_large_radius(self) -> None:
        """Test SO2 encoding with very large radius values."""
        enc = SO2EquivariantFeatureMap()
        x = np.array([1e10, 1e15])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_cyclic_very_small_values(self) -> None:
        """Test cyclic encoding with very small input values."""
        enc = CyclicEquivariantFeatureMap(n_features=4)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_cyclic_very_large_values(self) -> None:
        """Test cyclic encoding with very large input values."""
        enc = CyclicEquivariantFeatureMap(n_features=4)
        x = np.array([1e10, 1e15, 1e20, 1e30])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    def test_mixed_extreme_magnitudes(self) -> None:
        """Test cyclic encoding with mixed extreme magnitude values."""
        enc = CyclicEquivariantFeatureMap(n_features=4)
        x = np.array([1e-20, 1e20, -1e15, 0.5])
        if HAS_QISKIT:
            circuit = enc.get_circuit(x, backend="qiskit")
            assert isinstance(circuit, QuantumCircuit)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_extreme_values_produce_normalized_state(self) -> None:
        """Test that extreme values still produce normalized quantum states."""
        enc = CyclicEquivariantFeatureMap(n_features=4)
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([1e-10, 1e10, 0.5, 0.1])
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

    def test_so2_equality_same_parameters(self) -> None:
        """Test that SO2 encodings with same parameters are equal."""
        enc1 = SO2EquivariantFeatureMap(max_angular_momentum=2)
        enc2 = SO2EquivariantFeatureMap(max_angular_momentum=2)
        assert enc1 == enc2

    def test_so2_equality_different_parameters(self) -> None:
        """Test that SO2 encodings with different parameters are not equal."""
        enc1 = SO2EquivariantFeatureMap(max_angular_momentum=1)
        enc2 = SO2EquivariantFeatureMap(max_angular_momentum=2)
        assert enc1 != enc2

    def test_cyclic_equality_same_parameters(self) -> None:
        """Test that cyclic encodings with same parameters are equal."""
        enc1 = CyclicEquivariantFeatureMap(n_features=4, reps=2)
        enc2 = CyclicEquivariantFeatureMap(n_features=4, reps=2)
        assert enc1 == enc2

    def test_cyclic_equality_different_n_features(self) -> None:
        """Test that cyclic encodings with different n_features are not equal."""
        enc1 = CyclicEquivariantFeatureMap(n_features=4)
        enc2 = CyclicEquivariantFeatureMap(n_features=8)
        assert enc1 != enc2

    def test_swap_equality_same_parameters(self) -> None:
        """Test that swap encodings with same parameters are equal."""
        enc1 = SwapEquivariantFeatureMap(n_features=4, reps=2)
        enc2 = SwapEquivariantFeatureMap(n_features=4, reps=2)
        assert enc1 == enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = CyclicEquivariantFeatureMap(n_features=4)
        enc2 = CyclicEquivariantFeatureMap(n_features=4)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = CyclicEquivariantFeatureMap(n_features=4)
        enc2 = CyclicEquivariantFeatureMap(n_features=4)  # Same as enc1
        enc3 = CyclicEquivariantFeatureMap(n_features=8)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = CyclicEquivariantFeatureMap(n_features=4)
        enc2 = CyclicEquivariantFeatureMap(n_features=4)

        d = {enc1: "four features"}
        assert d[enc2] == "four features"  # enc2 should find same key

    def test_different_encoding_types_not_equal(self) -> None:
        """Test that different encoding types are not equal."""
        cyclic = CyclicEquivariantFeatureMap(n_features=4)
        swap = SwapEquivariantFeatureMap(n_features=4)
        assert cyclic != swap


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for string representation."""

    def test_so2_repr_contains_class_name(self) -> None:
        """Test that SO2 repr contains the class name."""
        enc = SO2EquivariantFeatureMap()
        repr_str = repr(enc)
        assert "SO2EquivariantFeatureMap" in repr_str

    def test_so2_repr_contains_parameters(self) -> None:
        """Test that SO2 repr contains key parameters."""
        enc = SO2EquivariantFeatureMap(max_angular_momentum=2, radial_sigma=0.5)
        repr_str = repr(enc)
        assert "max_angular_momentum=2" in repr_str or "2" in repr_str

    def test_cyclic_repr_contains_class_name(self) -> None:
        """Test that cyclic repr contains the class name."""
        enc = CyclicEquivariantFeatureMap(n_features=4)
        repr_str = repr(enc)
        assert "CyclicEquivariantFeatureMap" in repr_str

    def test_cyclic_repr_contains_parameters(self) -> None:
        """Test that cyclic repr contains key parameters."""
        enc = CyclicEquivariantFeatureMap(n_features=4, reps=3)
        repr_str = repr(enc)
        assert "n_features=4" in repr_str or "4" in repr_str
        assert "reps=3" in repr_str or "3" in repr_str

    def test_swap_repr_contains_class_name(self) -> None:
        """Test that swap repr contains the class name."""
        enc = SwapEquivariantFeatureMap(n_features=4)
        repr_str = repr(enc)
        assert "SwapEquivariantFeatureMap" in repr_str

    def test_repr_is_string(self) -> None:
        """Test that repr returns a string."""
        enc = CyclicEquivariantFeatureMap(n_features=4)
        assert isinstance(repr(enc), str)


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for invalid backend handling."""

    def test_so2_invalid_backend(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test that SO2 invalid backend raises ValueError."""
        x = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="Unknown backend"):
            so2_encoding.get_circuit(x, backend="invalid")

    def test_cyclic_invalid_backend(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that cyclic invalid backend raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="Unknown backend"):
            cyclic_encoding.get_circuit(x, backend="invalid")

    def test_swap_invalid_backend(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test that swap invalid backend raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="Unknown backend"):
            swap_encoding.get_circuit(x, backend="invalid")

    def test_empty_backend_string(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that empty backend string raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError):
            cyclic_encoding.get_circuit(x, backend="")

    def test_case_sensitive_backend(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that backend names are case-sensitive."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="Unknown backend"):
            cyclic_encoding.get_circuit(x, backend="Qiskit")  # Capital Q


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_so2_pickle_roundtrip(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test that SO2 encoding can be pickled and unpickled."""
        pickled = pickle.dumps(so2_encoding)
        restored = pickle.loads(pickled)

        assert restored.n_features == so2_encoding.n_features
        assert restored.max_angular_momentum == so2_encoding.max_angular_momentum
        assert restored.radial_function == so2_encoding.radial_function

    def test_cyclic_pickle_roundtrip(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that cyclic encoding can be pickled and unpickled."""
        pickled = pickle.dumps(cyclic_encoding)
        restored = pickle.loads(pickled)

        assert restored.n_features == cyclic_encoding.n_features
        assert restored.reps == cyclic_encoding.reps

    def test_swap_pickle_roundtrip(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test that swap encoding can be pickled and unpickled."""
        pickled = pickle.dumps(swap_encoding)
        restored = pickle.loads(pickled)

        assert restored.n_features == swap_encoding.n_features
        assert restored.n_pairs == swap_encoding.n_pairs

    def test_pickle_equality(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that pickled and restored encoding equals original."""
        pickled = pickle.dumps(cyclic_encoding)
        restored = pickle.loads(pickled)

        assert cyclic_encoding == restored
        assert hash(cyclic_encoding) == hash(restored)

    def test_pickle_circuit_generation_after_restore(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that circuit generation works after unpickling."""
        pickled = pickle.dumps(cyclic_encoding)
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

    def test_concurrent_circuit_generation(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for i in range(num_circuits_per_thread):
                    x = np.random.randn(4)
                    circuit = cyclic_encoding.get_circuit(x, backend="qiskit")
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

    def test_concurrent_property_access(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that concurrent property access is thread-safe."""
        num_threads = 20

        results: list[Any] = []
        errors: list[Exception] = []

        def access_properties() -> None:
            try:
                props = cyclic_encoding.properties
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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_concurrent_equivariance_verification(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test concurrent equivariance verification."""
        num_threads = 5
        results: list[bool] = []
        errors: list[Exception] = []

        def verify_equivariance(shift: int) -> None:
            try:
                result = cyclic_encoding.verify_equivariance(
                    sample_data_4d, shift, atol=1e-6
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(verify_equivariance, k) for k in range(4)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0
        assert all(results)


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
    def test_all_backends_produce_valid_states(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that all backends produce valid normalized quantum states."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane
        pl_circuit = cyclic_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=cyclic_encoding.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = pl_full()
        assert np.isclose(np.sum(np.abs(pl_state) ** 2), 1.0, atol=1e-10)

        # Qiskit
        qk_circuit = cyclic_encoding.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        assert np.isclose(np.sum(np.abs(qk_sv.data) ** 2), 1.0, atol=1e-10)

        # Cirq
        cirq_circuit = cyclic_encoding.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = cirq_result.final_state_vector
        assert np.isclose(np.sum(np.abs(cirq_state) ** 2), 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that different inputs produce different quantum states."""
        dev = qml.device("default.qubit", wires=cyclic_encoding.n_qubits)

        inputs = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]

        states = []
        for x in inputs:
            circuit_fn = cyclic_encoding.get_circuit(x, backend="pennylane")

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
    def test_reproducibility(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that same input always produces same state."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        dev = qml.device("default.qubit", wires=cyclic_encoding.n_qubits)

        states = []
        for _ in range(5):
            circuit_fn = cyclic_encoding.get_circuit(x, backend="pennylane")

            @qml.qnode(dev)
            def circuit():
                circuit_fn()
                return qml.state()

            states.append(circuit())

        # All states should be identical
        for i in range(1, len(states)):
            assert np.allclose(states[0], states[i], atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_so2_equivariance_detailed_full_circle(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test SO2 equivariance across full rotation circle."""
        x = np.array([0.5, 0.3])
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        for phi in angles:
            result = so2_encoding.verify_equivariance_detailed(x, phi, atol=1e-5)
            assert result["equivariant"], f"Equivariance failed at angle {phi}"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_cyclic_equivariance_with_various_inputs(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic equivariance with various input patterns."""
        test_inputs = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([-0.5, 0.5, -0.5, 0.5]),
        ]

        for x in test_inputs:
            for k in range(cyclic_encoding.n_features):
                assert cyclic_encoding.verify_equivariance(
                    x, k, atol=1e-5
                ), f"Failed for input {x} with shift {k}"

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT),
        reason="PennyLane and Qiskit required",
    )
    @pytest.mark.cross_backend
    def test_cyclic_pennylane_qiskit_fidelity(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test PennyLane-Qiskit statevector fidelity for cyclic encoding.

        Both backends use RZZ-type entangling gates that must agree on the
        parameter convention.  A previous bug applied a spurious 2x factor
        in the Qiskit backend; this test guards against regressions.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane statevector
        pl_fn = cyclic_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=cyclic_encoding.n_qubits)

        @qml.qnode(dev)
        def pl_circuit():
            pl_fn()
            return qml.state()

        sv_pl = pl_circuit()

        # Qiskit statevector (reverse qubit order to match PennyLane MSB)
        qk_circuit = cyclic_encoding.get_circuit(x, backend="qiskit")
        sv_qk = np.array(Statevector(qk_circuit))

        # Qiskit uses LSB ordering; reverse to match PennyLane MSB
        n = cyclic_encoding.n_qubits
        sv_qk_reordered = sv_qk.reshape([2] * n).transpose(
            list(range(n))[::-1]
        ).flatten()

        fidelity = np.abs(np.vdot(sv_pl, sv_qk_reordered)) ** 2

        assert fidelity > 0.9999, (
            f"PennyLane-Qiskit fidelity is {fidelity:.6f} for cyclic encoding. "
            f"RZZ parameter convention may be inconsistent."
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_CIRQ),
        reason="PennyLane and Cirq required",
    )
    @pytest.mark.cross_backend
    def test_cyclic_pennylane_cirq_fidelity(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test PennyLane-Cirq statevector fidelity for cyclic encoding.

        Cirq's ZZPowGate uses a different parameter convention
        (exponent-based) from PennyLane's IsingZZ.  This test verifies
        the conversion is correct.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane statevector
        pl_fn = cyclic_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=cyclic_encoding.n_qubits)

        @qml.qnode(dev)
        def pl_circuit():
            pl_fn()
            return qml.state()

        sv_pl = pl_circuit()

        # Cirq statevector
        cirq_circuit = cyclic_encoding.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        sv_cirq = cirq_result.final_state_vector

        fidelity = np.abs(np.vdot(sv_pl, sv_cirq)) ** 2

        assert fidelity > 0.9999, (
            f"PennyLane-Cirq fidelity is {fidelity:.6f} for cyclic encoding. "
            f"ZZPowGate exponent conversion may be inconsistent."
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT),
        reason="PennyLane and Qiskit required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize(
        "coupling_strength",
        [np.pi / 8, np.pi / 4, np.pi / 2, np.pi],
        ids=["pi/8", "pi/4", "pi/2", "pi"],
    )
    def test_cyclic_fidelity_varying_coupling(
        self, coupling_strength: float
    ) -> None:
        """Test cross-backend fidelity across different coupling strengths.

        Ensures the RZZ parameter fix is correct for all coupling values,
        not just the default π/4.
        """
        enc = CyclicEquivariantFeatureMap(
            n_features=4, reps=1, coupling_strength=coupling_strength
        )
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane
        pl_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_circuit():
            pl_fn()
            return qml.state()

        sv_pl = pl_circuit()

        # Qiskit (reorder to MSB)
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        sv_qk = np.array(Statevector(qk_circuit))
        n = enc.n_qubits
        sv_qk_reordered = sv_qk.reshape([2] * n).transpose(
            list(range(n))[::-1]
        ).flatten()

        fidelity = np.abs(np.vdot(sv_pl, sv_qk_reordered)) ** 2

        assert fidelity > 0.9999, (
            f"Fidelity {fidelity:.6f} at coupling_strength={coupling_strength:.4f}"
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT),
        reason="PennyLane and Qiskit required",
    )
    @pytest.mark.cross_backend
    def test_swap_pennylane_qiskit_fidelity(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test PennyLane-Qiskit statevector fidelity for swap encoding.

        Verifies that both backends produce the same quantum state for the
        swap equivariant circuit using direct RY encoding and CZ entanglement.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane statevector
        pl_fn = swap_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=swap_encoding.n_qubits)

        @qml.qnode(dev)
        def pl_circuit():
            pl_fn()
            return qml.state()

        sv_pl = pl_circuit()

        # Qiskit statevector (reverse qubit order to match PennyLane MSB)
        qk_circuit = swap_encoding.get_circuit(x, backend="qiskit")
        sv_qk = np.array(Statevector(qk_circuit))

        n = swap_encoding.n_qubits
        sv_qk_reordered = sv_qk.reshape([2] * n).transpose(
            list(range(n))[::-1]
        ).flatten()

        fidelity = np.abs(np.vdot(sv_pl, sv_qk_reordered)) ** 2

        assert fidelity > 0.9999, (
            f"PennyLane-Qiskit fidelity is {fidelity:.6f} for swap encoding."
        )

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_CIRQ),
        reason="PennyLane and Cirq required",
    )
    @pytest.mark.cross_backend
    def test_swap_pennylane_cirq_fidelity(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test PennyLane-Cirq statevector fidelity for swap encoding.

        Verifies that both backends produce the same quantum state for the
        swap equivariant circuit using direct RY encoding and CZ entanglement.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # PennyLane statevector
        pl_fn = swap_encoding.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=swap_encoding.n_qubits)

        @qml.qnode(dev)
        def pl_circuit():
            pl_fn()
            return qml.state()

        sv_pl = pl_circuit()

        # Cirq statevector
        cirq_circuit = swap_encoding.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator()
        cirq_result = cirq_sim.simulate(cirq_circuit)
        sv_cirq = cirq_result.final_state_vector

        fidelity = np.abs(np.vdot(sv_pl, sv_cirq)) ** 2

        assert fidelity > 0.9999, (
            f"PennyLane-Cirq fidelity is {fidelity:.6f} for swap encoding."
        )


# =============================================================================
# Test Class: Statistical Verification
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
@pytest.mark.statistical_verification
class TestStatisticalVerification:
    """Tests for statistical equivariance verification.

    Statistical verification uses measurement sampling to compare probability
    distributions, making it scalable to larger systems where exact state
    vector verification would require exponential memory.
    """

    def test_so2_statistical_verification_basic(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test SO2 statistical verification with basic input."""
        x = np.array([0.5, 0.3])
        phi = np.pi / 4

        result = so2_encoding.verify_equivariance_statistical(
            x, phi, n_shots=5000, significance=0.05
        )

        assert "equivariant" in result
        assert "p_value" in result
        assert "test_statistic" in result
        assert "significance" in result
        assert "n_shots" in result
        assert "group_element" in result
        assert "method" in result
        assert "confidence_level" in result

        # For a correct equivariant encoding, should pass the test
        assert result["equivariant"] is True
        assert result["confidence_level"] == 0.95

    def test_cyclic_statistical_verification(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic statistical verification."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        result = cyclic_encoding.verify_equivariance_statistical(
            x, 1, n_shots=10000, significance=0.01
        )

        assert result["equivariant"] is True
        assert result["n_shots"] == 10000
        assert result["group_element"] == 1

    def test_swap_statistical_verification(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swap statistical verification."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        swaps = [True, False]

        result = swap_encoding.verify_equivariance_statistical(
            x, swaps, n_shots=10000, significance=0.01
        )

        assert result["equivariant"] is True
        assert result["group_element"] == swaps

    def test_statistical_verification_higher_confidence(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test statistical verification with higher confidence (more shots)."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        result = cyclic_encoding.verify_equivariance_statistical(
            x, 1, n_shots=10000, significance=0.01
        )

        assert result["equivariant"] is True
        assert result["confidence_level"] == 0.99

    def test_statistical_verification_invalid_n_shots_too_low(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that n_shots < 100 raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="n_shots must be at least 100"):
            cyclic_encoding.verify_equivariance_statistical(x, 1, n_shots=50)

    def test_statistical_verification_invalid_significance_zero(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that significance=0 raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="significance must be in"):
            cyclic_encoding.verify_equivariance_statistical(
                x, 1, n_shots=1000, significance=0.0
            )

    def test_statistical_verification_invalid_significance_one(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that significance=1 raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="significance must be in"):
            cyclic_encoding.verify_equivariance_statistical(
                x, 1, n_shots=1000, significance=1.0
            )

    def test_statistical_verification_invalid_significance_negative(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that negative significance raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="significance must be in"):
            cyclic_encoding.verify_equivariance_statistical(
                x, 1, n_shots=1000, significance=-0.1
            )

    def test_statistical_verification_result_types(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test that statistical verification returns correct types."""
        x = np.array([0.5, 0.3])

        result = so2_encoding.verify_equivariance_statistical(
            x, np.pi / 4, n_shots=1000
        )

        assert isinstance(result["equivariant"], bool)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["test_statistic"], float)
        assert isinstance(result["significance"], float)
        assert isinstance(result["n_shots"], int)
        assert isinstance(result["method"], str)
        assert isinstance(result["confidence_level"], float)

    def test_statistical_verification_multiple_angles(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test SO2 statistical verification across multiple rotation angles.

        Note: Statistical tests are inherently probabilistic. We use high shot
        counts and low significance levels to minimize false negatives while
        still testing the statistical verification infrastructure.
        """
        x = np.array([0.5, 0.3])
        angles = [np.pi / 6, np.pi / 4, np.pi / 2]

        # Track successes - allow one failure due to statistical variation
        successes = 0
        for phi in angles:
            result = so2_encoding.verify_equivariance_statistical(
                x, phi, n_shots=8000, significance=0.01
            )
            if result["equivariant"]:
                successes += 1

        # At least 2 of 3 should pass (allows for statistical variation)
        assert successes >= 2, f"Only {successes}/3 angles passed statistical test"

    def test_statistical_verification_all_cyclic_shifts(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic statistical verification for all shift values.

        Note: Statistical tests are inherently probabilistic. We use high shot
        counts and track success rates rather than requiring all tests to pass.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4])
        n = cyclic_encoding.n_features

        # Track successes - allow one failure due to statistical variation
        successes = 0
        for k in range(n):
            result = cyclic_encoding.verify_equivariance_statistical(
                x, k, n_shots=8000, significance=0.01
            )
            if result["equivariant"]:
                successes += 1

        # At least n-1 should pass (allows for one statistical variation)
        assert successes >= n - 1, f"Only {successes}/{n} shifts passed statistical test"


# =============================================================================
# Test Class: Auto Verification Method Selection
# =============================================================================


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
class TestAutoVerification:
    """Tests for automatic verification method selection.

    The verify_equivariance_auto method automatically chooses between exact
    and statistical verification based on the number of qubits.
    """

    def test_so2_auto_verification(
        self,
        so2_encoding: SO2EquivariantFeatureMap,
        sample_data_2d: NDArray[np.floating],
    ) -> None:
        """Test SO2 auto verification selects correct method."""
        # SO2 with max_m=1 has 2 qubits, should use exact verification
        result = so2_encoding.verify_equivariance_auto(sample_data_2d, np.pi / 4)
        assert result is True

    def test_cyclic_auto_verification(
        self,
        cyclic_encoding: CyclicEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test cyclic auto verification selects correct method."""
        # 4 qubits should use exact verification
        result = cyclic_encoding.verify_equivariance_auto(sample_data_4d, 1)
        assert result is True

    def test_swap_auto_verification(
        self,
        swap_encoding: SwapEquivariantFeatureMap,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test swap auto verification selects correct method."""
        result = swap_encoding.verify_equivariance_auto(
            sample_data_4d, [True, False]
        )
        assert result is True

    def test_auto_verification_identity_element(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test auto verification with identity element (shift by 0)."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        result = cyclic_encoding.verify_equivariance_auto(x, 0)
        assert result is True

    def test_auto_verification_multiple_shifts(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test auto verification with multiple cyclic shifts."""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        for k in range(cyclic_encoding.n_features):
            result = cyclic_encoding.verify_equivariance_auto(x, k)
            assert result is True, f"Auto verification failed for shift k={k}"

    def test_auto_verification_with_custom_tolerance(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test auto verification with custom tolerance."""
        x = np.array([0.5, 0.3])
        result = so2_encoding.verify_equivariance_auto(x, np.pi / 4, atol=1e-8)
        assert result is True


# =============================================================================
# Test Class: Resource Summary
# =============================================================================


class TestResourceSummary:
    """Tests for resource_summary() method across all equivariant encodings."""

    # --- SO2 Resource Summary ---

    def test_so2_resource_summary_structure(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test SO2 resource summary contains all expected keys."""
        summary = so2_encoding.resource_summary()

        # Circuit structure
        assert "n_qubits" in summary
        assert "n_features" in summary
        assert "depth" in summary
        assert "max_angular_momentum" in summary
        assert "radial_function" in summary
        assert "radial_sigma" in summary

        # Gate counts
        assert "gate_counts" in summary
        assert isinstance(summary["gate_counts"], dict)

        # Encoding characteristics
        assert "is_entangling" in summary
        assert "simulability" in summary
        assert "trainability_estimate" in summary

        # Symmetry information
        assert "symmetry_group" in summary
        assert "angular_momenta" in summary
        assert "n_angular_states" in summary

        # Verification cost
        assert "verification_cost" in summary
        assert "exact" in summary["verification_cost"]
        assert "statistical" in summary["verification_cost"]

        # Hardware requirements
        assert "hardware_requirements" in summary
        assert "connectivity" in summary["hardware_requirements"]
        assert "native_gates" in summary["hardware_requirements"]

        # Entanglement details
        assert "n_entanglement_pairs" in summary
        assert "entanglement_pairs" in summary

        # Verification methods
        assert "verification_methods" in summary

    def test_so2_resource_summary_values(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test SO2 resource summary returns correct values."""
        summary = so2_encoding.resource_summary()

        assert summary["n_qubits"] == 2
        assert summary["n_features"] == 2
        assert summary["max_angular_momentum"] == 1
        assert summary["symmetry_group"] == "SO(2)"
        assert summary["angular_momenta"] == [-1, 0, 1]
        assert summary["n_angular_states"] == 3
        assert summary["is_entangling"] is True
        assert summary["radial_function"] == "gaussian"
        assert summary["radial_sigma"] == 1.0

    def test_so2_resource_summary_gate_counts(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test SO2 resource summary gate count breakdown."""
        summary = so2_encoding.resource_summary()
        gate_counts = summary["gate_counts"]

        assert "state_preparation" in gate_counts
        assert "phase_gates" in gate_counts
        assert "total_single_qubit" in gate_counts
        assert "total_two_qubit" in gate_counts
        assert "total" in gate_counts

        # Gate counts should be non-negative integers
        assert gate_counts["total"] >= 0
        assert gate_counts["total_single_qubit"] >= 0
        assert gate_counts["total_two_qubit"] >= 0

    def test_so2_resource_summary_custom_parameters(self) -> None:
        """Test SO2 resource summary with custom parameters."""
        enc = SO2EquivariantFeatureMap(
            max_angular_momentum=2,
            radial_function="uniform",
            radial_sigma=0.5,
        )
        summary = enc.resource_summary()

        assert summary["max_angular_momentum"] == 2
        assert summary["radial_function"] == "uniform"
        assert summary["radial_sigma"] == 0.5
        assert summary["angular_momenta"] == [-2, -1, 0, 1, 2]
        assert summary["n_angular_states"] == 5

    # --- Cyclic Resource Summary ---

    def test_cyclic_resource_summary_structure(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic resource summary contains all expected keys."""
        summary = cyclic_encoding.resource_summary()

        # Circuit structure
        assert "n_qubits" in summary
        assert "n_features" in summary
        assert "depth" in summary
        assert "reps" in summary
        assert "coupling_strength" in summary

        # Gate counts
        assert "gate_counts" in summary
        gate_counts = summary["gate_counts"]
        assert "ry" in gate_counts
        assert "rzz" in gate_counts
        assert "rx" in gate_counts
        assert "total_single_qubit" in gate_counts
        assert "total_two_qubit" in gate_counts
        assert "total" in gate_counts

        # Symmetry information
        assert "symmetry_group" in summary
        assert "cyclic_order" in summary

        # Hardware requirements
        assert "hardware_requirements" in summary
        assert summary["hardware_requirements"]["connectivity"] == "ring"

        # Entanglement details
        assert "n_entanglement_pairs" in summary
        assert "entanglement_pairs" in summary

    def test_cyclic_resource_summary_values(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic resource summary returns correct values."""
        summary = cyclic_encoding.resource_summary()

        assert summary["n_qubits"] == 4
        assert summary["n_features"] == 4
        assert summary["reps"] == 2
        assert summary["symmetry_group"] == "Z_4"
        assert summary["cyclic_order"] == 4
        assert summary["n_entanglement_pairs"] == 4  # Ring of 4 qubits
        assert len(summary["entanglement_pairs"]) == 4

    def test_cyclic_resource_summary_entanglement_pairs(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic entanglement pairs form a ring topology."""
        summary = cyclic_encoding.resource_summary()
        pairs = summary["entanglement_pairs"]

        # Should include (0,1), (1,2), (2,3), (3,0) for ring of 4
        expected_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert pairs == expected_pairs

    def test_cyclic_resource_summary_gate_counts(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic gate count breakdown is correct."""
        summary = cyclic_encoding.resource_summary()
        gate_counts = summary["gate_counts"]
        n = cyclic_encoding.n_features
        reps = cyclic_encoding.reps

        # Per rep: n RY gates, n RZZ gates, n RX gates
        assert gate_counts["ry"] == n * reps
        assert gate_counts["rzz"] == n * reps
        assert gate_counts["rx"] == n * reps
        assert gate_counts["total_single_qubit"] == gate_counts["ry"] + gate_counts["rx"]
        assert gate_counts["total_two_qubit"] == gate_counts["rzz"]

    # --- Swap Resource Summary ---

    def test_swap_resource_summary_structure(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swap resource summary contains all expected keys."""
        summary = swap_encoding.resource_summary()

        # Circuit structure
        assert "n_qubits" in summary
        assert "n_features" in summary
        assert "depth" in summary
        assert "reps" in summary
        assert "n_pairs" in summary

        # Gate counts
        assert "gate_counts" in summary
        gate_counts = summary["gate_counts"]
        assert "ry" in gate_counts
        assert "hadamard" in gate_counts
        assert "cz" in gate_counts
        assert "total_single_qubit" in gate_counts
        assert "total_two_qubit" in gate_counts
        assert "total" in gate_counts

        # Symmetry information
        assert "symmetry_group" in summary
        assert "n_pair_swaps" in summary

        # Hardware requirements
        assert "hardware_requirements" in summary
        assert summary["hardware_requirements"]["connectivity"] == "pairs"

    def test_swap_resource_summary_values(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swap resource summary returns correct values."""
        summary = swap_encoding.resource_summary()

        assert summary["n_qubits"] == 4
        assert summary["n_features"] == 4
        assert summary["n_pairs"] == 2
        assert summary["reps"] == 2
        assert summary["symmetry_group"] == "S_2^2"
        assert summary["n_pair_swaps"] == 2
        assert summary["n_entanglement_pairs"] == 2  # 2 pairs

    def test_swap_resource_summary_entanglement_pairs(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swap entanglement pairs are within feature pairs."""
        summary = swap_encoding.resource_summary()
        pairs = summary["entanglement_pairs"]

        # Should be [(0,1), (2,3)] for 4 features with 2 pairs
        expected_pairs = [(0, 1), (2, 3)]
        assert pairs == expected_pairs

    def test_swap_resource_summary_gate_counts(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swap gate count breakdown is correct."""
        summary = swap_encoding.resource_summary()
        gate_counts = summary["gate_counts"]
        n = swap_encoding.n_features
        reps = swap_encoding.reps
        n_pairs = swap_encoding.n_pairs

        # Per rep: n RY gates, n H gates, n_pairs CZ gates
        assert gate_counts["ry"] == n * reps
        assert gate_counts["hadamard"] == n * reps
        assert gate_counts["cz"] == n_pairs * reps
        assert gate_counts["total_single_qubit"] == gate_counts["ry"] + gate_counts["hadamard"]
        assert gate_counts["total_two_qubit"] == gate_counts["cz"]

    def test_swap_resource_summary_six_features(self) -> None:
        """Test swap resource summary with 6 features (3 pairs)."""
        enc = SwapEquivariantFeatureMap(n_features=6, reps=3)
        summary = enc.resource_summary()

        assert summary["n_qubits"] == 6
        assert summary["n_features"] == 6
        assert summary["n_pairs"] == 3
        assert summary["symmetry_group"] == "S_2^3"
        assert summary["n_entanglement_pairs"] == 3
        assert summary["entanglement_pairs"] == [(0, 1), (2, 3), (4, 5)]

    # --- Verification Cost Information ---

    def test_verification_cost_structure(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test verification cost information is complete."""
        summary = so2_encoding.resource_summary()
        cost = summary["verification_cost"]

        # Exact verification cost
        assert "memory" in cost["exact"]
        assert "time" in cost["exact"]
        assert "recommended_max_qubits" in cost["exact"]

        # Statistical verification cost
        assert "memory" in cost["statistical"]
        assert "time" in cost["statistical"]
        assert "default_shots" in cost["statistical"]
        assert "scalable" in cost["statistical"]

        # Statistical should be scalable
        assert cost["statistical"]["scalable"] is True

    def test_verification_methods_listed(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test that verification methods are properly listed."""
        summary = cyclic_encoding.resource_summary()

        assert "verification_methods" in summary
        methods = summary["verification_methods"]

        assert any("exact" in m for m in methods)
        assert any("statistical" in m for m in methods)
        assert any("auto" in m for m in methods)


# =============================================================================
# Test Class: Get Entanglement Pairs
# =============================================================================


class TestGetEntanglementPairs:
    """Tests for get_entanglement_pairs() method."""

    def test_so2_entanglement_pairs_empty(
        self, so2_encoding: SO2EquivariantFeatureMap
    ) -> None:
        """Test SO2 entanglement pairs are empty (uses state preparation)."""
        pairs = so2_encoding.get_entanglement_pairs()
        assert pairs == []

    def test_cyclic_entanglement_pairs_ring(
        self, cyclic_encoding: CyclicEquivariantFeatureMap
    ) -> None:
        """Test cyclic entanglement pairs form a ring."""
        pairs = cyclic_encoding.get_entanglement_pairs()

        # 4 qubits should have 4 pairs in ring: (0,1), (1,2), (2,3), (3,0)
        assert len(pairs) == 4
        assert (0, 1) in pairs
        assert (1, 2) in pairs
        assert (2, 3) in pairs
        assert (3, 0) in pairs

    def test_swap_entanglement_pairs_within_pairs(
        self, swap_encoding: SwapEquivariantFeatureMap
    ) -> None:
        """Test swap entanglement pairs are within feature pairs."""
        pairs = swap_encoding.get_entanglement_pairs()

        # 4 qubits = 2 pairs: (0,1), (2,3)
        assert len(pairs) == 2
        assert (0, 1) in pairs
        assert (2, 3) in pairs

    def test_cyclic_entanglement_pairs_various_sizes(self) -> None:
        """Test cyclic entanglement pairs for various sizes."""
        for n in [2, 3, 5, 8]:
            enc = CyclicEquivariantFeatureMap(n_features=n)
            pairs = enc.get_entanglement_pairs()

            # Ring has n pairs
            assert len(pairs) == n

            # Each pair should be (i, (i+1) % n)
            for i in range(n):
                assert (i, (i + 1) % n) in pairs

    def test_swap_entanglement_pairs_various_sizes(self) -> None:
        """Test swap entanglement pairs for various sizes."""
        for n in [2, 4, 6, 8]:
            enc = SwapEquivariantFeatureMap(n_features=n)
            pairs = enc.get_entanglement_pairs()

            # n_pairs = n // 2
            assert len(pairs) == n // 2

            # Each pair should be (2i, 2i+1)
            for i in range(n // 2):
                assert (2 * i, 2 * i + 1) in pairs
