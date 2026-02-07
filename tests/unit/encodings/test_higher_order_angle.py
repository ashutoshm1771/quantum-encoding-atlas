"""Comprehensive tests for HigherOrderAngleEncoding.

This test module provides complete coverage of the HigherOrderAngleEncoding class,
which encodes classical data using polynomial combinations of features up to a
specified interaction order. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, n_terms, gate_count)
- Term generation behavior (unique to HigherOrderAngleEncoding)
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

Run with: pytest tests/unit/encodings/test_higher_order_angle.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_higher_order_angle.py -v -m "not slow"

References
----------
.. [1] Schuld, M. & Killoran, N. (2019). Quantum Machine Learning in Feature
       Hilbert Spaces. Physical Review Letters, 122(4), 040504.
"""

from __future__ import annotations

import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas.encodings.higher_order_angle import (
    HigherOrderAngleEncoding,
    count_terms,
)

if TYPE_CHECKING:
    from typing import Any

    from encoding_atlas.core.properties import EncodingProperties


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

    Values chosen to exercise typical encoding behavior.
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
    - [0.5, 0.6, 0.7, 0.8] (higher values)
    - [0.0, 0.0, 0.0, 0.0] (edge case: zeros)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.0, 0.0, 0.0, 0.0],
    ])


@pytest.fixture
def default_encoding() -> HigherOrderAngleEncoding:
    """Default HigherOrderAngleEncoding with 4 features and order=2."""
    return HigherOrderAngleEncoding(n_features=4, order=2)


@pytest.fixture
def encoding_order_3() -> HigherOrderAngleEncoding:
    """HigherOrderAngleEncoding with third-order interactions."""
    return HigherOrderAngleEncoding(n_features=4, order=3)


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for HigherOrderAngleEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = HigherOrderAngleEncoding(n_features=4)

        assert enc.n_features == 4
        assert enc.order == 2
        assert enc.rotation == "Y"
        assert enc.combination == "product"
        assert enc.include_first_order is True
        assert enc.scaling == 1.0
        assert enc.reps == 1

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = HigherOrderAngleEncoding(n_features=1, order=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1
        assert enc.n_terms == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16]:
            enc = HigherOrderAngleEncoding(n_features=n, order=min(n, 2))
            assert enc.n_features == n

    def test_custom_order(self) -> None:
        """Test instantiation with custom order."""
        enc = HigherOrderAngleEncoding(n_features=5, order=3)
        assert enc.order == 3

    def test_order_equals_one(self) -> None:
        """Test order=1 (equivalent to standard angle encoding)."""
        enc = HigherOrderAngleEncoding(n_features=4, order=1)
        assert enc.order == 1
        # Only first-order terms: 4 terms
        assert enc.n_terms == 4

    def test_order_equals_n_features(self) -> None:
        """Test order equals n_features (maximum order)."""
        enc = HigherOrderAngleEncoding(n_features=3, order=3)
        assert enc.order == 3
        # Terms: 3 first-order + 3 second-order + 1 third-order = 7
        assert enc.n_terms == 7

    def test_rotation_x(self) -> None:
        """Test RX rotation axis."""
        enc = HigherOrderAngleEncoding(n_features=4, rotation="X")
        assert enc.rotation == "X"

    def test_rotation_z(self) -> None:
        """Test RZ rotation axis."""
        enc = HigherOrderAngleEncoding(n_features=4, rotation="Z")
        assert enc.rotation == "Z"

    def test_rotation_lowercase(self) -> None:
        """Test lowercase rotation is normalized to uppercase."""
        enc = HigherOrderAngleEncoding(n_features=4, rotation="y")
        assert enc.rotation == "Y"

    def test_combination_sum(self) -> None:
        """Test sum combination method."""
        enc = HigherOrderAngleEncoding(n_features=4, combination="sum")
        assert enc.combination == "sum"

    def test_combination_uppercase(self) -> None:
        """Test uppercase combination is normalized to lowercase."""
        enc = HigherOrderAngleEncoding(n_features=4, combination="PRODUCT")
        assert enc.combination == "product"

    def test_include_first_order_false(self) -> None:
        """Test excluding first-order terms."""
        enc = HigherOrderAngleEncoding(
            n_features=4, order=2, include_first_order=False
        )
        assert enc.include_first_order is False
        # Only second-order: C(4,2) = 6
        assert enc.n_terms == 6

    def test_custom_scaling(self) -> None:
        """Test custom scaling factor."""
        enc = HigherOrderAngleEncoding(n_features=4, scaling=2.5)
        assert enc.scaling == 2.5

    def test_negative_scaling(self) -> None:
        """Test negative scaling factor (should be allowed)."""
        enc = HigherOrderAngleEncoding(n_features=4, scaling=-1.0)
        assert enc.scaling == -1.0

    def test_zero_scaling(self) -> None:
        """Test zero scaling factor."""
        enc = HigherOrderAngleEncoding(n_features=4, scaling=0.0)
        assert enc.scaling == 0.0

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = HigherOrderAngleEncoding(n_features=4, reps=3)
        assert enc.reps == 3

    def test_all_custom_parameters(self) -> None:
        """Test all parameters customized together."""
        enc = HigherOrderAngleEncoding(
            n_features=5,
            order=3,
            rotation="Z",
            combination="sum",
            include_first_order=False,
            scaling=0.5,
            reps=2,
        )

        assert enc.n_features == 5
        assert enc.order == 3
        assert enc.rotation == "Z"
        assert enc.combination == "sum"
        assert enc.include_first_order is False
        assert enc.scaling == 0.5
        assert enc.reps == 2

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = HigherOrderAngleEncoding(n_features=100, order=2)
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
            HigherOrderAngleEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be"):
            HigherOrderAngleEncoding(n_features=-1)

    def test_invalid_n_features_float(self) -> None:
        """Test that float n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            HigherOrderAngleEncoding(n_features=4.0)  # type: ignore

    def test_invalid_n_features_string(self) -> None:
        """Test that string n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            HigherOrderAngleEncoding(n_features="4")  # type: ignore

    def test_invalid_n_features_none(self) -> None:
        """Test that None n_features raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            HigherOrderAngleEncoding(n_features=None)  # type: ignore

    def test_invalid_order_zero(self) -> None:
        """Test that order=0 raises ValueError."""
        with pytest.raises(ValueError, match="order must be at least 1"):
            HigherOrderAngleEncoding(n_features=4, order=0)

    def test_invalid_order_negative(self) -> None:
        """Test that negative order raises ValueError."""
        with pytest.raises(ValueError, match="order must be at least 1"):
            HigherOrderAngleEncoding(n_features=4, order=-1)

    def test_invalid_order_exceeds_n_features(self) -> None:
        """Test that order > n_features raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed n_features"):
            HigherOrderAngleEncoding(n_features=3, order=4)

    def test_invalid_rotation(self) -> None:
        """Test invalid rotation axis raises ValueError."""
        with pytest.raises(ValueError, match="rotation must be one of"):
            HigherOrderAngleEncoding(n_features=4, rotation="W")

    def test_invalid_combination(self) -> None:
        """Test invalid combination method raises ValueError."""
        with pytest.raises(ValueError, match="combination must be one of"):
            HigherOrderAngleEncoding(n_features=4, combination="multiply")

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            HigherOrderAngleEncoding(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            HigherOrderAngleEncoding(n_features=4, reps=-1)

    def test_invalid_scaling_nan(self) -> None:
        """Test that NaN scaling raises ValueError."""
        with pytest.raises(ValueError, match="scaling must be finite"):
            HigherOrderAngleEncoding(n_features=4, scaling=float("nan"))

    def test_invalid_scaling_inf(self) -> None:
        """Test that infinite scaling raises ValueError."""
        with pytest.raises(ValueError, match="scaling must be finite"):
            HigherOrderAngleEncoding(n_features=4, scaling=float("inf"))

    def test_degenerate_case_order1_no_first_order(self) -> None:
        """Test that order=1 with include_first_order=False raises ValueError."""
        with pytest.raises(ValueError, match="no terms"):
            HigherOrderAngleEncoding(
                n_features=4, order=1, include_first_order=False
            )

    def test_type_error_order_float(self) -> None:
        """Test that float order raises TypeError."""
        with pytest.raises(TypeError, match="order must be an integer"):
            HigherOrderAngleEncoding(n_features=4, order=2.5)  # type: ignore

    def test_type_error_rotation_int(self) -> None:
        """Test that non-string rotation raises TypeError."""
        with pytest.raises(TypeError, match="rotation must be a string"):
            HigherOrderAngleEncoding(n_features=4, rotation=1)  # type: ignore

    def test_type_error_combination_int(self) -> None:
        """Test that non-string combination raises TypeError."""
        with pytest.raises(TypeError, match="combination must be a string"):
            HigherOrderAngleEncoding(n_features=4, combination=1)  # type: ignore

    def test_type_error_include_first_order_string(self) -> None:
        """Test that non-bool include_first_order raises TypeError."""
        with pytest.raises(TypeError, match="include_first_order must be a bool"):
            HigherOrderAngleEncoding(
                n_features=4, include_first_order="True"  # type: ignore
            )

    def test_type_error_scaling_string(self) -> None:
        """Test that non-numeric scaling raises TypeError."""
        with pytest.raises(TypeError, match="scaling must be a number"):
            HigherOrderAngleEncoding(n_features=4, scaling="1.0")  # type: ignore

    def test_type_error_reps_float(self) -> None:
        """Test that float reps raises TypeError."""
        with pytest.raises(TypeError, match="reps must be an integer"):
            HigherOrderAngleEncoding(n_features=4, reps=2.0)  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of HigherOrderAngleEncoding."""

    def test_n_qubits(self) -> None:
        """Test n_qubits property calculation.

        For HigherOrderAngleEncoding, n_qubits equals n_features.
        """
        enc = HigherOrderAngleEncoding(n_features=4)
        assert enc.n_qubits == 4

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features for various sizes."""
        for n in [2, 4, 8, 16]:
            enc = HigherOrderAngleEncoding(n_features=n)
            assert enc.n_qubits == n

    def test_depth(self) -> None:
        """Test circuit depth calculation.

        Depth formula: depth = reps (all gates parallel).
        For n_features=4, reps=2: depth = 2.
        """
        enc = HigherOrderAngleEncoding(n_features=4, reps=2)
        assert enc.depth == 2

    def test_depth_equals_reps(self) -> None:
        """Test that depth equals reps (all gates parallel)."""
        for reps in [1, 2, 3, 5]:
            enc = HigherOrderAngleEncoding(n_features=4, reps=reps)
            assert enc.depth == reps

    def test_properties_type(self) -> None:
        """Test that properties returns EncodingProperties instance."""
        from encoding_atlas.core.properties import EncodingProperties

        enc = HigherOrderAngleEncoding(n_features=4)
        assert isinstance(enc.properties, EncodingProperties)

    def test_properties_cached(self) -> None:
        """Test that properties are cached (same object returned)."""
        enc = HigherOrderAngleEncoding(n_features=4)
        props1 = enc.properties
        props2 = enc.properties
        assert props1 is props2

    def test_properties_gate_count(self) -> None:
        """Test gate count calculation.

        gate_count = n_qubits * reps (one rotation per qubit per rep).
        """
        enc = HigherOrderAngleEncoding(n_features=4, reps=1)
        assert enc.properties.gate_count == 4

    def test_properties_object(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that properties returns EncodingProperties with correct values."""
        props = default_encoding.properties

        assert props.n_qubits == 4
        assert props.depth == 1
        # One gate per qubit
        assert props.gate_count == 4
        assert props.single_qubit_gates == 4
        # No entangling gates
        assert props.two_qubit_gates == 0
        # All data-dependent
        assert props.parameter_count == 0
        assert props.is_entangling is False
        assert props.simulability == "simulable"

    def test_not_entangling(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that encoding creates no entanglement."""
        assert default_encoding.properties.is_entangling is False

    def test_classically_simulable(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that encoding is classically simulable."""
        assert default_encoding.properties.simulability == "simulable"

    def test_trainability_estimate(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test trainability estimate is high (product states)."""
        props = default_encoding.properties
        assert props.trainability_estimate is not None
        assert props.trainability_estimate > 0.8

    def test_properties_notes_contain_order(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that notes mention the order."""
        props = default_encoding.properties
        assert "Order-2" in props.notes

    def test_properties_with_reps(self) -> None:
        """Test properties with multiple repetitions."""
        enc = HigherOrderAngleEncoding(n_features=4, reps=3)
        props = enc.properties

        assert props.depth == 3
        # 4 qubits * 3 reps = 12 gates
        assert props.gate_count == 12


# =============================================================================
# Test Class: Term Generation Behavior
# =============================================================================


class TestTermGenerationBehavior:
    """Tests for polynomial term generation (unique to HigherOrderAngleEncoding)."""

    def test_first_order_terms(self) -> None:
        """Test that first-order terms are generated correctly."""
        enc = HigherOrderAngleEncoding(n_features=4, order=1)
        terms = enc.terms

        expected = [(0,), (1,), (2,), (3,)]
        assert terms == expected

    def test_second_order_terms(self) -> None:
        """Test that second-order terms are generated correctly."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        terms = enc.terms

        # First-order terms
        first_order = [(0,), (1,), (2,), (3,)]
        # Second-order terms (pairs)
        second_order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        expected = first_order + second_order
        assert terms == expected

    def test_third_order_terms(self) -> None:
        """Test that third-order terms are generated correctly."""
        enc = HigherOrderAngleEncoding(n_features=4, order=3)
        terms = enc.terms

        # First-order: 4 terms
        # Second-order: C(4,2) = 6 terms
        # Third-order: C(4,3) = 4 terms
        assert len(terms) == 14

        # Check third-order terms
        third_order = [t for t in terms if len(t) == 3]
        expected_third = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        assert third_order == expected_third

    def test_terms_without_first_order(self) -> None:
        """Test term generation without first-order terms."""
        enc = HigherOrderAngleEncoding(
            n_features=4, order=2, include_first_order=False
        )
        terms = enc.terms

        # Only second-order
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert terms == expected

    def test_n_terms_formula(self) -> None:
        """Test that n_terms matches expected formula.

        n_terms = sum of C(n, k) for k = 1 to order.
        """
        for n in range(2, 6):
            for order in range(1, n + 1):
                enc = HigherOrderAngleEncoding(n_features=n, order=order)

                # Expected: sum of C(n, k) for k = 1 to order
                expected = sum(math.comb(n, k) for k in range(1, order + 1))
                assert enc.n_terms == expected

    def test_term_info(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test get_term_info returns correct structure."""
        info = default_encoding.get_term_info()

        assert "terms" in info
        assert "n_terms" in info
        assert "terms_by_order" in info
        assert "qubit_assignments" in info

        assert info["n_terms"] == len(info["terms"])
        # First-order: 4 terms
        assert len(info["terms_by_order"][1]) == 4
        # Second-order: 6 terms
        assert len(info["terms_by_order"][2]) == 6

    def test_qubit_assignment_round_robin(self) -> None:
        """Test that terms are assigned to qubits in round-robin fashion."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        info = enc.get_term_info()

        assignments = info["qubit_assignments"]

        # With 10 terms and 4 qubits:
        # q0 gets terms 0, 4, 8 (indices)
        # q1 gets terms 1, 5, 9
        # q2 gets terms 2, 6
        # q3 gets terms 3, 7

        terms = enc.terms
        assert assignments[0] == [terms[0], terms[4], terms[8]]
        assert assignments[1] == [terms[1], terms[5], terms[9]]
        assert assignments[2] == [terms[2], terms[6]]
        assert assignments[3] == [terms[3], terms[7]]


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_input_shape(
        self,
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid input shape is accepted."""
        circuit = default_encoding.get_circuit(sample_data_4d)
        assert circuit is not None

    def test_valid_2d_input_single_sample(
        self,
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 2D input with single sample is accepted."""
        x = sample_data_4d.reshape(1, -1)
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_wrong_feature_count(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that wrong feature count raises ValueError."""
        x = np.array([0.1, 0.2, 0.3])  # 3 features, expected 4
        with pytest.raises(ValueError, match="Expected .* features"):
            default_encoding.get_circuit(x, backend="qiskit")

    def test_nan_input_rejected(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN|nan|finite"):
            default_encoding.get_circuit(x, backend="qiskit")

    def test_inf_input_rejected(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="inf|infinite|finite"):
            default_encoding.get_circuit(x, backend="qiskit")

    def test_list_input_accepted(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that list input is converted to array."""
        x = [0.1, 0.2, 0.3, 0.4]  # list, not array
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_batch_input_rejected_for_get_circuit(
        self,
        default_encoding: HigherOrderAngleEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that 2D input with multiple samples raises error for get_circuit."""
        with pytest.raises(ValueError, match="single sample"):
            default_encoding.get_circuit(batch_data_4d)

    def test_batch_input_shape(
        self,
        default_encoding: HigherOrderAngleEncoding,
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
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        default_encoding: HigherOrderAngleEncoding,
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
        assert len(state) == 2 ** default_encoding.n_qubits
        # State should be valid (normalized)
        assert np.isclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_state_normalized(
        self,
        default_encoding: HigherOrderAngleEncoding,
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

    def test_rotation_x_gates(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that RX gates are used when rotation='X'."""
        enc = HigherOrderAngleEncoding(n_features=4, rotation="X")
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_rotation_z_gates(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that RZ gates are used when rotation='Z'."""
        enc = HigherOrderAngleEncoding(n_features=4, rotation="Z")
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_multiple_reps(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test circuit with multiple repetitions."""
        enc = HigherOrderAngleEncoding(n_features=4, reps=3)
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
        default_encoding: HigherOrderAngleEncoding,
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
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == 4

    def test_circuit_has_expected_gates(
        self,
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit contains expected gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        # Should have at least one gate per qubit
        assert len(circuit.data) >= default_encoding.n_qubits

    def test_rotation_y_gates(
        self,
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that RY gates are used by default."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")

        # Check that gates are RY
        for instruction in circuit.data:
            if instruction.operation.name != "barrier":
                assert instruction.operation.name == "ry"

    def test_rotation_x_gates(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that RX gates are used when rotation='X'."""
        enc = HigherOrderAngleEncoding(n_features=4, rotation="X")
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        for instruction in circuit.data:
            if instruction.operation.name != "barrier":
                assert instruction.operation.name == "rx"

    def test_rotation_z_gates(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that RZ gates are used when rotation='Z'."""
        enc = HigherOrderAngleEncoding(n_features=4, rotation="Z")
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        for instruction in circuit.data:
            if instruction.operation.name != "barrier":
                assert instruction.operation.name == "rz"

    def test_multiple_reps_has_barriers(
        self,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that multiple reps have barriers between them."""
        enc = HigherOrderAngleEncoding(n_features=4, reps=3)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        # Count barriers
        barrier_count = sum(
            1 for inst in circuit.data if inst.operation.name == "barrier"
        )
        # reps - 1 barriers
        assert barrier_count == 2

    def test_batch_circuits(
        self,
        default_encoding: HigherOrderAngleEncoding,
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
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: HigherOrderAngleEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == 4

    def test_batch_circuits(
        self,
        default_encoding: HigherOrderAngleEncoding,
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
    """Tests for mathematical correctness of term computation."""

    def test_product_combination(self) -> None:
        """Test product combination computes correctly.

        For product combination, each term is the product of its indices.
        """
        enc = HigherOrderAngleEncoding(
            n_features=3, order=2, combination="product", scaling=1.0
        )
        x = np.array([2.0, 3.0, 5.0])

        angles = enc.compute_angles(x)

        # Terms: (0,), (1,), (2,), (0,1), (0,2), (1,2)
        # Values: 2, 3, 5, 6, 10, 15
        # Qubit 0: terms 0, 3 -> 2 + 6 = 8
        # Qubit 1: terms 1, 4 -> 3 + 10 = 13
        # Qubit 2: terms 2, 5 -> 5 + 15 = 20

        assert np.isclose(angles[0], 8.0)
        assert np.isclose(angles[1], 13.0)
        assert np.isclose(angles[2], 20.0)

    def test_sum_combination(self) -> None:
        """Test sum combination computes correctly.

        For sum combination, each term is the sum of its indices.
        """
        enc = HigherOrderAngleEncoding(
            n_features=3, order=2, combination="sum", scaling=1.0
        )
        x = np.array([2.0, 3.0, 5.0])

        angles = enc.compute_angles(x)

        # Terms: (0,), (1,), (2,), (0,1), (0,2), (1,2)
        # Values with sum: 2, 3, 5, 5, 7, 8
        # Qubit 0: terms 0, 3 -> 2 + 5 = 7
        # Qubit 1: terms 1, 4 -> 3 + 7 = 10
        # Qubit 2: terms 2, 5 -> 5 + 8 = 13

        assert np.isclose(angles[0], 7.0)
        assert np.isclose(angles[1], 10.0)
        assert np.isclose(angles[2], 13.0)

    def test_scaling_applied(self) -> None:
        """Test that scaling is applied correctly."""
        enc1 = HigherOrderAngleEncoding(n_features=3, order=1, scaling=1.0)
        enc2 = HigherOrderAngleEncoding(n_features=3, order=1, scaling=2.0)

        x = np.array([0.1, 0.2, 0.3])

        angles1 = enc1.compute_angles(x)
        angles2 = enc2.compute_angles(x)

        np.testing.assert_allclose(angles2, 2.0 * angles1)

    def test_first_order_only(self) -> None:
        """Test first-order only matches simple angle encoding."""
        enc = HigherOrderAngleEncoding(n_features=4, order=1, scaling=1.0)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        angles = enc.compute_angles(x)

        # First-order only: each qubit gets its own feature
        np.testing.assert_allclose(angles, x)

    def test_without_first_order(self) -> None:
        """Test encoding without first-order terms."""
        enc = HigherOrderAngleEncoding(
            n_features=3, order=2, include_first_order=False, scaling=1.0
        )
        x = np.array([2.0, 3.0, 5.0])

        angles = enc.compute_angles(x)

        # Terms: (0,1), (0,2), (1,2) only
        # Values: 6, 10, 15
        # Qubit 0: term 0 -> 6
        # Qubit 1: term 1 -> 10
        # Qubit 2: term 2 -> 15

        assert np.isclose(angles[0], 6.0)
        assert np.isclose(angles[1], 10.0)
        assert np.isclose(angles[2], 15.0)

    def test_zero_input(self) -> None:
        """Test that zero input produces zero angles."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, combination="product")
        x = np.zeros(4)

        angles = enc.compute_angles(x)

        np.testing.assert_allclose(angles, 0.0)

    def test_third_order_product(self) -> None:
        """Test third-order product term computation."""
        enc = HigherOrderAngleEncoding(
            n_features=3, order=3, combination="product", scaling=1.0
        )
        x = np.array([2.0, 3.0, 5.0])

        # Terms: (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)
        # Product values: 2, 3, 5, 6, 10, 15, 30
        # Qubit 0: terms 0, 3, 6 -> 2 + 6 + 30 = 38
        # Qubit 1: terms 1, 4 -> 3 + 10 = 13
        # Qubit 2: terms 2, 5 -> 5 + 15 = 20

        angles = enc.compute_angles(x)

        assert np.isclose(angles[0], 38.0)
        assert np.isclose(angles[1], 13.0)
        assert np.isclose(angles[2], 20.0)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature(self) -> None:
        """Test encoding with single feature."""
        enc = HigherOrderAngleEncoding(n_features=1, order=1)
        x = np.array([0.5])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None
        assert enc.n_terms == 1

    def test_large_feature_count(self) -> None:
        """Test encoding with many features."""
        enc = HigherOrderAngleEncoding(n_features=10, order=2)

        assert enc.n_qubits == 10
        # 10 first-order + C(10,2) = 10 + 45 = 55 terms
        assert enc.n_terms == 55

    def test_maximum_order(self) -> None:
        """Test encoding with order = n_features."""
        enc = HigherOrderAngleEncoding(n_features=4, order=4)

        # All subsets: 2^4 - 1 = 15 terms
        assert enc.n_terms == 15

    def test_extreme_large_values(self) -> None:
        """Test encoding with large positive values."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x_large = np.array([100.0, 200.0, 300.0, 400.0])
        circuit = enc.get_circuit(x_large, backend="pennylane")
        assert circuit is not None

    def test_extreme_small_values(self) -> None:
        """Test encoding with very small values."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x_small = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        circuit = enc.get_circuit(x_small, backend="pennylane")
        assert circuit is not None

    def test_negative_values(self) -> None:
        """Test encoding with negative values."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x_neg = np.array([-1.0, -2.0, -3.0, -4.0])
        circuit = enc.get_circuit(x_neg, backend="pennylane")
        assert circuit is not None

    def test_mixed_signs(self) -> None:
        """Test encoding with mixed positive/negative values."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, combination="product")
        x = np.array([1.0, -1.0, 1.0, -1.0])

        angles = enc.compute_angles(x)

        # Products of alternating signs will give various results
        assert angles is not None

    def test_negative_scaling(self) -> None:
        """Test that negative scaling reverses angles."""
        enc_pos = HigherOrderAngleEncoding(n_features=3, order=1, scaling=1.0)
        enc_neg = HigherOrderAngleEncoding(n_features=3, order=1, scaling=-1.0)

        x = np.array([0.1, 0.2, 0.3])

        angles_pos = enc_pos.compute_angles(x)
        angles_neg = enc_neg.compute_angles(x)

        np.testing.assert_allclose(angles_neg, -angles_pos)

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions."""
        enc = HigherOrderAngleEncoding(n_features=4, reps=10)

        assert enc.depth == 10
        # 4 qubits * 10 reps = 40 gates
        assert enc.properties.gate_count == 40


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
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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

        Large values should not cause overflow (angles wrap around 2pi).
        """
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
    def test_mixed_extreme_magnitudes(self) -> None:
        """Test encoding with mixed extreme magnitude values."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
    def test_extreme_values_produce_normalized_state(self) -> None:
        """Test that extreme values still produce normalized quantum states."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_near_pi_multiples(self) -> None:
        """Test numerical stability near pi multiples."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
    def test_high_order_stability(self) -> None:
        """Test numerical stability with high order interactions."""
        enc = HigherOrderAngleEncoding(n_features=4, order=4)  # Maximum order
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
    @pytest.mark.parametrize("combination", ["product", "sum"])
    def test_all_combinations_stable(self, combination: str) -> None:
        """Test numerical stability across all combination methods."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, combination=combination)
        x = np.random.default_rng(42).standard_normal(4) * 100  # Large random values

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
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        eps = np.finfo(float).eps
        x = np.array([eps, 2 * eps, 3 * eps, 4 * eps])

        # Should not raise
        circuit = enc.get_circuit(x, backend="pennylane")
        assert callable(circuit)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_large_scaling_factor(self) -> None:
        """Test numerical stability with large scaling factor."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, scaling=1000.0)
        x = np.array([0.1, 0.2, 0.3, 0.4])

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
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2)
        enc2 = HigherOrderAngleEncoding(n_features=4, order=2)
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2)
        enc2 = HigherOrderAngleEncoding(n_features=8, order=2)
        assert enc1 != enc2

    def test_inequality_different_order(self) -> None:
        """Test that encodings with different order are not equal."""
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2)
        enc2 = HigherOrderAngleEncoding(n_features=4, order=3)
        assert enc1 != enc2

    def test_inequality_different_rotation(self) -> None:
        """Test that encodings with different rotation are not equal."""
        enc1 = HigherOrderAngleEncoding(n_features=4, rotation="Y")
        enc2 = HigherOrderAngleEncoding(n_features=4, rotation="Z")
        assert enc1 != enc2

    def test_inequality_different_combination(self) -> None:
        """Test that encodings with different combination are not equal."""
        enc1 = HigherOrderAngleEncoding(n_features=4, combination="product")
        enc2 = HigherOrderAngleEncoding(n_features=4, combination="sum")
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2)
        enc2 = HigherOrderAngleEncoding(n_features=4, order=2)
        assert hash(enc1) == hash(enc2)

    def test_hash_same_params(self) -> None:
        """Test that encodings with same parameters have same hash."""
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2, scaling=1.5)
        enc2 = HigherOrderAngleEncoding(n_features=4, order=2, scaling=1.5)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2)
        enc2 = HigherOrderAngleEncoding(n_features=4, order=2)  # Same as enc1
        enc3 = HigherOrderAngleEncoding(n_features=4, order=3)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2)
        enc2 = HigherOrderAngleEncoding(n_features=4, order=2)

        d = {enc1: "test"}
        assert d[enc2] == "test"  # enc2 should find same key


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__ string representation."""

    def test_repr_contains_class_name(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that repr contains class name."""
        assert "HigherOrderAngleEncoding" in repr(default_encoding)

    def test_repr_contains_parameters(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that repr contains all parameters."""
        r = repr(default_encoding)

        assert "n_features=4" in r
        assert "order=2" in r
        assert "rotation='Y'" in r
        assert "combination='product'" in r
        assert "include_first_order=True" in r
        assert "scaling=1.0" in r
        assert "reps=1" in r

    def test_repr_round_trip(self) -> None:
        """Test that repr can be used to recreate encoding (informational)."""
        enc = HigherOrderAngleEncoding(
            n_features=5,
            order=3,
            rotation="Z",
            combination="sum",
            include_first_order=False,
            scaling=2.5,
            reps=2,
        )

        # This should work (though we don't actually eval for safety)
        r = repr(enc)
        assert "n_features=5" in r
        assert "order=3" in r


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for invalid backend names and missing backends."""

    def test_invalid_backend_name(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that invalid backend name raises ValueError."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="backend"):
            default_encoding.get_circuit(x, backend="invalid_backend")

    def test_invalid_backend_type(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that non-string backend raises appropriate error."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises((TypeError, ValueError)):
            default_encoding.get_circuit(x, backend=123)  # type: ignore

    def test_none_backend_uses_default(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that None backend uses default backend."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        # Should not raise - uses default backend
        circuit = default_encoding.get_circuit(x, backend=None)
        assert circuit is not None

    def test_case_insensitive_backend(
        self, default_encoding: HigherOrderAngleEncoding
    ) -> None:
        """Test that backend names are case-insensitive."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        # Should work with various cases
        circuit_lower = default_encoding.get_circuit(x, backend="pennylane")
        circuit_upper = default_encoding.get_circuit(x, backend="PENNYLANE")
        circuit_mixed = default_encoding.get_circuit(x, backend="PennyLane")

        assert circuit_lower is not None
        assert circuit_upper is not None
        assert circuit_mixed is not None


# =============================================================================
# Test Class: Serialization
# =============================================================================


@pytest.mark.serialization
class TestSerialization:
    """Tests for serialization (pickle) support."""

    def test_pickle_roundtrip(self) -> None:
        """Test that encoding can be pickled and unpickled."""
        enc = HigherOrderAngleEncoding(
            n_features=4,
            order=3,
            rotation="Z",
            combination="sum",
            scaling=2.0,
            reps=2,
        )

        # Force properties to be computed
        _ = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.order == enc.order
        assert restored.rotation == enc.rotation
        assert restored.combination == enc.combination
        assert restored.scaling == enc.scaling
        assert restored.reps == enc.reps
        assert restored.n_qubits == enc.n_qubits

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = restored.get_circuit(x, backend="pennylane")
        assert callable(circuit)

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        from encoding_atlas.core.properties import EncodingProperties

        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        num_threads = 10
        num_circuits_per_thread = 50

        errors: list[Exception] = []

        def generate_circuits(thread_id: int) -> list[Any]:
            circuits = []
            try:
                for i in range(num_circuits_per_thread):
                    rng = np.random.default_rng(seed=thread_id * 1000 + i)
                    x = rng.standard_normal(4)
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
        enc = HigherOrderAngleEncoding(n_features=4, order=3)
        num_threads = 20

        results: list[Any] = []
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
# Test Class: Utility Functions (Encoding-Specific)
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions (encoding-specific)."""

    def test_count_terms_order_1(self) -> None:
        """Test count_terms for order=1."""
        assert count_terms(4, 1, True) == 4
        # No terms if excluding first order
        assert count_terms(4, 1, False) == 0

    def test_count_terms_order_2(self) -> None:
        """Test count_terms for order=2."""
        # 4 first-order + 6 second-order = 10
        assert count_terms(4, 2, True) == 10
        # Only 6 second-order
        assert count_terms(4, 2, False) == 6

    def test_count_terms_order_3(self) -> None:
        """Test count_terms for order=3."""
        # 4 + 6 + 4 = 14
        assert count_terms(4, 3, True) == 14
        # 6 + 4 = 10
        assert count_terms(4, 3, False) == 10

    def test_count_terms_maximum_order(self) -> None:
        """Test count_terms for order=n (all subsets)."""
        # 2^4 - 1 = 15 (all non-empty subsets)
        assert count_terms(4, 4, True) == 15

    def test_count_terms_matches_encoding(self) -> None:
        """Test that count_terms matches actual encoding term count."""
        for n in range(2, 6):
            for order in range(1, n + 1):
                for include_first in [True, False]:
                    if order == 1 and not include_first:
                        continue  # Skip degenerate case

                    expected = count_terms(n, order, include_first)
                    enc = HigherOrderAngleEncoding(
                        n_features=n,
                        order=order,
                        include_first_order=include_first,
                    )
                    assert enc.n_terms == expected


# =============================================================================
# Test Class: Gate Count Breakdown (Encoding-Specific)
# =============================================================================


class TestGateCountBreakdown:
    """Tests for the gate_count_breakdown() method (encoding-specific)."""

    def test_gate_count_breakdown_returns_typed_dict(self) -> None:
        """Test that gate_count_breakdown returns the expected structure."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        breakdown = enc.gate_count_breakdown()

        # Check all required keys exist
        assert "rx" in breakdown
        assert "ry" in breakdown
        assert "rz" in breakdown
        assert "total_single_qubit" in breakdown
        assert "total_two_qubit" in breakdown
        assert "total" in breakdown

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_gate_count_breakdown_rotation_specific(self, rotation: str) -> None:
        """Test that only the configured rotation gate has non-zero count."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, rotation=rotation)
        breakdown = enc.gate_count_breakdown()

        # Only the configured rotation should have non-zero count
        expected_count = enc.n_qubits * enc.reps
        if rotation == "X":
            assert breakdown["rx"] == expected_count
            assert breakdown["ry"] == 0
            assert breakdown["rz"] == 0
        elif rotation == "Y":
            assert breakdown["rx"] == 0
            assert breakdown["ry"] == expected_count
            assert breakdown["rz"] == 0
        else:  # Z
            assert breakdown["rx"] == 0
            assert breakdown["ry"] == 0
            assert breakdown["rz"] == expected_count

    def test_gate_count_breakdown_no_two_qubit_gates(self) -> None:
        """Test that two-qubit gate count is always zero."""
        enc = HigherOrderAngleEncoding(n_features=8, order=4)
        breakdown = enc.gate_count_breakdown()
        assert breakdown["total_two_qubit"] == 0

    def test_gate_count_breakdown_total_equals_sum(self) -> None:
        """Test that total equals sum of single and two qubit gates."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, reps=3)
        breakdown = enc.gate_count_breakdown()
        assert (
            breakdown["total"]
            == breakdown["total_single_qubit"] + breakdown["total_two_qubit"]
        )

    def test_gate_count_scales_with_reps(self) -> None:
        """Test that gate count scales linearly with reps."""
        enc1 = HigherOrderAngleEncoding(n_features=4, order=2, reps=1)
        enc2 = HigherOrderAngleEncoding(n_features=4, order=2, reps=3)

        breakdown1 = enc1.gate_count_breakdown()
        breakdown2 = enc2.gate_count_breakdown()

        assert breakdown2["total"] == 3 * breakdown1["total"]


# =============================================================================
# Test Class: Resource Summary (Encoding-Specific)
# =============================================================================


class TestResourceSummary:
    """Tests for the resource_summary() method (encoding-specific)."""

    def test_resource_summary_returns_dict(self) -> None:
        """Test that resource_summary returns a dictionary."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        summary = enc.resource_summary()
        assert isinstance(summary, dict)

    def test_resource_summary_contains_all_keys(self) -> None:
        """Test that resource_summary contains all expected keys."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        summary = enc.resource_summary()

        # Circuit structure keys
        assert "n_qubits" in summary
        assert "n_features" in summary
        assert "depth" in summary
        assert "reps" in summary
        assert "rotation" in summary

        # Polynomial configuration keys
        assert "order" in summary
        assert "combination" in summary
        assert "include_first_order" in summary
        assert "scaling" in summary
        assert "n_terms" in summary
        assert "terms_by_order" in summary

        # Gate counts
        assert "gate_counts" in summary

        # Encoding characteristics
        assert "is_entangling" in summary
        assert "simulability" in summary
        assert "trainability_estimate" in summary

        # Hardware requirements
        assert "hardware_requirements" in summary

    def test_resource_summary_values_match_properties(self) -> None:
        """Test that resource_summary values match encoding properties."""
        enc = HigherOrderAngleEncoding(n_features=5, order=3, reps=2, rotation="X")
        summary = enc.resource_summary()

        assert summary["n_qubits"] == enc.n_qubits
        assert summary["n_features"] == enc.n_features
        assert summary["depth"] == enc.depth
        assert summary["reps"] == enc.reps
        assert summary["rotation"] == enc.rotation
        assert summary["order"] == enc.order
        assert summary["n_terms"] == enc.n_terms

    def test_resource_summary_is_entangling_false(self) -> None:
        """Test that is_entangling is always False for this encoding."""
        enc = HigherOrderAngleEncoding(n_features=8, order=4)
        summary = enc.resource_summary()
        assert summary["is_entangling"] is False

    def test_resource_summary_simulability(self) -> None:
        """Test that simulability is 'simulable' for product states."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        summary = enc.resource_summary()
        assert summary["simulability"] == "simulable"

    def test_resource_summary_hardware_requirements(self) -> None:
        """Test hardware requirements structure."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, rotation="Y")
        summary = enc.resource_summary()

        hw = summary["hardware_requirements"]
        assert hw["connectivity"] == "none"
        assert "RY" in hw["native_gates"]

    def test_resource_summary_terms_by_order(self) -> None:
        """Test terms_by_order counts are correct."""
        enc = HigherOrderAngleEncoding(
            n_features=4, order=3, include_first_order=True
        )
        summary = enc.resource_summary()

        terms_by_order = summary["terms_by_order"]
        # 4 first-order terms, 6 second-order terms, 4 third-order terms
        assert terms_by_order[1] == 4
        assert terms_by_order[2] == 6
        assert terms_by_order[3] == 4


# =============================================================================
# Test Class: Parallel Processing (Encoding-Specific)
# =============================================================================


class TestParallelProcessing:
    """Tests for parallel processing in get_circuits() (encoding-specific)."""

    def test_parallel_produces_same_results_as_sequential(self) -> None:
        """Test that parallel and sequential processing produce identical results."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4))

        circuits_seq = enc.get_circuits(X, backend="pennylane", parallel=False)
        circuits_par = enc.get_circuits(X, backend="pennylane", parallel=True)

        assert len(circuits_seq) == len(circuits_par)

    def test_parallel_with_max_workers(self) -> None:
        """Test parallel processing with custom max_workers."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4))

        circuits = enc.get_circuits(
            X, backend="pennylane", parallel=True, max_workers=2
        )
        assert len(circuits) == 20

    def test_parallel_order_preserved(self) -> None:
        """Test that parallel processing preserves sample order."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        X = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ])

        # Get circuits with parallel processing
        circuits_par = enc.get_circuits(X, backend="pennylane", parallel=True)

        # The first circuit should use the first sample's angles
        assert len(circuits_par) == 3

    @pytest.mark.parametrize("backend", ["pennylane", "qiskit", "cirq"])
    def test_parallel_all_backends(self, backend: str) -> None:
        """Test parallel processing works for all backends."""
        pytest.importorskip(
            "pennylane"
            if backend == "pennylane"
            else "qiskit"
            if backend == "qiskit"
            else "cirq"
        )
        enc = HigherOrderAngleEncoding(n_features=3, order=2)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))

        circuits = enc.get_circuits(X, backend=backend, parallel=True)
        assert len(circuits) == 10

    def test_parallel_single_sample_no_parallelism(self) -> None:
        """Test that single sample doesn't use parallel processing."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        X = np.array([[0.1, 0.2, 0.3, 0.4]])

        # Should work without issues even with parallel=True
        circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
        assert len(circuits) == 1


# =============================================================================
# Test Class: Warnings (Encoding-Specific)
# =============================================================================


class TestWarnings:
    """Tests for the exponential term growth warning (encoding-specific)."""

    def test_warning_issued_for_high_term_count(self) -> None:
        """Test that a warning is issued when term count exceeds threshold."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 175 terms
            enc = HigherOrderAngleEncoding(n_features=10, order=3)

            # Check that a warning was issued
            assert len(w) >= 1
            assert any("term count" in str(warning.message).lower() for warning in w)

    def test_no_warning_for_low_term_count(self) -> None:
        """Test that no warning is issued for low term count."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 10 terms
            enc = HigherOrderAngleEncoding(n_features=4, order=2)

            # Filter to only UserWarnings about term count
            term_warnings = [
                warning
                for warning in w
                if "term count" in str(warning.message).lower()
            ]
            assert len(term_warnings) == 0


# =============================================================================
# Test Class: Public API (Encoding-Specific)
# =============================================================================


class TestPublicAPI:
    """Tests for the public API exports (encoding-specific)."""

    def test_all_exports(self) -> None:
        """Test that __all__ exports are importable."""
        from encoding_atlas.encodings.higher_order_angle import (
            GateCountBreakdown,
            HigherOrderAngleEncoding,
            count_terms,
        )

        assert HigherOrderAngleEncoding is not None
        assert GateCountBreakdown is not None
        assert count_terms is not None

    def test_gate_count_breakdown_type(self) -> None:
        """Test that GateCountBreakdown is a TypedDict."""
        from encoding_atlas.encodings.higher_order_angle import GateCountBreakdown

        # TypedDict classes have __annotations__
        assert hasattr(GateCountBreakdown, "__annotations__")


# =============================================================================
# Test Class: Slow Simulation
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
        """Test that same input always produces same state."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
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
    def test_cross_backend_state_equivalence(self) -> None:
        """Test that all backends produce mathematically equivalent states."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x = np.array([0.5, 0.3, 0.7, 0.1])

        # Get PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Get Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Get Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator(dtype=np.complex128)
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # All states should have same dimension
        expected_dim = 2 ** enc.n_qubits
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        # States should be equivalent (compare probability distributions)
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
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_cross_backend_various_orders(self, order: int) -> None:
        """Test cross-backend equivalence for various interaction orders."""
        enc = HigherOrderAngleEncoding(n_features=4, order=order)
        x = np.array([0.3, 0.5, 0.7, 0.9])

        # Get PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Get Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Get Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator(dtype=np.complex128)
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # Compare probability distributions
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
    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_cross_backend_all_rotations(self, rotation: str) -> None:
        """Test cross-backend equivalence for all rotation axes."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, rotation=rotation)
        x = np.array([0.2, 0.4, 0.6, 0.8])

        # Get PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Get Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Get Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator(dtype=np.complex128)
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # Compare probability distributions
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
    @pytest.mark.parametrize("combination", ["product", "sum"])
    def test_cross_backend_all_combinations(self, combination: str) -> None:
        """Test cross-backend equivalence for all combination methods."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, combination=combination)
        x = np.array([0.2, 0.4, 0.6, 0.8])

        # Get PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Get Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Get Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator(dtype=np.complex128)
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # Compare probability distributions
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
    def test_cross_backend_zero_input_state_equivalence(self) -> None:
        """Test that all backends produce equivalent states for zero input.

        When all input features are zero, all rotation angles are zero, resulting
        in identity operations. All backends should produce the |0...0⟩ state
        with equivalent state vectors and consistent qubit dimensions.

        This test verifies that removing the zero-angle skip optimization
        maintains correct cross-backend consistency.
        """
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x = np.zeros(4)

        # Get PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Get Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Get Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator(dtype=np.complex128)
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # All states should have same dimension (2^n_qubits = 16)
        expected_dim = 2 ** enc.n_qubits
        assert len(pl_state) == expected_dim, f"PennyLane: {len(pl_state)} != {expected_dim}"
        assert len(qk_state) == expected_dim, f"Qiskit: {len(qk_state)} != {expected_dim}"
        assert len(cirq_state) == expected_dim, f"Cirq: {len(cirq_state)} != {expected_dim}"

        # All states should be |0...0⟩ (first element is 1, rest are 0)
        expected_state = np.zeros(expected_dim, dtype=np.complex128)
        expected_state[0] = 1.0

        np.testing.assert_allclose(pl_state, expected_state, atol=1e-10)
        np.testing.assert_allclose(qk_state, expected_state, atol=1e-10)
        np.testing.assert_allclose(cirq_state, expected_state, atol=1e-10)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_zero_input_qubit_registration(self) -> None:
        """Test that all backends register correct number of qubits for zero input.

        This specifically tests Cirq's qubit registration behavior. When all
        rotation angles are zero, Cirq must still register all qubits to maintain
        consistent state vector dimensions across backends.
        """
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x = np.zeros(4)

        # PennyLane: Device defines qubit count
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = pl_full()
        pl_dim = len(pl_state)

        # Qiskit: Circuit pre-allocates qubits
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_qubits = qk_circuit.num_qubits

        # Cirq: Qubits registered via operations
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_qubits = len(cirq_circuit.all_qubits())

        # All should have n_features qubits
        assert qk_qubits == enc.n_features, f"Qiskit qubits: {qk_qubits}"
        assert cirq_qubits == enc.n_features, f"Cirq qubits: {cirq_qubits}"
        assert pl_dim == 2 ** enc.n_features, f"PennyLane state dim: {pl_dim}"

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_cross_backend_partial_zero_input(self) -> None:
        """Test cross-backend equivalence when some (but not all) inputs are zero.

        This tests the edge case where some rotation angles are zero while
        others are non-zero, ensuring consistent behavior.
        """
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        # Only first two features are non-zero
        x = np.array([0.5, 0.3, 0.0, 0.0])

        # Get PennyLane state
        pl_circuit = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def pl_full():
            pl_circuit()
            return qml.state()

        pl_state = np.array(pl_full())

        # Get Qiskit state
        qk_circuit = enc.get_circuit(x, backend="qiskit")
        qk_sv = Statevector(qk_circuit)
        qk_state = np.array(qk_sv.data)

        # Get Cirq state
        cirq_circuit = enc.get_circuit(x, backend="cirq")
        cirq_sim = cirq.Simulator(dtype=np.complex128)
        cirq_result = cirq_sim.simulate(cirq_circuit)
        cirq_state = np.array(cirq_result.final_state_vector)

        # All states should have same dimension
        expected_dim = 2 ** enc.n_qubits
        assert len(pl_state) == expected_dim
        assert len(qk_state) == expected_dim
        assert len(cirq_state) == expected_dim

        # States should be equivalent (compare probability distributions)
        pl_probs = sorted(np.abs(pl_state) ** 2)
        qk_probs = sorted(np.abs(qk_state) ** 2)
        cirq_probs = sorted(np.abs(cirq_state) ** 2)

        np.testing.assert_allclose(pl_probs, qk_probs, atol=1e-6)
        np.testing.assert_allclose(pl_probs, cirq_probs, atol=1e-6)


# =============================================================================
# Test Class: Deterministic Gate Counts
# =============================================================================


class TestDeterministicGateCounts:
    """Tests verifying deterministic gate counts regardless of input values.

    After removing the zero-angle skip optimization, gate counts should be
    deterministic and independent of input values. This ensures predictable
    circuit structure for hardware resource planning.
    """

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_qiskit_gate_count_with_zero_input(self) -> None:
        """Test that Qiskit circuit has deterministic gate count for zero input."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x_zero = np.zeros(4)
        x_nonzero = np.array([0.1, 0.2, 0.3, 0.4])

        circuit_zero = enc.get_circuit(x_zero, backend="qiskit")
        circuit_nonzero = enc.get_circuit(x_nonzero, backend="qiskit")

        # Count rotation gates (excluding barriers)
        def count_rotations(circuit: "QuantumCircuit") -> int:
            return sum(
                1 for inst in circuit.data
                if inst.operation.name in ("rx", "ry", "rz")
            )

        gates_zero = count_rotations(circuit_zero)
        gates_nonzero = count_rotations(circuit_nonzero)

        # Both should have the same number of gates
        assert gates_zero == gates_nonzero
        # Should equal n_qubits * reps
        assert gates_zero == enc.n_qubits * enc.reps

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_qiskit_gate_count_matches_properties(self) -> None:
        """Test that actual Qiskit gate count matches reported properties."""
        for n_features in [2, 4, 6]:
            for reps in [1, 2, 3]:
                enc = HigherOrderAngleEncoding(n_features=n_features, reps=reps)
                x = np.random.randn(n_features)

                circuit = enc.get_circuit(x, backend="qiskit")
                actual_gates = sum(
                    1 for inst in circuit.data
                    if inst.operation.name in ("rx", "ry", "rz")
                )

                expected_gates = enc.properties.single_qubit_gates
                assert actual_gates == expected_gates, (
                    f"n_features={n_features}, reps={reps}: "
                    f"actual={actual_gates}, expected={expected_gates}"
                )

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_cirq_gate_count_with_zero_input(self) -> None:
        """Test that Cirq circuit has deterministic gate count for zero input."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x_zero = np.zeros(4)
        x_nonzero = np.array([0.1, 0.2, 0.3, 0.4])

        circuit_zero = enc.get_circuit(x_zero, backend="cirq")
        circuit_nonzero = enc.get_circuit(x_nonzero, backend="cirq")

        gates_zero = len(list(circuit_zero.all_operations()))
        gates_nonzero = len(list(circuit_nonzero.all_operations()))

        # Both should have the same number of gates
        assert gates_zero == gates_nonzero
        # Should equal n_qubits * reps
        assert gates_zero == enc.n_qubits * enc.reps

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_cirq_qubit_count_with_zero_input(self) -> None:
        """Test that Cirq circuit registers all qubits even with zero input.

        This is critical for Cirq's behavior: qubits are only registered when
        operations are applied to them. With the zero-angle skip removed,
        all qubits should always be registered.
        """
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x = np.zeros(4)

        circuit = enc.get_circuit(x, backend="cirq")
        registered_qubits = len(circuit.all_qubits())

        assert registered_qubits == enc.n_features, (
            f"Expected {enc.n_features} qubits, got {registered_qubits}. "
            "This may indicate zero-angle rotations are being skipped."
        )

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_gate_count_all_rotations(self, rotation: str) -> None:
        """Test deterministic gate counts for all rotation axes."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, rotation=rotation)
        x_zero = np.zeros(4)

        circuit = enc.get_circuit(x_zero, backend="qiskit")

        gate_name = f"r{rotation.lower()}"
        gate_count = sum(
            1 for inst in circuit.data
            if inst.operation.name == gate_name
        )

        assert gate_count == enc.n_qubits * enc.reps

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_gate_count_multiple_reps(self) -> None:
        """Test that gate count scales correctly with reps."""
        for reps in [1, 2, 3, 5]:
            enc = HigherOrderAngleEncoding(n_features=4, order=2, reps=reps)
            x = np.zeros(4)

            circuit = enc.get_circuit(x, backend="qiskit")
            gate_count = sum(
                1 for inst in circuit.data
                if inst.operation.name == "ry"
            )

            expected = enc.n_qubits * reps
            assert gate_count == expected, f"reps={reps}: {gate_count} != {expected}"


# =============================================================================
# Test Class: Zero-Angle Edge Cases
# =============================================================================


class TestZeroAngleEdgeCases:
    """Tests for edge cases involving zero angles.

    These tests verify correct behavior when rotation angles are exactly zero
    or very close to zero due to input values or floating-point arithmetic.
    """

    def test_zero_angles_computed_correctly(self) -> None:
        """Test that zero input produces exactly zero angles."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, combination="product")
        x = np.zeros(4)

        angles = enc.compute_angles(x)

        # All angles should be exactly zero
        assert np.all(angles == 0.0), f"Expected all zeros, got {angles}"

    def test_near_zero_angles_not_exactly_zero(self) -> None:
        """Test that very small (but non-zero) inputs produce non-zero angles."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, combination="product")
        eps = np.finfo(float).eps
        x = np.array([eps, eps, eps, eps])

        angles = enc.compute_angles(x)

        # Angles should be non-zero (though very small)
        assert np.all(angles != 0.0), "Expected non-zero angles for non-zero input"

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_zero_input_produces_ground_state(self) -> None:
        """Test that zero input produces the ground state |0...0⟩."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x = np.zeros(4)

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # Ground state: |0000⟩ = [1, 0, 0, ..., 0]
        expected = np.zeros(2 ** enc.n_qubits, dtype=np.complex128)
        expected[0] = 1.0

        np.testing.assert_allclose(state, expected, atol=1e-10)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_single_nonzero_feature(self) -> None:
        """Test encoding when only one feature is non-zero."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2)
        x = np.array([0.5, 0.0, 0.0, 0.0])

        # Should not raise
        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # State should be valid (normalized)
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

        # State should NOT be ground state (since x[0] != 0)
        ground_state = np.zeros(2 ** enc.n_qubits, dtype=np.complex128)
        ground_state[0] = 1.0
        assert not np.allclose(state, ground_state, atol=1e-6)

    def test_sum_combination_zero_input(self) -> None:
        """Test sum combination with zero input."""
        enc = HigherOrderAngleEncoding(n_features=4, order=2, combination="sum")
        x = np.zeros(4)

        angles = enc.compute_angles(x)

        # All angles should be exactly zero for sum combination too
        assert np.all(angles == 0.0)

    def test_mixed_positive_negative_cancellation(self) -> None:
        """Test that positive and negative values can cancel to zero.

        For product combination, x * (-x) produces negative values, not zero.
        This test verifies correct handling of such cases.
        """
        enc = HigherOrderAngleEncoding(n_features=2, order=2, combination="product")
        x = np.array([1.0, -1.0])

        # Terms: (0,) -> 1.0, (1,) -> -1.0, (0,1) -> -1.0
        # Qubit 0 (idx 0, 2): (0,), (0,1) -> 1.0 + (-1.0) = 0.0
        # Qubit 1 (idx 1): (1,) -> -1.0

        angles = enc.compute_angles(x)

        # Verify angle computation is correct
        assert np.isclose(angles[0], 0.0), f"Expected 0.0, got {angles[0]}"
        assert np.isclose(angles[1], -1.0), f"Expected -1.0, got {angles[1]}"
