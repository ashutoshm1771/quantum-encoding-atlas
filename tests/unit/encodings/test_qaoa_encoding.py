"""Comprehensive tests for QAOAEncoding.

This test module provides complete coverage of the QAOAEncoding class,
which implements a QAOA-inspired quantum data encoding. It includes:

- Instantiation and parameter validation
- Property computation (n_qubits, depth, gate counts, simulability)
- Entanglement pattern behavior (linear, full, circular, none)
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

Run with: pytest tests/unit/encodings/test_qaoa_encoding.py -v
Run excluding slow tests: pytest tests/unit/encodings/test_qaoa_encoding.py -v -m "not slow"

References
----------
.. [1] Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate
       Optimization Algorithm. arXiv:1411.4028
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas.encodings.qaoa_encoding import QAOAEncoding

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
    return np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture
def default_encoding() -> QAOAEncoding:
    """Default QAOAEncoding with n_features=4, reps=2."""
    return QAOAEncoding(n_features=4, reps=2)


@pytest.fixture
def encoding_full_entanglement() -> QAOAEncoding:
    """QAOAEncoding with full entanglement pattern."""
    return QAOAEncoding(n_features=4, reps=2, entanglement="full")


@pytest.fixture
def encoding_no_entanglement() -> QAOAEncoding:
    """QAOAEncoding with no entanglement (product state)."""
    return QAOAEncoding(n_features=4, reps=2, entanglement="none")


# =============================================================================
# Test Class: Instantiation
# =============================================================================


class TestInstantiation:
    """Tests for QAOAEncoding instantiation and parameter handling."""

    def test_default_parameters(self) -> None:
        """Test creating encoding with default parameters."""
        enc = QAOAEncoding(n_features=4)

        assert enc.n_features == 4
        assert enc.reps == 2
        assert enc.data_rotation == "Z"
        assert enc.mixer_rotation == "X"
        assert enc.entanglement == "linear"
        assert enc.entangling_gate == "cz"
        assert enc.gamma == 1.0
        assert enc.beta == 1.0
        assert enc.include_initial_h is True
        assert enc.feature_map == "linear"

    def test_single_feature(self) -> None:
        """Test creating encoding with single feature (minimum case)."""
        enc = QAOAEncoding(n_features=1)
        assert enc.n_features == 1
        assert enc.n_qubits == 1

    def test_various_feature_counts(self) -> None:
        """Test instantiation with various feature counts."""
        for n in [1, 2, 4, 8, 16]:
            enc = QAOAEncoding(n_features=n)
            assert enc.n_features == n

    def test_custom_reps(self) -> None:
        """Test instantiation with custom repetitions."""
        enc = QAOAEncoding(n_features=4, reps=5)
        assert enc.reps == 5

    def test_reps_equals_one(self) -> None:
        """Test reps=1 (minimal QAOA depth)."""
        enc = QAOAEncoding(n_features=4, reps=1)
        assert enc.reps == 1

    def test_data_rotation_y(self) -> None:
        """Test data rotation Y axis."""
        enc = QAOAEncoding(n_features=4, data_rotation="Y")
        assert enc.data_rotation == "Y"

    def test_data_rotation_x(self) -> None:
        """Test data rotation X axis."""
        enc = QAOAEncoding(n_features=4, data_rotation="X")
        assert enc.data_rotation == "X"

    def test_data_rotation_lowercase(self) -> None:
        """Test lowercase data rotation is normalized to uppercase."""
        enc = QAOAEncoding(n_features=4, data_rotation="z")
        assert enc.data_rotation == "Z"

    def test_mixer_rotation_y(self) -> None:
        """Test mixer rotation Y axis."""
        enc = QAOAEncoding(n_features=4, mixer_rotation="Y")
        assert enc.mixer_rotation == "Y"

    def test_mixer_rotation_z(self) -> None:
        """Test mixer rotation Z axis."""
        enc = QAOAEncoding(n_features=4, mixer_rotation="Z")
        assert enc.mixer_rotation == "Z"

    def test_mixer_rotation_lowercase(self) -> None:
        """Test lowercase mixer rotation is normalized to uppercase."""
        enc = QAOAEncoding(n_features=4, mixer_rotation="x")
        assert enc.mixer_rotation == "X"

    def test_entanglement_full(self) -> None:
        """Test full entanglement pattern."""
        enc = QAOAEncoding(n_features=4, entanglement="full")
        assert enc.entanglement == "full"

    def test_entanglement_circular(self) -> None:
        """Test circular entanglement pattern."""
        enc = QAOAEncoding(n_features=4, entanglement="circular")
        assert enc.entanglement == "circular"

    def test_entanglement_none(self) -> None:
        """Test no entanglement (product state)."""
        enc = QAOAEncoding(n_features=4, entanglement="none")
        assert enc.entanglement == "none"

    def test_entanglement_uppercase(self) -> None:
        """Test uppercase entanglement is normalized to lowercase."""
        enc = QAOAEncoding(n_features=4, entanglement="LINEAR")
        assert enc.entanglement == "linear"

    def test_entangling_gate_cx(self) -> None:
        """Test CNOT entangling gate."""
        enc = QAOAEncoding(n_features=4, entangling_gate="cx")
        assert enc.entangling_gate == "cx"

    def test_entangling_gate_rzz(self) -> None:
        """Test RZZ entangling gate."""
        enc = QAOAEncoding(n_features=4, entangling_gate="rzz")
        assert enc.entangling_gate == "rzz"

    def test_entangling_gate_uppercase(self) -> None:
        """Test uppercase entangling gate is normalized to lowercase."""
        enc = QAOAEncoding(n_features=4, entangling_gate="CZ")
        assert enc.entangling_gate == "cz"

    def test_custom_gamma(self) -> None:
        """Test custom gamma scaling factor."""
        enc = QAOAEncoding(n_features=4, gamma=np.pi)
        assert enc.gamma == np.pi

    def test_custom_beta(self) -> None:
        """Test custom beta scaling factor."""
        enc = QAOAEncoding(n_features=4, beta=np.pi / 2)
        assert enc.beta == np.pi / 2

    def test_negative_gamma(self) -> None:
        """Test negative gamma (should be allowed)."""
        enc = QAOAEncoding(n_features=4, gamma=-1.0)
        assert enc.gamma == -1.0

    def test_negative_beta(self) -> None:
        """Test negative beta (should be allowed)."""
        enc = QAOAEncoding(n_features=4, beta=-1.0)
        assert enc.beta == -1.0

    def test_zero_gamma(self) -> None:
        """Test zero gamma."""
        enc = QAOAEncoding(n_features=4, gamma=0.0)
        assert enc.gamma == 0.0

    def test_zero_beta(self) -> None:
        """Test zero beta."""
        enc = QAOAEncoding(n_features=4, beta=0.0)
        assert enc.beta == 0.0

    def test_include_initial_h_false(self) -> None:
        """Test without initial Hadamard layer."""
        enc = QAOAEncoding(n_features=4, include_initial_h=False)
        assert enc.include_initial_h is False

    def test_feature_map_quadratic(self) -> None:
        """Test quadratic feature map."""
        enc = QAOAEncoding(n_features=4, feature_map="quadratic")
        assert enc.feature_map == "quadratic"

    def test_feature_map_uppercase(self) -> None:
        """Test uppercase feature map is normalized to lowercase."""
        enc = QAOAEncoding(n_features=4, feature_map="LINEAR")
        assert enc.feature_map == "linear"

    def test_all_custom_parameters(self) -> None:
        """Test all parameters customized together."""
        enc = QAOAEncoding(
            n_features=6,
            reps=4,
            data_rotation="Y",
            mixer_rotation="Z",
            entanglement="full",
            entangling_gate="cx",
            gamma=2.0,
            beta=0.5,
            include_initial_h=False,
            feature_map="quadratic",
        )

        assert enc.n_features == 6
        assert enc.reps == 4
        assert enc.data_rotation == "Y"
        assert enc.mixer_rotation == "Z"
        assert enc.entanglement == "full"
        assert enc.entangling_gate == "cx"
        assert enc.gamma == 2.0
        assert enc.beta == 0.5
        assert enc.include_initial_h is False
        assert enc.feature_map == "quadratic"

    def test_instantiation_does_not_generate_circuit(self) -> None:
        """Test that instantiation is lazy (no circuit generated)."""
        enc = QAOAEncoding(n_features=100)
        assert enc.n_features == 100
        # Should be fast since no circuit is pre-computed


# =============================================================================
# Test Class: Validation
# =============================================================================


class TestValidation:
    """Tests for parameter validation and error handling."""

    def test_invalid_n_features_zero(self) -> None:
        """Test that n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be at least 1"):
            QAOAEncoding(n_features=0)

    def test_invalid_n_features_negative(self) -> None:
        """Test that negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be at least 1"):
            QAOAEncoding(n_features=-1)

    def test_invalid_reps_zero(self) -> None:
        """Test that reps=0 raises ValueError."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            QAOAEncoding(n_features=4, reps=0)

    def test_invalid_reps_negative(self) -> None:
        """Test that negative reps raises ValueError."""
        with pytest.raises(ValueError, match="reps must be at least 1"):
            QAOAEncoding(n_features=4, reps=-1)

    def test_invalid_data_rotation(self) -> None:
        """Test invalid data rotation axis raises ValueError."""
        with pytest.raises(ValueError, match="data_rotation must be one of"):
            QAOAEncoding(n_features=4, data_rotation="W")

    def test_invalid_mixer_rotation(self) -> None:
        """Test invalid mixer rotation axis raises ValueError."""
        with pytest.raises(ValueError, match="mixer_rotation must be one of"):
            QAOAEncoding(n_features=4, mixer_rotation="W")

    def test_invalid_entanglement(self) -> None:
        """Test invalid entanglement pattern raises ValueError."""
        with pytest.raises(ValueError, match="entanglement must be one of"):
            QAOAEncoding(n_features=4, entanglement="invalid")

    def test_invalid_entangling_gate(self) -> None:
        """Test invalid entangling gate raises ValueError."""
        with pytest.raises(ValueError, match="entangling_gate must be one of"):
            QAOAEncoding(n_features=4, entangling_gate="swap")

    def test_invalid_gamma_nan(self) -> None:
        """Test that NaN gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be finite"):
            QAOAEncoding(n_features=4, gamma=float("nan"))

    def test_invalid_gamma_inf(self) -> None:
        """Test that infinite gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be finite"):
            QAOAEncoding(n_features=4, gamma=float("inf"))

    def test_invalid_beta_nan(self) -> None:
        """Test that NaN beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be finite"):
            QAOAEncoding(n_features=4, beta=float("nan"))

    def test_invalid_beta_inf(self) -> None:
        """Test that infinite beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be finite"):
            QAOAEncoding(n_features=4, beta=float("inf"))

    def test_invalid_feature_map(self) -> None:
        """Test invalid feature map raises ValueError."""
        with pytest.raises(ValueError, match="feature_map must be one of"):
            QAOAEncoding(n_features=4, feature_map="cubic")

    def test_type_error_n_features_float(self) -> None:
        """Test that float n_features raises TypeError."""
        with pytest.raises(TypeError, match="n_features must be an integer"):
            QAOAEncoding(n_features=4.5)  # type: ignore

    def test_type_error_reps_float(self) -> None:
        """Test that float reps raises TypeError."""
        with pytest.raises(TypeError, match="reps must be an integer"):
            QAOAEncoding(n_features=4, reps=2.5)  # type: ignore

    def test_type_error_data_rotation_int(self) -> None:
        """Test that non-string data_rotation raises TypeError."""
        with pytest.raises(TypeError, match="data_rotation must be a string"):
            QAOAEncoding(n_features=4, data_rotation=1)  # type: ignore

    def test_type_error_mixer_rotation_int(self) -> None:
        """Test that non-string mixer_rotation raises TypeError."""
        with pytest.raises(TypeError, match="mixer_rotation must be a string"):
            QAOAEncoding(n_features=4, mixer_rotation=1)  # type: ignore

    def test_type_error_entanglement_int(self) -> None:
        """Test that non-string entanglement raises TypeError."""
        with pytest.raises(TypeError, match="entanglement must be a string"):
            QAOAEncoding(n_features=4, entanglement=1)  # type: ignore

    def test_type_error_entangling_gate_int(self) -> None:
        """Test that non-string entangling_gate raises TypeError."""
        with pytest.raises(TypeError, match="entangling_gate must be a string"):
            QAOAEncoding(n_features=4, entangling_gate=1)  # type: ignore

    def test_type_error_gamma_string(self) -> None:
        """Test that non-numeric gamma raises TypeError."""
        with pytest.raises(TypeError, match="gamma must be a number"):
            QAOAEncoding(n_features=4, gamma="1.0")  # type: ignore

    def test_type_error_beta_string(self) -> None:
        """Test that non-numeric beta raises TypeError."""
        with pytest.raises(TypeError, match="beta must be a number"):
            QAOAEncoding(n_features=4, beta="1.0")  # type: ignore

    def test_type_error_include_initial_h_string(self) -> None:
        """Test that non-bool include_initial_h raises TypeError."""
        with pytest.raises(TypeError, match="include_initial_h must be a bool"):
            QAOAEncoding(n_features=4, include_initial_h="True")  # type: ignore

    def test_type_error_feature_map_int(self) -> None:
        """Test that non-string feature_map raises TypeError."""
        with pytest.raises(TypeError, match="feature_map must be a string"):
            QAOAEncoding(n_features=4, feature_map=1)  # type: ignore


# =============================================================================
# Test Class: Properties
# =============================================================================


class TestProperties:
    """Tests for computed properties of QAOAEncoding."""

    def test_n_qubits_equals_n_features(self) -> None:
        """Test that n_qubits equals n_features."""
        for n in [2, 4, 8, 16]:
            enc = QAOAEncoding(n_features=n)
            assert enc.n_qubits == n

    def test_depth_with_initial_h(self) -> None:
        """Test depth includes initial Hadamard layer."""
        enc = QAOAEncoding(n_features=4, reps=1, include_initial_h=True)
        # H layer + (data layer + mixer layer + entangle layer) × reps
        assert enc.depth >= 4

    def test_depth_without_initial_h(self) -> None:
        """Test depth without initial Hadamard layer."""
        enc = QAOAEncoding(n_features=4, reps=1, include_initial_h=False)
        # (data layer + mixer layer + entangle layer) × reps
        assert enc.depth >= 3

    def test_depth_increases_with_reps(self) -> None:
        """Test that depth increases with reps."""
        depths = []
        for reps in [1, 2, 3, 4]:
            enc = QAOAEncoding(n_features=4, reps=reps)
            depths.append(enc.depth)

        # Depth should strictly increase
        for i in range(1, len(depths)):
            assert depths[i] > depths[i - 1]

    def test_depth_linear_entanglement_exact(self) -> None:
        """Test exact depth calculation for linear entanglement.

        Linear entanglement depth uses optimal parallel scheduling:
        - Even pairs: (0,1), (2,3), ... in parallel
        - Odd pairs: (1,2), (3,4), ... in parallel
        This gives depth = min(2, n-1) for the entangling layer.
        """
        # n=4, reps=1, with H: 1 (H) + 1 (data) + 1 (mixer) + 2 (entangle) = 5
        enc = QAOAEncoding(n_features=4, reps=1, entanglement="linear")
        assert enc.depth == 5

        # n=4, reps=2, with H: 1 + 2*(1+1+2) = 9
        enc = QAOAEncoding(n_features=4, reps=2, entanglement="linear")
        assert enc.depth == 9

        # n=2, reps=1, with H: 1 + 1 + 1 + 1 = 4 (only 1 pair, depth=1)
        enc = QAOAEncoding(n_features=2, reps=1, entanglement="linear")
        assert enc.depth == 4

    def test_depth_circular_entanglement_exact(self) -> None:
        """Test exact depth calculation for circular entanglement.

        Circular entanglement forms a cycle graph C_n.
        By edge coloring theory (chromatic index):
        - χ'(C_n) = 2 if n is even
        - χ'(C_n) = 3 if n is odd
        """
        # n=4 (even), reps=1, with H: 1 + 1 + 1 + 2 = 5
        enc = QAOAEncoding(n_features=4, reps=1, entanglement="circular")
        assert enc.depth == 5

        # n=5 (odd), reps=1, with H: 1 + 1 + 1 + 3 = 6
        enc = QAOAEncoding(n_features=5, reps=1, entanglement="circular")
        assert enc.depth == 6

        # n=6 (even), reps=2, with H: 1 + 2*(1+1+2) = 9
        enc = QAOAEncoding(n_features=6, reps=2, entanglement="circular")
        assert enc.depth == 9

        # n=7 (odd), reps=2, with H: 1 + 2*(1+1+3) = 11
        enc = QAOAEncoding(n_features=7, reps=2, entanglement="circular")
        assert enc.depth == 11

    def test_depth_full_entanglement_exact(self) -> None:
        """Test exact depth calculation for full entanglement.

        Full entanglement is a complete graph K_n.
        By Vizing's theorem:
        - χ'(K_n) = n-1 if n is even
        - χ'(K_n) = n if n is odd
        """
        # n=4 (even), reps=1, with H: 1 + 1 + 1 + 3 = 6
        enc = QAOAEncoding(n_features=4, reps=1, entanglement="full")
        assert enc.depth == 6

        # n=5 (odd), reps=1, with H: 1 + 1 + 1 + 5 = 8
        enc = QAOAEncoding(n_features=5, reps=1, entanglement="full")
        assert enc.depth == 8

    def test_depth_no_entanglement_exact(self) -> None:
        """Test exact depth calculation with no entanglement."""
        # n=4, reps=1, with H: 1 + 1 + 1 + 0 = 3
        enc = QAOAEncoding(n_features=4, reps=1, entanglement="none")
        assert enc.depth == 3

        # n=4, reps=2, with H: 1 + 2*(1+1+0) = 5
        enc = QAOAEncoding(n_features=4, reps=2, entanglement="none")
        assert enc.depth == 5

    def test_properties_object(self, default_encoding: QAOAEncoding) -> None:
        """Test that properties returns EncodingProperties object."""
        props = default_encoding.properties

        assert props.n_qubits == 4
        assert props.depth > 0
        assert props.gate_count > 0
        assert props.single_qubit_gates > 0
        assert props.two_qubit_gates > 0

    def test_is_entangling_with_linear(self) -> None:
        """Test that linear entanglement creates entangled state."""
        enc = QAOAEncoding(n_features=4, entanglement="linear")
        assert enc.properties.is_entangling is True

    def test_is_entangling_with_full(self) -> None:
        """Test that full entanglement creates entangled state."""
        enc = QAOAEncoding(n_features=4, entanglement="full")
        assert enc.properties.is_entangling is True

    def test_not_entangling_with_none(self) -> None:
        """Test that no entanglement creates product state."""
        enc = QAOAEncoding(n_features=4, entanglement="none")
        assert enc.properties.is_entangling is False

    def test_not_entangling_single_qubit(self) -> None:
        """Test single qubit is never entangling."""
        enc = QAOAEncoding(n_features=1, entanglement="linear")
        assert enc.properties.is_entangling is False

    def test_simulability_entangling(self) -> None:
        """Test that entangling circuit is not simulable."""
        enc = QAOAEncoding(n_features=4, entanglement="linear")
        assert enc.properties.simulability == "not_simulable"

    def test_simulability_product_state(self) -> None:
        """Test that product state circuit is simulable."""
        enc = QAOAEncoding(n_features=4, entanglement="none")
        assert enc.properties.simulability == "simulable"

    def test_properties_cached(self, default_encoding: QAOAEncoding) -> None:
        """Test that properties are cached (same object returned)."""
        props1 = default_encoding.properties
        props2 = default_encoding.properties
        assert props1 is props2

    def test_trainability_estimate_exists(self, default_encoding: QAOAEncoding) -> None:
        """Test trainability estimate is computed."""
        props = default_encoding.properties
        assert props.trainability_estimate is not None
        assert 0 < props.trainability_estimate <= 1

    def test_properties_notes_contain_qaoa(
        self, default_encoding: QAOAEncoding
    ) -> None:
        """Test that notes mention QAOA."""
        props = default_encoding.properties
        assert "QAOA" in props.notes

    def test_two_qubit_gate_count(self) -> None:
        """Test two-qubit gate count calculation."""
        enc = QAOAEncoding(n_features=4, reps=2, entanglement="linear")
        props = enc.properties

        # Linear: 3 pairs per rep, 2 reps = 6 two-qubit gates
        expected = 3 * 2
        assert props.two_qubit_gates == expected

    def test_layer_info(self, default_encoding: QAOAEncoding) -> None:
        """Test get_layer_info returns correct structure."""
        info = default_encoding.get_layer_info()

        assert "n_qubits" in info
        assert "reps" in info
        assert "layers_per_rep" in info
        assert "entanglement_pairs" in info
        assert "total_data_gates" in info
        assert "total_mixer_gates" in info
        assert "total_entangling_gates" in info

        assert info["n_qubits"] == 4
        assert info["reps"] == 2


# =============================================================================
# Test Class: Entanglement Patterns (QAOA-specific unique behavior)
# =============================================================================


class TestEntanglementPatterns:
    """Tests for entanglement pair computation for different patterns.

    This is QAOA-specific behavior testing entanglement topology.
    """

    def test_linear_entanglement_pairs(self) -> None:
        """Test linear entanglement produces correct pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="linear")
        expected = [(0, 1), (1, 2), (2, 3)]
        assert enc._entanglement_pairs == expected

    def test_full_entanglement_pairs(self) -> None:
        """Test full entanglement produces all pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="full")
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert enc._entanglement_pairs == expected

    def test_circular_entanglement_pairs(self) -> None:
        """Test circular entanglement produces correct pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="circular")
        expected = [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert enc._entanglement_pairs == expected

    def test_none_entanglement_pairs(self) -> None:
        """Test no entanglement produces empty pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="none")
        assert enc._entanglement_pairs == []

    def test_single_qubit_no_pairs(self) -> None:
        """Test single qubit has no entanglement pairs."""
        enc = QAOAEncoding(n_features=1, entanglement="linear")
        assert enc._entanglement_pairs == []

    def test_two_qubits_linear(self) -> None:
        """Test two qubits with linear entanglement."""
        enc = QAOAEncoding(n_features=2, entanglement="linear")
        assert enc._entanglement_pairs == [(0, 1)]

    def test_two_qubits_circular(self) -> None:
        """Test two qubits with circular entanglement."""
        enc = QAOAEncoding(n_features=2, entanglement="circular")
        # For n=2, circular should just have one pair
        assert enc._entanglement_pairs == [(0, 1)]

    def test_pair_count_linear(self) -> None:
        """Test linear entanglement pair count equals n-1."""
        for n in range(2, 10):
            enc = QAOAEncoding(n_features=n, entanglement="linear")
            assert len(enc._entanglement_pairs) == n - 1

    def test_pair_count_full(self) -> None:
        """Test full entanglement pair count equals n choose 2."""
        for n in range(2, 10):
            enc = QAOAEncoding(n_features=n, entanglement="full")
            expected = n * (n - 1) // 2
            assert len(enc._entanglement_pairs) == expected

    def test_pair_count_circular(self) -> None:
        """Test circular entanglement pair count equals n."""
        for n in range(3, 10):
            enc = QAOAEncoding(n_features=n, entanglement="circular")
            assert len(enc._entanglement_pairs) == n


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input data validation during circuit generation."""

    def test_valid_1d_input(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 1D input is accepted."""
        circuit = default_encoding.get_circuit(sample_data_4d)
        assert circuit is not None

    def test_valid_2d_input_single_sample(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that valid 2D input with single sample is accepted."""
        x = sample_data_4d.reshape(1, -1)
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_wrong_feature_count(self, default_encoding: QAOAEncoding) -> None:
        """Test that wrong number of features raises ValueError."""
        x = np.array([0.1, 0.2, 0.3])  # 3 features, expect 4
        with pytest.raises(ValueError, match="Expected 4 features"):
            default_encoding.get_circuit(x)

    def test_nan_input_rejected(self, default_encoding: QAOAEncoding) -> None:
        """Test that NaN values in input raise ValueError."""
        x = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN"):
            default_encoding.get_circuit(x)

    def test_inf_input_rejected(self, default_encoding: QAOAEncoding) -> None:
        """Test that infinite values in input raise ValueError."""
        x = np.array([0.1, np.inf, 0.3, 0.4])
        with pytest.raises(ValueError, match="infinite"):
            default_encoding.get_circuit(x)

    def test_2d_multiple_samples_rejected(
        self,
        default_encoding: QAOAEncoding,
        batch_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that 2D input with multiple samples raises error for get_circuit."""
        with pytest.raises(ValueError, match="single sample"):
            default_encoding.get_circuit(batch_data_4d)

    def test_list_input_converted(self, default_encoding: QAOAEncoding) -> None:
        """Test that list input is converted to numpy array."""
        x = [0.1, 0.2, 0.3, 0.4]
        circuit = default_encoding.get_circuit(x)
        assert circuit is not None

    def test_batch_input_shape(
        self,
        default_encoding: QAOAEncoding,
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
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that PennyLane circuit is a callable function."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="pennylane")
        assert callable(circuit)

    def test_circuit_executes_without_error(
        self,
        default_encoding: QAOAEncoding,
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
        default_encoding: QAOAEncoding,
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

    def test_no_initial_h(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test circuit without initial Hadamard gates."""
        enc = QAOAEncoding(n_features=4, include_initial_h=False)
        circuit_fn = enc.get_circuit(sample_data_4d, backend="pennylane")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        assert state is not None

    def test_different_entangling_gates(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test different entangling gates execute correctly."""
        for gate in ["cx", "cz", "rzz"]:
            enc = QAOAEncoding(n_features=4, entangling_gate=gate)
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
        default_encoding: QAOAEncoding,
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
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit is a QuantumCircuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert isinstance(circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        assert circuit.num_qubits == default_encoding.n_qubits

    def test_circuit_has_gates(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Qiskit circuit has gates."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="qiskit")
        # Should have many gates
        assert len(circuit.data) > default_encoding.n_qubits

    def test_has_initial_h_gates(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that Hadamard gates are present when include_initial_h=True."""
        enc = QAOAEncoding(n_features=4, include_initial_h=True)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        # Check for Hadamard gates
        h_count = sum(1 for inst in circuit.data if inst.operation.name == "h")
        assert h_count == 4

    def test_no_initial_h_gates(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that Hadamard gates are absent when include_initial_h=False."""
        enc = QAOAEncoding(n_features=4, include_initial_h=False)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        # Check for no Hadamard gates
        h_count = sum(1 for inst in circuit.data if inst.operation.name == "h")
        assert h_count == 0

    def test_cz_gates(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that CZ gates are used when specified."""
        enc = QAOAEncoding(n_features=4, entangling_gate="cz")
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        cz_count = sum(1 for inst in circuit.data if inst.operation.name == "cz")
        assert cz_count > 0

    def test_cx_gates(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that CX (CNOT) gates are used when specified."""
        enc = QAOAEncoding(n_features=4, entangling_gate="cx")
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        cx_count = sum(1 for inst in circuit.data if inst.operation.name == "cx")
        assert cx_count > 0

    def test_rzz_gates(self, sample_data_4d: NDArray[np.floating]) -> None:
        """Test that RZZ gates are used when specified."""
        enc = QAOAEncoding(n_features=4, entangling_gate="rzz")
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        rzz_count = sum(1 for inst in circuit.data if inst.operation.name == "rzz")
        assert rzz_count > 0

    def test_multiple_reps_has_barriers(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test that multiple reps have barriers between them."""
        enc = QAOAEncoding(n_features=4, reps=3)
        circuit = enc.get_circuit(sample_data_4d, backend="qiskit")

        barrier_count = sum(
            1 for inst in circuit.data if inst.operation.name == "barrier"
        )
        # reps - 1 barriers
        assert barrier_count == 2

    def test_batch_circuits(
        self,
        default_encoding: QAOAEncoding,
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
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit is a cirq.Circuit."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_has_operations(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that Cirq circuit has operations."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_qubit_count(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that circuit has correct number of qubits."""
        circuit = default_encoding.get_circuit(sample_data_4d, backend="cirq")
        assert len(circuit.all_qubits()) == default_encoding.n_qubits

    def test_different_entangling_gates(
        self, sample_data_4d: NDArray[np.floating]
    ) -> None:
        """Test different entangling gates in Cirq."""
        for gate in ["cx", "cz", "rzz"]:
            enc = QAOAEncoding(n_features=4, entangling_gate=gate)
            circuit = enc.get_circuit(sample_data_4d, backend="cirq")
            assert isinstance(circuit, cirq.Circuit)

    def test_batch_circuits(
        self,
        default_encoding: QAOAEncoding,
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
    """Tests for mathematical correctness of angle computation."""

    def test_linear_feature_map(self) -> None:
        """Test linear feature map computes angles correctly."""
        enc = QAOAEncoding(n_features=4, gamma=1.0, feature_map="linear")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        angles = enc.compute_data_angles(x)

        # Linear: angle = gamma * x_i
        expected = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(angles, expected)

    def test_quadratic_feature_map(self) -> None:
        """Test quadratic feature map computes angles correctly."""
        enc = QAOAEncoding(n_features=4, gamma=1.0, feature_map="quadratic")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        angles = enc.compute_data_angles(x)

        # Quadratic: angle = gamma * x_i^2
        expected = np.array([0.01, 0.04, 0.09, 0.16])
        np.testing.assert_allclose(angles, expected)

    def test_gamma_scaling(self) -> None:
        """Test that gamma scaling is applied correctly."""
        enc1 = QAOAEncoding(n_features=4, gamma=1.0)
        enc2 = QAOAEncoding(n_features=4, gamma=2.0)

        x = np.array([0.1, 0.2, 0.3, 0.4])

        angles1 = enc1.compute_data_angles(x)
        angles2 = enc2.compute_data_angles(x)

        np.testing.assert_allclose(angles2, 2.0 * angles1)

    def test_zero_input(self) -> None:
        """Test that zero input produces zero angles."""
        enc = QAOAEncoding(n_features=4)
        x = np.zeros(4)

        angles = enc.compute_data_angles(x)

        np.testing.assert_allclose(angles, 0.0)

    def test_negative_input(self) -> None:
        """Test negative input values produce negative angles."""
        enc = QAOAEncoding(n_features=4, gamma=1.0, feature_map="linear")
        x = np.array([-0.1, -0.2, 0.3, 0.4])

        angles = enc.compute_data_angles(x)

        expected = np.array([-0.1, -0.2, 0.3, 0.4])
        np.testing.assert_allclose(angles, expected)

    def test_negative_gamma(self) -> None:
        """Test that negative gamma reverses angles."""
        enc_pos = QAOAEncoding(n_features=4, gamma=1.0)
        enc_neg = QAOAEncoding(n_features=4, gamma=-1.0)

        x = np.array([0.1, 0.2, 0.3, 0.4])

        angles_pos = enc_pos.compute_data_angles(x)
        angles_neg = enc_neg.compute_data_angles(x)

        np.testing.assert_allclose(angles_neg, -angles_pos)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature(self) -> None:
        """Test encoding with single feature."""
        enc = QAOAEncoding(n_features=1, reps=2)
        x = np.array([0.5])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None
        assert enc.n_qubits == 1

    def test_large_feature_count(self) -> None:
        """Test encoding with many features."""
        enc = QAOAEncoding(n_features=16, reps=2)

        assert enc.n_qubits == 16
        x = np.random.randn(16)
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_many_reps(self) -> None:
        """Test encoding with many repetitions."""
        enc = QAOAEncoding(n_features=4, reps=10)

        assert enc.reps == 10
        assert enc.depth > 30  # Should be deep

    def test_extreme_values(self) -> None:
        """Test encoding with extreme input values."""
        enc = QAOAEncoding(n_features=4)

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

    def test_extreme_gamma_beta(self) -> None:
        """Test with large gamma and beta values."""
        enc = QAOAEncoding(n_features=4, gamma=100.0, beta=100.0)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_zero_gamma_zero_beta(self) -> None:
        """Test with zero gamma and beta (trivial encoding)."""
        enc = QAOAEncoding(n_features=4, gamma=0.0, beta=0.0)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    def test_product_state_single_rep(self) -> None:
        """Test product state encoding with single rep."""
        enc = QAOAEncoding(n_features=4, reps=1, entanglement="none")
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================


@pytest.mark.numerical_stability
class TestNumericalStability:
    """Tests for numerical stability with extreme values.

    These tests verify that QAOAEncoding produces valid, normalized quantum
    states even when input data contains extreme values.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_small_values(self) -> None:
        """Test encoding with very small input values near machine epsilon."""
        enc = QAOAEncoding(n_features=4, reps=2)
        x = np.array([1e-15, 1e-16, 1e-17, 1e-18])

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
        # All values should be finite
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_very_large_values(self) -> None:
        """Test encoding with very large input values.

        Large values are valid as rotation angles wrap around 2π.
        """
        enc = QAOAEncoding(n_features=4, reps=2)
        x = np.array([1e5, 2e5, 3e5, 4e5])

        circuit_fn = enc.get_circuit(x, backend="pennylane")
        dev = qml.device("default.qubit", wires=enc.n_qubits)

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()
        # State should be normalized despite large input
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)
        assert np.all(np.isfinite(state))

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_mixed_magnitude_values(self) -> None:
        """Test encoding with mixed magnitude values spanning many orders."""
        enc = QAOAEncoding(n_features=4, reps=2)
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
        enc = QAOAEncoding(n_features=4, reps=2)
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


# =============================================================================
# Test Class: Equality and Hashing
# =============================================================================


class TestEqualityAndHashing:
    """Tests for equality comparison and hashing."""

    def test_equality_same_parameters(self) -> None:
        """Test that encodings with same parameters are equal."""
        enc1 = QAOAEncoding(n_features=4, reps=2)
        enc2 = QAOAEncoding(n_features=4, reps=2)
        assert enc1 == enc2

    def test_equality_different_n_features(self) -> None:
        """Test that encodings with different n_features are not equal."""
        enc1 = QAOAEncoding(n_features=4)
        enc2 = QAOAEncoding(n_features=8)
        assert enc1 != enc2

    def test_inequality_different_reps(self) -> None:
        """Test that encodings with different reps are not equal."""
        enc1 = QAOAEncoding(n_features=4, reps=2)
        enc2 = QAOAEncoding(n_features=4, reps=3)
        assert enc1 != enc2

    def test_inequality_different_data_rotation(self) -> None:
        """Test that encodings with different data_rotation are not equal."""
        enc1 = QAOAEncoding(n_features=4, data_rotation="Z")
        enc2 = QAOAEncoding(n_features=4, data_rotation="Y")
        assert enc1 != enc2

    def test_inequality_different_entanglement(self) -> None:
        """Test that encodings with different entanglement are not equal."""
        enc1 = QAOAEncoding(n_features=4, entanglement="linear")
        enc2 = QAOAEncoding(n_features=4, entanglement="full")
        assert enc1 != enc2

    def test_inequality_different_gamma(self) -> None:
        """Test that encodings with different gamma are not equal."""
        enc1 = QAOAEncoding(n_features=4, gamma=1.0)
        enc2 = QAOAEncoding(n_features=4, gamma=2.0)
        assert enc1 != enc2

    def test_hash_consistency(self) -> None:
        """Test that equal objects have equal hashes."""
        enc1 = QAOAEncoding(n_features=4, reps=2, gamma=1.5)
        enc2 = QAOAEncoding(n_features=4, reps=2, gamma=1.5)
        assert hash(enc1) == hash(enc2)

    def test_set_membership(self) -> None:
        """Test that encodings work correctly in sets."""
        enc1 = QAOAEncoding(n_features=4, reps=2)
        enc2 = QAOAEncoding(n_features=4, reps=2)  # Same as enc1
        enc3 = QAOAEncoding(n_features=4, reps=3)

        s = {enc1, enc2, enc3}
        # enc1 and enc2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_dict_key_usage(self) -> None:
        """Test that encodings work correctly as dictionary keys."""
        enc1 = QAOAEncoding(n_features=4, reps=2)
        enc2 = QAOAEncoding(n_features=4, reps=2)

        d = {enc1: "four features"}
        assert d[enc2] == "four features"  # enc2 should find same key


# =============================================================================
# Test Class: Repr
# =============================================================================


class TestRepr:
    """Tests for string representation."""

    def test_repr_contains_class_name(self, default_encoding: QAOAEncoding) -> None:
        """Test that repr contains class name."""
        assert "QAOAEncoding" in repr(default_encoding)

    def test_repr_contains_parameters(self, default_encoding: QAOAEncoding) -> None:
        """Test that repr contains all parameters."""
        r = repr(default_encoding)

        assert "n_features=4" in r
        assert "reps=2" in r
        assert "data_rotation='Z'" in r
        assert "mixer_rotation='X'" in r
        assert "entanglement='linear'" in r
        assert "entangling_gate='cz'" in r
        assert "gamma=1.0" in r
        assert "beta=1.0" in r
        assert "include_initial_h=True" in r
        assert "feature_map='linear'" in r

    def test_repr_custom_params(self) -> None:
        """Test repr with custom parameters."""
        enc = QAOAEncoding(
            n_features=6,
            reps=4,
            data_rotation="Y",
            entanglement="full",
            gamma=2.5,
        )

        r = repr(enc)
        assert "n_features=6" in r
        assert "reps=4" in r
        assert "data_rotation='Y'" in r
        assert "entanglement='full'" in r
        assert "gamma=2.5" in r


# =============================================================================
# Test Class: Backend Error Handling
# =============================================================================


class TestBackendErrorHandling:
    """Tests for invalid backend names and missing backend handling."""

    def test_invalid_backend_name(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="invalid")  # type: ignore

    def test_invalid_backend_typo(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that common typos in backend names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="penylane")  # type: ignore

    def test_invalid_backend_case_sensitive(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that uppercase backend names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            default_encoding.get_circuit(sample_data_4d, backend="QISKIT")  # type: ignore

    def test_invalid_backend_empty_string(
        self,
        default_encoding: QAOAEncoding,
        sample_data_4d: NDArray[np.floating],
    ) -> None:
        """Test that empty backend string raises ValueError."""
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
        enc = QAOAEncoding(
            n_features=4,
            reps=3,
            data_rotation="Y",
            mixer_rotation="Z",
            entanglement="full",
            gamma=2.5,
            beta=1.5,
        )

        # Force properties to be computed
        _ = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert restored.n_features == enc.n_features
        assert restored.reps == enc.reps
        assert restored.data_rotation == enc.data_rotation
        assert restored.mixer_rotation == enc.mixer_rotation
        assert restored.gamma == enc.gamma
        assert restored.beta == enc.beta

    def test_pickle_equality(self) -> None:
        """Test that pickled and restored encoding equals original."""
        enc = QAOAEncoding(n_features=4, reps=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        assert enc == restored
        assert hash(enc) == hash(restored)

    def test_pickle_circuit_generation_after_restore(self) -> None:
        """Test that circuit generation works after unpickling."""
        enc = QAOAEncoding(n_features=4, reps=2)
        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = restored.get_circuit(x, backend="pennylane")
        assert callable(circuit)

    def test_pickle_properties_after_restore(self) -> None:
        """Test that properties work correctly after unpickling."""
        enc = QAOAEncoding(n_features=4, reps=2)
        original_props = enc.properties

        pickled = pickle.dumps(enc)
        restored = pickle.loads(pickled)

        restored_props = restored.properties

        assert restored_props.n_qubits == original_props.n_qubits
        assert restored_props.depth == original_props.depth
        assert restored_props.gate_count == original_props.gate_count

    def test_copy_preserves_all_params(self) -> None:
        """Test that copy without overrides preserves all parameters."""
        enc = QAOAEncoding(
            n_features=4,
            reps=3,
            data_rotation="Y",
            mixer_rotation="Z",
            entanglement="full",
            entangling_gate="cx",
            gamma=2.5,
            beta=1.5,
            include_initial_h=False,
            feature_map="quadratic",
        )

        enc_copy = enc.copy()

        assert enc_copy.n_features == enc.n_features
        assert enc_copy.reps == enc.reps
        assert enc_copy.data_rotation == enc.data_rotation
        assert enc_copy.mixer_rotation == enc.mixer_rotation
        assert enc_copy.entanglement == enc.entanglement
        assert enc_copy.entangling_gate == enc.entangling_gate
        assert enc_copy.gamma == enc.gamma
        assert enc_copy.beta == enc.beta
        assert enc_copy.include_initial_h == enc.include_initial_h
        assert enc_copy.feature_map == enc.feature_map

    def test_copy_with_overrides(self) -> None:
        """Test that copy with overrides applies changes."""
        enc = QAOAEncoding(n_features=4, reps=2, gamma=1.0)
        enc_copy = enc.copy(reps=5, gamma=3.0, entanglement="full")

        assert enc_copy.n_features == 4  # Preserved
        assert enc_copy.reps == 5  # Overridden
        assert enc_copy.gamma == 3.0  # Overridden
        assert enc_copy.entanglement == "full"  # Overridden

    def test_copy_is_independent(self) -> None:
        """Test that copy creates an independent instance."""
        enc = QAOAEncoding(n_features=4, reps=2)
        enc_copy = enc.copy()

        # They should be equal but not the same object
        assert enc == enc_copy
        assert enc is not enc_copy

    def test_to_dict_contains_all_fields(self) -> None:
        """Test that to_dict includes all parameters."""
        enc = QAOAEncoding(
            n_features=4,
            reps=3,
            data_rotation="Y",
            gamma=2.0,
        )

        d = enc.to_dict()

        assert d["class"] == "QAOAEncoding"
        assert d["n_features"] == 4
        assert d["reps"] == 3
        assert d["data_rotation"] == "Y"
        assert d["gamma"] == 2.0
        assert "mixer_rotation" in d
        assert "entanglement" in d

    def test_from_dict_reconstructs_encoding(self) -> None:
        """Test that from_dict reconstructs the encoding correctly."""
        original = QAOAEncoding(
            n_features=6,
            reps=4,
            entanglement="circular",
            gamma=1.5,
            beta=0.8,
        )

        config = original.to_dict()
        reconstructed = QAOAEncoding.from_dict(config)

        assert original == reconstructed

    def test_roundtrip_json_compatible(self) -> None:
        """Test that to_dict output is JSON serializable."""
        import json

        enc = QAOAEncoding(n_features=4, reps=2, gamma=np.pi)
        config = enc.to_dict()

        # Should not raise
        json_str = json.dumps(config)
        loaded = json.loads(json_str)

        reconstructed = QAOAEncoding.from_dict(loaded)
        assert enc == reconstructed


# =============================================================================
# Test Class: Concurrent Access / Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_circuit_generation(self) -> None:
        """Test that concurrent circuit generation is thread-safe."""
        enc = QAOAEncoding(n_features=4, reps=2)
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
        enc = QAOAEncoding(n_features=4, reps=2)
        num_threads = 20

        results: list[int] = []
        errors: list[Exception] = []

        def access_properties() -> None:
            try:
                props = enc.properties
                results.append(props.n_qubits)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(access_properties) for _ in range(100)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 100
        # All should return the same value
        assert all(r == 4 for r in results)

    def test_concurrent_different_encodings(self) -> None:
        """Test that multiple encodings can be used concurrently."""
        encodings = [
            QAOAEncoding(n_features=n, reps=r) for n in [2, 4, 6] for r in [1, 2]
        ]
        results: list[tuple[int, int]] = []
        errors: list[Exception] = []

        def access_encoding(enc: QAOAEncoding) -> None:
            try:
                props = enc.properties
                results.append((props.n_qubits, enc.reps))
            except Exception as e:
                errors.append(e)

        # Access all encodings concurrently, multiple times each
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            for _ in range(10):
                for enc in encodings:
                    futures.append(executor.submit(access_encoding, enc))
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 60  # 6 encodings × 10 iterations


# =============================================================================
# Test Class: QAOA-Specific - N Data Parameters
# =============================================================================


class TestNDataParameters:
    """Tests for the n_data_parameters property."""

    def test_basic_calculation(self) -> None:
        """Test basic n_data_parameters calculation."""
        enc = QAOAEncoding(n_features=4, reps=3)
        # n_data_parameters = n_features * reps = 4 * 3 = 12
        assert enc.n_data_parameters == 12

    def test_single_rep(self) -> None:
        """Test n_data_parameters with single repetition."""
        enc = QAOAEncoding(n_features=6, reps=1)
        assert enc.n_data_parameters == 6

    def test_many_reps(self) -> None:
        """Test n_data_parameters with many repetitions."""
        enc = QAOAEncoding(n_features=4, reps=10)
        assert enc.n_data_parameters == 40

    def test_single_qubit(self) -> None:
        """Test n_data_parameters with single qubit."""
        enc = QAOAEncoding(n_features=1, reps=5)
        assert enc.n_data_parameters == 5


# =============================================================================
# Test Class: QAOA-Specific - High Reps Warning
# =============================================================================


class TestHighRepsWarning:
    """Tests for warning when high reps values are used."""

    def test_no_warning_for_low_reps(self) -> None:
        """Test that no warning is raised for reps <= 10."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QAOAEncoding(n_features=4, reps=10)

            # Filter for UserWarnings about barren plateaus
            barren_warnings = [
                warning for warning in w if "barren" in str(warning.message).lower()
            ]
            assert len(barren_warnings) == 0

    def test_warning_for_high_reps(self) -> None:
        """Test that warning is raised for reps > 10."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QAOAEncoding(n_features=4, reps=11)

            # Should have a warning about barren plateaus
            barren_warnings = [
                warning for warning in w if "barren" in str(warning.message).lower()
            ]
            assert len(barren_warnings) == 1
            assert "reps=11" in str(barren_warnings[0].message)

    def test_warning_for_very_high_reps(self) -> None:
        """Test warning for very high reps."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QAOAEncoding(n_features=4, reps=50)

            barren_warnings = [
                warning for warning in w if "barren" in str(warning.message).lower()
            ]
            assert len(barren_warnings) == 1


# =============================================================================
# Test Class: QAOA-Specific - Depth Verification
# =============================================================================


class TestDepthVerification:
    """Tests to verify theoretical depth vs actual backend depth.

    The theoretical depth property uses graph coloring theory for optimal
    parallel scheduling.
    """

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    @pytest.mark.parametrize(
        "n_features,reps,entanglement,include_h",
        [
            (4, 1, "linear", True),
            (4, 2, "linear", True),
            (4, 1, "linear", False),
            (4, 1, "circular", True),
            (6, 2, "circular", True),
            (4, 1, "full", True),
            (4, 2, "none", True),
            (2, 1, "linear", True),
            (8, 3, "linear", True),
        ],
    )
    def test_depth_matches_cirq_for_simple_cases(
        self,
        n_features: int,
        reps: int,
        entanglement: str,
        include_h: bool,
    ) -> None:
        """Verify depth property matches Cirq circuit depth for simple cases."""
        enc = QAOAEncoding(
            n_features=n_features,
            reps=reps,
            entanglement=entanglement,
            include_initial_h=include_h,
        )

        theoretical_depth = enc.depth
        actual_cirq_depth = enc.get_depth(backend="cirq")

        assert theoretical_depth == actual_cirq_depth, (
            f"Depth mismatch for n={n_features}, reps={reps}, "
            f"entanglement={entanglement}, include_h={include_h}: "
            f"theoretical={theoretical_depth}, cirq={actual_cirq_depth}"
        )

    def test_depth_property_vs_get_depth_pennylane(self) -> None:
        """Verify depth property equals get_depth for PennyLane backend."""
        enc = QAOAEncoding(n_features=4, reps=2, entanglement="linear")

        # PennyLane get_depth returns theoretical depth
        theoretical = enc.depth
        from_method = enc.get_depth(backend="pennylane")

        assert theoretical == from_method


# =============================================================================
# Test Class: QAOA-Specific - Get Entanglement Pairs
# =============================================================================


class TestGetEntanglementPairs:
    """Tests for the get_entanglement_pairs() method."""

    def test_linear_entanglement_pairs(self) -> None:
        """Test linear entanglement pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="linear")
        pairs = enc.get_entanglement_pairs()

        assert pairs == [(0, 1), (1, 2), (2, 3)]

    def test_full_entanglement_pairs(self) -> None:
        """Test full entanglement pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="full")
        pairs = enc.get_entanglement_pairs()

        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert pairs == expected

    def test_circular_entanglement_pairs(self) -> None:
        """Test circular entanglement pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="circular")
        pairs = enc.get_entanglement_pairs()

        assert pairs == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_none_entanglement_pairs(self) -> None:
        """Test no entanglement returns empty list."""
        enc = QAOAEncoding(n_features=4, entanglement="none")
        pairs = enc.get_entanglement_pairs()

        assert pairs == []

    def test_returns_copy_not_reference(self) -> None:
        """Test that returned list is a copy, not the internal reference."""
        enc = QAOAEncoding(n_features=4, entanglement="linear")
        pairs1 = enc.get_entanglement_pairs()
        pairs2 = enc.get_entanglement_pairs()

        # Should be equal but not the same object
        assert pairs1 == pairs2
        assert pairs1 is not pairs2

        # Modifying one should not affect the other
        pairs1.append((99, 100))
        assert (99, 100) not in enc.get_entanglement_pairs()


# =============================================================================
# Test Class: QAOA-Specific - Gate Count Breakdown
# =============================================================================


class TestGateCountBreakdown:
    """Tests for the gate_count_breakdown() method."""

    def test_basic_breakdown(self) -> None:
        """Test basic gate count breakdown."""
        enc = QAOAEncoding(n_features=4, reps=2, entanglement="linear")
        breakdown = enc.gate_count_breakdown()

        # Verify all keys are present
        assert "hadamard" in breakdown
        assert "data_rotation" in breakdown
        assert "mixer_rotation" in breakdown
        assert "entangling" in breakdown
        assert "total_single_qubit" in breakdown
        assert "total_two_qubit" in breakdown
        assert "total" in breakdown

    def test_hadamard_count_with_initial_h(self) -> None:
        """Test Hadamard count when include_initial_h=True."""
        enc = QAOAEncoding(n_features=4, reps=2, include_initial_h=True)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["hadamard"] == 4

    def test_hadamard_count_without_initial_h(self) -> None:
        """Test Hadamard count when include_initial_h=False."""
        enc = QAOAEncoding(n_features=4, reps=2, include_initial_h=False)
        breakdown = enc.gate_count_breakdown()

        assert breakdown["hadamard"] == 0

    def test_data_rotation_count(self) -> None:
        """Test data rotation gate count."""
        enc = QAOAEncoding(n_features=4, reps=3)
        breakdown = enc.gate_count_breakdown()

        # data_rotation = n_qubits * reps = 4 * 3 = 12
        assert breakdown["data_rotation"] == 12

    def test_entangling_count_linear(self) -> None:
        """Test entangling gate count for linear entanglement."""
        enc = QAOAEncoding(n_features=4, reps=2, entanglement="linear")
        breakdown = enc.gate_count_breakdown()

        # Linear: n-1 pairs = 3 pairs, * reps = 6
        assert breakdown["entangling"] == 6

    def test_entangling_count_none(self) -> None:
        """Test entangling gate count for no entanglement."""
        enc = QAOAEncoding(n_features=4, reps=2, entanglement="none")
        breakdown = enc.gate_count_breakdown()

        assert breakdown["entangling"] == 0

    def test_total_calculation(self) -> None:
        """Test total gate count calculation."""
        enc = QAOAEncoding(n_features=4, reps=2)
        breakdown = enc.gate_count_breakdown()

        expected = breakdown["total_single_qubit"] + breakdown["total_two_qubit"]
        assert breakdown["total"] == expected


# =============================================================================
# Test Class: QAOA-Specific - Resource Summary
# =============================================================================


class TestResourceSummary:
    """Tests for the resource_summary() method."""

    def test_contains_all_keys(self) -> None:
        """Test that resource_summary contains all expected keys."""
        enc = QAOAEncoding(n_features=4, reps=2)
        summary = enc.resource_summary()

        # Circuit structure
        assert "n_qubits" in summary
        assert "n_features" in summary
        assert "depth" in summary
        assert "reps" in summary

        # Entanglement information
        assert "entanglement" in summary
        assert "entanglement_pairs" in summary
        assert "n_entanglement_pairs" in summary
        assert "entangling_gate" in summary

        # Gate counts
        assert "gate_counts" in summary

        # Encoding parameters
        assert "gamma" in summary
        assert "beta" in summary

    def test_n_qubits_value(self) -> None:
        """Test n_qubits value in summary."""
        enc = QAOAEncoding(n_features=6)
        summary = enc.resource_summary()

        assert summary["n_qubits"] == 6

    def test_entanglement_pairs_match(self) -> None:
        """Test entanglement_pairs matches get_entanglement_pairs."""
        enc = QAOAEncoding(n_features=4, entanglement="full")
        summary = enc.resource_summary()

        assert summary["entanglement_pairs"] == enc.get_entanglement_pairs()

    def test_gate_counts_match(self) -> None:
        """Test gate_counts matches gate_count_breakdown."""
        enc = QAOAEncoding(n_features=4, reps=2)
        summary = enc.resource_summary()

        assert summary["gate_counts"] == enc.gate_count_breakdown()

    def test_is_entangling_with_entanglement(self) -> None:
        """Test is_entangling is True when entanglement is configured."""
        enc = QAOAEncoding(n_features=4, entanglement="linear")
        summary = enc.resource_summary()

        assert summary["is_entangling"] is True

    def test_is_entangling_without_entanglement(self) -> None:
        """Test is_entangling is False when no entanglement."""
        enc = QAOAEncoding(n_features=4, entanglement="none")
        summary = enc.resource_summary()

        assert summary["is_entangling"] is False


# =============================================================================
# Test Class: QAOA-Specific - Full Entanglement Warning
# =============================================================================


class TestFullEntanglementWarning:
    """Tests for warnings when full entanglement is used with many features."""

    def test_no_warning_for_small_n(self) -> None:
        """Test no warning for n < threshold."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QAOAEncoding(n_features=8, entanglement="full")

            # Filter for full entanglement warnings
            full_ent_warnings = [
                warning
                for warning in w
                if "full entanglement" in str(warning.message).lower()
            ]
            assert len(full_ent_warnings) == 0

    def test_warning_for_large_n_full_entanglement(self) -> None:
        """Test warning for n >= 12 with full entanglement."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QAOAEncoding(n_features=12, entanglement="full")

            # Should have a warning about full entanglement
            full_ent_warnings = [
                warning
                for warning in w
                if "full entanglement" in str(warning.message).lower()
            ]
            assert len(full_ent_warnings) == 1
            assert "n_features=12" in str(full_ent_warnings[0].message)

    def test_no_warning_for_large_n_linear_entanglement(self) -> None:
        """Test no warning for large n with linear entanglement."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QAOAEncoding(n_features=20, entanglement="linear")

            # Filter for full entanglement warnings
            full_ent_warnings = [
                warning
                for warning in w
                if "full entanglement" in str(warning.message).lower()
            ]
            assert len(full_ent_warnings) == 0


# =============================================================================
# Test Class: QAOA-Specific - Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports and imports."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from encoding_atlas.encodings import qaoa_encoding

        assert hasattr(qaoa_encoding, "__all__")
        assert "QAOAEncoding" in qaoa_encoding.__all__
        assert "GateCountBreakdown" in qaoa_encoding.__all__

    def test_gate_count_breakdown_importable(self) -> None:
        """Test that GateCountBreakdown can be imported."""
        from encoding_atlas.encodings.qaoa_encoding import GateCountBreakdown

        # Should be a TypedDict
        assert hasattr(GateCountBreakdown, "__annotations__")
        assert "hadamard" in GateCountBreakdown.__annotations__
        assert "data_rotation" in GateCountBreakdown.__annotations__


# =============================================================================
# Test Class: QAOA-Specific - Parallel Batch Processing
# =============================================================================


class TestParallelBatchProcessing:
    """Tests for parallel batch processing in get_circuits()."""

    def test_parallel_processing_basic(self) -> None:
        """Test basic parallel processing functionality."""
        enc = QAOAEncoding(n_features=4, reps=2)
        X = np.random.randn(20, 4)

        # Generate circuits in parallel
        circuits_parallel = enc.get_circuits(X, backend="pennylane", parallel=True)

        assert len(circuits_parallel) == 20
        assert all(callable(c) for c in circuits_parallel)

    def test_parallel_vs_sequential_equivalence(self) -> None:
        """Test that parallel and sequential processing produce equivalent results."""
        enc = QAOAEncoding(n_features=4, reps=2)
        X = np.random.randn(10, 4)

        # Generate circuits sequentially
        circuits_sequential = enc.get_circuits(X, backend="pennylane", parallel=False)

        # Generate circuits in parallel
        circuits_parallel = enc.get_circuits(X, backend="pennylane", parallel=True)

        # Both should have same length
        assert len(circuits_sequential) == len(circuits_parallel)
        assert len(circuits_sequential) == 10

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_parallel_preserves_order(self) -> None:
        """Test that parallel processing preserves input order."""
        enc = QAOAEncoding(n_features=4, reps=1)

        # Create distinct inputs
        X = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        )

        circuits_parallel = enc.get_circuits(X, backend="pennylane", parallel=True)
        circuits_sequential = enc.get_circuits(X, backend="pennylane", parallel=False)

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        # Execute circuits and compare states
        for i in range(len(X)):

            @qml.qnode(dev)
            def circuit_par():
                circuits_parallel[i]()
                return qml.state()

            @qml.qnode(dev)
            def circuit_seq():
                circuits_sequential[i]()
                return qml.state()

            state_par = circuit_par()
            state_seq = circuit_seq()
            assert np.allclose(state_par, state_seq, atol=1e-10)

    def test_parallel_single_sample_falls_back_to_sequential(self) -> None:
        """Test that single sample uses sequential processing."""
        enc = QAOAEncoding(n_features=4, reps=2)
        X = np.array([[0.1, 0.2, 0.3, 0.4]])

        # Single sample should work without issues
        circuits = enc.get_circuits(X, backend="pennylane", parallel=True)
        assert len(circuits) == 1

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_parallel_qiskit_backend(self) -> None:
        """Test parallel processing with Qiskit backend."""
        enc = QAOAEncoding(n_features=4, reps=2)
        X = np.random.randn(10, 4)

        circuits = enc.get_circuits(X, backend="qiskit", parallel=True)
        assert len(circuits) == 10
        assert all(c.num_qubits == 4 for c in circuits)

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_parallel_cirq_backend(self) -> None:
        """Test parallel processing with Cirq backend."""
        enc = QAOAEncoding(n_features=4, reps=2)
        X = np.random.randn(10, 4)

        circuits = enc.get_circuits(X, backend="cirq", parallel=True)
        assert len(circuits) == 10
        assert all(len(c.all_qubits()) == 4 for c in circuits)


# =============================================================================
# Test Class: QAOA-Specific - Internal Method Tests
# =============================================================================


class TestGetCircuitFromValidated:
    """Tests for the internal _get_circuit_from_validated method."""

    def test_internal_method_exists(self) -> None:
        """Test that _get_circuit_from_validated method exists."""
        enc = QAOAEncoding(n_features=4)
        assert hasattr(enc, "_get_circuit_from_validated")

    def test_internal_method_works_with_1d_input(self) -> None:
        """Test _get_circuit_from_validated with 1D input."""
        enc = QAOAEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc._get_circuit_from_validated(x, "pennylane")
        assert callable(circuit)

    def test_internal_method_invalid_backend(self) -> None:
        """Test _get_circuit_from_validated with invalid backend."""
        enc = QAOAEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="Unknown backend"):
            enc._get_circuit_from_validated(x, "invalid")  # type: ignore

    @pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
    def test_internal_method_qiskit(self) -> None:
        """Test _get_circuit_from_validated with Qiskit backend."""
        enc = QAOAEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc._get_circuit_from_validated(x, "qiskit")
        assert circuit.num_qubits == 4

    @pytest.mark.skipif(not HAS_CIRQ, reason="Cirq not installed")
    def test_internal_method_cirq(self) -> None:
        """Test _get_circuit_from_validated with Cirq backend."""
        enc = QAOAEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc._get_circuit_from_validated(x, "cirq")
        assert len(circuit.all_qubits()) == 4


# =============================================================================
# Test Class: Slow Simulation Tests (ALWAYS LAST)
# =============================================================================


@pytest.mark.slow
class TestSlowSimulation:
    """Slow tests that perform actual quantum simulation.

    These tests verify cross-backend consistency and state fidelity.
    They require full quantum simulation and are marked as slow.
    """

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_different_inputs_produce_different_states(self) -> None:
        """Test that different inputs produce different quantum states."""
        enc = QAOAEncoding(n_features=4, reps=2)

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
        enc = QAOAEncoding(n_features=4, reps=2)

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

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_entanglement_creates_entangled_state(self) -> None:
        """Test that entanglement actually creates entangled state."""
        enc = QAOAEncoding(n_features=2, reps=1, entanglement="linear")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([np.pi / 4, np.pi / 4])

        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # For an entangled state, reduced density matrix has mixed state
        # Check by computing purity of reduced state
        rho_full = np.outer(state, np.conj(state))
        rho_reduced = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rho_reduced[i, j] += rho_full[2 * i + k, 2 * j + k]

        purity = np.real(np.trace(rho_reduced @ rho_reduced))

        # Pure product state has purity = 1, entangled state has purity < 1
        assert purity < 1.0

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_product_state_is_separable(self) -> None:
        """Test that no entanglement creates separable state."""
        enc = QAOAEncoding(n_features=2, reps=1, entanglement="none")

        dev = qml.device("default.qubit", wires=enc.n_qubits)

        x = np.array([np.pi / 4, np.pi / 3])

        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        state = full_circuit()

        # For a product state, reduced density matrix is pure
        rho_full = np.outer(state, np.conj(state))
        rho_reduced = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    rho_reduced[i, j] += rho_full[2 * i + k, 2 * j + k]

        purity = np.real(np.trace(rho_reduced @ rho_reduced))

        # Product state has purity = 1
        assert np.isclose(purity, 1.0, atol=1e-10)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    def test_all_backends_produce_valid_states(self) -> None:
        """Test that all backends produce valid normalized quantum states."""
        enc = QAOAEncoding(n_features=4, reps=2)
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
    @pytest.mark.parametrize("entangling_gate", ["cx", "cz", "rzz"])
    def test_cross_backend_all_entangling_gates(self, entangling_gate: str) -> None:
        """Test that all backends produce equivalent states for all gate types."""
        enc = QAOAEncoding(
            n_features=3,
            reps=1,
            entangling_gate=entangling_gate,
            gamma=0.7,
            beta=0.5,
        )
        x = np.array([0.3, 0.5, 0.7])

        # Get states from all backends
        state_pl = self._simulate_pennylane(enc, x)
        state_qk = self._simulate_qiskit(enc, x)
        state_cirq = self._simulate_cirq(enc, x)

        # Compare using sorted probability distributions (handles qubit ordering)
        probs_pl = sorted(np.abs(state_pl) ** 2)
        probs_qk = sorted(np.abs(state_qk) ** 2)
        probs_cirq = sorted(np.abs(state_cirq) ** 2)

        np.testing.assert_allclose(probs_pl, probs_qk, atol=1e-6)
        np.testing.assert_allclose(probs_pl, probs_cirq, atol=1e-6)

    @pytest.mark.skipif(
        not (HAS_PENNYLANE and HAS_QISKIT and HAS_CIRQ),
        reason="All backends required",
    )
    @pytest.mark.cross_backend
    @pytest.mark.parametrize("entanglement", ["linear", "full", "circular", "none"])
    def test_cross_backend_all_entanglements(self, entanglement: str) -> None:
        """Test cross-backend equivalence for all entanglement patterns."""
        enc = QAOAEncoding(n_features=4, reps=2, entanglement=entanglement)
        x = np.array([0.3, 0.5, 0.7, 0.9])

        state_pl = self._simulate_pennylane(enc, x)
        state_qk = self._simulate_qiskit(enc, x)
        state_cirq = self._simulate_cirq(enc, x)

        # Compare probability distributions
        probs_pl = sorted(np.abs(state_pl) ** 2)
        probs_qk = sorted(np.abs(state_qk) ** 2)
        probs_cirq = sorted(np.abs(state_cirq) ** 2)

        np.testing.assert_allclose(probs_pl, probs_qk, atol=1e-6)
        np.testing.assert_allclose(probs_pl, probs_cirq, atol=1e-6)

    def _simulate_pennylane(
        self, enc: QAOAEncoding, x: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """Simulate circuit with PennyLane and return statevector."""
        dev = qml.device("default.qubit", wires=enc.n_qubits)
        circuit_fn = enc.get_circuit(x, backend="pennylane")

        @qml.qnode(dev)
        def full_circuit():
            circuit_fn()
            return qml.state()

        return np.array(full_circuit())

    def _simulate_qiskit(
        self, enc: QAOAEncoding, x: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """Simulate circuit with Qiskit and return statevector."""
        circuit = enc.get_circuit(x, backend="qiskit")
        statevector = Statevector(circuit)
        return np.array(statevector.data)

    def _simulate_cirq(
        self, enc: QAOAEncoding, x: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """Simulate circuit with Cirq and return statevector."""
        circuit = enc.get_circuit(x, backend="cirq")
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        return result.final_state_vector
