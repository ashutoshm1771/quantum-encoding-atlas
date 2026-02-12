"""Reference implementation comparison tests.

This test suite compares our encoding implementations against Qiskit's
built-in feature maps to validate mathematical correctness.

Test Categories:
1. ZZFeatureMap comparison with qiskit.circuit.library.ZZFeatureMap
2. PauliFeatureMap comparison with qiskit.circuit.library.PauliFeatureMap
3. Various parameter configurations
4. Edge cases and boundary conditions
"""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import PauliFeatureMap, ZZFeatureMap

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def qiskit_statevector():
    """Import Qiskit Statevector."""
    qiskit_quantum_info = pytest.importorskip("qiskit.quantum_info")
    return qiskit_quantum_info.Statevector


@pytest.fixture
def qiskit_zz_feature_map():
    """Import Qiskit's ZZFeatureMap."""
    qiskit_library = pytest.importorskip("qiskit.circuit.library")
    return qiskit_library.ZZFeatureMap


@pytest.fixture
def qiskit_pauli_feature_map():
    """Import Qiskit's PauliFeatureMap."""
    qiskit_library = pytest.importorskip("qiskit.circuit.library")
    return qiskit_library.PauliFeatureMap


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute fidelity between two quantum states.

    Handles global phase differences by comparing |<s1|s2>|^2.
    """
    return float(np.abs(np.vdot(state1, state2)) ** 2)


# =============================================================================
# ZZ FEATURE MAP REFERENCE COMPARISON
# =============================================================================


class TestZZFeatureMapReference:
    """Compare ZZFeatureMap against Qiskit's reference implementation."""

    def test_zz_matches_qiskit_basic(
        self, qiskit_statevector, qiskit_zz_feature_map
    ) -> None:
        """Verify ZZFeatureMap produces equivalent states to Qiskit's implementation."""
        # Our implementation
        our_enc = ZZFeatureMap(n_features=4, reps=2, entanglement="full")

        # Qiskit's implementation
        qiskit_enc = qiskit_zz_feature_map(
            feature_dimension=4,
            reps=2,
            entanglement="full",
        )

        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Get our circuit and simulate
        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        # Bind parameters to Qiskit's circuit
        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        # Compare states (up to global phase)
        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low vs Qiskit reference"

    def test_zz_matches_qiskit_linear_entanglement(
        self, qiskit_statevector, qiskit_zz_feature_map
    ) -> None:
        """Test ZZFeatureMap with linear entanglement."""
        our_enc = ZZFeatureMap(n_features=4, reps=1, entanglement="linear")

        qiskit_enc = qiskit_zz_feature_map(
            feature_dimension=4,
            reps=1,
            entanglement="linear",
        )

        x = np.array([0.5, 0.6, 0.7, 0.8])

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low"

    @pytest.mark.parametrize("n_features", [2, 3, 4, 5])
    def test_zz_matches_qiskit_various_sizes(
        self, qiskit_statevector, qiskit_zz_feature_map, n_features: int
    ) -> None:
        """Test ZZFeatureMap matches Qiskit for various feature sizes."""
        our_enc = ZZFeatureMap(n_features=n_features, reps=1, entanglement="full")

        qiskit_enc = qiskit_zz_feature_map(
            feature_dimension=n_features,
            reps=1,
            entanglement="full",
        )

        rng = np.random.default_rng(seed=n_features * 42)
        x = rng.uniform(-np.pi, np.pi, size=n_features)

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low for n={n_features}"

    @pytest.mark.parametrize("reps", [1, 2, 3])
    def test_zz_matches_qiskit_various_reps(
        self, qiskit_statevector, qiskit_zz_feature_map, reps: int
    ) -> None:
        """Test ZZFeatureMap matches Qiskit for various repetition counts."""
        our_enc = ZZFeatureMap(n_features=3, reps=reps, entanglement="full")

        qiskit_enc = qiskit_zz_feature_map(
            feature_dimension=3,
            reps=reps,
            entanglement="full",
        )

        x = np.array([0.3, 0.6, 0.9])

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low for reps={reps}"

    def test_zz_matches_qiskit_extreme_values(
        self, qiskit_statevector, qiskit_zz_feature_map
    ) -> None:
        """Test ZZFeatureMap with extreme input values."""
        our_enc = ZZFeatureMap(n_features=3, reps=1)

        qiskit_enc = qiskit_zz_feature_map(
            feature_dimension=3,
            reps=1,
        )

        # Large values (multiple rotations)
        x = np.array([10.0, -5.0, 15.0])

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low for extreme values"

    def test_zz_matches_qiskit_zero_input(
        self, qiskit_statevector, qiskit_zz_feature_map
    ) -> None:
        """Test ZZFeatureMap with zero inputs."""
        our_enc = ZZFeatureMap(n_features=3, reps=1)

        qiskit_enc = qiskit_zz_feature_map(
            feature_dimension=3,
            reps=1,
        )

        x = np.array([0.0, 0.0, 0.0])

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low for zero input"


# =============================================================================
# PAULI FEATURE MAP REFERENCE COMPARISON
# =============================================================================


class TestPauliFeatureMapReference:
    """Compare PauliFeatureMap against Qiskit's reference implementation."""

    def test_pauli_matches_qiskit_z_only(
        self, qiskit_statevector, qiskit_pauli_feature_map
    ) -> None:
        """Verify PauliFeatureMap with Z paulis matches Qiskit."""
        our_enc = PauliFeatureMap(n_features=3, reps=1, paulis=["Z"])

        qiskit_enc = qiskit_pauli_feature_map(
            feature_dimension=3,
            reps=1,
            paulis=["Z"],
        )

        x = np.array([0.5, 0.6, 0.7])

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low vs Qiskit reference"

    def test_pauli_matches_qiskit_zz(
        self, qiskit_statevector, qiskit_pauli_feature_map
    ) -> None:
        """Verify PauliFeatureMap with ZZ paulis matches Qiskit."""
        our_enc = PauliFeatureMap(n_features=3, reps=1, paulis=["Z", "ZZ"])

        qiskit_enc = qiskit_pauli_feature_map(
            feature_dimension=3,
            reps=1,
            paulis=["Z", "ZZ"],
        )

        x = np.array([0.5, 0.6, 0.7])

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low"

    @pytest.mark.parametrize("n_features", [2, 3, 4])
    def test_pauli_matches_qiskit_various_sizes(
        self, qiskit_statevector, qiskit_pauli_feature_map, n_features: int
    ) -> None:
        """Test PauliFeatureMap matches Qiskit for various sizes."""
        our_enc = PauliFeatureMap(n_features=n_features, reps=1, paulis=["Z", "ZZ"])

        qiskit_enc = qiskit_pauli_feature_map(
            feature_dimension=n_features,
            reps=1,
            paulis=["Z", "ZZ"],
        )

        rng = np.random.default_rng(seed=n_features * 123)
        x = rng.uniform(-np.pi, np.pi, size=n_features)

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low for n={n_features}"

    @pytest.mark.parametrize("reps", [1, 2])
    def test_pauli_matches_qiskit_various_reps(
        self, qiskit_statevector, qiskit_pauli_feature_map, reps: int
    ) -> None:
        """Test PauliFeatureMap matches Qiskit for various reps."""
        our_enc = PauliFeatureMap(n_features=3, reps=reps, paulis=["Z"])

        qiskit_enc = qiskit_pauli_feature_map(
            feature_dimension=3,
            reps=reps,
            paulis=["Z"],
        )

        x = np.array([0.2, 0.4, 0.6])

        our_circuit = our_enc.get_circuit(x, backend="qiskit")
        our_state = qiskit_statevector(our_circuit).data

        param_dict = dict(zip(qiskit_enc.parameters, x))
        qiskit_bound = qiskit_enc.assign_parameters(param_dict)
        qiskit_state = qiskit_statevector(qiskit_bound).data

        fidelity = compute_state_fidelity(our_state, qiskit_state)
        assert fidelity > 0.99, f"Fidelity {fidelity:.6f} too low for reps={reps}"


# =============================================================================
# CROSS-VALIDATION TESTS
# =============================================================================


class TestCrossValidation:
    """Cross-validate our implementations using multiple approaches."""

    def test_zz_state_normalization(self, qiskit_statevector) -> None:
        """Verify ZZFeatureMap always produces normalized states."""
        enc = ZZFeatureMap(n_features=4, reps=2)

        rng = np.random.default_rng(seed=42)
        for _ in range(10):
            x = rng.uniform(-10, 10, size=4)
            circuit = enc.get_circuit(x, backend="qiskit")
            state = qiskit_statevector(circuit).data
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

    def test_pauli_state_normalization(self, qiskit_statevector) -> None:
        """Verify PauliFeatureMap always produces normalized states."""
        enc = PauliFeatureMap(n_features=4, reps=2, paulis=["Z", "ZZ"])

        rng = np.random.default_rng(seed=42)
        for _ in range(10):
            x = rng.uniform(-10, 10, size=4)
            circuit = enc.get_circuit(x, backend="qiskit")
            state = qiskit_statevector(circuit).data
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized: {norm}"

    def test_deterministic_output(self, qiskit_statevector) -> None:
        """Verify same input produces same output."""
        enc = ZZFeatureMap(n_features=3, reps=2)
        x = np.array([0.1, 0.2, 0.3])

        circuit1 = enc.get_circuit(x, backend="qiskit")
        circuit2 = enc.get_circuit(x, backend="qiskit")

        state1 = qiskit_statevector(circuit1).data
        state2 = qiskit_statevector(circuit2).data

        np.testing.assert_allclose(state1, state2, atol=1e-14)

    def test_different_inputs_different_outputs(self, qiskit_statevector) -> None:
        """Verify different inputs produce different outputs."""
        enc = ZZFeatureMap(n_features=3, reps=1)

        x1 = np.array([0.1, 0.2, 0.3])
        x2 = np.array([0.4, 0.5, 0.6])

        circuit1 = enc.get_circuit(x1, backend="qiskit")
        circuit2 = enc.get_circuit(x2, backend="qiskit")

        state1 = qiskit_statevector(circuit1).data
        state2 = qiskit_statevector(circuit2).data

        fidelity = compute_state_fidelity(state1, state2)
        assert fidelity < 0.99, "Different inputs should produce different states"
