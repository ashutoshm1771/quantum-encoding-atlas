"""Cross-backend consistency tests for quantum encodings.

This module verifies that different simulation backends (PennyLane, Qiskit, Cirq)
produce consistent results for the same encodings. This is critical for ensuring
that analysis functions (expressibility, entanglement, trainability) produce
reliable results regardless of which backend is used.

The tests focus on:
1. Qubit ordering convention consistency (MSB = qubit 0)
2. Statevector fidelity between backends
3. Observable expectation value consistency
4. Gradient computation consistency

These tests require both PennyLane and Qiskit to be installed. Tests are
skipped if the required backends are not available.

Example
-------
Run these tests with pytest:

    pytest tests/integration/test_cross_backend_consistency.py -v

Notes
-----
These tests are marked as integration tests and may take longer to run
than unit tests due to circuit simulation overhead.
"""

import numpy as np
import pytest

# Try to import backends - tests will be skipped if not available
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


# Skip markers for tests requiring specific backends
requires_pennylane = pytest.mark.skipif(
    not HAS_PENNYLANE, reason="PennyLane not installed"
)
requires_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
requires_both_backends = pytest.mark.skipif(
    not (HAS_PENNYLANE and HAS_QISKIT),
    reason="Both PennyLane and Qiskit are required for cross-backend tests",
)


@pytest.fixture
def sample_input_2q():
    """Sample input for 2-qubit encodings."""
    return np.array([0.5, 1.0])


@pytest.fixture
def sample_input_4q():
    """Sample input for 4-qubit encodings."""
    return np.array([0.25, 0.5, 0.75, 1.0])


@requires_both_backends
class TestCrossBackendStatevectorConsistency:
    """Test that PennyLane and Qiskit produce consistent statevectors.

    These tests verify that the qubit ordering convention is consistent
    between backends, which is essential for reliable analysis results.
    """

    def test_angle_encoding_statevector_consistency(self, sample_input_2q):
        """Test AngleEncoding produces consistent statevectors across backends.

        AngleEncoding applies single-qubit rotations, which should produce
        identical statevectors (up to numerical precision) in both backends.
        """
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = AngleEncoding(n_features=2)
        x = sample_input_2q

        # Get statevectors from both backends
        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        # Compute fidelity between the two statevectors
        # F = |<ψ_pl|ψ_qk>|² should be very close to 1.0
        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"Statevector fidelity between PennyLane and Qiskit is {fidelity:.6f}, "
            f"expected > 0.9999. This may indicate a qubit ordering mismatch."
        )

    def test_iqp_encoding_statevector_consistency(self, sample_input_2q):
        """Test IQPEncoding produces consistent statevectors across backends.

        IQPEncoding includes entangling gates (CZ), which makes it a good
        test for verifying that two-qubit gate conventions are consistent.
        """
        from encoding_atlas import IQPEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = IQPEncoding(n_features=2, reps=1)
        x = sample_input_2q

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"IQP statevector fidelity between backends is {fidelity:.6f}. "
            f"This may indicate an entangling gate convention mismatch."
        )

    def test_amplitude_encoding_statevector_consistency(self):
        """Test AmplitudeEncoding produces consistent statevectors across backends.

        AmplitudeEncoding directly sets amplitudes rather than applying
        individual gates, so it specifically tests that the MSB/LSB qubit
        ordering conversion in _to_qiskit() is correct.  Without the
        conversion, amplitude indices would be scrambled by the
        simulation utility's _reverse_qubit_order() call.
        """
        from encoding_atlas import AmplitudeEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.5, 0.3, 0.7, 0.1])

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        # Verify element-wise match (not just sorted probabilities)
        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"AmplitudeEncoding statevector fidelity between PennyLane and "
            f"Qiskit is {fidelity:.6f}, expected > 0.9999. "
            f"This may indicate that _to_qiskit() is not correctly "
            f"converting amplitude indices from MSB to LSB ordering."
        )

        # Additionally verify that the probability distributions match
        # at each index (unsorted), not just as a set
        probs_pl = np.abs(sv_pennylane) ** 2
        probs_qk = np.abs(sv_qiskit) ** 2
        np.testing.assert_allclose(
            probs_pl,
            probs_qk,
            atol=1e-6,
            err_msg="Per-index probability mismatch between PennyLane and Qiskit",
        )

    def test_zz_feature_map_statevector_consistency(self, sample_input_2q):
        """Test ZZFeatureMap produces consistent statevectors across backends.

        ZZFeatureMap uses CNOT gates for entanglement, providing another
        test case for two-qubit gate consistency.
        """
        from encoding_atlas import ZZFeatureMap
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = ZZFeatureMap(n_features=2, reps=1)
        x = sample_input_2q

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"ZZFeatureMap statevector fidelity is {fidelity:.6f}. "
            f"CNOT gate conventions may differ between backends."
        )

    def test_cyclic_equivariant_statevector_consistency(self, sample_input_4q):
        """Test CyclicEquivariantFeatureMap produces consistent statevectors.

        This encoding uses RZZ entangling gates whose parameter convention
        differs between PennyLane (IsingZZ), Qiskit (RZZ), and Cirq
        (ZZPowGate).  A previous bug applied a spurious 2x factor in the
        Qiskit and Cirq backends; this test guards against regressions.
        """
        from encoding_atlas.analysis._utils import simulate_encoding_statevector
        from encoding_atlas.encodings.equivariant_feature_map import (
            CyclicEquivariantFeatureMap,
        )

        enc = CyclicEquivariantFeatureMap(n_features=4, reps=1)
        x = sample_input_4q

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"CyclicEquivariantFeatureMap statevector fidelity between "
            f"PennyLane and Qiskit is {fidelity:.6f}, expected > 0.9999. "
            f"This may indicate an RZZ parameter convention mismatch."
        )

    @pytest.mark.parametrize(
        "coupling_strength",
        [np.pi / 8, np.pi / 4, np.pi / 2, np.pi],
        ids=["pi/8", "pi/4", "pi/2", "pi"],
    )
    def test_cyclic_equivariant_consistency_varying_coupling(
        self, sample_input_4q, coupling_strength
    ):
        """Test cross-backend consistency across a range of coupling strengths.

        Verifies that the RZZ convention fix holds for all coupling strengths,
        not just the default value.
        """
        from encoding_atlas.analysis._utils import simulate_encoding_statevector
        from encoding_atlas.encodings.equivariant_feature_map import (
            CyclicEquivariantFeatureMap,
        )

        enc = CyclicEquivariantFeatureMap(
            n_features=4, reps=1, coupling_strength=coupling_strength
        )
        x = sample_input_4q

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"Fidelity {fidelity:.6f} at coupling_strength={coupling_strength:.4f}. "
            f"RZZ parameter conversion may be incorrect."
        )


@requires_both_backends
class TestCrossBackendQubitOrderingConsistency:
    """Test qubit ordering convention consistency.

    The MSB (most significant bit) convention should be consistent:
    - Qubit 0 = leftmost bit = most significant bit
    - Index 0 = |00...0⟩, Index 1 = |00...1⟩, etc.
    """

    def test_first_qubit_state_consistency(self, sample_input_2q):
        """Test that 'first qubit' means the same thing in both backends.

        This test applies a rotation only to qubit 0 and verifies that
        both backends agree on which amplitudes are affected.
        """
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        # Create encoding that applies RY only to first qubit
        # by setting second feature to 0
        x = np.array([np.pi / 2, 0.0])  # RY(π/2) on qubit 0, identity on qubit 1

        enc = AngleEncoding(n_features=2, rotation="Y")

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        # After RY(π/2) on qubit 0 from |00⟩:
        # |ψ⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
        # In MSB convention: indices 0 and 2 should have amplitude 1/√2

        expected_nonzero_indices = {0, 2}  # |00⟩ and |10⟩

        for idx in range(4):
            prob_pl = np.abs(sv_pennylane[idx]) ** 2
            prob_qk = np.abs(sv_qiskit[idx]) ** 2

            if idx in expected_nonzero_indices:
                assert prob_pl > 0.4, f"PennyLane: Index {idx} should have prob ~0.5"
                assert prob_qk > 0.4, f"Qiskit: Index {idx} should have prob ~0.5"
            else:
                assert prob_pl < 0.01, f"PennyLane: Index {idx} should have prob ~0"
                assert prob_qk < 0.01, f"Qiskit: Index {idx} should have prob ~0"

    def test_second_qubit_state_consistency(self, sample_input_2q):
        """Test that 'second qubit' (LSB) is consistent between backends."""
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        # Apply RY only to qubit 1 (second qubit, LSB)
        x = np.array([0.0, np.pi / 2])  # Identity on qubit 0, RY(π/2) on qubit 1

        enc = AngleEncoding(n_features=2, rotation="Y")

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        # After RY(π/2) on qubit 1 from |00⟩:
        # |ψ⟩ = |0⟩ ⊗ (|0⟩ + |1⟩)/√2 = (|00⟩ + |01⟩)/√2
        # In MSB convention: indices 0 and 1 should have amplitude 1/√2

        expected_nonzero_indices = {0, 1}  # |00⟩ and |01⟩

        for idx in range(4):
            prob_pl = np.abs(sv_pennylane[idx]) ** 2
            prob_qk = np.abs(sv_qiskit[idx]) ** 2

            if idx in expected_nonzero_indices:
                assert prob_pl > 0.4, f"PennyLane: Index {idx} should have prob ~0.5"
                assert prob_qk > 0.4, f"Qiskit: Index {idx} should have prob ~0.5"
            else:
                assert prob_pl < 0.01, f"PennyLane: Index {idx} should have prob ~0"
                assert prob_qk < 0.01, f"Qiskit: Index {idx} should have prob ~0"


@requires_both_backends
class TestCrossBackendObservableConsistency:
    """Test that observable expectations are consistent across backends."""

    def test_local_z_expectation_consistency(self):
        """Test ⟨Z₀⟩ (local Z on first qubit) consistency."""
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis._utils import (
            _compute_local_z_expectation,
            simulate_encoding_statevector,
        )

        enc = AngleEncoding(n_features=2, rotation="Y")

        # Test with state that has known ⟨Z₀⟩
        # RY(π) on qubit 0 gives |1⟩ state on qubit 0, so ⟨Z₀⟩ = -1
        x = np.array([np.pi, 0.0])

        sv_pl = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qk = simulate_encoding_statevector(enc, x, backend="qiskit")

        z0_pl = _compute_local_z_expectation(sv_pl, n_qubits=2)
        z0_qk = _compute_local_z_expectation(sv_qk, n_qubits=2)

        assert np.isclose(z0_pl, -1.0, atol=0.01), f"PennyLane ⟨Z₀⟩ = {z0_pl}"
        assert np.isclose(z0_qk, -1.0, atol=0.01), f"Qiskit ⟨Z₀⟩ = {z0_qk}"
        assert np.isclose(
            z0_pl, z0_qk, atol=0.001
        ), f"⟨Z₀⟩ mismatch: PennyLane={z0_pl}, Qiskit={z0_qk}"

    def test_global_z_expectation_consistency(self):
        """Test ⟨Z⊗Z⟩ (global Z) consistency."""
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = AngleEncoding(n_features=2, rotation="Y")
        x = np.array([np.pi / 4, np.pi / 4])

        sv_pl = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qk = simulate_encoding_statevector(enc, x, backend="qiskit")

        def compute_global_z(sv):
            """Compute ⟨Z⊗Z⟩."""
            n = len(sv)
            eigenvalues = np.array(
                [1.0 - 2.0 * (bin(i).count("1") % 2) for i in range(n)]
            )
            return float(np.real(np.sum(eigenvalues * np.abs(sv) ** 2)))

        zz_pl = compute_global_z(sv_pl)
        zz_qk = compute_global_z(sv_qk)

        assert np.isclose(
            zz_pl, zz_qk, atol=0.001
        ), f"⟨Z⊗Z⟩ mismatch: PennyLane={zz_pl}, Qiskit={zz_qk}"


@requires_both_backends
class TestCrossBackendDataReuploadingConsistency:
    """Test DataReuploading produces consistent results across backends.

    DataReuploading applies repeated layers of RY data-encoding rotations
    interleaved with CNOT entangling ladders.  This tests that the layered
    structure and cyclic feature mapping are handled identically by both
    PennyLane and Qiskit.
    """

    def test_data_reuploading_statevector_consistency(self, sample_input_4q):
        """Test basic statevector fidelity between PennyLane and Qiskit."""
        from encoding_atlas import DataReuploading
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = DataReuploading(n_features=4, n_layers=3)
        x = sample_input_4q

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"DataReuploading statevector fidelity between PennyLane and "
            f"Qiskit is {fidelity:.6f}, expected > 0.9999."
        )

        probs_pl = np.abs(sv_pennylane) ** 2
        probs_qk = np.abs(sv_qiskit) ** 2
        np.testing.assert_allclose(
            probs_pl,
            probs_qk,
            atol=1e-6,
            err_msg="Per-index probability mismatch between PennyLane and Qiskit",
        )

    def test_data_reuploading_single_layer_consistency(self, sample_input_2q):
        """Test single-layer DataReuploading consistency across backends."""
        from encoding_atlas import DataReuploading
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = DataReuploading(n_features=2, n_layers=1)
        x = sample_input_2q

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert (
            fidelity > 0.9999
        ), f"Single-layer DataReuploading fidelity is {fidelity:.6f}."

    def test_data_reuploading_cyclic_mapping_consistency(self):
        """Test cyclic feature mapping (n_features > n_qubits) consistency.

        When more features than qubits are used, features are mapped
        cyclically to qubits (feature i -> qubit i % n_qubits).  This
        must be handled identically across backends.
        """
        from encoding_atlas import DataReuploading
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = DataReuploading(n_features=6, n_qubits=3, n_layers=2)
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert fidelity > 0.9999, (
            f"Cyclic DataReuploading (6 features, 3 qubits) fidelity "
            f"is {fidelity:.6f}, expected > 0.9999."
        )

    @pytest.mark.parametrize("n_layers", [1, 2, 5], ids=["L=1", "L=2", "L=5"])
    def test_data_reuploading_consistency_varying_layers(
        self, sample_input_4q, n_layers
    ):
        """Test cross-backend consistency across different layer counts.

        More layers mean more repeated encoding + entangling blocks, which
        amplifies any convention mismatch between backends.
        """
        from encoding_atlas import DataReuploading
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = DataReuploading(n_features=4, n_layers=n_layers)
        x = sample_input_4q

        sv_pennylane = simulate_encoding_statevector(enc, x, backend="pennylane")
        sv_qiskit = simulate_encoding_statevector(enc, x, backend="qiskit")

        fidelity = np.abs(np.vdot(sv_pennylane, sv_qiskit)) ** 2

        assert (
            fidelity > 0.9999
        ), f"DataReuploading (L={n_layers}) fidelity is {fidelity:.6f}."


@requires_both_backends
class TestCrossBackendAnalysisConsistency:
    """Test that analysis functions produce consistent results across backends."""

    @pytest.mark.slow
    def test_expressibility_consistency(self, sample_input_2q):
        """Test expressibility is consistent across backends.

        This is a statistical test, so we use a relaxed tolerance.
        """
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis import compute_expressibility

        enc = AngleEncoding(n_features=2)

        # Compute expressibility with both backends
        # Use same seed for reproducibility
        result_pl = compute_expressibility(
            enc, n_samples=100, seed=42, backend="pennylane"
        )
        result_qk = compute_expressibility(
            enc, n_samples=100, seed=42, backend="qiskit"
        )

        # Results should be similar (within statistical error)
        assert np.isclose(result_pl, result_qk, atol=0.1), (
            f"Expressibility mismatch: PennyLane={result_pl:.4f}, "
            f"Qiskit={result_qk:.4f}"
        )

    @pytest.mark.slow
    def test_entanglement_consistency(self, sample_input_2q):
        """Test entanglement capability is consistent across backends."""
        from encoding_atlas import IQPEncoding
        from encoding_atlas.analysis import compute_entanglement_capability

        enc = IQPEncoding(n_features=2, reps=1)

        # compute_entanglement_capability returns a float by default
        result_pl = compute_entanglement_capability(
            enc, n_samples=50, seed=42, backend="pennylane"
        )
        result_qk = compute_entanglement_capability(
            enc, n_samples=50, seed=42, backend="qiskit"
        )

        # For entanglement, we expect very close results since it's
        # a direct computation from the statevector
        assert np.isclose(result_pl, result_qk, atol=0.05), (
            f"Entanglement mismatch: PennyLane={result_pl:.4f}, "
            f"Qiskit={result_qk:.4f}"
        )
