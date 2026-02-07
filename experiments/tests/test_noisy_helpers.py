"""Tests for noisy analysis helper functions.

This module tests the internal helper functions in
:mod:`experiments.noisy_analysis` that are used by the public API
(``compute_noisy_expressibility``, ``compute_noisy_entanglement``,
``compute_fidelity_decay``) but have no dedicated test coverage:

- ``partial_trace_single_qubit`` — used by noisy entanglement
- ``_is_density_matrix_invalid`` — used by all three public functions
- ``compute_pure_mixed_fidelity`` — used by fidelity decay
- ``_compute_haar_distribution`` — used by noisy expressibility

Run with: pytest experiments/tests/test_noisy_helpers.py -v
"""

import numpy as np
import pytest

from experiments.noisy_analysis import (
    _compute_haar_distribution,
    _is_density_matrix_invalid,
    compute_linear_entropy,
    compute_mixed_state_fidelity,
    compute_pure_mixed_fidelity,
    partial_trace_single_qubit,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def bell_state_dm():
    """Density matrix for |Phi+> = (|00> + |11>) / sqrt(2)."""
    bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    return np.outer(bell, bell.conj())


@pytest.fixture
def ghz3_dm():
    """Density matrix for 3-qubit GHZ = (|000> + |111>) / sqrt(2)."""
    ghz = np.zeros(8, dtype=np.complex128)
    ghz[0] = 1.0 / np.sqrt(2)
    ghz[7] = 1.0 / np.sqrt(2)
    return np.outer(ghz, ghz.conj())


@pytest.fixture
def maximally_mixed_2q():
    """Maximally mixed state I/4 for 2 qubits."""
    return np.eye(4, dtype=np.complex128) / 4


# =============================================================================
# partial_trace_single_qubit
# =============================================================================


class TestPartialTrace:
    """Tests for partial trace of multi-qubit density matrices."""

    def test_product_state_00(self):
        """Tracing out either qubit of |00><00| yields |0><0|."""
        rho = np.zeros((4, 4), dtype=np.complex128)
        rho[0, 0] = 1.0

        for qubit in [0, 1]:
            rho_q = partial_trace_single_qubit(rho, n_qubits=2, keep_qubit=qubit)
            assert rho_q.shape == (2, 2)
            assert np.isclose(rho_q[0, 0], 1.0)
            assert np.isclose(rho_q[1, 1], 0.0)

    def test_product_state_01(self):
        """Tracing |01><01|: qubit 0 -> |0><0|, qubit 1 -> |1><1|."""
        state = np.array([0, 1, 0, 0], dtype=np.complex128)
        rho = np.outer(state, state.conj())

        rho_0 = partial_trace_single_qubit(rho, n_qubits=2, keep_qubit=0)
        assert np.isclose(rho_0[0, 0], 1.0)
        assert np.isclose(rho_0[1, 1], 0.0)

        rho_1 = partial_trace_single_qubit(rho, n_qubits=2, keep_qubit=1)
        assert np.isclose(rho_1[0, 0], 0.0)
        assert np.isclose(rho_1[1, 1], 1.0)

    def test_bell_state_gives_maximally_mixed(self, bell_state_dm):
        """Partial trace of Bell state yields I/2 for both qubits."""
        for qubit in [0, 1]:
            rho_q = partial_trace_single_qubit(
                bell_state_dm, n_qubits=2, keep_qubit=qubit
            )
            assert np.allclose(rho_q, np.eye(2) / 2, atol=1e-10)

    def test_ghz_state_gives_maximally_mixed(self, ghz3_dm):
        """Partial trace of 3-qubit GHZ yields I/2 for each qubit."""
        for qubit in range(3):
            rho_q = partial_trace_single_qubit(
                ghz3_dm, n_qubits=3, keep_qubit=qubit
            )
            assert rho_q.shape == (2, 2)
            assert np.allclose(rho_q, np.eye(2) / 2, atol=1e-10)

    def test_trace_preserved_random_state(self):
        """Reduced density matrix must have Tr = 1 for a random pure state."""
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

        for qubit in range(3):
            rho_q = partial_trace_single_qubit(rho, n_qubits=3, keep_qubit=qubit)
            assert np.isclose(np.trace(rho_q), 1.0, atol=1e-10)

    def test_hermiticity_preserved(self, bell_state_dm):
        """Reduced state must be Hermitian."""
        rho_q = partial_trace_single_qubit(bell_state_dm, n_qubits=2, keep_qubit=0)
        assert np.allclose(rho_q, rho_q.conj().T, atol=1e-12)

    def test_positive_semidefinite(self):
        """Reduced state must have non-negative eigenvalues."""
        rng = np.random.default_rng(99)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

        for qubit in range(4):
            rho_q = partial_trace_single_qubit(rho, n_qubits=4, keep_qubit=qubit)
            eigenvalues = np.linalg.eigvalsh(rho_q)
            assert np.all(eigenvalues >= -1e-12)

    def test_wrong_shape_raises(self):
        """Density matrix with inconsistent shape must raise ValueError."""
        rho = np.eye(3, dtype=np.complex128)
        with pytest.raises(ValueError, match="doesn't match"):
            partial_trace_single_qubit(rho, n_qubits=2, keep_qubit=0)

    def test_separable_state(self):
        """For |0> tensor |1>, tracing is exact."""
        # |0> tensor |1> = [0, 1, 0, 0]
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        psi = np.kron(psi_0, psi_1)
        rho = np.outer(psi, psi.conj())

        rho_0 = partial_trace_single_qubit(rho, n_qubits=2, keep_qubit=0)
        assert np.allclose(rho_0, np.outer(psi_0, psi_0.conj()), atol=1e-12)

        rho_1 = partial_trace_single_qubit(rho, n_qubits=2, keep_qubit=1)
        assert np.allclose(rho_1, np.outer(psi_1, psi_1.conj()), atol=1e-12)


# =============================================================================
# _is_density_matrix_invalid
# =============================================================================


class TestDensityMatrixValidation:
    """Tests for the density matrix validity checker."""

    def test_valid_pure_state(self):
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert _is_density_matrix_invalid(rho) is False

    def test_valid_maximally_mixed(self):
        rho = np.eye(2, dtype=np.complex128) / 2
        assert _is_density_matrix_invalid(rho) is False

    def test_valid_bell_state(self, bell_state_dm):
        assert _is_density_matrix_invalid(bell_state_dm) is False

    def test_nan_detected(self):
        rho = np.array([[np.nan, 0], [0, 1]], dtype=np.complex128)
        assert _is_density_matrix_invalid(rho) is True

    def test_inf_detected(self):
        rho = np.array([[np.inf, 0], [0, 0]], dtype=np.complex128)
        assert _is_density_matrix_invalid(rho) is True

    def test_bad_trace_detected(self):
        """Trace far from 1 should be flagged."""
        rho = np.eye(2, dtype=np.complex128) * 2.0  # Tr = 4
        assert _is_density_matrix_invalid(rho) is True

    def test_large_negative_eigenvalue_detected(self):
        """Eigenvalue < -0.1 should be flagged."""
        rho = np.array([[1.5, 0], [0, -0.5]], dtype=np.complex128)
        assert _is_density_matrix_invalid(rho) is True

    def test_small_negative_eigenvalue_tolerated(self):
        """Small numerical noise in eigenvalues should be tolerated."""
        # Tr = 1.0, min eigenvalue = -0.001 (above -0.1 threshold)
        rho = np.array([[1.001, 0], [0, -0.001]], dtype=np.complex128)
        assert _is_density_matrix_invalid(rho) is False

    def test_zero_matrix_invalid(self):
        """Zero matrix has Tr = 0, which is not close to 1."""
        rho = np.zeros((2, 2), dtype=np.complex128)
        assert _is_density_matrix_invalid(rho) is True

    def test_four_qubit_valid(self):
        """Valid 4-qubit density matrix should pass."""
        rho = np.eye(16, dtype=np.complex128) / 16
        assert _is_density_matrix_invalid(rho) is False


# =============================================================================
# compute_pure_mixed_fidelity
# =============================================================================


class TestPureMixedFidelity:
    """Tests for fidelity between a pure state and a density matrix."""

    def test_pure_state_with_itself(self):
        psi = np.array([1, 0], dtype=np.complex128)
        rho = np.outer(psi, psi.conj())
        assert np.isclose(compute_pure_mixed_fidelity(psi, rho), 1.0)

    def test_orthogonal_states(self):
        psi = np.array([1, 0], dtype=np.complex128)
        rho = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        assert np.isclose(compute_pure_mixed_fidelity(psi, rho), 0.0)

    def test_pure_with_maximally_mixed(self):
        psi = np.array([1, 0], dtype=np.complex128)
        rho = np.eye(2, dtype=np.complex128) / 2
        assert np.isclose(compute_pure_mixed_fidelity(psi, rho), 0.5)

    def test_plus_state_with_maximally_mixed(self):
        psi = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        rho = np.eye(2, dtype=np.complex128) / 2
        assert np.isclose(compute_pure_mixed_fidelity(psi, rho), 0.5)

    def test_result_in_unit_interval(self):
        """Fidelity must lie in [0, 1] for random inputs."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            psi /= np.linalg.norm(psi)
            rho = np.eye(4, dtype=np.complex128) / 4
            f = compute_pure_mixed_fidelity(psi, rho)
            assert 0.0 <= f <= 1.0

    def test_four_qubit_pure_with_mixed(self):
        """Works for 4-qubit (16-dim) systems."""
        psi = np.zeros(16, dtype=np.complex128)
        psi[0] = 1.0
        rho = np.eye(16, dtype=np.complex128) / 16
        f = compute_pure_mixed_fidelity(psi, rho)
        assert np.isclose(f, 1.0 / 16)

    def test_bell_state_partial_trace_fidelity(self, bell_state_dm):
        """F(|0>, Tr_1(|Phi+><Phi+|)) should be 0.5."""
        rho_reduced = partial_trace_single_qubit(
            bell_state_dm, n_qubits=2, keep_qubit=0
        )
        psi_0 = np.array([1, 0], dtype=np.complex128)
        f = compute_pure_mixed_fidelity(psi_0, rho_reduced)
        assert np.isclose(f, 0.5)


# =============================================================================
# compute_mixed_state_fidelity (additional edge cases)
# =============================================================================


class TestMixedStateFidelityEdgeCases:
    """Additional edge-case tests for mixed-state fidelity.

    Basic tests already exist in ``test_noise_smoke.py``. These cover
    edge cases and the pure-state fast path.
    """

    def test_identical_mixed_states(self):
        """F(rho, rho) = 1 for any valid state."""
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=np.complex128)
        assert np.isclose(compute_mixed_state_fidelity(rho, rho), 1.0, atol=1e-6)

    def test_pure_state_fast_path(self):
        """When rho is pure (Tr(rho^2)=1), the fast path is used."""
        psi = np.array([1, 0], dtype=np.complex128)
        rho = np.outer(psi, psi.conj())  # Pure: Tr(rho^2) = 1
        sigma = np.eye(2, dtype=np.complex128) / 2

        f = compute_mixed_state_fidelity(rho, sigma)
        assert np.isclose(f, 0.5, atol=1e-6)

    def test_orthogonal_pure_states(self):
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        sigma = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        assert np.isclose(compute_mixed_state_fidelity(rho, sigma), 0.0, atol=1e-6)

    def test_symmetry(self):
        """F(rho, sigma) = F(sigma, rho)."""
        rng = np.random.default_rng(7)
        # Random valid density matrices via partial trace
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        full_rho = np.outer(psi, psi.conj())
        rho = partial_trace_single_qubit(full_rho, n_qubits=3, keep_qubit=0)

        psi2 = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi2 /= np.linalg.norm(psi2)
        full_sigma = np.outer(psi2, psi2.conj())
        sigma = partial_trace_single_qubit(full_sigma, n_qubits=3, keep_qubit=0)

        f1 = compute_mixed_state_fidelity(rho, sigma)
        f2 = compute_mixed_state_fidelity(sigma, rho)
        assert np.isclose(f1, f2, atol=1e-6)


# =============================================================================
# compute_linear_entropy (additional coverage)
# =============================================================================


class TestLinearEntropyEdgeCases:
    """Additional edge-case tests for linear entropy.

    Basic tests exist in ``test_noise_smoke.py``. These cover
    boundary values and multi-qubit states.
    """

    def test_pure_state_zero_entropy(self):
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert np.isclose(compute_linear_entropy(rho), 0.0)

    def test_maximally_mixed_qubit(self):
        """S_L(I/2) = 1 - 1/2 = 0.5."""
        rho = np.eye(2, dtype=np.complex128) / 2
        assert np.isclose(compute_linear_entropy(rho), 0.5)

    def test_maximally_mixed_two_qubits(self):
        """S_L(I/4) = 1 - 1/4 = 0.75."""
        rho = np.eye(4, dtype=np.complex128) / 4
        assert np.isclose(compute_linear_entropy(rho), 0.75)

    def test_bell_state_reduced_entropy(self, bell_state_dm):
        """Reduced state of Bell pair has S_L = 0.5."""
        rho_q = partial_trace_single_qubit(
            bell_state_dm, n_qubits=2, keep_qubit=0
        )
        assert np.isclose(compute_linear_entropy(rho_q), 0.5)

    def test_product_state_zero_reduced_entropy(self):
        """Reduced state of product state has S_L = 0."""
        psi = np.array([1, 0, 0, 0], dtype=np.complex128)  # |00>
        rho = np.outer(psi, psi.conj())
        rho_q = partial_trace_single_qubit(rho, n_qubits=2, keep_qubit=0)
        assert np.isclose(compute_linear_entropy(rho_q), 0.0)


# =============================================================================
# _compute_haar_distribution
# =============================================================================


class TestHaarDistribution:
    """Tests for the Haar-random fidelity distribution P_Haar(F)."""

    def test_normalized_to_one(self):
        F = np.linspace(0.01, 0.99, 100)
        P = _compute_haar_distribution(n_qubits=2, fidelity_values=F)
        assert np.isclose(P.sum(), 1.0, atol=0.02)

    def test_non_negative(self):
        F = np.linspace(0.0, 1.0, 100)
        P = _compute_haar_distribution(n_qubits=3, fidelity_values=F)
        assert np.all(P >= 0)

    def test_single_qubit_uniform(self):
        """For d=2 (1 qubit), P_Haar is uniform."""
        F = np.linspace(0.01, 0.99, 50)
        P = _compute_haar_distribution(n_qubits=1, fidelity_values=F)
        # All values should be equal (uniform)
        assert np.allclose(P, P[0], atol=1e-10)

    def test_concentration_near_zero_for_many_qubits(self):
        """For d >> 1, Haar distribution concentrates near F = 0."""
        F = np.linspace(0.01, 0.99, 100)
        P = _compute_haar_distribution(n_qubits=4, fidelity_values=F)
        low_fidelity_mass = P[:20].sum()
        high_fidelity_mass = P[80:].sum()
        assert low_fidelity_mass > high_fidelity_mass

    def test_large_qubit_count_no_nan_or_inf(self):
        """Log-space path for d > 100 must produce finite values."""
        F = np.linspace(0.01, 0.99, 50)
        P = _compute_haar_distribution(n_qubits=8, fidelity_values=F)
        assert not np.any(np.isnan(P))
        assert not np.any(np.isinf(P))
        assert np.isclose(P.sum(), 1.0, atol=0.02)

    def test_two_qubit_shape(self):
        """P_Haar(F) = (d-1)(1-F)^(d-2) for d = 2^n_qubits."""
        # For 2 qubits: d = 4, P(F) = 3(1-F)^2
        F = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        P = _compute_haar_distribution(n_qubits=2, fidelity_values=F)
        # Check relative shape (not absolute values due to normalization)
        expected_raw = 3 * (1 - F) ** 2
        expected_normalized = expected_raw / expected_raw.sum()
        assert np.allclose(P, expected_normalized, atol=1e-10)

    def test_empty_input(self):
        """Empty fidelity array should return empty array."""
        F = np.array([])
        P = _compute_haar_distribution(n_qubits=2, fidelity_values=F)
        assert len(P) == 0
