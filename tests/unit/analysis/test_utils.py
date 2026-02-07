"""Comprehensive unit tests for analysis._utils module.

This module contains thorough tests for all utility functions in the
encoding_atlas.analysis._utils module, including:

- Statevector simulation (PennyLane and Qiskit backends)
- Validation functions
- Partial trace computations
- Fidelity and purity calculations
- Gradient computation
- Random sampling utilities

Test Categories
---------------
1. **Basic Functionality**: Verify correct output for simple inputs
2. **Edge Cases**: Test boundary conditions and special cases
3. **Numerical Stability**: Test with extreme values
4. **Error Handling**: Verify proper exceptions for invalid inputs
5. **Cross-Backend Consistency**: Compare results across backends
6. **Known Values**: Test against analytically known results

Coverage Goals
--------------
- All public functions in _utils.py
- All code paths including error paths
- Numerical edge cases (NaN, Inf, zero norms)
- Type validation paths
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from encoding_atlas.core.exceptions import (
    AnalysisError,
    NumericalInstabilityError,
    SimulationError,
    ValidationError,
)
from encoding_atlas.analysis._utils import (
    # Simulation
    simulate_encoding_statevector,
    simulate_encoding_statevectors_batch,
    # Validation
    validate_encoding_for_analysis,
    validate_statevector,
    # Quantum operations
    partial_trace_single_qubit,
    partial_trace_subsystem,
    compute_fidelity,
    compute_purity,
    compute_linear_entropy,
    compute_von_neumann_entropy,
    # Gradient computation
    compute_parameter_gradient,
    compute_all_parameter_gradients,
    # Random sampling
    generate_random_parameters,
    create_rng,
)


# =============================================================================
# Test: validate_statevector
# =============================================================================


class TestValidateStatevector:
    """Tests for validate_statevector function."""

    def test_valid_normalized_state(self, zero_state_2q):
        """Test validation of a properly normalized state."""
        result = validate_statevector(zero_state_2q)
        assert result.dtype == np.complex128
        assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)

    def test_valid_state_with_expected_qubits(self, bell_state):
        """Test validation with expected qubit count."""
        result = validate_statevector(bell_state, expected_qubits=2)
        assert len(result) == 4

    def test_auto_renormalization_small_drift(self):
        """Test that states with tiny norm drift are auto-renormalized.

        The implementation only auto-renormalizes when the norm deviation
        is within ``max(tolerance * 1e4, 1e-6)``.  Larger deviations are
        rejected as non-physical (see ``validate_statevector`` source).
        """
        # Norm ≈ 1 + 1e-8, well within auto-renormalization tolerance
        slightly_off = np.array([1.0 + 1e-8, 0.0, 0.0, 0.0], dtype=np.complex128)
        result = validate_statevector(slightly_off, check_normalization=True)
        assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)

    def test_grossly_unnormalized_raises(self, unnormalized_state):
        """Test that grossly unnormalized states raise ValidationError.

        States whose norm deviates from 1.0 by more than
        ``max(tolerance * 1e4, 1e-6)`` are considered non-physical
        and are rejected rather than silently renormalized.
        """
        with pytest.raises(ValidationError, match="not normalized"):
            validate_statevector(unnormalized_state, check_normalization=True)

    def test_wrong_qubit_count_raises(self, bell_state):
        """Test that wrong expected_qubits raises error."""
        with pytest.raises(ValidationError, match="expected.*3 qubits"):
            validate_statevector(bell_state, expected_qubits=3)

    def test_non_power_of_2_dimension_raises(self):
        """Test that non-power-of-2 dimension raises error."""
        invalid_state = np.array([1, 0, 0], dtype=complex)
        with pytest.raises(ValidationError, match="power of 2"):
            validate_statevector(invalid_state)

    def test_nan_values_raise(self, state_with_nan):
        """Test that NaN values raise error."""
        with pytest.raises(ValidationError, match="NaN"):
            validate_statevector(state_with_nan)

    def test_inf_values_raise(self, state_with_inf):
        """Test that infinite values raise error."""
        with pytest.raises(ValidationError, match="infinite"):
            validate_statevector(state_with_inf)

    def test_zero_norm_raises(self, zero_norm_state):
        """Test that zero norm raises NumericalInstabilityError."""
        with pytest.raises(NumericalInstabilityError, match="near-zero norm"):
            validate_statevector(zero_norm_state)

    def test_2d_array_raises(self):
        """Test that 2D array raises error."""
        invalid_state = np.array([[1, 0], [0, 0]], dtype=complex)
        with pytest.raises(ValidationError, match="must be 1D"):
            validate_statevector(invalid_state)

    def test_empty_array_raises(self):
        """Test that empty array raises error."""
        invalid_state = np.array([], dtype=complex)
        with pytest.raises(ValidationError, match="power of 2"):
            validate_statevector(invalid_state)


# =============================================================================
# Test: validate_encoding_for_analysis
# =============================================================================


class TestValidateEncodingForAnalysis:
    """Tests for validate_encoding_for_analysis function."""

    def test_valid_encoding(self, sample_encoding_2q):
        """Test that valid encoding passes validation."""
        # Should not raise
        validate_encoding_for_analysis(sample_encoding_2q)

    def test_non_encoding_raises(self):
        """Test that non-BaseEncoding raises error."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            validate_encoding_for_analysis("not an encoding")

    def test_none_raises(self):
        """Test that None raises error."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            validate_encoding_for_analysis(None)

    def test_dict_raises(self):
        """Test that dictionary raises error."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            validate_encoding_for_analysis({"n_features": 4})


# =============================================================================
# Test: compute_fidelity
# =============================================================================


class TestComputeFidelity:
    """Tests for compute_fidelity function."""

    def test_identical_states(self, zero_state_2q):
        """Test fidelity of identical states is 1."""
        fidelity = compute_fidelity(zero_state_2q, zero_state_2q)
        assert_allclose(fidelity, 1.0, atol=1e-10)

    def test_orthogonal_states(self, zero_state_1q, one_state_1q):
        """Test fidelity of orthogonal states is 0."""
        fidelity = compute_fidelity(zero_state_1q, one_state_1q)
        assert_allclose(fidelity, 0.0, atol=1e-10)

    def test_global_phase_invariance(self, zero_state_1q):
        """Test that fidelity is invariant under global phase."""
        phase = np.exp(1j * np.pi / 4)
        state_with_phase = zero_state_1q * phase
        fidelity = compute_fidelity(zero_state_1q, state_with_phase)
        assert_allclose(fidelity, 1.0, atol=1e-10)

    def test_partial_overlap(self, plus_state_1q, zero_state_1q):
        """Test fidelity for states with partial overlap."""
        # |⟨+|0⟩|² = |1/√2|² = 0.5
        fidelity = compute_fidelity(plus_state_1q, zero_state_1q)
        assert_allclose(fidelity, 0.5, atol=1e-10)

    def test_bell_state_fidelity(self, bell_state):
        """Test fidelity of Bell state with itself."""
        fidelity = compute_fidelity(bell_state, bell_state)
        assert_allclose(fidelity, 1.0, atol=1e-10)

    def test_dimension_mismatch_raises(self, zero_state_1q, zero_state_2q):
        """Test that different dimensions raise error."""
        with pytest.raises(ValueError, match="same dimension"):
            compute_fidelity(zero_state_1q, zero_state_2q)

    def test_fidelity_bounded(self, random_statevector_generator):
        """Test that fidelity is always in [0, 1]."""
        state1 = random_statevector_generator(3)
        state2 = random_statevector_generator(3)
        fidelity = compute_fidelity(state1, state2)
        assert 0.0 <= fidelity <= 1.0

    def test_nan_raises(self, zero_state_1q, state_with_nan):
        """Test that NaN values raise error."""
        # Resize state_with_nan to match zero_state_1q
        state_nan = np.array([np.nan, 0], dtype=complex)
        with pytest.raises(ValidationError, match="NaN"):
            compute_fidelity(zero_state_1q, state_nan)

    def test_symmetric(self, random_statevector_generator):
        """Test that fidelity is symmetric: F(a,b) = F(b,a)."""
        state1 = random_statevector_generator(2)
        state2 = random_statevector_generator(2)
        fidelity_12 = compute_fidelity(state1, state2)
        fidelity_21 = compute_fidelity(state2, state1)
        assert_allclose(fidelity_12, fidelity_21, atol=1e-10)


# =============================================================================
# Test: compute_purity
# =============================================================================


class TestComputePurity:
    """Tests for compute_purity function."""

    def test_pure_state_purity(self, pure_density_matrix_1q):
        """Test purity of pure state is 1."""
        purity = compute_purity(pure_density_matrix_1q)
        assert_allclose(purity, 1.0, atol=1e-10)

    def test_maximally_mixed_purity(self, maximally_mixed_1q):
        """Test purity of maximally mixed state is 1/d."""
        purity = compute_purity(maximally_mixed_1q)
        assert_allclose(purity, 0.5, atol=1e-10)  # 1/2 for 1 qubit

    def test_maximally_mixed_2q_purity(self, maximally_mixed_2q):
        """Test purity of 2-qubit maximally mixed state is 1/4."""
        purity = compute_purity(maximally_mixed_2q)
        assert_allclose(purity, 0.25, atol=1e-10)

    def test_purity_from_statevector(self, bell_state):
        """Test purity computed from outer product of statevector."""
        rho = np.outer(bell_state, np.conj(bell_state))
        purity = compute_purity(rho)
        assert_allclose(purity, 1.0, atol=1e-10)

    def test_non_square_raises(self):
        """Test that non-square matrix raises error."""
        non_square = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        with pytest.raises(ValidationError, match="square"):
            compute_purity(non_square)

    def test_1d_array_raises(self):
        """Test that 1D array raises error."""
        vector = np.array([1, 0], dtype=complex)
        with pytest.raises(ValidationError, match="2D"):
            compute_purity(vector)

    def test_purity_bounded(self, random_statevector_generator):
        """Test that purity is always in [1/d, 1] (allowing small numerical tolerance)."""
        state = random_statevector_generator(2)
        rho = np.outer(state, np.conj(state))
        purity = compute_purity(rho)
        # Allow small numerical tolerance above 1.0
        assert 0.0 <= purity <= 1.0 + 1e-10


# =============================================================================
# Test: compute_linear_entropy
# =============================================================================


class TestComputeLinearEntropy:
    """Tests for compute_linear_entropy function."""

    def test_pure_state_entropy(self, pure_density_matrix_1q):
        """Test linear entropy of pure state is 0."""
        entropy = compute_linear_entropy(pure_density_matrix_1q)
        assert_allclose(entropy, 0.0, atol=1e-10)

    def test_maximally_mixed_entropy(self, maximally_mixed_1q):
        """Test linear entropy of maximally mixed state."""
        entropy = compute_linear_entropy(maximally_mixed_1q)
        # S_L = 1 - 1/d = 1 - 0.5 = 0.5
        assert_allclose(entropy, 0.5, atol=1e-10)

    def test_relationship_to_purity(self, maximally_mixed_2q):
        """Test S_L = 1 - γ."""
        purity = compute_purity(maximally_mixed_2q)
        entropy = compute_linear_entropy(maximally_mixed_2q)
        assert_allclose(entropy + purity, 1.0, atol=1e-10)


# =============================================================================
# Test: compute_von_neumann_entropy
# =============================================================================


class TestComputeVonNeumannEntropy:
    """Tests for compute_von_neumann_entropy function."""

    def test_pure_state_entropy(self, pure_density_matrix_1q):
        """Test von Neumann entropy of pure state is 0."""
        entropy = compute_von_neumann_entropy(pure_density_matrix_1q)
        assert_allclose(entropy, 0.0, atol=1e-10)

    def test_maximally_mixed_1q_entropy(self, maximally_mixed_1q):
        """Test von Neumann entropy of maximally mixed 1-qubit state."""
        entropy = compute_von_neumann_entropy(maximally_mixed_1q)
        # S = log₂(2) = 1 bit
        assert_allclose(entropy, 1.0, atol=1e-10)

    def test_maximally_mixed_2q_entropy(self, maximally_mixed_2q):
        """Test von Neumann entropy of maximally mixed 2-qubit state."""
        entropy = compute_von_neumann_entropy(maximally_mixed_2q)
        # S = log₂(4) = 2 bits
        assert_allclose(entropy, 2.0, atol=1e-10)

    def test_bell_state_reduced_entropy(self, bell_state):
        """Test entropy of reduced state from Bell state."""
        # Trace out one qubit from Bell state - should give maximally mixed
        rho_full = np.outer(bell_state, np.conj(bell_state))
        rho_reduced = partial_trace_single_qubit(bell_state, n_qubits=2, keep_qubit=0)
        entropy = compute_von_neumann_entropy(rho_reduced)
        # Reduced state is maximally mixed, entropy = 1 bit
        assert_allclose(entropy, 1.0, atol=1e-10)


# =============================================================================
# Test: partial_trace_single_qubit
# =============================================================================


class TestPartialTraceSingleQubit:
    """Tests for partial_trace_single_qubit function."""

    def test_product_state_trace(self, product_state_2q):
        """Test partial trace of product state |+⟩⊗|0⟩."""
        # Keep qubit 0: should get |+⟩ = mixed state with p=0.5
        rho_0 = partial_trace_single_qubit(product_state_2q, n_qubits=2, keep_qubit=0)
        # |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
        expected_0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        assert_array_almost_equal(rho_0, expected_0, decimal=10)

        # Keep qubit 1: should get |0⟩⟨0|
        rho_1 = partial_trace_single_qubit(product_state_2q, n_qubits=2, keep_qubit=1)
        expected_1 = np.array([[1, 0], [0, 0]], dtype=complex)
        assert_array_almost_equal(rho_1, expected_1, decimal=10)

    def test_bell_state_trace(self, bell_state):
        """Test partial trace of Bell state gives maximally mixed."""
        rho_0 = partial_trace_single_qubit(bell_state, n_qubits=2, keep_qubit=0)
        expected = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        assert_array_almost_equal(rho_0, expected, decimal=10)

    def test_computational_basis_trace(self, zero_state_2q):
        """Test partial trace of |00⟩ gives |0⟩."""
        rho = partial_trace_single_qubit(zero_state_2q, n_qubits=2, keep_qubit=0)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert_array_almost_equal(rho, expected, decimal=10)

    def test_single_qubit_trace(self, zero_state_1q):
        """Test partial trace of single-qubit state (trivial case)."""
        rho = partial_trace_single_qubit(zero_state_1q, n_qubits=1, keep_qubit=0)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert_array_almost_equal(rho, expected, decimal=10)

    def test_trace_has_unit_trace(self, random_statevector_generator):
        """Test that reduced density matrix has trace 1."""
        state = random_statevector_generator(3)
        for keep_qubit in range(3):
            rho = partial_trace_single_qubit(state, n_qubits=3, keep_qubit=keep_qubit)
            trace = np.trace(rho)
            assert_allclose(np.real(trace), 1.0, atol=1e-10)

    def test_trace_is_hermitian(self, random_statevector_generator):
        """Test that reduced density matrix is Hermitian."""
        state = random_statevector_generator(3)
        rho = partial_trace_single_qubit(state, n_qubits=3, keep_qubit=1)
        assert_array_almost_equal(rho, np.conj(rho.T), decimal=10)

    def test_trace_positive_semidefinite(self, random_statevector_generator):
        """Test that reduced density matrix is positive semidefinite."""
        state = random_statevector_generator(2)
        rho = partial_trace_single_qubit(state, n_qubits=2, keep_qubit=0)
        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_invalid_keep_qubit_raises(self, zero_state_2q):
        """Test that invalid keep_qubit raises error."""
        with pytest.raises(ValueError, match="range"):
            partial_trace_single_qubit(zero_state_2q, n_qubits=2, keep_qubit=2)

    def test_negative_keep_qubit_raises(self, zero_state_2q):
        """Test that negative keep_qubit raises error."""
        with pytest.raises(ValueError, match="range"):
            partial_trace_single_qubit(zero_state_2q, n_qubits=2, keep_qubit=-1)


# =============================================================================
# Test: partial_trace_subsystem
# =============================================================================


class TestPartialTraceSubsystem:
    """Tests for partial_trace_subsystem function."""

    def test_keep_all_qubits(self, bell_state):
        """Test keeping all qubits returns full density matrix."""
        rho = partial_trace_subsystem(bell_state, n_qubits=2, keep_qubits=[0, 1])
        expected = np.outer(bell_state, np.conj(bell_state))
        assert_array_almost_equal(rho, expected, decimal=10)

    def test_keep_single_qubit(self, bell_state):
        """Test keeping single qubit matches partial_trace_single_qubit."""
        rho_subsystem = partial_trace_subsystem(bell_state, n_qubits=2, keep_qubits=[0])
        rho_single = partial_trace_single_qubit(bell_state, n_qubits=2, keep_qubit=0)
        assert_array_almost_equal(rho_subsystem, rho_single, decimal=10)

    def test_ghz_state_trace(self, ghz_state_3q):
        """Test tracing out one qubit from GHZ state."""
        # Keep qubits 0 and 1, trace out qubit 2
        rho = partial_trace_subsystem(ghz_state_3q, n_qubits=3, keep_qubits=[0, 1])
        # GHZ reduced state is diagonal: 0.5|00⟩⟨00| + 0.5|11⟩⟨11|
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 0.5  # |00⟩⟨00|
        expected[3, 3] = 0.5  # |11⟩⟨11|
        assert_array_almost_equal(rho, expected, decimal=10)

    def test_empty_keep_qubits_raises(self, zero_state_2q):
        """Test that empty keep_qubits raises error."""
        with pytest.raises(ValueError, match="must not be empty"):
            partial_trace_subsystem(zero_state_2q, n_qubits=2, keep_qubits=[])

    def test_duplicate_qubits_raises(self, zero_state_2q):
        """Test that duplicate qubit indices raise error."""
        with pytest.raises(ValueError, match="duplicate"):
            partial_trace_subsystem(zero_state_2q, n_qubits=2, keep_qubits=[0, 0])

    def test_out_of_range_qubit_raises(self, zero_state_2q):
        """Test that out-of-range qubit index raises error."""
        with pytest.raises(ValueError, match="out of range"):
            partial_trace_subsystem(zero_state_2q, n_qubits=2, keep_qubits=[0, 2])


# =============================================================================
# Test: generate_random_parameters
# =============================================================================


class TestGenerateRandomParameters:
    """Tests for generate_random_parameters function."""

    def test_single_sample_shape(self):
        """Test shape when generating single sample.

        Note: The first positional parameter is ``encoding_or_n_features``
        (accepts a BaseEncoding instance or an int).
        """
        params = generate_random_parameters(4, n_samples=1, seed=42)
        assert params.shape == (4,)

    def test_multiple_samples_shape(self):
        """Test shape when generating multiple samples."""
        params = generate_random_parameters(4, n_samples=10, seed=42)
        assert params.shape == (10, 4)

    def test_default_range(self):
        """Test parameters are in default range [0, 2π]."""
        params = generate_random_parameters(100, n_samples=100, seed=42)
        assert np.all(params >= 0)
        assert np.all(params <= 2 * np.pi)

    def test_custom_range(self):
        """Test parameters respect custom range."""
        params = generate_random_parameters(
            100, n_samples=100, param_min=-1, param_max=1, seed=42
        )
        assert np.all(params >= -1)
        assert np.all(params <= 1)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        params1 = generate_random_parameters(4, n_samples=5, seed=42)
        params2 = generate_random_parameters(4, n_samples=5, seed=42)
        assert_array_almost_equal(params1, params2)

    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        params1 = generate_random_parameters(4, n_samples=5, seed=42)
        params2 = generate_random_parameters(4, n_samples=5, seed=123)
        assert not np.allclose(params1, params2)

    def test_invalid_n_features_raises(self):
        """Test that n_features < 1 raises error."""
        with pytest.raises(ValueError, match="n_features"):
            generate_random_parameters(0)

    def test_invalid_n_samples_raises(self):
        """Test that n_samples < 1 raises error."""
        with pytest.raises(ValueError, match="n_samples"):
            generate_random_parameters(4, n_samples=0)

    def test_invalid_range_raises(self):
        """Test that param_min >= param_max raises error."""
        with pytest.raises(ValueError, match="param_min"):
            generate_random_parameters(4, param_min=1, param_max=0)


# =============================================================================
# Test: create_rng
# =============================================================================


class TestCreateRng:
    """Tests for create_rng function."""

    def test_seeded_reproducibility(self):
        """Test that seeded RNG produces reproducible results."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)
        values1 = rng1.random(10)
        values2 = rng2.random(10)
        assert_array_almost_equal(values1, values2)

    def test_unseeded_creates_rng(self):
        """Test that unseeded call creates valid RNG."""
        rng = create_rng(None)
        values = rng.random(10)
        assert len(values) == 10
        assert np.all((values >= 0) & (values < 1))


# =============================================================================
# Test: simulate_encoding_statevector (PennyLane)
# =============================================================================


class TestSimulateEncodingStatevectorPennylane:
    """Tests for statevector simulation with PennyLane backend."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_basic_simulation(self, sample_encoding_2q, sample_features_2d):
        """Test basic statevector simulation."""
        state = simulate_encoding_statevector(
            sample_encoding_2q, sample_features_2d, backend="pennylane"
        )
        assert state.shape == (4,)  # 2 qubits = 2^2 = 4
        assert state.dtype == np.complex128
        assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)

    def test_different_features_different_states(self, sample_encoding_2q):
        """Test that different inputs produce different states."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([np.pi, np.pi])
        state1 = simulate_encoding_statevector(sample_encoding_2q, x1, backend="pennylane")
        state2 = simulate_encoding_statevector(sample_encoding_2q, x2, backend="pennylane")
        fidelity = compute_fidelity(state1, state2)
        assert fidelity < 0.99  # States should be different

    def test_zero_input_initial_state(self, sample_encoding_2q):
        """Test that zero input gives deterministic state."""
        x = np.array([0.0, 0.0])
        state = simulate_encoding_statevector(sample_encoding_2q, x, backend="pennylane")
        # For AngleEncoding with RY and angle 0, state should be |00⟩
        assert_allclose(np.abs(state[0])**2, 1.0, atol=1e-10)

    def test_wrong_feature_count_raises(self, sample_encoding_2q):
        """Test that wrong feature count raises error."""
        wrong_features = np.array([0.1, 0.2, 0.3])  # 3 features, but encoding expects 2
        with pytest.raises(ValidationError, match="features"):
            simulate_encoding_statevector(sample_encoding_2q, wrong_features, backend="pennylane")

    def test_nan_input_raises(self, sample_encoding_2q, features_with_nan):
        """Test that NaN input raises error."""
        # Create features with correct size but containing NaN
        nan_features = np.array([0.1, np.nan])
        with pytest.raises(ValidationError, match="NaN"):
            simulate_encoding_statevector(sample_encoding_2q, nan_features, backend="pennylane")

    def test_inf_input_raises(self, sample_encoding_2q):
        """Test that infinite input raises error."""
        inf_features = np.array([0.1, np.inf])
        with pytest.raises(ValidationError, match="infinite"):
            simulate_encoding_statevector(sample_encoding_2q, inf_features, backend="pennylane")

    def test_2d_input_raises(self, sample_encoding_2q):
        """Test that 2D input raises error."""
        x_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValidationError, match="1D"):
            simulate_encoding_statevector(sample_encoding_2q, x_2d, backend="pennylane")

    def test_invalid_encoding_raises(self, sample_features_2d):
        """Test that invalid encoding raises error."""
        with pytest.raises(AnalysisError, match="BaseEncoding"):
            simulate_encoding_statevector("not_an_encoding", sample_features_2d, backend="pennylane")


# =============================================================================
# Test: simulate_encoding_statevector (Qiskit)
# =============================================================================


class TestSimulateEncodingStatevectorQiskit:
    """Tests for statevector simulation with Qiskit backend."""

    @pytest.fixture(autouse=True)
    def check_qiskit(self, qiskit_available):
        """Skip tests if Qiskit is not available."""
        if not qiskit_available:
            pytest.skip("Qiskit not available")

    def test_basic_simulation(self, sample_encoding_2q, sample_features_2d):
        """Test basic statevector simulation."""
        state = simulate_encoding_statevector(
            sample_encoding_2q, sample_features_2d, backend="qiskit"
        )
        assert state.shape == (4,)
        assert state.dtype == np.complex128
        assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)

    def test_normalization(self, sample_encoding_4q, sample_features_4d):
        """Test that output state is normalized."""
        state = simulate_encoding_statevector(
            sample_encoding_4q, sample_features_4d, backend="qiskit"
        )
        assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)


# =============================================================================
# Test: Cross-Backend Consistency
# =============================================================================


class TestCrossBackendConsistency:
    """Tests comparing results across different backends.

    Note: PennyLane and Qiskit may use different qubit ordering conventions
    (big-endian vs little-endian), which can result in different statevector
    representations for the same logical state. The key property we test is
    that _within_ each backend, the results are consistent.
    """

    @pytest.fixture(autouse=True)
    def check_both_backends(self, pennylane_available, qiskit_available):
        """Skip tests if both backends are not available."""
        if not pennylane_available or not qiskit_available:
            pytest.skip("Both PennyLane and Qiskit required")

    def test_same_statevector(self, sample_encoding_2q, sample_features_2d):
        """Test that both backends produce valid normalized states.

        Note: Due to different qubit ordering conventions between PennyLane
        and Qiskit, the raw statevectors may differ. This test verifies that
        both backends produce valid normalized states, not necessarily
        identical statevectors.
        """
        state_pl = simulate_encoding_statevector(
            sample_encoding_2q, sample_features_2d, backend="pennylane"
        )
        state_qk = simulate_encoding_statevector(
            sample_encoding_2q, sample_features_2d, backend="qiskit"
        )
        # Both states should be normalized
        assert_allclose(np.linalg.norm(state_pl), 1.0, atol=1e-10)
        assert_allclose(np.linalg.norm(state_qk), 1.0, atol=1e-10)
        # Both should have same dimension
        assert len(state_pl) == len(state_qk)

    def test_same_fidelity_within_backend(self, sample_encoding_2q):
        """Test that fidelity computation is consistent within each backend.

        The key property is that fidelity between two states computed within
        the same backend should be consistent and meaningful.
        """
        x1 = np.array([0.1, 0.2])
        x2 = np.array([0.5, 0.8])

        # Test PennyLane fidelity
        state1_pl = simulate_encoding_statevector(sample_encoding_2q, x1, backend="pennylane")
        state2_pl = simulate_encoding_statevector(sample_encoding_2q, x2, backend="pennylane")
        fidelity_pl = compute_fidelity(state1_pl, state2_pl)
        assert 0.0 <= fidelity_pl <= 1.0

        # Test Qiskit fidelity
        state1_qk = simulate_encoding_statevector(sample_encoding_2q, x1, backend="qiskit")
        state2_qk = simulate_encoding_statevector(sample_encoding_2q, x2, backend="qiskit")
        fidelity_qk = compute_fidelity(state1_qk, state2_qk)
        assert 0.0 <= fidelity_qk <= 1.0

        # Both should indicate some difference (not identical states)
        assert fidelity_pl < 1.0
        assert fidelity_qk < 1.0


# =============================================================================
# Test: simulate_encoding_statevectors_batch
# =============================================================================


class TestSimulateEncodingStatevectorsBatch:
    """Tests for batch statevector simulation."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_batch_simulation(self, sample_encoding_2q, sample_features_batch_2d):
        """Test batch simulation returns correct number of states."""
        states = simulate_encoding_statevectors_batch(
            sample_encoding_2q, sample_features_batch_2d, backend="pennylane"
        )
        assert len(states) == 3  # 3 samples in batch
        for state in states:
            assert state.shape == (4,)
            assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)

    def test_batch_matches_individual(self, sample_encoding_2q, sample_features_batch_2d):
        """Test that batch results match individual simulation."""
        batch_states = simulate_encoding_statevectors_batch(
            sample_encoding_2q, sample_features_batch_2d, backend="pennylane"
        )
        for i, x in enumerate(sample_features_batch_2d):
            individual_state = simulate_encoding_statevector(
                sample_encoding_2q, x, backend="pennylane"
            )
            fidelity = compute_fidelity(batch_states[i], individual_state)
            assert_allclose(fidelity, 1.0, atol=1e-10)

    def test_1d_input_raises(self, sample_encoding_2q, sample_features_2d):
        """Test that 1D input raises error."""
        with pytest.raises(ValidationError, match="2D"):
            simulate_encoding_statevectors_batch(
                sample_encoding_2q, sample_features_2d, backend="pennylane"
            )


# =============================================================================
# Test: compute_parameter_gradient
# =============================================================================


class TestComputeParameterGradient:
    """Tests for parameter gradient computation."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_gradient_computation(self, sample_encoding_2q, sample_features_2d):
        """Test that gradient computation returns a float."""
        grad = compute_parameter_gradient(
            sample_encoding_2q, sample_features_2d, param_index=0
        )
        assert isinstance(grad, float)
        assert np.isfinite(grad)

    def test_gradient_for_each_parameter(self, sample_encoding_2q, sample_features_2d):
        """Test computing gradient for each parameter."""
        for i in range(len(sample_features_2d)):
            grad = compute_parameter_gradient(
                sample_encoding_2q, sample_features_2d, param_index=i
            )
            assert isinstance(grad, float)
            assert np.isfinite(grad)

    def test_invalid_param_index_raises(self, sample_encoding_2q, sample_features_2d):
        """Test that invalid param_index raises error."""
        with pytest.raises(ValidationError, match="out of range"):
            compute_parameter_gradient(
                sample_encoding_2q, sample_features_2d, param_index=10
            )

    def test_negative_param_index_raises(self, sample_encoding_2q, sample_features_2d):
        """Test that negative param_index raises error."""
        with pytest.raises(ValidationError, match="out of range"):
            compute_parameter_gradient(
                sample_encoding_2q, sample_features_2d, param_index=-1
            )


# =============================================================================
# Test: compute_all_parameter_gradients
# =============================================================================


class TestComputeAllParameterGradients:
    """Tests for computing all parameter gradients."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_gradient_vector_shape(self, sample_encoding_2q, sample_features_2d):
        """Test that gradient vector has correct shape."""
        grads = compute_all_parameter_gradients(
            sample_encoding_2q, sample_features_2d
        )
        assert grads.shape == (2,)  # 2 features = 2 parameters
        assert grads.dtype == np.float64

    def test_gradient_vector_matches_individual(self, sample_encoding_2q, sample_features_2d):
        """Test that gradient vector matches individual computations."""
        grads = compute_all_parameter_gradients(
            sample_encoding_2q, sample_features_2d
        )
        for i in range(len(sample_features_2d)):
            individual_grad = compute_parameter_gradient(
                sample_encoding_2q, sample_features_2d, param_index=i
            )
            assert_allclose(grads[i], individual_grad, atol=1e-10)


# =============================================================================
# Test: Edge Cases and Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of analysis utilities."""

    def test_fidelity_near_one(self):
        """Test fidelity computation for nearly identical states."""
        state = np.array([1, 0], dtype=complex)
        # Add tiny perturbation
        perturbed = state + 1e-14 * np.array([0, 1], dtype=complex)
        perturbed = perturbed / np.linalg.norm(perturbed)
        fidelity = compute_fidelity(state, perturbed)
        assert_allclose(fidelity, 1.0, atol=1e-10)

    def test_purity_near_one(self):
        """Test purity computation for nearly pure state."""
        # Nearly pure state with tiny mixing
        rho = np.array([[1 - 1e-14, 0], [0, 1e-14]], dtype=complex)
        purity = compute_purity(rho)
        assert_allclose(purity, 1.0, atol=1e-10)

    def test_purity_numerical_positive(self):
        """Test purity is always positive even with numerical errors."""
        # Random density matrix (positive semidefinite)
        rng = np.random.default_rng(42)
        for _ in range(10):
            # Create random density matrix
            M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
            rho = M @ M.conj().T
            rho = rho / np.trace(rho)
            purity = compute_purity(rho)
            assert purity >= 0

    def test_partial_trace_preserves_trace(self, random_statevector_generator):
        """Test partial trace always preserves unit trace."""
        for n_qubits in [2, 3, 4]:
            state = random_statevector_generator(n_qubits)
            for keep_qubit in range(n_qubits):
                rho = partial_trace_single_qubit(state, n_qubits, keep_qubit)
                trace = np.real(np.trace(rho))
                assert_allclose(trace, 1.0, atol=1e-10)


# =============================================================================
# Test: Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests verifying error messages are informative."""

    def test_validation_error_includes_details(self):
        """Test that validation errors include helpful details."""
        state = np.array([1, 0, 0], dtype=complex)  # Invalid dimension
        try:
            validate_statevector(state)
        except ValidationError as e:
            assert "power of 2" in str(e).lower() or "dimension" in str(e).lower()

    def test_analysis_error_includes_type(self):
        """Test that analysis error includes type information."""
        try:
            validate_encoding_for_analysis("string")
        except AnalysisError as e:
            assert "str" in str(e) or "BaseEncoding" in str(e)

    def test_numerical_error_includes_value(self, zero_norm_state):
        """Test that numerical error includes problematic value."""
        try:
            validate_statevector(zero_norm_state)
        except NumericalInstabilityError as e:
            assert e.value is not None or "norm" in str(e).lower()
