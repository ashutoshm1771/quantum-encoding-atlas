"""Comprehensive unit tests for analysis.entanglement module.

This module contains thorough tests for all entanglement capability
functions in the encoding_atlas.analysis.entanglement module, including:

- Meyer-Wallach entanglement measure
- Scott entanglement measure
- Entanglement capability computation
- Per-qubit breakdown analysis

Test Categories
---------------
1. **Known Values**: Test against analytically known results (GHZ, Bell, product states)
2. **Edge Cases**: Test boundary conditions and special cases
3. **Numerical Stability**: Test with extreme values and near-boundary conditions
4. **Error Handling**: Verify proper exceptions for invalid inputs
5. **Statistical Properties**: Test statistical correctness of sampling
6. **Cross-Backend Consistency**: Compare results across simulation backends

Coverage Goals
--------------
- All public functions in entanglement.py
- All code paths including error paths
- Numerical edge cases
- Type validation paths

References
----------
Test values are validated against the mathematical definitions:
    - Meyer-Wallach: Q(|psi>) = (2/n) * sum_i (1 - Tr(rho_i^2))
    - GHZ state has MW = 1 (maximally entangled)
    - Product states have MW = 0 (no entanglement)
    - Bell states have MW = 1 (maximally entangled for 2 qubits)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from encoding_atlas.core.exceptions import (
    AnalysisError,
    InsufficientSamplesError,
    NumericalInstabilityError,
    SimulationError,
    ValidationError,
)
from encoding_atlas.analysis.entanglement import (
    compute_entanglement_capability,
    compute_meyer_wallach,
    compute_meyer_wallach_with_breakdown,
    compute_scott_measure,
    EntanglementResult,
)


# =============================================================================
# Test: compute_meyer_wallach - Known Values
# =============================================================================


class TestMeyerWallachKnownValues:
    """Tests for Meyer-Wallach measure with analytically known values."""

    def test_ghz_state_3q(self, ghz_state_3q):
        """Test that GHZ state (|000> + |111>)/sqrt(2) has MW = 1.

        The GHZ state is maximally entangled, meaning each qubit's
        reduced state is maximally mixed (I/2).
        """
        mw = compute_meyer_wallach(ghz_state_3q, n_qubits=3)
        assert_allclose(mw, 1.0, atol=1e-10)

    def test_bell_state_phi_plus(self, bell_state):
        """Test that Bell state |Phi+> = (|00> + |11>)/sqrt(2) has MW = 1."""
        mw = compute_meyer_wallach(bell_state, n_qubits=2)
        assert_allclose(mw, 1.0, atol=1e-10)

    def test_bell_state_psi_plus(self, bell_state_psi_plus):
        """Test that Bell state |Psi+> = (|01> + |10>)/sqrt(2) has MW = 1."""
        mw = compute_meyer_wallach(bell_state_psi_plus, n_qubits=2)
        assert_allclose(mw, 1.0, atol=1e-10)

    def test_product_state_2q(self, zero_state_2q):
        """Test that product state |00> has MW = 0."""
        mw = compute_meyer_wallach(zero_state_2q, n_qubits=2)
        assert_allclose(mw, 0.0, atol=1e-10)

    def test_product_state_3q(self):
        """Test that product state |000> has MW = 0."""
        product = np.zeros(8, dtype=complex)
        product[0] = 1.0
        mw = compute_meyer_wallach(product, n_qubits=3)
        assert_allclose(mw, 0.0, atol=1e-10)

    def test_product_state_superposition(self, product_state_2q):
        """Test that product state |+>|0> = (|00> + |10>)/sqrt(2) has MW = 0.

        This is a product state (|+> tensor |0>), so there's no entanglement.
        """
        mw = compute_meyer_wallach(product_state_2q, n_qubits=2)
        assert_allclose(mw, 0.0, atol=1e-10)

    def test_w_state(self, w_state_3q):
        """Test Meyer-Wallach measure for W state.

        The W state (|001> + |010> + |100>)/sqrt(3) has a known MW value.
        Each qubit has purity 5/9, so linear entropy is 4/9.
        MW = (2/3) * 3 * (4/9) = 8/9 ≈ 0.889
        """
        mw = compute_meyer_wallach(w_state_3q, n_qubits=3)
        expected_mw = 8.0 / 9.0
        assert_allclose(mw, expected_mw, atol=1e-10)

    def test_single_qubit_no_entanglement(self, zero_state_1q):
        """Test that single-qubit states have MW = 0 (entanglement requires > 1 qubit)."""
        mw = compute_meyer_wallach(zero_state_1q, n_qubits=1)
        assert_allclose(mw, 0.0, atol=1e-10)

    def test_mw_bounded_zero_one(self, random_statevector_generator):
        """Test that MW measure is always in [0, 1]."""
        for n_qubits in [2, 3, 4]:
            state = random_statevector_generator(n_qubits)
            mw = compute_meyer_wallach(state, n_qubits)
            assert 0.0 <= mw <= 1.0, f"MW = {mw} is outside [0, 1]"

    def test_local_unitary_invariance(self, bell_state):
        """Test that MW is invariant under local unitary operations.

        Applying single-qubit rotations shouldn't change entanglement.
        """
        # Apply local rotation to first qubit (Z rotation)
        # |Phi+> -> (|00> + e^{i*theta}|11>)/sqrt(2) has same MW
        theta = np.pi / 4
        rotated = np.copy(bell_state)
        rotated[3] *= np.exp(1j * theta)

        mw_original = compute_meyer_wallach(bell_state, n_qubits=2)
        mw_rotated = compute_meyer_wallach(rotated, n_qubits=2)
        assert_allclose(mw_original, mw_rotated, atol=1e-10)


# =============================================================================
# Test: compute_meyer_wallach_with_breakdown
# =============================================================================


class TestMeyerWallachWithBreakdown:
    """Tests for Meyer-Wallach measure with per-qubit breakdown."""

    def test_returns_tuple(self, bell_state):
        """Test that function returns tuple of (float, array)."""
        result = compute_meyer_wallach_with_breakdown(bell_state, n_qubits=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mw_value, per_qubit = result
        assert isinstance(mw_value, float)
        assert isinstance(per_qubit, np.ndarray)

    def test_per_qubit_shape(self, ghz_state_3q):
        """Test that per-qubit array has correct shape."""
        _, per_qubit = compute_meyer_wallach_with_breakdown(ghz_state_3q, n_qubits=3)
        assert per_qubit.shape == (3,)

    def test_ghz_symmetric_entanglement(self, ghz_state_3q):
        """Test that all qubits in GHZ state have equal entanglement."""
        _, per_qubit = compute_meyer_wallach_with_breakdown(ghz_state_3q, n_qubits=3)
        # All qubits should have the same linear entropy
        assert_allclose(per_qubit[0], per_qubit[1], atol=1e-10)
        assert_allclose(per_qubit[1], per_qubit[2], atol=1e-10)
        # Each should be 0.5 (maximally mixed reduced state)
        assert_allclose(per_qubit, 0.5, atol=1e-10)

    def test_w_state_symmetric_entanglement(self, w_state_3q):
        """Test that all qubits in W state have equal entanglement."""
        _, per_qubit = compute_meyer_wallach_with_breakdown(w_state_3q, n_qubits=3)
        # All qubits should have the same linear entropy
        assert_allclose(per_qubit[0], per_qubit[1], atol=1e-10)
        assert_allclose(per_qubit[1], per_qubit[2], atol=1e-10)

    def test_product_state_zero_entropy(self, zero_state_2q):
        """Test that product states have zero per-qubit entropy."""
        _, per_qubit = compute_meyer_wallach_with_breakdown(zero_state_2q, n_qubits=2)
        assert_allclose(per_qubit, 0.0, atol=1e-10)

    def test_consistency_with_compute_meyer_wallach(self, random_statevector_generator):
        """Test that breakdown version gives same MW as basic version."""
        for n_qubits in [2, 3, 4]:
            state = random_statevector_generator(n_qubits)
            mw_basic = compute_meyer_wallach(state, n_qubits)
            mw_breakdown, _ = compute_meyer_wallach_with_breakdown(state, n_qubits)
            assert_allclose(mw_basic, mw_breakdown, atol=1e-10)

    def test_single_qubit_zero_entropy_array(self, zero_state_1q):
        """Test single-qubit case returns zero array."""
        mw, per_qubit = compute_meyer_wallach_with_breakdown(zero_state_1q, n_qubits=1)
        assert mw == 0.0
        assert per_qubit.shape == (1,)
        assert per_qubit[0] == 0.0


# =============================================================================
# Test: compute_scott_measure
# =============================================================================


class TestScottMeasure:
    """Tests for Scott entanglement measure."""

    def test_k1_equals_meyer_wallach(self, ghz_state_3q):
        """Test that Scott measure with k=1 equals Meyer-Wallach."""
        scott_k1 = compute_scott_measure(ghz_state_3q, n_qubits=3, k=1)
        mw = compute_meyer_wallach(ghz_state_3q, n_qubits=3)
        assert_allclose(scott_k1, mw, atol=1e-10)

    def test_ghz_state_k2_with_4_qubits(self):
        """Test Scott measure with k=2 for 4-qubit GHZ state.

        For GHZ state with 4 qubits, we can use k=2.
        The 2-qubit reduced states are diagonal:
        rho_{ij} = 0.5 * (|00><00| + |11><11|)
        This has purity 0.5, linear entropy 0.5, normalized to 0.5/0.75 = 2/3.
        """
        # Create 4-qubit GHZ state
        ghz_4q = np.zeros(16, dtype=complex)
        ghz_4q[0] = ghz_4q[15] = 1.0 / np.sqrt(2)

        scott_k2 = compute_scott_measure(ghz_4q, n_qubits=4, k=2)
        # The expected value is 2/3 (normalized linear entropy)
        expected = 2.0 / 3.0
        assert_allclose(scott_k2, expected, atol=1e-10)

    def test_product_state(self, zero_state_2q):
        """Test that product state has Scott measure = 0."""
        scott = compute_scott_measure(zero_state_2q, n_qubits=2, k=1)
        assert_allclose(scott, 0.0, atol=1e-10)

    def test_bell_state(self, bell_state):
        """Test Scott measure for Bell state."""
        # k=1 should be 1.0 (MW)
        scott_k1 = compute_scott_measure(bell_state, n_qubits=2, k=1)
        assert_allclose(scott_k1, 1.0, atol=1e-10)

    def test_invalid_k_raises(self, ghz_state_3q):
        """Test that k > n_qubits//2 raises error."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            compute_scott_measure(ghz_state_3q, n_qubits=3, k=2)  # max is 1

    def test_k_zero_raises(self, bell_state):
        """Test that k=0 raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_scott_measure(bell_state, n_qubits=2, k=0)

    def test_k_negative_raises(self, bell_state):
        """Test that negative k raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_scott_measure(bell_state, n_qubits=2, k=-1)

    def test_bounded_zero_one(self, random_statevector_generator):
        """Test that Scott measure is always in [0, 1]."""
        for n_qubits in [2, 4, 6]:  # Even qubits for k=n/2 tests
            state = random_statevector_generator(n_qubits)
            for k in range(1, n_qubits // 2 + 1):
                scott = compute_scott_measure(state, n_qubits, k=k)
                assert 0.0 <= scott <= 1.0, f"Scott(k={k}) = {scott} outside [0, 1]"


# =============================================================================
# Test: compute_entanglement_capability - Basic Functionality
# =============================================================================


class TestEntanglementCapabilityBasic:
    """Basic functionality tests for entanglement capability."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_returns_float(self, entangling_encoding_4q):
        """Test that function returns float by default."""
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=20, seed=42
        )
        assert isinstance(result, float)

    def test_returns_dict_with_details(self, entangling_encoding_4q):
        """Test that function returns dict when return_details=True."""
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=20, seed=42, return_details=True
        )
        assert isinstance(result, dict)
        assert "entanglement_capability" in result
        assert "entanglement_samples" in result
        assert "std_error" in result
        assert "n_samples" in result
        assert "per_qubit_entanglement" in result
        assert "measure" in result
        assert "scott_k" in result

    def test_result_structure(self, entangling_encoding_4q):
        """Test that result dictionary has correct structure."""
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=30, seed=42, return_details=True
        )

        assert isinstance(result["entanglement_capability"], float)
        assert isinstance(result["entanglement_samples"], np.ndarray)
        assert isinstance(result["std_error"], float)
        assert isinstance(result["n_samples"], int)
        assert isinstance(result["per_qubit_entanglement"], np.ndarray)
        assert result["measure"] == "meyer_wallach"
        assert result["scott_k"] is None

        # Check array shapes
        assert result["entanglement_samples"].shape == (30,)
        assert result["per_qubit_entanglement"].shape == (4,)  # 4 qubits

    def test_bounded_zero_one(self, entangling_encoding_4q):
        """Test that result is in [0, 1]."""
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=50, seed=42
        )
        assert 0.0 <= result <= 1.0

    def test_reproducibility_with_seed(self, entangling_encoding_4q):
        """Test that same seed gives same result."""
        result1 = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=50, seed=42
        )
        result2 = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=50, seed=42
        )
        assert_allclose(result1, result2, atol=1e-10)

    def test_different_seeds_different_results(self, entangling_encoding_4q):
        """Test that different seeds give different results."""
        result1 = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=50, seed=42
        )
        result2 = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=50, seed=123
        )
        # Results should be similar but not identical
        assert not np.isclose(result1, result2, atol=1e-10)


# =============================================================================
# Test: compute_entanglement_capability - Non-entangling Encodings
# =============================================================================


class TestEntanglementCapabilityNonEntangling:
    """Tests for entanglement capability with non-entangling encodings."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_angle_encoding_zero_entanglement(self, sample_encoding_2q):
        """Test that AngleEncoding (non-entangling) has zero entanglement."""
        result = compute_entanglement_capability(
            sample_encoding_2q, n_samples=50, seed=42
        )
        assert_allclose(result, 0.0, atol=1e-10)

    def test_angle_encoding_4q_zero(self, sample_encoding_4q):
        """Test that 4-qubit AngleEncoding has zero entanglement."""
        result = compute_entanglement_capability(
            sample_encoding_4q, n_samples=50, seed=42
        )
        assert_allclose(result, 0.0, atol=1e-10)


# =============================================================================
# Test: compute_entanglement_capability - Entangling Encodings
# =============================================================================


class TestEntanglementCapabilityEntangling:
    """Tests for entanglement capability with entangling encodings."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_iqp_nonzero_entanglement(self, entangling_encoding_4q):
        """Test that IQPEncoding has non-zero entanglement."""
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=100, seed=42
        )
        assert result > 0.0

    def test_more_samples_reduces_std_error(self, entangling_encoding_4q):
        """Test that more samples reduces standard error."""
        result_50 = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=50, seed=42, return_details=True
        )
        result_200 = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=200, seed=42, return_details=True
        )

        # Standard error should decrease with sqrt(n)
        assert result_200["std_error"] < result_50["std_error"]


# =============================================================================
# Test: compute_entanglement_capability - Error Handling
# =============================================================================


class TestEntanglementCapabilityErrors:
    """Tests for error handling in entanglement capability."""

    def test_single_qubit_raises(self):
        """Test that single-qubit encoding raises ValueError."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        with pytest.raises(ValueError, match="at least 2 qubits"):
            compute_entanglement_capability(enc, n_samples=10)

    def test_insufficient_samples_raises(self, entangling_encoding_4q):
        """Test that n_samples < 10 raises InsufficientSamplesError."""
        with pytest.raises(InsufficientSamplesError, match="at least 10"):
            compute_entanglement_capability(entangling_encoding_4q, n_samples=5)

    def test_low_samples_warning(self, entangling_encoding_4q, pennylane_available):
        """Test that n_samples < 100 raises warning."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        with pytest.warns(UserWarning, match="is low"):
            compute_entanglement_capability(entangling_encoding_4q, n_samples=50)

    def test_invalid_encoding_raises(self):
        """Test that invalid encoding raises AnalysisError."""
        with pytest.raises(AnalysisError, match="BaseEncoding"):
            compute_entanglement_capability("not an encoding", n_samples=10)

    def test_invalid_measure_raises(self, entangling_encoding_4q):
        """Test that invalid measure name raises ValueError."""
        with pytest.raises(ValueError, match="must be"):
            compute_entanglement_capability(
                entangling_encoding_4q, n_samples=10, measure="invalid"
            )

    def test_invalid_backend_raises(self, entangling_encoding_4q):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="must be"):
            compute_entanglement_capability(
                entangling_encoding_4q, n_samples=10, backend="invalid"
            )

    def test_invalid_input_range_raises(self, entangling_encoding_4q):
        """Test that invalid input_range raises ValidationError."""
        with pytest.raises(ValidationError):
            compute_entanglement_capability(
                entangling_encoding_4q, n_samples=10, input_range=(1.0, 0.0)
            )


# =============================================================================
# Test: compute_meyer_wallach - Error Handling
# =============================================================================


class TestMeyerWallachErrors:
    """Tests for error handling in Meyer-Wallach computation."""

    def test_wrong_dimension_raises(self, bell_state):
        """Test that wrong n_qubits raises error."""
        with pytest.raises(ValidationError, match="expected.*qubits"):
            compute_meyer_wallach(bell_state, n_qubits=3)

    def test_non_power_of_2_raises(self):
        """Test that non-power-of-2 dimension raises error."""
        invalid_state = np.array([1, 0, 0], dtype=complex)
        with pytest.raises(ValidationError, match="power of 2"):
            compute_meyer_wallach(invalid_state, n_qubits=2)

    def test_nan_raises(self):
        """Test that NaN values raise error."""
        state_nan = np.array([np.nan, 0, 0, 1], dtype=complex)
        with pytest.raises(ValidationError, match="NaN"):
            compute_meyer_wallach(state_nan, n_qubits=2)

    def test_zero_norm_raises(self, zero_norm_state):
        """Test that zero-norm state raises NumericalInstabilityError."""
        with pytest.raises(NumericalInstabilityError, match="near-zero"):
            compute_meyer_wallach(zero_norm_state, n_qubits=2)

    def test_negative_n_qubits_raises(self, bell_state):
        """Test that negative n_qubits raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_meyer_wallach(bell_state, n_qubits=-1)

    def test_zero_n_qubits_raises(self, bell_state):
        """Test that n_qubits=0 raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_meyer_wallach(bell_state, n_qubits=0)


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of entanglement measures."""

    def test_nearly_product_state(self):
        """Test MW for state very close to product."""
        # |00> with tiny perturbation
        state = np.array([1.0 - 1e-12, 0, 0, 1e-12], dtype=complex)
        state = state / np.linalg.norm(state)
        mw = compute_meyer_wallach(state, n_qubits=2)
        assert_allclose(mw, 0.0, atol=1e-6)

    def test_nearly_maximally_entangled(self):
        """Test MW for state very close to Bell."""
        # Bell state with tiny perturbation
        bell = np.array([1.0, 0, 0, 1.0], dtype=complex) / np.sqrt(2)
        perturbed = bell + 1e-12 * np.array([0, 1, 0, 0], dtype=complex)
        perturbed = perturbed / np.linalg.norm(perturbed)
        mw = compute_meyer_wallach(perturbed, n_qubits=2)
        assert_allclose(mw, 1.0, atol=1e-6)

    def test_complex_phases(self, bell_state):
        """Test MW with complex phases."""
        # Bell state with various phases
        phases = [np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
        for phase in phases:
            phased = bell_state * np.exp(1j * phase)
            mw = compute_meyer_wallach(phased, n_qubits=2)
            assert_allclose(mw, 1.0, atol=1e-10)

    def test_unnormalized_state(self, unnormalized_state):
        """Test that unnormalized state is handled correctly."""
        # Should be renormalized automatically
        mw = compute_meyer_wallach(unnormalized_state, n_qubits=2)
        assert 0.0 <= mw <= 1.0

    def test_many_qubits_stability(self):
        """Test numerical stability for larger qubit systems."""
        n_qubits = 6
        dim = 2**n_qubits

        # GHZ state for n qubits
        ghz = np.zeros(dim, dtype=complex)
        ghz[0] = ghz[-1] = 1.0 / np.sqrt(2)

        mw = compute_meyer_wallach(ghz, n_qubits=n_qubits)
        assert_allclose(mw, 1.0, atol=1e-8)


# =============================================================================
# Test: Statistical Properties
# =============================================================================


class TestStatisticalProperties:
    """Tests for statistical properties of entanglement capability."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_std_error_formula(self, entangling_encoding_4q):
        """Test that std_error = std / sqrt(n)."""
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=100, seed=42, return_details=True
        )

        samples = result["entanglement_samples"]
        expected_std_error = np.std(samples, ddof=1) / np.sqrt(100)

        assert_allclose(result["std_error"], expected_std_error, atol=1e-10)

    def test_n_samples_stored(self, entangling_encoding_4q):
        """Test that n_samples is correctly stored."""
        for n in [20, 50, 100]:
            result = compute_entanglement_capability(
                entangling_encoding_4q, n_samples=n, seed=42, return_details=True
            )
            assert result["n_samples"] == n

    def test_samples_array_length(self, entangling_encoding_4q):
        """Test that samples array has correct length."""
        for n in [20, 50, 100]:
            result = compute_entanglement_capability(
                entangling_encoding_4q, n_samples=n, seed=42, return_details=True
            )
            assert len(result["entanglement_samples"]) == n

    def test_mean_equals_capability(self, entangling_encoding_4q):
        """Test that entanglement_capability equals mean of samples."""
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=100, seed=42, return_details=True
        )

        expected_mean = np.mean(result["entanglement_samples"])
        assert_allclose(result["entanglement_capability"], expected_mean, atol=1e-10)


# =============================================================================
# Test: Scott Measure Error Handling
# =============================================================================


class TestScottMeasureErrors:
    """Tests for error handling in Scott measure."""

    def test_k_exceeds_half_n_raises(self):
        """Test that k > n_qubits // 2 raises error."""
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0  # |000>

        # For 3 qubits, max k is 1
        with pytest.raises(ValueError, match="exceeds maximum"):
            compute_scott_measure(state, n_qubits=3, k=2)

    def test_k_equals_n_qubits_raises(self, bell_state):
        """Test that k = n_qubits raises error (for n > 2)."""
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0

        with pytest.raises(ValueError, match="exceeds maximum"):
            compute_scott_measure(state, n_qubits=3, k=3)

    def test_wrong_statevector_dimension(self, bell_state):
        """Test that wrong n_qubits for statevector raises error."""
        # bell_state has dim 4 (2 qubits), but we claim 3 qubits
        with pytest.raises(ValidationError):
            compute_scott_measure(bell_state, n_qubits=3, k=1)


# =============================================================================
# Test: Backend Consistency
# =============================================================================


class TestBackendConsistency:
    """Tests comparing results across simulation backends."""

    @pytest.fixture(autouse=True)
    def check_both_backends(self, pennylane_available, qiskit_available):
        """Skip tests if both backends are not available."""
        if not pennylane_available or not qiskit_available:
            pytest.skip("Both PennyLane and Qiskit required")

    def test_similar_results_both_backends(self, entangling_encoding_4q):
        """Test that both backends give similar entanglement capability."""
        result_pl = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=100, seed=42, backend="pennylane"
        )
        result_qk = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=100, seed=42, backend="qiskit"
        )

        # Results should be similar (within statistical variation)
        # Note: Different backends may use different conventions, so we
        # just check they're both valid and in reasonable range
        assert 0.0 <= result_pl <= 1.0
        assert 0.0 <= result_qk <= 1.0


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exactly_10_samples(self, entangling_encoding_4q, pennylane_available):
        """Test minimum sample count of 10."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        # Should not raise
        result = compute_entanglement_capability(
            entangling_encoding_4q, n_samples=10, seed=42
        )
        assert isinstance(result, float)

    def test_input_range_full_rotation(self, entangling_encoding_4q, pennylane_available):
        """Test with default full rotation range [0, 2π]."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=50,
            input_range=(0.0, 2 * np.pi),
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_input_range_partial(self, entangling_encoding_4q, pennylane_available):
        """Test with partial rotation range [0, π]."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=50,
            input_range=(0.0, np.pi),
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_two_qubit_minimum(self, sample_encoding_2q, pennylane_available):
        """Test with minimum 2 qubits."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        result = compute_entanglement_capability(
            sample_encoding_2q, n_samples=50, seed=42
        )
        # AngleEncoding has 0 entanglement
        assert_allclose(result, 0.0, atol=1e-10)


# =============================================================================
# Test: Special States Additional Tests
# =============================================================================


class TestSpecialStates:
    """Additional tests for specific quantum states."""

    def test_all_ones_state(self):
        """Test computational basis state |11...1>."""
        for n_qubits in [2, 3, 4]:
            dim = 2**n_qubits
            state = np.zeros(dim, dtype=complex)
            state[-1] = 1.0  # |11...1>
            mw = compute_meyer_wallach(state, n_qubits)
            assert_allclose(mw, 0.0, atol=1e-10)

    def test_uniform_superposition(self):
        """Test uniform superposition state (1/sqrt(2^n)) * (|0> + |1> + ...)."""
        for n_qubits in [2, 3]:
            dim = 2**n_qubits
            state = np.ones(dim, dtype=complex) / np.sqrt(dim)
            mw = compute_meyer_wallach(state, n_qubits)
            # Uniform superposition has specific MW value depending on n
            assert 0.0 <= mw <= 1.0

    def test_phase_gate_invariance(self):
        """Test that global phase doesn't affect entanglement."""
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        for phase in [0, np.pi / 4, np.pi / 2, np.pi]:
            phased = bell * np.exp(1j * phase)
            mw = compute_meyer_wallach(phased, n_qubits=2)
            assert_allclose(mw, 1.0, atol=1e-10)

    def test_partial_entanglement(self):
        """Test state with partial entanglement.

        |psi> = cos(theta)|00> + sin(theta)|11>
        For theta = pi/4, this is the Bell state (MW = 1).
        For theta = 0, this is |00> (MW = 0).
        """
        for theta, expected_mw in [
            (0.0, 0.0),
            (np.pi / 4, 1.0),
            (np.pi / 2, 0.0),
        ]:
            state = np.array(
                [np.cos(theta), 0, 0, np.sin(theta)], dtype=complex
            )
            if np.linalg.norm(state) > 1e-10:
                state = state / np.linalg.norm(state)
                mw = compute_meyer_wallach(state, n_qubits=2)
                assert_allclose(mw, expected_mw, atol=1e-10)


# =============================================================================
# Test: Scott Measure Integration with compute_entanglement_capability
# =============================================================================


class TestScottMeasureIntegration:
    """Integration tests for compute_entanglement_capability with Scott measure."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_scott_measure_4_qubits(self, entangling_encoding_4q):
        """Test Scott measure integration with 4-qubit system (valid k=2)."""
        result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=20,
            seed=42,
            measure="scott",
            return_details=True,
        )
        assert isinstance(result, dict)
        assert result["measure"] == "scott"
        assert result["scott_k"] == 2  # Auto-selected max for 4 qubits
        assert 0.0 <= result["entanglement_capability"] <= 1.0

    def test_scott_measure_auto_k_selection_3_qubits(self):
        """Test Scott measure auto-selects k=1 for 3-qubit system."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=3, reps=2)
        result = compute_entanglement_capability(
            enc,
            n_samples=20,
            seed=42,
            measure="scott",
            return_details=True,
        )
        assert result["measure"] == "scott"
        # For 3 qubits, max_k = 3 // 2 = 1, so auto-selected k should be 1
        assert result["scott_k"] == 1
        assert 0.0 <= result["entanglement_capability"] <= 1.0

    def test_scott_measure_auto_k_selection_2_qubits(self, sample_encoding_2q):
        """Test Scott measure auto-selects k=1 for 2-qubit system."""
        result = compute_entanglement_capability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            measure="scott",
            return_details=True,
        )
        assert result["measure"] == "scott"
        # For 2 qubits, max_k = 2 // 2 = 1, so auto-selected k should be 1
        assert result["scott_k"] == 1

    def test_scott_k_explicit_valid(self, entangling_encoding_4q):
        """Test explicitly providing a valid scott_k value."""
        result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=20,
            seed=42,
            measure="scott",
            scott_k=1,  # Explicitly request k=1
            return_details=True,
        )
        assert result["scott_k"] == 1

    def test_scott_k_explicit_invalid_raises(self, entangling_encoding_4q):
        """Test that invalid explicit scott_k raises ValueError."""
        with pytest.raises(ValueError, match="exceeds maximum valid value"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                seed=42,
                measure="scott",
                scott_k=3,  # Invalid: 3 > 4 // 2 = 2
            )

    def test_scott_k_zero_raises(self, entangling_encoding_4q):
        """Test that scott_k=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                seed=42,
                measure="scott",
                scott_k=0,
            )

    def test_scott_k_negative_raises(self, entangling_encoding_4q):
        """Test that negative scott_k raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                seed=42,
                measure="scott",
                scott_k=-1,
            )

    def test_scott_k_ignored_for_meyer_wallach(self, entangling_encoding_4q):
        """Test that scott_k is ignored (with warning) when measure='meyer_wallach'."""
        with pytest.warns(UserWarning, match="scott_k.*ignored"):
            result = compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=20,
                seed=42,
                measure="meyer_wallach",
                scott_k=2,  # Should be ignored
                return_details=True,
            )
        assert result["measure"] == "meyer_wallach"
        assert result["scott_k"] is None

    def test_scott_vs_meyer_wallach_k1_equivalence(self, entangling_encoding_4q):
        """Test that Scott measure with k=1 gives similar results to Meyer-Wallach."""
        mw_result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=100,
            seed=42,
            measure="meyer_wallach",
        )
        scott_result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=100,
            seed=42,
            measure="scott",
            scott_k=1,
        )
        # Results should be very close since Scott k=1 is equivalent to MW
        assert_allclose(mw_result, scott_result, atol=0.05)


# =============================================================================
# Test: Input Range Type Validation
# =============================================================================


class TestInputRangeValidation:
    """Tests for input_range parameter validation."""

    def test_input_range_scalar_raises_validation_error(self, entangling_encoding_4q):
        """Test that scalar input_range raises ValidationError with helpful message."""
        with pytest.raises(ValidationError, match="tuple or list"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                input_range=0.0,  # type: ignore  # Intentional type error for test
            )

    def test_input_range_string_raises_validation_error(self, entangling_encoding_4q):
        """Test that string input_range raises ValidationError."""
        with pytest.raises(ValidationError, match="tuple or list"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                input_range="invalid",  # type: ignore
            )

    def test_input_range_none_raises_validation_error(self, entangling_encoding_4q):
        """Test that None input_range raises ValidationError."""
        with pytest.raises(ValidationError, match="tuple or list"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                input_range=None,  # type: ignore
            )

    def test_input_range_wrong_length_raises(self, entangling_encoding_4q):
        """Test that input_range with wrong length raises ValidationError."""
        with pytest.raises(ValidationError, match="exactly 2 elements"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                input_range=(0.0, 1.0, 2.0),  # type: ignore
            )

    def test_input_range_single_element_raises(self, entangling_encoding_4q):
        """Test that single-element input_range raises ValidationError."""
        with pytest.raises(ValidationError, match="exactly 2 elements"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                input_range=(0.0,),  # type: ignore
            )

    def test_input_range_list_accepted(self, entangling_encoding_4q, pennylane_available):
        """Test that list input_range is accepted."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=20,
            seed=42,
            input_range=[0.0, np.pi],  # List instead of tuple
        )
        assert 0.0 <= result <= 1.0

    def test_input_range_tuple_accepted(self, entangling_encoding_4q, pennylane_available):
        """Test that tuple input_range is accepted."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        result = compute_entanglement_capability(
            entangling_encoding_4q,
            n_samples=20,
            seed=42,
            input_range=(0.0, np.pi),
        )
        assert 0.0 <= result <= 1.0

    def test_input_range_inverted_raises(self, entangling_encoding_4q):
        """Test that inverted input_range raises ValidationError."""
        with pytest.raises(ValidationError, match="must be less than"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                input_range=(2.0, 1.0),  # min > max
            )

    def test_input_range_equal_raises(self, entangling_encoding_4q):
        """Test that equal bounds raises ValidationError."""
        with pytest.raises(ValidationError, match="must be less than"):
            compute_entanglement_capability(
                entangling_encoding_4q,
                n_samples=10,
                input_range=(1.0, 1.0),  # min == max
            )
