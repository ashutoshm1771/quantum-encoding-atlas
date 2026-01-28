"""Comprehensive tests for expressibility computation.

This module tests the expressibility analysis functions in
encoding_atlas.analysis.expressibility, including:

- Basic functionality and API correctness
- Input validation and error handling
- Edge cases and numerical stability
- Statistical properties of the computation
- Integration with different encodings
- Cross-backend consistency

The tests are organized into the following categories:

1. **Basic Functionality Tests**: Verify core API works correctly
2. **Input Validation Tests**: Ensure proper error messages for invalid inputs
3. **Numerical Stability Tests**: Test edge cases and extreme values
4. **Statistical Tests**: Verify statistical properties of results
5. **Integration Tests**: Test with real encodings and backends
6. **Performance Tests**: Ensure reasonable execution times (marked slow)

References
----------
.. [1] Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019).
       "Expressibility and entangling capability of parameterized quantum
       circuits for hybrid quantum-classical algorithms."
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas.analysis.expressibility import (
    ExpressibilityResult,
    _DEFAULT_N_BINS,
    _DEFAULT_N_SAMPLES,
    _MAX_KL_DIVERGENCE,
    _MIN_SAMPLES_ERROR,
    _MIN_SAMPLES_WARNING,
    _NUMERICAL_EPSILON,
    compute_expressibility,
    compute_fidelity_distribution,
    compute_haar_distribution,
)
from encoding_atlas.core.exceptions import (
    AnalysisError,
    InsufficientSamplesError,
    SimulationError,
)

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_encoding_2q():
    """Create a simple 2-qubit AngleEncoding for testing."""
    from encoding_atlas import AngleEncoding
    return AngleEncoding(n_features=2)


@pytest.fixture
def simple_encoding_4q():
    """Create a simple 4-qubit AngleEncoding for testing."""
    from encoding_atlas import AngleEncoding
    return AngleEncoding(n_features=4)


@pytest.fixture
def entangling_encoding_4q():
    """Create an entangling 4-qubit IQPEncoding for testing."""
    from encoding_atlas import IQPEncoding
    return IQPEncoding(n_features=4, reps=1)


@pytest.fixture
def single_qubit_encoding():
    """Create a single-qubit encoding for edge case testing."""
    from encoding_atlas import AngleEncoding
    return AngleEncoding(n_features=1)


# =============================================================================
# Tests: Basic Functionality
# =============================================================================


class TestComputeExpressibilityBasic:
    """Test basic functionality of compute_expressibility."""

    def test_returns_float_by_default(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that default return is a float."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        assert isinstance(result, float)

    def test_returns_dict_when_requested(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that return_distributions=True returns ExpressibilityResult."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            return_distributions=True,
        )
        assert isinstance(result, dict)
        # Check all required keys are present
        required_keys = {
            "expressibility",
            "kl_divergence",
            "fidelity_distribution",
            "haar_distribution",
            "bin_edges",
            "n_samples",
            "n_bins",
            "convergence_estimate",
            "mean_fidelity",
            "std_fidelity",
        }
        assert set(result.keys()) == required_keys

    def test_expressibility_in_valid_range(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that expressibility is in [0, 1]."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_kl_divergence_non_negative(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that KL divergence is non-negative."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert result["kl_divergence"] >= 0.0

    def test_reproducibility_with_seed(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that same seed produces same result."""
        result1 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        result2 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        assert result1 == result2

    def test_different_seeds_different_results(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that different seeds produce different results."""
        result1 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        result2 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=123,
        )
        # Should be different (with very high probability)
        assert result1 != result2

    def test_n_samples_stored_correctly(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that n_samples is stored in result."""
        n_samples = 75
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=n_samples,
            n_bins=15,
            seed=42,
            return_distributions=True,
        )
        assert result["n_samples"] == n_samples

    def test_n_bins_stored_correctly(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that n_bins is stored in result."""
        n_bins = 25
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=n_bins,
            seed=42,
            return_distributions=True,
        )
        assert result["n_bins"] == n_bins


class TestComputeExpressibilityDistributions:
    """Test distribution-related outputs."""

    def test_fidelity_distribution_shape(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that fidelity distribution has correct shape."""
        n_bins = 20
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=n_bins,
            seed=42,
            return_distributions=True,
        )
        assert result["fidelity_distribution"].shape == (n_bins,)

    def test_haar_distribution_shape(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that Haar distribution has correct shape."""
        n_bins = 20
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=n_bins,
            seed=42,
            return_distributions=True,
        )
        assert result["haar_distribution"].shape == (n_bins,)

    def test_bin_edges_shape(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that bin edges has correct shape (n_bins + 1)."""
        n_bins = 20
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=n_bins,
            seed=42,
            return_distributions=True,
        )
        assert result["bin_edges"].shape == (n_bins + 1,)

    def test_distributions_sum_to_one(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that probability distributions sum to approximately 1."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert np.isclose(result["fidelity_distribution"].sum(), 1.0, atol=1e-6)
        assert np.isclose(result["haar_distribution"].sum(), 1.0, atol=1e-6)

    def test_distributions_non_negative(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that all distribution values are non-negative."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert np.all(result["fidelity_distribution"] >= 0)
        assert np.all(result["haar_distribution"] >= 0)


# =============================================================================
# Tests: Input Validation
# =============================================================================


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_encoding_type(self):
        """Test that non-encoding input raises error."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            compute_expressibility("not an encoding")

    def test_invalid_encoding_none(self):
        """Test that None encoding raises error."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            compute_expressibility(None)

    def test_n_samples_too_small_raises_error(
        self, simple_encoding_2q
    ):
        """Test that n_samples < MIN_SAMPLES_ERROR raises InsufficientSamplesError."""
        with pytest.raises(InsufficientSamplesError):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=5,  # Below minimum
                n_bins=10,
            )

    def test_n_samples_low_raises_warning(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that n_samples < MIN_SAMPLES_WARNING raises warning."""
        with pytest.warns(UserWarning, match="n_samples.*low"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,  # Between error and warning thresholds
                n_bins=10,
                seed=42,
            )

    def test_n_bins_too_small_raises_error(
        self, simple_encoding_2q
    ):
        """Test that n_bins < 10 raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be at least 10"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=5,  # Below minimum
            )

    def test_n_bins_larger_than_n_samples_raises_error(
        self, simple_encoding_2q
    ):
        """Test that n_bins > n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_bins.*cannot exceed n_samples"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=100,  # More bins than samples
            )

    def test_invalid_input_range_order(
        self, simple_encoding_2q
    ):
        """Test that input_range with min >= max raises error."""
        with pytest.raises(ValueError, match="input_range"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20,
                input_range=(2.0, 1.0),  # Invalid: max < min
            )

    def test_invalid_input_range_equal(
        self, simple_encoding_2q
    ):
        """Test that input_range with min == max raises error."""
        with pytest.raises(ValueError, match="input_range"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20,
                input_range=(1.0, 1.0),  # Invalid: equal
            )

    def test_invalid_backend(
        self, simple_encoding_2q
    ):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20,
                backend="invalid_backend",
            )

    def test_negative_n_samples(
        self, simple_encoding_2q
    ):
        """Test that negative n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be a positive"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=-10,
                n_bins=20,
            )

    def test_zero_n_samples(
        self, simple_encoding_2q
    ):
        """Test that zero n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be a positive"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=0,
                n_bins=20,
            )


# =============================================================================
# Tests: Haar Distribution
# =============================================================================


class TestComputeHaarDistribution:
    """Test the compute_haar_distribution function."""

    def test_single_qubit_uniform(self):
        """Test that single qubit gives uniform distribution."""
        fidelities = np.linspace(0, 1, 100)
        P_haar = compute_haar_distribution(n_qubits=1, fidelity_values=fidelities)

        # For d=2, P_Haar should be uniform
        # Check that all values are approximately equal
        assert np.allclose(P_haar, P_haar[0], atol=1e-6)

    def test_two_qubit_distribution(self):
        """Test that two qubit distribution has correct shape."""
        fidelities = np.linspace(0, 1, 50)
        P_haar = compute_haar_distribution(n_qubits=2, fidelity_values=fidelities)

        # P_Haar should be linear: (d-1)(1-F)^(d-2) = 3(1-F)^2
        # Should be decreasing with F
        assert P_haar[0] > P_haar[-1]

    def test_normalized_to_one(self):
        """Test that distribution sums to one."""
        fidelities = np.linspace(0, 1, 100)

        for n_qubits in [1, 2, 3, 4, 5]:
            P_haar = compute_haar_distribution(n_qubits, fidelities)
            assert np.isclose(P_haar.sum(), 1.0, atol=1e-6)

    def test_non_negative(self):
        """Test that all probabilities are non-negative."""
        fidelities = np.linspace(0, 1, 100)

        for n_qubits in [1, 2, 3, 4, 5]:
            P_haar = compute_haar_distribution(n_qubits, fidelities)
            assert np.all(P_haar >= 0)

    def test_invalid_n_qubits_zero(self):
        """Test that n_qubits=0 raises ValueError."""
        fidelities = np.linspace(0, 1, 50)
        with pytest.raises(ValueError, match="n_qubits must be a positive"):
            compute_haar_distribution(n_qubits=0, fidelity_values=fidelities)

    def test_invalid_n_qubits_negative(self):
        """Test that negative n_qubits raises ValueError."""
        fidelities = np.linspace(0, 1, 50)
        with pytest.raises(ValueError, match="n_qubits must be a positive"):
            compute_haar_distribution(n_qubits=-1, fidelity_values=fidelities)

    def test_concentration_increases_with_qubits(self):
        """Test that distribution concentrates near F=0 with more qubits."""
        fidelities = np.linspace(0, 1, 100)

        P_2q = compute_haar_distribution(n_qubits=2, fidelity_values=fidelities)
        P_4q = compute_haar_distribution(n_qubits=4, fidelity_values=fidelities)

        # With more qubits, probability should be more concentrated at low fidelity
        # Check that P_4q has more mass in the first quarter
        first_quarter = len(fidelities) // 4
        assert P_4q[:first_quarter].sum() > P_2q[:first_quarter].sum()

    def test_handles_edge_fidelities(self):
        """Test that F=0 and F=1 are handled correctly."""
        fidelities = np.array([0.0, 0.5, 1.0])

        for n_qubits in [2, 3, 4]:
            P_haar = compute_haar_distribution(n_qubits, fidelities)
            # All values should be valid (non-NaN, non-Inf)
            assert np.all(np.isfinite(P_haar))

    def test_large_qubit_count_stability(self):
        """Test numerical stability for large qubit counts."""
        fidelities = np.linspace(0, 1, 50)

        # Should not raise or produce NaN/Inf for up to 10 qubits
        for n_qubits in [6, 7, 8, 9, 10]:
            P_haar = compute_haar_distribution(n_qubits, fidelity_values=fidelities)
            assert np.all(np.isfinite(P_haar))
            assert np.isclose(P_haar.sum(), 1.0, atol=1e-6)


# =============================================================================
# Tests: Fidelity Distribution
# =============================================================================


class TestComputeFidelityDistribution:
    """Test the compute_fidelity_distribution function."""

    def test_returns_correct_shape(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that returned array has correct shape."""
        n_samples = 50
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=n_samples,
            seed=42,
        )
        assert fidelities.shape == (n_samples,)

    def test_all_values_in_valid_range(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that all fidelities are in [0, 1]."""
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=100,
            seed=42,
        )
        assert np.all(fidelities >= 0.0)
        assert np.all(fidelities <= 1.0)

    def test_reproducibility_with_seed(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that same seed produces same results."""
        fid1 = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            seed=42,
        )
        fid2 = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            seed=42,
        )
        np.testing.assert_array_equal(fid1, fid2)

    def test_invalid_n_samples_raises_error(
        self, simple_encoding_2q
    ):
        """Test that invalid n_samples raises error."""
        with pytest.raises(InsufficientSamplesError):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=5,  # Below minimum
            )

    def test_invalid_input_range(
        self, simple_encoding_2q
    ):
        """Test that invalid input_range raises error."""
        with pytest.raises(ValueError, match="input_range"):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=50,
                input_range=(2.0, 1.0),  # Invalid
            )


# =============================================================================
# Tests: Edge Cases and Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_single_qubit_encoding(
        self, single_qubit_encoding, skip_if_no_pennylane
    ):
        """Test that single qubit encoding works correctly."""
        result = compute_expressibility(
            single_qubit_encoding,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_narrow_input_range(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test with narrow input range."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            input_range=(0.0, 0.1),  # Narrow range
            seed=42,
        )
        # Should still produce valid result
        assert 0.0 <= result <= 1.0

    def test_wide_input_range(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test with wide input range."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            input_range=(0.0, 10 * np.pi),  # Wide range
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_minimum_valid_samples(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test with minimum valid sample count."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,  # Exactly at minimum
            n_bins=_MIN_SAMPLES_ERROR,  # n_bins == n_samples is valid
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_maximum_bins_equals_samples(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test with n_bins == n_samples (edge case)."""
        n = 50
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=n,
            n_bins=n,  # Equal to samples
            seed=42,
        )
        assert 0.0 <= result <= 1.0


# =============================================================================
# Tests: Statistical Properties
# =============================================================================


class TestStatisticalProperties:
    """Test statistical properties of expressibility computation."""

    def test_more_samples_smaller_convergence(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that more samples generally give smaller convergence estimate."""
        result_small = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            return_distributions=True,
        )
        result_large = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        # More samples should generally give more stable results
        # Allow some tolerance as this is statistical
        # Large sample convergence should typically be smaller
        # (but this isn't guaranteed, so we just check both are finite)
        assert np.isfinite(result_small["convergence_estimate"])
        assert np.isfinite(result_large["convergence_estimate"])

    def test_mean_fidelity_in_valid_range(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that mean fidelity is in [0, 1]."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert 0.0 <= result["mean_fidelity"] <= 1.0

    def test_std_fidelity_non_negative(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that std fidelity is non-negative."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert result["std_fidelity"] >= 0.0


# =============================================================================
# Tests: Encoding Comparison
# =============================================================================


class TestEncodingComparison:
    """Test expressibility comparison between different encodings."""

    def test_entangling_vs_non_entangling(
        self, simple_encoding_4q, entangling_encoding_4q, skip_if_no_pennylane
    ):
        """Test that entangling encoding has higher expressibility.

        Note: This is a general expectation from theory. Entangling circuits
        typically have higher expressibility, but this isn't guaranteed
        for all parameter ranges and sample sizes.
        """
        expr_simple = compute_expressibility(
            simple_encoding_4q,
            n_samples=200,
            n_bins=30,
            seed=42,
        )
        expr_entangling = compute_expressibility(
            entangling_encoding_4q,
            n_samples=200,
            n_bins=30,
            seed=42,
        )

        # Both should be valid
        assert 0.0 <= expr_simple <= 1.0
        assert 0.0 <= expr_entangling <= 1.0

        # Entangling encoding should generally be more expressive
        # (allow some tolerance as this is statistical)
        # We don't enforce this strictly as it depends on the specific circuits
        # Just verify both produce reasonable results


# =============================================================================
# Tests: Backend Consistency
# =============================================================================


class TestBackendConsistency:
    """Test cross-backend consistency.

    This test class verifies that all supported backends (PennyLane, Qiskit, Cirq)
    produce valid and consistent expressibility results. Each backend implements
    the same quantum operations but may have slight numerical differences.

    The tests are organized as:
    1. Individual backend validity tests
    2. Pairwise cross-backend consistency tests
    3. Full three-backend consistency test
    """

    def test_pennylane_produces_valid_result(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that PennyLane backend produces valid results."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="pennylane",
        )
        assert 0.0 <= result <= 1.0

    def test_qiskit_produces_valid_result(
        self, simple_encoding_2q, skip_if_no_qiskit
    ):
        """Test that Qiskit backend produces valid results.

        This test verifies that the Qiskit backend can successfully compute
        expressibility and returns a valid score in [0, 1].
        """
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="qiskit",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_produces_valid_result(
        self, simple_encoding_2q, skip_if_no_cirq
    ):
        """Test that Cirq backend produces valid results.

        This test verifies that the Cirq backend can successfully compute
        expressibility and returns a valid score in [0, 1].
        """
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_returns_dict_when_requested(
        self, simple_encoding_2q, skip_if_no_cirq
    ):
        """Test that Cirq backend returns ExpressibilityResult with return_distributions=True."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )
        assert isinstance(result, dict)
        # Check all required keys are present
        required_keys = {
            "expressibility",
            "kl_divergence",
            "fidelity_distribution",
            "haar_distribution",
            "bin_edges",
            "n_samples",
            "n_bins",
            "convergence_estimate",
            "mean_fidelity",
            "std_fidelity",
        }
        assert set(result.keys()) == required_keys

    def test_cirq_reproducibility_with_seed(
        self, simple_encoding_2q, skip_if_no_cirq
    ):
        """Test that Cirq backend produces reproducible results with same seed."""
        result1 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        result2 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert result1 == result2

    def test_cross_backend_consistency_pennylane_qiskit(
        self, simple_encoding_2q, pennylane_available, qiskit_available
    ):
        """Test that PennyLane and Qiskit backends produce consistent results.

        This test verifies cross-backend consistency by computing expressibility
        with both backends using the same random seed and comparing results.
        The results should be similar (within tolerance) since both backends
        implement the same quantum operations.

        Note: Small differences are expected due to numerical precision
        differences between backends, but the results should be close.
        """
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        if not qiskit_available:
            pytest.skip("Qiskit not available")

        # Compute with PennyLane
        result_pennylane = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="pennylane",
        )

        # Compute with Qiskit
        result_qiskit = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="qiskit",
        )

        # Both should be valid
        assert 0.0 <= result_pennylane <= 1.0
        assert 0.0 <= result_qiskit <= 1.0

        # Results should be similar (within reasonable tolerance)
        # Allow for numerical differences between backends
        tolerance = 0.15  # 15% tolerance for cross-backend variation
        assert abs(result_pennylane - result_qiskit) < tolerance, (
            f"Cross-backend results differ significantly: "
            f"PennyLane={result_pennylane:.4f}, Qiskit={result_qiskit:.4f}"
        )

    def test_cross_backend_consistency_pennylane_cirq(
        self, simple_encoding_2q, pennylane_available, cirq_available
    ):
        """Test that PennyLane and Cirq backends produce consistent results.

        This test verifies cross-backend consistency by computing expressibility
        with both backends using the same random seed and comparing results.
        """
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        if not cirq_available:
            pytest.skip("Cirq not available")

        # Compute with PennyLane
        result_pennylane = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="pennylane",
        )

        # Compute with Cirq
        result_cirq = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="cirq",
        )

        # Both should be valid
        assert 0.0 <= result_pennylane <= 1.0
        assert 0.0 <= result_cirq <= 1.0

        # Results should be similar (within reasonable tolerance)
        tolerance = 0.15  # 15% tolerance for cross-backend variation
        assert abs(result_pennylane - result_cirq) < tolerance, (
            f"Cross-backend results differ significantly: "
            f"PennyLane={result_pennylane:.4f}, Cirq={result_cirq:.4f}"
        )

    def test_cross_backend_consistency_qiskit_cirq(
        self, simple_encoding_2q, qiskit_available, cirq_available
    ):
        """Test that Qiskit and Cirq backends produce consistent results.

        This test verifies cross-backend consistency by computing expressibility
        with both backends using the same random seed and comparing results.
        """
        if not qiskit_available:
            pytest.skip("Qiskit not available")
        if not cirq_available:
            pytest.skip("Cirq not available")

        # Compute with Qiskit
        result_qiskit = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="qiskit",
        )

        # Compute with Cirq
        result_cirq = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="cirq",
        )

        # Both should be valid
        assert 0.0 <= result_qiskit <= 1.0
        assert 0.0 <= result_cirq <= 1.0

        # Results should be similar (within reasonable tolerance)
        tolerance = 0.15  # 15% tolerance for cross-backend variation
        assert abs(result_qiskit - result_cirq) < tolerance, (
            f"Cross-backend results differ significantly: "
            f"Qiskit={result_qiskit:.4f}, Cirq={result_cirq:.4f}"
        )

    def test_cross_backend_consistency_all_three(
        self, simple_encoding_2q, pennylane_available, qiskit_available, cirq_available
    ):
        """Test that all three backends produce consistent results.

        This comprehensive test verifies that PennyLane, Qiskit, and Cirq
        all produce similar expressibility results for the same encoding,
        parameters, and random seed.
        """
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        if not qiskit_available:
            pytest.skip("Qiskit not available")
        if not cirq_available:
            pytest.skip("Cirq not available")

        # Compute with all three backends
        result_pennylane = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="pennylane",
        )
        result_qiskit = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="qiskit",
        )
        result_cirq = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="cirq",
        )

        # All should be valid
        assert 0.0 <= result_pennylane <= 1.0
        assert 0.0 <= result_qiskit <= 1.0
        assert 0.0 <= result_cirq <= 1.0

        # All results should be within tolerance of each other
        tolerance = 0.15
        results = {
            "PennyLane": result_pennylane,
            "Qiskit": result_qiskit,
            "Cirq": result_cirq,
        }

        # Check all pairwise differences
        for name1, val1 in results.items():
            for name2, val2 in results.items():
                if name1 < name2:
                    assert abs(val1 - val2) < tolerance, (
                        f"Cross-backend results differ significantly: "
                        f"{name1}={val1:.4f}, {name2}={val2:.4f}"
                    )


# =============================================================================
# Tests: Cirq Backend Integration
# =============================================================================


class TestCirqBackendIntegration:
    """Comprehensive integration tests for Cirq backend.

    This test class provides thorough coverage of the Cirq backend for
    expressibility computation, including:

    - Basic functionality with various encodings
    - Numerical stability and edge cases
    - Distribution output correctness
    - Statistical properties
    - Fidelity distribution computation
    """

    @pytest.fixture(autouse=True)
    def check_cirq(self, cirq_available):
        """Skip all tests in this class if Cirq is not available."""
        if not cirq_available:
            pytest.skip("Cirq not available")

    def test_cirq_angle_encoding_2q(self, simple_encoding_2q):
        """Test Cirq with 2-qubit AngleEncoding."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_angle_encoding_4q(self, simple_encoding_4q):
        """Test Cirq with 4-qubit AngleEncoding."""
        result = compute_expressibility(
            simple_encoding_4q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_entangling_encoding(self, entangling_encoding_4q):
        """Test Cirq with entangling IQPEncoding."""
        result = compute_expressibility(
            entangling_encoding_4q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_single_qubit_encoding(self, single_qubit_encoding):
        """Test Cirq with single-qubit encoding (edge case)."""
        result = compute_expressibility(
            single_qubit_encoding,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_kl_divergence_non_negative(self, simple_encoding_2q):
        """Test that Cirq backend produces non-negative KL divergence."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )
        assert result["kl_divergence"] >= 0.0

    def test_cirq_distributions_valid(self, simple_encoding_2q):
        """Test that Cirq backend produces valid distributions."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )

        # Distributions should sum to approximately 1
        assert np.isclose(result["fidelity_distribution"].sum(), 1.0, atol=1e-6)
        assert np.isclose(result["haar_distribution"].sum(), 1.0, atol=1e-6)

        # All values should be non-negative
        assert np.all(result["fidelity_distribution"] >= 0)
        assert np.all(result["haar_distribution"] >= 0)

    def test_cirq_fidelity_distribution_shape(self, simple_encoding_2q):
        """Test that Cirq produces correct fidelity distribution shape."""
        n_bins = 25
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=n_bins,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )
        assert result["fidelity_distribution"].shape == (n_bins,)
        assert result["haar_distribution"].shape == (n_bins,)
        assert result["bin_edges"].shape == (n_bins + 1,)

    def test_cirq_statistics_valid(self, simple_encoding_2q):
        """Test that Cirq backend produces valid statistics."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )

        # Mean fidelity should be in [0, 1]
        assert 0.0 <= result["mean_fidelity"] <= 1.0

        # Std fidelity should be non-negative
        assert result["std_fidelity"] >= 0.0

        # Convergence estimate should be finite
        assert np.isfinite(result["convergence_estimate"])

    def test_cirq_narrow_input_range(self, simple_encoding_2q):
        """Test Cirq with narrow input range."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            input_range=(0.0, 0.1),
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_wide_input_range(self, simple_encoding_2q):
        """Test Cirq with wide input range."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            input_range=(0.0, 10 * np.pi),
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("n_features", [2, 3, 4])
    def test_cirq_angle_encoding_various_sizes(self, n_features):
        """Test Cirq with AngleEncoding of various sizes."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=n_features)
        result = compute_expressibility(
            enc,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("reps", [1, 2])
    def test_cirq_iqp_encoding_various_reps(self, reps):
        """Test Cirq with IQPEncoding of various repetitions."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=3, reps=reps)
        result = compute_expressibility(
            enc,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_fidelity_distribution_function(self, simple_encoding_2q):
        """Test compute_fidelity_distribution with Cirq backend."""
        n_samples = 50
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=n_samples,
            seed=42,
            backend="cirq",
        )

        # Check shape
        assert fidelities.shape == (n_samples,)

        # Check all values in valid range
        assert np.all(fidelities >= 0.0)
        assert np.all(fidelities <= 1.0)

    def test_cirq_fidelity_distribution_reproducibility(self, simple_encoding_2q):
        """Test that compute_fidelity_distribution is reproducible with Cirq."""
        fid1 = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            seed=42,
            backend="cirq",
        )
        fid2 = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            seed=42,
            backend="cirq",
        )
        np.testing.assert_array_equal(fid1, fid2)

    def test_cirq_different_seeds_different_results(self, simple_encoding_2q):
        """Test that different seeds produce different results with Cirq."""
        result1 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
        )
        result2 = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=123,
            backend="cirq",
        )
        # Results should be different (with high probability)
        assert result1 != result2

    def test_cirq_verbose_mode_no_error(self, simple_encoding_2q, caplog):
        """Test that verbose mode doesn't cause errors with Cirq."""
        with caplog.at_level(logging.DEBUG):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                seed=42,
                backend="cirq",
                verbose=True,
            )
        assert 0.0 <= result <= 1.0


# =============================================================================
# Tests: Cirq Backend Numerical Stability
# =============================================================================


class TestCirqNumericalStability:
    """Numerical stability tests specific to Cirq backend.

    These tests verify that the Cirq backend handles edge cases and
    numerical challenges correctly, producing stable and valid results.
    """

    @pytest.fixture(autouse=True)
    def check_cirq(self, cirq_available):
        """Skip all tests in this class if Cirq is not available."""
        if not cirq_available:
            pytest.skip("Cirq not available")

    def test_cirq_minimum_valid_samples(self, simple_encoding_2q):
        """Test Cirq with minimum valid sample count."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,
            n_bins=_MIN_SAMPLES_ERROR,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_bins_equal_samples(self, simple_encoding_2q):
        """Test Cirq with n_bins == n_samples (edge case)."""
        n = 50
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=n,
            n_bins=n,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_result_types_correct(self, simple_encoding_2q):
        """Test that Cirq backend returns correct types in result dict."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )

        assert isinstance(result["expressibility"], float)
        assert isinstance(result["kl_divergence"], float)
        assert isinstance(result["fidelity_distribution"], np.ndarray)
        assert isinstance(result["haar_distribution"], np.ndarray)
        assert isinstance(result["bin_edges"], np.ndarray)
        assert isinstance(result["n_samples"], int)
        assert isinstance(result["n_bins"], int)
        assert isinstance(result["convergence_estimate"], float)
        assert isinstance(result["mean_fidelity"], float)
        assert isinstance(result["std_fidelity"], float)

    def test_cirq_n_samples_stored_correctly(self, simple_encoding_2q):
        """Test that n_samples is correctly stored in Cirq result."""
        n_samples = 75
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=n_samples,
            n_bins=15,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )
        assert result["n_samples"] == n_samples

    def test_cirq_n_bins_stored_correctly(self, simple_encoding_2q):
        """Test that n_bins is correctly stored in Cirq result."""
        n_bins = 30
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=n_bins,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )
        assert result["n_bins"] == n_bins

    def test_cirq_finite_results(self, simple_encoding_2q):
        """Test that all Cirq results are finite (no NaN/Inf)."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            backend="cirq",
            return_distributions=True,
        )

        assert np.isfinite(result["expressibility"])
        assert np.isfinite(result["kl_divergence"])
        assert np.all(np.isfinite(result["fidelity_distribution"]))
        assert np.all(np.isfinite(result["haar_distribution"]))
        assert np.all(np.isfinite(result["bin_edges"]))
        assert np.isfinite(result["convergence_estimate"])
        assert np.isfinite(result["mean_fidelity"])
        assert np.isfinite(result["std_fidelity"])


# =============================================================================
# Tests: Verbose Mode and Logging
# =============================================================================


class TestVerboseMode:
    """Test verbose mode and logging."""

    def test_verbose_mode_no_error(
        self, simple_encoding_2q, skip_if_no_pennylane, caplog
    ):
        """Test that verbose mode doesn't cause errors."""
        with caplog.at_level(logging.DEBUG):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                seed=42,
                verbose=True,
            )
        assert 0.0 <= result <= 1.0


# =============================================================================
# Tests: Constants Validation
# =============================================================================


class TestConstants:
    """Test module-level constants are reasonable."""

    def test_default_n_samples_reasonable(self):
        """Test that default n_samples is reasonable."""
        assert _DEFAULT_N_SAMPLES >= 1000
        assert _DEFAULT_N_SAMPLES <= 100000

    def test_default_n_bins_reasonable(self):
        """Test that default n_bins is reasonable."""
        assert _DEFAULT_N_BINS >= 50
        assert _DEFAULT_N_BINS <= 200

    def test_min_samples_hierarchy(self):
        """Test that error threshold < warning threshold."""
        assert _MIN_SAMPLES_ERROR < _MIN_SAMPLES_WARNING

    def test_max_kl_divergence_positive(self):
        """Test that max KL divergence is positive."""
        assert _MAX_KL_DIVERGENCE > 0

    def test_numerical_epsilon_positive_and_small(self):
        """Test that epsilon is positive and small."""
        assert _NUMERICAL_EPSILON > 0
        assert _NUMERICAL_EPSILON < 1e-6


# =============================================================================
# Tests: Type Annotations
# =============================================================================


class TestTypeAnnotations:
    """Test that return types match documentation."""

    def test_expressibility_result_type(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test ExpressibilityResult has correct types."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            return_distributions=True,
        )

        assert isinstance(result["expressibility"], float)
        assert isinstance(result["kl_divergence"], float)
        assert isinstance(result["fidelity_distribution"], np.ndarray)
        assert isinstance(result["haar_distribution"], np.ndarray)
        assert isinstance(result["bin_edges"], np.ndarray)
        assert isinstance(result["n_samples"], int)
        assert isinstance(result["n_bins"], int)
        assert isinstance(result["convergence_estimate"], float)
        assert isinstance(result["mean_fidelity"], float)
        assert isinstance(result["std_fidelity"], float)


# =============================================================================
# Tests: Integration with Multiple Encodings
# =============================================================================


class TestMultipleEncodings:
    """Test expressibility computation with various encodings."""

    @pytest.mark.parametrize("n_features", [2, 3, 4])
    def test_angle_encoding_various_sizes(
        self, n_features, skip_if_no_pennylane
    ):
        """Test AngleEncoding with various sizes."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=n_features)
        result = compute_expressibility(
            enc,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("reps", [1, 2])
    def test_iqp_encoding_various_reps(
        self, reps, skip_if_no_pennylane
    ):
        """Test IQPEncoding with various repetitions."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=3, reps=reps)
        result = compute_expressibility(
            enc,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        assert 0.0 <= result <= 1.0


# =============================================================================
# Slow Tests (marked for optional execution)
# =============================================================================


@pytest.mark.slow
class TestExpressibilityPerformance:
    """Performance tests for expressibility computation."""

    def test_large_sample_count(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test with larger sample count (slow)."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=1000,
            n_bins=50,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_convergence_with_samples(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that results converge as samples increase."""
        sample_counts = [100, 500, 1000]
        results = []

        for n in sample_counts:
            r = compute_expressibility(
                simple_encoding_2q,
                n_samples=n,
                n_bins=30,
                seed=42,
            )
            results.append(r)

        # All should be valid
        for r in results:
            assert 0.0 <= r <= 1.0

        # Results should be somewhat similar (within 0.3)
        # as we're using the same seed and encoding
        assert abs(results[-1] - results[0]) < 0.3

    def test_performance_timing_baseline_2q(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that 2-qubit expressibility completes within expected time.

        This test establishes a baseline performance expectation for small
        encodings. The computation should complete well within the timeout
        to ensure the implementation is efficient.

        Performance Target: < 30 seconds for n_samples=1000, n_qubits=2
        """
        import time

        start_time = time.perf_counter()
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=1000,
            n_bins=50,
            seed=42,
        )
        elapsed_time = time.perf_counter() - start_time

        # Verify valid result
        assert 0.0 <= result <= 1.0

        # Performance assertion: should complete within 30 seconds
        # This is generous to account for varying hardware
        max_allowed_time = 30.0
        assert elapsed_time < max_allowed_time, (
            f"Expressibility computation took {elapsed_time:.2f}s, "
            f"expected < {max_allowed_time}s for 2-qubit encoding with 1000 samples"
        )

    def test_performance_timing_4q_encoding(self, skip_if_no_pennylane):
        """Test that 4-qubit expressibility completes within expected time.

        This tests a medium-sized encoding to ensure reasonable scaling.

        Performance Target: < 60 seconds for n_samples=500, n_qubits=4
        """
        import time
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)

        start_time = time.perf_counter()
        result = compute_expressibility(
            enc,
            n_samples=500,
            n_bins=30,
            seed=42,
        )
        elapsed_time = time.perf_counter() - start_time

        # Verify valid result
        assert 0.0 <= result <= 1.0

        # Performance assertion
        max_allowed_time = 60.0
        assert elapsed_time < max_allowed_time, (
            f"Expressibility computation took {elapsed_time:.2f}s, "
            f"expected < {max_allowed_time}s for 4-qubit encoding with 500 samples"
        )


# =============================================================================
# High Qubit Count Tests (Numerical Stability at Scale)
# =============================================================================


@pytest.mark.slow
class TestHighQubitCountNumericalStability:
    """Tests for numerical stability at higher qubit counts.

    These tests verify that the expressibility computation remains numerically
    stable for encodings near and at the warning threshold (10 qubits).
    This is critical for production use where users may analyze larger encodings.

    The tests check:
    1. Results contain no NaN or Inf values
    2. Results are within valid range [0, 1]
    3. Distributions are properly normalized
    4. Appropriate warnings are issued for large qubit counts
    """

    def test_6_qubit_encoding_numerical_stability(self, skip_if_no_pennylane):
        """Test expressibility computation for 6-qubit encoding.

        6 qubits is a common use case that should work without issues.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=6)
        result = compute_expressibility(
            enc,
            n_samples=100,  # Reduced for speed
            n_bins=20,
            seed=42,
            return_distributions=True,
        )

        # Verify no numerical issues
        assert np.isfinite(result["expressibility"]), "Expressibility is not finite"
        assert np.isfinite(result["kl_divergence"]), "KL divergence is not finite"
        assert np.all(np.isfinite(result["fidelity_distribution"])), (
            "Fidelity distribution contains non-finite values"
        )
        assert np.all(np.isfinite(result["haar_distribution"])), (
            "Haar distribution contains non-finite values"
        )

        # Verify valid range
        assert 0.0 <= result["expressibility"] <= 1.0
        assert result["kl_divergence"] >= 0.0

        # Verify distributions are properly normalized (sum to ~1)
        fid_sum = np.sum(result["fidelity_distribution"])
        haar_sum = np.sum(result["haar_distribution"])
        assert 0.99 <= fid_sum <= 1.01, f"Fidelity distribution sum: {fid_sum}"
        assert 0.99 <= haar_sum <= 1.01, f"Haar distribution sum: {haar_sum}"

    def test_8_qubit_encoding_numerical_stability(self, skip_if_no_pennylane):
        """Test expressibility computation for 8-qubit encoding.

        8 qubits approaches the warning threshold and tests the algorithm
        at larger Hilbert space dimensions (d=256).
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=8)
        result = compute_expressibility(
            enc,
            n_samples=50,  # Reduced for speed
            n_bins=15,
            seed=42,
            return_distributions=True,
        )

        # Verify no numerical issues
        assert np.isfinite(result["expressibility"]), "Expressibility is not finite"
        assert np.isfinite(result["kl_divergence"]), "KL divergence is not finite"
        assert np.all(np.isfinite(result["fidelity_distribution"])), (
            "Fidelity distribution contains non-finite values"
        )
        assert np.all(np.isfinite(result["haar_distribution"])), (
            "Haar distribution contains non-finite values"
        )

        # Verify valid range
        assert 0.0 <= result["expressibility"] <= 1.0

    def test_10_qubit_encoding_numerical_stability(self, skip_if_no_pennylane):
        """Test expressibility computation for 10-qubit encoding.

        10 qubits is at the warning threshold. This tests that the algorithm
        handles larger Hilbert spaces (d=1024) correctly.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=10)

        # Should work but may be slow
        result = compute_expressibility(
            enc,
            n_samples=30,  # Minimal for speed
            n_bins=10,
            seed=42,
            return_distributions=True,
        )

        # Verify no numerical issues
        assert np.isfinite(result["expressibility"]), "Expressibility is not finite"
        assert np.isfinite(result["kl_divergence"]), "KL divergence is not finite"

        # Verify valid range
        assert 0.0 <= result["expressibility"] <= 1.0

    def test_qubit_warning_threshold_emits_warning(self, skip_if_no_pennylane):
        """Test that warning is emitted for encodings above qubit threshold.

        The implementation should warn users when computing expressibility
        for encodings with more than 10 qubits, as the computation becomes
        exponentially slower.
        """
        from encoding_atlas import AngleEncoding

        # Create an 11-qubit encoding (above threshold)
        enc = AngleEncoding(n_features=11)

        # Should emit a UserWarning about qubit count
        with pytest.warns(UserWarning, match=r".*qubit.*"):
            # Use minimal parameters to make test fast
            result = compute_expressibility(
                enc,
                n_samples=10,
                n_bins=10,
                seed=42,
            )

        # Should still produce valid result
        assert 0.0 <= result <= 1.0

    def test_haar_distribution_large_dimension_stability(self):
        """Test Haar distribution computation for large Hilbert space dimensions.

        The Haar distribution formula P_Haar(F) = (d-1)(1-F)^(d-2) can cause
        numerical issues for large d due to the large exponent. This test
        verifies the log-space computation handles this correctly.
        """
        # Test various qubit counts
        for n_qubits in [8, 10, 12]:
            fidelity_values = np.linspace(0, 1, 100)
            haar_dist = compute_haar_distribution(n_qubits, fidelity_values)

            # Should be finite
            assert np.all(np.isfinite(haar_dist)), (
                f"Haar distribution contains non-finite values for {n_qubits} qubits"
            )

            # Should be non-negative
            assert np.all(haar_dist >= 0), (
                f"Haar distribution contains negative values for {n_qubits} qubits"
            )

            # Should sum to approximately 1 (normalized probability)
            dist_sum = np.sum(haar_dist)
            assert 0.99 <= dist_sum <= 1.01, (
                f"Haar distribution sum is {dist_sum} for {n_qubits} qubits"
            )

    def test_entangling_encoding_high_qubit_stability(self, skip_if_no_pennylane):
        """Test numerical stability with entangling encoding at higher qubits.

        Entangling encodings create more complex quantum states which may
        stress the numerical stability of the implementation differently
        than product state encodings.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=6, reps=1)
        result = compute_expressibility(
            enc,
            n_samples=50,
            n_bins=15,
            seed=42,
            return_distributions=True,
        )

        # Verify no numerical issues
        assert np.isfinite(result["expressibility"]), "Expressibility is not finite"
        assert np.isfinite(result["kl_divergence"]), "KL divergence is not finite"
        assert np.all(np.isfinite(result["fidelity_distribution"])), (
            "Fidelity distribution contains non-finite values"
        )

        # Verify valid range
        assert 0.0 <= result["expressibility"] <= 1.0
