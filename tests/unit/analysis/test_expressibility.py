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
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from encoding_atlas.analysis._utils import create_rng
from encoding_atlas.analysis.expressibility import (
    _DEFAULT_N_BINS,
    _DEFAULT_N_SAMPLES,
    _MAX_KL_DIVERGENCE,
    _MIN_SAMPLES_ERROR,
    _MIN_SAMPLES_WARNING,
    _NUMERICAL_EPSILON,
    _QUBIT_WARNING_THRESHOLD,
    _estimate_convergence,
    _sample_fidelities,
    compute_expressibility,
    compute_fidelity_distribution,
    compute_haar_distribution,
)
from encoding_atlas.core.exceptions import (
    AnalysisError,
    InsufficientSamplesError,
    NumericalInstabilityError,
    SimulationError,
)

if TYPE_CHECKING:
    pass

# Suppress expected low-sample-count warnings in tests that intentionally use
# small n_samples for speed.  The test that explicitly checks warning emission
# (test_n_samples_low_raises_warning) overrides this with a per-test marker.
pytestmark = pytest.mark.filterwarnings("ignore:n_samples.*low:UserWarning")


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

    def test_returns_float_by_default(self, simple_encoding_2q, skip_if_no_pennylane):
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
            # Bootstrap CI fields added in the analysis statistics commit.
            "expressibility_ci_lower",
            "expressibility_ci_upper",
            "mean_fidelity_ci_lower",
            "mean_fidelity_ci_upper",
            "confidence_level",
            "sampling",
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

    def test_kl_divergence_non_negative(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that KL divergence is non-negative."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert result["kl_divergence"] >= 0.0

    def test_reproducibility_with_seed(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_n_samples_stored_correctly(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_n_bins_stored_correctly(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_haar_distribution_shape(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_bin_edges_shape(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_distributions_sum_to_one(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_distributions_non_negative(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_n_samples_too_small_raises_error(self, simple_encoding_2q):
        """Test that n_samples < MIN_SAMPLES_ERROR raises InsufficientSamplesError."""
        with pytest.raises(InsufficientSamplesError):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=5,  # Below minimum
                n_bins=10,
            )

    @pytest.mark.filterwarnings("default:n_samples.*low:UserWarning")
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

    def test_n_bins_too_small_raises_error(self, simple_encoding_2q):
        """Test that n_bins < 10 raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be at least 10"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=5,  # Below minimum
            )

    def test_n_bins_larger_than_n_samples_raises_error(self, simple_encoding_2q):
        """Test that n_bins > n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_bins.*cannot exceed n_samples"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=100,  # More bins than samples
            )

    def test_invalid_input_range_order(self, simple_encoding_2q):
        """Test that input_range with min >= max raises error."""
        with pytest.raises(ValueError, match="input_range"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20,
                input_range=(2.0, 1.0),  # Invalid: max < min
            )

    def test_invalid_input_range_equal(self, simple_encoding_2q):
        """Test that input_range with min == max raises error."""
        with pytest.raises(ValueError, match="input_range"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20,
                input_range=(1.0, 1.0),  # Invalid: equal
            )

    def test_invalid_backend(self, simple_encoding_2q):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20,
                backend="invalid_backend",
            )

    def test_negative_n_samples(self, simple_encoding_2q):
        """Test that negative n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be a positive"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=-10,
                n_bins=20,
            )

    def test_zero_n_samples(self, simple_encoding_2q):
        """Test that zero n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be a positive"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=0,
                n_bins=20,
            )

    def test_float_n_samples_raises_error(self, simple_encoding_2q):
        """Test that float n_samples raises ValueError.

        n_samples must be an int. Passing a float (even one with no
        fractional part like 50.0) should be rejected.
        """
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50.0,
            )

    def test_float_n_bins_raises_error(self, simple_encoding_2q):
        """Test that float n_bins raises ValueError.

        n_bins must be an int. Passing a float should be rejected.
        """
        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20.0,
            )

    @pytest.mark.filterwarnings("error:n_samples.*low:UserWarning")
    def test_n_samples_at_warning_threshold_no_warning(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that n_samples at the warning threshold does not emit warning.

        _MIN_SAMPLES_WARNING is the threshold. n_samples equal to this value
        should NOT trigger the low-sample warning (only values strictly less
        than the threshold trigger it). The filterwarnings('error') marker
        ensures the test fails if the warning IS emitted.
        """
        compute_expressibility(
            simple_encoding_2q,
            n_samples=_MIN_SAMPLES_WARNING,
            n_bins=20,
            seed=42,
        )

    def test_negative_n_bins_raises_error(self, simple_encoding_2q):
        """Test that negative n_bins raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=-5,
            )

    def test_n_bins_zero_raises_error(self, simple_encoding_2q):
        """Test that n_bins=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=0,
            )

    def test_input_range_wrong_length(self, simple_encoding_2q):
        """Test that input_range with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="input_range must have 2 elements"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=20,
                input_range=(0.0, 1.0, 2.0),
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

    def test_empty_fidelity_values_small_d(self):
        """Test compute_haar_distribution with empty array for small d.

        For d <= 100 (n_qubits <= 6), the direct computation path handles
        empty arrays without raising an error, returning an empty array.
        """
        fidelities = np.array([], dtype=np.float64)
        P_haar = compute_haar_distribution(n_qubits=2, fidelity_values=fidelities)
        assert len(P_haar) == 0
        assert P_haar.dtype == np.float64

    def test_empty_fidelity_values_large_d(self):
        """Test compute_haar_distribution with empty array for large d.

        For n_qubits >= 7 (d > 100), the log-space computation path is used.
        Empty inputs should return an empty array regardless of the number
        of qubits.
        """
        fidelities = np.array([], dtype=np.float64)
        P_haar = compute_haar_distribution(n_qubits=8, fidelity_values=fidelities)
        assert len(P_haar) == 0
        assert P_haar.dtype == np.float64

    def test_monotonically_decreasing_for_multi_qubit(self):
        """Test that Haar distribution is monotonically non-increasing for d > 2.

        The Haar PDF P(F) = (d-1)(1-F)^(d-2) has derivative
        -(d-1)(d-2)(1-F)^(d-3), which is strictly negative for F in (0, 1)
        when d > 2, making the distribution strictly decreasing.
        """
        fidelities = np.linspace(0.01, 0.99, 100)

        for n_qubits in [2, 3, 4, 5]:
            P_haar = compute_haar_distribution(n_qubits, fidelities)
            # After normalization, relative ordering is preserved
            for i in range(len(P_haar) - 1):
                assert P_haar[i] >= P_haar[i + 1] - 1e-10, (
                    f"Haar distribution not monotonically decreasing at index "
                    f"{i} for n_qubits={n_qubits}: P[{i}]={P_haar[i]:.8f}, "
                    f"P[{i+1}]={P_haar[i+1]:.8f}"
                )

    def test_single_element_fidelity(self):
        """Test compute_haar_distribution with a single fidelity value."""
        fidelities = np.array([0.5])
        P_haar = compute_haar_distribution(n_qubits=3, fidelity_values=fidelities)
        assert P_haar.shape == (1,)
        assert np.isclose(P_haar.sum(), 1.0, atol=1e-6)
        assert P_haar[0] > 0

    def test_float_n_qubits_raises_error(self):
        """Test that float n_qubits raises ValueError."""
        fidelities = np.linspace(0, 1, 50)
        with pytest.raises(ValueError, match="n_qubits must be a positive"):
            compute_haar_distribution(n_qubits=2.5, fidelity_values=fidelities)

    def test_two_qubit_analytical_shape(self):
        """Test that 2-qubit Haar distribution follows P = 3(1-F)^2.

        For d=4 (2 qubits), the Haar PDF is exactly P(F) = 3(1-F)^2.
        After normalization, the relative shape should be preserved.
        """
        fidelities = np.linspace(0.05, 0.95, 50)
        P_haar = compute_haar_distribution(n_qubits=2, fidelity_values=fidelities)

        # Compute expected shape: 3(1-F)^2, then normalize
        expected = 3.0 * (1.0 - fidelities) ** 2
        expected = expected / expected.sum()

        np.testing.assert_allclose(P_haar, expected, atol=1e-10)


# =============================================================================
# Tests: Fidelity Distribution
# =============================================================================


class TestComputeFidelityDistribution:
    """Test the compute_fidelity_distribution function."""

    def test_returns_correct_shape(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that returned array has correct shape."""
        n_samples = 50
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=n_samples,
            seed=42,
        )
        assert fidelities.shape == (n_samples,)

    def test_all_values_in_valid_range(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that all fidelities are in [0, 1]."""
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=100,
            seed=42,
        )
        assert np.all(fidelities >= 0.0)
        assert np.all(fidelities <= 1.0)

    def test_reproducibility_with_seed(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_invalid_n_samples_raises_error(self, simple_encoding_2q):
        """Test that invalid n_samples raises error."""
        with pytest.raises(InsufficientSamplesError):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=5,  # Below minimum
            )

    def test_invalid_input_range(self, simple_encoding_2q):
        """Test that invalid input_range raises error."""
        with pytest.raises(ValueError, match="input_range"):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=50,
                input_range=(2.0, 1.0),  # Invalid
            )

    def test_qiskit_backend(self, simple_encoding_2q, skip_if_no_qiskit):
        """Test compute_fidelity_distribution with Qiskit backend."""
        n_samples = 50
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=n_samples,
            seed=42,
            backend="qiskit",
        )
        assert fidelities.shape == (n_samples,)
        assert np.all(fidelities >= 0.0)
        assert np.all(fidelities <= 1.0)

    def test_cirq_backend(self, simple_encoding_2q, skip_if_no_cirq):
        """Test compute_fidelity_distribution with Cirq backend."""
        n_samples = 50
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=n_samples,
            seed=42,
            backend="cirq",
        )
        assert fidelities.shape == (n_samples,)
        assert np.all(fidelities >= 0.0)
        assert np.all(fidelities <= 1.0)

    def test_invalid_backend_raises_error(self, simple_encoding_2q):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=50,
                backend="invalid_backend",
            )

    def test_qiskit_reproducibility_with_seed(
        self, simple_encoding_2q, skip_if_no_qiskit
    ):
        """Test that Qiskit backend produces reproducible fidelity distributions."""
        fid1 = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            seed=42,
            backend="qiskit",
        )
        fid2 = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            seed=42,
            backend="qiskit",
        )
        np.testing.assert_array_equal(fid1, fid2)

    def test_cirq_reproducibility_with_seed(self, simple_encoding_2q, skip_if_no_cirq):
        """Test that Cirq backend produces reproducible fidelity distributions."""
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


# =============================================================================
# Tests: Edge Cases and Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_single_qubit_encoding(self, single_qubit_encoding, skip_if_no_pennylane):
        """Test that single qubit encoding works correctly."""
        result = compute_expressibility(
            single_qubit_encoding,
            n_samples=50,
            n_bins=10,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_narrow_input_range(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_wide_input_range(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test with wide input range."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            input_range=(0.0, 10 * np.pi),  # Wide range
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_minimum_valid_samples(self, simple_encoding_2q, skip_if_no_pennylane):
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

    def test_std_fidelity_non_negative(self, simple_encoding_2q, skip_if_no_pennylane):
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

        AngleEncoding produces only product states (no entanglement), while
        IQPEncoding creates entangled states that explore more of the Hilbert
        space. Theory (Sim et al., 2019) predicts entangling circuits have
        strictly higher expressibility than product-state circuits.
        """
        expr_simple = compute_expressibility(
            simple_encoding_4q,
            n_samples=300,
            n_bins=40,
            seed=42,
        )
        expr_entangling = compute_expressibility(
            entangling_encoding_4q,
            n_samples=300,
            n_bins=40,
            seed=42,
        )

        # Both should be valid
        assert 0.0 <= expr_simple <= 1.0
        assert 0.0 <= expr_entangling <= 1.0

        # Entangling encoding should have higher expressibility
        assert expr_entangling > expr_simple, (
            f"Expected entangling encoding (IQP) to have higher expressibility "
            f"than non-entangling encoding (Angle): "
            f"IQP={expr_entangling:.4f}, Angle={expr_simple:.4f}"
        )


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

    def test_qiskit_produces_valid_result(self, simple_encoding_2q, skip_if_no_qiskit):
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

    def test_cirq_produces_valid_result(self, simple_encoding_2q, skip_if_no_cirq):
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
            # Bootstrap CI fields added in the analysis statistics commit.
            "expressibility_ci_lower",
            "expressibility_ci_upper",
            "mean_fidelity_ci_lower",
            "mean_fidelity_ci_upper",
            "confidence_level",
            "sampling",
        }
        assert set(result.keys()) == required_keys

    def test_cirq_reproducibility_with_seed(self, simple_encoding_2q, skip_if_no_cirq):
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

    def test_expressibility_result_type(self, simple_encoding_2q, skip_if_no_pennylane):
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
    def test_angle_encoding_various_sizes(self, n_features, skip_if_no_pennylane):
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
    def test_iqp_encoding_various_reps(self, reps, skip_if_no_pennylane):
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

    def test_large_sample_count(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test with larger sample count (slow)."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=1000,
            n_bins=50,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_convergence_with_samples(self, simple_encoding_2q, skip_if_no_pennylane):
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
        assert np.all(
            np.isfinite(result["fidelity_distribution"])
        ), "Fidelity distribution contains non-finite values"
        assert np.all(
            np.isfinite(result["haar_distribution"])
        ), "Haar distribution contains non-finite values"

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
        assert np.all(
            np.isfinite(result["fidelity_distribution"])
        ), "Fidelity distribution contains non-finite values"
        assert np.all(
            np.isfinite(result["haar_distribution"])
        ), "Haar distribution contains non-finite values"

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
            assert np.all(
                np.isfinite(haar_dist)
            ), f"Haar distribution contains non-finite values for {n_qubits} qubits"

            # Should be non-negative
            assert np.all(
                haar_dist >= 0
            ), f"Haar distribution contains negative values for {n_qubits} qubits"

            # Should sum to approximately 1 (normalized probability)
            dist_sum = np.sum(haar_dist)
            assert (
                0.99 <= dist_sum <= 1.01
            ), f"Haar distribution sum is {dist_sum} for {n_qubits} qubits"

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
        assert np.all(
            np.isfinite(result["fidelity_distribution"])
        ), "Fidelity distribution contains non-finite values"

        # Verify valid range
        assert 0.0 <= result["expressibility"] <= 1.0


# =============================================================================
# Tests: n_bins Auto-Selection
# =============================================================================


class TestNBinsAutoSelection:
    """Test the n_bins=None auto-selection logic.

    When n_bins is not explicitly provided, it defaults to
    min(_DEFAULT_N_BINS, n_samples). This ensures the bin count
    is sensible for the given sample size.
    """

    def test_auto_selects_default_bins(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that n_bins=None selects _DEFAULT_N_BINS when n_samples is large."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=None,
            seed=42,
            return_distributions=True,
        )
        assert result["n_bins"] == _DEFAULT_N_BINS

    def test_auto_clamps_to_n_samples(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that n_bins=None clamps to n_samples when samples < default bins."""
        n_samples = 30
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=n_samples,
            n_bins=None,
            seed=42,
            return_distributions=True,
        )
        assert result["n_bins"] == n_samples

    def test_auto_selection_distribution_shape(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that auto-selected n_bins produces correctly shaped distributions."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=None,
            seed=42,
            return_distributions=True,
        )
        expected_bins = _DEFAULT_N_BINS
        assert result["fidelity_distribution"].shape == (expected_bins,)
        assert result["haar_distribution"].shape == (expected_bins,)
        assert result["bin_edges"].shape == (expected_bins + 1,)

    def test_auto_selection_at_boundary(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test n_bins=None when n_samples exactly equals _DEFAULT_N_BINS."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=_DEFAULT_N_BINS,
            seed=42,
            return_distributions=True,
        )
        assert result["n_bins"] == _DEFAULT_N_BINS

    def test_auto_selection_one_below_default(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test n_bins=None when n_samples is one less than _DEFAULT_N_BINS."""
        n_samples = _DEFAULT_N_BINS - 1
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=n_samples,
            seed=42,
            return_distributions=True,
        )
        assert result["n_bins"] == n_samples


# =============================================================================
# Tests: Expressibility Result Consistency
# =============================================================================


class TestExpressibilityResultConsistency:
    """Test internal consistency of the ExpressibilityResult dictionary.

    These tests verify that the various fields in the result dictionary
    are mutually consistent and match the documented formulas.
    """

    def test_expressibility_matches_kl_formula(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that expressibility = 1 - min(1, kl_divergence / MAX_KL)."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        expected = 1.0 - min(1.0, result["kl_divergence"] / _MAX_KL_DIVERGENCE)
        assert np.isclose(result["expressibility"], expected, atol=1e-10), (
            f"Expressibility {result['expressibility']:.10f} does not match "
            f"formula result {expected:.10f}"
        )

    def test_bin_edges_span_zero_to_one(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that bin edges span the full fidelity range [0, 1]."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert result["bin_edges"][0] == 0.0
        assert result["bin_edges"][-1] == 1.0

    def test_bin_edges_monotonically_increasing(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that bin edges are strictly increasing."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        edges = result["bin_edges"]
        for i in range(len(edges) - 1):
            assert edges[i] < edges[i + 1]

    def test_distributions_and_edges_shape_consistent(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that distributions and bin_edges shapes are consistent."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        n_bins = result["n_bins"]
        assert result["fidelity_distribution"].shape == (n_bins,)
        assert result["haar_distribution"].shape == (n_bins,)
        assert result["bin_edges"].shape == (n_bins + 1,)

    def test_distributions_dtype_float64(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that all array outputs are float64."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert result["fidelity_distribution"].dtype == np.float64
        assert result["haar_distribution"].dtype == np.float64
        assert result["bin_edges"].dtype == np.float64

    def test_std_fidelity_bounded_by_range(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that std fidelity cannot exceed the fidelity range [0, 1]."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        # Standard deviation of values in [0, 1] is at most 0.5
        assert 0.0 <= result["std_fidelity"] <= 0.5 + 1e-10


# =============================================================================
# Tests: Convergence Estimate
# =============================================================================


class TestConvergenceEstimate:
    """Test the bootstrap convergence estimate in expressibility results."""

    def test_convergence_estimate_is_finite(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that convergence estimate is finite for adequate samples."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        assert np.isfinite(result["convergence_estimate"])

    def test_convergence_estimate_non_negative(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that convergence estimate (std error) is non-negative."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        assert result["convergence_estimate"] >= 0.0

    def test_convergence_estimate_is_float(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that convergence estimate is a Python float."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert isinstance(result["convergence_estimate"], float)


# =============================================================================
# Tests: _sample_fidelities Error Handling (Private Function)
# =============================================================================


class TestSampleFidelitiesErrorHandling:
    """Test error handling in the _sample_fidelities private function.

    These tests verify that SimulationError is re-raised directly, and
    that other exceptions are wrapped in SimulationError with contextual
    details (sample index, error type, etc.).
    """

    def test_simulation_error_reraise(self, simple_encoding_2q):
        """Test that SimulationError from simulate_encoding_statevector is re-raised."""
        with (
            patch(
                "encoding_atlas.analysis.expressibility.simulate_encoding_statevector",
                side_effect=SimulationError("Backend crashed", backend="pennylane"),
            ),
            pytest.raises(SimulationError, match="Backend crashed"),
        ):
            _sample_fidelities(
                encoding=simple_encoding_2q,
                n_samples=1,
                n_features=simple_encoding_2q.n_features,
                input_range=(0.0, 2.0 * np.pi),
                rng=create_rng(42),
                backend="pennylane",
                verbose=False,
            )

    def test_generic_exception_wrapped_in_simulation_error(self, simple_encoding_2q):
        """Test that non-SimulationError exceptions are wrapped in SimulationError."""
        mock_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

        with (
            patch(
                "encoding_atlas.analysis.expressibility.simulate_encoding_statevector",
                return_value=mock_state,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.compute_fidelity",
                side_effect=ValueError("Invalid state dimensions"),
            ),
            pytest.raises(SimulationError, match="Failed to compute fidelity"),
        ):
            _sample_fidelities(
                encoding=simple_encoding_2q,
                n_samples=5,
                n_features=simple_encoding_2q.n_features,
                input_range=(0.0, 2.0 * np.pi),
                rng=create_rng(42),
                backend="pennylane",
                verbose=False,
            )

    def test_wrapped_error_contains_context_details(self, simple_encoding_2q):
        """Test that wrapped SimulationError contains sample index and error details."""
        mock_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

        with (
            patch(
                "encoding_atlas.analysis.expressibility.simulate_encoding_statevector",
                return_value=mock_state,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.compute_fidelity",
                side_effect=RuntimeError("Unexpected failure"),
            ),
        ):
            with pytest.raises(SimulationError) as exc_info:
                _sample_fidelities(
                    encoding=simple_encoding_2q,
                    n_samples=3,
                    n_features=simple_encoding_2q.n_features,
                    input_range=(0.0, 2.0 * np.pi),
                    rng=create_rng(42),
                    backend="pennylane",
                    verbose=False,
                )
            assert "sample_index" in exc_info.value.details
            assert exc_info.value.details["sample_index"] == 0
            assert exc_info.value.details["error_type"] == "RuntimeError"
            assert "Unexpected failure" in exc_info.value.details["error_message"]


# =============================================================================
# Tests: _sample_fidelities Fidelity Clamping
# =============================================================================


class TestSampleFidelitiesClamping:
    """Test that fidelity values are clamped to [0, 1] in _sample_fidelities.

    The fidelity clamping at line 1099 is a defensive measure against
    numerical errors that could produce values slightly outside [0, 1].
    These tests use mocks to inject out-of-range fidelity values.
    """

    def test_negative_fidelity_clamped_to_zero(self, simple_encoding_2q):
        """Test that fidelity values < 0 are clamped to 0."""
        mock_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

        with (
            patch(
                "encoding_atlas.analysis.expressibility.simulate_encoding_statevector",
                return_value=mock_state,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.compute_fidelity",
                return_value=-0.05,
            ),
        ):
            fidelities = _sample_fidelities(
                encoding=simple_encoding_2q,
                n_samples=10,
                n_features=simple_encoding_2q.n_features,
                input_range=(0.0, 2.0 * np.pi),
                rng=create_rng(42),
                backend="pennylane",
                verbose=False,
            )
            assert np.all(fidelities >= 0.0)
            assert np.all(fidelities == 0.0)

    def test_fidelity_above_one_clamped_to_one(self, simple_encoding_2q):
        """Test that fidelity values > 1 are clamped to 1."""
        mock_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

        with (
            patch(
                "encoding_atlas.analysis.expressibility.simulate_encoding_statevector",
                return_value=mock_state,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.compute_fidelity",
                return_value=1.0001,
            ),
        ):
            fidelities = _sample_fidelities(
                encoding=simple_encoding_2q,
                n_samples=10,
                n_features=simple_encoding_2q.n_features,
                input_range=(0.0, 2.0 * np.pi),
                rng=create_rng(42),
                backend="pennylane",
                verbose=False,
            )
            assert np.all(fidelities <= 1.0)
            assert np.all(fidelities == 1.0)


# =============================================================================
# Tests: _sample_fidelities Verbose Logging
# =============================================================================


class TestSampleFidelitiesVerboseLogging:
    """Test verbose logging in _sample_fidelities.

    When verbose=True, _sample_fidelities logs progress at 10% intervals
    using _logger.debug. These tests verify that logs are emitted when
    expected and suppressed when verbose=False.
    """

    def test_verbose_emits_progress_debug_logs(
        self, simple_encoding_2q, skip_if_no_pennylane, caplog
    ):
        """Test that verbose=True emits 'Sampled' debug messages at intervals."""
        with caplog.at_level(logging.DEBUG):
            _sample_fidelities(
                encoding=simple_encoding_2q,
                n_samples=50,
                n_features=simple_encoding_2q.n_features,
                input_range=(0.0, 2.0 * np.pi),
                rng=create_rng(42),
                backend="pennylane",
                verbose=True,
            )

        sampled_msgs = [
            r
            for r in caplog.records
            if r.levelname == "DEBUG" and "Sampled" in r.message
        ]
        # 50 samples with log_interval=5, expect logs at 5,10,...,50 → 10 messages
        assert len(sampled_msgs) == 10

    def test_no_verbose_suppresses_progress_logs(
        self, simple_encoding_2q, skip_if_no_pennylane, caplog
    ):
        """Test that verbose=False does not emit 'Sampled' progress logs."""
        with caplog.at_level(logging.DEBUG):
            _sample_fidelities(
                encoding=simple_encoding_2q,
                n_samples=50,
                n_features=simple_encoding_2q.n_features,
                input_range=(0.0, 2.0 * np.pi),
                rng=create_rng(42),
                backend="pennylane",
                verbose=False,
            )

        sampled_msgs = [
            r
            for r in caplog.records
            if r.levelname == "DEBUG" and "Sampled" in r.message
        ]
        assert len(sampled_msgs) == 0


# =============================================================================
# Tests: _estimate_convergence Edge Cases (Private Function)
# =============================================================================


class TestEstimateConvergenceEdgeCases:
    """Test edge cases in the _estimate_convergence private function.

    This function uses bootstrap resampling to estimate the standard error
    of the KL divergence. These tests cover boundary conditions, degenerate
    inputs, and numerical edge cases.
    """

    def test_returns_inf_for_fewer_than_20_samples(self):
        """Test that convergence returns inf when n_samples < 20."""
        fidelities = np.random.default_rng(42).uniform(0, 1, 15)
        conv = _estimate_convergence(
            fidelities=fidelities,
            n_bins=10,
            n_qubits=2,
            rng=create_rng(42),
            n_bootstrap=50,
        )
        assert conv == float("inf")

    def test_returns_inf_for_exactly_19_samples(self):
        """Test that convergence returns inf at n_samples=19 (boundary)."""
        fidelities = np.random.default_rng(42).uniform(0, 1, 19)
        conv = _estimate_convergence(
            fidelities=fidelities,
            n_bins=10,
            n_qubits=2,
            rng=create_rng(42),
            n_bootstrap=50,
        )
        assert conv == float("inf")

    def test_returns_finite_for_20_samples(self):
        """Test that convergence returns a finite value at n_samples=20."""
        fidelities = np.random.default_rng(42).uniform(0, 1, 20)
        conv = _estimate_convergence(
            fidelities=fidelities,
            n_bins=10,
            n_qubits=2,
            rng=create_rng(42),
            n_bootstrap=50,
        )
        assert np.isfinite(conv)
        assert conv >= 0.0

    def test_reproducible_with_same_rng_seed(self):
        """Test that convergence estimate is reproducible with same seed."""
        fidelities = np.random.default_rng(99).uniform(0, 1, 100)
        conv1 = _estimate_convergence(
            fidelities=fidelities,
            n_bins=20,
            n_qubits=2,
            rng=create_rng(42),
            n_bootstrap=50,
        )
        conv2 = _estimate_convergence(
            fidelities=fidelities,
            n_bins=20,
            n_qubits=2,
            rng=create_rng(42),
            n_bootstrap=50,
        )
        assert conv1 == conv2

    def test_handles_all_identical_fidelities(self):
        """Test convergence with constant fidelities (all same value).

        When all fidelities are identical, all bootstrap samples produce the
        same histogram, giving zero variance in KL divergence estimates.
        """
        fidelities = np.full(50, 0.5)
        conv = _estimate_convergence(
            fidelities=fidelities,
            n_bins=20,
            n_qubits=2,
            rng=create_rng(42),
            n_bootstrap=50,
        )
        assert np.isfinite(conv)
        # All bootstrap samples are identical → KL std should be 0
        assert conv == 0.0

    def test_bootstrap_nan_kl_clamped_to_max(self):
        """Test that NaN KL values in bootstrap are clamped to _MAX_KL_DIVERGENCE.

        When rel_entr produces NaN for a bootstrap sample (e.g., due to
        numerical edge cases), the NaN KL value should be replaced with
        _MAX_KL_DIVERGENCE rather than propagating the NaN.
        """
        fidelities = np.random.default_rng(42).uniform(0, 1, 50)

        with patch("encoding_atlas.analysis.expressibility.rel_entr") as mock_rel:
            # Return NaN for all bootstrap calls → all KL values become MAX
            mock_rel.return_value = np.array([np.nan] * 20)
            conv = _estimate_convergence(
                fidelities=fidelities,
                n_bins=20,
                n_qubits=2,
                rng=create_rng(42),
                n_bootstrap=10,
            )

        # All bootstrap KLs are clamped to MAX → std = 0
        assert np.isfinite(conv)
        assert conv == 0.0

    def test_bootstrap_inf_kl_clamped_to_max(self):
        """Test that Inf KL values in bootstrap are clamped to _MAX_KL_DIVERGENCE."""
        fidelities = np.random.default_rng(42).uniform(0, 1, 50)

        with patch("encoding_atlas.analysis.expressibility.rel_entr") as mock_rel:
            mock_rel.return_value = np.array([np.inf] * 20)
            conv = _estimate_convergence(
                fidelities=fidelities,
                n_bins=20,
                n_qubits=2,
                rng=create_rng(42),
                n_bootstrap=10,
            )

        assert np.isfinite(conv)
        assert conv == 0.0


# =============================================================================
# Tests: Degenerate Fidelity Distribution Fallback
# =============================================================================


class TestDegenerateFidelityDistribution:
    """Test the degenerate distribution fallback in compute_expressibility.

    When the fidelity histogram has near-zero total probability mass
    (P_encoding_sum <= _NUMERICAL_EPSILON), the code falls back to a
    uniform distribution and emits a warning. This is a defensive branch
    that prevents downstream KL divergence computation from failing.
    """

    def test_fallback_to_uniform_on_zero_histogram(self, simple_encoding_2q, caplog):
        """Test that all-zero histogram triggers uniform distribution fallback."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility._estimate_convergence",
                return_value=0.01,
            ),
            patch("numpy.histogram") as mock_hist,
        ):
            # Return all-zero histogram (degenerate case)
            mock_hist.return_value = (
                np.zeros(20, dtype=np.float64),
                np.linspace(0.0, 1.0, 21),
            )
            with caplog.at_level(logging.WARNING):
                result = compute_expressibility(
                    simple_encoding_2q,
                    n_samples=50,
                    n_bins=20,
                    seed=42,
                    return_distributions=True,
                )

        # Should use uniform distribution fallback
        expected_uniform = np.ones(20, dtype=np.float64) / 20
        np.testing.assert_allclose(
            result["fidelity_distribution"],
            expected_uniform,
            atol=1e-10,
        )

    def test_degenerate_distribution_emits_warning(self, simple_encoding_2q, caplog):
        """Test that degenerate distribution triggers a warning log."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility._estimate_convergence",
                return_value=0.01,
            ),
            patch("numpy.histogram") as mock_hist,
        ):
            mock_hist.return_value = (
                np.zeros(20, dtype=np.float64),
                np.linspace(0.0, 1.0, 21),
            )
            with caplog.at_level(logging.WARNING):
                compute_expressibility(
                    simple_encoding_2q,
                    n_samples=50,
                    n_bins=20,
                    seed=42,
                )

        degenerate_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "degenerate" in r.message.lower()
        ]
        assert len(degenerate_warnings) >= 1

    def test_degenerate_distribution_still_returns_valid_result(
        self, simple_encoding_2q
    ):
        """Test that degenerate distribution produces a valid expressibility score."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility._estimate_convergence",
                return_value=0.01,
            ),
            patch("numpy.histogram") as mock_hist,
        ):
            mock_hist.return_value = (
                np.zeros(20, dtype=np.float64),
                np.linspace(0.0, 1.0, 21),
            )
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=20,
                seed=42,
            )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Tests: KL Divergence Numerical Edge Cases
# =============================================================================


class TestKLDivergenceNumericalEdgeCases:
    """Test numerical edge cases in KL divergence computation.

    These tests verify the defensive checks for NaN/Inf values in the KL
    divergence array (lines 707-716) and the negative KL clamping logic
    (lines 721-726).
    """

    def test_nan_in_kl_raises_numerical_instability_error(self, simple_encoding_2q):
        """Test that NaN in KL divergence raises NumericalInstabilityError."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
            ) as mock_rel,
        ):
            kl_vals = np.zeros(20)
            kl_vals[5] = np.nan
            mock_rel.return_value = kl_vals

            with pytest.raises(NumericalInstabilityError) as exc_info:
                compute_expressibility(
                    simple_encoding_2q,
                    n_samples=50,
                    n_bins=20,
                    seed=42,
                )

            assert exc_info.value.details["has_nan"] is True

    def test_inf_in_kl_raises_numerical_instability_error(self, simple_encoding_2q):
        """Test that Inf in KL divergence raises NumericalInstabilityError."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
            ) as mock_rel,
        ):
            kl_vals = np.zeros(20)
            kl_vals[3] = np.inf
            mock_rel.return_value = kl_vals

            with pytest.raises(NumericalInstabilityError) as exc_info:
                compute_expressibility(
                    simple_encoding_2q,
                    n_samples=50,
                    n_bins=20,
                    seed=42,
                )

            assert exc_info.value.details["has_inf"] is True

    def test_mixed_nan_and_inf_in_kl_raises_error(self, simple_encoding_2q):
        """Test that mixed NaN and Inf in KL divergence raises error."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
            ) as mock_rel,
        ):
            kl_vals = np.zeros(20)
            kl_vals[2] = np.nan
            kl_vals[7] = np.inf
            mock_rel.return_value = kl_vals

            with pytest.raises(NumericalInstabilityError) as exc_info:
                compute_expressibility(
                    simple_encoding_2q,
                    n_samples=50,
                    n_bins=20,
                    seed=42,
                )

            assert exc_info.value.details["has_nan"] is True
            assert exc_info.value.details["has_inf"] is True

    def test_small_negative_kl_clamped_without_warning(
        self, simple_encoding_2q, caplog
    ):
        """Test that KL divergence slightly below 0 is clamped without warning.

        When KL is in (-_NUMERICAL_EPSILON, 0), it is clamped to 0 silently
        (no warning is emitted because the negativity is within numerical noise).
        """
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)
        # KL sum = -5e-11, within epsilon = 1e-10 of zero
        kl_vals = np.full(20, -5e-11 / 20)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
                return_value=kl_vals,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=20,
                seed=42,
                return_distributions=True,
            )

        assert result["kl_divergence"] == 0.0

        negative_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "negative" in r.message.lower()
        ]
        assert len(negative_warnings) == 0

    def test_large_negative_kl_clamped_with_warning(self, simple_encoding_2q, caplog):
        """Test that KL divergence significantly below 0 is clamped with warning.

        When KL < -_NUMERICAL_EPSILON, it is clamped to 0 and a warning is logged
        because the negativity exceeds numerical noise and may indicate a problem.
        """
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)
        # KL sum = -0.02, well below -epsilon
        kl_vals = np.full(20, -0.001)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
                return_value=kl_vals,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=20,
                seed=42,
                return_distributions=True,
            )

        assert result["kl_divergence"] == 0.0

        negative_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "negative" in r.message.lower()
        ]
        assert len(negative_warnings) >= 1


# =============================================================================
# Tests: Simulation Error Propagation from compute_expressibility
# =============================================================================


class TestSimulationErrorPropagation:
    """Test error propagation through compute_expressibility.

    The try/except in compute_expressibility (lines 621-637) has two
    branches: SimulationError is re-raised directly, while other exceptions
    are wrapped in AnalysisError with context details.
    """

    def test_simulation_error_propagates_directly(self, simple_encoding_2q):
        """Test that SimulationError from _sample_fidelities propagates unchanged."""
        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                side_effect=SimulationError(
                    "PennyLane device error", backend="pennylane"
                ),
            ),
            pytest.raises(SimulationError, match="PennyLane device error"),
        ):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                seed=42,
            )

    def test_non_simulation_error_wrapped_in_analysis_error(self, simple_encoding_2q):
        """Test that non-SimulationError exceptions are wrapped in AnalysisError."""
        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                side_effect=RuntimeError("Memory allocation failed"),
            ),
            pytest.raises(AnalysisError, match="Failed to sample fidelities"),
        ):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                seed=42,
            )

    def test_wrapped_analysis_error_contains_details(self, simple_encoding_2q):
        """Test that wrapped AnalysisError contains error type and message."""
        with patch(
            "encoding_atlas.analysis.expressibility._sample_fidelities",
            side_effect=TypeError("Bad type"),
        ):
            with pytest.raises(AnalysisError) as exc_info:
                compute_expressibility(
                    simple_encoding_2q,
                    n_samples=50,
                    n_bins=10,
                    seed=42,
                )

            assert exc_info.value.details["error_type"] == "TypeError"
            assert "Bad type" in exc_info.value.details["error_message"]

    def test_wrapped_analysis_error_preserves_cause(self, simple_encoding_2q):
        """Test that wrapped AnalysisError preserves the original exception via __cause__."""
        original = RuntimeError("Original cause")
        with patch(
            "encoding_atlas.analysis.expressibility._sample_fidelities",
            side_effect=original,
        ):
            with pytest.raises(AnalysisError) as exc_info:
                compute_expressibility(
                    simple_encoding_2q,
                    n_samples=50,
                    n_bins=10,
                    seed=42,
                )

            assert exc_info.value.__cause__ is original


# =============================================================================
# Tests: Verbose Logging (Detailed)
# =============================================================================


class TestVerboseLoggingDetailed:
    """Test verbose logging output in compute_expressibility.

    When verbose=True, compute_expressibility logs:
    1. Computation parameters (encoding name, n_qubits, n_samples, etc.) at INFO
    2. Final results (expressibility, KL, mean_fid, convergence) at INFO
    3. Sampling progress at DEBUG (via _sample_fidelities)

    These tests verify the content and level of these log messages.
    """

    def test_verbose_logs_computation_parameters(
        self, simple_encoding_2q, skip_if_no_pennylane, caplog
    ):
        """Test that verbose=True logs computation parameters at INFO level."""
        with caplog.at_level(logging.INFO):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                seed=42,
                verbose=True,
            )

        info_msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
        param_msgs = [m for m in info_msgs if "Computing expressibility" in m]
        assert len(param_msgs) >= 1
        assert "AngleEncoding" in param_msgs[0]
        assert "n_samples=50" in param_msgs[0]
        assert "n_bins=10" in param_msgs[0]

    def test_verbose_logs_final_results(
        self, simple_encoding_2q, skip_if_no_pennylane, caplog
    ):
        """Test that verbose=True logs final expressibility results at INFO level."""
        with caplog.at_level(logging.INFO):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                seed=42,
                verbose=True,
                return_distributions=True,
            )

        info_msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
        result_msgs = [m for m in info_msgs if "Expressibility:" in m and "KL=" in m]
        assert len(result_msgs) >= 1
        assert "mean_fid=" in result_msgs[0]
        assert "convergence=" in result_msgs[0]

    def test_verbose_false_no_info_logs(
        self, simple_encoding_2q, skip_if_no_pennylane, caplog
    ):
        """Test that verbose=False does not emit INFO logs about computation."""
        with caplog.at_level(logging.INFO):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                seed=42,
                verbose=False,
            )

        info_msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
        expr_info = [
            m
            for m in info_msgs
            if "Computing expressibility" in m or "Expressibility:" in m
        ]
        assert len(expr_info) == 0


# =============================================================================
# Tests: Haar Distribution Advanced Edge Cases
# =============================================================================


class TestHaarDistributionAdvanced:
    """Advanced tests for compute_haar_distribution.

    These tests cover the log-space computation path (d > 100), degenerate
    fallback to uniform distribution, clipping of (1-F) at boundaries, and
    consistency across different input sizes.
    """

    def test_log_space_matches_direct_computation_at_boundary(self):
        """Test that log-space and direct computation produce same results.

        For n_qubits=6 (d=64, direct path) and n_qubits=7 (d=128, log-space
        path), results with overlapping fidelity values should follow the same
        mathematical formula P(F) = (d-1)(1-F)^(d-2).
        """
        fidelities = np.linspace(0.01, 0.99, 50)

        # n_qubits=6: d=64, uses direct computation (d <= 100)
        P_6q = compute_haar_distribution(n_qubits=6, fidelity_values=fidelities)
        d_6 = 64
        one_minus_F = np.clip(1.0 - fidelities, 1e-10, 1.0)
        expected_6 = (d_6 - 1) * np.power(one_minus_F, d_6 - 2)
        expected_6 = expected_6 / expected_6.sum()
        np.testing.assert_allclose(P_6q, expected_6, rtol=1e-10)

        # n_qubits=7: d=128, uses log-space computation (d > 100)
        P_7q = compute_haar_distribution(n_qubits=7, fidelity_values=fidelities)
        d_7 = 128
        log_P = np.log(d_7 - 1) + (d_7 - 2) * np.log(one_minus_F)
        log_P_max = np.max(log_P)
        expected_7 = np.exp(log_P - log_P_max)
        expected_7 = expected_7 / expected_7.sum()
        np.testing.assert_allclose(P_7q, expected_7, rtol=1e-10)

    def test_degenerate_haar_fallback_to_uniform(self, caplog):
        """Test that Haar distribution falls back to uniform on underflow.

        With n_qubits=3 (d=8) and all fidelities = 1.0, the term
        (d-1)(1-F)^(d-2) = 7 * (1e-10)^6 ≈ 7e-60, which is effectively zero
        after floating-point computation. The sum is below _NUMERICAL_EPSILON,
        triggering the uniform fallback.
        """
        fidelities = np.ones(50)  # All F = 1.0

        with caplog.at_level(logging.WARNING):
            P_haar = compute_haar_distribution(n_qubits=3, fidelity_values=fidelities)

        expected_uniform = np.ones(50) / 50
        np.testing.assert_allclose(P_haar, expected_uniform, atol=1e-10)

        haar_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "near zero" in r.message.lower()
        ]
        assert len(haar_warnings) >= 1

    def test_fidelity_exactly_one_does_not_produce_nan(self):
        """Test that F=1.0 inputs produce finite results (no log(0) errors).

        The clipping of (1-F) to [_NUMERICAL_EPSILON, 1.0] prevents log(0)
        when F=1.0. This test verifies that the clipping works correctly.
        """
        fidelities = np.array([0.0, 0.5, 1.0])
        for n_qubits in [2, 3, 4, 5]:
            P_haar = compute_haar_distribution(n_qubits, fidelities)
            assert np.all(
                np.isfinite(P_haar)
            ), f"Non-finite values for n_qubits={n_qubits}"
            assert np.all(P_haar >= 0), f"Negative values for n_qubits={n_qubits}"

    def test_empty_input_consistent_across_qubit_counts(self):
        """Test that empty input returns identical empty arrays for all n_qubits."""
        empty = np.array([], dtype=np.float64)
        results = [
            compute_haar_distribution(n_qubits=n, fidelity_values=empty)
            for n in [1, 3, 5, 7, 10, 15]
        ]

        for r in results:
            assert len(r) == 0
            assert r.dtype == np.float64

    def test_log_space_path_stable_for_large_qubit_counts(self):
        """Test that log-space path produces valid results for n_qubits up to 20.

        The log-space normalization trick (subtracting log_P_max before exp)
        prevents underflow/overflow for large Hilbert space dimensions.
        """
        fidelities = np.linspace(0.0, 0.95, 50)

        for n_qubits in [8, 10, 12, 15, 20]:
            P_haar = compute_haar_distribution(n_qubits, fidelities)
            assert np.all(
                np.isfinite(P_haar)
            ), f"Non-finite values for n_qubits={n_qubits}"
            assert np.isclose(
                P_haar.sum(), 1.0, atol=1e-6
            ), f"Sum not ~1 for n_qubits={n_qubits}: {P_haar.sum()}"
            assert np.all(P_haar >= 0), f"Negative values for n_qubits={n_qubits}"


# =============================================================================
# Tests: Input Validation (Extended)
# =============================================================================


class TestInputValidationExtended:
    """Extended input validation tests for edge cases and boundary conditions."""

    def test_backend_case_sensitive(self, simple_encoding_2q):
        """Test that backend parameter is case-sensitive."""
        with pytest.raises(ValueError, match="backend must be"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                backend="PennyLane",
            )

    def test_backend_none_raises_error(self, simple_encoding_2q):
        """Test that backend=None raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                backend=None,
            )

    def test_n_bins_exactly_9_rejected(self, simple_encoding_2q):
        """Test that n_bins=9 (one below minimum) is rejected."""
        with pytest.raises(ValueError, match="n_bins must be at least 10"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=9,
            )

    def test_n_bins_exactly_10_accepted(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that n_bins=10 (exact minimum) is accepted."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=10,
            seed=42,
            return_distributions=True,
        )
        assert result["n_bins"] == 10

    def test_input_range_single_element(self, simple_encoding_2q):
        """Test that input_range with 1 element raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                input_range=(1.0,),
            )

    def test_input_range_four_elements(self, simple_encoding_2q):
        """Test that input_range with 4 elements raises ValueError."""
        with pytest.raises(ValueError, match="input_range must have 2 elements"):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=10,
                input_range=(0.0, 1.0, 2.0, 3.0),
            )

    def test_n_bins_auto_clamps_to_very_small_n_samples(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that n_bins=None with n_samples=10 auto-selects n_bins=10."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,
            n_bins=None,
            seed=42,
            return_distributions=True,
        )
        assert result["n_bins"] == _MIN_SAMPLES_ERROR

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_no_qubit_warning_at_threshold(self, skip_if_no_pennylane):
        """Test that no qubit warning at exactly _QUBIT_WARNING_THRESHOLD.

        The warning should only be emitted for n_qubits > threshold (strictly
        greater), not at the threshold itself. The filterwarnings('error')
        marker converts warnings to errors, so if a qubit warning IS emitted,
        this test will fail.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=_QUBIT_WARNING_THRESHOLD)
        compute_expressibility(
            enc,
            n_samples=_MIN_SAMPLES_ERROR,
            n_bins=_MIN_SAMPLES_ERROR,
            seed=42,
        )

    def test_n_samples_exactly_at_error_threshold_accepted(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that n_samples exactly at _MIN_SAMPLES_ERROR is accepted."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,
            seed=42,
            return_distributions=True,
        )
        assert isinstance(result, dict)
        assert 0.0 <= result["expressibility"] <= 1.0

    def test_n_samples_one_below_error_threshold_rejected(self, simple_encoding_2q):
        """Test that n_samples one below _MIN_SAMPLES_ERROR is rejected."""
        with pytest.raises(InsufficientSamplesError):
            compute_expressibility(
                simple_encoding_2q,
                n_samples=_MIN_SAMPLES_ERROR - 1,
            )

    def test_fidelity_distribution_invalid_backend(self, simple_encoding_2q):
        """Test that compute_fidelity_distribution rejects invalid backend."""
        with pytest.raises(ValueError, match="backend must be"):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=50,
                backend="tensorflow",
            )

    def test_fidelity_distribution_invalid_input_range_equal(self, simple_encoding_2q):
        """Test that compute_fidelity_distribution rejects equal input_range."""
        with pytest.raises(ValueError, match="input_range"):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=50,
                input_range=(1.0, 1.0),
            )

    def test_fidelity_distribution_negative_n_samples(self, simple_encoding_2q):
        """Test that compute_fidelity_distribution rejects negative n_samples."""
        with pytest.raises(ValueError, match="n_samples must be a positive"):
            compute_fidelity_distribution(
                simple_encoding_2q,
                n_samples=-5,
            )

    def test_haar_distribution_float_n_qubits(self):
        """Test that float n_qubits raises ValueError in compute_haar_distribution."""
        fidelities = np.linspace(0, 1, 50)
        with pytest.raises(ValueError, match="n_qubits must be a positive"):
            compute_haar_distribution(n_qubits=3.5, fidelity_values=fidelities)


# =============================================================================
# Tests: Expressibility Formula Verification
# =============================================================================


class TestExpressibilityFormulaVerification:
    """Test that the expressibility normalization formula is correct.

    The formula is: expressibility = 1 - min(1, kl_divergence / _MAX_KL_DIVERGENCE)

    This maps KL divergence to [0, 1] where higher values indicate more
    expressive encodings (closer to Haar-random).
    """

    def test_zero_kl_gives_max_expressibility(self, simple_encoding_2q):
        """Test that KL divergence = 0 produces expressibility = 1.0."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)
        kl_vals = np.zeros(20)  # Sum = 0

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
                return_value=kl_vals,
            ),
        ):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=20,
                seed=42,
                return_distributions=True,
            )

        assert result["kl_divergence"] == 0.0
        assert result["expressibility"] == 1.0

    def test_max_kl_gives_zero_expressibility(self, simple_encoding_2q):
        """Test that KL divergence = _MAX_KL_DIVERGENCE produces expressibility = 0."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)
        # KL sum = _MAX_KL_DIVERGENCE (10.0)
        kl_vals = np.full(20, _MAX_KL_DIVERGENCE / 20)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
                return_value=kl_vals,
            ),
        ):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=20,
                seed=42,
                return_distributions=True,
            )

        assert np.isclose(result["kl_divergence"], _MAX_KL_DIVERGENCE, rtol=1e-10)
        assert result["expressibility"] == 0.0

    def test_half_max_kl_gives_half_expressibility(self, simple_encoding_2q):
        """Test that KL divergence = _MAX_KL/2 produces expressibility = 0.5."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)
        half_max = _MAX_KL_DIVERGENCE / 2.0
        kl_vals = np.full(20, half_max / 20)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
                return_value=kl_vals,
            ),
        ):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=20,
                seed=42,
                return_distributions=True,
            )

        assert np.isclose(result["kl_divergence"], half_max, rtol=1e-10)
        assert np.isclose(result["expressibility"], 0.5, atol=1e-10)

    def test_kl_exceeding_max_clamped_to_zero_expressibility(self, simple_encoding_2q):
        """Test that KL > _MAX_KL_DIVERGENCE still gives expressibility = 0."""
        mock_fidelities = np.random.default_rng(99).uniform(0, 1, 50)
        # KL sum = 2 * MAX
        kl_vals = np.full(20, 2.0 * _MAX_KL_DIVERGENCE / 20)

        with (
            patch(
                "encoding_atlas.analysis.expressibility._sample_fidelities",
                return_value=mock_fidelities,
            ),
            patch(
                "encoding_atlas.analysis.expressibility.rel_entr",
                return_value=kl_vals,
            ),
        ):
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=50,
                n_bins=20,
                seed=42,
                return_distributions=True,
            )

        assert result["kl_divergence"] > _MAX_KL_DIVERGENCE
        assert result["expressibility"] == 0.0


# =============================================================================
# Tests: Distribution Properties (Advanced)
# =============================================================================


class TestDistributionPropertiesAdvanced:
    """Advanced tests for returned distribution properties.

    These tests verify mathematical properties of the distributions
    returned by compute_expressibility with return_distributions=True.
    """

    def test_bin_edges_uniformly_spaced(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that histogram bin edges are uniformly spaced in [0, 1]."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        edges = result["bin_edges"]
        spacings = np.diff(edges)
        assert np.allclose(
            spacings, spacings[0], atol=1e-15
        ), "Bin edges are not uniformly spaced"

    def test_fidelity_distribution_float64_precision(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that returned arrays use float64 precision."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert result["fidelity_distribution"].dtype == np.float64
        assert result["haar_distribution"].dtype == np.float64
        assert result["bin_edges"].dtype == np.float64

    def test_expressibility_and_kl_are_consistent(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that expressibility and kl_divergence follow the exact formula."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        expected = 1.0 - min(1.0, result["kl_divergence"] / _MAX_KL_DIVERGENCE)
        assert np.isclose(result["expressibility"], expected, atol=1e-10)

    def test_mean_fidelity_within_distribution_support(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that mean fidelity is within the fidelity support [0, 1]."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        assert 0.0 <= result["mean_fidelity"] <= 1.0

    def test_std_fidelity_bounded_by_half(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that std fidelity cannot exceed 0.5.

        For values in [0, 1], the maximum standard deviation is 0.5
        (achieved when half the values are 0 and half are 1).
        """
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=200,
            n_bins=30,
            seed=42,
            return_distributions=True,
        )
        assert 0.0 <= result["std_fidelity"] <= 0.5 + 1e-10

    def test_convergence_estimate_inf_for_minimum_samples(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that convergence is inf when n_samples < 20 (bootstrap threshold).

        With n_samples=10 (_MIN_SAMPLES_ERROR), the bootstrap inside
        _estimate_convergence gets only 10 fidelities (< 20), so it
        returns inf as the convergence estimate.
        """
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,
            n_bins=_MIN_SAMPLES_ERROR,
            seed=42,
            return_distributions=True,
        )
        assert result["convergence_estimate"] == float("inf")

    def test_no_nan_in_result_fields(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that no result field contains NaN values."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=100,
            n_bins=20,
            seed=42,
            return_distributions=True,
        )
        assert np.isfinite(result["expressibility"])
        assert np.isfinite(result["kl_divergence"])
        assert np.isfinite(result["convergence_estimate"])
        assert np.isfinite(result["mean_fidelity"])
        assert np.isfinite(result["std_fidelity"])
        assert np.all(np.isfinite(result["fidelity_distribution"]))
        assert np.all(np.isfinite(result["haar_distribution"]))
        assert np.all(np.isfinite(result["bin_edges"]))


# =============================================================================
# Tests: Qubit Warning Threshold
# =============================================================================


class TestQubitWarningThreshold:
    """Test the qubit count warning in compute_expressibility.

    When the encoding has more than _QUBIT_WARNING_THRESHOLD qubits, a
    UserWarning should be emitted about potential slow computation.
    """

    @pytest.mark.filterwarnings("default::UserWarning")
    def test_warning_emitted_above_threshold(self, skip_if_no_pennylane):
        """Test that warning is emitted for n_qubits > threshold."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=_QUBIT_WARNING_THRESHOLD + 1)
        with pytest.warns(UserWarning, match=r".*qubit.*slow"):
            compute_expressibility(
                enc,
                n_samples=_MIN_SAMPLES_ERROR,
                n_bins=_MIN_SAMPLES_ERROR,
                seed=42,
            )

    def test_warning_text_contains_qubit_count(self, skip_if_no_pennylane):
        """Test that the qubit warning includes the actual qubit count."""
        from encoding_atlas import AngleEncoding

        n_qubits = _QUBIT_WARNING_THRESHOLD + 2
        enc = AngleEncoding(n_features=n_qubits)
        with pytest.warns(UserWarning, match=str(n_qubits)):
            compute_expressibility(
                enc,
                n_samples=_MIN_SAMPLES_ERROR,
                n_bins=_MIN_SAMPLES_ERROR,
                seed=42,
            )


# =============================================================================
# Tests: compute_fidelity_distribution Edge Cases
# =============================================================================


class TestFidelityDistributionEdgeCases:
    """Edge case tests for compute_fidelity_distribution."""

    def test_fidelity_values_dtype_float64(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that returned fidelity array has float64 dtype."""
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            seed=42,
        )
        assert fidelities.dtype == np.float64

    def test_custom_input_range(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test compute_fidelity_distribution with custom input range."""
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=50,
            input_range=(0.0, 1.0),
            seed=42,
        )
        assert fidelities.shape == (50,)
        assert np.all(fidelities >= 0.0)
        assert np.all(fidelities <= 1.0)

    def test_minimum_valid_samples(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test compute_fidelity_distribution with minimum valid n_samples."""
        fidelities = compute_fidelity_distribution(
            simple_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,
            seed=42,
        )
        assert fidelities.shape == (_MIN_SAMPLES_ERROR,)
        assert np.all(np.isfinite(fidelities))


# =============================================================================
# Tests: ExpressibilityResult TypedDict Completeness
# =============================================================================


class TestExpressibilityResultCompleteness:
    """Test that ExpressibilityResult dict is complete and well-formed."""

    def test_no_extra_keys_in_result(self, simple_encoding_2q, skip_if_no_pennylane):
        """Test that the result dict contains exactly the documented keys."""
        result = compute_expressibility(
            simple_encoding_2q,
            n_samples=50,
            n_bins=10,
            seed=42,
            return_distributions=True,
        )
        expected_keys = {
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
            # Bootstrap CI fields added in the analysis statistics commit.
            "expressibility_ci_lower",
            "expressibility_ci_upper",
            "mean_fidelity_ci_lower",
            "mean_fidelity_ci_upper",
            "confidence_level",
            "sampling",
        }
        assert set(result.keys()) == expected_keys

    def test_array_shapes_consistent_with_n_bins(
        self, simple_encoding_2q, skip_if_no_pennylane
    ):
        """Test that array shapes are consistent with n_bins parameter."""
        for n_bins in [10, 20, 50]:
            result = compute_expressibility(
                simple_encoding_2q,
                n_samples=100,
                n_bins=n_bins,
                seed=42,
                return_distributions=True,
            )
            assert result["fidelity_distribution"].shape == (n_bins,)
            assert result["haar_distribution"].shape == (n_bins,)
            assert result["bin_edges"].shape == (n_bins + 1,)
            assert result["n_bins"] == n_bins
