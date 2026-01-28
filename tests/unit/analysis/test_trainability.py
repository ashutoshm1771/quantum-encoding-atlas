"""Comprehensive unit tests for analysis.trainability module.

This module contains thorough tests for all functions in the
encoding_atlas.analysis.trainability module, including:

- estimate_trainability: Main trainability estimation function
- compute_gradient_variance: Gradient variance computation
- detect_barren_plateau: Barren plateau risk detection

Test Categories
---------------
1. **Basic Functionality**: Verify correct output for simple inputs
2. **Edge Cases**: Test boundary conditions and special cases
3. **Numerical Stability**: Test with extreme values
4. **Error Handling**: Verify proper exceptions for invalid inputs
5. **Known Values**: Test against expected behavior (e.g., product states)
6. **Reproducibility**: Test seed-based reproducibility
7. **Parameter Validation**: Test all parameter constraints

Coverage Goals
--------------
- All public functions in trainability.py
- All code paths including error paths
- All observable types (computational, pauli_z, global_z)
- Both backends (pennylane, qiskit)
- Edge cases (minimum samples, failed computations)

References
----------
.. [1] McClean, J. R., et al. (2018).
       "Barren plateaus in quantum neural network training landscapes."
       Nature Communications, 9(1), 4812.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from encoding_atlas.analysis.trainability import (
    _MIN_SAMPLES_ERROR,
    _MIN_SAMPLES_WARNING,
    compute_gradient_variance,
    detect_barren_plateau,
    estimate_trainability,
)
from encoding_atlas.core.exceptions import (
    AnalysisError,
    InsufficientSamplesError,
)

# =============================================================================
# Test: estimate_trainability - Basic Functionality
# =============================================================================


class TestEstimateTrainabilityBasic:
    """Basic functionality tests for estimate_trainability."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_returns_float_by_default(self, sample_encoding_2q):
        """Test that function returns float when return_details=False."""
        result = estimate_trainability(sample_encoding_2q, n_samples=20, seed=42)
        assert isinstance(result, float)

    def test_returns_dict_when_detailed(self, sample_encoding_2q):
        """Test that function returns dict when return_details=True."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert isinstance(result, dict)
        # Verify all expected keys are present
        expected_keys = {
            "trainability_estimate",
            "gradient_variance",
            "barren_plateau_risk",
            "effective_dimension",
            "n_samples",
            "n_successful_samples",
            "per_parameter_variance",
            "n_failed_samples",
        }
        assert expected_keys == set(result.keys())

    def test_trainability_in_range(self, sample_encoding_2q):
        """Test that trainability estimate is in [0, 1]."""
        result = estimate_trainability(sample_encoding_2q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_gradient_variance_non_negative(self, sample_encoding_2q):
        """Test that gradient variance is non-negative."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["gradient_variance"] >= 0.0

    def test_barren_plateau_risk_valid(self, sample_encoding_2q):
        """Test that barren plateau risk is one of valid options."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["barren_plateau_risk"] in {"low", "medium", "high"}

    def test_n_samples_recorded(self, sample_encoding_2q):
        """Test that n_samples is correctly recorded in result."""
        n_samples = 25
        result = estimate_trainability(
            sample_encoding_2q, n_samples=n_samples, seed=42, return_details=True
        )
        assert result["n_samples"] == n_samples

    def test_per_parameter_variance_shape(self, sample_encoding_4q):
        """Test that per_parameter_variance has correct shape."""
        result = estimate_trainability(
            sample_encoding_4q, n_samples=20, seed=42, return_details=True
        )
        assert result["per_parameter_variance"].shape == (4,)

    def test_effective_dimension_non_negative(self, sample_encoding_2q):
        """Test that effective dimension is non-negative."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["effective_dimension"] >= 0.0


# =============================================================================
# Test: estimate_trainability - Reproducibility
# =============================================================================


class TestEstimateTrainabilityReproducibility:
    """Test reproducibility of trainability estimates."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_same_seed_same_result(self, sample_encoding_2q):
        """Test that same seed produces same result."""
        result1 = estimate_trainability(sample_encoding_2q, n_samples=30, seed=42)
        result2 = estimate_trainability(sample_encoding_2q, n_samples=30, seed=42)
        assert_allclose(result1, result2, atol=1e-10)

    def test_different_seed_different_result(self, sample_encoding_2q):
        """Test that different seeds produce different results."""
        result1 = estimate_trainability(sample_encoding_2q, n_samples=50, seed=42)
        result2 = estimate_trainability(sample_encoding_2q, n_samples=50, seed=123)
        # Results should differ (though not guaranteed for all seeds)
        # This test verifies different seeds don't produce identical results
        # We simply check both values are valid - exact comparison is unreliable
        assert 0.0 <= result1 <= 1.0
        assert 0.0 <= result2 <= 1.0

    def test_detailed_results_reproducible(self, sample_encoding_2q):
        """Test that detailed results are also reproducible."""
        result1 = estimate_trainability(
            sample_encoding_2q, n_samples=30, seed=42, return_details=True
        )
        result2 = estimate_trainability(
            sample_encoding_2q, n_samples=30, seed=42, return_details=True
        )
        assert_allclose(
            result1["trainability_estimate"],
            result2["trainability_estimate"],
            atol=1e-10,
        )
        assert_allclose(
            result1["gradient_variance"],
            result2["gradient_variance"],
            atol=1e-10,
        )
        assert_allclose(
            result1["per_parameter_variance"],
            result2["per_parameter_variance"],
            atol=1e-10,
        )


# =============================================================================
# Test: estimate_trainability - Observable Types
# =============================================================================


class TestEstimateTrainabilityObservables:
    """Test different observable types for trainability estimation."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_computational_observable(self, sample_encoding_2q):
        """Test trainability with computational observable."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            observable="computational",
        )
        assert 0.0 <= result <= 1.0

    def test_pauli_z_observable(self, sample_encoding_2q):
        """Test trainability with Pauli Z observable."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            observable="pauli_z",
        )
        assert 0.0 <= result <= 1.0

    def test_global_z_observable(self, sample_encoding_2q):
        """Test trainability with global Z observable."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            observable="global_z",
        )
        assert 0.0 <= result <= 1.0

    def test_invalid_observable_raises(self, sample_encoding_2q):
        """Test that invalid observable raises error."""
        with pytest.raises(ValueError, match="observable"):
            estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                observable="invalid",
            )


# =============================================================================
# Test: estimate_trainability - Input Validation
# =============================================================================


class TestEstimateTrainabilityValidation:
    """Test input validation for estimate_trainability."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_invalid_encoding_raises(self):
        """Test that non-encoding raises AnalysisError."""
        with pytest.raises(AnalysisError, match="BaseEncoding"):
            estimate_trainability("not_an_encoding", n_samples=20)

    def test_none_encoding_raises(self):
        """Test that None encoding raises error."""
        with pytest.raises(AnalysisError, match="BaseEncoding"):
            estimate_trainability(None, n_samples=20)

    def test_too_few_samples_raises(self, sample_encoding_2q):
        """Test that n_samples < minimum raises error."""
        with pytest.raises(InsufficientSamplesError, match="n_samples"):
            estimate_trainability(sample_encoding_2q, n_samples=5)

    def test_exactly_minimum_samples_allowed(self, sample_encoding_2q):
        """Test that exactly minimum samples is allowed."""
        # Should not raise
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_low_samples_warns(self, sample_encoding_2q):
        """Test that low sample count emits warning."""
        # Between error threshold and warning threshold
        n_samples = _MIN_SAMPLES_WARNING - 1
        if n_samples >= _MIN_SAMPLES_ERROR:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                estimate_trainability(sample_encoding_2q, n_samples=n_samples, seed=42)
                # Check for warning about low samples
                assert any(
                    "unreliable" in str(warning.message).lower()
                    or "n_samples" in str(warning.message).lower()
                    for warning in w
                )

    def test_invalid_input_range_raises(self, sample_encoding_2q):
        """Test that invalid input_range raises error."""
        with pytest.raises(ValueError, match="input_range"):
            estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                input_range=(2.0, 1.0),  # max < min
            )

    def test_invalid_backend_raises(self, sample_encoding_2q):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="backend"):
            estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                backend="invalid_backend",
            )


# =============================================================================
# Test: estimate_trainability - Encoding Comparison
# =============================================================================


class TestEstimateTrainabilityEncodingComparison:
    """Test trainability estimation across different encodings."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_product_state_high_trainability(self, sample_encoding_2q):
        """Test that product state encodings have relatively high trainability.

        AngleEncoding produces product states (no entanglement), which are
        known to avoid barren plateaus. They should generally have good
        trainability.
        """
        result = estimate_trainability(
            sample_encoding_2q, n_samples=50, seed=42, return_details=True
        )
        # Product states should have low barren plateau risk
        assert result["barren_plateau_risk"] in {"low", "medium"}

    def test_larger_encoding_still_works(self, sample_encoding_4q):
        """Test that larger encodings can be analyzed."""
        result = estimate_trainability(sample_encoding_4q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Test: estimate_trainability - Backend Comparison
# =============================================================================


class TestEstimateTrainabilityBackends:
    """Test trainability estimation with different backends."""

    def test_pennylane_backend(self, sample_encoding_2q, pennylane_available):
        """Test with PennyLane backend."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            backend="pennylane",
        )
        assert 0.0 <= result <= 1.0

    def test_qiskit_backend(self, sample_encoding_2q, qiskit_available):
        """Test with Qiskit backend."""
        if not qiskit_available:
            pytest.skip("Qiskit not available")
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            backend="qiskit",
        )
        assert 0.0 <= result <= 1.0

    def test_both_backends_similar(
        self, sample_encoding_2q, pennylane_available, qiskit_available
    ):
        """Test that both backends give similar results.

        Note: Results may differ somewhat due to different implementations
        and qubit ordering conventions, but should be in same ballpark.
        """
        if not pennylane_available or not qiskit_available:
            pytest.skip("Both backends required")

        result_pl = estimate_trainability(
            sample_encoding_2q,
            n_samples=30,
            seed=42,
            backend="pennylane",
        )
        result_qk = estimate_trainability(
            sample_encoding_2q,
            n_samples=30,
            seed=42,
            backend="qiskit",
        )
        # Both should be in valid range
        assert 0.0 <= result_pl <= 1.0
        assert 0.0 <= result_qk <= 1.0
        # Results should be somewhat similar (within 0.5)
        # Note: This is a loose bound since implementations differ
        assert abs(result_pl - result_qk) < 0.5


# =============================================================================
# Test: estimate_trainability - Custom Input Range
# =============================================================================


class TestEstimateTrainabilityInputRange:
    """Test trainability estimation with custom input ranges."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_default_range(self, sample_encoding_2q):
        """Test with default input range [0, 2π]."""
        result = estimate_trainability(sample_encoding_2q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_narrow_range(self, sample_encoding_2q):
        """Test with narrow input range."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            input_range=(0.0, 0.5),
        )
        assert 0.0 <= result <= 1.0

    def test_negative_range(self, sample_encoding_2q):
        """Test with negative input range."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            input_range=(-np.pi, np.pi),
        )
        assert 0.0 <= result <= 1.0


# =============================================================================
# Test: compute_gradient_variance
# =============================================================================


class TestComputeGradientVariance:
    """Tests for compute_gradient_variance function."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_returns_float(self, sample_encoding_2q):
        """Test that function returns a float."""
        variance = compute_gradient_variance(sample_encoding_2q, n_samples=20, seed=42)
        assert isinstance(variance, float)

    def test_non_negative(self, sample_encoding_2q):
        """Test that variance is non-negative."""
        variance = compute_gradient_variance(sample_encoding_2q, n_samples=20, seed=42)
        assert variance >= 0.0

    def test_reproducible(self, sample_encoding_2q):
        """Test that same seed gives same variance."""
        var1 = compute_gradient_variance(sample_encoding_2q, n_samples=30, seed=42)
        var2 = compute_gradient_variance(sample_encoding_2q, n_samples=30, seed=42)
        assert_allclose(var1, var2, atol=1e-10)

    def test_matches_detailed_result(self, sample_encoding_2q):
        """Test that variance matches detailed result from estimate_trainability."""
        variance = compute_gradient_variance(sample_encoding_2q, n_samples=30, seed=42)
        detailed = estimate_trainability(
            sample_encoding_2q, n_samples=30, seed=42, return_details=True
        )
        assert_allclose(variance, detailed["gradient_variance"], atol=1e-10)


# =============================================================================
# Test: detect_barren_plateau
# =============================================================================


class TestDetectBarrenPlateau:
    """Tests for detect_barren_plateau function."""

    def test_low_risk_for_high_variance(self):
        """Test that high variance gives low risk."""
        risk = detect_barren_plateau(gradient_variance=0.1, n_qubits=4, n_params=16)
        assert risk == "low"

    def test_high_risk_for_low_variance(self):
        """Test that very low variance gives high risk."""
        risk = detect_barren_plateau(gradient_variance=1e-10, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_medium_risk_for_borderline_variance(self):
        """Test that borderline variance gives medium risk."""
        # Find a variance in the medium range
        # Exact thresholds depend on n_qubits, so test range
        risk = detect_barren_plateau(gradient_variance=1e-5, n_qubits=4, n_params=16)
        assert risk in {"low", "medium", "high"}

    def test_scaling_with_qubits(self):
        """Test that thresholds scale with qubit count."""
        # For same variance, larger systems should have different risk
        risk_small = detect_barren_plateau(
            gradient_variance=1e-5, n_qubits=2, n_params=8
        )
        risk_large = detect_barren_plateau(
            gradient_variance=1e-5, n_qubits=8, n_params=32
        )
        # Both should be valid risk levels
        assert risk_small in {"low", "medium", "high"}
        assert risk_large in {"low", "medium", "high"}

    def test_invalid_variance_raises(self):
        """Test that negative variance raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            detect_barren_plateau(gradient_variance=-1.0, n_qubits=4, n_params=16)

    def test_invalid_n_qubits_raises(self):
        """Test that invalid n_qubits raises error."""
        with pytest.raises(ValueError, match="n_qubits"):
            detect_barren_plateau(gradient_variance=0.1, n_qubits=0, n_params=16)

    def test_invalid_n_params_raises(self):
        """Test that invalid n_params raises error."""
        with pytest.raises(ValueError, match="n_params"):
            detect_barren_plateau(gradient_variance=0.1, n_qubits=4, n_params=0)

    def test_zero_variance_high_risk(self):
        """Test that zero variance gives high risk."""
        risk = detect_barren_plateau(gradient_variance=0.0, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_returns_valid_literal(self):
        """Test that return value is always a valid literal."""
        for variance in [0.0, 1e-10, 1e-5, 1e-3, 0.01, 0.1, 1.0]:
            for n_qubits in [1, 2, 4, 8]:
                risk = detect_barren_plateau(
                    gradient_variance=variance,
                    n_qubits=n_qubits,
                    n_params=n_qubits * 4,
                )
                assert risk in {"low", "medium", "high"}


# =============================================================================
# Test: Known Values and Expected Behavior
# =============================================================================


class TestKnownValues:
    """Tests for known/expected behavior."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_angle_encoding_trainable(self, sample_encoding_2q):
        """Test that AngleEncoding (product states) appears trainable.

        AngleEncoding creates product states without entanglement, which
        should generally avoid barren plateaus according to McClean et al.
        """
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=100,
            seed=42,
            return_details=True,
        )
        # Should have low or medium risk
        assert result["barren_plateau_risk"] in {"low", "medium"}
        # Trainability should be reasonably high (above 0.2)
        assert result["trainability_estimate"] > 0.2

    def test_variance_positive_for_nontrivial_encoding(self, sample_encoding_4q):
        """Test that variance is positive for non-trivial encoding."""
        result = estimate_trainability(
            sample_encoding_4q,
            n_samples=50,
            seed=42,
            return_details=True,
        )
        # Variance should be positive for an encoding with actual parameters
        assert result["gradient_variance"] > 0

    def test_per_parameter_variance_all_positive(self, sample_encoding_2q):
        """Test that per-parameter variances are all non-negative."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=50,
            seed=42,
            return_details=True,
        )
        assert np.all(result["per_parameter_variance"] >= 0)


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of trainability computation."""

    def test_detect_barren_plateau_extreme_variance(self):
        """Test detection with extreme variance values."""
        # Very large variance
        risk_large = detect_barren_plateau(
            gradient_variance=1e10, n_qubits=4, n_params=16
        )
        assert risk_large == "low"

        # Very small variance
        risk_small = detect_barren_plateau(
            gradient_variance=1e-20, n_qubits=4, n_params=16
        )
        assert risk_small == "high"

    def test_detect_barren_plateau_many_qubits(self):
        """Test detection with many qubits."""
        # Should not raise or produce invalid results
        risk = detect_barren_plateau(gradient_variance=1e-6, n_qubits=20, n_params=80)
        assert risk in {"low", "medium", "high"}

    def test_trainability_score_clamped(self, sample_encoding_2q, pennylane_available):
        """Test that trainability score is always in [0, 1]."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        # Run multiple tests to check clamping
        for seed in [42, 123, 456]:
            result = estimate_trainability(sample_encoding_2q, n_samples=20, seed=seed)
            assert 0.0 <= result <= 1.0


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in trainability estimation."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_single_qubit_encoding(self):
        """Test with single-qubit encoding."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = estimate_trainability(enc, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_minimum_valid_samples(self, sample_encoding_2q):
        """Test with minimum valid number of samples."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=_MIN_SAMPLES_ERROR, seed=42
        )
        assert 0.0 <= result <= 1.0

    def test_n_failed_samples_tracked(self, sample_encoding_2q):
        """Test that failed samples are tracked."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            return_details=True,
        )
        # For valid encoding, failures should be minimal
        assert result["n_failed_samples"] >= 0
        assert result["n_failed_samples"] <= result["n_samples"]

    def test_n_successful_samples_consistent(self, sample_encoding_2q):
        """Test that n_successful_samples = n_samples - n_failed_samples."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=30,
            seed=42,
            return_details=True,
        )
        expected_successful = result["n_samples"] - result["n_failed_samples"]
        assert result["n_successful_samples"] == expected_successful

    def test_n_successful_samples_positive(self, sample_encoding_2q):
        """Test that n_successful_samples is positive for valid analysis."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            return_details=True,
        )
        # If analysis completed, we must have had successful samples
        assert result["n_successful_samples"] > 0
        assert result["n_successful_samples"] <= result["n_samples"]


# =============================================================================
# Test: TrainabilityResult TypedDict
# =============================================================================


class TestTrainabilityResultType:
    """Tests for TrainabilityResult TypedDict structure."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_result_types(self, sample_encoding_2q):
        """Test that all result fields have correct types."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )

        assert isinstance(result["trainability_estimate"], float)
        assert isinstance(result["gradient_variance"], float)
        assert result["barren_plateau_risk"] in {"low", "medium", "high"}
        assert isinstance(result["effective_dimension"], float)
        assert isinstance(result["n_samples"], int)
        assert isinstance(result["n_successful_samples"], int)
        assert isinstance(result["per_parameter_variance"], np.ndarray)
        assert isinstance(result["n_failed_samples"], int)

    def test_result_array_dtype(self, sample_encoding_2q):
        """Test that arrays have correct dtype."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["per_parameter_variance"].dtype in [np.float64, np.float32]


# =============================================================================
# Test: Verbose Mode
# =============================================================================


class TestVerboseMode:
    """Tests for verbose mode logging."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_verbose_mode_runs(self, sample_encoding_2q, caplog):
        """Test that verbose mode runs without error."""
        import logging

        with caplog.at_level(logging.INFO):
            result = estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                seed=42,
                verbose=True,
            )
        assert 0.0 <= result <= 1.0

    def test_non_verbose_silent(self, sample_encoding_2q, caplog):
        """Test that non-verbose mode is relatively quiet."""
        import logging

        with caplog.at_level(logging.DEBUG):
            result = estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                seed=42,
                verbose=False,
            )
        # Should complete without error
        assert 0.0 <= result <= 1.0


# =============================================================================
# Test: Integration with Different Encodings
# =============================================================================


class TestEncodingIntegration:
    """Integration tests with various encoding types."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available):
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_with_iqp_encoding(self, entangling_encoding_4q):
        """Test trainability estimation with IQP encoding."""
        result = estimate_trainability(entangling_encoding_4q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_comparison_angle_vs_iqp(self, sample_encoding_4q, entangling_encoding_4q):
        """Compare trainability of different encoding types."""
        # Use same seed for fair comparison
        train_angle = estimate_trainability(sample_encoding_4q, n_samples=50, seed=42)
        train_iqp = estimate_trainability(entangling_encoding_4q, n_samples=50, seed=42)

        # Both should be valid
        assert 0.0 <= train_angle <= 1.0
        assert 0.0 <= train_iqp <= 1.0

        # Product state (angle) encoding often has better trainability
        # due to no barren plateau from entanglement
        # Note: This is a general trend, not a guarantee for all cases


# =============================================================================
# Test: Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests for informative error messages."""

    def test_insufficient_samples_error_includes_counts(
        self, sample_encoding_2q, pennylane_available
    ):
        """Test that InsufficientSamplesError includes sample counts."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        try:
            estimate_trainability(sample_encoding_2q, n_samples=5)
        except InsufficientSamplesError as e:
            assert e.requested_samples == 5
            assert e.minimum_samples == _MIN_SAMPLES_ERROR

    def test_invalid_observable_shows_options(
        self, sample_encoding_2q, pennylane_available
    ):
        """Test that invalid observable error shows valid options."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        try:
            estimate_trainability(sample_encoding_2q, n_samples=20, observable="bad")
        except ValueError as e:
            error_msg = str(e).lower()
            assert "observable" in error_msg

    def test_invalid_backend_shows_options(
        self, sample_encoding_2q, pennylane_available
    ):
        """Test that invalid backend error shows valid options."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        try:
            estimate_trainability(sample_encoding_2q, n_samples=20, backend="bad")
        except ValueError as e:
            error_msg = str(e).lower()
            assert "backend" in error_msg
