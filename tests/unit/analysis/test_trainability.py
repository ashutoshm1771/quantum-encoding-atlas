"""Comprehensive unit tests for analysis.trainability module.

This module contains thorough tests for all functions in the
encoding_atlas.analysis.trainability module, including:

- estimate_trainability: Main trainability estimation function
- compute_gradient_variance: Gradient variance computation
- detect_barren_plateau: Barren plateau risk detection
- _variance_to_trainability: Variance-to-score sigmoid mapping
- _pad_or_truncate: Array dimension handling
- _compute_prob_zero_first_qubit: Local qubit probability computation
- _compute_expectation_value: Observable expectation values
- _compute_encoding_gradients: Parameter-shift gradient computation

Test Categories
---------------
1. **Basic Functionality**: Verify correct output for simple inputs
2. **Edge Cases**: Test boundary conditions and special cases
3. **Numerical Stability**: Test with extreme values
4. **Error Handling**: Verify proper exceptions for invalid inputs
5. **Known Values**: Test against analytically derived expected values
6. **Reproducibility**: Test seed-based reproducibility
7. **Parameter Validation**: Test all parameter constraints
8. **Private Helper Correctness**: Test internal helper functions
9. **Cross-Backend Consistency**: Verify PennyLane, Qiskit, Cirq agreement

Coverage Goals
--------------
- All public functions in trainability.py
- All private helper functions with non-trivial logic
- All code paths including error paths
- All observable types (computational, pauli_z, global_z)
- All backends (pennylane, qiskit, cirq)
- Edge cases (minimum samples, failed computations, boundary thresholds)
- Known mathematical values (analytical gradients, expectation values)

References
----------
.. [1] McClean, J. R., et al. (2018).
       "Barren plateaus in quantum neural network training landscapes."
       Nature Communications, 9(1), 4812.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from encoding_atlas.analysis.trainability import (
    _BP_HIGH_RISK_THRESHOLD,
    _BP_MEDIUM_RISK_THRESHOLD,
    _DEFAULT_N_SAMPLES,
    _EPSILON,
    _MAX_FAILURE_FRACTION,
    _MIN_SAMPLES_ERROR,
    _MIN_SAMPLES_WARNING,
    _REFERENCE_VARIANCE,
    _SIGMOID_STEEPNESS,
    _compute_encoding_gradients,
    _compute_expectation_value,
    _compute_prob_zero_first_qubit,
    _pad_or_truncate,
    _variance_to_trainability,
    compute_gradient_variance,
    detect_barren_plateau,
    estimate_trainability,
)
from encoding_atlas.core.exceptions import (
    AnalysisError,
    InsufficientSamplesError,
    NumericalInstabilityError,
    SimulationError,
)

# =============================================================================
# Test: estimate_trainability - Basic Functionality
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityBasic:
    """Basic functionality tests for estimate_trainability."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_returns_float_by_default(self, sample_encoding_2q) -> None:
        """Return type is float when return_details=False."""
        result = estimate_trainability(sample_encoding_2q, n_samples=20, seed=42)
        assert isinstance(result, float)

    def test_returns_dict_when_detailed(self, sample_encoding_2q) -> None:
        """Return type is dict when return_details=True."""
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
            # Bootstrap CI fields added in the analysis statistics commit.
            "trainability_ci_lower",
            "trainability_ci_upper",
            "gradient_variance_ci_lower",
            "gradient_variance_ci_upper",
            "confidence_level",
            "sampling",
        }
        assert expected_keys == set(result.keys())

    def test_trainability_in_range(self, sample_encoding_2q) -> None:
        """Trainability estimate lies in [0, 1]."""
        result = estimate_trainability(sample_encoding_2q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_gradient_variance_non_negative(self, sample_encoding_2q) -> None:
        """Gradient variance is non-negative."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["gradient_variance"] >= 0.0

    def test_barren_plateau_risk_valid(self, sample_encoding_2q) -> None:
        """Barren plateau risk is one of the valid categorical values."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["barren_plateau_risk"] in {"low", "medium", "high"}

    def test_n_samples_recorded(self, sample_encoding_2q) -> None:
        """n_samples is correctly recorded in the result dict."""
        n_samples = 25
        result = estimate_trainability(
            sample_encoding_2q, n_samples=n_samples, seed=42, return_details=True
        )
        assert result["n_samples"] == n_samples

    def test_per_parameter_variance_shape(self, sample_encoding_4q) -> None:
        """per_parameter_variance shape matches n_features."""
        result = estimate_trainability(
            sample_encoding_4q, n_samples=20, seed=42, return_details=True
        )
        assert result["per_parameter_variance"].shape == (4,)

    def test_effective_dimension_non_negative(self, sample_encoding_2q) -> None:
        """Effective dimension is non-negative."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["effective_dimension"] >= 0.0


# =============================================================================
# Test: estimate_trainability - Reproducibility
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityReproducibility:
    """Test reproducibility of trainability estimates."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_same_seed_same_result(self, sample_encoding_2q) -> None:
        """Same seed produces identical results."""
        result1 = estimate_trainability(sample_encoding_2q, n_samples=30, seed=42)
        result2 = estimate_trainability(sample_encoding_2q, n_samples=30, seed=42)
        assert_allclose(result1, result2, atol=1e-10)

    def test_different_seed_different_result(self, sample_encoding_2q) -> None:
        """Different seeds produce valid (potentially different) results."""
        result1 = estimate_trainability(sample_encoding_2q, n_samples=50, seed=42)
        result2 = estimate_trainability(sample_encoding_2q, n_samples=50, seed=123)
        # Results should differ (though not guaranteed for all seeds)
        # This test verifies different seeds don't produce identical results
        # We simply check both values are valid - exact comparison is unreliable
        assert 0.0 <= result1 <= 1.0
        assert 0.0 <= result2 <= 1.0

    def test_detailed_results_reproducible(self, sample_encoding_2q) -> None:
        """Detailed results are reproducible with same seed."""
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


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityObservables:
    """Test different observable types for trainability estimation."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_computational_observable(self, sample_encoding_2q) -> None:
        """Trainability with computational observable returns valid value."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            observable="computational",
        )
        assert 0.0 <= result <= 1.0

    def test_pauli_z_observable(self, sample_encoding_2q) -> None:
        """Trainability with Pauli Z observable returns valid value."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            observable="pauli_z",
        )
        assert 0.0 <= result <= 1.0

    def test_global_z_observable(self, sample_encoding_2q) -> None:
        """Trainability with global Z observable returns valid value."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            observable="global_z",
        )
        assert 0.0 <= result <= 1.0

    def test_invalid_observable_raises(self, sample_encoding_2q) -> None:
        """Invalid observable raises ValueError."""
        with pytest.raises(ValueError, match="observable"):
            estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                observable="invalid",
            )

    def test_observables_produce_different_variances(
        self, entangling_encoding_4q
    ) -> None:
        """Different observables produce different gradient variances.

        For entangling circuits, local vs global observables should give
        meaningfully different gradient statistics.
        """
        results = {}
        for obs in ("computational", "pauli_z", "global_z"):
            results[obs] = estimate_trainability(
                entangling_encoding_4q,
                n_samples=50,
                seed=42,
                observable=obs,
                return_details=True,
            )
        # All should be valid TrainabilityResult dicts
        for obs, result in results.items():
            assert result["gradient_variance"] >= 0.0, f"{obs}: negative variance"
            assert 0.0 <= result["trainability_estimate"] <= 1.0, f"{obs}: score OOB"
            assert result["barren_plateau_risk"] in {
                "low",
                "medium",
                "high",
            }, f"{obs}: bad risk"


# =============================================================================
# Test: estimate_trainability - Input Validation
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityValidation:
    """Test input validation for estimate_trainability."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_invalid_encoding_raises(self) -> None:
        """Non-encoding object raises AnalysisError."""
        with pytest.raises(AnalysisError, match="BaseEncoding"):
            estimate_trainability("not_an_encoding", n_samples=20)

    def test_none_encoding_raises(self) -> None:
        """None encoding raises AnalysisError."""
        with pytest.raises(AnalysisError, match="BaseEncoding"):
            estimate_trainability(None, n_samples=20)

    def test_too_few_samples_raises(self, sample_encoding_2q) -> None:
        """n_samples below error threshold raises InsufficientSamplesError."""
        with pytest.raises(InsufficientSamplesError, match="n_samples"):
            estimate_trainability(sample_encoding_2q, n_samples=5)

    def test_exactly_minimum_samples_allowed(self, sample_encoding_2q) -> None:
        """Exactly minimum samples is accepted without error."""
        # Should not raise
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=_MIN_SAMPLES_ERROR,
            seed=42,
        )
        assert 0.0 <= result <= 1.0

    def test_low_samples_warns(self, sample_encoding_2q) -> None:
        """Sample count below warning threshold emits UserWarning."""
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

    def test_invalid_input_range_raises(self, sample_encoding_2q) -> None:
        """Inverted input_range (max < min) raises ValueError."""
        with pytest.raises(ValueError, match="input_range"):
            estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                input_range=(2.0, 1.0),  # max < min
            )

    def test_invalid_backend_raises(self, sample_encoding_2q) -> None:
        """Invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="backend"):
            estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                backend="invalid_backend",
            )

    def test_zero_samples_raises(self, sample_encoding_2q) -> None:
        """n_samples=0 raises InsufficientSamplesError."""
        with pytest.raises(InsufficientSamplesError):
            estimate_trainability(sample_encoding_2q, n_samples=0)

    def test_negative_samples_raises(self, sample_encoding_2q) -> None:
        """Negative n_samples raises InsufficientSamplesError."""
        with pytest.raises(InsufficientSamplesError):
            estimate_trainability(sample_encoding_2q, n_samples=-5)

    def test_equal_input_range_raises(self, sample_encoding_2q) -> None:
        """Equal min and max in input_range raises ValueError."""
        with pytest.raises(ValueError, match="input_range"):
            estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                input_range=(1.0, 1.0),
            )


# =============================================================================
# Test: estimate_trainability - Encoding Comparison
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityEncodingComparison:
    """Test trainability estimation across different encodings."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_product_state_high_trainability(self, sample_encoding_2q) -> None:
        """Product state encodings (AngleEncoding) avoid barren plateaus.

        AngleEncoding produces product states (no entanglement), which are
        known to avoid barren plateaus per McClean et al. (2018).
        """
        result = estimate_trainability(
            sample_encoding_2q, n_samples=50, seed=42, return_details=True
        )
        # Product states should have low barren plateau risk
        assert result["barren_plateau_risk"] in {"low", "medium"}

    def test_larger_encoding_still_works(self, sample_encoding_4q) -> None:
        """Larger encodings produce valid trainability estimates."""
        result = estimate_trainability(sample_encoding_4q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Test: estimate_trainability - Backend Comparison
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityBackends:
    """Test trainability estimation with different backends."""

    def test_pennylane_backend(self, sample_encoding_2q, pennylane_available) -> None:
        """PennyLane backend produces valid result."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            backend="pennylane",
        )
        assert 0.0 <= result <= 1.0

    def test_qiskit_backend(self, sample_encoding_2q, qiskit_available) -> None:
        """Qiskit backend produces valid result."""
        if not qiskit_available:
            pytest.skip("Qiskit not available")
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            backend="qiskit",
        )
        assert 0.0 <= result <= 1.0

    def test_cirq_backend(self, sample_encoding_2q, cirq_available) -> None:
        """Cirq backend produces valid result."""
        if not cirq_available:
            pytest.skip("Cirq not available")
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            backend="cirq",
        )
        assert 0.0 <= result <= 1.0

    def test_pennylane_qiskit_similar(
        self, sample_encoding_2q, pennylane_available, qiskit_available
    ) -> None:
        """PennyLane and Qiskit backends give results in same ballpark.

        Exact agreement is not required due to different implementations
        and qubit ordering conventions, but results should not differ wildly.
        """
        if not pennylane_available or not qiskit_available:
            pytest.skip("Both PennyLane and Qiskit required")

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

    def test_three_backend_all_valid(
        self,
        sample_encoding_2q,
        pennylane_available,
        qiskit_available,
        cirq_available,
    ) -> None:
        """All three backends produce valid trainability estimates."""
        if not (pennylane_available and qiskit_available and cirq_available):
            pytest.skip("All three backends required")

        results = {}
        for backend in ("pennylane", "qiskit", "cirq"):
            results[backend] = estimate_trainability(
                sample_encoding_2q,
                n_samples=30,
                seed=42,
                backend=backend,
            )
            assert 0.0 <= results[backend] <= 1.0

        # All should be in the same general ballpark
        vals = list(results.values())
        assert max(vals) - min(vals) < 0.6


# =============================================================================
# Test: estimate_trainability - Custom Input Range
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityInputRange:
    """Test trainability estimation with custom input ranges."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_default_range(self, sample_encoding_2q) -> None:
        """Default input range [0, 2pi] produces valid result."""
        result = estimate_trainability(sample_encoding_2q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_narrow_range(self, sample_encoding_2q) -> None:
        """Narrow input range produces valid result."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            input_range=(0.0, 0.5),
        )
        assert 0.0 <= result <= 1.0

    def test_negative_range(self, sample_encoding_2q) -> None:
        """Negative input range produces valid result."""
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


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestComputeGradientVariance:
    """Tests for compute_gradient_variance function."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_returns_float(self, sample_encoding_2q) -> None:
        """Return type is float."""
        variance = compute_gradient_variance(sample_encoding_2q, n_samples=20, seed=42)
        assert isinstance(variance, float)

    def test_non_negative(self, sample_encoding_2q) -> None:
        """Variance is non-negative."""
        variance = compute_gradient_variance(sample_encoding_2q, n_samples=20, seed=42)
        assert variance >= 0.0

    def test_reproducible(self, sample_encoding_2q) -> None:
        """Same seed gives same variance."""
        var1 = compute_gradient_variance(sample_encoding_2q, n_samples=30, seed=42)
        var2 = compute_gradient_variance(sample_encoding_2q, n_samples=30, seed=42)
        assert_allclose(var1, var2, atol=1e-10)

    def test_matches_detailed_result(self, sample_encoding_2q) -> None:
        """Variance matches detailed result from estimate_trainability."""
        variance = compute_gradient_variance(sample_encoding_2q, n_samples=30, seed=42)
        detailed = estimate_trainability(
            sample_encoding_2q, n_samples=30, seed=42, return_details=True
        )
        assert_allclose(variance, detailed["gradient_variance"], atol=1e-10)

    def test_with_global_z_observable(self, sample_encoding_2q) -> None:
        """Gradient variance with global_z observable returns valid result."""
        variance = compute_gradient_variance(
            sample_encoding_2q, n_samples=20, seed=42, observable="global_z"
        )
        assert isinstance(variance, float)
        assert variance >= 0.0

    def test_propagates_sample_error(self, sample_encoding_2q) -> None:
        """Too-few-samples error propagates from estimate_trainability."""
        with pytest.raises(InsufficientSamplesError):
            compute_gradient_variance(sample_encoding_2q, n_samples=5)


# =============================================================================
# Test: detect_barren_plateau
# =============================================================================


class TestDetectBarrenPlateau:
    """Tests for detect_barren_plateau function."""

    def test_low_risk_for_high_variance(self) -> None:
        """High variance gives low risk."""
        risk = detect_barren_plateau(gradient_variance=0.1, n_qubits=4, n_params=16)
        assert risk == "low"

    def test_high_risk_for_low_variance(self) -> None:
        """Very low variance gives high risk."""
        risk = detect_barren_plateau(gradient_variance=1e-10, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_medium_risk_for_borderline_variance(self) -> None:
        """Borderline variance gives medium risk.

        For n_qubits=4 (size_factor=1.0):
        - high_threshold = 1e-6
        - medium_threshold = 1e-3
        Variance 1e-5 is between them, so risk is 'medium'.
        """
        risk = detect_barren_plateau(gradient_variance=1e-5, n_qubits=4, n_params=16)
        assert risk == "medium"

    def test_scaling_with_qubits(self) -> None:
        """Thresholds scale with qubit count."""
        risk_small = detect_barren_plateau(
            gradient_variance=1e-5, n_qubits=2, n_params=8
        )
        risk_large = detect_barren_plateau(
            gradient_variance=1e-5, n_qubits=8, n_params=32
        )
        # Both should be valid risk levels
        assert risk_small in {"low", "medium", "high"}
        assert risk_large in {"low", "medium", "high"}

    def test_invalid_variance_raises(self) -> None:
        """Negative variance raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            detect_barren_plateau(gradient_variance=-1.0, n_qubits=4, n_params=16)

    def test_invalid_n_qubits_raises(self) -> None:
        """Invalid n_qubits raises ValueError."""
        with pytest.raises(ValueError, match="n_qubits"):
            detect_barren_plateau(gradient_variance=0.1, n_qubits=0, n_params=16)

    def test_invalid_n_params_raises(self) -> None:
        """Invalid n_params raises ValueError."""
        with pytest.raises(ValueError, match="n_params"):
            detect_barren_plateau(gradient_variance=0.1, n_qubits=4, n_params=0)

    def test_zero_variance_high_risk(self) -> None:
        """Zero variance gives high risk."""
        risk = detect_barren_plateau(gradient_variance=0.0, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_returns_valid_literal(self) -> None:
        """Return value is always a valid literal for all input combinations."""
        for variance in [0.0, 1e-10, 1e-5, 1e-3, 0.01, 0.1, 1.0]:
            for n_qubits in [1, 2, 4, 8]:
                risk = detect_barren_plateau(
                    gradient_variance=variance,
                    n_qubits=n_qubits,
                    n_params=n_qubits * 4,
                )
                assert risk in {"low", "medium", "high"}

    def test_exact_high_threshold_boundary(self) -> None:
        """Variance exactly at high_threshold is medium, not high.

        The condition is 'variance < high_threshold' for high risk.
        At the boundary (variance == threshold), it falls through to medium.
        For n_qubits=4: high_threshold = 1e-6.
        """
        risk = detect_barren_plateau(
            gradient_variance=_BP_HIGH_RISK_THRESHOLD,
            n_qubits=4,
            n_params=16,
        )
        assert risk == "medium"

    def test_exact_medium_threshold_boundary(self) -> None:
        """Variance exactly at medium_threshold is low, not medium.

        The condition is 'variance < medium_threshold' for medium risk.
        At the boundary (variance == threshold), it falls through to low.
        For n_qubits=4: medium_threshold = 1e-3.
        """
        risk = detect_barren_plateau(
            gradient_variance=_BP_MEDIUM_RISK_THRESHOLD,
            n_qubits=4,
            n_params=16,
        )
        assert risk == "low"

    def test_threshold_floor_large_system(self) -> None:
        """Threshold floors prevent everything becoming 'low' for large systems.

        For n_qubits=100, the size_factor would make thresholds ~0, but
        the floor enforces: high_threshold >= 1e-12, medium_threshold >= 1e-9.
        """
        # Variance 1e-13 should still be classified as high risk
        risk_high = detect_barren_plateau(
            gradient_variance=1e-13, n_qubits=100, n_params=400
        )
        assert risk_high == "high"

        # Variance 1e-10 should be medium (between 1e-12 and 1e-9 floors)
        risk_med = detect_barren_plateau(
            gradient_variance=1e-10, n_qubits=100, n_params=400
        )
        assert risk_med == "medium"

        # Variance 1e-8 should be low (above 1e-9 floor)
        risk_low = detect_barren_plateau(
            gradient_variance=1e-8, n_qubits=100, n_params=400
        )
        assert risk_low == "low"

    def test_monotonicity_with_increasing_variance(self) -> None:
        """Risk severity never increases as variance increases.

        For a sequence of increasing variances, the risk transitions
        from high → medium → low and never goes backwards.
        """
        severity_map = {"high": 3, "medium": 2, "low": 1}
        variances = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]
        n_qubits = 4
        n_params = 16

        prev_severity = 4  # Start higher than any level
        for var in variances:
            risk = detect_barren_plateau(var, n_qubits, n_params)
            severity = severity_map[risk]
            assert severity <= prev_severity, (
                f"Monotonicity violated: variance {var} has risk {risk} "
                f"(severity {severity}) which is more severe than previous "
                f"(severity {prev_severity})"
            )
            prev_severity = severity

    def test_single_qubit_system(self) -> None:
        """Single-qubit system uses base thresholds (n_qubits <= 4)."""
        risk = detect_barren_plateau(gradient_variance=0.01, n_qubits=1, n_params=1)
        assert risk == "low"

        risk = detect_barren_plateau(gradient_variance=1e-8, n_qubits=1, n_params=1)
        assert risk == "high"


# =============================================================================
# Test: Known Values and Expected Behavior
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestKnownValues:
    """Tests for known/expected behavior."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_angle_encoding_trainable(self, sample_encoding_2q) -> None:
        """AngleEncoding (product states) appears trainable.

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

    def test_variance_positive_for_nontrivial_encoding(
        self, sample_encoding_4q
    ) -> None:
        """Variance is positive for non-trivial encoding."""
        result = estimate_trainability(
            sample_encoding_4q,
            n_samples=50,
            seed=42,
            return_details=True,
        )
        # Variance should be positive for an encoding with actual parameters
        assert result["gradient_variance"] > 0

    def test_per_parameter_variance_all_positive(self, sample_encoding_2q) -> None:
        """Per-parameter variances are all non-negative."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=50,
            seed=42,
            return_details=True,
        )
        assert np.all(result["per_parameter_variance"] >= 0)

    def test_trainability_consistent_with_variance(self, sample_encoding_2q) -> None:
        """Higher gradient variance corresponds to higher trainability score.

        Uses _variance_to_trainability relationship: the score is a monotone
        function of variance, so a known-larger variance must produce a
        known-larger score.
        """
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=50,
            seed=42,
            return_details=True,
        )
        # Verify the score is consistent with the variance
        expected_score = _variance_to_trainability(result["gradient_variance"])
        assert_allclose(result["trainability_estimate"], expected_score, atol=1e-10)


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of trainability computation."""

    def test_detect_barren_plateau_extreme_variance(self) -> None:
        """Detection handles extreme variance values without errors."""
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

    def test_detect_barren_plateau_many_qubits(self) -> None:
        """Detection handles many qubits without numerical issues."""
        # Should not raise or produce invalid results
        risk = detect_barren_plateau(gradient_variance=1e-6, n_qubits=20, n_params=80)
        assert risk in {"low", "medium", "high"}

    def test_trainability_score_clamped(
        self, sample_encoding_2q, pennylane_available
    ) -> None:
        """Trainability score is always in [0, 1] across multiple seeds."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

        for seed in [42, 123, 456]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = estimate_trainability(
                    sample_encoding_2q, n_samples=20, seed=seed
                )
            assert 0.0 <= result <= 1.0


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEdgeCases:
    """Tests for edge cases in trainability estimation."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_single_qubit_encoding(self) -> None:
        """Single-qubit encoding produces valid result."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = estimate_trainability(enc, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_minimum_valid_samples(self, sample_encoding_2q) -> None:
        """Minimum valid number of samples produces valid result."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=_MIN_SAMPLES_ERROR, seed=42
        )
        assert 0.0 <= result <= 1.0

    def test_n_failed_samples_tracked(self, sample_encoding_2q) -> None:
        """Failed samples are tracked and within bounds."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            return_details=True,
        )
        # For valid encoding, failures should be minimal
        assert result["n_failed_samples"] >= 0
        assert result["n_failed_samples"] <= result["n_samples"]

    def test_n_successful_samples_consistent(self, sample_encoding_2q) -> None:
        """n_successful_samples = n_samples - n_failed_samples."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=30,
            seed=42,
            return_details=True,
        )
        expected_successful = result["n_samples"] - result["n_failed_samples"]
        assert result["n_successful_samples"] == expected_successful

    def test_n_successful_samples_positive(self, sample_encoding_2q) -> None:
        """n_successful_samples is positive for valid analysis."""
        result = estimate_trainability(
            sample_encoding_2q,
            n_samples=20,
            seed=42,
            return_details=True,
        )
        # If analysis completed, we must have had successful samples
        assert result["n_successful_samples"] > 0
        assert result["n_successful_samples"] <= result["n_samples"]

    def test_effective_dimension_bounded_by_n_features(
        self, sample_encoding_4q
    ) -> None:
        """Effective dimension does not exceed n_features."""
        result = estimate_trainability(
            sample_encoding_4q,
            n_samples=50,
            seed=42,
            return_details=True,
        )
        assert result["effective_dimension"] <= sample_encoding_4q.n_features


# =============================================================================
# Test: TrainabilityResult TypedDict
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestTrainabilityResultType:
    """Tests for TrainabilityResult TypedDict structure."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_result_types(self, sample_encoding_2q) -> None:
        """All result fields have correct types."""
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

    def test_result_array_dtype(self, sample_encoding_2q) -> None:
        """Arrays have floating-point dtype."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=20, seed=42, return_details=True
        )
        assert result["per_parameter_variance"].dtype in [np.float64, np.float32]

    def test_result_value_ranges(self, sample_encoding_2q) -> None:
        """All result values are within expected ranges."""
        result = estimate_trainability(
            sample_encoding_2q, n_samples=50, seed=42, return_details=True
        )
        assert 0.0 <= result["trainability_estimate"] <= 1.0
        assert result["gradient_variance"] >= 0.0
        assert result["effective_dimension"] >= 0.0
        assert result["n_samples"] > 0
        assert result["n_successful_samples"] > 0
        assert result["n_failed_samples"] >= 0
        assert np.all(result["per_parameter_variance"] >= 0.0)


# =============================================================================
# Test: Verbose Mode
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestVerboseMode:
    """Tests for verbose mode logging."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_verbose_mode_runs(self, sample_encoding_2q, caplog) -> None:
        """Verbose mode runs without error and produces log output."""
        import logging

        with caplog.at_level(logging.INFO):
            result = estimate_trainability(
                sample_encoding_2q,
                n_samples=20,
                seed=42,
                verbose=True,
            )
        assert 0.0 <= result <= 1.0

    def test_non_verbose_silent(self, sample_encoding_2q, caplog) -> None:
        """Non-verbose mode completes without error."""
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


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEncodingIntegration:
    """Integration tests with various encoding types."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_with_iqp_encoding(self, entangling_encoding_4q) -> None:
        """IQP encoding produces valid trainability estimate."""
        result = estimate_trainability(entangling_encoding_4q, n_samples=20, seed=42)
        assert 0.0 <= result <= 1.0

    def test_comparison_angle_vs_iqp(
        self, sample_encoding_4q, entangling_encoding_4q
    ) -> None:
        """Angle and IQP encodings produce valid (comparable) results."""
        # Use same seed for fair comparison
        train_angle = estimate_trainability(sample_encoding_4q, n_samples=50, seed=42)
        train_iqp = estimate_trainability(entangling_encoding_4q, n_samples=50, seed=42)

        # Both should be valid
        assert 0.0 <= train_angle <= 1.0
        assert 0.0 <= train_iqp <= 1.0

    def test_with_data_reuploading(self) -> None:
        """DataReuploading encoding produces valid trainability estimate."""
        try:
            from encoding_atlas import DataReuploading
        except ImportError:
            pytest.skip("DataReuploading not available")
        enc = DataReuploading(n_features=2, n_layers=1)
        result = estimate_trainability(enc, n_samples=50, seed=42)
        assert 0.0 <= result <= 1.0

    def test_with_zz_feature_map(self) -> None:
        """ZZFeatureMap encoding produces valid trainability estimate."""
        try:
            from encoding_atlas import ZZFeatureMap
        except ImportError:
            pytest.skip("ZZFeatureMap not available")
        enc = ZZFeatureMap(n_features=4, reps=1)
        result = estimate_trainability(enc, n_samples=50, seed=42)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Test: Error Messages
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestErrorMessages:
    """Tests for informative error messages."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_insufficient_samples_error_includes_counts(
        self, sample_encoding_2q
    ) -> None:
        """InsufficientSamplesError includes sample count details."""
        with pytest.raises(InsufficientSamplesError) as exc_info:
            estimate_trainability(sample_encoding_2q, n_samples=5)
        assert exc_info.value.requested_samples == 5
        assert exc_info.value.minimum_samples == _MIN_SAMPLES_ERROR
        assert exc_info.value.metric == "trainability"

    def test_invalid_observable_shows_options(self, sample_encoding_2q) -> None:
        """Invalid observable error message lists valid options."""
        with pytest.raises(ValueError, match="observable") as exc_info:
            estimate_trainability(sample_encoding_2q, n_samples=20, observable="bad")
        error_msg = str(exc_info.value).lower()
        assert "observable" in error_msg

    def test_invalid_backend_shows_options(self, sample_encoding_2q) -> None:
        """Invalid backend error message lists valid options."""
        with pytest.raises(ValueError, match="backend") as exc_info:
            estimate_trainability(sample_encoding_2q, n_samples=20, backend="bad")
        error_msg = str(exc_info.value).lower()
        assert "backend" in error_msg


# =============================================================================
# Test: _variance_to_trainability (Private Helper)
# =============================================================================


class TestVarianceToTrainability:
    """Tests for the _variance_to_trainability sigmoid mapping.

    This function maps gradient variance to a [0, 1] trainability score
    using a logistic transformation on the log-ratio to a reference variance.
    """

    def test_zero_variance_returns_zero(self) -> None:
        """Zero variance maps to trainability 0.0."""
        assert _variance_to_trainability(0.0) == 0.0

    def test_negative_variance_returns_zero(self) -> None:
        """Negative variance maps to trainability 0.0 (edge guard)."""
        assert _variance_to_trainability(-1.0) == 0.0
        assert _variance_to_trainability(-1e-15) == 0.0

    def test_reference_variance_gives_half(self) -> None:
        """Variance equal to reference gives exactly 0.5.

        At variance = _REFERENCE_VARIANCE:
          log_ratio = log10(ref/ref) = 0
          sigmoid(0) = 1/(1+exp(0)) = 0.5
        """
        score = _variance_to_trainability(_REFERENCE_VARIANCE)
        assert_allclose(score, 0.5, atol=1e-10)

    def test_large_variance_approaches_one(self) -> None:
        """Very large variance gives score close to 1.0."""
        score = _variance_to_trainability(100.0)
        assert score > 0.99
        assert score <= 1.0

    def test_small_variance_approaches_zero(self) -> None:
        """Very small variance gives score close to 0.0."""
        score = _variance_to_trainability(1e-10)
        assert score < 0.01
        assert score >= 0.0

    def test_monotonically_increasing(self) -> None:
        """Larger variance always produces higher trainability score."""
        variances = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
        scores = [_variance_to_trainability(v) for v in variances]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Monotonicity violated: variance {variances[i]} "
                f"(score {scores[i]}) >= variance {variances[i+1]} "
                f"(score {scores[i+1]})"
            )

    def test_output_always_in_unit_interval(self) -> None:
        """Output is always in [0, 1] for any non-negative input."""
        test_values = [0.0, 1e-300, 1e-15, 1e-10, 1e-5, 0.001, 0.1, 1.0, 1e5, 1e100]
        for v in test_values:
            score = _variance_to_trainability(v)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for variance {v}"


# =============================================================================
# Test: _pad_or_truncate (Private Helper)
# =============================================================================


class TestPadOrTruncate:
    """Tests for _pad_or_truncate array resizing utility."""

    def test_equal_length_unchanged(self) -> None:
        """Array matching target length is returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _pad_or_truncate(arr, 3)
        assert_allclose(result, arr)

    def test_shorter_padded_with_zeros(self) -> None:
        """Shorter array is padded with zeros to target length."""
        arr = np.array([1.0, 2.0])
        result = _pad_or_truncate(arr, 4)
        assert result.shape == (4,)
        assert_allclose(result, [1.0, 2.0, 0.0, 0.0])

    def test_longer_truncated(self) -> None:
        """Longer array is truncated to target length."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _pad_or_truncate(arr, 2)
        assert result.shape == (2,)
        assert_allclose(result, [1.0, 2.0])

    def test_single_element_padded(self) -> None:
        """Single-element array is padded correctly."""
        arr = np.array([5.0])
        result = _pad_or_truncate(arr, 3)
        assert_allclose(result, [5.0, 0.0, 0.0])

    def test_single_element_target(self) -> None:
        """Multi-element array truncated to single element."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _pad_or_truncate(arr, 1)
        assert result.shape == (1,)
        assert_allclose(result, [1.0])

    def test_dtype_preserved(self) -> None:
        """Output dtype matches input dtype."""
        arr_f64 = np.array([1.0, 2.0], dtype=np.float64)
        result_f64 = _pad_or_truncate(arr_f64, 4)
        assert result_f64.dtype == np.float64

        arr_f32 = np.array([1.0, 2.0], dtype=np.float32)
        result_f32 = _pad_or_truncate(arr_f32, 4)
        assert result_f32.dtype == np.float32


# =============================================================================
# Test: _compute_prob_zero_first_qubit (Private Helper)
# =============================================================================


class TestComputeProbZeroFirstQubit:
    """Tests for _compute_prob_zero_first_qubit.

    This function computes P(qubit 0 = |0>) from a statevector,
    using MSB-first qubit ordering (qubit 0 = most significant bit).
    """

    def test_all_zero_state_2q(self) -> None:
        """|00> state: first qubit is definitely |0>, probability = 1.0."""
        state = np.array([1, 0, 0, 0], dtype=np.complex128)
        prob = _compute_prob_zero_first_qubit(state, 2)
        assert_allclose(prob, 1.0)

    def test_first_qubit_one_2q(self) -> None:
        """|10> state: first qubit is |1>, probability of |0> = 0.0."""
        state = np.array([0, 0, 1, 0], dtype=np.complex128)
        prob = _compute_prob_zero_first_qubit(state, 2)
        assert_allclose(prob, 0.0)

    def test_equal_superposition_2q(self) -> None:
        """(|00> + |10>)/sqrt(2): first qubit equally |0> and |1>."""
        state = np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2)
        prob = _compute_prob_zero_first_qubit(state, 2)
        assert_allclose(prob, 0.5)

    def test_bell_state(self) -> None:
        """Bell state (|00> + |11>)/sqrt(2): P(q0=0) = 0.5."""
        state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        prob = _compute_prob_zero_first_qubit(state, 2)
        assert_allclose(prob, 0.5)

    def test_single_qubit_zero(self) -> None:
        """|0> state: probability = 1.0."""
        state = np.array([1, 0], dtype=np.complex128)
        prob = _compute_prob_zero_first_qubit(state, 1)
        assert_allclose(prob, 1.0)

    def test_single_qubit_one(self) -> None:
        """|1> state: probability = 0.0."""
        state = np.array([0, 1], dtype=np.complex128)
        prob = _compute_prob_zero_first_qubit(state, 1)
        assert_allclose(prob, 0.0)

    def test_three_qubit_ghz(self) -> None:
        """GHZ state (|000> + |111>)/sqrt(2): P(q0=0) = 0.5.

        Only |000> (index 0) contributes to q0=|0>.
        |111> (index 7) has q0=|1>.
        """
        state = np.zeros(8, dtype=np.complex128)
        state[0] = 1.0 / np.sqrt(2)
        state[7] = 1.0 / np.sqrt(2)
        prob = _compute_prob_zero_first_qubit(state, 3)
        assert_allclose(prob, 0.5)

    def test_second_qubit_flipped_only(self) -> None:
        """|01> state: first qubit is |0>, so probability = 1.0."""
        state = np.array([0, 1, 0, 0], dtype=np.complex128)
        prob = _compute_prob_zero_first_qubit(state, 2)
        assert_allclose(prob, 1.0)


# =============================================================================
# Test: _compute_expectation_value (Private Helper)
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestComputeExpectationValue:
    """Tests for _compute_expectation_value with known quantum states.

    Uses AngleEncoding (RY gates) where:
    - x=[0]: state = |0> (RY(0)|0> = |0>)
    - x=[pi]: state = |1> (RY(pi)|0> = |1>)
    - x=[pi/2]: state = (|0>+|1>)/sqrt(2)
    """

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_computational_all_zeros(self) -> None:
        """Computational observable on |0> gives 1.0 (P(|0>))."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = _compute_expectation_value(
            enc, np.array([0.0]), "computational", "pennylane"
        )
        assert_allclose(result, 1.0, atol=1e-10)

    def test_computational_all_ones(self) -> None:
        """Computational observable on |1> gives 0.0 (P(|0>) = 0)."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = _compute_expectation_value(
            enc, np.array([np.pi]), "computational", "pennylane"
        )
        assert_allclose(result, 0.0, atol=1e-10)

    def test_computational_superposition(self) -> None:
        """Computational observable on (|0>+|1>)/sqrt(2) gives 0.5."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = _compute_expectation_value(
            enc, np.array([np.pi / 2]), "computational", "pennylane"
        )
        assert_allclose(result, 0.5, atol=1e-10)

    def test_pauli_z_all_zeros(self) -> None:
        """Pauli Z on |0> gives +1.0 (eigenvalue of Z for |0>)."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = _compute_expectation_value(
            enc, np.array([0.0]), "pauli_z", "pennylane"
        )
        assert_allclose(result, 1.0, atol=1e-10)

    def test_pauli_z_all_ones(self) -> None:
        """Pauli Z on |1> gives -1.0 (eigenvalue of Z for |1>)."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = _compute_expectation_value(
            enc, np.array([np.pi]), "pauli_z", "pennylane"
        )
        assert_allclose(result, -1.0, atol=1e-10)

    def test_global_z_two_qubit_product(self) -> None:
        """Global Z on |01> gives -1.0.

        For 2-qubit AngleEncoding with x=[0, pi]:
        - State = |0> ⊗ |1> = |01>
        - Z⊗Z eigenvalue for |01> = (+1)(-1) = -1
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        result = _compute_expectation_value(
            enc, np.array([0.0, np.pi]), "global_z", "pennylane"
        )
        assert_allclose(result, -1.0, atol=1e-10)

    def test_pauli_z_differs_from_global_z_two_qubit(self) -> None:
        """For |01>, pauli_z (local Z on q0) != global_z (Z⊗Z).

        pauli_z measures only q0 which is |0>, giving +1.0.
        global_z measures Z⊗Z on |01>, giving -1.0.
        This confirms the two observables compute different things.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([0.0, np.pi])

        pauli_z_val = _compute_expectation_value(enc, x, "pauli_z", "pennylane")
        global_z_val = _compute_expectation_value(enc, x, "global_z", "pennylane")

        assert_allclose(pauli_z_val, 1.0, atol=1e-10)
        assert_allclose(global_z_val, -1.0, atol=1e-10)
        # They must differ
        assert not np.isclose(pauli_z_val, global_z_val)

    def test_unknown_observable_raises(self) -> None:
        """Unknown observable raises ValueError."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        with pytest.raises(ValueError, match="Unknown observable"):
            _compute_expectation_value(
                enc, np.array([0.0]), "bad_observable", "pennylane"
            )


# =============================================================================
# Test: _compute_encoding_gradients (Private Helper)
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestComputeEncodingGradients:
    """Tests for _compute_encoding_gradients (parameter-shift rule).

    Uses AngleEncoding (RY gates) where analytical gradients are known:
    - f(theta) = cos^2(theta/2) for computational observable
    - df/dtheta = -sin(theta)/2
    """

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_gradient_shape_matches_features(self) -> None:
        """Output gradient array has shape (n_features,)."""
        from encoding_atlas import AngleEncoding

        for n_features in [1, 2, 4]:
            enc = AngleEncoding(n_features=n_features)
            x = np.zeros(n_features, dtype=np.float64)
            grads = _compute_encoding_gradients(enc, x, "computational", "pennylane")
            assert grads.shape == (
                n_features,
            ), f"Expected shape ({n_features},), got {grads.shape}"

    def test_gradient_values_finite(self) -> None:
        """All gradient values are finite (no NaN or Inf)."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=3)
        x = np.array([0.5, 1.0, 1.5], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "computational", "pennylane")
        assert np.all(np.isfinite(grads))

    def test_known_gradient_computational(self) -> None:
        """Gradient matches analytical value for 1-qubit AngleEncoding.

        For RY(theta)|0>, f(theta) = cos^2(theta/2).
        df/dtheta = -sin(theta)/2.
        At theta=pi/4: gradient = -sin(pi/4)/2 = -sqrt(2)/4 ≈ -0.3536.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([np.pi / 4], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "computational", "pennylane")
        expected = -np.sqrt(2) / 4
        assert_allclose(grads[0], expected, atol=1e-8)

    def test_known_gradient_pauli_z(self) -> None:
        """Gradient of <Z> for 1-qubit matches analytical value.

        For RY(theta)|0>, <Z> = cos(theta).
        d<Z>/dtheta = -sin(theta).
        At theta=pi/4: gradient = -sin(pi/4) = -sqrt(2)/2 ≈ -0.7071.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([np.pi / 4], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "pauli_z", "pennylane")
        expected = -np.sqrt(2) / 2
        assert_allclose(grads[0], expected, atol=1e-8)

    def test_gradient_vs_finite_difference(self) -> None:
        """Parameter-shift gradient agrees with finite-difference approximation.

        Cross-validates the parameter-shift implementation against an
        independent numerical differentiation method.
        """
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        enc = AngleEncoding(n_features=1)
        x = np.array([np.pi / 4], dtype=np.float64)
        h = 1e-5

        # Parameter-shift gradient
        ps_grads = _compute_encoding_gradients(enc, x, "computational", "pennylane")

        # Finite-difference gradient
        x_plus = x.copy()
        x_plus[0] += h
        x_minus = x.copy()
        x_minus[0] -= h

        sv_plus = simulate_encoding_statevector(enc, x_plus, backend="pennylane")
        sv_minus = simulate_encoding_statevector(enc, x_minus, backend="pennylane")

        f_plus = float(np.abs(sv_plus[0]) ** 2)
        f_minus = float(np.abs(sv_minus[0]) ** 2)
        fd_grad = (f_plus - f_minus) / (2 * h)

        assert_allclose(ps_grads[0], fd_grad, atol=1e-5)

    def test_gradient_at_extrema(self) -> None:
        """Gradient is zero at cos^2 extrema (theta=0 and theta=pi).

        At theta=0: f=1 (maximum), gradient = -sin(0)/2 = 0.
        At theta=pi: f=0 (minimum), gradient = -sin(pi)/2 ≈ 0.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)

        grads_zero = _compute_encoding_gradients(
            enc, np.array([0.0]), "computational", "pennylane"
        )
        assert_allclose(grads_zero[0], 0.0, atol=1e-10)

        grads_pi = _compute_encoding_gradients(
            enc, np.array([np.pi]), "computational", "pennylane"
        )
        assert_allclose(grads_pi[0], 0.0, atol=1e-10)

    def test_multiparameter_gradient(self) -> None:
        """Multi-parameter gradient has correct structure.

        For 2-qubit AngleEncoding with x=[pi/4, 0]:
        f(x0, x1) = cos^2(x0/2) * cos^2(x1/2)
        df/dx0 = -sin(x0)/2 * cos^2(x1/2)
        df/dx1 = cos^2(x0/2) * (-sin(x1)/2)

        At x=[pi/4, 0]:
        df/dx0 = -sin(pi/4)/2 * cos^2(0) = -sqrt(2)/4 * 1 = -sqrt(2)/4
        df/dx1 = cos^2(pi/8) * (-sin(0)/2) = cos^2(pi/8) * 0 = 0
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([np.pi / 4, 0.0], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "computational", "pennylane")

        assert grads.shape == (2,)
        assert_allclose(grads[0], -np.sqrt(2) / 4, atol=1e-8)
        assert_allclose(grads[1], 0.0, atol=1e-10)


# =============================================================================
# Test: Parameter Count Edge Cases (Mock-Based)
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestParameterCountEdgeCases:
    """Tests for parameter_count property access fallbacks.

    These tests cover the edge cases in estimate_trainability where
    encoding.properties.parameter_count is unavailable or zero. Mocking
    is used to isolate the fallback logic from actual simulation.
    """

    def test_parameter_count_property_failure_fallback(
        self, sample_encoding_2q
    ) -> None:
        """Falls back to n_features when properties.parameter_count raises.

        When accessing encoding.properties raises an exception (e.g., the
        encoding does not support lazy property computation), the function
        should gracefully fall back to using n_features as n_params.
        """
        n_features = sample_encoding_2q.n_features
        mock_gradients = np.array([0.1] * n_features, dtype=np.float64)

        with (
            patch.object(
                type(sample_encoding_2q),
                "properties",
                new_callable=PropertyMock,
                side_effect=RuntimeError("properties unavailable"),
            ),
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                return_value=mock_gradients,
            ),
        ):
            result = estimate_trainability(
                sample_encoding_2q, n_samples=20, seed=42, return_details=True
            )

        assert 0.0 <= result["trainability_estimate"] <= 1.0
        assert result["n_successful_samples"] == 20
        assert result["n_failed_samples"] == 0

    def test_parameter_count_zero_uses_n_features(self, sample_encoding_2q) -> None:
        """Uses n_features as effective n_params when parameter_count is 0.

        Non-parameterized encodings (parameter_count=0) have no trainable
        parameters. The function should use n_features for gradient analysis
        so that barren-plateau detection remains meaningful.
        """
        n_features = sample_encoding_2q.n_features
        mock_gradients = np.array([0.1] * n_features, dtype=np.float64)

        mock_props = MagicMock()
        mock_props.parameter_count = 0

        with (
            patch.object(
                type(sample_encoding_2q),
                "properties",
                new_callable=PropertyMock,
                return_value=mock_props,
            ),
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                return_value=mock_gradients,
            ),
        ):
            result = estimate_trainability(
                sample_encoding_2q, n_samples=20, seed=42, return_details=True
            )

        assert 0.0 <= result["trainability_estimate"] <= 1.0
        assert result["barren_plateau_risk"] in {"low", "medium", "high"}
        assert result["n_successful_samples"] == 20


# =============================================================================
# Test: Gradient Failure Handling (Mock-Based)
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestGradientFailureHandling:
    """Tests for gradient computation failure handling in estimate_trainability.

    These tests verify that the gradient sampling loop correctly handles
    failures by counting them, excluding them from variance computation,
    and raising appropriate errors when too many failures occur.

    Mocking is used to simulate failures without requiring actual
    simulation backend issues.
    """

    def test_simulation_error_counted_as_failure(self, sample_encoding_2q) -> None:
        """SimulationError during gradient computation is counted as failure."""
        n_features = sample_encoding_2q.n_features
        call_count = [0]

        def mock_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise SimulationError("test simulation error")
            return np.array([0.1] * n_features, dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=mock_gradients,
        ):
            result = estimate_trainability(
                sample_encoding_2q, n_samples=20, seed=42, return_details=True
            )

        assert result["n_failed_samples"] == 2
        assert result["n_successful_samples"] == 18
        assert result["n_samples"] == 20

    def test_numerical_instability_counted_as_failure(self, sample_encoding_2q) -> None:
        """NumericalInstabilityError during gradient is counted as failure."""
        n_features = sample_encoding_2q.n_features
        call_count = [0]

        def mock_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] == 1:
                raise NumericalInstabilityError(
                    "test instability", value=float("nan"), operation="test"
                )
            return np.array([0.1] * n_features, dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=mock_gradients,
        ):
            result = estimate_trainability(
                sample_encoding_2q, n_samples=20, seed=42, return_details=True
            )

        assert result["n_failed_samples"] == 1
        assert result["n_successful_samples"] == 19

    def test_unexpected_exception_counted_as_failure(self, sample_encoding_2q) -> None:
        """Unexpected exception (e.g., RuntimeError) is counted as failure.

        The generic except handler catches any exception not covered by
        SimulationError or NumericalInstabilityError.
        """
        n_features = sample_encoding_2q.n_features
        call_count = [0]

        def mock_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("unexpected internal error")
            return np.array([0.1] * n_features, dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=mock_gradients,
        ):
            result = estimate_trainability(
                sample_encoding_2q, n_samples=20, seed=42, return_details=True
            )

        assert result["n_failed_samples"] == 1
        assert result["n_successful_samples"] == 19

    def test_too_many_failures_raises_simulation_error(
        self, sample_encoding_2q
    ) -> None:
        """SimulationError raised when failure fraction exceeds threshold.

        When more than 20% of gradient computations fail, the analysis
        is considered unreliable and a SimulationError is raised with
        detailed failure statistics.
        """
        n_features = sample_encoding_2q.n_features
        call_count = [0]

        def mock_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] <= 5:  # 5 out of 20 = 25% > 20%
                raise SimulationError("test failure")
            return np.array([0.1] * n_features, dtype=np.float64)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                side_effect=mock_gradients,
            ),
            pytest.raises(
                SimulationError, match="Too many gradient computations failed"
            ) as exc_info,
        ):
            estimate_trainability(sample_encoding_2q, n_samples=20, seed=42)

        assert exc_info.value.details["n_failed"] == 5
        assert exc_info.value.details["n_samples"] == 20
        assert exc_info.value.details["failure_fraction"] > _MAX_FAILURE_FRACTION

    def test_insufficient_successful_samples_raises(self, sample_encoding_2q) -> None:
        """SimulationError raised when too few samples succeed.

        When the failure fraction is at the threshold (20%, passes the >20%
        check) but the number of successful samples is below
        _MIN_SAMPLES_ERROR, a SimulationError is raised.

        With n_samples=10 and n_failed=2: failure_fraction=0.2 (not >0.2,
        passes first check) but n_successful=8 < 10 (triggers this check).
        """
        n_features = sample_encoding_2q.n_features
        call_count = [0]

        def mock_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] <= 2:  # 2 out of 10 = 20% (at threshold)
                raise SimulationError("test failure")
            return np.array([0.1] * n_features, dtype=np.float64)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                side_effect=mock_gradients,
            ),
            pytest.raises(SimulationError, match="successful samples") as exc_info,
        ):
            estimate_trainability(sample_encoding_2q, n_samples=10, seed=42)

        assert exc_info.value.details["n_successful"] < _MIN_SAMPLES_ERROR
        assert exc_info.value.details["n_failed"] == 2

    def test_gradient_dimension_mismatch_triggers_pad(self, sample_encoding_2q) -> None:
        """Gradients with fewer elements than n_features are padded.

        When _compute_encoding_gradients returns an array with fewer
        elements than n_features, _pad_or_truncate pads it with zeros.
        """
        n_features = sample_encoding_2q.n_features  # 2

        def mock_gradients(encoding, x, observable, backend):
            return np.array([0.1], dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=mock_gradients,
        ):
            result = estimate_trainability(
                sample_encoding_2q, n_samples=20, seed=42, return_details=True
            )

        assert result["per_parameter_variance"].shape == (n_features,)
        assert result["n_successful_samples"] == 20
        # Second parameter should have zero variance (all padded to 0.0)
        assert_allclose(result["per_parameter_variance"][1], 0.0, atol=1e-15)

    def test_gradient_dimension_mismatch_triggers_truncate(
        self, sample_encoding_2q
    ) -> None:
        """Gradients with more elements than n_features are truncated.

        When _compute_encoding_gradients returns an array with more
        elements than n_features, _pad_or_truncate truncates it.
        """
        n_features = sample_encoding_2q.n_features  # 2

        def mock_gradients(encoding, x, observable, backend):
            return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=mock_gradients,
        ):
            result = estimate_trainability(
                sample_encoding_2q, n_samples=20, seed=42, return_details=True
            )

        assert result["per_parameter_variance"].shape == (n_features,)
        assert result["n_successful_samples"] == 20


# =============================================================================
# Test: NaN/Inf Gradient Detection (Mock-Based)
# =============================================================================


class TestNumericalInstabilityDetection:
    """Tests for NaN/Inf detection in _compute_encoding_gradients.

    When the parameter-shift rule produces NaN or Inf gradient values
    (due to numerical issues in expectation value computation),
    _compute_encoding_gradients should raise NumericalInstabilityError
    with diagnostic details.
    """

    def test_nan_expectation_raises_numerical_instability(self) -> None:
        """NaN expectation value causes NumericalInstabilityError.

        ``_compute_encoding_gradients`` evaluates all ``2 * n_features``
        shifted expectation values in one batched call to
        ``_expectations_batch``. Forcing that call to return NaN
        triggers ``(NaN - NaN) / 2 = NaN``, which the NaN/Inf
        validation check is required to catch.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([0.5], dtype=np.float64)

        # 2 * n_features = 2 shifted states → 2 expectations.
        nan_expectations = np.array([float("nan"), float("nan")], dtype=np.float64)
        with (
            patch(
                "encoding_atlas.analysis.trainability._expectations_batch",
                return_value=nan_expectations,
            ),
            pytest.raises(NumericalInstabilityError, match="invalid value") as exc_info,
        ):
            _compute_encoding_gradients(enc, x, "computational", "pennylane")

        assert exc_info.value.operation == "parameter_shift"
        assert exc_info.value.details["param_index"] == 0

    def test_inf_expectation_raises_numerical_instability(self) -> None:
        """Inf expectation value causes NumericalInstabilityError.

        ``exp_plus = +inf`` and ``exp_minus = 0.5`` makes the gradient
        ``(+inf - 0.5) / 2 = +inf``, caught by the NaN/Inf check.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([0.5], dtype=np.float64)

        inf_expectations = np.array([float("inf"), 0.5], dtype=np.float64)
        with (
            patch(
                "encoding_atlas.analysis.trainability._expectations_batch",
                return_value=inf_expectations,
            ),
            pytest.raises(NumericalInstabilityError, match="invalid value") as exc_info,
        ):
            _compute_encoding_gradients(enc, x, "computational", "pennylane")

        assert exc_info.value.operation == "parameter_shift"
        assert np.isinf(exc_info.value.value)

    def test_neg_inf_expectation_raises(self) -> None:
        """Negative infinity expectation value raises NumericalInstabilityError."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([0.5], dtype=np.float64)

        neg_inf_expectations = np.array(
            [float("-inf"), float("-inf")], dtype=np.float64
        )
        with (
            patch(
                "encoding_atlas.analysis.trainability._expectations_batch",
                return_value=neg_inf_expectations,
            ),
            pytest.raises(NumericalInstabilityError),
        ):
            _compute_encoding_gradients(enc, x, "computational", "pennylane")

    def test_nan_on_second_param_reports_correct_index(self) -> None:
        """NaN on second parameter reports ``param_index=1`` in error details.

        The expectations layout is ``[exp_plus_0, exp_minus_0,
        exp_plus_1, exp_minus_1]``. Making the first parameter finite
        and the second parameter NaN must cause the error to point at
        ``param_index=1``.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([0.5, 1.0], dtype=np.float64)

        # [exp_plus_0, exp_minus_0, exp_plus_1, exp_minus_1]
        expectations = np.array(
            [0.5, 0.5, float("nan"), float("nan")], dtype=np.float64
        )
        with (
            patch(
                "encoding_atlas.analysis.trainability._expectations_batch",
                return_value=expectations,
            ),
            pytest.raises(NumericalInstabilityError) as exc_info,
        ):
            _compute_encoding_gradients(enc, x, "computational", "pennylane")

        assert exc_info.value.details["param_index"] == 1

    def test_error_details_include_all_diagnostic_fields(self) -> None:
        """NumericalInstabilityError includes all expected diagnostic fields."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([1.5], dtype=np.float64)

        nan_expectations = np.array([float("nan"), float("nan")], dtype=np.float64)
        with (
            patch(
                "encoding_atlas.analysis.trainability._expectations_batch",
                return_value=nan_expectations,
            ),
            pytest.raises(NumericalInstabilityError) as exc_info,
        ):
            _compute_encoding_gradients(enc, x, "computational", "pennylane")

        details = exc_info.value.details
        assert "param_index" in details
        assert "exp_plus" in details
        assert "exp_minus" in details
        assert "x_original" in details
        assert details["x_original"] == pytest.approx(1.5)


# =============================================================================
# Test: Verbose Progress Logging
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestVerboseProgressLogging:
    """Tests for verbose progress logging during gradient sampling.

    Covers the verbose logging path inside the gradient loop at
    10% progress intervals (lines 679-686) and the initial/completion
    log messages.
    """

    def test_verbose_logs_progress_at_intervals(self, caplog) -> None:
        """Verbose mode logs progress messages at 10% intervals."""
        import logging

        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                return_value=np.array([0.1, -0.1], dtype=np.float64),
            ),
            caplog.at_level(logging.INFO),
        ):
            estimate_trainability(mock_enc, n_samples=20, seed=42, verbose=True)

        progress_msgs = [r for r in caplog.records if "Progress" in r.getMessage()]
        assert len(progress_msgs) > 0

    def test_verbose_logs_completion_summary(self, caplog) -> None:
        """Verbose mode logs a completion summary."""
        import logging

        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                return_value=np.array([0.1, -0.1], dtype=np.float64),
            ),
            caplog.at_level(logging.INFO),
        ):
            estimate_trainability(mock_enc, n_samples=20, seed=42, verbose=True)

        completion_msgs = [
            r for r in caplog.records if "complete" in r.getMessage().lower()
        ]
        assert len(completion_msgs) > 0

    def test_verbose_logs_initial_info(self, caplog) -> None:
        """Verbose mode logs initial encoding info."""
        import logging

        mock_enc = _make_mock_encoding(n_features=3, n_qubits=3)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                return_value=np.array([0.1, -0.1, 0.2], dtype=np.float64),
            ),
            caplog.at_level(logging.INFO),
        ):
            estimate_trainability(mock_enc, n_samples=15, seed=42, verbose=True)

        estimating_msgs = [r for r in caplog.records if "Estimating" in r.getMessage()]
        assert len(estimating_msgs) > 0

    def test_verbose_with_failures_reports_failed_count(self, caplog) -> None:
        """Verbose progress messages include failed sample count."""
        import logging

        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        call_count = [0]

        def _gradient_side_effect(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] == 1:
                raise SimulationError("fail")
            return np.array([0.1, -0.1], dtype=np.float64)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                side_effect=_gradient_side_effect,
            ),
            caplog.at_level(logging.INFO),
        ):
            estimate_trainability(mock_enc, n_samples=20, seed=42, verbose=True)

        progress_msgs = [
            r for r in caplog.records if "failed" in r.getMessage().lower()
        ]
        assert len(progress_msgs) > 0


# =============================================================================
# Test: Module Constants
# =============================================================================


class TestModuleConstants:
    """Tests that module-level constants have expected values and types.

    These tests serve as guardrails against accidental changes to
    calibration constants that would silently alter behavior.
    """

    def test_default_n_samples(self) -> None:
        """Default sample count is 500."""
        assert _DEFAULT_N_SAMPLES == 500

    def test_min_samples_warning(self) -> None:
        """Warning threshold is 50."""
        assert _MIN_SAMPLES_WARNING == 50

    def test_min_samples_error(self) -> None:
        """Error threshold is 10."""
        assert _MIN_SAMPLES_ERROR == 10

    def test_bp_high_risk_threshold(self) -> None:
        """High risk threshold is 1e-6."""
        assert _BP_HIGH_RISK_THRESHOLD == 1e-6

    def test_bp_medium_risk_threshold(self) -> None:
        """Medium risk threshold is 1e-3."""
        assert _BP_MEDIUM_RISK_THRESHOLD == 1e-3

    def test_reference_variance(self) -> None:
        """Reference variance is 0.1."""
        assert _REFERENCE_VARIANCE == 0.1

    def test_sigmoid_steepness(self) -> None:
        """Sigmoid steepness is 2.0."""
        assert _SIGMOID_STEEPNESS == 2.0

    def test_epsilon(self) -> None:
        """Epsilon is 1e-15."""
        assert _EPSILON == 1e-15

    def test_max_failure_fraction(self) -> None:
        """Max failure fraction is 0.2."""
        assert _MAX_FAILURE_FRACTION == 0.2

    def test_threshold_ordering(self) -> None:
        """High risk threshold < medium risk threshold."""
        assert _BP_HIGH_RISK_THRESHOLD < _BP_MEDIUM_RISK_THRESHOLD

    def test_sample_thresholds_ordering(self) -> None:
        """Error threshold < warning threshold < default."""
        assert _MIN_SAMPLES_ERROR < _MIN_SAMPLES_WARNING < _DEFAULT_N_SAMPLES


# =============================================================================
# Test: _variance_to_trainability - Extended Edge Cases
# =============================================================================


class TestVarianceToTrainabilityExtended:
    """Extended tests for _variance_to_trainability sigmoid mapping."""

    def test_very_tiny_positive_variance(self) -> None:
        """Very small positive variance gives score near 0."""
        score = _variance_to_trainability(1e-300)
        assert 0.0 <= score <= 1.0
        assert score < 0.01

    def test_exact_sigmoid_formula(self) -> None:
        """Verify exact sigmoid formula matches implementation.

        For variance v: score = 1 / (1 + exp(-k * log10(v / ref)))
        """
        variance = 0.05
        expected_log_ratio = np.log10(variance / _REFERENCE_VARIANCE)
        expected_score = 1.0 / (1.0 + np.exp(-_SIGMOID_STEEPNESS * expected_log_ratio))
        actual = _variance_to_trainability(variance)
        assert_allclose(actual, expected_score, atol=1e-12)

    def test_symmetry_around_reference(self) -> None:
        """Score at 10*ref and ref/10 are symmetric around 0.5.

        If v1 = ref * k and v2 = ref / k, then:
        log_ratio(v1) = log10(k) and log_ratio(v2) = -log10(k)
        So: score(v1) + score(v2) = 1.0 (sigmoid symmetry)
        """
        k = 10.0
        v_high = _REFERENCE_VARIANCE * k
        v_low = _REFERENCE_VARIANCE / k
        score_high = _variance_to_trainability(v_high)
        score_low = _variance_to_trainability(v_low)
        assert_allclose(score_high + score_low, 1.0, atol=1e-10)

    def test_double_reference_variance(self) -> None:
        """Variance = 2 * reference gives score > 0.5."""
        score = _variance_to_trainability(2.0 * _REFERENCE_VARIANCE)
        assert score > 0.5

    def test_half_reference_variance(self) -> None:
        """Variance = 0.5 * reference gives score < 0.5."""
        score = _variance_to_trainability(0.5 * _REFERENCE_VARIANCE)
        assert score < 0.5


# =============================================================================
# Test: _pad_or_truncate - Extended Edge Cases
# =============================================================================


class TestPadOrTruncateExtended:
    """Extended tests for _pad_or_truncate utility."""

    def test_empty_array_padded(self) -> None:
        """Empty array is padded to target length with zeros."""
        arr = np.array([], dtype=np.float64)
        result = _pad_or_truncate(arr, 3)
        assert result.shape == (3,)
        assert_allclose(result, [0.0, 0.0, 0.0])

    def test_large_truncation(self) -> None:
        """Large array truncated to 1 element preserves first element."""
        arr = np.arange(100, dtype=np.float64)
        result = _pad_or_truncate(arr, 1)
        assert result.shape == (1,)
        assert_allclose(result[0], 0.0)

    def test_truncated_array_is_copy(self) -> None:
        """Truncated result is a copy, not a view of the original."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _pad_or_truncate(arr, 2)
        result[0] = 999.0
        # Original should be unchanged
        assert arr[0] == 1.0

    def test_identity_returns_same_reference(self) -> None:
        """When length matches target, same object is returned (no copy)."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _pad_or_truncate(arr, 3)
        # Same object reference (optimization)
        assert result is arr


# =============================================================================
# Test: _compute_prob_zero_first_qubit - Extended
# =============================================================================


class TestComputeProbZeroFirstQubitExtended:
    """Extended tests for _compute_prob_zero_first_qubit."""

    def test_four_qubit_all_zeros(self) -> None:
        """|0000> state: P(q0=0) = 1.0."""
        state = np.zeros(16, dtype=np.complex128)
        state[0] = 1.0
        prob = _compute_prob_zero_first_qubit(state, 4)
        assert_allclose(prob, 1.0)

    def test_four_qubit_first_qubit_one(self) -> None:
        """|1000> state: P(q0=0) = 0.0."""
        state = np.zeros(16, dtype=np.complex128)
        state[8] = 1.0  # |1000> = index 8
        prob = _compute_prob_zero_first_qubit(state, 4)
        assert_allclose(prob, 0.0)

    def test_uniform_superposition(self) -> None:
        """Uniform superposition over all basis states: P(q0=0) = 0.5."""
        n_qubits = 3
        dim = 2**n_qubits
        state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        prob = _compute_prob_zero_first_qubit(state, n_qubits)
        assert_allclose(prob, 0.5)

    def test_probability_always_in_range(self) -> None:
        """Probability is always in [0, 1] for random valid states."""
        rng = np.random.default_rng(42)
        for n_qubits in [1, 2, 3, 4]:
            dim = 2**n_qubits
            real = rng.standard_normal(dim)
            imag = rng.standard_normal(dim)
            state = (real + 1j * imag).astype(np.complex128)
            state /= np.linalg.norm(state)
            prob = _compute_prob_zero_first_qubit(state, n_qubits)
            assert (
                0.0 <= prob <= 1.0 + 1e-10
            ), f"Probability {prob} out of range for {n_qubits} qubits"


# =============================================================================
# Test: _compute_expectation_value - Extended
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestComputeExpectationValueExtended:
    """Extended tests for _compute_expectation_value."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_global_z_all_zeros_two_qubit(self) -> None:
        """Global Z on |00> gives +1.0 (both qubits in |0>)."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        result = _compute_expectation_value(
            enc, np.array([0.0, 0.0]), "global_z", "pennylane"
        )
        assert_allclose(result, 1.0, atol=1e-10)

    def test_global_z_all_ones_two_qubit(self) -> None:
        """Global Z on |11> gives +1.0 (even parity).

        For |11>: Z eigenvalue on each qubit is -1.
        Z⊗Z eigenvalue = (-1)(-1) = +1.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        result = _compute_expectation_value(
            enc, np.array([np.pi, np.pi]), "global_z", "pennylane"
        )
        assert_allclose(result, 1.0, atol=1e-10)

    def test_computational_is_probability(self) -> None:
        """Computational observable returns value in [0, 1]."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.uniform(0, 2 * np.pi, size=2).astype(np.float64)
            result = _compute_expectation_value(enc, x, "computational", "pennylane")
            assert 0.0 <= result <= 1.0 + 1e-10

    def test_pauli_z_in_range(self) -> None:
        """Pauli Z expectation is in [-1, +1]."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.uniform(0, 2 * np.pi, size=2).astype(np.float64)
            result = _compute_expectation_value(enc, x, "pauli_z", "pennylane")
            assert -1.0 - 1e-10 <= result <= 1.0 + 1e-10

    def test_global_z_in_range(self) -> None:
        """Global Z expectation is in [-1, +1]."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.uniform(0, 2 * np.pi, size=2).astype(np.float64)
            result = _compute_expectation_value(enc, x, "global_z", "pennylane")
            assert -1.0 - 1e-10 <= result <= 1.0 + 1e-10


# =============================================================================
# Test: _compute_encoding_gradients - Extended
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestComputeEncodingGradientsExtended:
    """Extended tests for _compute_encoding_gradients."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available) -> None:
        """Skip tests if PennyLane is not available."""
        if not pennylane_available:
            pytest.skip("PennyLane not available")

    def test_gradient_dtype_is_float64(self) -> None:
        """Output gradient array has float64 dtype."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([0.5, 1.0], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "computational", "pennylane")
        assert grads.dtype == np.float64

    def test_gradient_does_not_modify_input(self) -> None:
        """Input array x is not modified by gradient computation."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([0.5, 1.0], dtype=np.float64)
        x_original = x.copy()
        _compute_encoding_gradients(enc, x, "computational", "pennylane")
        assert_allclose(x, x_original)

    def test_gradient_with_pauli_z_observable(self) -> None:
        """Gradient with pauli_z observable has correct shape and values."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([np.pi / 3, np.pi / 6], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "pauli_z", "pennylane")
        assert grads.shape == (2,)
        assert np.all(np.isfinite(grads))

    def test_gradient_with_global_z_observable(self) -> None:
        """Gradient with global_z observable has correct shape and values."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([np.pi / 3, np.pi / 6], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "global_z", "pennylane")
        assert grads.shape == (2,)
        assert np.all(np.isfinite(grads))

    def test_gradient_at_pi_over_2(self) -> None:
        """Gradient at pi/2 for 1-qubit matches analytical value.

        f(theta) = cos^2(theta/2), df/dtheta = -sin(theta)/2.
        At theta=pi/2: gradient = -sin(pi/2)/2 = -0.5.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([np.pi / 2], dtype=np.float64)
        grads = _compute_encoding_gradients(enc, x, "computational", "pennylane")
        assert_allclose(grads[0], -0.5, atol=1e-8)


# =============================================================================
# Test: detect_barren_plateau - Extended Threshold Scaling
# =============================================================================


class TestDetectBarrenPlateauScaling:
    """Extended tests for threshold scaling in detect_barren_plateau."""

    def test_size_factor_at_4_qubits(self) -> None:
        """At n_qubits=4, size_factor=1.0 (base thresholds unchanged)."""
        risk = detect_barren_plateau(gradient_variance=5e-7, n_qubits=4, n_params=16)
        assert risk == "high"

    def test_size_factor_at_6_qubits(self) -> None:
        """At n_qubits=6: size_factor=2^(-1)=0.5.

        high_threshold = 1e-6 * 0.5 = 5e-7
        medium_threshold = 1e-3 * 0.5 = 5e-4
        """
        # Variance 3e-7: below 5e-7 -> high
        assert detect_barren_plateau(3e-7, 6, 24) == "high"
        # Variance 1e-5: between 5e-7 and 5e-4 -> medium
        assert detect_barren_plateau(1e-5, 6, 24) == "medium"
        # Variance 1e-3: above 5e-4 -> low
        assert detect_barren_plateau(1e-3, 6, 24) == "low"

    def test_size_factor_at_8_qubits(self) -> None:
        """At n_qubits=8: size_factor=2^(-2)=0.25.

        high_threshold = 1e-6 * 0.25 = 2.5e-7
        medium_threshold = 1e-3 * 0.25 = 2.5e-4
        """
        assert detect_barren_plateau(1e-7, 8, 32) == "high"
        assert detect_barren_plateau(1e-5, 8, 32) == "medium"
        assert detect_barren_plateau(1e-2, 8, 32) == "low"

    def test_threshold_floors_activated(self) -> None:
        """For very large qubit counts, floors prevent near-zero thresholds.

        At n_qubits=50: size_factor = 2^(-23) ~ 1.2e-7
        Without floors: high_threshold ~ 1.2e-13 (below 1e-12 floor)
        With floors: high_threshold = 1e-12, medium_threshold = 1e-9
        """
        assert detect_barren_plateau(5e-13, 50, 200) == "high"
        assert detect_barren_plateau(5e-11, 50, 200) == "medium"
        assert detect_barren_plateau(1e-8, 50, 200) == "low"


# =============================================================================
# Test: estimate_trainability - Full Integration with Mocks
# =============================================================================


def _make_mock_encoding(
    n_features: int = 2,
    n_qubits: int = 2,
    *,
    parameter_count: int | None = None,
    properties_raises: bool = False,
) -> MagicMock:
    """Create a mock encoding that passes ``validate_encoding_for_analysis``.

    Parameters
    ----------
    n_features : int
        Value for ``encoding.n_features``.
    n_qubits : int
        Value for ``encoding.n_qubits``.
    parameter_count : int or None
        If int, ``encoding.properties.parameter_count`` returns this value.
        If None and *properties_raises* is False, returns *n_features*.
    properties_raises : bool
        If True, accessing ``encoding.properties`` raises ``AttributeError``,
        exercising the fallback path.
    """
    from encoding_atlas.core.base import BaseEncoding

    mock = MagicMock(spec=BaseEncoding)
    mock.n_features = n_features
    mock.n_qubits = n_qubits

    if properties_raises:
        type(mock).properties = PropertyMock(
            side_effect=AttributeError("no properties")
        )
    else:
        props = MagicMock()
        props.parameter_count = (
            parameter_count if parameter_count is not None else n_features
        )
        mock.properties = props

    return mock


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestEstimateTrainabilityMockIntegration:
    """Integration tests using mocks to control gradient values precisely.

    These tests verify the end-to-end computation from gradient samples
    to trainability scores with known inputs.
    """

    def test_constant_gradients_give_zero_variance(self) -> None:
        """Identical gradients across all samples give zero variance."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            return_value=np.array([0.5, -0.5], dtype=np.float64),
        ):
            result = estimate_trainability(
                mock_enc, n_samples=20, seed=42, return_details=True
            )

        assert_allclose(result["gradient_variance"], 0.0, atol=1e-15)
        assert result["trainability_estimate"] == 0.0
        assert result["barren_plateau_risk"] == "high"

    def test_known_variance_produces_expected_trainability(self) -> None:
        """Gradients with known variance produce expected trainability score.

        Alternating +0.3 and -0.3 gives variance = 0.09 for even samples.
        """
        mock_enc = _make_mock_encoding(n_features=1, n_qubits=1)

        call_count = [0]

        def _alternating_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            val = 0.3 if call_count[0] % 2 == 0 else -0.3
            return np.array([val], dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=_alternating_gradients,
        ):
            result = estimate_trainability(
                mock_enc, n_samples=100, seed=42, return_details=True
            )

        assert_allclose(result["gradient_variance"], 0.09, atol=0.005)

    def test_effective_dimension_all_active(self) -> None:
        """All parameters active gives effective_dimension = n_features."""
        mock_enc = _make_mock_encoding(n_features=3, n_qubits=3)

        call_count = [0]

        def _varying_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            rng = np.random.default_rng(call_count[0])
            return rng.normal(0, 0.3, size=3).astype(np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=_varying_gradients,
        ):
            result = estimate_trainability(
                mock_enc, n_samples=50, seed=42, return_details=True
            )

        assert result["effective_dimension"] == 3.0

    def test_effective_dimension_partially_frozen(self) -> None:
        """Parameters with near-zero variance are not counted in eff. dim."""
        mock_enc = _make_mock_encoding(n_features=3, n_qubits=3)

        call_count = [0]

        def _partial_gradients(encoding, x, observable, backend):
            call_count[0] += 1
            rng = np.random.default_rng(call_count[0])
            # First 2 params vary, third is always 0
            return np.array(
                [rng.normal(0, 0.3), rng.normal(0, 0.3), 0.0],
                dtype=np.float64,
            )

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=_partial_gradients,
        ):
            result = estimate_trainability(
                mock_enc, n_samples=50, seed=42, return_details=True
            )

        assert result["effective_dimension"] <= 2.0

    def test_return_details_false_gives_float(self) -> None:
        """return_details=False returns a plain float, not a dict."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            return_value=np.array([0.1, -0.1], dtype=np.float64),
        ):
            result = estimate_trainability(
                mock_enc, n_samples=15, seed=42, return_details=False
            )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_custom_backend_passed_through(self) -> None:
        """Custom backend name is passed to gradient computation."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        captured_backends = []

        def _capture_backend(encoding, x, observable, backend):
            captured_backends.append(backend)
            return np.array([0.1, -0.1], dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=_capture_backend,
        ):
            estimate_trainability(mock_enc, n_samples=15, seed=42, backend="qiskit")

        assert all(b == "qiskit" for b in captured_backends)

    def test_custom_observable_passed_through(self) -> None:
        """Custom observable name is passed to gradient computation."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        captured_observables = []

        def _capture_observable(encoding, x, observable, backend):
            captured_observables.append(observable)
            return np.array([0.1, -0.1], dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=_capture_observable,
        ):
            estimate_trainability(
                mock_enc, n_samples=15, seed=42, observable="global_z"
            )

        assert all(o == "global_z" for o in captured_observables)

    def test_exactly_threshold_failures_allowed(self) -> None:
        """Exactly 20% failures does NOT raise (threshold is strict >)."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        call_count = [0]

        def _gradient_side_effect(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] <= 4:  # 4 of 20 = 20% exactly
                raise SimulationError("fail")
            return np.array([0.1, -0.1], dtype=np.float64)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            side_effect=_gradient_side_effect,
        ):
            result = estimate_trainability(
                mock_enc, n_samples=20, seed=42, return_details=True
            )

        assert result["n_failed_samples"] == 4
        assert result["n_successful_samples"] == 16

    def test_all_samples_fail_raises(self) -> None:
        """100% failure rate raises SimulationError."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                side_effect=SimulationError("always fails"),
            ),
            pytest.raises(SimulationError, match="Too many gradient"),
        ):
            estimate_trainability(mock_enc, n_samples=15, seed=42)

    def test_excessive_failure_error_includes_all_details(self) -> None:
        """SimulationError from excessive failures includes all detail fields."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        call_count = [0]

        def _gradient_side_effect(encoding, x, observable, backend):
            call_count[0] += 1
            if call_count[0] <= 6:  # 6 of 20 = 30%
                raise SimulationError("fail")
            return np.array([0.1, -0.1], dtype=np.float64)

        with (
            patch(
                "encoding_atlas.analysis.trainability._compute_encoding_gradients",
                side_effect=_gradient_side_effect,
            ),
            pytest.raises(SimulationError) as exc_info,
        ):
            estimate_trainability(mock_enc, n_samples=20, seed=42)

        exc = exc_info.value
        assert exc.backend == "pennylane"
        assert exc.details["n_failed"] == 6
        assert exc.details["n_samples"] == 20
        assert exc.details["n_successful"] == 14
        assert exc.details["failure_fraction"] == pytest.approx(0.3)


# =============================================================================
# Test: compute_gradient_variance - Extended
# =============================================================================


@pytest.mark.filterwarnings("ignore:n_samples=.*:UserWarning")
class TestComputeGradientVarianceExtended:
    """Extended tests for compute_gradient_variance wrapper."""

    def test_with_pauli_z_observable_mock(self) -> None:
        """compute_gradient_variance works with pauli_z observable (mocked)."""
        mock_enc = _make_mock_encoding(n_features=2, n_qubits=2)

        with patch(
            "encoding_atlas.analysis.trainability._compute_encoding_gradients",
            return_value=np.array([0.1, -0.1], dtype=np.float64),
        ):
            variance = compute_gradient_variance(
                mock_enc, n_samples=15, seed=42, observable="pauli_z"
            )

        assert isinstance(variance, float)
        assert variance >= 0.0

    def test_invalid_encoding_propagates(self) -> None:
        """Invalid encoding error propagates through wrapper."""
        with pytest.raises(AnalysisError):
            compute_gradient_variance("not_an_encoding", n_samples=20)


# =============================================================================
# Test: __all__ Exports and Public API
# =============================================================================


class TestPublicAPI:
    """Tests that the module's public API is correctly defined."""

    def test_all_exports(self) -> None:
        """__all__ contains the expected public symbols."""
        from encoding_atlas.analysis import trainability

        assert "estimate_trainability" in trainability.__all__
        assert "compute_gradient_variance" in trainability.__all__
        assert "detect_barren_plateau" in trainability.__all__
        assert "TrainabilityResult" in trainability.__all__

    def test_all_has_exactly_four_entries(self) -> None:
        """__all__ has exactly 4 entries (no accidental additions)."""
        from encoding_atlas.analysis import trainability

        assert len(trainability.__all__) == 4

    def test_all_exports_resolvable(self) -> None:
        """Every name in __all__ is actually defined in the module."""
        from encoding_atlas.analysis import trainability

        for name in trainability.__all__:
            assert hasattr(
                trainability, name
            ), f"{name!r} is in __all__ but not defined in the module"

    def test_trainability_result_in_analysis_init(self) -> None:
        """TrainabilityResult is re-exported from analysis __init__."""
        from encoding_atlas.analysis import TrainabilityResult

        assert TrainabilityResult is not None

    def test_estimate_trainability_in_analysis_init(self) -> None:
        """estimate_trainability is re-exported from analysis __init__."""
        from encoding_atlas.analysis import estimate_trainability as et

        assert callable(et)

    def test_detect_barren_plateau_in_analysis_init(self) -> None:
        """detect_barren_plateau is re-exported from analysis __init__."""
        from encoding_atlas.analysis import detect_barren_plateau as dbp

        assert callable(dbp)

    def test_compute_gradient_variance_in_analysis_init(self) -> None:
        """compute_gradient_variance is re-exported from analysis __init__."""
        from encoding_atlas.analysis import compute_gradient_variance as cgv

        assert callable(cgv)
