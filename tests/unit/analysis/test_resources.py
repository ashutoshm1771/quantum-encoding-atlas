"""Comprehensive tests for encoding_atlas.analysis.resources module.

This test suite covers all public functions in the resources module:
- count_resources
- get_resource_summary
- get_gate_breakdown
- compare_resources
- estimate_execution_time

Tests cover:
- Normal operation with all 16 encoding types
- Edge cases (empty inputs, zero values, single features, boundary values)
- Error handling (invalid inputs, missing required parameters, invalid metrics)
- Data-dependent encodings (BasisEncoding)
- Protocol-based resource counting (ResourceAnalyzable)
- Properties fallback path (non-protocol encodings)
- Parameter validation (parallelization_factor, gate times, metric names)
- Cross-function consistency (summary vs detailed, count vs breakdown)
- Type safety and return value structure
- Mathematical correctness of derived metrics
- cnot/cx alias behavior
"""

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from encoding_atlas.analysis.resources import (
    _safe_divide,
    _validate_encoding,
    _validate_input_data,
    compare_resources,
    count_resources,
    estimate_execution_time,
    get_gate_breakdown,
    get_resource_summary,
)
from encoding_atlas.core.exceptions import AnalysisError, ValidationError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def angle_encoding_2q():
    """Create AngleEncoding with 2 features."""
    from encoding_atlas import AngleEncoding

    return AngleEncoding(n_features=2)


@pytest.fixture
def angle_encoding_4q():
    """Create AngleEncoding with 4 features."""
    from encoding_atlas import AngleEncoding

    return AngleEncoding(n_features=4)


@pytest.fixture
def iqp_encoding_4q():
    """Create IQPEncoding with 4 features."""
    from encoding_atlas import IQPEncoding

    return IQPEncoding(n_features=4, reps=2, entanglement="full")


@pytest.fixture
def iqp_encoding_linear():
    """Create IQPEncoding with linear entanglement."""
    from encoding_atlas import IQPEncoding

    return IQPEncoding(n_features=4, reps=1, entanglement="linear")


@pytest.fixture
def basis_encoding_4q():
    """Create BasisEncoding with 4 features (data-dependent)."""
    from encoding_atlas import BasisEncoding

    return BasisEncoding(n_features=4)


@pytest.fixture
def sample_binary_input() -> NDArray[np.floating[Any]]:
    """Sample binary input for BasisEncoding (2 ones)."""
    return np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float64)


@pytest.fixture
def sample_all_ones() -> NDArray[np.floating[Any]]:
    """Input that binarizes to all ones."""
    return np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float64)


@pytest.fixture
def sample_all_zeros() -> NDArray[np.floating[Any]]:
    """Input that binarizes to all zeros."""
    return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)


@pytest.fixture
def multiple_encodings():
    """List of various encodings for comparison."""
    from encoding_atlas import AngleEncoding, IQPEncoding

    return [
        AngleEncoding(n_features=4),
        IQPEncoding(n_features=4, reps=1),
        IQPEncoding(n_features=4, reps=2),
    ]


# =============================================================================
# Tests for count_resources
# =============================================================================


class TestCountResources:
    """Tests for the count_resources function."""

    def test_basic_angle_encoding(self, angle_encoding_4q):
        """Test basic resource counting for AngleEncoding."""
        result = count_resources(angle_encoding_4q)

        # Check all required keys are present
        assert "n_qubits" in result
        assert "depth" in result
        assert "gate_count" in result
        assert "single_qubit_gates" in result
        assert "two_qubit_gates" in result
        assert "parameter_count" in result
        assert "encoding_name" in result

        # Check values for AngleEncoding
        assert result["n_qubits"] == 4
        assert result["encoding_name"] == "AngleEncoding"
        assert result["two_qubit_gates"] == 0  # AngleEncoding has no 2Q gates
        assert result["is_data_dependent"] is False

    def test_iqp_encoding_resources(self, iqp_encoding_4q):
        """Test resource counting for IQPEncoding."""
        result = count_resources(iqp_encoding_4q)

        assert result["n_qubits"] == 4
        assert result["encoding_name"] == "IQPEncoding"
        assert result["two_qubit_gates"] > 0  # IQP has CNOT gates
        assert result["is_data_dependent"] is False

        # Check that two_qubit_ratio is calculated correctly
        expected_ratio = result["two_qubit_gates"] / result["gate_count"]
        assert abs(result["two_qubit_ratio"] - expected_ratio) < 1e-10

    def test_detailed_breakdown(self, iqp_encoding_4q):
        """Test detailed gate breakdown."""
        result = count_resources(iqp_encoding_4q, detailed=True)

        # Check detailed breakdown keys
        assert "rx" in result
        assert "ry" in result
        assert "rz" in result
        assert "h" in result
        assert "cnot" in result
        assert "total_single_qubit" in result
        assert "total_two_qubit" in result
        assert "total" in result
        assert "encoding_name" in result

        # Verify totals add up
        single_qubit_sum = sum(
            [
                result["rx"],
                result["ry"],
                result["rz"],
                result["h"],
                result["x"],
                result["y"],
                result["z"],
                result["s"],
                result["t"],
            ]
        )
        assert single_qubit_sum == result["total_single_qubit"]

    def test_data_dependent_encoding(self, basis_encoding_4q, sample_binary_input):
        """Test resource counting for data-dependent BasisEncoding."""
        result = count_resources(basis_encoding_4q, x=sample_binary_input)

        assert result["encoding_name"] == "BasisEncoding"
        assert result["is_data_dependent"] is True
        assert result["two_qubit_gates"] == 0  # BasisEncoding has no 2Q gates

    def test_data_dependent_requires_x(self, basis_encoding_4q):
        """Test that data-dependent encodings require input x."""
        with pytest.raises(ValidationError) as exc_info:
            count_resources(basis_encoding_4q)

        assert "data-dependent" in str(exc_info.value).lower()

    def test_data_dependent_varies_with_input(
        self, basis_encoding_4q, sample_all_ones, sample_all_zeros
    ):
        """Test that data-dependent gate count varies with input."""
        result_ones = count_resources(basis_encoding_4q, x=sample_all_ones)
        result_zeros = count_resources(basis_encoding_4q, x=sample_all_zeros)

        # All ones should have more X gates than all zeros
        assert result_ones["gate_count"] > result_zeros["gate_count"]

    def test_invalid_encoding_type(self):
        """Test error handling for invalid encoding type."""
        with pytest.raises(AnalysisError) as exc_info:
            count_resources("not an encoding")

        assert "BaseEncoding" in str(exc_info.value)

    def test_invalid_encoding_none(self):
        """Test error handling for None encoding."""
        with pytest.raises(AnalysisError):
            count_resources(None)

    def test_input_validation_wrong_shape(self, basis_encoding_4q):
        """Test input validation for wrong shape."""
        wrong_shape = np.array([0.5, 0.5, 0.5])  # 3 features instead of 4

        with pytest.raises(ValidationError) as exc_info:
            count_resources(basis_encoding_4q, x=wrong_shape)

        assert "features" in str(exc_info.value).lower()

    def test_input_validation_nan(self, basis_encoding_4q):
        """Test input validation for NaN values."""
        nan_input = np.array([0.5, np.nan, 0.5, 0.5])

        with pytest.raises(ValidationError) as exc_info:
            count_resources(basis_encoding_4q, x=nan_input)

        assert "nan" in str(exc_info.value).lower()

    def test_input_validation_inf(self, basis_encoding_4q):
        """Test input validation for infinite values."""
        inf_input = np.array([0.5, np.inf, 0.5, 0.5])

        with pytest.raises(ValidationError) as exc_info:
            count_resources(basis_encoding_4q, x=inf_input)

        assert (
            "infinite" in str(exc_info.value).lower()
            or "inf" in str(exc_info.value).lower()
        )

    def test_2d_input_single_sample(self, basis_encoding_4q):
        """Test that 2D input with single sample is accepted."""
        x_2d = np.array([[0.2, 0.8, 0.3, 0.9]])

        result = count_resources(basis_encoding_4q, x=x_2d)
        assert result["gate_count"] >= 0

    def test_2d_input_multiple_samples_rejected(self, basis_encoding_4q):
        """Test that 2D input with multiple samples is rejected."""
        x_2d = np.array([[0.2, 0.8, 0.3, 0.9], [0.1, 0.9, 0.2, 0.8]])

        with pytest.raises(ValidationError):
            count_resources(basis_encoding_4q, x=x_2d)


# =============================================================================
# Tests for get_resource_summary
# =============================================================================


class TestGetResourceSummary:
    """Tests for the get_resource_summary function."""

    def test_basic_summary(self, angle_encoding_4q):
        """Test basic resource summary."""
        result = get_resource_summary(angle_encoding_4q)

        assert result["n_qubits"] == 4
        assert result["encoding_name"] == "AngleEncoding"
        assert "depth" in result
        assert "gate_count" in result

    def test_summary_uses_properties(self, iqp_encoding_4q):
        """Test that summary uses encoding properties."""
        result = get_resource_summary(iqp_encoding_4q)

        # Should match properties
        props = iqp_encoding_4q.properties
        assert result["n_qubits"] == props.n_qubits
        assert result["depth"] == props.depth

    def test_summary_for_data_dependent(self, basis_encoding_4q):
        """Test summary returns worst-case for data-dependent encodings."""
        result = get_resource_summary(basis_encoding_4q)

        # Should return worst-case (max gates) without requiring x
        assert result["is_data_dependent"] is True
        assert result["gate_count"] == basis_encoding_4q.properties.gate_count

    def test_derived_metrics(self, iqp_encoding_4q):
        """Test that derived metrics are calculated."""
        result = get_resource_summary(iqp_encoding_4q)

        assert "two_qubit_ratio" in result
        assert "gates_per_qubit" in result

        # Verify calculations
        if result["gate_count"] > 0:
            expected_ratio = result["two_qubit_gates"] / result["gate_count"]
            assert abs(result["two_qubit_ratio"] - expected_ratio) < 1e-10

    def test_invalid_encoding(self):
        """Test error handling for invalid encoding."""
        with pytest.raises(AnalysisError):
            get_resource_summary("not an encoding")


# =============================================================================
# Tests for get_gate_breakdown
# =============================================================================


class TestGetGateBreakdown:
    """Tests for the get_gate_breakdown function."""

    def test_breakdown_structure(self, iqp_encoding_4q):
        """Test gate breakdown structure."""
        result = get_gate_breakdown(iqp_encoding_4q)

        # Check all gate types are present
        gate_types = [
            "rx",
            "ry",
            "rz",
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "cnot",
            "cx",
            "cz",
            "swap",
        ]
        for gate in gate_types:
            assert gate in result

        # Check totals are present
        assert "total_single_qubit" in result
        assert "total_two_qubit" in result
        assert "total" in result

    def test_breakdown_for_data_dependent(self, basis_encoding_4q, sample_binary_input):
        """Test breakdown for data-dependent encoding."""
        result = get_gate_breakdown(basis_encoding_4q, x=sample_binary_input)

        # BasisEncoding only uses X gates
        assert result["x"] == result["total"]
        assert result["total_two_qubit"] == 0

    def test_breakdown_totals_match(self, iqp_encoding_4q):
        """Test that breakdown totals match sum of individual gates."""
        result = get_gate_breakdown(iqp_encoding_4q)

        single_qubit_sum = sum(
            [
                result["rx"],
                result["ry"],
                result["rz"],
                result["h"],
                result["x"],
                result["y"],
                result["z"],
                result["s"],
                result["t"],
            ]
        )

        two_qubit_sum = result["cnot"] + result["cz"] + result["swap"]

        assert single_qubit_sum == result["total_single_qubit"]
        assert two_qubit_sum == result["total_two_qubit"]
        assert single_qubit_sum + two_qubit_sum == result["total"]


# =============================================================================
# Tests for compare_resources
# =============================================================================


class TestCompareResources:
    """Tests for the compare_resources function."""

    def test_basic_comparison(self, multiple_encodings):
        """Test basic encoding comparison."""
        result = compare_resources(multiple_encodings)

        assert "encoding_name" in result
        assert "gate_count" in result
        assert len(result["encoding_name"]) == 3
        assert len(result["gate_count"]) == 3

    def test_comparison_order_preserved(self, multiple_encodings):
        """Test that comparison order matches input order."""
        result = compare_resources(multiple_encodings)

        assert result["encoding_name"][0] == "AngleEncoding"
        assert result["encoding_name"][1] == "IQPEncoding"
        assert result["encoding_name"][2] == "IQPEncoding"

    def test_specific_metrics(self, multiple_encodings):
        """Test comparison with specific metrics."""
        metrics = ["gate_count", "two_qubit_gates"]
        result = compare_resources(multiple_encodings, metrics=metrics)

        assert "gate_count" in result
        assert "two_qubit_gates" in result
        assert "depth" not in result  # Not requested

    def test_without_names(self, multiple_encodings):
        """Test comparison without encoding names."""
        result = compare_resources(multiple_encodings, include_names=False)

        assert "encoding_name" not in result

    def test_empty_list_error(self):
        """Test error for empty encoding list."""
        with pytest.raises(ValueError) as exc_info:
            compare_resources([])

        assert "empty" in str(exc_info.value).lower()

    def test_single_encoding(self, angle_encoding_4q):
        """Test comparison with single encoding."""
        result = compare_resources([angle_encoding_4q])

        assert len(result["encoding_name"]) == 1
        assert result["encoding_name"][0] == "AngleEncoding"

    def test_gate_counts_increase_with_reps(self):
        """Test that gate counts increase with repetitions."""
        from encoding_atlas import IQPEncoding

        encodings = [
            IQPEncoding(n_features=4, reps=1),
            IQPEncoding(n_features=4, reps=2),
            IQPEncoding(n_features=4, reps=3),
        ]

        result = compare_resources(encodings)

        # Gate count should increase with reps
        assert result["gate_count"][1] > result["gate_count"][0]
        assert result["gate_count"][2] > result["gate_count"][1]


# =============================================================================
# Tests for estimate_execution_time
# =============================================================================


class TestEstimateExecutionTime:
    """Tests for the estimate_execution_time function."""

    def test_basic_estimation(self, iqp_encoding_4q):
        """Test basic execution time estimation."""
        result = estimate_execution_time(iqp_encoding_4q)

        assert "serial_time_us" in result
        assert "estimated_time_us" in result
        assert "single_qubit_time_us" in result
        assert "two_qubit_time_us" in result
        assert "measurement_time_us" in result
        assert "parallelization_factor" in result

    def test_serial_time_positive(self, iqp_encoding_4q):
        """Test that serial time is positive for non-trivial circuits."""
        result = estimate_execution_time(iqp_encoding_4q)

        assert result["serial_time_us"] > 0
        assert result["estimated_time_us"] > 0

    def test_estimated_less_than_serial(self, iqp_encoding_4q):
        """Test that estimated time is less than or equal to serial."""
        result = estimate_execution_time(iqp_encoding_4q)

        # With parallelization, estimated should be less or equal
        assert result["estimated_time_us"] <= result["serial_time_us"]

    def test_custom_gate_times(self, iqp_encoding_4q):
        """Test with custom gate times."""
        result_default = estimate_execution_time(iqp_encoding_4q)
        result_slow = estimate_execution_time(
            iqp_encoding_4q,
            single_qubit_gate_time_us=1.0,
            two_qubit_gate_time_us=10.0,
        )

        # Slower gates should give longer times
        assert result_slow["serial_time_us"] > result_default["serial_time_us"]

    def test_without_measurement(self, iqp_encoding_4q):
        """Test estimation without measurement time."""
        result_with = estimate_execution_time(iqp_encoding_4q, include_measurement=True)
        result_without = estimate_execution_time(
            iqp_encoding_4q, include_measurement=False
        )

        assert result_without["measurement_time_us"] == 0.0
        assert result_with["serial_time_us"] > result_without["serial_time_us"]

    def test_parallelization_factor(self, iqp_encoding_4q):
        """Test different parallelization factors."""
        result_no_parallel = estimate_execution_time(
            iqp_encoding_4q, parallelization_factor=0.0
        )
        result_full_parallel = estimate_execution_time(
            iqp_encoding_4q, parallelization_factor=1.0
        )

        # Serial time should be independent of parallelization
        assert (
            result_no_parallel["serial_time_us"]
            == result_full_parallel["serial_time_us"]
        )

        # Full parallel should be faster or equal (bounded by critical path)
        assert (
            result_full_parallel["estimated_time_us"]
            <= result_no_parallel["estimated_time_us"]
        )

    def test_returns_parallelization_factor(self, iqp_encoding_4q):
        """Test that parallelization factor is returned."""
        factor = 0.7
        result = estimate_execution_time(iqp_encoding_4q, parallelization_factor=factor)

        assert result["parallelization_factor"] == factor

    def test_zero_gate_encoding(self):
        """Test estimation for encoding with minimal gates."""
        from encoding_atlas import AngleEncoding

        # AngleEncoding has no 2Q gates
        enc = AngleEncoding(n_features=2)
        result = estimate_execution_time(enc)

        assert result["two_qubit_time_us"] == 0.0
        assert result["serial_time_us"] > 0


class TestEstimateExecutionTimeMathematicalCorrectness:
    """Tests for mathematical correctness of estimate_execution_time.

    These tests verify that the timing formulas produce physically meaningful
    results at edge cases and boundary conditions, ensuring the parallelization
    model behaves correctly.
    """

    def test_full_parallelization_bounded_by_critical_path(self):
        """Test that full parallelization is bounded by critical path time.

        When parallelization_factor=1.0, the estimated time should not be less
        than the critical path time (depth * slowest_gate_time + measurement).
        This is a fundamental physical constraint - no amount of parallelization
        can make a circuit faster than its critical path.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)
        summary = get_resource_summary(enc)

        # Use default gate times
        two_qubit_gate_time_us = 0.2  # Default value
        measurement_time_us = 1.0  # Default value

        result = estimate_execution_time(enc, parallelization_factor=1.0)

        # Critical path = depth * max(single_qubit, two_qubit) + measurement
        # For IQP (has 2Q gates): max(0.02, 0.2) = 0.2
        expected_critical_path = (
            summary["depth"] * two_qubit_gate_time_us + measurement_time_us
        )

        # Estimated time must be >= critical path (physical constraint)
        assert result["estimated_time_us"] >= expected_critical_path - 1e-10, (
            f"Estimated time {result['estimated_time_us']:.6f} μs is less than "
            f"critical path time {expected_critical_path:.6f} μs"
        )

    def test_critical_path_with_custom_gate_times(self):
        """Test critical path calculation with custom gate times."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        summary = get_resource_summary(enc)

        # Use custom gate times (simulating trapped ion hardware)
        single_qubit_time = 1.0  # 1 μs
        two_qubit_time = 100.0  # 100 μs (much slower)
        meas_time = 10.0  # 10 μs

        result = estimate_execution_time(
            enc,
            single_qubit_gate_time_us=single_qubit_time,
            two_qubit_gate_time_us=two_qubit_time,
            measurement_time_us=meas_time,
            parallelization_factor=1.0,
        )

        # Critical path = depth * max(single_qubit, two_qubit) + meas_time
        # For IQP (has 2Q gates): max(1.0, 100.0) = 100.0
        expected_critical_path = (
            summary["depth"] * max(single_qubit_time, two_qubit_time) + meas_time
        )

        assert result["estimated_time_us"] >= expected_critical_path - 1e-10, (
            f"With custom times: estimated {result['estimated_time_us']:.2f} μs "
            f"< critical path {expected_critical_path:.2f} μs"
        )

    def test_serial_time_formula_correctness(self):
        """Test that serial time is calculated correctly.

        serial_time = (single_qubit_gates * single_qubit_time) +
                      (two_qubit_gates * two_qubit_time) +
                      measurement_time
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        summary = get_resource_summary(enc)

        # Use specific gate times for easy verification
        single_qubit_time = 0.1
        two_qubit_time = 1.0
        meas_time = 5.0

        result = estimate_execution_time(
            enc,
            single_qubit_gate_time_us=single_qubit_time,
            two_qubit_gate_time_us=two_qubit_time,
            measurement_time_us=meas_time,
        )

        expected_serial = (
            summary["single_qubit_gates"] * single_qubit_time
            + summary["two_qubit_gates"] * two_qubit_time
            + meas_time
        )

        assert abs(result["serial_time_us"] - expected_serial) < 1e-10, (
            f"Serial time {result['serial_time_us']:.6f} μs != "
            f"expected {expected_serial:.6f} μs"
        )

    def test_parallelization_monotonicity(self):
        """Test that more parallelization never increases estimated time.

        As parallelization_factor increases from 0 to 1, the estimated time
        should monotonically decrease (or stay constant if bounded by
        critical path).
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)

        factors = [0.0, 0.25, 0.5, 0.75, 1.0]
        times = []

        for factor in factors:
            result = estimate_execution_time(enc, parallelization_factor=factor)
            times.append(result["estimated_time_us"])

        # Each time should be <= the previous time
        for i in range(1, len(times)):
            assert times[i] <= times[i - 1] + 1e-10, (
                f"Time increased with more parallelization: "
                f"factor {factors[i-1]} -> {times[i-1]:.4f} μs, "
                f"factor {factors[i]} -> {times[i]:.4f} μs"
            )

    def test_no_parallelization_equals_serial_minus_measurement(self):
        """Test that parallelization_factor=0 gives gate_time + measurement.

        With zero parallelization, the estimated_time should equal the serial
        gate time plus measurement time.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)

        result = estimate_execution_time(enc, parallelization_factor=0.0)

        # With factor=0: estimated = gate_time * (1 - 0) + meas = serial_time
        # But may still be bounded by critical path
        expected_no_parallel = (
            result["single_qubit_time_us"]
            + result["two_qubit_time_us"]
            + result["measurement_time_us"]
        )

        # The estimated time should be at least the unbounded calculation
        # (could be higher if critical path bound kicks in, but for factor=0
        # the unbounded calc IS the serial time, so it should match)
        assert (
            abs(result["estimated_time_us"] - result["serial_time_us"]) < 1e-10
            or result["estimated_time_us"] >= expected_no_parallel - 1e-10
        )

    def test_single_qubit_only_encoding_critical_path(self):
        """Test critical path for encodings with only single-qubit gates.

        For encodings without two-qubit gates (like AngleEncoding), the
        critical path uses single_qubit_gate_time (not two_qubit_gate_time)
        since no layer contains two-qubit gates.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4, reps=1)
        summary = get_resource_summary(enc)

        # Verify no two-qubit gates
        assert summary["two_qubit_gates"] == 0

        single_qubit_time = 0.02  # Default
        meas_time = 1.0

        result = estimate_execution_time(
            enc,
            single_qubit_gate_time_us=single_qubit_time,
            measurement_time_us=meas_time,
            parallelization_factor=1.0,
        )

        # Critical path = depth * single_qubit_time + meas_time
        # (uses single_qubit_time because circuit has no two-qubit gates)
        expected_critical_path = summary["depth"] * single_qubit_time + meas_time

        assert result["estimated_time_us"] >= expected_critical_path - 1e-10

    def test_zero_measurement_time(self):
        """Test estimation with zero measurement time."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)

        result = estimate_execution_time(
            enc,
            measurement_time_us=0.0,
            include_measurement=True,  # Should still be 0 since meas_time=0
        )

        assert result["measurement_time_us"] == 0.0
        # Serial time should still be positive from gate times
        assert result["serial_time_us"] > 0

    def test_deep_circuit_critical_path_dominates(self):
        """Test that for deep circuits, critical path dominates at high parallelization.

        For circuits with many repetitions (high depth), the critical path
        should become the dominant factor when parallelization is high.
        """
        from encoding_atlas import IQPEncoding

        # Create a deep circuit
        enc_deep = IQPEncoding(n_features=4, reps=5)
        summary = get_resource_summary(enc_deep)

        two_qubit_time = 0.2
        meas_time = 1.0

        result = estimate_execution_time(
            enc_deep,
            two_qubit_gate_time_us=two_qubit_time,
            measurement_time_us=meas_time,
            parallelization_factor=1.0,
        )

        expected_critical_path = summary["depth"] * two_qubit_time + meas_time

        # For full parallelization, estimated time should be exactly critical path
        # (or very close, accounting for floating point)
        assert abs(result["estimated_time_us"] - expected_critical_path) < 1e-10, (
            f"Deep circuit: estimated {result['estimated_time_us']:.4f} μs "
            f"!= critical path {expected_critical_path:.4f} μs"
        )

    def test_component_times_sum_correctly(self):
        """Test that component times (single_qubit + two_qubit + meas) equal serial."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)

        result = estimate_execution_time(enc, include_measurement=True)

        expected_serial = (
            result["single_qubit_time_us"]
            + result["two_qubit_time_us"]
            + result["measurement_time_us"]
        )

        assert abs(result["serial_time_us"] - expected_serial) < 1e-10, (
            f"Component times don't sum to serial: "
            f"{result['single_qubit_time_us']:.4f} + "
            f"{result['two_qubit_time_us']:.4f} + "
            f"{result['measurement_time_us']:.4f} = {expected_serial:.4f} "
            f"!= {result['serial_time_us']:.4f}"
        )

    @pytest.mark.parametrize("n_features", [2, 4, 8])
    @pytest.mark.parametrize("reps", [1, 2, 3])
    def test_scaling_with_circuit_size(self, n_features, reps):
        """Test that times scale appropriately with circuit size.

        Note: The critical path uses max(single_qubit_time, two_qubit_time)
        per layer for circuits with two-qubit gates, or single_qubit_time
        for single-qubit-only circuits. This can exceed serial time for
        certain circuit configurations. This is documented behavior.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=n_features, reps=reps)

        result = estimate_execution_time(enc)

        # All times should be positive
        assert result["serial_time_us"] > 0
        assert result["estimated_time_us"] > 0

        # Critical path bound should hold
        # IQPEncoding always has 2Q gates, so per_layer_time = max(0.02, 0.2) = 0.2
        summary = get_resource_summary(enc)
        per_layer_time = max(0.02, 0.2)  # Default single/two qubit gate times
        critical_path = summary["depth"] * per_layer_time + 1.0  # + default meas time

        assert result["estimated_time_us"] >= critical_path - 1e-10

        # Estimated time is bounded by max(formula_result, critical_path)
        # The key invariant is: estimated_time >= critical_path
        gate_time = result["single_qubit_time_us"] + result["two_qubit_time_us"]
        formula_result = (
            gate_time * 0.5 + result["measurement_time_us"]
        )  # default factor=0.5
        expected_estimated = max(formula_result, critical_path)

        assert abs(result["estimated_time_us"] - expected_estimated) < 1e-10


# =============================================================================
# Tests for Private Helper Functions
# =============================================================================


class TestPrivateHelpers:
    """Tests for private helper functions."""

    def test_safe_divide_normal(self):
        """Test safe division with normal values."""
        assert _safe_divide(10, 5) == 2.0
        assert _safe_divide(1, 4) == 0.25

    def test_safe_divide_zero_denominator(self):
        """Test safe division with zero denominator."""
        assert _safe_divide(10, 0) == 0.0
        assert _safe_divide(0, 0) == 0.0

    def test_safe_divide_small_denominator(self):
        """Test safe division with very small denominator."""
        assert _safe_divide(10, 1e-20) == 0.0

    def test_validate_encoding_invalid_type(self):
        """Test encoding validation with invalid type."""
        with pytest.raises(AnalysisError):
            _validate_encoding("not an encoding")

    def test_validate_encoding_none(self):
        """Test encoding validation with None."""
        with pytest.raises(AnalysisError):
            _validate_encoding(None)

    def test_validate_input_data_valid(self):
        """Test input data validation with valid data."""
        result = _validate_input_data(np.array([0.1, 0.2, 0.3]), 3)
        assert len(result) == 3

    def test_validate_input_data_wrong_features(self):
        """Test input validation with wrong number of features."""
        with pytest.raises(ValidationError):
            _validate_input_data(np.array([0.1, 0.2]), 3)

    def test_validate_input_data_nan(self):
        """Test input validation with NaN."""
        with pytest.raises(ValidationError):
            _validate_input_data(np.array([0.1, np.nan, 0.3]), 3)

    def test_validate_input_data_inf(self):
        """Test input validation with infinity."""
        with pytest.raises(ValidationError):
            _validate_input_data(np.array([0.1, np.inf, 0.3]), 3)

    def test_validate_input_data_list(self):
        """Test input validation accepts lists."""
        result = _validate_input_data([0.1, 0.2, 0.3], 3)
        assert len(result) == 3


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_qubit_encoding(self):
        """Test encoding with single qubit."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = count_resources(enc)

        assert result["n_qubits"] == 1
        assert result["gate_count"] >= 1

    def test_large_encoding(self):
        """Test encoding with many qubits."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=10)
        result = count_resources(enc)

        assert result["n_qubits"] == 10
        assert result["gates_per_qubit"] > 0

    def test_different_entanglement_patterns(self):
        """Test IQP with different entanglement patterns."""
        from encoding_atlas import IQPEncoding

        enc_full = IQPEncoding(n_features=4, entanglement="full")
        enc_linear = IQPEncoding(n_features=4, entanglement="linear")
        enc_circular = IQPEncoding(n_features=4, entanglement="circular")

        res_full = count_resources(enc_full)
        res_linear = count_resources(enc_linear)
        res_circular = count_resources(enc_circular)

        # Full entanglement should have more 2Q gates
        assert res_full["two_qubit_gates"] > res_linear["two_qubit_gates"]
        assert res_full["two_qubit_gates"] > res_circular["two_qubit_gates"]

    def test_consistent_results(self, iqp_encoding_4q):
        """Test that repeated calls give consistent results."""
        result1 = count_resources(iqp_encoding_4q)
        result2 = count_resources(iqp_encoding_4q)

        assert result1["gate_count"] == result2["gate_count"]
        assert result1["n_qubits"] == result2["n_qubits"]
        assert result1["depth"] == result2["depth"]


# =============================================================================
# Tests for Type Annotations
# =============================================================================


class TestTypeAnnotations:
    """Tests for type annotations and return types."""

    def test_count_resources_returns_dict(self, angle_encoding_4q):
        """Test that count_resources returns a dict."""
        result = count_resources(angle_encoding_4q)
        assert isinstance(result, dict)

    def test_detailed_has_different_keys(self, angle_encoding_4q):
        """Test that detailed=True returns different keys."""
        summary = count_resources(angle_encoding_4q, detailed=False)
        detailed = count_resources(angle_encoding_4q, detailed=True)

        # Summary should have is_data_dependent
        assert "is_data_dependent" in summary

        # Detailed should have rx, ry, rz
        assert "rx" in detailed
        assert "ry" in detailed
        assert "rz" in detailed

    def test_compare_resources_returns_dict_of_lists(self, multiple_encodings):
        """Test that compare_resources returns dict of lists."""
        result = compare_resources(multiple_encodings)

        assert isinstance(result, dict)
        for value in result.values():
            assert isinstance(value, list)

    def test_estimate_execution_time_returns_dict(self, iqp_encoding_4q):
        """Test that estimate_execution_time returns dict."""
        result = estimate_execution_time(iqp_encoding_4q)

        assert isinstance(result, dict)
        for value in result.values():
            assert isinstance(value, float)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_workflow_compare_then_detail(self, multiple_encodings):
        """Test typical workflow: compare then get details."""
        # First, compare encodings
        comparison = compare_resources(multiple_encodings)

        # Find the encoding with most gates
        max_idx = comparison["gate_count"].index(max(comparison["gate_count"]))
        selected_encoding = multiple_encodings[max_idx]

        # Get detailed breakdown
        breakdown = get_gate_breakdown(selected_encoding)

        assert breakdown["total"] == comparison["gate_count"][max_idx]

    def test_workflow_resource_then_time(self, iqp_encoding_4q):
        """Test workflow: count resources then estimate time."""
        resources = count_resources(iqp_encoding_4q)
        time_estimate = estimate_execution_time(iqp_encoding_4q)

        # Time should correlate with gate count
        # More gates generally means more time
        assert time_estimate["serial_time_us"] > 0

        # Verify resource data matches time estimate components
        # (single_qubit_gates + two_qubit_gates should drive the time)
        assert resources["gate_count"] > 0
        assert (
            resources["gate_count"]
            == resources["single_qubit_gates"] + resources["two_qubit_gates"]
        )

    def test_summary_matches_count(self, angle_encoding_4q):
        """Test that get_resource_summary matches count_resources."""
        summary = get_resource_summary(angle_encoding_4q)
        count = count_resources(angle_encoding_4q)

        assert summary["n_qubits"] == count["n_qubits"]
        assert summary["depth"] == count["depth"]
        assert summary["gate_count"] == count["gate_count"]


# =============================================================================
# Tests for All Encoding Types
# =============================================================================


class TestAllEncodingTypes:
    """Test count_resources against every encoding type in the library.

    This ensures the protocol-based dispatch works for all 16 encodings
    and that gate counts, totals, and invariants hold universally.
    """

    @pytest.fixture
    def amplitude_encoding(self):
        from encoding_atlas import AmplitudeEncoding

        return AmplitudeEncoding(n_features=4)

    @pytest.fixture
    def zz_feature_map(self):
        from encoding_atlas import ZZFeatureMap

        return ZZFeatureMap(n_features=4, reps=2, entanglement="full")

    @pytest.fixture
    def pauli_feature_map(self):
        from encoding_atlas import PauliFeatureMap

        return PauliFeatureMap(n_features=4, reps=1, entanglement="full")

    @pytest.fixture
    def hardware_efficient(self):
        from encoding_atlas import HardwareEfficientEncoding

        return HardwareEfficientEncoding(n_features=4, reps=2, entanglement="linear")

    @pytest.fixture
    def data_reuploading(self):
        from encoding_atlas import DataReuploading

        return DataReuploading(n_features=4, n_layers=2)

    @pytest.fixture
    def higher_order_angle(self):
        from encoding_atlas import HigherOrderAngleEncoding

        return HigherOrderAngleEncoding(n_features=4, order=2)

    @pytest.fixture
    def qaoa_encoding(self):
        from encoding_atlas import QAOAEncoding

        return QAOAEncoding(n_features=4, reps=2, entanglement="linear")

    @pytest.fixture
    def hamiltonian_encoding(self):
        from encoding_atlas import HamiltonianEncoding

        return HamiltonianEncoding(n_features=4, reps=2, entanglement="full")

    @pytest.fixture
    def symmetry_inspired(self):
        from encoding_atlas import SymmetryInspiredFeatureMap

        return SymmetryInspiredFeatureMap(n_features=4, reps=2)

    @pytest.fixture
    def trainable_encoding(self):
        from encoding_atlas import TrainableEncoding

        return TrainableEncoding(n_features=4, n_layers=2, entanglement="linear")

    @pytest.fixture
    def so2_equivariant(self):
        from encoding_atlas import SO2EquivariantFeatureMap

        return SO2EquivariantFeatureMap(n_features=2, max_angular_momentum=1)

    @pytest.fixture
    def cyclic_equivariant(self):
        from encoding_atlas import CyclicEquivariantFeatureMap

        return CyclicEquivariantFeatureMap(n_features=4, reps=2)

    @pytest.fixture
    def swap_equivariant(self):
        from encoding_atlas import SwapEquivariantFeatureMap

        return SwapEquivariantFeatureMap(n_features=4, reps=2)

    @pytest.fixture(
        params=[
            "amplitude_encoding",
            "zz_feature_map",
            "pauli_feature_map",
            "hardware_efficient",
            "data_reuploading",
            "higher_order_angle",
            "qaoa_encoding",
            "hamiltonian_encoding",
            "symmetry_inspired",
            "trainable_encoding",
            "so2_equivariant",
            "cyclic_equivariant",
            "swap_equivariant",
        ]
    )
    def any_encoding(self, request):
        """Parametrize over all non-trivial encoding types."""
        return request.getfixturevalue(request.param)

    def test_count_resources_summary_all_keys_present(self, any_encoding):
        """Every encoding must produce a summary with all required keys."""
        result = count_resources(any_encoding)

        required_keys = {
            "n_qubits",
            "depth",
            "gate_count",
            "single_qubit_gates",
            "two_qubit_gates",
            "parameter_count",
            "cnot_count",
            "cz_count",
            "t_gate_count",
            "hadamard_count",
            "rotation_gates",
            "two_qubit_ratio",
            "gates_per_qubit",
            "encoding_name",
            "is_data_dependent",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys for {any_encoding.__class__.__name__}: "
            f"{required_keys - result.keys()}"
        )

    def test_count_resources_detailed_all_keys_present(self, any_encoding):
        """Every encoding must produce a detailed breakdown with all keys."""
        result = count_resources(any_encoding, detailed=True)

        required_keys = {
            "rx",
            "ry",
            "rz",
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "cnot",
            "cx",
            "cz",
            "swap",
            "total_single_qubit",
            "total_two_qubit",
            "total",
            "encoding_name",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys for {any_encoding.__class__.__name__}: "
            f"{required_keys - result.keys()}"
        )

    def test_gate_count_equals_sum_of_single_and_two_qubit(self, any_encoding):
        """gate_count must equal single_qubit_gates + two_qubit_gates."""
        result = count_resources(any_encoding)

        assert result["gate_count"] == (
            result["single_qubit_gates"] + result["two_qubit_gates"]
        ), (
            f"{any_encoding.__class__.__name__}: "
            f"gate_count={result['gate_count']} != "
            f"single({result['single_qubit_gates']}) + two({result['two_qubit_gates']})"
        )

    def test_detailed_totals_internally_consistent(self, any_encoding):
        """total must equal total_single_qubit + total_two_qubit."""
        result = count_resources(any_encoding, detailed=True)

        assert (
            result["total"] == result["total_single_qubit"] + result["total_two_qubit"]
        ), (
            f"{any_encoding.__class__.__name__}: "
            f"total={result['total']} != "
            f"1Q({result['total_single_qubit']}) + 2Q({result['total_two_qubit']})"
        )

    def test_mapped_gates_do_not_exceed_totals(self, any_encoding):
        """Mapped individual gates must not exceed the totals.

        Note: Some encodings use custom gate names (e.g., 'data_ry',
        'mixer_rx') that aren't mapped to the standard rx/ry/rz/h/x/y/z/s/t
        keys. So mapped gates may be LESS than total, but never MORE.
        """
        result = count_resources(any_encoding, detailed=True)

        single_qubit_sum = sum(
            [
                result["rx"],
                result["ry"],
                result["rz"],
                result["h"],
                result["x"],
                result["y"],
                result["z"],
                result["s"],
                result["t"],
            ]
        )

        # Use cnot (not cx) to avoid double-counting the alias
        two_qubit_sum = result["cnot"] + result["cz"] + result["swap"]

        assert single_qubit_sum <= result["total_single_qubit"], (
            f"{any_encoding.__class__.__name__}: "
            f"mapped 1Q sum={single_qubit_sum} > total_single_qubit={result['total_single_qubit']}"
        )

        assert two_qubit_sum <= result["total_two_qubit"], (
            f"{any_encoding.__class__.__name__}: "
            f"mapped 2Q sum={two_qubit_sum} > total_two_qubit={result['total_two_qubit']}"
        )

    def test_summary_and_detailed_totals_consistent(self, any_encoding):
        """Summary gate_count must equal detailed total."""
        summary = count_resources(any_encoding, detailed=False)
        detailed = count_resources(any_encoding, detailed=True)

        assert summary["gate_count"] == detailed["total"], (
            f"{any_encoding.__class__.__name__}: "
            f"summary gate_count={summary['gate_count']} != "
            f"detailed total={detailed['total']}"
        )
        assert summary["single_qubit_gates"] == detailed["total_single_qubit"]
        assert summary["two_qubit_gates"] == detailed["total_two_qubit"]

    def test_all_gate_counts_non_negative(self, any_encoding):
        """No gate count may be negative."""
        result = count_resources(any_encoding, detailed=True)

        for key, value in result.items():
            if isinstance(value, int):
                assert value >= 0, (
                    f"{any_encoding.__class__.__name__}: " f"{key}={value} is negative"
                )

    def test_n_qubits_positive(self, any_encoding):
        """Every encoding must report at least 1 qubit."""
        result = count_resources(any_encoding)
        assert result["n_qubits"] >= 1

    def test_depth_non_negative(self, any_encoding):
        """Depth must be non-negative."""
        result = count_resources(any_encoding)
        assert result["depth"] >= 0

    def test_two_qubit_ratio_in_range(self, any_encoding):
        """two_qubit_ratio must be in [0, 1]."""
        result = count_resources(any_encoding)
        assert 0.0 <= result["two_qubit_ratio"] <= 1.0

    def test_gates_per_qubit_non_negative(self, any_encoding):
        """gates_per_qubit must be non-negative."""
        result = count_resources(any_encoding)
        assert result["gates_per_qubit"] >= 0.0

    def test_is_data_dependent_false_for_standard_encodings(self, any_encoding):
        """All tested encodings here are NOT data-dependent."""
        result = count_resources(any_encoding)
        assert result["is_data_dependent"] is False

    def test_encoding_name_matches_class(self, any_encoding):
        """encoding_name must match the class name."""
        result = count_resources(any_encoding)
        assert result["encoding_name"] == any_encoding.__class__.__name__

    def test_get_resource_summary_works(self, any_encoding):
        """get_resource_summary must work for every encoding."""
        result = get_resource_summary(any_encoding)
        assert result["n_qubits"] >= 1
        assert result["gate_count"] >= 0

    def test_get_gate_breakdown_works(self, any_encoding):
        """get_gate_breakdown must work for every encoding."""
        result = get_gate_breakdown(any_encoding)
        assert result["total"] >= 0

    def test_estimate_execution_time_works(self, any_encoding):
        """estimate_execution_time must work for every encoding."""
        result = estimate_execution_time(any_encoding)
        assert result["serial_time_us"] >= 0
        assert result["estimated_time_us"] >= 0

    def test_compare_resources_single(self, any_encoding):
        """compare_resources must work with a single encoding."""
        result = compare_resources([any_encoding])
        assert len(result["gate_count"]) == 1

    @pytest.mark.parametrize(
        "enc_cls,enc_kwargs",
        [
            ("AngleEncoding", dict(n_features=4)),
            ("IQPEncoding", dict(n_features=4, reps=2, entanglement="full")),
            ("HardwareEfficientEncoding", dict(n_features=4, reps=2)),
            ("HigherOrderAngleEncoding", dict(n_features=4, order=2)),
            ("HamiltonianEncoding", dict(n_features=4, reps=1)),
        ],
    )
    def test_detailed_totals_exact_for_well_mapped_encodings(self, enc_cls, enc_kwargs):
        """For encodings with standard gate names, individual gates must sum
        exactly to the totals (no unmapped gates)."""
        import encoding_atlas

        cls = getattr(encoding_atlas, enc_cls)
        enc = cls(**enc_kwargs)
        result = count_resources(enc, detailed=True)

        single_qubit_sum = sum(
            [
                result["rx"],
                result["ry"],
                result["rz"],
                result["h"],
                result["x"],
                result["y"],
                result["z"],
                result["s"],
                result["t"],
            ]
        )
        two_qubit_sum = result["cnot"] + result["cz"] + result["swap"]

        assert single_qubit_sum == result["total_single_qubit"], (
            f"{enc_cls}: 1Q sum={single_qubit_sum} != "
            f"total_single_qubit={result['total_single_qubit']}"
        )
        assert two_qubit_sum == result["total_two_qubit"], (
            f"{enc_cls}: 2Q sum={two_qubit_sum} != "
            f"total_two_qubit={result['total_two_qubit']}"
        )


# =============================================================================
# Tests for Parameter Validation (New Validation Code)
# =============================================================================


class TestEstimateExecutionTimeValidation:
    """Tests for the parameter validation added to estimate_execution_time."""

    @pytest.fixture
    def encoding(self):
        from encoding_atlas import IQPEncoding

        return IQPEncoding(n_features=4, reps=1)

    def test_negative_parallelization_factor(self, encoding):
        """Negative parallelization_factor must raise ValidationError."""
        with pytest.raises(ValidationError, match="parallelization_factor"):
            estimate_execution_time(encoding, parallelization_factor=-0.1)

    def test_parallelization_factor_above_one(self, encoding):
        """parallelization_factor > 1 must raise ValidationError."""
        with pytest.raises(ValidationError, match="parallelization_factor"):
            estimate_execution_time(encoding, parallelization_factor=1.5)

    def test_parallelization_factor_large(self, encoding):
        """Very large parallelization_factor must raise ValidationError."""
        with pytest.raises(ValidationError, match="parallelization_factor"):
            estimate_execution_time(encoding, parallelization_factor=100.0)

    def test_parallelization_factor_boundary_zero(self, encoding):
        """parallelization_factor=0.0 (boundary) must succeed."""
        result = estimate_execution_time(encoding, parallelization_factor=0.0)
        assert result["parallelization_factor"] == 0.0

    def test_parallelization_factor_boundary_one(self, encoding):
        """parallelization_factor=1.0 (boundary) must succeed."""
        result = estimate_execution_time(encoding, parallelization_factor=1.0)
        assert result["parallelization_factor"] == 1.0

    def test_negative_single_qubit_gate_time(self, encoding):
        """Negative single_qubit_gate_time_us must raise ValidationError."""
        with pytest.raises(ValidationError, match="single_qubit_gate_time_us"):
            estimate_execution_time(encoding, single_qubit_gate_time_us=-0.01)

    def test_negative_two_qubit_gate_time(self, encoding):
        """Negative two_qubit_gate_time_us must raise ValidationError."""
        with pytest.raises(ValidationError, match="two_qubit_gate_time_us"):
            estimate_execution_time(encoding, two_qubit_gate_time_us=-1.0)

    def test_negative_measurement_time(self, encoding):
        """Negative measurement_time_us must raise ValidationError."""
        with pytest.raises(ValidationError, match="measurement_time_us"):
            estimate_execution_time(encoding, measurement_time_us=-0.5)

    def test_zero_gate_times_accepted(self, encoding):
        """Zero gate times are valid (edge case)."""
        result = estimate_execution_time(
            encoding,
            single_qubit_gate_time_us=0.0,
            two_qubit_gate_time_us=0.0,
            measurement_time_us=0.0,
            include_measurement=False,
        )
        assert result["serial_time_us"] == 0.0

    def test_invalid_encoding_raises_analysis_error(self):
        """estimate_execution_time must validate encoding type."""
        with pytest.raises(AnalysisError):
            estimate_execution_time("not an encoding")


# =============================================================================
# Tests for compare_resources Metric Validation
# =============================================================================


class TestCompareResourcesMetricValidation:
    """Tests for metric name validation in compare_resources."""

    @pytest.fixture
    def encodings(self):
        from encoding_atlas import AngleEncoding

        return [AngleEncoding(n_features=4)]

    def test_unknown_metric_raises_value_error(self, encodings):
        """Unknown metric names must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_resources(encodings, metrics=["gate_cont"])  # typo

    def test_unknown_metric_lists_valid_options(self, encodings):
        """Error message must list valid metric names."""
        with pytest.raises(ValueError, match="Valid metrics"):
            compare_resources(encodings, metrics=["nonexistent"])

    def test_multiple_unknown_metrics(self, encodings):
        """Multiple unknown metrics must all be listed."""
        with pytest.raises(ValueError, match="Unknown metric") as exc_info:
            compare_resources(encodings, metrics=["foo", "bar", "gate_count"])
        error_msg = str(exc_info.value)
        assert "foo" in error_msg or "bar" in error_msg

    def test_all_valid_metrics_accepted(self, encodings):
        """Every valid metric name must be accepted."""
        valid_metrics = [
            "n_qubits",
            "depth",
            "gate_count",
            "single_qubit_gates",
            "two_qubit_gates",
            "parameter_count",
            "cnot_count",
            "cz_count",
            "t_gate_count",
            "hadamard_count",
            "rotation_gates",
            "two_qubit_ratio",
            "gates_per_qubit",
        ]
        result = compare_resources(encodings, metrics=valid_metrics)
        for metric in valid_metrics:
            assert metric in result

    def test_empty_metrics_list_raises_no_error(self, encodings):
        """Empty metrics list produces empty result (no crash)."""
        result = compare_resources(encodings, metrics=[])
        # Only encoding_name should be present (from include_names=True)
        assert "encoding_name" in result
        assert len(result) == 1

    def test_data_dependent_encoding_in_comparison(self):
        """compare_resources handles data-dependent encodings via worst-case."""
        from encoding_atlas import AngleEncoding, BasisEncoding

        encodings = [
            AngleEncoding(n_features=4),
            BasisEncoding(n_features=4),
        ]
        result = compare_resources(encodings)

        assert result["encoding_name"][0] == "AngleEncoding"
        assert result["encoding_name"][1] == "BasisEncoding"
        assert len(result["gate_count"]) == 2
        # Both should have non-None gate counts (worst-case for BasisEncoding)
        assert all(gc is not None for gc in result["gate_count"])


# =============================================================================
# Tests for Cross-Function Consistency
# =============================================================================


class TestCrossFunctionConsistency:
    """Tests verifying that different functions agree on values."""

    @pytest.fixture(
        params=[
            ("AngleEncoding", dict(n_features=4)),
            ("IQPEncoding", dict(n_features=4, reps=2, entanglement="full")),
            ("IQPEncoding", dict(n_features=4, reps=1, entanglement="linear")),
            ("ZZFeatureMap", dict(n_features=4, reps=1)),
            ("HardwareEfficientEncoding", dict(n_features=4, reps=2)),
            ("AmplitudeEncoding", dict(n_features=4)),
        ]
    )
    def encoding(self, request):
        import encoding_atlas

        cls_name, kwargs = request.param
        cls = getattr(encoding_atlas, cls_name)
        return cls(**kwargs)

    def test_summary_gate_count_equals_detailed_total(self, encoding):
        """count_resources summary gate_count == detailed total."""
        summary = count_resources(encoding, detailed=False)
        detailed = count_resources(encoding, detailed=True)

        assert summary["gate_count"] == detailed["total"]

    def test_summary_single_qubit_equals_detailed(self, encoding):
        """count_resources summary single_qubit_gates == detailed total_single_qubit."""
        summary = count_resources(encoding, detailed=False)
        detailed = count_resources(encoding, detailed=True)

        assert summary["single_qubit_gates"] == detailed["total_single_qubit"]

    def test_summary_two_qubit_equals_detailed(self, encoding):
        """count_resources summary two_qubit_gates == detailed total_two_qubit."""
        summary = count_resources(encoding, detailed=False)
        detailed = count_resources(encoding, detailed=True)

        assert summary["two_qubit_gates"] == detailed["total_two_qubit"]

    def test_get_gate_breakdown_matches_detailed(self, encoding):
        """get_gate_breakdown must return the same dict as count_resources(detailed=True)."""
        breakdown = get_gate_breakdown(encoding)
        detailed = count_resources(encoding, detailed=True)

        for key in detailed:
            assert breakdown[key] == detailed[key], (
                f"{encoding.__class__.__name__}: "
                f"get_gate_breakdown['{key}']={breakdown[key]} != "
                f"count_resources(detailed=True)['{key}']={detailed[key]}"
            )

    def test_get_resource_summary_matches_properties(self, encoding):
        """get_resource_summary fields must match encoding.properties."""
        summary = get_resource_summary(encoding)
        props = encoding.properties

        assert summary["n_qubits"] == props.n_qubits
        assert summary["depth"] == props.depth
        assert summary["gate_count"] == props.gate_count
        assert summary["single_qubit_gates"] == props.single_qubit_gates
        assert summary["two_qubit_gates"] == props.two_qubit_gates
        assert summary["parameter_count"] == props.parameter_count


# =============================================================================
# Tests for Data-Dependent Encoding (BasisEncoding) — Extended
# =============================================================================


class TestBasisEncodingExtended:
    """Extended tests for data-dependent resource counting with BasisEncoding."""

    @pytest.fixture
    def basis_enc(self):
        from encoding_atlas import BasisEncoding

        return BasisEncoding(n_features=4)

    def test_all_zeros_gate_count_is_zero(self, basis_enc):
        """All-zeros input should produce 0 gates."""
        x = np.array([0.1, 0.2, 0.3, 0.4])  # All below 0.5 threshold
        result = count_resources(basis_enc, x=x)
        assert result["gate_count"] == 0

    def test_all_ones_gate_count_equals_n_features(self, basis_enc):
        """All-ones input should produce n_features gates."""
        x = np.array([0.6, 0.7, 0.8, 0.9])  # All above 0.5 threshold
        result = count_resources(basis_enc, x=x)
        assert result["gate_count"] == 4

    def test_threshold_boundary_below(self, basis_enc):
        """Values exactly at threshold (0.5) should NOT trigger X gates."""
        x = np.array([0.5, 0.5, 0.5, 0.5])  # Exactly at threshold
        result = count_resources(basis_enc, x=x)
        # threshold check is `> threshold`, not `>= threshold`
        assert result["gate_count"] == 0

    def test_threshold_boundary_above(self, basis_enc):
        """Values just above threshold should trigger X gates."""
        x = np.array([0.500001, 0.500001, 0.500001, 0.500001])
        result = count_resources(basis_enc, x=x)
        assert result["gate_count"] == 4

    def test_detailed_only_has_x_gates(self, basis_enc):
        """Detailed breakdown should only have X gates for BasisEncoding."""
        x = np.array([0.1, 0.9, 0.3, 0.8])  # 2 above threshold
        result = count_resources(basis_enc, x=x, detailed=True)

        assert result["x"] == 2
        assert result["rx"] == 0
        assert result["ry"] == 0
        assert result["rz"] == 0
        assert result["h"] == 0
        assert result["cnot"] == 0
        assert result["total_single_qubit"] == 2
        assert result["total_two_qubit"] == 0
        assert result["total"] == 2

    def test_detailed_consistent_with_summary(self, basis_enc):
        """Detailed and summary must agree for data-dependent encoding."""
        x = np.array([0.1, 0.9, 0.3, 0.8])
        summary = count_resources(basis_enc, x=x, detailed=False)
        detailed = count_resources(basis_enc, x=x, detailed=True)

        assert summary["gate_count"] == detailed["total"]
        assert summary["single_qubit_gates"] == detailed["total_single_qubit"]
        assert summary["two_qubit_gates"] == detailed["total_two_qubit"]

    def test_depth_is_constant(self, basis_enc):
        """BasisEncoding depth should not vary with input."""
        x_few = np.array([0.1, 0.9, 0.1, 0.1])
        x_many = np.array([0.9, 0.9, 0.9, 0.9])

        res_few = count_resources(basis_enc, x=x_few)
        res_many = count_resources(basis_enc, x=x_many)

        assert res_few["depth"] == res_many["depth"]

    def test_get_gate_breakdown_requires_x(self, basis_enc):
        """get_gate_breakdown must raise ValidationError without x for BasisEncoding."""
        with pytest.raises(ValidationError, match="data-dependent"):
            get_gate_breakdown(basis_enc)

    def test_get_gate_breakdown_with_x_works(self, basis_enc):
        """get_gate_breakdown with x must work for BasisEncoding."""
        x = np.array([0.1, 0.9, 0.3, 0.8])
        result = get_gate_breakdown(basis_enc, x=x)
        assert result["total"] == 2

    def test_two_qubit_ratio_is_zero(self, basis_enc):
        """BasisEncoding should always have two_qubit_ratio == 0."""
        x = np.array([0.9, 0.9, 0.9, 0.9])
        result = count_resources(basis_enc, x=x)
        assert result["two_qubit_ratio"] == 0.0

    def test_parameter_count_is_zero(self, basis_enc):
        """BasisEncoding should have parameter_count == 0."""
        x = np.array([0.9, 0.9, 0.9, 0.9])
        result = count_resources(basis_enc, x=x)
        assert result["parameter_count"] == 0

    def test_gates_per_qubit_varies_with_input(self, basis_enc):
        """gates_per_qubit should change based on input for BasisEncoding."""
        x_none = np.array([0.1, 0.2, 0.3, 0.4])  # 0 gates
        x_all = np.array([0.9, 0.9, 0.9, 0.9])  # 4 gates

        res_none = count_resources(basis_enc, x=x_none)
        res_all = count_resources(basis_enc, x=x_all)

        assert res_none["gates_per_qubit"] == 0.0
        assert res_all["gates_per_qubit"] == 1.0

    def test_worst_case_from_get_resource_summary(self, basis_enc):
        """get_resource_summary should return worst-case without x."""
        summary = get_resource_summary(basis_enc)

        # Worst case = all qubits get X gates
        assert summary["gate_count"] == basis_enc.n_features
        assert summary["is_data_dependent"] is True


# =============================================================================
# Tests for get_resource_summary — Extended
# =============================================================================


class TestGetResourceSummaryExtended:
    """Extended tests for get_resource_summary return values."""

    @pytest.fixture
    def iqp_enc(self):
        from encoding_atlas import IQPEncoding

        return IQPEncoding(n_features=4, reps=1, entanglement="full")

    def test_all_15_keys_present(self, iqp_enc):
        """All 15 keys of ResourceCountSummary must be present."""
        result = get_resource_summary(iqp_enc)

        expected_keys = {
            "n_qubits",
            "depth",
            "gate_count",
            "single_qubit_gates",
            "two_qubit_gates",
            "parameter_count",
            "cnot_count",
            "cz_count",
            "t_gate_count",
            "hadamard_count",
            "rotation_gates",
            "two_qubit_ratio",
            "gates_per_qubit",
            "encoding_name",
            "is_data_dependent",
        }
        assert expected_keys == set(result.keys())

    def test_cnot_count_approximation(self, iqp_enc):
        """cnot_count should approximate two_qubit_gates."""
        result = get_resource_summary(iqp_enc)
        # In get_resource_summary (not count_resources), cnot_count = two_qubit_gates
        assert result["cnot_count"] == result["two_qubit_gates"]

    def test_cz_count_always_zero(self, iqp_enc):
        """cz_count should be 0 in get_resource_summary (no detailed analysis)."""
        result = get_resource_summary(iqp_enc)
        assert result["cz_count"] == 0

    def test_t_gate_count_always_zero(self, iqp_enc):
        """t_gate_count should be 0 in get_resource_summary."""
        result = get_resource_summary(iqp_enc)
        assert result["t_gate_count"] == 0

    def test_hadamard_count_always_zero(self, iqp_enc):
        """hadamard_count should be 0 in get_resource_summary."""
        result = get_resource_summary(iqp_enc)
        assert result["hadamard_count"] == 0

    def test_rotation_gates_approximation(self, iqp_enc):
        """rotation_gates should approximate single_qubit_gates."""
        result = get_resource_summary(iqp_enc)
        assert result["rotation_gates"] == result["single_qubit_gates"]

    def test_gates_per_qubit_correctness(self, iqp_enc):
        """gates_per_qubit must equal gate_count / n_qubits."""
        result = get_resource_summary(iqp_enc)
        expected = result["gate_count"] / result["n_qubits"]
        assert abs(result["gates_per_qubit"] - expected) < 1e-10

    def test_two_qubit_ratio_correctness(self, iqp_enc):
        """two_qubit_ratio must equal two_qubit_gates / gate_count."""
        result = get_resource_summary(iqp_enc)
        expected = result["two_qubit_gates"] / result["gate_count"]
        assert abs(result["two_qubit_ratio"] - expected) < 1e-10

    def test_return_value_types(self, iqp_enc):
        """Each field must have the correct type."""
        result = get_resource_summary(iqp_enc)

        int_fields = [
            "n_qubits",
            "depth",
            "gate_count",
            "single_qubit_gates",
            "two_qubit_gates",
            "parameter_count",
            "cnot_count",
            "cz_count",
            "t_gate_count",
            "hadamard_count",
            "rotation_gates",
        ]
        for field in int_fields:
            assert isinstance(result[field], int), f"{field} should be int"

        assert isinstance(result["two_qubit_ratio"], float)
        assert isinstance(result["gates_per_qubit"], float)
        assert isinstance(result["encoding_name"], str)
        assert isinstance(result["is_data_dependent"], bool)


# =============================================================================
# Tests for cnot/cx Alias Behavior
# =============================================================================


class TestCnotCxAlias:
    """Tests verifying that cnot and cx are properly aliased."""

    def test_cnot_equals_cx_for_iqp(self):
        """cnot and cx must have the same value for IQPEncoding."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)
        result = count_resources(enc, detailed=True)
        assert result["cnot"] == result["cx"]

    def test_cnot_equals_cx_for_angle(self):
        """cnot and cx must both be 0 for AngleEncoding (no 2Q gates)."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        result = count_resources(enc, detailed=True)
        assert result["cnot"] == 0
        assert result["cx"] == 0
        assert result["cnot"] == result["cx"]

    def test_cnot_equals_cx_for_hardware_efficient(self):
        """cnot and cx must match for HardwareEfficientEncoding."""
        from encoding_atlas import HardwareEfficientEncoding

        enc = HardwareEfficientEncoding(n_features=4, reps=2)
        result = count_resources(enc, detailed=True)
        assert result["cnot"] == result["cx"]

    def test_total_two_qubit_uses_cnot_not_both(self):
        """total_two_qubit should not double-count cnot and cx."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        result = count_resources(enc, detailed=True)

        # total_two_qubit should equal cnot + cz + swap, NOT cnot + cx + cz + swap
        expected_two_qubit = result["cnot"] + result["cz"] + result["swap"]
        assert result["total_two_qubit"] == expected_two_qubit


# =============================================================================
# Tests for Private Helpers — Extended
# =============================================================================


class TestPrivateHelpersExtended:
    """Extended tests for private helper functions."""

    # _safe_divide edge cases
    def test_safe_divide_negative_values(self):
        """Safe divide works with negative numerator."""
        assert _safe_divide(-10, 5) == -2.0

    def test_safe_divide_both_negative(self):
        """Safe divide works with both negative."""
        assert _safe_divide(-10, -5) == 2.0

    def test_safe_divide_float_inputs(self):
        """Safe divide works with float inputs."""
        assert abs(_safe_divide(1.5, 3.0) - 0.5) < 1e-10

    def test_safe_divide_very_large_values(self):
        """Safe divide works with very large values."""
        result = _safe_divide(1e18, 1e9)
        assert abs(result - 1e9) < 1.0

    def test_safe_divide_negative_zero(self):
        """Safe divide handles -0.0 denominator."""
        assert _safe_divide(10, -0.0) == 0.0

    # _validate_encoding edge cases
    def test_validate_encoding_integer(self):
        """Integer is not a valid encoding."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            _validate_encoding(42)

    def test_validate_encoding_dict(self):
        """Dict is not a valid encoding."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            _validate_encoding({"n_qubits": 4})

    def test_validate_encoding_list(self):
        """List is not a valid encoding."""
        with pytest.raises(AnalysisError, match="Expected BaseEncoding"):
            _validate_encoding([1, 2, 3])

    def test_validate_encoding_valid(self):
        """Valid encoding should pass without error."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        _validate_encoding(enc)  # Should not raise

    # _validate_input_data edge cases
    def test_validate_input_data_3d_array(self):
        """3D array should be rejected."""
        with pytest.raises(ValidationError):
            _validate_input_data(np.ones((2, 3, 4)), 4)

    def test_validate_input_data_scalar(self):
        """0D scalar should be rejected."""
        with pytest.raises(ValidationError):
            _validate_input_data(np.float64(0.5), 1)

    def test_validate_input_data_negative_inf(self):
        """Negative infinity should be rejected."""
        with pytest.raises(ValidationError, match="infinite|inf|NaN"):
            _validate_input_data(np.array([0.1, -np.inf, 0.3]), 3)

    def test_validate_input_data_2d_single_sample(self):
        """2D array with shape (1, n) should be accepted."""
        result = _validate_input_data(np.array([[0.1, 0.2, 0.3]]), 3)
        assert result.ndim == 1
        assert len(result) == 3

    def test_validate_input_data_2d_multiple_samples(self):
        """2D array with shape (k>1, n) should be rejected."""
        with pytest.raises(ValidationError, match="single sample"):
            _validate_input_data(np.array([[0.1, 0.2], [0.3, 0.4]]), 2)

    def test_validate_input_data_tuple_input(self):
        """Tuple input should be accepted (converted to array)."""
        result = _validate_input_data((0.1, 0.2, 0.3), 3)
        assert len(result) == 3

    def test_validate_input_data_integer_list(self):
        """Integer list should be accepted (converted to float64)."""
        result = _validate_input_data([0, 1, 0, 1], 4)
        assert result.dtype == np.float64
        assert len(result) == 4

    def test_validate_input_data_empty_array_zero_features(self):
        """Empty array for zero features should match."""
        result = _validate_input_data(np.array([]), 0)
        assert len(result) == 0

    def test_validate_input_data_non_convertible(self):
        """Non-numeric input should raise ValidationError."""
        with pytest.raises(ValidationError, match="Cannot convert"):
            _validate_input_data(["hello", "world"], 2)


# =============================================================================
# Tests for Properties Fallback Path
# =============================================================================


class TestPropertiesFallbackPath:
    """Tests for the _get_summary_from_properties fallback.

    All 16 encodings implement ResourceAnalyzable, so the fallback is never
    reached in normal usage. We test it by mocking the protocol check.
    """

    @pytest.fixture
    def encoding(self):
        from encoding_atlas import AngleEncoding

        return AngleEncoding(n_features=4)

    def test_fallback_summary_returns_correct_structure(self, encoding):
        """Fallback path should return all required summary keys."""
        with (
            patch(
                "encoding_atlas.analysis.resources.is_resource_analyzable",
                return_value=False,
            ),
            patch(
                "encoding_atlas.analysis.resources.is_data_dependent_resource_analyzable",
                return_value=False,
            ),
        ):
            result = count_resources(encoding, detailed=False)

        required_keys = {
            "n_qubits",
            "depth",
            "gate_count",
            "single_qubit_gates",
            "two_qubit_gates",
            "parameter_count",
            "cnot_count",
            "cz_count",
            "t_gate_count",
            "hadamard_count",
            "rotation_gates",
            "two_qubit_ratio",
            "gates_per_qubit",
            "encoding_name",
            "is_data_dependent",
        }
        assert required_keys.issubset(result.keys())

    def test_fallback_summary_matches_properties(self, encoding):
        """Fallback path should use encoding.properties values."""
        with (
            patch(
                "encoding_atlas.analysis.resources.is_resource_analyzable",
                return_value=False,
            ),
            patch(
                "encoding_atlas.analysis.resources.is_data_dependent_resource_analyzable",
                return_value=False,
            ),
        ):
            result = count_resources(encoding, detailed=False)

        props = encoding.properties
        assert result["n_qubits"] == props.n_qubits
        assert result["depth"] == props.depth
        assert result["gate_count"] == props.gate_count

    def test_fallback_detailed_returns_correct_structure(self, encoding):
        """Fallback detailed path should return all required detailed keys."""
        with (
            patch(
                "encoding_atlas.analysis.resources.is_resource_analyzable",
                return_value=False,
            ),
            patch(
                "encoding_atlas.analysis.resources.is_data_dependent_resource_analyzable",
                return_value=False,
            ),
        ):
            result = count_resources(encoding, detailed=True)

        required_keys = {
            "rx",
            "ry",
            "rz",
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "cnot",
            "cx",
            "cz",
            "swap",
            "total_single_qubit",
            "total_two_qubit",
            "total",
            "encoding_name",
        }
        assert required_keys.issubset(result.keys())

    def test_fallback_detailed_totals_consistent(self, encoding):
        """Fallback detailed totals must be consistent."""
        with (
            patch(
                "encoding_atlas.analysis.resources.is_resource_analyzable",
                return_value=False,
            ),
            patch(
                "encoding_atlas.analysis.resources.is_data_dependent_resource_analyzable",
                return_value=False,
            ),
        ):
            result = count_resources(encoding, detailed=True)

        # total must equal sum of all individual gates (using cnot, not cx)
        single_sum = sum(
            [
                result["rx"],
                result["ry"],
                result["rz"],
                result["h"],
                result["x"],
                result["y"],
                result["z"],
                result["s"],
                result["t"],
            ]
        )
        two_sum = result["cnot"] + result["cz"] + result["swap"]

        assert single_sum == result["total_single_qubit"]
        assert two_sum == result["total_two_qubit"]
        assert single_sum + two_sum == result["total"]

    def test_fallback_detailed_attributes_all_1q_to_ry(self, encoding):
        """Fallback assumes all single-qubit gates are RY."""
        with (
            patch(
                "encoding_atlas.analysis.resources.is_resource_analyzable",
                return_value=False,
            ),
            patch(
                "encoding_atlas.analysis.resources.is_data_dependent_resource_analyzable",
                return_value=False,
            ),
        ):
            result = count_resources(encoding, detailed=True)

        props = encoding.properties
        assert result["ry"] == props.single_qubit_gates
        assert result["rx"] == 0
        assert result["rz"] == 0

    def test_fallback_detailed_attributes_all_2q_to_cnot(self, encoding):
        """Fallback assumes all two-qubit gates are CNOT."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)

        with (
            patch(
                "encoding_atlas.analysis.resources.is_resource_analyzable",
                return_value=False,
            ),
            patch(
                "encoding_atlas.analysis.resources.is_data_dependent_resource_analyzable",
                return_value=False,
            ),
        ):
            result = count_resources(enc, detailed=True)

        props = enc.properties
        assert result["cnot"] == props.two_qubit_gates
        assert result["cz"] == 0
        assert result["swap"] == 0


# =============================================================================
# Tests for Entangling vs Non-Entangling Encoding Properties
# =============================================================================


class TestEntanglingVsNonEntangling:
    """Tests that verify resource differences between entangling and
    non-entangling encodings."""

    def test_non_entangling_has_zero_two_qubit(self):
        """Non-entangling encodings should have zero two-qubit gates."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        result = count_resources(enc)

        assert result["two_qubit_gates"] == 0
        assert result["two_qubit_ratio"] == 0.0

    def test_entangling_has_nonzero_two_qubit(self):
        """Entangling encodings should have non-zero two-qubit gates."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        result = count_resources(enc)

        assert result["two_qubit_gates"] > 0
        assert result["two_qubit_ratio"] > 0.0

    def test_angle_encoding_rotation_axes(self):
        """Test AngleEncoding with different rotation axes."""
        from encoding_atlas import AngleEncoding

        for axis in ["X", "Y", "Z"]:
            enc = AngleEncoding(n_features=4, rotation=axis)
            result = count_resources(enc, detailed=True)

            # Only the selected axis should have non-zero count
            if axis == "X":
                assert result["rx"] == 4
                assert result["ry"] == 0
                assert result["rz"] == 0
            elif axis == "Y":
                assert result["rx"] == 0
                assert result["ry"] == 4
                assert result["rz"] == 0
            else:  # Z
                assert result["rx"] == 0
                assert result["ry"] == 0
                assert result["rz"] == 4

    def test_angle_encoding_reps_scaling(self):
        """Gate count should scale linearly with reps for AngleEncoding."""
        from encoding_atlas import AngleEncoding

        enc_1 = AngleEncoding(n_features=4, reps=1)
        enc_2 = AngleEncoding(n_features=4, reps=2)
        enc_3 = AngleEncoding(n_features=4, reps=3)

        r1 = count_resources(enc_1)
        r2 = count_resources(enc_2)
        r3 = count_resources(enc_3)

        assert r2["gate_count"] == 2 * r1["gate_count"]
        assert r3["gate_count"] == 3 * r1["gate_count"]

    def test_iqp_entanglement_patterns_gate_ordering(self):
        """full > circular > linear for IQP two-qubit gate count."""
        from encoding_atlas import IQPEncoding

        enc_full = IQPEncoding(n_features=6, reps=1, entanglement="full")
        enc_circ = IQPEncoding(n_features=6, reps=1, entanglement="circular")
        enc_lin = IQPEncoding(n_features=6, reps=1, entanglement="linear")

        r_full = count_resources(enc_full)
        r_circ = count_resources(enc_circ)
        r_lin = count_resources(enc_lin)

        assert r_full["two_qubit_gates"] > r_circ["two_qubit_gates"]
        assert r_circ["two_qubit_gates"] > r_lin["two_qubit_gates"]


# =============================================================================
# Tests for Specific Encoding Gate Breakdowns
# =============================================================================


class TestSpecificEncodingBreakdowns:
    """Tests verifying specific gate compositions for known encodings."""

    def test_iqp_has_hadamard_and_cnot(self):
        """IQPEncoding must have Hadamard and CNOT gates."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        result = count_resources(enc, detailed=True)

        assert result["h"] > 0, "IQP should have Hadamard gates"
        assert result["cnot"] > 0, "IQP should have CNOT gates"
        assert result["rz"] > 0, "IQP should have RZ gates"

    def test_iqp_detailed_h_count(self):
        """IQP Hadamard count should be n_features * reps."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)
        result = count_resources(enc, detailed=True)

        assert result["h"] == 4 * 2  # n_features * reps

    def test_amplitude_encoding_resources(self):
        """AmplitudeEncoding resource counting should work."""
        from encoding_atlas import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        result = count_resources(enc)

        assert result["n_qubits"] >= 2  # log2(4) = 2
        assert result["gate_count"] > 0

    def test_higher_order_angle_encoding(self):
        """HigherOrderAngleEncoding with order=2 should produce more gates."""
        from encoding_atlas import AngleEncoding, HigherOrderAngleEncoding

        enc_simple = AngleEncoding(n_features=4)
        enc_higher = HigherOrderAngleEncoding(n_features=4, order=2)

        r_simple = count_resources(enc_simple)
        r_higher = count_resources(enc_higher)

        # Higher order should use at least as many gates
        assert r_higher["gate_count"] >= r_simple["gate_count"]


# =============================================================================
# Tests for estimate_execution_time — Extended
# =============================================================================


class TestEstimateExecutionTimeExtended:
    """Extended tests for estimate_execution_time edge cases."""

    def test_all_zero_gate_times_produce_zero_serial(self):
        """All-zero gate times should give zero serial time."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)

        result = estimate_execution_time(
            enc,
            single_qubit_gate_time_us=0.0,
            two_qubit_gate_time_us=0.0,
            measurement_time_us=0.0,
            include_measurement=False,
        )
        assert result["serial_time_us"] == 0.0

    def test_measurement_excluded_when_flag_false(self):
        """include_measurement=False should zero out measurement time."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)

        result = estimate_execution_time(enc, include_measurement=False)
        assert result["measurement_time_us"] == 0.0

    def test_various_encodings_produce_positive_times(self):
        """All encoding types should produce positive timing estimates."""
        from encoding_atlas import (
            AngleEncoding,
            DataReuploading,
            HardwareEfficientEncoding,
            IQPEncoding,
            ZZFeatureMap,
        )

        for enc in [
            AngleEncoding(n_features=4),
            IQPEncoding(n_features=4, reps=1),
            ZZFeatureMap(n_features=4, reps=1),
            HardwareEfficientEncoding(n_features=4, reps=1),
            DataReuploading(n_features=4, n_layers=1),
        ]:
            result = estimate_execution_time(enc)
            assert (
                result["serial_time_us"] > 0
            ), f"{enc.__class__.__name__} serial_time should be positive"

    def test_trapped_ion_gate_times(self):
        """Trapped ion gate times (slow 2Q) should produce longer times."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)

        result_sc = estimate_execution_time(enc)  # superconducting defaults
        result_ti = estimate_execution_time(
            enc,
            single_qubit_gate_time_us=1.0,
            two_qubit_gate_time_us=100.0,
            measurement_time_us=10.0,
        )

        assert result_ti["serial_time_us"] > result_sc["serial_time_us"]


# =============================================================================
# Tests for compare_resources — Extended
# =============================================================================


class TestCompareResourcesExtended:
    """Extended tests for compare_resources edge cases."""

    def test_mixed_encoding_types(self):
        """compare_resources works with a diverse mix of encoding types."""
        from encoding_atlas import (
            AmplitudeEncoding,
            AngleEncoding,
            HardwareEfficientEncoding,
            IQPEncoding,
        )

        encodings = [
            AngleEncoding(n_features=4),
            IQPEncoding(n_features=4, reps=1),
            AmplitudeEncoding(n_features=4),
            HardwareEfficientEncoding(n_features=4, reps=1),
        ]
        result = compare_resources(encodings)

        assert len(result["gate_count"]) == 4
        assert len(result["encoding_name"]) == 4

    def test_all_default_metrics_present(self):
        """Default metrics should include the standard 8 metrics."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)

        result = compare_resources([enc])

        default_metrics = [
            "n_qubits",
            "depth",
            "gate_count",
            "single_qubit_gates",
            "two_qubit_gates",
            "parameter_count",
            "two_qubit_ratio",
            "gates_per_qubit",
        ]
        for metric in default_metrics:
            assert metric in result, f"Default metric '{metric}' missing"

    def test_duplicate_metrics(self):
        """Duplicate metric names cause repeated appends to the same key.

        This is a known quirk: the dict comprehension deduplicates keys,
        but the iteration loop still runs once per occurrence in the input
        list, appending to the same list each time.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)

        result = compare_resources([enc], metrics=["gate_count", "gate_count"])
        # The inner loop iterates twice for "gate_count", appending twice
        assert len(result["gate_count"]) == 2
        assert result["gate_count"][0] == result["gate_count"][1]

    def test_include_names_true_adds_encoding_name(self):
        """include_names=True should add encoding_name key."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)

        result = compare_resources([enc], include_names=True)
        assert "encoding_name" in result

    def test_include_names_false_no_encoding_name(self):
        """include_names=False should not add encoding_name key."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)

        result = compare_resources([enc], include_names=False)
        assert "encoding_name" not in result

    def test_large_comparison(self):
        """compare_resources works with many encodings."""
        from encoding_atlas import AngleEncoding

        encodings = [AngleEncoding(n_features=i) for i in range(1, 11)]
        result = compare_resources(encodings)

        assert len(result["gate_count"]) == 10
        # Gate count should increase with n_features for AngleEncoding
        for i in range(1, 10):
            assert result["gate_count"][i] >= result["gate_count"][i - 1]


# =============================================================================
# Tests for Detailed Gate Breakdown Return Value Types
# =============================================================================


class TestDetailedBreakdownTypes:
    """Tests that all values in DetailedGateBreakdown have correct types."""

    def test_all_gate_counts_are_int(self):
        """All gate count fields must be int."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        result = count_resources(enc, detailed=True)

        int_fields = [
            "rx",
            "ry",
            "rz",
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "cnot",
            "cx",
            "cz",
            "swap",
            "total_single_qubit",
            "total_two_qubit",
            "total",
        ]
        for field in int_fields:
            assert isinstance(
                result[field], int
            ), f"Field '{field}' should be int, got {type(result[field]).__name__}"

    def test_encoding_name_is_str(self):
        """encoding_name must be a str."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        result = count_resources(enc, detailed=True)

        assert isinstance(result["encoding_name"], str)


# =============================================================================
# Tests for Idempotency and Determinism
# =============================================================================


class TestIdempotencyAndDeterminism:
    """Tests that resource counting is deterministic and repeatable."""

    def test_count_resources_deterministic(self):
        """Multiple calls should return identical results."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)

        results = [count_resources(enc) for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]

    def test_detailed_deterministic(self):
        """Multiple detailed calls should return identical results."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)

        results = [count_resources(enc, detailed=True) for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]

    def test_basis_encoding_deterministic_with_same_input(self):
        """BasisEncoding with same input should return identical results."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)
        x = np.array([0.1, 0.9, 0.3, 0.8])

        results = [count_resources(enc, x=x) for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]

    def test_different_encoding_instances_same_config(self):
        """Two separate instances with same config should give same results."""
        from encoding_atlas import IQPEncoding

        enc1 = IQPEncoding(n_features=4, reps=2, entanglement="full")
        enc2 = IQPEncoding(n_features=4, reps=2, entanglement="full")

        r1 = count_resources(enc1)
        r2 = count_resources(enc2)

        assert r1 == r2


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestIntegrationExtended:
    """Extended integration tests combining multiple functions."""

    def test_compare_then_breakdown_consistency(self):
        """Values from compare_resources should match individual breakdowns."""
        from encoding_atlas import AngleEncoding, IQPEncoding

        encodings = [
            AngleEncoding(n_features=4),
            IQPEncoding(n_features=4, reps=1),
        ]

        comparison = compare_resources(encodings)

        for i, enc in enumerate(encodings):
            summary = get_resource_summary(enc)
            assert comparison["gate_count"][i] == summary["gate_count"]
            assert comparison["n_qubits"][i] == summary["n_qubits"]
            assert comparison["depth"][i] == summary["depth"]

    def test_full_workflow_all_functions(self):
        """Test the complete analysis workflow end-to-end."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)

        # Step 1: Quick summary
        summary = get_resource_summary(enc)
        assert summary["gate_count"] > 0

        # Step 2: Full count
        resources = count_resources(enc)
        assert resources["gate_count"] == summary["gate_count"]

        # Step 3: Detailed breakdown
        breakdown = get_gate_breakdown(enc)
        assert breakdown["total"] == resources["gate_count"]

        # Step 4: Time estimate
        timing = estimate_execution_time(enc)
        assert timing["serial_time_us"] > 0

        # Step 5: Comparison
        comparison = compare_resources([enc])
        assert comparison["gate_count"][0] == resources["gate_count"]

    def test_data_dependent_full_workflow(self):
        """Test complete workflow for data-dependent encoding."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)
        x = np.array([0.1, 0.9, 0.3, 0.8])

        # Worst-case summary (no x required)
        summary = get_resource_summary(enc)
        assert summary["gate_count"] == 4  # worst case

        # Actual count (requires x)
        actual = count_resources(enc, x=x)
        assert actual["gate_count"] == 2  # only 2 values > 0.5
        assert actual["gate_count"] <= summary["gate_count"]

        # Detailed breakdown
        detailed = count_resources(enc, x=x, detailed=True)
        assert detailed["x"] == 2
        assert detailed["total"] == 2

        # Timing uses worst-case (from summary)
        timing = estimate_execution_time(enc)
        assert timing["serial_time_us"] > 0

    def test_x_provided_but_not_needed_is_ignored(self):
        """x parameter should be silently ignored for non-data-dependent encodings."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        result_with_x = count_resources(enc, x=x)
        result_without_x = count_resources(enc)

        assert result_with_x == result_without_x


# =============================================================================
# Tests for _validate_encoding Defensive Branches (Mock-Based)
# =============================================================================


class TestValidateEncodingDefensiveBranches:
    """Tests for _validate_encoding edge cases that can't be triggered with
    real encodings (since BaseEncoding.__init__ already validates).

    These test defensive branches using mocks to ensure robustness
    against future encoding implementations or corrupted state.
    """

    def _make_mock_encoding(self, n_features=4, n_qubits=4, depth=1):
        """Create a minimal mock that passes isinstance(enc, BaseEncoding).

        Uses a real AngleEncoding and patches its properties to return
        controlled values.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        return enc

    def test_n_qubits_less_than_one(self):
        """Encoding with n_qubits < 1 should raise AnalysisError."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with (
            patch.object(
                type(enc), "n_qubits", new_callable=PropertyMock, return_value=0
            ),
            pytest.raises(AnalysisError, match="n_qubits=0"),
        ):
            _validate_encoding(enc)

    def test_n_features_less_than_one(self):
        """Encoding with n_features < 1 should raise AnalysisError."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with (
            patch.object(
                type(enc), "n_features", new_callable=PropertyMock, return_value=0
            ),
            pytest.raises(AnalysisError, match="n_features=0"),
        ):
            _validate_encoding(enc)

    def test_negative_depth(self):
        """Encoding with depth < 0 should raise AnalysisError."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with (
            patch.object(
                type(enc), "depth", new_callable=PropertyMock, return_value=-1
            ),
            pytest.raises(AnalysisError, match="depth=-1"),
        ):
            _validate_encoding(enc)

    def test_property_access_raises_exception(self):
        """Encoding whose properties raise should produce AnalysisError."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with (
            patch.object(
                type(enc),
                "n_qubits",
                new_callable=PropertyMock,
                side_effect=RuntimeError("corrupted state"),
            ),
            pytest.raises(AnalysisError, match="invalid properties"),
        ):
            _validate_encoding(enc)

    def test_error_details_include_encoding_name(self):
        """AnalysisError for n_qubits < 1 should include encoding name in details."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with patch.object(
            type(enc), "n_qubits", new_callable=PropertyMock, return_value=0
        ):
            with pytest.raises(AnalysisError) as exc_info:
                _validate_encoding(enc)

            assert exc_info.value.details["encoding"] == "AngleEncoding"
            assert exc_info.value.details["n_qubits"] == 0

    def test_error_details_for_n_features(self):
        """AnalysisError for n_features < 1 should include details."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with patch.object(
            type(enc), "n_features", new_callable=PropertyMock, return_value=-1
        ):
            with pytest.raises(AnalysisError) as exc_info:
                _validate_encoding(enc)

            assert exc_info.value.details["n_features"] == -1

    def test_error_details_for_depth(self):
        """AnalysisError for depth < 0 should include details."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with patch.object(
            type(enc), "depth", new_callable=PropertyMock, return_value=-5
        ):
            with pytest.raises(AnalysisError) as exc_info:
                _validate_encoding(enc)

            assert exc_info.value.details["depth"] == -5

    def test_property_access_error_includes_original_message(self):
        """AnalysisError from property access failure should include original error."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        with (
            patch.object(
                type(enc),
                "n_features",
                new_callable=PropertyMock,
                side_effect=AttributeError("missing attribute"),
            ),
            pytest.raises(AnalysisError, match="missing attribute"),
        ):
            _validate_encoding(enc)


# =============================================================================
# Tests for Data-Dependent Fallback Paths
# =============================================================================


class TestDataDependentFallbackPaths:
    """Tests for fallback branches in _count_data_dependent_resources.

    These test the exception handling paths when resource_summary() or
    actual_gate_count() behave unexpectedly.

    Note: BasisEncoding uses __slots__, so we must patch on the CLASS
    (not the instance) to override methods.
    """

    @pytest.fixture
    def basis_enc(self):
        from encoding_atlas import BasisEncoding

        return BasisEncoding(n_features=4)

    def test_resource_summary_no_arg_fallback_failure_logs_warning(
        self, basis_enc, caplog
    ):
        """When both resource_summary(x) and resource_summary() fail,
        a warning should be logged and resource_dict should be empty."""
        from encoding_atlas import BasisEncoding

        x = np.array([0.1, 0.9, 0.3, 0.8])

        call_count = 0

        def broken_resource_summary(self_inner, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("unexpected arg")
            raise RuntimeError("broken method")

        with patch.object(BasisEncoding, "resource_summary", broken_resource_summary):
            import logging

            with caplog.at_level(
                logging.WARNING, logger="encoding_atlas.analysis.resources"
            ):
                result = count_resources(basis_enc, x=x)

        assert "Failed to get resource_summary" in caplog.text
        assert result["gate_count"] >= 0

    def test_actual_gate_count_failure_falls_back_to_properties(
        self, basis_enc, caplog
    ):
        """When actual_gate_count(x) fails, should fall back to properties
        and log a warning."""
        from encoding_atlas import BasisEncoding

        x = np.array([0.1, 0.9, 0.3, 0.8])

        def broken_actual_gate_count(self_inner, x_inner):
            raise RuntimeError("computation failed")

        with patch.object(BasisEncoding, "actual_gate_count", broken_actual_gate_count):
            import logging

            with caplog.at_level(
                logging.WARNING, logger="encoding_atlas.analysis.resources"
            ):
                result = count_resources(basis_enc, x=x)

        assert "Failed to get actual_gate_count" in caplog.text
        assert result["gate_count"] == basis_enc.properties.gate_count

    def test_no_resource_summary_method_still_works(self, basis_enc):
        """If hasattr(encoding, 'resource_summary') returns False, should
        still work via actual_gate_count."""
        x = np.array([0.1, 0.9, 0.3, 0.8])

        original_hasattr = builtins.hasattr

        def patched_hasattr(obj, name):
            if name == "resource_summary" and isinstance(obj, type(basis_enc)):
                return False
            return original_hasattr(obj, name)

        with patch.object(builtins, "hasattr", side_effect=patched_hasattr):
            result = count_resources(basis_enc, x=x)

        assert result["gate_count"] >= 0

    def test_no_actual_gate_count_method_defaults_to_zero(self, basis_enc):
        """If hasattr(encoding, 'actual_gate_count') returns False,
        actual_gates defaults to 0."""
        x = np.array([0.1, 0.9, 0.3, 0.8])

        original_hasattr = builtins.hasattr

        def patched_hasattr(obj, name):
            if name == "actual_gate_count" and isinstance(obj, type(basis_enc)):
                return False
            return original_hasattr(obj, name)

        with patch.object(builtins, "hasattr", side_effect=patched_hasattr):
            result = count_resources(basis_enc, x=x)

        assert result["gate_count"] == 0

    def test_data_dependent_detailed_with_actual_gate_count_failure(
        self, basis_enc, caplog
    ):
        """Detailed breakdown should work even when actual_gate_count fails."""
        from encoding_atlas import BasisEncoding

        x = np.array([0.1, 0.9, 0.3, 0.8])

        def broken_actual_gate_count(self_inner, x_inner):
            raise RuntimeError("boom")

        with patch.object(BasisEncoding, "actual_gate_count", broken_actual_gate_count):
            import logging

            with caplog.at_level(
                logging.WARNING, logger="encoding_atlas.analysis.resources"
            ):
                result = count_resources(basis_enc, x=x, detailed=True)

        assert result["total"] == basis_enc.properties.gate_count
        assert result["x"] == basis_enc.properties.gate_count


# =============================================================================
# Regression Tests for Bug Fixes
# =============================================================================


class TestBugFixRegressions:
    """Regression tests for specific bugs that were fixed.

    Each test documents the original bug and verifies the fix.
    """

    def test_compare_resources_encoding_name_not_duplicated(self):
        """Bug #1: When encoding_name is in metrics AND include_names=True,
        encoding_name should NOT be duplicated.

        Before fix: compare_resources([enc], metrics=["encoding_name"])
        would produce {"encoding_name": ["AngleEncoding", "AngleEncoding"]}
        because both the include_names logic and the metrics loop appended.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        result = compare_resources([enc], metrics=["encoding_name"], include_names=True)

        # Should have exactly 1 entry per encoding, not 2
        assert len(result["encoding_name"]) == 1
        assert result["encoding_name"][0] == "AngleEncoding"

    def test_compare_resources_encoding_name_in_metrics_without_include_names(self):
        """encoding_name in metrics with include_names=False should still work."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        result = compare_resources(
            [enc], metrics=["encoding_name"], include_names=False
        )

        assert len(result["encoding_name"]) == 1
        assert result["encoding_name"][0] == "AngleEncoding"

    def test_compare_resources_encoding_name_with_multiple_encodings(self):
        """Bug #1 regression: multiple encodings should each appear once."""
        from encoding_atlas import AngleEncoding, IQPEncoding

        encodings = [
            AngleEncoding(n_features=4),
            IQPEncoding(n_features=4, reps=1),
        ]
        result = compare_resources(
            encodings, metrics=["encoding_name", "gate_count"], include_names=True
        )

        assert len(result["encoding_name"]) == 2
        assert result["encoding_name"][0] == "AngleEncoding"
        assert result["encoding_name"][1] == "IQPEncoding"
        assert len(result["gate_count"]) == 2

    def test_compare_resources_is_data_dependent_as_metric(self):
        """is_data_dependent as a metric should not cause issues."""
        from encoding_atlas import AngleEncoding, BasisEncoding

        encodings = [AngleEncoding(n_features=4), BasisEncoding(n_features=4)]
        result = compare_resources(
            encodings, metrics=["is_data_dependent", "gate_count"]
        )

        assert len(result["is_data_dependent"]) == 2
        assert result["is_data_dependent"][0] is False
        assert result["is_data_dependent"][1] is True

    def test_critical_path_single_qubit_only_uses_single_qubit_time(self):
        """Bug #3: For single-qubit-only circuits (no 2Q gates), the
        critical path should use single_qubit_gate_time_us, not
        two_qubit_gate_time_us.

        Before fix: AngleEncoding critical path was depth * two_qubit_time,
        giving an inflated time for circuits with zero two-qubit gates.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4, reps=1)
        summary = get_resource_summary(enc)
        assert summary["two_qubit_gates"] == 0  # Precondition

        single_qubit_time = 0.05
        two_qubit_time = 10.0  # Very large — should NOT affect result
        meas_time = 0.0

        result = estimate_execution_time(
            enc,
            single_qubit_gate_time_us=single_qubit_time,
            two_qubit_gate_time_us=two_qubit_time,
            measurement_time_us=meas_time,
            include_measurement=False,
            parallelization_factor=1.0,
        )

        # Critical path = depth * single_qubit_time (NOT two_qubit_time)
        expected_critical_path = summary["depth"] * single_qubit_time
        assert abs(result["estimated_time_us"] - expected_critical_path) < 1e-10, (
            f"Expected critical path {expected_critical_path:.4f} μs "
            f"(depth={summary['depth']} * single_qubit={single_qubit_time}), "
            f"got {result['estimated_time_us']:.4f} μs. "
            f"The two_qubit_time ({two_qubit_time}) should NOT be used."
        )

    def test_critical_path_with_two_qubit_gates_uses_max_gate_time(self):
        """For circuits with 2Q gates, critical path uses max(1Q, 2Q) time."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        summary = get_resource_summary(enc)
        assert summary["two_qubit_gates"] > 0  # Precondition

        single_qubit_time = 5.0  # Deliberately larger than two_qubit_time
        two_qubit_time = 1.0
        meas_time = 0.0

        result = estimate_execution_time(
            enc,
            single_qubit_gate_time_us=single_qubit_time,
            two_qubit_gate_time_us=two_qubit_time,
            measurement_time_us=meas_time,
            include_measurement=False,
            parallelization_factor=1.0,
        )

        # Critical path should use max(5.0, 1.0) = 5.0 per layer
        expected_critical_path = summary["depth"] * max(
            single_qubit_time, two_qubit_time
        )
        assert result["estimated_time_us"] >= expected_critical_path - 1e-10

    def test_silent_exception_swallowing_now_logs_warning(self, caplog):
        """Bug #2: Inner except in _count_data_dependent_resources should
        now log a warning instead of silently swallowing."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)
        x = np.array([0.1, 0.9, 0.3, 0.8])

        call_count = 0

        def broken_resource_summary(self_inner, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("no x support")
            raise ValueError("internal error")

        with patch.object(BasisEncoding, "resource_summary", broken_resource_summary):
            import logging

            with caplog.at_level(
                logging.WARNING, logger="encoding_atlas.analysis.resources"
            ):
                count_resources(enc, x=x)

        assert "Failed to get resource_summary" in caplog.text
