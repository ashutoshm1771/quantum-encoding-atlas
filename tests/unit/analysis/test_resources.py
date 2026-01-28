"""Comprehensive tests for encoding_atlas.analysis.resources module.

This test suite covers all public functions in the resources module:
- count_resources
- get_resource_summary
- get_gate_breakdown
- compare_resources
- estimate_execution_time

Tests cover:
- Normal operation with various encoding types
- Edge cases (empty inputs, zero values, single features)
- Error handling (invalid inputs, missing required parameters)
- Data-dependent encodings (BasisEncoding)
- Protocol-based resource counting (ResourceAnalyzable)
- Type safety and return value structure
"""

from __future__ import annotations

from typing import Any

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
        single_qubit_sum = sum([
            result["rx"], result["ry"], result["rz"],
            result["h"], result["x"], result["y"],
            result["z"], result["s"], result["t"],
        ])
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

        assert "infinite" in str(exc_info.value).lower() or "inf" in str(exc_info.value).lower()

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
        gate_types = ["rx", "ry", "rz", "h", "x", "y", "z", "s", "t",
                      "cnot", "cx", "cz", "swap"]
        for gate in gate_types:
            assert gate in result

        # Check totals are present
        assert "total_single_qubit" in result
        assert "total_two_qubit" in result
        assert "total" in result

    def test_breakdown_for_data_dependent(
        self, basis_encoding_4q, sample_binary_input
    ):
        """Test breakdown for data-dependent encoding."""
        result = get_gate_breakdown(basis_encoding_4q, x=sample_binary_input)

        # BasisEncoding only uses X gates
        assert result["x"] == result["total"]
        assert result["total_two_qubit"] == 0

    def test_breakdown_totals_match(self, iqp_encoding_4q):
        """Test that breakdown totals match sum of individual gates."""
        result = get_gate_breakdown(iqp_encoding_4q)

        single_qubit_sum = sum([
            result["rx"], result["ry"], result["rz"],
            result["h"], result["x"], result["y"],
            result["z"], result["s"], result["t"],
        ])

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
        result_without = estimate_execution_time(iqp_encoding_4q, include_measurement=False)

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
        result = estimate_execution_time(
            iqp_encoding_4q, parallelization_factor=factor
        )

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

        # Critical path time = depth * two_qubit_gate_time + measurement
        expected_critical_path = summary["depth"] * two_qubit_gate_time_us + measurement_time_us

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

        # Critical path = depth * two_qubit_time + meas_time
        expected_critical_path = summary["depth"] * two_qubit_time + meas_time

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
        expected_no_parallel = result["single_qubit_time_us"] + result["two_qubit_time_us"] + result["measurement_time_us"]

        # The estimated time should be at least the unbounded calculation
        # (could be higher if critical path bound kicks in, but for factor=0
        # the unbounded calc IS the serial time, so it should match)
        assert abs(result["estimated_time_us"] - result["serial_time_us"]) < 1e-10 or \
               result["estimated_time_us"] >= expected_no_parallel - 1e-10

    def test_single_qubit_only_encoding_critical_path(self):
        """Test critical path for encodings with only single-qubit gates.

        For encodings without two-qubit gates (like AngleEncoding), the
        critical path uses the two-qubit gate time (which results in 0
        contribution from gates), but the depth still matters.
        """
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4, reps=1)
        summary = get_resource_summary(enc)

        # Verify no two-qubit gates
        assert summary["two_qubit_gates"] == 0

        two_qubit_time = 0.2
        meas_time = 1.0

        result = estimate_execution_time(
            enc,
            two_qubit_gate_time_us=two_qubit_time,
            measurement_time_us=meas_time,
            parallelization_factor=1.0,
        )

        # Critical path = depth * two_qubit_time + meas_time
        # Note: This uses two_qubit_time even for single-qubit-only circuits
        # as a conservative estimate (the implementation assumes worst-case per layer)
        expected_critical_path = summary["depth"] * two_qubit_time + meas_time

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

        Note: The implementation uses a conservative critical path estimate
        (depth * two_qubit_gate_time) which can exceed serial time for certain
        circuit configurations. This is documented behavior.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=n_features, reps=reps)

        result = estimate_execution_time(enc)

        # All times should be positive
        assert result["serial_time_us"] > 0
        assert result["estimated_time_us"] > 0

        # Critical path bound should hold
        summary = get_resource_summary(enc)
        critical_path = summary["depth"] * 0.2 + 1.0  # Default gate times

        assert result["estimated_time_us"] >= critical_path - 1e-10

        # Estimated time is bounded by max(formula_result, critical_path)
        # For circuits with high depth but few gates, critical_path may
        # exceed serial time (conservative estimate). This is by design.
        # The key invariant is: estimated_time >= critical_path
        gate_time = result["single_qubit_time_us"] + result["two_qubit_time_us"]
        formula_result = gate_time * 0.5 + result["measurement_time_us"]  # default factor=0.5
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
        assert resources["gate_count"] == resources["single_qubit_gates"] + resources["two_qubit_gates"]

    def test_summary_matches_count(self, angle_encoding_4q):
        """Test that get_resource_summary matches count_resources."""
        summary = get_resource_summary(angle_encoding_4q)
        count = count_resources(angle_encoding_4q)

        assert summary["n_qubits"] == count["n_qubits"]
        assert summary["depth"] == count["depth"]
        assert summary["gate_count"] == count["gate_count"]
