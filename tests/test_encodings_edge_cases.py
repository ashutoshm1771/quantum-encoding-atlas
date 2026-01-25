"""
Critical Edge Case Tests - Real-World User Mistakes & Edge Cases
=================================================================
These tests cover scenarios that REAL USERS will encounter but that
the basic test suite might miss.

Categories:
1. Negative values (z-scored data, StandardScaler output)
2. Values outside [0,1] (raw/unnormalized data)
3. Batch processing (multiple samples)
4. Input type handling (lists, integers, etc.)
5. High-dimensional data
6. Outliers and extreme values
7. Common user mistakes

Usage:
    python tests/test_encodings_edge_cases.py
    python tests/test_encodings_edge_cases.py --verbose
"""

import sys
import io
import argparse
import math
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

sys.path.insert(0, 'src')


def _fix_windows_console():
    """Fix Windows console encoding for Unicode (only when run directly)."""
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


@dataclass
class EdgeCaseResult:
    name: str
    category: str
    passed: bool
    behavior: str  # "works", "raises_error", "silent_wrong"
    message: str
    expected_behavior: str
    details: Dict[str, Any] = field(default_factory=dict)


def print_result(result: EdgeCaseResult, verbose: bool = False) -> None:
    if result.passed:
        icon = "[OK]"
    else:
        icon = "[!!]" if result.behavior == "silent_wrong" else "[X]"

    print(f"  {icon} {result.name}")
    print(f"      Behavior: {result.behavior}")
    print(f"      Expected: {result.expected_behavior}")
    if not result.passed or verbose:
        print(f"      Message: {result.message}")
        if result.details:
            for k, v in result.details.items():
                print(f"      {k}: {v}")


# ============================================================================
# 1. NEGATIVE VALUES (Z-SCORED DATA)
# ============================================================================

def test_negative_values() -> List[EdgeCaseResult]:
    """Test encodings with negative values (common from StandardScaler)."""
    from encoding_atlas import get_encoding

    results = []

    # Typical z-scored data (mean=0, std=1)
    z_scored_data = [
        np.array([-1.5, 0.5, 1.2, -0.3]),  # Typical z-scores
        np.array([-2.5, -1.0, 0.0, 1.0, 2.5]),  # Symmetric around 0
        np.array([-0.1, -0.05, 0.0, 0.05, 0.1]),  # Small negative
        np.array([-3.0, -3.0, -3.0, -3.0]),  # All negative
    ]

    encodings_to_test = [
        ("angle", {}),
        ("zz_feature_map", {"reps": 1}),
        ("iqp", {"reps": 1}),
        ("pauli_feature_map", {"paulis": ["Z", "ZZ"]}),
        ("hardware_efficient", {"reps": 1}),
        ("data_reuploading", {"n_layers": 1}),
    ]

    for data_name, x in [("typical_zscore", z_scored_data[0]),
                          ("symmetric_zscore", z_scored_data[1]),
                          ("small_negative", z_scored_data[2]),
                          ("all_negative", z_scored_data[3])]:

        for enc_name, config in encodings_to_test:
            try:
                n_features = len(x)
                encoding = get_encoding(enc_name, n_features=n_features, **config)
                circuit = encoding.get_circuit(x, backend="qiskit")

                # Check if circuit was created
                if circuit is None:
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/{data_name}",
                        category="negative_values",
                        passed=False,
                        behavior="returns_none",
                        message="Circuit is None",
                        expected_behavior="Should either work or raise clear error"
                    ))
                    continue

                # Check for NaN/Inf in parameters
                has_nan = False
                params = []
                for instruction in circuit.data:
                    for param in instruction.operation.params:
                        if isinstance(param, (int, float)):
                            params.append(param)
                            if math.isnan(param) or math.isinf(param):
                                has_nan = True

                if has_nan:
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/{data_name}",
                        category="negative_values",
                        passed=False,
                        behavior="silent_wrong",
                        message="Circuit has NaN/Inf parameters (DANGEROUS!)",
                        expected_behavior="Should handle negative values or raise error",
                        details={"input": list(x), "bad_params": [p for p in params if math.isnan(p) or math.isinf(p)]}
                    ))
                else:
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/{data_name}",
                        category="negative_values",
                        passed=True,
                        behavior="works",
                        message=f"Circuit created with {len(params)} parameters",
                        expected_behavior="Should handle negative values",
                        details={"input": list(x), "depth": circuit.depth()}
                    ))

            except Exception as e:
                error_type = type(e).__name__
                # Raising an error is acceptable IF it's a clear validation error
                is_validation_error = "validation" in str(e).lower() or "negative" in str(e).lower() or "range" in str(e).lower()

                results.append(EdgeCaseResult(
                    name=f"{enc_name}/{data_name}",
                    category="negative_values",
                    passed=is_validation_error,  # Pass if it's a clear error
                    behavior="raises_error",
                    message=f"{error_type}: {str(e)[:100]}",
                    expected_behavior="Should either work OR raise clear validation error",
                    details={"input": list(x)}
                ))

    return results


# ============================================================================
# 2. VALUES OUTSIDE [0,1] - RAW/UNNORMALIZED DATA
# ============================================================================

def test_unnormalized_values() -> List[EdgeCaseResult]:
    """Test with raw data that users forgot to normalize."""
    from encoding_atlas import get_encoding

    results = []

    # Common user mistakes - forgetting to normalize
    raw_data = [
        ("raw_iris", np.array([5.1, 3.5, 1.4, 0.2])),  # Raw Iris (cm)
        ("raw_pixels", np.array([128, 255, 0, 64, 192, 32, 224, 96])),  # Raw pixel values 0-255
        ("large_values", np.array([1000.0, 2500.0, 500.0, 3000.0])),  # Large scale
        ("mixed_scales", np.array([0.001, 1000, 0.5, 500])),  # Mixed scales
        ("angle_radians", np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])),  # Angles in [0, 2pi]
    ]

    encodings_to_test = [
        ("angle", {}),
        ("zz_feature_map", {"reps": 1}),
        ("amplitude", {"normalize": True}),  # Should handle via normalization
        ("amplitude", {"normalize": False}),  # Should fail or warn
    ]

    for data_name, x in raw_data:
        for enc_name, config in encodings_to_test:
            try:
                n_features = len(x)
                # Skip amplitude if not power of 2
                if enc_name == "amplitude" and n_features not in [2, 4, 8]:
                    continue

                encoding = get_encoding(enc_name, n_features=n_features, **config)
                circuit = encoding.get_circuit(x.astype(float), backend="qiskit")

                if circuit is None:
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/{data_name}",
                        category="unnormalized",
                        passed=False,
                        behavior="returns_none",
                        message="Circuit is None",
                        expected_behavior="Should work, warn, or raise clear error"
                    ))
                    continue

                # Check for issues
                has_nan = False
                for instruction in circuit.data:
                    for param in instruction.operation.params:
                        if isinstance(param, (int, float)) and (math.isnan(param) or math.isinf(param)):
                            has_nan = True

                if has_nan:
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/{data_name}",
                        category="unnormalized",
                        passed=False,
                        behavior="silent_wrong",
                        message="NaN/Inf in circuit (user should be warned!)",
                        expected_behavior="Should validate input range or handle gracefully"
                    ))
                else:
                    # Note: Working is fine, but users should understand angles wrap
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/{data_name}",
                        category="unnormalized",
                        passed=True,
                        behavior="works",
                        message=f"Processes large values (angles will wrap mod 2pi)",
                        expected_behavior="Works but user should normalize for best results",
                        details={"max_input": float(np.max(x)), "depth": circuit.depth()}
                    ))

            except Exception as e:
                error_type = type(e).__name__
                results.append(EdgeCaseResult(
                    name=f"{enc_name}/{data_name}",
                    category="unnormalized",
                    passed=True,  # Raising error for bad input is acceptable
                    behavior="raises_error",
                    message=f"{error_type}: {str(e)[:80]}",
                    expected_behavior="Error for invalid input is acceptable"
                ))

    return results


# ============================================================================
# 3. BATCH PROCESSING
# ============================================================================

def test_batch_processing() -> List[EdgeCaseResult]:
    """Test get_circuits with multiple samples."""
    from encoding_atlas import get_encoding

    results = []

    # Batch of samples (2D array)
    batch_sizes = [1, 5, 10, 100]
    n_features = 4

    encodings_to_test = [
        ("angle", {}),
        ("zz_feature_map", {"reps": 1}),
        ("iqp", {"reps": 1}),
        ("hardware_efficient", {"reps": 1}),
    ]

    for batch_size in batch_sizes:
        X = np.random.rand(batch_size, n_features)

        for enc_name, config in encodings_to_test:
            try:
                encoding = get_encoding(enc_name, n_features=n_features, **config)
                circuits = encoding.get_circuits(X, backend="qiskit")

                # Verify we got the right number of circuits
                if len(circuits) != batch_size:
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/batch_{batch_size}",
                        category="batch",
                        passed=False,
                        behavior="wrong_count",
                        message=f"Expected {batch_size} circuits, got {len(circuits)}",
                        expected_behavior=f"Should return {batch_size} circuits"
                    ))
                else:
                    # Verify each circuit is valid
                    all_valid = all(c is not None and c.num_qubits > 0 for c in circuits)
                    results.append(EdgeCaseResult(
                        name=f"{enc_name}/batch_{batch_size}",
                        category="batch",
                        passed=all_valid,
                        behavior="works" if all_valid else "some_invalid",
                        message=f"Created {len(circuits)} circuits",
                        expected_behavior="Each sample should produce valid circuit"
                    ))

            except Exception as e:
                results.append(EdgeCaseResult(
                    name=f"{enc_name}/batch_{batch_size}",
                    category="batch",
                    passed=False,
                    behavior="raises_error",
                    message=f"{type(e).__name__}: {str(e)[:80]}",
                    expected_behavior="Should handle batched input"
                ))

    return results


# ============================================================================
# 4. INPUT TYPE HANDLING
# ============================================================================

def test_input_types() -> List[EdgeCaseResult]:
    """Test with various input types users might provide."""
    from encoding_atlas import get_encoding

    results = []

    # Different input types users might try
    inputs = [
        ("python_list", [0.5, 0.3, 0.7, 0.2]),
        ("numpy_float64", np.array([0.5, 0.3, 0.7, 0.2], dtype=np.float64)),
        ("numpy_float32", np.array([0.5, 0.3, 0.7, 0.2], dtype=np.float32)),
        ("numpy_int", np.array([1, 0, 1, 0], dtype=np.int32)),
        ("python_ints", [1, 0, 1, 0]),
        ("mixed_int_float", [0.5, 1, 0.3, 0]),
        ("tuple", (0.5, 0.3, 0.7, 0.2)),
    ]

    encoding = get_encoding("angle", n_features=4, rotation="Y")

    for type_name, x in inputs:
        try:
            circuit = encoding.get_circuit(x, backend="qiskit")

            if circuit is None:
                results.append(EdgeCaseResult(
                    name=f"input_type/{type_name}",
                    category="input_type",
                    passed=False,
                    behavior="returns_none",
                    message="Circuit is None",
                    expected_behavior="Should handle common input types"
                ))
            else:
                results.append(EdgeCaseResult(
                    name=f"input_type/{type_name}",
                    category="input_type",
                    passed=True,
                    behavior="works",
                    message=f"Accepted {type_name}, circuit depth={circuit.depth()}",
                    expected_behavior="Should convert to float array internally"
                ))

        except Exception as e:
            results.append(EdgeCaseResult(
                name=f"input_type/{type_name}",
                category="input_type",
                passed=False,
                behavior="raises_error",
                message=f"{type(e).__name__}: {str(e)[:80]}",
                expected_behavior="Should handle or give clear error"
            ))

    return results


# ============================================================================
# 5. HIGH DIMENSIONAL DATA
# ============================================================================

def test_high_dimensional() -> List[EdgeCaseResult]:
    """Test with high-dimensional data (many features)."""
    from encoding_atlas import get_encoding

    results = []

    dimensions = [10, 20, 50]  # Skip very large to keep tests fast

    for n_features in dimensions:
        x = np.random.rand(n_features)

        # Angle encoding scales linearly
        try:
            encoding = get_encoding("angle", n_features=n_features)
            circuit = encoding.get_circuit(x, backend="qiskit")

            results.append(EdgeCaseResult(
                name=f"angle/dim_{n_features}",
                category="high_dim",
                passed=circuit is not None,
                behavior="works" if circuit else "fails",
                message=f"{n_features} qubits, depth={circuit.depth() if circuit else 'N/A'}",
                expected_behavior="Should scale to high dimensions"
            ))
        except Exception as e:
            results.append(EdgeCaseResult(
                name=f"angle/dim_{n_features}",
                category="high_dim",
                passed=False,
                behavior="raises_error",
                message=str(e)[:80],
                expected_behavior="Should handle high dimensions"
            ))

        # ZZ feature map - O(n^2) interactions
        try:
            encoding = get_encoding("zz_feature_map", n_features=n_features, reps=1, entanglement="linear")
            circuit = encoding.get_circuit(x, backend="qiskit")

            results.append(EdgeCaseResult(
                name=f"zz_linear/dim_{n_features}",
                category="high_dim",
                passed=circuit is not None,
                behavior="works" if circuit else "fails",
                message=f"{n_features} qubits, depth={circuit.depth() if circuit else 'N/A'}",
                expected_behavior="Linear entanglement scales well"
            ))
        except Exception as e:
            results.append(EdgeCaseResult(
                name=f"zz_linear/dim_{n_features}",
                category="high_dim",
                passed=False,
                behavior="raises_error",
                message=str(e)[:80],
                expected_behavior="Should handle high dimensions"
            ))

    return results


# ============================================================================
# 6. OUTLIERS AND EXTREME VALUES
# ============================================================================

def test_outliers() -> List[EdgeCaseResult]:
    """Test with outliers that might appear in real data."""
    from encoding_atlas import get_encoding

    results = []

    outlier_data = [
        ("single_outlier", np.array([0.5, 0.5, 100.0, 0.5])),  # One extreme value
        ("negative_outlier", np.array([0.5, 0.5, -50.0, 0.5])),
        ("inf_value", np.array([0.5, 0.5, np.inf, 0.5])),
        ("neg_inf", np.array([0.5, np.NINF, 0.5, 0.5])),
        ("nan_value", np.array([0.5, np.nan, 0.5, 0.5])),
    ]

    encoding = get_encoding("angle", n_features=4)

    for name, x in outlier_data:
        try:
            circuit = encoding.get_circuit(x, backend="qiskit")

            # Check what happened
            has_nan = False
            has_inf = False
            for instruction in circuit.data:
                for param in instruction.operation.params:
                    if isinstance(param, (int, float)):
                        if math.isnan(param):
                            has_nan = True
                        if math.isinf(param):
                            has_inf = True

            if has_nan or has_inf:
                results.append(EdgeCaseResult(
                    name=f"outlier/{name}",
                    category="outliers",
                    passed=False,
                    behavior="silent_wrong",
                    message="Circuit has NaN/Inf (should be caught!)",
                    expected_behavior="Should raise ValidationError for NaN/Inf input"
                ))
            else:
                results.append(EdgeCaseResult(
                    name=f"outlier/{name}",
                    category="outliers",
                    passed=True,
                    behavior="works",
                    message="Handled outlier (values wrap for angles)",
                    expected_behavior="Should work or raise clear error"
                ))

        except Exception as e:
            error_msg = str(e).lower()
            is_validation = "nan" in error_msg or "inf" in error_msg or "finite" in error_msg or "validation" in error_msg

            results.append(EdgeCaseResult(
                name=f"outlier/{name}",
                category="outliers",
                passed=is_validation,  # Good if it catches bad input
                behavior="raises_error",
                message=f"{type(e).__name__}: {str(e)[:60]}",
                expected_behavior="Should reject NaN/Inf with clear error"
            ))

    return results


# ============================================================================
# 7. COMMON USER MISTAKES
# ============================================================================

def test_common_mistakes() -> List[EdgeCaseResult]:
    """Test common mistakes users make."""
    from encoding_atlas import get_encoding

    results = []

    # Mistake 1: Wrong number of features
    try:
        encoding = get_encoding("angle", n_features=4)
        circuit = encoding.get_circuit(np.array([0.5, 0.5]), backend="qiskit")  # Only 2 features!

        results.append(EdgeCaseResult(
            name="wrong_n_features",
            category="mistakes",
            passed=False,
            behavior="silent_wrong",
            message="Accepted wrong feature count without error",
            expected_behavior="Should raise error for mismatched dimensions"
        ))
    except Exception as e:
        results.append(EdgeCaseResult(
            name="wrong_n_features",
            category="mistakes",
            passed=True,
            behavior="raises_error",
            message=f"Correctly rejected: {str(e)[:60]}",
            expected_behavior="Should raise error for mismatched dimensions"
        ))

    # Mistake 2: Empty array
    try:
        encoding = get_encoding("angle", n_features=0)
        results.append(EdgeCaseResult(
            name="zero_features",
            category="mistakes",
            passed=False,
            behavior="silent_wrong",
            message="Accepted n_features=0",
            expected_behavior="Should raise error"
        ))
    except Exception as e:
        results.append(EdgeCaseResult(
            name="zero_features",
            category="mistakes",
            passed=True,
            behavior="raises_error",
            message=f"Correctly rejected: {str(e)[:60]}",
            expected_behavior="Should raise error for n_features=0"
        ))

    # Mistake 3: 2D array for single sample
    try:
        encoding = get_encoding("angle", n_features=4)
        x_2d = np.array([[0.5, 0.3, 0.7, 0.2]])  # Shape (1, 4) instead of (4,)
        circuit = encoding.get_circuit(x_2d, backend="qiskit")

        # This might be acceptable if handled correctly
        results.append(EdgeCaseResult(
            name="2d_single_sample",
            category="mistakes",
            passed=True,  # Handling 2D gracefully is fine
            behavior="works",
            message="Handled 2D input for single sample",
            expected_behavior="Should handle or give clear error"
        ))
    except Exception as e:
        results.append(EdgeCaseResult(
            name="2d_single_sample",
            category="mistakes",
            passed=True,  # Error is also acceptable
            behavior="raises_error",
            message=f"{type(e).__name__}: {str(e)[:60]}",
            expected_behavior="Error for ambiguous input is acceptable"
        ))

    # Mistake 4: String input
    try:
        encoding = get_encoding("angle", n_features=4)
        circuit = encoding.get_circuit(["0.5", "0.3", "0.7", "0.2"], backend="qiskit")

        results.append(EdgeCaseResult(
            name="string_input",
            category="mistakes",
            passed=False,
            behavior="silent_wrong",
            message="Accepted string input",
            expected_behavior="Should raise TypeError"
        ))
    except Exception as e:
        results.append(EdgeCaseResult(
            name="string_input",
            category="mistakes",
            passed=True,
            behavior="raises_error",
            message=f"Correctly rejected: {type(e).__name__}",
            expected_behavior="Should raise error for string input"
        ))

    return results


# ============================================================================
# 8. AMPLITUDE ENCODING SPECIFIC EDGE CASES
# ============================================================================

def test_amplitude_edge_cases() -> List[EdgeCaseResult]:
    """Test amplitude encoding specific issues."""
    from encoding_atlas import get_encoding

    results = []

    # Case 1: Zero vector (can't normalize)
    try:
        encoding = get_encoding("amplitude", n_features=4, normalize=True)
        circuit = encoding.get_circuit(np.array([0.0, 0.0, 0.0, 0.0]), backend="qiskit")

        # Check if amplitudes are valid
        for instruction in circuit.data:
            if instruction.operation.name == 'initialize':
                amps = instruction.operation.params
                norm = sum(abs(a)**2 for a in amps)
                if abs(norm - 1.0) > 1e-6 or any(math.isnan(a) for a in amps):
                    results.append(EdgeCaseResult(
                        name="amplitude/zero_vector",
                        category="amplitude",
                        passed=False,
                        behavior="silent_wrong",
                        message=f"Invalid amplitudes: norm={norm}",
                        expected_behavior="Should raise error for zero vector"
                    ))
                else:
                    results.append(EdgeCaseResult(
                        name="amplitude/zero_vector",
                        category="amplitude",
                        passed=True,
                        behavior="works",
                        message="Handled zero vector somehow",
                        expected_behavior="Should raise error or use uniform"
                    ))
                break
    except Exception as e:
        results.append(EdgeCaseResult(
            name="amplitude/zero_vector",
            category="amplitude",
            passed=True,
            behavior="raises_error",
            message=f"Correctly rejected: {str(e)[:60]}",
            expected_behavior="Should raise error for zero vector"
        ))

    # Case 2: Non-power-of-2 features (library correctly pads to next power of 2)
    try:
        encoding = get_encoding("amplitude", n_features=3)
        # This is CORRECT behavior - amplitude encoding pads 3 features to 4 (2 qubits)
        # [x1, x2, x3] -> [x1, x2, x3, 0] normalized
        circuit = encoding.get_circuit(np.array([0.5, 0.3, 0.8]), backend="qiskit")

        # Verify it works correctly
        if circuit is not None and encoding.n_qubits == 2:  # ceil(log2(3)) = 2
            results.append(EdgeCaseResult(
                name="amplitude/non_pow2",
                category="amplitude",
                passed=True,
                behavior="works",
                message=f"Correctly pads 3 features to 4 amplitudes (2 qubits)",
                expected_behavior="Should pad to next power of 2"
            ))
        else:
            results.append(EdgeCaseResult(
                name="amplitude/non_pow2",
                category="amplitude",
                passed=False,
                behavior="wrong_output",
                message=f"Unexpected n_qubits: {encoding.n_qubits}",
                expected_behavior="Should use 2 qubits for 3 features"
            ))
    except Exception as e:
        results.append(EdgeCaseResult(
            name="amplitude/non_pow2",
            category="amplitude",
            passed=False,
            behavior="raises_error",
            message=f"Unexpected error: {str(e)[:60]}",
            expected_behavior="Should pad to next power of 2, not raise error"
        ))

    # Case 3: Already normalized (should not double-normalize)
    try:
        encoding = get_encoding("amplitude", n_features=4, normalize=True)
        x_normalized = np.array([0.5, 0.5, 0.5, 0.5])  # Already unit vector
        circuit = encoding.get_circuit(x_normalized, backend="qiskit")

        for instruction in circuit.data:
            if instruction.operation.name == 'initialize':
                amps = list(instruction.operation.params)[:4]
                expected = list(x_normalized / np.linalg.norm(x_normalized))
                # Handle complex amplitudes by comparing magnitudes
                match = all(
                    abs(complex(amps[i]).real - expected[i]) < 1e-6
                    for i in range(4)
                )

                # Format for display, handling complex numbers
                amps_display = [round(complex(a).real, 4) for a in amps]

                results.append(EdgeCaseResult(
                    name="amplitude/already_normalized",
                    category="amplitude",
                    passed=match,
                    behavior="works" if match else "wrong_output",
                    message=f"Amplitudes: {amps_display}",
                    expected_behavior="Should normalize consistently"
                ))
                break
    except Exception as e:
        results.append(EdgeCaseResult(
            name="amplitude/already_normalized",
            category="amplitude",
            passed=False,
            behavior="raises_error",
            message=str(e)[:60],
            expected_behavior="Should handle already-normalized input"
        ))

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_all_edge_tests(verbose: bool = False) -> Dict[str, List[EdgeCaseResult]]:
    """Run all edge case tests."""

    all_tests = [
        ("NEGATIVE VALUES (Z-SCORED DATA)", test_negative_values),
        ("UNNORMALIZED VALUES", test_unnormalized_values),
        ("BATCH PROCESSING", test_batch_processing),
        ("INPUT TYPES", test_input_types),
        ("HIGH DIMENSIONAL", test_high_dimensional),
        ("OUTLIERS", test_outliers),
        ("COMMON MISTAKES", test_common_mistakes),
        ("AMPLITUDE EDGE CASES", test_amplitude_edge_cases),
    ]

    all_results = {}
    total_passed = 0
    total_failed = 0
    total_dangerous = 0  # silent_wrong is dangerous

    print("=" * 70)
    print("CRITICAL EDGE CASE TESTS - REAL USER SCENARIOS")
    print("=" * 70)

    for category_name, test_func in all_tests:
        print(f"\n{'=' * 70}")
        print(f"{category_name}")
        print("=" * 70)

        results = test_func()
        all_results[category_name] = results

        for r in results:
            print_result(r, verbose)

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        dangerous = sum(1 for r in results if r.behavior == "silent_wrong")

        total_passed += passed
        total_failed += failed
        total_dangerous += dangerous

        print(f"\n  Summary: {passed} passed, {failed} failed" +
              (f", {dangerous} DANGEROUS (silent wrong)" if dangerous > 0 else ""))

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    if total_dangerous > 0:
        print(f"  DANGEROUS (silent wrong): {total_dangerous}")
        print("  WARNING: Silent wrong means the code produces incorrect output")
        print("           without raising an error - users won't know it's wrong!")

    return all_results


def main():
    _fix_windows_console()
    parser = argparse.ArgumentParser(description="Edge case tests for encodings")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_all_edge_tests(verbose=args.verbose)


if __name__ == "__main__":
    main()
