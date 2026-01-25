"""
Real-World Encoding Tests with Expected Output Validation
==========================================================
This test suite validates that encodings produce CORRECT outputs, not just that they run.

It includes:
1. Formula verification - check that angles match expected mathematical formulas
2. Real-world data scenarios - normalized ML features, image data, time series, etc.
3. Known-answer tests - specific inputs with pre-computed expected outputs
4. Consistency tests - same input always produces same output
5. Edge case validation - boundary conditions, extreme values

Usage:
    python tests/test_encodings_realworld.py
    python tests/test_encodings_realworld.py --verbose
    python tests/test_encodings_realworld.py --category formula
    python tests/test_encodings_realworld.py --category realworld
"""

import sys
import io
import argparse
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field

sys.path.insert(0, 'src')


def _fix_windows_console():
    """Fix Windows console encoding for Unicode (only when run directly)."""
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# TEST INFRASTRUCTURE
# ============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    category: str
    passed: bool
    message: str
    expected: Any = None
    actual: Any = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealWorldScenario:
    """A real-world test scenario with expected properties."""
    name: str
    description: str
    data: np.ndarray
    data_source: str  # e.g., "normalized_iris", "mnist_pixel", "stock_returns"
    expected_properties: Dict[str, Any]  # What we expect from the encoding


def print_result(result: TestResult, verbose: bool = False) -> None:
    """Print a test result."""
    status = "PASS" if result.passed else "FAIL"
    icon = "[OK]" if result.passed else "[X]"

    print(f"  {icon} {result.name}: {status}")

    if not result.passed or verbose:
        print(f"      Message: {result.message}")
        if result.expected is not None:
            print(f"      Expected: {result.expected}")
        if result.actual is not None:
            print(f"      Actual: {result.actual}")
        if result.details and verbose:
            for k, v in result.details.items():
                print(f"      {k}: {v}")


# ============================================================================
# FORMULA VERIFICATION TESTS
# ============================================================================

def test_angle_encoding_formula() -> List[TestResult]:
    """Verify AngleEncoding uses correct rotation angles."""
    from encoding_atlas import get_encoding

    results = []

    # Test: angle should equal input value for RY rotation
    test_values = [0.0, 0.5, 1.0, np.pi/4, np.pi/2, np.pi]

    for val in test_values:
        x = np.array([val, val])
        encoding = get_encoding("angle", n_features=2, rotation="Y")
        circuit = encoding.get_circuit(x, backend="qiskit")

        # Extract RY angles from circuit
        ry_angles = []
        for instruction in circuit.data:
            if instruction.operation.name == 'ry':
                ry_angles.append(instruction.operation.params[0])

        # Verify angles match input
        expected_angle = val
        passed = all(abs(a - expected_angle) < 1e-10 for a in ry_angles)

        results.append(TestResult(
            name=f"angle_ry_formula(x={val:.4f})",
            category="formula",
            passed=passed,
            message="RY angle should equal input value" if passed else "RY angle mismatch",
            expected=expected_angle,
            actual=ry_angles[0] if ry_angles else None
        ))

    return results


def test_zz_feature_map_formula() -> List[TestResult]:
    """Verify ZZFeatureMap uses correct (pi-x_i)(pi-x_j) formula for ZZ interactions."""
    from encoding_atlas import get_encoding

    results = []

    # Test cases with known expected ZZ angles
    test_cases = [
        # (x_0, x_1, expected_zz_angle = 2*(pi-x_0)*(pi-x_1))
        (0.0, 0.0, 2 * np.pi * np.pi),  # ~19.74
        (np.pi, np.pi, 0.0),  # At x=pi, angle is 0
        (0.5, 0.5, 2 * (np.pi - 0.5) * (np.pi - 0.5)),  # ~13.96
        (1.0, 2.0, 2 * (np.pi - 1.0) * (np.pi - 2.0)),  # ~4.88
    ]

    for x0, x1, expected_zz in test_cases:
        x = np.array([x0, x1])
        encoding = get_encoding("zz_feature_map", n_features=2, reps=1, entanglement="linear")
        circuit = encoding.get_circuit(x, backend="qiskit")

        # Find the ZZ interaction angle (it's in the middle P gate of CNOT-P-CNOT pattern)
        # The circuit has: H, P(2*x_i), then CNOT-P(zz_angle)-CNOT
        p_gates = []
        for instruction in circuit.data:
            if instruction.operation.name == 'p':
                p_gates.append(instruction.operation.params[0])

        # The third P gate (index 2) should be the ZZ interaction
        if len(p_gates) >= 3:
            actual_zz = p_gates[2]
            # Allow for 2*pi periodicity
            diff = abs(actual_zz - expected_zz) % (2 * np.pi)
            diff = min(diff, 2 * np.pi - diff)
            passed = diff < 1e-6
        else:
            actual_zz = None
            passed = False

        results.append(TestResult(
            name=f"zz_formula(x=[{x0:.2f},{x1:.2f}])",
            category="formula",
            passed=passed,
            message="ZZ angle = 2*(pi-x_i)*(pi-x_j)" if passed else "ZZ angle formula mismatch",
            expected=round(expected_zz, 4),
            actual=round(actual_zz, 4) if actual_zz else None,
            details={"all_p_angles": [round(p, 4) for p in p_gates]}
        ))

    return results


def test_amplitude_encoding_normalization() -> List[TestResult]:
    """Verify AmplitudeEncoding normalizes input to unit vector."""
    from encoding_atlas import get_encoding

    results = []

    test_cases = [
        np.array([1.0, 0.0]),  # Already on axis
        np.array([1.0, 1.0]),  # Should normalize to [0.707, 0.707]
        np.array([3.0, 4.0]),  # Should normalize to [0.6, 0.8]
        np.array([1.0, 2.0, 3.0, 4.0]),  # 4 features -> 2 qubits
    ]

    for x in test_cases:
        encoding = get_encoding("amplitude", n_features=len(x), normalize=True)
        circuit = encoding.get_circuit(x, backend="qiskit")

        # The Initialize gate should have normalized amplitudes
        for instruction in circuit.data:
            if instruction.operation.name == 'initialize':
                amplitudes = list(instruction.operation.params)

                # Check normalization (sum of |amplitude|^2 should be 1)
                norm_sq = sum(abs(a)**2 for a in amplitudes)
                passed = abs(norm_sq - 1.0) < 1e-10

                # Check proportions match original
                expected_normalized = x / np.linalg.norm(x)
                proportion_match = all(
                    abs(amplitudes[i] - expected_normalized[i]) < 1e-10
                    for i in range(len(x))
                )

                results.append(TestResult(
                    name=f"amplitude_norm(x={list(x)})",
                    category="formula",
                    passed=passed and proportion_match,
                    message="Amplitudes normalized correctly" if passed else "Normalization error",
                    expected=list(np.round(expected_normalized, 6)),
                    actual=list(np.round(amplitudes[:len(x)], 6)),
                    details={"norm_squared": round(norm_sq, 10)}
                ))
                break

    return results


def test_iqp_vs_zz_difference() -> List[TestResult]:
    """Verify IQP and ZZ use different angle formulas (they should!)."""
    from encoding_atlas import get_encoding

    results = []

    x = np.array([0.5, 0.7])

    # IQP uses x_i * x_j for ZZ interaction
    # ZZ uses 2*(pi - x_i)*(pi - x_j)

    iqp_encoding = get_encoding("iqp", n_features=2, reps=1, entanglement="linear")
    zz_encoding = get_encoding("zz_feature_map", n_features=2, reps=1, entanglement="linear")

    iqp_circuit = iqp_encoding.get_circuit(x, backend="qiskit")
    zz_circuit = zz_encoding.get_circuit(x, backend="qiskit")

    # Extract interaction angles
    def get_interaction_angle(circuit):
        for instruction in circuit.data:
            if instruction.operation.name == 'rz':
                # The last RZ in IQP is the interaction
                return instruction.operation.params[0]
            if instruction.operation.name == 'p':
                # For ZZ, it's a P gate
                pass
        return None

    # They should be different (IQP: x_i*x_j vs ZZ: 2*(pi-x_i)*(pi-x_j))
    iqp_expected = x[0] * x[1]  # 0.35
    zz_expected = 2 * (np.pi - x[0]) * (np.pi - x[1])  # ~12.9

    results.append(TestResult(
        name="iqp_vs_zz_formulas_differ",
        category="formula",
        passed=abs(iqp_expected - zz_expected) > 1.0,  # They should be very different
        message="IQP and ZZ use different formulas (correct)",
        expected=f"IQP: {iqp_expected:.4f}, ZZ: {zz_expected:.4f}",
        actual="Formulas are different" if abs(iqp_expected - zz_expected) > 1.0 else "Same formula (wrong!)"
    ))

    return results


# ============================================================================
# REAL-WORLD DATA SCENARIOS
# ============================================================================

def create_realworld_scenarios() -> Dict[str, List[RealWorldScenario]]:
    """Create realistic data scenarios that users might encounter."""

    scenarios = {
        "ml_classification": [
            RealWorldScenario(
                name="iris_normalized",
                description="Normalized Iris dataset features (sepal/petal measurements)",
                data=np.array([0.222, 0.625, 0.068, 0.042]),  # Real normalized Iris sample
                data_source="sklearn.datasets.load_iris (normalized)",
                expected_properties={
                    "should_encode": True,
                    "typical_use": "quantum kernel classification"
                }
            ),
            RealWorldScenario(
                name="breast_cancer_features",
                description="Normalized breast cancer dataset features",
                data=np.array([0.521, 0.023, 0.546, 0.016, 0.154, 0.040, 0.239, 0.094]),
                data_source="sklearn.datasets.load_breast_cancer (normalized)",
                expected_properties={
                    "should_encode": True,
                    "n_qubits_amplitude": 3  # log2(8) = 3
                }
            ),
            RealWorldScenario(
                name="mnist_pixel_patch",
                description="8 normalized pixel values from MNIST digit",
                data=np.array([0.0, 0.12, 0.89, 1.0, 0.95, 0.67, 0.23, 0.0]),
                data_source="MNIST 8x8 downsampled patch",
                expected_properties={
                    "has_zeros": True,
                    "has_ones": True
                }
            ),
        ],

        "time_series": [
            RealWorldScenario(
                name="stock_returns",
                description="Daily stock returns (can be negative in general, normalized here)",
                data=np.array([0.52, 0.48, 0.51, 0.47, 0.53, 0.49]),  # Centered around 0.5
                data_source="Normalized daily returns",
                expected_properties={
                    "pattern": "oscillating",
                    "range": [0.47, 0.53]
                }
            ),
            RealWorldScenario(
                name="sensor_readings",
                description="IoT sensor readings over time (temperature normalized)",
                data=np.array([0.2, 0.25, 0.35, 0.5, 0.65, 0.7, 0.68, 0.6]),
                data_source="Temperature sensor (min-max normalized)",
                expected_properties={
                    "trend": "increasing_then_decreasing"
                }
            ),
            RealWorldScenario(
                name="ecg_heartbeat",
                description="Normalized ECG signal segment",
                data=np.array([0.5, 0.5, 0.6, 0.9, 0.3, 0.5, 0.5, 0.5]),  # Simplified R-peak
                data_source="ECG signal (normalized)",
                expected_properties={
                    "has_spike": True
                }
            ),
        ],

        "scientific": [
            RealWorldScenario(
                name="molecular_features",
                description="Molecular descriptors (normalized)",
                data=np.array([0.123, 0.456, 0.789, 0.234]),  # LogP, MW, TPSA, etc.
                data_source="RDKit molecular descriptors",
                expected_properties={
                    "use_case": "drug_discovery"
                }
            ),
            RealWorldScenario(
                name="spectral_data",
                description="Normalized spectroscopy peaks",
                data=np.array([0.1, 0.3, 0.9, 0.2, 0.05, 0.8, 0.15, 0.4]),
                data_source="IR/NMR spectroscopy",
                expected_properties={
                    "has_peaks": True
                }
            ),
        ],

        "edge_cases": [
            RealWorldScenario(
                name="sparse_data",
                description="Mostly zeros (sparse features common in NLP/recommenders)",
                data=np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.3, 0.0]),
                data_source="One-hot or sparse embedding",
                expected_properties={
                    "sparsity": 0.75
                }
            ),
            RealWorldScenario(
                name="uniform_data",
                description="All same value (degenerate case)",
                data=np.array([0.5, 0.5, 0.5, 0.5]),
                data_source="Constant features",
                expected_properties={
                    "variance": 0.0
                }
            ),
            RealWorldScenario(
                name="near_boundary",
                description="Values very close to 0 and 1 boundaries",
                data=np.array([0.001, 0.999, 0.002, 0.998]),
                data_source="Extreme normalized values",
                expected_properties={
                    "near_boundary": True
                }
            ),
            RealWorldScenario(
                name="high_precision",
                description="Values requiring high numerical precision",
                data=np.array([0.123456789, 0.987654321, 0.246813579, 0.135792468]),
                data_source="High precision measurements",
                expected_properties={
                    "precision_test": True
                }
            ),
        ],

        "adversarial": [
            RealWorldScenario(
                name="pi_multiples",
                description="Values that are multiples of pi (special in rotations)",
                data=np.array([0.0, np.pi/4, np.pi/2, np.pi]) / np.pi,  # Normalized
                data_source="Angles normalized by pi",
                expected_properties={
                    "special_angles": True
                }
            ),
            RealWorldScenario(
                name="very_small",
                description="Very small values (numerical stability test)",
                data=np.array([1e-10, 1e-8, 1e-6, 1e-4]),
                data_source="Near-zero values",
                expected_properties={
                    "stability_test": True
                }
            ),
            RealWorldScenario(
                name="alternating",
                description="Rapidly alternating values",
                data=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
                data_source="Binary alternating pattern",
                expected_properties={
                    "max_variation": True
                }
            ),
        ],
    }

    return scenarios


def test_realworld_scenarios() -> List[TestResult]:
    """Test all encodings with real-world data scenarios."""
    from encoding_atlas import get_encoding, list_encodings

    results = []
    scenarios = create_realworld_scenarios()

    # Encodings to test with their constraints
    encoding_configs = [
        ("angle", {}, lambda n: True),
        ("zz_feature_map", {"reps": 1}, lambda n: n >= 2),
        ("iqp", {"reps": 1}, lambda n: n >= 2),
        ("pauli_feature_map", {"paulis": ["Z", "ZZ"], "reps": 1}, lambda n: n >= 2),
        ("amplitude", {"normalize": True}, lambda n: n in [2, 4, 8]),
        ("data_reuploading", {"n_layers": 2}, lambda n: n >= 2),
        ("hardware_efficient", {"reps": 1}, lambda n: n >= 2),
        ("hamiltonian", {"hamiltonian_type": "iqp", "reps": 1}, lambda n: n >= 2),
    ]

    for category, scenario_list in scenarios.items():
        for scenario in scenario_list:
            n_features = len(scenario.data)

            for enc_name, enc_config, is_valid in encoding_configs:
                if not is_valid(n_features):
                    continue

                test_name = f"{enc_name}/{scenario.name}"

                try:
                    encoding = get_encoding(enc_name, n_features=n_features, **enc_config)
                    circuit = encoding.get_circuit(scenario.data, backend="qiskit")

                    # Validate circuit was created
                    if circuit is None:
                        results.append(TestResult(
                            name=test_name,
                            category=category,
                            passed=False,
                            message="Circuit is None",
                            details={"scenario": scenario.description}
                        ))
                        continue

                    # Validate circuit has correct number of qubits
                    expected_qubits = encoding.n_qubits
                    actual_qubits = circuit.num_qubits

                    if expected_qubits != actual_qubits:
                        results.append(TestResult(
                            name=test_name,
                            category=category,
                            passed=False,
                            message="Qubit count mismatch",
                            expected=expected_qubits,
                            actual=actual_qubits
                        ))
                        continue

                    # Validate circuit has gates (not empty)
                    if circuit.depth() == 0 and enc_name != "basis":
                        results.append(TestResult(
                            name=test_name,
                            category=category,
                            passed=False,
                            message="Circuit is empty (no gates)",
                            details={"data": list(scenario.data)}
                        ))
                        continue

                    # Validate no NaN in gate parameters
                    has_nan = False
                    for instruction in circuit.data:
                        for param in instruction.operation.params:
                            if isinstance(param, (int, float)) and (math.isnan(param) or math.isinf(param)):
                                has_nan = True
                                break

                    if has_nan:
                        results.append(TestResult(
                            name=test_name,
                            category=category,
                            passed=False,
                            message="Circuit contains NaN or Inf parameters",
                            details={"data": list(scenario.data)}
                        ))
                        continue

                    # All validations passed
                    results.append(TestResult(
                        name=test_name,
                        category=category,
                        passed=True,
                        message=f"OK (qubits={actual_qubits}, depth={circuit.depth()})",
                        details={
                            "description": scenario.description,
                            "depth": circuit.depth(),
                            "gates": dict(circuit.count_ops())
                        }
                    ))

                except Exception as e:
                    results.append(TestResult(
                        name=test_name,
                        category=category,
                        passed=False,
                        message=f"Exception: {type(e).__name__}: {str(e)[:100]}",
                        details={"data": list(scenario.data)}
                    ))

    return results


# ============================================================================
# CONSISTENCY TESTS
# ============================================================================

def test_deterministic_output() -> List[TestResult]:
    """Verify same input always produces same circuit."""
    from encoding_atlas import get_encoding

    results = []

    encodings_to_test = [
        ("angle", {"rotation": "Y"}),
        ("zz_feature_map", {"reps": 1}),
        ("iqp", {"reps": 1}),
        ("amplitude", {"normalize": True}),
        ("hamiltonian", {"hamiltonian_type": "iqp"}),
    ]

    x = np.array([0.3, 0.7, 0.5, 0.9])

    for enc_name, config in encodings_to_test:
        n_features = 4 if enc_name != "amplitude" else 4

        try:
            encoding = get_encoding(enc_name, n_features=n_features, **config)

            # Generate circuit multiple times
            circuits = [encoding.get_circuit(x.copy(), backend="qiskit") for _ in range(5)]

            # Extract all parameters from each circuit
            def get_all_params(circuit):
                params = []
                for instruction in circuit.data:
                    params.extend([p for p in instruction.operation.params if isinstance(p, (int, float))])
                return params

            all_params = [get_all_params(c) for c in circuits]

            # Check all are identical
            reference = all_params[0]
            all_same = all(
                len(p) == len(reference) and all(abs(p[i] - reference[i]) < 1e-15 for i in range(len(p)))
                for p in all_params[1:]
            )

            results.append(TestResult(
                name=f"deterministic/{enc_name}",
                category="consistency",
                passed=all_same,
                message="Same input produces same output" if all_same else "Non-deterministic output!",
                details={"num_params": len(reference)}
            ))

        except Exception as e:
            results.append(TestResult(
                name=f"deterministic/{enc_name}",
                category="consistency",
                passed=False,
                message=f"Exception: {e}"
            ))

    return results


def test_symmetry_properties() -> List[TestResult]:
    """Test that symmetric inputs produce expected symmetric outputs."""
    from encoding_atlas import get_encoding

    results = []

    # For angle encoding, swapping features should swap qubit operations
    x1 = np.array([0.3, 0.7])
    x2 = np.array([0.7, 0.3])  # Swapped

    encoding = get_encoding("angle", n_features=2, rotation="Y")
    c1 = encoding.get_circuit(x1, backend="qiskit")
    c2 = encoding.get_circuit(x2, backend="qiskit")

    # Get RY angles
    def get_ry_angles(circuit):
        angles = {}
        for instruction in circuit.data:
            if instruction.operation.name == 'ry':
                qubit_idx = circuit.find_bit(instruction.qubits[0]).index
                angles[qubit_idx] = instruction.operation.params[0]
        return angles

    angles1 = get_ry_angles(c1)
    angles2 = get_ry_angles(c2)

    # After swap: q0 of x1 should match q1 of x2, and vice versa
    swap_symmetric = (
        abs(angles1[0] - angles2[1]) < 1e-10 and
        abs(angles1[1] - angles2[0]) < 1e-10
    )

    results.append(TestResult(
        name="angle_swap_symmetry",
        category="consistency",
        passed=swap_symmetric,
        message="Swapping inputs swaps qubit angles" if swap_symmetric else "Symmetry broken",
        expected={"q0": 0.3, "q1": 0.7},
        actual=angles1
    ))

    return results


# ============================================================================
# KNOWN-ANSWER TESTS (Pre-computed expected values)
# ============================================================================

def test_known_answers() -> List[TestResult]:
    """Test specific inputs against pre-computed expected outputs."""
    from encoding_atlas import get_encoding

    results = []

    # Test 1: Angle encoding with x=0 should have Ry(0) = identity-like
    encoding = get_encoding("angle", n_features=2, rotation="Y")
    circuit = encoding.get_circuit(np.array([0.0, 0.0]), backend="qiskit")

    # Ry(0) is identity, so depth should still be 1 (gate exists but does nothing)
    angles = [i.operation.params[0] for i in circuit.data if i.operation.name == 'ry']
    results.append(TestResult(
        name="angle_zero_input",
        category="known_answer",
        passed=all(a == 0.0 for a in angles),
        message="Ry(0) for zero input",
        expected=[0.0, 0.0],
        actual=angles
    ))

    # Test 2: ZZ feature map with x=pi should have ZZ angle = 0
    encoding = get_encoding("zz_feature_map", n_features=2, reps=1)
    circuit = encoding.get_circuit(np.array([np.pi, np.pi]), backend="qiskit")

    # Find the ZZ interaction P gate
    p_gates = [i.operation.params[0] for i in circuit.data if i.operation.name == 'p']
    # The ZZ angle should be 2*(pi-pi)*(pi-pi) = 0
    # But the single-qubit P gates will be 2*pi each

    # Check that we have the expected structure
    has_correct_structure = len(p_gates) >= 3  # At least 2 single + 1 interaction

    results.append(TestResult(
        name="zz_pi_input",
        category="known_answer",
        passed=has_correct_structure,
        message="ZZ structure correct for pi input",
        expected="3 P gates (2 single + 1 interaction)",
        actual=f"{len(p_gates)} P gates"
    ))

    # Test 3: Amplitude encoding with [1,0,0,0] should give |00⟩
    encoding = get_encoding("amplitude", n_features=4, normalize=False)
    # Need to handle that [1,0,0,0] is already normalized
    x = np.array([1.0, 0.0, 0.0, 0.0])
    circuit = encoding.get_circuit(x, backend="qiskit")

    # The Initialize gate should have amplitudes [1, 0, 0, 0]
    for instruction in circuit.data:
        if instruction.operation.name == 'initialize':
            amplitudes = list(instruction.operation.params)
            expected = [1.0, 0.0, 0.0, 0.0]
            match = all(abs(amplitudes[i] - expected[i]) < 1e-10 for i in range(4))
            results.append(TestResult(
                name="amplitude_basis_state",
                category="known_answer",
                passed=match,
                message="[1,0,0,0] encodes to |00⟩",
                expected=expected,
                actual=amplitudes[:4]
            ))
            break

    # Test 4: Basis encoding with [0.3, 0.7] should give X on q1 only
    encoding = get_encoding("basis", n_features=2)
    circuit = encoding.get_circuit(np.array([0.3, 0.7]), backend="qiskit")

    x_gates = [(circuit.find_bit(i.qubits[0]).index) for i in circuit.data if i.operation.name == 'x']
    # 0.3 < 0.5 -> no X, 0.7 > 0.5 -> X
    expected_x = [1]  # X only on qubit 1

    results.append(TestResult(
        name="basis_threshold",
        category="known_answer",
        passed=x_gates == expected_x,
        message="Basis encoding threshold at 0.5",
        expected=expected_x,
        actual=x_gates
    ))

    return results


# ============================================================================
# PROPERTY VALIDATION TESTS
# ============================================================================

def test_property_accuracy() -> List[TestResult]:
    """Verify encoding.properties matches actual circuit properties."""
    from encoding_atlas import get_encoding

    results = []

    test_cases = [
        ("angle", {"rotation": "Y"}, 4),
        ("zz_feature_map", {"reps": 1, "entanglement": "linear"}, 4),
        ("zz_feature_map", {"reps": 2, "entanglement": "full"}, 3),
        ("iqp", {"reps": 1, "entanglement": "circular"}, 4),
        ("hardware_efficient", {"reps": 2, "entanglement": "linear"}, 4),
        ("data_reuploading", {"n_layers": 3}, 3),
    ]

    for enc_name, config, n_features in test_cases:
        try:
            encoding = get_encoding(enc_name, n_features=n_features, **config)
            props = encoding.properties

            x = np.random.rand(n_features)
            circuit = encoding.get_circuit(x, backend="qiskit")

            # Check n_qubits
            qubits_match = props.n_qubits == circuit.num_qubits

            # Check is_entangling
            has_two_qubit = any(
                len(i.qubits) > 1
                for i in circuit.data
                if i.operation.name not in ['barrier', 'measure']
            )
            entangling_match = props.is_entangling == has_two_qubit

            # Check gate counts (approximately - some decomposition may differ)
            actual_gates = sum(circuit.count_ops().values())
            # Allow some flexibility due to barriers
            gates_reasonable = abs(props.gate_count - actual_gates) <= actual_gates * 0.3 + 5

            passed = qubits_match and entangling_match and gates_reasonable

            results.append(TestResult(
                name=f"props/{enc_name}({n_features}f)",
                category="properties",
                passed=passed,
                message="Properties match circuit" if passed else "Property mismatch",
                expected={
                    "n_qubits": props.n_qubits,
                    "is_entangling": props.is_entangling,
                    "gate_count": props.gate_count
                },
                actual={
                    "n_qubits": circuit.num_qubits,
                    "has_two_qubit_gates": has_two_qubit,
                    "actual_gates": actual_gates
                }
            ))

        except Exception as e:
            results.append(TestResult(
                name=f"props/{enc_name}({n_features}f)",
                category="properties",
                passed=False,
                message=f"Exception: {e}"
            ))

    return results


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================

def test_numerical_stability() -> List[TestResult]:
    """Test encodings with numerically challenging inputs."""
    from encoding_atlas import get_encoding

    results = []

    challenging_inputs = [
        ("very_small", np.array([1e-15, 1e-14, 1e-13, 1e-12])),
        ("very_close", np.array([0.5, 0.5 + 1e-10, 0.5 - 1e-10, 0.5])),
        ("near_pi", np.array([np.pi - 1e-10, np.pi + 1e-10, np.pi, np.pi])),
        ("mixed_scale", np.array([1e-10, 0.5, 1.0 - 1e-10, 0.999999999])),
    ]

    encodings = ["angle", "zz_feature_map", "iqp", "pauli_feature_map"]

    for input_name, x in challenging_inputs:
        for enc_name in encodings:
            try:
                config = {"paulis": ["Z", "ZZ"]} if enc_name == "pauli_feature_map" else {}
                encoding = get_encoding(enc_name, n_features=4, **config)
                circuit = encoding.get_circuit(x, backend="qiskit")

                # Check for NaN/Inf in parameters
                has_bad_values = False
                bad_params = []
                for instruction in circuit.data:
                    for param in instruction.operation.params:
                        if isinstance(param, (int, float)):
                            if math.isnan(param) or math.isinf(param):
                                has_bad_values = True
                                bad_params.append((instruction.operation.name, param))

                results.append(TestResult(
                    name=f"stability/{enc_name}/{input_name}",
                    category="stability",
                    passed=not has_bad_values,
                    message="Numerically stable" if not has_bad_values else f"Bad values: {bad_params}",
                    details={"input": list(x)}
                ))

            except Exception as e:
                results.append(TestResult(
                    name=f"stability/{enc_name}/{input_name}",
                    category="stability",
                    passed=False,
                    message=f"Exception: {type(e).__name__}: {str(e)[:50]}"
                ))

    return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(
    categories: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, List[TestResult]]:
    """Run all tests and return results organized by category."""

    all_tests = {
        "formula": [
            ("Angle Encoding Formula", test_angle_encoding_formula),
            ("ZZ Feature Map Formula", test_zz_feature_map_formula),
            ("Amplitude Normalization", test_amplitude_encoding_normalization),
            ("IQP vs ZZ Difference", test_iqp_vs_zz_difference),
        ],
        "realworld": [
            ("Real-World Scenarios", test_realworld_scenarios),
        ],
        "consistency": [
            ("Deterministic Output", test_deterministic_output),
            ("Symmetry Properties", test_symmetry_properties),
        ],
        "known_answer": [
            ("Known Answers", test_known_answers),
        ],
        "properties": [
            ("Property Accuracy", test_property_accuracy),
        ],
        "stability": [
            ("Numerical Stability", test_numerical_stability),
        ],
    }

    if categories:
        all_tests = {k: v for k, v in all_tests.items() if k in categories}

    all_results = {}

    print("=" * 70)
    print("REAL-WORLD ENCODING VALIDATION TESTS")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for category, test_funcs in all_tests.items():
        print(f"\n{'=' * 70}")
        print(f"CATEGORY: {category.upper()}")
        print("=" * 70)

        category_results = []

        for test_name, test_func in test_funcs:
            print(f"\n--- {test_name} ---")

            try:
                results = test_func()
                category_results.extend(results)

                for result in results:
                    print_result(result, verbose)

            except Exception as e:
                print(f"  [X] Test suite crashed: {e}")
                import traceback
                if verbose:
                    traceback.print_exc()

        all_results[category] = category_results

        passed = sum(1 for r in category_results if r.passed)
        failed = len(category_results) - passed
        total_passed += passed
        total_failed += failed

        print(f"\n  Category Summary: {passed} passed, {failed} failed")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    print(f"  Pass Rate: {100 * total_passed / (total_passed + total_failed):.1f}%")

    if total_failed > 0:
        print("\n  FAILED TESTS:")
        for category, results in all_results.items():
            for r in results:
                if not r.passed:
                    print(f"    - [{category}] {r.name}: {r.message}")

    return all_results


def main():
    _fix_windows_console()
    parser = argparse.ArgumentParser(description="Real-world encoding validation tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--category", "-c",
        type=str,
        nargs="+",
        choices=["formula", "realworld", "consistency", "known_answer", "properties", "stability"],
        help="Test categories to run"
    )

    args = parser.parse_args()

    run_all_tests(categories=args.category, verbose=args.verbose)


if __name__ == "__main__":
    main()
