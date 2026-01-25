"""
Encoding Visualizer & Debug Script
===================================
This script tests all implemented encodings and shows what's happening in the pipeline.
Run this to see inputs, outputs, circuit structure, and properties for each encoding.

Usage:
    python tests/debug_encodings_visualizer.py
    python tests/debug_encodings_visualizer.py --encoding amplitude
    python tests/debug_encodings_visualizer.py --backend qiskit
    python tests/debug_encodings_visualizer.py --verbose
"""

import sys
import io
import argparse
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Add the src directory to the path
sys.path.insert(0, 'src')

# ============================================================================
# HELPER CLASSES AND UTILITIES
# ============================================================================

@dataclass
class TestScenario:
    """Defines a test scenario for an encoding."""
    name: str
    n_features: int
    input_data: np.ndarray
    description: str
    extra_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_config is None:
            self.extra_config = {}


def create_divider(title: str = "", char: str = "=", width: int = 80) -> str:
    """Create a visual divider for output."""
    if title:
        padding = (width - len(title) - 2) // 2
        return f"\n{char * padding} {title} {char * padding}\n"
    return f"\n{char * width}\n"


def format_array(arr: np.ndarray, precision: int = 4) -> str:
    """Format numpy array for display."""
    if arr.ndim == 1:
        return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"
    return str(np.round(arr, precision))


def print_properties(props, indent: int = 2) -> None:
    """Print encoding properties in a readable format."""
    prefix = " " * indent
    print(f"{prefix}📊 ENCODING PROPERTIES:")
    print(f"{prefix}  • n_qubits: {props.n_qubits}")
    print(f"{prefix}  • depth: {props.depth}")
    print(f"{prefix}  • gate_count: {props.gate_count}")
    print(f"{prefix}  • single_qubit_gates: {props.single_qubit_gates}")
    print(f"{prefix}  • two_qubit_gates: {props.two_qubit_gates}")
    print(f"{prefix}  • parameter_count: {props.parameter_count}")
    print(f"{prefix}  • is_entangling: {props.is_entangling}")
    print(f"{prefix}  • simulability: {props.simulability}")
    if props.notes:
        print(f"{prefix}  • notes: {props.notes[:100]}..." if len(props.notes) > 100 else f"{prefix}  • notes: {props.notes}")


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def get_test_scenarios() -> Dict[str, List[TestScenario]]:
    """Create test scenarios for different situations."""

    scenarios = {
        # Basic scenarios with small feature counts
        "small": [
            TestScenario(
                name="2_features_uniform",
                n_features=2,
                input_data=np.array([0.5, 0.5]),
                description="2 features with uniform values (0.5, 0.5)"
            ),
            TestScenario(
                name="2_features_extreme",
                n_features=2,
                input_data=np.array([0.0, 1.0]),
                description="2 features with extreme values (0, 1)"
            ),
            TestScenario(
                name="3_features_random",
                n_features=3,
                input_data=np.array([0.2, 0.7, 0.4]),
                description="3 features with varied values"
            ),
        ],

        # Medium scenarios
        "medium": [
            TestScenario(
                name="4_features_linear",
                n_features=4,
                input_data=np.array([0.25, 0.5, 0.75, 1.0]),
                description="4 features with linear progression"
            ),
            TestScenario(
                name="4_features_symmetric",
                n_features=4,
                input_data=np.array([0.1, 0.9, 0.9, 0.1]),
                description="4 features with symmetric pattern"
            ),
            TestScenario(
                name="6_features_sinusoidal",
                n_features=6,
                input_data=np.sin(np.linspace(0, np.pi, 6)),
                description="6 features following sinusoidal pattern"
            ),
        ],

        # Large scenarios (for amplitude encoding)
        "large": [
            TestScenario(
                name="8_features_random",
                n_features=8,
                input_data=np.random.RandomState(42).rand(8),
                description="8 features with random values (seed=42)"
            ),
            TestScenario(
                name="8_features_normalized",
                n_features=8,
                input_data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) / np.linalg.norm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                description="8 features pre-normalized (unit vector)"
            ),
        ],

        # Edge cases
        "edge_cases": [
            TestScenario(
                name="all_zeros",
                n_features=4,
                input_data=np.array([0.0, 0.0, 0.0, 0.0]),
                description="All zero inputs"
            ),
            TestScenario(
                name="all_ones",
                n_features=4,
                input_data=np.array([1.0, 1.0, 1.0, 1.0]),
                description="All one inputs"
            ),
            TestScenario(
                name="near_zero",
                n_features=4,
                input_data=np.array([1e-6, 1e-5, 1e-4, 1e-3]),
                description="Very small values"
            ),
            TestScenario(
                name="binary_pattern",
                n_features=4,
                input_data=np.array([0.0, 1.0, 0.0, 1.0]),
                description="Binary alternating pattern (for basis encoding)"
            ),
        ],

        # Special scenarios for specific encodings
        "special": [
            TestScenario(
                name="pi_scaled",
                n_features=4,
                input_data=np.array([0.0, np.pi/4, np.pi/2, np.pi]) / np.pi,  # Normalized to [0,1]
                description="Pi-scaled values (useful for angle encodings)"
            ),
            TestScenario(
                name="quantum_kernel_optimal",
                n_features=4,
                input_data=np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/6]) / np.pi,
                description="Values optimized for quantum kernel methods"
            ),
        ],
    }

    return scenarios


# ============================================================================
# ENCODING-SPECIFIC CONFIGURATIONS
# ============================================================================

def get_encoding_configs() -> Dict[str, List[Dict[str, Any]]]:
    """Get different configurations to test for each encoding type."""

    return {
        "angle": [
            {"rotation": "Y"},
            {"rotation": "X"},
            {"rotation": "Z"},
            {"rotation": "Y", "reps": 2},
        ],

        "amplitude": [
            {"normalize": True},
            {"normalize": False},
        ],

        "basis": [
            {},  # Default config
        ],

        "iqp": [
            {"reps": 1, "entanglement": "linear"},
            {"reps": 1, "entanglement": "circular"},
            {"reps": 2, "entanglement": "full"},
        ],

        "zz_feature_map": [
            {"reps": 1, "entanglement": "linear"},
            {"reps": 2, "entanglement": "circular"},
            {"reps": 1, "entanglement": "full"},
        ],

        "pauli_feature_map": [
            {"paulis": ["Z"], "reps": 1},
            {"paulis": ["Z", "ZZ"], "reps": 1, "entanglement": "linear"},
            {"paulis": ["Y", "YY"], "reps": 1, "entanglement": "circular"},
        ],

        "data_reuploading": [
            {"n_layers": 1},
            {"n_layers": 2},
            {"n_layers": 3},
        ],

        "hardware_efficient": [
            {"reps": 1, "entanglement": "linear"},
            {"reps": 2, "entanglement": "circular"},
        ],

        "higher_order_angle": [
            {"order": 1},
            {"order": 2, "combination": "product"},
            {"order": 2, "combination": "sum"},
        ],

        "qaoa_encoding": [
            {"reps": 1, "data_rotation": "X", "mixer_rotation": "Y"},
            {"reps": 2, "feature_map": "quadratic"},
        ],

        "hamiltonian": [
            {"hamiltonian_type": "iqp", "evolution_time": 1.0, "reps": 1},
            {"hamiltonian_type": "xy", "evolution_time": 0.5, "reps": 1},
            {"hamiltonian_type": "heisenberg", "evolution_time": 1.0, "reps": 1},
            {"hamiltonian_type": "pauli_z", "evolution_time": 1.0, "reps": 1},
        ],

        "symmetry_inspired_feature_map": [
            {"symmetry": "cyclic", "reps": 1},
            {"symmetry": "reflection", "reps": 1},
            {"symmetry": "rotation", "reps": 1},  # requires even n_features
        ],
    }


# ============================================================================
# CIRCUIT VISUALIZATION
# ============================================================================

def visualize_circuit(circuit, backend: str, encoding_name: str, max_width: int = 100) -> str:
    """Visualize a circuit based on the backend."""

    output_lines = []

    if backend == "pennylane":
        # PennyLane returns a callable/QNode
        output_lines.append(f"  📝 PennyLane Circuit (callable QNode)")
        output_lines.append(f"     Type: {type(circuit)}")
        if hasattr(circuit, 'tape') and circuit.tape is not None:
            output_lines.append(f"     Operations: {len(circuit.tape.operations)}")
        output_lines.append(f"     Note: Call this function to execute the circuit")

    elif backend == "qiskit":
        # Qiskit returns a QuantumCircuit
        output_lines.append(f"  📝 Qiskit QuantumCircuit:")
        output_lines.append(f"     • Qubits: {circuit.num_qubits}")
        output_lines.append(f"     • Depth: {circuit.depth()}")
        output_lines.append(f"     • Gate counts: {dict(circuit.count_ops())}")

        # Draw the circuit
        try:
            circuit_str = circuit.draw(output='text', fold=max_width)
            output_lines.append(f"\n     Circuit diagram:")
            for line in str(circuit_str).split('\n'):
                output_lines.append(f"     {line}")
        except Exception as e:
            output_lines.append(f"     (Could not draw circuit: {e})")

    elif backend == "cirq":
        # Cirq returns a Circuit
        output_lines.append(f"  📝 Cirq Circuit:")
        output_lines.append(f"     • Qubits: {len(circuit.all_qubits())}")
        output_lines.append(f"     • Moments: {len(circuit)}")

        # Draw the circuit
        try:
            circuit_str = str(circuit)
            output_lines.append(f"\n     Circuit diagram:")
            for line in circuit_str.split('\n')[:20]:  # Limit lines
                output_lines.append(f"     {line}")
            if circuit_str.count('\n') > 20:
                output_lines.append(f"     ... (truncated)")
        except Exception as e:
            output_lines.append(f"     (Could not draw circuit: {e})")

    return '\n'.join(output_lines)


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_encoding(
    encoding_name: str,
    scenario: TestScenario,
    config: Dict[str, Any],
    backends: List[str],
    verbose: bool = False
) -> Dict[str, Any]:
    """Test a single encoding with a specific scenario and configuration."""

    from encoding_atlas import get_encoding
    from encoding_atlas.core.exceptions import EncodingError, ValidationError

    results = {
        "encoding": encoding_name,
        "scenario": scenario.name,
        "config": config,
        "success": False,
        "circuits": {},
        "properties": None,
        "error": None,
    }

    print(f"\n  ⚙️  Config: {config}")
    print(f"  📥 INPUT:")
    print(f"     • n_features: {scenario.n_features}")
    print(f"     • data: {format_array(scenario.input_data)}")
    print(f"     • description: {scenario.description}")

    try:
        # Create the encoding
        encoding = get_encoding(encoding_name, n_features=scenario.n_features, **config)

        # Get properties
        props = encoding.properties
        results["properties"] = props

        if verbose:
            print_properties(props)
        else:
            print(f"\n  📊 Properties: qubits={props.n_qubits}, depth={props.depth}, "
                  f"gates={props.gate_count}, entangling={props.is_entangling}")

        # Test each backend
        for backend in backends:
            print(f"\n  🔧 Backend: {backend.upper()}")
            try:
                circuit = encoding.get_circuit(scenario.input_data, backend=backend)
                results["circuits"][backend] = circuit

                print(visualize_circuit(circuit, backend, encoding_name))

                # For Qiskit, show what measurements would look like
                if backend == "qiskit" and verbose:
                    try:
                        from qiskit import transpile
                        from qiskit_aer import AerSimulator

                        # Add measurements
                        circuit_with_meas = circuit.copy()
                        circuit_with_meas.measure_all()

                        # Simulate
                        simulator = AerSimulator()
                        transpiled = transpile(circuit_with_meas, simulator)
                        job = simulator.run(transpiled, shots=1024)
                        counts = job.result().get_counts()

                        # Sort by count and show top results
                        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        print(f"\n     🎯 Simulation Results (top 5 of {len(counts)} outcomes):")
                        for state, count in sorted_counts:
                            prob = count / 1024
                            bar = "█" * int(prob * 20)
                            print(f"        |{state}⟩: {count:4d} ({prob:.2%}) {bar}")
                    except Exception as e:
                        print(f"     (Simulation skipped: {e})")

            except Exception as e:
                print(f"     ❌ Error: {e}")
                results["circuits"][backend] = None

        results["success"] = True

    except (EncodingError, ValidationError) as e:
        print(f"  ❌ Encoding Error: {e}")
        results["error"] = str(e)
    except Exception as e:
        print(f"  ❌ Unexpected Error: {type(e).__name__}: {e}")
        results["error"] = str(e)
        import traceback
        if verbose:
            traceback.print_exc()

    return results


def run_all_tests(
    encodings: Optional[List[str]] = None,
    backends: Optional[List[str]] = None,
    scenario_groups: Optional[List[str]] = None,
    verbose: bool = False,
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """Run tests for all encodings with all scenarios."""

    from encoding_atlas import list_encodings

    # Get available encodings
    available_encodings = list_encodings()

    if encodings:
        # Filter to requested encodings
        test_encodings = [e for e in encodings if e in available_encodings]
        if not test_encodings:
            print(f"❌ No matching encodings found. Available: {available_encodings}")
            return []
    else:
        test_encodings = available_encodings

    # Set up backends
    if backends is None:
        backends = ["qiskit"]  # Default to qiskit for visualization

    # Get scenarios
    all_scenarios = get_test_scenarios()
    if scenario_groups:
        scenarios_to_test = {k: v for k, v in all_scenarios.items() if k in scenario_groups}
    elif quick:
        scenarios_to_test = {"small": all_scenarios["small"][:1]}  # Just one small scenario
    else:
        scenarios_to_test = all_scenarios

    # Get encoding configs
    all_configs = get_encoding_configs()

    print(create_divider("QUANTUM ENCODING VISUALIZER"))
    print(f"Testing {len(test_encodings)} encodings with {sum(len(v) for v in scenarios_to_test.values())} scenarios")
    print(f"Backends: {backends}")
    print(f"Mode: {'Verbose' if verbose else 'Normal'}")

    all_results = []

    for encoding_name in test_encodings:
        print(create_divider(f"ENCODING: {encoding_name.upper()}", "="))

        # Get configs for this encoding
        configs = all_configs.get(encoding_name, [{}])
        if quick:
            configs = configs[:1]  # Just first config in quick mode

        for group_name, scenarios in scenarios_to_test.items():
            print(create_divider(f"Scenario Group: {group_name}", "-"))

            # Filter scenarios based on encoding requirements
            valid_scenarios = []
            for scenario in scenarios:
                # Special handling for different encodings
                if encoding_name == "amplitude":
                    # Amplitude encoding needs power of 2 features
                    if scenario.n_features in [2, 4, 8]:
                        valid_scenarios.append(scenario)
                elif encoding_name == "symmetry_inspired_feature_map":
                    # Some configs need even features
                    valid_scenarios.append(scenario)
                else:
                    if scenario.n_features >= 2:  # Most encodings need at least 2 features
                        valid_scenarios.append(scenario)

            if not valid_scenarios:
                print(f"  ⏭️  No valid scenarios for {encoding_name} in {group_name}")
                continue

            for scenario in valid_scenarios:
                print(f"\n  📋 Scenario: {scenario.name}")

                for config in configs:
                    # Skip invalid configs for rotation symmetry (requires even n_features)
                    if encoding_name in ("symmetry_inspired_feature_map", "symmetry_inspired"):
                        # rotation is the default, so skip if symmetry is rotation OR not specified
                        symmetry = config.get("symmetry", "rotation")
                        if symmetry == "rotation" and scenario.n_features % 2 != 0:
                            continue

                    result = test_encoding(
                        encoding_name=encoding_name,
                        scenario=scenario,
                        config=config,
                        backends=backends,
                        verbose=verbose,
                    )
                    all_results.append(result)

    # Summary
    print(create_divider("SUMMARY"))
    successful = sum(1 for r in all_results if r["success"])
    failed = len(all_results) - successful
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for r in all_results:
            if not r["success"]:
                print(f"  • {r['encoding']} / {r['scenario']} / {r['config']}: {r['error']}")

    return all_results


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Run an interactive session for exploring encodings."""

    from encoding_atlas import get_encoding, list_encodings

    print(create_divider("INTERACTIVE ENCODING EXPLORER"))
    print("Commands:")
    print("  list                    - List all available encodings")
    print("  test <encoding>         - Test a specific encoding")
    print("  compare <enc1> <enc2>   - Compare two encodings")
    print("  help                    - Show this help")
    print("  quit                    - Exit")

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0]

        if action == "quit" or action == "exit":
            print("Goodbye!")
            break

        elif action == "list":
            encodings = list_encodings()
            print(f"\nAvailable encodings ({len(encodings)}):")
            for enc in sorted(encodings):
                print(f"  • {enc}")

        elif action == "test" and len(parts) > 1:
            encoding_name = parts[1]
            run_all_tests(
                encodings=[encoding_name],
                backends=["qiskit"],
                scenario_groups=["small"],
                verbose=True,
            )

        elif action == "compare" and len(parts) > 2:
            enc1, enc2 = parts[1], parts[2]
            print(f"\nComparing {enc1} vs {enc2}:")
            run_all_tests(
                encodings=[enc1, enc2],
                backends=["qiskit"],
                scenario_groups=["small"],
                verbose=False,
            )

        elif action == "help":
            print("Commands:")
            print("  list                    - List all available encodings")
            print("  test <encoding>         - Test a specific encoding")
            print("  compare <enc1> <enc2>   - Compare two encodings")
            print("  help                    - Show this help")
            print("  quit                    - Exit")
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    # Fix Windows console encoding for Unicode (only when run directly)
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="Visualize and test quantum encodings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_encodings_visualizer.py                    # Test all encodings
  python debug_encodings_visualizer.py --encoding angle   # Test specific encoding
  python debug_encodings_visualizer.py --backend cirq     # Use specific backend
  python debug_encodings_visualizer.py --verbose          # Show detailed output
  python debug_encodings_visualizer.py --quick            # Quick test (minimal scenarios)
  python debug_encodings_visualizer.py --interactive      # Interactive mode
        """
    )

    parser.add_argument(
        "--encoding", "-e",
        type=str,
        nargs="+",
        help="Specific encoding(s) to test"
    )

    parser.add_argument(
        "--backend", "-b",
        type=str,
        nargs="+",
        choices=["pennylane", "qiskit", "cirq"],
        default=["qiskit"],
        help="Backend(s) to test (default: qiskit)"
    )

    parser.add_argument(
        "--scenarios", "-s",
        type=str,
        nargs="+",
        choices=["small", "medium", "large", "edge_cases", "special"],
        help="Scenario groups to test (default: all)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including properties and simulation results"
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: test only first scenario and config for each encoding"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        run_all_tests(
            encodings=args.encoding,
            backends=args.backend,
            scenario_groups=args.scenarios,
            verbose=args.verbose,
            quick=args.quick,
        )


if __name__ == "__main__":
    main()
