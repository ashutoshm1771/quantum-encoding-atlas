"""Analysis tools for quantum encoding properties.

This module provides quantitative analysis capabilities for quantum encodings,
including resource counting, simulability analysis, expressibility computation,
entanglement capability measurement, and trainability estimation.

Analysis Functions
------------------
The main analysis functions are:

- :func:`count_resources`: Count computational resources (gates, depth, qubits)
- :func:`get_resource_summary`: Quick resource summary from encoding properties
- :func:`get_gate_breakdown`: Detailed gate-by-gate breakdown
- :func:`compare_resources`: Compare resources across multiple encodings
- :func:`estimate_execution_time`: Estimate circuit execution time
- :func:`check_simulability`: Determine if classical simulation is efficient
- :func:`get_simulability_reason`: Get a concise simulability explanation
- :func:`is_clifford_circuit`: Check if encoding uses only Clifford gates
- :func:`is_matchgate_circuit`: Check if encoding uses only matchgate operations
- :func:`estimate_entanglement_bound`: Estimate upper bound on entanglement entropy
- :func:`compute_expressibility`: Measure Hilbert space coverage
- :func:`compute_entanglement_capability`: Measure entanglement generation
- :func:`estimate_trainability`: Detect barren plateau risk

Utility Functions
-----------------
Lower-level utilities for custom analysis:

- :func:`simulate_encoding_statevector`: Simulate circuit to get statevector
- :func:`compute_fidelity`: Compute fidelity between pure states
- :func:`compute_purity`: Compute purity of a density matrix
- :func:`partial_trace_single_qubit`: Compute reduced density matrix

Examples
--------
Basic usage:

>>> from encoding_atlas import AngleEncoding, IQPEncoding
>>> from encoding_atlas.analysis import count_resources, check_simulability
>>>
>>> # Compare resources between encodings
>>> enc_simple = AngleEncoding(n_features=4)
>>> enc_complex = IQPEncoding(n_features=4, reps=2)
>>>
>>> res_simple = count_resources(enc_simple)
>>> res_complex = count_resources(enc_complex)
>>>
>>> print(f"Simple encoding: {res_simple['gate_count']} gates")
>>> print(f"Complex encoding: {res_complex['gate_count']} gates")

Advanced analysis:

>>> from encoding_atlas.analysis import simulate_encoding_statevector
>>> import numpy as np
>>>
>>> enc = AngleEncoding(n_features=2)
>>> x = np.array([0.5, 1.0])
>>> state = simulate_encoding_statevector(enc, x)
>>> print(f"State dimension: {len(state)}")

See Also
--------
encoding_atlas.core.properties : Encoding property dataclasses.
encoding_atlas.core.exceptions : Analysis-related exceptions.
"""

from __future__ import annotations

# Utility functions
from encoding_atlas.analysis._utils import (
    compute_all_parameter_gradients,
    compute_fidelity,
    compute_linear_entropy,
    # Gradient computation
    compute_parameter_gradient,
    compute_purity,
    compute_von_neumann_entropy,
    create_rng,
    # Random sampling
    generate_random_parameters,
    # Quantum operations
    partial_trace_single_qubit,
    partial_trace_subsystem,
    # Simulation
    simulate_encoding_statevector,
    simulate_encoding_statevectors_batch,
    # Validation
    validate_encoding_for_analysis,
    validate_statevector,
)
from encoding_atlas.analysis.entanglement import (
    EntanglementResult,
    compute_entanglement_capability,
    compute_meyer_wallach,
    compute_meyer_wallach_with_breakdown,
    compute_scott_measure,
)

# Main analysis functions
from encoding_atlas.analysis.expressibility import (
    ExpressibilityResult,
    compute_expressibility,
    compute_fidelity_distribution,
    compute_haar_distribution,
)

# Generalization diagnostics (kernel-geometry metrics)
from encoding_atlas.analysis.generalization import (
    centered_kernel_target_alignment,
    compute_effective_dimension,
    compute_fidelity_kernel,
    compute_geometric_difference,
    compute_kernel_target_alignment,
    geometric_difference,
    kernel_effective_dimension,
    kernel_target_alignment,
)

# Noise-resilience analysis (depolarizing noise model)
from encoding_atlas.analysis.noise import (
    NOISE_LEVELS,
    NoiseResilienceResult,
    compute_noise_resilience,
    simulate_noisy_density_matrix,
)

# Unified encoding profiler
from encoding_atlas.analysis.profile import (
    EncodingCharacterization,
    atlas_comparable_metrics,
    compare_to_atlas,
    profile_encoding,
)
from encoding_atlas.analysis.resources import (
    compare_resources,
    count_resources,
    estimate_execution_time,
    get_gate_breakdown,
    get_resource_summary,
)
from encoding_atlas.analysis.simulability import (
    SimulabilityResult,
    check_simulability,
    estimate_entanglement_bound,
    get_simulability_reason,
    is_clifford_circuit,
    is_matchgate_circuit,
)
from encoding_atlas.analysis.trainability import (
    TrainabilityResult,
    compute_gradient_variance,
    detect_barren_plateau,
    estimate_trainability,
)

__all__ = [
    # Main analysis functions
    "compute_expressibility",
    "compute_fidelity_distribution",
    "compute_haar_distribution",
    "ExpressibilityResult",
    "compute_entanglement_capability",
    "compute_meyer_wallach",
    "compute_meyer_wallach_with_breakdown",
    "compute_scott_measure",
    "count_resources",
    "get_resource_summary",
    "get_gate_breakdown",
    "compare_resources",
    "estimate_execution_time",
    "estimate_trainability",
    "compute_gradient_variance",
    "detect_barren_plateau",
    "check_simulability",
    "get_simulability_reason",
    "is_clifford_circuit",
    "is_matchgate_circuit",
    "estimate_entanglement_bound",
    # Generalization diagnostics
    "compute_fidelity_kernel",
    "compute_kernel_target_alignment",
    "compute_geometric_difference",
    "compute_effective_dimension",
    "kernel_target_alignment",
    "centered_kernel_target_alignment",
    "geometric_difference",
    "kernel_effective_dimension",
    # Noise-resilience analysis
    "compute_noise_resilience",
    "simulate_noisy_density_matrix",
    "NOISE_LEVELS",
    # Unified profiler
    "profile_encoding",
    "compare_to_atlas",
    "atlas_comparable_metrics",
    # Type definitions
    "EncodingCharacterization",
    "NoiseResilienceResult",
    "SimulabilityResult",
    "EntanglementResult",
    "TrainabilityResult",
    # Simulation utilities
    "simulate_encoding_statevector",
    "simulate_encoding_statevectors_batch",
    # Validation utilities
    "validate_encoding_for_analysis",
    "validate_statevector",
    # Quantum operation utilities
    "partial_trace_single_qubit",
    "partial_trace_subsystem",
    "compute_fidelity",
    "compute_purity",
    "compute_linear_entropy",
    "compute_von_neumann_entropy",
    # Gradient utilities
    "compute_parameter_gradient",
    "compute_all_parameter_gradients",
    # Random sampling utilities
    "generate_random_parameters",
    "create_rng",
]
