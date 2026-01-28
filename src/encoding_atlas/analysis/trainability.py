"""Trainability analysis for quantum encodings.

This module implements trainability estimation based on gradient variance
analysis, as described in McClean et al. (2018) on barren plateaus. It
provides tools to assess how easy it is to optimize parameters in a
variational quantum circuit.

Mathematical Background
-----------------------
Trainability of variational quantum circuits is limited by "barren plateaus"
- regions of parameter space where gradients vanish exponentially with
system size. This module estimates trainability by:

1. Computing parameter gradients at random initialization points
2. Measuring the variance of these gradients
3. Comparing to expected scaling for trainable vs. untrainable circuits

A circuit is considered trainable if gradient variance does not decay
exponentially with qubit count.

**Gradient Variance Scaling:**

For a parameterized circuit U(θ), the variance of ∂⟨O⟩/∂θᵢ across random
initializations indicates trainability:

- Barren plateau (untrainable): Var[∂⟨O⟩/∂θᵢ] ∝ 1/2ⁿ (exponential decay)
- Trainable: Var[∂⟨O⟩/∂θᵢ] ∝ const (no decay with system size)

**Cost Function Landscape:**

We analyze the cost function landscape by:
1. Computing gradients at random parameter points using parameter-shift rule
2. Measuring the variance of these gradients
3. Checking for exponential decay with system size

**Observable Types:**

The choice of observable affects barren plateau detection:
- Local observables (single-qubit): Less prone to barren plateaus
- Global observables (many-qubit): More prone to barren plateaus

This module uses local observables by default for more reliable detection.

References
----------
.. [1] McClean, J. R., et al. (2018).
       "Barren plateaus in quantum neural network training landscapes."
       Nature Communications, 9(1), 4812.

.. [2] Cerezo, M., et al. (2021).
       "Cost function dependent barren plateaus in shallow parametrized
       quantum circuits."
       Nature Communications, 12(1), 1791.

.. [3] Larocca, M., et al. (2022).
       "Diagnosing Barren Plateaus with Tools from Quantum Optimal Control."
       Quantum, 6, 824.

.. [4] Schuld, M., et al. (2019).
       "Evaluating analytic gradients on quantum hardware."
       Physical Review A, 99(3), 032331.
       (Parameter-shift rule for gradient computation)

Examples
--------
Basic trainability estimation:

>>> from encoding_atlas import AngleEncoding, IQPEncoding
>>> from encoding_atlas.analysis import estimate_trainability
>>>
>>> # Non-entangling encoding (typically high trainability)
>>> enc_simple = AngleEncoding(n_features=4)
>>> train_simple = estimate_trainability(enc_simple, n_samples=100, seed=42)
>>> print(f"Simple encoding trainability: {train_simple:.4f}")
>>>
>>> # Entangling encoding (may have lower trainability)
>>> enc_complex = IQPEncoding(n_features=4, reps=2)
>>> train_complex = estimate_trainability(enc_complex, n_samples=100, seed=42)
>>> print(f"Complex encoding trainability: {train_complex:.4f}")

Detailed analysis with barren plateau risk:

>>> result = estimate_trainability(
...     enc_complex,
...     n_samples=200,
...     seed=42,
...     return_details=True
... )
>>> print(f"Trainability: {result['trainability_estimate']:.4f}")
>>> print(f"Gradient variance: {result['gradient_variance']:.2e}")
>>> print(f"Barren plateau risk: {result['barren_plateau_risk']}")

Comparing shallow vs deep circuits:

>>> from encoding_atlas import DataReuploading
>>> shallow = DataReuploading(n_features=4, n_layers=1)
>>> deep = DataReuploading(n_features=4, n_layers=5)
>>> print(f"Shallow: {estimate_trainability(shallow, seed=42):.4f}")
>>> print(f"Deep: {estimate_trainability(deep, seed=42):.4f}")

Notes
-----
The trainability score should be interpreted as a relative measure for
comparing encodings, not an absolute guarantee of training success.
Actual training performance depends on many factors including:

- Optimizer choice and learning rate
- Cost function structure
- Hardware noise characteristics
- Initial parameter values

For production use, consider running multiple seeds and averaging results
for more reliable estimates.

See Also
--------
compute_expressibility : Measure Hilbert space coverage.
compute_entanglement_capability : Measure entanglement generation.
count_resources : Count computational resources.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, TypedDict, overload

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.analysis._utils import (
    create_rng,
    simulate_encoding_statevector,
    validate_encoding_for_analysis,
)
from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.exceptions import (
    InsufficientSamplesError,
    NumericalInstabilityError,
    SimulationError,
)

# =============================================================================
# Module-Level Logger
# =============================================================================

_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "estimate_trainability",
    "compute_gradient_variance",
    "detect_barren_plateau",
    "TrainabilityResult",
]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default number of random parameter samples for gradient variance estimation.
#
# RATIONALE: Trainability uses the FEWEST samples (500) compared to other
# analysis metrics (expressibility: 5000, entanglement: 1000) because:
#
# 1. COMPUTATIONAL COST: Each sample requires computing gradients using the
#    parameter-shift rule. For n_features parameters, this means 2*n_features
#    circuit simulations per sample. This is significantly more expensive than
#    expressibility (2 simulations/sample) or entanglement (1 simulation/sample).
#
# 2. VARIANCE ESTIMATION: We're estimating Var(∂⟨O⟩/∂θ), which converges
#    relatively quickly. Variance estimators have standard error ≈ σ²/√(n-1),
#    so 500 samples gives SE ≈ σ²/22, which is sufficient for barren plateau
#    detection (orders of magnitude matter, not decimal precision).
#
# 3. BARREN PLATEAU DETECTION: The goal is to distinguish trainable circuits
#    (variance > 10⁻³) from untrainable ones (variance < 10⁻⁶). This 3-order-
#    of-magnitude gap is easily detectable even with moderate sample counts.
#
# 4. LITERATURE STANDARD: Barren plateau studies typically use 100-1000
#    parameter samples (McClean et al., 2018; Cerezo et al., 2021).
#
# For a 4-feature encoding, 500 samples = 500 * 2 * 4 = 4000 circuit evaluations,
# which is comparable to expressibility's 5000 * 2 = 10000 evaluations.
_DEFAULT_N_SAMPLES: int = 500

# Warning threshold for sample count.
#
# RATIONALE: 50 samples is the minimum for reasonable variance estimation.
# This is LOWER than expressibility/entanglement (100) because:
# 1. Each trainability sample is more expensive, so users may need fewer
# 2. Variance estimation needs fewer samples than histogram estimation
# 3. We only need order-of-magnitude accuracy for barren plateau detection
_MIN_SAMPLES_WARNING: int = 50

# Error threshold for sample count.
# Below this value, we refuse to compute as variance estimates would be
# statistically meaningless and numerically unstable.
_MIN_SAMPLES_ERROR: int = 10

# Default parameter sampling range.
# Parameters are sampled uniformly from [0, 2π], which covers the full
# rotation range for standard quantum gates (RX, RY, RZ).
_DEFAULT_INPUT_RANGE: tuple[float, float] = (0.0, 2.0 * np.pi)

# Barren plateau detection thresholds.
# These thresholds are calibrated based on literature values and empirical
# testing. They should be adjusted based on the specific circuit architecture.
#
# HIGH RISK: Variance below this threshold indicates severe barren plateau.
# The circuit is likely untrainable with gradient-based optimization.
_BP_HIGH_RISK_THRESHOLD: float = 1e-6

# MEDIUM RISK: Variance below this threshold indicates potential trainability
# issues. Training may be slow or require careful hyperparameter tuning.
_BP_MEDIUM_RISK_THRESHOLD: float = 1e-3

# Reference variance for trainability scoring.
#
# MATHEMATICAL JUSTIFICATION:
# For a single-qubit rotation gate Rᵧ(θ) with parameter θ uniform over [0, 2π],
# the expectation value of Z is ⟨Z⟩ = cos(θ). Using the parameter-shift rule:
#
#     ∂⟨Z⟩/∂θ = (⟨Z⟩_{θ+π/2} - ⟨Z⟩_{θ-π/2}) / 2 = -sin(θ)
#
# The variance of -sin(θ) for θ ~ Uniform[0, 2π] is:
#
#     Var[sin(θ)] = E[sin²(θ)] - E[sin(θ)]² = 1/2 - 0 = 0.5
#
# For multi-parameter circuits, gradients typically have lower variance due to:
# 1. Parameter interference effects reducing individual contributions
# 2. Entanglement distributing information across parameters
# 3. Observable locality (local observables see partial circuit effects)
#
# The value 0.1 is chosen as a conservative reference because:
# - It's below the single-parameter maximum (~0.5)
# - It's representative of shallow, structured circuits (2-4 qubits, 1-2 layers)
# - It's well above the barren plateau regime (< 10⁻³)
# - It aligns with empirical observations in QML benchmarks (Cerezo et al., 2021)
#
# The trainability score sigmoid maps variance relative to this reference:
# - variance >> 0.1 → trainability → 1 (highly trainable)
# - variance ≈ 0.1 → trainability ≈ 0.5 (moderately trainable)
# - variance << 0.1 → trainability → 0 (poorly trainable)
_REFERENCE_VARIANCE: float = 0.1

# Numerical stability constant to prevent division by zero.
_EPSILON: float = 1e-15

# Maximum number of failed gradient computations before aborting.
# If more than this fraction of samples fail, the analysis is unreliable.
_MAX_FAILURE_FRACTION: float = 0.2

# Sigmoid steepness for variance-to-trainability mapping.
# Higher values create sharper transition between trainable/untrainable.
_SIGMOID_STEEPNESS: float = 2.0


# =============================================================================
# Type Definitions
# =============================================================================


class TrainabilityResult(TypedDict):
    """Result of trainability estimation.

    This TypedDict defines the structure of the detailed result returned
    by :func:`estimate_trainability` when ``return_details=True``.

    Attributes
    ----------
    trainability_estimate : float
        Overall trainability score in the range [0, 1], where higher values
        indicate better trainability. This score is derived from the gradient
        variance using a logistic transformation.
    gradient_variance : float
        Average variance of parameter gradients computed over successful
        samples only. This is the primary metric for barren plateau detection.
        Values near zero indicate a barren plateau.
    barren_plateau_risk : {"low", "medium", "high"}
        Categorical assessment of barren plateau risk based on gradient
        variance and circuit size:

        - "low": Variance is healthy, training should proceed normally
        - "medium": Variance is borderline, training may be slow
        - "high": Variance is very low, likely barren plateau

    effective_dimension : float
        Estimated effective parameter dimension (HEURISTIC METRIC), computed
        as the count of parameters with variance exceeding 1% of the mean
        variance. This indicates how many parameters contribute meaningfully
        to the optimization landscape. Note: This is not a standard metric
        in barren plateau literature; it is provided for diagnostic purposes.
        For rigorous analysis, examine ``per_parameter_variance`` directly.
    n_samples : int
        Total number of random parameter samples requested for the estimate.
    n_successful_samples : int
        Number of samples where gradient computation succeeded. This is the
        actual sample size used for variance computation. Equal to
        ``n_samples - n_failed_samples``.
    per_parameter_variance : NDArray[np.floating]
        Array of variance values for each individual parameter, computed
        over successful samples only. This can reveal if specific parameters
        are more trainable than others.
    n_failed_samples : int
        Number of samples where gradient computation failed. High values
        may indicate issues with the encoding or simulation backend.
        Failed samples are excluded from variance computation to ensure
        unbiased statistical estimates.
    """

    trainability_estimate: float
    gradient_variance: float
    barren_plateau_risk: Literal["low", "medium", "high"]
    effective_dimension: float
    n_samples: int
    n_successful_samples: int
    per_parameter_variance: NDArray[np.floating[Any]]
    n_failed_samples: int


# =============================================================================
# Main Public Function
# =============================================================================


@overload
def estimate_trainability(
    encoding: BaseEncoding,
    n_samples: int = ...,
    input_range: tuple[float, float] = ...,
    observable: Literal["computational", "pauli_z", "global_z"] = ...,
    seed: int | None = ...,
    backend: Literal["pennylane", "qiskit", "cirq"] = ...,
    return_details: Literal[False] = ...,
    verbose: bool = ...,
) -> float: ...


@overload
def estimate_trainability(
    encoding: BaseEncoding,
    n_samples: int = ...,
    input_range: tuple[float, float] = ...,
    observable: Literal["computational", "pauli_z", "global_z"] = ...,
    seed: int | None = ...,
    backend: Literal["pennylane", "qiskit", "cirq"] = ...,
    return_details: Literal[True] = ...,
    verbose: bool = ...,
) -> TrainabilityResult: ...


def estimate_trainability(
    encoding: BaseEncoding,
    n_samples: int = _DEFAULT_N_SAMPLES,
    input_range: tuple[float, float] = _DEFAULT_INPUT_RANGE,
    observable: Literal["computational", "pauli_z", "global_z"] = "computational",
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
    return_details: bool = False,
    verbose: bool = False,
) -> float | TrainabilityResult:
    """Estimate the trainability of a quantum encoding.

    Trainability is estimated by analyzing the variance of gradients
    with respect to encoding parameters. Low variance (especially
    exponentially decaying with system size) indicates barren plateaus
    and poor trainability.

    This function samples random parameter initializations, computes
    gradients using the parameter-shift rule, and analyzes the variance
    of these gradients to detect barren plateaus.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to analyze. Must be a valid encoding that
        implements the ``get_circuit()`` method.
    n_samples : int, default=500
        Number of random parameter initializations to sample. Higher
        values give more accurate estimates but take longer to compute.
        Minimum value is 10, recommended minimum is 50.
    input_range : tuple[float, float], default=(0, 2π)
        Range for sampling random input/parameter values as ``(min, max)``.
        The default range covers full rotation angles for quantum gates.
    observable : {"computational", "pauli_z", "global_z"}, default="computational"
        Observable to use for gradient computation:

        - ``"computational"``: Probability of |0...0⟩ state (local)
        - ``"pauli_z"``: Expectation value of Z on first qubit (local)
        - ``"global_z"``: Expectation value of Z⊗Z⊗...⊗Z (global)

        Local observables are less prone to false positives for barren
        plateau detection and are recommended for most use cases.
    seed : int, optional
        Random seed for reproducibility. If None, uses system entropy.
        For reproducible results, always specify a seed.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Backend for circuit simulation and gradient computation:

        - ``"pennylane"``: Uses PennyLane's default.qubit simulator
        - ``"qiskit"``: Uses Qiskit's Statevector simulator
        - ``"cirq"``: Uses Cirq's Simulator for statevector simulation
    return_details : bool, default=False
        If True, return full result dictionary with detailed statistics.
        If False, return only the trainability estimate as a float.
    verbose : bool, default=False
        If True, log progress information during computation.

    Returns
    -------
    float or TrainabilityResult
        If ``return_details=False``: Trainability estimate in [0, 1],
        where higher values indicate better trainability.

        If ``return_details=True``: Dictionary containing:

        - ``trainability_estimate``: float in [0, 1]
        - ``gradient_variance``: variance computed over successful samples only
        - ``barren_plateau_risk``: "low", "medium", or "high"
        - ``effective_dimension``: estimated effective parameter dimension
        - ``n_samples``: total number of samples requested
        - ``n_successful_samples``: samples used for variance computation
        - ``per_parameter_variance``: variance for each parameter
        - ``n_failed_samples``: number of failed gradient computations

    Raises
    ------
    InsufficientSamplesError
        If ``n_samples`` < 10.
    SimulationError
        If too many gradient computations fail (> 20% of samples).
    AnalysisError
        If the encoding is invalid or cannot be analyzed.

    Warnings
    --------
    UserWarning
        If ``n_samples`` < 50, a warning is issued about potential
        statistical unreliability.

    Notes
    -----
    **Interpretation:**

    The trainability score should be interpreted as a relative measure
    for comparing encodings, not an absolute guarantee. A score of 0.8
    means the encoding appears more trainable than one with score 0.5,
    but actual training success depends on many other factors.

    **Barren Plateau Risk Levels:**

    - "low" (variance ≥ 1e-3): Training should proceed normally
    - "medium" (1e-6 ≤ variance < 1e-3): May need careful tuning
    - "high" (variance < 1e-6): Likely barren plateau, consider:
      - Using fewer qubits
      - Shallower circuits
      - Local cost functions
      - Parameter initialization strategies

    **Computational Cost:**

    Each sample requires computing gradients for all parameters, which
    involves 2 × n_parameters circuit evaluations (parameter-shift rule).
    Total circuit evaluations = 2 × n_parameters × n_samples.

    **Failed Sample Handling:**

    Gradient computation may fail for certain parameter configurations
    due to numerical instabilities or backend issues. Failed samples are
    completely excluded from variance computation to ensure unbiased
    statistical estimates. If more than 20% of samples fail, a
    :exc:`SimulationError` is raised. The number of successful samples
    used is reported in ``n_successful_samples`` when ``return_details=True``.

    **Best Practices:**

    1. Use at least 100 samples for reliable estimates
    2. Compare encodings using the same seed for fair comparison
    3. Use local observables ("computational" or "pauli_z") unless
       specifically studying global cost functions
    4. For publication, report both trainability score and gradient
       variance with confidence intervals

    Examples
    --------
    Basic usage:

    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> train = estimate_trainability(enc, seed=42)
    >>> print(f"Trainability: {train:.4f}")

    Detailed analysis:

    >>> result = estimate_trainability(
    ...     enc, n_samples=200, seed=42, return_details=True
    ... )
    >>> print(f"Variance: {result['gradient_variance']:.2e}")
    >>> print(f"Risk level: {result['barren_plateau_risk']}")

    Comparing encodings:

    >>> from encoding_atlas import IQPEncoding
    >>> enc1 = AngleEncoding(n_features=4)
    >>> enc2 = IQPEncoding(n_features=4, reps=2)
    >>> t1 = estimate_trainability(enc1, seed=42)
    >>> t2 = estimate_trainability(enc2, seed=42)
    >>> print(f"Angle: {t1:.4f}, IQP: {t2:.4f}")

    See Also
    --------
    compute_gradient_variance : Compute only gradient variance.
    detect_barren_plateau : Detect barren plateau from variance.
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    validate_encoding_for_analysis(encoding)

    if n_samples < _MIN_SAMPLES_ERROR:
        raise InsufficientSamplesError(
            f"n_samples must be at least {_MIN_SAMPLES_ERROR}, got {n_samples}. "
            f"For reliable trainability estimates, use at least "
            f"{_MIN_SAMPLES_WARNING} samples.",
            requested_samples=n_samples,
            minimum_samples=_MIN_SAMPLES_ERROR,
            metric="trainability",
        )

    if n_samples < _MIN_SAMPLES_WARNING:
        warnings.warn(
            f"n_samples={n_samples} is below the recommended minimum of "
            f"{_MIN_SAMPLES_WARNING}. Trainability estimates may be unreliable. "
            f"Consider increasing n_samples for more accurate results.",
            UserWarning,
            stacklevel=2,
        )

    if input_range[0] >= input_range[1]:
        raise ValueError(
            f"input_range must satisfy min < max, got {input_range}. "
            f"Provide a valid range like (0.0, 2*np.pi)."
        )

    valid_observables = {"computational", "pauli_z", "global_z"}
    if observable not in valid_observables:
        raise ValueError(
            f"observable must be one of {sorted(valid_observables)}, "
            f"got {observable!r}."
        )

    valid_backends = {"pennylane", "qiskit", "cirq"}
    if backend not in valid_backends:
        raise ValueError(
            f"backend must be one of {sorted(valid_backends)}, got {backend!r}."
        )

    # =========================================================================
    # Setup
    # =========================================================================
    rng = create_rng(seed)
    n_features = encoding.n_features
    n_qubits = encoding.n_qubits

    # Estimate number of parameters from encoding properties
    try:
        n_params = encoding.properties.parameter_count
    except Exception:
        # Fallback: assume n_features parameters
        n_params = n_features
        _logger.debug(
            "Could not get parameter_count from properties, "
            "using n_features=%d as fallback",
            n_features,
        )

    if verbose:
        _logger.info(
            "Estimating trainability for %s (n_qubits=%d, n_params=%d, "
            "n_samples=%d, observable=%r, backend=%r)",
            encoding.__class__.__name__,
            n_qubits,
            n_params,
            n_samples,
            observable,
            backend,
        )

    # =========================================================================
    # Sample Gradients
    # =========================================================================
    #
    # COMPUTATIONAL COMPLEXITY:
    # Each sample requires computing gradients for all n_features parameters.
    # Using the parameter-shift rule, this requires 2 circuit evaluations per
    # parameter. Total circuit simulations:
    #
    #     Total = 2 × n_features × n_samples
    #
    # For default values (n_samples=500) with typical encodings:
    #     - 4 features: 4,000 simulations
    #     - 8 features: 8,000 simulations
    #     - 16 features: 16,000 simulations
    #
    # Each simulation is O(2^n_qubits) for statevector computation.
    #
    # MEMORY LAYOUT:
    # Pre-allocate array for gradient samples. Successful samples are stored
    # contiguously starting from index 0. Failed samples are not stored,
    # ensuring variance computation uses only valid data.
    #
    gradient_samples = np.zeros((n_samples, n_features), dtype=np.float64)
    n_successful = 0
    n_failed = 0

    # Log progress at 10% intervals
    log_interval = max(1, n_samples // 10)

    for i in range(n_samples):
        # Generate random input parameters
        x = rng.uniform(input_range[0], input_range[1], size=n_features)
        x = x.astype(np.float64)

        try:
            # Compute gradients using parameter-shift rule
            gradients = _compute_encoding_gradients(
                encoding=encoding,
                x=x,
                observable=observable,
                backend=backend,
            )

            # Handle dimension mismatch (some encodings have different param count)
            if len(gradients) != n_features:
                gradients = _pad_or_truncate(gradients, n_features)

            # Store successful sample contiguously at the next available index
            gradient_samples[n_successful, :] = gradients
            n_successful += 1

        except (SimulationError, NumericalInstabilityError) as e:
            # Log but continue - single failure shouldn't abort entire analysis.
            # Failed samples are NOT stored, ensuring unbiased variance estimation.
            n_failed += 1
            _logger.debug(
                "Gradient computation failed at sample %d/%d: %s",
                i + 1,
                n_samples,
                str(e),
            )

        except Exception as e:
            # Unexpected error - log warning but continue.
            # Failed samples are excluded from variance computation.
            n_failed += 1
            _logger.warning(
                "Unexpected error during gradient computation at sample %d: %s",
                i + 1,
                str(e),
            )

        if verbose and (i + 1) % log_interval == 0:
            _logger.info(
                "Progress: %d/%d samples completed (%d successful, %d failed)",
                i + 1,
                n_samples,
                n_successful,
                n_failed,
            )

    # Check if too many samples failed
    failure_fraction = n_failed / n_samples
    if failure_fraction > _MAX_FAILURE_FRACTION:
        raise SimulationError(
            f"Too many gradient computations failed: {n_failed}/{n_samples} "
            f"({100*failure_fraction:.1f}%). This may indicate issues with the "
            f"encoding or the simulation backend. Try a different backend or "
            f"check if the encoding is valid.",
            backend=backend,
            details={
                "n_failed": n_failed,
                "n_samples": n_samples,
                "n_successful": n_successful,
                "failure_fraction": failure_fraction,
            },
        )

    # =========================================================================
    # Compute Statistics
    # =========================================================================
    # Validate we have enough successful samples for meaningful statistics
    if n_successful < _MIN_SAMPLES_ERROR:
        raise SimulationError(
            f"Only {n_successful} successful samples, need at least "
            f"{_MIN_SAMPLES_ERROR}. Gradient computation is failing too often.",
            backend=backend,
            details={"n_successful": n_successful, "n_failed": n_failed},
        )

    # Extract only the successful samples for variance computation.
    # This ensures unbiased estimation - failed samples are completely excluded.
    successful_gradients = gradient_samples[:n_successful, :]

    # Per-parameter variance across successful samples only.
    # Using ddof=1 for unbiased sample variance (Bessel's correction).
    per_param_variance = np.var(successful_gradients, axis=0, ddof=1)

    # Handle edge case: single feature
    if len(per_param_variance) == 0:
        per_param_variance = np.array([0.0], dtype=np.float64)

    # Overall gradient variance (mean of per-parameter variances)
    # This is the primary metric for barren plateau detection
    gradient_variance = float(np.mean(per_param_variance))

    # =========================================================================
    # Effective Dimension Computation (Heuristic Metric)
    # =========================================================================
    #
    # DEFINITION:
    # Effective dimension counts how many parameters have "meaningful"
    # gradient variance, indicating they contribute to the optimization.
    #
    # THRESHOLD RATIONALE:
    # We use 1% of the mean variance as the significance threshold. This is
    # a HEURISTIC choice, not derived from theory, with the following logic:
    #
    # 1. Parameters with variance < 1% of mean contribute negligibly to
    #    the average gradient signal and may be effectively "frozen"
    # 2. The 1% threshold (two orders of magnitude) is commonly used in
    #    statistics for distinguishing signal from noise
    # 3. Using a relative threshold (vs. absolute) adapts to different
    #    circuit architectures and observable choices
    #
    # INTERPRETATION:
    # - effective_dimension ≈ n_params: All parameters are trainable
    # - effective_dimension << n_params: Many parameters are frozen/redundant
    # - effective_dimension = 0: Complete barren plateau (all variances tiny)
    #
    # NOTE: This metric is provided for diagnostic purposes only. It is not
    # a standard measure in the barren plateau literature and should be
    # interpreted with caution. For rigorous analysis, examine the full
    # per_parameter_variance distribution instead.
    #
    variance_threshold = max(gradient_variance * 0.01, _EPSILON)
    effective_dimension = float(np.sum(per_param_variance > variance_threshold))

    _logger.debug(
        "Gradient statistics: variance=%.2e, effective_dim=%.1f, "
        "n_successful=%d, n_failed=%d",
        gradient_variance,
        effective_dimension,
        n_successful,
        n_failed,
    )

    # =========================================================================
    # Barren Plateau Detection
    # =========================================================================
    barren_plateau_risk = detect_barren_plateau(
        gradient_variance=gradient_variance,
        n_qubits=n_qubits,
        n_params=n_params,
    )

    # =========================================================================
    # Compute Trainability Score
    # =========================================================================
    trainability_estimate = _variance_to_trainability(variance=gradient_variance)

    if verbose:
        _logger.info(
            "Trainability estimation complete: score=%.4f, variance=%.2e, "
            "risk=%s, effective_dim=%.1f, samples_used=%d/%d",
            trainability_estimate,
            gradient_variance,
            barren_plateau_risk,
            effective_dimension,
            n_successful,
            n_samples,
        )

    # =========================================================================
    # Return Results
    # =========================================================================
    if return_details:
        return TrainabilityResult(
            trainability_estimate=trainability_estimate,
            gradient_variance=gradient_variance,
            barren_plateau_risk=barren_plateau_risk,
            effective_dimension=effective_dimension,
            n_samples=n_samples,
            n_successful_samples=n_successful,
            per_parameter_variance=per_param_variance,
            n_failed_samples=n_failed,
        )

    return trainability_estimate


# =============================================================================
# Supporting Public Functions
# =============================================================================


def compute_gradient_variance(
    encoding: BaseEncoding,
    n_samples: int = _DEFAULT_N_SAMPLES,
    input_range: tuple[float, float] = _DEFAULT_INPUT_RANGE,
    observable: Literal["computational", "pauli_z", "global_z"] = "computational",
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> float:
    """Compute the average variance of parameter gradients.

    This is the core metric used for trainability estimation. It measures
    how much the gradients vary across random parameter initializations.
    Low variance indicates a flat optimization landscape (barren plateau).

    This function is a convenience wrapper that calls :func:`estimate_trainability`
    and extracts only the gradient variance.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    n_samples : int, default=500
        Number of random parameter samples.
    input_range : tuple[float, float], default=(0, 2π)
        Range for sampling random parameters.
    observable : {"computational", "pauli_z", "global_z"}, default="computational"
        Observable for gradient computation.
    seed : int, optional
        Random seed for reproducibility.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Backend for gradient computation.

    Returns
    -------
    float
        Average gradient variance across parameters. Values close to zero
        indicate a barren plateau.

    Examples
    --------
    >>> from encoding_atlas import AngleEncoding
    >>> enc = AngleEncoding(n_features=4)
    >>> variance = compute_gradient_variance(enc, n_samples=100, seed=42)
    >>> print(f"Gradient variance: {variance:.2e}")

    See Also
    --------
    estimate_trainability : Full trainability analysis with risk assessment.
    detect_barren_plateau : Categorize variance into risk levels.
    """
    result = estimate_trainability(
        encoding=encoding,
        n_samples=n_samples,
        input_range=input_range,
        observable=observable,
        seed=seed,
        backend=backend,
        return_details=True,
    )
    return result["gradient_variance"]


def detect_barren_plateau(
    gradient_variance: float,
    n_qubits: int,
    n_params: int,
) -> Literal["low", "medium", "high"]:
    """Detect barren plateau risk based on gradient variance.

    Compares observed gradient variance to theoretical scaling for
    barren plateaus (exponential decay with qubit count) and returns
    a categorical risk assessment.

    The thresholds are adjusted based on system size to account for
    the natural scaling of gradient variance with qubit count.

    Parameters
    ----------
    gradient_variance : float
        Observed gradient variance from gradient sampling. Must be
        non-negative.
    n_qubits : int
        Number of qubits in the circuit. Used to adjust thresholds
        based on expected scaling.
    n_params : int
        Number of parameters in the circuit. Currently used for
        logging but may be used for more sophisticated detection
        in future versions.

    Returns
    -------
    {"low", "medium", "high"}
        Barren plateau risk level:

        - "low": Variance is healthy, gradient-based training should work
        - "medium": Variance is borderline, may need careful tuning
        - "high": Variance is very low, likely barren plateau

    Notes
    -----
    **Risk Threshold Calibration:**

    The risk thresholds are calibrated based on literature values:

    - McClean et al. (2018) showed variance scales as O(1/2ⁿ) for
      random circuits with global cost functions
    - Cerezo et al. (2021) showed local cost functions have better
      scaling: O(1/n) for certain circuit architectures

    The thresholds are adjusted as follows:

    - Base high-risk threshold: 1e-6
    - Base medium-risk threshold: 1e-3
    - Size adjustment: thresholds scale with 2^(-n) to account for
      natural variance decay

    **Practical Interpretation:**

    - "low" risk: Proceed with standard optimization
    - "medium" risk: Consider using:
      - Adaptive learning rates
      - Better parameter initialization (e.g., identity-initialized)
      - Noise injection during training
    - "high" risk: Fundamental changes needed:
      - Reduce circuit depth
      - Use local cost functions
      - Try hardware-efficient ansatze
      - Consider layer-wise training

    Examples
    --------
    >>> risk = detect_barren_plateau(
    ...     gradient_variance=1e-2,
    ...     n_qubits=4,
    ...     n_params=16
    ... )
    >>> print(f"Risk level: {risk}")
    Risk level: low

    >>> risk = detect_barren_plateau(
    ...     gradient_variance=1e-8,
    ...     n_qubits=10,
    ...     n_params=40
    ... )
    >>> print(f"Risk level: {risk}")
    Risk level: high

    See Also
    --------
    estimate_trainability : Complete trainability analysis.
    compute_gradient_variance : Compute gradient variance.
    """
    # Validate inputs
    if gradient_variance < 0:
        raise ValueError(
            f"gradient_variance must be non-negative, got {gradient_variance}"
        )
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be at least 1, got {n_qubits}")
    if n_params < 1:
        raise ValueError(f"n_params must be at least 1, got {n_params}")

    # =========================================================================
    # Threshold Scaling Based on System Size
    # =========================================================================
    #
    # THEORETICAL BACKGROUND:
    # Barren plateau severity depends on system size. McClean et al. (2018)
    # showed that for deep random circuits with global cost functions:
    #
    #     Var[∂⟨O⟩/∂θ] ∝ 2^(-n)  [exponential decay]
    #
    # However, Cerezo et al. (2021) demonstrated that local cost functions
    # have more favorable scaling:
    #
    #     Var[∂⟨O⟩/∂θ] ∝ 1/poly(n)  [polynomial decay]
    #
    # SCALING FORMULA DERIVATION:
    # We use a moderate scaling factor: 2^(-(n-4)/2) for n > 4
    #
    # This formula is chosen because:
    # 1. It interpolates between full exponential (2^(-n)) and no decay (1)
    # 2. The exponent -(n-4)/2 means:
    #    - n=4: factor = 1.0 (baseline, no adjustment)
    #    - n=6: factor = 2^(-1) = 0.5
    #    - n=8: factor = 2^(-2) = 0.25
    #    - n=10: factor = 2^(-3) = 0.125
    # 3. This ~O(2^(-n/2)) scaling matches the geometric mean of:
    #    - Worst case: O(2^(-n)) for random circuits [McClean et al.]
    #    - Best case: O(1) for very shallow/structured circuits
    # 4. The offset (n-4) ensures small circuits (≤4 qubits) use
    #    unmodified thresholds, as barren plateaus are less severe there.
    #
    # PRACTICAL EFFECT:
    # As system size increases, we lower our expectations for gradient
    # variance, making the detection more lenient for larger systems
    # where some natural variance decay is expected even for trainable circuits.
    #
    # References:
    # - McClean et al. (2018), Nature Communications 9(1), 4812
    # - Cerezo et al. (2021), Nature Communications 12(1), 1791
    #
    if n_qubits <= 4:  # noqa: SIM108
        # Small systems: use base thresholds without adjustment
        size_factor = 1.0
    else:
        # Larger systems: apply moderate exponential scaling
        # Formula: 2^(-(n-4)/2) gives ~O(2^(-n/2)) overall scaling
        size_factor = 2.0 ** (-(n_qubits - 4) / 2.0)

    high_threshold = _BP_HIGH_RISK_THRESHOLD * size_factor
    medium_threshold = _BP_MEDIUM_RISK_THRESHOLD * size_factor

    # Enforce minimum thresholds to prevent numerical issues.
    # These floors ensure we don't classify everything as "low risk"
    # for very large systems where size_factor becomes tiny.
    # - 1e-12 for high risk: ~40 qubits at full 2^(-n) scaling
    # - 1e-9 for medium risk: maintains 3 orders of magnitude gap
    high_threshold = max(high_threshold, 1e-12)
    medium_threshold = max(medium_threshold, 1e-9)

    _logger.debug(
        "Barren plateau detection: variance=%.2e, n_qubits=%d, "
        "high_thresh=%.2e, medium_thresh=%.2e",
        gradient_variance,
        n_qubits,
        high_threshold,
        medium_threshold,
    )

    if gradient_variance < high_threshold:
        return "high"
    elif gradient_variance < medium_threshold:
        return "medium"
    else:
        return "low"


# =============================================================================
# Private Helper Functions
# =============================================================================


def _compute_encoding_gradients(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    observable: str,
    backend: str,
) -> NDArray[np.floating[Any]]:
    """Compute gradients of expectation value with respect to input parameters.

    Uses the parameter-shift rule for exact (analytical) gradient computation.
    This is the standard method for differentiating quantum circuits with
    gates of the form exp(-iθG/2) where G² = I (e.g., Pauli rotations).

    Mathematical Background
    -----------------------
    For a parameterized gate U(θ) = exp(-iθG/2) where G is a generator with
    eigenvalues ±1, the parameter-shift rule gives the exact gradient:

        ∂⟨O⟩/∂θ = (⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}) / 2

    This formula is exact (not an approximation) and does not suffer from
    the numerical instabilities of finite-difference methods.

    Reference: Schuld et al. (2019), "Evaluating analytic gradients on
    quantum hardware", Physical Review A, 99(3), 032331.

    Computational Complexity
    ------------------------
    This function requires **2 × n_features** circuit simulations:
    - For each of the n_features parameters, we evaluate the circuit twice
      (forward shift θ+π/2 and backward shift θ-π/2)

    For the full trainability estimation with n_samples:
        Total simulations = 2 × n_features × n_samples

    Example: 500 samples, 4 features → 4,000 circuit evaluations

    Memory Optimization
    -------------------
    This implementation reuses a single working array for shifted parameters
    instead of allocating 2×n_features copies, reducing memory pressure
    especially for large feature counts or when called repeatedly.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance to differentiate.
    x : NDArray[np.floating]
        Input parameter vector of shape (n_features,). Not modified.
    observable : str
        Observable type: "computational", "pauli_z", or "global_z".
    backend : str
        Backend for circuit simulation.

    Returns
    -------
    NDArray[np.floating]
        Gradient vector of shape (n_features,), where element i is
        ∂⟨O⟩/∂xᵢ evaluated at the input point x.

    Raises
    ------
    SimulationError
        If statevector simulation fails for any shifted parameter.
    NumericalInstabilityError
        If gradient computation produces NaN or Inf values, indicating
        numerical issues in the expectation value computation.

    Notes
    -----
    Future optimization: If the backend supports batch simulation, the
    2×n_features evaluations could be parallelized. Currently, simulations
    are performed sequentially due to backend API constraints.
    """
    n_features = len(x)
    gradients = np.zeros(n_features, dtype=np.float64)

    # Parameter-shift value: π/2 for standard Pauli rotation gates.
    # This is exact for gates of the form exp(-iθP/2) where P is a Pauli matrix.
    shift = np.pi / 2.0

    # Pre-allocate a single working array for shifted parameters.
    # This avoids 2×n_features array allocations in the loop.
    x_shifted = x.copy()

    for i in range(n_features):
        # Forward shift: evaluate at θᵢ + π/2
        x_shifted[i] = x[i] + shift
        exp_plus = _compute_expectation_value(
            encoding=encoding,
            x=x_shifted,
            observable=observable,
            backend=backend,
        )

        # Backward shift: evaluate at θᵢ - π/2
        x_shifted[i] = x[i] - shift
        exp_minus = _compute_expectation_value(
            encoding=encoding,
            x=x_shifted,
            observable=observable,
            backend=backend,
        )

        # Restore parameter for next iteration
        x_shifted[i] = x[i]

        # Compute gradient using parameter-shift rule
        # This is mathematically exact for Pauli rotation gates
        gradient = (exp_plus - exp_minus) / 2.0

        # Validate numerical stability
        if np.isnan(gradient) or np.isinf(gradient):
            raise NumericalInstabilityError(
                f"Gradient computation produced invalid value: {gradient}",
                value=gradient,
                operation="parameter_shift",
                details={
                    "param_index": i,
                    "exp_plus": exp_plus,
                    "exp_minus": exp_minus,
                    "x_original": float(x[i]),
                },
            )

        gradients[i] = gradient

    return gradients


def _compute_expectation_value(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    observable: str,
    backend: str,
) -> float:
    """Compute expectation value of observable for given input parameters.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding instance.
    x : NDArray[np.floating]
        Input parameter vector.
    observable : str
        Observable type.
    backend : str
        Simulation backend.

    Returns
    -------
    float
        Expectation value of the observable.

    Raises
    ------
    SimulationError
        If statevector simulation fails.
    ValueError
        If observable type is unknown.
    """
    # Simulate to get statevector
    statevector = simulate_encoding_statevector(encoding, x, backend=backend)
    n_qubits = encoding.n_qubits

    if observable == "computational":
        # ⟨0...0|ρ|0...0⟩ = |⟨0...0|ψ⟩|² = |ψ[0]|²
        # Probability of measuring all zeros
        expectation = float(np.abs(statevector[0]) ** 2)

    elif observable == "pauli_z":
        # ⟨Z₀⟩ = P(0 on qubit 0) - P(1 on qubit 0)
        # For local observable on first qubit
        prob_zero = _compute_prob_zero_first_qubit(statevector, n_qubits)
        expectation = 2.0 * prob_zero - 1.0  # P(0) - P(1) = 2*P(0) - 1

    elif observable == "global_z":
        # ⟨Z⊗Z⊗...⊗Z⟩ - global Pauli Z string
        # Eigenvalue is (-1)^(number of 1s in bitstring)
        dim = len(statevector)
        z_eigenvalues = np.array(
            [1.0 - 2.0 * (bin(i).count("1") % 2) for i in range(dim)],
            dtype=np.float64,
        )
        probabilities = np.abs(statevector) ** 2
        expectation = float(np.sum(z_eigenvalues * probabilities))

    else:
        raise ValueError(f"Unknown observable type: {observable!r}")

    return expectation


def _compute_prob_zero_first_qubit(
    statevector: NDArray[np.complexfloating[Any, Any]],
    n_qubits: int,
) -> float:
    """Compute probability of measuring 0 on the first qubit.

    The first qubit corresponds to the most significant bit (MSB) in the
    computational basis ordering used by this module.

    Mathematical Details
    --------------------
    In the computational basis |i⟩ where i ∈ {0, 1, ..., 2ⁿ-1}, the first
    qubit (qubit 0) corresponds to the most significant bit. For a state
    |ψ⟩ = Σᵢ αᵢ|i⟩, the probability of measuring 0 on the first qubit is:

        P(q₀ = 0) = Σᵢ |αᵢ|² where bit (n-1) of i is 0

    This is equivalent to summing over the first half of basis states
    (indices 0 to 2^(n-1) - 1), since these are exactly the states where
    the MSB is 0.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        Statevector of shape (2^n_qubits,). Must be a valid quantum state
        (normalized, though normalization is not enforced here).
    n_qubits : int
        Number of qubits. Must satisfy len(statevector) == 2^n_qubits.

    Returns
    -------
    float
        Probability of measuring 0 on the first qubit, in range [0, 1].

    Notes
    -----
    This implementation uses vectorized NumPy operations for efficiency.
    The mask selects indices where the MSB (bit n_qubits-1) is 0, which
    corresponds to the first half of the Hilbert space.
    """
    dim = len(statevector)

    # Vectorized computation using NumPy boolean indexing.
    # The MSB (first qubit) is bit (n_qubits - 1) in the binary representation.
    # We select all basis states where this bit is 0.
    #
    # Mathematical equivalence:
    #   indices where (i >> (n_qubits - 1)) & 1 == 0
    #   ≡ indices in range [0, 2^(n_qubits-1))
    #   ≡ first half of the Hilbert space
    indices = np.arange(dim, dtype=np.uint64)
    mask = ((indices >> (n_qubits - 1)) & 1) == 0

    # Sum of |αᵢ|² for all i where first qubit is 0
    prob_zero = float(np.sum(np.abs(statevector[mask]) ** 2))

    return prob_zero


def _variance_to_trainability(variance: float) -> float:
    """Convert gradient variance to trainability score in [0, 1].

    Uses a logistic (sigmoid) transformation on the log-ratio of variance
    to a reference value, mapping the infinite range of possible variances
    to a bounded [0, 1] score.

    Mathematical Formulation
    ------------------------
    The transformation is:

        r = log₁₀(σ² / σ²_ref)
        trainability = 1 / (1 + exp(-k · r))

    where:
    - σ² is the observed gradient variance
    - σ²_ref = 0.1 is the reference variance (see _REFERENCE_VARIANCE)
    - k = 2.0 is the sigmoid steepness (see _SIGMOID_STEEPNESS)

    Properties:
    - σ² = σ²_ref → r = 0 → trainability = 0.5
    - σ² >> σ²_ref → r >> 0 → trainability → 1
    - σ² << σ²_ref → r << 0 → trainability → 0

    The log-ratio ensures equal multiplicative changes in variance produce
    equal additive changes in the sigmoid input, making the score invariant
    to the scale of variance (only relative magnitude matters).

    Parameters
    ----------
    variance : float
        Gradient variance (must be non-negative).

    Returns
    -------
    float
        Trainability score in [0, 1].
    """
    # Handle edge case: zero or negative variance
    if variance <= 0:
        return 0.0

    # Reference variance for comparison
    reference = _REFERENCE_VARIANCE

    # Compute log ratio
    # Positive when variance > reference (good trainability)
    # Negative when variance < reference (poor trainability)
    log_ratio = np.log10(variance / reference)

    # Sigmoid transformation: maps (-∞, +∞) to (0, 1)
    # trainability = 1 / (1 + exp(-k * log_ratio))
    # Higher k = sharper transition
    trainability = 1.0 / (1.0 + np.exp(-_SIGMOID_STEEPNESS * log_ratio))

    # Clamp to [0, 1] for numerical safety
    trainability = float(np.clip(trainability, 0.0, 1.0))

    return trainability


def _pad_or_truncate(
    arr: NDArray[np.floating[Any]],
    target_length: int,
) -> NDArray[np.floating[Any]]:
    """Pad or truncate array to target length.

    Used to handle dimension mismatches between encoding parameters
    and computed gradients.

    Parameters
    ----------
    arr : NDArray[np.floating]
        Input array to resize.
    target_length : int
        Desired length.

    Returns
    -------
    NDArray[np.floating]
        Array of exactly target_length elements.
    """
    current_length = len(arr)

    if current_length == target_length:
        return arr
    elif current_length < target_length:
        # Pad with zeros
        padded = np.zeros(target_length, dtype=arr.dtype)
        padded[:current_length] = arr
        return padded
    else:
        # Truncate
        return arr[:target_length].copy()
