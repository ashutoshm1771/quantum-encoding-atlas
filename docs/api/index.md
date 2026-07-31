# API Reference

Complete programmatic interface for the Quantum Encoding Atlas. All public classes and functions are documented below with their signatures, parameters, and return types.

---

## Encodings

All encodings inherit from `BaseEncoding` and share a unified interface.

### Base Class

::: encoding_atlas.core.base.BaseEncoding
    options:
      show_root_heading: true
      members:
        - n_qubits
        - depth
        - properties
        - config
        - get_circuit

### Properties

::: encoding_atlas.core.properties.EncodingProperties
    options:
      show_root_heading: true

---

## Encoding Classes

### Angle Encoding

::: encoding_atlas.AngleEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Amplitude Encoding

::: encoding_atlas.AmplitudeEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Basis Encoding

::: encoding_atlas.BasisEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### IQP Encoding

::: encoding_atlas.IQPEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### ZZ Feature Map

::: encoding_atlas.ZZFeatureMap
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Pauli Feature Map

::: encoding_atlas.PauliFeatureMap
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Data Re-uploading

::: encoding_atlas.DataReuploading
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Hardware Efficient Encoding

::: encoding_atlas.HardwareEfficientEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

---

## Analysis Module

::: encoding_atlas.analysis
    options:
      show_root_heading: true
      show_source: false

### Kernel-Concentration Diagnostics

How an encoding's fidelity kernel approaches the Haar floor as the circuit
widens — the axis that says whether a fixed-width result transfers. See
[Kernel Concentration](../concepts/kernel-concentration.md) for the theory.

::: encoding_atlas.analysis.compute_kernel_concentration
    options:
      show_root_heading: true

::: encoding_atlas.analysis.estimate_concentration_scaling
    options:
      show_root_heading: true

::: encoding_atlas.analysis.haar_kernel_moments
    options:
      show_root_heading: true

::: encoding_atlas.analysis.ConcentrationResult
    options:
      show_root_heading: true
      members:
        - shots_for_dataset

::: encoding_atlas.analysis.ScalingResult
    options:
      show_root_heading: true
      members:
        - concentration_horizon
        - shots_per_entry_at
        - concentration_ratios
        - offdiagonal_variances
        - offdiagonal_means

### Finite-Shot Kernel Estimation

::: encoding_atlas.analysis.sample_shot_kernel
    options:
      show_root_heading: true

---

## Guide Module

::: encoding_atlas.guide.recommender.recommend_encoding
    options:
      show_root_heading: true

::: encoding_atlas.guide.recommender.Recommendation
    options:
      show_root_heading: true

::: encoding_atlas.guide.rules.get_matching_encodings
    options:
      show_root_heading: true

---

## Benchmark Module

Evaluate encodings on classification tasks with variational quantum classifiers
and quantum-kernel SVMs, paired stratified cross-validation, classical
baselines, and statistical comparison (Wilcoxon + Holm–Bonferroni + Cliff's
delta).

::: encoding_atlas.benchmark.EncodingBenchmark
    options:
      show_root_heading: true
      members:
        - run
        - statistical_tests
        - plot_comparison
        - save_results

::: encoding_atlas.benchmark.evaluate_encoding
    options:
      show_root_heading: true

::: encoding_atlas.benchmark.VQCClassifier
    options:
      show_root_heading: true
      members:
        - fit
        - predict
        - score

::: encoding_atlas.benchmark.QuantumKernelClassifier
    options:
      show_root_heading: true
      members:
        - fit
        - predict
        - score

::: encoding_atlas.benchmark.compute_kernel_matrix
    options:
      show_root_heading: true

::: encoding_atlas.benchmark.kernel_target_alignment
    options:
      show_root_heading: true

::: encoding_atlas.benchmark.compare_encodings_corrected
    options:
      show_root_heading: true

---

## Atlas Module

The empirical benchmark results — measured circuit resources, simulability,
expressibility, entanglement, trainability, noise resilience, and downstream
VQC / quantum-kernel accuracy for all 16 encodings — bundled with the package
as a queryable, read-only API.

::: encoding_atlas.atlas.get_encoding_profile
    options:
      show_root_heading: true

::: encoding_atlas.atlas.rank_encodings
    options:
      show_root_heading: true

::: encoding_atlas.atlas.pareto_front
    options:
      show_root_heading: true

::: encoding_atlas.atlas.hypothesis_verdicts
    options:
      show_root_heading: true

::: encoding_atlas.atlas.atlas_metadata
    options:
      show_root_heading: true

::: encoding_atlas.atlas.EncodingProfile
    options:
      show_root_heading: true

### Kernel-Concentration Scan

The companion dataset recording how each encoding's fidelity kernel approaches
the Haar floor across 2–8 qubits. The accuracy numbers above were measured at
2–4 qubits, so this is the axis that says whether they transfer.

::: encoding_atlas.atlas.get_concentration_profile
    options:
      show_root_heading: true

::: encoding_atlas.atlas.list_concentration_profiles
    options:
      show_root_heading: true

::: encoding_atlas.atlas.concentrated_encodings
    options:
      show_root_heading: true

::: encoding_atlas.atlas.concentration_metadata
    options:
      show_root_heading: true

::: encoding_atlas.atlas.ConcentrationProfile
    options:
      show_root_heading: true
      members:
        - at_features
        - is_concentrated_at

::: encoding_atlas.atlas.ConcentrationPoint
    options:
      show_root_heading: true

---

## Exceptions

::: encoding_atlas.core.exceptions
    options:
      show_root_heading: true
      show_source: false
