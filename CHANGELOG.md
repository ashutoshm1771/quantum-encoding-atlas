# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Data-driven encoding screening
- `encoding_atlas.guide.screen_encodings` — ranks candidate encodings by
  kernel-target alignment measured on the caller's own ``(X, y)``, with no
  training. This is the workflow the benchmark's validated predictor implies:
  score, keep the encodings that do well, train only those. All 16 candidates
  on 100 samples take about a second.
- `ScreeningResult` (with `top()`, `names()`, `best()`, `get()`) and
  `ScreenedEncoding`, which carries the built instance so the shortlist can be
  trained without rebuilding. Encodings that cannot be constructed at the
  caller's feature count are reported in `skipped` with the reason instead of
  raising.
- Supports restricted candidate sets, stratified and seeded sub-sampling for
  large inputs, finite-shot kernels via `shots=`, and optional kernel
  concentration annotation. Concentration deliberately does **not** affect the
  ranking: at a fixed circuit width alignment already ranks the Haar-floor
  encodings last, so a veto would be redundant.
- Documented honestly: on the benchmark's eight datasets the top-3 shortlist
  reaches 0.973 mean accuracy against an oracle's 0.974, but the single top
  pick (0.960) is not meaningfully better than always choosing `angle`
  (0.958). The shortlist is the deliverable, not the top-1.
- New guide page: *Screening on Your Data*.

#### Kernel-target alignment in the atlas
- `kernel_target_alignment` is now a first-class, rankable atlas metric:
  `rank_encodings(by="kernel_target_alignment")`. Previously the atlas
  exposed `expressibility` — the predictor the study *refutes* (rho = -0.68) —
  while the predictor it *validates* (rho = +0.91) was measured by the pipeline
  and then dropped during consolidation.
- `experiments/report.py` now carries the measured alignment through, so the
  column is derived from existing Stage 6b measurements with no new simulation,
  averaged over (configuration, dataset) pairs — the identical rule the
  `kernel_accuracy` column already used.

#### Analysis
- `encoding_atlas.analysis.summarize_kernel_concentration` — derives the
  concentration statistics from an already-computed kernel, so a caller that
  holds one (screening, for instance) does not pay for a second simulation.

### Fixed
- Kernel-target alignment assumed labels were literally `{0, 1}` and silently
  returned wrong values for any other two-class convention: an ideal kernel
  scored 1.00 for `{0, 1}` but 0.20 for `{1, 2}` and 0.80 for `{-1, +1}`.
  Two-class labels are now mapped to `{-1, +1}` by partition, so the score
  depends on the split rather than on how the classes are spelled. `{0, 1}`
  behaviour, and therefore every published number, is unchanged; continuous
  and multi-class targets keep the previous linear mapping.
- `encoding_atlas.benchmark.kernel` had a second, independently maintained copy
  of both alignment functions that had drifted from the analysis package's.
  It now re-exports the single definition, so the metric the benchmark records
  and the metric users screen with cannot disagree.

### Changed
- The benchmark-matched encoding parameters now live in the package
  (`encoding_atlas.guide._candidates.BENCHMARK_PARAMS`) as the single source of
  truth for both screening and `experiments/concentration_scan.py`, which
  re-exports them as `ENCODING_PARAMS` for backwards compatibility.

#### Kernel-concentration analysis
- `encoding_atlas.analysis.compute_kernel_concentration` — measures how close an
  encoding's fidelity kernel sits to the Haar floor at a given circuit width.
  The order parameter is `concentration_ratio`, the off-diagonal kernel
  variance divided by the Haar-random variance `(d-1)/(d²(d+1))`; a value near
  1 means the kernel has collapsed to the identity up to sampling noise and a
  kernel method cannot generalize from it at any shot budget.
- `encoding_atlas.analysis.estimate_concentration_scaling` — sweeps a factory
  across circuit widths and fits a log-linear decay model, returning
  `decay_rate`, `mean_decay_rate`, `haar_normalized_slope`, `r_squared`, a
  `concentration_horizon()`, and `shots_per_entry_at()` extrapolation. Widths a
  factory rejects are recorded in `skipped` rather than aborting the sweep.
- `encoding_atlas.analysis.haar_kernel_moments` — the Haar reference moments.
- `ConcentrationResult` (with `shots_for_dataset()`) and `ScalingResult`.
- New concept page: *Kernel Concentration*, documenting why the kernel variance
  and not its mean is the order parameter — a product of independent random
  single-qubit states has mean overlap exactly `2⁻ⁿ`, identical to Haar.

#### Finite-shot kernel estimation
- `encoding_atlas.analysis.sample_shot_kernel` — models the compute-uncompute
  estimator, whose all-zeros count is exactly `Binomial(shots, K)`, so shot
  realism costs one random draw rather than a second simulation.
- `shots=` / `seed=` on `compute_fidelity_kernel`,
  `compute_kernel_target_alignment`, `compute_geometric_difference`, and
  `compute_effective_dimension`. Sampled kernels stay symmetric with an exact
  unit diagonal but are not PSD; project with `benchmark.kernel.ensure_psd`.

#### Bundled concentration dataset
- `encoding_atlas/atlas/data/concentration.json` — the scan for all 16
  encodings across 2–8 qubits, generated by the new deterministic
  `experiments/concentration_scan.py` (`--check` verifies the committed file
  still matches the code).
- Atlas API: `get_concentration_profile`, `list_concentration_profiles`,
  `concentrated_encodings`, `concentration_metadata`, plus the
  `ConcentrationProfile` / `ConcentrationPoint` types.
- The four encodings whose kernels reach the Haar floor (`iqp`,
  `zz_feature_map`, `pauli_feature_map`, `hamiltonian`) are exactly the four
  with expressibility ≈ 0.999 and four of the five worst-ranked in the
  benchmark — the mechanism behind the refuted hypothesis H1.

### Changed
- `profile_encoding` now always computes a concentration axis
  (`kernel_concentration_ratio`, `kernel_offdiagonal_mean`,
  `kernel_shots_per_entry`, `kernel_is_concentrated`) and accepts
  `concentration_samples`. The axis is data-free by default and uses `X` when
  supplied. Existing metrics and failure handling are unchanged.
- `guide.Recommendation` gains a `scale_warning` field (default `None`), set
  when the recommended encoding is measured at the Haar floor for the caller's
  feature count. It is kept out of `explanation` so existing callers that
  format the rationale are unaffected.

## [0.1.0] - 2026-02-07

### Added

#### Core Framework
- Abstract base class `BaseEncoding` with thread-safe property caching
- `EncodingProperties` dataclass for encoding metrics (qubits, depth, gates, simulability)
- Encoding registry system with `register_encoding`, `get_encoding`, `list_encodings`
- Type definitions and protocols for static type checking (`py.typed` marker)

#### Encoding Implementations (16 encodings)
- **AngleEncoding**: Single-qubit rotations (RX, RY, RZ) with configurable repetitions
- **AmplitudeEncoding**: Logarithmic qubit encoding via state amplitudes
- **BasisEncoding**: Binary encoding into computational basis states
- **IQPEncoding**: Instantaneous Quantum Polynomial circuits with diagonal gates
- **ZZFeatureMap**: Pauli-ZZ entangling feature map
- **PauliFeatureMap**: Configurable Pauli rotation feature maps
- **HardwareEfficientEncoding**: NISQ-friendly ansatz with hardware-native gates
- **DataReuploading**: Multi-layer data re-uploading with trainable parameters
- **SymmetryInspiredFeatureMap**: Symmetry-inspired quantum feature maps
- **HamiltonianEncoding**: Time-evolution based encoding
- **HigherOrderAngleEncoding**: Higher-order polynomial angle encoding
- **QAOAEncoding**: QAOA-inspired encoding structure
- **TrainableEncoding**: Parameterized encoding with learnable gate parameters
- **SO2EquivariantFeatureMap**: SO(2) rotation-equivariant quantum feature map
- **CyclicEquivariantFeatureMap**: Cyclic group equivariant quantum feature map
- **SwapEquivariantFeatureMap**: Permutation-equivariant quantum feature map

#### Multi-Backend Support
- PennyLane backend (primary)
- Qiskit backend with full circuit generation
- Cirq backend with moment-based circuit construction
- Consistent quantum states across all backends (verified by cross-backend tests)

#### Analysis Tools
- Expressibility calculation
- Entanglement capability metrics
- Trainability estimation via variance of parameter-shift gradients
- Classical simulability analysis (Clifford detection, matchgate detection, entanglement-based)
- Resource counting and estimation (gate counts, depth, qubit requirements)

#### Experiment Framework
- Experiment runner with checkpointing support
- VQC (Variational Quantum Classifier) experiment pipeline
- Kernel-based classification experiment pipeline
- Noise model support for realistic hardware simulation

#### Utilities
- Benchmarking framework for encoding comparison
- Decision guide system for encoding selection
- Visualization tools (`pip install encoding-atlas[visualization]` for matplotlib support)

#### Quality & Infrastructure
- Comprehensive test suite with 80%+ coverage requirement
- Thread safety with double-checked locking pattern
- Pickle serialization support for distributed computing
- Input validation with clear error messages (NaN, Inf, complex, shape)
- Defensive copying for thread-safe input handling

#### Documentation
- NumPy-style docstrings with mathematical background
- Academic references for each encoding
- Usage examples in all public APIs
- Module-level documentation with preprocessing guidance

### Backend Requirements
- **Core**: NumPy ≥1.21, SciPy ≥1.7, PennyLane ≥0.33, scikit-learn ≥1.0
- **Optional**: Qiskit ≥1.0 (`pip install encoding-atlas[qiskit]`)
- **Optional**: Cirq ≥1.0 (`pip install encoding-atlas[cirq]`)
- **Optional**: matplotlib ≥3.5 (`pip install encoding-atlas[visualization]`)
- **All backends**: `pip install encoding-atlas[all]`

### Python Support
- Python 3.9, 3.10, 3.11, 3.12

---

[Unreleased]: https://github.com/encoding-atlas/quantum-encoding-atlas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/encoding-atlas/quantum-encoding-atlas/releases/tag/v0.1.0
