# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Renamed `CovariantFeatureMap` to `SymmetryInspiredFeatureMap` to better
  reflect its heuristic nature. The class provides symmetry-inspired inductive bias but
  does NOT guarantee mathematical equivariance (U(g)|ψ(x)⟩ = |ψ(g·x)⟩).
  - `CovariantFeatureMap` remains available as a backwards-compatible alias
  - Registry keys `covariant` and `covariant_feature_map` still work
  - New recommended keys: `symmetry_inspired` and `symmetry_inspired_feature_map`
  - File renamed from `covariant_feature_map.py` to `symmetry_inspired_feature_map.py`

### Added
- Added detailed implementation guide for true `EquivariantFeatureMap` in
  `external_drivers/EQUIVARIANT_FEATURE_MAP_IMPLEMENTATION_GUIDE.md`

## [0.1.0] - 2026-01-24

### Added

#### Core Framework
- Abstract base class `BaseEncoding` with thread-safe property caching
- `EncodingProperties` dataclass for encoding metrics (qubits, depth, gates, simulability)
- Encoding registry system with `register_encoding`, `get_encoding`, `list_encodings`
- Type definitions and protocols for static type checking (`py.typed` marker)

#### Encoding Implementations
- **AngleEncoding**: Single-qubit rotations (RX, RY, RZ) with configurable repetitions
- **AmplitudeEncoding**: Logarithmic qubit encoding via state amplitudes
- **BasisEncoding**: Binary encoding into computational basis states
- **IQPEncoding**: Instantaneous Quantum Polynomial circuits with diagonal gates
- **ZZFeatureMap**: Pauli-ZZ entangling feature map
- **PauliFeatureMap**: Configurable Pauli rotation feature maps
- **HardwareEfficientEncoding**: NISQ-friendly ansatz with hardware-native gates
- **DataReuploading**: Multi-layer data re-uploading with trainable parameters
- **SymmetryInspiredFeatureMap**: Symmetry-inspired quantum feature maps (formerly CovariantFeatureMap)
- **HamiltonianEncoding**: Time-evolution based encoding
- **HigherOrderAngleEncoding**: Higher-order polynomial angle encoding
- **QAOAEncoding**: QAOA-inspired encoding structure

#### Multi-Backend Support
- PennyLane backend (primary)
- Qiskit backend with full circuit generation
- Cirq backend with moment-based circuit construction
- Consistent quantum states across all backends (verified by cross-backend tests)

#### Analysis Tools
- Expressibility calculation
- Entanglement capability metrics
- Resource estimation (qubits, depth, gate counts)

#### Utilities
- Benchmarking framework for encoding comparison
- Decision guide system for encoding selection
- Visualization tools

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
- **All backends**: `pip install encoding-atlas[all]`

### Python Support
- Python 3.9, 3.10, 3.11, 3.12

---

[Unreleased]: https://github.com/ashutoshm1771/quantum-encoding-atlas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ashutoshm1771/quantum-encoding-atlas/releases/tag/v0.1.0
