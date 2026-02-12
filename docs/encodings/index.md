# Encodings Reference

The Quantum Encoding Atlas implements **16 quantum data encoding methods** with a unified API across PennyLane, Qiskit, and Cirq backends. Each encoding transforms classical feature vectors into quantum states using different circuit structures, offering distinct tradeoffs in expressibility, trainability, hardware cost, and noise resilience.

---

## At a Glance

| Encoding | Qubits | Entangling | Depth | Best For |
|:---------|:------:|:----------:|:-----:|:---------|
| [Angle](angle/angle_encoding.md) | *n* | No | O(1) | Low-noise baselines, few features |
| [Amplitude](amplitude/amplitude_encoding.md) | log *n* | Yes | O(2^n) | Qubit-limited regimes, many features |
| [Basis](basis/basis_encoding.md) | *n* | No | O(1) | Binary / categorical data |
| [IQP](iqp/iqp_encoding.md) | *n* | Yes | O(n^2) | Quantum kernels, provable hardness |
| [ZZ Feature Map](zz_feature_map/zz_feature_map.md) | *n* | Yes | O(n^2) | Pairwise interactions, kernel methods |
| [Pauli Feature Map](pauli_feature_map/pauli_feature_map.md) | *n* | Yes | O(n^2) | Configurable Pauli rotations |
| [Data Re-uploading](data_reuploading/data_reuploading.md) | *n* | Yes | O(L) | Universal approximation, small circuits |
| [Hardware Efficient](hardware_efficient/hardware_efficient.md) | *n* | Yes | O(L) | NISQ devices, native gate sets |
| [Higher-Order Angle](higher_order_angle/higher_order_angle_encoding.md) | *n* | No | O(n^d) | Polynomial features without entanglement |
| [QAOA-Inspired](qaoa/qaoa.md) | *n* | Yes | O(L) | Combinatorial structure in data |
| [Hamiltonian](hamiltonian/hamiltonian_encoding.md) | *n* | Yes | O(L) | Physics-informed encoding |
| [Trainable](trainable/trainable_encoding.md) | *n* | Yes | O(L) | Learnable encoding layers |
| [Symmetry-Inspired](symmetry_inspired/symmetry_inspired_feature_map.md) | *n* | Yes | O(L) | Heuristic symmetry-aware bias |
| [SO(2) Equivariant](so2_equivariant/so2_equivariant_feature_map.md) | *n* | Yes | O(L) | 2D rotational symmetry |
| [Cyclic Equivariant](cyclic_equivariant/cyclic_equivariant.md) | *n* | Yes | O(L) | Z_n cyclic shift symmetry |
| [Swap Equivariant](swap_equivariant/swap_equivariant_feature_map.md) | *n* | Yes | O(L) | S_2 pair-swap symmetry |

---

## Choosing an Encoding

Not sure where to start? The [Decision Guide](../guide/index.md) walks you through selecting an encoding based on your data, hardware, and priorities.

As a quick rule of thumb:

```
                    Do your features have known symmetry?
                    ├── Yes ──► SO(2) / Cyclic / Swap Equivariant
                    └── No
                         │
                    Do you need provable quantum advantage?
                    ├── Yes ──► IQP / ZZ Feature Map
                    └── No
                         │
                    Are you qubit-limited?
                    ├── Yes ──► Amplitude Encoding
                    └── No
                         │
                    Do you want a trainable feature map?
                    ├── Yes ──► Data Re-uploading / Trainable
                    └── No ──► Angle Encoding (safe default)
```

---

## Taxonomy

### By Circuit Structure

**Product-state encodings** — no entanglement, classically simulable:

- [Angle Encoding](angle/angle_encoding.md) — single-qubit rotations
- [Basis Encoding](basis/basis_encoding.md) — computational basis mapping
- [Higher-Order Angle](higher_order_angle/higher_order_angle_encoding.md) — polynomial rotations

**Diagonal-phase encodings** — Hadamard + phase + entanglement layers:

- [IQP Encoding](iqp/iqp_encoding.md) — provably hard to simulate
- [ZZ Feature Map](zz_feature_map/zz_feature_map.md) — pairwise ZZ interactions
- [Pauli Feature Map](pauli_feature_map/pauli_feature_map.md) — configurable Pauli strings

**Variational encodings** — interleaved data and parametric layers:

- [Data Re-uploading](data_reuploading/data_reuploading.md) — repeated data injection
- [Hardware Efficient](hardware_efficient/hardware_efficient.md) — native gate sets
- [Trainable Encoding](trainable/trainable_encoding.md) — learnable parameters
- [QAOA-Inspired](qaoa/qaoa.md) — cost-mixer structure

**Global state preparation** — amplitude-level encoding:

- [Amplitude Encoding](amplitude/amplitude_encoding.md) — exponential compression

**Equivariant encodings** — symmetry-preserving circuits:

- [SO(2) Equivariant](so2_equivariant/so2_equivariant_feature_map.md) — continuous rotations
- [Cyclic Equivariant](cyclic_equivariant/cyclic_equivariant.md) — discrete cyclic shifts
- [Swap Equivariant](swap_equivariant/swap_equivariant_feature_map.md) — pairwise permutations
- [Symmetry-Inspired](symmetry_inspired/symmetry_inspired_feature_map.md) — heuristic symmetry

### By Simulability

| Classically Simulable | Not Classically Simulable |
|:----------------------|:--------------------------|
| Angle, Basis, Higher-Order Angle | IQP, ZZ, Pauli, Amplitude, Data Re-uploading, Hardware Efficient, QAOA, Hamiltonian, Trainable, all Equivariant maps |

---

## Unified API

Every encoding shares the same interface:

```python
from encoding_atlas import IQPEncoding

enc = IQPEncoding(n_features=4, reps=2)

# Properties
enc.n_qubits          # Number of qubits
enc.depth             # Circuit depth
enc.properties        # Full EncodingProperties dataclass

# Circuit generation
circuit = enc.get_circuit(x, backend='pennylane')

# Configuration
enc.config            # Read-only dict of parameters
```

See the [Quick Start](../quickstart.md) for a hands-on introduction and the [API Reference](../api/index.md) for the complete programmatic interface.
