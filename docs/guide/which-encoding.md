# Which Encoding Should I Use?

This page helps you narrow down the right encoding by walking through the key decision factors. Answer the questions below to arrive at a recommendation.

---

## Step 1: What Kind of Data?

| Data Type | Recommended Encodings |
|:----------|:---------------------|
| **Continuous features** (floats) | Angle, IQP, ZZ, Data Re-uploading, Trainable |
| **Binary / categorical** | Basis Encoding |
| **High-dimensional** (>16 features) | Amplitude Encoding (qubit-efficient) or Angle (simple) |
| **Data with known symmetry** | SO(2), Cyclic, Swap Equivariant |

---

## Step 2: What Is Your Priority?

### Accuracy

Choose encodings with high expressibility and entanglement capability:

- **IQP Encoding** — provably hard kernel, excellent for classification
- **Data Re-uploading** — universal approximation with enough layers
- **ZZ Feature Map** — strong pairwise feature interactions

### Trainability

Choose shallow encodings with limited entanglement:

- **Angle Encoding** — no entanglement, no barren plateaus
- **Hardware Efficient** — shallow, tunable depth
- **Trainable Encoding** — learnable parameters, controllable expressibility

### Hardware Compatibility

Choose low-depth encodings with linear connectivity:

- **Angle Encoding** — depth 1, no two-qubit gates
- **IQP (linear entanglement)** — O(n) CNOTs instead of O(n^2)
- **Hardware Efficient** — designed for native gate sets

### Qubit Efficiency

- **Amplitude Encoding** — log(n) qubits for n features (but exponential depth)

---

## Step 3: Constraints

| Constraint | Avoid | Prefer |
|:-----------|:------|:-------|
| < 10 qubits available | Angle with many features | Amplitude, or reduce features |
| Noisy hardware (high error rates) | Deep circuits (IQP full, Amplitude) | Angle, Hardware Efficient |
| Need classical baseline comparison | Simulable encodings (can't show advantage) | IQP, ZZ (non-simulable) |
| Training budget is small | High-rep entangling encodings | Angle, shallow Data Re-uploading |

---

## Decision Matrix

| Encoding | Expressibility | Trainability | NISQ-Friendly | Quantum Advantage |
|:---------|:-:|:-:|:-:|:-:|
| Angle | Low | High | Yes | No |
| Amplitude | Maximal | Moderate | No | Possible |
| Basis | Minimal | High | Yes | No |
| IQP | High | Moderate | Moderate | Yes |
| ZZ Feature Map | High | Moderate | Moderate | Yes |
| Pauli Feature Map | High | Moderate | Moderate | Depends |
| Data Re-uploading | High | Moderate | Yes (if shallow) | Possible |
| Hardware Efficient | Moderate | High | Yes | Possible |
| Equivariant | Targeted | High | Moderate | On symmetric data |

---

## Still Unsure?

Use the programmatic recommender:

```python
from encoding_atlas.guide import recommend_encoding

rec = recommend_encoding(
    n_features=4,
    n_samples=500,
    task='classification',
    hardware='simulator',
    priority='accuracy'
)
print(rec.encoding_name, '-', rec.explanation)
```

Or see the [Decision Flowchart](decision-flowchart.md) for a visual walkthrough.
