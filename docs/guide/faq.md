# Frequently Asked Questions

---

### What is the simplest encoding to start with?

**Angle Encoding.** It maps one feature per qubit using single-qubit rotations. No entanglement, no complex circuit structure, fast simulation, and no risk of barren plateaus. It serves as an excellent baseline for comparison.

```python
from encoding_atlas import AngleEncoding
enc = AngleEncoding(n_features=4, rotation='Y')
```

---

### Which encoding gives the best accuracy?

There is no universally best encoding — performance depends on the dataset. However, encodings with high expressibility and entanglement capability (IQP, ZZ Feature Map, Data Re-uploading) tend to perform well on non-trivial classification tasks.

The [Benchmarking tutorial](../tutorials/benchmarking.md) shows how to compare encodings systematically on your specific data.

---

### Can I use quantum encoding without a quantum computer?

Yes. All encodings in the atlas can be simulated classically using PennyLane's `default.qubit` simulator. This is sufficient for research and development up to about 20-25 qubits.

For larger problems, you would need access to quantum hardware (IBM Quantum, Google, IonQ, etc.) through the Qiskit or Cirq backends.

---

### What's the difference between IQP and ZZ Feature Map?

Both use the same H-RZ-ZZ circuit structure and are structurally very similar. The primary difference is the phase convention:

- **IQP Encoding** follows the Havlicek et al. (2019) convention with 2x scaling in RZ gates
- **ZZ Feature Map** may use different parameterisation

For most practical purposes, they behave similarly. IQP is the more standard reference in the quantum advantage literature.

---

### How many qubits do I need?

| Encoding | Qubits for *n* features |
|:---------|:----------------------:|
| Angle, IQP, ZZ, Basis, etc. | *n* |
| Amplitude | ceil(log2(*n*)) |

For 4 features: most encodings need 4 qubits; amplitude encoding needs 2.

---

### What does "not classically simulable" mean?

It means that, under widely believed complexity-theoretic assumptions, no classical computer can efficiently compute the measurement outcomes of the encoding circuit. This is a prerequisite for quantum advantage.

Encodings that **are** classically simulable (Angle, Basis, Higher-Order Angle) can always be replaced by a classical feature map at equal or lower computational cost.

---

### How do I handle features of different scales?

Scale your features to a consistent range before encoding. Most encodings work best with features in [0, pi] or [-pi/2, pi/2]:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)
```

This is especially important for encodings with pairwise interactions (IQP, ZZ) where unscaled products can wrap around 2pi multiple times, wasting information.

---

### Can I combine multiple encodings?

Not directly through the library's API, but you can compose encoding circuits manually by generating circuits from multiple encodings and concatenating them. This is an area of active research (hybrid encoding strategies).

---

### How do equivariant encodings work?

Equivariant encodings build a symmetry constraint into the circuit: if the input data is transformed by a group action (rotation, permutation, cyclic shift), the quantum state transforms accordingly. This acts as a strong inductive bias, reducing the effective hypothesis space and improving generalisation on problems with the matching symmetry.

See the [Concepts: Quantum Advantage](../concepts/quantum-advantage.md) page for the theoretical background.

---

### Where can I learn more?

- **This library's documentation:** [Concepts](../concepts/index.md) and [Encodings Reference](../encodings/index.md)
- **Schuld & Petruccione** (2021), *Machine Learning with Quantum Computers*, Springer
- **Havlicek et al.** (2019), "Supervised learning with quantum-enhanced feature spaces", Nature 567
- **Schuld & Killoran** (2019), "Quantum machine learning in feature Hilbert spaces", PRL 122
