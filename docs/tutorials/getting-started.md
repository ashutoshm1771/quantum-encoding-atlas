# Getting Started

This tutorial walks you through the core workflow of the Quantum Encoding Atlas: creating encodings, inspecting their properties, and generating quantum circuits.

!!! abstract "What you'll learn"
    - Creating encoding instances with different configurations
    - Accessing encoding properties (qubits, depth, gates, simulability)
    - Generating circuits for PennyLane, Qiskit, and Cirq
    - Using the recommendation engine

---

## Creating Your First Encoding

```python
from encoding_atlas import AngleEncoding, IQPEncoding

# A simple product-state encoding
angle = AngleEncoding(n_features=4, rotation='Y')

# An entangling encoding with provable hardness
iqp = IQPEncoding(n_features=4, reps=2, entanglement='full')
```

Every encoding takes `n_features` as its first argument — the number of classical features to encode. Additional parameters control the circuit structure.

---

## Inspecting Properties

```python
props = iqp.properties

print(f"Qubits:      {props.n_qubits}")
print(f"Depth:       {props.depth}")
print(f"Gate count:  {props.gate_count}")
print(f"Entangling:  {props.is_entangling}")
print(f"Simulable:   {props.simulability}")
```

The `properties` attribute returns an `EncodingProperties` dataclass computed lazily on first access and cached for subsequent calls. See [Encoding Properties](../concepts/encoding-properties.md) for what each metric means.

---

## Generating Circuits

```python
import numpy as np

x = np.array([0.1, 0.2, 0.3, 0.4])

# PennyLane circuit
pl_circuit = iqp.get_circuit(x, backend='pennylane')

# Qiskit circuit (requires: pip install encoding-atlas[qiskit])
qk_circuit = iqp.get_circuit(x, backend='qiskit')

# Cirq circuit (requires: pip install encoding-atlas[cirq])
cirq_circuit = iqp.get_circuit(x, backend='cirq')
```

---

## Comparing Two Encodings

```python
from encoding_atlas import AngleEncoding, IQPEncoding

encodings = {
    'Angle': AngleEncoding(n_features=4),
    'IQP':   IQPEncoding(n_features=4, reps=2),
}

for name, enc in encodings.items():
    p = enc.properties
    print(f"{name:10s}  qubits={p.n_qubits}  depth={p.depth}  "
          f"entangling={p.is_entangling}")
```

---

## Getting Recommendations

```python
from encoding_atlas.guide import recommend_encoding

rec = recommend_encoding(
    n_features=4,
    n_samples=500,
    task='classification',
    hardware='simulator',
    priority='accuracy'
)

print(f"Recommended: {rec.encoding_name}")
print(f"Reason:      {rec.explanation}")
```

---

## Next Steps

- [Comparing Encodings](comparing-encodings.md) — run deeper analysis with expressibility and entanglement metrics
- [Encodings Reference](../encodings/index.md) — detailed documentation for all 16 encodings
- [API Reference](../api/index.md) — full programmatic interface
