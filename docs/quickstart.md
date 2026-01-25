# Quick Start

This guide will get you started with the Quantum Encoding Atlas in 5 minutes.

## Creating Encodings

```python
from encoding_atlas import AngleEncoding, IQPEncoding

# Simple angle encoding
angle = AngleEncoding(n_features=4, rotation='Y')

# IQP encoding with entanglement
iqp = IQPEncoding(n_features=4, reps=2, entanglement='full')
```

## Checking Properties

```python
# Get encoding properties
props = iqp.properties

print(f"Qubits: {props.n_qubits}")
print(f"Depth: {props.depth}")
print(f"Gate count: {props.gate_count}")
print(f"Entangling: {props.is_entangling}")
print(f"Simulability: {props.simulability}")
```

## Generating Circuits

```python
import numpy as np

# Sample data
x = np.array([0.1, 0.2, 0.3, 0.4])

# Generate circuit for PennyLane
circuit = iqp.get_circuit(x, backend='pennylane')

# Generate circuit for Qiskit
qiskit_circuit = iqp.get_circuit(x, backend='qiskit')
```

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
print(f"Reason: {rec.explanation}")
```
