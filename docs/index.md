# Quantum Encoding Atlas

Welcome to the Quantum Encoding Atlas documentation.

## Overview

The Quantum Encoding Atlas is a comprehensive library for quantum data encodings
in machine learning. It provides:

- **15+ encoding methods** with a unified API
- **Multi-framework support** for PennyLane, Qiskit, and Cirq
- **Analysis tools** for encoding properties
- **Benchmarking framework** for systematic comparison
- **Decision guide** for encoding selection

## Quick Start

```python
from encoding_atlas import IQPEncoding, AngleEncoding
from encoding_atlas.analysis import count_resources

# Create an encoding
encoding = IQPEncoding(n_features=4, reps=2)

# Check properties
print(f"Qubits: {encoding.n_qubits}")
print(f"Depth: {encoding.depth}")
print(f"Entangling: {encoding.properties.is_entangling}")
```

## Installation

```bash
pip install encoding-atlas
```

## Contents

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Tutorials](tutorials/)
- [Encodings Reference](encodings/)
- [API Reference](api/)
- [Decision Guide](guide/)
