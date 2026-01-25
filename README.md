<div align="center">

# Quantum Encoding Atlas

**The comprehensive library for quantum data encodings in machine learning**

[![PyPI version](https://badge.fury.io/py/encoding-atlas.svg)](https://badge.fury.io/py/encoding-atlas)
[![Python versions](https://img.shields.io/pypi/pyversions/encoding-atlas.svg)](https://pypi.org/project/encoding-atlas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ashutoshm1771/quantum-encoding-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/ashutoshm1771/quantum-encoding-atlas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ashutoshm1771/quantum-encoding-atlas/branch/main/graph/badge.svg)](https://codecov.io/gh/ashutoshm1771/quantum-encoding-atlas)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://q-encoding-atlas.web.app/documentation)

[Documentation](https://q-encoding-atlas.web.app/documentation) |
[Tutorials](https://q-encoding-atlas.web.app/documentation) |
[API Reference](https://q-encoding-atlas.web.app/documentation)

</div>

---

## Overview

The **Quantum Encoding Atlas** is the definitive open-source resource for understanding, comparing, and selecting quantum data encodings for machine learning applications.

## Features

- 📊 **15+ Encoding Methods** — Comprehensive implementations of all major quantum data encodings
- 🔀 **Multi-Framework Support** — Works seamlessly with PennyLane, Qiskit, and Cirq
- 📈 **Analysis Tools** — Compute expressibility, entanglement capability, and trainability
- 🧪 **Benchmarking Framework** — Systematic comparison infrastructure
- 🧭 **Decision Guide** — Evidence-based encoding recommendations
- 📚 **Extensive Documentation** — Tutorials, API docs, and theoretical background

## Installation

```bash
pip install encoding-atlas
```

With optional backends:

```bash
# With Qiskit support
pip install encoding-atlas[qiskit]

# With Cirq support
pip install encoding-atlas[cirq]

# With all backends
pip install encoding-atlas[all]

# Development installation
pip install encoding-atlas[dev]
```

## Quick Start

```python
from encoding_atlas import IQPEncoding, AngleEncoding
from encoding_atlas.analysis import compute_expressibility
import numpy as np

# Create encodings
iqp = IQPEncoding(n_features=4, reps=2)
angle = AngleEncoding(n_features=4, rotation='Y')

# Generate circuits (PennyLane by default)
X = np.random.randn(10, 4)
circuit = iqp.get_circuit(X[0])

# Analyze properties
print(f"IQP qubits: {iqp.n_qubits}")
print(f"IQP depth: {iqp.depth}")
print(f"IQP expressibility: {compute_expressibility(iqp, n_samples=500):.4f}")

# Get encoding recommendation
from encoding_atlas.guide import recommend_encoding

rec = recommend_encoding(
    n_features=4,
    n_samples=500,
    task='classification',
    hardware='simulator'
)
print(f"Recommended: {rec.encoding_name}")
print(f"Reason: {rec.explanation}")
```

## Supported Encodings

| Category | Encodings |
|----------|-----------|
| **Amplitude-based** | Amplitude, Approximate Amplitude |
| **Angle-based** | RX, RY, RZ, Multi-axis, Higher-order |
| **Basis** | Binary, One-hot |
| **Entangling** | IQP, ZZ Feature Map, Pauli Feature Map |
| **Advanced** | Data Re-uploading, Hardware-efficient |

See the [full encoding list](https://q-encoding-atlas.web.app/documentation) for details.

## Documentation

- [Installation Guide](https://q-encoding-atlas.web.app/documentation)
- [Quick Start Tutorial](https://q-encoding-atlas.web.app/documentation)
- [Encoding Selection Guide](https://q-encoding-atlas.web.app/documentation)
- [API Reference](https://q-encoding-atlas.web.app/documentation)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{Mishra2026encoding,
  title={Quantum Encoding Atlas: A Comprehensive Library for Quantum Data Encodings},
  author={Mishra, Ashutosh},
  year={2026},
  url={https://github.com/ashutoshm1771/quantum-encoding-atlas},
  version={0.1.0}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
