<div align="center">

# Quantum Encoding Atlas

**The comprehensive library for quantum data encodings in machine learning**

[![PyPI version](https://badge.fury.io/py/encoding-atlas.svg)](https://pypi.org/project/encoding-atlas/)
[![Python versions](https://img.shields.io/pypi/pyversions/encoding-atlas.svg)](https://pypi.org/project/encoding-atlas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/encoding-atlas/quantum-encoding-atlas/ci.yml?branch=master&logo=github&label=CI)](https://github.com/encoding-atlas/quantum-encoding-atlas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/encoding-atlas/quantum-encoding-atlas/branch/master/graph/badge.svg)](https://codecov.io/gh/encoding-atlas/quantum-encoding-atlas)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://encoding-atlas.github.io/quantum-encoding-atlas/)
[![Website](https://img.shields.io/badge/Website-live-brightgreen?logo=data:image/png;base64,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)](https://q-encoding-atlas.web.app)

[Documentation](https://encoding-atlas.github.io/quantum-encoding-atlas/) |
[Website](https://q-encoding-atlas.web.app) |
[Tutorials](https://encoding-atlas.github.io/quantum-encoding-atlas/tutorials/) |
[API Reference](https://encoding-atlas.github.io/quantum-encoding-atlas/api/) |
[PyPI](https://pypi.org/project/encoding-atlas/)

</div>

---

## Overview

The **Quantum Encoding Atlas** is the definitive open-source resource for understanding, comparing, and selecting quantum data encodings for machine learning applications.

## Features

- 📊 **16 Encoding Methods** — Comprehensive implementations of all major quantum data encodings
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
| **Amplitude-based** | AmplitudeEncoding |
| **Angle-based** | AngleEncoding (RX/RY/RZ), HigherOrderAngleEncoding |
| **Basis** | BasisEncoding |
| **Entangling** | IQPEncoding, ZZFeatureMap, PauliFeatureMap |
| **Advanced** | DataReuploading, HardwareEfficientEncoding, QAOAEncoding, HamiltonianEncoding |
| **Symmetry & Equivariant** | SymmetryInspiredFeatureMap, SO2EquivariantFeatureMap, CyclicEquivariantFeatureMap, SwapEquivariantFeatureMap |
| **Trainable** | TrainableEncoding |

See the [full encoding list](https://encoding-atlas.github.io/quantum-encoding-atlas/encodings/) for details.

## Documentation

- [Installation Guide](https://encoding-atlas.github.io/quantum-encoding-atlas/installation/)
- [Quick Start Tutorial](https://encoding-atlas.github.io/quantum-encoding-atlas/quickstart/)
- [Encoding Selection Guide](https://encoding-atlas.github.io/quantum-encoding-atlas/guide/which-encoding/)
- [API Reference](https://encoding-atlas.github.io/quantum-encoding-atlas/api/)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{Mishra2026encoding,
  title={Quantum Encoding Atlas: A Comprehensive Library for Quantum Data Encodings},
  author={Mishra, Ashutosh},
  year={2026},
  url={https://github.com/encoding-atlas/quantum-encoding-atlas},
  version={0.1.0}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
