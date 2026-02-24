# Installation

## Basic Installation

```bash
pip install encoding-atlas
```

## With Optional Backends

```bash
# With Qiskit support
pip install encoding-atlas[qiskit]

# With Cirq support
pip install encoding-atlas[cirq]

# With visualization support
pip install encoding-atlas[visualization]

# With all optional dependencies
pip install encoding-atlas[all]
```

## Development Installation

```bash
git clone https://github.com/encoding-atlas/quantum-encoding-atlas.git
cd quantum-encoding-atlas
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- PennyLane >= 0.33
- scikit-learn >= 1.0

### Optional Dependencies

- Qiskit >= 1.0 (for Qiskit backend)
- Cirq >= 1.0 (for Cirq backend)
- Matplotlib >= 3.5 (for visualization)
