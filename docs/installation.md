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

# With all backends
pip install encoding-atlas[all]
```

## Development Installation

```bash
git clone https://github.com/ashutoshm1771/quantum-encoding-atlas.git
cd quantum-encoding-atlas
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- PennyLane >= 0.33

### Optional Dependencies

- Qiskit >= 1.0 (for Qiskit backend)
- Cirq >= 1.0 (for Cirq backend)
