# Contributing to Quantum Encoding Atlas

Thank you for your interest in contributing to Quantum Encoding Atlas! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Adding New Encodings](#adding-new-encodings)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/quantum-encoding-atlas.git
   cd quantum-encoding-atlas
   ```

3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/encoding-atlas/quantum-encoding-atlas.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev,docs,all]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Verify your setup:
   ```bash
   pytest
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our [code style guidelines](#code-style)

3. Add tests for your changes

4. Run the test suite:
   ```bash
   pytest
   ```

5. Run linting:
   ```bash
   ruff check src tests
   black --check src tests
   mypy src
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "Add meaningful commit message"
   ```

## Adding New Encodings

To add a new encoding:

1. Create a new file in `src/encoding_atlas/encodings/` (or add to existing file if it's a variant)

2. Implement the encoding class inheriting from `BaseEncoding`:
   ```python
   from encoding_atlas.core.base import BaseEncoding
   from encoding_atlas.core.properties import EncodingProperties

   class MyNewEncoding(BaseEncoding):
       """Description of the encoding.

       Parameters
       ----------
       n_features : int
           Number of features to encode.
       my_param : type
           Description of parameter.

       Examples
       --------
       >>> encoding = MyNewEncoding(n_features=4)
       >>> circuit = encoding.get_circuit(data)
       """

       def __init__(self, n_features: int, my_param: type = default) -> None:
           super().__init__(n_features, my_param=my_param)
           self.my_param = my_param

       @property
       def n_qubits(self) -> int:
           # Return number of qubits needed
           ...

       @property
       def depth(self) -> int:
           # Return circuit depth
           ...

       def get_circuit(self, x, backend="pennylane"):
           # Generate circuit
           ...

       def get_circuits(self, X, backend="pennylane"):
           # Generate circuits for batch
           ...

       def _compute_properties(self) -> EncodingProperties:
           # Return encoding properties
           ...
   ```

3. Register the encoding in `src/encoding_atlas/encodings/_registry.py`

4. Export it in `src/encoding_atlas/encodings/__init__.py`

5. Add comprehensive tests in `tests/unit/encodings/test_my_new_encoding.py`

6. Add documentation in `docs/encodings/my-new-encoding.md`

## Testing

Run the full test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=encoding_atlas --cov-report=html
```

Run specific tests:
```bash
pytest tests/unit/encodings/test_angle.py -v
```

Run tests across Python versions:
```bash
nox -s tests
```

## Documentation

Build documentation locally:
```bash
mkdocs serve
```

Then visit `http://localhost:8000` to preview.

Documentation style:
- Use Google-style docstrings
- Include examples in docstrings
- Add mathematical notation where appropriate
- Reference original papers

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Create a pull request with a clear description

### PR Checklist

- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Code follows style guidelines
- [ ] Commits are clean and well-described

## Code Style

We use:
- **Black** for code formatting (line length 88)
- **Ruff** for linting
- **mypy** for type checking
- **Google-style** docstrings

Type hints are required for all public functions:
```python
def compute_expressibility(
    encoding: BaseEncoding,
    n_samples: int = 1000,
    seed: int | None = None,
) -> float:
    """Compute expressibility of an encoding.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    n_samples : int, default=1000
        Number of samples.
    seed : int or None, default=None
        Random seed.

    Returns
    -------
    float
        Expressibility value.
    """
    ...
```

## Questions?

Feel free to open an issue for any questions about contributing!
