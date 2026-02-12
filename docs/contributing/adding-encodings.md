# Adding Encodings

This guide explains how to add a new quantum encoding to the Quantum Encoding Atlas. Every encoding follows the same structure and integrates with the unified API, analysis tools, and benchmarking framework.

---

## Overview

Adding a new encoding involves four steps:

1. **Implement** the encoding class
2. **Write documentation** for the encoding
3. **Add tests** for the encoding
4. **Register** the encoding in the package

---

## Step 1: Implement the Encoding

Create a new file in `src/encoding_atlas/encodings/`:

```python
# src/encoding_atlas/encodings/my_encoding.py

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties


class MyEncoding(BaseEncoding):
    """One-line description of the encoding.

    Longer description explaining the encoding strategy,
    circuit structure, and relevant references.

    Args:
        n_features: Number of classical features to encode.
        reps: Number of circuit repetitions.

    References:
        Author et al., "Paper Title", Journal, Year.
    """

    __slots__ = ('_reps',)  # Required: declare instance attributes

    def __init__(self, n_features: int, reps: int = 1):
        self._reps = reps
        super().__init__(n_features=n_features)

    def _build_circuit(self, x, backend='pennylane'):
        """Build the encoding circuit for input x."""
        ...

    def _compute_properties(self) -> EncodingProperties:
        """Compute encoding properties."""
        ...
```

### Key Requirements

- Inherit from `BaseEncoding`
- Use `__slots__` for all instance attributes
- Implement `_build_circuit` and `_compute_properties`
- Support all three backends: `pennylane`, `qiskit`, `cirq`
- Include Google-style docstrings with references

---

## Step 2: Write Documentation

Create a documentation page in `docs/encodings/<name>/`:

```
docs/encodings/my_encoding/my_encoding.md
```

Follow the structure of existing encoding pages (see [IQP Encoding](../encodings/iqp/iqp_encoding.md) for the gold standard):

1. Title and one-line description
2. Core idea with ASCII diagram
3. Circuit structure
4. Mathematics
5. Key properties table
6. Example walkthrough
7. Resource scaling table
8. Strengths and limitations
9. Use cases
10. References

---

## Step 3: Add Tests

Create a test file in `tests/unit/encodings/`:

```python
# tests/unit/encodings/test_my_encoding.py

import pytest
from encoding_atlas.encodings.my_encoding import MyEncoding


class TestMyEncoding:
    def test_basic_creation(self):
        enc = MyEncoding(n_features=4)
        assert enc.n_qubits == 4

    def test_properties(self):
        enc = MyEncoding(n_features=4)
        props = enc.properties
        assert props.gate_count > 0

    def test_circuit_generation(self):
        ...
```

Ensure tests cover:

- Construction with default and custom parameters
- Property computation
- Circuit generation for each backend
- Edge cases (n_features=1, large n_features)

---

## Step 4: Register the Encoding

1. Export from the encodings module:

    ```python
    # src/encoding_atlas/encodings/__init__.py
    from .my_encoding import MyEncoding
    ```

2. Export from the top-level package:

    ```python
    # src/encoding_atlas/__init__.py
    from .encodings.my_encoding import MyEncoding
    ```

3. Add to the nav in `mkdocs.yml`:

    ```yaml
    - My Encoding: encodings/my_encoding/my_encoding.md
    ```

---

## Checklist

Before submitting a PR for a new encoding:

- [ ] Encoding class inherits from `BaseEncoding` with `__slots__`
- [ ] `_build_circuit` supports pennylane, qiskit, and cirq backends
- [ ] `_compute_properties` returns a valid `EncodingProperties`
- [ ] Google-style docstring with description and references
- [ ] Documentation page with circuit diagrams, math, and examples
- [ ] Test file with >90% coverage of the new code
- [ ] Encoding exported from `__init__.py`
- [ ] Added to `mkdocs.yml` nav
- [ ] All existing tests still pass (`pytest`)
- [ ] Linting passes (`ruff check src tests`)
- [ ] Formatting passes (`black --check src tests`)
- [ ] Type checking passes (`mypy src`)
