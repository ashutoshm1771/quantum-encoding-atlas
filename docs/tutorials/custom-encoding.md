# Custom Encodings

This tutorial shows how to create your own quantum data encoding by subclassing `BaseEncoding`. This lets you define arbitrary circuit structures while inheriting the full analysis, benchmarking, and multi-backend infrastructure of the Quantum Encoding Atlas.

!!! abstract "What you'll learn"
    - The `BaseEncoding` interface and required methods
    - Implementing a custom encoding step by step
    - Registering your encoding for use with the analysis tools

---

## The BaseEncoding Interface

Every encoding in the atlas inherits from `BaseEncoding` and implements a small set of methods:

```python
from encoding_atlas.core.base import BaseEncoding

class MyEncoding(BaseEncoding):
    """A custom quantum data encoding."""

    def __init__(self, n_features: int, **kwargs):
        super().__init__(n_features=n_features, **kwargs)

    def _build_circuit(self, x, backend='pennylane'):
        """Build the encoding circuit for input x."""
        ...

    def _compute_properties(self):
        """Compute and return an EncodingProperties dataclass."""
        ...
```

| Method | Purpose | Required |
|:-------|:--------|:--------:|
| `__init__` | Set encoding parameters | Yes |
| `_build_circuit` | Define the quantum circuit | Yes |
| `_compute_properties` | Compute encoding properties | Yes |

---

## Example: Custom Rotation Encoding

!!! info "Under Development"
    A complete worked example with a custom rotation encoding, property computation, backend support, and integration with the analysis tools is being prepared.

    **Planned content:**

    - Step-by-step implementation of a custom `RXRZEncoding`
    - Computing circuit depth, gate count, and simulability
    - Registering the encoding in the atlas registry
    - Running expressibility and entanglement analysis on the custom encoding
    - Testing with the benchmarking framework

---

## Next Steps

- [Benchmarking](benchmarking.md) — test your custom encoding against the built-in ones
- [Adding Encodings](../contributing/adding-encodings.md) — contribute your encoding to the project
- [API Reference](../api/index.md) — full `BaseEncoding` interface documentation
