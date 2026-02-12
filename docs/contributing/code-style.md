# Code Style

The Quantum Encoding Atlas follows consistent coding conventions enforced by automated tools. This page documents the style rules and how to apply them.

---

## Tools

| Tool | Purpose | Config Location |
|:-----|:--------|:---------------|
| **Black** | Code formatting | `[tool.black]` in `pyproject.toml` |
| **Ruff** | Linting (replaces flake8, isort, pyflakes) | `[tool.ruff]` in `pyproject.toml` |
| **mypy** | Static type checking | `[tool.mypy]` in `pyproject.toml` |

---

## Formatting with Black

Black is an opinionated code formatter. It makes all formatting decisions for you.

```bash
# Check formatting (CI runs this)
black --check src tests

# Auto-format everything
black src tests
```

**Key settings:**

- Line length: **88 characters**
- Target versions: Python 3.9-3.12

---

## Linting with Ruff

Ruff catches common errors, style issues, and import ordering problems.

```bash
# Check for issues
ruff check src tests

# Auto-fix what can be fixed
ruff check --fix src tests
```

**Enabled rule sets:**

- `E`, `W` — pycodestyle (errors and warnings)
- `F` — pyflakes (undefined names, unused imports)
- `I` — isort (import ordering)
- `B` — flake8-bugbear (common bugs)
- `C4` — flake8-comprehensions (better list/dict comprehensions)
- `UP` — pyupgrade (modern Python syntax)
- `ARG` — flake8-unused-arguments
- `SIM` — flake8-simplify

---

## Type Annotations with mypy

All public functions and methods must have type annotations. mypy runs in strict mode.

```bash
mypy src
```

**Key settings:**

- `disallow_untyped_defs = true` — every function needs annotations
- `warn_return_any = true` — flag functions returning `Any`
- `strict_equality = true` — flag comparisons between incompatible types

Third-party libraries without type stubs (PennyLane, Qiskit, Cirq, SciPy) are configured with `ignore_missing_imports`.

---

## Docstrings

Use **Google-style** docstrings. This is required for mkdocstrings to generate API documentation correctly.

```python
def compute_expressibility(
    encoding: BaseEncoding,
    n_samples: int = 1000,
) -> float:
    """Compute the expressibility of an encoding.

    Estimates the KL divergence between the encoding's fidelity
    distribution and the Haar-random distribution.

    Args:
        encoding: The encoding to analyse.
        n_samples: Number of random input pairs to sample.

    Returns:
        The KL divergence (lower = more expressive).

    Raises:
        AnalysisError: If the encoding circuit cannot be simulated.
    """
```

---

## Import Ordering

Imports are sorted by Ruff (isort rules) into three groups:

```python
# 1. Standard library
import math
from typing import Optional

# 2. Third-party
import numpy as np
import pennylane as qml

# 3. First-party (this project)
from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties
```

---

## Naming Conventions

| Entity | Convention | Example |
|:-------|:----------|:--------|
| Classes | PascalCase | `IQPEncoding` |
| Functions / methods | snake_case | `compute_expressibility` |
| Constants | UPPER_SNAKE_CASE | `VALID_STAGES` |
| Private attributes | `_prefix` | `self._config` |
| Type variables | PascalCase | `EncodingT` |

---

## Running All Checks

Before submitting a PR, run all quality checks:

```bash
ruff check src tests && black --check src tests && mypy src && pytest
```

Or use the one-liner from the project root. All four must pass for CI to be green.
