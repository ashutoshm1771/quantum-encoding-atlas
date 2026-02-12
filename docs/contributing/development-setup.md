# Development Setup

This guide walks you through setting up a local development environment for the Quantum Encoding Atlas.

---

## Prerequisites

- **Python 3.9 or later** (3.11 recommended)
- **Git**
- A virtual environment tool (`venv`, `conda`, or `virtualenv`)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/ashutoshm1771/quantum-encoding-atlas.git
cd quantum-encoding-atlas
```

---

## Step 2: Create a Virtual Environment

=== "venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate    # Linux / macOS
    .venv\Scripts\activate       # Windows
    ```

=== "conda"

    ```bash
    conda create -n encoding-atlas python=3.11
    conda activate encoding-atlas
    ```

---

## Step 3: Install in Development Mode

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with all development dependencies (pytest, ruff, black, mypy, hypothesis, etc.).

To also install optional backends and documentation tools:

```bash
pip install -e ".[dev,all,docs]"
```

---

## Step 4: Verify the Installation

```bash
# Run the test suite
pytest

# Run linting
ruff check src tests

# Run formatting check
black --check src tests

# Run type checking
mypy src
```

All four should pass without errors on a fresh clone.

---

## Step 5: Build Documentation Locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000` in your browser to preview the documentation site with live reloading.

---

## Project Layout

```
quantum-encoding-atlas/
├── src/
│   └── encoding_atlas/         # Main package
│       ├── core/               # BaseEncoding, properties, exceptions
│       ├── encodings/          # All 16 encoding implementations
│       ├── analysis/           # Expressibility, entanglement, etc.
│       ├── guide/              # Recommendation engine
│       ├── benchmark/          # Benchmarking framework
│       └── visualization/      # Plotting utilities
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests by module
│   └── ...
├── docs/                       # Documentation source (Markdown)
├── experiments/                # Experiment framework (not part of package)
├── pyproject.toml              # Build config, dependencies, tool settings
└── mkdocs.yml                  # Documentation site config
```

---

## Running Specific Tests

```bash
# Run a specific test file
pytest tests/unit/encodings/test_iqp_encoding.py

# Run tests matching a keyword
pytest -k "iqp"

# Skip slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=encoding_atlas --cov-report=term-missing
```

---

## Next Steps

- [Code Style](code-style.md) — formatting, linting, and type annotation conventions
- [Adding Encodings](adding-encodings.md) — how to implement a new encoding
