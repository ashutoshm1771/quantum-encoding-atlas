# Contributing

The Quantum Encoding Atlas is an open-source project and contributions are welcome. Whether you're fixing a bug, adding a new encoding, improving documentation, or suggesting an idea — your input is valued.

---

## Ways to Contribute

| Contribution | Guide |
|:-------------|:------|
| Report a bug or suggest a feature | [Open an issue](https://github.com/encoding-atlas/quantum-encoding-atlas/issues) |
| Set up a development environment | [Development Setup](development-setup.md) |
| Add a new quantum encoding | [Adding Encodings](adding-encodings.md) |
| Understand the code conventions | [Code Style](code-style.md) |
| Understand the release process | [Release Process](release-process.md) |

---

## Quick Start for Contributors

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/quantum-encoding-atlas.git
cd quantum-encoding-atlas

# 2. Install in development mode
pip install -e ".[dev]"

# 3. Run the tests
pytest

# 4. Make your changes, then run quality checks
ruff check src tests
black src tests
mypy src
pytest --cov=encoding_atlas
```

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. By participating, you agree to uphold a welcoming, inclusive, and respectful environment for everyone.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://github.com/encoding-atlas/quantum-encoding-atlas/blob/master/LICENSE).
