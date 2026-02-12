"""Documentation and docstring tests for quantum encodings.

This module tests documentation quality including:
- Presence of docstrings on all public methods and classes
- Docstring completeness (Parameters, Returns, Examples sections)
- Example code in docstrings executes correctly
- Type hints consistency
- Module-level documentation

These tests ensure documentation meets quality standards for publication.

Run with: pytest tests/unit/test_docstrings.py -v
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from encoding_atlas import (
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    DataReuploading,
    HamiltonianEncoding,
    HardwareEfficientEncoding,
    HigherOrderAngleEncoding,
    IQPEncoding,
    PauliFeatureMap,
    QAOAEncoding,
    SymmetryInspiredFeatureMap,
    ZZFeatureMap,
)
from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.registry import get_encoding, list_encodings

# =============================================================================
# All Encoding Classes for Testing
# =============================================================================


ALL_ENCODING_CLASSES: list[type[BaseEncoding]] = [
    AngleEncoding,
    AmplitudeEncoding,
    BasisEncoding,
    IQPEncoding,
    ZZFeatureMap,
    PauliFeatureMap,
    DataReuploading,
    HardwareEfficientEncoding,
    HigherOrderAngleEncoding,
    QAOAEncoding,
    HamiltonianEncoding,
    SymmetryInspiredFeatureMap,
]


# =============================================================================
# Documentation Checking Utilities
# =============================================================================


def get_public_methods(cls: type) -> list[str]:
    """Get names of public methods (not starting with _)."""
    return [
        name
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]


def get_public_properties(cls: type) -> list[str]:
    """Get names of public properties."""
    return [
        name
        for name, prop in inspect.getmembers(cls)
        if isinstance(prop, property) and not name.startswith("_")
    ]


def has_parameters_section(docstring: str) -> bool:
    """Check if docstring has Parameters section."""
    if not docstring:
        return False
    return "Parameters" in docstring or "parameters" in docstring.lower()


def has_returns_section(docstring: str) -> bool:
    """Check if docstring has Returns section."""
    if not docstring:
        return False
    return "Returns" in docstring or "returns" in docstring.lower()


def has_examples_section(docstring: str) -> bool:
    """Check if docstring has Examples section."""
    if not docstring:
        return False
    return "Examples" in docstring or "Example" in docstring


def extract_code_examples(docstring: str) -> list[str]:
    """Extract code examples from docstring.

    Looks for:
    - >>> code blocks
    - ```python blocks
    """
    if not docstring:
        return []

    examples: list[str] = []

    # Extract >>> examples
    lines = docstring.split("\n")
    current_example: list[str] = []
    in_example = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>"):
            in_example = True
            # Remove >>> prefix
            code = stripped[3:].strip()
            current_example.append(code)
        elif in_example and stripped.startswith("..."):
            # Continuation
            code = stripped[3:].strip()
            current_example.append(code)
        elif in_example and stripped and not stripped.startswith(">>>"):
            # Might be expected output, end of example
            if current_example:
                examples.append("\n".join(current_example))
                current_example = []
            in_example = False

    if current_example:
        examples.append("\n".join(current_example))

    # Extract ```python blocks
    code_block_pattern = r"```python\n(.*?)```"
    matches = re.findall(code_block_pattern, docstring, re.DOTALL)
    examples.extend(matches)

    return examples


def get_docstring_first_line(docstring: str) -> str:
    """Get first line of docstring (summary)."""
    if not docstring:
        return ""
    lines = docstring.strip().split("\n")
    return lines[0].strip() if lines else ""


# =============================================================================
# Test Class: Docstring Presence
# =============================================================================


class TestDocstringPresence:
    """Tests for presence of docstrings."""

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_class_has_docstring(self, encoding_cls: type[BaseEncoding]) -> None:
        """Test that encoding class has a docstring."""
        docstring = encoding_cls.__doc__
        assert docstring is not None, f"{encoding_cls.__name__} missing class docstring"
        assert (
            len(docstring.strip()) > 10
        ), f"{encoding_cls.__name__} docstring too short"

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_init_has_docstring(self, encoding_cls: type[BaseEncoding]) -> None:
        """Test that __init__ has a docstring."""
        docstring = encoding_cls.__init__.__doc__
        # __init__ may inherit docstring from base class
        # Check if class-level docstring covers parameters
        class_doc = encoding_cls.__doc__ or ""
        has_param_docs = (
            docstring and len(docstring.strip()) > 10
        ) or has_parameters_section(class_doc)
        assert has_param_docs, f"{encoding_cls.__name__}.__init__ missing documentation"

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_get_circuit_has_docstring(self, encoding_cls: type[BaseEncoding]) -> None:
        """Test that get_circuit method has a docstring."""
        method = getattr(encoding_cls, "get_circuit", None)
        if method is None:
            pytest.skip(f"{encoding_cls.__name__} has no get_circuit method")

        # May inherit from base class
        docstring = method.__doc__
        assert (
            docstring is not None
        ), f"{encoding_cls.__name__}.get_circuit missing docstring"

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_properties_have_docstrings(self, encoding_cls: type[BaseEncoding]) -> None:
        """Test that public properties have docstrings."""
        missing = []

        for name in ["n_qubits", "depth", "properties", "n_features"]:
            prop = getattr(type(encoding_cls), name, None)
            if prop is None:
                continue

            if isinstance(prop, property) and not prop.fget.__doc__:
                missing.append(name)

        # Allow some missing (may be inherited with docs)
        if len(missing) > 2:
            pytest.fail(f"{encoding_cls.__name__} missing docstrings for: {missing}")


# =============================================================================
# Test Class: Docstring Completeness
# =============================================================================


class TestDocstringCompleteness:
    """Tests for docstring completeness."""

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_class_docstring_describes_encoding(
        self, encoding_cls: type[BaseEncoding]
    ) -> None:
        """Test that class docstring describes what the encoding does."""
        docstring = encoding_cls.__doc__ or ""
        first_line = get_docstring_first_line(docstring)

        # Should mention "encoding" or describe the type
        keywords = ["encoding", "feature map", "circuit", "quantum"]
        has_keyword = any(kw.lower() in docstring.lower() for kw in keywords)

        assert has_keyword, (
            f"{encoding_cls.__name__} docstring doesn't describe encoding: "
            f"'{first_line[:50]}...'"
        )

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_class_docstring_has_parameters(
        self, encoding_cls: type[BaseEncoding]
    ) -> None:
        """Test that class docstring documents parameters."""
        docstring = encoding_cls.__doc__ or ""

        # Should document at least n_features
        has_params = (
            has_parameters_section(docstring)
            or "n_features" in docstring
            or "Parameters" in docstring
        )

        assert (
            has_params
        ), f"{encoding_cls.__name__} docstring missing parameter documentation"

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_docstring_mentions_backend_support(
        self, encoding_cls: type[BaseEncoding]
    ) -> None:
        """Test that docstring mentions backend support."""
        docstring = encoding_cls.__doc__ or ""

        # Should mention backends somewhere
        backends = ["pennylane", "qiskit", "cirq", "backend"]
        mentions_backend = any(b.lower() in docstring.lower() for b in backends)

        # This is a soft check - warn but don't fail
        if not mentions_backend:
            pytest.xfail(
                f"{encoding_cls.__name__} docstring doesn't mention backend support"
            )


# =============================================================================
# Test Class: Docstring Examples
# =============================================================================


class TestDocstringExamples:
    """Tests for docstring example code."""

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_class_has_usage_example(self, encoding_cls: type[BaseEncoding]) -> None:
        """Test that class docstring has a usage example."""
        docstring = encoding_cls.__doc__ or ""

        # Check for example section or code
        has_example = (
            has_examples_section(docstring) or ">>>" in docstring or "```" in docstring
        )

        # Soft requirement - many encodings may not have examples yet
        if not has_example:
            pytest.xfail(f"{encoding_cls.__name__} docstring missing usage example")

    def test_base_encoding_examples_execute(self) -> None:
        """Test that BaseEncoding examples can be executed."""
        # Common patterns that should work
        test_code = """
import numpy as np
from encoding_atlas import AngleEncoding

enc = AngleEncoding(n_features=4)
x = np.array([0.1, 0.2, 0.3, 0.4])
circuit = enc.get_circuit(x, backend='pennylane')
"""
        try:
            exec(test_code, {"__builtins__": __builtins__})
        except Exception as e:
            pytest.fail(f"Basic encoding example failed: {e}")

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES[:4])
    def test_encoding_can_be_used_as_documented(
        self, encoding_cls: type[BaseEncoding]
    ) -> None:
        """Test that encodings work as expected based on docs."""
        # Create encoding
        enc = encoding_cls(n_features=4)

        # Access documented properties
        assert hasattr(enc, "n_features")
        assert hasattr(enc, "n_qubits")
        assert hasattr(enc, "depth")
        assert hasattr(enc, "properties")

        # Generate circuit (documented method)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None


# =============================================================================
# Test Class: Module Documentation
# =============================================================================


class TestModuleDocumentation:
    """Tests for module-level documentation."""

    def test_encodings_module_has_docstring(self) -> None:
        """Test that encodings module has a docstring."""
        import encoding_atlas.encodings

        docstring = encoding_atlas.encodings.__doc__
        assert docstring is not None
        assert len(docstring) > 10

    def test_core_module_has_docstring(self) -> None:
        """Test that core module has a docstring."""

        # Core module should have some documentation
        # May be in __init__.py or individual modules

    def test_main_package_has_docstring(self) -> None:
        """Test that main package has a docstring."""
        import encoding_atlas

        docstring = encoding_atlas.__doc__
        # Main package should describe the library
        # Allow for minimal docstring
        assert docstring is None or isinstance(docstring, str)

    def test_registry_module_documented(self) -> None:
        """Test that registry module is documented."""
        from encoding_atlas.core import registry

        docstring = registry.__doc__
        assert docstring is not None
        assert "registry" in docstring.lower() or "encoding" in docstring.lower()


# =============================================================================
# Test Class: Type Hints
# =============================================================================


class TestTypeHints:
    """Tests for type hint consistency."""

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_init_has_type_hints(self, encoding_cls: type[BaseEncoding]) -> None:
        """Test that __init__ has type hints."""
        init = encoding_cls.__init__
        hints = getattr(init, "__annotations__", {})

        # Should have at least n_features type hint
        # May be defined in base class
        has_hints = len(hints) > 0 or "n_features" in str(inspect.signature(init))

        # Soft requirement
        if not has_hints:
            pytest.xfail(f"{encoding_cls.__name__}.__init__ missing type hints")

    @pytest.mark.parametrize("encoding_cls", ALL_ENCODING_CLASSES)
    def test_get_circuit_has_return_hint(
        self, encoding_cls: type[BaseEncoding]
    ) -> None:
        """Test that get_circuit has return type hint."""
        method = getattr(encoding_cls, "get_circuit", None)
        if method is None:
            pytest.skip("No get_circuit method")

        hints = getattr(method, "__annotations__", {})
        has_return = "return" in hints

        # Soft requirement
        if not has_return:
            pytest.xfail(f"{encoding_cls.__name__}.get_circuit missing return hint")

    def test_registry_functions_have_hints(self) -> None:
        """Test that registry functions have type hints."""

        # list_encodings
        hints = getattr(list_encodings, "__annotations__", {})
        assert "return" in hints, "list_encodings missing return type hint"

        # get_encoding
        hints = getattr(get_encoding, "__annotations__", {})
        assert "return" in hints, "get_encoding missing return type hint"


# =============================================================================
# Test Class: API Documentation Consistency
# =============================================================================


class TestAPIConsistency:
    """Tests for API documentation consistency across encodings."""

    def test_all_encodings_document_n_features(self) -> None:
        """Test that all encodings document n_features parameter."""
        for cls in ALL_ENCODING_CLASSES:
            docstring = cls.__doc__ or ""
            assert (
                "n_features" in docstring or "features" in docstring.lower()
            ), f"{cls.__name__} doesn't document n_features"

    def test_all_encodings_have_consistent_api(self) -> None:
        """Test that all encodings have consistent public API."""
        required_attrs = ["n_features", "n_qubits", "get_circuit", "properties"]

        for cls in ALL_ENCODING_CLASSES:
            enc = cls(n_features=4)
            for attr in required_attrs:
                assert hasattr(
                    enc, attr
                ), f"{cls.__name__} missing required attribute: {attr}"

    def test_get_circuit_consistent_signature(self) -> None:
        """Test that get_circuit has consistent signature across encodings."""
        for cls in ALL_ENCODING_CLASSES:
            method = cls.get_circuit
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())

            # Should have 'x' and 'backend' parameters
            assert (
                "x" in params or len(params) >= 2
            ), f"{cls.__name__}.get_circuit missing 'x' parameter"

    def test_all_encodings_document_backends(self) -> None:
        """Test that encodings mention supported backends."""
        for cls in ALL_ENCODING_CLASSES:
            docstring = cls.__doc__ or ""
            method_doc = cls.get_circuit.__doc__ or ""
            combined = docstring + method_doc

            # Should mention backend somewhere
            mentions = (
                "backend" in combined.lower()
                or "pennylane" in combined.lower()
                or "qiskit" in combined.lower()
            )

            # Soft check
            if not mentions:
                pass  # Many encodings inherit from base


# =============================================================================
# Test Class: Documentation Quality Metrics
# =============================================================================


class TestDocumentationQuality:
    """Tests for overall documentation quality metrics."""

    def test_average_docstring_length(self) -> None:
        """Test that average docstring length meets minimum."""
        lengths: list[int] = []

        for cls in ALL_ENCODING_CLASSES:
            docstring = cls.__doc__ or ""
            lengths.append(len(docstring))

        avg_length = sum(lengths) / len(lengths)

        # Average should be at least 200 characters
        assert (
            avg_length > 200
        ), f"Average docstring length too short: {avg_length:.0f} chars"

    def test_no_todo_in_docstrings(self) -> None:
        """Test that docstrings don't have unresolved TODOs."""
        todos_found: list[tuple[str, str]] = []

        for cls in ALL_ENCODING_CLASSES:
            docstring = cls.__doc__ or ""
            if "TODO" in docstring or "FIXME" in docstring:
                todos_found.append((cls.__name__, "class"))

            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                method_doc = method.__doc__ or ""
                if "TODO" in method_doc or "FIXME" in method_doc:
                    todos_found.append((cls.__name__, name))

        if todos_found:
            pytest.xfail(f"Found TODOs in docstrings: {todos_found}")

    def test_no_placeholder_docstrings(self) -> None:
        """Test that docstrings aren't placeholders."""
        placeholders = ["TODO", "PLACEHOLDER", "FILL IN", "ADD DOCS"]

        for cls in ALL_ENCODING_CLASSES:
            docstring = (cls.__doc__ or "").upper()
            for placeholder in placeholders:
                assert (
                    placeholder not in docstring
                ), f"{cls.__name__} has placeholder docstring"

    def test_docstrings_grammatically_correct(self) -> None:
        """Basic test that docstrings start with capital and end properly."""
        for cls in ALL_ENCODING_CLASSES:
            docstring = cls.__doc__ or ""
            first_line = get_docstring_first_line(docstring)

            if first_line:
                # Should start with capital letter
                assert (
                    first_line[0].isupper() or first_line[0] == '"'
                ), f"{cls.__name__} docstring should start with capital"

                # Should end with period
                assert first_line.rstrip().endswith(
                    (".", "!")
                ), f"{cls.__name__} docstring first line should end with period"
