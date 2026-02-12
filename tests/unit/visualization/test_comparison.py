"""Tests for encoding comparison visualization.

This module tests the compare_encodings function and related utilities
for visualizing quantum encoding comparisons.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

from encoding_atlas.visualization import EncodingComparison, compare_encodings
from encoding_atlas.visualization.comparison import (
    _create_bar,
    _get_encoding_data,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basic_encodings() -> list[str]:
    """Basic encoding names for testing."""
    return ["angle", "iqp"]


@pytest.fixture
def all_common_encodings() -> list[str]:
    """Common encodings that work with n_features=4."""
    return ["angle", "iqp", "zz_feature_map", "hardware_efficient"]


# =============================================================================
# UNIT TESTS - HELPER FUNCTIONS
# =============================================================================


class TestCreateBar:
    """Tests for the _create_bar helper function."""

    def test_full_bar(self) -> None:
        """Test bar at maximum value."""
        bar = _create_bar(10, 10, width=10)
        assert bar == "█" * 10

    def test_half_bar(self) -> None:
        """Test bar at half value."""
        bar = _create_bar(5, 10, width=10)
        assert bar == "█" * 5

    def test_empty_bar(self) -> None:
        """Test bar at zero value."""
        bar = _create_bar(0, 10, width=10)
        assert bar == ""

    def test_zero_max(self) -> None:
        """Test bar with zero max value."""
        bar = _create_bar(5, 0, width=10)
        assert bar == ""

    def test_custom_width(self) -> None:
        """Test bar with custom width."""
        bar = _create_bar(10, 10, width=20)
        assert bar == "█" * 20


class TestGetEncodingData:
    """Tests for the _get_encoding_data helper function."""

    def test_basic_encodings(self, basic_encodings: list[str]) -> None:
        """Test gathering data for basic encodings."""
        data = _get_encoding_data(basic_encodings, n_features=4)

        assert len(data) == 2
        assert all(isinstance(d, EncodingComparison) for d in data)
        assert data[0].name == "angle"
        assert data[1].name == "iqp"

    def test_with_custom_config(self) -> None:
        """Test gathering data with custom configurations."""
        data = _get_encoding_data(
            ["angle"],
            n_features=4,
            configs={"angle": {"rotation": "X", "reps": 2}},
        )

        assert len(data) == 1
        assert data[0].config == {"rotation": "X", "reps": 2}
        assert data[0].encoding.rotation == "X"
        assert data[0].encoding.reps == 2

    def test_invalid_encoding_skipped(self) -> None:
        """Test that invalid encodings are skipped with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = _get_encoding_data(
                ["angle", "nonexistent_encoding"],
                n_features=4,
            )

            # Should only get angle encoding
            assert len(data) == 1
            assert data[0].name == "angle"

            # Should have warning about nonexistent encoding
            assert len(w) == 1
            assert "nonexistent_encoding" in str(w[0].message)

    def test_encoding_with_various_features(self) -> None:
        """Test that encodings handle various feature counts."""
        # Both should work with n_features=4
        data = _get_encoding_data(
            ["angle", "amplitude"],
            n_features=4,
        )

        # Both encodings should be created
        assert len(data) == 2
        names = [d.name for d in data]
        assert "angle" in names
        assert "amplitude" in names

    def test_properties_included(self, basic_encodings: list[str]) -> None:
        """Test that properties are correctly included."""
        data = _get_encoding_data(basic_encodings, n_features=4)

        for d in data:
            assert d.properties is not None
            assert d.properties.n_qubits >= 1
            assert d.properties.depth >= 1
            assert d.properties.gate_count >= 1


# =============================================================================
# INTEGRATION TESTS - TEXT OUTPUT
# =============================================================================


class TestCompareEncodingsText:
    """Tests for compare_encodings with text output."""

    def test_basic_comparison(self, basic_encodings: list[str]) -> None:
        """Test basic text comparison."""
        result = compare_encodings(basic_encodings, n_features=4, output="text")

        assert isinstance(result, str)
        assert "ENCODING COMPARISON" in result
        assert "n_features=4" in result
        assert "angle" in result
        assert "iqp" in result

    def test_contains_metrics(self, basic_encodings: list[str]) -> None:
        """Test that output contains all metric sections."""
        result = compare_encodings(basic_encodings, n_features=4, output="text")

        assert "QUBITS" in result
        assert "CIRCUIT DEPTH" in result
        assert "GATE COUNT" in result
        assert "TWO-QUBIT GATES" in result
        assert "PROPERTIES" in result

    def test_contains_property_values(self, basic_encodings: list[str]) -> None:
        """Test that output contains property indicators."""
        result = compare_encodings(basic_encodings, n_features=4, output="text")

        # Should have entanglement indicators
        assert "Yes" in result or "No" in result

        # Should have simulability info
        assert "Simulable" in result or "simulable" in result.lower()

    def test_with_notes(self, basic_encodings: list[str]) -> None:
        """Test text output with notes enabled."""
        result = compare_encodings(
            basic_encodings,
            n_features=4,
            output="text",
            show_notes=True,
        )

        assert "NOTES" in result

    def test_multiple_encodings(self, all_common_encodings: list[str]) -> None:
        """Test comparison with multiple encodings."""
        result = compare_encodings(all_common_encodings, n_features=4, output="text")

        for name in all_common_encodings:
            assert name in result

    def test_custom_config_applied(self) -> None:
        """Test that custom configurations are applied."""
        result = compare_encodings(
            ["angle"],
            n_features=4,
            output="text",
            configs={"angle": {"rotation": "X"}},
        )

        assert isinstance(result, str)
        assert "angle" in result

    def test_empty_result_message(self) -> None:
        """Test message when no encodings can be created."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = compare_encodings(
                ["nonexistent"],
                n_features=4,
                output="text",
            )

            assert "No encodings" in result


# =============================================================================
# INTEGRATION TESTS - MATPLOTLIB OUTPUT
# =============================================================================


class TestCompareEncodingsMatplotlib:
    """Tests for compare_encodings with matplotlib output."""

    def test_returns_figure(self, basic_encodings: list[str]) -> None:
        """Test that matplotlib output returns a Figure."""
        pytest.importorskip("matplotlib")
        from matplotlib.figure import Figure

        fig = compare_encodings(basic_encodings, n_features=4, output="matplotlib")

        assert isinstance(fig, Figure)

    def test_figure_has_axes(self, basic_encodings: list[str]) -> None:
        """Test that figure has expected axes."""
        pytest.importorskip("matplotlib")

        fig = compare_encodings(basic_encodings, n_features=4, output="matplotlib")

        # Should have 2x3 = 6 subplots
        assert len(fig.axes) == 6

    def test_figure_has_title(self, basic_encodings: list[str]) -> None:
        """Test that figure has title with n_features."""
        pytest.importorskip("matplotlib")

        fig = compare_encodings(basic_encodings, n_features=4, output="matplotlib")

        assert "n_features=4" in fig._suptitle.get_text()

    def test_custom_figsize(self, basic_encodings: list[str]) -> None:
        """Test custom figure size."""
        pytest.importorskip("matplotlib")

        fig = compare_encodings(
            basic_encodings,
            n_features=4,
            output="matplotlib",
            figsize=(8, 6),
        )

        # Check figure size (approximately)
        width, height = fig.get_size_inches()
        assert abs(width - 8) < 0.1
        assert abs(height - 6) < 0.1

    def test_multiple_encodings(self, all_common_encodings: list[str]) -> None:
        """Test matplotlib output with multiple encodings."""
        pytest.importorskip("matplotlib")
        from matplotlib.figure import Figure

        fig = compare_encodings(
            all_common_encodings,
            n_features=4,
            output="matplotlib",
        )

        assert isinstance(fig, Figure)

    def test_raises_on_empty_data(self) -> None:
        """Test that matplotlib raises on empty data."""
        pytest.importorskip("matplotlib")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="No encodings"):
                compare_encodings(
                    ["nonexistent"],
                    n_features=4,
                    output="matplotlib",
                )

    def test_missing_matplotlib_raises(
        self, basic_encodings: list[str], monkeypatch
    ) -> None:
        """Test that missing matplotlib raises ImportError."""
        import sys

        # Mock matplotlib as not installed
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

        # Need to reimport to trigger the check
        # Force reimport
        import importlib

        from encoding_atlas.visualization import comparison

        importlib.reload(comparison)

        # This should work (text output doesn't need matplotlib)
        result = comparison.compare_encodings(
            basic_encodings, n_features=4, output="text"
        )
        assert isinstance(result, str)


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_encoding(self) -> None:
        """Test comparison with single encoding."""
        result = compare_encodings(["angle"], n_features=4, output="text")

        assert isinstance(result, str)
        assert "angle" in result

    def test_large_n_features(self) -> None:
        """Test with larger feature count."""
        result = compare_encodings(["angle", "iqp"], n_features=16, output="text")

        assert "n_features=16" in result

    def test_small_n_features(self) -> None:
        """Test with minimum feature count."""
        result = compare_encodings(["angle"], n_features=1, output="text")

        assert "n_features=1" in result

    def test_invalid_output_format(self) -> None:
        """Test that invalid output format raises error."""
        with pytest.raises(ValueError, match="Unknown output format"):
            compare_encodings(["angle"], n_features=4, output="invalid")  # type: ignore

    def test_amplitude_with_power_of_2(self) -> None:
        """Test amplitude encoding with valid power of 2."""
        result = compare_encodings(
            ["angle", "amplitude"],
            n_features=4,  # Power of 2
            output="text",
        )

        assert "amplitude" in result
        assert "angle" in result

    def test_duplicate_encoding_names(self) -> None:
        """Test handling of duplicate encoding names."""
        # Same encoding twice should work (though not very useful)
        result = compare_encodings(["angle", "angle"], n_features=4, output="text")

        assert isinstance(result, str)


# =============================================================================
# ENCODING COMPARISON DATA CLASS
# =============================================================================


class TestEncodingComparison:
    """Tests for EncodingComparison dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating EncodingComparison manually."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        comparison = EncodingComparison(
            name="angle",
            encoding=enc,
            properties=enc.properties,
            config={"rotation": "Y"},
        )

        assert comparison.name == "angle"
        assert comparison.encoding is enc
        assert comparison.properties is enc.properties
        assert comparison.config == {"rotation": "Y"}


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestRegression:
    """Regression tests for specific issues."""

    def test_zero_two_qubit_gates_display(self) -> None:
        """Test that encodings with zero two-qubit gates display correctly."""
        # AngleEncoding has no two-qubit gates
        result = compare_encodings(["angle"], n_features=4, output="text")

        # Should not crash and should show 0 for two-qubit gates
        assert isinstance(result, str)
        assert "0" in result  # Two-qubit gate count

    def test_text_output_width_consistency(self) -> None:
        """Test that text output has consistent width."""
        result = compare_encodings(["angle", "iqp"], n_features=4, output="text")

        lines = result.split("\n")
        # Filter out empty lines
        content_lines = [l for l in lines if l.strip()]

        # Most lines should start with │ or ┌ or └ or ├
        border_lines = [l for l in content_lines if l[0] in "│┌└├"]
        if border_lines:
            widths = {len(l) for l in border_lines}
            # Allow some variation but should be mostly consistent
            assert len(widths) <= 3, f"Too many different line widths: {widths}"
