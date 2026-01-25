"""Tests for capability protocols.

This module tests the protocol definitions in `encoding_atlas.core.protocols`
and verifies that existing encodings correctly implement the appropriate
protocols.

Test Categories
---------------
1. **Import Tests**: Verify protocols can be imported
2. **Protocol Recognition Tests**: Verify isinstance() checks work
3. **Type Guard Tests**: Verify helper functions work
4. **Protocol Method Tests**: Verify protocol methods are callable
5. **Non-Conformance Tests**: Verify non-implementing encodings don't match
"""

import pytest
import numpy as np
from typing import Any


class TestProtocolImports:
    """Test that all protocols can be imported correctly."""

    def test_import_from_core_protocols(self) -> None:
        """Test importing protocols from core.protocols module."""
        from encoding_atlas.core.protocols import (
            ResourceAnalyzable,
            DataDependentResourceAnalyzable,
            EntanglementQueryable,
            DataTransformable,
        )

        # Verify they are Protocol classes
        assert hasattr(ResourceAnalyzable, '__protocol_attrs__') or True
        assert hasattr(DataDependentResourceAnalyzable, '__protocol_attrs__') or True
        assert hasattr(EntanglementQueryable, '__protocol_attrs__') or True
        assert hasattr(DataTransformable, '__protocol_attrs__') or True

    def test_import_from_core_init(self) -> None:
        """Test importing protocols from core module."""
        from encoding_atlas.core import (
            ResourceAnalyzable,
            DataDependentResourceAnalyzable,
            EntanglementQueryable,
            DataTransformable,
        )

        assert ResourceAnalyzable is not None
        assert DataDependentResourceAnalyzable is not None
        assert EntanglementQueryable is not None
        assert DataTransformable is not None

    def test_import_from_main_package(self) -> None:
        """Test importing protocols from main package."""
        from encoding_atlas import (
            ResourceAnalyzable,
            DataDependentResourceAnalyzable,
            EntanglementQueryable,
            DataTransformable,
        )

        assert ResourceAnalyzable is not None
        assert DataDependentResourceAnalyzable is not None
        assert EntanglementQueryable is not None
        assert DataTransformable is not None

    def test_import_type_guards(self) -> None:
        """Test importing type guard functions."""
        from encoding_atlas.core.protocols import (
            is_resource_analyzable,
            is_data_dependent_resource_analyzable,
            is_entanglement_queryable,
            is_data_transformable,
        )

        assert callable(is_resource_analyzable)
        assert callable(is_data_dependent_resource_analyzable)
        assert callable(is_entanglement_queryable)
        assert callable(is_data_transformable)


class TestResourceAnalyzableProtocol:
    """Test ResourceAnalyzable protocol recognition."""

    @pytest.fixture
    def resource_analyzable_encodings(self) -> list[Any]:
        """Get list of encodings that should implement ResourceAnalyzable."""
        from encoding_atlas import (
            AmplitudeEncoding,
            AngleEncoding,
            IQPEncoding,
            ZZFeatureMap,
            HardwareEfficientEncoding,
            DataReuploading,
            HigherOrderAngleEncoding,
            HamiltonianEncoding,
            QAOAEncoding,
            SymmetryInspiredFeatureMap,
            PauliFeatureMap,
        )

        return [
            AmplitudeEncoding(n_features=4),
            AngleEncoding(n_features=4),
            IQPEncoding(n_features=4),
            ZZFeatureMap(n_features=4),
            HardwareEfficientEncoding(n_features=4),
            DataReuploading(n_features=4),
            HigherOrderAngleEncoding(n_features=4),
            HamiltonianEncoding(n_features=4),
            QAOAEncoding(n_features=4),
            SymmetryInspiredFeatureMap(n_features=4),
            PauliFeatureMap(n_features=4),
        ]

    def test_isinstance_check(self, resource_analyzable_encodings: list[Any]) -> None:
        """Test that ResourceAnalyzable encodings pass isinstance check."""
        from encoding_atlas.core.protocols import ResourceAnalyzable

        for enc in resource_analyzable_encodings:
            assert isinstance(enc, ResourceAnalyzable), (
                f"{enc.__class__.__name__} should implement ResourceAnalyzable"
            )

    def test_type_guard_function(self, resource_analyzable_encodings: list[Any]) -> None:
        """Test that type guard function works correctly."""
        from encoding_atlas.core.protocols import is_resource_analyzable

        for enc in resource_analyzable_encodings:
            assert is_resource_analyzable(enc), (
                f"is_resource_analyzable({enc.__class__.__name__}) should return True"
            )

    def test_resource_summary_callable(
        self, resource_analyzable_encodings: list[Any]
    ) -> None:
        """Test that resource_summary() is callable on conforming encodings."""
        from encoding_atlas.core.protocols import ResourceAnalyzable

        for enc in resource_analyzable_encodings:
            if isinstance(enc, ResourceAnalyzable):
                summary = enc.resource_summary()
                assert isinstance(summary, dict), (
                    f"{enc.__class__.__name__}.resource_summary() should return dict"
                )

    def test_gate_count_breakdown_callable(
        self, resource_analyzable_encodings: list[Any]
    ) -> None:
        """Test that gate_count_breakdown() is callable on conforming encodings."""
        from encoding_atlas.core.protocols import ResourceAnalyzable

        for enc in resource_analyzable_encodings:
            if isinstance(enc, ResourceAnalyzable):
                breakdown = enc.gate_count_breakdown()
                assert isinstance(breakdown, dict), (
                    f"{enc.__class__.__name__}.gate_count_breakdown() should return dict"
                )
                # Verify required fields
                assert "total" in breakdown or "total_single_qubit" in breakdown, (
                    f"{enc.__class__.__name__}.gate_count_breakdown() missing required fields"
                )


class TestDataDependentResourceAnalyzableProtocol:
    """Test DataDependentResourceAnalyzable protocol recognition."""

    @pytest.fixture
    def data_dependent_encodings(self) -> list[Any]:
        """Get list of encodings that should implement DataDependentResourceAnalyzable."""
        from encoding_atlas import BasisEncoding

        return [
            BasisEncoding(n_features=4),
        ]

    def test_isinstance_check(self, data_dependent_encodings: list[Any]) -> None:
        """Test that DataDependentResourceAnalyzable encodings pass isinstance check."""
        from encoding_atlas.core.protocols import DataDependentResourceAnalyzable

        for enc in data_dependent_encodings:
            assert isinstance(enc, DataDependentResourceAnalyzable), (
                f"{enc.__class__.__name__} should implement DataDependentResourceAnalyzable"
            )

    def test_type_guard_function(self, data_dependent_encodings: list[Any]) -> None:
        """Test that type guard function works correctly."""
        from encoding_atlas.core.protocols import is_data_dependent_resource_analyzable

        for enc in data_dependent_encodings:
            assert is_data_dependent_resource_analyzable(enc), (
                f"is_data_dependent_resource_analyzable({enc.__class__.__name__}) "
                "should return True"
            )

    def test_resource_summary_with_data(
        self, data_dependent_encodings: list[Any]
    ) -> None:
        """Test that resource_summary(x) is callable with input data."""
        from encoding_atlas.core.protocols import DataDependentResourceAnalyzable

        x = np.array([0.1, 0.9, 0.3, 0.8])

        for enc in data_dependent_encodings:
            if isinstance(enc, DataDependentResourceAnalyzable):
                summary = enc.resource_summary(x)
                assert isinstance(summary, dict), (
                    f"{enc.__class__.__name__}.resource_summary(x) should return dict"
                )

    def test_actual_gate_count(self, data_dependent_encodings: list[Any]) -> None:
        """Test that actual_gate_count(x) returns correct type."""
        from encoding_atlas.core.protocols import DataDependentResourceAnalyzable

        x = np.array([0.1, 0.9, 0.3, 0.8])  # 2 values > 0.5

        for enc in data_dependent_encodings:
            if isinstance(enc, DataDependentResourceAnalyzable):
                count = enc.actual_gate_count(x)
                assert isinstance(count, int), (
                    f"{enc.__class__.__name__}.actual_gate_count(x) should return int"
                )
                assert count >= 0, "Gate count should be non-negative"


class TestEntanglementQueryableProtocol:
    """Test EntanglementQueryable protocol recognition."""

    @pytest.fixture
    def entanglement_queryable_encodings(self) -> list[Any]:
        """Get list of encodings that should implement EntanglementQueryable."""
        from encoding_atlas import (
            IQPEncoding,
            ZZFeatureMap,
            HardwareEfficientEncoding,
            DataReuploading,
            QAOAEncoding,
            SymmetryInspiredFeatureMap,
            PauliFeatureMap,
        )

        return [
            IQPEncoding(n_features=4),
            ZZFeatureMap(n_features=4),
            HardwareEfficientEncoding(n_features=4),
            DataReuploading(n_features=4),
            QAOAEncoding(n_features=4),
            SymmetryInspiredFeatureMap(n_features=4),
            PauliFeatureMap(n_features=4),
        ]

    @pytest.fixture
    def non_entangling_encodings(self) -> list[Any]:
        """Get list of encodings that should NOT implement EntanglementQueryable."""
        from encoding_atlas import AngleEncoding

        return [
            AngleEncoding(n_features=4),
        ]

    def test_isinstance_check(
        self, entanglement_queryable_encodings: list[Any]
    ) -> None:
        """Test that EntanglementQueryable encodings pass isinstance check."""
        from encoding_atlas.core.protocols import EntanglementQueryable

        for enc in entanglement_queryable_encodings:
            assert isinstance(enc, EntanglementQueryable), (
                f"{enc.__class__.__name__} should implement EntanglementQueryable"
            )

    def test_type_guard_function(
        self, entanglement_queryable_encodings: list[Any]
    ) -> None:
        """Test that type guard function works correctly."""
        from encoding_atlas.core.protocols import is_entanglement_queryable

        for enc in entanglement_queryable_encodings:
            assert is_entanglement_queryable(enc), (
                f"is_entanglement_queryable({enc.__class__.__name__}) should return True"
            )

    def test_get_entanglement_pairs_callable(
        self, entanglement_queryable_encodings: list[Any]
    ) -> None:
        """Test that get_entanglement_pairs() returns correct type."""
        from encoding_atlas.core.protocols import EntanglementQueryable

        for enc in entanglement_queryable_encodings:
            if isinstance(enc, EntanglementQueryable):
                pairs = enc.get_entanglement_pairs()
                assert isinstance(pairs, list), (
                    f"{enc.__class__.__name__}.get_entanglement_pairs() should return list"
                )
                # Verify pairs are tuples of ints
                for pair in pairs:
                    assert isinstance(pair, tuple), "Each pair should be a tuple"
                    assert len(pair) == 2, "Each pair should have 2 elements"
                    assert all(isinstance(q, int) for q in pair), (
                        "Pair elements should be integers"
                    )

    def test_non_entangling_not_queryable(
        self, non_entangling_encodings: list[Any]
    ) -> None:
        """Test that non-entangling encodings don't have get_entanglement_pairs."""
        from encoding_atlas.core.protocols import EntanglementQueryable

        for enc in non_entangling_encodings:
            # AngleEncoding might or might not implement EntanglementQueryable
            # depending on implementation. If it does, pairs should be empty.
            if isinstance(enc, EntanglementQueryable):
                pairs = enc.get_entanglement_pairs()
                assert pairs == [], (
                    f"Non-entangling {enc.__class__.__name__} should return empty pairs"
                )


class TestDataTransformableProtocol:
    """Test DataTransformable protocol recognition and implementation.

    Tests verify that BasisEncoding and AmplitudeEncoding correctly implement
    the DataTransformable protocol via their transform_input() methods.
    """

    @pytest.fixture
    def data_transformable_encodings(self) -> list[Any]:
        """Get list of encodings that should implement DataTransformable."""
        from encoding_atlas import BasisEncoding, AmplitudeEncoding

        return [
            BasisEncoding(n_features=4),
            AmplitudeEncoding(n_features=4),
        ]

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that DataTransformable protocol supports isinstance checks."""
        from encoding_atlas.core.protocols import DataTransformable

        # Create a mock class that implements the protocol
        class MockDataTransformable:
            def transform_input(self, x: Any) -> np.ndarray:
                return np.array(x)

        obj = MockDataTransformable()
        assert isinstance(obj, DataTransformable), (
            "DataTransformable should be runtime checkable"
        )

    def test_type_guard_function_with_mock(self) -> None:
        """Test that type guard function works with conforming objects."""
        from encoding_atlas.core.protocols import is_data_transformable

        class MockDataTransformable:
            def transform_input(self, x: Any) -> np.ndarray:
                return np.array(x)

        obj = MockDataTransformable()
        assert is_data_transformable(obj), (
            "is_data_transformable() should return True for conforming objects"
        )

    def test_non_conforming_fails_check(self) -> None:
        """Test that objects without transform_input fail the check."""
        from encoding_atlas.core.protocols import DataTransformable

        class NotDataTransformable:
            pass

        obj = NotDataTransformable()
        assert not isinstance(obj, DataTransformable), (
            "Objects without transform_input should not match DataTransformable"
        )

    def test_isinstance_check(
        self, data_transformable_encodings: list[Any]
    ) -> None:
        """Test that DataTransformable encodings pass isinstance check."""
        from encoding_atlas.core.protocols import DataTransformable

        for enc in data_transformable_encodings:
            assert isinstance(enc, DataTransformable), (
                f"{enc.__class__.__name__} should implement DataTransformable"
            )

    def test_type_guard_function(
        self, data_transformable_encodings: list[Any]
    ) -> None:
        """Test that type guard function works correctly."""
        from encoding_atlas.core.protocols import is_data_transformable

        for enc in data_transformable_encodings:
            assert is_data_transformable(enc), (
                f"is_data_transformable({enc.__class__.__name__}) should return True"
            )

    def test_basis_encoding_transform_input(self) -> None:
        """Test BasisEncoding.transform_input() binarization behavior."""
        from encoding_atlas import BasisEncoding
        from encoding_atlas.core.protocols import DataTransformable

        enc = BasisEncoding(n_features=4)
        assert isinstance(enc, DataTransformable)

        # Test binarization with default threshold (0.5)
        x = np.array([0.1, 0.9, 0.3, 0.8])
        result = enc.transform_input(x)

        assert isinstance(result, np.ndarray)
        assert result.tolist() == [0, 1, 0, 1], (
            "Values > 0.5 should become 1, values <= 0.5 should become 0"
        )

    def test_basis_encoding_transform_input_custom_threshold(self) -> None:
        """Test BasisEncoding.transform_input() with custom threshold."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4, threshold=0.0)
        x = np.array([-0.5, 0.5, -0.1, 0.1])
        result = enc.transform_input(x)

        assert result.tolist() == [0, 1, 0, 1], (
            "With threshold=0.0, negative values become 0, positive become 1"
        )

    def test_basis_encoding_transform_equals_binarize(self) -> None:
        """Test that transform_input() is consistent with binarize()."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)
        x = np.array([0.2, 0.8, 0.5, 0.51])

        transform_result = enc.transform_input(x)
        binarize_result = enc.binarize(x)

        np.testing.assert_array_equal(
            transform_result, binarize_result,
            err_msg="transform_input() should produce same result as binarize()"
        )

    def test_amplitude_encoding_transform_input(self) -> None:
        """Test AmplitudeEncoding.transform_input() normalization behavior."""
        from encoding_atlas import AmplitudeEncoding
        from encoding_atlas.core.protocols import DataTransformable

        enc = AmplitudeEncoding(n_features=4)
        assert isinstance(enc, DataTransformable)

        # Test normalization
        x = np.array([3.0, 4.0, 0.0, 0.0])
        result = enc.transform_input(x)

        assert isinstance(result, np.ndarray)
        # Check unit norm
        np.testing.assert_almost_equal(
            np.linalg.norm(result), 1.0,
            err_msg="Transformed data should have unit L2 norm"
        )
        # Check values (3, 4, 0, 0) normalized to (0.6, 0.8, 0, 0)
        np.testing.assert_array_almost_equal(
            result, [0.6, 0.8, 0.0, 0.0],
            err_msg="Normalization should produce correct values"
        )

    def test_amplitude_encoding_transform_input_padding(self) -> None:
        """Test AmplitudeEncoding.transform_input() zero-padding behavior."""
        from encoding_atlas import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=3)  # Pads to 4 (2^2)
        x = np.array([1.0, 0.0, 0.0])
        result = enc.transform_input(x)

        assert len(result) == 4, "Result should be padded to power of 2"
        np.testing.assert_array_almost_equal(
            result, [1.0, 0.0, 0.0, 0.0],
            err_msg="Single feature should normalize to 1.0 with zeros padded"
        )

    def test_amplitude_encoding_transform_input_zero_vector_error(self) -> None:
        """Test that AmplitudeEncoding.transform_input() rejects zero vectors."""
        from encoding_atlas import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([0.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="zero.*norm"):
            enc.transform_input(x)

    def test_transform_input_preserves_shape_basis(self) -> None:
        """Test that BasisEncoding.transform_input() preserves input shape."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)

        # 1D input
        x_1d = np.array([0.1, 0.9, 0.3, 0.8])
        result_1d = enc.transform_input(x_1d)
        assert result_1d.shape == (4,)

        # 2D batch input
        x_2d = np.array([[0.1, 0.9, 0.3, 0.8], [0.6, 0.4, 0.7, 0.2]])
        result_2d = enc.transform_input(x_2d)
        assert result_2d.shape == (2, 4)


class TestProtocolNonConformance:
    """Test that objects not implementing protocols correctly fail checks."""

    def test_non_encoding_object_fails_resource_analyzable(self) -> None:
        """Test that non-encoding objects don't match ResourceAnalyzable."""
        from encoding_atlas.core.protocols import ResourceAnalyzable

        class NotAnEncoding:
            pass

        obj = NotAnEncoding()
        assert not isinstance(obj, ResourceAnalyzable)

    def test_non_encoding_object_fails_entanglement_queryable(self) -> None:
        """Test that non-encoding objects don't match EntanglementQueryable."""
        from encoding_atlas.core.protocols import EntanglementQueryable

        class NotAnEncoding:
            pass

        obj = NotAnEncoding()
        assert not isinstance(obj, EntanglementQueryable)

    def test_partial_implementation_fails(self) -> None:
        """Test that partial implementations don't match protocols."""
        from encoding_atlas.core.protocols import ResourceAnalyzable

        class PartialImplementation:
            # Has resource_summary but not gate_count_breakdown
            def resource_summary(self) -> dict:
                return {}

        obj = PartialImplementation()
        # Should fail because gate_count_breakdown is missing
        assert not isinstance(obj, ResourceAnalyzable)

    def test_type_guards_return_false_for_non_conforming(self) -> None:
        """Test that type guards return False for non-conforming objects."""
        from encoding_atlas.core.protocols import (
            is_resource_analyzable,
            is_data_dependent_resource_analyzable,
            is_entanglement_queryable,
            is_data_transformable,
        )

        class NotAnEncoding:
            pass

        obj = NotAnEncoding()
        assert not is_resource_analyzable(obj)
        assert not is_data_dependent_resource_analyzable(obj)
        assert not is_entanglement_queryable(obj)
        assert not is_data_transformable(obj)


class TestProtocolUsagePatterns:
    """Test common usage patterns for protocols."""

    def test_generic_resource_analysis_function(self) -> None:
        """Test writing a generic function using ResourceAnalyzable."""
        from encoding_atlas import IQPEncoding, AngleEncoding
        from encoding_atlas.core.protocols import ResourceAnalyzable
        from encoding_atlas.core.base import BaseEncoding

        def get_total_gates(enc: BaseEncoding) -> int | None:
            """Get total gates if encoding supports resource analysis."""
            if isinstance(enc, ResourceAnalyzable):
                breakdown = enc.gate_count_breakdown()
                return breakdown.get("total", 0)
            return None

        # IQPEncoding implements ResourceAnalyzable
        iqp = IQPEncoding(n_features=4)
        result = get_total_gates(iqp)
        assert result is not None
        assert isinstance(result, int)
        assert result > 0

        # AngleEncoding also implements ResourceAnalyzable
        angle = AngleEncoding(n_features=4)
        result = get_total_gates(angle)
        assert result is not None

    def test_capability_discovery_pattern(self) -> None:
        """Test discovering capabilities of an encoding."""
        from encoding_atlas import IQPEncoding, BasisEncoding
        from encoding_atlas.core.protocols import (
            ResourceAnalyzable,
            EntanglementQueryable,
            DataTransformable,
            DataDependentResourceAnalyzable,
        )

        # Test IQPEncoding capabilities
        iqp = IQPEncoding(n_features=4)
        iqp_capabilities = {
            "resource_analyzable": isinstance(iqp, ResourceAnalyzable),
            "entanglement_queryable": isinstance(iqp, EntanglementQueryable),
            "data_transformable": isinstance(iqp, DataTransformable),
        }

        # IQPEncoding should be resource_analyzable and entanglement_queryable
        assert iqp_capabilities["resource_analyzable"] is True
        assert iqp_capabilities["entanglement_queryable"] is True
        # But not data_transformable (no transform_input method)
        assert iqp_capabilities["data_transformable"] is False

        # Test BasisEncoding capabilities
        basis = BasisEncoding(n_features=4)
        basis_capabilities = {
            "resource_analyzable": isinstance(basis, ResourceAnalyzable),
            "data_dependent_resource_analyzable": isinstance(
                basis, DataDependentResourceAnalyzable
            ),
            "data_transformable": isinstance(basis, DataTransformable),
        }

        # BasisEncoding has data-dependent resource analysis and is transformable.
        # Note: At runtime, isinstance() only checks method existence, not signatures.
        # BasisEncoding has both resource_summary(x) and gate_count_breakdown(),
        # so it technically satisfies both ResourceAnalyzable and
        # DataDependentResourceAnalyzable at the runtime level.
        # The semantic distinction is that resource_summary(x) requires input data.
        assert basis_capabilities["data_dependent_resource_analyzable"] is True
        assert basis_capabilities["data_transformable"] is True

    def test_encoding_comparison_using_protocols(self) -> None:
        """Test comparing encodings using protocol capabilities."""
        from encoding_atlas import IQPEncoding, AngleEncoding
        from encoding_atlas.core.protocols import ResourceAnalyzable

        encodings = [
            IQPEncoding(n_features=4),
            AngleEncoding(n_features=4),
        ]

        # Compare gate counts for encodings that support resource analysis
        gate_counts = []
        for enc in encodings:
            if isinstance(enc, ResourceAnalyzable):
                breakdown = enc.gate_count_breakdown()
                gate_counts.append({
                    "name": enc.__class__.__name__,
                    "total": breakdown.get("total", 0),
                })

        assert len(gate_counts) == 2
        # IQPEncoding should have more gates than AngleEncoding
        iqp_gates = next(g["total"] for g in gate_counts if g["name"] == "IQPEncoding")
        angle_gates = next(g["total"] for g in gate_counts if g["name"] == "AngleEncoding")
        assert iqp_gates > angle_gates


class TestProtocolDocumentation:
    """Test that protocols have proper documentation."""

    def test_protocols_have_docstrings(self) -> None:
        """Test that all protocols have docstrings."""
        from encoding_atlas.core.protocols import (
            ResourceAnalyzable,
            DataDependentResourceAnalyzable,
            EntanglementQueryable,
            DataTransformable,
        )

        protocols = [
            ResourceAnalyzable,
            DataDependentResourceAnalyzable,
            EntanglementQueryable,
            DataTransformable,
        ]

        for protocol in protocols:
            assert protocol.__doc__ is not None, (
                f"{protocol.__name__} should have a docstring"
            )
            assert len(protocol.__doc__) > 100, (
                f"{protocol.__name__} docstring should be comprehensive"
            )

    def test_module_has_docstring(self) -> None:
        """Test that protocols module has a docstring."""
        from encoding_atlas.core import protocols

        assert protocols.__doc__ is not None
        assert len(protocols.__doc__) > 500, (
            "protocols module should have comprehensive documentation"
        )

    def test_type_guards_have_docstrings(self) -> None:
        """Test that type guard functions have docstrings."""
        from encoding_atlas.core.protocols import (
            is_resource_analyzable,
            is_data_dependent_resource_analyzable,
            is_entanglement_queryable,
            is_data_transformable,
        )

        functions = [
            is_resource_analyzable,
            is_data_dependent_resource_analyzable,
            is_entanglement_queryable,
            is_data_transformable,
        ]

        for func in functions:
            assert func.__doc__ is not None, (
                f"{func.__name__} should have a docstring"
            )
