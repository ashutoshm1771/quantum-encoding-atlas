"""Comprehensive tests for the encoding registry system.

This module tests the registry functionality including:
- Listing all registered encodings
- Getting encodings by name
- Dynamic registration
- Error handling for invalid names
- Registry state management
- Thread safety of registry operations

Run with: pytest tests/unit/core/test_registry.py -v
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from encoding_atlas.core.exceptions import RegistryError
from encoding_atlas.core.registry import (
    _ENCODING_REGISTRY,
    _clear_registry,
    get_encoding,
    list_encodings,
    register_encoding,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants for Special Encodings
# =============================================================================

# Some encodings have mathematical constraints on the number of features they accept.
# This mapping defines the required n_features for encodings with such constraints.
# Encodings not listed here default to using n_features=4 in tests.
_ENCODING_REQUIRED_N_FEATURES: dict[str, int] = {
    # SO(2) equivariant encoding requires exactly 2D input (Cartesian coordinates)
    # due to the mathematical properties of the SO(2) rotation group.
    "so2_equivariant": 2,
    "so2_equivariant_feature_map": 2,
}


def _get_test_n_features(encoding_name: str) -> int:
    """Get appropriate n_features for testing a specific encoding.

    Some encodings have mathematical constraints on their input dimensionality.
    This helper returns the appropriate n_features for testing each encoding.

    Parameters
    ----------
    encoding_name : str
        Name of the encoding to test.

    Returns
    -------
    int
        The appropriate n_features value for testing this encoding.
    """
    return _ENCODING_REQUIRED_N_FEATURES.get(encoding_name, 4)


def _get_test_data(encoding_name: str) -> np.ndarray:
    """Get appropriate test data for a specific encoding.

    Returns test data with the correct number of features for the encoding.

    Parameters
    ----------
    encoding_name : str
        Name of the encoding to test.

    Returns
    -------
    np.ndarray
        Test data with appropriate dimensions.
    """
    n_features = _get_test_n_features(encoding_name)
    # Generate deterministic test data based on feature count
    return np.array([0.1 * (i + 1) for i in range(n_features)])


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_registry():
    """Fixture that clears and restores registry state.

    This ensures tests don't interfere with each other.
    """
    # Store original state
    original_registry = dict(_ENCODING_REGISTRY)

    yield

    # Restore original state
    _clear_registry()
    for name, cls in original_registry.items():
        try:
            register_encoding(name)(cls)
        except RegistryError:
            pass  # Already registered in restore


# =============================================================================
# Test Class: Basic Registry Operations
# =============================================================================


class TestRegistryBasicOperations:
    """Tests for basic registry operations."""

    def test_list_encodings_returns_list(self) -> None:
        """Test that list_encodings returns a list."""
        encodings = list_encodings()
        assert isinstance(encodings, list)

    def test_list_encodings_sorted(self) -> None:
        """Test that list_encodings returns sorted names."""
        encodings = list_encodings()
        assert encodings == sorted(encodings)

    def test_list_encodings_contains_standard_encodings(self) -> None:
        """Test that standard encodings are registered."""
        encodings = list_encodings()

        expected = [
            "angle",
            "amplitude",
            "basis",
            "iqp",
            "zz_feature_map",
            "pauli_feature_map",
            "data_reuploading",
            "hardware_efficient",
            "higher_order_angle",
            "qaoa",
            "hamiltonian",
            "symmetry_inspired",
        ]

        for name in expected:
            assert name in encodings, f"Expected '{name}' in registry"

    def test_list_encodings_not_empty(self) -> None:
        """Test that registry is not empty."""
        encodings = list_encodings()
        assert len(encodings) > 0

    def test_get_encoding_basic(self) -> None:
        """Test getting an encoding by name."""
        encoding = get_encoding("angle", n_features=4)
        assert encoding.n_features == 4
        assert encoding.__class__.__name__ == "AngleEncoding"

    def test_get_encoding_returns_instance(self) -> None:
        """Test that get_encoding returns an instance, not class."""
        encoding = get_encoding("amplitude", n_features=4)
        # Should be an instance, not a type
        assert not isinstance(encoding, type)
        assert hasattr(encoding, "get_circuit")

    def test_get_unknown_encoding_raises(self) -> None:
        """Test that unknown encoding raises RegistryError."""
        with pytest.raises(RegistryError, match="Unknown encoding"):
            get_encoding("nonexistent_encoding", n_features=4)

    def test_get_unknown_encoding_lists_available(self) -> None:
        """Test that error message lists available encodings."""
        with pytest.raises(RegistryError) as exc_info:
            get_encoding("nonexistent", n_features=4)

        error_msg = str(exc_info.value)
        # Should mention some available encodings
        assert "angle" in error_msg or "Available" in error_msg


# =============================================================================
# Test Class: Registry Listing All Encodings
# =============================================================================


class TestRegistryListingAllEncodings:
    """Tests for listing all registered encodings.

    Note: Some encodings have mathematical constraints on their input
    dimensionality (e.g., SO2EquivariantFeatureMap requires exactly 2D input).
    Tests use helper functions to determine appropriate n_features and test
    data for each encoding.
    """

    def test_all_listed_encodings_can_be_instantiated(self) -> None:
        """Test that every listed encoding can be instantiated."""
        encodings = list_encodings()

        for name in encodings:
            try:
                n_features = _get_test_n_features(name)
                enc = get_encoding(name, n_features=n_features)
                assert enc is not None
                assert hasattr(enc, "n_features")
                assert enc.n_features == n_features
            except Exception as e:
                pytest.fail(f"Failed to instantiate '{name}': {e}")

    def test_all_encodings_have_get_circuit(self) -> None:
        """Test that all encodings have get_circuit method."""
        encodings = list_encodings()

        for name in encodings:
            n_features = _get_test_n_features(name)
            enc = get_encoding(name, n_features=n_features)
            assert hasattr(enc, "get_circuit"), f"'{name}' missing get_circuit"
            assert callable(enc.get_circuit)

    def test_all_encodings_have_properties(self) -> None:
        """Test that all encodings have properties attribute."""
        encodings = list_encodings()

        for name in encodings:
            n_features = _get_test_n_features(name)
            enc = get_encoding(name, n_features=n_features)
            assert hasattr(enc, "properties"), f"'{name}' missing properties"
            assert hasattr(enc, "n_qubits"), f"'{name}' missing n_qubits"
            assert hasattr(enc, "depth"), f"'{name}' missing depth"

    def test_encoding_count(self) -> None:
        """Test that we have expected number of encodings."""
        encodings = list_encodings()

        # We should have at least 12 distinct encoding types
        # (some have aliases)
        assert len(encodings) >= 12

    def test_all_encodings_generate_valid_circuits(self) -> None:
        """Test that all encodings generate valid circuits."""
        encodings = list_encodings()

        for name in encodings:
            x = _get_test_data(name)
            n_features = _get_test_n_features(name)
            enc = get_encoding(name, n_features=n_features)
            circuit = enc.get_circuit(x, backend="pennylane")
            assert circuit is not None, f"'{name}' returned None circuit"


# =============================================================================
# Test Class: Dynamic Registration
# =============================================================================


class TestDynamicRegistration:
    """Tests for dynamic encoding registration."""

    def test_register_new_encoding(self, clean_registry) -> None:
        """Test registering a new encoding class."""
        from encoding_atlas.core.base import BaseEncoding

        @register_encoding("test_encoding")
        class TestEncoding(BaseEncoding):
            def __init__(self, n_features: int):
                super().__init__(n_features=n_features)

            @property
            def n_qubits(self) -> int:
                return self.n_features

            @property
            def depth(self) -> int:
                return 1

            def _compute_properties(self):
                from encoding_atlas.core.properties import EncodingProperties

                return EncodingProperties(
                    n_qubits=self.n_features,
                    depth=1,
                    n_parameters=0,
                    is_entangling=False,
                    simulability="classically_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

        # Should be able to get the new encoding
        assert "test_encoding" in list_encodings()
        enc = get_encoding("test_encoding", n_features=4)
        assert enc.n_features == 4

    def test_register_duplicate_raises(self, clean_registry) -> None:
        """Test that registering duplicate name raises error."""
        from encoding_atlas.core.base import BaseEncoding

        @register_encoding("duplicate_test")
        class FirstEncoding(BaseEncoding):
            def __init__(self, n_features: int):
                super().__init__(n_features=n_features)

            @property
            def n_qubits(self) -> int:
                return self.n_features

            @property
            def depth(self) -> int:
                return 1

            def _compute_properties(self):
                from encoding_atlas.core.properties import EncodingProperties

                return EncodingProperties(
                    n_qubits=self.n_features,
                    depth=1,
                    n_parameters=0,
                    is_entangling=False,
                    simulability="classically_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

        # Second registration should fail
        with pytest.raises(RegistryError, match="already registered"):

            @register_encoding("duplicate_test")
            class SecondEncoding(BaseEncoding):
                def __init__(self, n_features: int):
                    super().__init__(n_features=n_features)

                @property
                def n_qubits(self) -> int:
                    return self.n_features

                @property
                def depth(self) -> int:
                    return 1

                def _compute_properties(self):
                    from encoding_atlas.core.properties import EncodingProperties

                    return EncodingProperties(
                        n_qubits=self.n_features,
                        depth=1,
                        n_parameters=0,
                        is_entangling=False,
                        simulability="classically_simulable",
                    )

                def _get_circuit_from_validated(self, x, backend):
                    return None

    def test_registration_preserves_class(self, clean_registry) -> None:
        """Test that registration returns the original class."""
        from encoding_atlas.core.base import BaseEncoding

        class OriginalClass(BaseEncoding):
            def __init__(self, n_features: int):
                super().__init__(n_features=n_features)

            @property
            def n_qubits(self) -> int:
                return self.n_features

            @property
            def depth(self) -> int:
                return 1

            def _compute_properties(self):
                from encoding_atlas.core.properties import EncodingProperties

                return EncodingProperties(
                    n_qubits=self.n_features,
                    depth=1,
                    n_parameters=0,
                    is_entangling=False,
                    simulability="classically_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

        registered_class = register_encoding("original_test")(OriginalClass)
        assert registered_class is OriginalClass


# =============================================================================
# Test Class: Error Handling
# =============================================================================


class TestRegistryErrorHandling:
    """Tests for registry error handling."""

    def test_invalid_encoding_name_empty(self) -> None:
        """Test error for empty encoding name."""
        with pytest.raises(RegistryError, match="Unknown encoding"):
            get_encoding("", n_features=4)

    def test_invalid_encoding_name_none(self) -> None:
        """Test error for None encoding name."""
        with pytest.raises((RegistryError, TypeError)):
            get_encoding(None, n_features=4)  # type: ignore

    def test_invalid_encoding_name_number(self) -> None:
        """Test error for numeric encoding name."""
        with pytest.raises((RegistryError, TypeError)):
            get_encoding(123, n_features=4)  # type: ignore

    def test_case_sensitivity(self) -> None:
        """Test that encoding names are case-sensitive."""
        # "angle" exists
        enc = get_encoding("angle", n_features=4)
        assert enc is not None

        # "ANGLE" or "Angle" should not exist (case-sensitive)
        with pytest.raises(RegistryError, match="Unknown encoding"):
            get_encoding("ANGLE", n_features=4)

    def test_encoding_with_missing_required_params(self) -> None:
        """Test error when required parameters are missing."""
        with pytest.raises(TypeError):
            # n_features is required for all encodings
            get_encoding("angle")

    def test_encoding_with_invalid_params(self) -> None:
        """Test error when invalid parameters are passed."""
        with pytest.raises(ValueError):
            get_encoding("angle", n_features=-1)

    def test_whitespace_in_name(self) -> None:
        """Test error for names with whitespace."""
        with pytest.raises(RegistryError, match="Unknown encoding"):
            get_encoding(" angle", n_features=4)

        with pytest.raises(RegistryError, match="Unknown encoding"):
            get_encoding("angle ", n_features=4)


# =============================================================================
# Test Class: Encoding Aliases
# =============================================================================


class TestEncodingAliases:
    """Tests for encoding name aliases."""

    def test_qaoa_aliases(self) -> None:
        """Test that QAOA encoding has multiple aliases."""
        # Both should work
        enc1 = get_encoding("qaoa", n_features=4)
        enc2 = get_encoding("qaoa_encoding", n_features=4)

        assert type(enc1) == type(enc2)

    def test_hamiltonian_aliases(self) -> None:
        """Test that Hamiltonian encoding has aliases."""
        enc1 = get_encoding("hamiltonian", n_features=4)
        enc2 = get_encoding("hamiltonian_encoding", n_features=4)

        assert type(enc1) == type(enc2)

    def test_covariant_aliases(self) -> None:
        """Test that Covariant feature map has aliases."""
        enc1 = get_encoding("covariant", n_features=4)
        enc2 = get_encoding("covariant_feature_map", n_features=4)

        assert type(enc1) == type(enc2)

    def test_angle_aliases(self) -> None:
        """Test AngleEncoding aliases."""
        enc1 = get_encoding("angle", n_features=4)
        enc2 = get_encoding("angle_ry", n_features=4)

        assert type(enc1) == type(enc2)


# =============================================================================
# Test Class: Thread Safety
# =============================================================================


@pytest.mark.thread_safety
class TestRegistryThreadSafety:
    """Tests for thread safety of registry operations."""

    def test_concurrent_get_encoding(self) -> None:
        """Test concurrent access to get_encoding."""
        results: list[Any] = []
        errors: list[Exception] = []

        def get_angle_encoding():
            try:
                enc = get_encoding("angle", n_features=4)
                results.append(enc)
            except Exception as e:
                errors.append(e)

        # Run concurrent accesses
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_angle_encoding) for _ in range(100)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 100

    def test_concurrent_list_encodings(self) -> None:
        """Test concurrent access to list_encodings."""
        results: list[list[str]] = []
        errors: list[Exception] = []

        def list_all():
            try:
                encodings = list_encodings()
                results.append(encodings)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(list_all) for _ in range(100)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert len(results) == 100

        # All results should be identical
        first = results[0]
        for r in results[1:]:
            assert r == first

    def test_concurrent_mixed_operations(self) -> None:
        """Test concurrent mixed read operations."""
        errors: list[Exception] = []

        def mixed_ops():
            try:
                _ = list_encodings()
                _ = get_encoding("angle", n_features=4)
                _ = get_encoding("iqp", n_features=4)
                _ = list_encodings()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mixed_ops) for _ in range(50)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Concurrent errors: {errors}"


# =============================================================================
# Test Class: Registry State Management
# =============================================================================


class TestRegistryStateManagement:
    """Tests for registry state management."""

    def test_clear_registry(self, clean_registry) -> None:
        """Test that _clear_registry works."""
        # Get initial count
        initial = len(list_encodings())
        assert initial > 0

        # Clear
        _clear_registry()
        assert len(list_encodings()) == 0

    def test_registry_isolation(self, clean_registry) -> None:
        """Test that registry changes are isolated."""
        from encoding_atlas.core.base import BaseEncoding

        initial_count = len(list_encodings())

        @register_encoding("isolated_test")
        class IsolatedEncoding(BaseEncoding):
            def __init__(self, n_features: int):
                super().__init__(n_features=n_features)

            @property
            def n_qubits(self) -> int:
                return self.n_features

            @property
            def depth(self) -> int:
                return 1

            def _compute_properties(self):
                from encoding_atlas.core.properties import EncodingProperties

                return EncodingProperties(
                    n_qubits=self.n_features,
                    depth=1,
                    n_parameters=0,
                    is_entangling=False,
                    simulability="classically_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

        # Count should increase
        new_count = len(list_encodings())
        assert new_count == initial_count + 1


# =============================================================================
# Test Class: Integration with Encodings
# =============================================================================


class TestRegistryEncodingIntegration:
    """Integration tests with actual encoding implementations."""

    @pytest.mark.parametrize(
        "name,params",
        [
            ("angle", {"n_features": 4}),
            ("amplitude", {"n_features": 4}),
            ("basis", {"n_features": 4}),
            ("iqp", {"n_features": 4}),
            ("zz_feature_map", {"n_features": 4}),
            ("pauli_feature_map", {"n_features": 4}),
            ("data_reuploading", {"n_features": 4}),
            ("hardware_efficient", {"n_features": 4}),
            ("higher_order_angle", {"n_features": 4}),
            ("qaoa", {"n_features": 4}),
            ("hamiltonian", {"n_features": 4}),
            ("covariant", {"n_features": 4}),
        ],
    )
    def test_encoding_via_registry_produces_valid_circuit(
        self, name: str, params: dict
    ) -> None:
        """Test that encodings from registry produce valid circuits."""
        enc = get_encoding(name, **params)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = enc.get_circuit(x, backend="pennylane")
        assert circuit is not None

    @pytest.mark.parametrize(
        "name",
        ["angle", "iqp", "zz_feature_map", "hardware_efficient"],
    )
    def test_encoding_via_registry_batch_processing(self, name: str) -> None:
        """Test batch processing via registry-obtained encodings."""
        enc = get_encoding(name, n_features=4)
        batch = np.random.randn(10, 4)

        circuits = enc.get_circuits(batch, backend="pennylane")
        assert len(circuits) == 10

    def test_encoding_via_registry_equality(self) -> None:
        """Test that encodings from registry support equality."""
        enc1 = get_encoding("angle", n_features=4)
        enc2 = get_encoding("angle", n_features=4)
        enc3 = get_encoding("angle", n_features=8)

        assert enc1 == enc2
        assert enc1 != enc3

    def test_encoding_via_registry_hashing(self) -> None:
        """Test that encodings from registry support hashing."""
        enc1 = get_encoding("angle", n_features=4)
        enc2 = get_encoding("angle", n_features=4)

        # Should be usable in sets
        encoding_set = {enc1, enc2}
        assert len(encoding_set) == 1  # Duplicates removed

        # Should be usable as dict keys
        encoding_dict = {enc1: "value1"}
        assert encoding_dict[enc2] == "value1"
