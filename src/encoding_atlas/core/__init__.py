"""Core abstractions for quantum encodings.

This module provides the foundational components for the Quantum Encoding Atlas:

Core Classes
------------
- :class:`BaseEncoding` : Abstract base class for all encodings
- :class:`EncodingProperties` : Immutable encoding property container
- :class:`ResourceSummary` : Resource usage summary

Capability Protocols
--------------------
- :class:`ResourceAnalyzable` : Data-independent resource analysis
- :class:`DataDependentResourceAnalyzable` : Data-dependent resource analysis
- :class:`EntanglementQueryable` : Entanglement structure queries
- :class:`DataTransformable` : Data transformation exposure

Registry Functions
------------------
- :func:`get_encoding` : Get encoding by name
- :func:`list_encodings` : List all registered encodings
- :func:`register_encoding` : Register a new encoding

Type Aliases
------------
- :data:`BackendType` : Supported quantum backends
- :data:`CircuitType` : Circuit types from all backends
- :data:`FloatArray` : NumPy float array type

Exceptions
----------
- :exc:`EncodingError` : Base encoding exception
- :exc:`ValidationError` : Input validation errors
- :exc:`BackendError` : Backend-specific errors
- :exc:`RegistryError` : Registry operation errors
"""

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.exceptions import (
    BackendError,
    EncodingError,
    RegistryError,
    ValidationError,
)
from encoding_atlas.core.properties import EncodingProperties, ResourceSummary
from encoding_atlas.core.protocols import (
    DataDependentResourceAnalyzable,
    DataTransformable,
    EntanglementQueryable,
    ResourceAnalyzable,
    is_data_dependent_resource_analyzable,
    is_data_transformable,
    is_entanglement_queryable,
    is_resource_analyzable,
)
from encoding_atlas.core.registry import get_encoding, list_encodings, register_encoding
from encoding_atlas.core.types import BackendType, CircuitType, FloatArray

__all__ = [
    # Core classes
    "BaseEncoding",
    "EncodingProperties",
    "ResourceSummary",
    # Registry functions
    "get_encoding",
    "list_encodings",
    "register_encoding",
    # Exceptions
    "EncodingError",
    "ValidationError",
    "BackendError",
    "RegistryError",
    # Type aliases
    "BackendType",
    "CircuitType",
    "FloatArray",
    # Capability protocols
    "ResourceAnalyzable",
    "DataDependentResourceAnalyzable",
    "EntanglementQueryable",
    "DataTransformable",
    # Type guards
    "is_resource_analyzable",
    "is_data_dependent_resource_analyzable",
    "is_entanglement_queryable",
    "is_data_transformable",
]
