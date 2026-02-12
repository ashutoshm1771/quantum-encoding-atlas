"""
Quantum Encoding Atlas
======================

A comprehensive library for quantum data encodings in machine learning.

Quick Start
-----------
>>> from encoding_atlas import AngleEncoding, IQPEncoding
>>> encoding = IQPEncoding(n_features=4, reps=2)
>>> circuit = encoding.get_circuit(data)

Capability Protocols
--------------------
Check encoding capabilities at runtime using protocols:

>>> from encoding_atlas import IQPEncoding
>>> from encoding_atlas.core.protocols import ResourceAnalyzable, EntanglementQueryable
>>>
>>> enc = IQPEncoding(n_features=4)
>>>
>>> # Check capabilities
>>> if isinstance(enc, ResourceAnalyzable):
...     summary = enc.resource_summary()
...     print(f"Total gates: {summary['gate_counts']['total']}")
>>>
>>> if isinstance(enc, EntanglementQueryable):
...     pairs = enc.get_entanglement_pairs()
...     print(f"Entanglement pairs: {len(pairs)}")

Available Protocols
~~~~~~~~~~~~~~~~~~~
- ``ResourceAnalyzable``: Data-independent resource analysis
- ``DataDependentResourceAnalyzable``: Data-dependent resource analysis
- ``EntanglementQueryable``: Query entanglement structure
- ``DataTransformable``: Expose data preprocessing logic

For more information, see the documentation at:
https://q-encoding-atlas.web.app/documentation
"""

from encoding_atlas import analysis, benchmark, guide, visualization
from encoding_atlas._version import __version__
from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties

# Capability protocols (Layer 2 of the Layered Contract Architecture)
from encoding_atlas.core.protocols import (
    DataDependentResourceAnalyzable,
    DataTransformable,
    EntanglementQueryable,
    ResourceAnalyzable,
)
from encoding_atlas.core.registry import get_encoding, list_encodings, register_encoding

# Encodings
from encoding_atlas.encodings import (
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    CovariantFeatureMap,  # Backwards compatibility alias
    CyclicEquivariantFeatureMap,
    DataReuploading,
    HamiltonianEncoding,
    HardwareEfficientEncoding,
    HigherOrderAngleEncoding,
    IQPEncoding,
    PauliFeatureMap,
    QAOAEncoding,
    SO2EquivariantFeatureMap,
    SwapEquivariantFeatureMap,
    SymmetryInspiredFeatureMap,
    TrainableEncoding,
    ZZFeatureMap,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "BaseEncoding",
    "EncodingProperties",
    # Registry functions
    "get_encoding",
    "list_encodings",
    "register_encoding",
    # Capability protocols
    "ResourceAnalyzable",
    "DataDependentResourceAnalyzable",
    "EntanglementQueryable",
    "DataTransformable",
    # Encodings
    "AmplitudeEncoding",
    "AngleEncoding",
    "BasisEncoding",
    "CovariantFeatureMap",  # Backwards compatibility alias
    "CyclicEquivariantFeatureMap",
    "DataReuploading",
    "HamiltonianEncoding",
    "HardwareEfficientEncoding",
    "HigherOrderAngleEncoding",
    "IQPEncoding",
    "PauliFeatureMap",
    "QAOAEncoding",
    "SO2EquivariantFeatureMap",
    "SwapEquivariantFeatureMap",
    "SymmetryInspiredFeatureMap",
    "TrainableEncoding",
    "ZZFeatureMap",
    # Modules
    "analysis",
    "benchmark",
    "guide",
    "visualization",
]
