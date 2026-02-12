"""Quantum data encoding implementations."""

from encoding_atlas.encodings.amplitude import AmplitudeEncoding
from encoding_atlas.encodings.angle import AngleEncoding
from encoding_atlas.encodings.basis import BasisEncoding
from encoding_atlas.encodings.data_reuploading import DataReuploading
from encoding_atlas.encodings.equivariant_feature_map import (
    CyclicEquivariantFeatureMap,
    SO2EquivariantFeatureMap,
    SwapEquivariantFeatureMap,
)
from encoding_atlas.encodings.hamiltonian import HamiltonianEncoding
from encoding_atlas.encodings.hardware_efficient import HardwareEfficientEncoding
from encoding_atlas.encodings.higher_order_angle import HigherOrderAngleEncoding
from encoding_atlas.encodings.iqp import IQPEncoding
from encoding_atlas.encodings.pauli_feature_map import PauliFeatureMap
from encoding_atlas.encodings.qaoa_encoding import QAOAEncoding
from encoding_atlas.encodings.symmetry_inspired_feature_map import (
    CovariantFeatureMap,  # Backwards compatibility alias
    SymmetryInspiredFeatureMap,
)
from encoding_atlas.encodings.trainable import TrainableEncoding
from encoding_atlas.encodings.zz_feature_map import ZZFeatureMap

__all__ = [
    "AngleEncoding",
    "AmplitudeEncoding",
    "BasisEncoding",
    "IQPEncoding",
    "ZZFeatureMap",
    "PauliFeatureMap",
    "DataReuploading",
    "HardwareEfficientEncoding",
    "HigherOrderAngleEncoding",
    "QAOAEncoding",
    "HamiltonianEncoding",
    "SymmetryInspiredFeatureMap",
    "CovariantFeatureMap",  # Backwards compatibility alias
    "TrainableEncoding",
    "SO2EquivariantFeatureMap",
    "CyclicEquivariantFeatureMap",
    "SwapEquivariantFeatureMap",
]

# Register all encodings
from encoding_atlas.encodings import _registry  # noqa: F401
