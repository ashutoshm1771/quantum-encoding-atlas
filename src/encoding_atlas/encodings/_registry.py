"""Internal registry setup for encodings."""

from encoding_atlas.core.registry import register_encoding
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
    SymmetryInspiredFeatureMap,
)
from encoding_atlas.encodings.trainable import TrainableEncoding
from encoding_atlas.encodings.zz_feature_map import ZZFeatureMap

# Register encodings with the global registry
_encodings_to_register = [
    ("angle", AngleEncoding),
    ("angle_ry", AngleEncoding),
    ("amplitude", AmplitudeEncoding),
    ("basis", BasisEncoding),
    ("iqp", IQPEncoding),
    ("zz_feature_map", ZZFeatureMap),
    ("pauli_feature_map", PauliFeatureMap),
    ("data_reuploading", DataReuploading),
    ("hardware_efficient", HardwareEfficientEncoding),
    ("higher_order_angle", HigherOrderAngleEncoding),
    ("qaoa", QAOAEncoding),
    ("qaoa_encoding", QAOAEncoding),
    ("hamiltonian", HamiltonianEncoding),
    ("hamiltonian_encoding", HamiltonianEncoding),
    ("symmetry_inspired_feature_map", SymmetryInspiredFeatureMap),
    ("symmetry_inspired", SymmetryInspiredFeatureMap),
    ("covariant_feature_map", SymmetryInspiredFeatureMap),  # Backwards compatibility
    ("covariant", SymmetryInspiredFeatureMap),  # Backwards compatibility
    ("trainable", TrainableEncoding),
    ("trainable_encoding", TrainableEncoding),
    ("so2_equivariant", SO2EquivariantFeatureMap),
    ("so2_equivariant_feature_map", SO2EquivariantFeatureMap),
    ("cyclic_equivariant", CyclicEquivariantFeatureMap),
    ("cyclic_equivariant_feature_map", CyclicEquivariantFeatureMap),
    ("swap_equivariant", SwapEquivariantFeatureMap),
    ("swap_equivariant_feature_map", SwapEquivariantFeatureMap),
]

for name, cls in _encodings_to_register:
    try:
        register_encoding(name)(cls)
    except Exception:
        pass  # Already registered
