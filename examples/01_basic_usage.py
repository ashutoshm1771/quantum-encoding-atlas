"""Basic usage example for Quantum Encoding Atlas."""

import numpy as np
from encoding_atlas import AngleEncoding, IQPEncoding, get_encoding, list_encodings

# List all available encodings
print("Available encodings:")
print(list_encodings())

# Create an angle encoding
angle_enc = AngleEncoding(n_features=4, rotation="Y")
print(f"\nAngle Encoding: {angle_enc}")
print(f"  Qubits: {angle_enc.n_qubits}")
print(f"  Depth: {angle_enc.depth}")
print(f"  Entangling: {angle_enc.properties.is_entangling}")

# Create an IQP encoding
iqp_enc = IQPEncoding(n_features=4, reps=2)
print(f"\nIQP Encoding: {iqp_enc}")
print(f"  Qubits: {iqp_enc.n_qubits}")
print(f"  Depth: {iqp_enc.depth}")
print(f"  Entangling: {iqp_enc.properties.is_entangling}")

# Get encoding from registry
enc = get_encoding("zz_feature_map", n_features=4, reps=2)
print(f"\nFrom registry: {enc}")
