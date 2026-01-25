"""Example: Comparing multiple encodings."""

import numpy as np
from encoding_atlas import (
    AngleEncoding,
    IQPEncoding,
    ZZFeatureMap,
    AmplitudeEncoding,
)
from encoding_atlas.analysis import count_resources

# Create encodings
encodings = [
    ("Angle (RY)", AngleEncoding(n_features=4)),
    ("IQP (2 reps)", IQPEncoding(n_features=4, reps=2)),
    ("ZZ Feature Map", ZZFeatureMap(n_features=4, reps=2)),
    ("Amplitude", AmplitudeEncoding(n_features=4)),
]

print("Encoding Comparison")
print("=" * 60)
print(f"{'Name':<20} {'Qubits':<8} {'Depth':<8} {'Entangling':<12} {'Simulable':<12}")
print("-" * 60)

for name, enc in encodings:
    props = enc.properties
    print(
        f"{name:<20} {props.n_qubits:<8} {props.depth:<8} "
        f"{str(props.is_entangling):<12} {props.simulability:<12}"
    )
