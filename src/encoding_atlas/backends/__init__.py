"""Backend adapters for quantum frameworks."""

from encoding_atlas.backends.base import BaseBackend
from encoding_atlas.backends._detection import (
    is_pennylane_available,
    is_qiskit_available,
    is_cirq_available,
    get_available_backends,
)

__all__ = [
    "BaseBackend",
    "is_pennylane_available",
    "is_qiskit_available",
    "is_cirq_available",
    "get_available_backends",
]
