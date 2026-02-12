"""Backend adapters for quantum frameworks."""

from encoding_atlas.backends._detection import (
    get_available_backends,
    is_cirq_available,
    is_pennylane_available,
    is_qiskit_available,
)
from encoding_atlas.backends.base import BaseBackend

__all__ = [
    "BaseBackend",
    "is_pennylane_available",
    "is_qiskit_available",
    "is_cirq_available",
    "get_available_backends",
]
