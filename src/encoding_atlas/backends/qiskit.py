"""Qiskit backend implementation."""

from typing import Any
from encoding_atlas.backends.base import BaseBackend
from encoding_atlas.backends._detection import is_qiskit_available


class QiskitBackend(BaseBackend):
    """Qiskit backend for circuit generation."""

    @property
    def name(self) -> str:
        return "qiskit"

    def is_available(self) -> bool:
        return is_qiskit_available()

    def create_circuit(self, n_qubits: int) -> Any:
        raise NotImplementedError("Use encoding.get_circuit() directly")

    def add_rotation(
        self, circuit: Any, qubit: int, axis: str, angle: float
    ) -> Any:
        raise NotImplementedError("Use encoding.get_circuit() directly")

    def add_cnot(self, circuit: Any, control: int, target: int) -> Any:
        raise NotImplementedError("Use encoding.get_circuit() directly")
