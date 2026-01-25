"""Abstract backend interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseBackend(ABC):
    """Abstract base class for quantum framework backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        ...

    @abstractmethod
    def create_circuit(self, n_qubits: int) -> Any:
        """Create a new circuit."""
        ...

    @abstractmethod
    def add_rotation(
        self, circuit: Any, qubit: int, axis: str, angle: float
    ) -> Any:
        """Add a rotation gate."""
        ...

    @abstractmethod
    def add_cnot(self, circuit: Any, control: int, target: int) -> Any:
        """Add a CNOT gate."""
        ...
