"""Encoding properties data structures."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class EncodingProperties:
    """Properties computed for a quantum encoding.

    Attributes
    ----------
    n_qubits : int
        Number of qubits required.
    depth : int
        Circuit depth (number of time steps).
    gate_count : int
        Total number of gates.
    single_qubit_gates : int
        Number of single-qubit gates.
    two_qubit_gates : int
        Number of two-qubit gates.
    parameter_count : int
        Number of data-dependent parameters.
    is_entangling : bool
        Whether the encoding creates entanglement.
    simulability : str
        Classical simulability status.
    """

    n_qubits: int
    depth: int
    gate_count: int
    single_qubit_gates: int
    two_qubit_gates: int
    parameter_count: int
    is_entangling: bool
    simulability: Literal["simulable", "conditionally_simulable", "not_simulable"]

    expressibility: Optional[float] = None
    entanglement_capability: Optional[float] = None
    trainability_estimate: Optional[float] = None
    noise_resilience_estimate: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate properties after initialization."""
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if self.depth < 0:
            raise ValueError("depth cannot be negative")
        if self.gate_count < 0:
            raise ValueError("gate_count cannot be negative")
        if self.single_qubit_gates + self.two_qubit_gates != self.gate_count:
            raise ValueError(
                "single_qubit_gates + two_qubit_gates must equal gate_count"
            )

    def to_dict(self) -> dict:
        """Convert properties to dictionary."""
        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "gate_count": self.gate_count,
            "single_qubit_gates": self.single_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "parameter_count": self.parameter_count,
            "is_entangling": self.is_entangling,
            "simulability": self.simulability,
            "expressibility": self.expressibility,
            "entanglement_capability": self.entanglement_capability,
            "trainability_estimate": self.trainability_estimate,
            "noise_resilience_estimate": self.noise_resilience_estimate,
            "notes": self.notes,
        }


@dataclass
class ResourceSummary:
    """Summary of computational resources for an encoding."""

    n_qubits: int
    depth: int
    gate_count: int
    two_qubit_gates: int

    @property
    def two_qubit_ratio(self) -> float:
        """Ratio of two-qubit gates to total gates."""
        if self.gate_count == 0:
            return 0.0
        return self.two_qubit_gates / self.gate_count
