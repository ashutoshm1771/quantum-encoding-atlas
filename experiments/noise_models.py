"""Noise model definitions and utilities for Stage 5b.

This module provides noise level definitions and utilities for creating
noisy PennyLane devices for density matrix simulation with depolarizing
noise channels.

Noise Levels
------------
Three noise levels are defined based on realistic hardware error rates:

- **Low**: Single-qubit error rate 0.1%, two-qubit error rate 1%
  Represents near-term fault-tolerant or high-quality superconducting hardware.

- **Medium**: Single-qubit error rate 0.5%, two-qubit error rate 5%
  Represents typical NISQ-era superconducting or trapped-ion hardware.

- **High**: Single-qubit error rate 1%, two-qubit error rate 10%
  Represents noisy NISQ hardware or edge cases for robustness testing.

References
----------
.. [1] Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond."
       Quantum, 2, 79.

.. [2] Arute, F., et al. (2019). "Quantum supremacy using a programmable
       superconducting processor." Nature, 574, 505-510.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

__all__ = [
    "NOISE_LEVELS",
    "get_noise_params",
    "create_noisy_device",
    "apply_depolarizing_noise",
]

_logger = logging.getLogger(__name__)

# =============================================================================
# Noise Level Definitions
# =============================================================================

NOISE_LEVELS: dict[str, dict[str, float]] = {
    "low": {"single_qubit": 0.001, "two_qubit": 0.01},
    "medium": {"single_qubit": 0.005, "two_qubit": 0.05},
    "high": {"single_qubit": 0.01, "two_qubit": 0.10},
}
"""Predefined noise levels with depolarizing error rates.

Each level specifies:
- ``single_qubit``: Depolarizing probability after single-qubit gates.
- ``two_qubit``: Depolarizing probability after two-qubit gates.

The depolarizing channel with probability p replaces the state ρ with
the maximally mixed state with probability p:
    E(ρ) = (1-p)ρ + p·I/d

where d is the dimension (2 for single qubit, 4 for two qubits).
"""


def get_noise_params(noise_level: str) -> dict[str, float]:
    """Get noise parameters for a given noise level.

    Parameters
    ----------
    noise_level : str
        One of "low", "medium", or "high".

    Returns
    -------
    dict[str, float]
        Dictionary with keys "single_qubit" and "two_qubit" containing
        the depolarizing error probabilities.

    Raises
    ------
    ValueError
        If noise_level is not recognized.
    """
    if noise_level not in NOISE_LEVELS:
        raise ValueError(
            f"Unknown noise level '{noise_level}'. "
            f"Valid options: {list(NOISE_LEVELS.keys())}"
        )
    return NOISE_LEVELS[noise_level].copy()


# =============================================================================
# Device Creation
# =============================================================================


def create_noisy_device(n_qubits: int):
    """Create a PennyLane mixed-state device for noisy simulation.

    This creates a ``default.mixed`` device that supports density matrix
    simulation with quantum channels (noise operations).

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the device.

    Returns
    -------
    qml.Device
        A PennyLane device configured for mixed-state simulation.

    Raises
    ------
    ImportError
        If PennyLane is not installed.
    ValueError
        If n_qubits is invalid.

    Notes
    -----
    Memory scales as O(4^n) for density matrices, compared to O(2^n) for
    statevectors. For 8 qubits:
    - Statevector: 2^8 × 16 bytes = 4 KB
    - Density matrix: 4^8 × 16 bytes = 1 MB

    Keep n_qubits ≤ 8 for reasonable memory usage.
    """
    try:
        import pennylane as qml
    except ImportError as e:
        raise ImportError(
            "PennyLane is required for noisy simulation. "
            "Install it with: pip install pennylane"
        ) from e

    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")

    # Memory warning for large qubit counts
    if n_qubits > 8:
        density_matrix_bytes = (4 ** n_qubits) * 16
        density_matrix_mb = density_matrix_bytes / (1024 ** 2)
        _logger.warning(
            "Creating noisy device with %d qubits. Density matrix will require "
            "%.1f MB of memory. Consider reducing qubit count for faster "
            "simulation.",
            n_qubits,
            density_matrix_mb,
        )

    return qml.device("default.mixed", wires=n_qubits)


# =============================================================================
# Noise Application Utilities
# =============================================================================


def apply_depolarizing_noise(
    wire: int,
    noise_prob: float,
) -> None:
    """Apply a depolarizing channel to a single qubit.

    This function should be called within a PennyLane circuit context
    (inside a QNode function) to apply noise after gates.

    Parameters
    ----------
    wire : int
        The qubit wire to apply noise to.
    noise_prob : float
        Depolarizing probability in [0, 1].

    Notes
    -----
    The depolarizing channel transforms a single-qubit state as:

        E(ρ) = (1-p)ρ + p·I/2

    where p is the noise probability. This models the effect of random
    Pauli errors (X, Y, Z applied with equal probability p/3 each).
    """
    import pennylane as qml

    if noise_prob > 0:
        qml.DepolarizingChannel(noise_prob, wires=wire)


def apply_two_qubit_depolarizing_noise(
    wire0: int,
    wire1: int,
    noise_prob: float,
) -> None:
    """Apply depolarizing noise after a two-qubit gate.

    Since PennyLane's standard depolarizing channel is single-qubit,
    we apply independent single-qubit depolarizing channels to both
    qubits involved in the two-qubit gate. This is a common approximation
    for two-qubit gate errors.

    Parameters
    ----------
    wire0 : int
        First qubit wire.
    wire1 : int
        Second qubit wire.
    noise_prob : float
        Depolarizing probability for each qubit.

    Notes
    -----
    For a more accurate two-qubit depolarizing model, one could implement
    a custom 4×4 depolarizing channel. However, independent single-qubit
    channels provide a reasonable approximation and are more efficient
    to simulate.
    """
    import pennylane as qml

    if noise_prob > 0:
        qml.DepolarizingChannel(noise_prob, wires=wire0)
        qml.DepolarizingChannel(noise_prob, wires=wire1)


# =============================================================================
# Circuit Execution Utilities
# =============================================================================


def execute_noisy_circuit(
    encoding,
    x: NDArray[np.floating[Any]],
    noise_params: dict[str, float],
) -> NDArray[np.complexfloating[Any, Any]]:
    """Execute an encoding circuit with depolarizing noise and return density matrix.

    This function builds a noisy version of the encoding circuit by manually
    inserting depolarizing channels after each gate operation.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to simulate.
    x : NDArray[np.floating]
        Input feature vector of shape (n_features,).
    noise_params : dict[str, float]
        Noise parameters with keys "single_qubit" and "two_qubit".

    Returns
    -------
    NDArray[np.complexfloating]
        Density matrix of shape (2^n_qubits, 2^n_qubits).

    Raises
    ------
    SimulationError
        If the noisy simulation fails.

    Notes
    -----
    This function uses PennyLane's tape mechanism to extract gates from
    the encoding circuit, then rebuilds the circuit with noise channels
    inserted after each gate.
    """
    import pennylane as qml

    n_qubits = encoding.n_qubits
    single_qubit_noise = noise_params.get("single_qubit", 0.0)
    two_qubit_noise = noise_params.get("two_qubit", 0.0)

    # Create the mixed-state device
    dev = create_noisy_device(n_qubits)

    # Get the circuit operations from the encoding
    circuit_fn = encoding.get_circuit(x, backend="pennylane")

    if not callable(circuit_fn):
        raise ValueError(
            f"Encoding returned non-callable circuit: {type(circuit_fn).__name__}"
        )

    @qml.qnode(dev)
    def noisy_circuit():
        """Execute circuit with noise injection."""
        # Record the operations from the original circuit
        with qml.queuing.AnnotatedQueue() as q:
            circuit_fn()

        tape = qml.tape.QuantumScript.from_queue(q)

        # Replay operations with noise channels
        for op in tape.operations:
            # Apply the original operation
            qml.apply(op)

            # Insert depolarizing noise based on gate type
            wires = list(op.wires)
            if len(wires) == 1:
                # Single-qubit gate
                apply_depolarizing_noise(wires[0], single_qubit_noise)
            elif len(wires) == 2:
                # Two-qubit gate
                apply_two_qubit_depolarizing_noise(
                    wires[0], wires[1], two_qubit_noise
                )
            elif len(wires) > 2:
                # Multi-qubit gate: apply noise to each qubit
                for wire in wires:
                    apply_depolarizing_noise(wire, two_qubit_noise)

        return qml.density_matrix(wires=range(n_qubits))

    # Execute and return density matrix
    rho = noisy_circuit()
    return np.array(rho, dtype=np.complex128)


def execute_ideal_circuit(
    encoding,
    x: NDArray[np.floating[Any]],
) -> NDArray[np.complexfloating[Any, Any]]:
    """Execute an encoding circuit without noise and return the statevector.

    This is a convenience wrapper that uses the standard PennyLane
    default.qubit simulator for pure-state simulation.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to simulate.
    x : NDArray[np.floating]
        Input feature vector of shape (n_features,).

    Returns
    -------
    NDArray[np.complexfloating]
        Statevector of shape (2^n_qubits,).
    """
    import pennylane as qml

    n_qubits = encoding.n_qubits

    # Create the pure-state device
    dev = qml.device("default.qubit", wires=n_qubits)

    # Get the circuit operations from the encoding
    circuit_fn = encoding.get_circuit(x, backend="pennylane")

    if not callable(circuit_fn):
        raise ValueError(
            f"Encoding returned non-callable circuit: {type(circuit_fn).__name__}"
        )

    @qml.qnode(dev)
    def ideal_circuit():
        circuit_fn()
        return qml.state()

    statevector = ideal_circuit()
    return np.array(statevector, dtype=np.complex128)
