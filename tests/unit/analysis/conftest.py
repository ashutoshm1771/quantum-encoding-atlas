"""Pytest fixtures for analysis module tests.

This module provides shared fixtures for testing the analysis functions
in encoding_atlas.analysis. It includes:

- Standard test states (computational basis, Bell states, GHZ states)
- Sample encodings for testing
- Random parameter generators
- Backend availability checkers

Usage
-----
These fixtures are automatically available to all tests in the
tests/unit/analysis/ directory.

>>> def test_something(sample_encoding_2q, bell_state):
...     # Fixtures are injected automatically by pytest
...     pass
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding


# =============================================================================
# Backend Availability Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pennylane_available() -> bool:
    """Check if PennyLane is available for testing."""
    try:
        import pennylane as qml
        # Quick check that it actually works
        dev = qml.device("default.qubit", wires=1)
        return True
    except (ImportError, Exception):
        return False


@pytest.fixture(scope="session")
def qiskit_available() -> bool:
    """Check if Qiskit is available for testing."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        # Quick check that it actually works
        qc = QuantumCircuit(1)
        Statevector(qc)
        return True
    except (ImportError, Exception):
        return False


@pytest.fixture(scope="session")
def cirq_available() -> bool:
    """Check if Cirq is available for testing."""
    try:
        import cirq
        # Quick check that it actually works
        qubit = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(qubit))
        simulator = cirq.Simulator()
        simulator.simulate(circuit)
        return True
    except (ImportError, Exception):
        return False


@pytest.fixture
def skip_if_no_pennylane(pennylane_available: bool):
    """Skip test if PennyLane is not available."""
    if not pennylane_available:
        pytest.skip("PennyLane not available")


@pytest.fixture
def skip_if_no_qiskit(qiskit_available: bool):
    """Skip test if Qiskit is not available."""
    if not qiskit_available:
        pytest.skip("Qiskit not available")


@pytest.fixture
def skip_if_no_cirq(cirq_available: bool):
    """Skip test if Cirq is not available."""
    if not cirq_available:
        pytest.skip("Cirq not available")


# =============================================================================
# Standard Quantum State Fixtures
# =============================================================================


@pytest.fixture
def zero_state_1q() -> NDArray[np.complexfloating]:
    """Single-qubit |0⟩ state."""
    return np.array([1.0, 0.0], dtype=np.complex128)


@pytest.fixture
def one_state_1q() -> NDArray[np.complexfloating]:
    """Single-qubit |1⟩ state."""
    return np.array([0.0, 1.0], dtype=np.complex128)


@pytest.fixture
def plus_state_1q() -> NDArray[np.complexfloating]:
    """Single-qubit |+⟩ = (|0⟩ + |1⟩)/√2 state."""
    return np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)


@pytest.fixture
def minus_state_1q() -> NDArray[np.complexfloating]:
    """Single-qubit |-⟩ = (|0⟩ - |1⟩)/√2 state."""
    return np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2)


@pytest.fixture
def zero_state_2q() -> NDArray[np.complexfloating]:
    """Two-qubit |00⟩ state."""
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0
    return state


@pytest.fixture
def bell_state() -> NDArray[np.complexfloating]:
    """Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2."""
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)  # |00⟩
    state[3] = 1.0 / np.sqrt(2)  # |11⟩
    return state


@pytest.fixture
def bell_state_psi_plus() -> NDArray[np.complexfloating]:
    """Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2."""
    state = np.zeros(4, dtype=np.complex128)
    state[1] = 1.0 / np.sqrt(2)  # |01⟩
    state[2] = 1.0 / np.sqrt(2)  # |10⟩
    return state


@pytest.fixture
def ghz_state_3q() -> NDArray[np.complexfloating]:
    """Three-qubit GHZ state (|000⟩ + |111⟩)/√2."""
    state = np.zeros(8, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)  # |000⟩
    state[7] = 1.0 / np.sqrt(2)  # |111⟩
    return state


@pytest.fixture
def w_state_3q() -> NDArray[np.complexfloating]:
    """Three-qubit W state (|001⟩ + |010⟩ + |100⟩)/√3."""
    state = np.zeros(8, dtype=np.complex128)
    state[1] = 1.0 / np.sqrt(3)  # |001⟩
    state[2] = 1.0 / np.sqrt(3)  # |010⟩
    state[4] = 1.0 / np.sqrt(3)  # |100⟩
    return state


@pytest.fixture
def product_state_2q() -> NDArray[np.complexfloating]:
    """Two-qubit product state |+⟩⊗|0⟩."""
    # |+⟩ = (|0⟩ + |1⟩)/√2
    # |+⟩⊗|0⟩ = (|00⟩ + |10⟩)/√2
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)  # |00⟩
    state[2] = 1.0 / np.sqrt(2)  # |10⟩
    return state


@pytest.fixture
def maximally_mixed_1q() -> NDArray[np.complexfloating]:
    """Maximally mixed single-qubit density matrix I/2."""
    return np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)


@pytest.fixture
def maximally_mixed_2q() -> NDArray[np.complexfloating]:
    """Maximally mixed two-qubit density matrix I/4."""
    return np.eye(4, dtype=np.complex128) / 4.0


@pytest.fixture
def pure_density_matrix_1q() -> NDArray[np.complexfloating]:
    """Pure single-qubit density matrix |0⟩⟨0|."""
    return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


# =============================================================================
# Sample Encoding Fixtures
# =============================================================================


@pytest.fixture
def sample_encoding_2q():
    """Sample AngleEncoding with 2 qubits/features."""
    from encoding_atlas import AngleEncoding
    return AngleEncoding(n_features=2)


@pytest.fixture
def sample_encoding_4q():
    """Sample AngleEncoding with 4 qubits/features."""
    from encoding_atlas import AngleEncoding
    return AngleEncoding(n_features=4)


@pytest.fixture
def entangling_encoding_4q():
    """Sample IQPEncoding with 4 qubits (entangling)."""
    from encoding_atlas import IQPEncoding
    return IQPEncoding(n_features=4, reps=1)


@pytest.fixture
def sample_encoding_factory():
    """Factory fixture for creating encodings with specific parameters."""
    def _create_encoding(encoding_type: str, n_features: int, **kwargs):
        if encoding_type == "angle":
            from encoding_atlas import AngleEncoding
            return AngleEncoding(n_features=n_features, **kwargs)
        elif encoding_type == "iqp":
            from encoding_atlas import IQPEncoding
            return IQPEncoding(n_features=n_features, **kwargs)
        elif encoding_type == "amplitude":
            from encoding_atlas import AmplitudeEncoding
            return AmplitudeEncoding(n_features=n_features, **kwargs)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    return _create_encoding


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_features_2d() -> NDArray[np.floating]:
    """Sample 2D feature vector."""
    return np.array([0.5, 1.0], dtype=np.float64)


@pytest.fixture
def sample_features_4d() -> NDArray[np.floating]:
    """Sample 4D feature vector."""
    return np.array([0.1, 0.5, 1.0, 1.5], dtype=np.float64)


@pytest.fixture
def sample_features_batch_2d() -> NDArray[np.floating]:
    """Batch of 2D feature vectors."""
    return np.array([
        [0.1, 0.2],
        [0.5, 1.0],
        [1.5, 2.0],
    ], dtype=np.float64)


@pytest.fixture
def sample_features_batch_4d() -> NDArray[np.floating]:
    """Batch of 4D feature vectors."""
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [1.0, 1.2, 1.4, 1.6],
    ], dtype=np.float64)


# =============================================================================
# Random Generator Fixtures
# =============================================================================


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def random_features_generator(seeded_rng: np.random.Generator):
    """Generator for random feature vectors."""
    def _generate(n_features: int, n_samples: int = 1) -> NDArray[np.floating]:
        if n_samples == 1:
            return seeded_rng.uniform(0, 2 * np.pi, size=n_features).astype(np.float64)
        return seeded_rng.uniform(0, 2 * np.pi, size=(n_samples, n_features)).astype(np.float64)
    return _generate


@pytest.fixture
def random_statevector_generator(seeded_rng: np.random.Generator):
    """Generator for random normalized statevectors."""
    def _generate(n_qubits: int) -> NDArray[np.complexfloating]:
        dim = 2**n_qubits
        # Random complex amplitudes
        real_parts = seeded_rng.standard_normal(dim)
        imag_parts = seeded_rng.standard_normal(dim)
        state = real_parts + 1j * imag_parts
        # Normalize
        state = state / np.linalg.norm(state)
        return state.astype(np.complex128)
    return _generate


# =============================================================================
# Helper Fixtures for Testing Edge Cases
# =============================================================================


@pytest.fixture
def unnormalized_state() -> NDArray[np.complexfloating]:
    """Unnormalized statevector for testing normalization."""
    return np.array([2.0, 0.0, 0.0, 0.0], dtype=np.complex128)


@pytest.fixture
def state_with_nan() -> NDArray[np.complexfloating]:
    """Statevector with NaN value for testing validation."""
    return np.array([np.nan, 0.0, 0.0, 1.0], dtype=np.complex128)


@pytest.fixture
def state_with_inf() -> NDArray[np.complexfloating]:
    """Statevector with infinite value for testing validation."""
    return np.array([np.inf, 0.0, 0.0, 0.0], dtype=np.complex128)


@pytest.fixture
def zero_norm_state() -> NDArray[np.complexfloating]:
    """Zero-norm statevector for testing edge cases."""
    return np.zeros(4, dtype=np.complex128)


@pytest.fixture
def features_with_nan() -> NDArray[np.floating]:
    """Feature vector with NaN for testing validation."""
    return np.array([0.5, np.nan, 1.0], dtype=np.float64)


@pytest.fixture
def features_with_inf() -> NDArray[np.floating]:
    """Feature vector with infinity for testing validation."""
    return np.array([0.5, np.inf, 1.0], dtype=np.float64)


# =============================================================================
# Parametrized Test Data
# =============================================================================


# Common test dimensions
QUBIT_COUNTS = [1, 2, 3, 4]
FEATURE_COUNTS = [2, 3, 4, 8]

# Backends for parametrized tests
BACKENDS = ["pennylane", "qiskit", "cirq"]


@pytest.fixture(params=QUBIT_COUNTS)
def n_qubits(request) -> int:
    """Parametrized number of qubits for tests."""
    return request.param


@pytest.fixture(params=FEATURE_COUNTS)
def n_features(request) -> int:
    """Parametrized number of features for tests."""
    return request.param


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    """Parametrized backend for tests."""
    return request.param
