"""Quantum tomography tests for encodings.

This test suite mathematically verifies that encoded quantum states
match their expected theoretical form for small systems where
analytical solutions are tractable.

Test Categories:
1. Angle encoding tomography (RX, RY, RZ)
2. Amplitude encoding tomography
3. IQP encoding tomography
4. Multi-qubit state verification
5. Special angle verification (0, π/4, π/2, π)
"""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import (
    AngleEncoding,
    AmplitudeEncoding,
    IQPEncoding,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def qiskit_statevector():
    """Import Qiskit Statevector."""
    qiskit_quantum_info = pytest.importorskip("qiskit.quantum_info")
    return qiskit_quantum_info.Statevector


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ry_state(theta: float) -> np.ndarray:
    """Compute RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩."""
    return np.array([np.cos(theta / 2), np.sin(theta / 2)])


def rx_state(theta: float) -> np.ndarray:
    """Compute RX(θ)|0⟩ = cos(θ/2)|0⟩ - i*sin(θ/2)|1⟩."""
    return np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)])


def rz_state(theta: float) -> np.ndarray:
    """Compute RZ(θ)|0⟩ = e^(-iθ/2)|0⟩."""
    return np.array([np.exp(-1j * theta / 2), 0])


def tensor_product(*states) -> np.ndarray:
    """Compute tensor product of multiple states.

    Uses Qiskit's little-endian ordering where q[0] is the rightmost qubit.
    For states [q0, q1, q2, ...], computes |...q2 q1 q0⟩ (reverse order).
    """
    # Reverse the order to match Qiskit's little-endian convention
    reversed_states = states[::-1]
    result = reversed_states[0]
    for state in reversed_states[1:]:
        result = np.kron(result, state)
    return result


def states_equivalent(state1: np.ndarray, state2: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if two states are equivalent up to global phase."""
    # Find first non-zero element to determine phase
    for i in range(len(state1)):
        if np.abs(state1[i]) > atol and np.abs(state2[i]) > atol:
            phase = state1[i] / state2[i]
            phase = phase / np.abs(phase)  # Normalize to unit phase
            return np.allclose(state1, phase * state2, atol=atol)

    # If we get here, compare directly
    return np.allclose(np.abs(state1), np.abs(state2), atol=atol)


# =============================================================================
# ANGLE ENCODING TOMOGRAPHY - RY ROTATION
# =============================================================================


class TestAngleEncodingRYTomography:
    """Verify RY angle encoding matches expected state."""

    def test_ry_single_qubit_pi_over_4(self, qiskit_statevector) -> None:
        """Test RY(π/4)|0⟩ state."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([np.pi / 4])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        expected = ry_state(np.pi / 4)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_ry_single_qubit_pi_over_2(self, qiskit_statevector) -> None:
        """Test RY(π/2)|0⟩ = (|0⟩ + |1⟩)/√2."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([np.pi / 2])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # RY(π/2)|0⟩ = cos(π/4)|0⟩ + sin(π/4)|1⟩ = (|0⟩ + |1⟩)/√2
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_ry_single_qubit_pi(self, qiskit_statevector) -> None:
        """Test RY(π)|0⟩ = |1⟩."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([np.pi])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # RY(π)|0⟩ = cos(π/2)|0⟩ + sin(π/2)|1⟩ = |1⟩
        expected = np.array([0, 1])

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_ry_single_qubit_zero(self, qiskit_statevector) -> None:
        """Test RY(0)|0⟩ = |0⟩."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([0.0])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        expected = np.array([1, 0])

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_ry_two_qubit_tomography(self, qiskit_statevector) -> None:
        """Test RY(π/4)|0⟩ ⊗ RY(π/2)|0⟩."""
        enc = AngleEncoding(n_features=2, rotation="Y")
        x = np.array([np.pi / 4, np.pi / 2])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # Expected: RY(π/4)|0⟩ ⊗ RY(π/2)|0⟩
        q0 = ry_state(np.pi / 4)
        q1 = ry_state(np.pi / 2)
        expected = tensor_product(q0, q1)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_ry_three_qubit_tomography(self, qiskit_statevector) -> None:
        """Test three-qubit RY encoding."""
        enc = AngleEncoding(n_features=3, rotation="Y")
        x = np.array([np.pi / 6, np.pi / 4, np.pi / 3])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        q0 = ry_state(np.pi / 6)
        q1 = ry_state(np.pi / 4)
        q2 = ry_state(np.pi / 3)
        expected = tensor_product(q0, q1, q2)

        np.testing.assert_allclose(actual, expected, atol=1e-10)


# =============================================================================
# ANGLE ENCODING TOMOGRAPHY - RX ROTATION
# =============================================================================


class TestAngleEncodingRXTomography:
    """Verify RX angle encoding matches expected state."""

    def test_rx_single_qubit_pi_over_2(self, qiskit_statevector) -> None:
        """Test RX(π/2)|0⟩ = (|0⟩ - i|1⟩)/√2."""
        enc = AngleEncoding(n_features=1, rotation="X")
        x = np.array([np.pi / 2])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        expected = rx_state(np.pi / 2)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_rx_single_qubit_pi(self, qiskit_statevector) -> None:
        """Test RX(π)|0⟩ = -i|1⟩."""
        enc = AngleEncoding(n_features=1, rotation="X")
        x = np.array([np.pi])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # RX(π)|0⟩ = cos(π/2)|0⟩ - i*sin(π/2)|1⟩ = -i|1⟩
        expected = np.array([0, -1j])

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_rx_two_qubit_tomography(self, qiskit_statevector) -> None:
        """Test two-qubit RX encoding."""
        enc = AngleEncoding(n_features=2, rotation="X")
        x = np.array([np.pi / 2, np.pi])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        q0 = rx_state(np.pi / 2)
        q1 = rx_state(np.pi)
        expected = tensor_product(q0, q1)

        np.testing.assert_allclose(actual, expected, atol=1e-10)


# =============================================================================
# ANGLE ENCODING TOMOGRAPHY - RZ ROTATION
# =============================================================================


class TestAngleEncodingRZTomography:
    """Verify RZ angle encoding matches expected state."""

    def test_rz_single_qubit_pi_over_4(self, qiskit_statevector) -> None:
        """Test RZ(π/4)|0⟩ state."""
        enc = AngleEncoding(n_features=1, rotation="Z")
        x = np.array([np.pi / 4])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        expected = rz_state(np.pi / 4)

        # Compare up to global phase
        assert states_equivalent(actual, expected, atol=1e-10)

    def test_rz_only_adds_phase(self, qiskit_statevector) -> None:
        """Test that RZ on |0⟩ only adds global phase, probability stays at |0⟩."""
        enc = AngleEncoding(n_features=1, rotation="Z")

        for angle in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            x = np.array([angle])
            circuit = enc.get_circuit(x, backend="qiskit")
            actual = qiskit_statevector(circuit).data

            # Probability should be entirely in |0⟩
            prob_0 = np.abs(actual[0]) ** 2
            prob_1 = np.abs(actual[1]) ** 2

            assert np.isclose(prob_0, 1.0, atol=1e-10), f"P(0) = {prob_0} for angle {angle}"
            assert np.isclose(prob_1, 0.0, atol=1e-10), f"P(1) = {prob_1} for angle {angle}"


# =============================================================================
# AMPLITUDE ENCODING TOMOGRAPHY
# =============================================================================


class TestAmplitudeEncodingTomography:
    """Verify amplitude encoding creates correct superposition."""

    def test_amplitude_encoding_uniform(self, qiskit_statevector) -> None:
        """Test uniform amplitude encoding: [1,1,1,1] → |++⟩."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1.0, 1.0, 1.0, 1.0])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # Expected: uniform superposition, all amplitudes = 1/2
        expected_amplitudes = np.array([0.5, 0.5, 0.5, 0.5])

        np.testing.assert_allclose(np.abs(actual[:4]), expected_amplitudes, atol=1e-10)

    def test_amplitude_encoding_single_basis(self, qiskit_statevector) -> None:
        """Test amplitude encoding with single basis state: [1,0,0,0] → |00⟩."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1.0, 0.0, 0.0, 0.0])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # Should have amplitude 1 at |00⟩
        assert np.isclose(np.abs(actual[0]), 1.0, atol=1e-10)
        assert np.allclose(np.abs(actual[1:4]), 0.0, atol=1e-10)

    def test_amplitude_encoding_normalized_amplitudes(self, qiskit_statevector) -> None:
        """Test that amplitudes are correctly normalized."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([1.0, 2.0, 3.0, 4.0])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # Expected: normalized amplitudes
        expected = x / np.linalg.norm(x)

        # Amplitude encoding puts values in first n_features amplitudes
        np.testing.assert_allclose(np.abs(actual[:4]), np.abs(expected), atol=1e-10)

    def test_amplitude_encoding_state_normalized(self, qiskit_statevector) -> None:
        """Test that output state is always normalized."""
        enc = AmplitudeEncoding(n_features=4, normalize=True)

        test_cases = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        ]

        for x in test_cases:
            circuit = enc.get_circuit(x, backend="qiskit")
            actual = qiskit_statevector(circuit).data
            norm = np.sum(np.abs(actual) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized for x={x}"


# =============================================================================
# IQP ENCODING TOMOGRAPHY
# =============================================================================


class TestIQPEncodingTomography:
    """Verify IQP encoding structure."""

    def test_iqp_hadamard_layer_zero_input(self, qiskit_statevector) -> None:
        """Test IQP with zero input: should be uniform superposition."""
        enc = IQPEncoding(n_features=2, reps=1)
        x = np.array([0.0, 0.0])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # With zero rotations, should be uniform superposition
        # |++⟩ = 1/2 * (|00⟩ + |01⟩ + |10⟩ + |11⟩)
        expected_probs = np.array([0.25, 0.25, 0.25, 0.25])
        actual_probs = np.abs(actual) ** 2

        np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-10)

    def test_iqp_state_normalized(self, qiskit_statevector) -> None:
        """Test that IQP always produces normalized state."""
        enc = IQPEncoding(n_features=4, reps=2)

        test_cases = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([np.pi, np.pi, np.pi, np.pi]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([-1.0, 2.0, -3.0, 4.0]),
        ]

        for x in test_cases:
            circuit = enc.get_circuit(x, backend="qiskit")
            actual = qiskit_statevector(circuit).data
            norm = np.sum(np.abs(actual) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10), f"State not normalized for x={x}"


# =============================================================================
# SPECIAL ANGLES VERIFICATION
# =============================================================================


class TestSpecialAngles:
    """Verify behavior at special angle values."""

    @pytest.mark.parametrize(
        "angle,expected_prob_0",
        [
            (0, 1.0),              # |0⟩
            (np.pi / 2, 0.5),      # Equal superposition
            (np.pi, 0.0),          # |1⟩
            (3 * np.pi / 2, 0.5),  # Equal superposition (opposite phase)
            (2 * np.pi, 1.0),      # Back to |0⟩
        ],
    )
    def test_ry_special_angles(
        self, qiskit_statevector, angle: float, expected_prob_0: float
    ) -> None:
        """Test RY at special angles produces expected probabilities."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([angle])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        prob_0 = np.abs(actual[0]) ** 2

        assert np.isclose(prob_0, expected_prob_0, atol=1e-10), (
            f"P(0) = {prob_0}, expected {expected_prob_0} for angle {angle}"
        )

    def test_hadamard_angle(self, qiskit_statevector) -> None:
        """Test RY(π/2) produces Hadamard-like state."""
        enc = AngleEncoding(n_features=1, rotation="Y")
        x = np.array([np.pi / 2])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # RY(π/2)|0⟩ = (|0⟩ + |1⟩)/√2 (like H|0⟩ but real)
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

        np.testing.assert_allclose(actual, expected, atol=1e-10)


# =============================================================================
# MULTI-QUBIT STATE STRUCTURE
# =============================================================================


class TestMultiQubitStateStructure:
    """Verify structure of multi-qubit encoded states."""

    def test_product_state_structure(self, qiskit_statevector) -> None:
        """Verify AngleEncoding produces product states (no entanglement)."""
        enc = AngleEncoding(n_features=2, rotation="Y")
        x = np.array([np.pi / 4, np.pi / 3])

        circuit = enc.get_circuit(x, backend="qiskit")
        actual = qiskit_statevector(circuit).data

        # For product state |ψ₁⟩⊗|ψ₂⟩, we can factor the state
        q0 = ry_state(np.pi / 4)
        q1 = ry_state(np.pi / 3)
        expected = tensor_product(q0, q1)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_state_dimension_matches_qubits(self, qiskit_statevector) -> None:
        """Verify state dimension is 2^n_qubits."""
        for n in [1, 2, 3, 4]:
            enc = AngleEncoding(n_features=n, rotation="Y")
            x = np.random.rand(n)

            circuit = enc.get_circuit(x, backend="qiskit")
            actual = qiskit_statevector(circuit).data

            expected_dim = 2 ** n
            assert len(actual) == expected_dim, (
                f"State dimension {len(actual)} != 2^{n} = {expected_dim}"
            )

    def test_all_amplitudes_in_valid_range(self, qiskit_statevector) -> None:
        """Verify all amplitudes have magnitude ≤ 1."""
        enc = AngleEncoding(n_features=4, rotation="Y")

        for _ in range(10):
            x = np.random.uniform(-10, 10, size=4)
            circuit = enc.get_circuit(x, backend="qiskit")
            actual = qiskit_statevector(circuit).data

            magnitudes = np.abs(actual)
            assert np.all(magnitudes <= 1.0 + 1e-10), (
                f"Amplitude magnitude > 1: max = {np.max(magnitudes)}"
            )
