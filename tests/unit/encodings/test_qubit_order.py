"""Tests for the MSB->LSB amplitude permutation.

The permutation is tested against an explicit bit-reversal reference rather
than against itself, because a self-consistent-but-wrong implementation is
exactly the failure mode that shipped: ``SO2EquivariantFeatureMap`` omitted
this conversion and silently prepared a bit-reversed state on Qiskit.
"""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas.encodings._qubit_order import msb_to_lsb_amplitudes
from tests._backends import require_backend


def _reference(amplitudes: np.ndarray, n_qubits: int) -> np.ndarray:
    """Bit-reverse indices with an explicit loop — the definition, slowly."""
    out = np.empty_like(amplitudes)
    for index in range(len(amplitudes)):
        bits = format(index, f"0{n_qubits}b")
        out[int(bits[::-1], 2)] = amplitudes[index]
    return out


class TestAgainstTheDefinition:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5, 6])
    def test_matches_explicit_bit_reversal(self, n_qubits: int) -> None:
        rng = np.random.default_rng(n_qubits)
        amplitudes = rng.normal(size=2**n_qubits)
        assert np.array_equal(
            msb_to_lsb_amplitudes(amplitudes, n_qubits),
            _reference(amplitudes, n_qubits),
        )

    def test_two_qubit_case_by_hand(self) -> None:
        # Indices 1 (01) and 2 (10) swap; 0 and 3 are palindromes.
        assert np.array_equal(
            msb_to_lsb_amplitudes(np.array([0.0, 1.0, 2.0, 3.0]), 2),
            np.array([0.0, 2.0, 1.0, 3.0]),
        )

    def test_single_qubit_is_identity(self) -> None:
        amplitudes = np.array([0.6, 0.8])
        assert np.array_equal(msb_to_lsb_amplitudes(amplitudes, 1), amplitudes)


class TestProperties:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
    def test_is_an_involution(self, n_qubits: int) -> None:
        rng = np.random.default_rng(n_qubits)
        amplitudes = rng.normal(size=2**n_qubits)
        once = msb_to_lsb_amplitudes(amplitudes, n_qubits)
        assert np.array_equal(msb_to_lsb_amplitudes(once, n_qubits), amplitudes)

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_is_a_permutation(self, n_qubits: int) -> None:
        amplitudes = np.arange(2**n_qubits, dtype=np.float64)
        result = msb_to_lsb_amplitudes(amplitudes, n_qubits)
        assert sorted(result.tolist()) == sorted(amplitudes.tolist())

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_preserves_norm(self, n_qubits: int) -> None:
        rng = np.random.default_rng(0)
        amplitudes = rng.normal(size=2**n_qubits)
        amplitudes /= np.linalg.norm(amplitudes)
        assert np.linalg.norm(msb_to_lsb_amplitudes(amplitudes, n_qubits)) == (
            pytest.approx(1.0)
        )

    def test_supports_complex_amplitudes(self) -> None:
        amplitudes = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
        assert np.array_equal(
            msb_to_lsb_amplitudes(amplitudes, 2),
            np.array([1 + 2j, 5 + 6j, 3 + 4j, 7 + 8j]),
        )

    def test_does_not_alias_or_mutate_the_input(self) -> None:
        amplitudes = np.array([0.0, 1.0, 2.0, 3.0])
        original = amplitudes.copy()
        result = msb_to_lsb_amplitudes(amplitudes, 2)
        result[0] = 99.0
        assert np.array_equal(amplitudes, original)

    def test_returns_contiguous_array(self) -> None:
        """``QuantumCircuit.initialize`` is happier with contiguous input."""
        result = msb_to_lsb_amplitudes(np.arange(8.0), 3)
        assert result.flags["C_CONTIGUOUS"]


class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_bad_qubit_count_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            msb_to_lsb_amplitudes(np.arange(4.0), bad)  # type: ignore[arg-type]

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 1D"):
            msb_to_lsb_amplitudes(np.zeros((2, 2)), 2)

    @pytest.mark.parametrize("length", [3, 5, 6, 7])
    def test_wrong_length_raises(self, length: int) -> None:
        with pytest.raises(ValueError, match="entries"):
            msb_to_lsb_amplitudes(np.zeros(length), 2)


class TestUsedWhereItMatters:
    """Both amplitude-vector state preparations must apply the permutation."""

    def test_amplitude_encoding_round_trips_through_qiskit(self) -> None:
        require_backend(
            "qiskit", reason="AmplitudeEncoding's MSB->LSB round trip did not run"
        )
        from encoding_atlas import AmplitudeEncoding
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        encoding = AmplitudeEncoding(n_features=4, normalize=True)
        x = np.array([0.1, 0.5, -0.3, 0.8])
        reference = simulate_encoding_statevector(encoding, x, backend="pennylane")
        qiskit_state = simulate_encoding_statevector(encoding, x, backend="qiskit")
        assert abs(np.vdot(reference, qiskit_state)) ** 2 == pytest.approx(1.0)

    def test_so2_round_trips_through_qiskit(self) -> None:
        require_backend("qiskit", reason="the SO(2) MSB->LSB round trip did not run")
        from encoding_atlas import SO2EquivariantFeatureMap
        from encoding_atlas.analysis._utils import simulate_encoding_statevector

        encoding = SO2EquivariantFeatureMap(n_features=2, max_angular_momentum=2)
        x = np.array([0.3, 0.7])
        reference = simulate_encoding_statevector(encoding, x, backend="pennylane")
        qiskit_state = simulate_encoding_statevector(encoding, x, backend="qiskit")
        assert abs(np.vdot(reference, qiskit_state)) ** 2 == pytest.approx(1.0)
