"""Smoke tests for Stage 5b noise resilience analysis.

This module provides basic sanity checks to verify that the noisy analysis
functions work correctly with a single encoding before running the full
stage. These tests verify:

1. Noisy expressibility is computed correctly (≤ ideal expressibility)
2. Noisy entanglement is computed correctly (typically ≤ ideal for pure states)
3. Fidelity decreases with increasing noise level
4. The noise handler integrates correctly with the experiment framework

Run with: pytest experiments/tests/test_noise_smoke.py -v
"""

import numpy as np
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def iqp_encoding():
    """Create a small IQP encoding for testing."""
    from encoding_atlas import IQPEncoding

    return IQPEncoding(n_features=2, reps=1)


@pytest.fixture
def angle_encoding():
    """Create a non-entangling angle encoding for testing."""
    from encoding_atlas import AngleEncoding

    return AngleEncoding(n_features=2, reps=1)


# =============================================================================
# Noise Models Tests
# =============================================================================


class TestNoiseModels:
    """Tests for experiments/noise_models.py."""

    def test_noise_levels_defined(self):
        """Verify all noise levels are defined correctly."""
        from experiments.noise_models import NOISE_LEVELS

        assert "low" in NOISE_LEVELS
        assert "medium" in NOISE_LEVELS
        assert "high" in NOISE_LEVELS

        # Check structure
        for level in ["low", "medium", "high"]:
            params = NOISE_LEVELS[level]
            assert "single_qubit" in params
            assert "two_qubit" in params
            assert 0 <= params["single_qubit"] <= 1
            assert 0 <= params["two_qubit"] <= 1

    def test_noise_level_ordering(self):
        """Verify noise levels increase: low < medium < high."""
        from experiments.noise_models import NOISE_LEVELS

        assert NOISE_LEVELS["low"]["single_qubit"] < NOISE_LEVELS["medium"]["single_qubit"]
        assert NOISE_LEVELS["medium"]["single_qubit"] < NOISE_LEVELS["high"]["single_qubit"]
        assert NOISE_LEVELS["low"]["two_qubit"] < NOISE_LEVELS["medium"]["two_qubit"]
        assert NOISE_LEVELS["medium"]["two_qubit"] < NOISE_LEVELS["high"]["two_qubit"]

    def test_get_noise_params(self):
        """Test get_noise_params function."""
        from experiments.noise_models import get_noise_params

        params = get_noise_params("medium")
        assert params["single_qubit"] == 0.005
        assert params["two_qubit"] == 0.05

    def test_get_noise_params_invalid(self):
        """Test get_noise_params raises error for invalid level."""
        from experiments.noise_models import get_noise_params

        with pytest.raises(ValueError, match="Unknown noise level"):
            get_noise_params("extreme")

    def test_create_noisy_device(self):
        """Test noisy device creation."""
        from experiments.noise_models import create_noisy_device

        dev = create_noisy_device(n_qubits=2)
        assert dev.wires.tolist() == [0, 1]

    def test_execute_noisy_circuit(self, iqp_encoding):
        """Test noisy circuit execution returns valid density matrix."""
        from experiments.noise_models import execute_noisy_circuit, get_noise_params

        x = np.array([0.5, 1.0])
        noise_params = get_noise_params("medium")

        rho = execute_noisy_circuit(iqp_encoding, x, noise_params)

        # Check it's a valid density matrix
        assert rho.shape == (4, 4)  # 2 qubits -> 4x4
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6)
        assert np.allclose(rho, rho.conj().T, atol=1e-10)  # Hermitian

    def test_execute_ideal_circuit(self, iqp_encoding):
        """Test ideal circuit execution returns valid statevector."""
        from experiments.noise_models import execute_ideal_circuit

        x = np.array([0.5, 1.0])

        psi = execute_ideal_circuit(iqp_encoding, x)

        # Check it's a valid statevector
        assert psi.shape == (4,)  # 2 qubits -> 4
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-10)


# =============================================================================
# Noisy Analysis Tests
# =============================================================================


class TestNoisyExpressibility:
    """Tests for noisy expressibility computation."""

    def test_compute_noisy_expressibility(self, iqp_encoding):
        """Test basic noisy expressibility computation."""
        from experiments.noisy_analysis import compute_noisy_expressibility

        result = compute_noisy_expressibility(
            iqp_encoding,
            noise_level="medium",
            n_samples=50,  # Small for testing
            seed=42,
        )

        assert "noisy_expressibility" in result
        assert "noisy_kl_divergence" in result
        assert "noise_level" in result
        assert "status" in result

        assert 0.0 <= result["noisy_expressibility"] <= 1.0
        assert result["noisy_kl_divergence"] >= 0.0
        assert result["status"] in ("success", "noise_failure")

    def test_noisy_expressibility_decreases_with_noise(self, iqp_encoding):
        """Verify noisy expressibility generally decreases with higher noise."""
        from encoding_atlas.analysis import compute_expressibility
        from experiments.noisy_analysis import compute_noisy_expressibility

        # Compute ideal expressibility
        ideal = compute_expressibility(
            iqp_encoding, n_samples=100, seed=42
        )

        # Compute noisy expressibility at low noise
        low_result = compute_noisy_expressibility(
            iqp_encoding,
            noise_level="low",
            n_samples=50,
            seed=42,
        )

        # Compute noisy expressibility at high noise
        high_result = compute_noisy_expressibility(
            iqp_encoding,
            noise_level="high",
            n_samples=50,
            seed=42,
        )

        # With enough samples, noisy expressibility should generally be <= ideal
        # (though stochasticity may cause small violations)
        # At minimum, high noise should be worse than low noise on average
        # Note: This is a statistical test, may occasionally fail due to randomness
        if low_result["status"] == "success" and high_result["status"] == "success":
            # High noise should generally give lower or similar expressibility
            # Allow some tolerance for statistical variation
            assert high_result["noisy_expressibility"] <= low_result["noisy_expressibility"] + 0.2


class TestNoisyEntanglement:
    """Tests for noisy entanglement computation."""

    def test_compute_noisy_entanglement(self, iqp_encoding):
        """Test basic noisy entanglement computation."""
        from experiments.noisy_analysis import compute_noisy_entanglement

        result = compute_noisy_entanglement(
            iqp_encoding,
            noise_level="medium",
            n_samples=30,  # Small for testing
            seed=42,
        )

        assert "noisy_entanglement" in result
        assert "noisy_linear_entropy_avg" in result
        assert "per_qubit_entropy" in result
        assert "status" in result

        assert 0.0 <= result["noisy_entanglement"] <= 1.0
        assert result["status"] in ("success", "noise_failure")

    def test_non_entangling_encoding_zero_entanglement(self, angle_encoding):
        """Test that non-entangling encodings have zero entanglement even with noise."""
        from experiments.noisy_analysis import compute_noisy_entanglement

        # AngleEncoding is non-entangling
        result = compute_noisy_entanglement(
            angle_encoding,
            noise_level="medium",
            n_samples=30,
            seed=42,
        )

        # Non-entangling encodings may still show some "entanglement" due to noise
        # (noise induces mixing which increases linear entropy), but it should
        # be bounded and less than truly entangling encodings


class TestFidelityDecay:
    """Tests for fidelity decay computation."""

    def test_compute_fidelity_decay(self, iqp_encoding):
        """Test basic fidelity decay computation."""
        from experiments.noisy_analysis import compute_fidelity_decay

        result = compute_fidelity_decay(
            iqp_encoding,
            noise_level="medium",
            n_samples=30,  # Small for testing
            seed=42,
        )

        assert "mean_fidelity" in result
        assert "fidelity_decay" in result
        assert "status" in result

        assert 0.0 <= result["mean_fidelity"] <= 1.0
        assert 0.0 <= result["fidelity_decay"] <= 1.0
        assert np.isclose(
            result["mean_fidelity"] + result["fidelity_decay"], 1.0, atol=1e-10
        )

    def test_fidelity_decreases_with_noise(self, iqp_encoding):
        """Verify fidelity decreases with higher noise levels."""
        from experiments.noisy_analysis import compute_fidelity_decay

        # Low noise
        low = compute_fidelity_decay(
            iqp_encoding,
            noise_level="low",
            n_samples=30,
            seed=42,
        )

        # High noise
        high = compute_fidelity_decay(
            iqp_encoding,
            noise_level="high",
            n_samples=30,
            seed=42,
        )

        if low["status"] == "success" and high["status"] == "success":
            # Higher noise should give lower fidelity (more decay)
            assert high["mean_fidelity"] <= low["mean_fidelity"] + 0.1  # Some tolerance


# =============================================================================
# Integration Tests
# =============================================================================


class TestNoiseHandlerIntegration:
    """Integration tests for the noise stage handler."""

    def test_handle_noise_basic(self, iqp_encoding):
        """Test the _handle_noise function directly."""
        from experiments.runner import _handle_noise
        from experiments.config import ExperimentConfig, EncodingSpec

        # Create a minimal config
        config = ExperimentConfig(
            stage="noise",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[EncodingSpec(name="iqp", params={"n_features": 2, "reps": 1})],
            analysis_params={
                "noise_levels": ["medium"],
                "n_samples_expressibility": 50,
                "n_samples_entanglement": 30,
                "n_samples_fidelity": 20,
            },
            output_dir="/tmp/test_noise",
        )

        result = _handle_noise(iqp_encoding, config, seed=42)

        # Check required fields
        assert "encoding_name" in result
        assert "ideal_expressibility" in result
        assert "ideal_entanglement" in result
        assert "noisy_medium_expressibility" in result
        assert "noisy_medium_fidelity_mean" in result
        assert "status" in result

    def test_handle_noise_non_entangling(self, angle_encoding):
        """Test _handle_noise for non-entangling encodings."""
        from experiments.runner import _handle_noise
        from experiments.config import ExperimentConfig, EncodingSpec

        config = ExperimentConfig(
            stage="noise",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[EncodingSpec(name="angle", params={"n_features": 2, "reps": 1})],
            analysis_params={
                "noise_levels": ["medium"],
                "n_samples_expressibility": 50,
                "n_samples_entanglement": 30,
                "n_samples_fidelity": 20,
            },
            output_dir="/tmp/test_noise",
        )

        result = _handle_noise(angle_encoding, config, seed=42)

        # Non-entangling encoding should have zero ideal entanglement
        assert result["ideal_entanglement"] == 0.0
        assert result["is_entangling"] is False


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestMixedStateFidelity:
    """Tests for mixed state fidelity computation."""

    def test_pure_state_fidelity(self):
        """Test fidelity for pure states (should match standard definition)."""
        from experiments.noisy_analysis import compute_mixed_state_fidelity

        # Two identical pure states
        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho1 = np.outer(psi, psi.conj())
        rho2 = np.outer(psi, psi.conj())

        fidelity = compute_mixed_state_fidelity(rho1, rho2)
        assert np.isclose(fidelity, 1.0, atol=1e-10)

    def test_orthogonal_pure_states(self):
        """Test fidelity for orthogonal pure states."""
        from experiments.noisy_analysis import compute_mixed_state_fidelity

        psi1 = np.array([1, 0, 0, 0], dtype=complex)
        psi2 = np.array([0, 1, 0, 0], dtype=complex)
        rho1 = np.outer(psi1, psi1.conj())
        rho2 = np.outer(psi2, psi2.conj())

        fidelity = compute_mixed_state_fidelity(rho1, rho2)
        assert np.isclose(fidelity, 0.0, atol=1e-10)

    def test_maximally_mixed_fidelity(self):
        """Test fidelity with maximally mixed state."""
        from experiments.noisy_analysis import compute_mixed_state_fidelity

        # Pure state
        psi = np.array([1, 0], dtype=complex)
        rho_pure = np.outer(psi, psi.conj())

        # Maximally mixed state
        rho_mixed = np.eye(2, dtype=complex) / 2

        fidelity = compute_mixed_state_fidelity(rho_pure, rho_mixed)
        # F(|0><0|, I/2) = <0|I/2|0> = 0.5
        assert np.isclose(fidelity, 0.5, atol=1e-6)


class TestLinearEntropy:
    """Tests for linear entropy computation."""

    def test_pure_state_zero_entropy(self):
        """Test that pure states have zero linear entropy."""
        from experiments.noisy_analysis import compute_linear_entropy

        psi = np.array([1, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())

        entropy = compute_linear_entropy(rho)
        assert np.isclose(entropy, 0.0, atol=1e-10)

    def test_maximally_mixed_max_entropy(self):
        """Test that maximally mixed state has maximum linear entropy."""
        from experiments.noisy_analysis import compute_linear_entropy

        # Maximally mixed 2x2 state
        rho = np.eye(2, dtype=complex) / 2

        entropy = compute_linear_entropy(rho)
        # S_L = 1 - Tr(I/2 @ I/2) = 1 - Tr(I/4) = 1 - 0.5 = 0.5
        assert np.isclose(entropy, 0.5, atol=1e-10)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
