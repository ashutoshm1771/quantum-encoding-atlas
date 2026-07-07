"""Tests for noise-resilience analysis (depolarizing noise model).

Asserts verifiable invariants — zero noise gives unit fidelity, resilience
decreases monotonically with the noise level, and entangling encodings decay far
more than non-entangling ones (the benchmark's noise result) — plus density-
matrix validity and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import (
    AngleEncoding,
    HigherOrderAngleEncoding,
    IQPEncoding,
    ZZFeatureMap,
)
from encoding_atlas.analysis.noise import (
    NOISE_LEVELS,
    NoiseResilienceResult,
    compute_noise_resilience,
    simulate_noisy_density_matrix,
)


class TestInvariants:
    def test_zero_noise_is_unit_fidelity(self) -> None:
        r = compute_noise_resilience(
            AngleEncoding(n_features=2),
            noise_params={"single_qubit": 0.0, "two_qubit": 0.0},
            n_samples=3,
            seed=0,
        )
        assert r.retained_fidelity == pytest.approx(1.0)
        assert r.fidelity_decay == pytest.approx(0.0)

    def test_monotonic_decay_with_noise_level(self) -> None:
        enc = AngleEncoding(n_features=3)
        fids = {
            lvl: compute_noise_resilience(
                enc, noise_level=lvl, n_samples=4, seed=0
            ).retained_fidelity
            for lvl in ("low", "medium", "high")
        }
        assert fids["low"] > fids["medium"] > fids["high"]

    def test_retained_fidelity_in_unit_interval(self) -> None:
        r = compute_noise_resilience(
            IQPEncoding(n_features=3, reps=2), noise_level="high", n_samples=4, seed=0
        )
        assert 0.0 <= r.retained_fidelity <= 1.0
        assert 0.0 <= r.fidelity_decay <= 1.0
        assert r.fidelity_decay == pytest.approx(1.0 - r.retained_fidelity)

    def test_entangling_decays_more_than_non_entangling(self) -> None:
        """The benchmark's noise finding: entangling encodings (IQP, ZZ) lose
        far more fidelity than non-entangling ones (angle, higher-order angle)."""
        non_ent = min(
            compute_noise_resilience(
                enc, noise_level="medium", n_samples=4, seed=0
            ).retained_fidelity
            for enc in (
                AngleEncoding(n_features=4),
                HigherOrderAngleEncoding(n_features=4, order=2),
            )
        )
        entangling = max(
            compute_noise_resilience(
                enc, noise_level="medium", n_samples=4, seed=0
            ).retained_fidelity
            for enc in (
                IQPEncoding(n_features=4, reps=2),
                ZZFeatureMap(n_features=4, reps=2),
            )
        )
        assert non_ent > entangling

    def test_determinism_same_seed(self) -> None:
        enc = AngleEncoding(n_features=2)
        a = compute_noise_resilience(enc, n_samples=4, seed=7)
        b = compute_noise_resilience(enc, n_samples=4, seed=7)
        assert a.retained_fidelity == b.retained_fidelity


class TestResultObject:
    def test_fields(self) -> None:
        r = compute_noise_resilience(
            AngleEncoding(n_features=2), noise_level="medium", n_samples=5, seed=0
        )
        assert isinstance(r, NoiseResilienceResult)
        assert r.noise_level == "medium"
        assert r.single_qubit_error == NOISE_LEVELS["medium"]["single_qubit"]
        assert r.two_qubit_error == NOISE_LEVELS["medium"]["two_qubit"]
        assert r.n_samples == 5
        assert r.min_fidelity <= r.retained_fidelity <= r.max_fidelity
        assert r.std_fidelity >= 0.0

    def test_custom_noise_params_override(self) -> None:
        r = compute_noise_resilience(
            AngleEncoding(n_features=2),
            noise_params={"single_qubit": 0.02, "two_qubit": 0.2},
            n_samples=3,
            seed=0,
        )
        assert r.noise_level == "custom"
        assert r.single_qubit_error == 0.02
        assert r.two_qubit_error == 0.2


class TestNoiseLevels:
    def test_preset_structure(self) -> None:
        assert set(NOISE_LEVELS) == {"low", "medium", "high"}
        for params in NOISE_LEVELS.values():
            assert params["single_qubit"] < params["two_qubit"]


class TestNoisyDensityMatrix:
    def test_valid_density_matrix(self) -> None:
        enc = AngleEncoding(n_features=2)
        x = np.array([0.5, 1.0])
        rho = simulate_noisy_density_matrix(
            enc, x, single_qubit_error=0.05, two_qubit_error=0.05
        )
        dim = 2**enc.n_qubits
        assert rho.shape == (dim, dim)
        assert np.trace(rho).real == pytest.approx(1.0, abs=1e-6)  # unit trace
        assert np.allclose(rho, rho.conj().T)  # Hermitian
        assert np.linalg.eigvalsh(rho).min() > -1e-9  # PSD


class TestValidation:
    def test_zero_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            compute_noise_resilience(AngleEncoding(n_features=2), n_samples=0)

    def test_unknown_noise_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown noise_level"):
            compute_noise_resilience(AngleEncoding(n_features=2), noise_level="extreme")

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability must be in"):
            compute_noise_resilience(
                AngleEncoding(n_features=2), noise_params={"single_qubit": 1.5}
            )

    def test_qubit_cap_raises(self) -> None:
        # Guard triggers before any simulation, so this stays cheap.
        with pytest.raises(ValueError, match="limited to"):
            compute_noise_resilience(AngleEncoding(n_features=13), n_samples=1)
