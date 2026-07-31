"""Tests for kernel-geometry generalization diagnostics.

Matrix-level tests assert exact, verifiable invariants (``g(K, K) = 1``,
``d_eff(I_n, 1) = n/2``, ``KTA(yy^T, y) = 1``); encoding-level tests check the
metrics on real fidelity kernels and confirm they track the accuracy ranking
that expressibility does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.analysis.generalization import (
    centered_kernel_target_alignment,
    compute_effective_dimension,
    compute_fidelity_kernel,
    compute_geometric_difference,
    compute_kernel_target_alignment,
    geometric_difference,
    kernel_effective_dimension,
    kernel_target_alignment,
    sample_shot_kernel,
)


def _psd_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.5, 0.2, (8, 2)), rng.normal(2.7, 0.2, (8, 2))])
    y = np.array([0] * 8 + [1] * 8, dtype=np.intp)
    return X, y


# =====================================================================
# Fidelity kernel
# =====================================================================


class TestFidelityKernel:
    def test_properties(self) -> None:
        enc = AngleEncoding(n_features=2, rotation="Y")
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2 * np.pi, (10, 2))
        K = compute_fidelity_kernel(enc, X)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)
        assert np.allclose(np.diag(K), 1.0)
        assert np.linalg.eigvalsh(K).min() > -1e-9  # PSD
        assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-9


# =====================================================================
# Kernel-target alignment
# =====================================================================


class TestKernelTargetAlignment:
    def test_ideal_kernel_alignment_is_one(self) -> None:
        y = np.array([0, 0, 1, 1])
        ys = 2 * y - 1
        K_ideal = np.outer(ys, ys).astype(float)
        assert kernel_target_alignment(K_ideal, y) == pytest.approx(1.0)

    def test_anticorrelated_is_minus_one(self) -> None:
        y = np.array([0, 0, 1, 1])
        ys = 2 * y - 1
        assert kernel_target_alignment(
            -np.outer(ys, ys).astype(float), y
        ) == pytest.approx(-1.0)

    def test_centered_ideal_is_one(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1])
        ys = 2 * y - 1
        assert centered_kernel_target_alignment(
            np.outer(ys, ys).astype(float), y
        ) == pytest.approx(1.0)

    def test_single_class_centered_is_zero(self) -> None:
        # Centering the all-same label matrix yields the zero matrix, so the
        # centered alignment is 0 (no class structure to align to). The
        # uncentered variant lacks this property but must stay finite/in range.
        y = np.array([1, 1, 1, 1])
        K = _psd_matrix(4)
        assert centered_kernel_target_alignment(K, y) == 0.0
        assert -1.0 <= kernel_target_alignment(K, y) <= 1.0

    def test_centered_tiny_returns_zero(self) -> None:
        assert centered_kernel_target_alignment(np.array([[1.0]]), np.array([1])) == 0.0

    def test_range(self, separable: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = separable
        K = compute_fidelity_kernel(AngleEncoding(n_features=2), X)
        assert -1.0 <= kernel_target_alignment(K, y) <= 1.0
        assert -1.0 <= centered_kernel_target_alignment(K, y) <= 1.0


# =====================================================================
# Geometric difference
# =====================================================================


class TestGeometricDifference:
    def test_identical_kernels_is_one(self) -> None:
        K = _psd_matrix(8)
        assert geometric_difference(K, K, regularization=1e-10) == pytest.approx(
            1.0, abs=1e-4
        )

    def test_identity_kernels_is_one(self) -> None:
        identity = np.eye(6)
        assert geometric_difference(
            identity, identity, regularization=1e-10
        ) == pytest.approx(1.0, abs=1e-5)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shapes differ"):
            geometric_difference(np.eye(4), np.eye(5))

    def test_positive_and_finite(self) -> None:
        g = geometric_difference(_psd_matrix(8, 1), _psd_matrix(8, 2))
        assert np.isfinite(g) and g > 0.0

    def test_encoding_level_positive(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        g = compute_geometric_difference(AngleEncoding(n_features=2), X)
        assert np.isfinite(g) and g > 0.0

    def test_invalid_classical_kernel_raises(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        with pytest.raises(ValueError, match="rbf.*linear"):
            compute_geometric_difference(AngleEncoding(n_features=2), X, classical_kernel="poly")  # type: ignore[arg-type]


# =====================================================================
# Effective dimension
# =====================================================================


class TestEffectiveDimension:
    @pytest.mark.parametrize("n", [4, 10, 16])
    def test_identity_gives_half_n(self, n: int) -> None:
        # d_eff(I_n, lambda=1) = sum 1/(1+1) = n/2.
        assert kernel_effective_dimension(
            np.eye(n), regularization=1.0
        ) == pytest.approx(n / 2)

    def test_rank_one_is_small(self) -> None:
        # All-ones kernel has one eigenvalue n, rest 0 -> d_eff = n/(n+1) < 1.5.
        assert kernel_effective_dimension(np.ones((10, 10)), regularization=1.0) < 1.5

    def test_in_zero_to_n(self) -> None:
        K = _psd_matrix(12)
        d = kernel_effective_dimension(K)
        assert 0.0 <= d <= 12.0

    def test_larger_regularization_lowers_dimension(self) -> None:
        K = _psd_matrix(8)
        assert kernel_effective_dimension(
            K, regularization=5.0
        ) < kernel_effective_dimension(K, regularization=0.1)

    def test_invalid_regularization_raises(self) -> None:
        with pytest.raises(ValueError, match="regularization must be positive"):
            kernel_effective_dimension(np.eye(4), regularization=0.0)

    def test_encoding_level_range(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        d = compute_effective_dimension(AngleEncoding(n_features=2), X)
        assert 0.0 < d <= len(X)


# =====================================================================
# The payoff: metrics track accuracy where expressibility does not
# =====================================================================


class TestExplainsAccuracyRanking:
    def test_alignment_orders_angle_above_iqp(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Centered KTA reproduces the benchmark ranking (angle #1 > IQP #16),
        which expressibility inverts. Deterministic: seeded data, exact kernels."""
        X, y = separable
        angle_kta = compute_kernel_target_alignment(AngleEncoding(n_features=2), X, y)
        iqp_kta = compute_kernel_target_alignment(
            IQPEncoding(n_features=2, reps=2), X, y
        )
        assert angle_kta > iqp_kta


# =====================================================================
# Finite-shot kernel estimation
# =====================================================================


class TestSampleShotKernel:
    """The compute-uncompute estimator: all-zeros count ~ Binomial(shots, K)."""

    def test_structure_is_preserved(self) -> None:
        K = np.array([[1.0, 0.4, 0.7], [0.4, 1.0, 0.2], [0.7, 0.2, 1.0]])
        estimate = sample_shot_kernel(K, shots=500, seed=0)
        assert estimate.shape == K.shape
        assert np.allclose(estimate, estimate.T)  # symmetric
        assert np.allclose(np.diag(estimate), 1.0)  # exact unit diagonal
        assert estimate.min() >= 0.0 and estimate.max() <= 1.0

    def test_entries_are_multiples_of_one_over_shots(self) -> None:
        K = np.full((5, 5), 0.37)
        np.fill_diagonal(K, 1.0)
        shots = 64
        estimate = sample_shot_kernel(K, shots=shots, seed=1)
        counts = estimate * shots
        assert np.allclose(counts, np.round(counts))

    def test_estimator_is_unbiased(self) -> None:
        """Averaging many independent draws converges to the exact kernel."""
        K = np.array([[1.0, 0.3], [0.3, 1.0]])
        draws = [sample_shot_kernel(K, shots=200, seed=s)[0, 1] for s in range(400)]
        assert float(np.mean(draws)) == pytest.approx(0.3, abs=0.01)

    def test_variance_matches_binomial(self) -> None:
        """Per-entry variance is K(1 - K)/shots — the basis of the shot budget."""
        p, shots = 0.3, 200
        K = np.array([[1.0, p], [p, 1.0]])
        draws = [sample_shot_kernel(K, shots=shots, seed=s)[0, 1] for s in range(800)]
        assert float(np.var(draws)) == pytest.approx(p * (1 - p) / shots, rel=0.2)

    def test_more_shots_reduce_error(self) -> None:
        enc = AngleEncoding(n_features=3)
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2 * np.pi, (12, 3))
        exact = compute_fidelity_kernel(enc, X)

        def error(shots: int) -> float:
            estimate = sample_shot_kernel(exact, shots=shots, seed=7)
            return float(np.abs(estimate - exact).mean())

        assert error(10_000) < error(100)

    def test_determinism_same_seed(self) -> None:
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert np.array_equal(
            sample_shot_kernel(K, shots=100, seed=3),
            sample_shot_kernel(K, shots=100, seed=3),
        )

    def test_single_sample_kernel_is_identity(self) -> None:
        assert np.array_equal(
            sample_shot_kernel(np.array([[1.0]]), shots=10, seed=0), np.eye(1)
        )

    @pytest.mark.parametrize("bad", [0, -5, 2.5, True, "100"])
    def test_invalid_shots_raises(self, bad: object) -> None:
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="shots must be a positive integer"):
            sample_shot_kernel(K, shots=bad)  # type: ignore[arg-type]

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square 2D matrix"):
            sample_shot_kernel(np.zeros((2, 3)), shots=10)

    def test_out_of_range_entries_raise(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sample_shot_kernel(np.array([[1.0, 1.5], [1.5, 1.0]]), shots=10)


class TestShotAwareDiagnostics:
    """``shots=`` threads through the kernel into every downstream metric."""

    def test_default_is_exact(self, separable: tuple[np.ndarray, np.ndarray]) -> None:
        X, _ = separable
        enc = AngleEncoding(n_features=2)
        assert np.array_equal(
            compute_fidelity_kernel(enc, X), compute_fidelity_kernel(enc, X, shots=None)
        )

    def test_kernel_shots_are_applied(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        enc = AngleEncoding(n_features=2)
        noisy = compute_fidelity_kernel(enc, X, shots=50, seed=0)
        assert not np.array_equal(noisy, compute_fidelity_kernel(enc, X))
        assert np.allclose(np.diag(noisy), 1.0)

    def test_alignment_survives_finite_shots(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """KTA aggregates over all n(n-1)/2 entries, so shot noise averages out.

        This is why the atlas's KTA-based screening rule is usable on hardware
        even though the kernel matrix itself is not.
        """
        X, y = separable
        enc = AngleEncoding(n_features=2)
        exact = compute_kernel_target_alignment(enc, X, y)
        sampled = compute_kernel_target_alignment(enc, X, y, shots=2000, seed=0)
        assert sampled == pytest.approx(exact, abs=0.05)

    def test_geometric_difference_accepts_shots(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        g = compute_geometric_difference(
            AngleEncoding(n_features=2), X, shots=500, seed=0
        )
        assert np.isfinite(g) and g > 0.0

    def test_effective_dimension_accepts_shots(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        d = compute_effective_dimension(
            AngleEncoding(n_features=2), X, shots=500, seed=0
        )
        assert 0.0 < d <= len(X)

    def test_shot_noise_breaks_positive_semidefiniteness(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The documented caveat: sampled kernels need ensure_psd before use."""
        from encoding_atlas.benchmark.kernel import ensure_psd

        X, _ = separable
        noisy = compute_fidelity_kernel(
            AngleEncoding(n_features=2), X, shots=20, seed=0
        )
        assert float(np.linalg.eigvalsh(noisy).min()) < -1e-9
        repaired, was_clipped = ensure_psd(noisy)
        assert was_clipped
        assert float(np.linalg.eigvalsh(repaired).min()) > -1e-9
